"""Moyenne de plusieurs entrainements du meme modele (graines) : u, v moyens, w par continuite.

Chaque membre est un `WindFieldModel` complet (poids et norm_stats.json de SON run_dir), sur une
geometrie commune chargee une fois. La sortie servie de la moyenne :

    u = (1/N) somme_g u_g      v = (1/N) somme_g v_g      (m/s, apres denormalisation par membre)
    w = Geometry.w_continuite(u, v)                         (comme le modele servi)

w par continuite est lineaire en (u, v) : il vaut aussi la moyenne des w par continuite des membres.
Les passes des membres se font l'une apres l'autre ; la passe servie (`predict`, DSO4) ne garde que
les sommes de u et v, donc la memoire du GPU est celle des poids des N membres plus UNE passe.
Pour l'evaluation (`predire`, signature des predicteurs de evaluate.metriques_loader et de
scripts/dso_hauteur_sol.mesurer), les sorties des membres de l'instant sont gardees : `predire_membre`
les relit, chaque graine est donc evaluee sur les memes passes que la moyenne, sans les refaire.
La dispersion entre graines (`dispersion`) est reliability.ensemble_sigma autour de la moyenne.

Aucune ponderation, aucun poids appris : moyenne simple (configs/national/regle_ensemble_cp.json).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config, pick_device
from .dataset import assemble_input
from .geometry import N_SURF, Geometry
from .inference import Volume, WindFieldModel, caracteristiques
from .reliability import ensemble_sigma


class ModeleMoyen:
    """N entrainements du meme modele, servis par leur moyenne (u, v), w par continuite."""

    def __init__(self, cfg: Config, runs: list[str | Path],
                 checkpoints: list[str | Path | None] | None = None,
                 noms: list[str] | None = None, device: str | None = None,
                 verbose: bool = True) -> None:
        if len(runs) < 2:
            raise ValueError("une moyenne demande au moins deux entrainements")
        checkpoints = list(checkpoints) if checkpoints is not None else [None] * len(runs)
        if len(checkpoints) != len(runs):
            raise ValueError("un checkpoint (ou None) par run_dir")
        self.cfg = cfg
        self.device = pick_device(device)
        self.geom = Geometry(cfg, device=self.device)
        self.noms = list(noms) if noms is not None else [Path(r).name for r in runs]
        if len(set(self.noms)) != len(self.noms):
            raise ValueError(f"noms de membres en double : {self.noms}")
        self.membres: list[WindFieldModel] = []
        for run, ck in zip(runs, checkpoints):
            c = copy.deepcopy(cfg)
            c.apply_overrides({"paths.run_dir": Path(run)})
            self.membres.append(WindFieldModel(c, checkpoint=ck, device=self.device,
                                               verbose=verbose, geom=self.geom))
        archis = {repr(m.cfg.model) for m in self.membres}
        if len(archis) != 1:
            raise ValueError("membres d'architectures differentes : pas une moyenne de graines")
        self._cache: tuple[object, list] | None = None

    @property
    def n(self) -> int:
        return len(self.membres)

    # ------------------------------------------------------------------ passes
    @torch.no_grad()
    def passe(self, i: int, s_raw: torch.Tensor, sensor: torch.Tensor, tfeat: torch.Tensor,
              avec_w: bool = True) -> tuple[torch.Tensor, ...]:
        """Passe servie du membre i -> (u, v, w) en m/s (B, nz, ny, nx), comme
        Evaluator.predire et WindFieldModel.predict (une passe, w par continuite si la config
        le dit) ; avec_w=False : (u, v) seulement (passe servie de la moyenne)."""
        m = self.membres[i]
        with torch.autocast(device_type=self.device.type, dtype=m.amp_dtype, enabled=m.use_amp):
            pred = m.model(assemble_input(s_raw, sensor, tfeat, self.geom, m.norm))
        u, v, w = m.norm.denorm(pred.float())
        if not avec_w:
            return u, v
        if m.cfg.inference.w_continuite:
            w = self.geom.w_continuite(u, v)
        return u, v, w

    @torch.no_grad()
    def passes(self, s_raw, sensor, tfeat) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Les passes de tous les membres, dans l'ordre de `noms`."""
        return [self.passe(i, s_raw, sensor, tfeat) for i in range(self.n)]

    def moyenne(self, sorties: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(u, v) moyens des membres, w par continuite de (u, v) moyens."""
        u = sum(s[0] for s in sorties) / len(sorties)
        v = sum(s[1] for s in sorties) / len(sorties)
        return u, v, self.geom.w_continuite(u, v)

    @staticmethod
    def dispersion(moyenne: tuple, sorties: list[tuple]) -> dict[str, np.ndarray]:
        """Dispersion entre graines autour de la moyenne servie, par noeud : sigma_c =
        sqrt((1/N) somme_g (c_g - c_moyenne)^2), c = u, v, w, vitesse (reliability.ensemble_sigma ;
        champs (nz, ny, nx), un seul instant). Toujours >= 0."""
        def np3(a):
            a = a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
            return a[0] if a.ndim == 4 else a
        return ensemble_sigma(tuple(np3(a) for a in moyenne),
                              [tuple(np3(a) for a in s) for s in sorties])

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def predire(self, s_raw, sensor, tfeat, y, j: int, stem: str) -> list[torch.Tensor]:
        """Predicteur « moyenne » (signature de metriques_loader / dso_hauteur_sol.mesurer). Garde
        les sorties des membres de l'instant pour `predire_membre`."""
        sorties = self.passes(s_raw, sensor, tfeat)
        self._cache = ((int(j), str(stem)), sorties)
        return list(self.moyenne(sorties))

    def predire_membre(self, i: int):
        """Predicteur du membre i seul : relit les passes de `predire` pour le meme instant (memes
        sorties, aucune passe de plus), sinon fait sa passe."""
        def f(s_raw, sensor, tfeat, y, j: int, stem: str) -> list[torch.Tensor]:
            if self._cache is not None and self._cache[0] == (int(j), str(stem)):
                return list(self._cache[1][i])
            return list(self.passe(i, s_raw, sensor, tfeat))
        return f

    def predicteurs(self, nom_moyenne: str = "moyenne") -> dict:
        """{nom_moyenne: predire, <nom du membre>: predire_membre(i)} : la moyenne en premier
        (les membres relisent ses passes)."""
        out = {nom_moyenne: self.predire}
        for i, nom in enumerate(self.noms):
            out[nom] = self.predire_membre(i)
        return out

    # ------------------------------------------------------------------ service (DSO4)
    def surface_from_cache(self, stem: str) -> tuple[np.ndarray, float, int, str]:
        """Entree d'un instant en cache (entree de production pour un modele reel) : la meme
        pour tous les membres (meme config hors run_dir)."""
        return self.membres[0].surface_from_cache(stem)

    @torch.no_grad()
    def predict(self, S: np.ndarray, hour: float, doy: int, timestamp: str | None = None,
                mask_nan: bool = True, forcage: np.ndarray | None = None) -> Volume:
        """Passe SERVIE de la moyenne, comme WindFieldModel.predict (tous les capteurs visibles de
        S[7], une passe par membre, sans fiabilite) : les membres passent l'un apres l'autre, seules
        les sommes de u et v sont gardees ; w par continuite de la moyenne. `inference_s` = temps
        de la valeur (DSO4) : N passes, moyenne, w, copie vers le CPU, synchronisation CUDA."""
        geom = self.geom
        sensor = S[7] > 0.5
        extra = forcage if forcage is not None else self.membres[0]._forcage(timestamp)
        tfeat = caracteristiques(S, sensor, hour, doy, geom, extra)
        s_t = torch.from_numpy(np.ascontiguousarray(S[:N_SURF]))[None].to(self.device)
        m_t = torch.from_numpy(sensor.astype(np.float32))[None].to(self.device)
        t_t = torch.from_numpy(tfeat)[None].to(self.device)

        t0 = time.perf_counter()
        su = sv = None
        for i in range(self.n):
            u_i, v_i = self.passe(i, s_t, m_t, t_t, avec_w=False)
            su = u_i if su is None else su + u_i
            sv = v_i if sv is None else sv + v_i
            del u_i, v_i
        u_t, v_t = su / self.n, sv / self.n
        w_t = geom.w_continuite(u_t, v_t)
        u, v, w = (a[0].float().cpu().numpy() for a in (u_t, v_t, w_t))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        del su, sv, u_t, v_t, w_t

        if mask_nan:
            u = np.where(geom.valid, u, np.nan)
            v = np.where(geom.valid, v, np.nan)
            w = np.where(geom.valid, w, np.nan)
        return Volume(u, v, w, geom, timestamp=timestamp, n_sensors=int(sensor.sum()),
                      inference_s=elapsed)

    def benchmark(self, S: np.ndarray, hour: float, doy: int, n: int = 20, warmup: int = 3,
                  timestamp: str | None = None, membres: bool = True) -> dict:
        """DSO4 de la moyenne, mesure comme WindFieldModel.benchmark (chauffe, puis n appels de la
        passe servie ; moyenne, p95, max ; conforme si moyenne < dso.time_s). membres : temps de la
        passe servie de chaque membre seul (WindFieldModel.benchmark, sans fiabilite), pour
        information. Sur GPU : memoire des poids et pic de la passe de la moyenne."""
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize()
            poids = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        for _ in range(warmup):
            self.predict(S, hour, doy, timestamp=timestamp, mask_nan=False)
        times = []
        for _ in range(n):
            times.append(self.predict(S, hour, doy, timestamp=timestamp,
                                      mask_nan=False).inference_s)
        out = {"n": n, "chauffe": warmup, "membres": self.n, "mean_s": float(np.mean(times)),
               "p95_s": float(np.quantile(times, 0.95)), "max_s": float(np.max(times)),
               "seuil_s": float(self.cfg.dso.time_s),
               "conforme": bool(np.mean(times) < self.cfg.dso.time_s),
               "device": str(self.device)}
        if cuda:
            out["gpu"] = torch.cuda.get_device_name(self.device)
            out["memoire_gpu_go"] = {
                "poids_et_geometrie": poids / 1e9,
                "pic_passe_servie": torch.cuda.max_memory_allocated(self.device) / 1e9,
                "total": torch.cuda.get_device_properties(self.device).total_memory / 1e9}
        if membres:
            out["par_membre"] = {
                nom: {k: v for k, v in m.benchmark(S, hour, doy, n=n, warmup=warmup,
                                                   reliability=False, timestamp=timestamp).items()
                      if k in ("mean_s", "p95_s", "max_s")}
                for nom, m in zip(self.noms, self.membres)}
        return out
