"""Coeur d'inference partage par l'evaluation, la reconstruction et l'API.

`WindFieldModel` charge une fois la geometrie, les statistiques et les poids, puis
reconstruit le volume (u, v, w) d'un instant a partir des seules mesures de surface.
C'est l'objet que le service HTTP garde en memoire.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config, pick_device
from .dataset import assemble_input
from .features import time_geo_features
from .geometry import N_SURF, SURF_VARS, Geometry
from .model import build_model, model_cfg_from_checkpoint
from .normalize import Normalizer


class Volume:
    """Resultat d'une reconstruction : u, v, w (nz, ny, nx), NaN sous le relief."""

    def __init__(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, geom: Geometry,
                 timestamp: str | None = None, n_sensors: int = 0,
                 inference_s: float = 0.0, correction: dict | None = None) -> None:
        self.u, self.v, self.w = u, v, w
        self.geom = geom
        self.timestamp = timestamp
        self.n_sensors = n_sensors
        self.inference_s = inference_s
        # correction post-modele appliquee (fichier, sha256...), None si modele seul
        self.correction = correction

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.u, self.v)

    @property
    def direction(self) -> np.ndarray:
        """Convention meteo : 0 = vent venant du nord, 90 = de l'est."""
        return (270.0 - np.degrees(np.arctan2(self.v, self.u))) % 360.0

    def at_altitude(self, z: float, field: np.ndarray | None = None) -> np.ndarray:
        """Interpolation lineaire entre niveaux ; NaN sous le relief."""
        levels, nz = self.geom.z_levels, self.geom.nz
        if z < levels[0] or z > levels[-1]:
            raise ValueError(f"Altitude hors domaine [{levels[0]:.0f}, {levels[-1]:.0f}] m")
        f = self.speed if field is None else field
        k = int(np.clip(np.searchsorted(levels, z) - 1, 0, nz - 2))
        a = (z - levels[k]) / (levels[k + 1] - levels[k])
        out = (1 - a) * f[k] + a * f[k + 1]
        return np.where(z > self.geom.zt, out, np.nan)

    def at_point(self, x_m: float, y_m: float, z_asl_m: float) -> dict:
        """Vent au point le plus proche en (x, y), interpole en altitude."""
        g = self.geom
        i = int(np.clip(round(x_m / g.dx), 0, g.nx - 1))
        j = int(np.clip(round(y_m / g.dy), 0, g.ny - 1))
        vals = {}
        for name, field in (("u", self.u), ("v", self.v), ("w", self.w)):
            col = self.at_altitude(z_asl_m, field)
            vals[name] = float(col[j, i])
        z_terrain = float(g.zt[j, i])
        # un point sous le relief n'a pas de vent : on le DIT, au lieu de renvoyer des null
        # sans explication (le relief atteint 677 m sur le domaine reel : 120 m ASL est
        # souvent sous terre)
        underground = not np.isfinite(vals["u"])

        def clean(x: float) -> float | None:
            return None if not np.isfinite(x) else x

        speed = float(np.hypot(vals["u"], vals["v"]))
        direction = float((270.0 - np.degrees(np.arctan2(vals["v"], vals["u"]))) % 360.0)
        return {
            "ix": i * g.stride, "iy": j * g.stride,
            "x_m": i * g.dx, "y_m": j * g.dy,
            "z_asl_m": z_asl_m, "z_terrain_m": z_terrain,
            "z_agl_m": round(z_asl_m - z_terrain, 2),
            "sous_le_relief": underground,
            "wind_u_m_s": clean(vals["u"]), "wind_v_m_s": clean(vals["v"]),
            "wind_w_m_s": clean(vals["w"]),
            "wind_speed_m_s": clean(speed),
            "wind_direction_deg": clean(direction),
            "timestamp_utc": self.timestamp,
        }

    def to_nodes_frame(self) -> pd.DataFrame:
        """Un noeud 3D du dataset par ligne (format attendu par la base ANEO)."""
        g = self.geom
        ok = g.node_flat >= 0
        pid = np.nonzero(ok)[0] + g.node_id_start
        k, j, i = np.unravel_index(g.node_flat[ok], g.shape)
        speed, direction = self.speed, self.direction
        return pd.DataFrame({
            "point_id": pid, "timestamp_utc": self.timestamp,
            "ix": i * g.stride, "iy": j * g.stride,
            "z_asl_m": g.z_levels[k], "z_agl_m": np.round(g.z_levels[k] - g.zt[j, i], 2),
            "wind_u_m_s": np.round(self.u[k, j, i], 3),
            "wind_v_m_s": np.round(self.v[k, j, i], 3),
            "wind_w_m_s": np.round(self.w[k, j, i], 4),
            "wind_speed_m_s": np.round(speed[k, j, i], 3),
            "wind_direction_deg": np.round(direction[k, j, i], 1),
        })

    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        g = self.geom
        np.savez_compressed(
            path, u=self.u.astype(np.float32), v=self.v.astype(np.float32),
            w=self.w.astype(np.float32), speed=self.speed.astype(np.float32),
            direction=self.direction.astype(np.float32), z_levels=g.z_levels,
            x_m=np.arange(g.nx) * g.dx, y_m=np.arange(g.ny) * g.dy,
            z_terrain=g.zt.astype(np.float32), timestamp=str(self.timestamp),
            # cle presente SEULEMENT si une correction a ete appliquee : desactivee, le .npz
            # garde exactement les cles de main
            **({"correction": json.dumps(self.correction)} if self.correction else {}))
        return path


@torch.no_grad()
def ensemble_forward(net: torch.nn.Module, geom: Geometry, norm: Normalizer,
                     s_raw: torch.Tensor, sensor: torch.Tensor, hour: float, doy: int,
                     k: int = 1, keep: float = 0.7, seed: int = 0,
                     amp_dtype=torch.float32, use_amp: bool = False,
                     tfeat: torch.Tensor | None = None) -> torch.Tensor:
    """Moyenne de k passes avant, chacune sur un sous-ensemble different des capteurs.

    Passe 0 : les capteurs fournis. Passes 1..k-1 : chacun garde avec probabilite `keep`.
    Pour chaque passe, les 8 caracteristiques temps / physique (vent geostrophique, vent
    thermique) sont RECALCULEES sur les capteurs gardes : c'est ainsi que le modele les a
    vues a l'entrainement. Les reutiliser telles quelles — ce que faisait le BLOC 10c du
    notebook — donne a chaque passe une entree qui n'existe pas dans la distribution
    d'entrainement (des capteurs masques, mais un vent geostrophique calcule sans masque).

    Entrees pour un seul instant : s_raw (1, 7, ny, nx), sensor (1, ny, nx).
    Sortie : prediction NORMALISEE (1, 3, nz, ny, nx), comme une passe simple.
    """
    device = s_raw.device
    if tfeat is None:
        tfeat = torch.from_numpy(time_geo_features(
            s_raw[0].float().cpu().numpy(), sensor[0].cpu().numpy() > 0.5, hour, doy, geom)
        )[None].to(device)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        total = net(assemble_input(s_raw, sensor, tfeat, geom, norm)).float()
    if k <= 1:
        return total

    s_np = s_raw[0].float().cpu().numpy()
    base = sensor[0].cpu().numpy() > 0.5
    rng = np.random.default_rng(seed)
    for _ in range(k - 1):
        kept = base & (rng.random(base.shape) < keep)
        if kept.sum() == 0:                       # garde-fou : jamais une passe sans capteur
            kept = base
        sensor_i = torch.from_numpy(kept.astype(np.float32))[None].to(device)
        tfeat_i = torch.from_numpy(time_geo_features(s_np, kept, hour, doy, geom))[None].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            total = total + net(assemble_input(s_raw, sensor_i, tfeat_i, geom, norm)).float()
    return total / k


class WindFieldModel:
    """Modele pret a l'emploi : geometrie + statistiques + poids, charges une seule fois."""

    def __init__(self, cfg: Config, checkpoint: str | Path | None = None,
                 device: str | None = None, verbose: bool = True) -> None:
        self.cfg = cfg
        self.device = pick_device(device)
        self.geom = Geometry(cfg, device=self.device)

        self.checkpoint_path = Path(checkpoint) if checkpoint \
            else cfg.paths.checkpoint_for_inference()
        # Sur CPU, les poids sont MAPPES depuis le fichier au lieu d'etre copies en RAM, et
        # le modele est construit sur le peripherique `meta` (aucune allocation) puis
        # rattache directement a ces tenseurs. Mesure sur la v7 (1,81 Go) : +0,03 Go de RAM
        # au chargement, contre ~3,6 Go en chargeant le fichier puis en le recopiant dans un
        # modele deja alloue. Les pages des poids appartiennent au cache du fichier : le
        # systeme peut les evincer sous pression memoire et les relire du disque.
        mapped = self.device.type == "cpu"
        # str() : sous torch 2.2, mmap exige un nom de fichier, pas un Path
        ck = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False,
                        mmap=mapped)
        saved = model_cfg_from_checkpoint(ck)
        if saved is not None:
            # l'architecture du checkpoint fait foi : un YAML modifie ne doit pas la casser
            cfg.model = saved
            self.geom.pool_surface_enabled = saved.pool_surface
        if mapped:
            with torch.device("meta"):
                self.model = build_model(cfg, self.geom.c_in)
            # strict : la moindre cle manquante ou de forme differente est une erreur
            self.model.load_state_dict(ck["model"], strict=True, assign=True)
        else:
            self.model = build_model(cfg, self.geom.c_in).to(self.device)
            self.model.load_state_dict(ck["model"])
        self.model.eval()

        self.norm = Normalizer.from_file(cfg.paths.norm_stats, self.geom,
                                         cfg.loss.w_level_max)
        self.amp_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.use_amp = self.device.type == "cuda"
        # une seule passe avant a la fois : sur CPU, la v7 prend ~5 Go par passe, et
        # l'application web lance 3-4 requetes en parallele a chaque instant. Sans verrou,
        # chacune lancerait sa propre passe et la machine saturerait sa memoire.
        self._lock = threading.Lock()
        # correction post-modele du vent moyen par niveau (lot M1) : strictement opt-in
        from .correction import check_top_level, load_correction, refuse_tta
        self.correction = load_correction(cfg.inference.correction)
        if self.correction is not None:
            refuse_tta(cfg.inference.tta, self.correction)   # ajustee sur une seule passe
            check_top_level(self.geom)                        # refus au chargement, pas en service
        self.epoch = ck.get("epoch")
        self.val_score = ck.get("val_score")
        if verbose:
            print(f"modele : {self.checkpoint_path.name} (epoque {(self.epoch or 0) + 1}) "
                  f"| {self.device} | {self.geom.summary()}")

    # ------------------------------------------------------------------ entrees
    def surface_from_csv(self, path: str | Path) -> tuple[np.ndarray, float, int, str]:
        """Mesures de surface d'un instant -> (S (8, ny, nx), heure, jour de l'annee, date)."""
        cols = ["point_id", "timestamp_utc", "mask_i_obs"] + SURF_VARS
        df = pd.read_csv(path, usecols=cols)
        df = df[df["point_id"] < self.geom.node_id_start]
        if df.empty:
            raise ValueError(f"{path} : aucune mesure de surface (point_id < "
                             f"{self.geom.node_id_start})")
        first = df["timestamp_utc"].iloc[0]
        df = df[df["timestamp_utc"] == first]
        stamp = pd.Timestamp(first)
        S = np.zeros((8, self.geom.ny_s * self.geom.nx_s), dtype=np.float32)
        flat = self.geom.pid_to_surf[df["point_id"].values]
        for ci, var in enumerate(SURF_VARS):
            S[ci, flat] = df[var].values
        S[7, flat] = df["mask_i_obs"].values
        S = self.geom.pool_surface(S)
        hour = stamp.hour + stamp.minute / 60 + stamp.second / 3600
        return S, hour, int(stamp.dayofyear), str(stamp)

    def surface_from_cache(self, stem: str) -> tuple[np.ndarray, float, int, str]:
        """Instant deja en cache (ex. un jour de test) : "004_120000"."""
        from .cache import load_index

        index = load_index(self.cfg)
        rows = index[index["stem"] == stem]
        if rows.empty:
            raise KeyError(f"instant {stem} absent de {self.cfg.paths.index_csv}")
        row = rows.iloc[0]
        S = np.load(self.cfg.paths.cache_dir / f"{stem}_S.npy")
        return S, float(row["hour"]), int(row["doy"]), str(row["timestamp"])

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, S: np.ndarray, hour: float, doy: int, timestamp: str | None = None,
                keep: float = 1.0, seed: int = 0, mask_nan: bool = True,
                tta: int | None = None, tta_keep: float | None = None) -> Volume:
        """Reconstruit le volume a partir de la surface agregee S (8, ny, nx).

        tta / tta_keep : moyenne de plusieurs passes (voir `ensemble_forward`). Par defaut,
        la valeur de la configuration (`inference.tta`, 1 = une seule passe).
        """
        geom = self.geom
        k = self.cfg.inference.tta if tta is None else int(tta)
        k_keep = self.cfg.inference.tta_keep if tta_keep is None else float(tta_keep)
        if self.correction is not None:
            from .correction import refuse_tta
            refuse_tta(k, self.correction)
        rng = np.random.default_rng(seed)
        sensor = (S[7] > 0.5) & (rng.random(S[7].shape) < keep)
        tfeat = time_geo_features(S, sensor, hour, doy, geom)

        s_t = torch.from_numpy(np.ascontiguousarray(S[:N_SURF]))[None].to(self.device)
        m_t = torch.from_numpy(sensor.astype(np.float32))[None].to(self.device)
        t_t = torch.from_numpy(tfeat)[None].to(self.device)

        with self._lock:
            t0 = time.perf_counter()
            pred = ensemble_forward(self.model, geom, self.norm, s_t, m_t, hour, doy, k=k,
                                    keep=k_keep, seed=seed, amp_dtype=self.amp_dtype,
                                    use_amp=self.use_amp, tfeat=t_t)
            u, v, w = (a[0].float().cpu().numpy() for a in self.norm.denorm(pred.float()))
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            del pred

        corr_info = None
        if self.correction is not None:
            # meme ordre que le BLOC 12 (etape 2b) : prediction du modele, puis recalage de la
            # moyenne des niveaux ; le temps de la correction compte dans l'inference (DSO4)
            t1 = time.perf_counter()
            u, v, n_corr = self.correction.apply(u, v, S, sensor, tfeat, geom)
            elapsed += time.perf_counter() - t1
            corr_info = {**self.correction.info(), "appliquee_a": n_corr}
        if mask_nan:
            u = np.where(geom.valid, u, np.nan)
            v = np.where(geom.valid, v, np.nan)
            w = np.where(geom.valid, w, np.nan)
        return Volume(u, v, w, geom, timestamp=timestamp, n_sensors=int(sensor.sum()),
                      inference_s=elapsed, correction=corr_info)

    def predict_stem(self, stem: str, keep: float = 1.0, seed: int = 0) -> Volume:
        S, hour, doy, stamp = self.surface_from_cache(stem)
        return self.predict(S, hour, doy, timestamp=stamp, keep=keep, seed=seed)

    def predict_csv(self, path: str | Path, keep: float = 1.0, seed: int = 0) -> Volume:
        S, hour, doy, stamp = self.surface_from_csv(path)
        return self.predict(S, hour, doy, timestamp=stamp, keep=keep, seed=seed)

    def benchmark(self, S: np.ndarray, hour: float, doy: int, n: int = 20,
                  warmup: int = 3) -> dict:
        """DSO4 — temps d'inference par instant."""
        for _ in range(warmup):
            self.predict(S, hour, doy, mask_nan=False)
        times = [self.predict(S, hour, doy, mask_nan=False).inference_s for _ in range(n)]
        return {"n": n, "mean_s": float(np.mean(times)), "p95_s": float(np.quantile(times, 0.95)),
                "max_s": float(np.max(times)),
                "conforme": bool(np.mean(times) < self.cfg.dso.time_s)}

    def meta(self) -> dict:
        """Description de la sortie, pour l'API et le visualiseur."""
        g = self.geom
        return {
            "modele": self.checkpoint_path.name,
            "epoque": (self.epoch + 1) if self.epoch is not None else None,
            "device": str(self.device),
            "grille": {"nz": g.nz, "ny": g.ny, "nx": g.nx, "dx_m": g.dx, "dy_m": g.dy},
            "z_levels_asl_m": g.z_levels.tolist(),
            "noeuds_valides": int(g.valid.sum()),
            "passes_moyennees": self.cfg.inference.tta,
            "correction": self.correction.info() if self.correction is not None else None,
            "variables": ["wind_u_m_s", "wind_v_m_s", "wind_w_m_s", "wind_speed_m_s",
                          "wind_direction_deg"],
            "conventions": g.meta.raw.get("conventions", {}),
        }
