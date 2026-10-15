"""BLOC 10 — evaluation sur le test et verdict sur les DSO.

Produit, dans `run_dir/eval_<modele>/` :
    test_par_niveau_synoptique.csv   RMSE et R2 par niveau (instants synoptiques)
    test_par_niveau_episodes.csv     idem sur les episodes rapides (test de stress)
    test_par_instant.csv             une ligne par instant (RMSE, residu, rugosite, direction)
    robustesse_capteurs.csv          RMSE max selon la fraction de capteurs visibles
    test_summary.json                les verdicts DSO1 a DSO5
    dso6_<calibration>.json          (option) verdict DSO6 de la fiabilite, voir `dso6`

et, sur la VALIDATION, la calibration de la fiabilite (`calibrate_reliability`).

Les metriques sont CUMULEES sur tous les instants (somme des carres), pas moyennees par
instant : c'est ce que demande un R2 global.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .dataset import ANEO3D, assemble_input, eval_loader, is_synoptic, make_splits
from .dso import dso3_conforme
from .inference import WindFieldModel, ensemble_forward
from .losses import PhysicsLoss
from .reliability import (COMPOSANTES, Calibration, DSO6Accumulator, RatioHistogram,
                          ensemble_sigma, fit_calibration, member_masks)

ROBUSTNESS_KEEPS = (0.1, 0.25, 0.5, 1.0)
ROBUSTNESS_INSTANTS = 40
EVAL_NAMESPACE = "eval_pcfno3d"     # run_dir/eval_pcfno3d/<modele>[_ttaN]/
RELIABILITY_NAMESPACE = "fiabilite_pcfno3d"   # run_dir/fiabilite_pcfno3d/<modele>[_ttaN]/


class Evaluator:
    def __init__(self, cfg: Config, checkpoint: str | Path | None = None,
                 device: str | None = None, verbose: bool = True,
                 tta: int | None = None, tta_keep: float | None = None) -> None:
        from .cache import load_index

        self.cfg = cfg
        self.verbose = verbose
        self.wf = WindFieldModel(cfg, checkpoint=checkpoint, device=device, verbose=verbose)
        self.geom = self.wf.geom
        self.norm = self.wf.norm
        self.index = load_index(cfg)
        self.splits = make_splits(cfg, self.index)
        self.loss_fn = PhysicsLoss(cfg, self.geom, self.norm)
        # moyenne de passes (voir inference.ensemble_forward) ; par defaut celle de la config
        self.tta = cfg.inference.tta if tta is None else int(tta)
        self.tta_keep = cfg.inference.tta_keep if tta_keep is None else float(tta_keep)
        if tta is not None:
            # le benchmark DSO4 passe par wf.predict : il doit mesurer la meme configuration
            cfg.inference.tta, cfg.inference.tta_keep = self.tta, self.tta_keep
        self._time_of = dict(zip(self.index["stem"],
                                 zip(self.index["hour"], self.index["doy"])))
        suffix = f"_tta{self.tta}" if self.tta > 1 else ""
        # sous-dossier propre au paquet : le notebook v7 ecrit deja run_dir/eval_<modele>,
        # et ces resultats sont la REFERENCE a laquelle on compare le paquet. Les ecraser
        # detruirait precisement ce qu'on cherche a verifier.
        self.out_dir = (cfg.paths.run_dir / EVAL_NAMESPACE
                        / f"{self.wf.checkpoint_path.stem}{suffix}")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ metriques
    @torch.no_grad()
    def evaluate_loader(self, loader, stems) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Metriques par niveau (cumulees) et par instant."""
        geom, norm, device = self.geom, self.norm, self.wf.device
        nz = geom.nz
        acc = {c: {k: np.zeros(4) for k in range(nz)} for c in "uvw"}   # n, sse, sum, sum2
        spd_sum = np.zeros(nz)
        per_instant: list[dict] = []

        for s_raw, sensor, tfeat, y, j in loader:
            s_raw, sensor, tfeat, y = (t.to(device) for t in (s_raw, sensor, tfeat, y))
            stem = stems[int(j)]
            if self.tta > 1:
                hour, doy = self._time_of[stem]
                pred = ensemble_forward(self.wf.model, geom, norm, s_raw, sensor,
                                        float(hour), int(doy), k=self.tta, keep=self.tta_keep,
                                        seed=int(j), amp_dtype=self.wf.amp_dtype,
                                        use_amp=self.wf.use_amp, tfeat=tfeat)
            else:
                with torch.autocast(device_type=device.type, dtype=self.wf.amp_dtype,
                                    enabled=self.wf.use_amp):
                    pred = self.wf.model(assemble_input(s_raw, sensor, tfeat, geom, norm))
            u, v, w = norm.denorm(pred.float())
            row = {"stem": stem, "timestamp": self.splits.stamp_of[stem]}

            for c, p in zip("uvw", (u, v, w)):
                t = y[:, "uvw".index(c)]
                # noeuds valides ET connus : une cible reelle peut etre partielle (NaN)
                known = geom.valid_t & torch.isfinite(t[0])
                for k in range(nz):
                    m = known[k]
                    if m.sum() == 0:
                        continue
                    pk, tk = p[0, k][m].double(), t[0, k][m].double()
                    acc[c][k] += np.array([tk.numel(), ((pk - tk) ** 2).sum().item(),
                                           tk.sum().item(), (tk ** 2).sum().item()])
                pv, tv = p[0][known], t[0][known]
                row[f"rmse_{c}"] = (torch.sqrt(((pv - tv) ** 2).mean()).item()
                                    if pv.numel() else float("nan"))

            known_uv = geom.valid_t & torch.isfinite(y[0, 0]) & torch.isfinite(y[0, 1])
            for k in range(nz):
                m = known_uv[k]
                if m.sum() > 0:
                    spd_sum[k] += torch.hypot(y[0, 0, k][m], y[0, 1, k][m]).double().sum().item()

            # erreur de direction, hors vents calmes (direction non definie sous 1 m/s)
            dp = torch.rad2deg(torch.atan2(v[0], u[0]))[known_uv]
            dt = torch.rad2deg(torch.atan2(y[0, 1], y[0, 0]))[known_uv]
            dd = torch.abs(dp - dt) % 360
            dd = torch.minimum(dd, 360 - dd)
            calm = torch.hypot(y[0, 0], y[0, 1])[known_uv] < 1.0
            row["mae_dir_deg"] = ((dd[~calm].mean() if (~calm).any() else dd.mean()).item()
                                  if dd.numel() else float("nan"))

            r = geom.mass_residual(y[:, 3], u, v, w)
            row["mass_rms"] = torch.sqrt((r[:, geom.mass_mask_t] ** 2).mean()).item()
            row["roughness_ratio"] = self.loss_fn.roughness_ratio(u, v, y).item()
            per_instant.append(row)

        levels = []
        for k in range(nz):
            rec = {"z_m": int(geom.z_levels[k]), "couverture": float(geom.cover[k]),
                   "vitesse_moyenne": float(spd_sum[k] / max(acc["u"][k][0], 1))}
            for c in "uvw":
                n, sse, st, st2 = acc[c][k]
                if n == 0:
                    continue
                sst = st2 - st ** 2 / n
                rec[f"rmse_{c}"] = float(np.sqrt(sse / n))
                rec[f"r2_{c}"] = float(1 - sse / sst) if sst > 0 else float("nan")
            levels.append(rec)
        return pd.DataFrame(levels), pd.DataFrame(per_instant)

    # ------------------------------------------------------------------ campagne
    def run(self) -> dict:
        cfg, geom = self.cfg, self.geom
        te_s = self.splits.test[0]
        if len(te_s) == 0:
            raise ValueError("Aucun instant de test dans l'index")
        syn = np.array([is_synoptic(self.splits.stamp_of[s]) for s in te_s])
        syn_idx, ep_idx = np.nonzero(syn)[0], np.nonzero(~syn)[0]
        if len(syn_idx) == 0:          # dataset sans instant synoptique (mini-dataset)
            syn_idx, ep_idx = np.arange(len(te_s)), np.zeros(0, dtype=int)

        levels, res = self.evaluate_loader(*eval_loader(cfg, geom, self.splits, syn_idx, 2000))
        if len(ep_idx):
            levels_ep, res_ep = self.evaluate_loader(
                *eval_loader(cfg, geom, self.splits, ep_idx, 2500))
        else:
            levels_ep, res_ep = pd.DataFrame(), pd.DataFrame()

        res["type"] = "synoptique"
        if len(res_ep):
            res_ep["type"] = "episode"
        levels.to_csv(self.out_dir / "test_par_niveau_synoptique.csv", index=False)
        if len(levels_ep):
            levels_ep.to_csv(self.out_dir / "test_par_niveau_episodes.csv", index=False)
        res_all = pd.concat([res, res_ep], ignore_index=True)
        res_all.to_csv(self.out_dir / "test_par_instant.csv", index=False)

        # DSO4 : temps d'inference sur un instant reel
        S, hour, doy, _ = self.wf.surface_from_cache(te_s[0])
        timing = self.wf.benchmark(S, hour, doy, n=20)
        if self.wf.reliability is not None:
            # DSO4 mesure AVEC la fiabilite (le temps reel de livraison) et SANS
            timing["sans_fiabilite"] = self.wf.benchmark(S, hour, doy, n=20, reliability=False)

        # DSO5 : robustesse a un reseau de capteurs degrade
        sub = syn_idx[np.unique(np.linspace(0, len(syn_idx) - 1,
                                            min(ROBUSTNESS_INSTANTS, len(syn_idx))).astype(int))]
        robust_rows = []
        for keep in ROBUSTNESS_KEEPS:
            lv, _ = self.evaluate_loader(
                *eval_loader(cfg, geom, self.splits, sub, 3000, keep=keep))
            lv = lv[lv["couverture"] >= cfg.dso.min_level_cover]
            robust_rows.append({"capteurs_visibles": keep,
                                "rmse_u_max": lv["rmse_u"].max(), "rmse_v_max": lv["rmse_v"].max(),
                                "r2_u_min": lv["r2_u"].min(), "r2_v_min": lv["r2_v"].min()})
        robust = pd.DataFrame(robust_rows)
        robust.to_csv(self.out_dir / "robustesse_capteurs.csv", index=False)

        summary = self._verdicts(levels, levels_ep, res, res_ep, res_all, robust, timing)
        (self.out_dir / "test_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.verbose:
            print(format_report(summary, levels, levels_ep, robust, cfg))
            print(f"resultats -> {self.out_dir}")
        return summary

    # ------------------------------------------------------------------ fiabilite (M2)
    def _reliability_instants(self, jeu: str, n_instants: int | None,
                              keep: float | None) -> ANEO3D:
        """Instants synoptiques du jeu, memes graines que DSO1 (test) et la validation de
        l'entrainement, et memes capteurs visibles (`mask.keep_eval` par defaut)."""
        if jeu == "validation":
            s, h, d = self.splits.val
            idx, seed = self.splits.val_selection, 1000
        elif jeu == "test":
            s, h, d = self.splits.test
            syn = np.array([is_synoptic(self.splits.stamp_of[x]) for x in s], dtype=bool)
            idx, seed = (np.nonzero(syn)[0] if syn.any() else np.arange(len(s))), 2000
        else:
            raise ValueError(f"jeu inconnu : {jeu!r} (validation ou test)")
        if n_instants is not None and len(idx) > n_instants:
            idx = idx[np.unique(np.linspace(0, len(idx) - 1, n_instants).astype(int))]
        if len(idx) == 0:
            raise ValueError(f"aucun instant de {jeu} en cache")
        return ANEO3D(s[idx], h[idx], d[idx], self.geom, train=False, seed=seed,
                      keep_eval=keep)

    @torch.no_grad()
    def _reliability_loop(self, jeu: str, n_instants: int | None, keep: float | None,
                          on_instant) -> int:
        """Pour chaque instant : passe servie, ensemble masque, sigma BRUT, erreurs.
        `on_instant(k, comp, err_abs, sigma_brut)` recoit les noeuds valides et connus."""
        geom, wf, rc = self.geom, self.wf, self.cfg.reliability
        levels = geom.eval_levels()
        ds = self._reliability_instants(jeu, n_instants, keep)
        for n, (s_raw, sensor, _tfeat, y, j) in enumerate(ds):
            s_np, m_np = s_raw.numpy(), sensor.numpy() > 0.5
            hour, doy = float(ds.hours[j]), int(ds.doys[j])
            if self.tta > 1:
                pred = ensemble_forward(
                    wf.model, geom, self.norm, s_raw[None].to(wf.device),
                    sensor[None].to(wf.device), hour, doy, k=self.tta, keep=self.tta_keep,
                    seed=int(j), amp_dtype=wf.amp_dtype, use_amp=wf.use_amp)
                served = tuple(a[0].float().cpu().numpy() for a in self.norm.denorm(pred))
            else:
                served = wf._member(s_np, m_np, hour, doy)
            members = [wf._member(s_np, m, hour, doy)
                       for m in member_masks(m_np, rc.members, rc.keep, rc.seed)]
            sig = ensemble_sigma(served, members)
            y_np = y.numpy()
            truth = {"u": y_np[0], "v": y_np[1], "w": y_np[2],
                     "speed": np.hypot(y_np[0], y_np[1])}
            pred = {"u": served[0], "v": served[1], "w": served[2],
                    "speed": np.hypot(served[0], served[1])}
            for c in COMPOSANTES:
                known = geom.valid & np.isfinite(truth[c])
                for k in levels:
                    m = known[k]
                    if m.any():
                        on_instant(k, c, np.abs(pred[c][k][m] - truth[c][k][m]), sig[c][k][m])
            if self.verbose and (n + 1) % 10 == 0:
                print(f"  fiabilite ({jeu}) : {n + 1}/{len(ds)} instants", flush=True)
        return len(ds)

    def reliability_dir(self) -> Path:
        """Meme nom que le dossier d'evaluation (modele, tta, correction...) : une
        calibration ne vaut que pour la sortie exacte sur laquelle elle a ete mesuree."""
        return self.cfg.paths.run_dir / RELIABILITY_NAMESPACE / self.out_dir.name

    def calibrate_reliability(self, out: str | Path | None = None,
                              n_instants: int | None = None,
                              keep: float | None = None) -> dict:
        """Calibre les facteurs k sur la VALIDATION et ecrit le JSON gele.

        N'ecrase jamais un fichier existant : la calibration retenue est une reference.
        """
        out = Path(out) if out else self.reliability_dir() / "calibration_validation.json"
        if out.exists():
            raise FileExistsError(f"{out} existe deja : choisis un autre --out (une "
                                  "calibration retenue n'est jamais ecrasee)")
        rc = self.cfg.reliability
        hist = RatioHistogram(self.geom.nz)
        n = self._reliability_loop(
            "validation", n_instants, keep,
            lambda k, c, e, sg: hist.add(c, k, e, sg, rc.sigma_floor_m_s))
        keep_eff = self.cfg.mask.keep_eval if keep is None else keep
        data = fit_calibration(hist, self.geom, "validation", rc, n, extra={
            "modele": self.wf.checkpoint_path.name, "capteurs_visibles": keep_eff,
            "tta": self.tta})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.verbose:
            cov = data["couverture80_validation"]
            kg = "  ".join(f"{c} {data['k_global'][c]:.3f}" for c in COMPOSANTES)
            cv = "  ".join(f"{c} {cov[c]:.3f}" for c in COMPOSANTES)
            print(f"calibration ({n} instants de validation) -> {out}\n"
                  f"  k global : {kg}\n  couverture 80 % obtenue : {cv}")
        return {**data, "fichier": str(out)}

    def dso6(self, calibration: str | Path | None = None, n_instants: int | None = None,
             keep: float | None = None, out: str | Path | None = None) -> dict:
        """Verdict DSO6 sur le TEST : couverture de l'intervalle a 80 % et correlation
        (Spearman) entre |erreur| et sigma calibre. La calibration vient de la validation."""
        src = calibration if calibration is not None else self.cfg.inference.reliability
        if src is None:
            raise ValueError("DSO6 : aucune calibration (--calibration ou "
                             "inference.reliability)")
        cal = Calibration.load(src)
        acc = DSO6Accumulator(self.geom.nz)
        n = self._reliability_loop(
            "test", n_instants, keep,
            lambda k, c, e, sg: acc.add(c, k, e, sg * np.float32(cal.factor(c, k))))
        rc = self.cfg.reliability
        summary = {"produit_par": "pcfno3d", "modele": self.wf.checkpoint_path.name,
                   "jeu": "test", "instants_synoptiques": n,
                   "capteurs_visibles": self.cfg.mask.keep_eval if keep is None else keep,
                   "tta": self.tta, "uncertainty": cal.uncertainty_method(rc.members),
                   "calibration_sha256": cal.sha256, "keep_membres": rc.keep,
                   "graine": rc.seed, **acc.summary(self.geom, rc.coverage_target)}
        tag = cal.sha256[:8] if cal.sha256 else "non_calibre"
        path = Path(out) if out else self.out_dir / f"dso6_{tag}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.verbose:
            g = summary["global"]
            print(f"DSO6 ({n} instants de test, calibration {tag})\n  couverture 80 % : "
                  + "  ".join(f"{c} {g[f'couverture80_{c}']:.3f}" for c in COMPOSANTES)
                  + "\n  Spearman |erreur|, sigma : "
                  + "  ".join(f"{c} {g[f'spearman_{c}']:.3f}" for c in COMPOSANTES)
                  + f"\n  -> {'CONFORME' if summary['dso6_conforme'] else 'NON CONFORME'}"
                  f"\nresultats -> {path}")
        return {**summary, "fichier": str(path)}

    def _verdicts(self, levels, levels_ep, res, res_ep, res_all, robust, timing) -> dict:
        cfg = self.cfg
        # tolerance OMM du niveau : sert au DSO1 revise propose (voir DSOCfg)
        levels["seuil_omm"] = np.maximum(cfg.dso.omm_floor,
                                         cfg.dso.omm_rel * levels["vitesse_moyenne"])
        ev = levels[levels["couverture"] >= cfg.dso.min_level_cover]
        dso = cfg.dso

        # RMSE global sur tous les noeuds 3D : moyenne quadratique ponderee par la couverture
        glob = {c: float(np.sqrt((ev[f"rmse_{c}"] ** 2 * ev["couverture"]).sum()
                                 / ev["couverture"].sum())) for c in "uv"}
        r25 = robust[robust["capteurs_visibles"] == 0.25]
        rmse25 = float(max(r25["rmse_u_max"].iloc[0], r25["rmse_v_max"].iloc[0])) \
            if len(r25) else float("nan")
        rough_mean = float(res["roughness_ratio"].mean())

        return {
            "produit_par": "pcfno3d",
            "modele": self.wf.checkpoint_path.name,
            "epoque": (self.wf.epoch + 1) if self.wf.epoch is not None else None,
            "instants_synoptiques": int(len(res)),
            "instants_episodes": int(len(res_ep)),
            "capteurs_visibles": cfg.mask.keep_eval,
            "tta": self.tta,
            "tta_keep": self.tta_keep if self.tta > 1 else None,
            "niveaux_evalues": [int(z) for z in ev["z_m"]],
            "par_niveau_synoptique": levels.to_dict(orient="records"),
            "par_niveau_episodes": levels_ep.to_dict(orient="records") if len(levels_ep) else [],
            "dso1_rmse": bool((ev[["rmse_u", "rmse_v"]].max(axis=1) <= dso.rmse).all()),
            "dso1_r2": bool((ev[["r2_u", "r2_v"]].min(axis=1) >= dso.r2).all()),
            "dso1_r2_uvw": bool((ev[["r2_u", "r2_v", "r2_w"]].min(axis=1) >= dso.r2_all).all()),
            "dso1_rmse_global_u": glob["u"],
            "dso1_rmse_global_v": glob["v"],
            "dso1_rmse_global_conforme": bool(max(glob.values()) <= dso.rmse),
            # DSO1 revise PROPOSE : RMSE <= tolerance OMM du niveau (le DSO1 officiel reste
            # `dso1_rmse`, juge au seuil fixe `dso.rmse`)
            "critere_omm": bool((ev[["rmse_u", "rmse_v"]].max(axis=1) <= ev["seuil_omm"]).all()),
            "dso2_rms_moyen": float(res["mass_rms"].mean()),
            "dso2_rms_p95": float(res["mass_rms"].quantile(0.95)),
            "dso2_rms_max": float(res["mass_rms"].max()),
            "dso2_rms_tous_instants": float(res_all["mass_rms"].mean()),
            "dso2_conforme": bool(res["mass_rms"].mean() <= dso.mass_rms),
            "dso3_rugosite": rough_mean,
            "dso3_rugosite_tous_instants": float(res_all["roughness_ratio"].mean()),
            "dso3_conforme": dso3_conforme(rough_mean, dso.roughness_range),
            "dso4_inference_s": timing["mean_s"],
            "dso4_conforme": bool(timing["conforme"]),
            "dso4_fiabilite_incluse": self.wf.reliability is not None,
            "dso4_inference_s_sans_fiabilite": timing["sans_fiabilite"]["mean_s"]
            if "sans_fiabilite" in timing else None,
            "dso5_rmse_25pct": rmse25,
            "dso5_conforme": bool(rmse25 <= dso.rmse_25pct),
            "mae_direction_deg": float(res_all["mae_dir_deg"].mean()),
            "robustesse": robust.to_dict(orient="records"),
        }


def format_report(s: dict, levels: pd.DataFrame, levels_ep: pd.DataFrame,
                  robust: pd.DataFrame, cfg: Config) -> str:
    """Le tableau que le notebook imprimait, en chaine (reutilisable dans un rapport)."""
    def ok(b: bool) -> str:
        return "CONFORME" if b else "NON CONFORME"

    dso, cover = cfg.dso, cfg.dso.min_level_cover
    L = ["=" * 78,
         f"TEST SYNOPTIQUE (DSO1) : {s['instants_synoptiques']} instants | "
         f"capteurs visibles : {s['capteurs_visibles']:.0%}"
         + (f" | moyenne de {s['tta']} passes (garde {s['tta_keep']:.0%})"
            if s.get("tta", 1) > 1 else " | une seule passe"),
         " Niveau | couverture | RMSE u  RMSE v  RMSE w |  R2 u    R2 v    R2 w"]
    for _, r in levels.iterrows():
        flag = "" if r["couverture"] >= cover else "  (exclu : couverture < 5 %)"
        L.append(f" {int(r['z_m']):5d} m |   {r['couverture']:5.1%}   | {r['rmse_u']:.3f}   "
                 f"{r['rmse_v']:.3f}   {r['rmse_w']:.3f} | {r['r2_u']:.4f}  {r['r2_v']:.4f}  "
                 f"{r['r2_w']:.4f}{flag}")
    L += ["-" * 78,
          f"DSO1  RMSE u, v <= {dso.rmse} m/s a chaque niveau   -> {ok(s['dso1_rmse'])}",
          f"      R2 u, v >= {dso.r2} a chaque niveau          -> {ok(s['dso1_r2'])}",
          f"      R2 u, v, w >= {dso.r2_all} a chaque niveau       -> {ok(s['dso1_r2_uvw'])}",
          f"      RMSE GLOBAL u = {s['dso1_rmse_global_u']:.3f}  v = {s['dso1_rmse_global_v']:.3f}"
          f" m/s (seuil {dso.rmse}) -> {ok(s['dso1_rmse_global_conforme'])}",
          f"      (info) critere OMM : RMSE <= max({dso.omm_floor} ; "
          f"{dso.omm_rel:.0%} de la vitesse) -> {ok(s['critere_omm'])}"
          "  [DSO1 revise PROPOSE, a valider avec Tukhnanutha]",
          f"DSO2  RMS residu moyen = {s['dso2_rms_moyen']:.2e} (seuil {dso.mass_rms:.0e}) -> "
          f"{ok(s['dso2_conforme'])} | p95 = {s['dso2_rms_p95']:.2e} | "
          f"max = {s['dso2_rms_max']:.2e}",
          f"DSO3  rugosite prediction / verite = {s['dso3_rugosite']:.3f} "
          f"(dans [{dso.roughness_range[0]} ; {dso.roughness_range[1]}]) "
          f"-> {ok(s['dso3_conforme'])}",
          f"DSO4  inference = {s['dso4_inference_s'] * 1000:.0f} ms (seuil {dso.time_s} s)"
          f"      -> {ok(s['dso4_conforme'])}",
          f"DSO5  RMSE max a 25 % de capteurs = {s['dso5_rmse_25pct']:.3f} m/s "
          f"(seuil {dso.rmse_25pct}) -> {ok(s['dso5_conforme'])}",
          f"Direction : MAE = {s['mae_direction_deg']:.2f} deg (vents >= 1 m/s)",
          "=" * 78]
    if len(levels_ep):
        L += [f"EPISODES RAPIDES (rafales, front, brise) : {s['instants_episodes']} instants "
              f"— test de stress, hors DSO1",
              " Niveau | RMSE u  RMSE v |  R2 u    R2 v"]
        for _, r in levels_ep[levels_ep["couverture"] >= cover].iterrows():
            L.append(f" {int(r['z_m']):5d} m | {r['rmse_u']:.3f}   {r['rmse_v']:.3f} | "
                     f"{r['r2_u']:.4f}  {r['r2_v']:.4f}")
        L.append("=" * 78)
    L += ["Robustesse (instants synoptiques repartis sur tous les jours de test) :",
          robust.round(4).to_string(index=False), "=" * 78]
    return "\n".join(L)


def evaluate(cfg: Config, checkpoint: str | Path | None = None, device: str | None = None,
             verbose: bool = True, tta: int | None = None,
             tta_keep: float | None = None) -> dict:
    return Evaluator(cfg, checkpoint=checkpoint, device=device, verbose=verbose,
                     tta=tta, tta_keep=tta_keep).run()
