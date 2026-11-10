"""Phase 1 nationale, diagnostic 6 : forme de l'erreur sur v (modele national v2 - verite d'entrainement).

Regle : configs/national/regle_phase1_complements.json (empreinte verifiee avant tout calcul).
Pour chacun des instants synoptiques du TEST 2022 (ceux du diagnostic 2), a 300 et 800 m au-dessus du sol,
ajustement par moindres carres sur les colonnes valides de e = a + b y + c x (y, x en km depuis le centre),
pour v et pour u (comparaison). Part de l'energie de l'erreur expliquee par le plan et par la moyenne seule,
puis correlations de Spearman (sur les instants) de a, b, c avec l'heure UTC, le mois, la vitesse du vent
geostrophique ERA5 (era5_ug, era5_vg de index.csv) et le sinus / cosinus de sa direction.
Aucun entrainement, aucun reglage.

    python scripts/diagnostic_forme_erreur_v.py -c national_eval_v2
Sortie : <run_dir>/diagnostic_forme_erreur_v/ (par_instant.csv, parts.csv, correlations.csv, verdict.json, .md)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
REGLE = REPO / "configs" / "national" / "regle_phase1_complements.json"
HAUTEURS = (300.0, 800.0)


def verifier_regle() -> str:
    h = hashlib.sha256(REGLE.read_bytes()).hexdigest()
    attendu = (REGLE.parent / (REGLE.name + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"regle modifiee depuis son hachage : {h} != {attendu}")
    return h


def ajuster_plan(e: np.ndarray, yk: np.ndarray, xk: np.ndarray) -> dict:
    """e = a + b y + c x sur les points finis ; energies totale, residuelle (plan), residuelle (moyenne)."""
    m = np.isfinite(e)
    ev, A = e[m], np.stack([np.ones(m.sum()), yk[m], xk[m]], 1)
    coef, *_ = np.linalg.lstsq(A, ev, rcond=None)
    res = ev - A @ coef
    return {"a": float(coef[0]), "b_par_100km": float(coef[1] * 100), "c_par_100km": float(coef[2] * 100),
            "ss_tot": float((ev ** 2).sum()), "ss_plan": float((res ** 2).sum()),
            "ss_moy": float(((ev - ev.mean()) ** 2).sum()), "n": int(m.sum()), "rmse": float(np.sqrt((ev ** 2).mean()))}


def main() -> int:
    import torch
    from scipy.stats import spearmanr

    from pcfno3d.config import Config
    from pcfno3d.dataset import eval_loader, is_synoptic
    from pcfno3d.evaluate import Evaluator

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--max-instants", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    h_regle = verifier_regle()

    cfg = Config.load(a.config)
    ev = Evaluator(cfg, device=a.device, verbose=False)
    geom, splits = ev.geom, ev.splits
    out = cfg.paths.run_dir / "diagnostic_forme_erreur_v"
    out.mkdir(parents=True, exist_ok=True)

    te_s = splits.test[0]
    idx = np.nonzero([is_synoptic(splits.stamp_of[s]) for s in te_s])[0]
    if a.max_instants:
        idx = idx[:a.max_instants]
    loader, stems = eval_loader(cfg, geom, splits, idx, 2000)
    stems = [str(s) for s in stems]
    index = pd.read_csv(cfg.paths.index_csv).astype({"stem": str}).set_index("stem")

    z = np.asarray(geom.z_levels, float)
    zt = np.asarray(geom.zt, float)
    valid = np.asarray(geom.valid, bool)
    ny, nx = zt.shape
    yk = (np.arange(ny)[:, None] - (ny - 1) / 2) * geom.dy / 1000.0 + 0 * zt
    xk = (np.arange(nx)[None, :] - (nx - 1) / 2) * geom.dx / 1000.0 + 0 * zt

    def a_hauteur(c3, h):
        zz = zt + h
        i = np.clip(np.searchsorted(z, zz) - 1, 0, len(z) - 2)
        t = np.clip((zz - z[i]) / (z[i + 1] - z[i]), 0, 1)
        lo = np.take_along_axis(c3, i[None], 0)[0]
        hi = np.take_along_axis(c3, (i + 1)[None], 0)[0]
        ok = (zz >= z[0]) & (zz <= z[-1]) & np.take_along_axis(valid, i[None], 0)[0] \
            & np.take_along_axis(valid, (i + 1)[None], 0)[0]
        return np.where(ok, (1 - t) * lo + t * hi, np.nan)

    lignes = []
    with torch.no_grad():
        for s_raw, sensor, tfeat, y, j in loader:
            s_raw, sensor, tfeat, y = (t.to(geom.device) for t in (s_raw, sensor, tfeat, y))
            stem = stems[int(j)]
            u, v, _ = ev.predire(s_raw, sensor, tfeat, y, int(j), stem)
            P = {"u": u[0].float().cpu().numpy(), "v": v[0].float().cpu().numpy()}
            Y = {"u": y[0, 0].float().cpu().numpy(), "v": y[0, 1].float().cpu().numpy()}
            r = index.loc[stem]
            ts = pd.Timestamp(str(r["timestamp"]))
            ug, vg = float(r["era5_ug"]), float(r["era5_vg"])
            base = {"stem": stem, "instant_utc": str(ts), "heure_utc": ts.hour, "mois": ts.month,
                    "vg_vitesse": float(np.hypot(ug, vg)),
                    # direction meteorologique (d'ou vient le vent) : sin = -ug / |g|, cos = -vg / |g|
                    "vg_dir_sin": float(np.sin(np.arctan2(-ug, -vg))), "vg_dir_cos": float(np.cos(np.arctan2(-ug, -vg)))}
            for h in HAUTEURS:
                for c in "uv":
                    f = ajuster_plan(a_hauteur(P[c] - Y[c], h), yk, xk)
                    lignes.append({**base, "hauteur_sol_m": int(h), "composante": c, **f})
            print(stem, flush=True)

    t = pd.DataFrame(lignes)
    t.to_csv(out / "par_instant.csv", index=False)
    parts, cor = [], []
    for (h, c), g in t.groupby(["hauteur_sol_m", "composante"]):
        par_inst_plan = 1 - g.ss_plan / g.ss_tot
        parts.append({"hauteur_sol_m": h, "composante": c, "instants": len(g),
                      "part_plan_ensemble": float(1 - g.ss_plan.sum() / g.ss_tot.sum()),
                      "part_moyenne_ensemble": float(1 - g.ss_moy.sum() / g.ss_tot.sum()),
                      "part_plan_mediane_instant": float(par_inst_plan.median()),
                      "rmse": float(np.sqrt(g.ss_tot.sum() / g.n.sum())),
                      "a_moyen": float(g.a.mean()), "a_ecart_type": float(g.a.std())})
        for coef in ("a", "b_par_100km", "c_par_100km"):
            for var in ("heure_utc", "mois", "vg_vitesse", "vg_dir_sin", "vg_dir_cos"):
                rho, p = spearmanr(g[coef], g[var])
                cor.append({"hauteur_sol_m": h, "composante": c, "coefficient": coef, "variable": var,
                            "rho": float(rho), "p": float(p), "signalee": bool(abs(rho) >= 0.3)})
    P_ = pd.DataFrame(parts)
    K = pd.DataFrame(cor)
    P_.to_csv(out / "parts.csv", index=False)
    K.to_csv(out / "correlations.csv", index=False)
    pv = P_[P_.composante == "v"]
    grande = bool((pv.part_plan_ensemble >= 0.5).all() and len(pv) == len(HAUTEURS))
    verdict = {"regle": "configs/national/regle_phase1_complements.json", "sha256_regle": h_regle,
               "instants": int(t.stem.nunique()),
               "erreur_v_correction_grande_echelle_par_instant": grande,
               "correlations_signalees": K[K.signalee & (K.composante == "v")].round(3).to_dict("records")}
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    L = ["# Phase 1, diagnostic 6 : forme de l'erreur sur v", "",
         f"Regle sha256 `{h_regle}` (verifiee). {verdict['instants']} instants synoptiques du test 2022.", "",
         P_.round(3).to_string(index=False), "",
         "Correlations de Spearman (|rho| >= 0,3 signalees) :", "",
         K[K.signalee].round(3).to_string(index=False) if K.signalee.any() else "aucune", "",
         f"**Erreur sur v = correction de grande echelle par instant** : {'oui' if grande else 'non'}"]
    (out / "diagnostic_forme_erreur_v.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
