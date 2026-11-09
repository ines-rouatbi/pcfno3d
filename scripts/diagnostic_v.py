"""Diagnostic de v (modele national) : v est 1,4 a 1,5 fois pire que u au-dessus de 200 m.

Sans modele ni entrainement. Sur les instants synoptiques du TEST, niveau par niveau :
  - ERA5 seule (vent ERA5 aux noeuds, l'entree du modele residuel) contre la verite : RMSE de u
    et de v, biais, correlation ; si v est aussi mauvais avec ERA5 seule, le defaut vient des
    donnees (verite ou forcage), pas du modele ;
  - correlations croisees (ERA5 u contre verite v, ERA5 v contre verite u) : un echange ou un
    signe inverse de u et v les ferait dominer ;
  - ecart-type de la verite et du detail (verite - ERA5) pour u et v, et anisotropie du detail :
    energie de ses variations le long de x (est-ouest) et de y (nord-sud) ;
  - statistiques de normalisation (norm_stats.json) de u et v.

    python scripts/diagnostic_v.py -c national_1km_colab [--max-instants N]
Sortie : <run_dir>/diagnostic_v/diagnostic_v.csv et .md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--max-instants", type=int, default=None)
    a = ap.parse_args()

    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.dataset import is_synoptic, make_splits
    from pcfno3d.forcage import horodatage
    from pcfno3d.geometry import Geometry

    cfg = Config.load(a.config)
    geom = Geometry(cfg, device="cpu")
    if geom.interp_era5 is None:
        raise SystemExit("paths.forcage_era5 requis")
    index = load_index(cfg)
    splits = make_splits(cfg, index)
    stems = [s for s in splits.test[0] if is_synoptic(splits.stamp_of[s])]
    if a.max_instants:
        stems = stems[:a.max_instants]
    nz = geom.nz
    valid = np.asarray(geom.valid, bool)
    n_c = geom.forcage.n_cellules
    # accumulateurs par niveau : pour u et v, sommes pour RMSE, biais, correlations, ecarts-types
    keys = ("n", "se_u", "se_v", "b_u", "b_v", "t_u", "t_v", "t2_u", "t2_v", "e_u", "e_v",
            "e2_u", "e2_v", "te_uu", "te_vv", "te_uv", "te_vu", "d2_u", "d2_v",
            "gx_du", "gy_du", "gx_dv", "gy_dv", "gx_tu", "gy_tu", "gx_tv", "gy_tv")
    acc = {k: np.zeros(nz) for k in keys}
    for i, stem in enumerate(stems):
        Y = np.load(cfg.paths.cache_dir / f"{stem}_Y.npy").astype(np.float64)
        vec = geom.forcage.vecteur(horodatage(geom, stem))
        E = geom.interp_era5.dense_np(vec[len(vec) - n_c:]).astype(np.float64)   # (2, nz, ny, nx)
        for k in range(nz):
            m = valid[k] & np.isfinite(Y[0, k]) & np.isfinite(Y[1, k])
            if not m.any():
                continue
            tu, tv, eu, ev = Y[0, k][m], Y[1, k][m], E[0, k][m], E[1, k][m]
            acc["n"][k] += m.sum()
            acc["se_u"][k] += ((eu - tu) ** 2).sum(); acc["se_v"][k] += ((ev - tv) ** 2).sum()
            acc["b_u"][k] += (eu - tu).sum(); acc["b_v"][k] += (ev - tv).sum()
            acc["t_u"][k] += tu.sum(); acc["t_v"][k] += tv.sum()
            acc["t2_u"][k] += (tu ** 2).sum(); acc["t2_v"][k] += (tv ** 2).sum()
            acc["e_u"][k] += eu.sum(); acc["e_v"][k] += ev.sum()
            acc["e2_u"][k] += (eu ** 2).sum(); acc["e2_v"][k] += (ev ** 2).sum()
            acc["te_uu"][k] += (tu * eu).sum(); acc["te_vv"][k] += (tv * ev).sum()
            acc["te_uv"][k] += (tv * eu).sum(); acc["te_vu"][k] += (tu * ev).sum()
            # detail = verite - ERA5 ; variations le long de x et de y (paires valides)
            for c, nom in ((0, "u"), (1, "v")):
                d = np.where(valid[k], Y[c, k] - E[c, k], np.nan)
                t = np.where(valid[k], Y[c, k], np.nan)
                acc[f"d2_{nom}"][k] += np.nansum(d ** 2)
                acc[f"gx_d{nom}"][k] += np.nanmean(np.diff(d, axis=1) ** 2) if np.isfinite(np.diff(d, axis=1)).any() else 0
                acc[f"gy_d{nom}"][k] += np.nanmean(np.diff(d, axis=0) ** 2) if np.isfinite(np.diff(d, axis=0)).any() else 0
                acc[f"gx_t{nom}"][k] += np.nanmean(np.diff(t, axis=1) ** 2) if np.isfinite(np.diff(t, axis=1)).any() else 0
                acc[f"gy_t{nom}"][k] += np.nanmean(np.diff(t, axis=0) ** 2) if np.isfinite(np.diff(t, axis=0)).any() else 0
        if (i + 1) % 30 == 0:
            print(f"{i + 1}/{len(stems)} instants", flush=True)

    def corr(sab, sa, sb, sa2, sb2, n):
        cov = sab / n - (sa / n) * (sb / n)
        return cov / np.sqrt((sa2 / n - (sa / n) ** 2) * (sb2 / n - (sb / n) ** 2))

    n, ni = acc["n"], len(stems)
    r = pd.DataFrame({
        "z_m": [int(z) for z in geom.z_levels], "couverture": np.round(geom.cover, 3),
        "era5_rmse_u": np.sqrt(acc["se_u"] / n), "era5_rmse_v": np.sqrt(acc["se_v"] / n),
        "era5_biais_u": acc["b_u"] / n, "era5_biais_v": acc["b_v"] / n,
        "corr_u_u": corr(acc["te_uu"], acc["t_u"], acc["e_u"], acc["t2_u"], acc["e2_u"], n),
        "corr_v_v": corr(acc["te_vv"], acc["t_v"], acc["e_v"], acc["t2_v"], acc["e2_v"], n),
        "corr_vrai_v_era5_u": corr(acc["te_uv"], acc["t_v"], acc["e_u"], acc["t2_v"], acc["e2_u"], n),
        "corr_vrai_u_era5_v": corr(acc["te_vu"], acc["t_u"], acc["e_v"], acc["t2_u"], acc["e2_v"], n),
        "ect_verite_u": np.sqrt(acc["t2_u"] / n - (acc["t_u"] / n) ** 2),
        "ect_verite_v": np.sqrt(acc["t2_v"] / n - (acc["t_v"] / n) ** 2),
        "detail_rms_u": np.sqrt(acc["d2_u"] / n), "detail_rms_v": np.sqrt(acc["d2_v"] / n),
        # variations du detail d'un noeud au voisin, le long de x (est-ouest) et de y (nord-sud)
        "detail_dx_u": np.sqrt(acc["gx_du"] / ni), "detail_dy_u": np.sqrt(acc["gy_du"] / ni),
        "detail_dx_v": np.sqrt(acc["gx_dv"] / ni), "detail_dy_v": np.sqrt(acc["gy_dv"] / ni),
    })
    r["era5_rapport_v_sur_u"] = r["era5_rmse_v"] / r["era5_rmse_u"]
    out = cfg.paths.run_dir / "diagnostic_v"
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "diagnostic_v.csv", index=False)
    ns = cfg.paths.run_dir / "norm_stats.json"
    stats = json.loads(ns.read_text()) if ns.exists() else {}
    L = [f"# Diagnostic de v ({ni} instants synoptiques de test)", "",
         f"Normalisation (norm_stats.json) : y_mu u, v = {stats.get('y_mu', ['?', '?'])[:2]} ; "
         f"y_sd u, v = {stats.get('y_sd', ['?', '?'])[:2]}", "",
         r.round(3).to_string(index=False)]
    (out / "diagnostic_v.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
