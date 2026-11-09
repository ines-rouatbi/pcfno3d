"""Modele national : le forcage SCALAIRE (moyenne du domaine) explique-t-il l'erreur sur v ?

Diagnostic sans entraînement. Le modele recoit 7 scalaires de forcage par instant
(scripts/forcage_era5.py : ug, vg a 850 hPa, cisaillement 850-700 hPa, rotation du vent 10 m -
850 hPa, stabilite, contraste terre-mer), MOYENNES SUR TOUT LE DOMAINE (828 x 388 km), en plus du
vent ERA5 aux noeuds (16 niveaux). Sur les instants synoptiques du TEST :
  - carte du RMSE de v (et u) du modele contre la verite a 800 m au-dessus du sol ;
  - par region (nord > 35,5 N, centre 33,5-35,5 N, sud < 33,5 N) et par instant : ecart entre le
    cisaillement et la rotation LOCAUX d'ERA5 (moyenne des cellules de la region) et les valeurs
    du domaine (meme calcul sur toutes les cellules : ce que voit le scalaire) ;
  - correlation, sur les couples (instant, region), entre cet ecart et le RMSE de v du modele.
Cisaillement et rotation sont pris sur le profil ERA5 par cellule (forcage_era5.npz, hauteurs
au-dessus du sol) : cisaillement entre 1000 et 2500 m, rotation entre 10 m et 1500 m.

    python scripts/diagnostic_forcage_regional.py -c national_eval_v2 [--max-instants N]
Sortie : <run_dir>/diagnostic_forcage/ (carte_rmse_800m.png, regions.csv, resume.json, .md)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
REGIONS = (("nord", 35.5, 90.0), ("centre", 33.5, 35.5), ("sud", -90.0, 33.5))
H_CARTE = 800.0


def profil_local(ch, i_t: int, h_bas: float, h_haut: float, h_rot: float):
    """Par cellule ERA5 : cisaillement (du, dv)/dz entre h_bas et h_haut, rotation (deg) 10 m -> h_rot."""
    hs = np.asarray(ch["hauteurs_sol"], float)

    def a(h):
        k = int(np.clip(np.searchsorted(hs, h) - 1, 0, len(hs) - 2))
        t = (h - hs[k]) / (hs[k + 1] - hs[k])
        return ((1 - t) * ch["u"][i_t, k] + t * ch["u"][i_t, k + 1],
                (1 - t) * ch["v"][i_t, k] + t * ch["v"][i_t, k + 1])
    ub, vb = a(h_bas)
    uh, vh = a(h_haut)
    u0, v0 = a(10.0)
    ur, vr = a(h_rot)
    cis = np.hypot(uh - ub, vh - vb) / (h_haut - h_bas)
    rot = (np.degrees(np.arctan2(v0, u0)) - np.degrees(np.arctan2(vr, ur)) + 180.0) % 360.0 - 180.0
    return cis, rot


def main() -> int:
    import torch

    from pcfno3d.config import Config
    from pcfno3d.dataset import eval_loader, is_synoptic
    from pcfno3d.evaluate import Evaluator
    from pcfno3d.forcage import horodatage

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--max-instants", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--hauteur", type=float, default=H_CARTE, help="hauteur au-dessus du sol (m)")
    a = ap.parse_args()
    h_carte = a.hauteur

    cfg = Config.load(a.config)
    ev = Evaluator(cfg, device=a.device, verbose=False)
    geom, splits = ev.geom, ev.splits
    te_s = splits.test[0]
    idx = np.nonzero([is_synoptic(splits.stamp_of[s]) for s in te_s])[0]
    if a.max_instants:
        idx = idx[:a.max_instants]
    loader, stems = eval_loader(cfg, geom, splits, idx, 2000)
    stems = [str(s) for s in stems]
    with np.load(Path(cfg.paths.forcage_era5).with_suffix(".npz")) as d:
        ch = {k: d[k] for k in d.files}
    lat_e, lon_e = np.asarray(ch["lat"]), np.asarray(ch["lon"])
    LAT_E = np.broadcast_to(lat_e[:, None], (len(lat_e), len(lon_e)))
    # cellules ERA5 dans le domaine du modele
    lat_n, lon_n = geom.interp_era5.lat, geom.interp_era5.lon
    dans = ((LAT_E >= lat_n.min()) & (LAT_E <= lat_n.max())
            & (np.broadcast_to(lon_e[None], LAT_E.shape) >= lon_n.min())
            & (np.broadcast_to(lon_e[None], LAT_E.shape) <= lon_n.max()))

    z = np.asarray(geom.z_levels, float)
    zt = np.asarray(geom.zt, float)
    valid = np.asarray(geom.valid, bool)

    def a_hauteur(c3, h):
        zz = zt + h
        i = np.clip(np.searchsorted(z, zz) - 1, 0, len(z) - 2)
        t = np.clip((zz - z[i]) / (z[i + 1] - z[i]), 0, 1)
        lo = np.take_along_axis(c3, i[None], 0)[0]
        hi = np.take_along_axis(c3, (i + 1)[None], 0)[0]
        ok = (zz >= z[0]) & (zz <= z[-1]) & np.take_along_axis(valid, i[None], 0)[0] \
            & np.take_along_axis(valid, (i + 1)[None], 0)[0]
        return np.where(ok, (1 - t) * lo + t * hi, np.nan)

    reg_noeud = {nom: (lat_n >= lo) & (lat_n < hi) for nom, lo, hi in REGIONS}
    reg_cell = {nom: dans & (LAT_E >= lo) & (LAT_E < hi) for nom, lo, hi in REGIONS}
    carte = np.zeros((3, *zt.shape))                  # n, se_u, se_v
    lignes = []
    t_ch = np.asarray(ch["temps"], np.int64)
    with torch.no_grad():
        for s_raw, sensor, tfeat, y, j in loader:
            s_raw, sensor, tfeat, y = (t.to(geom.device) for t in (s_raw, sensor, tfeat, y))
            stem = stems[int(j)]
            u, v, _ = ev.predire(s_raw, sensor, tfeat, y, int(j), stem)
            eu = a_hauteur(u[0].float().cpu().numpy() - y[0, 0].float().cpu().numpy(), h_carte)
            evv = a_hauteur(v[0].float().cpu().numpy() - y[0, 1].float().cpu().numpy(), h_carte)
            m = np.isfinite(evv)
            carte[0] += m
            carte[1] += np.where(m, eu ** 2, 0)
            carte[2] += np.where(m, evv ** 2, 0)
            ts = pd.Timestamp(horodatage(geom, stem)).value // 10 ** 9
            i_t = int(np.argmin(np.abs(t_ch - ts)))
            cis, rot = profil_local(ch, i_t, 1000.0, 2500.0, 1500.0)
            cis_dom, rot_dom = float(cis[dans].mean()), float(rot[dans].mean())
            for nom in reg_noeud:
                mm = reg_noeud[nom] & m
                rc = reg_cell[nom]
                if not mm.any() or not rc.any():
                    continue
                lignes.append({"stem": stem, "region": nom,
                               "rmse_v_800": float(np.sqrt((evv[mm] ** 2).mean())),
                               "rmse_u_800": float(np.sqrt((eu[mm] ** 2).mean())),
                               "ecart_cisaillement": abs(float(cis[rc].mean()) - cis_dom),
                               "ecart_rotation_deg": abs(float(rot[rc].mean()) - rot_dom)})
    t = pd.DataFrame(lignes, columns=["stem", "region", "rmse_v_800", "rmse_u_800",
                                      "ecart_cisaillement", "ecart_rotation_deg"])
    out = cfg.paths.run_dir / "diagnostic_forcage"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "regions.csv", index=False)
    par_reg = t.groupby("region").agg(rmse_v_800=("rmse_v_800", lambda x: float(np.sqrt((x ** 2).mean()))),
                                      rmse_u_800=("rmse_u_800", lambda x: float(np.sqrt((x ** 2).mean()))),
                                      ecart_cisaillement=("ecart_cisaillement", "mean"),
                                      ecart_rotation_deg=("ecart_rotation_deg", "mean")).reindex([r[0] for r in REGIONS])
    from scipy.stats import spearmanr
    corr = {}
    for col in ("ecart_cisaillement", "ecart_rotation_deg"):
        if len(t) < 3:
            corr[col] = None
            continue
        r_all = spearmanr(t[col], t["rmse_v_800"]).correlation
        # dans chaque region (enleve l'effet « region »), puis moyenne
        r_in = np.nanmean([spearmanr(g[col], g["rmse_v_800"]).correlation for _, g in t.groupby("region")
                           if len(g) >= 3] or [np.nan])
        corr[col] = {"spearman_global": float(r_all), "spearman_dans_region_moyen": float(r_in)}
    resume = {"instants": len(stems), "hauteur_sol_m": h_carte, "par_region": par_reg.round(4).to_dict(orient="index"),
              "correlations_avec_rmse_v": corr,
              "note": "forcage scalaire = moyenne du domaine (scripts/forcage_era5.py) ; le vent ERA5 aux noeuds "
                      "(16 niveaux) est aussi en entree"}
    (out / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = np.where(carte[0] > 0, carte[0], np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(11, 9), constrained_layout=True)
    ext = [float(lon_n.min()), float(lon_n.max()), float(lat_n.min()), float(lat_n.max())]
    for a_, (k, titre) in zip(ax, ((2, "RMSE v"), (1, "RMSE u"))):
        im = a_.imshow(np.sqrt(carte[k] / n), origin="lower", extent=ext, cmap="magma", vmin=0, vmax=2,
                       aspect=1 / np.cos(np.radians(lat_n.mean())))
        for _, lo, hi in REGIONS:
            for l in (lo, hi):
                if ext[2] < l < ext[3]:
                    a_.axhline(l, color="w", lw=0.6, ls="--")
        a_.set_title(f"{titre}, modèle − vérité, {int(h_carte)} m au-dessus du sol")
        a_.set_xlabel("longitude (°E)")
        fig.colorbar(im, ax=a_, shrink=0.6, label="m/s")
    ax[0].set_ylabel("latitude (°N)")
    fig.savefig(out / f"carte_rmse_{int(h_carte)}m.png", dpi=130)
    plt.close(fig)
    L = [f"Diagnostic forcage regional ({len(stems)} instants de test)", "",
         par_reg.round(4).to_string(), "", json.dumps(corr, indent=1)]
    (out / "diagnostic_forcage.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"resultats -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
