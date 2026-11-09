"""Cartes de l'ecart verite generee - ERA5 (modele national) et biais du generateur aux aeroports.

Sans modele ni entrainement. Sur les instants synoptiques du TEST :
  - cartes du RMSE et du biais de v (et de u, pour comparaison) entre la verite generee et ERA5
    interpole, a 300 m et 800 m AU-DESSUS DU SOL (interpolation lineaire dans chaque colonne) ;
  - biais du generateur aux 12 aeroports (stations_paires.csv de la validation reelle, vent a
    10 m) : moyenne de generateur - ERA5 et de generateur - mesure, pour u, v et la vitesse.

    python scripts/carte_v_generateur.py -c national_1km_colab
Sorties : <run_dir>/diagnostic_v/carte_v_<h>m.png, cartes.npz, biais_aeroports.csv / .md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]
HAUTEURS = (300.0, 800.0)


def a_hauteur(champ3, z_niv, zt, valid, h):
    """(nz, ny, nx) aux niveaux ASL -> a h m au-dessus du sol ; NaN si la colonne n'y monte pas."""
    z = zt + h
    i = np.clip(np.searchsorted(z_niv, z) - 1, 0, len(z_niv) - 2)
    t = np.clip((z - z_niv[i]) / (z_niv[i + 1] - z_niv[i]), 0.0, 1.0)
    lo = np.take_along_axis(champ3, i[None], 0)[0]
    hi = np.take_along_axis(champ3, (i + 1)[None], 0)[0]
    ok = (z >= z_niv[0]) & (z <= z_niv[-1]) & np.take_along_axis(valid, i[None], 0)[0] \
        & np.take_along_axis(valid, (i + 1)[None], 0)[0]
    return np.where(ok, (1 - t) * lo + t * hi, np.nan)


def biais_aeroports(paires: pd.DataFrame) -> pd.DataFrame:
    p = paires.dropna(subset=["u_obs", "v_obs", "u_era5", "v_era5", "u_generateur", "v_generateur"])
    lignes = []
    for s, g in p.groupby("station"):
        vg, ve, vo = (np.hypot(g[f"u_{x}"], g[f"v_{x}"]) for x in ("generateur", "era5", "obs"))
        lignes.append({
            "station": s, "n": len(g),
            "gen_moins_era5_u": (g.u_generateur - g.u_era5).mean(),
            "gen_moins_era5_v": (g.v_generateur - g.v_era5).mean(),
            "gen_moins_era5_vitesse": (vg - ve).mean(),
            "gen_moins_obs_u": (g.u_generateur - g.u_obs).mean(),
            "gen_moins_obs_v": (g.v_generateur - g.v_obs).mean(),
            "gen_moins_obs_vitesse": (vg - vo).mean(),
            "era5_moins_obs_vitesse": (ve - vo).mean(),
            "rmse_v_gen_obs": float(np.sqrt(((g.v_generateur - g.v_obs) ** 2).mean())),
            "rmse_v_era5_obs": float(np.sqrt(((g.v_era5 - g.v_obs) ** 2).mean()))})
    return pd.DataFrame(lignes)


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
    from pcfno3d.stations_reelles import cellules_codes, noms_stations

    cfg = Config.load(a.config)
    geom = Geometry(cfg, device="cpu")
    index = load_index(cfg)
    splits = make_splits(cfg, index)
    stems = [s for s in splits.test[0] if is_synoptic(splits.stamp_of[s])]
    if a.max_instants:
        stems = stems[:a.max_instants]
    z_niv = np.asarray(geom.z_levels, float)
    zt = np.asarray(geom.zt, float)
    valid = np.asarray(geom.valid, bool)
    n_c = geom.forcage.n_cellules
    ny, nx = zt.shape
    acc = {(h, c): np.zeros((3, ny, nx)) for h in HAUTEURS for c in "uv"}   # n, somme, somme2
    for i, stem in enumerate(stems):
        Y = np.load(cfg.paths.cache_dir / f"{stem}_Y.npy").astype(np.float64)
        vec = geom.forcage.vecteur(horodatage(geom, stem))
        E = geom.interp_era5.dense_np(vec[len(vec) - n_c:]).astype(np.float64)
        for h in HAUTEURS:
            for ci, c in enumerate("uv"):
                d = a_hauteur(Y[ci], z_niv, zt, valid, h) - a_hauteur(E[ci], z_niv, zt, valid, h)
                m = np.isfinite(d)
                acc[(h, c)][0] += m
                acc[(h, c)][1] += np.where(m, d, 0.0)
                acc[(h, c)][2] += np.where(m, d * d, 0.0)
        if (i + 1) % 30 == 0:
            print(f"{i + 1}/{len(stems)} instants", flush=True)

    out = cfg.paths.run_dir / "diagnostic_v"
    out.mkdir(parents=True, exist_ok=True)
    cartes = {}
    for (h, c), s in acc.items():
        n = np.where(s[0] > 0, s[0], np.nan)
        cartes[f"rmse_{c}_{int(h)}m"] = np.sqrt(s[2] / n)
        cartes[f"biais_{c}_{int(h)}m"] = s[1] / n
    np.savez_compressed(out / "cartes.npz", lat=geom.interp_era5.lat, lon=geom.interp_era5.lon, **cartes)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:                                    # aeroports (stations.json du jeu) : facultatif
        cel, noms = cellules_codes(geom), noms_stations(geom)
    except FileNotFoundError:
        cel, noms = {}, {}
    lat, lon = geom.interp_era5.lat, geom.interp_era5.lon
    ext = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    for h in HAUTEURS:
        fig, ax = plt.subplots(1, 3, figsize=(15, 9), constrained_layout=True)
        for a_, (cle, titre, cmap, vmin, vmax) in zip(ax, (
                (f"rmse_v_{int(h)}m", f"RMSE v, vérité − ERA5 ({int(h)} m sol)", "magma", 0, 4),
                (f"rmse_u_{int(h)}m", f"RMSE u, vérité − ERA5 ({int(h)} m sol)", "magma", 0, 4),
                (f"biais_v_{int(h)}m", f"Biais v, vérité − ERA5 ({int(h)} m sol)", "RdBu_r", -2, 2))):
            im = a_.imshow(cartes[cle], origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect=1 / np.cos(np.radians(lat.mean())))
            for code, (iy, ix) in cel.items():
                a_.plot(lon[iy, ix], lat[iy, ix], "k^", ms=5)
                a_.annotate(noms.get(code, code), (lon[iy, ix], lat[iy, ix]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
            a_.set_title(titre, fontsize=10)
            a_.set_xlabel("longitude (°E)")
            fig.colorbar(im, ax=a_, shrink=0.6, label="m/s")
        ax[0].set_ylabel("latitude (°N)")
        fig.suptitle(f"Générateur national : écart à ERA5 à {int(h)} m au-dessus du sol "
                     f"({len(stems)} instants synoptiques de test 2022)")
        fig.savefig(out / f"carte_v_{int(h)}m.png", dpi=130)
        plt.close(fig)

    L = [f"# Écart vérité générée - ERA5 ({len(stems)} instants de test)", "",
         "| hauteur sol | RMSE u (moyenne domaine) | RMSE v | biais v moyen |", "|---|---|---|---|"]
    for h in HAUTEURS:
        ru, rv, bv = (cartes[f"{k}_{int(h)}m"] for k in ("rmse_u", "rmse_v", "biais_v"))
        L.append(f"| {int(h)} m | {np.sqrt(np.nanmean(ru ** 2)):.3f} | {np.sqrt(np.nanmean(rv ** 2)):.3f} "
                 f"| {np.nanmean(bv):+.3f} |")
    sp = cfg.paths.run_dir / "validation_reelle" / "test" / "stations_paires.csv"
    if sp.exists():
        b = biais_aeroports(pd.read_csv(sp))
        b.to_csv(out / "biais_aeroports.csv", index=False)
        L += ["", "## Biais du générateur aux 12 aéroports (10 m, test 2022)", "",
              b.round(3).to_string(index=False)]
    (out / "cartes_v.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"resultats -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
