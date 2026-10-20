"""v9 contre les VRAIS radiosondages (Tunis-Carthage, Tozeur) : profil vertical de 0 a 1 km.

Pour chaque ballon dont l'heure nominale est un instant de test du cache, reconstruit le
volume a partir des anemometres seuls, prend la colonne de la maille du site et la compare
au ballon, a la meme hauteur au-dessus du sol ; fait de meme pour la verite ERA5-Land + ERA5.

    python scripts/v9_radiosondages.py -c v9_colab --igra <dossier des radiosondages_*.csv>

Sortie dans run_dir/radiosondages/<split>/ : paires.csv (un point de ballon par ligne) et
bilan.csv / bilan.json (par site et tranche de 250 m : erreur vectorielle RMS et biais de
vitesse, modele et ERA5).
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
    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.inference import WindFieldModel
    from pcfno3d import radiosondages as R
    from pcfno3d.tunisie_fno3d import H_MAX, H_MIN

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="v9_colab")
    ap.add_argument("--igra", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    modele = WindFieldModel(cfg, checkpoint=args.checkpoint, verbose=False)
    geom = modele.geom
    meta = json.loads((cfg.paths.data_dir / "meta.json").read_text(encoding="utf-8"))
    points = pd.read_csv(cfg.paths.data_dir / "aneo_grid_points.csv")
    surf = points[points["point_id"] < meta["noeuds_3d"]["point_id_debut"]]
    zt = np.zeros((geom.ny, geom.nx))
    zt[surf["iy"], surf["ix"]] = surf["z_terrain_m"]
    z = np.asarray(geom.z_levels, dtype=np.float64)

    index = load_index(cfg)
    index = index[index["split"] == args.split]
    stems = {pd.Timestamp(t): s for t, s in zip(index["timestamp"], index["stem"])}

    paires = []
    for code, (nom, lat, lon) in R.SITES.items():
        f = Path(args.igra) / f"radiosondages_{code}.csv"
        if not f.exists():
            print(f"{nom} : {f.name} absent")
            continue
        obs = pd.read_csv(f)
        obs["temps"] = pd.to_datetime(obs["temps"]).dt.tz_localize(None)
        obs = obs[obs["h_agl_m"].between(H_MIN, H_MAX)]
        iy, ix = R.maille(meta, lat, lon, geom.ny, geom.nx)
        h_noeuds = z - zt[iy, ix]
        valide = (h_noeuds >= H_MIN) & (h_noeuds <= H_MAX)
        n = 0
        for t, ballon in obs.groupby("temps"):
            stem = stems.get(t)
            if stem is None:
                continue
            S, heure, jour, _ = modele.surface_from_cache(stem)
            vol = modele.predict(S, heure, jour, reliability=False)
            Y = np.load(cfg.paths.cache_dir / f"{stem}_Y.npy")
            h = ballon["h_agl_m"].to_numpy()
            hv = np.where(valide, h_noeuds, np.nan)
            um, vm = R.colonne_aux_hauteurs(hv, vol.u[:, iy, ix], vol.v[:, iy, ix], h)
            ue, ve = R.colonne_aux_hauteurs(hv, Y[0, :, iy, ix], Y[1, :, iy, ix], h)
            paires.append(pd.DataFrame({
                "site": nom, "temps": t, "h_agl_m": h, "u_obs": ballon["u"].to_numpy(),
                "v_obs": ballon["v"].to_numpy(), "u_modele": um, "v_modele": vm,
                "u_era5": ue, "v_era5": ve}))
            n += 1
        print(f"{nom} : maille ({iy}, {ix}), relief {zt[iy, ix]:.0f} m ; {n} radiosondages "
              f"apparies sur le jeu {args.split}")

    sortie = cfg.paths.run_dir / "radiosondages" / args.split
    sortie.mkdir(parents=True, exist_ok=True)
    if not paires:
        print("aucun radiosondage apparie (heures 00/12 UTC absentes du cache ?)")
        return 1
    toutes = pd.concat(paires).dropna()
    toutes.to_csv(sortie / "paires.csv", index=False)
    b = R.bilan(toutes)
    b.to_csv(sortie / "bilan.csv", index=False)
    (sortie / "bilan.json").write_text(b.to_json(orient="records", indent=2), encoding="utf-8")
    print(b.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
