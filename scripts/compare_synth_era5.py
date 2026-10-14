"""Le vent synthetique de Tukhnanutha ressemble-t-il au vent reel de Tunisie ? (ERA5 2025)

Le monde synthetique n'est PAS pilote par la meteo reelle : comparer instant par instant
n'a pas de sens. On compare donc des STATISTIQUES, niveau par niveau :

    synthetique : vent moyen du domaine par instant (scripts/synth_moyennes.py, 100 jours)
    reel        : ERA5 au centre du domaine place a l'ancrage SUPPOSE (config real.origin_*),
                  toutes les heures telechargees, interpole aux niveaux du modele

    vitesse moyenne, ecart-type, P10 / P50 / P90, direction dominante, part de vents forts

ERA5 (~31 km) represente la grande echelle : celle du vent moyen du domaine, justement.

    python scripts/compare_synth_era5.py -c local_v7
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]
SECTEURS = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]


def era5_column(pattern: str, lat: float, lon: float, z_levels) -> pd.DataFrame:
    """ERA5 niveaux de pression -> colonne (u, v) au point (lat, lon), aux altitudes du modele."""
    import xarray as xr

    from pcfno3d.realdata import G0

    frames = []
    for f in sorted(glob.glob(pattern)):
        ds = xr.open_dataset(f)
        t = "valid_time" if "valid_time" in ds.dims else "time"
        lev = "pressure_level" if "pressure_level" in ds.dims else "level"
        col = ds[["u", "v", "z"]].interp(latitude=lat, longitude=lon)
        z = col["z"].transpose(t, lev).values / G0                 # (nt, nlev)
        u = col["u"].transpose(t, lev).values
        v = col["v"].transpose(t, lev).values
        order = np.argsort(z.mean(axis=0))
        z, u, v = z[:, order], u[:, order], v[:, order]
        times = pd.to_datetime(col[t].values)
        for zk in z_levels:
            uu = np.array([np.interp(zk, z[i], u[i], left=np.nan, right=np.nan)
                           for i in range(len(times))])
            vv = np.array([np.interp(zk, z[i], v[i], left=np.nan, right=np.nan)
                           for i in range(len(times))])
            frames.append(pd.DataFrame({"timestamp": times, "z_m": int(zk), "u": uu, "v": vv}))
    if not frames:
        raise FileNotFoundError(f"aucun fichier ERA5 : {pattern}")
    df = pd.concat(frames, ignore_index=True).dropna()
    df["vitesse"] = np.hypot(df["u"], df["v"])
    return df.drop_duplicates(["timestamp", "z_m"])


def secteur(u, v) -> np.ndarray:
    d = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0           # d'ou vient le vent
    return np.array(SECTEURS)[(((d + 22.5) % 360) // 45).astype(int)]


def stats(df: pd.DataFrame) -> dict:
    s = df["vitesse"]
    sect = pd.Series(secteur(df["u"].values, df["v"].values)).value_counts(normalize=True)
    return {"n": len(df), "moy": s.mean(), "ecart_type": s.std(), "p10": s.quantile(0.1),
            "p50": s.median(), "p90": s.quantile(0.9), "fort_10ms": (s > 10).mean(),
            "dir_dominante": f"{sect.index[0]} ({sect.iloc[0]:.0%})"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="local_v7")
    base = REPO / "artifacts" / "reanalysis"
    ap.add_argument("--era5", default=str(base / "tunisie_2025" / "era5_pl_*.nc"))
    ap.add_argument("--synth", default=str(base / "comparaison" / "synth_moyennes.csv"))
    ap.add_argument("--out", default=str(base / "comparaison"))
    args = ap.parse_args()

    from pcfno3d.config import Config
    from pcfno3d.geometry import Geometry
    from pcfno3d.realdata import GeoRef

    cfg = Config.load(args.config)
    geom, geo = Geometry(cfg, device="cpu"), GeoRef.from_cfg(cfg)
    lat_c, lon_c = (float(a) for a in geo.to_latlon(geom.nx * geom.dx / 2, geom.ny * geom.dy / 2))
    niveaux = [int(z) for z in geom.z_levels if z >= 100]

    synth = pd.read_csv(args.synth, parse_dates=["timestamp"])
    synth = synth[synth["z_m"].isin(niveaux)]
    era5 = era5_column(args.era5, lat_c, lon_c, niveaux)

    print(f"ancrage {'SUPPOSE' if cfg.real.origin_assumed else 'mesure'} : centre du domaine "
          f"{lat_c:.3f} N {lon_c:.3f} E")
    print(f"synthetique : {synth['timestamp'].nunique()} instants "
          f"({synth['timestamp'].min():%Y-%m-%d} -> {synth['timestamp'].max():%Y-%m-%d})")
    print(f"ERA5        : {era5['timestamp'].nunique()} heures "
          f"({era5['timestamp'].min():%Y-%m-%d} -> {era5['timestamp'].max():%Y-%m-%d})\n")
    print(" niveau |   vitesse moyenne   |    ecart-type     |  P90 (vents forts) | > 10 m/s "
          "          | direction dominante")
    print("        |  synth   /  ERA5    |  synth  /  ERA5   |  synth  /  ERA5    | synth / ERA5"
          "        | synth      / ERA5")
    lignes = []
    for z in niveaux:
        a, b = stats(synth[synth["z_m"] == z]), stats(era5[era5["z_m"] == z])
        lignes.append({"z_m": z, **{f"synth_{k}": v for k, v in a.items()},
                       **{f"era5_{k}": v for k, v in b.items()}})
        print(f"  {z:5d} | {a['moy']:5.2f}  / {b['moy']:5.2f} m/s | {a['ecart_type']:5.2f} / "
              f"{b['ecart_type']:5.2f}   | {a['p90']:5.2f} / {b['p90']:5.2f}     | "
              f"{a['fort_10ms']:4.0%} / {b['fort_10ms']:4.0%}         | {a['dir_dominante']:10s} / "
              f"{b['dir_dominante']}")
    pd.DataFrame(lignes).to_csv(Path(args.out) / "synth_vs_era5_statistiques.csv", index=False)

    commun = synth.merge(era5, on=["timestamp", "z_m"], suffixes=("_s", "_e"))
    if len(commun):
        n = commun["timestamp"].nunique()
        print(f"\nmemes instants ({n}) : correlation de la vitesse synth/ERA5 par niveau "
              "(attendue ~0 : le synthetique n'est pas pilote par la meteo reelle)")
        for z, g in commun.groupby("z_m"):
            c = np.corrcoef(g["vitesse_s"], g["vitesse_e"])[0, 1] if len(g) > 2 else np.nan
            print(f"  {z:5d} m : {c:+.2f}")
    print(f"\n-> {Path(args.out) / 'synth_vs_era5_statistiques.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
