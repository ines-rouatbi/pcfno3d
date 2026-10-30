"""Previsions GFS archivees (NOAA, 0,25 deg) -> forcage du modele Grand Tunis, pour l'OS7 (cahier v12,
section 18.10) : le meme modele, nourri par la prevision a +24 h ou +48 h au lieu d'ERA5.

Source : l'archive officielle NOAA sur AWS (noaa-gfs-bdp-pds, depuis janvier 2021), via Herbie, en
ne telechargeant que les champs utiles (fichiers .idx, requetes par plages d'octets) : ~34 Mo par
echeance pour 38 champs, decoupes aussitot sur les mailles du domaine puis effaces. L'API « Previous
Runs » d'Open-Meteo n'a pas le vent prevu avant 2025 (verifie le 30/10/2026).

Pour un instant de validite t et une echeance visee L (24 ou 48 h) : le run est le dernier cycle
GFS (00, 06, 12, 18 UTC) tel que t - run >= L, donc une echeance effective L a L + 5 h, notee.

Les champs sont ramenes au format de `pcfno3d.temps_reel.telecharger_gfs` (vitesse, direction,
geopotentiel par niveau ; 10 m ; 100 m ; T, HR a 2 m ; pression ; rayonnement) et passent par
`temps_reel.forcage_gfs` : les MEMES formules que le forcage ERA5 et que le temps reel. Relief des
mailles = orographie GFS (HGT:surface).

    python -m pcfno3d.gfs --config grand_tunis_v2_long_colab --echeance 24 --sortie forcage_gfs_f24.csv
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

NIVEAUX_GFS = (1000, 975, 950, 925, 900, 850, 800, 750, 700)
RECHERCHE = (r":(?:UGRD|VGRD|HGT):(?:" + "|".join(str(p) for p in NIVEAUX_GFS) + r") mb:"
             r"|:(?:UGRD|VGRD):(?:10|100) m above ground:|:(?:TMP|RH):2 m above ground:"
             r"|:(?:PRES|HGT|LAND):surface:|:PRMSL:mean sea level:|:DSWRF:surface:")
CYCLE_H = 6


def run_pour(valide: pd.Timestamp, echeance_h: int) -> tuple[pd.Timestamp, int]:
    """(run, echeance effective en h) : dernier cycle GFS a au moins `echeance_h` de `valide`."""
    t = pd.Timestamp(valide)
    limite = t - pd.Timedelta(hours=echeance_h)
    run = limite.floor(f"{CYCLE_H}h")
    return run, int(round((t - run) / pd.Timedelta(hours=1)))


def _vitesse_direction(u, v):
    return np.hypot(u, v), (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def extraire(datasets, lat: np.ndarray, lon: np.ndarray) -> dict:
    """Les datasets cfgrib d'un fichier GFS partiel -> dict « Open-Meteo » d'un instant, chaque
    champ (1, ny, nx) aux mailles (lat, lon) : point le plus proche de la grille 0,25 deg (les
    mailles ERA5 du forcage sont sur la meme grille 0,25 deg)."""
    import xarray as xr

    la, lo = np.meshgrid(np.asarray(lat, float), np.mod(np.asarray(lon, float), 360.0), indexing="ij")
    sel = {"latitude": xr.DataArray(la.ravel(), dims="p"), "longitude": xr.DataArray(lo.ravel(), dims="p")}

    def grille(ds, nom):
        vals = ds[nom].sel(**sel, method="nearest").values
        return np.asarray(vals, float).reshape(np.shape(vals)[:-1] + la.shape)

    out: dict = {}
    for ds in datasets:
        noms = set(ds.data_vars)
        if {"u", "v", "gh"} <= noms and "isobaricInhPa" in ds.coords:
            niveaux = np.atleast_1d(ds["isobaricInhPa"].values).astype(int)
            u, v, gh = (grille(ds, n).reshape((len(niveaux),) + la.shape) for n in ("u", "v", "gh"))
            for i, p in enumerate(niveaux):
                if int(p) in NIVEAUX_GFS:
                    sp, di = _vitesse_direction(u[i], v[i])
                    out[f"wind_speed_{p}hPa"], out[f"wind_direction_{p}hPa"] = sp[None], di[None]
                    out[f"geopotential_height_{p}hPa"] = gh[i][None]
        for h in (10, 100):
            if {f"u{h}", f"v{h}"} <= noms:
                sp, di = _vitesse_direction(grille(ds, f"u{h}"), grille(ds, f"v{h}"))
                out[f"wind_speed_{h}m"], out[f"wind_direction_{h}m"] = sp[None], di[None]
        if "t2m" in noms:
            out["temperature_2m"] = grille(ds, "t2m")[None] - 273.15
        if "r2" in noms:
            out["relative_humidity_2m"] = grille(ds, "r2")[None]
        if "sp" in noms:
            out["surface_pressure"] = grille(ds, "sp")[None] / 100.0
        if "orog" in noms:
            out["elevation"] = grille(ds, "orog")
        if "lsm" in noms:
            out["terre"] = grille(ds, "lsm") >= 0.5
        for nom in ("sdswrf", "dswrf"):
            if nom in noms:
                out["shortwave_radiation"] = grille(ds, nom)[None]
        for nom in ("prmsl", "msl"):
            if nom in noms:
                out["pressure_msl"] = grille(ds, nom)[None] / 100.0
    return out


REQUIS = ["wind_speed_10m", "wind_speed_100m", "temperature_2m", "relative_humidity_2m",
          "surface_pressure", "pressure_msl", "elevation"] + \
         [f"wind_speed_{p}hPa" for p in NIVEAUX_GFS] + [f"geopotential_height_{p}hPa" for p in NIVEAUX_GFS]


def telecharger_echeance(run: pd.Timestamp, echeance_h: int, lat, lon, dossier: Path) -> dict:
    """Une echeance GFS decoupee aux mailles ; le fichier GRIB partiel est efface aussitot."""
    import cfgrib
    from herbie import Herbie

    H = Herbie(run.strftime("%Y-%m-%d %H:%M"), model="gfs", product="pgrb2.0p25", fxx=int(echeance_h),
               priority=["aws", "google", "azure", "nomads"], save_dir=str(dossier), verbose=False)
    if H.grib is None:
        raise FileNotFoundError(f"GFS {run} f{echeance_h:03d} introuvable")
    f = Path(H.download(RECHERCHE, verbose=False))
    try:
        ds = cfgrib.open_datasets(str(f), backend_kwargs={"indexpath": ""})
        c = extraire(ds, np.asarray(lat), np.asarray(lon))
        for d in ds:
            d.close()
    finally:
        f.unlink(missing_ok=True)
    manque = [k for k in REQUIS if k not in c]
    if manque:
        raise ValueError(f"GFS {run} f{echeance_h:03d} : champs absents {manque}")
    c["source"] = str(H.grib_source)
    return c


def assembler(instants: list[tuple[pd.Timestamp, dict]]) -> dict:
    """Une liste (instant de validite, champs d'un instant) -> dict « Open-Meteo » sur le temps."""
    instants = sorted(instants, key=lambda x: x[0])
    c0 = instants[0][1]
    out = {"temps": np.array([int(t.value // 10 ** 9) for t, _ in instants], np.int64),
           "elevation": c0["elevation"]}
    if "terre" in c0:
        out["terre"] = c0["terre"]
    for k in c0:
        if k in ("elevation", "terre", "source"):
            continue
        out[k] = np.concatenate([c[k] for _, c in instants], axis=0)
    return out


def ecrire_forcage(c: dict, hauteurs, lat, lon, chemin_csv: Path, echeances: dict) -> pd.DataFrame:
    """CSV (scalaires BRUTS, comme forcage_era5.csv) + .npz (cellules par hauteur au-dessus du
    sol), lisibles tels quels par ForcageERA5 : paths.forcage_era5 peut pointer dessus."""
    from .forcage import SCALAIRES
    from .temps_reel import _boucher, forcage_gfs

    temps, scal, u, v = forcage_gfs(c, hauteurs)
    # hauteur hors du profil GFS (sous 10 m, au-dessus de 700 hPa) : la valeur voisine, comme le
    # temps reel (ForcageDirect)
    u, v = _boucher(np.asarray(u, np.float32)), _boucher(np.asarray(v, np.float32))
    df = pd.DataFrame({"timestamp_utc": pd.to_datetime(temps, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ")})
    for j, (k, f) in enumerate(SCALAIRES):
        df[k] = scal[:, j] / f
    df["run_utc"] = [echeances[int(t)][0] for t in temps]
    df["echeance_h"] = [echeances[int(t)][1] for t in temps]
    chemin_csv = Path(chemin_csv)
    chemin_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(chemin_csv, index=False, float_format="%.6g")
    np.savez_compressed(chemin_csv.with_suffix(".npz"), temps=temps, lat=np.asarray(lat),
                        lon=np.asarray(lon), hauteurs_sol=np.asarray(hauteurs, float),
                        u=u.astype(np.float32), v=v.astype(np.float32))
    return df


def construire(instants: list[pd.Timestamp], echeance_h: int, lat, lon, hauteurs,
               chemin_csv: Path, reprise: Path, travailleurs: int = 6, verbose: bool = True) -> pd.DataFrame:
    """Telecharge (en parallele, avec reprise : un .npz par instant dans `reprise`) puis ecrit
    le forcage GFS de tous les instants de validite."""
    reprise = Path(reprise)
    reprise.mkdir(parents=True, exist_ok=True)
    a_faire, faits = [], []
    echeances = {}
    for t in sorted(set(pd.Timestamp(x) for x in instants)):
        run, l_eff = run_pour(t, echeance_h)
        echeances[int(t.value // 10 ** 9)] = (run.strftime("%Y-%m-%dT%H:%MZ"), l_eff)
        f = reprise / f"{t:%Y%m%dT%H%M}_f{echeance_h}.npz"
        (faits if f.exists() else a_faire).append((t, run, l_eff, f))
    if verbose:
        print(f"GFS +{echeance_h} h : {len(a_faire)} echeances a telecharger, {len(faits)} deja faites",
              flush=True)
    t0, erreurs = time.time(), []

    def tache(t, run, l_eff, f):
        with tempfile.TemporaryDirectory(dir=reprise) as tmp:
            c = telecharger_echeance(run, l_eff, lat, lon, Path(tmp))
        np.savez_compressed(f, **{k: v for k, v in c.items() if k != "source"})
        return t

    with ThreadPoolExecutor(max_workers=travailleurs) as ex:
        futs = {ex.submit(tache, *x): x for x in a_faire}
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                fu.result()
            except Exception as exc:                                  # noqa: BLE001
                erreurs.append((futs[fu][0], str(exc)[:200]))
            if verbose and (n % 25 == 0 or n == len(futs)):
                print(f"  {n}/{len(futs)} ({time.time() - t0:.0f} s, {len(erreurs)} erreurs)", flush=True)
    instants_ok = []
    for t in sorted(set(pd.Timestamp(x) for x in instants)):
        f = reprise / f"{t:%Y%m%dT%H%M}_f{echeance_h}.npz"
        if f.exists():
            with np.load(f) as d:
                instants_ok.append((t, {k: d[k] for k in d.files}))
    if not instants_ok:
        raise RuntimeError(f"aucune echeance GFS disponible ; erreurs : {erreurs[:3]}")
    c = assembler(instants_ok)
    df = ecrire_forcage(c, hauteurs, lat, lon, chemin_csv, echeances)
    if erreurs:
        (Path(chemin_csv).with_suffix(".erreurs.json")).write_text(
            json.dumps([{"instant": str(t), "erreur": e} for t, e in erreurs], indent=1), encoding="utf-8")
    if verbose:
        print(f"-> {chemin_csv} : {len(df)} instants ({len(erreurs)} manquants)", flush=True)
    return df


def main() -> int:
    from .cache import load_index
    from .config import Config
    from .forcage import ForcageERA5

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="configuration du modele (forcage ERA5 = mailles)")
    ap.add_argument("--echeance", type=int, required=True, choices=(24, 48))
    ap.add_argument("--sortie", required=True, help="forcage_gfs_f24.csv (le .npz est ecrit a cote)")
    ap.add_argument("--reprise", help="dossier des echeances deja telechargees (defaut : a cote)")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--max-instants", type=int, help="essai")
    ap.add_argument("--travailleurs", type=int, default=6)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    ref = ForcageERA5(cfg.paths.forcage_era5)                         # mailles et hauteurs
    idx = load_index(cfg)
    idx = idx[idx["split"].isin(args.splits.split(","))]
    instants = list(pd.to_datetime(idx["timestamp"], utc=True).dt.tz_localize(None).unique())
    if args.max_instants:
        instants = instants[:: max(1, math.ceil(len(instants) / args.max_instants))][:args.max_instants]
    sortie = Path(args.sortie)
    reprise = Path(args.reprise) if args.reprise else sortie.parent / f"gfs_f{args.echeance}_instants"
    # jusqu'a 3 passes : une echeance refusee (reseau, archive lente) est retentee ; deux instants
    # manquants consecutifs bloqueraient l'entrainement (pas d'echeance a moins de 3 h)
    erreurs = sortie.with_suffix(".erreurs.json")
    for passe in range(3):
        erreurs.unlink(missing_ok=True)
        construire(instants, args.echeance, ref.lat, ref.lon, ref.hauteurs, sortie, reprise,
                   travailleurs=args.travailleurs if passe == 0 else 2)
        if not erreurs.exists():
            break
        print(f"passe {passe + 1} : echeances manquantes, nouvelle tentative", flush=True)
    return 0 if not erreurs.exists() else 4


if __name__ == "__main__":
    raise SystemExit(main())
