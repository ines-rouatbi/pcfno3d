"""Forcage ERA5 d'une zone pour le generateur reel : une ligne par instant (toutes les 3 h).

La verite generee ne depend QUE d'ERA5, du relief et du sol : aucune station n'entre ici.
Chaque grandeur est une moyenne ERA5 sur les mailles de la zone (avec une marge) :

    g_ms, dir_deg, ug, vg   vent a 850 hPa (~1 500 m) : vent de grande echelle du generateur
    shear_u, shear_v        (V700 - V850) / (z700 - z850) : cisaillement au-dessus de la couche limite
    veer_deg                rotation du vent entre 850 hPa et 10 m (Ekman), bornee a [0, 45]
    stabilite               tanh(dtheta / 3 K), dtheta = theta(couche 1000-925 hPa, Tv - 1,5 K) - theta(2 m)
    t_mer_c, p_mer_pa       temperature ramenee au niveau de la mer, pression au niveau de la mer
    dt_terre_mer_k          temperature de peau des terres - SST (slx) ; a defaut t2m terres - t2m mer
    h_cl_m                  hauteur de couche limite ERA5 (slx), vide si absente
    u_era5_<z>, v_era5_<z>  vent moyen ERA5 aux niveaux du modele (m au-dessus de la mer) :
                            reference du controle « moyenne du champ genere = ERA5 a chaque niveau »
et, a cote (meme nom, .npz), le vent ERA5 maille par maille aux niveaux du modele (temps, niveau,
lat, lon) : le generateur y recale sa grande echelle (recalage spatial).

    python scripts/forcage_era5.py --era5 <dossier> --grille <grille_reelle.json> --sortie forcage.csv
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

G = 9.80665
R_AIR = 287.05
KAPPA = 0.2857
LAPSE = 0.0065
NIVEAUX = (25., 50., 75., 100., 150., 200., 300., 400., 500., 650., 800., 1000.)   # v7
# hauteurs AU-DESSUS DU SOL ERA5 des champs par maille (recalage qui suit le relief)
HAUTEURS_SOL = (10., 25., 50., 75., 100., 150., 200., 300., 400., 500., 650., 800., 1000., 1200.)


def _ouvrir(fichiers, variables):
    import xarray as xr

    ds = []
    for f in fichiers:
        d = xr.open_dataset(f)
        garde = [v for v in variables if v in d]
        if garde:
            ds.append(d[garde])
    if not ds:
        return None
    tdim = "valid_time" if "valid_time" in ds[0].dims else "time"
    tout = xr.concat(ds, dim=tdim)
    _, i = np.unique(tout[tdim].values, return_index=True)
    return tout.isel({tdim: np.sort(i)}).rename({tdim: "temps"})


def _boite(ds, info, marge):
    z = info["zone"]
    lat = ds["latitude"].values
    sel = {"latitude": slice(z["lat_nord"] + marge, z["lat_sud"] - marge) if lat[0] > lat[-1]
           else slice(z["lat_sud"] - marge, z["lat_nord"] + marge),
           "longitude": slice(z["lon_ouest"] - marge, z["lon_est"] + marge)}
    return ds.sel(sel)


def forcage(era5: Path, info: dict, marge: float = 0.15, slx: bool = True,
            niveaux=NIVEAUX, hauteurs_sol=None) -> pd.DataFrame:

    if hauteurs_sol is None:
        # national : niveaux au-dessus de 1000 m -> champs par maille jusqu'a ce niveau
        hauteurs_sol = HAUTEURS_SOL + tuple(h for h in (1500., 2000., 2500., 3000.)
                                            if h - 500. < max(niveaux) and h > HAUTEURS_SOL[-1])
    d = Path(era5)
    pl = _ouvrir(sorted(d.glob("era5_plx_*.nc")), ["u", "v", "z"])
    sl = _ouvrir(sorted(d.glob("era5_sl_*.nc")), ["u10", "v10", "u100", "v100", "t2m", "sp", "msl", "z"])
    # sans_slx : une seule regle pour tout le jeu (contraste t2m, couche limite par formule),
    # quand la SST / temperature de peau ne couvre qu'une partie des dates
    slx = _ouvrir(sorted(d.glob("era5_slx_*.nc")), ["sst", "skt", "blh"]) if slx else None
    import xarray as xr
    with xr.open_dataset(d / "masque_terre_mer.nc") as m:
        lsm = _boite(m["lsm"].squeeze(drop=True).load(), info, marge)
    pl, sl = _boite(pl, info, marge), _boite(sl, info, marge)
    communs = np.intersect1d(pl["temps"].values, sl["temps"].values)
    pl, sl = pl.sel(temps=communs).load(), sl.sel(temps=communs).load()
    pdim = "pressure_level" if "pressure_level" in pl.dims else "level"
    terre = (lsm.values >= 0.5)
    if slx is not None:
        slx = _boite(slx, info, marge)
        slx = slx.reindex(temps=communs).load()

    def niveau(nom, p):
        return pl[nom].sel({pdim: p}).values                       # (t, y, x)

    u850, v850 = niveau("u", 850).mean((1, 2)), niveau("v", 850).mean((1, 2))
    u700, v700 = niveau("u", 700).mean((1, 2)), niveau("v", 700).mean((1, 2))
    dz = (niveau("z", 700) - niveau("z", 850)).mean((1, 2)) / G
    u10, v10 = sl["u10"].values.mean((1, 2)), sl["v10"].values.mean((1, 2))
    g = np.hypot(u850, v850)
    dirm = (270.0 - np.degrees(np.arctan2(v850, u850))) % 360.0
    veer = (np.degrees(np.arctan2(v10, u10)) - np.degrees(np.arctan2(v850, u850)) + 180.0) % 360.0 - 180.0
    veer = np.where(np.hypot(u10, v10) > 1.0, np.clip(veer, 0.0, 45.0), 25.0)

    # stabilite : theta de la couche 1000-925 hPa (epaisseur -> Tv) contre theta a 2 m
    tv = (niveau("z", 925) - niveau("z", 1000)) / (R_AIR * np.log(1000.0 / 925.0))
    th_couche = (tv * (1000.0 / 962.5) ** KAPPA).mean((1, 2))
    th_2m = (sl["t2m"].values * (100000.0 / sl["sp"].values) ** KAPPA).mean((1, 2))
    # la couche est en temperature VIRTUELLE (epaisseur) : ~1,5 K de plus que la temperature
    # seche a l'humidite tunisienne ; on la retire avant de comparer a theta 2 m
    stab = np.tanh((th_couche - 1.5 - th_2m) / 3.0)

    oro = sl["z"].values[0] / G
    t_mer = (sl["t2m"].values + LAPSE * oro[None]).mean((1, 2)) - 273.15
    t2m = sl["t2m"].values
    dt_tm = t2m[:, terre].mean(1) - t2m[:, ~terre].mean(1) if terre.any() and (~terre).any() \
        else np.zeros(len(communs))
    h_cl = np.full(len(communs), np.nan)
    if slx is not None:
        skt, sst = slx["skt"].values, slx["sst"].values
        ok = np.isfinite(skt).all((1, 2))
        dt_slx = np.nanmean(np.where(terre, skt, np.nan), (1, 2)) - np.nanmean(sst, (1, 2))
        dt_tm = np.where(ok & np.isfinite(dt_slx), dt_slx, dt_tm)
        h_cl = np.nanmean(slx["blh"].values, (1, 2))

    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(communs).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "g_ms": g, "dir_deg": dirm, "ug": u850, "vg": v850,
        "shear_u": (u700 - u850) / dz, "shear_v": (v700 - v850) / dz, "veer_deg": veer,
        "stabilite": stab, "t_mer_c": t_mer, "p_mer_pa": sl["msl"].values.mean((1, 2)),
        "dt_terre_mer_k": dt_tm, "h_cl_m": h_cl})

    # profil ERA5 moyen aux niveaux du modele (au-dessus de la mer) : 10 m, 100 m et niveaux de
    # pression, interpoles en hauteur colonne par colonne puis moyennes sur la zone
    from pcfno3d.tunisie import interpoler_hauteurs
    champs = {}
    zl = pl["z"].values / G                                        # (t, p, y, x) au-dessus de la mer
    h = np.concatenate([np.broadcast_to(oro + 10.0, (len(communs),) + oro.shape)[None],
                        np.broadcast_to(oro + 100.0, (len(communs),) + oro.shape)[None],
                        np.moveaxis(np.where(zl > oro[None, None] + 100.0, zl, np.nan), 1, 0)])
    for c, c10, c100 in (("u", "u10", "u100"), ("v", "v10", "v100")):
        val = np.concatenate([sl[c10].values[None], sl[c100].values[None],
                              np.moveaxis(pl[c].values, 1, 0)])
        prof = interpoler_hauteurs(h, val, list(niveaux))            # (nz, t, y, x), NaN sous le sol
        for k, zk in enumerate(niveaux):
            df[f"{c}_era5_{int(zk)}"] = np.nanmean(prof[k], axis=(1, 2))
        # champ par maille pour le recalage spatial, en HAUTEUR AU-DESSUS DU SOL ERA5 : le
        # generateur le relit a la meme hauteur au-dessus de SON relief (suivi du terrain)
        prof_s = interpoler_hauteurs(h, val, [oro + hh for hh in hauteurs_sol])   # (nh, t, y, x)
        # au-dessus du dernier niveau de pression (700 hPa, ~3 km) le profil est vide (NaN) : sur le
        # relief national (mailles a 500-1000 m), 2000-2500 m au-dessus du sol y sont. Valeur
        # valide la plus proche dans la colonne, comme le temps reel GFS (temps_reel._boucher) :
        # sans cela le generateur ecrivait un vent nul a ces noeuds et le modele recevait des NaN.
        trous = int(np.isnan(prof_s).sum())
        for k in range(1, prof_s.shape[0]):
            prof_s[k] = np.where(np.isfinite(prof_s[k]), prof_s[k], prof_s[k - 1])
        for k in range(prof_s.shape[0] - 2, -1, -1):
            prof_s[k] = np.where(np.isfinite(prof_s[k]), prof_s[k], prof_s[k + 1])
        if trous:
            print(f"{c} : {trous} valeurs au-dessus du profil ERA5 comblees par la plus proche "
                  f"({trous / prof_s.size:.2%})", flush=True)
        if np.isnan(prof_s).any():
            raise ValueError(f"{c} : colonnes ERA5 entierement vides")
        champs[c] = prof_s
    lat = pl["latitude"].values
    o = np.argsort(lat)
    df.attrs["champs"] = {"temps": pd.to_datetime(communs).values.astype("datetime64[s]").astype(np.int64),
                          "lat": lat[o], "lon": pl["longitude"].values,
                          "hauteurs_sol": np.array(hauteurs_sol),
                          "u": champs["u"][:, :, o, :].transpose(1, 0, 2, 3).astype(np.float32),
                          "v": champs["v"][:, :, o, :].transpose(1, 0, 2, 3).astype(np.float32)}
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--era5", required=True)
    ap.add_argument("--grille", required=True, help="grille_reelle.json (scripts/grille_reelle.py)")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--niveaux", default=None,
                    help="niveaux du modele (m au-dessus de la mer), comme generate_aneo_3d.py --niveaux")
    ap.add_argument("--sans-slx", action="store_true",
                    help="ignore SST / temperature de peau / couche limite (meme regle partout)")
    args = ap.parse_args()
    info = json.loads(Path(args.grille).read_text(encoding="utf-8"))
    niv = tuple(sorted(float(z) for z in args.niveaux.split(","))) if args.niveaux else NIVEAUX
    df = forcage(Path(args.era5), info, slx=not args.sans_slx, niveaux=niv)
    df.to_csv(args.sortie, index=False, float_format="%.5g")
    np.savez_compressed(Path(args.sortie).with_suffix(".npz"), **df.attrs["champs"])
    t = pd.to_datetime(df["timestamp_utc"])
    print(f"{len(df)} instants, {t.min():%Y-%m-%d} -> {t.max():%Y-%m-%d} ; mois : "
          f"{sorted(t.dt.strftime('%Y-%m').unique())}")
    print(df[["g_ms", "veer_deg", "stabilite", "t_mer_c", "dt_terre_mer_k", "h_cl_m"]].describe().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
