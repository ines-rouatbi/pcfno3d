"""PC-FNO3D Tunisie : le principe de la v7 sur la Tunisie entiere, avec ERA5 et les METAR.

Meme modele (FNO 3D), memes entrees (capteurs de surface + relief + geometrie + temps et vent
geostrophique), meme cible (u, v, w, rho aux noeuds 3D, niveaux en altitude au-dessus de la
mer), meme loss physique (donnees + gradients + conservation de la masse + condition au sol),
memes DSO : seul le jeu de donnees change. On l'ecrit au FORMAT EXACT du generateur et du
cache v7 (meta.json, split_days.json, aneo_grid_points.csv ; <stem>_S.npy, <stem>_Y.npy,
index.csv), pour que `Geometry`, `ANEO3D`, `Trainer` et `Evaluator` servent sans changement.

    grille  : la grille ERA5 de la boite Tunisie (0,25 deg ; 30,0-37,75 N, 7,25-11,75 E),
              32 x 19 mailles, sud -> nord (iy) et ouest -> est (ix). Pas metrique constant,
              pris a la latitude centrale : dx ~23 km, dy ~27,8 km (ecart < 5 % aux bords nord
              et sud ; le residu de masse de la verite le montre, voir `bilan_masse`).
    relief  : orographie ERA5 (geopotentiel de surface / g), deja lissee a l'echelle du modele.
    niveaux : NIVEAUX_ASL, de 10 a 2 200 m au-dessus de la mer ; un noeud existe si sa hauteur
              au-dessus du sol est dans [H_MIN, H_MAX] = [5, 1000] m : « du sol jusqu'a 1 km »
              partout, y compris sur les reliefs a ~1 200 m.
    capteurs: les anemometres des aeroports (METAR). Un anemometre a coupelles et girouette
              mesure le vent HORIZONTAL a sa hauteur (10 m, norme OMM) : u et v ; il ne mesure
              pas w (canal w = 0) ; la hauteur de l'anemometre entre dans le canal statique
              z_rel du capteur, comme les 20-30 m des capteurs ANEO dans la v7. Temperature,
              humidite et pression viennent de la meme station ; la pression est ramenee a
              l'altitude du capteur dans le modele (relief ERA5 + hauteur de l'anemometre), et
              la densite en est deduite (temperature virtuelle).
    verite  : ERA5. u, v : 10 m et 100 m (niveaux simples) puis niveaux de pression 1000-700 hPa
              (> 100 m au-dessus du sol), interpoles en hauteur ; entre 5 et 10 m, profil
              logarithmique depuis le vent a 10 m. rho : p / (R Tv), Tv tiree de l'epaisseur
              geopotentielle entre niveaux (hypsometrie) ; au sol sp / (R T2m). w = -omega /
              (rho g), et au sol la condition cinematique u.dzt/dx + v.dzt/dy (celle de L_bc).
    decoupage : 2021-2023 entrainement, 2024 validation, 2025 test (par jour).

Variante ERA5-Land (`grille="land"`, configs/tunisie_fno3d_land.yaml) : la grille, le relief
et la surface viennent d'ERA5-Land (0,1 deg, ~9 x 11 km). Vent 10 m, T 2 m et pression de
surface ERA5-Land sur les terres, ERA5 interpole sur la mer (ERA5-Land n'y a pas de valeurs).
Le vent 100 m, les niveaux de pression et omega restent ERA5 (0,25 deg), interpoles en
bilineaire sur la grille fine ; leurs hauteurs au-dessus du sol sont prises sur le relief fin.
Les niveaux montent jusqu'a 1 km au-dessus du plus haut relief fin.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .tunisie import ANNEES, entree_stations, interpoler_hauteurs, lire_metar

G = 9.80665
R_AIR = 287.05
NIVEAUX_ASL = (10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0,
               600.0, 700.0, 800.0, 900.0, 1000.0, 1150.0, 1300.0, 1450.0, 1600.0, 1800.0,
               2000.0)
H_MIN, H_MAX = 5.0, 1000.0           # noeud 3D : hauteur au-dessus du sol dans [H_MIN, H_MAX]
HAUTEUR_ANEMOMETRE = 10.0            # norme OMM, si l'archive METAR ne la donne pas
Z0_MER, Z0_TERRE = 2e-4, 0.05        # rugosite (m) selon le masque terre-mer ERA5
ID_NOEUDS = 100001                   # point_id du premier noeud 3D (surface : 1 .. ny*nx)
JOUR_0 = pd.Timestamp("2021-01-01")
MIN_STATIONS = 3                     # instant ignore avec moins de stations disponibles


# --------------------------------------------------------------------------- grille
def pas_metriques(lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    """dx, dy (m) de la grille reguliere en degres, a la latitude centrale."""
    lat0 = math.radians(float(np.mean(lat)))
    dlon = abs(float(lon[1] - lon[0]))
    dlat = abs(float(lat[1] - lat[0]))
    dx = dlon * (111412.84 * math.cos(lat0) - 93.5 * math.cos(3 * lat0))
    dy = dlat * (111132.92 - 559.82 * math.cos(2 * lat0) + 1.175 * math.cos(4 * lat0))
    return dx, dy


def f_coriolis(lat: np.ndarray) -> float:
    return 2 * 7.2921e-5 * math.sin(math.radians(float(np.mean(lat))))


def niveaux_pour(zt: np.ndarray, pas: float = 200.0) -> tuple[float, ...]:
    """NIVEAUX_ASL prolonges de `pas` en `pas` tant que des noeuds valides existent plus haut
    (relief fin au-dessus de 1 000 m : 1 km au-dessus du sol partout)."""
    niveaux = list(NIVEAUX_ASL)
    while niveaux[-1] + pas <= float(np.nanmax(zt)) + H_MAX:
        niveaux.append(niveaux[-1] + pas)
    return tuple(niveaux)


def vers_grille(a: np.ndarray, lat_c: np.ndarray, lon_c: np.ndarray, lat_f: np.ndarray,
                lon_f: np.ndarray) -> np.ndarray:
    """Interpolation bilineaire des deux derniers axes (lat, lon croissantes) de la grille
    grossiere vers la grille fine (incluse dans la grossiere)."""
    def poids(c, f):
        w = np.zeros((len(f), len(c)))
        i = np.clip(np.searchsorted(c, f) - 1, 0, len(c) - 2)
        r = np.clip((f - c[i]) / (c[i + 1] - c[i]), 0.0, 1.0)
        w[np.arange(len(f)), i] = 1.0 - r
        w[np.arange(len(f)), i + 1] += r
        return w
    return np.einsum("fy,...yx,gx->...fg", poids(lat_c, lat_f), a, poids(lon_c, lon_f))


def noeuds_valides(zt: np.ndarray, niveaux=NIVEAUX_ASL) -> np.ndarray:
    """(nz, ny, nx) : noeud a une hauteur au-dessus du sol dans [H_MIN, H_MAX]."""
    h = np.asarray(niveaux)[:, None, None] - zt[None]
    return (h >= H_MIN) & (h <= H_MAX)


def ecrire_jeu(dossier: Path, lat: np.ndarray, lon: np.ndarray, zt: np.ndarray,
               lsm: np.ndarray, hauteur_capteur: np.ndarray, niveaux=NIVEAUX_ASL,
               source: str = "ERA5") -> dict:
    """meta.json, split_days.json et aneo_grid_points.csv au format du generateur.

    lat croissante (sud -> nord), lon croissante ; zt, lsm, hauteur_capteur : (ny, nx)."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    ny, nx = zt.shape
    dx, dy = pas_metriques(lat, lon)
    dzdy, dzdx = np.gradient(zt, dy, dx)
    pente = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    orientation = np.degrees(np.arctan2(-dzdx, -dzdy)) % 360.0
    z0 = np.where(lsm >= 0.5, Z0_TERRE, Z0_MER)

    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    surface = pd.DataFrame({
        "point_id": 1 + (iy * nx + ix).ravel(), "ix": ix.ravel(), "iy": iy.ravel(),
        "z_terrain_m": zt.ravel(), "z_rel_m": hauteur_capteur.ravel(), "z0_m": z0.ravel(),
        "slope_deg": pente.ravel(), "aspect_deg": orientation.ravel()})
    valid = noeuds_valides(zt, niveaux)
    k, j, i = np.nonzero(valid)
    noeuds = pd.DataFrame({
        "point_id": ID_NOEUDS + np.arange(len(k)), "ix": i, "iy": j,
        "z_terrain_m": zt[j, i], "z_rel_m": np.asarray(niveaux)[k] - zt[j, i],
        "z0_m": z0[j, i], "slope_deg": pente[j, i], "aspect_deg": orientation[j, i]})
    pd.concat([surface, noeuds]).to_csv(dossier / "aneo_grid_points.csv", index=False)

    meta = {
        "producteur": f"pcfno3d.tunisie_fno3d ({source} + METAR)",
        "grille_surface": {"nx": nx, "ny": ny, "dx_m": dx, "dy_m": dy},
        "noeuds_3d": {"stride": 1, "nx": nx, "ny": ny, "dx_m": dx,
                      "z_levels_asl_m": list(map(float, niveaux)),
                      "min_z_rel_m": H_MIN, "max_z_rel_m": H_MAX,
                      "point_id_debut": ID_NOEUDS, "nombre": int(len(k))},
        "georeference": {"lat_sud": float(lat[0]), "lat_nord": float(lat[-1]),
                         "lon_ouest": float(lon[0]), "lon_est": float(lon[-1]),
                         "pas_deg": float(abs(lat[1] - lat[0])),
                         "pas_metrique": "constant, latitude centrale"},
        "conventions": {"u": "positif vers l'Est", "v": "positif vers le Nord",
                        "w": "positif vers le haut", "iy": "sud -> nord", "ix": "ouest -> est"},
        "f_coriolis": f_coriolis(lat),
        "source": f"{source} (Copernicus C3S) pour la verite ; METAR (IEM) pour les capteurs",
    }
    (dossier / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    jours = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    split = {"train_days": [], "val_days": [], "test_days": []}
    cle = {"entrainement": "train_days", "validation": "val_days", "test": "test_days"}
    for jeu, annees in ANNEES.items():
        split[cle[jeu]] = [int((d - JOUR_0).days) for d in jours if d.year in annees]
    (dossier / "split_days.json").write_text(json.dumps(split), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- verite ERA5
def densite_niveaux(phi: np.ndarray, p_hpa: np.ndarray) -> np.ndarray:
    """rho (t, p, y, x) aux niveaux de pression, par l'hypsometrie.

    phi : geopotentiel (t, p, y, x), p_hpa : (p,) decroissant. Tv d'une couche =
    (phi_haut - phi_bas) / (R ln(p_bas / p_haut)) ; Tv d'un niveau = moyenne des couches
    voisines (une seule aux extremites)."""
    p = np.asarray(p_hpa, np.float64) * 100.0
    tv_c = (phi[:, 1:] - phi[:, :-1]) / (R_AIR * np.log(p[:-1] / p[1:]))[None, :, None, None]
    tv = np.empty_like(phi, dtype=np.float64)
    tv[:, 0], tv[:, -1] = tv_c[:, 0], tv_c[:, -1]
    tv[:, 1:-1] = 0.5 * (tv_c[:, 1:] + tv_c[:, :-1])
    return p[None, :, None, None] / (R_AIR * tv)


def _colonnes(h: np.ndarray, val: np.ndarray, cibles: np.ndarray) -> np.ndarray:
    """Interpolation en hauteur colonne par colonne vers des cibles (nz, y, x) propres a
    chaque colonne ; h, val : (n, t, y, x). -> (t, nz, y, x)."""
    out = interpoler_hauteurs(h, val, list(cibles))           # (nz, t, y, x)
    return np.moveaxis(out, 0, 1)


def verite_era5(sl: dict, pl: dict, zt: np.ndarray, lsm: np.ndarray, dx: float, dy: float,
                niveaux=NIVEAUX_ASL) -> np.ndarray:
    """Y (t, 4, nz, ny, nx) = u, v, w, rho aux noeuds ; 0 hors des noeuds valides.

    sl : u10, v10, u100, v100, t2m, sp (t, y, x) ; pl : p_hpa (p,) decroissant, u, v, z, w
    (omega, Pa/s) (t, p, y, x). Tableaux deja orientes sud -> nord."""
    t = sl["u10"].shape[0]
    hz = np.asarray(niveaux)[:, None, None] - zt[None]              # (nz, y, x) au-dessus du sol
    valid = noeuds_valides(zt, niveaux)
    hpl = pl["z"] / G - zt[None, None]                               # (t, p, y, x)
    rho_pl = densite_niveaux(pl["z"], pl["p_hpa"])
    w_pl = -pl["w"] / (rho_pl * G)
    cible_h = np.clip(hz, 10.0, None)                                # 5-10 m : profil log ensuite

    # u, v : 10 m, 100 m, puis les niveaux de pression au-dessus de 100 m
    h_uv = np.concatenate([np.full((1, t) + zt.shape, 10.0), np.full((1, t) + zt.shape, 100.0),
                           np.moveaxis(np.where(hpl > 100.0, hpl, np.nan), 1, 0)])
    uv = []
    for c10, c100, c in (("u10", "u100", "u"), ("v10", "v100", "v")):
        val = np.concatenate([sl[c10][None], sl[c100][None], np.moveaxis(pl[c], 1, 0)])
        uv.append(_colonnes(h_uv, val, cible_h))
    z0 = np.where(lsm >= 0.5, Z0_TERRE, Z0_MER)
    bas = (hz < 10.0)[None]
    facteur = np.log(np.clip(hz, H_MIN, None) / z0) / np.log(10.0 / z0)
    uv = [np.where(bas, a * facteur[None], a) for a in uv]

    # w : condition cinematique au sol, puis -omega / (rho g) aux niveaux de pression
    dzdy, dzdx = np.gradient(zt, dy, dx)
    w_sol = sl["u10"] * dzdx[None] + sl["v10"] * dzdy[None]
    h_w = np.concatenate([np.zeros((1, t) + zt.shape),
                          np.moveaxis(np.where(hpl > 0.0, hpl, np.nan), 1, 0)])
    w = _colonnes(h_w, np.concatenate([w_sol[None], np.moveaxis(w_pl, 1, 0)]), hz)

    # rho : sp / (R T2m) a 2 m, puis les niveaux de pression
    rho_sol = sl["sp"] / (R_AIR * sl["t2m"])
    h_r = np.concatenate([np.full((1, t) + zt.shape, 2.0),
                          np.moveaxis(np.where(hpl > 2.0, hpl, np.nan), 1, 0)])
    rho = _colonnes(h_r, np.concatenate([rho_sol[None], np.moveaxis(rho_pl, 1, 0)]), hz)

    Y = np.stack([uv[0], uv[1], w, rho], axis=1).astype(np.float32)   # (t, 4, nz, y, x)
    trou = ~np.isfinite(Y) & valid[None, None]
    if trou.any():
        raise ValueError(f"verite ERA5 incomplete : {int(trou.sum())} valeurs absentes sur "
                         "des noeuds valides (niveaux de pression manquants ?)")
    return np.where(valid[None, None], np.nan_to_num(Y), 0.0).astype(np.float32)


# --------------------------------------------------------------------------- capteurs METAR
def stations_metar(metar_brut: pd.DataFrame, lat: np.ndarray, lon: np.ndarray,
                   zt: np.ndarray) -> pd.DataFrame:
    """Une ligne par station : maille la plus proche, altitude, hauteur de l'anemometre."""
    rows = []
    demi = 0.5 * abs(float(lat[1] - lat[0]))
    for st, g in metar_brut.groupby("station_id"):
        la, lo = float(g["lat"].median()), float(g["lon"].median())
        if not (lat[0] - demi <= la <= lat[-1] + demi and lon[0] - demi <= lo <= lon[-1] + demi):
            continue
        h = g["height_agl_m"].median() if "height_agl_m" in g else np.nan
        rows.append({"station": st, "lat": la, "lon": lo,
                     "iy": int(np.abs(lat - la).argmin()), "ix": int(np.abs(lon - lo).argmin()),
                     "elevation_m": float(g["elevation_m"].median()),
                     "hauteur_anemometre_m": float(h) if np.isfinite(h) else HAUTEUR_ANEMOMETRE})
    df = pd.DataFrame(rows)
    df["z_capteur_modele_m"] = zt[df["iy"], df["ix"]] + df["hauteur_anemometre_m"]
    return df


def humidite_specifique(t_c, hr_pct, p_pa):
    e = hr_pct / 100.0 * 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
    return 0.622 * e / np.maximum(p_pa - 0.378 * e, 1.0)


def surface_capteurs(X: np.ndarray, stations: pd.DataFrame, ny: int, nx: int) -> np.ndarray:
    """X (t, S, 6) de `entree_stations` -> S (t, 8, ny, nx) au format v7.

    Canaux : u, v, w (= 0 : l'anemometre ne le mesure pas), T (degC), rho, HR, p (Pa a
    l'altitude du capteur dans le modele), capteur disponible. Une station compte si u, v,
    T et p sont presents ; HR absente -> 50 %."""
    t = X.shape[0]
    S = np.zeros((t, 8, ny, nx), np.float32)
    cumul = np.zeros((t, ny, nx), np.float32)
    for k, st in stations.reset_index(drop=True).iterrows():
        u, v, tc, p_hpa, hr = (X[:, k, i].astype(np.float64) for i in range(5))
        ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(tc) & np.isfinite(p_hpa)
        hr = np.where(np.isfinite(hr), hr, 50.0)
        tk = np.where(ok, tc, 15.0) + 273.15
        # pression de la station (altitude reelle) -> altitude du capteur dans le modele
        dz = st["z_capteur_modele_m"] - (st["elevation_m"] + st["hauteur_anemometre_m"])
        p = np.where(ok, p_hpa, 1013.25) * 100.0 * np.exp(-G * dz / (R_AIR * tk))
        q = humidite_specifique(tk - 273.15, hr, p)
        rho = p / (R_AIR * tk * (1.0 + 0.608 * q))
        j, i = int(st["iy"]), int(st["ix"])
        vals = (u, v, np.zeros(t), tk - 273.15, rho, hr, p)
        for c, a in enumerate(vals):
            S[:, c, j, i] += np.where(ok, a, 0.0).astype(np.float32)
        cumul[:, j, i] += ok
    plein = cumul > 0
    S[:, :7] = np.where(plein[:, None], S[:, :7] / np.maximum(cumul, 1)[:, None], 0.0)
    S[:, 7] = plein
    return S


# --------------------------------------------------------------------------- lecture ERA5
def _orienter(a: np.ndarray, lat_desc: bool) -> np.ndarray:
    return a[..., ::-1, :] if lat_desc else a


def lire_statique(dossier: Path):
    """lat, lon (croissantes), zt (m), lsm : grille et relief ERA5."""
    import xarray as xr

    sl_f = sorted(Path(dossier).glob("era5_sl_*.nc"))[0]
    with xr.open_dataset(sl_f) as sl:
        tdim = "valid_time" if "valid_time" in sl.dims else "time"
        lat, lon = sl["latitude"].values, sl["longitude"].values
        zt = sl["z"].isel({tdim: 0}).values / G
    with xr.open_dataset(Path(dossier) / "masque_terre_mer.nc") as m:
        lsm = np.squeeze(m["lsm"].values)
    desc = lat[0] > lat[-1]
    return (lat[::-1] if desc else lat), lon, _orienter(zt, desc).copy(), \
        _orienter(lsm, desc).copy(), desc


def periodes_era5(dossier: Path):
    """Pour chaque fichier mensuel de niveaux de pression (plx) : les instants communs avec
    les niveaux simples et la vitesse verticale (plw), lus et orientes sud -> nord."""
    import xarray as xr

    d = Path(dossier)
    sls = [xr.open_dataset(f) for f in sorted(d.glob("era5_sl_*.nc"))]
    plws = [xr.open_dataset(f) for f in sorted(d.glob("era5_plw_*.nc"))]
    if not sls or not plws:
        raise FileNotFoundError(f"era5_sl_* et era5_plw_* requis dans {d}")
    tdim = "valid_time" if "valid_time" in sls[0].dims else "time"

    def dedoublonner(ds):
        _, garde = np.unique(ds[tdim].values, return_index=True)
        return ds.isel({tdim: np.sort(garde)})

    sl_tout = dedoublonner(xr.concat([s[["u10", "v10", "u100", "v100", "t2m", "sp"]]
                                      for s in sls], dim=tdim))
    w_tout = dedoublonner(xr.concat([w[["w"]] for w in plws], dim=tdim))
    try:
        for f in sorted(d.glob("era5_plx_*.nc")):
            with xr.open_dataset(f) as pl:
                communs = np.intersect1d(np.intersect1d(pl[tdim].values, sl_tout[tdim].values),
                                         w_tout[tdim].values)
                if len(communs) == 0:
                    continue
                pdim = "pressure_level" if "pressure_level" in pl.dims else "level"
                pl = pl.sortby(pdim, ascending=False).sel({tdim: communs}).load()
                ww = w_tout.sortby(pdim, ascending=False).sel({tdim: communs}).load()
                s = sl_tout.sel({tdim: communs}).load()
                desc = pl["latitude"].values[0] > pl["latitude"].values[-1]
                yield communs.astype("datetime64[s]"), \
                    {k: _orienter(s[k].values.astype(np.float64), desc) for k in s.data_vars}, \
                    {"p_hpa": pl[pdim].values.astype(np.float64),
                     **{k: _orienter(pl[k].values.astype(np.float64), desc) for k in ("u", "v", "z")},
                     "w": _orienter(ww["w"].values.astype(np.float64), desc)}
    finally:
        for x in sls + plws:
            x.close()


def lire_statique_land(dossier: Path):
    """lat, lon (croissantes), zt (m), lsm sur la grille ERA5-Land (era5_land_fixe.nc) ; sur
    la mer, relief et masque manquants pris dans ERA5 interpole."""
    import xarray as xr

    import zipfile

    lat_c, lon_c, zt_c, lsm_c, _ = lire_statique(dossier)
    fixe = Path(dossier) / "era5_land_fixe.nc"
    if zipfile.is_zipfile(fixe):              # le CDS livre relief et masque en zip (2 netCDF)
        extrait = Path(dossier) / "era5_land_fixe"
        with zipfile.ZipFile(fixe) as z:
            z.extractall(extrait)
        fichiers = sorted(extrait.glob("*.nc"))
    else:
        fichiers = [fixe]
    champs = {}
    for fichier in fichiers:                  # relief et masque n'ont pas la meme date
        with xr.open_dataset(fichier) as f:
            lat, lon = f["latitude"].values, f["longitude"].values
            for v in f.data_vars:
                a = f[v]
                champs[v] = a.isel({d: 0 for d in a.dims if d not in ("latitude", "longitude")}).values
    zt, lsm = champs["z"] / G, champs["lsm"]
    desc = lat[0] > lat[-1]
    lat, zt, lsm = (lat[::-1] if desc else lat), _orienter(zt, desc), _orienter(lsm, desc)
    zt = np.where(np.isfinite(zt), zt, vers_grille(zt_c, lat_c, lon_c, lat, lon))
    lsm = np.where(np.isfinite(lsm), lsm, vers_grille(lsm_c, lat_c, lon_c, lat, lon))
    return lat, lon, zt.copy(), lsm.copy()


def periodes_land(dossier: Path, lat_f: np.ndarray, lon_f: np.ndarray):
    """`periodes_era5` sur la grille ERA5-Land : tout est interpole d'ERA5 vers la grille
    fine, puis le vent 10 m, T 2 m et la pression de surface sont remplaces par ERA5-Land la
    ou il a des valeurs (terres). Instants communs avec ERA5-Land seulement."""
    import xarray as xr

    d = Path(dossier)
    lat_c, lon_c, *_ = lire_statique(d)
    lands = [xr.open_dataset(f) for f in sorted(d.glob("era5_land_20*.nc"))]
    if not lands:
        raise FileNotFoundError(f"era5_land_<annee>_* requis dans {d}")
    tdim = "valid_time" if "valid_time" in lands[0].dims else "time"
    tout = xr.concat([x[["u10", "v10", "t2m", "sp"]] for x in lands], dim=tdim)
    _, garde = np.unique(tout[tdim].values, return_index=True)
    tout = tout.isel({tdim: np.sort(garde)})
    t_land = tout[tdim].values.astype("datetime64[s]")
    desc = tout["latitude"].values[0] > tout["latitude"].values[-1]
    try:
        for temps, sl, pl in periodes_era5(d):
            communs = np.intersect1d(temps, t_land)
            if len(communs) == 0:
                continue
            k = np.searchsorted(temps, communs)

            def fin(a):
                return vers_grille(a[k], lat_c, lon_c, lat_f, lon_f)

            land = tout.isel({tdim: np.searchsorted(t_land, communs)}).load()
            sl_f = {c: fin(a) for c, a in sl.items()}
            for c in ("u10", "v10", "t2m", "sp"):
                v = _orienter(land[c].values.astype(np.float64), desc)
                sl_f[c] = np.where(np.isfinite(v), v, sl_f[c])
            pl_f = {"p_hpa": pl["p_hpa"], **{c: fin(pl[c]) for c in ("u", "v", "z", "w")}}
            yield communs, sl_f, pl_f
    finally:
        for x in lands:
            x.close()


# --------------------------------------------------------------------------- construction
def construire(era5_dir: Path, metar_fichiers, data_dir: Path, cache_dir: Path,
               verbose: bool = True, grille: str = "era5") -> pd.DataFrame:
    """Jeu (data_dir) + cache (cache_dir) au format v7, sur la grille ERA5 (0,25 deg) ou
    ERA5-Land (`grille="land"`, 0,1 deg). Reprise : les instants deja dans l'index sont
    ignores."""
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if grille == "land":
        lat, lon, zt, lsm = lire_statique_land(era5_dir)
        niveaux, source = niveaux_pour(zt), "ERA5-Land + ERA5"
        periodes = periodes_land(era5_dir, lat, lon)
    else:
        lat, lon, zt, lsm, _ = lire_statique(era5_dir)
        niveaux, source = NIVEAUX_ASL, "ERA5"
        periodes = periodes_era5(era5_dir)
    ny, nx = zt.shape
    dx, dy = pas_metriques(lat, lon)

    brut = pd.concat([pd.read_csv(f) for f in metar_fichiers], ignore_index=True)
    stations = stations_metar(brut, lat, lon, zt)
    hauteur = np.full((ny, nx), HAUTEUR_ANEMOMETRE)
    hauteur[stations["iy"], stations["ix"]] = stations["hauteur_anemometre_m"]
    meta = ecrire_jeu(data_dir, lat, lon, zt, lsm, hauteur, niveaux, source)
    stations.to_csv(data_dir / "stations_metar.csv", index=False)
    metar = lire_metar(metar_fichiers)
    if verbose:
        print(f"{source} : grille {len(niveaux)} x {ny} x {nx}, relief "
              f"{zt.min():.0f} a {zt.max():.0f} m, dx {dx/1000:.1f} km, dy {dy/1000:.1f} km, "
              f"{meta['noeuds_3d']['nombre']} noeuds 3D ; {len(stations)} stations : "
              + ", ".join(stations["station"]))

    index_path = cache_dir / "index.csv"
    faits = set(pd.read_csv(index_path)["timestamp"]) if index_path.exists() else set()
    split = json.loads((data_dir / "split_days.json").read_text())
    jeu_du_jour = {d: n for n, ds in (("train", split["train_days"]), ("val", split["val_days"]),
                                      ("test", split["test_days"])) for d in ds}
    for temps, sl, pl in periodes:
        ts = pd.to_datetime(temps)
        if all(str(t) in faits for t in ts):
            continue
        Y = verite_era5(sl, pl, zt, lsm, dx, dy, niveaux)
        S = surface_capteurs(entree_stations(metar, temps, list(stations["station"])),
                             stations, ny, nx)
        rows = []
        for n, t in enumerate(ts):
            if str(t) in faits or S[n, 7].sum() < MIN_STATIONS:
                continue
            day = int((t.normalize() - JOUR_0).days)
            stem = f"{day:04d}_{t.strftime('%H%M%S')}"
            np.save(cache_dir / f"{stem}_S.npy", S[n])
            np.save(cache_dir / f"{stem}_Y.npy", Y[n])
            rows.append({"stem": stem, "day": day, "timestamp": str(t),
                         "split": jeu_du_jour.get(day, "test"),
                         "hour": t.hour + t.minute / 60, "doy": t.dayofyear})
        if rows:
            pd.DataFrame(rows).to_csv(index_path, mode="a", header=not index_path.exists(),
                                      index=False)
            faits.update(r["timestamp"] for r in rows)
        if verbose:
            print(f"{ts[0]:%Y-%m} : {len(rows)} instants sur {len(ts)} "
                  f"(au moins {MIN_STATIONS} stations)", flush=True)
    index = pd.read_csv(index_path)
    if verbose:
        print(f"cache : {len(index)} instants {index['split'].value_counts().to_dict()}")
    return index


def bilan_masse(cfg, n: int = 50) -> dict:
    """Residu de continuite de la VERITE ERA5 avec l'operateur du modele : c'est le plancher
    que la loss de masse ne peut pas descendre sans s'ecarter des donnees ; il fixe
    `loss.mass_ref`."""
    import torch

    from .cache import load_index
    from .geometry import Geometry

    geom = Geometry(cfg)
    index = load_index(cfg)
    stems = index[index["split"] == "train"]["stem"].sample(min(n, len(index)), random_state=0)
    r2, div2 = [], []
    for stem in stems:
        Y = torch.from_numpy(np.load(cfg.paths.cache_dir / f"{stem}_Y.npy"))[None]
        r = geom.mass_residual(Y[:, 3], Y[:, 0], Y[:, 1], Y[:, 2])[:, geom.mass_mask_t]
        d = geom.mass_residual(Y[:, 3], Y[:, 0], Y[:, 1], torch.zeros_like(Y[:, 2]))
        r2.append(float((r ** 2).mean()))
        div2.append(float((d[:, geom.mass_mask_t] ** 2).mean()))
    return {"instants": len(r2), "residu_rms": float(np.sqrt(np.mean(r2))),
            "residu_rms_sans_w": float(np.sqrt(np.mean(div2))),
            "residu_moyen_carre": float(np.mean(r2))}
