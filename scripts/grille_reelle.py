"""Grille de surface REELLE pour le generateur ANEO 3D (scripts/generate_aneo_3d.py --input-grid).

Remplace le relief synthetique par le relief et l'occupation du sol reels d'une zone de
Tunisie (par defaut le Grand Tunis, 68 x 56 km a 50 m : 1360 x 1120 points, avec trois
stations reelles : Tunis-Carthage, Zaghouan, Nabeul) :
    relief   : Copernicus DEM GLO-30 (1 seconde d'arc, ~30 m), lisse puis interpole a 50 m ;
    rugosite : ESA WorldCover 2021 (10 m), z0 par classe, moyenne logarithmique sur la maille ;
    capteurs : hauteur des mats ANEO simulee (20-30 m), comme le generateur.
Projection locale : x vers l'Est, y vers le Nord, en metres depuis le coin sud-ouest, pas
metrique constant pris a la latitude centrale (comme pcfno3d.tunisie_fno3d.pas_metriques).

    python scripts/grille_reelle.py --dem <dossier des tuiles> --sortie <dossier>
    # tuiles : Copernicus_DSM_COG_10_N36_00_E010_00_DEM.tif (aws s3 copernicus-dem-30m) et
    #          ESA_WorldCover_10m_2021_v200_N36E009_Map.tif (aws s3 esa-worldcover)

Sorties : aneo_grid_points_surface.csv (colonnes du generateur + lat, lon), grille_reelle.json
(georeference, sources et attributions, stations reelles placees sur la grille), apercu.png.
Les tuiles telechargees sont des donnees : elles sont lues, jamais executees.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# classes ESA WorldCover -> (code d'occupation du generateur, z0 en m)
#   generateur : 1 eau, 2 plaine, 3 agricole, 4 foret, 5 urbain (z0 0,0002 / 0,03 / 0,1 / 1,0 / 1,5)
WORLDCOVER = {10: (4, 1.0), 20: (2, 0.2), 30: (2, 0.03), 40: (3, 0.1), 50: (5, 1.5),
              60: (2, 0.01), 70: (2, 0.001), 80: (1, 0.0002), 90: (1, 0.05), 95: (4, 0.5),
              100: (2, 0.03), 0: (1, 0.0002)}
NOMS = {1: "eau", 2: "plaine", 3: "agricole", 4: "foret", 5: "urbain"}
# stations reelles, coordonnees verifiees contre le relief Copernicus et l'altitude officielle :
#   Zaghouan : catalogue INM (36,3667 N 10,1167 E) -> 959 m, le sommet du djebel (arrondi faux) ;
#              NOAA ISD (36,433 N 10,083 E) -> 141 m pour 156 m officiels : retenu ;
#   Nabeul   : catalogue INM (36,45 N 10,7333 E) -> 9 m pour 1 m officiel : retenu ;
#              NOAA ISD (36,467 N 10,700 E) -> 55 m, douteux.
STATIONS = {"DTTA Tunis-Carthage (METAR, INM 60715, radiosondage IGRA TSM00060715)": (36.8504, 10.2156),
            "Zaghouan-Mograne (synop INM 60729, NOAA ISD 607290)": (36.433, 10.083),
            "Nabeul (synop INM 60728, NOAA ISD 607280)": (36.45, 10.7333),
            "Bizerte (INM 60714)": (37.25, 9.8), "Kelibia (INM 60720)": (36.85, 11.0833)}
SOURCES = {
    "relief": "Copernicus DEM GLO-30, doi:10.5270/ESA-c5d3d65 - (c) DLR e.V. 2010-2014 and (c) "
              "Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European "
              "Union and ESA; all rights reserved",
    "occupation_du_sol": "ESA WorldCover 10 m 2021 v200, doi:10.5281/zenodo.7254221 (CC BY 4.0)",
}


def metres_par_degre(lat_deg: float) -> tuple[float, float]:
    lat0 = math.radians(lat_deg)
    m_lon = 111412.84 * math.cos(lat0) - 93.5 * math.cos(3 * lat0)
    m_lat = 111132.92 - 559.82 * math.cos(2 * lat0) + 1.175 * math.cos(4 * lat0)
    return m_lon, m_lat


def _tuile(brut: bytes, p, th: int, tw: int) -> np.ndarray:
    """Une tuile deflate d'un GeoTIFF, decodee sans imagecodecs : zlib, puis le predicteur
    TIFF 3 (flottants : differences d'octets par ligne, octets ranges par plans, poids fort
    d'abord) ou 2 (differences horizontales d'entiers)."""
    import zlib

    if int(p.compression) != 8:
        raise ValueError(f"compression TIFF {p.compression} non geree (deflate attendu)")
    raw = np.frombuffer(zlib.decompress(brut), dtype=np.uint8)
    taille = np.dtype(p.dtype).itemsize
    pred = int(p.predictor)
    if pred == 3:
        octets = np.cumsum(raw.reshape(th, tw * taille), axis=1, dtype=np.uint8)
        plans = octets.reshape(th, taille, tw).transpose(0, 2, 1)       # (th, tw, octets)
        return np.ascontiguousarray(plans).view(np.dtype(p.dtype).newbyteorder(">"))[..., 0]             .astype(p.dtype)
    vals = raw.view(p.dtype).reshape(th, tw)
    if pred == 2:
        vals = np.cumsum(vals, axis=1, dtype=p.dtype)
    return vals


def lire_fenetre(chemin: Path, lat_s: float, lat_n: float, lon_o: float, lon_e: float):
    """Fenetre d'un GeoTIFF tuile (COG) en degres, sans charger l'image entiere.
    -> (tableau, lat du bord nord, lon du bord ouest, pas en degres)."""
    import tifffile

    with tifffile.TiffFile(chemin) as t:
        p = t.pages[0]
        lon0, lat0 = p.tags["ModelTiepointTag"].value[3:5]
        pas = p.tags["ModelPixelScaleTag"].value[0]
        r0, r1 = int((lat0 - lat_n) / pas), int(math.ceil((lat0 - lat_s) / pas))
        c0, c1 = int((lon_o - lon0) / pas), int(math.ceil((lon_e - lon0) / pas))
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, p.shape[0]), min(c1, p.shape[1])
        th, tw = p.tilelength, p.tilewidth
        ntx = math.ceil(p.shape[1] / tw)
        out = np.zeros((r1 - r0, c1 - c0), dtype=p.dtype)
        fh = t.filehandle
        for ty in range(r0 // th, (r1 - 1) // th + 1):
            for tx in range(c0 // tw, (c1 - 1) // tw + 1):
                k = ty * ntx + tx
                fh.seek(p.dataoffsets[k])
                seg = _tuile(fh.read(p.databytecounts[k]), p, th, tw)
                ya, yb = max(r0, ty * th), min(r1, (ty + 1) * th)
                xa, xb = max(c0, tx * tw), min(c1, (tx + 1) * tw)
                out[ya - r0:yb - r0, xa - c0:xb - c0] = seg[ya - ty * th:yb - ty * th,
                                                           xa - tx * tw:xb - tx * tw]
    return out, lat0 - r0 * pas, lon0 + c0 * pas, pas


def echantillonner(tab, lat_n, lon_o, pas, lat, lon, ordre=1):
    """Valeurs aux points (lat, lon) : bilineaire (ordre 1) ou plus proche voisin (0)."""
    fy = (lat_n - lat) / pas - 0.5           # centres de pixels
    fx = (lon - lon_o) / pas - 0.5
    if ordre == 0:
        i = np.clip(np.rint(fy).astype(int), 0, tab.shape[0] - 1)
        j = np.clip(np.rint(fx).astype(int), 0, tab.shape[1] - 1)
        return tab[i, j]
    i0 = np.clip(np.floor(fy).astype(int), 0, tab.shape[0] - 2)
    j0 = np.clip(np.floor(fx).astype(int), 0, tab.shape[1] - 2)
    a, b = np.clip(fy - i0, 0, 1), np.clip(fx - j0, 0, 1)
    return ((1 - a) * (1 - b) * tab[i0, j0] + (1 - a) * b * tab[i0, j0 + 1]
            + a * (1 - b) * tab[i0 + 1, j0] + a * b * tab[i0 + 1, j0 + 1])


def construire(dem_dir: Path, lat_sud: float, lon_ouest: float, nx: int, ny: int, dx: float,
               graine: int = 2026) -> tuple[pd.DataFrame, dict]:
    from scipy.ndimage import gaussian_filter

    lat_c = lat_sud + ny * dx / 2 / metres_par_degre(lat_sud)[1]
    m_lon, m_lat = metres_par_degre(lat_c)
    lat_n, lon_e = lat_sud + ny * dx / m_lat, lon_ouest + nx * dx / m_lon
    marge = 0.01
    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x, y = ix * dx, iy * dx
    lat, lon = lat_sud + y / m_lat, lon_ouest + x / m_lon

    # relief : tuile(s) Copernicus 1 deg x 1 deg ; lissage d'environ un demi-pas de grille
    zt = np.full((ny, nx), np.nan)
    for la in range(int(math.floor(lat_sud)), int(math.floor(lat_n)) + 1):
        for lo in range(int(math.floor(lon_ouest)), int(math.floor(lon_e)) + 1):
            f = dem_dir / f"Copernicus_DSM_COG_10_N{la:02d}_00_E{lo:03d}_00_DEM.tif"
            if not f.exists():
                raise FileNotFoundError(f"tuile de relief absente : {f.name}")
            dans = (lat >= la) & (lat < la + 1) & (lon >= lo) & (lon < lo + 1)
            if not dans.any():
                continue
            tab, n_, o_, pas = lire_fenetre(f, max(lat_sud - marge, la), min(lat_n + marge, la + 1),
                                           max(lon_ouest - marge, lo), min(lon_e + marge, lo + 1))
            tab = gaussian_filter(tab.astype(np.float64), sigma=0.5 * dx / (pas * m_lat))
            zt[dans] = echantillonner(tab, n_, o_, pas, lat[dans], lon[dans])
    if np.isnan(zt).any():
        raise ValueError("relief incomplet sur la grille")

    # occupation du sol : 5 x 5 sous-points par maille, z0 en moyenne logarithmique
    f = next(dem_dir.glob("ESA_WorldCover_10m_2021_v200_*_Map.tif"))
    tab, n_, o_, pas = lire_fenetre(f, lat_sud - marge, lat_n + marge, lon_ouest - marge, lon_e + marge)
    sous = (np.arange(5) + 0.5) / 5 - 0.5
    classes = np.stack([echantillonner(tab, n_, o_, pas, lat + sy * dx / m_lat, lon + sx * dx / m_lon, 0)
                        for sy in sous for sx in sous])                  # (25, ny, nx)
    z0_tab = np.vectorize(lambda c: WORLDCOVER.get(int(c), (2, 0.03))[1])(classes)
    code_tab = np.vectorize(lambda c: WORLDCOVER.get(int(c), (2, 0.03))[0])(classes)
    z0 = np.exp(np.log(z0_tab).mean(axis=0))
    occ = np.apply_along_axis(lambda v: np.bincount(v, minlength=6).argmax(), 0, code_tab)
    zt = np.where((occ == 1) & (zt < 1.0) & (zt > -1.0), 0.0, zt)       # mer et lagunes a 0 m

    gy, gx = np.gradient(zt, dx, dx)
    pente = np.degrees(np.arctan(np.hypot(gx, gy)))
    orient = np.degrees(np.arctan2(-gx, -gy)) % 360.0
    zrel = np.random.default_rng(graine).uniform(20.0, 30.0, size=(ny, nx))
    df = pd.DataFrame({
        "point_id": (iy * nx + ix + 1).ravel(), "ix": ix.ravel(), "iy": iy.ravel(),
        "x_m": x.ravel().astype(np.float64), "y_m": y.ravel().astype(np.float64),
        "z_terrain_m": np.round(zt.ravel(), 2), "z_rel_m": np.round(zrel.ravel(), 2),
        "z0_m": np.round(z0.ravel(), 5), "slope_deg": np.round(pente.ravel(), 3),
        "aspect_deg": np.round(orient.ravel(), 3), "land_use_code": occ.ravel(),
        "terrain_type": pd.Series(occ.ravel()).map(NOMS).values,
        "lat": np.round(lat.ravel(), 6), "lon": np.round(lon.ravel(), 6)})
    df["geom_wkt"] = "POINT (" + df["x_m"].map("{:.1f}".format) + " " + df["y_m"].map("{:.1f}".format) + ")"

    stations = {}
    for nom, (la, lo) in STATIONS.items():
        j, i = (la - lat_sud) * m_lat / dx, (lo - lon_ouest) * m_lon / dx
        dedans = 0 <= j < ny and 0 <= i < nx
        stations[nom] = {"lat": la, "lon": lo, "dans_la_zone": bool(dedans),
                         **({"ix": int(i), "iy": int(j), "z_terrain_m": float(zt[int(j), int(i)])}
                            if dedans else {})}
    info = {
        "zone": {"lat_sud": lat_sud, "lat_nord": round(lat_n, 6), "lon_ouest": lon_ouest,
                 "lon_est": round(lon_e, 6), "nx": nx, "ny": ny, "dx_m": dx,
                 "taille_km": [nx * dx / 1000, ny * dx / 1000]},
        "projection": {"type": "plan local, pas metrique constant a la latitude centrale",
                       "lat_centrale": round(lat_c, 6), "m_par_deg_lon": m_lon, "m_par_deg_lat": m_lat,
                       "lat": "lat_sud + y_m / m_par_deg_lat", "lon": "lon_ouest + x_m / m_par_deg_lon"},
        "f_coriolis": 2 * 7.2921e-5 * math.sin(math.radians(lat_c)),
        "relief_m": {"min": float(zt.min()), "max": float(zt.max()), "moyenne": float(zt.mean())},
        "part_eau": float((occ == 1).mean()),
        "occupation": {NOMS[k]: float((occ == k).mean()) for k in NOMS},
        "stations": stations, "sources": SOURCES,
    }
    return df, info


def apercu(df: pd.DataFrame, info: dict, chemin: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nx, ny = info["zone"]["nx"], info["zone"]["ny"]
    zt = df["z_terrain_m"].to_numpy().reshape(ny, nx)
    occ = df["land_use_code"].to_numpy().reshape(ny, nx)
    ext = [0, nx * info["zone"]["dx_m"] / 1000, 0, ny * info["zone"]["dx_m"] / 1000]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    im = ax[0].imshow(np.where(occ == 1, np.nan, zt), origin="lower", extent=ext, cmap="terrain")
    fig.colorbar(im, ax=ax[0], label="relief (m)")
    ax[1].imshow(occ, origin="lower", extent=ext, cmap="tab10", vmin=0, vmax=9)
    for a, titre in zip(ax, ("Relief Copernicus GLO-30", "Occupation ESA WorldCover (1 eau ... 5 urbain)")):
        a.set_title(titre); a.set_xlabel("x (km)"); a.set_ylabel("y (km)")
        for nom, s in info["stations"].items():
            if s["dans_la_zone"]:
                a.plot(s["ix"] * info["zone"]["dx_m"] / 1000, s["iy"] * info["zone"]["dx_m"] / 1000, "r^")
                a.annotate(nom.split(" (")[0], (s["ix"] * info["zone"]["dx_m"] / 1000,
                                               s["iy"] * info["zone"]["dx_m"] / 1000), color="r")
    fig.tight_layout()
    fig.savefig(chemin, dpi=110)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dem", required=True, help="dossier des tuiles Copernicus et WorldCover")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--lat-sud", type=float, default=36.395)
    ap.add_argument("--lon-ouest", type=float, default=10.04)
    ap.add_argument("--nx", type=int, default=1360)
    ap.add_argument("--ny", type=int, default=1120)
    ap.add_argument("--dx", type=float, default=50.0)
    args = ap.parse_args()
    df, info = construire(Path(args.dem), args.lat_sud, args.lon_ouest, args.nx, args.ny, args.dx)
    out = Path(args.sortie)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "aneo_grid_points_surface.csv", index=False, lineterminator="\n")
    (out / "grille_reelle.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    apercu(df, info, out / "apercu.png")
    print(json.dumps({k: info[k] for k in ("zone", "relief_m", "part_eau", "occupation", "stations")},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
