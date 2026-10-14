"""Telecharge ERA5 (Copernicus CDS) sur le domaine : fond d'altitude reel, forcage WRF.

Pre-requis (une fois, par l'utilisateur, JAMAIS dans le depot ni dans le salon) :
    fichier ~/.cdsapirc  (Windows : C:\\Users\\<toi>\\.cdsapirc) contenant
        url: https://cds.climate.copernicus.eu/api
        key: <ton Personal Access Token>
    + licences acceptees sur les pages des jeux ERA5 (onglet Download).

    python scripts/download_reanalysis.py --test
        1 heure, petite zone autour de Tunis : verifie le compte et les licences (~1 Mo)

    python scripts/download_reanalysis.py --start 2025-01-01 --end 2025-12-31
        zone = domaine georeference (config real.origin_*) + marge ; un fichier par mois
        et par produit dans artifacts/reanalysis/

    python scripts/download_reanalysis.py --bbox 37.2,9.8,36.4,10.6 --start ... --end ...
        zone explicite N,O,S,E en degres, si le domaine n'est pas encore georeference

Produits : ERA5 niveaux de pression 1000 a 850 hPa (~0 a 1500 m : u, v, w, T, geopotentiel)
et ERA5 niveaux simples (vent 10 m et 100 m, T 2 m, pressions, orographie). ERA5 est a
~31 km : quelques points autour d'un domaine de 50 x 25 km. C'est un FOND d'altitude, pas
une verite a 100 m. CERRA (5,5 km) se demande de la meme facon ; copier la requete exacte
depuis la page du jeu (bouton « Show API request ») avant de l'ajouter ici.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]

PRESSURE_LEVELS = ["850", "875", "900", "925", "950", "975", "1000"]
PL_VARS = ["u_component_of_wind", "v_component_of_wind", "vertical_velocity",
           "temperature", "geopotential"]
SL_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind",
           "100m_u_component_of_wind", "100m_v_component_of_wind", "2m_temperature",
           "surface_pressure", "mean_sea_level_pressure", "geopotential"]
TUNIS = (36.85, 10.23)               # aeroport de Tunis-Carthage, pour --test


def domain_bbox(cfg, margin_deg: float) -> list[float]:
    """[N, O, S, E] couvrant le domaine georeference, plus une marge."""
    from pcfno3d.geometry import Geometry
    from pcfno3d.realdata import GeoRef

    geo = GeoRef.from_cfg(cfg)
    geom = Geometry(cfg, device="cpu")
    xs = [0, geom.nx * geom.dx, 0, geom.nx * geom.dx]
    ys = [0, 0, geom.ny * geom.dy, geom.ny * geom.dy]
    lat, lon = geo.to_latlon(xs, ys)
    return [round(float(lat.max()) + margin_deg, 2), round(float(lon.min()) - margin_deg, 2),
            round(float(lat.min()) - margin_deg, 2), round(float(lon.max()) + margin_deg, 2)]


def request(dataset: str, variables, year: int, month: int, days, hours, area,
            levels=None) -> dict:
    req = {"product_type": ["reanalysis"], "variable": list(variables),
           "year": [str(year)], "month": [f"{month:02d}"], "day": [f"{d:02d}" for d in days],
           "time": [f"{h:02d}:00" for h in hours], "area": area,
           "data_format": "netcdf", "download_format": "unarchived"}
    if levels:
        req["pressure_level"] = list(levels)
    return req


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="local_v7")
    ap.add_argument("--test", action="store_true", help="1 heure autour de Tunis (verifie le compte)")
    ap.add_argument("--bbox", help="N,O,S,E en degres (sinon : domaine georeference)")
    ap.add_argument("--margin", type=float, default=0.75, help="marge autour du domaine (deg)")
    ap.add_argument("--start", help="AAAA-MM-JJ")
    ap.add_argument("--end", help="AAAA-MM-JJ (inclus)")
    ap.add_argument("--hours", default="0-23", help="ex. 0-23, ou 0,3,6,9,12,15,18,21")
    ap.add_argument("--products", default="pl,sl", help="pl (niveaux de pression), sl, ou les deux")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "reanalysis"))
    ap.add_argument("--chunk-days", type=int, default=4,
                    help="jours par requete (le CDS refuse les trop grosses)")
    args = ap.parse_args()

    try:
        import cdsapi
    except ImportError:
        print("pip install cdsapi xarray netCDF4", file=sys.stderr)
        return 2
    if not (Path.home() / ".cdsapirc").exists():
        print(f"Fichier {Path.home() / '.cdsapirc'} absent : cree-le toi-meme (url + key, voir "
              "l'en-tete de ce script). La cle ne se colle ni dans le salon, ni dans le depot, "
              "ni dans une conversation.", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    if args.test:
        lat, lon = TUNIS
        area = [lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25]
        target = out / "test_era5_tunis.nc"
        client.retrieve("reanalysis-era5-single-levels",
                        request("sl", ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                                2025, 1, [1], [12], area), str(target))
        print(f"COMPTE OK : {target} ({target.stat().st_size / 1e3:.0f} ko)")
        return 0

    if not (args.start and args.end):
        ap.error("--start et --end sont obligatoires (ou --test)")
    if args.bbox:
        area = [float(v) for v in args.bbox.split(",")]
    else:
        from pcfno3d.config import Config
        area = domain_bbox(Config.load(args.config), args.margin)
    if "-" in args.hours:
        a, b = (int(v) for v in args.hours.split("-"))
        hours = list(range(a, b + 1))
    else:
        hours = [int(v) for v in args.hours.split(",")]
    products = set(args.products.split(","))
    print(f"zone N,O,S,E = {area} | {args.start} -> {args.end} | heures {hours} | {products}")

    days = pd.date_range(args.start, args.end, freq="D")
    for (year, month), grp in days.to_series().groupby([days.year, days.month]):
        dd = sorted(grp.dt.day.tolist())
        for prod in sorted(products):
            # le CDS refuse les requetes trop grosses (« cost limits exceeded ») : un mois
            # entier de niveaux de pression horaires depasse la limite. On decoupe en blocs
            # de jours, et on coupe un bloc en deux chaque fois que le CDS le refuse.
            pending = [dd[i:i + args.chunk_days] for i in range(0, len(dd), args.chunk_days)]
            while pending:
                block = pending.pop(0)
                target = out / f"era5_{prod}_{year}{month:02d}_{block[0]:02d}-{block[-1]:02d}.nc"
                if target.exists():
                    print(f"  deja la : {target.name}")
                    continue
                if prod == "pl":
                    req = request("pl", PL_VARS, year, month, block, hours, area,
                                  PRESSURE_LEVELS)
                    name = "reanalysis-era5-pressure-levels"
                else:
                    req = request("sl", SL_VARS, year, month, block, hours, area)
                    name = "reanalysis-era5-single-levels"
                tmp = target.with_suffix(".part")
                try:
                    client.retrieve(name, req, str(tmp))
                except Exception as exc:              # noqa: BLE001
                    if "cost limits" in str(exc) and len(block) > 1:
                        half = len(block) // 2
                        print(f"  trop gros pour le CDS ({len(block)} jours) : en deux")
                        pending[:0] = [block[:half], block[half:]]
                        continue
                    raise
                tmp.replace(target)
                print(f"  {target.name} ({target.stat().st_size / 1e6:.1f} Mo)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
