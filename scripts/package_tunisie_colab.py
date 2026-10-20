"""Archive des DONNEES BRUTES de PC-FNO3D v9 (ERA5-Land, ERA5, METAR, radiosondages), pour Colab.

On n'envoie pas le cache (plusieurs Go) : Colab le reconstruit a partir des fichiers bruts
(< 1 Go). L'archive se depose une fois sur le Drive, dans MyDrive/pcfno3d_tunisie/ ; le code
part a part dans pcfno3d_code.zip (scripts/package_for_colab.py).

    python scripts/package_tunisie_colab.py
    # -> artifacts/tunisie_donnees.zip : era5/*.nc (ERA5, ERA5-Land), metar/metar_*.csv et
    #    igra/radiosondages_*.csv (scripts/igra_radiosondages.py)
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "pcfno3d"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--era5", default=str(BASE / "era5" / "tunisie"))
    ap.add_argument("--metar", default=str(BASE / "metar" / "historique"))
    ap.add_argument("--igra", default=str(BASE / "igra"))
    ap.add_argument("--sortie", default=str(REPO / "artifacts" / "tunisie_donnees.zip"))
    args = ap.parse_args()

    era5 = sorted(Path(args.era5).glob("era5_*.nc")) + [Path(args.era5) / "masque_terre_mer.nc"]
    metar = sorted(Path(args.metar).glob("metar_*.csv"))
    igra = sorted(Path(args.igra).glob("radiosondages_*.csv"))
    sortie = Path(args.sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    tmp = sortie.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w") as zf:
        for f in era5:                       # netCDF deja compresses : stockes tels quels
            zf.write(f, f"era5/{f.name}", compress_type=zipfile.ZIP_STORED)
        for f in metar:
            zf.write(f, f"metar/{f.name}", compress_type=zipfile.ZIP_DEFLATED)
        for f in igra:
            zf.write(f, f"igra/{f.name}", compress_type=zipfile.ZIP_DEFLATED)
    tmp.replace(sortie)
    print(f"{sortie} : {len(era5)} fichiers ERA5, {len(metar)} fichiers METAR, {len(igra)} radiosondages, "
          f"{sortie.stat().st_size / 1e6:.0f} Mo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
