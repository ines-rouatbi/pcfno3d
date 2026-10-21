"""Archives du modele reel (Grand Tunis) a deposer dans MyDrive/pcfno3d_grand_tunis/.

    python scripts/package_grand_tunis_colab.py
    -> artifacts/grand_tunis_donnees.zip : ce que lisent les carnets Colab (grille reelle,
       forcage ERA5, stations METAR / SYNOP, radiosondages) ; a redeposer quand le forcage
       est refait (nouveaux mois ERA5) ;
       artifacts/grand_tunis_sources.zip : toutes les donnees brutes utilisees (ERA5 et
       ERA5-Land .nc, tuiles Copernicus DEM et ESA WorldCover), pour l'archive et la tracabilite.
Jamais de cle, de jeton ni de fichier de configuration du CDS.
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "pcfno3d"
INTERDITS = {".cdsapirc", ".ecmwfdatastoresrc", "api_keys.json", ".env", ".dev.vars"}


def ajouter(zf: zipfile.ZipFile, fichiers, racine: Path, prefixe: str) -> int:
    n = 0
    for f in sorted(fichiers):
        if f.is_file() and f.name not in INTERDITS and not f.name.endswith((".part", ".tmp")):
            zf.write(f, f"{prefixe}/{f.relative_to(racine).as_posix()}")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--sortie", default=str(REPO / "artifacts"))
    args = ap.parse_args()
    base, out = Path(args.base), Path(args.sortie)
    out.mkdir(parents=True, exist_ok=True)

    gt = base / "grand_tunis"
    don = out / "grand_tunis_donnees.zip"
    with zipfile.ZipFile(don, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        n = ajouter(zf, [gt / x for x in ("aneo_grid_points_surface.csv", "grille_reelle.json",
                                           "forcage_era5.csv", "forcage_era5.npz", "apercu.png")],
                    gt, "grand_tunis")
        n += ajouter(zf, (gt / "brise_mer").glob("*.png"), gt, "grand_tunis")
        n += ajouter(zf, (gt / "brise_mer").glob("*.json"), gt, "grand_tunis")
        n += ajouter(zf, (base / "metar" / "historique").glob("metar_*.csv"), base / "metar" / "historique", "metar")
        n += ajouter(zf, (base / "isd").glob("synop_*.csv"), base / "isd", "isd")
        n += ajouter(zf, (base / "igra").glob("*.csv"), base / "igra", "igra")
    print(f"{don} : {n} fichiers, {don.stat().st_size / 1e6:.0f} Mo")

    src = out / "grand_tunis_sources.zip"
    with zipfile.ZipFile(src, "w", compression=zipfile.ZIP_STORED) as zf:
        e = base / "era5" / "tunisie"
        n = ajouter(zf, e.rglob("*"), e, "era5")
        n += ajouter(zf, (base / "dem" / "brut").glob("*.tif"), base / "dem" / "brut", "dem")
    print(f"{src} : {n} fichiers, {src.stat().st_size / 1e6:.0f} Mo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
