"""Archive des donnees du modele national pour Colab : national_donnees.zip, a deposer dans
MyDrive/pcfno3d_national/ (lue par scripts/make_national_nb.py).

    python scripts/package_national_colab.py --grille artifacts/national/grille \
        --forcage artifacts/national/forcage_era5.csv --sortie artifacts/national/national_donnees.zip

Contenu (dossier `national/` de l'archive) : aneo_grid_points_surface.csv et grille_reelle.json
(scripts/grille_reelle.py), forcage_era5.csv et .npz (scripts/forcage_era5.py --niveaux ...).
Option --obs-supp : synop_*.csv des stations de validation SUPPLEMENTAIRES (scripts/stations_supplementaires.py)
-> national/obs_supp/ (+ la liste configs/national/stations_supplementaires.csv) ; jamais en entree du modele,
lues seulement par scripts/valider_grand_tunis_reel.py --stations-supp.
Refuse une grille et un forcage dont les niveaux ne correspondent pas.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grille", required=True, help="dossier de sortie de grille_reelle.py")
    ap.add_argument("--forcage", required=True, help="forcage_era5.csv (le .npz a cote)")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--obs", default=None,
                    help="dossier des mesures reelles (metar_*.csv, radiosondages_*.csv) -> national/obs/")
    ap.add_argument("--obs-supp", default=None,
                    help="dossier des synop_*.csv des stations supplementaires -> national/obs_supp/")
    ap.add_argument("--niveaux", default="25,50,75,100,150,200,300,400,500,650,800,1000,1250,1500,2000,2500")
    args = ap.parse_args()
    g, f = Path(args.grille), Path(args.forcage)
    fichiers = {"aneo_grid_points_surface.csv": g / "aneo_grid_points_surface.csv",
                "grille_reelle.json": g / "grille_reelle.json",
                "forcage_era5.csv": f, "forcage_era5.npz": f.with_suffix(".npz")}
    manque = [str(p) for p in fichiers.values() if not p.exists()]
    if manque:
        raise SystemExit(f"fichiers absents : {manque}")
    niveaux = [int(float(z)) for z in args.niveaux.split(",")]
    colonnes = set(pd.read_csv(f, nrows=1).columns)
    absents = [z for z in niveaux if f"u_era5_{z}" not in colonnes]
    if absents:
        raise SystemExit(f"forcage sans les niveaux {absents} : relancer forcage_era5.py --niveaux {args.niveaux}")
    with np.load(f.with_suffix(".npz")) as d:
        print("hauteurs par maille :", d["hauteurs_sol"].tolist())
    info = json.loads(fichiers["grille_reelle.json"].read_text(encoding="utf-8"))
    print("domaine :", info["zone"], "| stations :", sum(s["dans_la_zone"] for s in info["stations"].values()))
    obs = []
    if args.obs:
        obs = sorted(Path(args.obs).glob("metar_*.csv")) + sorted(Path(args.obs).glob("radiosondages_*.csv"))
    if args.obs and not obs:
        raise SystemExit(f"aucun metar_*.csv / radiosondages_*.csv dans {args.obs}")
    supp = sorted(Path(args.obs_supp).glob("synop_*.csv")) if args.obs_supp else []
    if args.obs_supp and not supp:
        raise SystemExit(f"aucun synop_*.csv dans {args.obs_supp}")
    sortie = Path(args.sortie)
    tmp = sortie.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for nom, p in fichiers.items():
            zf.write(p, f"national/{nom}")
        for p in obs:
            zf.write(p, f"national/obs/{p.name}")
        for p in supp:
            zf.write(p, f"national/obs_supp/{p.name}")
        if supp:
            zf.write(Path(__file__).resolve().parents[1] / "configs" / "national" / "stations_supplementaires.csv",
                     "national/obs_supp/stations_supplementaires.csv")
    tmp.replace(sortie)
    print(f"-> {sortie} ({sortie.stat().st_size / 1e6:.0f} Mo), a deposer dans MyDrive/pcfno3d_national/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
