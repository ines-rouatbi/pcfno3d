"""Archive des donnees du modele national pour Colab : national_donnees.zip, a deposer dans
MyDrive/pcfno3d_national/ (lue par scripts/make_national_nb.py).

    python scripts/package_national_colab.py --grille artifacts/national/grille \
        --forcage artifacts/national/forcage_era5.csv --sortie artifacts/national/national_donnees.zip

Contenu (dossier `national/` de l'archive) : aneo_grid_points_surface.csv et grille_reelle.json
(scripts/grille_reelle.py), forcage_era5.csv et .npz (scripts/forcage_era5.py --niveaux ...).
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
    sortie = Path(args.sortie)
    tmp = sortie.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for nom, p in fichiers.items():
            zf.write(p, f"national/{nom}")
    tmp.replace(sortie)
    print(f"-> {sortie} ({sortie.stat().st_size / 1e6:.0f} Mo), a deposer dans MyDrive/pcfno3d_national/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
