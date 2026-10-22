"""Radiosondages reels (IGRA v2, NOAA NCEI, gratuit) -> profils verticaux de vent en CSV.

Seule mesure REELLE du profil vertical du vent en Tunisie : les ballons-sondes de
Tunis-Carthage (OMM 60715) et de Tozeur (60760), en general a 00 et 12 UTC. Sert a VALIDER
le profil reconstruit par PC-FNO3D (et ERA5), jamais d'entree du modele.

    python scripts/igra_radiosondages.py --brut <dossier des TSM000607xx-data.txt> \
        --debut 2021-01-01 --fin 2025-12-31 --sortie <dossier>

Sortie : radiosondages_<station>.csv, une ligne par niveau avec vent :
station, temps (UTC, heure nominale), lancement (UTC si connu), p_pa, z_asl_m (hauteur
geopotentielle), h_agl_m (au-dessus de la station), t_c, hr_pct, dir_deg, vit_ms, u, v,
type_niveau (1x standard, 2x autre ; x1 surface, x0 non) — format IGRA v2 (igra2-data-format).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALTITUDE = {"TSM00060715": 4.0, "TSM00060760": 86.0}      # m, liste des stations IGRA


def _val(s: str, echelle: float = 1.0) -> float:
    s = s.strip()
    if not s:
        return np.nan
    v = float(s.rstrip("ABab"))                                # drapeaux de qualite colles
    return np.nan if v in (-9999.0, -8888.0) else v / echelle


def lire_igra(chemin: Path, debut: str, fin: str) -> pd.DataFrame:
    """Fichier IGRA v2 (format a colonnes fixes) -> une ligne par niveau avec vent."""
    t0, t1 = pd.Timestamp(debut), pd.Timestamp(fin) + pd.Timedelta(days=1)
    lignes, garder, tete = [], False, {}
    with open(chemin, encoding="ascii", errors="replace") as f:
        for l in f:
            if l.startswith("#"):
                an, mo, jo, he = int(l[13:17]), int(l[18:20]), int(l[21:23]), int(l[24:26])
                if he == 99:                                   # heure nominale inconnue
                    garder = False
                    continue
                t = pd.Timestamp(an, mo, jo, he)
                garder = t0 <= t < t1
                rel = l[27:31]
                lanc = (pd.Timestamp(an, mo, jo, int(rel[:2]), int(rel[2:]))
                        if rel[:2] != "99" and rel[2:] != "99" else pd.NaT)
                # RELTIME n'a que l'heure : un ballon de 00 UTC lache a 22 h 30 l'a ete la
                # VEILLE (sinon le lancement tomberait ~22 h apres l'heure nominale)
                if pd.notna(lanc) and lanc - t > pd.Timedelta(hours=12):
                    lanc -= pd.Timedelta(days=1)
                tete = {"station": l[1:12], "temps": t, "lancement": lanc}
                continue
            if not garder:
                continue
            vit = _val(l[46:51], 10.0)
            dirn = _val(l[40:45])
            if not (np.isfinite(vit) and np.isfinite(dirn)):
                continue
            lignes.append({**tete, "type_niveau": l[0:2], "p_pa": _val(l[9:15]),
                           "z_asl_m": _val(l[16:21]), "t_c": _val(l[22:27], 10.0),
                           "hr_pct": _val(l[28:33], 10.0), "dir_deg": dirn, "vit_ms": vit})
    df = pd.DataFrame(lignes)
    if df.empty:
        return df
    # niveaux de vent sans hauteur (la moitie : niveaux significatifs du vent) : hauteur
    # interpolee en ln(p) entre les niveaux voisins du meme radiosondage
    for _, i in df.groupby("temps").groups.items():
        g = df.loc[i]
        ok = g["z_asl_m"].notna() & g["p_pa"].notna()
        if ok.sum() >= 2:
            lp = -np.log(g["p_pa"])
            z = np.interp(lp, lp[ok].sort_values(), g.loc[lp[ok].sort_values().index, "z_asl_m"],
                          left=np.nan, right=np.nan)
            df.loc[i, "z_asl_m"] = g["z_asl_m"].fillna(pd.Series(z, index=i))
    df["h_agl_m"] = df["z_asl_m"] - ALTITUDE.get(df["station"].iloc[0], 0.0)
    rad = np.radians(df["dir_deg"])
    df["u"] = -df["vit_ms"] * np.sin(rad)                      # direction d'ou vient le vent
    df["v"] = -df["vit_ms"] * np.cos(rad)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--brut", required=True)
    ap.add_argument("--debut", default="2021-01-01")
    ap.add_argument("--fin", default="2025-12-31")
    ap.add_argument("--sortie", required=True)
    args = ap.parse_args()
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(args.brut).glob("TSM000*-data.txt")):
        df = lire_igra(f, args.debut, args.fin)
        nom = f.name.split("-")[0]
        if df.empty:
            print(f"{nom} : aucun radiosondage avec vent sur {args.debut} -> {args.fin}")
            continue
        df.to_csv(sortie / f"radiosondages_{nom}.csv", index=False,
                  date_format="%Y-%m-%dT%H:%M:%SZ")
        bas = df[df["h_agl_m"].between(0, 1000)]
        par_an = df.groupby(df["temps"].dt.year)["temps"].nunique().to_dict()
        par_heure = df.groupby(df["temps"].dt.hour)["temps"].nunique().to_dict()
        prof = bas.groupby("temps").size()
        print(f"{nom} : {df['temps'].nunique()} radiosondages avec vent {par_an} ; heures "
              f"{par_heure} ; {bas['temps'].nunique()} avec du vent entre 0 et 1 000 m "
              f"(mediane {prof.median():.0f} niveaux par profil sous 1 km)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
