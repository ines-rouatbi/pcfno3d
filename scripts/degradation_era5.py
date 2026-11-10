"""Parametres de la degradation d'ERA5 du candidat D national (regle_v4_national.json), AVANT l'entrainement.

Ecart ERA5 10 m - METAR aux 12 aeroports sur les JOURS D'ENTRAINEMENT seulement (calendrier du jeu, METAR a
30 min au plus, instants du forcage toutes les 3 h) :
  sigma_u, sigma_v : RMS de l'ecart par composante (toutes stations et instants) ;
  L_km : longueur de correlation, moindres carres de exp(-d / L) sur la correlation des ecarts (moyenne u, v)
         entre paires d'aeroports (>= 30 instants communs) en fonction de la distance, bornee a [50 ; 400] km.
ERA5 aux aeroports : mailles du forcage a 10 m au-dessus du sol, bilineaire (comme le treillis d'entree).

    python scripts/degradation_era5.py --donnees <dossier national> --calendrier <calendrier.json> --sortie <json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--donnees", required=True)
    ap.add_argument("--calendrier", required=True)
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    from pcfno3d.stations_reelles import lire_observations, obs_a
    G = Path(a.donnees)
    info = json.loads((G / "grille_reelle.json").read_text(encoding="utf-8"))
    st = {s["code"]: (float(s["lat"]), float(s["lon"])) for s in info["stations"].values() if s.get("dans_la_zone")}
    obs = lire_observations(G / "obs", verbose=False, codes=list(st))
    cal = json.loads(Path(a.calendrier).read_text(encoding="utf-8"))
    dates = {d["day_index"]: d["date"] for d in cal["days"]}
    jours = {pd.Timestamp(dates[k]).normalize() for k in cal["split"]["train_days"]}
    with np.load(G / "forcage_era5.npz") as d:
        temps, lat, lon, hs = d["temps"], d["lat"], d["lon"], d["hauteurs_sol"]
        k10 = int(np.argmin(np.abs(hs - 10.0)))
        U, V = d["u"][:, k10], d["v"][:, k10]

    def bil(champ, la, lo):
        fy = np.clip((la - lat[0]) / (lat[1] - lat[0]), 0, len(lat) - 1.000001)
        fx = np.clip((lo - lon[0]) / (lon[1] - lon[0]), 0, len(lon) - 1.000001)
        i, j = int(fy), int(fx)
        p, q = fy - i, fx - j
        return float((1 - p) * (1 - q) * champ[i, j] + (1 - p) * q * champ[i, j + 1]
                     + p * (1 - q) * champ[i + 1, j] + p * q * champ[i + 1, j + 1])

    lignes = []
    for it, s in enumerate(temps):
        t = pd.Timestamp(int(s), unit="s")
        if t.normalize() not in jours:
            continue
        for code, (uo, vo) in obs_a(obs, t).items():
            la, lo = st[code]
            lignes.append({"t": t, "code": code, "du": bil(U[it], la, lo) - uo, "dv": bil(V[it], la, lo) - vo})
    E = pd.DataFrame(lignes)
    sig_u, sig_v = float(np.sqrt((E.du ** 2).mean())), float(np.sqrt((E.dv ** 2).mean()))
    pu, pv = E.pivot_table(index="t", columns="code", values="du"), E.pivot_table(index="t", columns="code", values="dv")
    paires = []
    codes = sorted(st)
    for i, c1 in enumerate(codes):
        for c2 in codes[i + 1:]:
            if c1 not in pu or c2 not in pu:
                continue
            m = pu[[c1, c2]].dropna()
            if len(m) < 30:
                continue
            r = 0.5 * (m[c1].corr(m[c2]) + pv[[c1, c2]].dropna().pipe(lambda x: x[c1].corr(x[c2])))
            (la1, lo1), (la2, lo2) = st[c1], st[c2]
            dk = 6371.0 * 2 * math.asin(math.sqrt(math.sin(math.radians(la2 - la1) / 2) ** 2 + math.cos(math.radians(la1))
                                                  * math.cos(math.radians(la2)) * math.sin(math.radians(lo2 - lo1) / 2) ** 2))
            paires.append({"paire": f"{c1}-{c2}", "distance_km": dk, "correlation": float(r), "instants": int(len(m))})
    P = pd.DataFrame(paires)
    grille = np.arange(50.0, 400.5, 1.0)
    cout = [float(((P.correlation - np.exp(-P.distance_km / L)) ** 2).sum()) for L in grille]
    L = float(grille[int(np.argmin(cout))])
    res = {"sigma_u": sig_u, "sigma_v": sig_v, "L_km": L, "ecarts": int(len(E)), "instants": int(E.t.nunique()),
           "jours_entrainement": len(jours), "stations": int(E.code.nunique()),
           "biais_u": float(E.du.mean()), "biais_v": float(E.dv.mean()),
           "paires": P.round(4).to_dict("records"),
           "note": "ecart ERA5 (maille 0,25 deg, 10 m) - anemometre ponctuel : contient l'erreur de representativite ; "
                   "la degradation du candidat D ne touche que ses ENTREES d'entrainement"}
    Path(a.sortie).parent.mkdir(parents=True, exist_ok=True)
    Path(a.sortie).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "paires"}, indent=2, ensure_ascii=False))
    print(P.sort_values("distance_km").round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
