"""Statistiques du canal d'innovation du candidat D2 national (configs/national/regle_d2_national.json,
PROPOSITION soumise a l'accord ecrit d'Ines), sur les JOURS D'ENTRAINEMENT seulement, AVANT l'entrainement.

Innovation a chaque aeroport et instant du forcage (toutes les 3 h, METAR a 30 min au plus) :
    d = vent METAR (10 m) - vent ERA5 a 10 m au-dessus du sol (mailles du forcage, bilineaire, comme
        scripts/degradation_era5.py et le treillis d'entree)
Methode de Hollingsworth et Loennberg (pcfno3d.innovation.ajuster_statistiques) : biais par station retire,
covariance des innovations entre deux aeroports (moyenne u, v ; >= 30 instants communs) = sigma_b^2 rho(d ; L),
variance a une station = sigma_b^2 + sigma_o^2 ; L sur la grille [20 ; 50] km (pas 1 km) et sigma_b^2 aux
moindres carres ponderes par les instants ; rapport_o_b = sigma_o / sigma_b.
Limite connue, ecrite dans la sortie : les aeroports sont distants de plusieurs dizaines de km ; L est alors
extrapole sous la plus petite distance entre aeroports et peut etre en butee (L_en_butee).

    python scripts/statistiques_innovation_d2.py --donnees <dossier national> --calendrier <calendrier.json> \
        --sortie <json> [--correlation gaussienne]
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


def haversine_km(la1, lo1, la2, lo2) -> float:
    return 6371.0 * 2 * math.asin(math.sqrt(math.sin(math.radians(la2 - la1) / 2) ** 2 + math.cos(math.radians(la1))
                                            * math.cos(math.radians(la2)) * math.sin(math.radians(lo2 - lo1) / 2) ** 2))


def innovations(obs_a, obs, st: dict, temps, U, V, lat, lon, jours) -> pd.DataFrame:
    """Table (t, code, du, dv) : METAR - ERA5 10 m (bilineaire) aux instants du forcage des `jours`."""
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
            if code not in st:
                continue
            la, lo = st[code]
            lignes.append({"t": t, "code": code, "du": uo - bil(U[it], la, lo), "dv": vo - bil(V[it], la, lo)})
    return pd.DataFrame(lignes, columns=["t", "code", "du", "dv"])


def paires_et_variances(E: pd.DataFrame, st: dict, min_instants: int = 30) -> tuple[list[dict], dict, dict]:
    """Covariances entre aeroports (biais par station retire, moyenne u, v), variances et biais par station."""
    E = E.copy()
    biais = E.groupby("code")[["du", "dv"]].mean()
    E["du"] -= E["code"].map(biais["du"])
    E["dv"] -= E["code"].map(biais["dv"])
    pu = E.pivot_table(index="t", columns="code", values="du")
    pv = E.pivot_table(index="t", columns="code", values="dv")
    variances = {c: float(0.5 * (pu[c].var() + pv[c].var())) for c in pu if pu[c].count() >= min_instants}
    paires = []
    codes = sorted(variances)
    for i, c1 in enumerate(codes):
        for c2 in codes[i + 1:]:
            mu, mv = pu[[c1, c2]].dropna(), pv[[c1, c2]].dropna()
            if len(mu) < min_instants:
                continue
            cov = 0.5 * (float(np.cov(mu[c1], mu[c2])[0, 1]) + float(np.cov(mv[c1], mv[c2])[0, 1]))
            paires.append({"paire": f"{c1}-{c2}", "distance_km": haversine_km(*st[c1], *st[c2]),
                           "covariance": cov, "correlation": cov / math.sqrt(variances[c1] * variances[c2]),
                           "instants": int(len(mu))})
    return paires, variances, {c: [float(biais.loc[c, "du"]), float(biais.loc[c, "dv"])] for c in biais.index}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--donnees", required=True)
    ap.add_argument("--calendrier", required=True)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--correlation", default="gaussienne", choices=["gaussienne", "exponentielle"])
    ap.add_argument("--bornes-km", default="20,50")
    a = ap.parse_args()
    from pcfno3d.innovation import ajuster_statistiques
    from pcfno3d.stations_reelles import lire_observations, obs_a

    sortie = Path(a.sortie)
    if sortie.exists():
        raise SystemExit(f"{sortie} existe deja : rien n'est recalcule ni ecrase")
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
    E = innovations(obs_a, obs, st, temps, U, V, lat, lon, jours)
    paires, variances, biais = paires_et_variances(E, st)
    bornes = tuple(float(x) for x in a.bornes_km.split(","))
    res = ajuster_statistiques(paires, variances, a.correlation, bornes)
    res.update({"jeu": "jours d'entrainement", "jours_entrainement": len(jours), "ecarts": int(len(E)),
                "instants": int(E.t.nunique()), "stations": sorted(variances), "variances": variances,
                "biais_par_station": biais, "paires_detail": paires,
                "regle": "configs/national/regle_d2_national.json (PROPOSITION soumise a l'accord ecrit d'Ines)",
                "note": "innovation = METAR (10 m) - ERA5 10 m ; sigma_o contient l'erreur de representativite "
                        "(maille 0,25 deg contre anemometre ponctuel)"})
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("paires_detail", "variances", "biais_par_station")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
