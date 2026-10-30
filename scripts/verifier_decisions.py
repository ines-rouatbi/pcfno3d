"""AGRI1 / AGRI2 (cahier v12, OS8) : les decisions agricoles d'ATOM Wind contre celles tirees des
mesures des stations de l'INM, sur le test 2022.

Regle FIGEE avant tout calcul, celle de l'espace agriculteur d'ATOM Wind (maquette, ACTS.spray :
vent a 10 m, en km/h) — jamais ajustee au resultat :

    traitement   < 3 attention (air trop calme) ; <= 15 oui ; <= 19 attention (derive) ; > 19 non
    aspersion    <= 15 oui ; <= 25 attention ; > 25 non

Donnees : stations_paires.csv de scripts/valider_grand_tunis_reel.py (instants synoptiques du
test, chaque station CACHEE au modele : u_modele, v_modele ; references ERA5 a 10 m et couche de
surface du generateur ; u_obs, v_obs = METAR / SYNOP). Une station en entree recopierait sa
mesure : l'accord serait artificiel.

    AGRI1  taux d'accord (oui / attention / non) et kappa de Cohen, par source, par station,
           toutes heures et heures de jour (6 h - 19 h, heure de Tunis)
    AGRI2  « faux oui » : part des heures ou la source dit « oui » et la station « non »
           (seuil du cahier : 5 % au plus, a valider) ; et part des « non » de la station
           ou la source dit « oui »

    python scripts/verifier_decisions.py --paires <run_dir>/validation_reelle/test/stations_paires.csv \
        --sortie <dossier>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KMH = 3.6
CLASSES = ("oui", "attention", "non")
SEUIL_AGRI2 = 0.05
UTC_LOCAL_H = 1


def decision_traitement(vitesse_m_s: np.ndarray) -> np.ndarray:
    k = np.round(np.asarray(vitesse_m_s, float) * KMH, 6)      # 15 km/h reste 15
    return np.select([k < 3, k <= 15, k <= 19], ["attention", "oui", "attention"], "non")


def decision_aspersion(vitesse_m_s: np.ndarray) -> np.ndarray:
    k = np.round(np.asarray(vitesse_m_s, float) * KMH, 6)      # 15 km/h reste 15
    return np.select([k <= 15, k <= 25], ["oui", "attention"], "non")


REGLES = {"traitement": decision_traitement, "aspersion": decision_aspersion}


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in CLASSES)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def scores(obs: np.ndarray, src: np.ndarray) -> dict:
    n = len(obs)
    non_obs = obs == "non"
    return {"n": int(n),
            "accord": float(np.mean(obs == src)) if n else float("nan"),
            "kappa": kappa(obs, src),
            "faux_oui": float(np.mean((src == "oui") & non_obs)) if n else float("nan"),
            "faux_oui_parmi_non": float(np.mean(src[non_obs] == "oui")) if non_obs.any() else float("nan"),
            "matrice": {f"{o}->{s}": int(np.sum((obs == o) & (src == s)))
                        for o in CLASSES for s in CLASSES}}


def evaluer(p: pd.DataFrame, sources: tuple[str, ...]) -> pd.DataFrame:
    p = p.dropna(subset=["u_obs", "v_obs"]).copy()
    heure = (pd.to_datetime(p["temps"]).dt.hour + UTC_LOCAL_H) % 24
    p["jour"] = (heure >= 6) & (heure <= 19)
    lignes = []
    for regle, f in REGLES.items():
        obs_all = f(np.hypot(p["u_obs"], p["v_obs"]))
        for src in sources:
            if f"u_{src}" not in p:
                continue
            ok = p[f"u_{src}"].notna() & p[f"v_{src}"].notna()
            dec = f(np.hypot(p[f"u_{src}"], p[f"v_{src}"]))
            for periode, m in (("toutes heures", np.ones(len(p), bool)), ("jour 6-19 h", p["jour"].to_numpy())):
                for station in ["toutes"] + sorted(p["station"].unique()):
                    sel = ok.to_numpy() & m & ((p["station"] == station).to_numpy() if station != "toutes" else True)
                    s = scores(obs_all[sel], dec[sel])
                    lignes.append({"regle": regle, "source": src, "periode": periode, "station": station,
                                   **{k: v for k, v in s.items() if k != "matrice"},
                                   "matrice": json.dumps(s["matrice"])})
    return pd.DataFrame(lignes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paires", required=True, nargs="+", help="stations_paires.csv (un ou plusieurs)")
    ap.add_argument("--sources", default="modele,era5,generateur")
    ap.add_argument("--sortie", required=True)
    args = ap.parse_args()
    p = pd.concat([pd.read_csv(f) for f in args.paires], ignore_index=True)
    t = evaluer(p, tuple(args.sources.split(",")))
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    t.to_csv(sortie / "decisions_agricoles.csv", index=False)
    g = t[(t["station"] == "toutes") & (t["regle"] == "traitement")]
    resume = {
        "regle": "traitement : <3 attention, <=15 oui, <=19 attention, >19 non (km/h, 10 m) ; "
                 "aspersion : <=15 oui, <=25 attention, >25 non",
        "donnees": [str(f) for f in args.paires],
        "agri1": {f"{r.source} / {r.periode}": {"accord": round(r.accord, 4), "kappa": round(r.kappa, 4), "n": r.n}
                  for r in g.itertuples()},
        "agri2": {f"{r.source} / {r.periode}": {"faux_oui": round(r.faux_oui, 4),
                                                "conforme": bool(r.faux_oui <= SEUIL_AGRI2)}
                  for r in g.itertuples()},
        "seuil_agri2": SEUIL_AGRI2,
        "note": "AGRI1 : la cible est fixee apres cette premiere mesure (cahier v12) ; la regle, elle, "
                "est celle de l'application, fixee avant.",
    }
    (sortie / "decisions_agricoles.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print(g[["source", "periode", "n", "accord", "kappa", "faux_oui"]].round(3).to_string(index=False))
    print(f"-> {sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
