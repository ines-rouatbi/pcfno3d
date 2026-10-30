"""Bilan de conformite du projet aux criteres d'acceptation du cahier des charges v12 (sections 9.2 et 9.3).

    python scripts/bilan_conformite.py --runs <dossier runs du Drive> [--modele grand_tunis_v4_dso]

Le script ne fait que LIRE des resultats deja ecrits (aucune passe du modele, aucun reglage) et classe
chaque critere : conforme, non conforme, a mesurer (le resultat n'existe pas encore) ou propose (critere
pas encore fixe avec l'entreprise). Chaque ligne dit d'ou vient son chiffre.

Sources lues dans --runs :
  <modele>/eval_pcfno3d/best_wcont (ou best)   DSO1 a DSO6 (pcfno3d.atom_wind.qualite)
  <modele>/validation_reelle/test              REEL1 (station exclue, contre ERA5 et une interpolation
                                               inverse a la distance des autres stations, calculee ici a
                                               partir de stations_paires.csv), REEL2 et EOL2 (radiosondages)
  **/prevision_bilan.json                      PREV1, PREV2 (scripts/evaluer_prevision.py)
  **/decisions_agricoles.json                  PREV3, AGRI1, AGRI2 (scripts/verifier_decisions.py)
  **/os6.json                                  NET1, NET2 (scripts/os6_reseau_aneo.py)
Les criteres mesures sur la v7 synthetique ou garantis par les tests automatiques du depot (DE1 a DE3,
AN1, MCP1, API1, HIS1, PROD1, EOL1) sont repris de leur mesure documentee, avec leur reference.

Sortie (dans --runs) : bilan_conformite.json et bilan_conformite.md.
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
sys.path.insert(0, str(REPO / "src"))

# stations du Grand Tunis (lat, lon) : cellules du modele dans data_dir/stations.json, memes points
STATIONS = {"Tunis-Carthage": (36.8504, 10.2156), "Zaghouan": (36.4333, 10.0833), "Nabeul": (36.45, 10.7333)}
SEUIL_NET1 = 0.15
SEUIL_AGRI2 = 0.05

# criteres repris de leur mesure documentee (CLAUDE.md, docs/) ou garantis par un test du depot
DOCUMENTES = [
    ("DE1", "Contrat de données : un fichier non conforme est rejeté avec un rapport d'erreurs", "conforme",
     "tests/test_contract*.py (lot M3, fusionné sur main)"),
    ("DE2", "Reproductibilité : même entrée, même version, même sortie", "conforme",
     "lot M0 (test de reproductibilité, tolérance documentée)"),
    ("DE3", "Traçabilité : chaque export référence modèle, statistiques et horodatage", "conforme",
     "lot M6, tests/test_publish*.py"),
    ("PROD1", "Parcours produit sans intervention manuelle", "conforme",
     "lot M6 : rejeu des 120 instants de test (scripts/m6_rejeu_colab.py)"),
    ("AN1", "Détection d'anomalies : ≥ 90 % détectés, < 5 % de fausses alertes", "conforme (2e lecture)",
     "docs/ALERTES.md : v1 non conforme (16/10) ; v2 conforme en seconde lecture, non indépendante "
     "(dérives 100 %, fausses alertes 2,2 %) ; mesuré sur la v7 synthétique"),
    ("VIS1", "Tableau de bord : reconstruction et indicateurs affichés en moins de 3 s", "à mesurer",
     "à chronométrer sur atom.zelqane.com une fois le service en ligne"),
    ("MCP1", "Serveur MCP : simulation lancée et résultat récupéré sans intervention", "conforme",
     "tests/test_webapp_and_mcp.py, tests/test_colab_serving.py (17 outils)"),
    ("API1", "Clé d'API : requête sans clé refusée, clé révoquée inactive aussitôt", "conforme",
     "lot M7, tests/test_auth*.py"),
    ("HIS1", "Historique : toute reconstruction passée retrouvée avec ses indicateurs", "conforme",
     "lot M6, tests contre tests/worker_simule.py"),
    ("EOL1", "Production éolienne : écart < 5 % entre vent reconstruit et vent vrai", "conforme (parc de démonstration)",
     "lot M5 : 1,71 % sur le parc de démonstration (v7) ; à refaire sur le parc réel"),
    ("REEL3", "Cohérence avec ERA5 : la moyenne du champ généré retrouve ERA5 à chaque niveau", "conforme",
     "porte du 21/10/2026 (scripts/test_generateur_reel.py) : moyenne du domaine ≈ ERA5 ; écart maximal à fixer"),
    ("REEL4", "Brise de mer : cycle journalier de la direction aux stations côtières", "conforme",
     "porte du 21/10/2026 (scripts/brise_mer.py), figure validée par Ines avant la génération"),
    ("UX1", "Compréhension : 4 utilisateurs sur 5 répondent en moins de 30 s", "à mesurer",
     "test avec 5 utilisateurs non spécialistes à organiser"),
    ("DSO7", "Précision à densité réaliste (un capteur tous les 500 m)", "proposé",
     "critère à fixer avec l'entreprise avant les essais (cahier v12, 18.7)"),
]


def _ligne(code, critere, statut, valeur=None, seuil=None, source="", detail=None) -> dict:
    return {"code": code, "critere": critere, "statut": statut, "valeur": valeur, "seuil": seuil,
            "source": source, **({"detail": detail} if detail else {})}


def _rmse(du, dv) -> float:
    return float(np.sqrt(np.mean(np.square(du) + np.square(dv)))) if len(du) else math.nan


def _km(a, b) -> float:
    (la1, lo1), (la2, lo2) = a, b
    return math.hypot((lo2 - lo1) * 111.32 * math.cos(math.radians((la1 + la2) / 2)), (la2 - la1) * 110.57)


def idw_stations(p: pd.DataFrame, puissance: float = 2.0) -> pd.DataFrame:
    """Ajoute u_idw, v_idw : pour chaque station jugee, l'interpolation inverse a la distance des mesures
    des AUTRES stations au meme instant (la station jugee n'y entre jamais)."""
    p = p.copy()
    obs = {(r.station, r.temps): (r.u_obs, r.v_obs) for r in p.itertuples()}
    uu, vv = [], []
    for r in p.itertuples():
        num_u = num_v = den = 0.0
        for autre, pos in STATIONS.items():
            if autre == r.station or r.station not in STATIONS:
                continue
            m = obs.get((autre, r.temps))
            if m is None or not (np.isfinite(m[0]) and np.isfinite(m[1])):
                continue
            w = 1.0 / _km(STATIONS[r.station], pos) ** puissance
            num_u, num_v, den = num_u + w * m[0], num_v + w * m[1], den + w
        uu.append(num_u / den if den else math.nan)
        vv.append(num_v / den if den else math.nan)
    p["u_idw"], p["v_idw"] = uu, vv
    return p


def reel1(dossier: Path) -> dict:
    f = dossier / "stations_paires.csv"
    if not f.exists():
        return _ligne("REEL1", "Station exclue : erreur du modèle < ERA5 et < interpolation (IDW)", "à mesurer",
                      source=str(f))
    p = idw_stations(pd.read_csv(f))
    sources = ("modele", "era5", "idw")
    ok = p[[f"{c}_{s}" for s in sources for c in "uv"] + ["u_obs", "v_obs"]].notna().all(axis=1)
    p = p[ok]
    par = {}
    for st, g in p.groupby("station"):
        par[st] = {s: round(_rmse(g[f"u_{s}"] - g["u_obs"], g[f"v_{s}"] - g["v_obs"]), 3) for s in sources}
        par[st]["n"] = int(len(g))
    conforme = bool(par) and all(v["modele"] < v["era5"] and v["modele"] < v["idw"] for v in par.values())
    return _ligne("REEL1", "Station exclue : erreur du modèle < ERA5 et < interpolation (IDW)",
                  "conforme" if conforme else "non conforme",
                  valeur={k: v["modele"] for k, v in par.items()}, seuil="< ERA5 et < IDW à chaque station",
                  source=str(f), detail=par)


def sondes(dossier: Path) -> list[dict]:
    f = dossier / "radiosondages_paires.csv"
    if not f.exists():
        return [_ligne(c, t, "à mesurer", source=str(f)) for c, t in
                (("REEL2", "Radiosondages 25-1 000 m : erreur du modèle ≤ ERA5"),
                 ("EOL2", "Radiosondages 0-250 m : erreur du modèle ≤ ERA5"))]
    p = pd.read_csv(f).dropna(subset=["u_obs", "v_obs", "u_modele", "v_modele", "u_era5", "v_era5"])
    out = []
    for code, texte, lo, hi in (("REEL2", "Radiosondages 25-1 000 m : erreur du modèle ≤ ERA5", 25, 1000),
                                ("EOL2", "Radiosondages 0-250 m : erreur du modèle ≤ ERA5", 0, 250)):
        q = p[(p["h_agl_m"] >= lo) & (p["h_agl_m"] <= hi)]
        m = _rmse(q["u_modele"] - q["u_obs"], q["v_modele"] - q["v_obs"])
        e = _rmse(q["u_era5"] - q["u_obs"], q["v_era5"] - q["v_obs"])
        statut = "à mesurer" if not len(q) else ("conforme" if m <= e else "non conforme")
        out.append(_ligne(code, texte, statut, valeur=round(m, 3), seuil=f"≤ ERA5 ({e:.3f} m/s)", source=str(f),
                          detail={"points": int(len(q)), "radiosondages": int(q["temps"].nunique()) if len(q) else 0,
                                  "note": "ERA5 assimile les radiosondages : critère non totalement indépendant"}))
    return out


def prevision(runs: Path) -> list[dict]:
    out, vus = [], set()
    for f in sorted(runs.rglob("prevision_bilan.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        t = r.get("verdicts", {}).get("toutes", {})
        code = r.get("critere", "PREV?")
        vus.add(code)
        out.append(_ligne(code, f"Prévision à +{r.get('echeance_h')} h aux stations : erreur < GFS brut et < persistance",
                          "conforme" if r.get("conforme") else "non conforme",
                          valeur=t.get("rmse_vect", {}).get("prevision"),
                          seuil=f"< GFS {t.get('rmse_vect', {}).get('gfs')} et < persistance "
                                f"{t.get('rmse_vect', {}).get('persistance')}",
                          source=str(f.relative_to(runs)), detail={"modele": r.get("modele"), "paires": r.get("paires")}))
    for code, h in (("PREV1", 24), ("PREV2", 48)):
        if code not in vus:
            out.append(_ligne(code, f"Prévision à +{h} h aux stations : erreur < GFS brut et < persistance",
                              "à mesurer", source="notebook OS7 (pcfno3d_grand_tunis_os7_colab.ipynb)"))
    return out


def decisions(runs: Path) -> list[dict]:
    fichiers = sorted(runs.rglob("decisions_agricoles.json"))
    if not fichiers:
        return [_ligne(c, t, "à mesurer", source="notebook OS7") for c, t in
                (("PREV3", "Conseil de la veille : décision prévue = décision mesurée le lendemain"),
                 ("AGRI1", "Décisions agricoles : taux d'accord avec les stations (cible après 1re mesure)"),
                 ("AGRI2", f"Erreur dangereuse : « oui » quand la station dit « non » ≤ {SEUIL_AGRI2:.0%}"))]
    out = []
    for f in fichiers:
        r = json.loads(f.read_text(encoding="utf-8"))
        rel = str(f.relative_to(runs))
        prevu = any(k.startswith("prevision") for k in r.get("agri1", {}))
        code1 = "PREV3" if prevu else "AGRI1"
        a1 = r.get("agri1", {})
        cle = next((k for k in a1 if k.startswith("prevision" if prevu else "modele")), None)
        out.append(_ligne(code1, "Décision de traitement : taux d'accord avec la mesure des stations", "mesuré",
                          valeur=a1.get(cle, {}).get("accord") if cle else None,
                          seuil="rapporté face à GFS brut" if prevu else "cible fixée après cette 1re mesure",
                          source=rel, detail=a1))
        a2 = r.get("agri2", {})
        cle2 = next((k for k in a2 if k.startswith("prevision" if prevu else "modele")), None)
        if cle2:
            out.append(_ligne("AGRI2", f"Erreur dangereuse ({'prévision' if prevu else 'temps réel'})",
                              "conforme" if a2[cle2]["conforme"] else "non conforme",
                              valeur=a2[cle2]["faux_oui"], seuil=f"≤ {SEUIL_AGRI2}", source=rel, detail=a2))
    return out


def reseau(runs: Path) -> list[dict]:
    fichiers = sorted(runs.rglob("os6.json"))
    if not fichiers:
        return [_ligne("NET1", "Placement optimisé : RMSE ≥ 15 % sous la grille régulière", "à mesurer",
                       source="notebook OS6"),
                _ligne("NET2", "Courbe erreur / nombre de capteurs de 10 à 200", "à mesurer", source="notebook OS6")]
    f = fichiers[-1]
    r = json.loads(f.read_text(encoding="utf-8"))
    comp = r.get("comparaison_net1", {})
    gains = {n: 1 - v["optimise"] / v["grille"] for n, v in comp.items() if v.get("grille")}
    g_min = min(gains.values()) if gains else None
    return [
        _ligne("NET1", "Placement optimisé : RMSE ≥ 15 % sous la grille régulière, à nombre égal",
               "à mesurer" if g_min is None else ("conforme" if g_min >= SEUIL_NET1 else "non conforme"),
               valeur=None if g_min is None else round(g_min, 3), seuil=f"gain ≥ {SEUIL_NET1} (seuil à valider)",
               source=str(f.relative_to(runs)), detail={str(k): round(v, 3) for k, v in gains.items()}),
        _ligne("NET2", "Courbe erreur / nombre de capteurs (10 à 200) et plus petit nombre suffisant",
               "conforme" if r.get("courbe") else "à mesurer", valeur=r.get("nombre_recommande"),
               seuil=f"pire RMSE de niveau ≤ {r.get('seuil_rmse_m_s')} m/s", source=str(f.relative_to(runs))),
    ]


def bilan(runs: Path, modele: str | None = None) -> dict:
    from pcfno3d import atom_wind
    from pcfno3d.config import DSOCfg

    runs = Path(runs)
    dec = runs / "decision_v4.json"
    if modele is None:
        modele = json.loads(dec.read_text(encoding="utf-8"))["modele_livre"] if dec.exists() else "grand_tunis_v2_long"
    lignes = []
    try:
        q = atom_wind.qualite(runs / modele, DSOCfg(), w_continuite=True)
        for l in q["dso"]:
            statut = {True: "conforme", False: "non conforme", None: "à mesurer"}[l["conforme"]]
            lignes.append(_ligne(l["code"], l["libelle"], statut, valeur=l["valeur"], source=f"{modele}/eval_pcfno3d/{q['evaluation']}",
                                 detail={"note": l["note"]} if l["note"] else None))
    except FileNotFoundError as exc:
        lignes += [_ligne(f"DSO{i}", "évaluation du modèle livré", "à mesurer", source=str(exc)) for i in range(1, 7)]
    reel = runs / modele / "validation_reelle" / "test"
    lignes.append(reel1(reel))
    lignes += sondes(reel)
    lignes += prevision(runs)
    lignes += decisions(runs)
    lignes += reseau(runs)
    for code, critere, statut, source in DOCUMENTES:
        lignes.append(_ligne(code, critere, statut, source=source))
    compte: dict[str, int] = {}
    for l in lignes:
        cle = l["statut"].split(" (")[0]
        compte[cle] = compte.get(cle, 0) + 1
    return {"modele": modele, "runs": str(runs), "criteres": lignes, "compte": compte,
            "regle": "lecture seule des resultats ecrits ; aucun reglage apres lecture du test"}


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        return ", ".join(f"{k} {_fmt(x)}" for k, x in v.items())
    if isinstance(v, float):
        return f"{v:.3g}".replace(".", ",")
    return str(v)


def rapport(b: dict) -> str:
    t = [f"# Bilan de conformité au cahier des charges v12 — modèle {b['modele']}", "",
         "Statuts : " + ", ".join(f"{k} {n}" for k, n in sorted(b["compte"].items())) + ".", "",
         "| Code | Critère | Valeur | Seuil | Statut | Source |", "|---|---|---|---|---|---|"]
    for l in b["criteres"]:
        t.append(f"| {l['code']} | {l['critere']} | {_fmt(l['valeur'])} | {_fmt(l['seuil'])} | {l['statut']} | "
                 f"{l['source']} |")
    t += ["", "Lecture seule des résultats écrits : aucune passe du modèle et aucun réglage après lecture du test. "
          "REEL1 : l'interpolation inverse à la distance n'utilise que les autres stations au même instant."]
    return "\n".join(t) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--modele", help="run du modèle livré (défaut : celui de decision_v4.json)")
    a = ap.parse_args(argv)
    b = bilan(Path(a.runs), a.modele)
    runs = Path(a.runs)
    (runs / "bilan_conformite.json").write_text(json.dumps(b, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    texte = rapport(b)
    (runs / "bilan_conformite.md").write_text(texte, encoding="utf-8")
    print(texte)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
