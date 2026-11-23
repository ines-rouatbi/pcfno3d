"""Regle configs/national/regle_national_gfs_v6.json (approuvee par Ines le 23/11/2026, hachee 9babee63...) : choix
sur la validation 2021 avec une MARGE (+3 % par aeroport), puis jugement UNIQUE sur le 1er-23 novembre 2026.

Trois commandes, dans cet ordre (dossier de sortie NEUF, MyDrive/pcfno3d_national/gfs_v6_regle/) :

    python scripts/juger_national_gfs_v6.py choisir --variantes <validation_2021_v5_2/variantes.json> --sortie <dossier>
    python scripts/juger_national_gfs_v6.py controle --cp ... --gfs ... --tn ... --obs <METAR 2021> --sortie <dossier>
    python scripts/juger_national_gfs_v6.py juger --cp ... --c1 ... --selection <dossier>/selection.json \
        --tn <GFS novembre f0> --tn <f24> --tn <f48> --obs <METAR novembre> --sortie <dossier> [--checkpoint-*]

choisir : aucun calcul de modele ; les 13 variantes deja jugees sur 2021 par regle_national_gfs_v5_2 (variantes.json,
dont l'empreinte de regle est verifiee) ; admissible = dans les trois cases, borne haute de l'IC 95 % <= 0 ET chaque
aeroport <= +3 % ; retenue = plus bas RMSE moyen des trois cases (ex aequo a 0,005 : C2, puis C1, puis C3). Aucune
admissible : ARRET, novembre n'est pas lu.
controle / juger : le code de scripts/juger_national_gfs_v5.py, avec l'empreinte de CETTE regle, les 184 instants
de novembre (toutes les 3 h ; 03/09/15/21 UTC par f003 / f027 / f051 comme le service) et sans C' s44 sans correction.
Sorties : selection.json, choix_2021_v6.md, controle_entree.json, paires.csv, bilan.csv, verdicts_gfs_v6.json,
gfs_v6.md. GFS et ERA5 ne sont jamais appeles mesures : seuls les METAR le sont.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]
REGLE = REPO / "configs" / "national" / "regle_national_gfs_v6.json"
REGLE_V5_2 = REPO / "configs" / "national" / "regle_national_gfs_v5_2.json"
# recopies de la regle (verifies par tests/test_gfs_v6_national.py contre son texte)
MARGE_CHOIX = 0.03
DEBUT, FIN, PAS_H = "2026-11-01T00:00", "2026-11-23T21:00", 3


def _module(nom: str, fichier: str):
    spec = importlib.util.spec_from_file_location(nom, REPO / "scripts" / fichier)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


J = _module("juger_gfs_v5_pour_v6", "juger_national_gfs_v5.py")
C = J.C
CASES, EX_AEQUO, ORDRE_SIMPLICITE = C.CASES, C.EX_AEQUO, C.ORDRE_SIMPLICITE


def verifier_regle(f: Path = REGLE) -> str:
    return C.verifier_regle(f)


def instants_novembre() -> list[pd.Timestamp]:
    """Les 184 instants juges : 2026-11-01 00 UTC -> 2026-11-23 21 UTC, toutes les 3 heures."""
    from pcfno3d.gfs import instants_periode
    return instants_periode(DEBUT, FIN, PAS_H)


# ---------------------------------------------------------------------- etape 1 : choix sur 2021
def admissible(r: dict) -> tuple[bool, dict]:
    """Une variante de variantes.json : (admissible, detail par case : borne haute de l'IC, pire degradation)."""
    det, ok = {}, True
    for case in CASES:
        d = r["details"][case]
        haut = float(d["ic95"][1])
        degr = {a: float(x["degradation"]) for a, x in d["par_aeroport"].items()}
        pire = max(degr.values()) if degr else float("nan")
        ok_case = bool(np.isfinite(haut) and haut <= 0 and degr and all(np.isfinite(x) and x <= MARGE_CHOIX
                                                                         for x in degr.values()))
        det[case] = {"ic95_haut": haut, "pire_degradation": pire,
                     "pire_aeroport": max(degr, key=degr.get) if degr else None, "admissible": ok_case}
        ok &= ok_case
    return ok, det


def choisir(variantes: dict) -> dict:
    """Retenue parmi les admissibles : plus bas RMSE moyen des trois cases ; ex aequo a EX_AEQUO : C2, C1, C3."""
    lignes = []
    for nom, r in variantes.items():
        ok, det = admissible(r)
        lignes.append({"nom": nom, "famille": nom.split("_")[0], "admissible": ok,
                       "rmse": float(r["rmse_moyen_trois_cases"]), "detail": det})
    adm = [x for x in lignes if x["admissible"] and np.isfinite(x["rmse"])]
    if not adm:
        return {"arret": True, "retenue": None, "lignes": lignes,
                "raison": "aucune variante admissible sur 2021 (IC <= 0 et chaque aeroport <= +3 % dans les trois "
                          "cases) : novembre 2026 n'est pas lu"}
    best = min(x["rmse"] for x in adm)
    proches = [x for x in adm if x["rmse"] <= best + EX_AEQUO]
    proches.sort(key=lambda x: (ORDRE_SIMPLICITE.index(x["famille"]), x["rmse"], x["nom"]))
    return {"arret": False, "retenue": proches[0]["nom"], "lignes": lignes}


def rapport_choix(sel: dict, sha: str) -> str:
    L = ["# Règle `regle_national_gfs_v6` : choix sur la validation 2021 (marge +3 %)", "",
         f"Règle `configs/national/regle_national_gfs_v6.json` (sha256 `{sha[:12]}…`), approuvée par Ines le 23/11/2026. "
         "Résultats 2021 déjà calculés par `regle_national_gfs_v5_2` (`variantes.json`), sans nouveau calcul. "
         "Admissible : dans les trois cases, borne haute de l'IC 95 % ≤ 0 et chaque aéroport ≤ +3 % face à GFS brut.",
         "", "| Variante | RMSE moyen (3 cases) | pire aéroport (analyse / +24 h / +48 h) | IC haut (3 cases) | admissible |",
         "|---|---:|---|---|---|"]
    for x in sorted(sel["lignes"], key=lambda x: x["rmse"]):
        d = x["detail"]
        pires = " / ".join(f"{d[c]['pire_aeroport']} {d[c]['pire_degradation']:+.1%}" for c in CASES)
        hauts = " / ".join(f"{d[c]['ic95_haut']:+.3f}" for c in CASES)
        L.append(f"| {x['nom']} | {x['rmse']:.3f} | {pires} | {hauts} | {'oui' if x['admissible'] else 'non'} |")
    L += ["", f"**Retenue : `{sel['retenue']}`**" if not sel["arret"] else f"**ARRÊT** : {sel['raison']}", ""]
    return "\n".join(L)


def commande_choisir(variantes_f: Path, sortie: Path) -> dict:
    sha = verifier_regle()
    sha_v52 = C.verifier_regle(REGLE_V5_2)
    sortie = Path(sortie)
    if (sortie / "selection.json").exists():
        raise SystemExit(f"{sortie} contient deja une selection : rien n'est recalcule ni ecrase")
    v = json.loads(Path(variantes_f).read_text(encoding="utf-8"))
    if v.get("regle_sha256") != sha_v52:
        raise SystemExit("variantes.json ne vient pas de regle_national_gfs_v5_2 (empreinte differente)")
    sel = choisir(v["variantes"])
    variante = next((x for x in C.variantes() if x["nom"] == sel["retenue"]), None)
    out = {"regle": REGLE.name, "regle_sha256": sha, "arret": sel["arret"], "retenue": sel["retenue"],
           "variante": variante, "source_2021": {"fichier": str(variantes_f), "regle_sha256": sha_v52},
           "marge_choix": MARGE_CHOIX, "lignes": sel["lignes"], "raison": sel.get("raison")}
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "choix_2021_v6.md").write_text(rapport_choix(sel, sha), encoding="utf-8")
    (sortie / "selection.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")
    print(rapport_choix(sel, sha))
    return out


# ---------------------------------------------------------------------- etape 2 : novembre 2026 (code de la v5.2)
def rapport_novembre(v: dict, b: pd.DataFrame, info: dict) -> str:
    L = ["# Règle `regle_national_gfs_v6` : jugement du 1er au 23 novembre 2026 aux 12 aéroports", "",
         f"Règle `configs/national/regle_national_gfs_v6.json` (sha256 `{v['regle_sha256'][:12]}…`), approuvée par Ines "
         "le 23/11/2026, vérifiée avant le calcul. Variante retenue sur la validation 2021 seulement (marge +3 %) : "
         f"`{(v.get('candidat') or {}).get('nom')}`. Période : 2026-11-01 00 UTC → 2026-11-23 21 UTC, toutes les 3 h "
         "(03/09/15/21 UTC par f003 / f027 / f051, comme le service), jamais utilisée, **lue une seule fois**. Entrée "
         "construite comme le service temps réel (archive GFS NOAA, METAR IEM des 11 autres aéroports à l'analyse, aucun "
         "aux échéances). Les seules mesures sont les METAR.", "",
         f"Instants : {info['instants']} sur {info['instants_attendus']} ({info['absents']} écartés pour toutes les "
         f"sources faute d'échéance GFS) ; {v['jours']} jours ; IC 95 % par {J.TIRAGES} tirages des jours (graine "
         f"{J.GRAINE}).", "",
         "| Case | Moyenne RMSE candidat | Moyenne RMSE GFS brut | Écart | IC 95 % | Aéroports > +5 % | Verdict |",
         "|---|---:|---:|---:|---|---|---|"]
    for case, d in v["details"].items():
        L.append(f"| {case} | {d['moyenne_rmse_source']:.3f} | {d['moyenne_rmse_gfs_brut']:.3f} | {d['ecart']:+.3f} | "
                 f"[{d['ic95'][0]:+.3f} ; {d['ic95'][1]:+.3f}] | "
                 f"{', '.join(d['aeroports_degrades_plus_5_pct']) or 'aucun'} | **{d['verdict']}** |")
    L += ["", "## Par aéroport (m/s)", ""]
    t = b.pivot_table(index=["echeance", "station"], columns="source", values="rmse_vect").round(3)
    L.append(C.md_table(t.reset_index()))
    return "\n".join(L) + "\n"


def configurer_v5_pour_v6() -> None:
    """Le code de juger_national_gfs_v5 avec l'empreinte, la periode et les sorties de CETTE regle."""
    J.REGLE = REGLE
    C.verifier_regle = (lambda f=REGLE, _v=C.verifier_regle: _v(f))
    J.echantillon_2026 = instants_novembre
    J.rapport = rapport_novembre
    J.FICHIER_VERDICTS, J.FICHIER_RAPPORT = "verdicts_gfs_v6.json", "gfs_v6.md"
    J.SORTIES = (J.FICHIER_VERDICTS, "paires.csv", "arret.json")


def main() -> int:
    from pcfno3d.config import Config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="commande", required=True)
    c = sp.add_parser("choisir", help="choix sur 2021 avec la marge (aucun calcul de modele)")
    c.add_argument("--variantes", required=True)
    c.add_argument("--sortie", required=True)
    a = sp.add_parser("controle", help="controle prealable sur la validation 2021 (avant toute lecture de novembre)")
    for x in ("--cp", "--gfs", "--obs", "--sortie"):
        a.add_argument(x, required=True)
    a.add_argument("--tn", action="append", required=True)
    b = sp.add_parser("juger", help="jugement unique du 1er au 23 novembre 2026")
    for x in ("--cp", "--c1", "--selection", "--obs", "--sortie"):
        b.add_argument(x, required=True)
    b.add_argument("--tn", action="append", required=True)
    b.add_argument("--checkpoint-cp")
    b.add_argument("--checkpoint-c1")
    b.add_argument("--device")
    x = ap.parse_args()
    if x.commande == "choisir":
        s = commande_choisir(Path(x.variantes), Path(x.sortie))
        return 6 if s["arret"] else 0
    configurer_v5_pour_v6()
    if x.commande == "controle":
        r = J.controler(Config.load(x.cp), Path(x.gfs), x.tn, Path(x.obs), Path(x.sortie))
        return 0 if r["passe"] else 5
    v = J.juger(Config.load(x.cp), Config.load(x.c1), Path(x.selection), x.tn, Path(x.obs), Path(x.sortie),
                x.checkpoint_cp, x.checkpoint_c1, x.device, avec_cp_s44=False)
    print(json.dumps(v["cases"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
