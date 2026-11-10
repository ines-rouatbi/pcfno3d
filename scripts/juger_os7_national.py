"""Applique la regle de l'OS7 national (configs/national/regle_os7_national.json, fixee et hachee avant tout
calcul) aux sorties ecrites par scripts/evaluer_prevision.py et scripts/verifier_decisions.py. Aucun seuil n'est
choisi ici : ils sont recopies de la regle, dont l'empreinte sha256 est verifiee avant toute lecture. Rien
n'est recalcule ni ajuste.

    python scripts/juger_os7_national.py --os7 <MyDrive/pcfno3d_national/os7> [--sortie DIR]

Dossier --os7 (carnet pcfno3d_national_os7_colab.ipynb) :
    v5_f24/, v5_f48/                     prevision_bilan.json, prevision_paires.csv (v5)
    v5_f24/decisions/, v5_f48/decisions/ decisions_agricoles.json (PREV3)
    sans_affinage_f24/, sans_affinage_f48/  la reference (v4 choisi + GFS, sans affinage)
Sorties : verdict_os7_national.json et .md dans --sortie (defaut : --os7). Un resultat absent est
declare « non mesure » ; rien n'est invente.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REGLE = REPO / "configs" / "national" / "regle_os7_national.json"
# recopies de la regle (verifies par test_os7_national.py contre son texte)
PREV3_MIN_PAIRES = 100
PREV3_MIN_STATIONS = 6
SEUIL_FAUX_OUI = 0.05


def verifier_regle(f: Path = REGLE) -> str:
    h = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    attendu = (Path(f).parent / (Path(f).name + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"regle modifiee depuis son hachage : {h} != {attendu}")
    return h


def _lire(f: Path) -> dict | None:
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def juger_prev(bilan: dict | None, code: str) -> dict:
    """PREV1 / PREV2 : RMSE vectoriel toutes paires du v5 < GFS brut ET < persistance."""
    if bilan is None:
        return {"critere": code, "verdict": "non mesure"}
    t = bilan.get("verdicts", {}).get("toutes", {})
    e = t.get("rmse_vect", {})
    if not e:
        return {"critere": code, "verdict": "non mesure", "raison": "aucune paire"}
    ok = bool(t.get("mieux_que_gfs")) and bool(t.get("mieux_que_persistance"))
    return {"critere": code, "verdict": "conforme" if ok else "non conforme",
            "rmse_vect": e, "paires": bilan.get("paires"), "instants": bilan.get("instants"),
            "aeroports_mieux_que_gfs": t.get("stations_mieux_que_gfs"),
            "aeroports_mieux_que_persistance": t.get("stations_mieux_que_persistance"),
            "aeroports": t.get("stations"), "ic95_ecart": t.get("ic95_ecart")}


def juger_prev3(decisions: dict | None, paires: Path) -> dict:
    """PREV3 : mesurable si >= 100 paires avec mesure et >= 6 aeroports ; conforme si accord du v5 > GFS
    brut et > persistance (toutes stations, toutes heures) et faux oui du v5 <= 5 %."""
    if decisions is None or not paires.exists():
        return {"critere": "PREV3", "verdict": "non mesure"}
    p = pd.read_csv(paires).dropna(subset=["u_obs", "v_obs"])
    n, stations = int(len(p)), int(p["station"].nunique())
    if n < PREV3_MIN_PAIRES or stations < PREV3_MIN_STATIONS:
        return {"critere": "PREV3", "verdict": "non mesurable", "paires": n, "aeroports": stations,
                "raison": f"il faut au moins {PREV3_MIN_PAIRES} paires et {PREV3_MIN_STATIONS} aeroports"}
    a1, a2 = decisions.get("agri1", {}), decisions.get("agri2", {})
    acc = {s: a1.get(f"{s} / toutes heures", {}).get("accord") for s in ("prevision", "gfs", "persistance")}
    faux = a2.get("prevision / toutes heures", {}).get("faux_oui")
    if any(v is None for v in acc.values()) or faux is None:
        return {"critere": "PREV3", "verdict": "non mesure", "raison": "sources absentes de decisions_agricoles.json"}
    ok = acc["prevision"] > acc["gfs"] and acc["prevision"] > acc["persistance"] and faux <= SEUIL_FAUX_OUI
    return {"critere": "PREV3", "verdict": "conforme" if ok else "non conforme", "accord": acc,
            "faux_oui_prevision": faux, "seuil_faux_oui": SEUIL_FAUX_OUI, "paires": n, "aeroports": stations,
            "kappa": {s: a1.get(f"{s} / toutes heures", {}).get("kappa") for s in acc}}


def juger(os7: Path) -> dict:
    h = verifier_regle()
    os7 = Path(os7)
    out = {"regle": str(REGLE.relative_to(REPO)).replace("\\", "/"), "regle_sha256": h,
           "PREV1": juger_prev(_lire(os7 / "v5_f24" / "prevision_bilan.json"), "PREV1"),
           "PREV2": juger_prev(_lire(os7 / "v5_f48" / "prevision_bilan.json"), "PREV2"),
           "PREV3": juger_prev3(_lire(os7 / "v5_f24" / "decisions" / "decisions_agricoles.json"),
                                os7 / "v5_f24" / "prevision_paires.csv"),
           "rapporte": {
               "PREV3_48h": juger_prev3(_lire(os7 / "v5_f48" / "decisions" / "decisions_agricoles.json"),
                                        os7 / "v5_f48" / "prevision_paires.csv"),
               "reference_sans_affinage_f24": juger_prev(_lire(os7 / "sans_affinage_f24" / "prevision_bilan.json"),
                                                          "PREV1 (reference)"),
               "reference_sans_affinage_f48": juger_prev(_lire(os7 / "sans_affinage_f48" / "prevision_bilan.json"),
                                                          "PREV2 (reference)")}}
    return out


def en_markdown(v: dict) -> str:
    L = ["# Verdicts de l'OS7 national\n", f"Regle `{v['regle']}` (sha256 `{v['regle_sha256'][:12]}...`, "
         "verifiee). Mesures : les 12 aeroports METAR ; GFS et la verite generee ne sont pas des mesures.\n",
         "| Critere | Verdict | v5 | GFS brut | Persistance |", "|---|---|---|---|---|"]

    def ligne(nom, r):
        e = r.get("rmse_vect") or r.get("accord") or {}
        f = (lambda x: "" if x is None else f"{x:.3f}")
        L.append(f"| {nom} | {r['verdict']} | {f(e.get('prevision'))} | {f(e.get('gfs'))} | {f(e.get('persistance'))} |")

    for k in ("PREV1", "PREV2", "PREV3"):
        ligne(k, v[k])
    for k, r in v["rapporte"].items():
        ligne(f"{k} (rapporte)", r)
    L.append("\nPREV1 / PREV2 : RMSE vectoriel a 10 m (m/s), toutes paires ; PREV3 : taux d'accord des decisions "
             "de traitement (faux oui du v5 : " + str(v["PREV3"].get("faux_oui_prevision")) + ").")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--os7", required=True)
    ap.add_argument("--sortie")
    args = ap.parse_args()
    v = juger(Path(args.os7))
    sortie = Path(args.sortie or args.os7)
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "verdict_os7_national.json").write_text(json.dumps(v, indent=2, ensure_ascii=False, default=str),
                                                      encoding="utf-8")
    (sortie / "verdict_os7_national.md").write_text(en_markdown(v), encoding="utf-8")
    print(en_markdown(v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
