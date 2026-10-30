"""Champion et challenger (cahier v12, E14) : comparaison automatique d'un nouveau modele au modele en
service, AVANT tout remplacement.

La regle est celle de la livraison v4 (scripts/decision_v4_grand_tunis.py, regles d'Ines du
2026-10-24), reprise telle quelle et appliquee a deux runs quelconques : le challenger ne remplace le
champion que s'il est AU MOINS AUSSI BON sur DSO1, DSO4, DSO5 et les vraies mesures (stations, et
radiosondages s'ils existent), avec les memes tolerances. Comme la decision v4, le script ne fait que
LIRE des evaluations deja ecrites (`pcfno3d evaluate`, `scripts/valider_grand_tunis_reel.py`) : il ne
relance rien et rien n'est regle apres lui.

    python scripts/champion_challenger.py --runs <dossier runs> --champion grand_tunis_v2_long \\
        --challenger grand_tunis_v6 [--dossier best_wcont]

Sortie (dans --runs) : champion_challenger_<challenger>.json et .md.
Code de sortie : 0 = le challenger remplace le champion, 3 = le champion reste, 2 = evaluations
incompletes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent


def regle_v4():
    spec = importlib.util.spec_from_file_location("decision_v4_grand_tunis", ICI / "decision_v4_grand_tunis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def comparer(runs: Path, champion: str, challenger: str, dossier: str = "best_wcont") -> dict | None:
    v4 = regle_v4()
    ev = {}
    for nom in (champion, challenger):
        r = v4.lire_eval(runs, nom, dossier)
        if r is None:
            print(f"evaluation absente : {nom}/eval_pcfno3d/{dossier}")
            return None
        st, rs = v4.lire_reel(runs, nom)
        if st is None:
            print(f"validation reelle absente : {nom}/validation_reelle/test/stations_bilan.csv")
            return None
        ev[nom] = (*r, st, rs)
    (s1, l1, st1, rs1), (s2, l2, st2, rs2) = ev[champion], ev[challenger]
    dec = v4.decider(s1, s2, l1, l2, st1, st2, rs1, rs2)
    remplace = bool(dec["v4_livree"])
    recul = [k for k, c in dec["criteres"].items() if not c["ok"]]
    return {"champion": champion, "challenger": challenger, "evaluation": dossier,
            "remplace": remplace, "modele_en_service": challenger if remplace else champion,
            "criteres": dec["criteres"], "tolerances": dec["tolerances"],
            "texte": (f"{challenger} remplace {champion} (au moins aussi bon sur DSO1, DSO4, DSO5 et les "
                      "vraies mesures)") if remplace else
                     f"{champion} reste en service ({challenger} recule sur : {', '.join(recul)})",
            "regle": "scripts/decision_v4_grand_tunis.py (regles d'Ines, 2026-10-24)"}


def _paire(v) -> str:
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return f"{v[0]:.4g} -> {v[1]:.4g}"
    return json.dumps(v, ensure_ascii=False)


def rapport(r: dict) -> str:
    lignes = [f"# Champion et challenger : {r['champion']} contre {r['challenger']}", "",
              f"**Decision : {r['texte']}.**", "",
              f"Evaluation lue : `eval_pcfno3d/{r['evaluation']}` ; regle : {r['regle']}.", "",
              "| Critere | Valeurs (champion -> challenger) | Conforme |", "|---|---|---|"]
    for k, c in r["criteres"].items():
        vals = "; ".join(f"{n} {_paire(v)}" for n, v in c.items() if n != "ok")
        lignes.append(f"| {k} | {vals} | {'oui' if c['ok'] else 'non'} |")
    return "\n".join(lignes) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--champion", required=True, help="run du modele en service")
    ap.add_argument("--challenger", required=True, help="run du nouveau modele")
    ap.add_argument("--dossier", default="best_wcont", help="dossier d'evaluation (forme servie)")
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    r = comparer(runs, a.champion, a.challenger, a.dossier)
    if r is None:
        return 2
    base = runs / f"champion_challenger_{a.challenger}"
    base.with_suffix(".json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    texte = rapport(r)
    base.with_suffix(".md").write_text(texte, encoding="utf-8")
    print(texte)
    return 0 if r["remplace"] else 3


if __name__ == "__main__":
    sys.exit(main())
