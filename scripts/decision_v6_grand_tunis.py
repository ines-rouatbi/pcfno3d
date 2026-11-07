"""Comparaison v4 / v6 (Grand Tunis) et regle de livraison de la v6.

Regle fixee le 2026-11-07, AVANT l'entrainement de la v6 et toute lecture de son test (demande
d'Ines : des resultats conformes aux DSO). Ce script ne fait que LIRE les evaluations deja ecrites
(`pcfno3d evaluate`, `scripts/valider_grand_tunis_reel.py`) ; il ne relance rien et rien n'est
regle apres lui. Les deux modeles sont compares sous leur forme livree (w par continuite).

La v6 est livree SEULEMENT si, sur le test 2022 :
  DSO1   conforme : RMSE de u et v <= dso.rmse a CHAQUE niveau evalue (la v4 echoue a 25 m), R2 de
         u et v conformes, RMSE globale conforme ;
  DSO3   conforme : rugosite dans dso.roughness_range (la v4 : 0,805) ;
  DSO2   conforme (garanti par construction avec w par continuite ; valeur mesuree publiee) ;
  DSO4   conforme ; DSO5 conforme et RMSE a 25 % des capteurs <= v4 x (1 + TOL) ;
  reel   erreur vectorielle moyenne des 3 stations <= v4 x (1 + TOL), aucune station plus de
         TOL_STATION au-dessus de la v4 ; radiosondages (si presents) <= v4 x (1 + TOL).
Sinon la v4 reste le modele livre. Le sous-verdict « R2 de u, v ET w par niveau » (dso1_r2_uvw),
non conforme pour la v4 (w par continuite aux niveaux hauts), n'est pas exige ici : il est publie
pour les deux modeles.

    python scripts/decision_v6_grand_tunis.py --runs /content/drive/MyDrive/pcfno3d_grand_tunis/runs

Sortie (dans --runs) : decision_v6.json, comparaison_v4_v6_par_niveau.csv, rapport_v6.md.
decision_v4.json (lu par le service) n'est PAS modifie : si la v6 est livree, le service bascule
sur la decision d'Ines (lire decision_v6.json).
Code de sortie : 0 = v6 livree, 3 = la v4 reste, 2 = evaluations incompletes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decision_v4_grand_tunis import dso2, evalues, lire_eval, lire_reel, reel_moyen  # noqa: E402

V4 = "grand_tunis_v4_dso"
V6 = "grand_tunis_v6_dso"
TOL = 0.01
TOL_STATION = 0.03


def par_niveau(l4: pd.DataFrame, l6: pd.DataFrame) -> pd.DataFrame:
    t = evalues(l4)[["z_m", "couverture"]].copy()
    t["z_m"] = t["z_m"].astype(int)
    for nom, lv in (("v4", l4), ("v6", l6)):
        e = evalues(lv).set_index("z_m")
        for col in ("rmse_u", "rmse_v", "r2_u", "r2_v", "r2_w", "rugosite"):
            if col in e:
                t[f"{col}_{nom}"] = t["z_m"].map(e[col])
    return t


def decider(s4: dict, s6: dict, l4: pd.DataFrame, l6: pd.DataFrame, st4, st6, rs4, rs6, rugosite) -> dict:
    lo, hi = rugosite
    pire = tuple(float(evalues(lv)[["rmse_u", "rmse_v"]].max().max()) for lv in (l4, l6))
    crit = {
        "DSO1": {"pire_rmse_niveau": pire,
                 "rmse_global": ((s4["dso1_rmse_global_u"], s4["dso1_rmse_global_v"]),
                                 (s6["dso1_rmse_global_u"], s6["dso1_rmse_global_v"])),
                 "ok": bool(s6["dso1_rmse"] and s6["dso1_r2"] and s6["dso1_rmse_global_conforme"])},
        "DSO3": {"rugosite": (s4["dso3_rugosite"], s6["dso3_rugosite"]), "intervalle": [lo, hi],
                 "ok": bool(s6["dso3_conforme"])},
        "DSO2": {"residu_masse": (dso2(s4), dso2(s6)), "ok": bool(s6["dso2_conforme"])},
        "DSO4": {"temps_s": (s4["dso4_inference_s"], s6["dso4_inference_s"]),
                 "ok": bool(s6["dso4_conforme"])},
        "DSO5": {"rmse_25pct": (s4["dso5_rmse_25pct"], s6["dso5_rmse_25pct"]),
                 "ok": bool(s6["dso5_conforme"] and s6["dso5_rmse_25pct"] <= s4["dso5_rmse_25pct"] * (1 + TOL))},
    }
    m4, par4 = reel_moyen(st4)
    m6, par6 = reel_moyen(st6)
    recul = [s for s in par4 if par6.get(s, float("inf")) > par4[s] * (1 + TOL_STATION)]
    reel = {"stations_moyenne": (m4, m6), "stations": {s: (par4[s], par6.get(s)) for s in par4},
            "stations_en_recul": recul}
    ok = m6 <= m4 * (1 + TOL) and not recul
    if rs4 is not None and rs6 is not None:
        r = (float(rs4["rmse_vect_modele"].mean()), float(rs6["rmse_vect_modele"].mean()))
        reel["radiosondages_moyenne"] = r
        ok = ok and r[1] <= r[0] * (1 + TOL)
    reel["ok"] = bool(ok)
    crit["mesures_reelles"] = reel
    livree = all(c["ok"] for c in crit.values())
    return {"criteres": crit, "v6_livree": livree, "modele_livre": V6 if livree else V4,
            "w_continuite": True, "dso2_w_continuite": "garanti par construction",
            "dso1_r2_uvw": {"v4": s4.get("dso1_r2_uvw"), "v6": s6.get("dso1_r2_uvw")},
            "texte": ("v6 livree : DSO1 et DSO3 conformes, aucun recul sur DSO2, DSO4, DSO5 et les "
                      "vraies mesures") if livree else
                     ("la v4 reste livree (la v6 manque : "
                      + ", ".join(k for k, c in crit.items() if not c["ok"]) + ")"),
            "tolerances": {"rmse_relative": TOL, "station_relative": TOL_STATION}}


def tableau_md(t: pd.DataFrame) -> str:
    """Tableau Markdown sans dependance (pas de tabulate)."""
    tete = "| " + " | ".join(map(str, t.columns)) + " |"
    sep = "|" + "---|" * len(t.columns)
    return "\n".join([tete, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)])


def rapport(t: pd.DataFrame, dec: dict) -> str:
    c = dec["criteres"]
    lignes = ["# Decision v6 (Grand Tunis) — regle fixee le 07/11/2026 avant le test", "",
              f"**{dec['texte']}**", "",
              "| critere | v4 | v6 | ok |", "|---|---|---|---|"]
    for k, v in c.items():
        vals = [x for n, x in v.items() if n not in ("ok", "stations", "stations_en_recul", "intervalle")]
        a = vals[0] if vals else ("", "")
        lignes.append(f"| {k} | {a[0]} | {a[1]} | {'oui' if v['ok'] else 'non'} |")
    lignes += ["", "DSO2 avec w par continuite : garanti par construction (valeurs mesurees ci-dessus).",
               f"dso1_r2_uvw (R2 de u, v ET w par niveau), publie, non exige : {dec['dso1_r2_uvw']}", "",
               "## Par niveau (test 2022, w par continuite)", "", tableau_md(t.round(3))]
    return "\n".join(lignes) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rugosite", default="0.9,1.1", help="dso.roughness_range")
    args = ap.parse_args()
    runs = Path(args.runs)
    e4, e6 = lire_eval(runs, V4, "best_wcont"), lire_eval(runs, V6, "best_wcont")
    if e4 is None or e6 is None:
        print("evaluation absente (best_wcont de la v4 ou de la v6)")
        return 2
    st4, rs4 = lire_reel(runs, V4)
    st6, rs6 = lire_reel(runs, V6)
    if st4 is None or st6 is None:
        print("validation reelle absente (stations_bilan.csv de la v4 ou de la v6)")
        return 2
    rug = tuple(float(x) for x in args.rugosite.split(","))
    dec = decider(e4[0], e6[0], e4[1], e6[1], st4, st6, rs4, rs6, rug)
    t = par_niveau(e4[1], e6[1])
    t.to_csv(runs / "comparaison_v4_v6_par_niveau.csv", index=False)
    (runs / "decision_v6.json").write_text(json.dumps(dec, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")
    texte = rapport(t, dec)
    (runs / "rapport_v6.md").write_text(texte, encoding="utf-8")
    print(texte)
    return 0 if dec["v6_livree"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
