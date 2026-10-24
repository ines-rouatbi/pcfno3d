"""Comparaison v2 / v4 (Grand Tunis) niveau par niveau et regle de livraison de la v4.

Regles d'Ines (2026-10-24), appliquees ici telles quelles :
  - le test n'est calcule qu'UNE fois : ce script ne fait que LIRE les evaluations deja ecrites
    (`pcfno3d evaluate`, `scripts/valider_grand_tunis_reel.py`, `pcfno3d dso6`) ; il ne relance
    rien et rien n'est regle apres lui ;
  - DSO1 et DSO3 cote a cote (v2 contre v4), niveau par niveau ; R2 de w par niveau avec et
    sans continuite ; DSO2 avec w par continuite = « garanti par construction » (la valeur
    mesuree est donnee a cote) ;
  - la v4 n'est livree que si elle est AU MOINS AUSSI BONNE que la v2 sur DSO1, DSO4, DSO5 et
    les vraies mesures ; sinon on livre la v2 avec w par continuite. Apres ca, plus aucun
    entrainement.

« Au moins aussi bonne » est fixe ICI, avant toute lecture du test de la v4 (les deux modeles
sont compares sous leur forme livree, w par continuite) :
  DSO1  tout sous-verdict DSO1 conforme en v2 l'est en v4 ; RMSE global u et v, pire RMSE de
        niveau (u, v) <= v2 x (1 + TOL) ; pire R2 de niveau (u, v, w) >= v2 - TOL_R2 ;
  DSO4  conforme en v4 si conforme en v2 ;
  DSO5  RMSE max a 25 % des capteurs <= v2 x (1 + TOL), et conforme si v2 conforme ;
  reel  erreur vectorielle moyenne des 3 stations <= v2 x (1 + TOL), aucune station plus de
        TOL_STATION au-dessus de la v2 ; radiosondages (si presents) : moyenne des tranches
        <= v2 x (1 + TOL).
TOL = 1 % couvre les arrondis et la variabilite de la mesure du temps, pas un recul reel.

    python scripts/decision_v4_grand_tunis.py --runs /content/drive/MyDrive/pcfno3d_grand_tunis/runs
    python scripts/decision_v4_grand_tunis.py --runs ... --dso6 <verdict dso6 du modele retenu>

Variantes PROPOSEES (DSO1 sur le R2 de u et v seulement, w a part ; DSO3 sur la partie previsible
de la verite, scripts/dso_variantes_grand_tunis.py) : publiees dans le rapport, jamais lues par la
decision tant que l'encadrant ne les a pas validees.

Sortie (dans --runs) : decision_v4.json, comparaison_v2_v4_par_niveau.csv, rapport_v4.md.
Code de sortie : 0 = v4 livree, 3 = v2 livree (w par continuite), 2 = evaluations incompletes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

V2 = "grand_tunis_v2_long"
V4 = "grand_tunis_v4_dso"
TOL = 0.01                 # 1 % sur les RMSE (arrondis, bruit de mesure du temps)
TOL_R2 = 0.005             # sur les R2
TOL_STATION = 0.03         # aucune station plus de 3 % au-dessus de la v2
MIN_COUVERTURE = 0.05      # niveaux evalues (dso.min_level_cover)
SOUS_VERDICTS_DSO1 = ("dso1_rmse", "dso1_r2", "dso1_r2_uvw", "dso1_rmse_global_conforme")


def lire_eval(runs: Path, run: str, dossier: str) -> tuple[dict, pd.DataFrame] | None:
    d = runs / run / "eval_pcfno3d" / dossier
    f, n = d / "test_summary.json", d / "test_par_niveau_synoptique.csv"
    if not (f.exists() and n.exists()):
        return None
    return json.loads(f.read_text(encoding="utf-8")), pd.read_csv(n)


def lire_reel(runs: Path, run: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    d = runs / run / "validation_reelle" / "test"
    st = d / "stations_bilan.csv"
    rs = d / "radiosondages_bilan.csv"
    return (pd.read_csv(st) if st.exists() else None,
            pd.read_csv(rs) if rs.exists() else None)


def evalues(lv: pd.DataFrame) -> pd.DataFrame:
    return lv[lv["couverture"] >= MIN_COUVERTURE].reset_index(drop=True)


def par_niveau(v2c: pd.DataFrame, v2b: pd.DataFrame, v4c: pd.DataFrame,
               v4b: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par niveau evalue : DSO1 (forme livree, w par continuite), R2 de w avec et
    sans continuite, DSO3 du niveau, pour la v2 et la v4."""
    t = pd.DataFrame({"z_m": evalues(v2c)["z_m"].astype(int),
                      "couverture": evalues(v2c)["couverture"]})
    for nom, cont, brut in (("v2", v2c, v2b), ("v4", v4c, v4b)):
        c, b = evalues(cont).set_index("z_m"), evalues(brut).set_index("z_m")
        for col in ("rmse_u", "rmse_v", "r2_u", "r2_v"):
            t[f"{col}_{nom}"] = t["z_m"].map(c[col])
        t[f"r2_w_continuite_{nom}"] = t["z_m"].map(c["r2_w"])
        t[f"r2_w_reseau_{nom}"] = t["z_m"].map(b["r2_w"])
        if "rugosite" in c:
            t[f"rugosite_{nom}"] = t["z_m"].map(c["rugosite"])
    return t


def reel_moyen(st: pd.DataFrame) -> tuple[float, dict]:
    m = st[st["source"] == "modele"].set_index("station")["rmse_vect"]
    return float(m.mean()), {k: float(x) for k, x in m.items()}


def decider(s2: dict, s4: dict, lv2: pd.DataFrame, lv4: pd.DataFrame,
            st2: pd.DataFrame, st4: pd.DataFrame,
            rs2: pd.DataFrame | None, rs4: pd.DataFrame | None) -> dict:
    """Les criteres de la docstring du module, chacun avec ses valeurs."""
    e2, e4 = evalues(lv2), evalues(lv4)
    crit: dict[str, dict] = {}

    perdus = [k for k in SOUS_VERDICTS_DSO1 if s2.get(k) and not s4.get(k)]
    pire_rmse = (float(e2[["rmse_u", "rmse_v"]].max().max()),
                 float(e4[["rmse_u", "rmse_v"]].max().max()))
    pire_r2 = (float(e2[["r2_u", "r2_v", "r2_w"]].min().min()),
               float(e4[["r2_u", "r2_v", "r2_w"]].min().min()))
    glob = {c: (s2[f"dso1_rmse_global_{c}"], s4[f"dso1_rmse_global_{c}"]) for c in "uv"}
    crit["DSO1"] = {
        "sous_verdicts_perdus": perdus,
        "rmse_global_u": glob["u"], "rmse_global_v": glob["v"],
        "pire_rmse_niveau": pire_rmse, "pire_r2_niveau_uvw": pire_r2,
        "ok": bool(not perdus
                   and all(b <= a * (1 + TOL) for a, b in glob.values())
                   and pire_rmse[1] <= pire_rmse[0] * (1 + TOL)
                   and pire_r2[1] >= pire_r2[0] - TOL_R2)}
    crit["DSO4"] = {"temps_s": (s2["dso4_inference_s"], s4["dso4_inference_s"]),
                    "ok": bool(s4["dso4_conforme"] or not s2["dso4_conforme"])}
    d5 = (s2["dso5_rmse_25pct"], s4["dso5_rmse_25pct"])
    crit["DSO5"] = {"rmse_25pct": d5,
                    "ok": bool(d5[1] <= d5[0] * (1 + TOL)
                               and (s4["dso5_conforme"] or not s2["dso5_conforme"]))}
    m2, par2 = reel_moyen(st2)
    m4, par4 = reel_moyen(st4)
    pires = [s for s in par2 if par4.get(s, float("inf")) > par2[s] * (1 + TOL_STATION)]
    reel = {"stations_moyenne": (m2, m4), "stations": {s: (par2[s], par4.get(s)) for s in par2},
            "stations_en_recul": pires}
    ok_reel = m4 <= m2 * (1 + TOL) and not pires
    if rs2 is not None and rs4 is not None:
        r = (float(rs2["rmse_vect_modele"].mean()), float(rs4["rmse_vect_modele"].mean()))
        reel["radiosondages_moyenne"] = r
        ok_reel = ok_reel and r[1] <= r[0] * (1 + TOL)
    reel["ok"] = bool(ok_reel)
    crit["mesures_reelles"] = reel

    v4_livree = all(c["ok"] for c in crit.values())
    return {"criteres": crit, "v4_livree": v4_livree,
            "modele_livre": V4 if v4_livree else V2,
            "w_continuite": True,
            "texte": ("v4 livree (au moins aussi bonne que la v2 sur DSO1, DSO4, DSO5 et les "
                      "vraies mesures), w par continuite") if v4_livree else
                     ("v2 livree avec w par continuite (la v4 recule sur : "
                      + ", ".join(k for k, c in crit.items() if not c["ok"]) + ")"),
            "tolerances": {"rmse_relative": TOL, "r2_absolue": TOL_R2,
                           "station_relative": TOL_STATION}}


STATUT_VARIANTES = ("proposees, NON utilisees pour decider (en attente de validation par "
                    "l'encadrant)")


def variantes(runs: Path, ev: dict, lv: dict) -> dict:
    """DSO1 sur le R2 de u et v seulement (w rapporte a part) et DSO3 sur la partie previsible
    de la verite (scripts/dso_variantes_grand_tunis.py), pour la v2 et la v4 (forme livree).
    Calculees et publiees ; la decision ne les lit pas."""
    out = {"statut": STATUT_VARIANTES}
    for nom, run, cle in (("v2", V2, "v2c"), ("v4", V4, "v4c")):
        e = evalues(lv[cle])
        f = runs / run / "eval_pcfno3d" / "best_wcont" / "dso3_previsible.json"
        d3 = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        out[nom] = {
            "dso1_uv_conforme": bool(ev[cle]["dso1_rmse"] and ev[cle]["dso1_r2"]),
            "pire_r2_uv": float(e[["r2_u", "r2_v"]].min().min()),
            "r2_w_par_niveau": {int(z): float(r) for z, r in zip(e["z_m"], e["r2_w"])},
            "dso3_verite": ev[cle]["dso3_rugosite"],
            "dso3_previsible": None if d3 is None else d3["dso3_rugosite_previsible"],
            "dso3_previsible_conforme": None if d3 is None else d3["dso3_previsible_conforme"],
            "verite_sur_previsible": None if d3 is None else d3["verite_sur_previsible"],
        }
    return out


def dso2(s: dict) -> float:
    return float(s["dso2_rms_moyen"])


def rapport(t: pd.DataFrame, ev: dict, dec: dict, dso6: dict | None,
            var: dict | None = None) -> str:
    f = lambda x: "—" if pd.isna(x) else f"{x:.3f}"            # noqa: E731
    L = ["# v2 contre v4 — Grand Tunis (test synoptique, calcule une seule fois)", "",
         f"**Decision : {dec['texte']}.** Apres ca, plus aucun entrainement.", "",
         "## DSO1 et DSO3 par niveau (forme livree : w par continuite)", "",
         "| z (m) | RMSE u v2 | RMSE u v4 | RMSE v v2 | RMSE v v4 | R2 u v2 | R2 u v4 "
         "| R2 v v2 | R2 v v4 | rugosite v2 | rugosite v4 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        L.append(f"| {int(r['z_m'])} | " + " | ".join(f(r.get(c)) for c in (
            "rmse_u_v2", "rmse_u_v4", "rmse_v_v2", "rmse_v_v4", "r2_u_v2", "r2_u_v4",
            "r2_v_v2", "r2_v_v4", "rugosite_v2", "rugosite_v4")) + " |")
    L += ["", "## R2 de w par niveau, avec et sans continuite", "",
          "| z (m) | v2 reseau | v2 continuite | v4 reseau | v4 continuite |", "|---|---|---|---|---|"]
    for _, r in t.iterrows():
        L.append(f"| {int(r['z_m'])} | " + " | ".join(f(r[c]) for c in (
            "r2_w_reseau_v2", "r2_w_continuite_v2", "r2_w_reseau_v4", "r2_w_continuite_v4")) + " |")
    L += ["", "## Verdicts globaux", "",
          "| | v2 reseau | v2 continuite | v4 reseau | v4 continuite |", "|---|---|---|---|---|"]
    cles = (("RMSE global u", lambda s: f(s["dso1_rmse_global_u"])),
            ("RMSE global v", lambda s: f(s["dso1_rmse_global_v"])),
            ("DSO2 residu de masse", lambda s: f"{dso2(s):.2e}"),
            ("DSO3 rugosite", lambda s: f(s["dso3_rugosite"])),
            ("DSO4 temps (s)", lambda s: f(s["dso4_inference_s"])),
            ("DSO5 RMSE max 25 %", lambda s: f(s["dso5_rmse_25pct"])))
    for nom, val in cles:
        L.append(f"| {nom} | " + " | ".join(val(ev[k]) for k in ("v2b", "v2c", "v4b", "v4c")) + " |")
    L += ["", "DSO2 avec w par continuite : **garanti par construction** (w est calcule pour "
          "annuler la divergence de rho·(u, v, w) colonne par colonne) ; la valeur mesuree "
          "ci-dessus, avec la densite de la verite, est donnee a titre de controle.", "",
          "## Criteres de livraison", ""]
    for k, c in dec["criteres"].items():
        det = {a: b for a, b in c.items() if a != "ok"}
        L.append(f"- **{k}** : {'OK' if c['ok'] else 'RECUL'} — `{json.dumps(det, ensure_ascii=False)}`")
    if dso6 is not None:
        g = dso6["global"]
        L += ["", f"## DSO6 — ensemble par masquage, modele retenu ({dso6.get('modele')}, sans "
              "reentrainement)", "",
              f"{dso6.get('instants_synoptiques')} instants de test, calibration sur la "
              f"validation (sha {str(dso6.get('calibration_sha256'))[:8]}), "
              f"{dso6.get('uncertainty')}.", "",
              "| | u | v | w | vitesse |", "|---|---|---|---|---|",
              "| couverture de l'intervalle 80 % | " + " | ".join(
                  f(g.get(f"couverture80_{c}")) for c in ("u", "v", "w", "speed")) + " |",
              "| Spearman incertitude / erreur | " + " | ".join(
                  f(g.get(f"spearman_{c}")) for c in ("u", "v", "w", "speed")) + " |", "",
              f"DSO6 : **{'CONFORME' if dso6.get('dso6_conforme') else 'NON CONFORME'}**"]
    if var is not None:
        def cellule(m: str, cle: str) -> str:
            x = var[m][cle]
            return "non calcule" if x is None else f(x)
        L += ["", f"## Variantes {var['statut']}", "",
              "| | v2 | v4 |", "|---|---|---|",
              "| DSO1 u, v (RMSE et R2 de u, v par niveau) | " + " | ".join(
                  "conforme" if var[m]["dso1_uv_conforme"] else "non conforme"
                  for m in ("v2", "v4")) + " |",
              "| pire R2 u, v par niveau | " + " | ".join(
                  cellule(m, "pire_r2_uv") for m in ("v2", "v4")) + " |",
              "| DSO3 sur la verite (officiel) | " + " | ".join(
                  cellule(m, "dso3_verite") for m in ("v2", "v4")) + " |",
              "| DSO3 sur la partie previsible | " + " | ".join(
                  cellule(m, "dso3_previsible") for m in ("v2", "v4")) + " |",
              "| rugosite verite / previsible | " + " | ".join(
                  cellule(m, "verite_sur_previsible") for m in ("v2", "v4")) + " |", "",
              "R2 de w, rapporte a part dans la variante DSO1 : tableau « R2 de w par niveau » "
              "ci-dessus. Aucun bruit n'est ajoute aux predictions."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--dso6", help="verdict DSO6 (JSON de `pcfno3d dso6`) du modele retenu")
    args = ap.parse_args()
    runs = Path(args.runs)
    ev, lv = {}, {}
    for cle, run, dossier in (("v2b", V2, "best"), ("v2c", V2, "best_wcont"),
                              ("v4b", V4, "best"), ("v4c", V4, "best_wcont")):
        r = lire_eval(runs, run, dossier)
        if r is None:
            print(f"evaluation absente : {run}/eval_pcfno3d/{dossier}")
            return 2
        ev[cle], lv[cle] = r
    st2, rs2 = lire_reel(runs, V2)
    st4, rs4 = lire_reel(runs, V4)
    if st2 is None or st4 is None:
        print("validation reelle absente (stations_bilan.csv de la v2 ou de la v4)")
        return 2

    t = par_niveau(lv["v2c"], lv["v2b"], lv["v4c"], lv["v4b"])
    dec = decider(ev["v2c"], ev["v4c"], lv["v2c"], lv["v4c"], st2, st4, rs2, rs4)
    dec["dso2_residu_masse"] = {k: dso2(ev[k]) for k in ev}
    dec["dso2_w_continuite"] = "garanti par construction"
    var = variantes(runs, ev, lv)                # publiees, jamais lues par `decider`
    dec["variantes_proposees"] = var
    dso6 = json.loads(Path(args.dso6).read_text(encoding="utf-8")) if args.dso6 else None
    if dso6 is not None:
        dec["dso6"] = {k: dso6.get(k) for k in ("modele", "dso6_conforme", "global",
                                                "calibration_sha256", "instants_synoptiques")}
    t.to_csv(runs / "comparaison_v2_v4_par_niveau.csv", index=False)
    (runs / "decision_v4.json").write_text(json.dumps(dec, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    texte = rapport(t, ev, dec, dso6, var)
    (runs / "rapport_v4.md").write_text(texte, encoding="utf-8")
    print(texte)
    return 0 if dec["v4_livree"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
