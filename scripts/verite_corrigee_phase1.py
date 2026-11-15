"""Phase 1 de la verite corrigee (regle configs/national/regle_verite_corrigee_phase1.json) : faisabilite, CPU.

Correction ERA5 10 m -> station avec les seules variables disponibles sur la grille nationale, retrecie d'un facteur k
choisi sur 2015-2020 (validation croisee par site), jugee en 2021 sur les 12 aeroports (site retire). 2022 et
2023-2024 ne sont pas lues.

    python scripts/verite_corrigee_phase1.py --paires artifacts/national/mos/paires_29 --sortie artifacts/national/verite_corrigee/phase1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
REGLE = REPO / "configs" / "national" / "regle_verite_corrigee_phase1.json"
KS = (1.0, 0.75, 0.5, 0.25)


def lire(dossier: Path, annees) -> pd.DataFrame:
    p = pd.concat([pd.read_csv(dossier / f"paires_{a}.csv.gz", parse_dates=["temps"]) for a in annees])
    return p[(p["temps"].dt.hour % 3 == 0) & (p["temps"].dt.minute == 0)].dropna(subset=["du", "dv"])


def rmse_site(p: pd.DataFrame, cu: str, cv: str) -> pd.Series:
    e = (p["u_obs"] - p[cu]) ** 2 + (p["v_obs"] - p[cv]) ** 2
    return np.sqrt(e.groupby(p["code"]).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paires", type=Path, required=True)
    ap.add_argument("--sortie", type=Path, required=True)
    a = ap.parse_args(argv)
    from pcfno3d.regles import verifier, sortie_neuve
    import mos_era5 as mos
    regle, h = verifier(REGLE)
    sortie = sortie_neuve(a.sortie, "bilan.json")
    print(f"regle {REGLE.name} {h[:12]}... verifiee", flush=True)
    var = regle["variables"]
    cfg = {"famille": "gradient_boosting", "learning_rate": 0.1, "max_depth": 3, "max_iter": 500,
           "min_samples_leaf": 100, "l2_regularization": 1.0}
    fit = lire(a.paires, range(2015, 2021))
    val = lire(a.paires, [2021])
    assert fit["temps"].dt.year.max() <= 2020 and val["temps"].dt.year.unique().tolist() == [2021]
    metar = set(pd.read_csv(REPO / "configs" / "national" / "stations_metar.csv")["code"])
    st = pd.read_csv(a.paires / "stations.csv").set_index("code")
    sites = sorted(fit["code"].unique())

    # 1. k sur 2015-2020 : validation croisee par site (chaque site retire de l'ajustement)
    pred_cv = []
    for s in sites:
        m = mos.ajuster(cfg, fit[fit["code"] != s], var, 2030)
        q = fit[fit["code"] == s]
        X = q[var].to_numpy(float)
        pred_cv.append(q[["code", "u_obs", "v_obs", "u10_era5", "v10_era5"]].assign(
            cu=m["du"].predict(X), cv=m["dv"].predict(X)))
        print(f"  cv {s}", flush=True)
    pc = pd.concat(pred_cv)
    r0 = rmse_site(pc, "u10_era5", "v10_era5")
    choix_k = []
    for k in KS:
        pk = pc.assign(uk=pc["u10_era5"] + k * pc["cu"], vk=pc["v10_era5"] + k * pc["cv"])
        rk = rmse_site(pk, "uk", "vk")
        choix_k.append({"k": k, "rmse_moyen": float(rk.mean()), "rmse_era5": float(r0.mean()),
                        "pire_degradation": float((rk / r0 - 1).max())})
    ck = pd.DataFrame(choix_k)
    ok = ck[ck["pire_degradation"] <= 0.05]
    print(ck.round(4).to_string(index=False), flush=True)
    if ok.empty:
        bilan = {"regle": REGLE.name, "regle_sha256": h, "choix_k_2015_2020": choix_k, "k": None,
                 "phase1_reussie": False, "motif": "aucun k ne garde tous les sites a +5 % au plus sur 2015-2020"}
        (sortie / "bilan.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(bilan, indent=1, ensure_ascii=False))
        return 0
    k = float(ok["k"].max())
    print(f"k retenu sur 2015-2020 : {k}", flush=True)

    # 2. jugement 2021 : modele refait sans le site juge, k fixe (reprise : predictions deterministes, graine 2030,
    # relues si une passe precedente les a ecrites avec le meme k)
    f_pred = sortie / "predictions_2021.csv.gz"
    if f_pred.exists() and (sortie / "k.txt").exists() and float((sortie / "k.txt").read_text()) == k:
        pv = pd.read_csv(f_pred, parse_dates=["temps"])
        print("predictions 2021 relues (meme k)", flush=True)
    else:
        pv = []
    for s in ([] if isinstance(pv, pd.DataFrame) else sorted(val["code"].unique())):
        m = mos.ajuster(cfg, fit[fit["code"] != s], var, 2030)
        q = val[val["code"] == s]
        X = q[var].to_numpy(float)
        pv.append(q[["code", "temps", "u_obs", "v_obs", "u10_era5", "v10_era5"]].assign(
            u_c=q["u10_era5"].to_numpy() + k * m["du"].predict(X), v_c=q["v10_era5"].to_numpy() + k * m["dv"].predict(X)))
    if isinstance(pv, list):
        pv = pd.concat(pv)
        pv["icao"] = pv["code"].map(st["icao"])
        pv["aeroport"] = pv["icao"].isin(metar)
        pv.to_csv(f_pred, index=False)
        (sortie / "k.txt").write_text(str(k))
    res = {}
    rng = np.random.default_rng(2030)
    for nom, sel in (("aeroports", pv[pv["aeroport"]]), ("toutes_stations", pv)):
        e0 = rmse_site(sel, "u10_era5", "v10_era5")
        e1 = rmse_site(sel, "u_c", "v_c")
        jours = sel["temps"].dt.normalize().to_numpy()
        ju = np.unique(jours)
        grp = [np.where(jours == j)[0] for j in ju]
        ec = []
        for _ in range(2000):
            idx = np.concatenate([grp[i] for i in rng.integers(0, len(ju), len(ju))])
            b = sel.iloc[idx]
            ec.append(rmse_site(b, "u_c", "v_c").mean() - rmse_site(b, "u10_era5", "v10_era5").mean())
        ic = (float(np.percentile(ec, 2.5)), float(np.percentile(ec, 97.5)))
        res[nom] = {"sites": int(len(e0)), "rmse_era5": float(e0.mean()), "rmse_corrige": float(e1.mean()),
                    "ecart": float(e1.mean() - e0.mean()), "ic95": ic, "ameliores": int((e1 < e0).sum()),
                    "pire_degradation": float((e1 / e0 - 1).max()),
                    "par_site": {c: [round(float(e0[c]), 3), round(float(e1[c]), 3)] for c in e0.index}}
    ra = res["aeroports"]
    reussie = bool(ra["ic95"][1] < 0 and ra["ameliores"] >= 8 and ra["pire_degradation"] <= 0.05)
    bilan = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "regle": REGLE.name,
             "regle_sha256": h, "choix_k_2015_2020": choix_k, "k": k, "jugement_2021": res,
             "phase1_reussie": reussie, "note": "2022 et 2023-2024 non lues ; ERA5 = reanalyse ; mesures = stations"}
    (sortie / "bilan.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({kk: v for kk, v in bilan.items() if kk != "jugement_2021"}, indent=1, ensure_ascii=False))
    for nom, r in res.items():
        print(nom, {kk: v for kk, v in r.items() if kk != "par_site"})
        print({st.loc[c, "nom"] if c in st.index else c: v for c, v in r["par_site"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
