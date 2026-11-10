"""OS6 — plan de deploiement du reseau ANEO, Grand Tunis ou national (OSSE, `pcfno3d.os6`).

    python scripts/os6_reseau_aneo.py -c grand_tunis_v2_service_colab \
        --sortie /content/drive/MyDrive/pcfno3d_grand_tunis/runs/os6_reseau_aneo

Placement glouton sur la VALIDATION, courbe de dimensionnement sur le TEST (calculee une fois),
plan exporte. Sorties (--sortie) :
    candidats.csv               emplacements possibles apres contraintes (BF74-BF77)
    ordre_validation.csv        ordre d'installation, gain de chaque etape (BF37, BF78)
    plan_aneo.csv / .geojson    plan de deploiement S18 (ordre, lat, lon, hauteur, gain)
    courbe_test_par_niveau.csv  erreur par niveau, par strategie et nombre de capteurs (BF35)
    courbe_test.csv             resume (RMSE vectoriel, pire niveau, seuil) (NET1, NET2)
    courbe.png, carte.png       figures
    os6.json                    resume : nombre recommande (BF36), contraintes, limites
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]


def main() -> int:
    from pcfno3d import os6
    from pcfno3d.config import Config
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.realdata import georef_grille

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="cle=valeur (comme pcfno3d)")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--contraintes", help="JSON des contraintes d'implantation (BF74-BF77)")
    ap.add_argument("--installes", help="CSV lat, lon des capteurs ANEO deja poses : le plan repart d'eux (BF78)")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--n-val", type=int, default=24, help="instants de validation du placement")
    ap.add_argument("--n-test", type=int, default=None, help="instants de test (defaut : tous)")
    ap.add_argument("--par-etape", type=int, default=5)
    ap.add_argument("--n-max", type=int, default=None, help="budget (defaut : contraintes)")
    ap.add_argument("--tailles", default=",".join(str(n) for n in os6.TAILLES))
    ap.add_argument("--lot", type=int, default=8, help="dispositions par passage du reseau")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg.apply_overrides({k: v})
    cfg.reliability.batch = args.lot
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device, verbose=False)
    g = wf.geom
    # projection du relief (plan a la latitude centrale) : sur le domaine national, celle du coin sud-ouest
    # (GeoRef) deplacerait l'est du plan d'environ 15 km ; Grand Tunis : moins de 0,15 km d'ecart
    georef = georef_grille(cfg, g)
    c = os6.Contraintes.charger(args.contraintes)
    if args.n_max:
        c.n_max = args.n_max
    cand = os6.candidats(g, c, georef)
    print(f"modele {wf.checkpoint_path.name} | {len(cand)} emplacements candidats "
          f"(pas {c.pas_candidats_m:g} m, altitude <= {c.altitude_max_m:g} m, pente <= "
          f"{c.pente_max_deg:g} deg, {len(c.exclusions)} zones exclues)", flush=True)
    import pandas as pd
    pd.DataFrame(cand, columns=["iy", "ix"]).to_csv(sortie / "candidats.csv", index=False)

    # 2. placement optimise (validation)
    f_ordre = sortie / "ordre_validation.csv"
    if f_ordre.exists():                       # reprise : le placement est deja fait
        ordre = pd.read_csv(f_ordre)
    else:
        val = os6.Banc(wf, "validation", args.n_val)
        installes = None
        if args.installes:
            if georef is None:
                raise SystemExit("--installes demande un domaine georeference (real.origin_lat/lon)")
            installes = os6.cellules_installes(g, georef, args.installes)
            print(f"OS6 : {len(installes)} capteur(s) deja installe(s) au depart du plan", flush=True)
        ordre = os6.placement_glouton(val, cand, c, par_etape=args.par_etape, installes=installes)
        ordre.to_csv(f_ordre, index=False)

    # 3. courbe de dimensionnement (test, une fois)
    f_courbe = sortie / "courbe_test_par_niveau.csv"
    if f_courbe.exists():
        t = pd.read_csv(f_courbe)
    else:
        test = os6.Banc(wf, "test", args.n_test)
        tailles = [int(x) for x in args.tailles.split(",")]
        t = os6.courbe(test, ordre, cand, c, tailles=tailles)
        t.to_csv(f_courbe, index=False)
    r = os6.resume_courbe(t, cfg.dso.rmse)
    r.to_csv(sortie / "courbe_test.csv", index=False)
    n_rec = os6.nombre_recommande(r)

    # 4. plan (S18)
    p = os6.plan(ordre, g, georef)
    p.to_csv(sortie / "plan_aneo.csv", index=False)
    meta = {"produit_par": "pcfno3d os6", "modele": wf.checkpoint_path.name,
            "configuration": cfg.name, "seuil_rmse_m_s": cfg.dso.rmse,
            "nombre_recommande": n_rec, "contraintes": os6.contraintes_dict(c)}
    if georef is not None:
        (sortie / "plan_aneo.geojson").write_text(
            json.dumps(os6.geojson(p, g, georef, meta), ensure_ascii=False), encoding="utf-8")

    actuel = r[r["strategie"] == "actuel"].iloc[0]
    resume = {
        **meta,
        "candidats": int(len(cand)), "capteurs_places": int(len(p)),
        "reseau_actuel": {"rmse_vect": float(actuel["rmse_vect"]),
                          "pire_rmse_niveau": float(actuel["pire_rmse_niveau"])},
        "courbe": r.to_dict(orient="records"),
        "comparaison_net1": {
            int(n): {s: float(r[(r["strategie"] == s) & (r["n_capteurs"] == n)]["rmse_vect"].iloc[0])
                     for s in ("optimise", "grille", "hasard")}
            for n in sorted(r[r["strategie"] == "optimise"]["n_capteurs"])},
        "limites": ("OSSE sur la verite generee (relief reel, forcage ERA5) ; le modele livre a "
                    "appris avec ERA5 + 3 stations : le gain mesure est celui du modele actuel. "
                    "Plan provisoire jusqu'a la validation sur les premiers capteurs (BF78)."),
        "duree_s": round(time.time() - t0, 1),
    }
    (sortie / "os6.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    figures(sortie, r, p, cand, g, cfg.dso.rmse)
    print(r.round(3).to_string(index=False))
    print(f"nombre recommande (pire RMSE de niveau <= {cfg.dso.rmse} m/s) : "
          f"{n_rec if n_rec is not None else 'non atteint dans la courbe'}")
    print(f"plan -> {sortie}")
    return 0


def figures(sortie: Path, r, p, cand, g, seuil: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for s, st in (("optimise", "o-"), ("grille", "s--"), ("hasard", "^:")):
        q = r[r["strategie"] == s]
        ax.plot(q["n_capteurs"], q["pire_rmse_niveau"], st, label=s)
    a = r[r["strategie"] == "actuel"]["pire_rmse_niveau"].iloc[0]
    ax.axhline(a, color="grey", lw=1, label="reseau actuel (ERA5 + 3 stations)")
    ax.axhline(seuil, color="red", lw=1, ls="--", label=f"seuil {seuil} m/s")
    ax.set_xlabel("capteurs ANEO ajoutes")
    ax.set_ylabel("pire RMSE de niveau, u ou v (m/s), test")
    ax.legend(fontsize=8)
    fig.savefig(sortie / "courbe.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6.6), constrained_layout=True)
    ext = (0, g.nx * g.dx / 1000, 0, g.ny * g.dy / 1000)
    im = ax.imshow(np.where(g.z0 >= 0.005, g.zt, np.nan), origin="lower", cmap="terrain",
                   extent=ext)
    fig.colorbar(im, ax=ax, shrink=0.7, label="altitude (m)")
    ax.scatter(cand[:, 1] * g.dx / 1000, cand[:, 0] * g.dy / 1000, s=1, c="k", alpha=0.15)
    sc = ax.scatter(p["x_m"] / 1000, p["y_m"] / 1000, c=p["ordre"], cmap="plasma_r", s=22,
                    edgecolors="k", linewidths=0.4)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="ordre d'installation")
    from pcfno3d.dataset import cellules_stations
    st = cellules_stations(g)
    if len(st):
        ax.plot(st[:, 1] * g.dx / 1000, st[:, 0] * g.dy / 1000, "w^", ms=10, mec="k",
                label="stations existantes")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("km")
    ax.set_title("Plan de deploiement ANEO propose (OS6)")
    fig.savefig(sortie / "carte.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
