"""OS6 — plan de deploiement du reseau ANEO (OSSE, `pcfno3d.os6`) : Grand Tunis, ou Tunisie entiere
avec --regle (phase 5 nationale).

    python scripts/os6_reseau_aneo.py -c grand_tunis_v2_service_colab \
        --sortie /content/drive/MyDrive/pcfno3d_grand_tunis/runs/os6_reseau_aneo

    python scripts/os6_reseau_aneo.py -c national_os6_D --regle configs/national/regle_os6_national.json \
        --modele-info <sortie>/choix_modele.json --sortie <Drive>/pcfno3d_national/runs/os6_national_D_s<g> \
        [--degradation <Drive>/pcfno3d_national/runs/degradation_era5_v4.json] [--zones-exclues zones.geojson]

Placement glouton sur la VALIDATION, courbe de dimensionnement sur le TEST (calculee une fois),
plan exporte. Sorties (--sortie), les memes pour le Grand Tunis et le national :
    candidats.csv               emplacements possibles apres contraintes (BF74-BF77)
    ordre_validation.csv        ordre d'installation, gain de chaque etape (BF37, BF78)
    plan_aneo.csv / .geojson    plan de deploiement S18 (ordre, lat, lon, hauteur, gain)
    courbe_test_par_niveau.csv  erreur par niveau, par strategie et nombre de capteurs (BF35)
    courbe_test.csv             resume (RMSE vectoriel, pire niveau, seuil) (NET1, NET2)
    courbe.png, carte.png       figures
    os6.json                    resume : nombre recommande (BF36), contraintes, limites
En plus, avec --regle (national) :
    regle_os6.json              copie de la regle verifiee (une reprise avec une autre regle est refusee)
    courbe_test_par_tranche.csv erreur par tranche de hauteur au-dessus du sol (0-1 km)
    courbe_test.csv             + colonnes rmse_vect_sol, rmse_max_uv_sol (critere du seuil)
    courbe_test_era5_degradee*.csv   courbe d'INFORMATION, ERA5 degradee (si --degradation)
Mode national : capteur = verite d'entrainement a la hauteur du mat + bruit (comme l'entrainement du
candidat D, `os6.CapteursSimules`), metrique 0-1 km au-dessus du sol, georeferencement de la grille.
"""
from __future__ import annotations

import argparse
import json
import shutil
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
    from pcfno3d.realdata import GeoRef

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="cle=valeur (comme pcfno3d)")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--contraintes", help="JSON des contraintes d'implantation (BF74-BF77)")
    ap.add_argument("--zones-exclues", help="GeoJSON de zones exclues (Polygon / MultiPolygon, lon, lat)")
    ap.add_argument("--installes", help="CSV lat, lon des capteurs ANEO deja poses : le plan repart d'eux (BF78)")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--n-val", type=int, default=24, help="instants de validation du placement")
    ap.add_argument("--n-test", type=int, default=None, help="instants de test (defaut : tous)")
    ap.add_argument("--par-etape", type=int, default=5)
    ap.add_argument("--n-max", type=int, default=None, help="budget (defaut : contraintes)")
    ap.add_argument("--tailles", default=",".join(str(n) for n in os6.TAILLES))
    ap.add_argument("--lot", type=int, default=8, help="dispositions par passage du reseau")
    # mode national (phase 5) : tout est fixe par la regle hachee ; les options seules servent aux essais
    ap.add_argument("--regle", help="regle OS6 nationale (JSON + .sha256 verifie) : fixe tous les reglages")
    ap.add_argument("--hauteur-capteur", type=float, default=None,
                    help="capteurs simules a cette hauteur au-dessus du sol (comme l'entrainement de D)")
    ap.add_argument("--graine-capteurs", type=int, default=0, help="graine du bruit des capteurs simules")
    ap.add_argument("--metrique", choices=("niveaux", "sol"), default="niveaux")
    ap.add_argument("--georef", choices=("cfg", "grille"), default="cfg")
    ap.add_argument("--rayon-influence-m", type=float, default=1500.0)
    ap.add_argument("--graines-hasard", default="0,1,2")
    ap.add_argument("--degradation", help="JSON de la degradation d'ERA5 (candidat D) : courbe d'information")
    ap.add_argument("--modele-info", help="JSON du choix du modele (graine retenue), recopie dans os6.json")
    args = ap.parse_args()

    regle, sha_regle = None, None
    if args.regle:
        regle, sha_regle = os6.charger_regle(args.regle)
        P = regle["parametres"]
        args.hauteur_capteur = float(P["hauteur_capteur_m"])
        args.graine_capteurs = int(P["graine_capteurs"])
        args.metrique, args.georef = P["metrique"], P["georeferencement"]
        args.rayon_influence_m = float(P["rayon_influence_m"])
        args.graines_hasard = ",".join(str(g) for g in P["graines_hasard"])
        args.n_val, args.n_test = int(P["n_val"]), P["n_test"]
        args.par_etape, args.n_max = int(P["par_etape"]), int(P["n_max"])
        args.tailles = ",".join(str(n) for n in P["tailles"])
        if not P.get("courbe_information_era5_degradee"):
            args.degradation = None
        print(f"regle {Path(args.regle).name} (sha256 {sha_regle[:12]}..., verifiee)", flush=True)

    cfg = Config.load(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg.apply_overrides({k: v})
    cfg.reliability.batch = args.lot
    seuil = float(regle["parametres"]["seuil_m_s"]) if regle else cfg.dso.rmse
    critere = "rmse_max_uv_sol" if args.metrique == "sol" else "pire_rmse_niveau"
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    if regle is not None:                     # un dossier = une regle : jamais de melange a la reprise
        copie = sortie / "regle_os6.json"
        if copie.exists() and copie.read_bytes() != Path(args.regle).read_bytes():
            raise SystemExit(f"{sortie} a ete calcule avec une autre regle : choisir un dossier neuf")
        shutil.copyfile(args.regle, copie)
    t0 = time.time()

    wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device, verbose=False)
    g = wf.geom
    if args.georef == "grille":
        georef = os6.GeoRefGrille.depuis_geometrie(g)
    else:
        try:
            georef = GeoRef.from_cfg(cfg)
        except ValueError:
            georef = None
    if regle is not None:                     # national : contraintes de la regle, pas d'un fichier libre
        if args.contraintes:
            raise SystemExit("--regle fixe les contraintes : --contraintes refuse (zones : --zones-exclues)")
        c = os6.Contraintes(**regle["parametres"]["contraintes"])
    else:
        c = os6.Contraintes.charger(args.contraintes)
    if args.n_max:
        c.n_max = args.n_max
    sha_zones = None
    if args.zones_exclues:
        import hashlib
        sha_zones = hashlib.sha256(Path(args.zones_exclues).read_bytes()).hexdigest()
        c.exclusions = list(c.exclusions) + os6.zones_geojson(args.zones_exclues)
    cand = os6.candidats(g, c, georef)
    print(f"modele {wf.checkpoint_path.name} | {len(cand)} emplacements candidats "
          f"(pas {c.pas_candidats_m:g} m, altitude <= {c.altitude_max_m:g} m, pente <= "
          f"{c.pente_max_deg:g} deg, {len(c.exclusions)} zones exclues)", flush=True)
    import pandas as pd
    pd.DataFrame(cand, columns=["iy", "ix"]).to_csv(sortie / "candidats.csv", index=False)

    capteurs = (os6.CapteursSimules(g, args.hauteur_capteur, args.graine_capteurs)
                if args.hauteur_capteur is not None else None)
    banc = dict(capteurs=capteurs, metrique=args.metrique)
    graines = tuple(int(x) for x in args.graines_hasard.split(","))

    # 2. placement optimise (validation)
    f_ordre = sortie / "ordre_validation.csv"
    if f_ordre.exists():                       # reprise : le placement est deja fait
        ordre = pd.read_csv(f_ordre)
    else:
        val = os6.Banc(wf, "validation", args.n_val, **banc)
        installes = None
        if args.installes:
            if georef is None:
                raise SystemExit("--installes demande un domaine georeference (real.origin_lat/lon)")
            installes = os6.cellules_installes(g, georef, args.installes)
            print(f"OS6 : {len(installes)} capteur(s) deja installe(s) au depart du plan", flush=True)
        ordre = os6.placement_glouton(val, cand, c, par_etape=args.par_etape,
                                      rayon_influence_m=args.rayon_influence_m, installes=installes)
        ordre.to_csv(f_ordre, index=False)

    # 3. courbe de dimensionnement (test, une fois ; une reprise relit les fichiers)
    tailles = [int(x) for x in args.tailles.split(",")]
    sol = args.metrique == "sol"

    def la_courbe(prefixe: str, degradation) -> pd.DataFrame:
        f = sortie / f"{prefixe}_par_niveau.csv"
        if f.exists():
            return pd.read_csv(f)
        test = os6.Banc(wf, "test", args.n_test, degradation=degradation, **banc)
        if sol:
            t, tt = os6.courbe(test, ordre, cand, c, tailles=tailles, graines=graines, par_tranche=True)
            tt.to_csv(sortie / f"{prefixe}_par_tranche.csv", index=False)
        else:
            t = os6.courbe(test, ordre, cand, c, tailles=tailles, graines=graines)
        t.to_csv(f, index=False)               # en dernier : sa presence marque la courbe finie
        return t

    t = la_courbe("courbe_test", None)
    r = os6.resume_courbe(t, seuil, critere)
    r.to_csv(sortie / "courbe_test.csv", index=False)
    n_rec = os6.nombre_recommande(r)
    r_deg = None
    if args.degradation:                       # information, sans effet sur le nombre recommande
        t_deg = la_courbe("courbe_test_era5_degradee", args.degradation)
        r_deg = os6.resume_courbe(t_deg, seuil, critere)
        r_deg.to_csv(sortie / "courbe_test_era5_degradee.csv", index=False)

    # 4. plan (S18)
    p = os6.plan(ordre, g, georef, hauteur_mat_m=args.hauteur_capteur)
    p.to_csv(sortie / "plan_aneo.csv", index=False)
    meta = {"produit_par": "pcfno3d os6", "modele": wf.checkpoint_path.name,
            "configuration": cfg.name, "seuil_rmse_m_s": seuil,
            "nombre_recommande": n_rec, "contraintes": os6.contraintes_dict(c)}
    if regle is not None or sol or capteurs is not None:
        meta.update({
            "regle": Path(args.regle).name if args.regle else None, "regle_sha256": sha_regle,
            "statut_regle": regle.get("statut") if regle else None,
            "modele_run_dir": str(cfg.paths.run_dir),
            "metrique_seuil": (f"{critere} : RMSE u, v maximal sur les tranches 0-100, 100-300, 300-600, "
                               "600-1000 m au-dessus du sol (scripts/dso_hauteur_sol.py)") if sol else critere,
            "capteur_simule": None if capteurs is None else {
                "hauteur_m": capteurs.hauteur_m, "bruit": "0,1 m/s + 2 % de la vitesse, par composante",
                "valeur": "verite d'entrainement a la hauteur du mat (aneo_entree.EntreeANEO.valeur_a_hauteur)",
                "graine": capteurs.graine},
            "georeferencement": args.georef, "condition_principale": "propre (entree de production)",
            "zones_exclues": None if not args.zones_exclues else {
                "fichier": Path(args.zones_exclues).name, "sha256": sha_zones}})
        if args.modele_info:
            meta["choix_modele"] = json.loads(Path(args.modele_info).read_text(encoding="utf-8"))
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
    if sol:
        resume["reseau_actuel"].update({k: float(actuel[k]) for k in ("rmse_vect_sol", "rmse_max_uv_sol")})
        resume["comparaison_net1_sol"] = {
            int(n): {s: float(r[(r["strategie"] == s) & (r["n_capteurs"] == n)]["rmse_vect_sol"].iloc[0])
                     for s in ("optimise", "grille", "hasard")}
            for n in sorted(r[r["strategie"] == "optimise"]["n_capteurs"])}
    if capteurs is not None:
        resume["limites"] = (
            "OSSE sur la verite d'entrainement generee (relief reel, forcage ERA5) ; capteurs simules = "
            "verite d'entrainement a la hauteur du mat + bruit de mesure, comme a l'entrainement du "
            "modele : ce ne sont pas des mesures. Le gain est celui de ce modele. Plan provisoire "
            "jusqu'a la validation sur les premiers capteurs (BF78).")
    if r_deg is not None:
        resume["information_era5_degradee"] = {
            "note": "information, sans effet sur le nombre recommande : ERA5 degradee par le champ correle "
                    "du candidat D (sans zones), memes dispositions, memes instants de test",
            "nombre_qui_respecterait_le_seuil": os6.nombre_recommande(r_deg),
            "courbe": r_deg.to_dict(orient="records")}
    (sortie / "os6.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    figures(sortie, r, p, cand, g, seuil, critere, r_deg)
    print(r.round(3).to_string(index=False))
    quoi = "RMSE u, v max 0-1 km au-dessus du sol" if sol else "pire RMSE de niveau"
    print(f"nombre recommande ({quoi} <= {seuil} m/s) : "
          f"{n_rec if n_rec is not None else 'non atteint dans la courbe'}")
    print(f"plan -> {sortie}")
    return 0


def figures(sortie: Path, r, p, cand, g, seuil: float, critere: str = "pire_rmse_niveau",
            r_deg=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    national = critere != "pire_rmse_niveau"
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for s, st in (("optimise", "o-"), ("grille", "s--"), ("hasard", "^:")):
        q = r[r["strategie"] == s]
        ax.plot(q["n_capteurs"], q[critere], st, label=s)
    if r_deg is not None:
        q = r_deg[r_deg["strategie"] == "optimise"]
        ax.plot(q["n_capteurs"], q[critere], "o-", color="grey", alpha=0.6,
                label="optimise, ERA5 degradee (information)")
    a = r[r["strategie"] == "actuel"][critere].iloc[0]
    ax.axhline(a, color="grey", lw=1,
               label="reseau actuel (ERA5 + 12 aeroports)" if national else "reseau actuel (ERA5 + 3 stations)")
    ax.axhline(seuil, color="red", lw=1, ls="--", label=f"seuil {seuil} m/s")
    ax.set_xlabel("capteurs ANEO ajoutes")
    ax.set_ylabel("RMSE u ou v max, 0-1 km au-dessus du sol (m/s), test" if national
                  else "pire RMSE de niveau, u ou v (m/s), test")
    ax.legend(fontsize=8)
    fig.savefig(sortie / "courbe.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 11) if g.ny > 2 * g.nx else (8, 6.6), constrained_layout=True)
    ext = (0, g.nx * g.dx / 1000, 0, g.ny * g.dy / 1000)
    im = ax.imshow(np.where(g.z0 >= 0.005, g.zt, np.nan), origin="lower", cmap="terrain",
                   extent=ext)
    fig.colorbar(im, ax=ax, shrink=0.7, label="altitude (m)")
    if len(cand) > 20000:                      # national : les candidats en voile, pas en points
        from pcfno3d.os6 import masque
        voile = np.where(masque(g, cand), 1.0, np.nan)
        ax.imshow(voile, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=4, alpha=0.35)
    else:
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
