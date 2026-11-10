"""Phase 6 de la feuille de route nationale : agriculture nationale (cahier v12, OS8), selon
configs/national/regle_agri_national.json (PROPOSITION, empreinte verifiee avant toute lecture).

Deux etapes, chacune refuse de recalculer une sortie existante :

  decisions   AGRI1 / AGRI2 aux 12 aeroports, test 2022 : post-traitement des paires « station
              retiree de l'entree » deja ecrites par scripts/valider_grand_tunis_reel.py
              (<run>/validation_reelle/test/stations_paires.csv) ; regle de traitement et
              d'aspersion FIGEE (celle du Grand Tunis, scripts/verifier_decisions.py) ; aucune
              nouvelle passe du modele.
  produit     fenetres de traitement et d'aspersion J0, J+1, J+2 et besoin en eau FAO-56 sur la
              grille nationale (1 km), vent a 2 et 10 m au-dessus du sol (pcfno3d.agriculture),
              pour les jours de base de la regle (validation 2021) ; T, HR et rayonnement de GFS
              (cube national pcfno3d.gfs) si fourni, sinon repli signale.

    python scripts/agri_national.py decisions --run <run_dir> --sortie <dossier>
    python scripts/agri_national.py produit -c <config> --sortie <dossier> [--gfs gfs_tunisie_f24.npz]

Sorties (decisions) : decisions_agricoles.csv / .json (format de verifier_decisions, lu par
scripts/bilan_conformite.py), verdict_agri_national.json et .md.
Sorties (produit) : agri_<D>/ (agri.npz, *.tif, carte.png, aeroports.csv), produit_agricole.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

REGLE = REPO / "configs" / "national" / "regle_agri_national.json"
SEUIL_AGRI2 = 0.05
ACCORD_GRAND_TUNIS = 0.627          # AGRI1 du Grand Tunis (OS7, 07/11/2026), pour reference seulement


# ---------------------------------------------------------------------- AGRI1 / AGRI2
def verdicts(t: pd.DataFrame) -> dict:
    """Verdicts de la regle a partir de la table de verifier_decisions.evaluer."""
    g = t[(t.regle == "traitement") & (t.periode == "toutes heures")]
    tout = g[g.station == "toutes"].set_index("source")
    if "modele" not in tout.index or "era5" not in tout.index:
        raise SystemExit("paires sans colonnes u_modele / u_era5 : AGRI1 impossible")
    m, e = tout.loc["modele"], tout.loc["era5"]
    par = g[g.station != "toutes"].pivot_table(index="station", columns="source", values="accord")
    stations_ok = int((par["modele"] >= par["era5"]).sum()) if {"modele", "era5"} <= set(par.columns) else None
    return {
        "agri1": {"accord_modele": float(m.accord), "accord_era5": float(e.accord),
                  "kappa_modele": float(m.kappa), "kappa_era5": float(e.kappa), "paires": int(m.n),
                  "conforme": bool(m.accord >= e.accord),
                  "stations_modele_au_moins_era5": stations_ok, "stations": int(len(par)),
                  "reference_grand_tunis": ACCORD_GRAND_TUNIS,
                  "critere": "accord du modele >= accord de l'ERA5 10 m sur les memes paires (PROPOSITION)"},
        "agri2": {"faux_oui_modele": float(m.faux_oui), "faux_oui_era5": float(e.faux_oui),
                  "faux_oui_parmi_non_modele": float(m.faux_oui_parmi_non), "seuil": SEUIL_AGRI2,
                  "conforme": bool(m.faux_oui <= SEUIL_AGRI2),
                  "critere": "faux oui du modele <= 0,05 (cahier v12, comme le Grand Tunis)"},
    }


def etape_decisions(args) -> int:
    from pcfno3d.regles import sortie_neuve, verifier

    regle, h = verifier(REGLE)
    paires = Path(args.paires) if args.paires else Path(args.run) / "validation_reelle" / "test" / "stations_paires.csv"
    if not paires.exists():
        raise SystemExit(f"{paires} absent : lancer d'abord scripts/valider_grand_tunis_reel.py --jeu test (une fois)")
    sortie = sortie_neuve(args.sortie, "verdict_agri_national.json")
    import verifier_decisions as vd
    p = pd.read_csv(paires)
    t = vd.evaluer(p, tuple(regle["agri1_agri2"]["sources"]))
    t.to_csv(sortie / "decisions_agricoles.csv", index=False)
    v = verdicts(t)
    g = t[(t.station == "toutes") & (t.regle == "traitement")]
    # meme format que scripts/verifier_decisions.py : lu tel quel par scripts/bilan_conformite.py
    from pcfno3d.agriculture import TEXTE_REGLE
    resume = {"regle": TEXTE_REGLE, "donnees": [str(paires)],
              "agri1": {f"{r.source} / {r.periode}": {"accord": round(r.accord, 4), "kappa": round(r.kappa, 4), "n": r.n}
                        for r in g.itertuples()},
              "agri2": {f"{r.source} / {r.periode}": {"faux_oui": round(r.faux_oui, 4),
                                                      "conforme": bool(r.faux_oui <= SEUIL_AGRI2)}
                        for r in g.itertuples()},
              "seuil_agri2": SEUIL_AGRI2,
              "note": "national, 12 aeroports METAR, station retiree de l'entree ; regle "
                      "configs/national/regle_agri_national.json (PROPOSITION)"}
    (sortie / "decisions_agricoles.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
    out = {"regle": str(REGLE.relative_to(REPO)), "sha256_regle": h, "statut_regle": "PROPOSITION",
           "run": str(args.run) if args.run else None, "paires": str(paires),
           "stations": sorted(p["station"].unique().tolist()), "instants": int(p["temps"].nunique()), **v,
           "mesures": "METAR des 12 aeroports ; ERA5 et la verite generee (« generateur ») sont la verite "
                      "d'entrainement, pas des mesures"}
    (sortie / "verdict_agri_national.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                                                       encoding="utf-8")
    par = t[(t.regle == "traitement") & (t.periode == "toutes heures") & (t.station != "toutes")]
    tab = par.pivot_table(index="station", columns="source", values=["accord", "faux_oui"]).round(3)
    try:
        tab_txt = tab.to_markdown()
    except ImportError:                                     # tabulate absent : table brute
        tab_txt = "```\n" + tab.to_string() + "\n```"
    L = [f"# AGRI1 / AGRI2 nationaux (PROPOSITION, regle sha256 {h[:12]}…)", "",
         f"{out['instants']} instants du test 2022, {len(out['stations'])} aeroports, chaque station retiree de "
         "l'entree quand elle est jugee. Regle de traitement figee (km/h a 10 m).", "",
         "| Critere | Modele | ERA5 10 m | Seuil | Verdict |", "|---|---|---|---|---|",
         f"| AGRI1 accord (traitement, toutes heures) | {v['agri1']['accord_modele']:.3f} | "
         f"{v['agri1']['accord_era5']:.3f} | >= ERA5 | {'conforme' if v['agri1']['conforme'] else 'non conforme'} |",
         f"| AGRI2 faux oui | {v['agri2']['faux_oui_modele']:.4f} | {v['agri2']['faux_oui_era5']:.4f} | "
         f"<= {SEUIL_AGRI2} | {'conforme' if v['agri2']['conforme'] else 'non conforme'} |", "",
         f"Kappa de Cohen : modele {v['agri1']['kappa_modele']:.3f}, ERA5 {v['agri1']['kappa_era5']:.3f}. "
         f"Aeroports ou le modele fait au moins aussi bien qu'ERA5 : {v['agri1']['stations_modele_au_moins_era5']} "
         f"sur {v['agri1']['stations']}. Grand Tunis (reference) : {ACCORD_GRAND_TUNIS:.1%}.", "",
         "Par aeroport :", "", tab_txt, "",
         "ERA5 et la verite generee sont la verite d'entrainement, pas des mesures."]
    (sortie / "verdict_agri_national.md").write_text("\n".join(L), encoding="utf-8")
    print(json.dumps(v, indent=1, ensure_ascii=False))
    print(f"-> {sortie}")
    return 0


# ---------------------------------------------------------------------- produit J0-J+2
def jours_de_base(index: pd.DataFrame, n: int = 3) -> list[str]:
    """Regle : les n jours locaux de la VALIDATION dont le plus de jours D, D+1, D+2 sont dans le
    cache (a egalite, le plus ancien)."""
    from pcfno3d.atom_wind import jours
    presents = {j["date"] for j in jours(index, split=None)}
    val = [j["date"] for j in jours(index, split="val")]
    def score(d):
        t = pd.Timestamp(d)
        return sum((t + pd.Timedelta(days=k)).strftime("%Y-%m-%d") in presents for k in range(3))
    return sorted(sorted(val), key=lambda d: (-score(d), d))[:n]


def _figure(jours_prod: list[dict], geom, chemin: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pcfno3d.agriculture import latlon_noeuds
    lat, lon = latlon_noeuds(geom)
    ext = (lon.min(), lon.max(), lat.min(), lat.max())
    fig, axes = plt.subplots(2, len(jours_prod), figsize=(4.2 * len(jours_prod), 9), squeeze=False,
                             constrained_layout=True)
    for c, j in enumerate(jours_prod):
        if j.get("absent"):
            for r in (0, 1):
                axes[r, c].set_title(f"{j['libelle']} {j['date']} : absent du cache", fontsize=9)
                axes[r, c].axis("off")
            continue
        im = axes[0, c].imshow(j["heures_oui_traitement"], origin="lower", extent=ext, cmap="Greens", vmin=0, vmax=24)
        axes[0, c].set_title(f"{j['libelle']} {j['date']} : heures « oui » traitement", fontsize=9)
        fig.colorbar(im, ax=axes[0, c], shrink=0.7)
        if j.get("et0_mm") is not None:
            im = axes[1, c].imshow(j["et0_mm"], origin="lower", extent=ext, cmap="YlOrRd")
            axes[1, c].set_title(f"ET0 FAO-56 (mm/j){' — Rs repli 0,72 Ra' if j['rayonnement_repli'] else ''}",
                                 fontsize=9)
            fig.colorbar(im, ax=axes[1, c], shrink=0.7)
        else:
            axes[1, c].set_title("ET0 : sans temperature / humidite", fontsize=9)
            axes[1, c].axis("off")
    fig.suptitle("Agriculture nationale — rejeu sur analyse (forcage ERA5), pas une prevision", fontsize=10)
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


def etape_produit(args) -> int:
    from pcfno3d import agriculture as ag
    from pcfno3d.atom_wind import heure_locale, jours
    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.regles import sortie_neuve, verifier
    from pcfno3d.sig import ecrire_geotiff, grille_geo
    from pcfno3d.stations_reelles import cellules_codes, noms_stations

    regle, h = verifier(REGLE)
    sortie = sortie_neuve(args.sortie, "produit_agricole.json")
    cfg = Config.load(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg.apply_overrides({k: v})
    wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device, verbose=False)
    geom = wf.geom
    index = load_index(cfg)
    par_jour = {j["date"]: j for j in jours(index, split=None)}
    bases = args.jours.split(",") if args.jours else jours_de_base(index, args.n_jours)
    meteo = ag.MeteoGFS(Path(args.gfs), geom) if args.gfs else None
    g = grille_geo(geom)
    cell = cellules_codes(geom)
    noms = noms_stations(geom)
    resume = {"regle": str(REGLE.relative_to(REPO)), "sha256_regle": h, "statut_regle": "PROPOSITION",
              "run": str(cfg.paths.run_dir), "checkpoint": wf.checkpoint_path.name,
              "entree": ("METAR reels aux aeroports (paths.stations_obs)" if cfg.paths.stations_obs
                         else "valeur du cache a la cellule des aeroports (pas de dossier obs)"),
              "source_du_vent": "rejeu : modele national sur les instants du cache (forcage de la configuration ; "
                                "ERA5 = analyse, pas une prevision)",
              "regle_de_decision": ag.TEXTE_REGLE, "jours_de_base": [], "gfs": str(args.gfs) if args.gfs else None,
              "choix_des_jours": ("impose par --jours (hors regle, dit ici)" if args.jours
                                  else "regle : validation, le plus de jours D..D+2 dans le cache")}
    for base in bases:
        d0 = pd.Timestamp(base)
        dossier = sortie / f"agri_{base}"
        dossier.mkdir(parents=True, exist_ok=True)
        prod, lignes, entree_jour = [], [], {"base": base, "jours": []}
        for k, lib in enumerate(("J0", "J+1", "J+2")):
            date = (d0 + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
            j = par_jour.get(date)
            if j is None:
                prod.append({"libelle": lib, "date": date, "absent": True})
                entree_jour["jours"].append({"libelle": lib, "date": date, "absent": True})
                continue
            instants, src_meteo = [], set()
            for it in j["instants"]:
                S, hour, doy, stamp = wf.surface_from_cache(it["stem"])
                vol = wf.predict(S, hour, doy, timestamp=stamp, reliability=False)
                vent = ag.vent_pres_du_sol(vol, geom)
                rec = {"heure_locale": heure_locale(stamp).hour + heure_locale(stamp).minute / 60,
                       "u10": vent[10.0][0], "v10": vent[10.0][1], "u2": vent[2.0][0], "v2": vent[2.0][1]}
                m = meteo.a(int(pd.Timestamp(stamp).value // 10 ** 9)) if meteo is not None else None
                if m and "t2m" in m and "hr2m" in m:
                    rec.update(t2m=m["t2m"], hr2m=m["hr2m"], ray=m.get("ray"))
                    src_meteo.add("GFS")
                else:
                    # repli signale : couche de surface du cache (verite d'entrainement issue d'ERA5)
                    rec.update(t2m=np.asarray(S[3], float), hr2m=np.clip(np.asarray(S[5], float), 1, 100), ray=None)
                    src_meteo.add("verite d'entrainement (ERA5)")
                instants.append(rec)
            jj = ag.journee(instants, geom, date)
            jj.update(libelle=lib)
            prod.append(jj)
            np.savez_compressed(dossier / f"agri_{lib.replace('+', 'p')}_{date}.npz",
                                traitement=jj["traitement"], aspersion=jj["aspersion"],
                                vitesse_10m=jj["vitesse_10m"], vitesse_2m=jj["vitesse_2m"],
                                et0_mm=jj["et0_mm"] if jj["et0_mm"] is not None else np.array([]))
            for cle in ("heures_oui_traitement", "heures_oui_aspersion", "et0_mm", "vent_moyen_10m", "vent_moyen_2m"):
                if jj.get(cle) is not None:
                    ecrire_geotiff(dossier / f"{cle}_{lib.replace('+', 'p')}_{date}.tif", jj[cle],
                                   g["lon0"], g["lat0"], g["dlon"], g["dlat"])
            for code, (iy, ix) in sorted(cell.items()):
                for regle_nom in ("traitement", "aspersion"):
                    cp = ag.creneaux_point(jj, iy, ix, regle_nom)
                    lignes.append({"jour": lib, "date": date, "code": code, "station": noms.get(code, code),
                                   "regle": regle_nom, "creneaux_oui": cp["creneaux"],
                                   "heures_oui": sum(v == "oui" for v in cp["heures"].values()),
                                   "heures_avec_valeur": len(cp["heures"]),
                                   "vent_moyen_10m_m_s": round(float(jj["vent_moyen_10m"][iy, ix]), 2),
                                   "et0_mm": (round(float(jj["et0_mm"][iy, ix]), 2) if jj["et0_mm"] is not None
                                              and np.isfinite(jj["et0_mm"][iy, ix]) else None)})
            entree_jour["jours"].append({
                "libelle": lib, "date": date, "instants_modele": jj["instants_modele"],
                "meteo": sorted(src_meteo), "rayonnement_repli": jj["rayonnement_repli"],
                "heures_oui_traitement_moyenne_terre": round(float(np.nanmean(
                    np.where(geom.z0 >= 0.005, jj["heures_oui_traitement"], np.nan))), 2),
                "et0_moyen_terre_mm": (round(float(np.nanmean(np.where(geom.z0 >= 0.005, jj["et0_mm"], np.nan))), 2)
                                       if jj["et0_mm"] is not None else None)})
        pd.DataFrame(lignes).to_csv(dossier / "aeroports.csv", index=False)
        _figure(prod, geom, dossier / "carte.png")
        resume["jours_de_base"].append(entree_jour)
        print(json.dumps(entree_jour, ensure_ascii=False), flush=True)
    resume["limites"] = ("4 sorties du modele par jour dans le cache (heures interpolees lineairement entre elles, "
                         "sans extrapolation) ; analyse et non prevision tant que la phase 4 (GFS national) n'est pas "
                         "faite ; T, HR, rayonnement : GFS si le cube national porte l'instant, sinon repli signale")
    (sortie / "produit_agricole.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False, default=str),
                                                  encoding="utf-8")
    print(f"-> {sortie}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="etape", required=True)
    a = sp.add_parser("decisions", help="AGRI1 / AGRI2 (paires station exclue du test 2022)")
    a.add_argument("--run", default=None, help="run_dir du modele (validation_reelle/test/stations_paires.csv)")
    a.add_argument("--paires", default=None)
    a.add_argument("--sortie", required=True)
    b = sp.add_parser("produit", help="fenetres J0-J+2 et FAO-56 sur la grille nationale")
    b.add_argument("-c", "--config", required=True)
    b.add_argument("--set", action="append", default=[])
    b.add_argument("--checkpoint")
    b.add_argument("--device")
    b.add_argument("--gfs", default=None, help="cube GFS national (pcfno3d.gfs, gfs_tunisie_f<L>.npz)")
    b.add_argument("--jours", default=None, help="jours de base AAAA-MM-JJ (defaut : regle, validation)")
    b.add_argument("--n-jours", type=int, default=3)
    b.add_argument("--sortie", required=True)
    args = ap.parse_args(argv)
    return etape_decisions(args) if args.etape == "decisions" else etape_produit(args)


if __name__ == "__main__":
    raise SystemExit(main())
