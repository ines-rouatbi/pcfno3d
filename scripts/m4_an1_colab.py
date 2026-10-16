"""Lot M4 — mesure du critere AN1 par simulation de pannes et de derives (session GPU), v2.

A lancer dans une session Colab, depuis le depot installe, avec le cache et le checkpoint :

    !python scripts/m4_an1_colab.py -c colab --deuxieme-lecture

Protocole du verdict (an1.PROTOCOLE, inchange depuis la v1) : 40 derives et 10 pannes par
jour, e_ref = 1 m/s, memes graines. La part de fausses alertes depend de la prevalence des
defauts : avec d'autres reglages (--derives-par-jour, --pannes-par-jour, --e-ref), le
resultat est marque « exploratoire » et AUCUN verdict officiel n'est produit
(an1_conforme = null).

0. ENTRAINEMENT (jours d'entrainement, instants synoptiques) : modeles de sites du
   detecteur spatial, un par rayon de la grille -> sites_R<r>.npz (jeu: entrainement).
1. VALIDATION : injection de derives (biais, gain, girouette) et de pannes (valeurs
   figees), puis TOUS les detecteurs de la grille fixee d'avance (an1.GRILLE_DETECTEURS :
   spatiale avec / sans correction de site, rayon, echelle, fenetre ; innovation,
   influence, combinee avec les memes passes). Pour chacun, le plus petit kappa dont la
   part de fausses alertes reste sous --fdr-cible ; on retient la meilleure detection.
   Regle des entrees (an1.GRILLE_ENTREES, vent exclu) choisie de meme. ->
   seuils_validation.json (jeu: validation, methode alertes-m4-v2)
2. TEST (autre graine) avec ces seuils FIGES : detection des derives, fausses alertes, a
   cote sains signales / sains, fausses alertes par instant, pannes (hors verdict), entrees
   (et, pour le diagnostic seulement, la regle des entrees de la v1). Rien n'y est ajuste.
   -> an1_test.json
3. Distributions sur la validation du residu de masse et de la condition au sol par instant.

--deuxieme-lecture : le test a deja ete lu une fois (v1, 16/10/2026, AN1 non conforme) et la
v2 a ete concue APRES cette lecture. Le verdict porte alors « seconde mesure sur le test,
apres une premiere lecture (16/10/2026, non conforme) » : ce n'est pas une mesure
independante. Sans l'option, an1_test.json porte « lecture : non declaree ».

Tout s'ecrit dans un dossier NEUF, run_dir/alertes_pcfno3d/<modele>/session_v2_<horodatage>/ :
le script n'ouvre en ecriture aucun fichier existant. Pour servir les seuils retenus :

    alerts:
      enabled: true
      seuils: <run_dir>/alertes_pcfno3d/<modele>/session_v2_.../seuils_validation.json

(le modele de sites retenu, s'il y en a un, est relu a cote du JSON).

Codes de retour : 0 campagne terminee (le verdict AN1 est dans an1_test.json et imprime, il
ne change pas le code) ; 1 erreur (pas de jour de validation ou de test en cache...).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]

PREMIERE_LECTURE = {"date": "2026-10-16", "version": "v1 (innovation, kappa 12,75)",
                    "an1_conforme": False}
NOTE_DEUXIEME = ("seconde mesure sur le test, apres une premiere lecture (16/10/2026, non "
                 "conforme) ; la v2 a ete concue apres cette lecture : ce n'est pas une "
                 "mesure independante")


def _jours(index, split: str, limite: int | None) -> list[list[str]]:
    """Instants synoptiques de chaque jour du jeu, dans l'ordre chronologique."""
    from pcfno3d.dataset import is_synoptic

    rows = index[index["split"] == split].sort_values("timestamp")
    out = []
    for _, d in rows.groupby("day", sort=True):
        stems = [s for s, t in zip(d["stem"], d["timestamp"]) if is_synoptic(t)]
        if stems:
            out.append(stems)
    return out[:limite] if limite else out


def _ecrire(path: Path, data: dict) -> None:
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False, default=float) + "\n")
                     .encode("utf-8"))


def _quantiles(x: list[float]) -> dict:
    import numpy as np
    a = np.asarray([v for v in x if v is not None], float)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "p50": float(np.quantile(a, 0.5)),
            "p99": float(np.quantile(a, 0.99)), "max": float(a.max())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="colab")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--jours-entrainement", type=int,
                    help="jours d'entrainement pour les modeles de sites (defaut : tous)")
    ap.add_argument("--jours-validation", type=int, help="defaut : tous les jours")
    ap.add_argument("--jours-test", type=int, help="defaut : tous les jours")
    ap.add_argument("--derives-par-jour", type=int, default=None,
                    help="defaut : protocole (40) ; autre valeur = exploratoire")
    ap.add_argument("--pannes-par-jour", type=int, default=None,
                    help="valeurs figees par jour ; defaut : protocole (10)")
    ap.add_argument("--e-ref", type=float, default=None,
                    help="amplitude effective (m/s) a partir de laquelle une derive compte "
                         "dans la detection (defaut : protocole, 1.0, le seuil du contrat)")
    ap.add_argument("--fdr-cible", type=float, default=0.025,
                    help="part de fausses alertes visee SUR LA VALIDATION (marge sous 5 %%)")
    ap.add_argument("--statistiques", default="spatiale,innovation,influence,combinee",
                    help="detecteurs candidats (defaut : les quatre ; « spatiale » seule "
                         "n'a besoin d'aucune passe sans capteur)")
    ap.add_argument("--deuxieme-lecture", action="store_true",
                    help="declare que le test a deja ete lu (v1, 16/10/2026, non conforme)")
    ap.add_argument("--graine", type=int, default=4)
    args = ap.parse_args(argv)

    import numpy as np

    from pcfno3d import an1
    from pcfno3d.alerts import STATISTIQUES, ModeleSites
    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.inference import WindFieldModel

    stats = tuple(s.strip() for s in args.statistiques.split(",") if s.strip())
    inconnues = set(stats) - set(STATISTIQUES)
    if inconnues or not stats:
        print(f"ERREUR : statistiques inconnues {sorted(inconnues)} (attendu : "
              f"{STATISTIQUES})", file=sys.stderr)
        return 1
    P = an1.PROTOCOLE
    n_der = P["derives_par_jour"] if args.derives_par_jour is None else args.derives_par_jour
    n_pan = P["pannes_par_jour"] if args.pannes_par_jour is None else args.pannes_par_jour
    e_ref = P["e_ref_m_s"] if args.e_ref is None else args.e_ref
    officiel = (n_der, n_pan, e_ref) == (P["derives_par_jour"], P["pannes_par_jour"],
                                         P["e_ref_m_s"])
    statut = "officiel" if officiel else "exploratoire"
    if not officiel:
        print(f"EXPLORATOIRE : {n_der} derives, {n_pan} pannes par jour, e_ref {e_ref} "
              f"(protocole : {P}) -> pas de verdict AN1 officiel")
    lecture = ({"lecture": "deuxieme", "note": NOTE_DEUXIEME,
                "premiere_lecture": PREMIERE_LECTURE} if args.deuxieme_lecture else
               {"lecture": "non declaree",
                "note": "le test a deja ete lu le 16/10/2026 (v1) : relancer avec "
                        "--deuxieme-lecture pour le declarer"})
    if not args.deuxieme_lecture:
        print("ATTENTION : test deja lu le 16/10/2026 (v1) ; --deuxieme-lecture absent")

    cfg = Config.load(args.config)
    cfg.inference.reliability = None          # passes simples : la fiabilite n'y joue pas
    wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device)
    index = load_index(cfg)
    val, test = _jours(index, "val", args.jours_validation), _jours(index, "test",
                                                                     args.jours_test)
    train = _jours(index, "train", args.jours_entrainement)
    if not val or not test:
        print("ERREUR : il faut des jours de validation ET de test dans le cache",
              file=sys.stderr)
        return 1
    dets = an1.detecteurs(stats)
    if any(d.sites for d in dets) and not train:
        print("ERREUR : correction de site sans jour d'entrainement dans le cache",
              file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (cfg.paths.run_dir / "alertes_pcfno3d" / wf.checkpoint_path.stem
           / f"session_v2_{stamp}")
    out.mkdir(parents=True, exist_ok=False)   # dossier neuf, ou erreur : rien n'est ecrase
    print(f"sortie -> {out}\n")
    a = cfg.alerts
    t0 = time.perf_counter()

    # 0. ENTRAINEMENT : modeles de sites (jamais la validation ni le test)
    rayons = sorted({d.rayon for d in dets if d.sites})
    sites = {}
    if rayons:
        stems_train = [s for j in train for s in j]
        sites = an1.ajuster_sites_entrainement(wf, stems_train, rayons, a.derive_min_voisins)
        for r, m in sites.items():
            m.sauver(out / an1.fichier_sites(r))
        print(f"sites : {len(stems_train)} instants d'entrainement, rayons {rayons} "
              f"({time.perf_counter() - t0:.0f} s)")

    # 1. VALIDATION : choix du detecteur, de kappa et de la regle des entrees
    rng = np.random.default_rng(args.graine)
    sim_val = [an1.simuler_jour(wf, j, a, rng, n_der, n_pan, dets, sites=sites)
               for j in val]
    choix = an1.choisir_detecteur(sim_val, fdr_cible=args.fdr_cible, e_ref=e_ref)
    for cle, c in choix["alternatives"].items():
        print(f"validation {cle:34s} kappa {c['kappa']:5.2f} | detection "
              f"{c['detection']} | fausses alertes {c['fausses_alertes']:.3f}"
              f"{'' if c['cible_atteinte'] else ' (cible non atteinte)'}")
    ent = an1.choisir_regle_entrees(sim_val, fdr_cible=args.fdr_cible)
    v1 = ent["regle_v1_validation"]
    print(f"entrees : regle v1 sur la validation -> fausses {v1['fausses_alertes']} "
          f"({v1['fausses_par_variable']}) ; retenue z_max {ent['entrees_z_max']}, "
          f"part_extreme {ent['entrees_part_extreme']} -> detection "
          f"{ent['validation']['detection']}, fausses {ent['validation']['fausses_alertes']}")
    seuils = {**{k: v for k, v in choix.items() if k != "alternatives"},
              "derive_min_m_s": a.derive_min_m_s,
              "derive_min_voisins": a.derive_min_voisins,
              "entrees_z_max": ent["entrees_z_max"],
              "entrees_part_extreme": ent["entrees_part_extreme"],
              "entrees_ecart_extreme": a.entrees_ecart_extreme,
              "entrees_variables": ent["entrees_variables"],
              "entrees_validation": {k: v for k, v in ent.items() if k != "jeu"},
              "modele": wf.checkpoint_path.name,
              "jours_entrainement_sites": [j[0].split("_")[0] for j in train] if rayons else [],
              "jours_validation": [j[0].split("_")[0] for j in val],
              "derives_par_jour": n_der, "pannes_par_jour": n_pan, "statut": statut,
              "graine": args.graine, "alternatives": choix["alternatives"]}
    _ecrire(out / "seuils_validation.json", seuils)
    print(f"  ({time.perf_counter() - t0:.0f} s) -> retenu : {choix['detecteur']}, "
          f"kappa {choix['derive_kappa']}\n")

    # 2. TEST : verdict avec les seuils figes
    t1 = time.perf_counter()
    rng = np.random.default_rng(args.graine + 1000)
    det = next(d for d in dets if d.cle == choix["detecteur"])
    sites_test = {}
    if det.sites:                             # relu depuis le fichier publie, comme le service
        sites_test = {det.rayon: ModeleSites.charger(out / an1.fichier_sites(det.rayon))}
    sim_test = [an1.simuler_jour(wf, j, a, rng, n_der, n_pan, [det], sites=sites_test,
                                 par_instant=True) for j in test]
    capteurs = an1.evaluer_an1(sim_test, det.cle, choix["derive_kappa"], e_ref)
    regle = {"z_max": ent["entrees_z_max"], "part_extreme": ent["entrees_part_extreme"],
             "variables": ent["entrees_variables"]}
    entrees = an1.evaluer_entrees(sim_test, regle)
    entrees_v1 = an1.evaluer_entrees(sim_test, an1.REGLE_ENTREES_V1)   # diagnostic seul
    verdict = (bool(capteurs["an1_capteurs_conforme"] and entrees["an1_entrees_conforme"])
               if officiel else None)
    an1_test = {"jeu": "test", "statut": statut, "methode": seuils["methode"], **lecture,
                "protocole": P,
                "reglages": {"derives_par_jour": n_der, "pannes_par_jour": n_pan,
                             "e_ref_m_s": e_ref},
                "seuils_validation": str(out / "seuils_validation.json"),
                "detecteur": det.cle, "kappa": choix["derive_kappa"],
                "jours_test": [j[0].split("_")[0] for j in test], "capteurs": capteurs,
                "entrees": entrees, "entrees_regle_v1_diagnostic": entrees_v1,
                "an1_conforme": verdict,
                "criteres": {"detection_min": an1.DETECTION_MIN,
                             "fausses_alertes_max": an1.FAUSSES_MAX}}
    _ecrire(out / "an1_test.json", an1_test)
    print(f"test : detection des derives {capteurs['detection']} | fausses alertes "
          f"{capteurs['fausses_alertes']:.3f} (sains signales / sains "
          f"{capteurs['taux_faux_positifs']}, par instant "
          f"{capteurs['fausses_alertes_par_instant']}) | par type "
          f"{capteurs['detection_par_type']} | pannes hors verdict "
          f"{capteurs['pannes_hors_verdict']['detection']} | entrees : detection "
          f"{entrees['detection']}, fausses {entrees['fausses_alertes']} (regle v1 : "
          f"{entrees_v1['fausses_alertes']}, {entrees_v1['fausses_par_variable']})  "
          f"({time.perf_counter() - t1:.0f} s)")

    # 3. physique sur la validation (instants avec derives injectees : effet local)
    phys = [p for j in sim_val for p in j["physique"]]
    resume = {"modele": wf.checkpoint_path.name, "dossier": str(out),
              "seuils": {k: seuils[k] for k in ("detecteur", "derive_statistique",
                                                "derive_kappa", "fdr_cible", "e_ref_m_s",
                                                "cible_atteinte", "entrees_z_max",
                                                "entrees_part_extreme")},
              "validation": choix["validation"], "test": capteurs, "entrees_test": entrees,
              "an1_conforme": verdict, "statut": statut, **lecture,
              "physique_validation": {
                  "mass_residual_rms": _quantiles([p["mass_residual_rms"] for p in phys]),
                  "ground_condition_rms_ms": _quantiles([p["ground_condition_rms_ms"]
                                                         for p in phys]),
                  "part_mass_residual_ok": float(np.mean([p["mass_residual_ok"]
                                                          for p in phys])),
                  "rho_source": a.rho_source},
              "duree_s": round(time.perf_counter() - t0, 1)}
    _ecrire(out / "resume.json", resume)
    mot = ("EXPLORATOIRE (pas de verdict)" if verdict is None
           else "CONFORME" if verdict else "NON CONFORME")
    if verdict is not None and args.deuxieme_lecture:
        mot += " (seconde mesure sur le test, apres une premiere lecture non conforme)"
    print(f"\nAN1 -> {mot} | resume -> {out / 'resume.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
