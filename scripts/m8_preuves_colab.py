"""Lot M8 — preuves (session GPU) : methodes de reference (L6), E1, E2, E3 (L9) et E5.

A lancer dans une session Colab, depuis le depot installe, avec le cache et le checkpoint :

    !python scripts/m8_preuves_colab.py -c colab

Etapes (`--etapes`, toutes par defaut, dans cet ordre) :

1. ajustement  VALIDATION (selection synoptique fixe, graine 1000) : variogramme du
               krigeage et alpha des lois de puissance. -> ajustement_validation.json
               (ou `--ajustement <json>` pour reprendre un ajustement deja fait)
2. comparaison TEST synoptique (graine 2000, `mask.keep_eval` des capteurs, les 120
               instants) : PC-FNO3D et references sur les memes instants, capteurs et
               noeuds, memes metriques que DSO1. -> comparaison_test.json / .csv
               Controle : sur tout le test, la ligne pcfno3d doit retrouver la table du
               notebook v7 (run_dir/eval_best_dso_v7/), ecart publie dans resume.json.
3. e1          densite du reseau : 5 a 100 % des capteurs, reseaux reguliers 300 m,
               500 m, 1 km. -> e1_densite.json / .csv
4. e2          placement optimal (incertitude de l'ensemble M2 sur la validation, gain
               mesure sur le test contre des placements au hasard). -> e2_placement.json,
               e2_cartes.npz
5. e3          capteur virtuel de secours. -> e3_secours.json / .csv
6. e5          sobriete : CodeCarbon s'il est installe, sinon nvidia-smi, sinon temps
               seul (energie « a mesurer ») ; inference de PC-FNO3D et des references,
               duree d'entrainement v7 reconstruite depuis le journal du notebook.
               -> e5_sobriete.json

Tout s'ecrit dans un dossier NEUF, run_dir/preuves_pcfno3d/<modele>/session_<horodatage>/ :
le script n'ouvre en ecriture aucun fichier existant. Le test ne regle rien : tout parametre
vient de la validation.

Codes de retour : 0 campagne terminee ; 1 erreur (instants absents, ajustement refuse...).
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

ETAPES = ("ajustement", "comparaison", "e1", "e2", "e3", "e5")


def _ecrire_json(path: Path, data) -> None:
    if path.exists():
        raise FileExistsError(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def _ecrire_csv(path: Path, df) -> None:
    if path.exists():
        raise FileExistsError(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        df.to_csv(f, index=False, lineterminator="\n")


def _sans_prive(d: dict) -> dict:
    """Retire les cles internes (prefixe _ : tableaux, DataFrame) avant JSON."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _liste(texte: str | None, typ=float):
    return None if texte is None else tuple(typ(x) for x in texte.split(",") if x.strip())


def controle_v7(cfg, comp: dict) -> dict | None:
    """Ecart max entre la ligne pcfno3d et la table du notebook v7, si elle existe."""
    import pandas as pd

    ref = cfg.paths.run_dir / "eval_best_dso_v7" / "test_par_niveau_synoptique.csv"
    if "pcfno3d" not in comp["methodes"] or not ref.exists():
        return None
    nb = pd.read_csv(ref)
    pk = pd.DataFrame(comp["methodes"]["pcfno3d"]["par_niveau"])
    m = nb.merge(pk, on="z_m", suffixes=("_notebook", "_m8"))
    m = m[m["couverture_m8"] >= cfg.dso.min_level_cover]
    ecart = max(float((m[f"rmse_{c}_notebook"] - m[f"rmse_{c}_m8"]).abs().max()) for c in "uv")
    return {"reference": str(ref), "niveaux": int(len(m)), "ecart_max_rmse_m_s": ecart}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="colab")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--etapes", default=",".join(ETAPES), help=f"parmi {', '.join(ETAPES)}")
    ap.add_argument("--sans-modele", action="store_true",
                    help="references seules (pas de poids) ; E2 est alors saute")
    ap.add_argument("--ajustement", help="ajustement de validation existant (JSON)")
    ap.add_argument("--references", help="noms separes par des virgules (defaut : toutes)")
    ap.add_argument("--instants", type=int, help="instants de test (defaut : tous, 120)")
    ap.add_argument("--instants-validation", type=int,
                    help="instants de validation pour l'ajustement (defaut : la selection)")
    ap.add_argument("--fractions", help="E1, ex. 0.05,0.1,0.25,0.5,0.75,1")
    ap.add_argument("--espacements", help="E1, en m, ex. 300,500,1000 ('' : aucun)")
    ap.add_argument("--references-e1", default="log_idw,puissance_idw",
                    help="E1 : references evaluees a chaque densite (le krigeage coute ~6 s "
                         "par instant sur la grille complete : 9 densites x 120 instants)")
    ap.add_argument("--e2-base-espacement", type=float, default=1000.0)
    ap.add_argument("--e2-etapes", type=int, default=4)
    ap.add_argument("--e2-par-etape", type=int)
    ap.add_argument("--e2-validation", type=int, default=8)
    ap.add_argument("--e2-hasard", type=int, default=3)
    ap.add_argument("--membres", type=int,
                    help="E2 : passes de l'ensemble d'incertitude (defaut reliability.members)")
    ap.add_argument("--e3-pannes", type=int, default=50)
    ap.add_argument("--e3-espacement", type=int, default=3)
    ap.add_argument("--e5-instants", type=int, default=20)
    ap.add_argument("--gpu-watts", type=float, help="E5 : puissance a supposer sans mesure")
    ap.add_argument("--intensite-gco2-kwh", type=float, help="E5 : intensite carbone")
    ap.add_argument("--sans-codecarbon", action="store_true")
    ap.add_argument("--journal-entrainement",
                    default=str(REPO / "notebooks" / "current" / "fno3d_v7.ipynb"),
                    help="E5 : notebook v7 (sorties du BLOC 9) ou history.json")
    args = ap.parse_args(argv)

    etapes = [e.strip() for e in args.etapes.split(",") if e.strip()]
    inconnues = [e for e in etapes if e not in ETAPES]
    if inconnues:
        print(f"ERREUR : etapes inconnues {inconnues}", file=sys.stderr)
        return 1

    import numpy as np
    import scipy
    import torch

    from pcfno3d import __version__
    from pcfno3d.baselines import (ajuster_sur_validation, charger_ajustement,
                                   ecrire_ajustement)
    from pcfno3d.config import Config
    from pcfno3d.preuves import (ESPACEMENTS_E1, FRACTIONS_E1, Banc, capteur_secours, comparer,
                                 etude_densite, placement_optimal, tableau_niveaux)
    from pcfno3d.sobriete import Sobriete, codecarbon_disponible, nvidia_smi_disponible

    cfg = Config.load(args.config)
    if args.membres:
        cfg.apply_overrides({"reliability.members": args.membres})
    cfg.inference.reliability = None          # la valeur servie, sans ensemble de fiabilite
    ev = None
    if not args.sans_modele:
        from pcfno3d.evaluate import Evaluator
        ev = Evaluator(cfg, checkpoint=args.checkpoint, device=args.device)
    nom_modele = ev.wf.checkpoint_path.stem if ev is not None else "references_seules"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.paths.run_dir / "preuves_pcfno3d" / nom_modele / f"session_{stamp}"
    out.mkdir(parents=True, exist_ok=False)    # dossier neuf, ou erreur : rien n'est ecrase
    print(f"sortie -> {out}\n")
    cc = codecarbon_disponible() and not args.sans_codecarbon
    sob = dict(gpu_watts=args.gpu_watts, intensite_gco2_kwh=args.intensite_gco2_kwh,
               utiliser_codecarbon=cc)
    resume: dict = {"modele": nom_modele, "dossier": str(out), "etapes": etapes,
                    "versions": {"pcfno3d": __version__, "torch": torch.__version__,
                                 "numpy": np.__version__, "scipy": scipy.__version__,
                                 "codecarbon": cc, "nvidia_smi": nvidia_smi_disponible()}}
    t0 = time.perf_counter()
    try:
        with Sobriete("session", out / "codecarbon" if cc else None, **sob) as s_session:
            # 1. ajustement (validation)
            ajust = None
            if args.ajustement:
                ajust = charger_ajustement(args.ajustement)
                resume["ajustement"] = {"fichier": args.ajustement, "repris": True}
            elif "ajustement" in etapes:
                print("1. ajustement sur la VALIDATION")
                ajust = ajuster_sur_validation(cfg, ev.geom if ev else _geom(cfg),
                                               ev.splits if ev else _splits(cfg),
                                               n_instants=args.instants_validation,
                                               verbose=True)
                ecrire_ajustement(ajust, out / "ajustement_validation.json")
                resume["ajustement"] = {k: ajust[k] for k in ("instants", "variogramme",
                                                               "alpha")}
            noms = _liste(args.references, str)
            banc = Banc(cfg, ev, ajust, noms, verbose=True)

            # 2. comparaison (L6)
            if "comparaison" in etapes:
                print("\n2. comparaison sur le TEST synoptique")
                comp = comparer(banc, args.instants)
                _ecrire_json(out / "comparaison_test.json", comp)
                _ecrire_csv(out / "comparaison_par_niveau.csv", tableau_niveaux(comp))
                resume["comparaison"] = {n: {k: r[k] for k in (
                    "rmse_max_uv", "niveau_rmse_max_m", "dso1_rmse_global_u",
                    "dso1_rmse_global_v", "r2_min_uv", "dso1_rmse", "dso2_rms_moyen",
                    "dso3_rugosite")} for n, r in comp["methodes"].items()}
                if args.instants is None:
                    resume["controle_v7"] = controle_v7(cfg, comp)
                _imprimer(comp)

            # 3. E1
            if "e1" in etapes:
                print("\n3. E1 densite du reseau")
                esp = ESPACEMENTS_E1 if args.espacements is None else _liste(args.espacements)
                banc_e1 = Banc(cfg, ev, ajust, _liste(args.references_e1, str), verbose=True)
                e1 = etude_densite(banc_e1, _liste(args.fractions) or FRACTIONS_E1, esp,
                                   n_instants=args.instants)
                _ecrire_json(out / "e1_densite.json", e1)
                import pandas as pd
                _ecrire_csv(out / "e1_densite.csv", pd.DataFrame(e1["lignes"]))
                resume["e1"] = e1["recommandation"]

            # 4. E2
            if "e2" in etapes and ev is not None:
                print("\n4. E2 placement optimal")
                e2 = placement_optimal(banc, base_espacement_m=args.e2_base_espacement,
                                       n_etapes=args.e2_etapes, par_etape=args.e2_par_etape,
                                       n_validation=args.e2_validation, n_test=args.instants,
                                       n_hasard=args.e2_hasard)
                cartes = e2["_cartes"]
                np.savez_compressed(out / "e2_cartes.npz",
                                    **{k: v for k, v in cartes.items() if v is not None})
                _ecrire_json(out / "e2_placement.json", _sans_prive(e2))
                resume["e2"] = e2["resume"]
            elif "e2" in etapes:
                resume["e2"] = "saute : demande le modele"

            # 5. E3
            if "e3" in etapes:
                print("\n5. E3 capteur virtuel de secours")
                e3 = capteur_secours(banc, args.instants, n_pannes=args.e3_pannes,
                                     espacement=args.e3_espacement)
                _ecrire_csv(out / "e3_secours.csv", e3["_lignes"])
                _ecrire_json(out / "e3_secours.json", _sans_prive(e3))
                resume["e3"] = e3["resume"]

            # 6. E5 (inference et references ; entrainement reconstruit)
            if "e5" in etapes:
                print("\n6. E5 sobriete")
                resume["e5"] = _e5(args, banc, ev, sob, out)
                _ecrire_json(out / "e5_sobriete.json", resume["e5"])
        resume["e5_session"] = s_session.resultat
    except (ValueError, FileExistsError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1
    resume["duree_s"] = round(time.perf_counter() - t0, 1)
    _ecrire_json(out / "resume.json", resume)
    print(f"\nresume -> {out / 'resume.json'}")
    return 0


def _geom(cfg):
    from pcfno3d.geometry import Geometry
    return Geometry(cfg, device="cpu")


def _splits(cfg):
    from pcfno3d.cache import load_index
    from pcfno3d.dataset import make_splits
    return make_splits(cfg, load_index(cfg))


def _imprimer(comp: dict) -> None:
    print(f"   {comp['instants']} instants, capteurs visibles "
          f"{comp['reseau'].get('capteurs_visibles')}")
    print("   methode              RMSE max u,v (niveau)   global u / v      R2 min   DSO1")
    for nom, r in comp["methodes"].items():
        rm = r["rmse_max_uv"]
        print(f"   {nom:20s} {rm if rm is not None else float('nan'):6.3f} "
              f"({r['niveau_rmse_max_m']} m)       {r['dso1_rmse_global_u']:.3f} / "
              f"{r['dso1_rmse_global_v']:.3f}   {r['r2_min_uv'] or float('nan'):.4f}   "
              f"{'conforme' if r['dso1_rmse'] else 'non conforme'}")


def _e5(args, banc, ev, sob: dict, out: Path) -> dict:
    from pcfno3d.sobriete import (Sobriete, energie_entrainement, historique_entrainement,
                                  par_unite)

    idx = banc.instants(args.e5_instants)
    stems = [str(banc.splits.test[0][i]) for i in idx]
    res: dict = {"definition": "E5, cahier v9 p. 42 (section 19.1), tache 33 p. 45",
                 "comparaison_simulation": "cahier v9 p. 8 : simulation numerique (CFD, WRF) "
                 "« plusieurs heures de calcul par instant » ; aucune mesure d'energie WRF "
                 "dans le depot (a sourcer)"}
    cc_dir = (out / "codecarbon") if sob.get("utiliser_codecarbon") else None
    if ev is not None:
        wf = ev.wf
        donnees = [wf.surface_from_cache(st) for st in stems]
        for S, hour, doy, _ in donnees[:2]:                  # chauffe, hors mesure
            wf.predict(S, hour, doy, mask_nan=False, reliability=False)
        with Sobriete("inference_pcfno3d", cc_dir, **sob) as m:
            for S, hour, doy, _ in donnees:
                wf.predict(S, hour, doy, mask_nan=False, reliability=False)
        res["inference_pcfno3d"] = {"phase": m.resultat,
                                    "par_reconstruction": par_unite(m.resultat, len(donnees)),
                                    "device": str(wf.device),
                                    "capteurs": "tous (cas de production)"}
    import numpy as np
    cache = banc.cfg.paths.cache_dir
    surfaces = [np.load(cache / f"{st}_S.npy") for st in stems]
    with Sobriete("references", cc_dir, **sob) as m:
        for S in surfaces:
            banc.calc.champs(S, S[7] > 0.5)
    res["references"] = {"phase": m.resultat,
                         "par_instant_toutes_references": par_unite(m.resultat, len(surfaces),
                                                                    "instant"),
                         "references": list(banc.refs), "calcul": "CPU (numpy / scipy)"}
    j = Path(args.journal_entrainement)
    if j.exists():
        h = historique_entrainement(j)
        res["entrainement_v7"] = {"historique": h, **energie_entrainement(
            h, args.gpu_watts, args.intensite_gco2_kwh)}
    else:
        res["entrainement_v7"] = {"note": f"journal introuvable : {j}"}
    return res


if __name__ == "__main__":
    sys.exit(main())
