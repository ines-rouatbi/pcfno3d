"""Lot M2 — calibration de la fiabilite et verdict DSO6 (session GPU), version 2.

A lancer dans une session Colab, depuis le depot installe, avec le cache et le checkpoint.
Le test a deja ete lu une fois (15/10/2026, DSO6 non conforme) : la seconde lecture se lance
AVEC --deuxieme-lecture, qui l'ecrit dans le verdict (champ `mesure`) :

    !python scripts/m2_calibration_colab.py -c colab --deuxieme-lecture

1. VALIDATION (selection synoptique fixe, graine 1000, `mask.keep_eval` des capteurs) :
   passe servie + N passes masquees par instant (par lots de `reliability.batch`), un
   histogramme de |erreur| / sigma brut par pli de jours (2 plis : jours de rang pair /
   impair). Marge de calibration par la regle « auto » fixee d'avance (validation croisee
   sur les plis, grille q = 0,80..0,90 ; reliability.choose_q), puis k finaux sur toute la
   validation. -> calibration_validation.json (regle, grille, couverture hors-pli, q retenu)
2. TEST synoptique (graine 2000, memes capteurs visibles que DSO1) avec ces k : couverture
   de l'intervalle a 80 % et Spearman(|erreur|, sigma) par niveau et global. -> dso6.json
   Le test ne sert QU'AU verdict : rien n'y est ajuste.
3. DSO4 sur un instant de test, sur `predict` comme en production : SANS fiabilite, AVEC
   (lot configure), et pour information avec des lots de 1, 4 et 8 passes (un lot trop
   gros pour la memoire est note, sans arreter la campagne).

Tout s'ecrit dans un dossier NEUF, run_dir/fiabilite_pcfno3d/<modele>/session_<horodatage>/ :
le script n'ouvre en ecriture aucun fichier existant (checkpoints, evaluations, JSON figes du
notebook, sessions precedentes). Pour servir la calibration retenue ensuite :

    inference:
      reliability: <run_dir>/fiabilite_pcfno3d/<modele>/session_.../calibration_validation.json

Codes de retour : 0 campagne terminee (le verdict DSO6 est dans dso6.json et imprime, il ne
change pas le code) ; 1 erreur (pas d'instant de validation ou de test en cache, pli vide...).
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

LOTS_INFO = (1, 4, 8)          # DSO4 pour information : taille du lot de l'ensemble


def _bench(wf, S, hour, doy, n: int, reliability: bool) -> dict:
    """benchmark DSO4 ; une erreur memoire (lot trop gros) est rendue, pas levee."""
    import torch

    try:
        return wf.benchmark(S, hour, doy, n=n, reliability=reliability)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {"erreur": f"memoire GPU insuffisante : {str(exc).splitlines()[0]}"}


def _ms(b: dict) -> str:
    if "erreur" in b:
        return b["erreur"]
    return f"{b['mean_s'] * 1000:.0f} ms ({'CONFORME' if b['conforme'] else 'NON CONFORME'})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="colab")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--instants-validation", type=int, help="defaut : toute la selection")
    ap.add_argument("--instants-test", type=int, help="defaut : tout le test synoptique")
    ap.add_argument("--keep", type=float, help="capteurs visibles (defaut : mask.keep_eval)")
    ap.add_argument("--membres", type=int, help="passes masquees (defaut : reliability.members)")
    ap.add_argument("--lot", type=int, help="passes par lot (defaut : reliability.batch)")
    ap.add_argument("--q", help="quantile vise : « auto » (regle fixee d'avance, defaut) ou "
                                "un nombre (defaut : reliability.calibration_q)")
    ap.add_argument("--benchmark", type=int, default=20, help="repetitions DSO4 (defaut 20)")
    ap.add_argument("--deuxieme-lecture", action="store_true",
                    help="le test a deja ete lu (15/10/2026, non conforme) : le verdict et le "
                         "resume portent le champ `mesure` (pas une mesure independante)")
    args = ap.parse_args(argv)

    from pcfno3d.config import Config
    from pcfno3d.evaluate import Evaluator
    from pcfno3d.reliability import MESURE_SECONDE_LECTURE, Calibration

    cfg = Config.load(args.config)
    if args.membres:
        cfg.apply_overrides({"reliability.members": args.membres})
    if args.lot:
        cfg.apply_overrides({"reliability.batch": args.lot})
    if args.q:
        cfg.apply_overrides({"reliability.calibration_q": args.q})
    cfg.inference.reliability = None          # le modele est charge SANS fiabilite
    ev = Evaluator(cfg, checkpoint=args.checkpoint, device=args.device)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ev.reliability_dir() / f"session_{stamp}"
    out.mkdir(parents=True, exist_ok=False)   # dossier neuf, ou erreur : rien n'est ecrase
    print(f"sortie -> {out}\n")
    if args.deuxieme_lecture:
        print(f"mesure : {MESURE_SECONDE_LECTURE}\n")

    t0 = time.perf_counter()
    try:
        cal = ev.calibrate_reliability(out=out / "calibration_validation.json",
                                       n_instants=args.instants_validation, keep=args.keep)
        print(f"  ({time.perf_counter() - t0:.0f} s)\n")
        t1 = time.perf_counter()
        dso6 = ev.dso6(calibration=cal["fichier"], n_instants=args.instants_test,
                       keep=args.keep, out=out / "dso6.json",
                       deuxieme_lecture=args.deuxieme_lecture)
        print(f"  ({time.perf_counter() - t1:.0f} s)\n")
    except ValueError as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1

    # DSO4 sans et avec la fiabilite, sur predict (chemin de production, tous les capteurs)
    rc = cfg.reliability
    stem = str(ev.splits.test[0][0])
    S, hour, doy, _ = ev.wf.surface_from_cache(stem)
    ev.wf.reliability = Calibration.load(cal["fichier"])
    lot_cfg = rc.batch
    sans = _bench(ev.wf, S, hour, doy, args.benchmark, reliability=False)
    avec = _bench(ev.wf, S, hour, doy, args.benchmark, reliability=True)
    par_lot = {}
    for lot in LOTS_INFO:                     # pour information
        rc.batch = lot
        par_lot[str(lot)] = _bench(ev.wf, S, hour, doy, args.benchmark, reliability=True)
    rc.batch = lot_cfg
    ev.wf.reliability = None
    dso4 = {"instant": stem, "sans_fiabilite": sans, "avec_fiabilite": avec,
            "lot": lot_cfg, "avec_fiabilite_par_lot": par_lot,
            "membres": rc.members, "seuil_s": cfg.dso.time_s,
            "device": str(ev.wf.device)}
    print(f"DSO4 : sans fiabilite {_ms(sans)} | avec (lot {lot_cfg}) {_ms(avec)} | "
          f"seuil {cfg.dso.time_s} s")
    print("  pour information, avec fiabilite par taille de lot : "
          + " | ".join(f"lot {k} : {_ms(v)}" for k, v in par_lot.items()))

    resume = {"modele": ev.wf.checkpoint_path.name, "dossier": str(out),
              **({"mesure": MESURE_SECONDE_LECTURE} if args.deuxieme_lecture else {}),
              "calibration": {k: cal.get(k) for k in ("k_global", "scale_factor", "quantile",
                                                       "calibration_q",
                                                       "couverture80_validation",
                                                       "coverage_80_validation", "instants",
                                                       "regle_quantile")},
              "calibration_sha256": dso6["calibration_sha256"],
              "dso6": {k: dso6[k] for k in ("coverage_80", "corr_error_sigma",
                                            "dso6_couverture_ok", "dso6_correlation_ok",
                                            "dso6_conforme", "instants_synoptiques")},
              "dso6_global": dso6["global"], "dso4": dso4,
              "duree_s": round(time.perf_counter() - t0, 1)}
    (out / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\nDSO6 -> {'CONFORME' if dso6['dso6_conforme'] else 'NON CONFORME'}"
          + (" (seconde lecture du test)" if args.deuxieme_lecture else "")
          + f" | resume -> {out / 'resume.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
