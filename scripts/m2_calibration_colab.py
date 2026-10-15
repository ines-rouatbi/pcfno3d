"""Lot M2 — calibration de la fiabilite et verdict DSO6 (session GPU).

A lancer dans une session Colab, depuis le depot installe, avec le cache et le checkpoint :

    !python scripts/m2_calibration_colab.py -c colab

1. VALIDATION (selection synoptique fixe, graine 1000, `mask.keep_eval` des capteurs) :
   passe servie + N passes masquees par instant, histogrammes de |erreur| / sigma brut,
   facteurs k par composante et par niveau. -> calibration_validation.json
2. TEST synoptique (graine 2000, memes capteurs visibles que DSO1) avec ces k : couverture
   de l'intervalle a 80 % et Spearman(|erreur|, sigma) par niveau et global. -> dso6.json
   Le test ne sert QU'AU verdict : rien n'y est ajuste.
3. DSO4 sur un instant de test : temps de livraison AVEC la fiabilite (passe servie +
   ensemble) et SANS (passe seule), sur `predict` comme en production.

Tout s'ecrit dans un dossier NEUF, run_dir/fiabilite_pcfno3d/<modele>/session_<horodatage>/ :
le script n'ouvre en ecriture aucun fichier existant (checkpoints, evaluations, JSON figes du
notebook). Pour servir la calibration retenue ensuite :

    inference:
      reliability: <run_dir>/fiabilite_pcfno3d/<modele>/session_.../calibration_validation.json

Codes de retour : 0 campagne terminee (le verdict DSO6 est dans dso6.json et imprime, il ne
change pas le code) ; 1 erreur (pas d'instant de validation ou de test en cache...).
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="colab")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--instants-validation", type=int, help="defaut : toute la selection")
    ap.add_argument("--instants-test", type=int, help="defaut : tout le test synoptique")
    ap.add_argument("--keep", type=float, help="capteurs visibles (defaut : mask.keep_eval)")
    ap.add_argument("--membres", type=int, help="passes masquees (defaut : reliability.members)")
    ap.add_argument("--q", type=float, help="quantile vise (defaut : reliability.calibration_q)")
    ap.add_argument("--benchmark", type=int, default=20, help="repetitions DSO4 (defaut 20)")
    args = ap.parse_args(argv)

    from pcfno3d.config import Config
    from pcfno3d.evaluate import Evaluator
    from pcfno3d.reliability import Calibration

    cfg = Config.load(args.config)
    if args.membres:
        cfg.apply_overrides({"reliability.members": args.membres})
    if args.q:
        cfg.apply_overrides({"reliability.calibration_q": args.q})
    cfg.inference.reliability = None          # le modele est charge SANS fiabilite
    ev = Evaluator(cfg, checkpoint=args.checkpoint, device=args.device)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ev.reliability_dir() / f"session_{stamp}"
    out.mkdir(parents=True, exist_ok=False)   # dossier neuf, ou erreur : rien n'est ecrase
    print(f"sortie -> {out}\n")

    t0 = time.perf_counter()
    try:
        cal = ev.calibrate_reliability(out=out / "calibration_validation.json",
                                       n_instants=args.instants_validation, keep=args.keep)
        print(f"  ({time.perf_counter() - t0:.0f} s)\n")
        t1 = time.perf_counter()
        dso6 = ev.dso6(calibration=cal["fichier"], n_instants=args.instants_test,
                       keep=args.keep, out=out / "dso6.json")
        print(f"  ({time.perf_counter() - t1:.0f} s)\n")
    except ValueError as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1

    # DSO4 avec et sans la fiabilite, sur predict (chemin de production, tous les capteurs)
    stem = str(ev.splits.test[0][0])
    S, hour, doy, _ = ev.wf.surface_from_cache(stem)
    ev.wf.reliability = Calibration.load(cal["fichier"])
    avec = ev.wf.benchmark(S, hour, doy, n=args.benchmark, reliability=True)
    sans = ev.wf.benchmark(S, hour, doy, n=args.benchmark, reliability=False)
    ev.wf.reliability = None
    dso4 = {"instant": stem, "avec_fiabilite": avec, "sans_fiabilite": sans,
            "membres": cfg.reliability.members, "seuil_s": cfg.dso.time_s}
    print(f"DSO4 : avec fiabilite {avec['mean_s'] * 1000:.0f} ms "
          f"({'CONFORME' if avec['conforme'] else 'NON CONFORME'}) | sans "
          f"{sans['mean_s'] * 1000:.0f} ms ({'CONFORME' if sans['conforme'] else 'NON CONFORME'})"
          f" | seuil {cfg.dso.time_s} s")

    resume = {"modele": ev.wf.checkpoint_path.name, "dossier": str(out),
              "calibration": {k: cal[k] for k in ("k_global", "scale_factor", "quantile",
                                                   "couverture80_validation",
                                                   "coverage_80_validation", "instants")},
              "calibration_sha256": dso6["calibration_sha256"],
              "dso6": {k: dso6[k] for k in ("coverage_80", "corr_error_sigma",
                                            "dso6_couverture_ok", "dso6_correlation_ok",
                                            "dso6_conforme", "instants_synoptiques")},
              "dso6_global": dso6["global"], "dso4": dso4,
              "duree_s": round(time.perf_counter() - t0, 1)}
    (out / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\nDSO6 -> {'CONFORME' if dso6['dso6_conforme'] else 'NON CONFORME'} | "
          f"resume -> {out / 'resume.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
