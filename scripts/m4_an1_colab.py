"""Lot M4 — mesure du critere AN1 par simulation de pannes et de derives (session GPU).

A lancer dans une session Colab, depuis le depot installe, avec le cache et le checkpoint :

    !python scripts/m4_an1_colab.py -c colab

1. VALIDATION (jours de validation, instants synoptiques dans l'ordre) : injection de
   derives (biais, gain, girouette) et de pannes (valeurs figees) sur des capteurs tires a
   graine fixe, chaine d'alertes complete, pour les DEUX statistiques de derive
   (« innovation » et « influence », memes passes). Pour chacune, le plus petit kappa
   dont la part de fausses alertes reste sous --fdr-cible ; la statistique retenue est
   celle qui detecte le mieux a ce kappa. -> seuils_validation.json (jeu: validation)
2. TEST (jours de test, autre graine) avec ces seuils FIGES : detection, fausses alertes,
   verdict AN1. Le test ne sert QU'AU verdict : rien n'y est ajuste. -> an1_test.json
3. Distributions sur la validation du residu de masse et de la condition au sol par instant
   (pour juger alerts.sol_rms_max_m_s, laisse a null tant qu'il n'est pas decide).

Tout s'ecrit dans un dossier NEUF, run_dir/alertes_pcfno3d/<modele>/session_<horodatage>/ :
le script n'ouvre en ecriture aucun fichier existant. Pour servir les seuils retenus :

    alerts:
      enabled: true
      seuils: <run_dir>/alertes_pcfno3d/<modele>/session_.../seuils_validation.json

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
    ap.add_argument("--jours-validation", type=int, help="defaut : tous les jours")
    ap.add_argument("--jours-test", type=int, help="defaut : tous les jours")
    ap.add_argument("--derives-par-jour", type=int, default=40)
    ap.add_argument("--figees-par-jour", type=int, default=10)
    ap.add_argument("--e-ref", type=float, default=1.0,
                    help="amplitude effective (m/s) a partir de laquelle une derive compte "
                         "dans la detection AN1 (defaut 1.0, le seuil du contrat)")
    ap.add_argument("--fdr-cible", type=float, default=0.025,
                    help="part de fausses alertes visee SUR LA VALIDATION (marge sous 5 %%)")
    ap.add_argument("--graine", type=int, default=4)
    args = ap.parse_args(argv)

    import numpy as np

    from pcfno3d import an1
    from pcfno3d.alerts import STATISTIQUES
    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.inference import WindFieldModel

    cfg = Config.load(args.config)
    cfg.inference.reliability = None          # passes simples : la fiabilite n'y joue pas
    wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device)
    index = load_index(cfg)
    val, test = _jours(index, "val", args.jours_validation), _jours(index, "test",
                                                                     args.jours_test)
    if not val or not test:
        print("ERREUR : il faut des jours de validation ET de test dans le cache",
              file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.paths.run_dir / "alertes_pcfno3d" / wf.checkpoint_path.stem / f"session_{stamp}"
    out.mkdir(parents=True, exist_ok=False)   # dossier neuf, ou erreur : rien n'est ecrase
    print(f"sortie -> {out}\n")
    a = cfg.alerts
    t0 = time.perf_counter()

    # 1. VALIDATION : reglage de kappa et choix de la statistique
    rng = np.random.default_rng(args.graine)
    sim_val = [an1.simuler_jour(wf, j, a, rng, args.derives_par_jour, args.figees_par_jour,
                                STATISTIQUES) for j in val]
    choix = {st: an1.choisir_kappa(sim_val, st, fdr_cible=args.fdr_cible, e_ref=args.e_ref)
             for st in STATISTIQUES}
    for st, c in choix.items():
        v = c["validation"]
        print(f"validation {st:10s} kappa {c['derive_kappa']:5.2f} | detection "
              f"{v['detection']:.3f} | fausses alertes {v['fausses_alertes']:.3f}"
              f"{'' if c['cible_atteinte'] else ' (cible non atteinte)'}")
    best = max(choix.values(), key=lambda c: (c["cible_atteinte"],
                                              c["validation"]["detection"] or 0.0,
                                              -c["validation"]["fausses_alertes"]))
    seuils = {**best, "derive_min_m_s": a.derive_min_m_s, "modele": wf.checkpoint_path.name,
              "jours_validation": [j[0].split("_")[0] for j in val],
              "derives_par_jour": args.derives_par_jour, "figees_par_jour": args.figees_par_jour,
              "graine": args.graine, "alternatives": {st: {k: c[k] for k in (
                  "derive_kappa", "cible_atteinte", "validation")} for st, c in choix.items()}}
    _ecrire(out / "seuils_validation.json", seuils)
    print(f"  ({time.perf_counter() - t0:.0f} s) -> retenu : {best['derive_statistique']}, "
          f"kappa {best['derive_kappa']}\n")

    # 2. TEST : verdict avec les seuils figes
    t1 = time.perf_counter()
    rng = np.random.default_rng(args.graine + 1000)
    st = best["derive_statistique"]
    sim_test = [an1.simuler_jour(wf, j, a, rng, args.derives_par_jour, args.figees_par_jour,
                                 (st,)) for j in test]
    capteurs = an1.evaluer_an1(sim_test, st, best["derive_kappa"], args.e_ref)
    entrees = an1.evaluer_entrees(sim_test)
    verdict = bool(capteurs["an1_capteurs_conforme"] and entrees["an1_entrees_conforme"])
    an1_test = {"jeu": "test", "seuils_validation": str(out / "seuils_validation.json"),
                "jours_test": [j[0].split("_")[0] for j in test], "capteurs": capteurs,
                "entrees": entrees, "an1_conforme": verdict,
                "criteres": {"detection_min": an1.DETECTION_MIN,
                             "fausses_alertes_max": an1.FAUSSES_MAX}}
    _ecrire(out / "an1_test.json", an1_test)
    print(f"test : detection {capteurs['detection']:.3f} (derives "
          f"{capteurs['detection_derives']}, pannes {capteurs['detection_pannes']}) | "
          f"fausses alertes {capteurs['fausses_alertes']:.3f} | entrees : detection "
          f"{entrees['detection']}, fausses {entrees['fausses_alertes']}  "
          f"({time.perf_counter() - t1:.0f} s)")

    # 3. physique sur la validation (instants avec derives injectees : effet local)
    phys = [p for j in sim_val for p in j["physique"]]
    resume = {"modele": wf.checkpoint_path.name, "dossier": str(out),
              "seuils": {k: seuils[k] for k in ("derive_statistique", "derive_kappa",
                                                "fdr_cible", "e_ref_m_s", "cible_atteinte")},
              "validation": best["validation"], "test": capteurs, "entrees_test": entrees,
              "an1_conforme": verdict,
              "physique_validation": {
                  "mass_residual_rms": _quantiles([p["mass_residual_rms"] for p in phys]),
                  "ground_condition_rms_ms": _quantiles([p["ground_condition_rms_ms"]
                                                         for p in phys]),
                  "part_mass_residual_ok": float(np.mean([p["mass_residual_ok"]
                                                          for p in phys])),
                  "rho_source": a.rho_source},
              "duree_s": round(time.perf_counter() - t0, 1)}
    _ecrire(out / "resume.json", resume)
    print(f"\nAN1 -> {'CONFORME' if verdict else 'NON CONFORME'} | resume -> "
          f"{out / 'resume.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
