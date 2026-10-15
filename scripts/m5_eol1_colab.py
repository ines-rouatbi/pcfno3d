"""Lot M5 — mesure du critere EOL1 (production du parc) en session GPU.

A lancer dans une session Colab, depuis le depot installe, avec le cache et le checkpoint :

    !python scripts/m5_eol1_colab.py -c colab --parc configs/eolien/parc_demo.yaml

EOL1 (docs/EOLIEN.md) : sur les instants synoptiques de TEST (ceux de DSO1), production du
parc calculee depuis le champ RECONSTRUIT et depuis la VERITE 3D du generateur (cache Y),
avec le meme parc, la meme densite (surface de l'instant) et la meme courbe :

    ecart = |E_rec - E_vrai| / E_vrai,   E = somme sur les instants de P_parc(t)
    conforme si ecart < wind_farm.eol1_max_rel (5 %)

Mesure a chaque fraction de capteurs visibles de --keep (defaut 1,0 = production, puis 0,5
= protocole DSO1) ; le verdict retenu est celui de la premiere valeur. Le test ne sert QU'AU
verdict : rien n'y est ajuste.

Tout s'ecrit dans un dossier NEUF, run_dir/eolien_pcfno3d/<modele>/eol1_<horodatage>/ : le
script n'ouvre en ecriture aucun fichier existant.

Codes de retour : 0 mesure terminee (le verdict est dans eol1.json, il ne change pas le
code) ; 1 erreur (aucun instant de test en cache, parc absent ou invalide...).
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
    ap.add_argument("--parc", help="declaration du parc (defaut : wind_farm.turbines, sinon "
                                   "configs/eolien/parc_demo.yaml)")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--instants", type=int, help="defaut : tout le test synoptique")
    ap.add_argument("--keep", type=float, nargs="+", default=[1.0, 0.5],
                    help="fractions de capteurs visibles (defaut : 1.0 0.5)")
    args = ap.parse_args(argv)

    from pcfno3d.config import Config
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.windfarm import eol1, load_farm

    cfg = Config.load(args.config)
    cfg.inference.reliability = None          # EOL1 ne porte que sur la valeur servie
    parc = args.parc or cfg.wind_farm.turbines or REPO / "configs/eolien/parc_demo.yaml"
    cfg.wind_farm.turbines = None             # le modele est charge sans parc attache
    try:
        wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device)
        farm = load_farm(parc, wf.geom, cfg)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.paths.run_dir / "eolien_pcfno3d" / wf.checkpoint_path.stem / f"eol1_{stamp}"
    out.mkdir(parents=True, exist_ok=False)   # dossier neuf, ou erreur : rien n'est ecrase
    print(f"parc : {farm.source} ({len(farm.turbines)} eoliennes, {farm.rated_kw:g} kW"
          f"{', EXEMPLE' if farm.exemple else ''}) | sortie -> {out}\n")

    t0 = time.perf_counter()
    mesures = []
    for keep in args.keep:
        t1 = time.perf_counter()
        try:
            r = eol1(wf, farm, keep=keep, n_instants=args.instants)
        except ValueError as exc:
            print(f"ERREUR : {exc}", file=sys.stderr)
            return 1
        r["duree_s"] = round(time.perf_counter() - t1, 1)
        mesures.append(r)
        ecart = r["ecart_relatif"]
        print(f"capteurs {keep:.0%} : {r['instants']} instants | ecart "
              + (f"{ecart:.2%}" if ecart is not None else "indefini (production vraie nulle)")
              + f" | nMAE {r['nmae_puissance']:.2%} | accord des etats "
              f"{r['accord_etats']:.0%} | {r['duree_s']} s")
        (out / f"eol1_keep{keep:g}.json").write_bytes(
            json.dumps(r, indent=2, ensure_ascii=False).encode("utf-8"))

    ref = mesures[0]
    resume = {"critere": "EOL1", "modele": wf.checkpoint_path.name, "dossier": str(out),
              "verdict_sur_capteurs": ref["capteurs_visibles"],
              "ecart_relatif": ref["ecart_relatif"], "seuil": ref["seuil"],
              "eol1_conforme": ref["eol1_conforme"],
              "par_capteurs": {f"{m['capteurs_visibles']:g}": {
                  k: m[k] for k in ("instants", "ecart_relatif", "eol1_conforme",
                                    "nmae_puissance", "accord_etats", "rmse_vent_moyeu_ms")}
                  for m in mesures},
              "parc": ref["parc"], "duree_s": round(time.perf_counter() - t0, 1)}
    (out / "eol1.json").write_bytes(json.dumps(resume, indent=2, ensure_ascii=False)
                                    .encode("utf-8"))
    verdict = {True: "CONFORME", False: "NON CONFORME", None: "INDEFINI"}[ref["eol1_conforme"]]
    print(f"\nEOL1 -> {verdict} | resume -> {out / 'eol1.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
