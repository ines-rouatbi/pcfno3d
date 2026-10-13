"""La v7 tient-elle avec la DENSITE d'un vrai reseau de capteurs ?

Toutes les evaluations jusqu'ici masquent au hasard une couche de surface SYNTHETIQUE quasi
continue (494 006 points a 50 m par instant) : a 50 %, le modele voit ~60 000 points. Un
vrai reseau ANEO a des points espaces de ~500 m a 1 km (Rayen, 2026-07-17). Ce script
compare, sur les memes instants de test synoptiques :

    reference : 50 % des points au hasard (protocole DSO1)
    reseau    : un capteur tous les 300 / 500 / 1000 m (grille reguliere, decalage tire)

C'est la densite d'un vrai reseau, pas sa disposition reelle (inconnue a ce jour).

    python scripts/eval_reseau.py -c local_v7 --max-instants 8
    python scripts/eval_reseau.py -c colab                     # les 120 synoptiques, GPU
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="local_v7")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--spacings", default="300,500,1000", help="espacements en m")
    ap.add_argument("--max-instants", type=int, help="limite (essai rapide sur CPU)")
    args = ap.parse_args()

    from pcfno3d.config import Config
    from pcfno3d.dataset import eval_loader, is_synoptic
    from pcfno3d.evaluate import Evaluator

    cfg = Config.load(args.config)
    ev = Evaluator(cfg, checkpoint=args.checkpoint, device=args.device, verbose=False)
    te_s = ev.splits.test[0]
    idx = np.array([i for i, s in enumerate(te_s) if is_synoptic(ev.splits.stamp_of[s])])
    if len(idx) == 0:
        idx = np.arange(len(te_s))
    if args.max_instants:
        idx = idx[:args.max_instants]
    cover = cfg.dso.min_level_cover
    print(f"modele {ev.wf.checkpoint_path.name} | {ev.wf.device} | {len(idx)} instants "
          f"de test synoptiques | niveaux evalues : couverture >= {cover:.0%}")

    cas = [("50 % au hasard (DSO1)", None)] + [
        (f"reseau {int(s)} m", float(s)) for s in args.spacings.split(",")]
    resultats = []
    for nom, spacing in cas:
        t0 = time.time()
        loader, stems = eval_loader(cfg, ev.geom, ev.splits, idx, seed=1234,
                                    keep=cfg.mask.keep_eval, spacing_m=spacing)
        n_capteurs = [int(b[1].sum()) for b in loader]          # meme graine : memes masques
        levels, _ = ev.evaluate_loader(loader, stems)
        lv = levels[levels["couverture"] >= cover]
        r = {"cas": nom, "espacement_m": spacing,
             "capteurs_moyen": float(np.mean(n_capteurs)),
             "rmse_u_max": float(lv["rmse_u"].max()), "rmse_v_max": float(lv["rmse_v"].max()),
             "niveau_pire_m": int(lv.loc[lv[["rmse_u", "rmse_v"]].max(axis=1).idxmax(), "z_m"]),
             "r2_min": float(lv[["r2_u", "r2_v"]].min().min()),
             "par_niveau": lv[["z_m", "rmse_u", "rmse_v", "r2_u", "r2_v"]].round(4)
                             .to_dict(orient="records"),
             "duree_s": round(time.time() - t0, 1)}
        resultats.append(r)
        print(f"{nom:24s} | capteurs ~{r['capteurs_moyen']:8.0f} | RMSE max u {r['rmse_u_max']:.3f}"
              f"  v {r['rmse_v_max']:.3f} (pire a {r['niveau_pire_m']} m) | R2 min "
              f"{r['r2_min']:.4f} | {r['duree_s']:.0f} s", flush=True)

    print("\nRMSE u / v par niveau :")
    print("  niveau | " + " | ".join(f"{r['cas'][:18]:^18s}" for r in resultats))
    for k, row in enumerate(resultats[0]["par_niveau"]):
        cells = [f"{r['par_niveau'][k]['rmse_u']:.3f} / {r['par_niveau'][k]['rmse_v']:.3f}"
                 for r in resultats]
        print(f"  {row['z_m']:5d} m | " + " | ".join(f"{c:^18s}" for c in cells))

    out = ev.out_dir.parent / f"reseau_{ev.wf.checkpoint_path.stem}_{len(idx)}instants.json"
    out.write_text(json.dumps({"config": args.config, "instants": len(idx),
                               "resultats": resultats}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
