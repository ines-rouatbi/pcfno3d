"""Atlas eolien local (cahier v12, E8) : vitesse moyenne, Weibull, facteur de charge et production
annuelle a hauteur de moyeu, a 100 m de resolution, a partir des reconstructions des instants du cache ;
trois meilleurs sites ; GeoTIFF (E15).

    python scripts/atlas_eolien.py -c grand_tunis_v2_service_colab --sortie <runs>/atlas_eolien

Sorties : atlas.json (resume, sites), atlas.npz, atlas_<couche>.tif, atlas.png. Le service les sert a
/api/atom-wind/atlas.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]


def main() -> int:
    from pcfno3d import atlas
    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.inference import WindFieldModel

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--splits", default="train,val,test", help="instants du cache utilises")
    ap.add_argument("--max-instants", type=int)
    ap.add_argument("--hub", type=float, default=100.0, help="hauteur de moyeu au-dessus du sol (m)")
    ap.add_argument("--courbe", default=atlas.COURBE_DEFAUT)
    ap.add_argument("--sortie")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg.apply_overrides({k: v})
    wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device, verbose=False)
    idx = load_index(cfg)
    stems = list(idx[idx["split"].isin(args.splits.split(","))].sort_values("timestamp")["stem"])
    if args.max_instants:
        stems = stems[:: max(1, len(stems) // args.max_instants)][:args.max_instants]
    courbe = atlas.charger_courbe(REPO / "configs" / "eolien" / "parc_demo.yaml", args.courbe)
    a = atlas.construire(wf, stems, hub_agl_m=args.hub, courbe=courbe)
    sortie = Path(args.sortie) if args.sortie else cfg.paths.run_dir.parent / "atlas_eolien"
    resume = atlas.ecrire(a, wf.geom, sortie)
    print(json.dumps({k: v for k, v in resume.items() if k != "limites"}, indent=1, ensure_ascii=False))
    print(f"-> {sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
