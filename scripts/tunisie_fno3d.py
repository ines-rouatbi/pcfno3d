"""PC-FNO3D Tunisie : jeu de donnees ERA5 + METAR au format v7 (voir pcfno3d.tunisie_fno3d).

    python scripts/tunisie_fno3d.py construire     # jeu (meta, grille, decoupage) + cache
    python scripts/tunisie_fno3d.py -c tunisie_fno3d_land construire   # grille ERA5-Land
    python scripts/tunisie_fno3d.py masse          # residu de masse de la verite -> mass_ref
puis l'entrainement et l'evaluation de la v7, sans changement :
    python -m pcfno3d -c tunisie_fno3d train
    python -m pcfno3d -c tunisie_fno3d evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]

BASE = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "pcfno3d"


def main() -> int:
    from pcfno3d.config import Config
    from pcfno3d import tunisie_fno3d as T

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("commande", choices=["construire", "masse"])
    ap.add_argument("-c", "--config", default="tunisie_fno3d")
    ap.add_argument("--era5", default=str(BASE / "era5" / "tunisie"))
    ap.add_argument("--metar", default=str(BASE / "metar" / "historique"))
    ap.add_argument("--instants", type=int, default=50)
    ap.add_argument("--grille", choices=["era5", "land"], default=None,
                    help="era5 (0,25 deg) ou land (ERA5-Land 0,1 deg) ; par defaut land si "
                         "le nom de la config contient _land")
    args = ap.parse_args()
    cfg = Config.load(args.config)

    if args.commande == "construire":
        fichiers = sorted(Path(args.metar).glob("metar_*.csv"))
        if not fichiers:
            print(f"aucune archive METAR dans {args.metar}", file=sys.stderr)
            return 2
        grille = args.grille or ("land" if "_land" in args.config else "era5")
        T.construire(Path(args.era5), fichiers, cfg.paths.data_dir, cfg.paths.cache_dir,
                     grille=grille)
        return 0

    bilan = T.bilan_masse(cfg, n=args.instants)
    # la verite elle-meme a ce residu : mass_ref a ~50 fois son carre moyen laisse au terme
    # de masse un poids de ~0,02 sur la verite, comme dans la v7 (seuil DSO2 / mass_ref)
    bilan["mass_ref_conseille"] = float(f"{50 * bilan['residu_moyen_carre']:.1e}")
    bilan["mass_ref_config"] = cfg.loss.mass_ref
    cfg.paths.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.run_dir / "bilan_masse_verite.json").write_text(json.dumps(bilan, indent=2))
    print(json.dumps(bilan, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
