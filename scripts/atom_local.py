"""Met le modele v7 de CE poste en ligne a atom-origin.zelqane.com (tunnel nomme).

Variante sans Colab de notebooks/current/pcfno3d_atom_colab.ipynb : meme tunnel, meme cle
de lecture (enregistree par son empreinte), mais le modele tourne ici, sur CPU (15 a 25 s
par instant nouveau, puis memorise).

Les identifiants du tunnel (le JSON remis par Rayen) sont lus dans un FICHIER HORS DU DEPOT,
passe en argument ; ils ne vont qu'au processus cloudflared et ne sont jamais affiches.

    python scripts/atom_local.py C:/Users/inesr/AppData/Local/pcfno3d/atom_tunnel.json

Le script reste au premier plan tant que le service vit ; Ctrl+C arrete tout proprement.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

TUNNEL_ID = "e552f718-2478-4fe3-82b4-8d33a4b9eb6e"
HOTE = "atom-origin.zelqane.com"
CLE_NOM = "atom-worker"
CLE_EMPREINTE = "ce7965c54284eb6d1a274119839a71d9785731609d04b17bd08fb6da7d2c30fd"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("identifiants", type=Path, help="fichier JSON des identifiants du tunnel")
    ap.add_argument("-c", "--config", default="local_v7")
    ap.add_argument("--port", type=int, default=8001,
                    help="port local (8001 : n'interfere pas avec un serveur deja sur 8000)")
    ap.add_argument("--export", action="store_true",
                    help="exporter d'abord les volumes 3D des instants de test en cache")
    args = ap.parse_args()

    chemin = args.identifiants.resolve()
    if REPO in chemin.parents:
        print("refus : les identifiants ne doivent pas vivre dans le depot", file=sys.stderr)
        return 2
    identifiants = chemin.read_text(encoding="utf-8").strip()

    if args.export:
        from pcfno3d.config import Config
        from pcfno3d.export import export_split
        from pcfno3d.inference import WindFieldModel
        cfg = Config.load(args.config)
        export_split(WindFieldModel(cfg, verbose=False), split="test", stride=2)

    from service.colab import serve_public, stop
    info = serve_public(config=args.config, port=args.port, key_name=CLE_NOM, scope="read",
                        key_hash=CLE_EMPREINTE, wait=600.0,
                        log_path=REPO / "artifacts" / "atom_service.log",
                        named_tunnel={"id": TUNNEL_ID, "hostname": HOTE,
                                      "credentials": identifiants})
    del identifiants
    try:
        while info["tunnel"].alive and info["process"].poll() is None:
            time.sleep(30)
        print("le tunnel ou le service s'est arrete ; journaux :", info["log_path"],
              info["tunnel"].log_path, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        stop(info)


if __name__ == "__main__":
    raise SystemExit(main())
