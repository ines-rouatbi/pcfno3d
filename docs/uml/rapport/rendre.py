"""Rend les .puml de ce dossier en PNG par le serveur PlantUML en ligne (https://www.plantuml.com/plantuml/png/...).
Seul le texte du diagramme est envoye (encodage PlantUML : deflate + base64 propre a PlantUML).
    python docs/uml/rapport/rendre.py [dossier_png]
"""
import sys
import urllib.request
import zlib
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = Path(sys.argv[1]) if len(sys.argv) > 1 else ICI
ALPHA = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"


def encoder(texte: str) -> str:
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    data = c.compress(texte.encode("utf-8")) + c.flush()
    out = []
    for i in range(0, len(data), 3):
        b = data[i:i + 3] + b"\0" * (3 - len(data[i:i + 3]))
        n = (b[0] << 16) | (b[1] << 8) | b[2]
        out.append("".join(ALPHA[(n >> s) & 0x3F] for s in (18, 12, 6, 0)))
    return "".join(out)


SORTIE.mkdir(parents=True, exist_ok=True)
for f in sorted(ICI.glob("*.puml")):
    url = "https://www.plantuml.com/plantuml/png/" + encoder(f.read_text(encoding="utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": "pcfno3d-rapport/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        png = r.read()
    assert png[:8] == b"\x89PNG\r\n\x1a\n", f.name
    (SORTIE / (f.stem + ".png")).write_bytes(png)
    print(f.stem, len(png), "octets")
