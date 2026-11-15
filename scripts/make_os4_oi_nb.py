"""Carnet Colab CPU : OS4 national par interpolation optimale (regle configs/national/regle_os4_oi_national.json),
simulation sur la VALIDATION, sans GPU. Reprend les cellules Drive / code / cache du carnet de test v4.

    python scripts/make_os4_oi_nb.py --commit <sha>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "notebooks" / "current" / "pcfno3d_national_v4_test_colab.ipynb"
SORTIE = REPO / "notebooks" / "current" / "pcfno3d_national_os4_oi_colab.ipynb"
SHA = (REPO / "configs" / "national" / "regle_os4_oi_national.json.sha256").read_text(encoding="ascii").split()[0]

CALCUL = r'''import shlex, traceback
from google.colab import runtime
DEG = NAT / "runs" / "degradation_era5_v4.json"
SORTIE = NAT / "os4_oi"
assert DEG.exists(), DEG
try:
    if not (SORTIE / "bilan.json").exists():
        lancer("os4_oi_national", f"python -u scripts/os4_oi_national.py -c national_1km_colab "
                                  f"--degradation {shlex.quote(str(DEG))} --sortie {shlex.quote(str(SORTIE))}")
    print((SORTIE / "bilan.md").read_text())
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()                          # fin de la session : plus aucune unite consommee
'''


def cellule(kind, texte):
    c = {"cell_type": kind, "metadata": {}, "source": texte.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    prep, cache = ("".join(c["source"]) for c in code[1:3])
    ancien = prep.split('COMMIT = "')[1].split('"')[0]
    prep = prep.replace(ancien, a.commit).replace(
        'for f, h in ((', f'for f, h in (("regle_os4_oi_national.json", "{SHA}"), (', 1)
    cache = cache.replace('lancer("controle", "python -u -m pcfno3d -c national_1km_colab check")',
                          "# controle de la verite deja fait par les carnets d'entrainement")
    nb["cells"] = [
        cellule("markdown", "# PC-FNO3D NATIONAL — OS4 : dimensionner le réseau ANEO par interpolation optimale\n\n"
                f"Règle `configs/national/regle_os4_oi_national.json` (`{SHA[:12]}…`). **Simulation** sur la "
                "validation : fond = ERA5 dégradée (statistiques ERA5 − METAR), capteurs simulés = vérité + bruit, "
                "analyse = fond + OI ; N = 10, 25, 50, 100, 200 ; N minimal pour 10 % de gain. **Sans GPU, sans "
                "entraînement, 2022 non lu.** La session se termine seule à la fin.\n"),
        cellule("markdown", "## 1. Drive, code et données"), cellule("code", prep),
        cellule("markdown", "## 2. Cache v3"), cellule("code", cache),
        cellule("markdown", "## 3. Courbe OS4, puis fin de la session"), cellule("code", CALCUL),
    ]
    nb["metadata"].pop("accelerator", None)
    nb["metadata"].setdefault("colab", {}).pop("gpuType", None)
    SORTIE.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(SORTIE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
