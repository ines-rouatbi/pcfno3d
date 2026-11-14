"""Carnet Colab : w servi (melange w par continuite / w du reseau) des entrainements C' finis, choisi sur la
VALIDATION (regle configs/national/regle_w_servi_v4.json) : R2 u, v, w par niveau et DSO2 pour chaque beta.
Inference seulement : une L4 suffit. Reprend les cellules Drive / code / cache du carnet d'entrainement v4.

    python scripts/make_w_servi_nb.py --commit <sha>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "notebooks" / "current" / "pcfno3d_national_v4_Cp_s44_colab.ipynb"
SORTIE = REPO / "notebooks" / "current" / "pcfno3d_national_v4_w_servi_colab.ipynb"
SHA_REGLE = open(REPO / "configs" / "national" / "regle_w_servi_v4.json.sha256").read().split()[0]


def cellule(kind: str, texte: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": texte.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    src = nb["cells"]
    prep = "".join(src[4]["source"])
    ancien = prep.split('COMMIT = "')[1].split('"')[0]
    prep = prep.replace(ancien, a.commit)
    prep = prep.replace('for f, h in (("regle_v4_national.json", ',
                        f'for f, h in (("regle_w_servi_v4.json", "{SHA_REGLE}"), ("regle_v4_national.json", ')
    cache = "".join(src[6]["source"]).replace('lancer("controle", "python -u -m pcfno3d -c national_1km_colab check")',
                                              "# controle de la verite deja fait par les carnets d'entrainement")
    calcul = '''import os, shlex
from google.colab import runtime
import traceback
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
RUNS = NAT / "runs"
try:
    for g in (44, 42, 43):                        # s44 d'abord : la meilleure graine (val_score)
        run = RUNS / f"national_1km_v4_Cp_s{g}"
        if not (run / "best.pt").exists():
            print("pas fini, ignore :", run.name)
            continue
        if (run / "w_servi_validation" / "bilan.json").exists():
            print("deja calcule :", run.name)
            print((run / "w_servi_validation" / "bilan.md").read_text()[:3000])
            continue
        lancer(f"w_servi_Cp_s{g}", f"python -u scripts/w_servi_v4.py -c national_1km_colab "
               f"--run {shlex.quote(str(run))} --regle configs/national/regle_w_servi_v4.json")
        print((run / "w_servi_validation" / "bilan.md").read_text()[:6000])
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''
    cells = [
        cellule("markdown", "# PC-FNO3D NATIONAL v4 — w servi : R² par niveau et DSO2 (validation, inférence seulement)\n\n"
                f"Règle `configs/national/regle_w_servi_v4.json` (sha256 `{SHA_REGLE[:12]}…`), vérifiée par "
                "le carnet. w = w_continuite + β (w_réseau − w_continuite), β choisi sur la validation. "
                "Entraînements C' finis (44, 42, 43). **Aucun entraînement, aucune "
                "lecture du test, aucun bruit ajouté.** Une L4 suffit (inférence). Vérité d'entraînement v3, "
                "jamais appelée « mesures ».\n"),
        cellule("code", "import subprocess as _sp\nprint(_sp.run(['nvidia-smi'], capture_output=True, text=True).stdout)\n"),
        cellule("markdown", "## 1. Drive, code et données\n"), cellule("code", prep),
        cellule("markdown", "## 2. Cache v3\n"), cellule("code", cache),
        cellule("markdown", "## 3. w servi : R² par niveau et DSO2 pour chaque β ; fin de la session\n"), cellule("code", calcul),
    ]
    nb["cells"] = cells
    SORTIE.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(SORTIE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
