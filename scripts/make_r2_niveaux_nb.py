"""Carnet Colab : R2 par niveau (avant / apres post-traitement) et DSO4 des entrainements C' finis, sur la
VALIDATION (regle configs/national/regle_postraitement_r2_v4.json). Inference seulement : une L4 suffit.
Reprend les cellules Drive / code / cache du carnet d'entrainement v4 (meme preparation), sans entrainement.

    python scripts/make_r2_niveaux_nb.py --commit <sha>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "notebooks" / "current" / "pcfno3d_national_v4_Cp_s44_colab.ipynb"
SORTIE = REPO / "notebooks" / "current" / "pcfno3d_national_v4_r2_niveaux_colab.ipynb"
SHA_ENS = open(REPO / "configs" / "national" / "regle_ensemble_cp.json.sha256").read().split()[0]
SHA_REGLE = "c82dff466abcd84e6d8d21f34037b4162d4428ee4c460a518193108ee49ea2a1"


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
                        f'for f, h in (("regle_postraitement_r2_v4.json", "{SHA_REGLE}"), '
                        f'("regle_ensemble_cp.json", "{SHA_ENS}"), ("regle_v4_national.json", ')
    cache = "".join(src[6]["source"]).replace('lancer("controle", "python -u -m pcfno3d -c national_1km_colab check")',
                                              "# controle de la verite deja fait par les carnets d'entrainement")
    calcul = '''import os, shlex
from google.colab import runtime
import traceback
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
RUNS = NAT / "runs"
try:
    for g in (42, 43, 44):
        run = RUNS / f"national_1km_v4_Cp_s{g}"
        if not (run / "best.pt").exists() or not (run / "history.json").exists():
            print("pas fini, ignore :", run.name)
            continue
        if (run / "r2_niveaux_validation" / "bilan.json").exists():
            print("deja calcule :", run.name)
            continue
        lancer(f"r2_niveaux_Cp_s{g}", f"python -u scripts/r2_niveaux_v4.py -c national_1km_colab "
               f"--run {shlex.quote(str(run))} --regle configs/national/regle_postraitement_r2_v4.json")
        print((run / "r2_niveaux_validation" / "bilan.md").read_text()[:6000])
    # moyenne des 3 graines de C' (regle_ensemble_cp, accord d'Ines du 13/11) : validation, DSO4, DSO6
    ENS = RUNS / "national_1km_v4_Cp_ens"
    if (ENS / "bilan_validation.json").exists():
        print("moyenne deja calculee :", ENS)
    elif all((RUNS / f"national_1km_v4_Cp_s{g}" / "best.pt").exists() for g in (42, 43, 44)):
        try:
            lancer("ensemble_graines_cp", f"python -u scripts/ensemble_graines.py -c national_1km_colab "
                   f"--runs {shlex.quote(str(RUNS))} --regle configs/national/regle_ensemble_cp.json")
        except RuntimeError as exc:                 # code 3 = criteres non remplis : bilan ecrit
            if "(code 3)" not in str(exc):
                raise
            print(exc)
        print((ENS / "bilan_validation.md").read_text(encoding="utf-8")[:8000])
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''
    cells = [
        cellule("markdown", "# PC-FNO3D NATIONAL v4 — R² par niveau et DSO4 (validation, inférence seulement)\n\n"
                f"Règle `configs/national/regle_postraitement_r2_v4.json` (sha256 `{SHA_REGLE[:12]}…`), vérifiée par "
                "le carnet. Entraînements C' finis (graines 42, 43, 44 si présente). **Aucun entraînement, aucune "
                "lecture du test, aucun bruit ajouté.** Une L4 suffit (inférence). Vérité d'entraînement v3, "
                "jamais appelée « mesures ».\n"),
        cellule("code", "import subprocess as _sp\nprint(_sp.run(['nvidia-smi'], capture_output=True, text=True).stdout)\n"),
        cellule("markdown", "## 1. Drive, code et données\n"), cellule("code", prep),
        cellule("markdown", "## 2. Cache v3\n"), cellule("code", cache),
        cellule("markdown", "## 3. R² par niveau, post-traitement, DSO4 ; moyenne des 3 graines (DSO6) ; fin de la session\n"), cellule("code", calcul),
    ]
    nb["cells"] = cells
    SORTIE.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(SORTIE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
