"""Carnet Colab CPU : OS4 national par interpolation optimale (regle configs/national/regle_os4_oi_v4_2.json),
simulation sur la VALIDATION, sans GPU. Reprend les cellules Drive / code / cache du carnet de test v4.

    python scripts/make_os4_oi_nb.py --commit <sha>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "notebooks" / "current" / "pcfno3d_national_v4_test_colab.ipynb"
SORTIE = REPO / "notebooks" / "current" / "pcfno3d_national_os4_oi_v4_colab.ipynb"
SHA = (REPO / "configs" / "national" / "regle_os4_oi_v4_2.json.sha256").read_text(encoding="ascii").split()[0]

CALCUL = r'''import shlex, traceback
from google.colab import runtime
DEG = NAT / "runs" / "degradation_era5_v4.json"
SORTIE = NAT / "os4_oi_v4_2"
assert DEG.exists(), DEG
base = (f"python -u scripts/os4_oi_v4.py -c national_1km_colab --degradation {shlex.quote(str(DEG))} "
        f"--sortie {shlex.quote(str(SORTIE))}")
try:
    # 1. essai du code sur 6 instants de VALIDATION (aucun parametre ne change apres) ; un echec arrete tout
    if not (Path(str(SORTIE) + "_essai") / "bilan.json").exists():
        lancer("os4_oi_v4_essai", base + " --max-instants 6")
    # 2. LECTURE UNIQUE des 180 instants de test 2022 (experience OSSE distincte de C')
    if not (SORTIE / "bilan.json").exists():
        lancer("os4_oi_v4_test", base + " --lecture-test")
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
        'for f, h in ((', f'for f, h in (("regle_os4_oi_v4_2.json", "{SHA}"), (', 1)
    cache = cache.replace('lancer("controle", "python -u -m pcfno3d -c national_1km_colab check")',
                          "# controle de la verite deja fait par les carnets d'entrainement")
    nb["cells"] = [
        cellule("markdown", "# PC-FNO3D NATIONAL — OS4 : dimensionner le réseau ANEO par interpolation optimale 3D\n\n"
                f"Règle `configs/national/regle_os4_oi_v4_2.json` (`{SHA[:12]}…`, approuvée par Ines le 16/11). "
                "**Expérience OSSE distincte de C'** : fond = ERA5 dégradée (σu 1,91, σv 2,02 m/s, L 55 km), "
                "capteurs simulés à 25 m (vérité + bruit 0,5 m/s), analyse = fond + OI 3D (L 150 km, λ 4, H 300 m) ; "
                "N = 10 à 400 ; placements glouton, grille régulière, aléatoire × 3. D'abord un **essai du code** sur "
                "6 instants de validation (aucun paramètre ne change ensuite), puis la **lecture unique des 180 instants "
                "de test 2022**. **CPU (High-RAM), sans entraînement.** La session se termine seule à la fin.\n"),
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
