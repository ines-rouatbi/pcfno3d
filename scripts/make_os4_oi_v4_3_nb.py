"""Carnet Colab CPU : OS4 national, regle configs/national/regle_os4_oi_v4_3.json (approuvee par Ines le 24/11/2026).
Reprend les cellules Drive / code / cache du carnet de test v4 national, comme le carnet de la v4_2.

    python scripts/make_os4_oi_v4_3_nb.py --commit <sha>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "notebooks" / "current" / "pcfno3d_national_v4_test_colab.ipynb"
SORTIE = REPO / "notebooks" / "current" / "pcfno3d_national_os4_oi_v4_3_colab.ipynb"
SHA = (REPO / "configs" / "national" / "regle_os4_oi_v4_3.json.sha256").read_text(encoding="ascii").split()[0]

CALCUL = r'''import shlex, traceback
from google.colab import runtime
DEG = NAT / "runs" / "degradation_era5_v4.json"
SORTIE = NAT / "os4_oi_v4_3"                    # dossier neuf ; os4_oi_v4_2/ n'est jamais touche
assert DEG.exists(), DEG
base = (f"python -u scripts/os4_oi_v4_3.py {{etape}} -c national_1km_colab --degradation {shlex.quote(str(DEG))} "
        f"--sortie {shlex.quote(str(SORTIE))}")
try:
    # 1a. essai du code de l'etape 1 sur 2 instants de validation (sortie a part, jamais lue ensuite)
    if not (SORTIE / "choix_validation_essai" / "choix.json").exists():
        lancer("os4_v4_3_choisir_essai", base.format(etape="choisir") + " --essai")
    # 1b. ETAPE 1 : choix du reglage d'OI sur TOUTE la validation 2021 (72 reglages, grille, N = 50 et 200)
    if not (SORTIE / "choix_validation" / "choix.json").exists():
        lancer("os4_v4_3_choisir", base.format(etape="choisir"))
    print((SORTIE / "choix_validation" / "choix.json").read_text())
    # 2a. essai du code du jugement sur 6 instants de VALIDATION (sortie os4_oi_v4_3_essai/)
    if not (Path(str(SORTIE) + "_essai") / "bilan.json").exists():
        lancer("os4_v4_3_juger_essai", base.format(etape="juger"))
    # 2b. ETAPE 2 : test 2022, deuxieme lecture pour l'OS4, non independante (dite dans la regle)
    if not (SORTIE / "bilan.json").exists():
        lancer("os4_v4_3_juger_test", base.format(etape="juger") + " --lecture-test")
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


def carnet(commit: str) -> dict:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    prep, cache = ("".join(c["source"]) for c in code[1:3])
    ancien = prep.split('COMMIT = "')[1].split('"')[0]
    prep = prep.replace(ancien, commit).replace(
        'for f, h in ((', f'for f, h in (("regle_os4_oi_v4_3.json", "{SHA}"), (', 1)
    cache = cache.replace('lancer("controle", "python -u -m pcfno3d -c national_1km_colab check")',
                          "# controle de la verite deja fait par les carnets d'entrainement")
    nb["cells"] = [
        cellule("markdown", "# PC-FNO3D NATIONAL — OS4 v4_3 : réseau ANEO (mâts à 25 m) par interpolation optimale 3D\n\n"
                f"Règle `configs/national/regle_os4_oi_v4_3.json` (`{SHA[:12]}…`, approuvée par Ines le 24/11/2026). "
                "**Expérience OSSE distincte de C'.** Capteurs simulés à **25 m** (vérité + bruit 0,5 m/s). "
                "Couche jugée **0–100 m** (RMSE vectorielle) : borne basse de l'IC 95 % du gain ≥ 10 % avec 50 capteurs ; "
                "10 m descriptif ; 0–1 km et pire tranche « information, non visée par des capteurs à 25 m ». "
                "Étape 1 : réglage de l'OI (forme, L, λ, H : 72 combinaisons) choisi sur **toute la validation 2021**. "
                "Étape 2 : **deuxième lecture du test 2022, non indépendante** (dite dans la règle). "
                "**CPU (High-RAM), sans GPU ni entraînement** ; la session se termine seule à la fin.\n"),
        cellule("markdown", "## 1. Drive, code et données"), cellule("code", prep),
        cellule("markdown", "## 2. Cache v3"), cellule("code", cache),
        cellule("markdown", "## 3. Étape 1 (validation), étape 2 (test), puis fin de la session"), cellule("code", CALCUL),
    ]
    nb["metadata"].pop("accelerator", None)
    nb["metadata"].setdefault("colab", {}).pop("gpuType", None)
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    SORTIE.write_text(json.dumps(carnet(a.commit), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(SORTIE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
