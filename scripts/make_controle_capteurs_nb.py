"""Carnets Colab du controle de branchement des capteurs (demande d'Ines, 11/11) — AUCUN entrainement.

    python scripts/make_controle_capteurs_nb.py
ecrit notebooks/current/pcfno3d_controle_capteurs_national_colab.ipynb (C' et D graine 42, controles 1 a 3)
et notebooks/current/pcfno3d_controle_capteurs_grand_tunis_colab.ipynb (Grand Tunis v4, controle 4), au commit
courant (HEAD, a pousser avant).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts")]
import make_grand_tunis_nb as GTN  # noqa: E402
import make_national_nb as NN  # noqa: E402
from make_grand_tunis_nb import carnet, code, md  # noqa: E402

NATIONAL = r'''
import shlex
from google.colab import runtime
import traceback
DEG = NAT / "runs" / "degradation_era5_v4.json"
try:
    for cand in ("Cp", "D"):
        RUN = NAT / "runs" / f"national_1km_v4_{cand}_s42"
        extra = ""
        if cand == "D":                    # controle 3 : lots d'entrainement de D (memes reglages que l'entrainement)
            extra = (f"--set mask.aneo_degradation={shlex.quote(str(DEG))} --set mask.aneo_p_zone=0.5 "
                     f"--set {shlex.quote('mask.aneo_capteurs=[20,200]')} --set mask.aneo_hauteur_capteur_m=25 "
                     "--entrainement --lots 5")
        lancer(f"controle_capteurs_{cand}_s42",
               f"python -u scripts/controle_capteurs.py -c national_1km_colab {OPTS} --run {shlex.quote(str(RUN))} "
               f"--instants 3 --n-capteurs 50 --delta 15 --rayon-km 5 {extra}")
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''

GRAND_TUNIS = r'''
import shlex
from google.colab import runtime
import traceback
RUN = GT / "runs" / "grand_tunis_v4_dso"
try:
    lancer("controle_capteurs_gt_v4",
           f"python -u scripts/controle_capteurs.py -c grand_tunis_v4_dso_colab --run {shlex.quote(str(RUN))} "
           "--set inference.w_continuite=true --instants 3 --n-capteurs 50 --delta 15 --rayon-km 5")
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''


def main() -> int:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    sha = (REPO / "configs" / "national" / "regle_v4_national.json.sha256").read_text(encoding="ascii").split()[0]
    git = NN.CODE_GIT.replace("@COMMIT@", commit).replace("@SHA_V4@", sha)
    tete = ("Contrôle demandé par Ines le 11/11 : **aucun entraînement, aucune lecture du test**. Test absurde "
            "(+15 m/s sur u aux 50 capteurs simulés et, en témoin, aux aéroports), chemin des capteurs jusqu'au "
            "tenseur d'entrée du réseau, et (national, D) capteurs présents dans 5 lots d'entraînement. "
            "Sortie : `<run>/controle_capteurs.json`, journaux `journaux/controle_capteurs_*.log`.")
    nat = carnet([
        md("# PC-FNO3D NATIONAL — contrôle du branchement des capteurs, C' et D graine 42 (Colab, A100)\n\n" + tete),
        md("## 1. Drive, code et données"), code(NN.COMMUN + git),
        md("## 2. Cache v3"), code(NN._ent_v3()),
        md("## 3. Mémoire du GPU"), code(NN.MEM),
        md("## 4. Contrôles, puis fin de la session"), code(NATIONAL)], gpu=True)
    gt = carnet([
        md("# PC-FNO3D Grand Tunis v4 — contrôle du branchement des capteurs (chemin de l'OS6) (Colab, A100)\n\n"
           + tete),
        md("## 1. Drive, code et données"), code(GTN.COMMUN + git),
        md("## 2. Cache"), code(GTN.ENT.replace("grand_tunis_colab", "grand_tunis_v4_dso_colab")),
        md("## 3. Contrôle, puis fin de la session"), code(GRAND_TUNIS)], gpu=True)
    for nom, nb in (("pcfno3d_controle_capteurs_national_colab.ipynb", nat),
                    ("pcfno3d_controle_capteurs_grand_tunis_colab.ipynb", gt)):
        f = REPO / "notebooks" / "current" / nom
        f.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f, "| code", commit[:7])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
