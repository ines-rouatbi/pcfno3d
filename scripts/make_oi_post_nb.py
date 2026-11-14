"""Carnet Colab : C' s44 + OI des innovations des aeroports (regle configs/national/regle_oi_post_v4.json), VALIDATION
2021 seulement (aucune lecture de 2022). Reprend les cellules GPU / Drive / cache / memoire du carnet de test v4.

    python scripts/make_oi_post_nb.py --commit <sha>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "notebooks" / "current" / "pcfno3d_national_v4_test_colab.ipynb"
SORTIE = REPO / "notebooks" / "current" / "pcfno3d_national_v4_oi_post_colab.ipynb"
SHA = (REPO / "configs" / "national" / "regle_oi_post_v4.json.sha256").read_text(encoding="ascii").split()[0]


def cellule(kind: str, texte: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": texte.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


CALCUL = r'''import shlex, yaml, traceback
from google.colab import runtime
RUN = NAT / "runs" / "national_1km_v4_Cp_s44"          # entrainement fixe par regle_choix_v4_Cp_s44
c = yaml.safe_load(open(CODE / "configs" / "national_1km_colab.yaml"))
c["name"] = "national_v4_eval"
c["paths"]["run_dir"], c["paths"]["local_dir"] = str(RUN), "/content/run_local_v4_sCp_s44"
yaml.safe_dump(c, open(CODE / "configs" / "national_v4_eval.yaml", "w"), sort_keys=False)
try:
    # VALIDATION 2021 seulement : chaque aeroport retire de l'entree quand il est juge (aucune lecture de 2022)
    if not (RUN / "validation_reelle" / "val" / "stations_paires.csv").exists():
        lancer("validation_reelle_v4_validation", f"python -u scripts/valider_grand_tunis_reel.py -c national_v4_eval "
                                                  f"--obs {G / 'obs'} --jeu val")
    lancer("oi_post_v4_validation", f"python -u scripts/oi_post_aeroports.py --run {shlex.quote(str(RUN))} --jeu val")
    print((RUN / "validation_reelle" / "val" / "oi_post" / "tableau.txt").read_text())
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    src = nb["cells"]
    code = [c for c in src if c["cell_type"] == "code"]
    garde, prep, cache, mem = ("".join(c["source"]) for c in code[:4])
    ancien = prep.split('COMMIT = "')[1].split('"')[0]
    prep = prep.replace(ancien, a.commit).replace(
        'for f, h in ((', f'for f, h in (("regle_oi_post_v4.json", "{SHA}"), (', 1)
    cache = cache.replace('lancer("controle", "python -u -m pcfno3d -c national_1km_colab check")',
                          "# controle de la verite deja fait par les carnets d'entrainement")
    cells = [
        cellule("markdown", "# PC-FNO3D NATIONAL v4 — C' s44 + OI des innovations des aéroports (VALIDATION 2021)\n\n"
                f"Règle `configs/national/regle_oi_post_v4.json` (sha256 `{SHA[:12]}…`), vérifiée par le carnet. "
                "Post-traitement sans réentraînement : vent à 10 m = modèle + interpolation optimale des écarts "
                "(mesure − modèle) des **autres** aéroports (L 150 km, λ 4, fixés sur 2015–2020). Chaque aéroport est "
                "jugé retiré. **Aucune lecture de 2022, aucun entraînement, aucun bruit ajouté.**\n"),
        cellule("markdown", "## 0. GPU"), cellule("code", garde),
        cellule("markdown", "## 1. Drive, code et données"), cellule("code", prep),
        cellule("markdown", "## 2. Cache v3"), cellule("code", cache),
        cellule("markdown", "## 3. Mémoire du GPU"), cellule("code", mem),
        cellule("markdown", "## 4. Aéroports (validation, aéroport retiré), puis OI ; fin de la session"),
        cellule("code", CALCUL),
    ]
    nb["cells"] = cells
    SORTIE.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(SORTIE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
