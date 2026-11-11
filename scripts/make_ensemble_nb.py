"""Carnet Colab de la MOYENNE DES GRAINES de C' (v4 nationale), regle configs/national/regle_ensemble_cp.json
(PROPOSITION soumise a l'accord ecrit d'Ines) : validation seulement, aucun entrainement, aucune lecture du test.

    python scripts/make_ensemble_nb.py [--commit <sha>]
    -> notebooks/current/pcfno3d_national_ensemble_cp_colab.ipynb   (A100)

Le code vient du depot public au commit fige (par defaut HEAD : le commit doit etre pousse avant de lancer
le carnet) ; l'empreinte de la regle est inscrite dans le carnet et verifiee avant tout calcul. Reprend les
conventions de scripts/make_national_nb.py (COMMUN, CODE_GIT, cache v3, lancer, choisir_poids).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_grand_tunis_nb import CHOIX_POIDS, carnet, code, md  # noqa: E402
from make_national_nb import CODE_GIT, COMMUN, _ent_v3  # noqa: E402

REGLE = REPO / "configs" / "national" / "regle_ensemble_cp.json"
NOM = "pcfno3d_national_ensemble_cp_colab.ipynb"

VERIF = '''
# regle de la moyenne des graines (PROPOSITION tant qu'Ines ne l'a pas validee) : empreinte inscrite ici
from pcfno3d.regles import verifier
REGLE = CODE / "configs" / "national" / "regle_ensemble_cp.json"
regle, sha_ens = verifier(REGLE)
assert sha_ens == "@SHA_ENS@", f"regle de la moyenne modifiee depuis la generation du carnet : {sha_ens}"
print("regle de la moyenne verifiee :", sha_ens[:12], "|", regle["statut"][:120])
'''

GPU = r'''
import torch
assert torch.cuda.is_available(), "pas de GPU : choisir A100 dans Execution > Modifier le type d'execution"
nom_gpu = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"{nom_gpu}, {total:.0f} Go")
if "A100" not in nom_gpu:
    print("ATTENTION : le DSO4 de la regle se mesure sur A100 ; sur un autre GPU, le temps n'est qu'indicatif")
# estimation (regle_ensemble_cp.json, non mesuree) : 3 x 1,81 Go de poids + une passe ~5-7 Go ; le script
# ecrit la memoire mesuree (dso4_moyenne.json)
'''

ENS = r'''
import shlex
import traceback
from google.colab import runtime
P = regle["parametres"]
RUNS = NAT / "runs"
SORTIE = RUNS / P["sortie"]
try:
    if (SORTIE / "bilan_validation.json").exists():
        print("deja calcule (rien n'est recalcule) :", SORTIE)
    else:
        CKPTS = " ".join(shlex.quote(str(choisir_poids(RUNS / f"{P['prefixe']}_s{g}"))) for g in P["graines"])
        try:
            # validation seulement : la moyenne, chaque graine, l'ERA5 seule ; DSO4 ; dispersion (DSO6)
            lancer("ensemble_graines_cp",
                   f"python -u scripts/ensemble_graines.py -c national_1km_colab --runs {shlex.quote(str(RUNS))} "
                   f"--regle {shlex.quote(str(REGLE))} --checkpoints {CKPTS}")
        except RuntimeError as exc:                 # code 3 = criteres non remplis : bilan ecrit
            if "(code 3)" not in str(exc):
                raise
            print(exc)
    print((SORTIE / "bilan_validation.md").read_text(encoding="utf-8")[:8000])
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''


def ensemble_cp(commit: str) -> dict:
    """Le carnet : code au commit fige, regles verifiees, cache v3, GPU, puis la moyenne et fin de session."""
    sha_v4 = (REPO / "configs" / "national" / "regle_v4_national.json.sha256").read_text(encoding="ascii").split()[0]
    sha = (REPO / "configs" / "national" / "regle_ensemble_cp.json.sha256").read_text(encoding="ascii").split()[0]
    regle = json.loads(REGLE.read_text(encoding="utf-8"))
    P = regle["parametres"]
    git = CODE_GIT.replace("@COMMIT@", commit).replace("@SHA_V4@", sha_v4)
    graines = ", ".join(str(g) for g in P["graines"])
    titre = f"""
# PC-FNO3D NATIONAL v4 — moyenne des graines de C' (validation seulement) (Colab, A100)

**À lancer uniquement avec l'accord écrit d'Ines.** Règle `configs/national/regle_ensemble_cp.json`
(sha256 `{sha[:12]}…`, statut : PROPOSITION soumise à l'accord écrit d'Ines), vérifiée par le carnet avant
tout calcul ; code du dépôt au commit figé `{commit[:12]}`. **Aucun entraînement, aucune lecture du test.**

La règle v4 (`regle_v4_national.json`, `{sha_v4[:12]}…`) choisit UN entraînement et ne prévoit pas
d'ensemble : cette règle s'y ajoute sans la modifier.

1. **modèle** : moyenne des graines {graines} de C' (`runs/{P["prefixe"]}_s<graine>`) — u, v moyens, w par
   continuité ; les run_dir des graines ne sont que lus ;
2. **validation** ({P["n_val"]} instants de la sélection d'entraînement, vérité d'entraînement v3) : tranches
   0–1 km au-dessus du sol pour la moyenne, chaque graine, la meilleure graine (val_score, fixée avant le
   calcul) et l'ERA5 seule, entrée complète puis 25 % des points d'entrée (DSO5) ;
3. **DSO4** : temps de la passe servie de la moyenne (3 passes sur l'A100), seuil 1 s ;
4. **DSO6** (estimation de validation, pas un verdict) : dispersion entre graines, calibrée sur la
   validation (2 plis de jours), couverture 80 % hors pli et Spearman ;
5. **bilan** : `MyDrive/pcfno3d_national/runs/{P["sortie"]}/` (dossier neuf ; une reprise relit le bilan).

Le test de la moyenne ne serait lu qu'avec un **accord écrit séparé d'Ines**, après lecture de ce bilan.
Ressources estimées (non mesurées) : ~12–14 Go de GPU sur 40, ~20–30 min. La session s'arrête à la fin.
"""
    return carnet([
        md(titre),
        md("## 1. Drive, code au commit figé, règles vérifiées, données"),
        code(COMMUN + git + VERIF.replace("@SHA_ENS@", sha)),
        md("## 2. Cache de la vérité v3"),
        code(_ent_v3()),
        md("## 3. GPU"),
        code(GPU),
        md("## 4. Moyenne des graines sur la validation, DSO4, dispersion — puis fin de la session"),
        code(CHOIX_POIDS + ENS),
    ], gpu=True, gpu_type="A100")


def main() -> int:
    if "--commit" in sys.argv:
        commit = sys.argv[sys.argv.index("--commit") + 1]
    else:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                                check=True).stdout.strip()
    f = REPO / "notebooks" / "current" / NOM
    f.write_text(json.dumps(ensemble_cp(commit), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
