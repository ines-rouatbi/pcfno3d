"""Carnet Colab de l'OS7 NATIONAL (phase 4 de docs/feuille_de_route_nationale.md) : prevision du vent 3D a
+24 h et +48 h sur toute la Tunisie, le modele national v4 choisi (runs/choix_v4.json) nourri par la
prevision GFS archivee, puis affine (v5, configs/national_v5_gfs_colab.yaml).

    python scripts/make_national_os7_nb.py [--commit <sha>]
    -> notebooks/current/pcfno3d_national_os7_colab.ipynb

Un seul carnet, deux environnements (le carnet detecte le sien) :
  - partie A, Colab CPU (memoire elevee) : previsions GFS f24 / f48 aux instants du cache de la verite v3
    (pcfno3d.gfs --national : boite nationale, echeances du Grand Tunis relues sans y ecrire), forcage
    national ecrit sur le Drive, puis fin de la session ;
  - partie B, Colab A100 : reference sans affinage, affinage v5, PREV1 / PREV2 / PREV3 aux 12 aeroports,
    champ 3D a +24 h contre la verite d'entrainement, verdicts de la regle, puis fin de la session.
Le code est celui du depot public au commit fige (CODE_GIT de make_national_nb.py) ; le carnet verifie
l'empreinte de configs/national/regle_v4_national.json ET de configs/national/regle_os7_national.json.
La regle est une PROPOSITION : le carnet ne doit etre lance qu'avec l'accord ecrit d'Ines (et refuse de
tourner tant que ACCORD_ECRIT_INES est vide). Ne modifie pas make_national_nb.py : il en importe les blocs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_grand_tunis_nb import CHOIX_POIDS, carnet, code, md  # noqa: E402
from make_national_nb import CODE_GIT, COMMUN, MEM, _ent_v3  # noqa: E402

NAT_REGLES = REPO / "configs" / "national"
NOM_CARNET = "pcfno3d_national_os7_colab.ipynb"


def empreinte(nom: str) -> str:
    """sha256 de la regle, verifie contre son fichier .sha256 (refus si la regle a change depuis)."""
    f = NAT_REGLES / nom
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    attendu = (NAT_REGLES / (nom + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"{nom} modifiee depuis son hachage ({h} != {attendu}) : rehacher avant le carnet")
    return h


ACCORD = r'''
# ACCORD ECRIT D'INES : la regle configs/national/regle_os7_national.json est une PROPOSITION. Ce carnet ne
# telecharge, n'affine et ne lit le test qu'une fois l'accord ecrit recu ; le recopier ici (date et termes).
ACCORD_ECRIT_INES = ""
if not ACCORD_ECRIT_INES.strip():
    raise SystemExit("accord ecrit d'Ines absent : rien n'est lance (remplir ACCORD_ECRIT_INES)")
print("accord :", ACCORD_ECRIT_INES)
'''

INSTALLE = '''
# Herbie : telechargement partiel des fichiers GFS (index .idx) ; cfgrib / eccodes : lecture GRIB2
!pip install -q herbie-data cfgrib eccodes 2>&1 | tail -1
'''

GIT_OS7 = '''
import torch
GPU = torch.cuda.is_available()
OS7N = NAT / "os7"                         # sorties de l'OS7 national (rien dans les dossiers du Grand Tunis)
OS7N.mkdir(exist_ok=True)
F = {e: OS7N / f"forcage_gfs_f{e}.csv" for e in (24, 48)}
print("environnement :", "GPU " + torch.cuda.get_device_name(0) if GPU else "CPU", "| partie", "B" if GPU else "A")
'''

PARTIE_A = r'''
# PARTIE A (Colab CPU) : previsions GFS f24 / f48 aux instants du cache de la verite v3, forcage national
from google.colab import runtime
import traceback
if GPU:
    print("GPU : la partie A (telechargement GFS) se fait en CPU ; on passe a la partie B")
else:
    try:
        # instants : l'index du cache v3 (les 3 parties), sans extraire les ~50 Go du cache
        CACHE = Path("/content/national/cache")
        CACHE.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        parts = []
        for k in (1, 2, 3):
            with zipfile.ZipFile(NAT / f"cache_v3_partie_{k}.zip") as zf:
                with zf.open(f"index_partie_{k}.csv") as fi:
                    parts.append(pd.read_csv(fi))
        pd.concat(parts).drop_duplicates("stem").to_csv(CACHE / "index.csv", index=False)
        print(pd.read_csv(CACHE / "index.csv")["split"].value_counts().to_string())
        for e in (24, 48):
            if not F[e].exists():
                # boite nationale ; les echeances deja telechargees par l'OS7 du Grand Tunis sont relues sans
                # y ecrire ; code 4 = plus de 2 % d'echeances absentes de l'archive (regle) : arret
                lancer(f"gfs_national_f{e}", f"python -u -m pcfno3d.gfs --config national_1km_colab --national "
                                             f"--echeance {e} --sortie {F[e]} --reprise {OS7N / f'gfs_f{e}_tn'} "
                                             f"--lecture {GT / 'os7' / f'gfs_f{e}_instants'} --travailleurs 8")
        for e in (24, 48):
            err = F[e].with_suffix(".erreurs.json")
            print(f"f{e} :", F[e].exists(), "| echeances absentes :",
                  len(json.loads(err.read_text())) if err.exists() else 0)
        print("partie A terminee : relancer ce carnet en A100 pour la partie B")
    except Exception:
        traceback.print_exc()
    finally:
        runtime.unassign()
    raise SystemExit("fin de la partie A")
'''

PARTIE_B_PREP = r'''
# PARTIE B (Colab A100) : il faut les deux forcages GFS de la partie A
for e in (24, 48):
    if not F[e].exists():
        raise FileNotFoundError(f"{F[e]} absent : faire d'abord la partie A (Colab CPU)")
OBS = G / "obs"                             # METAR des 12 aeroports (national_donnees.zip)
if not OBS.exists():
    raise FileNotFoundError(f"{OBS} absent : les mesures des aeroports sont requises (PREV)")
'''

PARTIE_B = r'''
from google.colab import runtime
import traceback
CHOIX = NAT / "runs" / "choix_v4.json"
if not CHOIX.exists():
    raise FileNotFoundError(f"{CHOIX} absent : le carnet de test v4 choisit l'entrainement de depart")
choix = json.loads(CHOIX.read_text())
if choix.get("regle_sha256") != "@SHA_V4@":
    raise RuntimeError(f"choix_v4.json ne vient pas de la regle v4 verifiee ({choix.get('regle_sha256')})")
V4 = NAT / "runs" / f"national_1km_v4_{choix['entrainement_choisi']}"
V5_CONF = "national_v5_gfs_colab"
R5 = NAT / "runs" / "national_1km_v5_gfs"
EPOQUES = 12                               # configs/national_v5_gfs_colab.yaml (train.epochs)
print("depart (v4 choisi sur la validation) :", V4.name, "| run v5 :", R5)

def forcage(e):                            # le forcage GFS d'une echeance, la ou la configuration le lit
    return f"--set paths.forcage_era5={G / F[e].name}"

def tolerer_code3(nom, cmd):               # code 3 = critere non conforme : le resultat est ecrit
    try:
        lancer(nom, cmd)
    except RuntimeError as exc:
        if "(code 3)" not in str(exc):
            raise
        print(exc)

for e, f in F.items():
    for suffixe in (".csv", ".npz"):
        shutil.copy2(f.with_suffix(suffixe), G / f.with_suffix(suffixe).name)
try:
    INIT = shlex.quote(str(choisir_poids(V4)))
    # 1. reference : le v4 choisi nourri par GFS, SANS affinage (son run_dir lu seulement : statistiques)
    for e in (24, 48):
        d = OS7N / f"sans_affinage_f{e}"
        if not (d / "prevision_bilan.json").exists():
            tolerer_code3(f"prevision_sans_affinage_f{e}",
                          f"python -u scripts/evaluer_prevision.py -c {V5_CONF} {OPTS} {forcage(e)} --checkpoint {INIT} "
                          f"--set paths.run_dir={shlex.quote(str(V4))} --obs {OBS} --echeance {e} --sortie {d} "
                          f"--bootstrap 5000 --graine 2030")

    # 2. affinage v5 (GFS +24 h en entree, verite v3 inchangee, epoque choisie sur la validation)
    h = R5 / "history.json"
    if not (h.exists() and any(r.get("epoch") == EPOQUES - 1 for r in json.loads(h.read_text()))):
        lancer("entrainer_v5_national", f"python -u -m pcfno3d -c {V5_CONF} {OPTS} {forcage(24)} "
                                        f"--set train.init_from={INIT} train")

    # 3. PREV1 (+24 h), PREV2 (+48 h) aux 12 aeroports ; PREV3 : decisions de traitement de la veille
    for e in (24, 48):
        d = OS7N / f"v5_f{e}"
        if not (d / "prevision_bilan.json").exists():
            tolerer_code3(f"prevision_v5_f{e}", f"python -u scripts/evaluer_prevision.py -c {V5_CONF} {OPTS} {forcage(e)} "
                                                f"--obs {OBS} --echeance {e} --sortie {d} --bootstrap 5000 --graine 2030")
        if (d / "prevision_paires.csv").exists() and not (d / "decisions" / "decisions_agricoles.json").exists():
            lancer(f"prev3_f{e}", f"python -u scripts/verifier_decisions.py --paires {d / 'prevision_paires.csv'} "
                                  f"--sources prevision,gfs,persistance --sortie {d / 'decisions'}")

    # 4. information : le champ 3D prevu a +24 h contre la verite d'entrainement v3 (test lu une fois)
    if not (R5 / "eval_pcfno3d" / "best_wcont" / "test_summary.json").exists():
        lancer("evaluer_v5_national_f24", f"python -u -m pcfno3d -c {V5_CONF} {OPTS} {forcage(24)} evaluate")
    # (la configuration v5 lit forcage_gfs_f24.csv la ou il vient d'etre copie : la prevision a +24 h)
    if not (R5 / "dso_hauteur_sol" / "regle.json").exists():
        lancer("dso_sol_regle_v5", f"python -u scripts/dso_hauteur_sol.py -c {V5_CONF} regle")
    if not (R5 / "dso_hauteur_sol" / "bilan.json").exists():
        tolerer_code3("dso_sol_v5", f"python -u scripts/dso_hauteur_sol.py -c {V5_CONF} mesure")

    # 5. verdicts de la regle (empreinte verifiee, seuils recopies de la regle, rien de recalcule)
    lancer("juger_os7_national", f"python -u scripts/juger_os7_national.py --os7 {OS7N}")
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''


def os7_national(commit: str) -> dict:
    """Le carnet, au commit de code `commit` (doit etre pousse sur le depot public avant de lancer)."""
    sha_v4 = empreinte("regle_v4_national.json")
    sha_os7 = empreinte("regle_os7_national.json")
    ancien = '(("regle_v4_national.json", "@SHA_V4@"),)'
    if ancien not in CODE_GIT:
        raise SystemExit("CODE_GIT de make_national_nb.py a change : adapter la verification des regles")
    git = (CODE_GIT.replace(ancien, '(("regle_v4_national.json", "@SHA_V4@"), ("regle_os7_national.json", "@SHA_OS7@"))')
           .replace('"| regle v4 verifiee"', '"| regles v4 et OS7 national verifiees"')
           .replace("@COMMIT@", commit).replace("@SHA_V4@", sha_v4).replace("@SHA_OS7@", sha_os7))
    regle = json.loads((NAT_REGLES / "regle_os7_national.json").read_text(encoding="utf-8"))
    titre = f"""
# PC-FNO3D NATIONAL — OS7 : prévision du vent 3D à +24 h et +48 h (Colab CPU puis A100)

**NE PAS LANCER sans l'accord écrit d'Ines.** La règle `configs/national/regle_os7_national.json`
(sha256 `{sha_os7[:12]}…`, fixée le {regle["fixee_le"][:10]}) est une **PROPOSITION soumise à l'accord écrit
d'Ines** ; le carnet vérifie son empreinte et celle de la règle v4, et refuse de tourner tant que
`ACCORD_ECRIT_INES` est vide. Code : dépôt public au commit `{commit[:12]}` (à pousser avant de lancer).

Le modèle national **v4 choisi** (`runs/choix_v4.json`, choisi sur la validation), nourri par la **prévision
GFS** (NOAA, AWS, `noaa-gfs-bdp-pds`, GRIB2 0,25°, lecture par plages d'octets via `.idx`) au lieu d'ERA5 :
mêmes mailles, mêmes hauteurs au-dessus du sol et mêmes formules que le forçage ERA5 national, aucune
station en entrée. Vérité d'entraînement v3 et découpage inchangés.

| Partie | Environnement | Contenu |
|---|---|---|
| A | **Colab CPU** (mémoire élevée) | GFS f24 et f48 aux instants du cache v3 (boîte nationale ; les échéances de l'OS7 du Grand Tunis sont relues sans y écrire) → `MyDrive/pcfno3d_national/os7/forcage_gfs_f<L>.csv/.npz` ; environ 6 à 8 h, reprise par instant |
| B | **Colab A100** | référence sans affinage ; affinage **v5** (12 époques, GFS f24, époque choisie sur la validation, `runs/national_1km_v5_gfs`) ; **PREV1** (+24 h), **PREV2** (+48 h) aux 12 aéroports METAR contre GFS brut et la persistance ; **PREV3** (décision de traitement de la veille) ; champ 3D à +24 h contre la vérité d'entraînement (information) ; verdicts par `scripts/juger_os7_national.py` |

Un seul **Tout exécuter** par environnement : le carnet détecte le sien. Le test 2022 n'est lu qu'une fois
par modèle et par échéance (une étape dont la sortie existe est sautée). La session s'arrête à la fin de
chaque partie. GFS et la vérité générée sont des prévisions et une vérité d'entraînement, jamais des
mesures ; les mesures sont les METAR.
"""
    return carnet([
        md(titre),
        md("## 1. Accord, Drive, code au commit figé, données"),
        code(ACCORD + COMMUN + INSTALLE + git + GIT_OS7),
        md("## 2. Partie A (Colab CPU) : prévisions GFS nationales, puis fin de la session"),
        code(PARTIE_A),
        md("## 3. Partie B (Colab A100) : cache v3 et contrôle de la vérité"),
        code(PARTIE_B_PREP + _ent_v3()),
        md("## 4. Mémoire du GPU"),
        code(MEM),
        md("## 5. Référence, affinage v5, PREV1-3, champ 3D, verdicts, fin de la session"),
        code(CHOIX_POIDS + PARTIE_B.replace("@SHA_V4@", sha_v4)),
    ], gpu=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", help="commit du code (defaut : HEAD)")
    args = ap.parse_args()
    commit = args.commit or subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                           text=True, check=True).stdout.strip()
    f = REPO / "notebooks" / "current" / NOM_CARNET
    f.write_text(json.dumps(os7_national(commit), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
