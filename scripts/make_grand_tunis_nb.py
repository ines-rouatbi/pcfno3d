"""Fabrique les carnets Colab du modele reel (Grand Tunis).

    python scripts/make_grand_tunis_nb.py
    -> notebooks/current/pcfno3d_grand_tunis_generation_colab.ipynb  (jeu, environnement CPU)
       notebooks/current/pcfno3d_grand_tunis_colab.ipynb             (entrainement, A100)
       notebooks/current/pcfno3d_grand_tunis_diagnostic_colab.ipynb  (diagnostic sur la
           validation, CPU : scripts/diag_grand_tunis.py)
       notebooks/current/pcfno3d_grand_tunis_v2_colab.ipynb       (v2 mode reel, A100)
       notebooks/current/pcfno3d_grand_tunis_v2_aneo_colab.ipynb  (v2 mode ANEO simule, A100)
       notebooks/current/pcfno3d_grand_tunis_validation_colab.ipynb (vraies mesures, A100)

Tout passe par le Drive, dossier MyDrive/pcfno3d_grand_tunis/ :
    pcfno3d_code.zip           code (python scripts/package_for_colab.py)
    grand_tunis_donnees.zip    donnees (python scripts/package_grand_tunis_colab.py)
    best_dso_v7.pt             poids v7 (depart de l'entrainement)
Le jeu (1 000 instants) est genere en 3 parties ; chaque partie terminee est un point de reprise
sur le Drive (cache_partie_<k>.zip), 3 au plus.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def md(texte: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texte.strip("\n").splitlines(True)}


def code(texte: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": texte.strip("\n").splitlines(True)}


def carnet(cellules: list[dict], gpu: bool, gpu_type: str = "A100") -> dict:
    meta = {"kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}, "colab": {"provenance": []}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = gpu_type
    else:
        meta["colab"]["machine_shape"] = "hm"         # CPU, memoire RAM elevee
    return {"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": cellules}


COMMUN = r'''
from google.colab import drive
import os as _os
if not _os.path.isdir("/content/drive/MyDrive"):   # deja monte (panneau Fichiers) : ne pas redemander
    drive.mount("/content/drive")

import json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

GT = Path("/content/drive/MyDrive/pcfno3d_grand_tunis")
GT.mkdir(parents=True, exist_ok=True)
JOURNAUX = GT / "journaux"
JOURNAUX.mkdir(exist_ok=True)

# code : archive deposee sur le Drive depuis ines-pc
CODE = Path("/content/PFE")
if CODE.exists():
    shutil.rmtree(CODE)
CODE.mkdir(parents=True)
with zipfile.ZipFile(GT / "pcfno3d_code.zip") as zf:
    zf.extractall(CODE)
tampon = CODE / "PCFNO3D_VERSION.txt"
print(tampon.read_text().strip() if tampon.exists() else "tampon : absent")
%cd /content/PFE
!pip install -q --no-deps -e . 2>&1 | tail -2
!pip install -q pandera xarray netCDF4 h5netcdf 2>&1 | tail -2
sys.path[:0] = ["/content/PFE", "/content/PFE/src"]

def lancer(nom: str, commande: str) -> None:
    """Commande obligatoire ; sortie affichee ET journalisee sur le Drive ; arret si echec."""
    t0 = time.time()
    with open(JOURNAUX / f"{nom}.log", "a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} : {commande}\n")
        p = subprocess.Popen(commande, shell=True, cwd="/content/PFE", stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for ligne in p.stdout:
            print(ligne, end="")
            log.write(ligne)
            log.flush()
        p.wait()
    print(f"[{nom}] {time.time() - t0:.0f} s, code {p.returncode}")
    if p.returncode:
        raise RuntimeError(f"{nom} a echoue (code {p.returncode}) : voir {JOURNAUX / (nom + '.log')}")

# donnees : grille reelle, forcage ERA5, stations (une archive sur le Drive)
DON = Path("/content/donnees")
if not (DON / "grand_tunis" / "forcage_era5.csv").exists():
    with zipfile.ZipFile(GT / "grand_tunis_donnees.zip") as zf:
        zf.extractall(DON)
# mises a jour deposees apres coup sur le Drive (morceaux de moins de 10 Mo) :
# maj_<fichier> ou maj_<fichier>.000, .001... -> remplacent <fichier> dans donnees/grand_tunis
import collections
morceaux = collections.defaultdict(list)
for f in sorted(GT.glob("maj_*")):
    nom = f.name[4:]
    tete, _, queue = nom.rpartition(".")
    morceaux[tete if queue.isdigit() and tete else nom].append(f)
for nom, fs in morceaux.items():
    with open(DON / "grand_tunis" / nom, "wb") as out:
        for f in sorted(fs):
            out.write(f.read_bytes())
    print("mise a jour :", nom, "<-", [f.name for f in sorted(fs)])
print(sorted(p.name for p in (DON / "grand_tunis").iterdir()))
'''

GEN = r'''
TEST = "2022-01-01:2022-03-31,2022-07-01:2022-08-31"
VAL = "2021-01-01:2021-02-28,2021-07-01:2021-07-31"
LOC = Path("/content/grand_tunis")
JEU = LOC / "jeu"
JEU.mkdir(parents=True, exist_ok=True)
G = DON / "grand_tunis"
INFO = json.loads((G / "grille_reelle.json").read_text(encoding="utf-8"))

# calendrier fige : cree au premier passage, garde sur le Drive, relu ensuite
if (GT / "calendrier.json").exists():
    shutil.copy2(GT / "calendrier.json", JEU / "calendrier.json")

PARTIES = {1: [("train", "1/2")], 2: [("train", "2/2"), ("val", None)], 3: [("test", None)]}
faites = [k for k in PARTIES if (GT / f"cache_partie_{k}.zip").exists()]
PARTIE = next((k for k in PARTIES if k not in faites), None)
print("parties deja sur le Drive :", faites, "| a faire maintenant :", PARTIE)
'''

GEN_RUN = r'''
if PARTIE is None:
    print("les 3 parties sont sur le Drive : passer au carnet d'entrainement")
else:
    cache = LOC / f"cache_p{PARTIE}"
    if cache.exists():
        shutil.rmtree(cache)                 # partie interrompue : on la refait entiere
    nw = max(1, (os.cpu_count() or 2) - 1)
    base = (f"python -u scripts/generate_aneo_3d.py --input-grid {G / 'aneo_grid_points_surface.csv'} "
            f"--output-dir {JEU} --cache-dir {cache} --forcage {G / 'forcage_era5.csv'} "
            f"--nx {INFO['zone']['nx']} --ny {INFO['zone']['ny']} --f-coriolis {INFO['f_coriolis']} "
            f"--perturbations 0.4 --test {TEST} --val {VAL} --marge-jours 7 "
            f"--instants 1000 --heures-par-jour 4 --workers {nw}")
    for jeux, part in PARTIES[PARTIE]:
        lancer(f"generation_p{PARTIE}", base + f" --jeux {jeux}" + (f" --partie {part}" if part else ""))
    if not (GT / "calendrier.json").exists():
        shutil.copy2(JEU / "calendrier.json", GT / "calendrier.json")

    # jours du calendrier sans forcage pour cette partie : refuse le point de reprise
    cal = json.loads((JEU / "calendrier.json").read_text())
    import pandas as pd
    t = pd.to_datetime(pd.read_csv(G / "forcage_era5.csv", usecols=["timestamp_utc"])["timestamp_utc"], utc=True)
    forces = set(t.dt.tz_localize(None).dt.normalize())
    idx = pd.read_csv(cache / "index.csv")
    attendus = {k for jeux, _ in PARTIES[PARTIE] for k in cal["split"][f"{jeux}_days"]}
    dates = {d["day_index"]: d["date"] for d in cal["days"]}
    manque = sorted(dates[k] for k in attendus if pd.Timestamp(dates[k]) not in forces)
    print(f"partie {PARTIE} : {len(idx)} instants generes ; jours sans forcage ERA5 : {len(manque)}")
    if manque and PARTIE in (2, 3):
        raise RuntimeError(f"forcage ERA5 absent pour {manque[:5]}... : deposer le forcage a jour "
                           "(grand_tunis_donnees.zip) puis relancer cette partie")

    # point de reprise : la partie entiere sur le Drive (index renomme : 3 index a fusionner)
    tmp = GT / f"cache_partie_{PARTIE}.zip.part"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in sorted(cache.iterdir()):
            nom = f"index_partie_{PARTIE}.csv" if f.name == "index.csv" else f.name
            zf.write(f, nom)
    tmp.replace(GT / f"cache_partie_{PARTIE}.zip")
    if not (GT / "jeu.zip").exists():
        with zipfile.ZipFile(GT / "jeu.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for f in sorted(JEU.iterdir()):
                if f.is_file():
                    zf.write(f, f.name)
    print(f"point de reprise {PARTIE}/3 sur le Drive : "
          f"{(GT / f'cache_partie_{PARTIE}.zip').stat().st_size / 1e9:.1f} Go")
'''


def generation() -> dict:
    return carnet([
        md("""
# PC-FNO3D réel (Grand Tunis) — génération du jeu (Colab, CPU)

Vérité d'entraînement = le générateur forcé par **ERA5** heure par heure, sur le relief
(Copernicus DEM) et le sol (ESA WorldCover) réels ; **aucune station n'y entre**.
1 000 instants répartis par saison et par heure, écrits directement au format du cache.

**Environnement : CPU** (Exécution → Modifier le type d'exécution → CPU, *Mémoire RAM élevée*).
Un seul **Tout exécuter** par partie : le carnet fait la partie suivante qui manque sur le Drive.

| partie | contenu | point de reprise |
|---|---|---|
| 1 | entraînement, 1re moitié | `cache_partie_1.zip` |
| 2 | entraînement, 2e moitié + validation (dont juillet 2021) | `cache_partie_2.zip` |
| 3 | test (janv.–mars et **juillet–août 2022 : en tout dernier**) | `cache_partie_3.zip` |

Une partie coupée en cours est refaite entière (3 points de reprise au plus sur le Drive).
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN),
        md("## 2. Calendrier et partie à faire"),
        code(GEN),
        md("## 3. Génération de la partie, puis point de reprise sur le Drive"),
        code(GEN_RUN),
    ], gpu=False)


ENT = r'''
# cache : les 3 parties du Drive -> disque local ; index fusionne
LOC = Path("/content/grand_tunis")
CACHE, JEU = LOC / "cache", LOC / "jeu"
for k in (1, 2, 3):
    if not (GT / f"cache_partie_{k}.zip").exists():
        raise FileNotFoundError(f"cache_partie_{k}.zip absent : finir le carnet de generation")
if not (CACHE / "index.csv").exists():
    CACHE.mkdir(parents=True, exist_ok=True)
    for k in (1, 2, 3):
        with zipfile.ZipFile(GT / f"cache_partie_{k}.zip") as zf:
            zf.extractall(CACHE)
    with zipfile.ZipFile(GT / "jeu.zip") as zf:
        zf.extractall(JEU)
    import pandas as pd
    idx = pd.concat([pd.read_csv(CACHE / f"index_partie_{k}.csv") for k in (1, 2, 3)])
    idx.drop_duplicates("stem").to_csv(CACHE / "index.csv", index=False)
import pandas as pd
print(pd.read_csv(CACHE / "index.csv")["split"].value_counts().to_string())

# la grille complete (aneo_grid_points.csv) est dans jeu.zip
if not (GT / "best_dso_v7.pt").exists():
    raise FileNotFoundError("deposer best_dso_v7.pt (poids v7) dans MyDrive/pcfno3d_grand_tunis/")
lancer("controle", "python -u -m pcfno3d -c grand_tunis_colab check")
'''

MEM = r'''
# memoire : un pas complet (avant + arriere) sur la grille entiere ; recalcul des activations
# couche par couche si le GPU ne suffit pas (memes resultats, ~30 % plus lent)
import torch
from pcfno3d.config import Config
from pcfno3d.geometry import Geometry
from pcfno3d.model import build_model

def essai(recalcul: bool) -> float:
    cfg = Config.load("configs/grand_tunis_colab.yaml")
    cfg.model.recalcul_activations = recalcul
    geom = Geometry(cfg, device="cuda")
    m = build_model(cfg, geom.c_in).cuda().train()
    x = torch.randn(cfg.train.batch_size, geom.c_in, *geom.shape, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(x).float().pow(2).mean().backward()
    pic = torch.cuda.max_memory_allocated() / 1e9
    del m, x
    torch.cuda.empty_cache()
    return pic

total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(torch.cuda.get_device_name(0), f"{total:.0f} Go")
try:
    pic = essai(False)
    RECALCUL = pic > 0.85 * total
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    pic, RECALCUL = float("nan"), True
if RECALCUL:
    pic = essai(True)
print(f"pic {pic:.1f} Go sur {total:.0f} | recalcul des activations : {RECALCUL}")
OPTS = f"--set model.recalcul_activations={str(RECALCUL).lower()}"
'''


V2 = {
    "grand_tunis_v2_colab": ("v2, mode réel", """
**v2, mode réel** : l'entrée de production (ERA5 10 m tous les 2,5 km + anémomètres) **enrichie
du vent ERA5 à chaque nœud et du forçage de l'instant** (ug, vg, cisaillement, rotation,
stabilité, contraste terre-mer) ; sortie u, v = ERA5 + correction apprise. Départ : `best.pt`
de la v1 (`runs/grand_tunis/`), u, v de départ = ERA5 (R² 0,95 / 0,97 en validation). 20 époques.
Résultats : `runs/grand_tunis_v2/`."""),
    "grand_tunis_v2_long_colab": ("v2 prolongé, mode réel", """
**v2 prolongé, mode réel** : mêmes entrées et sortie que la v2 ; départ = `best.pt` de la v2
(`runs/grand_tunis_v2/`), 40 époques de plus à LR 1e-4. Résultats : `runs/grand_tunis_v2_long/`."""),
    "grand_tunis_v2_aneo_colab": ("v2, mode ANEO simulé", """
**v2, mode ANEO simulé** : entrée = réseau dense de capteurs à 10 m (couche de surface du jeu,
masquée au hasard, comme la v7) + vent ERA5 à chaque nœud + forçage de l'instant ; sortie u, v =
ERA5 + correction apprise. Départ : poids v7. 20 époques. Résultats : `runs/grand_tunis_v2_aneo/`."""),
}


def entrainement(conf: str = "grand_tunis_colab", run: str = "grand_tunis") -> dict:
    nb = _entrainement()
    if conf == "grand_tunis_colab":
        return nb
    titre, texte = V2[conf]
    for c in nb["cells"]:
        src = "".join(c["source"]).replace("grand_tunis_colab", conf)
        src = src.replace('GT / "runs" / "grand_tunis"', f'GT / "runs" / "{run}"')
        src = src.replace("`runs/grand_tunis/`", f"`runs/{run}/`")
        src = src.replace("## 4. Entraînement (départ v7)", "## 4. Entraînement")
        if src.startswith("# PC-FNO3D réel (Grand Tunis) — entraînement"):
            src = (f"# PC-FNO3D réel (Grand Tunis) {titre} — entraînement (Colab, A100)\n"
                   + texte + "\n\n**Environnement : GPU A100.** Reprise automatique : relancer "
                   "**Tout exécuter** après une coupure reprend à la dernière copie sur le Drive.")
        c["source"] = src.splitlines(True)
    return nb


def _entrainement() -> dict:
    return carnet([
        md("""
# PC-FNO3D réel (Grand Tunis) — entraînement (Colab, A100)

**Le modèle déployé** : architecture v7, **départ des poids v7**, entraîné sur le jeu généré
(carnet de génération, 3 parties sur le Drive). Entrée = vent ERA5 à 10 m tous les 2,5 km
+ les anémomètres réels (Tunis-Carthage, Zaghouan, Nabeul), comme en production.

**Environnement : GPU A100.** Reprise automatique : relancer **Tout exécuter** après une
coupure reprend à la dernière copie sur le Drive (au plus 5 époques perdues).

Sur le Drive, `MyDrive/pcfno3d_grand_tunis/` : `pcfno3d_code.zip`, `grand_tunis_donnees.zip`,
`best_dso_v7.pt`, `cache_partie_1..3.zip`, `jeu.zip`. Résultats : `runs/grand_tunis/`.
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN),
        md("## 2. Cache (3 parties du Drive) et contrôle de la vérité"),
        code(ENT),
        md("## 3. Mémoire du GPU"),
        code(MEM),
        md("## 4. Entraînement (départ v7)"),
        code('lancer("entrainer", f"python -u -m pcfno3d -c grand_tunis_colab {OPTS} train")'),
        md("## 5. Test : erreurs par niveau (hiver et été 2022)"),
        code('''
lancer("evaluer", f"python -u -m pcfno3d -c grand_tunis_colab {OPTS} evaluate")
RUN = GT / "runs" / "grand_tunis"
for f in sorted(RUN.rglob("test_summary.json")):
    print(f); print(f.read_text()[:4000])
for f in sorted(RUN.rglob("test_par_niveau_synoptique.csv")):
    print(pd.read_csv(f).round(3).to_string())
'''),
        md("## 6. À rapatrier sur ines-pc\n\n`runs/grand_tunis/` : `best.pt`, `norm_stats.json`, résultats de test."),
        code('''
for f in sorted(RUN.iterdir()):
    print(f"{f.name:40s} {f.stat().st_size / 1e6:8.1f} Mo" if f.is_file() else f"{f.name}/")
'''),
    ], gpu=True)


DIAG = r'''
# validation seulement : index de la partie 2 (train 2/2 + val) et fichiers des instants de val
LOC = Path("/content/grand_tunis")
CACHE, JEU = LOC / "cache", LOC / "jeu"
CACHE.mkdir(parents=True, exist_ok=True)
JEU.mkdir(parents=True, exist_ok=True)
if (GT / "diag_grand_tunis.py").exists():           # script depose a part (plus recent que le zip)
    shutil.copy2(GT / "diag_grand_tunis.py", CODE / "scripts" / "diag_grand_tunis.py")
with zipfile.ZipFile(GT / "jeu.zip") as zf:
    zf.extractall(JEU)
import pandas as pd
with zipfile.ZipFile(GT / "cache_partie_2.zip") as zf:
    with zf.open("index_partie_2.csv") as f:
        idx = pd.read_csv(f)
    idx.to_csv(CACHE / "index.csv", index=False)
    val = set(idx.loc[idx["split"] == "val", "stem"])
    noms = [n for n in zf.namelist() if n.rsplit("_", 1)[0] in val]
    zf.extractall(CACHE, members=noms)
print(f"{len(val)} instants de validation, {len(noms)} fichiers")
'''


def diagnostic() -> dict:
    return carnet([
        md("""
# PC-FNO3D réel (Grand Tunis) — diagnostic sur la validation (Colab, CPU)

Avant toute relance de l'entraînement (plateau depuis l'époque 11) : sur les 64 instants de
validation, R² et RMSE par niveau d'**ERA5 interpolé**, du **vent moyen du domaine** (oracle) et
du **modèle** (`best.pt`) ; alignement entrée ERA5 / cible ; part de la variance de la cible due
aux perturbations aléatoires du générateur (instants régénérés sans perturbation).

**Environnement : CPU, mémoire RAM élevée.** Résultats : `MyDrive/pcfno3d_grand_tunis/diagnostic/`.
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN),
        md("## 2. Instants de validation (partie 2)"),
        code(DIAG),
        md("## 3. Diagnostic"),
        code('''
SORTIE = Path("/content/diag")
CKPT = Path("/content/best.pt")                     # copie locale : poids mappes en memoire
shutil.copy2(GT / "runs" / "grand_tunis" / "best.pt", CKPT)
lancer("diagnostic", f"python -u scripts/diag_grand_tunis.py -c grand_tunis_colab "
                     f"--donnees {DON / 'grand_tunis'} --ckpt {CKPT} --sortie {SORTIE}")
D = GT / "diagnostic"
D.mkdir(exist_ok=True)
for f in ("diagnostic.json", "alignement_par_instant.json", "r2_rmse_par_niveau.csv"):
    shutil.copy2(SORTIE / f, D / f)
print(sorted(p.name for p in D.iterdir()))
'''),
    ], gpu=False)


VALID = r'''
# instants de TEST des 3 parties (S, Y, E) + index fusionne ; observations reelles du domaine
LOC = Path("/content/grand_tunis")
CACHE, JEU = LOC / "cache", LOC / "jeu"
CACHE.mkdir(parents=True, exist_ok=True)
JEU.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(GT / "jeu.zip") as zf:
    zf.extractall(JEU)
import pandas as pd
index, n = [], 0
for k in (1, 2, 3):
    with zipfile.ZipFile(GT / f"cache_partie_{k}.zip") as zf:
        with zf.open(f"index_partie_{k}.csv") as f:
            idx = pd.read_csv(f)
        index.append(idx)
        test = set(idx.loc[idx["split"] == "test", "stem"])
        noms = [x for x in zf.namelist() if x.rsplit("_", 1)[0] in test]
        zf.extractall(CACHE, members=noms)
        n += len(noms)
pd.concat(index).drop_duplicates("stem").to_csv(CACHE / "index.csv", index=False)
OBS = Path("/content/obs")
with zipfile.ZipFile(GT / "grand_tunis_obs.zip") as zf:
    zf.extractall(OBS)
print(f"{n} fichiers de test ; observations : {sorted(p.name for p in OBS.iterdir())}")
'''


def validation() -> dict:
    return carnet([
        md("""
# PC-FNO3D réel (Grand Tunis) — validation sur de VRAIES mesures (Colab, GPU)

Sur les 180 instants de test (2022) : entrée de production avec les **vrais** anémomètres
(METAR Tunis-Carthage, SYNOP Zaghouan et Nabeul), chaque station **retirée de l'entrée** quand
on la juge ; vent à 10 m contre la mesure, comparé à ERA5 et au générateur ; profils contre les
**radiosondages** de Tunis-Carthage (0–1 000 m). Modèles : v2 prolongé, puis v2.
`scripts/valider_grand_tunis_reel.py`. La session s'arrête à la fin.

Sur le Drive : `grand_tunis_obs.zip` (python scripts/package_grand_tunis_colab.py).
Résultats : `runs/<modèle>/validation_reelle/test/`.
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN),
        md("## 2. Instants de test et observations"),
        code(VALID),
        md("## 3. Validation (v2 prolongé, puis v2), puis fin de la session"),
        code('''
from google.colab import runtime
try:
    for conf, run in (("grand_tunis_v2_long_colab", "grand_tunis_v2_long"),
                      ("grand_tunis_v2_colab", "grand_tunis_v2")):
        try:
            lancer(f"valider_{run}", f"python -u scripts/valider_grand_tunis_reel.py -c {conf} --obs {OBS}")
        except Exception as e:
            print(e)
finally:
    runtime.unassign()
'''),
    ], gpu=True)


V3 = ("grand_tunis_v3_stations", "grand_tunis_v3_cv_sans_dtta", "grand_tunis_v3_cv_sans_zaghouan",
      "grand_tunis_v3_cv_sans_nabeul")

OBS = r'''
# observations reelles du domaine (affinage et jugement)
OBS = Path("/content/obs")
with zipfile.ZipFile(GT / "grand_tunis_obs.zip") as zf:
    zf.extractall(OBS)
print(sorted(p.name for p in OBS.iterdir()))
'''


def affinage() -> dict:
    """v3 : affinage sur les stations (modele final, puis 3 plis de validation croisee spatiale),
    chacun entraine, evalue sur le test genere, puis juge sur les VRAIES mesures."""
    mem = MEM.replace("configs/grand_tunis_colab.yaml", "configs/grand_tunis_v3_stations_colab.yaml")
    chaine = '''
from google.colab import runtime
import traceback
try:
    for run in V3:
        conf = run + "_colab"
        for nom, cmd in ((f"entrainer_{run}", f"python -u -m pcfno3d -c {conf} {OPTS} train"),
                         (f"evaluer_{run}", f"python -u -m pcfno3d -c {conf} {OPTS} evaluate"),
                         (f"valider_{run}", f"python -u scripts/valider_grand_tunis_reel.py -c {conf} --obs {OBS}")):
            try:
                lancer(nom, cmd)
            except Exception:
                traceback.print_exc()
finally:
    runtime.unassign()
'''
    return carnet([
        md("""
# PC-FNO3D réel (Grand Tunis) v3 — affinage sur les stations (Colab, A100)

Depuis la v2 prolongée : à chaque instant d'entraînement, une station mesurée est tirée comme
**cible** et retirée de l'entrée ; son vent à 10 m prédit est rapproché de la **vraie mesure**
(`loss.lambda_station`), la vérité générée reste la cible partout ailleurs. Quatre runs, 10
époques chacun : le **modèle final** (3 stations cibles), puis **3 plis de validation croisée**
où une station n'est jamais cible — c'est elle qui juge honnêtement le gain. Chaque run est
évalué sur le test généré puis sur les vraies mesures. Seul `best.pt` va sur le Drive
(`train.sync_last: false`). La session s'arrête à la fin.
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN),
        md("## 2. Cache (3 parties) et observations"),
        code(ENT.replace("grand_tunis_colab", "grand_tunis_v3_stations_colab") + OBS),
        md("## 3. Mémoire du GPU"),
        code(mem),
        md("## 4. Affinage, test, vraies mesures (4 runs), puis fin de la session"),
        code(f"V3 = {V3!r}" + chr(10) + chaine),
    ], gpu=True)


def suite_v3() -> dict:
    """Apres les plis : bilan et regle de livraison, cartes v3 - v2, puis (seulement si les plis
    montrent un gain) v3b = modele final avec Nabeul en direction seule."""
    suite = '''
from google.colab import runtime
import traceback
RUNS = GT / "runs"
CARTES = RUNS / "comparaisons"
def essai(nom, cmd):
    try:
        lancer(nom, cmd)
        return 0
    except Exception as e:
        print(e)
        return 1
try:
    # 1. plis : la station jamais cible juge ; code 0 = gain (v3 livrable), 3 = la v2 reste livree
    p = subprocess.run(f"python -u scripts/bilan_plis_grand_tunis.py --runs {RUNS}", shell=True,
                       cwd="/content/PFE")
    gain = p.returncode == 0
    print("GAIN AUX STATIONS JAMAIS VUES :", gain)
    # 2. cartes de l'ecart v3 - v2 a 10 m et 100 m
    essai("cartes_v3", f"python -u scripts/comparer_modeles_grand_tunis.py --conf-a grand_tunis_v2_long_colab "
                       f"--conf-b grand_tunis_v3_stations_colab --obs {OBS} --sortie {CARTES}")
    # 3. v3b (Nabeul en direction seule), seulement si les plis montrent un gain
    if gain:
        conf = "grand_tunis_v3b_stations_colab"
        essai("entrainer_v3b", f"python -u -m pcfno3d -c {conf} {OPTS} train")
        essai("evaluer_v3b", f"python -u -m pcfno3d -c {conf} {OPTS} evaluate")
        essai("valider_v3b", f"python -u scripts/valider_grand_tunis_reel.py -c {conf} --obs {OBS}")
        essai("cartes_v3b", f"python -u scripts/comparer_modeles_grand_tunis.py --conf-a grand_tunis_v2_long_colab "
                            f"--conf-b {conf} --obs {OBS} --sortie {CARTES}")
    else:
        print("pas de gain aux stations jamais vues : la v2 reste le modele livre, pas de v3b")
finally:
    runtime.unassign()
'''
    mem = MEM.replace("configs/grand_tunis_colab.yaml", "configs/grand_tunis_v3b_stations_colab.yaml")
    return carnet([
        md("""
# PC-FNO3D réel (Grand Tunis) — suite de la v3 : plis, cartes, Nabeul (Colab, A100)

1. **Bilan des 3 plis** de validation croisée (`scripts/bilan_plis_grand_tunis.py`) : erreur à la
   station jamais cible, à côté de la v2 et d'ERA5. Règle : la v3 n'est livrée que si les plis
   montrent un gain aux stations jamais vues ; sinon la v2 reste livrée.
2. **Cartes de l'écart v3 − v2** à 10 m et 100 m (`scripts/comparer_modeles_grand_tunis.py`) :
   pas de taches autour des stations ?
3. **Seulement si gain** : v3b = modèle final avec **Nabeul en direction seule**
   (`loss.poids_vitesse_station`), entraîné, évalué, jugé sur les vraies mesures, cartes.

Résultats : `runs/bilan_plis_v3.*`, `runs/comparaisons/`, `runs/grand_tunis_v3b_stations/`.
La session s'arrête à la fin.
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN),
        md("## 2. Cache (3 parties) et observations"),
        code(ENT.replace("grand_tunis_colab", "grand_tunis_v3b_stations_colab") + OBS),
        md("## 3. Mémoire du GPU"),
        code(mem),
        md("## 4. Plis, cartes, v3b si gain, puis fin de la session"),
        code(suite),
    ], gpu=True)


def v3b() -> dict:
    """v3b demandee par Ines (2026-10-24) malgre l'absence de gain aux plis : modele final avec
    Nabeul en direction seule ; entrainement, test genere, vraies mesures, cartes v3b - v2."""
    chaine = '''
from google.colab import runtime
import traceback
conf = "grand_tunis_v3b_stations_colab"
CARTES = GT / "runs" / "comparaisons"
try:
    for nom, cmd in (("entrainer_v3b", f"python -u -m pcfno3d -c {conf} {OPTS} train"),
                     ("evaluer_v3b", f"python -u -m pcfno3d -c {conf} {OPTS} evaluate"),
                     ("valider_v3b", f"python -u scripts/valider_grand_tunis_reel.py -c {conf} --obs {OBS}"),
                     ("cartes_v3b", f"python -u scripts/comparer_modeles_grand_tunis.py --conf-a grand_tunis_v2_long_colab "
                                    f"--conf-b {conf} --obs {OBS} --sortie {CARTES}")):
        try:
            lancer(nom, cmd)
        except Exception:
            traceback.print_exc()
finally:
    runtime.unassign()
'''
    mem = MEM.replace("configs/grand_tunis_colab.yaml", "configs/grand_tunis_v3b_stations_colab.yaml")
    return carnet([
        md("""
# PC-FNO3D réel (Grand Tunis) v3b — affinage sur les stations, Nabeul en direction seule (Colab, A100)

Comme la v3 finale (les 3 stations cibles, 10 époques depuis la v2 prolongée), mais **Nabeul ne
compte que par sa direction** (`loss.poids_vitesse_station`) : station abritée, sa vitesse n'est
pas représentative. Les plis de validation croisée ont montré que l'affinage reste une correction
locale aux stations : la v2 prolongée reste le modèle livré ; la v3b documente l'effet de la
correction de Nabeul. Entraînement, test généré, vraies mesures, cartes v3b − v2 ; seul `best.pt`
va sur le Drive. La session s'arrête à la fin.
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN),
        md("## 2. Cache (3 parties) et observations"),
        code(ENT.replace("grand_tunis_colab", "grand_tunis_v3b_stations_colab") + OBS),
        md("## 3. Mémoire du GPU"),
        code(mem),
        md("## 4. v3b : entraînement, test, vraies mesures, cartes, puis fin de la session"),
        code(chaine),
    ], gpu=True)


def service_v2() -> dict:
    """Le modele LIVRE (v2 prolongee) servi a l'application, a l'adresse fixe atom.zelqane.com
    (tunnel nomme `atom`, Worker de Rayen) — meme schema que pcfno3d_atom_colab.ipynb (v7)."""
    conf = "grand_tunis_v2_service_colab"
    installe = '''
# le service web : aucune de ces bibliotheques ne touche a numpy
!pip install -q fastapi uvicorn python-multipart httpx mcp==2.3.0 2>&1 | tail -2
'''
    production = r'''
# modele livre (decision d'Ines, 2026-10-24) copie dans runs/grand_tunis_v9_production (poids,
# statistiques, historique, evaluations, validation reelle, fiabilite, figures) ; une seule fois.
# La v4 (affinage DSO) remplace la v2 prolongee SEULEMENT si runs/decision_v4.json le dit (regle :
# au moins aussi bonne sur DSO1, DSO4, DSO5 et les vraies mesures) ; w par continuite dans les deux cas.
RUNS = GT / "runs"
DECISION = RUNS / "decision_v4.json"
livre = json.loads(DECISION.read_text())["modele_livre"] if DECISION.exists() else "grand_tunis_v2_long"
SRC, PROD = RUNS / livre, RUNS / "grand_tunis_v9_production"
print("modele livre :", livre, "(decision_v4.json)" if DECISION.exists() else "(pas de decision v4 : v2)")
if (PROD / "PROVENANCE.txt").exists() and livre not in (PROD / "PROVENANCE.txt").read_text():
    raise RuntimeError(f"{PROD} contient un autre modele que {livre} : verifie avant de le remplacer")
if not (PROD / "best.pt").exists():
    PROD.mkdir(parents=True, exist_ok=True)
    for nom in ("norm_stats.json", "history.json", "best.pt"):
        shutil.copy2(SRC / nom, PROD / nom)
    for d in ("eval_pcfno3d", "validation_reelle", "fiabilite_pcfno3d", "figures"):
        if (SRC / d).exists():
            shutil.copytree(SRC / d, PROD / d, dirs_exist_ok=True)
    (PROD / "PROVENANCE.txt").write_text(
        f"Copie de runs/{livre} : le modele livre PC-FNO3D Grand Tunis, servi avec w recalcule par "
        "continuite (decision d'Ines, 2026-10-24 ; runs/decision_v4.json et rapport_v4.md). Les "
        "affinages sur les stations (v3, v3b) sont ecartes : correction de site autour des stations, "
        "pas de gain aux stations jamais vues.", encoding="utf-8")
# affinages v3 : resultats gardes pour le rapport, poids supprimes sauf le pli sans Nabeul
for run in ("grand_tunis_v3_stations", "grand_tunis_v3_cv_sans_dtta", "grand_tunis_v3_cv_sans_zaghouan",
            "grand_tunis_v3b_stations"):
    f = RUNS / run / "best.pt"
    if f.exists():
        f.unlink()
        print("supprime :", f)
print("production :", sorted(p.name for p in PROD.iterdir()))
'''
    exporte = f'''
# volumes 3D de la vue de Rayen (/api/export/<instant>/...) : un pas sur deux a l'horizontale
lancer("export_web", "python -u -m pcfno3d -c {conf} export-web --split test --stride 2")
'''
    sert = f'''
from google.colab import userdata
from service.colab import serve_public

TUNNEL_ID = "e552f718-2478-4fe3-82b4-8d33a4b9eb6e"
HOTE = "atom-origin.zelqane.com"
CLE_NOM = "atom-worker"                      # la cle de lecture du Worker de Rayen
CLE_EMPREINTE = "ce7965c54284eb6d1a274119839a71d9785731609d04b17bd08fb6da7d2c30fd"

try:
    identifiants = userdata.get("TUNNEL_CRED_CONTENTS")
except Exception as exc:                     # secret absent, ou acces non accorde
    raise RuntimeError(
        "secret Colab TUNNEL_CRED_CONTENTS introuvable ou non autorise : panneau de gauche, "
        "icone cle, ajoute-le (contenu remis par Rayen) et active « Acces au notebook ».") from None

info = serve_public(config="{conf}", key_name=CLE_NOM, scope="read", key_hash=CLE_EMPREINTE,
                    quota_minute=1200,
                    named_tunnel={{"id": TUNNEL_ID, "hostname": HOTE, "credentials": identifiants}})
del identifiants
'''
    verifie = r'''
import urllib.request
req = urllib.request.Request(f"https://{HOTE}/api/health", headers={"User-Agent": "pcfno3d-colab/1.0"})
with urllib.request.urlopen(req, timeout=60) as r:
    sante = json.loads(r.read().decode())
print(json.dumps(sante, ensure_ascii=False, indent=1))
assert sante["status"] == "ok", "le service ne repond pas correctement par le tunnel"
print(f"\nEN LIGNE : https://{HOTE} ({livre}, Grand Tunis) — previens Rayen (sans coller de cle).")
print(f"Front-end ATOM Wind : https://{HOTE}/app/atom-wind/ (la cle d'API se saisit dans le bandeau) ;"
      " routes /api/atom-wind/grille, /jours, /jour?date=..., /reseau-aneo (plan OS6).")
'''
    vie = '''
try:
    while True:
        tunnel_ok = info["tunnel"].alive
        service_ok = info["process"].poll() is None
        print(time.strftime("%H:%M"), "| tunnel", "ok" if tunnel_ok else "ARRETE",
              "| service", "ok" if service_ok else "ARRETE", flush=True)
        if not (tunnel_ok and service_ok):
            print("journaux :", info["log_path"], "et", info["tunnel"].log_path)
            break
        time.sleep(600)
except KeyboardInterrupt:
    print("surveillance arretee — le service tourne toujours")
'''
    arret = '''
# Garde-fou : « Tout executer » atteint aussi cette cellule des que la surveillance s'arrete.
# Mets ARRETER = True pour couper volontairement.
ARRETER = False
if ARRETER:
    from service.colab import stop
    stop(info)
    print("service et tunnel arretes : atom.zelqane.com repasse en « endormi »")
else:
    print("service laisse en marche (ARRETER = False)")
'''
    return carnet([
        md("""
# PC-FNO3D réel (Grand Tunis) — le modèle livré en ligne (atom.zelqane.com)

Sert le **modèle livré** (la v2 prolongée, ou la v4 si `runs/decision_v4.json` la retient), avec
w recalculé par continuité, copié dans `runs/grand_tunis_v9_production`, à l'adresse fixe :

```
navigateur / vue 3D --> atom.zelqane.com (Worker de Rayen, clé injectée côté serveur)
                    --> atom-origin.zelqane.com (tunnel nommé, URL fixe)
                    --> CE carnet : cloudflared --> service sur 127.0.0.1:8000 (GPU)
```

Entrée de production : ERA5 10 m sur le réseau régulier + les **vraies mesures** des 3 stations
(Tunis-Carthage, Zaghouan, Nabeul) + le forçage ERA5 de l'instant. Instants servis : les 180
instants de test 2022 du cache.

**Avant la première exécution** : secret Colab `TUNNEL_CRED_CONTENTS` (identifiants du tunnel
`atom`, remis par Rayen de la main à la main), avec « Accès au notebook ». La clé d'API n'est
présente que par son **empreinte**.

**GPU** : le carnet demande un **L4** (24 Go, devrait suffire pour servir : non encore essayé) ; si Colab n'en donne pas, ou en
cas de manque de mémoire, Exécution > Modifier le type d'exécution > **A100**.

**Exécution** : GPU, puis **Tout exécuter** (~20 min : code, données, instants de test,
volumes 3D, démarrage). Laisse ensuite l'onglet ouvert : le service vit tant que la session vit.
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN + installe),
        md("## 2. Modèle livré dans runs/grand_tunis_v9_production"),
        code(production),
        md("## 3. Instants de test (cache) et vraies mesures des stations"),
        code(VALID),
        md("## 4. Volumes 3D pour la visualisation"),
        code(exporte),
        md("## 5. Démarrer le service derrière le tunnel nommé"),
        code(sert),
        md("## 6. Vérifier de l'extérieur"),
        code(verifie),
        md("## 7. Garder en vie"),
        code(vie),
        code(arret),
    ], gpu=True, gpu_type="L4")


def v4_dso() -> dict:
    """v4 : tous les DSO (demande d'Ines, 2026-10-24). Mesure d'abord la v2 prolongee (brute et avec
    w par continuite, robustesse DSO5 corrigee), puis poids par niveau sur la validation, affinage
    selectionne sur le score DSO, test (avec et sans w par continuite), vraies mesures."""
    chaine = r'''
from google.colab import runtime
import traceback
RUNS = GT / "runs"
V2, V4 = "grand_tunis_v2_long_colab", "grand_tunis_v4_dso_colab"
R2, R4 = RUNS / "grand_tunis_v2_long", RUNS / "grand_tunis_v4_dso"
POIDS = R4 / "uv_level_weight.json"
WCONT = "--set inference.w_continuite=true"

# l'evaluation de la v2 faite avec le code d'avant (DSO5 mal mesure) reste pour le rapport
for d in ("best", "best_wcont"):
    a = R2 / "eval_pcfno3d" / d
    if a.exists() and not a.with_name(d + "_avant_v4").exists():
        shutil.copytree(a, a.with_name(d + "_avant_v4"))

def evalue(run, dossier):                    # le test n'est calcule qu'UNE fois (code actuel :
    f = run / "eval_pcfno3d" / dossier / "test_par_niveau_synoptique.csv"   # colonne rugosite)
    return f.exists() and "rugosite" in f.read_text().splitlines()[0]

def entraine():                              # les 15 epoques de la v4 sont deja faites
    h = R4 / "history.json"
    return h.exists() and any(r.get("epoch") == 14 for r in json.loads(h.read_text()))

etapes = (
    ("evaluer_v2_brut", f"python -u -m pcfno3d -c {V2} {OPTS} evaluate", lambda: evalue(R2, "best")),
    ("evaluer_v2_wcont", f"python -u -m pcfno3d -c {V2} {OPTS} {WCONT} evaluate", lambda: evalue(R2, "best_wcont")),
    ("poids_v4", f"python -u -m pcfno3d -c {V2} {OPTS} dso-weights --out {POIDS}", POIDS.exists),
    ("entrainer_v4", f"python -u -m pcfno3d -c {V4} {OPTS} --set loss.uv_level_weight={POIDS} train", entraine),
    ("evaluer_v4_wcont", f"python -u -m pcfno3d -c {V4} {OPTS} evaluate", lambda: evalue(R4, "best_wcont")),
    ("evaluer_v4_brut", f"python -u -m pcfno3d -c {V4} {OPTS} --set inference.w_continuite=false evaluate", lambda: evalue(R4, "best")),
    ("valider_v4", f"python -u scripts/valider_grand_tunis_reel.py -c {V4} --obs {OBS}",
     lambda: (R4 / "validation_reelle" / "test" / "stations_bilan.csv").exists()),
    # variantes PROPOSEES (DSO3 sur la partie previsible de la verite) : publiees dans le
    # rapport, jamais lues par la decision tant que l'encadrant ne les a pas validees
    ("variantes_v2", f"python -u scripts/dso_variantes_grand_tunis.py -c {V2} --donnees {DON / 'grand_tunis'} --w-continuite",
     lambda: (R2 / "eval_pcfno3d" / "best_wcont" / "dso3_previsible.json").exists()),
    ("variantes_v4", f"python -u scripts/dso_variantes_grand_tunis.py -c {V4} --donnees {DON / 'grand_tunis'} --w-continuite",
     lambda: (R4 / "eval_pcfno3d" / "best_wcont" / "dso3_previsible.json").exists()),
)
try:
    for nom, cmd, fait in etapes:
        if fait():
            print(f"[{nom}] deja fait : saute")
            continue
        try:
            lancer(nom, cmd)
        except Exception:
            traceback.print_exc()

    # decision (regle d'Ines) : la v4 seulement si au moins aussi bonne que la v2 sur DSO1, DSO4,
    # DSO5 et les vraies mesures ; sinon la v2 avec w par continuite. Plus aucun entrainement.
    DEC = "python -u scripts/decision_v4_grand_tunis.py --runs " + str(RUNS)
    code_dec = subprocess.run(DEC, shell=True, cwd="/content/PFE").returncode
    if code_dec not in (0, 3):
        raise RuntimeError(f"decision impossible (code {code_dec}) : evaluations incompletes")
    conf, run, opts = (V4, R4, OPTS) if code_dec == 0 else (V2, R2, OPTS + " " + WCONT)
    print("modele retenu :", run.name, "(w par continuite)")

    # DSO6 sur le modele retenu, SANS reentrainement : ensemble par masquage, calibration sur la
    # validation, puis couverture 80 % et correlation incertitude / erreur sur le test (une fois)
    CAL = run / "fiabilite_pcfno3d" / "best_wcont" / "calibration_validation.json"
    D6 = run / "eval_pcfno3d" / "best_wcont" / "dso6_modele_livre.json"
    if not CAL.exists():
        lancer("calibrer_dso6", f"python -u -m pcfno3d -c {conf} {opts} calibrate-reliability --out {CAL}")
    if not D6.exists():
        try:
            lancer("dso6", f"python -u -m pcfno3d -c {conf} {opts} dso6 --calibration {CAL} --out {D6}")
        except RuntimeError as exc:          # code 3 = NON CONFORME : le verdict est ecrit
            print(exc)
    subprocess.run(DEC + f" --dso6 {D6}", shell=True, cwd="/content/PFE")
    print("rapport ->", RUNS / "rapport_v4.md", "| decision ->", RUNS / "decision_v4.json")
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''
    mem = MEM.replace("configs/grand_tunis_colab.yaml", "configs/grand_tunis_v4_dso_colab.yaml")
    return carnet([
        md("""
# PC-FNO3D réel (Grand Tunis) v4 — affinage sur les DSO (Colab, A100)

Objectif : **tous les DSO**. La v2 prolongée manque DSO1 (25 m ; R² de w), DSO2 (masse) et DSO3
(rugosité). Étapes : (1) re-mesure de la v2 (DSO5 corrigé) brute et avec **w recalculé par
continuité** ; (2) poids u, v par niveau mesurés sur la **validation** ; (3) affinage 15 époques
sélectionné sur le **score DSO** de la validation (masse ×5, gradients 0,5, w ×2) ; (4) test avec et
sans w par continuité ; (5) vraies mesures ; (6) **décision** (`scripts/decision_v4_grand_tunis.py`,
règle fixée avant le test) : la v4 n'est livrée que si elle est au moins aussi bonne que la v2 sur
DSO1, DSO4, DSO5 et les vraies mesures, sinon la v2 avec w par continuité ; (7) **DSO6** sur le modèle
retenu, sans réentraînement (ensemble par masquage, calibration sur la validation, couverture 80 % et
corrélation incertitude / erreur sur le test). Le test n'est lu qu'une fois : une étape déjà faite est
sautée. Rapport : `runs/rapport_v4.md` (DSO1 et DSO3 niveau par niveau v2 / v4, R² de w avec et sans
continuité, DSO2 « garanti par construction »). **Variantes proposées**, calculées mais jamais
utilisées pour décider tant que l'encadrant ne les a pas validées : DSO3 sur la partie prévisible de
la vérité (instants régénérés sans les perturbations aléatoires du générateur) et DSO1 sur le R² de
u et v seulement, w rapporté à part. Aucun bruit n'est ajouté aux prédictions. Seul `best.pt` va sur le Drive. La session s'arrête
à la fin. Après ça, plus aucun entraînement.
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN),
        md("## 2. Cache (3 parties) et observations"),
        code(ENT.replace("grand_tunis_colab", "grand_tunis_v4_dso_colab") + OBS),
        md("## 3. Mémoire du GPU"),
        code(mem),
        md("## 4. v2 re-mesurée, poids, affinage v4, test, vraies mesures, décision, DSO6, fin de la session"),
        code(chaine),
    ], gpu=True)


def os6_reseau() -> dict:
    """OS6 : plan de deploiement du reseau ANEO (OSSE) sur le modele LIVRE (decision_v4.json),
    sans entrainement. Placement glouton sur la validation, courbe sur le test, plan S18."""
    chaine = r'''
from google.colab import runtime
import traceback
RUNS = GT / "runs"
DECISION = RUNS / "decision_v4.json"
livre = json.loads(DECISION.read_text())["modele_livre"] if DECISION.exists() else "grand_tunis_v2_long"
CONF = {"grand_tunis_v4_dso": "grand_tunis_v4_dso_colab",
        "grand_tunis_v2_long": "grand_tunis_v2_long_colab"}[livre]
SORTIE = RUNS / "os6_reseau_aneo"
CONTRAINTES = GT / "os6_contraintes.json"      # zones exclues, budget... (facultatif, depose sur le Drive)
print("modele livre :", livre, "| contraintes :", CONTRAINTES if CONTRAINTES.exists() else "par defaut")
try:
    lancer("os6_reseau_aneo",
           f"python -u scripts/os6_reseau_aneo.py -c {CONF} --set inference.w_continuite=true "
           f"--sortie {SORTIE} --n-val 24 --lot 4"
           + (f" --contraintes {CONTRAINTES}" if CONTRAINTES.exists() else ""))
    print(json.dumps({k: v for k, v in json.loads((SORTIE / "os6.json").read_text()).items()
                      if k in ("nombre_recommande", "candidats", "capteurs_places", "reseau_actuel",
                               "comparaison_net1")}, indent=1, ensure_ascii=False))
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''
    return carnet([
        md("""
# PC-FNO3D réel (Grand Tunis) — OS6 : plan de déploiement du réseau ANEO (Colab, L4)

Expériences de simulation de systèmes d'observation (OSSE) sur le **modèle livré** (celui que
désigne `runs/decision_v4.json`, w par continuité), **sans entraînement** : un capteur ANEO simulé
est une maille où le modèle voit la vérité de surface du cache, comme une station réelle.

1. emplacements candidats (terre ferme, altitude ≤ 650 m, pente ≤ 20°, à 1 km du bord, à 1 km des
   stations et entre capteurs ; zones exclues de `os6_contraintes.json` s'il est déposé sur le Drive) ;
2. placement glouton sur la **validation** : 5 capteurs par étape là où l'erreur résiduelle est la
   plus forte, gain mesuré à chaque étape (jusqu'à 200) ;
3. courbe de dimensionnement sur le **test** (une fois) : optimisé, grille régulière, hasard, à nombre
   égal ; nombre minimal pour un pire RMSE de niveau ≤ 0,5 m/s ;
4. plan S18 : `runs/os6_reseau_aneo/plan_aneo.csv`, `.geojson`, `carte.png`, `courbe.png`, `os6.json`.

Le front-end ATOM Wind lit le plan (`/api/atom-wind/reseau-aneo`) au prochain démarrage du service.
Reprise : une étape déjà écrite (placement, courbe) est relue. La session s'arrête à la fin.
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN),
        md("## 2. Cache (3 parties)"),
        code(ENT.replace("grand_tunis_colab", "grand_tunis_v2_long_colab")),
        md("## 3. Placement, courbe, plan — puis fin de la session"),
        code(chaine),
    ], gpu=True, gpu_type="L4")


def os7_prevision() -> dict:
    """OS7 (cahier v12, 18.10) : prevision du vent 3D a +24 et +48 h. Previsions GFS archivees (NOAA,
    AWS, Herbie) aux instants du cache, reference « modele livre + GFS sans affinage », affinage v5,
    PREV1 / PREV2 / PREV3 aux stations, test 3D, et AGRI1 / AGRI2 du modele livre."""
    installe = """
# Herbie : telechargement partiel des fichiers GFS (index .idx) ; cfgrib / eccodes : lecture GRIB2
!pip install -q herbie-data cfgrib eccodes 2>&1 | tail -1
"""
    chaine = r"""
from google.colab import runtime
import traceback
RUNS = GT / "runs"
OS7 = GT / "os7"
OS7.mkdir(exist_ok=True)
DECISION = RUNS / "decision_v4.json"
livre = json.loads(DECISION.read_text())["modele_livre"] if DECISION.exists() else "grand_tunis_v2_long"
CONF_LIVRE = {"grand_tunis_v4_dso": "grand_tunis_v4_dso_colab",
              "grand_tunis_v2_long": "grand_tunis_v2_long_colab"}[livre]
V5 = "grand_tunis_v5_gfs_colab"
R5 = RUNS / "grand_tunis_v5_gfs"
DONNEES = DON / "grand_tunis"
F = {e: OS7 / f"forcage_gfs_f{e}.csv" for e in (24, 48)}
print("modele livre :", livre)

def forcage(e):                    # le forcage GFS d'une echeance, copie la ou la configuration le lit
    return f"--set paths.forcage_era5={DONNEES / F[e].name}"

def copier_forcages():
    for e, f in F.items():
        for suffixe in (".csv", ".npz"):
            src = f.with_suffix(suffixe)
            if src.exists():
                shutil.copy2(src, DONNEES / src.name)

try:
    # 1. previsions GFS archivees : reprise par instant sur le Drive (quelques Ko chacun)
    for e in (24, 48):
        if not F[e].exists():
            lancer(f"gfs_f{e}", f"python -u -m pcfno3d.gfs --config {CONF_LIVRE} --echeance {e} "
                                f"--sortie {F[e]} --reprise {OS7 / f'gfs_f{e}_instants'} --travailleurs 8")
    copier_forcages()

    # 2. reference : le modele LIVRE nourri par GFS, sans affinage (meme entree que la v5)
    INIT = RUNS / livre / "best.pt"
    for e in (24, 48):
        d = OS7 / f"sans_affinage_f{e}"
        if not (d / "prevision_bilan.json").exists():
            try:
                lancer(f"prevision_sans_affinage_f{e}",
                       f"python -u scripts/evaluer_prevision.py -c {V5} {OPTS} {forcage(e)} --checkpoint {INIT} "
                       f"--obs {OBS} --echeance {e} --sortie {d}")
            except RuntimeError as exc:                 # code 3 = PREV non conforme : resultat ecrit
                print(exc)

    # 3. affinage v5 (GFS +24 h en entree, meme verite) depuis le modele livre
    h = R5 / "history.json"
    if not (h.exists() and any(r.get("epoch") == 11 for r in json.loads(h.read_text()))):
        lancer("entrainer_v5", f"python -u -m pcfno3d -c {V5} {OPTS} {forcage(24)} --set train.init_from={INIT} train")

    # 4. PREV1 (+24 h), PREV2 (+48 h) avec la v5 ; PREV3 : decisions de la veille
    for e in (24, 48):
        d = OS7 / f"v5_f{e}"
        if not (d / "prevision_bilan.json").exists():
            try:
                lancer(f"prevision_v5_f{e}", f"python -u scripts/evaluer_prevision.py -c {V5} {OPTS} {forcage(e)} "
                                             f"--obs {OBS} --echeance {e} --sortie {d}")
            except RuntimeError as exc:
                print(exc)
        if (d / "prevision_paires.csv").exists() and not (d / "decisions" / "decisions_agricoles.json").exists():
            lancer(f"prev3_f{e}", f"python -u scripts/verifier_decisions.py --paires {d / 'prevision_paires.csv'} "
                                  f"--sources prevision,gfs,persistance --sortie {d / 'decisions'}")

    # 5. le champ 3D prevu contre la verite generee (test 2022), a +24 h
    if not (R5 / "eval_pcfno3d" / "best_wcont" / "test_summary.json").exists():
        lancer("evaluer_v5_f24", f"python -u -m pcfno3d -c {V5} {OPTS} {forcage(24)} evaluate")

    # 6. AGRI1 / AGRI2 du modele livre : decisions a station cachee (validation reelle) contre les stations
    paires = RUNS / livre / "validation_reelle" / "test" / "stations_paires.csv"
    if paires.exists() and not (RUNS / livre / "decisions_agricoles" / "decisions_agricoles.json").exists():
        lancer("agri1_agri2", f"python -u scripts/verifier_decisions.py --paires {paires} "
                              f"--sources modele,era5,generateur --sortie {RUNS / livre / 'decisions_agricoles'}")
    for f in sorted(OS7.glob("*/prevision_bilan.json")):
        print(f.parent.name, json.dumps(json.loads(f.read_text())["verdicts"].get("toutes", {}), ensure_ascii=False))
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
"""
    mem = MEM.replace("configs/grand_tunis_colab.yaml", "configs/grand_tunis_v5_gfs_colab.yaml")
    return carnet([
        md("""
# PC-FNO3D réel (Grand Tunis) — OS7 : prévision du vent 3D à +24 et +48 h (Colab, A100)

Le **même modèle**, nourri par la **prévision GFS** au lieu d'ERA5 (cahier v12, section 18.10) :

1. prévisions GFS archivées (NOAA, AWS, depuis 2021) aux instants du cache, à +24 h et +48 h :
   seuls 38 champs par échéance (~34 Mo), découpés aussitôt sur les mailles du Grand Tunis
   (`pcfno3d.gfs`) ; reprise par instant sur le Drive (`os7/`) ;
2. **référence** : le modèle livré nourri par GFS, **sans affinage** ;
3. **affinage v5** : GFS +24 h en entrée, la même vérité, le même découpage, aucune station en entrée ;
4. **PREV1** (+24 h) et **PREV2** (+48 h) aux stations de l'INM sur le test 2022, contre GFS brut et
   la persistance ; **PREV3** : décision de traitement de la veille contre la décision mesurée ;
5. le champ 3D prévu contre la vérité générée (DSO1 à DSO5 du test) ;
6. **AGRI1 / AGRI2** du modèle livré (décisions à station cachée contre les stations).

Le test 2022 n'est lu qu'une fois par étape ; une étape déjà faite est sautée. La session s'arrête à
la fin. Limite : la qualité dépend de celle de GFS (le modèle n'avance pas l'atmosphère lui-même).
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN + installe),
        md("## 2. Cache (3 parties) et observations"),
        code(ENT.replace("grand_tunis_colab", "grand_tunis_v2_long_colab") + OBS),
        md("## 3. Mémoire du GPU"),
        code(mem),
        md("## 4. GFS, référence, affinage v5, PREV1-3, test 3D, AGRI1-2, fin de la session"),
        code(chaine),
    ], gpu=True)


def main() -> int:
    out = REPO / "notebooks" / "current"
    for nom, nb in (("pcfno3d_grand_tunis_generation_colab.ipynb", generation()),
                    ("pcfno3d_grand_tunis_colab.ipynb", entrainement()),
                    ("pcfno3d_grand_tunis_v2_colab.ipynb",
                     entrainement("grand_tunis_v2_colab", "grand_tunis_v2")),
                    ("pcfno3d_grand_tunis_v2_long_colab.ipynb",
                     entrainement("grand_tunis_v2_long_colab", "grand_tunis_v2_long")),
                    ("pcfno3d_grand_tunis_v2_aneo_colab.ipynb",
                     entrainement("grand_tunis_v2_aneo_colab", "grand_tunis_v2_aneo")),
                    ("pcfno3d_grand_tunis_diagnostic_colab.ipynb", diagnostic()),
                    ("pcfno3d_grand_tunis_validation_colab.ipynb", validation()),
                    ("pcfno3d_grand_tunis_v3_colab.ipynb", affinage()),
                    ("pcfno3d_grand_tunis_v3_suite_colab.ipynb", suite_v3()),
                    ("pcfno3d_grand_tunis_v3b_colab.ipynb", v3b()),
                    ("pcfno3d_grand_tunis_atom_colab.ipynb", service_v2()),
                    ("pcfno3d_grand_tunis_v4_dso_colab.ipynb", v4_dso()),
                    ("pcfno3d_grand_tunis_os6_colab.ipynb", os6_reseau()),
                    ("pcfno3d_grand_tunis_os7_colab.ipynb", os7_prevision())):
        (out / nom).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(out / nom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
