"""Fabrique les carnets Colab du modele reel (Grand Tunis).

    python scripts/make_grand_tunis_nb.py
    -> notebooks/current/pcfno3d_grand_tunis_generation_colab.ipynb  (jeu, environnement CPU)
       notebooks/current/pcfno3d_grand_tunis_colab.ipynb             (entrainement, A100)
       notebooks/current/pcfno3d_grand_tunis_diagnostic_colab.ipynb  (diagnostic sur la
           validation, CPU : scripts/diag_grand_tunis.py)
       notebooks/current/pcfno3d_grand_tunis_v2_colab.ipynb       (v2 mode reel, A100)
       notebooks/current/pcfno3d_grand_tunis_v2_aneo_colab.ipynb  (v2 mode ANEO simule, A100)

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


def carnet(cellules: list[dict], gpu: bool) -> dict:
    meta = {"kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}, "colab": {"provenance": []}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "A100"
    else:
        meta["colab"]["machine_shape"] = "hm"         # CPU, memoire RAM elevee
    return {"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": cellules}


COMMUN = r'''
from google.colab import drive
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


def main() -> int:
    out = REPO / "notebooks" / "current"
    for nom, nb in (("pcfno3d_grand_tunis_generation_colab.ipynb", generation()),
                    ("pcfno3d_grand_tunis_colab.ipynb", entrainement()),
                    ("pcfno3d_grand_tunis_v2_colab.ipynb",
                     entrainement("grand_tunis_v2_colab", "grand_tunis_v2")),
                    ("pcfno3d_grand_tunis_v2_aneo_colab.ipynb",
                     entrainement("grand_tunis_v2_aneo_colab", "grand_tunis_v2_aneo")),
                    ("pcfno3d_grand_tunis_diagnostic_colab.ipynb", diagnostic())):
        (out / nom).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(out / nom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
