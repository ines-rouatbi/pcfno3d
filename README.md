# PC-FNO3D — reconstruction du champ de vent 3D sur la Tunisie

Projet de fin d'études (ESPRIT, option Data) réalisé chez **Tukhnanutha**.
Objectif : reconstruire le champ de vent **(u, v, w)** sur toute la colonne atmosphérique
**0–1000 m** à partir du réseau de capteurs **ANEO** (mesures à ~25–30 m du sol), avec un
opérateur neuronal de Fourier 3D contraint par la physique.

Le contexte métier, les objectifs chiffrés (DSO) et l'état d'avancement vis-à-vis du
planning sont dans **[`tukhnanutha.md`](tukhnanutha.md)**. Le fonctionnement du code est
documenté dans **[`CLAUDE.md`](CLAUDE.md)**.

## Ce que fait le modèle

| | |
|---|---|
| Entrée | mesures de surface d'un instant (7 variables, capteurs partiellement manquants) |
| Sortie | volume 12 niveaux × 250 × 500 pour u, v, w — 1,5 million de nœuds |
| Grille | 100 m horizontal, niveaux ASL 25 → 1000 m |
| Contraintes | conservation de la masse, condition cinématique au sol, finesse spatiale |
| Cible | RMSE ≤ 0,5 m/s · R² ≥ 0,95 · résidu de masse ≤ 1e-4 · < 1 s par instant |

## Démarrage

```bash
pip install -e .                      # paquet + dépendances
pip install -e ".[serve,dev]"         # + API HTTP et outils de test
```

Validation complète sur un mini-dataset synthétique, sans Colab ni GPU (moins d'une minute) :

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke.ps1
```

Elle enchaîne : génération d'un mini-dataset → cache → entraînement (2 époques) →
évaluation DSO → reconstruction 3D → export web → tests unitaires.

## Pipeline sur les vraies données

```bash
python -m pcfno3d info                       # configuration effective, artefacts présents
python -m pcfno3d build-cache                # CSV journaliers -> tenseurs .npy  (une fois)
python -m pcfno3d check                      # géométrie et vérité terrain
python -m pcfno3d train                      # entraînement, reprise automatique
python -m pcfno3d evaluate                   # métriques de test + verdict DSO1-DSO5
python -m pcfno3d reconstruct --stem 004_120000 --altitude 120 --web
```

Toutes les commandes acceptent `-c/--config` (`default`, `test`, `colab`, ou un chemin) et
`--set section.cle=valeur` pour une surcharge ponctuelle :

```bash
python -m pcfno3d -c colab train --set train.epochs=200 --set model.width=64
```

Sur Colab, `configs/colab.yaml` garde le dataset et les résultats sur le Drive et le cache
sur le disque local de la machine :

```python
from google.colab import drive; drive.mount("/content/drive")
!pip install -e /content/drive/MyDrive/PFE
!python -m pcfno3d -c colab build-cache && python -m pcfno3d -c colab train
```

## Service d'inférence et visualiseur

```bash
pip install -r service/requirements.txt
python -m pcfno3d serve --port 8000          # http://127.0.0.1:8000/docs
```

| Route | Rôle |
|---|---|
| `GET /health`, `GET /meta` | état du service, description de la grille |
| `GET /instants` | instants disponibles en cache |
| `POST /reconstruct` | volume 3D d'un instant (+ export binaire optionnel) |
| `POST /reconstruct/upload` | reconstruction depuis un CSV de mesures de surface |
| `GET /field?z=120` | carte du vent à une altitude quelconque (interpolée) |
| `GET /point?x=&y=&z=` | vent en un point précis |

Image Docker CPU : `docker build -f service/Dockerfile -t pcfno3d-api .`

Le visualiseur `web/index.html` lit un export produit par `reconstruct --web`
(`manifest.json` + binaires Float32) :

```bash
python -m http.server 8080
# http://localhost:8080/web/?src=../artifacts/runs/v7/web/recon_004_120000
```

## Organisation du dépôt

```
src/pcfno3d/        le modèle et le pipeline (portage du notebook, un module par BLOC)
configs/            default.yaml (référence) · colab.yaml · test.yaml (mini-dataset)
service/            API HTTP d'inférence + Dockerfile
web/                visualiseur du volume reconstruit
scripts/            make_mini_dataset.py · smoke.ps1
tests/              30 tests, tous sur CPU et mini-dataset
notebooks/current/  fno3d_v7_1.ipynb — le notebook de référence (Colab)
notebooks/archive/  architectures explorées puis écartées (FNO2D, U-Net, GNN, DeepONet, ViT…)
notebooks/data_prep/01_EDA.ipynb · 02_cleaning.ipynb
docs/               présentations, notes techniques, Gantt, figures
artifacts/          cache, checkpoints, évaluations, exports — non versionné
```

Les données (`aneo_dataset_3d_v2/`, ~35 Go), les anciens jeux (`data_*`) et les archives
`.rar` restent à la racine mais sont exclus de Git : ce sont des artefacts du générateur,
pas des sources. Le générateur lui-même est dans
`aneo_physical_db_multitime_v3_progress/`.

## Correspondance notebook → modules

| Notebook | Module |
|---|---|
| BLOC 0/1 configuration | `config.py` |
| BLOC 2 grille, masques, opérateurs | `geometry.py` |
| BLOC 3/4 cache de tenseurs | `cache.py` |
| BLOC 5 normalisation | `normalize.py` |
| BLOC 6 features, dataset | `features.py`, `dataset.py` |
| BLOC 7 modèle | `model.py` |
| BLOC 8 loss physique | `losses.py` |
| BLOC 9/9b entraînement | `train.py` |
| BLOC 10 évaluation DSO | `evaluate.py` |
| BLOC 11/12 reconstruction, figures | `reconstruct.py`, `export.py`, `inference.py` |

Le notebook reste la référence pour l'expérimentation sur Colab ; le paquet est ce qui
s'exécute en production, se teste et se déploie.
