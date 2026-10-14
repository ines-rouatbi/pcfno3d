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

## État du modèle : v7 et corrections v8

La v7 (Colab, `notebooks/current/fno3d_v7.ipynb`) respecte **5 DSO sur 6**.
Seul le DSO1 — RMSE ≤ 0,5 m/s **à chaque niveau** — échoue, en haut de la colonne :
u = 0,564 / 0,632 / 0,665 m/s à 650 / 800 / 1000 m. Tout passe jusqu'à 500 m ; R² ≥ 0,991
partout ; masse 2,6e-5, rugosité 0,95, inférence 91 ms, robustesse 0,646 m/s.

Les corrections v8 sont dans le paquet, désactivées par défaut (la v7 reste reproductible) :

| option | cible |
|---|---|
| `evaluate --tta 8` | moyenne de passes **sans réentraîner** |
| `loss.uv_level_weight` | aligner la loss sur un maximum par niveau, poids mesurés sur la **validation** |
| `mask.p_full` | la v7 est pire avec 100 % des capteurs qu'avec 50 % |
| `train.init_from` + `select_by: dso` | affiner depuis la v7, garder le meilleur score DSO |

`notebooks/current/pcfno3d_v8_colab.ipynb` les exécute de la moins chère à la plus chère,
**après avoir vérifié que le paquet reproduit les chiffres v7**. Elles ne sont pas encore
mesurées sur les vraies données : rien ne dit encore qu'elles suffisent.

### Servir la v7 en attendant la v8

Deux voies, le même modèle :

| voie | vitesse | il faut |
|---|---|---|
| **Colab** (recommandée) | 0,091 s par instant (GPU) | exécuter `pcfno3d_api_colab.ipynb`, puis `pcfno3d link set <URL> --key <clé>` |
| **ce poste** | 13,7 s par instant (CPU, 5 Go) | télécharger du Drive `aneo_fno3d_runs/v7/best_dso_v7.pt` et `norm_stats.json` |

Pour la voie locale :

```bash
python scripts/install_v7_weights.py C:/Users/inesr/Downloads   # vérifie PUIS installe
python -m pcfno3d -c local_v7 serve --port 8000
```

L'installeur refuse un fichier tronqué, des poids d'une autre architecture
(chargement strict) ou des statistiques incohérentes, et fait une passe d'essai sur un vrai
instant avant de copier quoi que ce soit : un téléchargement raté ne remplace jamais un
modèle qui fonctionne. Sur CPU, le DSO4 (< 1 s) n'est pas tenu ; les valeurs, elles, sont
celles de la v7.

## Application web, API et serveur MCP

Un seul processus sert l'interface, l'API JSON et le serveur MCP :

```bash
pip install -e ".[serve]"
pcfno3d apikey create app-web --scope read   # le secret s'affiche UNE seule fois
pcfno3d serve --port 8000
```

| | |
|---|---|
| Application web | `http://127.0.0.1:8000/app/` |
| API interactive | `http://127.0.0.1:8000/api/docs` |
| Serveur MCP | `http://127.0.0.1:8000/mcp` |

L'application web donne : carte du vent à n'importe quelle altitude (clic = lecture du
point), profil vertical, verdict DSO, et la page d'intégration qui affiche l'URL MCP et
les exemples `curl` prêts à copier.

**Clés d'API** — seul le hash SHA-256 est conservé, dans `run_dir/api_keys.json` ; le
secret n'est affiché qu'à la création. Deux portées : `read` (lecture, reconstruction) et
`admin` (+ écriture d'exports, gestion des clés). La clé passe par `X-API-Key`,
`Authorization: Bearer`, ou `?api_key=`. Sans aucune clé, le service répond sans
authentification et le signale dans `/api/health` — confortable en local, à ne pas
exposer au réseau.

```bash
pcfno3d apikey list        # noms, portées, empreintes — jamais de secret
pcfno3d apikey revoke app-web
```

**Serveur MCP** — pour qu'une autre application ou un agent interroge le vent sans rien
savoir du modèle. Outils : `grid_meta`, `list_instants`, `wind_at_point`, `wind_profile`,
`wind_field_at_altitude`, `instant_summary`, `dso_report`, `export_instant` (admin).
Transport streamable HTTP sur `/mcp`, ou stdio avec `pcfno3d mcp`.

**Pour brancher une autre application, lire [`docs/INTEGRATION.md`](docs/INTEGRATION.md)** :
les trois interfaces, le format binaire, les conventions (direction météo, altitudes ASL,
`-9999` sous le relief) et les limites actuelles.

Image Docker CPU : `docker build -f service/Dockerfile -t pcfno3d-api .`

### Servir depuis Colab (le GPU et les poids y sont déjà)

Plutôt que de descendre les checkpoints, on ouvre le modèle là où il est entraîné : une
URL publique donne accès à l'API, au MCP et à l'application web, et une autre application
envoie ses requêtes directement au vrai modèle.

```bash
python scripts/package_for_colab.py      # artifacts/pcfno3d_code.zip, 106 Ko, code seul
```

Dépose l'archive à la racine de ton Drive, puis ouvre
[`notebooks/current/pcfno3d_api_colab.ipynb`](notebooks/current/pcfno3d_api_colab.ipynb)
dans Colab. Le cœur du notebook tient en un appel :

```python
from service.colab import serve_public
info = serve_public(config="colab", key_name="app-externe", scope="read")
# -> URL publique, clé d'API, URL MCP, déclaration client prête à copier
```

Il vérifie les poids, crée la clé, ouvre un tunnel Cloudflare, démarre uvicorn et attend
`/api/health`. Le tunnel est ouvert **avant** le service : son nom d'hôte est tiré au
hasard et le transport MCP refuse un `Host` non déclaré.

Deux limites à dire franchement : la session Colab est **éphémère** (inactivité ~90 min,
12 h au maximum) et l'URL change à chaque relance — ce n'est pas un hébergement ; et le
tunnel est **public**, donc c'est la clé d'API qui protège les données, pas l'URL.

### Relier l'application locale au modèle distant

Pour que l'URL utilisée par les applications et les clients MCP **ne change pas** quand le
tunnel Colab change, le service local peut se mettre en relais : aucun modèle chargé ici,
tout `/api` et `/mcp` partent vers l'instance distante.

```bash
pcfno3d link set https://xxx.trycloudflare.com --key pcf_...   # vérifie l'amont
pcfno3d serve                                                  # /app/ sert les données de Colab
pcfno3d link show                                              # état et santé de l'amont
pcfno3d link clear                                             # revenir au modèle local
```

Deux clés, et c'est voulu :

```
navigateur --(clé LOCALE)--> service local --(clé AMONT)--> Colab
```

La clé de l'instance Colab reste sur cette machine : le navigateur ne la voit jamais, et
révoquer une clé locale ne touche pas à l'amont. Quand la session Colab redémarre, une
seule commande est à refaire — `link set` avec la nouvelle URL. Si l'amont est tombé,
`/api/health` passe en `degraded` et les requêtes répondent `502` avec la commande de
réparation dans le message.

Un visualiseur autonome, sans serveur, lit aussi un export de
`reconstruct --web` :

```bash
python -m http.server 8080
# http://localhost:8080/web/?src=../artifacts/runs/v7/web/recon_004_120000
```

## Organisation du dépôt

```
src/pcfno3d/        le modèle et le pipeline (portage du notebook, un module par BLOC)
configs/            default.yaml (référence) · colab.yaml · test.yaml (mini-dataset)
service/            app.py · mcp_server.py · auth.py · link.py · colab.py · Dockerfile
webapp/             l'application web servie sur /app/
web/                visualiseur autonome d'un export, sans serveur
scripts/            make_mini_dataset.py · smoke.ps1 · package_for_colab.py · install_v7_weights.py
tests/              80 tests, tous sur CPU et mini-dataset
notebooks/current/  fno3d_v7_1 (référence) · fno3d_v7 (LA v7, avec ses sorties Colab) · pcfno3d_v8_colab · pcfno3d_api_colab
notebooks/archive/  architectures explorées puis écartées (FNO2D, U-Net, GNN, DeepONet, ViT…)
notebooks/data_prep/01_EDA.ipynb · 02_cleaning.ipynb
docs/               INTEGRATION.md · REPRODUCTIBILITE.md (DE2) · présentations, notes techniques, Gantt, figures
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
