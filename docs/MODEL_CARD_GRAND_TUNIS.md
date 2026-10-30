# Model card — PC-FNO3D réel, Grand Tunis v4 (modèle livré)

*Livrables L3 et L7 (cahier des charges v12, § 9.1), sortie S12. Chaque chiffre est suivi de sa source.
Un critère sans mesure est marqué « à mesurer » ; aucun n'est présenté comme atteint sans résultat écrit.
La fiche de la v7 synthétique (référence de méthode) reste `docs/MODEL_CARD.md`.*

## Identité

| Élément | Valeur | Source |
|---|---|---|
| Modèle livré | `grand_tunis_v4_dso`, servi avec **w recalculé par continuité** | `runs/decision_v4.json` du 24/10/2026 (`modele_livre`, `w_continuite: true`), lu sur le Drive le 30/10 |
| Copie de production | `runs/grand_tunis_v9_production` (poids, `norm_stats.json`, évaluations, validation réelle, `decision_v4.json`, `PROVENANCE.txt`) ; la copie précédente (v2 prolongée) est mise de côté, jamais supprimée | `scripts/make_grand_tunis_nb.py` (cellule « production ») |
| Lignée | v7 synthétique → Grand Tunis v1 → v2 (entrée ERA5 aux nœuds, résiduel) → v2 prolongée (40 époques à 1e-4) → **v4** (affinage DSO, 15 époques) | `CLAUDE.md` (« Grand Tunis v2 », « v4 ») |
| Écartés | v3 et v3b (affinage sur les stations : correction de site, pas de gain aux stations jamais vues) ; v2 ANEO (couche dense simulée, moins bonne) | `CLAUDE.md` ; `scripts/bilan_plis_grand_tunis.py` |
| Règle de livraison | v4 livrée seulement si elle n'est pas moins bonne que la v2 sur DSO1, DSO4, DSO5 et les vraies mesures (tolérance 1 %, aucune station à plus de 3 %) ; fixée avant la lecture du test | `scripts/decision_v4_grand_tunis.py` |

## Domaine et données

| Élément | Valeur | Source |
|---|---|---|
| Domaine | 68 × 56 km, 36,395–36,900 N, 10,04–10,80 E ; nœuds à 100 m ; surface 50 m (Copernicus DEM, ESA WorldCover) | `CLAUDE.md` (« PC-FNO3D réel ») ; `scripts/grille_reelle.py` |
| Vérité d'entraînement | le générateur physique forcé par **ERA5** (pas des mesures) : brise de mer, grande échelle recalée sur ERA5 maille par maille, w par continuité | `scripts/forcage_era5.py`, `scripts/generate_aneo_3d.py --forcage` |
| Entrée de production | vent ERA5 à 10 m tous les 2,5 km + anémomètres réels (Tunis-Carthage METAR, Zaghouan et Nabeul SYNOP) + forçage ERA5 (7 scalaires) et profil ERA5 aux nœuds ; **en temps réel, GFS remplace ERA5** (mêmes formules, mêmes mailles) | `mask.entree: era5_stations` ; `pcfno3d.temps_reel` |
| Découpage | test = janv.–mars 2022 + juil.–août 2022 ; validation = janv.–févr. 2021 + juil. 2021 ; entraînement = le reste, 7 jours de marge | `CLAUDE.md` |
| Exclusions | Nabeul 2023–2024 (station abritée) et mois d'anémomètre en panne (Nabeul juin, juillet, septembre 2021) | `stations_reelles.EXCLUSIONS` |

## Architecture et entraînement

| Élément | Valeur | Source |
|---|---|---|
| Opérateur | FNO 3D de la v7 (largeur 48, modes 6 × 32 × 32, 4 couches) ; **résiduel ERA5** : u, v = ERA5 au nœud + correction apprise | `configs/grand_tunis_v4_dso_colab.yaml` |
| Entrée | 34 canaux (les 25 de la v7 + 7 scalaires de forçage + u, v ERA5 aux nœuds) | `CLAUDE.md` (« Grand Tunis v2 ») |
| Affinage v4 | depuis la v2 prolongée, 15 époques, lr 5e-5, λ masse 5, λ gradient 0,5, poids u, v par niveau mesurés sur la validation, sélection sur le score DSO de la validation | `configs/grand_tunis_v4_dso_colab.yaml` |
| Sortie servie | u, v du réseau ; **w recalculé par continuité** à partir de u, v | `inference.w_continuite` ; `Geometry.w_continuite` |

## Résultats (test : 180 instants synoptiques, jamais lus pour régler)

| Critère | v4 (livrée) | v2 prolongée | Seuil | Source |
|---|---|---|---|---|
| RMSE global u / v | 0,380 / 0,373 m/s | 0,377 / 0,371 m/s | — | `decision_v4.json` |
| Pire RMSE de niveau (u, v) | 0,528 m/s | 0,530 m/s | ≤ 0,5 m/s (DSO1) | idem : **DSO1 non atteint au niveau le plus bas** (25 m) |
| DSO2 résidu de masse (w par continuité) | 1,05·10⁻⁵ | 1,05·10⁻⁵ | ≤ 10⁻⁴ | idem (sans continuité : v4 1,25·10⁻⁴, v2 1,31·10⁻⁴) |
| DSO4 temps d'une carte 3D | 0,302 s | 0,302 s | < 1 s | idem |
| DSO5 RMSE avec 25 % des capteurs | 0,535 m/s | 0,537 m/s | ≤ 0,7 m/s | idem |
| DSO6 couverture à 80 % (u / v / w) | 0,878 / 0,883 / 0,841 ; Spearman 0,16 / 0,17 / 0,50 | — | ≥ 80 %, corrélation positive | `decision_v4.json` (bloc `dso6`) : **conforme** |
| Vraies stations, 10 m, station exclue de l'entrée | moyenne 2,469 m/s (Nabeul 2,42 ; Tunis-Carthage 2,70 ; Zaghouan 2,29) | 2,467 m/s | ≤ v2 + 1 % | idem ; ERA5 au même point : 2,57 / 2,81 / 2,16 (v2, 23/10) |
| DSO3 rugosité | non relue dans la décision | 0,82 (v2) | [0,9 ; 1,1] | `CLAUDE.md` : **à confirmer pour la v4** |
| REEL2 / EOL2 radiosondages | à lire dans `bilan_conformite.md` | v2 : modèle meilleur qu'ERA5 sur 0–1 000 m | ≤ ERA5 | `scripts/bilan_conformite.py` |

Le bilan complet, critère par critère du cahier v12, est produit par `scripts/bilan_conformite.py`
(fin des notebooks OS6 et OS7) dans `runs/bilan_conformite.md`.

## Limites

- **La vérité est générée** (générateur forcé par ERA5) : aux stations, le modèle est au niveau d'ERA5 ;
  la limite est la vérité d'entraînement, pas le réseau (`CLAUDE.md`, validation réelle du 23/10).
- **Trois stations seulement** dans le domaine : loin d'elles, la valeur vient surtout de la grande
  échelle et du relief (voir l'influence des stations, E10/E11).
- **Temps réel** : la grande échelle vient de GFS au lieu d'ERA5 ; le modèle n'a pas été entraîné sur
  GFS (c'est l'objet d'OS7, affinage v5, pas encore lancé).
- Le niveau le plus bas (25 m) reste au-dessus du seuil DSO1 (0,53 m/s).
- Les rafales affichées par l'application sont un facteur heuristique, pas une sortie du modèle.

## Usages

Espace agriculteur (créneaux de traitement et d'irrigation, J0 à J+2), démonstration éolienne
(production d'éoliennes de référence, atlas), carte et vue 3D du vent, API REST et serveur MCP.
Pas d'usage réglementaire ni de sécurité des personnes sans vérification.

## Reproduire

1. Données : `scripts/grille_reelle.py`, `scripts/forcage_era5.py`, `scripts/generate_aneo_3d.py --forcage --cache-dir`.
2. Entraînement : notebooks `pcfno3d_grand_tunis_colab.ipynb` (v1), puis v2, v2 prolongée et
   `pcfno3d_grand_tunis_v4_dso_colab.ipynb` (A100).
3. Décision : `scripts/decision_v4_grand_tunis.py --runs <runs> --dso6 <verdict>`.
4. Service : `pcfno3d_grand_tunis_atom_colab.ipynb` (L4), qui copie le modèle désigné par la décision.
5. Le cache porte son manifeste (`pcfno3d manifeste-cache --verifier`).
