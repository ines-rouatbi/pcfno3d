# Contrôle qualité du jeu v3 national — résultat (10/11/2026, Colab CPU, aucun entraînement)

Protocole : `plan_v4_controle_et_place.md` § 1. Script `scripts/controle_jeu_v3.py` au commit `edfb2e9`, carnet
`notebooks/current/pcfno3d_national_controle_v3_colab.ipynb` (commit `cc8aea6`). Sortie complète :
`MyDrive/pcfno3d_national/controle_v3/bilan_controle_v3.json` et `paires_aeroports_10.csv`.
La vérité v3 est une **vérité d'entraînement** (générateur forcé par ERA5), pas une mesure.

## 1. Manifeste du cache : CONFORME

- `pcfno3d manifeste-cache --verifier` : conforme (code 0), `cache_manifest.json` écrit.
- 1000 instants : train 700, val 120, test 180, égaux aux nombres attendus.
- Calendrier de la v2 relu, aucun instant hors de son jeu, aucun fichier manquant, aucun instant non fini.

## 2. Conservation de la masse : conforme, au niveau de la v2

Résidu de masse (`geom.mass_residual` sur `geom.mass_mask`), w du cache :

| vérité | jeu | instants | RMS moyen | RMS max | max absolu |
|---|---|---|---|---|---|
| v3 | train | 700 | 1e-6 | 3e-6 | 4,4e-5 |
| v3 | val | 120 | 1e-6 | 3e-6 | 3,6e-5 |
| v3 | test | 180 | 1e-6 | 3e-6 | 4,7e-5 |
| v2 (référence) | val | 120 | 1e-6 | 3e-6 | 4,8e-5 |
| v2 (référence) | test | 180 | 1e-6 | 3e-6 | 1,1e-4 |

## 3. Dix instants de validation régénérés (graine 2031) : identiques au cache

- Instants : 0001_060000, 0034_180000, 0042_180000, 0182_000000, 0184_150000, 0191_000000, 0191_060000,
  0195_180000, 0206_210000, 0212_180000.
- Générateur relancé avec les réglages v3 (passe-haut, perturbations 0, brise 0,5, relief 1, β 0,5).
- Code de sortie 0. **Écart maximal sur Y = 0,0 pour les 10 instants** (bit à bit).

## 4. Chiffres du rapport recalculés sur ces 10 instants

Écart RMS vérité − ERA5 (m/s), à la même hauteur au-dessus du sol :

| | u 300 m | u 800 m | v 300 m | v 800 m |
|---|---|---|---|---|
| cache v3 | 0,2542 | 0,2424 | 0,3007 | 0,2573 |
| chemin de la phase 2 (`compute_fields`) | 0,2542 | 0,2424 | 0,3007 | 0,2573 |
| rapport d'acceptation (20 autres instants) | 0,370 | 0,274 | 0,434 | 0,320 |

- **Le cache et le chemin du rapport donnent les mêmes chiffres** (écart < 1e-5 m/s) : le jeu v3 est bien celui
  que le rapport a mesuré.
- Les valeurs sont plus basses que celles de l'acceptation, mais **ce ne sont pas les mêmes instants**. On ne
  compare donc que l'ordre de grandeur. Le rapport v/u reste le même à 300 m (1,18 ici, 1,17 à l'acceptation).
  À 800 m, il vaut 1,06 ici contre 1,17.

Aéroports (110 paires instant × station, METAR à 30 min au plus), RMSE vectoriel à 10 m (m/s) :

| | vérité (surface du cache, bruit de mesure simulé) | vérité (chemin de la phase 2) | ERA5 seule |
|---|---|---|---|
| 10 instants de validation | 3,160 | 3,160 | 3,138 |
| acceptation (20 instants) | 2,925 | | 2,940 |

- Sur ces 10 instants, la vérité est **0,7 % moins bonne** qu'ERA5 aux aéroports. À l'acceptation, elle était
  0,5 % meilleure. Dans les deux cas, l'écart à ERA5 est inférieur à 1 %, et 110 paires ne suffisent pas à
  distinguer les deux.
- Ce n'est pas une mesure de la qualité d'un modèle. C'est le contrôle que la vérité v3 reste au niveau d'ERA5
  aux stations.

## Verdict

**Le jeu v3 est conforme et cohérent** :
- son manifeste est conforme ;
- la masse est conservée comme en v2 ;
- la régénération reproduit le cache bit à bit ;
- le cache redonne exactement les chiffres du chemin du rapport.

Aucun entraînement n'a été lancé. La v4 (règle `6e344f0d…`) attend l'accord écrit d'Ines.
