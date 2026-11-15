# Modèle national PC-FNO3D v4 — bilan final (16/11/2026)

Décision d'Ines (16/11/2026, « 1 ») : arrêt ici ; le produit livré est celui décrit ci-dessous. Chaque chiffre vient
d'un calcul fait sous une règle datée et hachée **écrite avant** le calcul ; les échecs sont publiés tels quels.
Vérité d'entraînement = générateur forcé par l'ERA5 (jamais des mesures) ; mesures = METAR, SYNOP, radiosondages.

## 1. Le modèle livré

- **C' s44** : PC-FNO3D 453 M paramètres, grille 16 × 828 × 388 (nœuds 1 km, 25–2500 m), entrée ERA5 10 m + 12
  aéroports METAR, résiduel sur l'ERA5, affiné 30 époques depuis la v2 (vérité v3, terme de rugosité).
- Choisi sur la **validation** (`regle_choix_v4_Cp_s44`, accord écrit d'Ines) : meilleur score des trois graines de C'
  (s44 0,5816 ; s42 0,5818 ; s43 0,5856 ; D s42 2,317).
- w servi = w par continuité + 0,75 (w du réseau − w par continuité), choisi sur la validation
  (`regle_w_servi_v4`) ; les trois graines donnent le même β et R² w 48/48.
- `decision_v4_national.json` : les huit conditions de `regle_v4_national` sont remplies, **C' s44 remplace la v2**.

## 2. DSO — test 2022, lu une seule fois (180 instants synoptiques)

| DSO | Seuil | Résultat | |
|---|---|---|---|
| DSO1 RMSE u, v par niveau | ≤ 0,5 m/s | max 0,285 ; global 0,214 / 0,222 | conforme |
| DSO1 R² u, v, w par niveau | ≥ 0,95 | min 0,952 (w, 2000 m) ; u, v ≥ 0,997 | conforme |
| DSO2 résidu de masse | ≤ 1e-4 | 2,4e-5 | conforme |
| DSO3 rugosité | 0,9–1,1 | 0,995 | conforme |
| DSO4 temps | < 1 s | 264 ms (A100) | conforme |
| DSO5 25 % des capteurs | ≤ 0,7 m/s | 0,286 | conforme |
| DSO6 incertitude | couverture 80 % | 0,85–0,86 ; Spearman w 0,58, u/v 0,14–0,15 | conforme |

**Ni sur-apprentissage ni sous-apprentissage** : erreur de validation encore décroissante à l'époque 30 ; scores du
test égaux à ceux de la validation (R² u, v ≈ 0,998) ; erreurs 0,21–0,29 m/s, loin du seuil.

## 3. Face à l'ERA5

| Juge | Modèle | ERA5 | Lecture |
|---|---|---|---|
| Vérité, 0–1 km, pire tranche RMSE u, v | **0,235** | 0,299 | modèle meilleur (−21 %) |
| Vérité, w 0–1 km (R²) | **0,975** | 0,382 | modèle 5 × plus juste |
| 12 aéroports réels, réseau seul (2022) | 2,844 | 2,857 | au niveau |
| 12 aéroports réels, **couche 10 m ERA5 + OI** (2023–2024, indépendant) | **2,639** | 2,739 | **meilleur (−3,7 %, IC [−0,108 ; −0,092], 9/12, pire +2,1 %)** |

Produit servi (option i d'Ines, `regle_produit_vent10m_v4`) : champ 3D de C' s44 et, à 10 m, la couche ERA5 + OI des
mesures des aéroports (`pcfno3d.vent10m_oi`, L 150 km, λ 4).

## 4. Ce qui a été essayé et n'a pas passé sa règle

| Essai | Résultat | Condition manquée |
|---|---|---|
| Modèle + OI des aéroports (validation 2021) | −3,1 % sur l'ERA5, 8/12 | Monastir +10,9 % (> 5 %) |
| MOS complet de l'ERA5 (validation 2021) | −7,6 %, 71 % des sites | un site à +24,7 % |
| Vérité corrigée, phase 1 (validation 2021, k 0,25) | −4,2 %, 10/12, 26/29 stations | Tozeur +5,3 % (> 5 %) : réentraînement non lancé |

## 5. Ce qu'on peut écrire dans le rapport

- Le modèle national est conforme aux six DSO et bat l'ERA5 face à la vérité de référence.
- Aux mesures réelles, le réseau reproduit sa vérité d'entraînement, dont la précision est celle de l'ERA5 : c'est la
  limite de la vérité, pas du réseau.
- La couche 10 m « ERA5 + interpolation optimale des aéroports » bat l'ERA5 sur une période indépendante (2023–2024).
- À ne pas écrire : que le réseau seul bat l'ERA5 aux stations ; que la couche 10 m vaut loin des aéroports (au-delà
  d'environ 300 km elle vaut l'ERA5) ; que le gain est vérifié avec un fond GFS (temps réel), non mesuré.

## 6. Perspectives

1. Vérité d'entraînement corrigée près des côtes et dans l'intérieur venteux (phase 1 : −4,2 %, Tozeur +5,3 %) —
   régénération et réentraînement (≈ 50 Go de Drive, A100).
2. Vérifier la couche 10 m avec le fond GFS du temps réel.
3. Brancher le modèle national dans le service ATOM Wind (aujourd'hui : Grand Tunis v4).
4. Graines D s43 (coupée à 28/30) et D s44 (non lancée) : sans effet sur le choix, utiles pour la phase 5 (réseau ANEO).

## Fichiers

`docs/resultats/national/` : `v4_test_2022.md`, `diag_leviers_aeroports.md`, `oi_post_v4_validation.md`,
`independant_2023_2024.md`, `verite_corrigee_phase1.md`, `mos_era5_resultats.md`.
Règles : `configs/national/regle_*.json` (+ `.sha256`). Drive : `MyDrive/pcfno3d_national/runs/national_1km_v4_Cp_s44/`.
