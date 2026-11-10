# Phase 1 nationale : diagnostics complémentaires 5, 6 et 7 (10/11/2026)

Demande d'Ines du 10/11. Règle écrite et hachée avant tout calcul :
`configs/national/regle_phase1_complements.json`, sha256 `752e16ca…a8a105` (commit 003de24), vérifiée par chaque
script. Aucun entraînement, aucune génération complète. ERA5 et la vérité générée sont des vérités d'entraînement,
pas des mesures.

## 5. Cohérence de l'ERA5 : modèle contre générateur — COHÉRENT, aucun bug

`scripts/diagnostic_coherence_era5.py`, sur ines-pc, avec 20 instants de validation 2021 (graine 2028).

- **Côté modèle :** `ForcageERA5.vecteur` puis `InterpERA5.dense_np`, c'est-à-dire les canaux ERA5 de l'entrée et la
  base du résiduel.
- **Côté générateur :** `DayStateERA5.champ_cible` puis `_era5_au_sol`, c'est-à-dire le vent du recalage de grande
  échelle.

| Mesure | Résultat |
|---|---|
| RMSE modèle − générateur, u et v, chacun des 16 niveaux et à 300 / 800 m au-dessus du sol | ≤ **1,4·10⁻⁶ m/s** (arrondi float32) |
| Biais | ≤ 7·10⁻⁹ m/s |
| Position des nœuds (projection du modèle contre grille du générateur) | ≤ 0,07 m |
| Relief des nœuds, hauteur au-dessus du sol utilisée | identiques (écart 0) ; les deux côtés interpolent **au-dessus du sol**, plancher 5 m |
| Ordre des latitudes | fichier brut du nord au sud, retrié du sud au nord dans le npz ; le modèle et le générateur lisent le même npz |
| u10 et v10 du npz contre le fichier brut `era5_sl_2021_m01-06.nc`, 3 mailles × 3 instants | écart 0,000 m/s ; même signe de v |
| Instant UTC | temps du npz = `valid_time` du fichier brut ; horodatage du cache = jour du générateur + heure |

Le modèle reçoit exactement l'ERA5 qui sert au recalage de la vérité. L'erreur sur v ne vient pas d'un décalage
entre les deux.

## 6. Forme de l'erreur sur v — ce n'est pas une correction uniforme par instant

`scripts/diagnostic_forme_erreur_v.py`, sur Colab L4, avec le modèle national v2 et les 180 instants synoptiques du
test 2022. Par instant, on ajuste e = a + b·y + c·x sur l'erreur (modèle − vérité).

| Hauteur au-dessus du sol | Composante | Part de l'énergie expliquée par le plan | Expliquée par la moyenne seule | Médiane par instant (plan) | RMSE |
|---|---|---|---|---|---|
| 300 m | v | **0,321** | 0,028 | 0,194 | 0,837 |
| 800 m | v | **0,307** | 0,024 | 0,220 | 0,876 |
| 300 m | u | 0,080 | 0,018 | 0,062 | 0,592 |
| 800 m | u | 0,080 | 0,021 | 0,063 | 0,593 |

- **Verdict de la règle :** l'erreur sur v n'est pas une correction de grande échelle par instant, car le plan
  explique moins de 50 %.
- **La moyenne seule n'explique que 2 à 3 %.** L'erreur n'est donc pas un décalage uniforme de v.
- **La pente du plan, nord-sud et est-ouest, explique environ 30 % de l'énergie sur v, contre 8 % sur u**, soit
  quatre fois plus. Le reste (environ 70 %) n'est pas linéaire. Le diagnostic 2 place pourtant 78 à 85 % de
  l'énergie de l'erreur sur v au-delà de 200 km le long de y : l'erreur est donc courbe, mais à grande échelle.
- **Une seule corrélation dépasse |ρ| ≥ 0,3 :** la moyenne de l'erreur sur v à 300 m avec le cosinus de la
  direction du vent géostrophique (ρ = −0,32). Plus le vent vient du nord, plus le modèle sous-estime v en moyenne.
- **Corrélations sous le seuil, non signalées :**
  - moyenne à 300 m avec le mois (−0,27) et avec la vitesse géostrophique (+0,18) ;
  - à 800 m, avec la vitesse géostrophique (+0,26) et le sinus de la direction (−0,24) ;
  - pente nord-sud à 300 m avec l'heure (−0,20) ;
  - pente est-ouest à 300 m avec le cosinus de la direction (+0,27).

## 7. Mécanismes du générateur et correction de la phase 2, à instants égaux

**Diagnostic 4** (12 instants de validation à 3 h et 15 h) : `docs/resultats/national/phase1_diag23/verdict_phase1_diag234.md`.
L'amortissement de w est le seul mécanisme au-dessus de 30 %. La correction passe-haut fait passer l'écart sur v de
1,461 à 0,555 m/s à 300 m (−62 %) et de 1,381 à 0,456 m/s à 800 m (−67 %).

**Mise en garde sur les chiffres « 1,46 → 0,86 ».** Ces deux valeurs viennent de deux jeux d'instants différents :
le 1,46 du diagnostic 4 et le 0,86 de l'acceptation. Elles ne se comparent pas. Le générateur de la v2 a donc été
mesuré sur les mêmes instants que la calibration et l'acceptation (`scripts/calibrer_verite_v3.py --reference`,
`docs/resultats/national/verite_v3_reference_v2.csv`).

| Jeu (20 instants de validation) | Générateur | RMSE aéroports (ERA5 seule) | Écart v 300 m | v 800 m | u 300 m | u 800 m |
|---|---|---|---|---|---|---|
| Calibration (graine 2026) | v2 (actuel) | 2,981 (2,700) | 2,601 | 2,389 | 1,945 | 1,613 |
| | passe-haut, perturbations 0, brise 0,5 | 2,721 | 0,891 | 0,687 | 0,845 | 0,596 |
| | **+ repli β = 0,5** | **2,685** | **0,449** | **0,355** | 0,426 | 0,314 |
| Acceptation (graine 2027, jours disjoints) | v2 (actuel) | 3,114 (2,940) | 2,240 | 2,121 | 1,540 | 1,261 |
| | passe-haut, perturbations 0, brise 0,5 | 2,948 | 0,861 | 0,615 | 0,732 | 0,518 |
| | **+ repli β = 0,5** | **2,925** | **0,434** | **0,320** | 0,370 | 0,274 |

Sur l'acceptation, l'écart sur v passe donc de 2,24 à 0,86 (−62 %), puis à 0,43 (−81 %) avec β = 0,5. Aux
aéroports, la vérité de la v2 était **moins bonne qu'ERA5** (3,114 contre 2,940 m/s). La vérité β = 0,5 est
meilleure (2,925 m/s) et a été acceptée par la règle de repli.

**Ce que la correction change exactement dans `scripts/generate_aneo_3d.py`** (`compute_fields`) :

1. **`ANEO_GEN_AMORT_PASSE_HAUT=1`**, à l'étape 2 (amortissement vertical de w).
   - À chaque niveau, le générateur ajoute un écoulement potentiel horizontal grad(χ)/ρ. χ est obtenu par Poisson,
     par FFT périodique sur tout le domaine, et retire une fraction du flux vertical hérité du niveau inférieur.
   - Ce terme arrive **après** le recalage sur ERA5. Sa partie de grande échelle déplaçait donc la vérité loin d'ERA5.
   - La correction soustrait sa partie lisse : `gx -= passe_bas(gx, 12 km)`, et de même pour gy. Seules les échelles
     de moins de 12 km restent, et w est réintégré ensuite par la continuité, qui reste exacte.
   - C'est presque toute la baisse. Sur la calibration, les autres réglages de la variante choisie ne déplacent
     l'écart sur v à 300 m que de 0,918 à 0,891 (perturbations 0,4 → 0) et de 0,8917 à 0,8914 (brise 1 → 0,5).
2. **Réglages choisis sur la calibration** par la règle `regle_verite_v3.json` : facteur de perturbations 0
   (`ANEO_GEN_PERTURB`), brise 0,5 (`ANEO_GEN_BRISE`), relief 1 (`ANEO_GEN_RELIEF`). Cette variante a été
   **refusée** à l'acceptation (2,948 contre ERA5 2,940).
3. **Repli `ANEO_GEN_DETAIL=0,5`**, après l'amortissement et avant l'intégration finale de w.
   - Le générateur calcule u, v = ERA5 + 0,5 × (généré − ERA5) sur le volume et sur la surface, puis w par la
     continuité.
   - Le détail local (relief, brise, côte) est gardé à moitié autour d'ERA5.
   - Règle `regle_verite_v3_repli.json` : **acceptée**, 2,925 contre 2,940 m/s aux aéroports.

## Conclusion

- **Aucun bug d'ERA5** (diagnostic 5).
- **L'erreur du modèle sur v** est à grande échelle, mais ce n'est ni un décalage uniforme ni seulement un plan.
  Elle est la même à l'entraînement et au test (diagnostic 3).
- **Le générateur** ajoutait à la vérité, après le recalage sur ERA5, un écart à grande échelle que le modèle ne
  pouvait pas déduire de ses entrées (diagnostic 4 et point 7). La correction passe-haut et le repli β = 0,5
  réduisent cet écart de 81 % à instants égaux. La vérité devient alors meilleure qu'ERA5 aux aéroports.
- **La suite (phase 3, ablation C)** consiste à régénérer la vérité v3 avec β = 0,5 et à entraîner la v4. Elle
  attend l'accord écrit d'Ines.
