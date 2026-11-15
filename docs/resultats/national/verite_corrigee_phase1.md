# Vérité corrigée — phase 1 (faisabilité, CPU) : ÉCHEC selon la règle

Règle `configs/national/regle_verite_corrigee_phase1.json` (sha256 `3902185e…`, écrite et committée avant le calcul,
accord d'Ines « corrige la vérité et réentraîne »), `scripts/verite_corrigee_phase1.py`, 15–16/11/2026. Correction du
vent ERA5 à 10 m avec les seules variables disponibles sur la grille nationale (HistGB de regle_mos_era5, sans blh,
rayonnement ni nébulosité), réduite d'un facteur k. 2022 et 2023–2024 non lues. Incident : la première passe s'est
arrêtée sur une erreur de type de date dans le bootstrap, après l'écriture des prédictions 2021 (déterministes,
graine 2030, k = 0,25 imprimé dans le journal) ; la seconde passe a refait le choix de k (même résultat) et relu ces
prédictions.

## Choix de k (2015–2020, chaque site retiré de l'ajustement, 28 sites)

| k | RMSE vect. moyen | ERA5 | pire dégradation |
|---|---|---|---|
| 1 | 2,690 | 2,909 | +24,5 % |
| 0,75 | 2,666 | 2,909 | +16,6 % |
| 0,5 | 2,696 | 2,909 | +9,7 % |
| **0,25 (retenu)** | 2,779 | 2,909 | +4,1 % |

## Jugement 2021 (site retiré)

| | ERA5 | corrigé (k 0,25) | écart | IC 95 % | améliorés | pire dégradation |
|---|---|---|---|---|---|---|
| 12 aéroports | 2,793 | 2,676 | −0,117 (−4,2 %) | [−0,123 ; −0,111] | 10 / 12 | **+5,3 % (Tozeur)** |
| 29 stations | 2,882 | 2,759 | −0,122 (−4,2 %) | [−0,128 ; −0,116] | 26 / 29 | +5,3 % |

Conditions : (1) oui, (2) oui, **(3) non** (Tozeur 3,075 contre 2,919, +5,3 % > 5 %). **Phase 1 non réussie : la
phase 2 (vérité corrigée, réentraînement) n'est pas lancée** ; aucune unité Colab dépensée. La règle n'est pas
modifiée après coup.

Lecture : une correction calculable sur la grille gagne ~4 % en moyenne et améliore presque toutes les stations,
mais elle reste un peu trop forte à Tozeur (intérieur, venteux), comme le MOS complet (+24,7 % au même endroit).
