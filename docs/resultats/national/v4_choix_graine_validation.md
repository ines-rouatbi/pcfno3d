# v4 nationale — choix de la graine (validation seulement), 13/11/2026

Règle `configs/national/regle_v4_national.json` (sha256 `60671e84…`), appliquée telle quelle : choix = minimum de
`val_score` (`history.json`) parmi les **6** entraînements, sur la **validation** de la vérité d'entraînement v3
(64 instants). Le test n'est pas lu. Lu sur le Drive le 13/11 (`runs/national_1km_v4_<cand>_s<g>/history.json`,
`info_capteurs_validation.json`).

| Entraînement | Meilleure époque | Score DSO (`val_score`) | RMSE max (niveau ASL) | R² global u / v / w | r2_min (niveau) | Bande 0-1 km propre (ERA5 seule) | Bande 0-1 km ERA5 dégradée (ERA5 seule) |
|---|---|---|---|---|---|---|---|
| C' s42 | 30 | 0,58177 | 0,2909 m/s (2500 m) | 0,99858 / 0,99816 / 0,96581 | 0,8323 (niveau : à mesurer) | 0,2533 (0,3233) | 1,8975 (1,8929) |
| C' s43 | 30 | 0,58562 | 0,2928 m/s (2500 m) | 0,99856 / 0,99815 / 0,96584 | 0,8322 (niveau : à mesurer) | 0,2554 (0,3233) | 1,8998 (1,8929) |
| **C' s44** | 30 | **0,58164** | 0,2908 m/s (2500 m) | 0,99857 / 0,99815 / 0,96600 | 0,8325 (niveau : à mesurer) | 0,2547 (0,3233) | 1,8976 (1,8929) |
| D s42 | 27 | 2,317 | 1,159 m/s (25 m) | 0,983 / 0,979 / 0,937 (ép. 30) | — | 0,9395 (0,3233) | 1,3544 (1,8929) |
| D s43 | coupé à 28/30 (12/11) | ≤ 2,281 (essai coupé, non retenu) | — | — | — | — | — |
| D s44 | pas lancé | — | — | — | — | — | — |

C' : moyenne des scores 0,5830, écart-type 0,0022 (3 graines) : reproductible.

**Choix selon la règle : pas encore définitif.** La règle compare les 6 entraînements ; D s43 et D s44 manquent.
En l'état, la meilleure est **C' s44 (0,58164)**, devant C' s42 (0,58177, écart 0,02 %). D s42 est à 2,317 :
pour changer le choix, un D devrait gagner un facteur 4. Le choix définitif attend D s43 et D s44, ou une décision
écrite d'Ines (nouvelle règle datée et hachée).

**DSO1-R² par niveau non conforme sur la validation, pour les trois graines** : r2_min ≈ 0,832 < 0,95
(rapport `r2_uvw` ≈ 3,35). Ce critère n'est pas dans le score de sélection (`dso1_rmse`, `dso2_masse`,
`dso3_rugosite`). Le niveau et la composante en cause seront donnés par `scripts/r2_niveaux_v4.py`
(règle `regle_postraitement_r2_v4.json`, `c82dff46…`), avec le même calcul après post-traitement.

Sur-apprentissage / sous-apprentissage : pour les trois graines, le score de validation baisse à chaque époque
jusqu'à la 30e (C' s44 : 3,695 → 2,538 → … → 0,582), la perte de validation suit la perte d'entraînement
(≈ 0,038 contre ≈ 0,029) sans remonter : pas de sur-apprentissage visible ; R² u, v ≈ 0,998.
Capteurs simulés : gain nul pour C' (1e-7 m/s), attendu (C' ne voit pas de capteurs à l'entraînement).
ERA5 et la vérité v3 sont la vérité d'entraînement, pas des mesures.
