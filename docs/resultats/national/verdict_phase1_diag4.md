# Phase 1 nationale : verdicts selon la regle fixee avant les calculs

Regle : `configs/national/regle_phase1_diagnostics.json`, sha256 `298232278ce441282c7aea38abd1d7d705e29c6e7469781cfc9302c5af0bd1da` (verifie).

## Diagnostic 2 : spectres de l'erreur (test 2022)
Non execute.

## Diagnostic 3 : entrainement contre test
Non execute.

## Diagnostic 4 : mecanismes du generateur (12 instants de validation 2021)
| Configuration | u 300 m | v 300 m | u 800 m | v 800 m |
|---|---|---|---|---|
| actuel | 1.188 (+0 %) | 1.461 (+0 %) | 1.150 (+0 %) | 1.381 (+0 %) |
| passe_haut | 0.533 (+55 %) | 0.555 (+62 %) | 0.414 (+64 %) | 0.456 (+67 %) |
| sans_amort | 0.252 (+79 %) | 0.264 (+82 %) | 0.151 (+87 %) | 0.155 (+89 %) |
| sans_brise | 1.186 (+0 %) | 1.455 (+0 %) | 1.149 (+0 %) | 1.380 (+0 %) |
| sans_pente | 1.180 (+1 %) | 1.433 (+2 %) | 1.186 (-3 %) | 1.403 (-2 %) |
| sans_perturb | 1.182 (+0 %) | 1.456 (+0 %) | 1.146 (+0 %) | 1.377 (+0 %) |
| sans_profil_cl | 1.102 (+7 %) | 1.393 (+5 %) | 1.108 (+4 %) | 1.339 (+3 %) |
| sans_recalage | 3.565 (-200 %) | 2.761 (-89 %) | 3.200 (-178 %) | 2.460 (-78 %) |
| sans_relief | 1.191 (-0 %) | 1.462 (-0 %) | 1.150 (-0 %) | 1.381 (-0 %) |

Pourcentages : reduction de l'ecart par rapport a la configuration actuelle (positif = ecart plus petit).

**Mecanismes retenus pour correction en phase 2** (reduction >= 30 % sur v a 300 OU 800 m) : sans_amort
**Correction passe_haut** : v reduit d'au moins 30 % a 300 et 800 m : oui ; u augmente de 10 % au plus : oui -> RETENUE pour la phase 2
