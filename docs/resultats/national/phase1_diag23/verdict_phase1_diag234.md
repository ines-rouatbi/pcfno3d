# Phase 1 nationale : verdicts selon la regle fixee avant les calculs

Regle : `configs/national/regle_phase1_diagnostics.json`, sha256 `298232278ce441282c7aea38abd1d7d705e29c6e7469781cfc9302c5af0bd1da` (verifie).

## Diagnostic 2 : spectres de l'erreur (test 2022)
- 300 m : part 13-26 km le long de y, u 0.176 / v 0.051 ; part > 50 km de l'erreur sur v 0.872
- 800 m : part 13-26 km le long de y, u 0.087 / v 0.027 ; part > 50 km de l'erreur sur v 0.929
**Ablation modes [6, 64, 32] en phase 3** : non proposee
**Erreur sur v dominee par la grande echelle (> 50 km)** : oui, priorite a la phase 2

## Diagnostic 3 : entrainement contre test
   tranche  entrainement_rmse_u  entrainement_rmse_v  test_rmse_u  test_rmse_v  entrainement_v_sur_u  test_v_sur_u
   0-100 m                0.477                0.602        0.463        0.603                 1.261         1.302
 100-300 m                0.522                0.726        0.519        0.726                 1.390         1.400
 300-600 m                0.641                0.851        0.608        0.853                 1.327         1.402
600-1000 m                0.625                0.850        0.585        0.878                 1.360         1.502

**Sur-apprentissage** (test / entrainement > 1,15) : aucun
**Ecart v/u deja present a l'entrainement** (>= 1,3) : oui (100-300 m, 300-600 m, 600-1000 m) : il vient de la verite d'entrainement ou de l'architecture

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
