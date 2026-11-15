# MOS de l'ERA5 10 m aux stations tunisiennes : résultats

Règle `configs/national/regle_mos_era5.json` (sha256 `80fe361dd9394308a8cc067cf59141849cd6fb146888e882c494b3d1189e1cc8`), `scripts/mos_era5.py`. Validation croisée par site : chaque site retiré de l'ajustement (2015–2020), puis jugé. ERA5 servie par Open-Meteo (`models=era5`, maille la plus proche).

## Grille (validation 2021)

| Configuration | MOS | ERA5 seule | durée (s) |
|---|---|---|---|
| ridge_a0.1 | 2.6801 | 2.8915 | 56 |
| ridge_a1 | 2.6801 | 2.8915 | 46 |
| ridge_a10 | 2.6801 | 2.8915 | 47 |
| ridge_a100 | 2.6799 | 2.8915 | 41 |
| ridge_a1000 | 2.6786 | 2.8915 | 43 |
| gb_lr0.1_d3_n200 | 2.6755 | 2.8915 | 420 |
| gb_lr0.1_d3_n500 | 2.6730 | 2.8915 | 896 |
| gb_lr0.1_d6_n200 | 2.6732 | 2.8915 | 434 |
| gb_lr0.1_d6_n500 | 2.6747 | 2.8915 | 1097 |

Configuration choisie : **gb_lr0.1_d3_n500** ; anomalies : aucune ; test autorisé : **False**.

### Validation 2021

- moyenne sur 28 sites du RMSE vectoriel : **MOS 2.673** contre ERA5 seule 2.892 m/s ; écart -0.219 m/s, IC 95 % [-0.238, -0.200] ;
- sites améliorés : 71% ; pire dégradation relative +24.7% (INM60760) ;
- conditions (1) moyenne et IC : True, (2) 2 sites sur 3 : True, (3) aucun site > 5 % : False → **ne bat pas ERA5**.

- aeroports (12 sites) : MOS 2.629 contre ERA5 seule 2.793 m/s, 8 sites améliorés
- non_aeroports (16 sites) : MOS 2.706 contre ERA5 seule 2.965 m/s, 12 sites améliorés
- sans_abri_probable (24 sites) : MOS 2.764 contre ERA5 seule 2.872 m/s, 16 sites améliorés

| Station | n | RMSE vect. MOS | RMSE vect. ERA5 seule | écart relatif | biais vitesse MOS | biais vitesse ERA5 | MAE dir. MOS | MAE dir. ERA5 |
|---|---|---|---|---|---|---|---|---|
| INM60710 | 2877 | 2.652 | 2.885 | -8.0% | -1.18 | -0.13 | 48 | 46 |
| INM60714 | 2815 | 2.364 | 2.911 | -18.8% | -0.59 | +0.29 | 36 | 39 |
| INM60715 | 2878 | 2.085 | 2.645 | -21.2% | -0.79 | +0.19 | 29 | 35 |
| INM60720 | 2837 | 2.555 | 3.648 | -30.0% | +1.33 | +2.51 | 29 | 32 |
| INM60723 | 2132 | 2.617 | 3.396 | -22.9% | +0.04 | +1.31 | 74 | 71 |
| INM60725 | 2769 | 2.272 | 2.398 | -5.2% | -0.62 | -0.12 | 46 | 44 |
| INM60728 | 1926 | 3.315 | 3.504 | -5.4% | -1.02 | +0.83 | 35 | 26 |
| INM60729 | 2384 | 2.238 | 2.489 | -10.1% | -0.78 | +0.18 | 40 | 41 |
| INM60731 | 2873 | 3.067 | 3.103 | -1.2% | -1.08 | -0.11 | 37 | 34 |
| INM60732 | 2686 | 1.968 | 2.286 | -13.9% | -0.46 | +0.23 | 40 | 44 |
| INM60734 | 2543 | 2.167 | 2.306 | -6.0% | +0.05 | +0.54 | 42 | 44 |
| INM60735 | 2855 | 1.453 | 1.863 | -22.0% | -0.16 | +0.94 | 34 | 28 |
| INM60738 | 2714 | 5.070 | 4.401 | +15.2% | -3.75 | -3.00 | 38 | 28 |
| INM60739 | 2819 | 3.004 | 2.958 | +1.6% | -1.29 | -1.00 | 44 | 47 |
| ISD60740399999 | 2872 | 2.622 | 2.590 | +1.2% | -1.13 | -0.74 | 23 | 24 |
| INM60742 | 2633 | 2.606 | 3.090 | -15.7% | -0.63 | +0.03 | 29 | 38 |
| INM60745 | 2761 | 3.302 | 2.973 | +11.1% | -2.12 | -1.19 | 38 | 36 |
| INM60748 | 2663 | 1.983 | 2.045 | -3.0% | -0.63 | +0.16 | 37 | 33 |
| INM60750 | 2823 | 2.408 | 2.905 | -17.1% | +0.57 | +1.32 | 28 | 29 |
| INM60760 | 2836 | 3.639 | 2.919 | +24.7% | -2.32 | -1.55 | 34 | 27 |
| INM60764 | 2795 | 3.921 | 3.772 | +3.9% | -1.57 | +0.26 | 54 | 44 |
| INM60765 | 2873 | 2.459 | 3.558 | -30.9% | +1.55 | +2.80 | 24 | 25 |
| INM60767 | 2783 | 2.679 | 2.722 | -1.6% | -1.43 | -0.34 | 28 | 29 |
| INM60769 | 2870 | 3.030 | 2.655 | +14.1% | -0.87 | -0.13 | 31 | 25 |
| INM60770 | 2796 | 1.877 | 3.133 | -40.1% | +0.72 | +2.40 | 26 | 24 |
| INM60772 | 2797 | 2.199 | 2.597 | -15.3% | +0.17 | +1.04 | 32 | 31 |
| INM60775 | 2795 | 2.472 | 2.512 | -1.6% | +0.93 | +1.43 | 34 | 27 |
| INM60780 | 2223 | 2.821 | 2.698 | +4.6% | +0.39 | +0.25 | 25 | 26 |

### Test 2022

Non lu (la validation ne bat pas ERA5 selon la règle, ou anomalie).
