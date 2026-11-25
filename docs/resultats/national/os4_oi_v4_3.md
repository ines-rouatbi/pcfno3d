# OS4 national, règle v4_3 — réseau ANEO à 25 m jugé sur la couche 0–100 m (test 2022, deuxième lecture)

Dates du calcul : choix du réglage le 24/11/2026 au soir, jugement du test le 25/11/2026 de 00 h 25 à 02 h 35
(heure de Tunis). Colab CPU High-RAM, carnet `notebooks/current/pcfno3d_national_os4_oi_v4_3_colab.ipynb`
(commit `a522b75`, code épinglé sur `665568945731c67ae91175ec65d43f7a9084b04a`). Environ 1 unité Colab
(limite fixée par Ines : 10). La session s'est terminée seule à la fin (`runtime.unassign()`, vérifié :
« Reconnecter »). Rien n'a été supprimé ni renommé sur le Drive.

Règle : `configs/national/regle_os4_oi_v4_3.json`, sha256
`21cb71ba14afaeb574eeac5916343ee742236764000b7b8141a362144ff62541`, approuvée par Ines avant tout calcul
(« règle v4_3 OK », 24/11/2026). Sorties : `MyDrive/pcfno3d_national/os4_oi_v4_3/` (`bilan.md`, `bilan.json`,
`metriques.csv`, `courbe.csv`, `courbe.png`, `n_minimal.json`, `plan_grille.geojson`, `plan_glouton.geojson`,
`choix_validation/`), écrites à 02 h 35.

## Pourquoi une règle v4_3

Le 24/11/2026, Ines a corrigé l'objectif OS4. Un anémomètre ANEO mesure à la hauteur du mât (25 m) et ne
renseigne pas la couche 300–1000 m : juger toute la colonne 0–1 km (règle v4_2) était une **erreur de
formulation, corrigée pour une raison physique**, et non parce que la v4_2 n'est pas passée. La v4_3 juge la
couche 0–100 m, garde le vent à 10 m en descriptif et la couche 0–1 km et la pire tranche **pour information,
non visées par des capteurs à 25 m**. Elle corrige aussi les réglages de l'OI repris des aéroports réels
(λ = 4, L = 150 km), incohérents avec la simulation : forme de corrélation, L, λ et H sont choisis sur la
validation 2021 seulement.

**Deuxième lecture, non indépendante.** La règle a été écrite après la lecture du test 2022 par la v4_2 et relit
ce même test. Le résultat v4_2 reste publié tel quel (`os4_oi.md`, N minimal non atteint jusqu'à 400 sur la pire
tranche 0–1 km).

## Nature de l'expérience

Simulation OSSE, distincte du modèle C' : fond = ERA5 dégradée, capteurs ANEO simulés à 25 m (vérité générée
+ bruit), analyse = fond + OI 3D. La vérité est la vérité générée du jeu national : **ce ne sont pas des
mesures**. Placements : glouton (incertitude du fond), grille régulière, aléatoire ×3 ; N de 10 à 400.
Intervalles : bootstrap sur les jours, 2000 tirages, graine 2030.

## Réglage choisi sur la validation 2021

72 combinaisons (forme gaussienne / exponentielle × L ∈ {55, 100, 150} km × λ ∈ {0,065 ; 0,25 ; 1 ; 4} ×
H ∈ {300, 1000, 3000} m), critère = erreur 0–100 m moyenne à N = 50 et 200 sur la grille régulière,
120 instants de validation.

**Retenu : corrélation exponentielle, L = 100 km, λ = 0,065, H = 3000 m** (critère 1,560 m/s, un seul
meilleur, pas d'ex aequo).

## Verdict (test 2022)

Critère fixé avant le calcul : borne basse de l'IC 95 % du gain de RMSE vectorielle 0–100 m ≥ 10 % à N = 50.

| Stratégie | Gain 0–100 m à N = 50 | IC 95 % | Verdict | N minimal (IC 95 %) |
|---|---:|---|---|---:|
| glouton | +15,46 % | [+10,87 ; +19,25] | **CONFORME** | 50 |
| grille régulière | +28,29 % | [+27,56 ; +29,00] | **CONFORME** | 25 |
| aléatoire 1 | +20,67 % | [+19,84 ; +21,54] | **CONFORME** | 25 |
| aléatoire 2 | +20,12 % | [+19,17 ; +21,07] | **CONFORME** | 25 |
| aléatoire 3 | +20,49 % | [+19,65 ; +21,32] | **CONFORME** | 25 |

Sans réseau : 0–100 m 2,485 m/s ; 10 m 2,083 m/s ; 0–1 km 2,491 m/s (information) ; pire tranche
1,807 m/s (information).

## Courbe complète (gain de RMSE vectorielle)

| Stratégie | N | 0–100 m | IC 95 % | 10 m | 0–1 km (information) |
|---|---:|---:|---|---:|---:|
| glouton | 10 | −9,97 % | [−16,40 ; −4,50] | −3,03 % | −6,83 % |
| glouton | 25 | −0,62 % | [−7,24 ; +5,07] | +9,43 % | +2,63 % |
| glouton | 50 | +15,46 % | [+10,87 ; +19,25] | +19,91 % | +16,35 % |
| glouton | 100 | +27,28 % | [+23,79 ; +30,31] | +32,91 % | +27,63 % |
| glouton | 200 | +41,19 % | [+38,98 ; +43,07] | +47,29 % | +40,18 % |
| glouton | 400 | +51,19 % | [+49,74 ; +52,45] | +59,32 % | +49,58 % |
| grille régulière | 10 | +5,52 % | [+4,67 ; +6,41] | +6,14 % | +5,67 % |
| grille régulière | 25 | +17,85 % | [+16,93 ; +18,78] | +19,29 % | +17,63 % |
| grille régulière | 50 | +28,29 % | [+27,56 ; +29,00] | +32,65 % | +27,90 % |
| grille régulière | 100 | +38,90 % | [+38,17 ; +39,63] | +44,62 % | +37,70 % |
| grille régulière | 200 | +47,65 % | [+47,02 ; +48,31] | +56,08 % | +46,12 % |
| grille régulière | 400 | +53,00 % | [+52,36 ; +53,69] | +63,59 % | +51,43 % |
| aléatoire 1 | 10 | +7,21 % | [+6,31 ; +8,22] | +8,26 % | +7,09 % |
| aléatoire 1 | 25 | +14,16 % | [+13,20 ; +15,17] | +16,45 % | +13,83 % |
| aléatoire 1 | 50 | +20,67 % | [+19,84 ; +21,54] | +25,93 % | +20,78 % |
| aléatoire 1 | 100 | +30,50 % | [+29,72 ; +31,29] | +37,73 % | +30,17 % |
| aléatoire 1 | 200 | +38,21 % | [+37,45 ; +38,97] | +48,55 % | +38,00 % |
| aléatoire 1 | 400 | +44,69 % | [+43,88 ; +45,55] | +57,25 % | +44,35 % |
| aléatoire 2 | 10 | +5,43 % | [+4,74 ; +6,18] | +8,06 % | +5,92 % |
| aléatoire 2 | 25 | +13,54 % | [+12,82 ; +14,29] | +18,36 % | +14,33 % |
| aléatoire 2 | 50 | +20,12 % | [+19,17 ; +21,07] | +27,87 % | +20,93 % |
| aléatoire 2 | 100 | +28,69 % | [+27,55 ; +29,76] | +39,37 % | +29,31 % |
| aléatoire 2 | 200 | +35,99 % | [+34,97 ; +36,99] | +48,66 % | +36,46 % |
| aléatoire 2 | 400 | +44,00 % | [+43,07 ; +44,93] | +57,58 % | +43,78 % |
| aléatoire 3 | 10 | +6,97 % | [+6,26 ; +7,70] | +9,18 % | +7,26 % |
| aléatoire 3 | 25 | +13,58 % | [+12,69 ; +14,46] | +18,76 % | +14,04 % |
| aléatoire 3 | 50 | +20,49 % | [+19,65 ; +21,32] | +27,44 % | +20,89 % |
| aléatoire 3 | 100 | +27,44 % | [+26,67 ; +28,21] | +38,04 % | +28,18 % |
| aléatoire 3 | 200 | +35,72 % | [+34,96 ; +36,51] | +48,74 % | +36,27 % |
| aléatoire 3 | 400 | +41,52 % | [+40,78 ; +42,29] | +57,08 % | +41,99 % |

## Métriques descriptives à N = 50 (couche 0–100 m, `metriques.csv`)

| Stratégie | RMSE vect. (m/s) | RMSE u | RMSE v | R² u | R² v |
|---|---:|---:|---:|---:|---:|
| sans réseau | 2,485 | 1,712 | 1,802 | 0,856 | 0,745 |
| glouton | 2,101 | 1,517 | 1,453 | 0,887 | 0,834 |
| grille régulière | 1,782 | 1,226 | 1,293 | 0,926 | 0,869 |
| aléatoire 1 | 1,971 | 1,354 | 1,433 | 0,910 | 0,839 |
| aléatoire 2 | 1,985 | 1,366 | 1,440 | 0,908 | 0,837 |
| aléatoire 3 | 1,976 | 1,366 | 1,428 | 0,908 | 0,840 |

Les autres métriques descriptives de la règle (MAE de vitesse et de direction, biais, L∞, résidu de masse),
avec leur IC 95 %, sont dans `metriques.csv` sur le Drive ; elles ne sont pas recopiées ici.

## Lecture

- **Toutes les stratégies sont conformes à N = 50.** Le réseau de mâts à 25 m réduit nettement l'erreur de la
  couche 0–100 m, qu'il vise.
- **La grille régulière est la meilleure à chaque N** : +28 % à 50 mâts, +53 % à 400 ; elle franchit
  l'objectif dès 25 mâts (borne basse +16,9 %).
- **Le glouton est le moins bon sous 100 mâts** et dégrade l'analyse avec 10 mâts (−10 %) : placer d'abord les
  mâts là où le fond est le plus incertain concentre les mesures sans couvrir le pays. Il rattrape la grille vers
  400 mâts.
- La couche 0–1 km (information) suit la couche 0–100 m, car H = 3000 m, retenu sur la validation, propage la
  correction presque sans décroissance verticale. Ce n'est pas un objectif de la v4_3 : des capteurs à 25 m ne
  mesurent pas le vent à 1 km, et ce gain dépend entièrement de la forme supposée de la corrélation verticale.
- Limite : vérité simulée, deuxième lecture du test 2022 ; à confirmer sur des mesures réelles de mâts.
- Comparaison avec la v4_2 : sans réseau, la pire tranche (1,807 m/s) et le vent à 10 m (2,083 m/s) sont
  identiques dans les deux notes, mais la couche 0–100 m ne l'est pas (1,802 m/s en v4_2, 2,485 m/s en RMSE
  vectorielle v4_3) : les deux règles ne la mesurent pas de la même façon. Les gains 0–100 m d'une note et de
  l'autre ne se comparent donc pas directement.

## Recommandation pour Tukhnanutha

Une **grille régulière de 25 à 50 mâts ANEO à 25 m** : à 50 mâts, l'erreur du vent de 0 à 100 m baisse de
28 % (de 2,49 à 1,78 m/s) et celle du vent à 10 m de 33 %. Le service sert ce plan en grille dans l'onglet
« Réseau ANEO » (`plan_grille.geojson`), lu par `atom_wind.dossier_os4` depuis le redémarrage du 25/11/2026
(commit `0cb8ab1`).
