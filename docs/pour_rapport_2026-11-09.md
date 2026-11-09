# Résultats du 9 novembre 2026 pour le rapport et la soutenance

Note destinée à la session qui rédige le rapport (pfe-fd). Le message n'a pas pu lui être
transmis : les deux sessions ont des modes de permission différents. Les détails sont dans
`docs/regle_dso_0_1km.md` et `docs/diagnostic_v_national.md`.

## 0. Modèle national livré : la v2

Règle `configs/national/regle_v3_rugosite.json`, appliquée telle qu'écrite, au critère des
aéroports (RMSE vectoriel moyen sur les 12) :

| | Valeur |
|---|---|
| v2 | 3,045499 m/s |
| Seuil (1,01 × v2) | 3,075954 m/s |
| v3 | 3,075995 m/s, soit +1,0013 % pour +1 % autorisé |

La v3 échoue de très peu, mais la règle s'applique : la v2 reste livrée.

## 1. Grand Tunis v4 : DSO de 0 à 1 km au-dessus du sol (test 2022)

Mesure par `scripts/dso_hauteur_sol.py`, règle hachée avant le calcul.

| Tranche au-dessus du sol | Modèle, RMSE u / v | ERA5, RMSE u / v |
|---|---|---|
| 0–100 m | 0,486 / 0,478 | 1,018 / 0,946 |
| 100–300 m | 0,375 / 0,375 | 1,147 / 1,204 |
| 300–600 m | 0,348 / 0,322 | 0,892 / 0,981 |
| 600–1000 m | 0,286 / 0,287 | 0,426 / 0,475 |
| 0–1 km | 0,382 / 0,374 | 0,923 / 0,951 |

Sur 0–1 km :
- **DSO1 conforme** (0,486 m/s au pire), critère OMM aussi ;
- R² u, v conforme (0,986) ; R² w non conforme ;
- DSO2 conforme ;
- DSO3 non conforme (0,805) ;
- DSO5 conforme (0,486 m/s avec 25 % des entrées) ;
- direction : 3,3° contre 5,5° pour ERA5.

Vérifié **par niveau d'altitude**, le DSO1 reste non conforme (0,528 m/s au niveau 25 m). La
règle 0–1 km a été écrite **après** les tests : il faut publier les deux verdicts côte à côte,
et c'est à l'entreprise d'adopter ou non la règle.

## 2. National v2, même règle de 0 à 1 km

| Critère | National v2 | ERA5 |
|---|---|---|
| RMSE u / v sur 0–1 km | 0,557 / 0,779 m/s | 1,334 / 1,772 m/s |
| DSO1 | non conforme (0,881 au pire) | |
| DSO5 | non conforme (0,900) | |
| DSO3 | 0,786 | |
| R² u, v | conforme (0,969) | |

## 3. Écart entre la vérité générée et ERA5 (national, 180 instants de test)

| Hauteur au-dessus du sol | RMSE u | RMSE v |
|---|---|---|
| 300 m | 1,449 m/s | 2,083 m/s |
| 800 m | 1,353 m/s | 1,909 m/s |

Le biais moyen est à peu près nul. Les cartes sont sur le Drive :
`MyDrive/pcfno3d_national/runs/national_1km_v2/diagnostic_v/carte_v_300m.png` et
`carte_v_800m.png`.

## 4. Attribution par composante

L'amortissement de w explique 85 à 92 % de l'écart à 300 et 800 m au-dessus du sol. La brise,
les vents de pente, le relief et les perturbations aléatoires n'y pèsent que 0,01 à 0,05 m/s.
La correction du générateur (option passe-haut, commit 7b05b3d) ramène l'écart de 0,90–1,35 à
0,25–0,45 m/s.

## 5. Biais du générateur aux 12 aéroports (10 m, test 2022)

| Aéroport | générateur − ERA5 : u / v / vitesse | générateur − mesure : u / v / vitesse |
|---|---|---|
| Bizerte | −0,10 / +0,05 / −0,20 | −0,32 / +0,27 / −0,06 |
| Djerba-Zarzis | +0,32 / −0,39 / −0,35 | +0,01 / +0,27 / −0,11 |
| Enfidha | −0,21 / −0,06 / +0,23 | +0,66 / +0,03 / −0,47 |
| Gabès | −0,45 / +0,11 / +0,06 | −0,36 / +0,39 / −0,26 |
| Gafsa | −0,32 / −0,38 / +0,92 | +0,25 / +0,38 / −0,49 |
| Kairouan | −0,14 / +0,01 / +0,06 | −0,21 / −0,13 / +0,99 |
| Monastir | +0,24 / −0,38 / −0,22 | +0,86 / −0,46 / −0,82 |
| Remada | −0,52 / −0,14 / +0,91 | −1,51 / −0,58 / +2,30 |
| Sfax | −0,24 / −0,06 / +0,14 | −0,77 / +0,11 / +1,20 |
| Tabarka | −0,73 / −0,34 / +0,38 | −0,88 / −0,40 / +0,69 |
| Tozeur | +0,06 / +0,27 / +0,09 | +1,21 / −0,33 / −1,65 |
| Tunis-Carthage | −0,14 / +0,59 / −0,36 | −0,21 / +0,41 / +0,12 |

Au sol, le biais du générateur sur v reste modéré (au plus 0,6 m/s). Les grands biais de
vitesse face aux mesures (Remada, Tozeur, Sfax, Kairouan) sont déjà dans ERA5. Le générateur
n'en ajoute beaucoup qu'à Gafsa et Remada, environ +0,9 m/s.

## 6. Corrections des diapos de la soutenance

- **Diapo 16** : écrire « −9 % d'erreur face à GFS brut à +24 h (en moyenne ; meilleure que GFS à
  chaque station) ».
- **Diapo 14**, ligne M1 : ajouter « (50 % des capteurs) » après « sur le test synoptique ».
- **Diapo 11** : vérifier « 2,6·10⁻⁵ » et « 7,5 à 9,8× » dans `docs/BILAN_FINAL.md`.

## 7. Plan approuvé par Rayen (en cours)

- **Grand Tunis v7.** La règle de livraison est écrite avant l'entraînement
  (`configs/grand_tunis/regle_v7_rugosite.json`, commit 45d4e5a). L'entraînement n'est pas lancé :
  il faut d'abord racheter des unités Colab.
- **National.** Le test rapide des facteurs de perturbation, sur la validation, tourne en local.
