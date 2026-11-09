# Règle : domaine d'évaluation des DSO du modèle national

**Écrite le 9 novembre 2026, à la demande d'Ines.**

## Règle

> Les DSO sont évalués de **0 à 1 km au-dessus du sol** (document du 25/01/2026, ch. 1).
> Les niveaux au-dessus sont publiés **à titre d'information**, sans verdict.

## Mise en œuvre

Le modèle a 16 niveaux en altitude, de 25 à 2500 m au-dessus de la mer. Chaque nœud est donc
classé selon sa hauteur au-dessus du relief : altitude du niveau − altitude du terrain du nœud.
Les métriques sont cumulées par tranche :

| Tranche (m au-dessus du sol) | Statut |
|---|---|
| 0–100 | évaluée |
| 100–300 | évaluée |
| 300–600 | évaluée |
| 600–1000 | évaluée |
| > 1000 | information |

Dans chaque tranche, sur les mêmes instants, capteurs et nœuds :
- DSO1 : RMSE de u et de v ≤ 0,5 m/s, et R² de u et de v ≥ 0,95 ;
- critère OMM, rapporté : RMSE ≤ max(0,5 ; 10 % de la vitesse) ;
- DSO3 : rugosité dans [0,9 ; 1,1] ;
- DSO5 : RMSE maximal à 25 % des capteurs ≤ 0,7 m/s (cahier), et dégradation entre 100 % et 25 % des capteurs ≤ 10 % (DSO national).

Ces métriques sont calculées pour le modèle livré **et** pour ERA5 interpolé sur la même vérité.

Commande : `python -m pcfno3d -c <config> dso-tranches`. Elle écrit
`eval_pcfno3d/<modèle>/dso_tranches.json` et `.md` (code : `src/pcfno3d/dso_tranches.py`).

## Traçabilité

- La règle a été écrite **après** les tests 2022 de la national v2 et de la v3. Son origine
  est le document du 25/01/2026, pas les résultats. Le verdict par niveau d'altitude (16
  niveaux jusqu'à 2500 m) reste calculé et publié à côté, sans être retiré.
- À l'altitude de 2500 m, le nœud le plus proche du sol est encore à environ 950 m au-dessus
  du relief (point culminant 1544 m). Partout ailleurs, il est entre 1,5 et 2,5 km au-dessus
  du sol, donc hors du domaine demandé.

## Résultat : national v2 livrée, test 2022 (180 instants, 9 novembre 2026)

Mesure faite par `scripts/dso_hauteur_sol.py`, avec la règle hachée
(`runs/national_1km_v2/dso_hauteur_sol/regle.json`). Les nœuds valides vont de 5 à 2522 m
au-dessus du sol, dont 67,8 % dans le domaine 0–1 km.

| Tranche (au-dessus du sol) | Modèle, RMSE u / v | ERA5 interpolé, RMSE u / v | R² u / v (modèle) | R² w (modèle) |
|---|---|---|---|---|
| 0–100 m | 0,466 / 0,607 | 1,082 / 1,231 | 0,991 / 0,976 | 0,959 |
| 100–300 m | 0,515 / 0,728 | 1,399 / 1,771 | 0,992 / 0,976 | 0,904 |
| 300–600 m | 0,618 / 0,851 | 1,439 / 1,994 | 0,988 / 0,971 | 0,668 |
| 600–1000 m | 0,601 / 0,881 | 1,346 / 1,919 | 0,988 / 0,969 | 0,457 |
| **0–1 km (domaine)** | **0,557 / 0,779** | 1,334 / 1,772 | 0,990 / 0,972 | 0,824 |
| > 1 km (information) | 0,608 / 0,920 | 1,028 / 1,601 | 0,988 / 0,972 | 0,411 |

| Verdict sur 0–1 km | Modèle | ERA5 |
|---|---|---|
| DSO1, RMSE ≤ 0,5 m/s dans chaque tranche | **non conforme** (max 0,881 m/s, 600–1000 m) | non conforme (1,994) |
| DSO1, critère OMM | non conforme | non conforme |
| R² u, v ≥ 0,95 | **conforme** (min 0,969) | non conforme (0,839) |
| R² u, v, w ≥ 0,95 | non conforme (w 0,457) | non conforme |
| DSO2, masse | conforme (2,5·10⁻⁶) | conforme |
| DSO3, rugosité | non conforme (0,786) | non conforme (0,526) |
| Direction, MAE | 6,0° | 11,7° |
| DSO5, 25 % des entrées : RMSE max ≤ 0,7 m/s (cahier) | non conforme (0,900) | — |
| DSO5 national : dégradation ≤ 10 % | conforme (0,881 → 0,900, soit +2 %) | — |

Ce qu'il faut retenir :
- Restreindre l'évaluation à 0–1 km ne rend pas le DSO1 conforme. Même dans la tranche
  0–100 m, v vaut 0,61 m/s.
- Le domaine 0–1 km contient des nœuds au-dessus du relief, situés à des altitudes où
  l'artefact du générateur sur v est fort (docs/diagnostic_v_national.md).
- Contre la même vérité, le modèle divise l'erreur d'ERA5 par 2,3 : RMSE vectoriel 0,96 m/s
  contre 2,22 sur 0–1 km.
