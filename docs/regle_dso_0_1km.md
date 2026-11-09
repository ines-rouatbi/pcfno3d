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
