# DSO du modèle national — proposition pour le cahier des charges

*Rédigé le 9 novembre 2026. À insérer dans le cahier des charges par Ines, qui le valide.*

## 1. Pourquoi des DSO propres au modèle national

Les DSO1 à DSO6 du cahier ont été définis pour le Grand Tunis : un réseau de capteurs ANEO
dense, 12 niveaux jusqu'à 1000 m. Le modèle national (Tunisie entière, nœuds de 1 km) change
trois conditions qui fixent ce qu'un seuil peut raisonnablement exiger :

| | Grand Tunis | National |
|---|---|---|
| Capteurs d'entrée | réseau ANEO dense | 12 aéroports METAR sur 828 × 388 km, plus ERA5 10 m tous les 2 km |
| Niveaux | 12, jusqu'à 1000 m | 16, jusqu'à 2500 m au-dessus de la mer |
| Couverture du niveau le plus bas | quasi complète | 24 % à 25 m (relief jusqu'à 1544 m) |

Avec 12 aéroports, l'information vient surtout d'ERA5. La question utile est donc de savoir
si le modèle fait mieux qu'ERA5 interpolé, pas seulement s'il passe un seuil absolu.

**Traçabilité.** Ces DSO ont été fixés le 9 novembre 2026, **après** le test de la national
v2 et **avant** celui de la v3. Les seuils ne sont pas tirés des chiffres de la v2 : ils
viennent de l'OMM (10 %), de l'objectif initial (1000 m) et de la référence ERA5.

## 2. Les DSO nationaux

| DSO | Critère national | Seuil | Justification |
|---|---|---|---|
| **DSO1a** précision | RMSE de u et de v à chaque niveau jusqu'à 1000 m | ≤ max(0,5 m/s ; 10 % de la vitesse moyenne du niveau) | Tolérance de l'OMM sur la mesure du vent. Le vent croît avec la hauteur : un seuil absolu pénalise surtout les niveaux hauts. |
| **DSO1b** valeur ajoutée | Gain sur ERA5 interpolé, 1 − RMSE vect. modèle / RMSE vect. ERA5, à chaque niveau jusqu'à 1000 m | ≥ 0 | Un modèle national n'a d'utilité que s'il améliore ERA5, sa propre entrée. |
| **DSO1c** corrélation | R² de u et de v à chaque niveau jusqu'à 1000 m | ≥ 0,95 | Inchangé pour u et v. Le R² de w est rapporté sans verdict : w vient de l'équation de continuité, et sa cohérence est jugée par DSO2. |
| (rapporté) | Niveaux de 1250 à 2500 m | sans verdict | Au-delà de l'objectif initial (« jusqu'à 1000 m ») |
| **DSO2** masse | RMS du résidu de continuité | ≤ 10⁻⁴ | Inchangé (physique, indépendant de l'échelle) |
| **DSO3** rugosité | Rapport de rugosité prédiction / vérité, moyenne des niveaux couvrant au moins 50 % du domaine | dans [0,9 ; 1,1] | Inchangé, mais les niveaux surtout sous le relief (25 à 150 m) sont trop peu représentatifs |
| **DSO4** temps | Inférence par instant | < 1 s | Inchangé |
| **DSO5** robustesse | Dégradation du RMSE maximal entre 100 % et 25 % des capteurs | ≤ 10 % | 25 % des aéroports, c'est 3 stations. Le critère absolu (0,7 m/s) ne faisait que répéter DSO1. |
| **DSO6** fiabilité | Couverture de l'intervalle à 80 % | inchangé | Indépendant de l'échelle |
| **DSO7** (nouveau) mesures réelles | RMSE vectoriel moyen aux 12 aéroports, chaque station retirée de l'entrée quand elle est jugée, 2022 | ≤ celui d'ERA5 seul | La seule comparaison à de vraies mesures |

Les DSO du cahier (Grand Tunis) restent calculés et publiés à côté, pour comparaison.

## 3. Ce que ces DSO donnent pour la national v2 (test 2022)

| DSO | v2 | Verdict |
|---|---|---|
| DSO1a | erreur relative de 9 à 11 % jusqu'à 300 m, puis 14 à 18 % ; v = 0,74 m/s à 400 m pour un seuil de 0,63 | non conforme |
| DSO1b | à mesurer (`pcfno3d verdict-national`) | — |
| DSO1c | R² u, v ≥ 0,965 à tous les niveaux | conforme |
| DSO2 | 2,4·10⁻⁶ | conforme |
| DSO3 | 0,81 (niveaux couvrant au moins 50 %) | non conforme |
| DSO4 | 248 ms | conforme |
| DSO5 | −0,4 % (aucune dégradation) | conforme |
| DSO6 | couverture à 80 % : 0,82 à 0,83 | conforme |
| DSO7 | 3,05 contre 2,86 m/s pour ERA5 ; 4 aéroports sur 12 meilleurs | non conforme |

Changer les DSO ne rend pas la v2 conforme. Ça mesure ce qui compte pour un modèle national,
et ça désigne la limite : la vérité d'entraînement (le générateur) est elle-même moins juste
qu'ERA5 aux aéroports (3,11 contre 2,86 m/s). Le levier est donc le générateur, pas un
entraînement plus long.

## 4. Mise en œuvre

- Seuils : section `dso_national` de la configuration (`config.DSONationalCfg`).
- Calcul : `python -m pcfno3d -c national_1km_colab verdict-national`, après `evaluate`, `dso6`
  et `scripts/valider_grand_tunis_reel.py`. Écrit `verdict_national.json` et
  `verdict_national.md` dans `eval_pcfno3d/<modèle>/`.
- ERA5 interpolé : les deux derniers canaux d'entrée du modèle, sur les mêmes instants,
  capteurs et nœuds que le modèle (une seule lecture du test).
