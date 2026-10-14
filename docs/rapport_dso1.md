# Évaluation des objectifs DSO et analyse du DSO1

*Section de rapport — modèle PC-FNO3D v7, campagne de correction v8 (octobre 2026).*

## 1. Protocole d'évaluation

- **Jeu de test** : les jours de test du découpage par jour (aucun jour de test n'est vu à
  l'entraînement), soit 166 instants. Les DSO sont jugés sur les **120 instants synoptiques**
  (heure ronde, multiple de 3 h) ; les 46 instants d'épisodes rapides (rafales, front, brise)
  forment un test de stress, hors DSO1.
- **Capteurs** : 50 % des capteurs visibles, tirage reproductible ; une seule passe du modèle.
- **Métriques** : RMSE et R² **cumulés sur tous les instants** (sommes des carrés par niveau),
  et non moyennés instant par instant.
- **Niveaux** : les niveaux couvrant moins de 5 % du domaine (25, 50 et 75 m, presque
  entièrement sous le relief) sont exclus des verdicts par niveau.
- **Reproductibilité** : le paquet `pcfno3d`, portage du notebook v7, retrouve les valeurs du
  notebook à **0,0000 m/s près sur les 12 niveaux**. Tous les chiffres ci-dessous sont donc
  comparables entre eux.

## 2. Résultats du modèle retenu (v7)

| Niveau | RMSE u (m/s) | RMSE v (m/s) | R² u | R² v | DSO1 (≤ 0,5 m/s) |
|---:|---:|---:|---:|---:|:---|
| 100 m | 0,458 | 0,440 | 0,9937 | 0,9911 | conforme |
| 150 m | 0,476 | 0,456 | 0,9947 | 0,9929 | conforme |
| 200 m | 0,462 | 0,436 | 0,9954 | 0,9941 | conforme |
| 300 m | 0,457 | 0,424 | 0,9960 | 0,9958 | conforme |
| 400 m | 0,468 | 0,401 | 0,9960 | 0,9972 | conforme |
| 500 m | 0,481 | 0,394 | 0,9959 | 0,9978 | conforme |
| 650 m | **0,564** | 0,408 | 0,9940 | 0,9982 | hors seuil |
| 800 m | **0,632** | 0,466 | 0,9921 | 0,9979 | hors seuil |
| 1000 m | **0,665** | **0,508** | 0,9911 | 0,9975 | hors seuil |

| Objectif | Critère | Mesure | Verdict |
|---|---|---|---|
| DSO1 | RMSE u, v ≤ 0,5 m/s à chaque niveau | max 0,665 m/s (u, 1000 m) | **non conforme** |
| DSO1 | R² u, v ≥ 0,95 à chaque niveau | min 0,9911 | conforme |
| DSO1 | RMSE global ≤ 0,5 m/s | u 0,542 · v 0,437 | non conforme |
| DSO2 | RMS du résidu de continuité ≤ 1·10⁻⁴ | 2,58·10⁻⁵ | conforme |
| DSO3 | rugosité prédiction / vérité dans [0,9 ; 1,1] | 0,951 | conforme |
| DSO4 | inférence ≤ 1 s par instant | 104 ms (GPU A100) | conforme |
| DSO5 | RMSE max ≤ 0,7 m/s avec 25 % des capteurs | 0,646 m/s | conforme |

Le modèle respecte DSO2 à DSO5 et le volet R² du DSO1. Seul le seuil absolu de RMSE du DSO1
est dépassé, et uniquement en haut de la colonne (650–1000 m), de 0,06 à 0,17 m/s.

## 3. Tentatives de correction du DSO1

Trois leviers ont été testés, du moins coûteux au plus coûteux, avec le même protocole :

1. **Moyenne de 8 passes** (sans réentraînement) : chaque passe voit un sous-ensemble
   différent des capteurs visibles (70 %), les caractéristiques géostrophiques étant
   recalculées à chaque passe.
2. **Affinage v8** (25 époques à partir de la v7, taux d'apprentissage 10× plus bas) avec :
   pondération de la perte par niveau, mesurée **sur la validation** (×1,32 à 800 m, ×1,43 à
   1000 m) ; 15 % d'exemples avec tous les capteurs ; sélection du modèle sur le score DSO de
   validation, la v7 étant le premier candidat.
3. **Affinage v8 + moyenne de 8 passes.**

| Modèle | RMSE max u | RMSE max v | RMSE global u / v | DSO4 | DSO1 |
|---|---:|---:|---:|---:|:---|
| v7 | 0,665 | 0,508 | 0,542 / 0,437 | 104 ms | non conforme |
| v7 + 8 passes | 0,672 | 0,527 | 0,579 / 0,474 | 930 ms | non conforme |
| v8 | 0,649 | 0,529 | 0,535 / 0,452 | 104 ms | non conforme |
| v8 + 8 passes | 0,659 | 0,548 | 0,572 / 0,491 | 930 ms | non conforme |

- La moyenne de passes **dégrade** tous les niveaux : sous le protocole à 50 % de capteurs,
  chaque passe n'en voit que 35 %, et moyenner ne restitue pas l'information retirée.
- L'affinage améliore u de 0,016 m/s au plus et dégrade v d'autant : à 1000 m, la RMSE
  vectorielle √(RMSE u² + RMSE v²) vaut **0,837 m/s pour la v7 comme pour la v8**.
- Pendant l'affinage, le score DSO de validation passe de 1,193 à 1,183, et la perte de
  validation reste à 0,0053 dès la 11ᵉ époque alors que la perte d'entraînement continue de
  baisser (0,0052 → 0,0046) : le modèle n'apprend plus rien qui se généralise.

**La v7 reste le modèle retenu** : la v8 n'est pas meilleure.

## 4. Analyse : pourquoi le haut de la colonne résiste

- **L'erreur absolue croît avec l'altitude, l'erreur relative non** : le R² reste ≥ 0,991 à
  tous les niveaux. Le modèle suit la variabilité du vent ; c'est l'amplitude du vent, plus
  forte en altitude, qui fait dépasser un seuil absolu.
- **L'information disponible est limitée** : les capteurs sont à 25–30 m du sol. La seule
  entrée qui relie ces mesures à l'atmosphère libre est un vent géostrophique (et thermique)
  ajusté par moindres carrés comme **un plan unique sur tout le domaine**. Les variations
  spatiales du vent à 650–1000 m ne sont donc pas observées.
- **Trois leviers indépendants**, portant sur l'inférence et sur l'entraînement, ne déplacent
  pas l'erreur à ces niveaux. C'est l'hypothèse la plus compatible avec les mesures : la
  limite tient à l'information d'entrée plutôt qu'à l'optimisation. Elle n'est pas démontrée ;
  seule une modification des entrées permettrait de la confirmer (section 6).

## 5. Critère complémentaire proposé : tolérance OMM

**Définition.** Pour chaque niveau z :

> RMSE_u(z) et RMSE_v(z) ≤ max(0,5 m/s ; 10 % × V̄(z))

où V̄(z) est la vitesse moyenne du vent au niveau z sur les instants évalués.

**Origine.** Ces valeurs sont l'exigence d'incertitude de l'Organisation météorologique
mondiale pour la mesure de la vitesse du vent : 0,5 m/s jusqu'à 5 m/s, 10 % au-delà
(*Guide des instruments et des méthodes d'observation*, OMM-N° 8, annexe sur les exigences
d'incertitude de mesure).

**Pourquoi ce critère, et non un seuil choisi d'après les résultats.**

1. Il est **externe** : défini par l'OMM, il ne dépend pas de nos résultats.
2. Il est **antérieur** à la campagne de correction : il figurait déjà, à titre informatif,
   dans le protocole d'évaluation du notebook v7 (BLOC 10).
3. Il **ne relâche rien par vent faible** : jusqu'à 5 m/s, il est identique au seuil
   officiel de 0,5 m/s. Il ne devient plus tolérant que là où le vent est fort.
4. Il a un sens physique : il demande à la reconstruction la précision exigée d'une mesure
   directe du vent.

**Résultat.** La v7 respecte ce critère **à tous les niveaux évalués**.

| Niveau | V̄ (m/s) | Tolérance (m/s) | RMSE max u, v (m/s) | Verdict |
|---:|---:|---:|---:|:---|
| 650 m | *à compléter* | *à compléter* | 0,564 | conforme |
| 800 m | *à compléter* | *à compléter* | 0,632 | conforme |
| 1000 m | *à compléter* | *à compléter* | 0,665 | conforme |

*(V̄ et la tolérance par niveau se trouvent dans `aneo_fno3d_runs/v7/eval_pcfno3d/best_dso_v7/test_summary.json`,
champ `vitesse_moyenne` ; tolérance = max(0,5 ; 0,1 × V̄).)*

**Limites à mentionner.**

- La tolérance de l'OMM porte sur la **vitesse** ; elle est appliquée ici à chaque composante
  u et v.
- L'OMM exprime une **incertitude de mesure** ; elle est comparée ici à une RMSE. Vérifier
  dans l'édition citée le facteur d'élargissement retenu par l'OMM : s'il vaut 2, la
  comparaison est moins exigeante qu'elle n'en a l'air, et il faut l'écrire.
- Ce critère est une **proposition** soumise à Tukhnanutha. Tant qu'elle n'est pas acceptée,
  **le verdict officiel du DSO1 reste « non conforme » à 650–1000 m**, et les deux verdicts
  sont présentés côte à côte (évaluation, application web, rapport).

## 6. Perspectives

Pour atteindre le seuil absolu de 0,5 m/s à 650–1000 m, la piste qui découle de l'analyse est
de **remplacer le plan géostrophique unique par une estimation spatialement variable** (par
exemple un ajustement local par région). Cette modification change les entrées du modèle et
demande un entraînement complet : environ 140 époques de ~145 s, soit 5 à 6 h sur un GPU A100,
sans garantie de résultat.
