# Lot M1 — correction post-modèle du vent moyen par niveau

*Portage dans le paquet des BLOCS 10f, 10g, 10h et de l'étape 2b du BLOC 12 de `fno3d_v7.ipynb`
(Colab, 12–13/10/2026). Module : `src/pcfno3d/correction.py`.*

## Statut : prêt, **non retenu tant que 10i ne l'a pas validé**

La correction n'est retenue que si la session de confirmation **BLOC 10i** la valide (critères
C1, C2, C3, règle de Rayen). Le chiffre annoncé — pire niveau u 0,419 / v 0,382 m/s au test —
est **optimiste** : plusieurs variantes (10e, 10f, 10g, 10h à alpha 1 et 0,01) ont été regardées
sur le test pendant la conception. Ce portage est donc fusionné **après** cette décision, pas
avant.

## Ce que fait la correction

Au-dessus de 500 m, l'erreur de v7 est surtout un **décalage de tout le niveau**. Une régression
ridge, une par niveau et par composante (u, v), prédit le vent moyen vrai du niveau à partir de
30 caractéristiques calculées sur la surface. On décale ensuite **uniformément** les nœuds
valides du niveau : la structure spatiale prédite par v7 est conservée, et les nœuds sous le
relief ne sont jamais touchés.

| Bloc | Fonction du paquet | Contenu |
|---|---|---|
| 10f | `instant_features`, `ridge_fit`, `ridge_pred` | 13 caractéristiques : tfeat (8), moyennes u, v, T des capteurs visibles, moyennes u, v du niveau prédites par v7 |
| 10g | `geostrophic_v2`, `plane_residual_v1`, `geo_row`, `x2` | vent géostrophique v2 : Tv = p/(Rd·ρ), altitude comme régresseur du plan (indicateur de stabilité), second ajustement sans points aberrants, ρ au niveau de la mer 1,2 ou calculée |
| 10h | `x3`, `fit_correction` | rotation du géostrophique (¼, ½, ¾ de l'angle surface/géostrophique), vent thermique × stabilité (séparé par signe) : 30 caractéristiques ; alpha ∈ {0,01 … 100} choisi **sur la validation** |
| 12-2b | `MeanWindCorrection.apply` | application depuis le JSON `couche_limite_v7.json` |

## Utilisation

Strictement **opt-in** : `inference.correction` vaut `null` par défaut, et la sortie de v7 est
alors identique bit à bit (testé).

```yaml
inference:
  correction: /content/drive/MyDrive/aneo_fno3d_runs/v7/couche_limite_v7.json
```

Une fois activée, la correction s'applique partout où le paquet produit un volume : API, MCP,
`reconstruct`, `evaluate`. L'évaluation corrigée a son propre dossier,
`eval_pcfno3d/<modèle>_corr<sha8>/`. Le **sha256 du JSON** figure dans `meta()` (`/api/meta`),
dans le manifeste des exports web et dans les `.npz`. Son temps de calcul compte dans DSO4.
Désactivée, ni le manifeste ni les `.npz` n'ont de clé `correction` : les fichiers restent
ceux de `main` (testé).

Deux combinaisons sont **refusées** par une `ValueError` explicite, au chargement plutôt qu'en
service :

- `inference.tta > 1` avec une correction (configuration, `WindFieldModel`, `predict(tta=…)`,
  `Evaluator(tta=…)`) : voir l'écart 5 ci-dessous ;
- une grille dont le niveau du haut n'est pas évalué : `x3` lit le vent moyen des capteurs dans
  la ligne de ce niveau, qui serait nulle, et l'angle de rotation serait faux sans bruit.

## Fidélité au notebook (tests CPU, `tests/test_correction.py`)

Les copies exactes des cellules sont dans `tests/reference/notebook_v7_correction/`, contrôlées
par `SHA256SUMS`. Les tests exécutent **les fonctions du notebook elles-mêmes** sur les mêmes
tableaux, et comparent à 1e-6 :

- l'estimateur géostrophique v2 ;
- les caractéristiques (10f, x2, x3) ;
- la ridge ;
- l'application (2b, exécutée telle quelle).

S'y ajoutent les tests de comportement : désactivée = bit à bit, coefficients connus = décalage
exact, ajustement qui retrouve une relation connue.

**Écarts connus entre le notebook et le portage.** Les écarts 1 et 3 sont sans effet sur v7.
L'écart 2 est sans effet sur les **prédictions** de la ridge, mais **pas sur ses coefficients**.
Les écarts 4 et 5 sont des limites d'usage, pas de portage :

1. Moins de 100 capteurs visibles : `geostrophic_v2` renvoie des zéros (garde du 10g). L'étape 2b
   n'a pas cette garde et ajusterait le plan quand même. Le paquet suit 10g partout, pour que
   l'application voie exactement les caractéristiques de l'ajustement. Sans effet sur le domaine
   v7 (des milliers de capteurs).
2. Les moyennes des caractéristiques sont calculées en numpy float32. Le 10f les calcule en torch
   float32 (sur GPU, sortie bf16 convertie en fp32, pour les moyennes de niveau), et 2b en
   numpy : écarts de l'ordre de 1e-7 en relatif. Sur les **coefficients**, l'effet n'est pas
   négligeable : les 30 caractéristiques sont exactement colinéaires (su*stab = su*stab+ +
   su*stab-, idem sv) et alpha vaut 0,01, donc ces 1e-7 déplacent mu / sd / coef d'environ
   3e-4. Les prédictions, elles, ne bougent pas (la colinéarité se compense). C'est pourquoi
   le script de reproduction juge le portage sur les **prédictions**, pas sur les coefficients.
3. Niveaux de sélection : le notebook prend les niveaux ≥ 650 m. Sur une grille qui n'en a pas
   (le mini-jeu de test), le paquet prend tous les niveaux évalués. Sans effet sur v7.
4. **Fraction de capteurs.** L'ajustement (10f à 10h) et la confirmation (10i) se font à
   **50 %** des capteurs (protocole DSO1). En production, `predict()` applique la correction à
   `keep = 1` : les caractéristiques (moyennes des capteurs visibles, géostrophique, moyennes
   de niveau prédites) sortent alors de la distribution d'ajustement. Le gain DSO1 n'est
   **démontré qu'à 50 %**. Le script de reproduction mesure, à titre informatif, le test
   synoptique à `keep = 1,0` (modèle seul contre corrigé, RMSE par niveau) pour vérifier
   l'absence de dégradation ; ce n'est pas une démonstration du gain à `keep = 1`.
5. **Moyenne de passes (`tta > 1`).** La correction n'a été ajustée que sur la sortie d'une
   seule passe. Appliquée au champ moyenné de plusieurs passes, que l'ajustement n'a jamais
   vu, elle n'a pas de garantie : la combinaison est refusée (`ValueError`).

## Reproduction dans la session GPU

`scripts/m1_reproduction_colab.py`, à lancer **après 10i**, dans la même session :

- rejoue 10f + 10g + 10h avec le paquet ;
- écrit uniquement dans `aneo_fno3d_runs/v7/reproduction_paquet/`. Les JSON figés et leurs CSV
  ne sont jamais réécrits (vérifié par sha256 avant/après). Seule autre écriture possible :
  `Geometry` peut créer son cache `geometry_v1.npz` dans `paths.cache_dir`, qui est le disque
  local de Colab (`/content/aneo_cache_v7`) et non le Drive ;
- juge le portage sur trois critères, contre le `couche_limite_v7.json` figé :
  (a) même variante rho_sl, même alpha, mêmes niveaux « utilise » ;
  (b) mêmes **prédictions** de la ridge (vent moyen cible, instant par instant, chaque
  niveau / composante utilisé, sur la validation et sur le test synoptique) à 1e-4 m/s ;
  (c) RMSE par niveau du test synoptique (50 %) égales aux chiffres de desktop-d3, au millième ;
- imprime les écarts de mu / sd / coef **pour information** (écart 2) et les sha256 ;
- mesure, **pour information**, le test synoptique à `keep = 1,0` (écart 4).

Codes de retour : 0 portage valide (a, b et c), 1 portage non valide, 2 JSON figé absent,
3 JSON figé modifié pendant le script (contrôle explicite, qui survit à `python -O`).

## Réserve : transférabilité au réel

Les coefficients sont appris sur des données **synthétiques**, où la pression des capteurs est
**sans bruit** et où la pression au niveau de la mer est un plan exact. La correction repose
surtout sur le vent géostrophique estimé à partir de la pression des capteurs. Or le signal est
faible : environ **26 Pa pour 10 m/s** sur les 25 km du domaine en y. Des baromètres réels
portent une erreur de **±10 à 50 Pa**, du même ordre que ce signal, auquel s'ajoutent les
erreurs de réduction au niveau de la mer sur un relief réel. Sur données réelles, l'estimateur
et les coefficients sont donc à **recalibrer** (par exemple sur ERA5 et des radiosondes, voir
`docs/REAL_DATA.md`), et le gain annoncé n'est pas transférable tel quel.
