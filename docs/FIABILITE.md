# Lot M2 — fiabilité du champ reconstruit (DSO6)

*Module : `src/pcfno3d/reliability.py`. Besoins BF7, BF8, BF9 ; critère DSO6 ; contrat du
nœud virtuel (§ 8.4) et contrat ATOM (`common.schema.json` : `status_node`, `interval`,
`uncertainty_method`, `reliability_class`).*

## Statut : prêt, **verdict DSO6 en attente de la session GPU**

Les tests CPU vérifient le mécanisme, mais pas la qualité des σ sur v7. Le chiffre DSO6 vient
de `scripts/m2_calibration_colab.py`, à lancer sur GPU.

## Ce que chaque nœud reçoit

| Champ | Contenu |
|---|---|
| `sigma_u`, `sigma_v`, `sigma_w` | incertitude **calibrée** de chaque composante (m/s) |
| `speed_interval` | `p10`, `p50`, `p90` de la vitesse horizontale |
| `status` | `mesure` ou `estime` (`null` sous le relief) |
| `coverage` | indice de couverture par les capteurs visibles, dans [0, 1] |
| `reliability`, `reliability_class` | indice de fiabilité dans [0, 1] ; `elevee` / `moyenne` / `faible` |
| `uncertainty` (par volume) | `method`, `members`, `calibrated`, `scale_factor`, `coverage_80_validation`, exactement les clés du contrat |

## Méthode

**Valeur servie.** C'est la passe **unique** avec les capteurs fournis, inchangée (revue D6).
La moyenne de plusieurs passes dégrade DSO1 (étape A de v8) : l'ensemble sert uniquement à
mesurer la dispersion. Ainsi `p50 = speed`.

**σ brut.** On fait N passes masquées (défaut N = 8). Chacune garde chaque capteur visible
avec la probabilité `keep = 0,7`, avec des graines fixes. Les caractéristiques
géostrophiques et thermiques sont **recalculées** sur les capteurs gardés, comme pour le TTA.

    σ = sqrt( (1/N) · Σ_i (x_i − x_0)² )     x_0 = passe servie, pour u, v, w et la vitesse

**Calibration (validation uniquement).** Le facteur k est calculé par composante et par
niveau :

    k = quantile_q( |erreur| / σ_brut ) / 1,2816

avec q = `reliability.calibration_q`, par défaut 0,8. Il est mesuré sur la sélection
synoptique de validation (graine 1000), avec les capteurs visibles de DSO1
(`mask.keep_eval`). Les ratios passent par un histogramme log10 de 4 000 classes, ce qui
garde une mémoire constante. `fit_calibration` et `Calibration.load` **refusent** le test.
Le JSON gelé contient :

- les k ;
- la couverture **obtenue** sur la validation ;
- le modèle, les graines et les membres.

Son sha256 est exposé.

**Intervalle.** `p10, p90 = speed ∓ 1,2816 · σ_vitesse calibré`, avec p10 ≥ 0. C'est une
approximation gaussienne autour de la passe servie, **pas** les quantiles empiriques de
l'ensemble : avec 8 membres, un quantile à 10 % ne serait pas stable.

**Statut (revue D7).** Un nœud est « mesuré » seulement si trois conditions tiennent :

- les données sont **réelles** (`/api/reconstruct/stations`, `predict(real_sensors=True)`) ;
- un capteur visible se trouve dans la même colonne ;
- l'écart d'altitude vérifie |z_nœud − z_capteur| ≤ 12,5 m.

Sur le jeu synthétique, tout est « estimé ».

**Couverture.** Densité gaussienne des capteurs visibles (L = 1 km), saturée et atténuée
avec la hauteur au-dessus du sol (H = 500 m) :

    D = Σ_capteurs exp(−d²/2L²)      couverture = (1 − exp(−D/2)) · exp(−h_sol/H)

**Fiabilité.** L'indice vaut

    r = exp(−σ_vitesse / 1 m/s) · (0,5 + 0,5 · couverture)

La classe est `elevee` si r ≥ 0,6, `moyenne` si r ≥ 0,3, `faible` sinon.

Les constantes (L, H, D0, σ_ref, seuils de classe) sont des **choix d'ingénierie
documentés**, réglables dans la section `reliability`, et non des valeurs ajustées. Seuls
les k sont appris, sur la validation.

## Utilisation

La fiabilité est strictement **opt-in** : `inference.reliability: null` par défaut. Dans ce
cas, la sortie, le `.npz`, le manifeste web et le temps d'inférence sont identiques
(testé).

```yaml
inference:
  reliability: <run_dir>/fiabilite_pcfno3d/<modele>/session_.../calibration_validation.json
  # ou "non_calibre" : sigma bruts, calibrated: false
```

| Commande | Effet |
|---|---|
| `pcfno3d calibrate-reliability` | calibration sur la validation ; n'écrase jamais un JSON existant |
| `pcfno3d dso6 --calibration <json>` | verdict sur le test → `eval_pcfno3d/<modele>/dso6_<sha8>.json` |
| `scripts/m2_calibration_colab.py` | les deux, plus DSO4 avec et sans, dans un dossier de session neuf |

Une fois la fiabilité activée, elle apparaît partout :

- `Volume.reliability` ;
- le `.npz` : σ et couverture en float32, `status` et `reliability_class` en uint8, 255
  sous le relief ;
- la section `fiabilite` du manifeste web, avec des binaires de même grille ;
- `/api/profile`, avec les noms du contrat par niveau plus le bloc `uncertainty` ;
- `/api/meta`.

Le temps de l'ensemble est **compris** dans `inference_s`, donc dans DSO4.
`reliability_s` en donne la part, et l'évaluation publie DSO4 avec et sans
(`dso4_inference_s_sans_fiabilite`).

## DSO6

Deux conditions, mesurées sur le test synoptique avec la calibration de validation :

1. la couverture de l'intervalle à 80 % est ≥ 0,80, pour u, v et w ;
2. la corrélation de Spearman entre |erreur| et σ est > 0.

La couverture est un comptage exact. Spearman est calculé sur un échantillon à graine fixe
(20 000 couples par niveau et composante).

## Points à trancher (signalés, pas décidés)

1. **Marge de calibration.** Avec q = 0,8, la couverture sur le test oscille *autour* de
   0,80, et le verdict strict ≥ 0,80 dépend alors de l'échantillonnage (testé sur un bruit
   connu). Une marge (q = 0,82 par exemple) se décide **sur la validation, avant** de
   regarder le test.
2. **`scale_factor`.** Le contrat n'a qu'un facteur. On y met le k global de la vitesse, et
   les k appliqués (par composante et par niveau) sont dans le JSON.
3. **Capteurs visibles.** La calibration et DSO6 se mesurent à `mask.keep_eval` (50 %),
   comme DSO1. En production, `predict` voit 100 % des capteurs, donc des σ plus petits ;
   la calibration à 50 % est conservatrice ou non selon le comportement de k avec la
   densité, ce qui est à vérifier sur GPU (`--keep 1.0`).
4. **Correction M1.** Quand `inference.correction` est active, chaque passe de l'ensemble
   doit recevoir la même correction que la passe servie (même post-traitement dans
   `WindFieldModel._member`).
5. **Coût.** N passes supplémentaires par instant. DSO4 (< 1 s) sur v7 avec 8 membres est
   à mesurer sur GPU ; on peut réduire `reliability.members` si besoin.
