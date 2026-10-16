# Lot M2 — fiabilité du champ reconstruit (DSO6)

*Module : `src/pcfno3d/reliability.py`. Besoins BF7, BF8, BF9 ; critère DSO6 ; contrat du
nœud virtuel (§ 8.4) et contrat ATOM **révision 2** (`DECISIONS-rev2.md` : D6, D7, D13, B3,
B5 ; `common.schema.json` : `sigma_ms`, `status_node`, `interval`, `uncertainty_method`,
`reliability_class` ; `quality.schema.json`).*

## Statut : prêt, **verdict DSO6 en attente de la session GPU**

Les tests CPU vérifient le mécanisme, mais pas la qualité des σ sur v7. Le chiffre DSO6 vient
de `scripts/m2_calibration_colab.py`, à lancer sur GPU.

## Ce que chaque nœud reçoit

Les noms et les codages sont ceux de `meta.fields` dans la révision 2 (D13).

| Champ | Codage du contrat | Contenu |
|---|---|---|
| `sigma_speed` | float32 | écart-type **calibré** de la norme horizontale (m/s), qui donne l'intervalle |
| `sigma_u`, `sigma_v`, `sigma_w` | float32 | écart-type **calibré** de chaque composante (m/s) |
| `status` | uint8 | 1 mesuré, 0 estimé, 255 sous le relief ; `mesure` / `estime` / `null` en JSON |
| `coverage` | uint8 quantifié | indice de couverture par les capteurs visibles, dans [0, 1] |
| `reliability` | uint8 quantifié | indice de fiabilité dans [0, 1] |
| `reliability_class` | — | `elevee` / `moyenne` / `faible` (profils et quality) |
| `speed_interval` | — | `p10`, `p50`, `p90` de la vitesse |
| `uncertainty` (par volume) | — | `method`, `members`, `calibrated`, `scale_factor`, `coverage_80_validation` ; `{method: "aucune", members: 1, calibrated: false}` sans M2 (B5, `UNCERTAINTY_AUCUNE`) |

« uint8 quantifié » : octet = round(254·x), 255 = absent, relu en octet × 1/254
(`to_u8_unit`, `U8_SCALE`). M2 calcule en float32 (`Volume`, `.npz`, manifeste web) ; la
publication `fields/{var}.f32|u8` est le travail de M6, qui réutilise ces fonctions.

## Méthode

**Valeur servie.** C'est la passe **unique** avec les capteurs fournis, inchangée (D6).
La moyenne de plusieurs passes dégrade DSO1 (étape A de v8) : l'ensemble sert uniquement à
mesurer la dispersion. Ainsi `p50 = speed`.

**σ brut.** On fait N passes masquées (défaut N = 8). Chacune garde chaque capteur visible
avec la probabilité `keep = 0,7`, avec des graines fixes. Les caractéristiques
géostrophiques et thermiques sont **recalculées** sur les capteurs gardés, comme pour le TTA.
Si la correction M1 est active, chaque passe la reçoit aussi.

    σ = sqrt( (1/N) · Σ_i (x_i − x_0)² )     x_0 = passe servie, pour u, v, w et la vitesse

σ brut est ensuite borné en bas par `reliability.sigma_floor_m_s` (0,01 m/s), le même
plancher en calibration, en production et en DSO6 : un nœud où tous les membres coïncident
n'a pas un intervalle de largeur nulle.

**Calibration (validation uniquement).** Le facteur k est calculé par composante et par
niveau :

    k = quantile_q( |erreur| / σ_brut ) / 1,2816

avec q = `reliability.calibration_q`. Il est mesuré sur la sélection synoptique de
validation (graine 1000), avec les capteurs visibles de DSO1 (`mask.keep_eval`). Les ratios
passent par un histogramme log10 de 4 000 classes, ce qui garde une mémoire constante.
`fit_calibration` et `Calibration.load` **refusent** le test. Le JSON gelé contient :

- les k et le quantile visé ;
- la couverture **obtenue** sur la validation ;
- le modèle, les graines et les membres.

Son sha256 est exposé.

**Intervalle (D6).** C'est exactement la formule du contrat :

    p50 = speed      p10 = max(0, speed − 1,2816 · sigma_speed)      p90 = speed + 1,2816 · sigma_speed

avec `sigma_speed` calibré (testé à 1e-6 près, et nœud par nœud dans `/api/profile`). C'est
une dispersion symétrique autour de la passe servie, **pas** les quantiles empiriques de
l'ensemble, écartés par le contrat.

**Statut (D7).** Un nœud est « mesuré » (1) si :

- la colonne (i, j) porte un capteur **visible** ;
- et |z_nœud − z_capteur| ≤ 15 m (`reliability.dz_status_m`).

La règle vaut **pour les données réelles comme synthétiques** : les capteurs simulés comptent,
c'est la démonstration de BF7. Sinon le nœud est « estimé » (0), et 255 sous le relief.
La nature des données (`versions.data_kind`) est portée ailleurs ; `predict(real_sensors=…)`
n'est plus qu'une information recopiée dans `reliability.info["donnees_reelles"]`, sans effet
sur le statut.

**Couverture.** Densité gaussienne des capteurs visibles (L = 1 km), saturée et atténuée
avec la hauteur au-dessus du sol (H = 500 m) :

    D = Σ_capteurs exp(−d²/2L²)      couverture = (1 − exp(−D/2)) · exp(−h_sol/H)

**Fiabilité.** L'indice vaut

    r = exp(−σ_vitesse / 1 m/s) · (0,5 + 0,5 · couverture)

La classe est `elevee` si r ≥ 0,75, `moyenne` si r ≥ 0,5, `faible` sinon
(`reliability.class_high` / `class_mid`).

**Pourquoi 0,75 / 0,5 (et plus 0,6 / 0,3).** Le tableau de bord ne lit pas la classe : il
la **recalcule** depuis le champ `reliability` publié, avec 0,75 et 0,5, les seuils des
instantanés de test du contrat (`contract/tools/make-fixtures.mjs`, branche `contrat-v1` du
dépôt du tableau de bord). Avec 0,6 / 0,3 côté modèle, un même nœud aurait eu deux classes
(par exemple r = 0,65 : `elevee` dans `quality.json` et les profils, `moyenne` sur la
carte). Les seuils ont donc été alignés sur ceux du tableau de bord (décision du
coordinateur, 2026-10-16) ; ce sont des choix d'ingénierie, pas des valeurs ajustées sur
des données, et aucun verdict (DSO6 porte sur σ, pas sur la classe) n'en dépend.

**Classe prise sur la valeur publiée.** `reliability` est publié en uint8 (octet / 254,
D13). La classe est calculée sur cette valeur quantifiée (`classe_octet`), exactement
comme le tableau de bord la recalcule, et non sur r brut : sans cela, un nœud à
r ∈ [0,498 ; 0,5[ (publié 127/254 = 0,5) serait `faible` pour le modèle et `moyenne` pour
la carte. Conséquence : r = 0,75 tout juste (publié 190/254 ≈ 0,748) est `moyenne`. Les
profils publient la même valeur quantifiée (`coverage`, `reliability` à 10⁻⁴), et les
exports NetCDF / CSV recalculent la classe depuis l'octet relu avec la même règle : une
seule classe par nœud partout (testé : `test_classe_prise_sur_la_valeur_publiee`,
`test_classe_de_fiabilite_comme_le_tableau_de_bord`).

Les constantes (L, H, D0, σ_ref, seuils de classe) sont des **choix d'ingénierie
documentés**, réglables dans la section `reliability`, et non des valeurs ajustées. Seuls
les k sont appris, sur la validation.

**Blocs `quality` (`quality.schema.json`).** `quality_by_level` produit `by_level` : parts
`elevee` / `moyenne` / `faible`, médianes de `sigma_speed` et de la couverture, `null` pour un
niveau sans nœud. `calibration_block` produit `calibration` depuis le résumé DSO6
(`evaluated_on`, `coverage_80`, `corr_error_sigma`, `dso6_pass`), ou `null` tant qu'aucun
verdict n'existe : jamais un 0 pour une valeur inconnue (B5).

## Utilisation

La fiabilité est strictement **opt-in** : `inference.reliability: null` par défaut. Dans ce
cas, la sortie (comparée bit à bit au modèle appelé directement), le `.npz`, le manifeste
web, `/api/meta`, `/api/reconstruct`, `test_summary.json` et le temps d'inférence sont
identiques à ceux d'avant M2 : aucune clé n'est ajoutée.

```yaml
inference:
  reliability: <run_dir>/fiabilite_pcfno3d/<modele>/session_.../calibration_validation.json
  # ou "non_calibre" : sigma bruts, calibrated: false
```

| Commande | Effet |
|---|---|
| `pcfno3d calibrate-reliability` | calibration sur la validation ; n'écrase jamais un JSON existant |
| `pcfno3d dso6 --calibration <json>` | verdict sur le test → `eval_pcfno3d/<modele>/dso6_<sha8>_keep<k>_n<instants>.json`, jamais écrasé |
| `scripts/m2_calibration_colab.py` | les deux, plus DSO4 avec et sans, dans un dossier de session neuf |

Une fois la fiabilité activée, elle apparaît partout :

- `Volume.reliability`, dont `sigma_speed` ;
- le `.npz` : σ (dont `sigma_speed`), couverture et fiabilité en float32, `status` et
  `reliability_class` en uint8, 255 sous le relief ;
- la section `fiabilite` du manifeste web, avec des binaires de même grille
  (`sigma_speed.bin` compris) ;
- `/api/profile`, avec les noms du contrat par niveau (`sigma_speed` compris) plus le bloc
  `uncertainty` ;
- `/api/meta`.

Le temps de l'ensemble est **compris** dans `inference_s`, donc dans DSO4.
`reliability_s` en donne la part, et l'évaluation publie DSO4 avec et sans
(`dso4_inference_s_sans_fiabilite`).

## DSO6

Deux conditions, mesurées sur le test synoptique avec la calibration de validation :

1. la couverture de l'intervalle à 80 % est ≥ 0,80, pour u, v et w ;
2. la corrélation de Spearman entre |erreur| et σ est > 0.

La couverture est un comptage exact. Spearman est calculé sur un échantillon à graine fixe
de 20 000 couples par niveau et composante, tiré en « bottom-k » : chaque couple reçoit une
clé uniforme et on garde les plus petites. Tous les instants y sont donc représentés à parts
égales. Un re-tirage parmi anciens + nouveaux couples ne gardait en pratique que les
derniers instants (défaut trouvé à la vérification, corrigé et testé).

## Choix tranchés (relecture stage-b5, contrat rév. 2)

1. **Marge de calibration.** `calibration_q` vaut **0,8** par défaut. Avec q = 0,8 pile, la
   couverture sur le test oscille *autour* de 0,80, et le verdict strict ≥ 0,80 dépend alors
   de l'échantillonnage (testé sur un bruit connu). Une marge éventuelle (q = 0,82 par
   exemple) se décide **sur la validation, avant** de regarder le test, et elle est écrite
   dans le JSON (`quantile`).
2. **`scale_factor`.** Le contrat n'a qu'un facteur. On y met le **k global de la vitesse**
   (celui de `sigma_speed`, qui donne l'intervalle) ; les k appliqués, par composante et par
   niveau, sont dans le JSON.
3. **Capteurs visibles.** La calibration et DSO6 se mesurent à `mask.keep_eval` (**50 %**),
   comme DSO1, alors qu'en production `predict` voit **100 %** des capteurs, donc des σ plus
   petits. La mesure à keep 1,0 est **prévue en session GPU** (`--keep 1.0`), pour vérifier
   que k ne dépend pas de la densité ; sinon, la calibration servie sera celle à 1,0.
4. **Correction M1.** Quand `inference.correction` est active, chaque passe de l'ensemble
   reçoit la même correction que la passe servie (même post-traitement dans
   `WindFieldModel._member`). La fusion avec M1 est essayée localement (branche
   `essai/m2-plus-m1`, jamais poussée) et testée par
   `test_avec_correction_m1_chaque_passe_est_corrigee`, ignoré tant que M1 n'est pas sur
   `main` et qui échoue si `_member` oublie la correction. La calibration est rangée dans le
   même dossier que l'évaluation (`..._corr<sha8>`) : elle ne vaut que pour cette sortie.
5. **Upload et stations réelles.** `real_sensors` **ne compte plus** pour le statut (D7). On
   le garde seulement comme information : `/api/reconstruct/stations` passe `True`, l'upload
   ANEO et le cache passent `False`, sans effet sur `status`.
6. **Coût.** N passes supplémentaires par instant. DSO4 (< 1 s) sur v7 avec 8 membres est
   à mesurer sur GPU ; on peut réduire `reliability.members` si besoin.

## Écarts restants avec les schémas rév. 2 (hors M2)

- `/api/profile` n'est pas la route du contrat : celle-ci (`/v1/profiles/{id}/{point_id}`)
  est produite par M6. Nos niveaux portent en plus `sigma_u/v/w`, que le profil rév. 2
  n'accepte pas (`additionalProperties: false`), et gardent leurs anciens noms
  (`z_agl_m`, `sous_le_relief`, `wind_*`).
- `sigma_ms` borne σ à 40 m/s dans le contrat : aucune troncature côté M2, c'est au
  publieur de vérifier.
- `min_agl_m` (B1) : le masque `geom.valid` vient de la grille du générateur. Que
  « NaN ⇔ z − relief < 5 m » tienne exactement est à vérifier par M6 à la publication.
