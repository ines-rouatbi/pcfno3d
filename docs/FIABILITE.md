# Lot M2 — fiabilité du champ reconstruit (DSO6)

*Module : `src/pcfno3d/reliability.py`. Besoins BF7, BF8, BF9 ; critère DSO6 ; contrat du
nœud virtuel (§ 8.4) et contrat ATOM **révision 2** (`DECISIONS-rev2.md` : D6, D7, D13, B3,
B5 ; `common.schema.json` : `sigma_ms`, `status_node`, `interval`, `uncertainty_method`,
`reliability_class` ; `quality.schema.json`).*

## Statut : version 2, **seconde lecture du test en attente de la session GPU**

Première mesure GPU (15-16/10/2026) : DSO6 **non conforme** (couverture u et w sous 0,80) et
DSO4 **non conforme** avec la fiabilité (4,563 s pour 8 membres). Chiffres complets dans
« Historique des mesures DSO6 » ci-dessous. La version 2 change deux choses :

- **A. Ensemble par lots** (DSO4) : les N passes masquées passent par lots de
  `reliability.batch` (défaut 4) dans un seul passage avant du réseau ;
- **B. Marge de calibration par une règle fixée d'avance**, sur la validation seule
  (`reliability.calibration_q: "auto"`, nouveau défaut).

La règle B a été fixée **après** la première lecture du test : le second verdict DSO6 n'est
donc pas une mesure indépendante, et il le dit (champ `mesure`, option `--deuxieme-lecture`).
Les tests CPU vérifient le mécanisme, pas la qualité des σ sur v7 : les chiffres viennent de
`scripts/m2_calibration_colab.py`, sur GPU.

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

**Ensemble par lots (v2).** Les N passes ne sont plus faites une par une : pour chaque lot
de `reliability.batch` passes (défaut 4 ; 1 = comportement d'avant), on construit les entrées
(`assemble_input`, avec pour chaque passe **ses** caractéristiques recalculées sur ses
capteurs) et on fait **un seul** passage avant du réseau (`WindFieldModel._members`), sous
le verrou du modèle. Les sorties sont identiques au calcul passe par passe à la précision
flottante près (lot de 1 : bit à bit ; lots de 3, 4 et 8 : écart ≤ 1e-5, testé par
`test_ensemble_par_lots_identique_passe_par_passe`). Chaque passe reçoit ensuite le même
post-traitement que la passe servie dans `WindFieldModel._post_member`, **point d'accroche de
la correction M1** (identité tant que M1 n'est pas fusionnée). La passe servie, elle, reste
calculée seule : elle est inchangée bit à bit. La mémoire d'un passage est multipliée par la
taille du lot ; sur GPU, un lot qui ne tient pas est recoupé en deux (même résultat). Sur
CPU (service local, ~5 Go par passe pour v7), garder `batch: 1` si la mémoire est juste.

Temps CPU mesuré sur le mini-jeu (grille 4 × 10 × 20, 8 membres, médiane de 30 appels,
deux séries) : ensemble seul 47–52 ms en lot de 1, 22–28 ms en lot de 4, 17 ms en lot de 8.
Le calcul des caractéristiques (`time_geo_features`, numpy) coûte 0,18 ms par passe, soit
~1,4 ms pour 8 passes : il n'est pas le poste dominant et n'a pas été modifié. Le gain réel
se mesure sur GPU avec v7 (DSO4 par taille de lot dans le script Colab).

σ brut est ensuite borné en bas par `reliability.sigma_floor_m_s` (0,01 m/s), le même
plancher en calibration, en production et en DSO6 : un nœud où tous les membres coïncident
n'a pas un intervalle de largeur nulle.

**Calibration (validation uniquement).** Le facteur k est calculé par composante et par
niveau :

    k = quantile_q( |erreur| / σ_brut ) / 1,2816

avec q = `reliability.calibration_q`. Il est mesuré sur la sélection synoptique de
validation (graine 1000), avec les capteurs visibles de DSO1 (`mask.keep_eval`). Les ratios
passent par un histogramme log10 de 4 000 classes, ce qui garde une mémoire constante.
`fit_calibration` et `Calibration.load` **refusent** le test.

**Choix de q : règle fixée d'avance (v2, `calibration_q: "auto"`, défaut).** Sur la
validation seulement :

1. la sélection de validation est coupée en **2 plis de jours**, de façon déterministe :
   jours de rang pair (pli a) et impair (pli b), dans l'ordre croissant des jours présents ;
2. pour chaque q de la grille fixe 0,80 ; 0,81 ; … ; 0,90 (`Q_GRILLE`), les k sont ajustés
   sur un pli et la couverture de l'intervalle à 80 % est mesurée sur l'autre, puis
   inversement ;
3. q retenu = le **plus petit** q dont la couverture hors-pli minimale (sur u, v, w et les
   deux sens) atteint 0,80 (`coverage_target`) ; si aucun q de la grille n'y arrive, 0,90,
   avec `atteint: false` dans le JSON et un message à l'écran ;
4. les k finaux sont ajustés sur **toute** la validation avec ce q.

Une seule boucle sur la validation : un histogramme par pli, leur somme donne celui de toute
la validation ; la règle ne refait aucune passe. Le JSON contient, sous `regle_quantile` :
la règle en clair, la grille, la cible, les plis (jours, instants par pli), la couverture
hors-pli de chaque q (dans les deux sens, par composante, et le minimum u, v, w) et le q
retenu. Un nombre dans `calibration_q` garde l'ancien comportement (q imposé, pas de
`regle_quantile`). Seul cas de repli : une validation d'un seul jour (le mini-jeu des tests)
est coupée par instants de rang pair / impair, ce que le JSON signale (`plis`) ; le jeu réel
a plusieurs jours de validation. Un pli vide est une erreur.

Le JSON gelé contient :

- les k, le quantile retenu (`quantile`) et l'option (`calibration_q`) ;
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
| `pcfno3d dso6 --calibration <json>` | verdict sur le test → `eval_pcfno3d/<modele>/dso6_<sha8>_keep<k>_n<instants>.json`, jamais écrasé ; `--deuxieme-lecture` ajoute le champ `mesure` |
| `scripts/m2_calibration_colab.py` | les deux (règle auto, ensemble par lots), plus DSO4 sans et avec la fiabilité, et avec des lots de 1, 4 et 8 pour information, dans un dossier de session neuf ; `--deuxieme-lecture` |

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

## Historique des mesures DSO6

Les mesures s'ajoutent ici, dans l'ordre ; aucune n'est retirée ni corrigée après coup.

### Première mesure — Colab L4, v7, 15-16/10/2026 (version 1)

Calibration de la version 1 : q = 0,80 imposé, 8 membres, `keep` 0,7, capteurs visibles
`mask.keep_eval`.

| | u | v | w | speed |
|---|---|---|---|---|
| k global (64 instants de validation) | 4,946 | 5,155 | 1,593 | 5,015 |
| Couverture 80 % obtenue sur la validation | 0,801 | 0,801 | 0,801 | 0,801 |
| Couverture 80 % sur le test (120 instants) | **0,751** | 0,811 | **0,794** | 0,796 |
| Spearman(\|erreur\|, σ) sur le test | 0,289 | 0,307 | 0,229 | 0,290 |

Verdict DSO6 : **NON CONFORME** (couverture de u et de w sous 0,80 ; la corrélation est
positive partout). DSO4 : 0,448 s sans fiabilité (conforme) ; **4,563 s** avec 8 membres
(**non conforme**, seuil 1 s).

### Seconde mesure — à venir (version 2)

Après cette première lecture, l'utilisatrice (propriétaire de l'application) a décidé la
version 2 : ensemble par lots (A) et marge de calibration par une règle fixée d'avance sur la
validation (B). **La règle B a été fixée APRÈS la première lecture du test** : le second
verdict n'est donc **pas une mesure indépendante** du test. Il se lance avec
`--deuxieme-lecture`, et le JSON du verdict comme le résumé du script portent alors :

    "mesure": "seconde mesure sur le test, apres une premiere lecture (15/10/2026, non conforme)"

Les chiffres de la seconde mesure seront ajoutés ici, à la suite de la première.

## Choix tranchés (relecture stage-b5, contrat rév. 2)

1. **Marge de calibration.** Version 1 : `calibration_q` valait **0,8** par défaut. Avec
   q = 0,8 pile, la couverture sur le test oscille *autour* de 0,80, et le verdict strict
   ≥ 0,80 dépend alors de l'échantillonnage (testé sur un bruit connu). La première mesure
   GPU l'a confirmé (u 0,751, w 0,794). Version 2 : la marge est choisie par la **règle
   fixée d'avance** décrite dans « Méthode » (`calibration_q: "auto"`), sur la validation
   seule, et tout son calcul est écrit dans le JSON (`regle_quantile`). Cette règle a été
   écrite **après** la première lecture du test (voir l'historique ci-dessous).
2. **`scale_factor`.** Le contrat n'a qu'un facteur. On y met le **k global de la vitesse**
   (celui de `sigma_speed`, qui donne l'intervalle) ; les k appliqués, par composante et par
   niveau, sont dans le JSON.
3. **Capteurs visibles.** La calibration et DSO6 se mesurent à `mask.keep_eval` (**50 %**),
   comme DSO1, alors qu'en production `predict` voit **100 %** des capteurs, donc des σ plus
   petits. La mesure à keep 1,0 est **prévue en session GPU** (`--keep 1.0`), pour vérifier
   que k ne dépend pas de la densité ; sinon, la calibration servie sera celle à 1,0.
4. **Correction M1.** Quand `inference.correction` est active, chaque passe de l'ensemble
   reçoit la même correction que la passe servie. Depuis l'ensemble par lots (v2), le
   post-traitement d'une passe est dans `WindFieldModel._post_member`, appelé pour chaque
   passe de chaque lot ; `_member` (une passe) passe par le même chemin. La fusion avec M1
   est essayée localement (branche `essai/m2-plus-m1`, jamais poussée) et testée par
   `test_avec_correction_m1_chaque_passe_est_corrigee`, ignoré tant que M1 n'est pas sur
   `main` et qui échoue si `_member` oublie la correction : à la fusion, la correction va
   dans `_post_member`, sinon les passes par lots en seraient privées. La calibration est
   rangée dans le même dossier que l'évaluation (`..._corr<sha8>`) : elle ne vaut que pour
   cette sortie.
5. **Upload et stations réelles.** `real_sensors` **ne compte plus** pour le statut (D7). On
   le garde seulement comme information : `/api/reconstruct/stations` passe `True`, l'upload
   ANEO et le cache passent `False`, sans effet sur `status`.
6. **Coût.** N passes supplémentaires par instant. Première mesure sur L4 : 0,448 s sans
   fiabilité, 4,563 s avec 8 membres passés un par un (DSO4 non conforme, seuil 1 s). La v2
   les passe par lots (`reliability.batch`) ; le script Colab mesure DSO4 sans, avec, et par
   taille de lot (1, 4, 8). Si cela ne suffit pas, on peut réduire `reliability.members`.

## Écarts restants avec les schémas rév. 2 (hors M2)

- `/api/profile` n'est pas la route du contrat : celle-ci (`/v1/profiles/{id}/{point_id}`)
  est produite par M6. Nos niveaux portent en plus `sigma_u/v/w`, que le profil rév. 2
  n'accepte pas (`additionalProperties: false`), et gardent leurs anciens noms
  (`z_agl_m`, `sous_le_relief`, `wind_*`).
- `sigma_ms` borne σ à 40 m/s dans le contrat : aucune troncature côté M2, c'est au
  publieur de vérifier.
- `min_agl_m` (B1) : le masque `geom.valid` vient de la grille du générateur. Que
  « NaN ⇔ z − relief < 5 m » tienne exactement est à vérifier par M6 à la publication.
