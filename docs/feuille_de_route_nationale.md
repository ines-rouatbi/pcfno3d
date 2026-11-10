# Feuille de route nationale (9 novembre 2026)

Décision de Rayen et d'Ines : l'objectif du projet est le modèle **national**, de 0 à 1 km
au-dessus du sol. Le Grand Tunis v4 reste la preuve de concept et le zoom à 100 m ; plus aucun
entraînement n'y est prévu.

Discipline, à chaque phase :
- une règle écrite, datée et hachée avant le calcul ;
- réglages sur la validation seulement ;
- une seule lecture du test par modèle ;
- aucun entraînement ni aucune génération sans accord écrit ;
- aucun bruit ajouté aux prédictions ;
- ERA5 et vérité générée = « vérité d'entraînement », jamais « mesures » ;
- un `run_dir` par modèle.

## 0. Points a) à e)

| Point | Résultat |
|---|---|
| a) Renommage | Fait le 09/11 : `runs/grand_tunis_v4_dso/best (1).pt` renommé en `best_epoque10_copie.pt`, après vérification de son empreinte (époque 10, sha256 `e2f5658d…b483bb467`), sans suppression. `best.pt` : époque 11, sha256 `e6f7317462c8772e11e85155f96c940c461266b39b83fd23b1e2646e4ff74434`, noté dans `docs/MODEL_CARD_GRAND_TUNIS.md`. Attention : `runs/grand_tunis_v9_production/best.pt` n'a pas été trouvé à cet emplacement, donc l'empreinte de la copie servie reste à vérifier. |
| b) Mise à jour du forçage | `grand_tunis_donnees.zip` contient encore l'ancien forçage (SST, température de peau, couche limite d'ERA5). Le forçage `--sans-slx` du 21/10 (une seule règle pour tout le jeu) a été déposé à côté, sous la forme `maj_forcage_era5.csv/.npz`. Le disque `/content` de Colab étant vidé à chaque session, le carnet extrait l'archive puis recopie les `maj_*` par-dessus à chaque lancement. Le résultat est toujours le même forçage, celui de la vérité. Pour supprimer ce message, il faut reconstruire l'archive (modification du Drive, sur accord). |
| c) Modèle national livré | **v2**. La v3 échoue la troisième condition de `regle_v3_rugosite.json` : aéroports 3,075995 m/s pour un maximum de 1,01 × 3,045499 = 3,075954 m/s. Pas de relance de `dso_hauteur_sol.py` sur la v3. |
| d) Scripts fournis | `scripts/dso_hauteur_sol.py` est dans la branche (commit 1b25f31), avec une correction de l'index ERA5 dans `tfeat`. **`scripts/postraitement_grand_tunis.py` n'a pas été reçu** : il n'est ni dans les messages, ni dans le dépôt, ni sur le disque. |
| e) Post-traitement du Grand Tunis | Non fait, faute du script. Dès réception : étapes `regles` puis `validation`, et envoi de `validation_variantes.csv` avant l'étape `test`. |

## 1. Phase 1 : diagnostics nationaux (sans entraînement)

**État des 5 diagnostics (10/11/2026) : la phase 1 est terminée.**

| # | Diagnostic | État |
|---|---|---|
| 1 | Modes de Fourier, grille, plus petite longueur d'onde | **fait** (ci-dessous) |
| 2 | Spectres de l'erreur à 300 / 800 m au-dessus du sol, 180 instants de test | **fait** (10/11, Colab L4, ci-dessous) |
| 3 | RMSE u, v par tranche, 60 instants d'entraînement contre test | **fait** (10/11, Colab L4, ci-dessous) |
| 4 | Mécanismes du générateur coupés un par un, 12 instants de validation 2021 | **fait** (09/11, en local, ci-dessous) |
| 5 | Biais aux aéroports | **fait** (ci-dessous) |

**Diagnostics 2 et 3** (`scripts/diagnostic_v_phase1.py`, commit 1d3236a, carnet Colab cellule 4, L4, 10/11/2026
08:38). Sorties : `MyDrive/pcfno3d_national/runs/national_1km_v2/diagnostic_v_phase1/`. Verdicts de
`scripts/juger_phase1.py` (règle sha256 2982322…0bd1da, vérifiée) dans
`docs/resultats/national/phase1_diag23/verdict_phase1_diag234.md`. Les CSV de ce dossier sont recopiés du rapport
du Drive, arrondis à 3 décimales ; l'arrondi ne déplace aucun verdict, les écarts aux seuils étant d'au moins 0,03.

Part de l'énergie de l'erreur du modèle le long de y (nord-sud), test 2022 :

| Hauteur au-dessus du sol | u, bande 13–26 km | v, bande 13–26 km | v, > 50 km |
|---|---|---|---|
| 300 m | 0,176 | 0,051 | **0,872** |
| 800 m | 0,087 | 0,027 | **0,929** |

- **Ablation des modes [6, 64, 32] : non proposée.** L'erreur sur v n'est pas dans la bande 13–26 km
  (5 % et 3 %, seuil 25 %). La coupure spectrale nord-sud n'est donc pas la cause.
- **L'erreur sur v est dominée par la grande échelle** (87 % et 93 % au-delà de 50 km) : **priorité à la
  phase 2**, c'est-à-dire à la vérité d'entraînement. C'est cohérent avec le diagnostic 4.

RMSE (m/s) par tranche au-dessus du sol, 60 instants d'entraînement contre 60 instants de test :

| Tranche | Entraînement u | Entraînement v | Test u | Test v | v/u entraînement | v/u test |
|---|---|---|---|---|---|---|
| 0–100 m | 0,477 | 0,602 | 0,463 | 0,603 | 1,26 | 1,30 |
| 100–300 m | 0,522 | 0,726 | 0,519 | 0,726 | 1,39 | 1,40 |
| 300–600 m | 0,641 | 0,851 | 0,608 | 0,853 | 1,33 | 1,40 |
| 600–1000 m | 0,625 | 0,850 | 0,585 | 0,878 | 1,36 | 1,50 |

- **Pas de sur-apprentissage** : le rapport test / entraînement est d'au plus 1,03, sous le seuil de 1,15.
- **L'écart entre v et u existe déjà à l'entraînement** (rapport v/u de 1,33 à 1,39 au-dessus de 100 m). Il vient
  donc de la vérité d'entraînement ou de l'architecture, pas d'une mauvaise généralisation.

**Diagnostics complémentaires 5, 6, 7 (demande d'Ines du 10/11, règle 752e16c)** :
`docs/resultats/national/rapport_diagnostics_5_6_7.md`.
- **5. Cohérence de l'ERA5 :** cohérent, aucun bug. Le modèle et le générateur voient le même ERA5 à
  1,4·10⁻⁶ m/s près ; latitudes, instant UTC, interpolation au-dessus du sol et signe de v sont vérifiés.
- **6. Forme de l'erreur sur v :** un plan par instant n'explique que 31 à 32 % de son énergie (8 % pour u) et la
  moyenne seule 2 à 3 %. Ce n'est donc pas une correction uniforme par instant. Une seule corrélation est
  signalée : la moyenne à 300 m avec la direction géostrophique, ρ = −0,32.
- **7. Correction de la phase 2 :** à instants égaux (acceptation), l'écart sur v à 300 m passe de 2,24 (générateur
  v2) à 0,86 (passe-haut) puis à 0,43 (β = 0,5). Aux aéroports, la vérité passe de 3,114 à 2,925 m/s, contre 2,940
  pour ERA5.

**Conclusion de la phase 1.** Les diagnostics 2 à 4 désignent la même cause : la vérité d'entraînement
(amortissement de w dans le générateur), et pas les modes de Fourier ni le sur-apprentissage. La phase 3 garde
donc l'ablation C (même architecture, nouvelle vérité) et n'a pas besoin de l'ablation des modes.

**Diagnostic 1 : modes de Fourier du national v2** (`configs/national_1km_colab.yaml`)

| | z | y (nord-sud) | x (est-ouest) |
|---|---|---|---|
| Grille | 16 niveaux | 828 nœuds à 1 km | 388 nœuds à 1 km |
| Pad | 4 | 8 | 8 |
| Modes gardés | 6 | 32 | 32 |
| Plus petite longueur d'onde spectrale | — | (828 + 8) / 31 ≈ **27,0 km** | (388 + 8) / 31 ≈ **12,8 km** |

Largeur 48, convolution locale activée.

**Diagnostic 4 : mécanismes du générateur sur la VALIDATION 2021** (`scripts/attribution_v_generateur.py
--validation`, 12 instants à 3 h et 15 h UTC, 6 jours de janvier, février et juillet 2021). Écart rms entre la vérité
d'entraînement et ERA5 (m/s), au-dessus du sol. Entre parenthèses, la réduction par rapport au générateur actuel.
Verdict rendu par `scripts/juger_phase1.py` selon la règle b543873 (sha256 2982322…0bd1da), fixée avant le calcul.

| Configuration | u 300 m | v 300 m | u 800 m | v 800 m |
|---|---|---|---|---|
| actuel | 1,188 | 1,461 | 1,150 | 1,381 |
| sans amortissement de w | 0,252 (−79 %) | 0,264 (−82 %) | 0,151 (−87 %) | 0,155 (−89 %) |
| **correction passe-haut** | 0,533 (−55 %) | 0,555 (−62 %) | 0,414 (−64 %) | 0,456 (−67 %) |
| sans brise | 1,186 | 1,455 | 1,149 | 1,380 |
| sans vents de pente | 1,180 | 1,433 | 1,186 | 1,403 |
| sans relief | 1,191 | 1,462 | 1,150 | 1,381 |
| sans perturbations aléatoires | 1,182 | 1,456 | 1,146 | 1,377 |
| sans profil de couche limite | 1,102 (−7 %) | 1,393 (−5 %) | 1,108 (−4 %) | 1,339 (−3 %) |
| sans recalage par cellule | 3,565 (+200 %) | 2,761 (+89 %) | 3,200 (+178 %) | 2,460 (+78 %) |

Verdicts selon la règle :
- **Mécanisme à corriger en phase 2 : l'amortissement de w**, seul au-dessus de 30 %. Il porte 82 à 89 % de l'écart
  sur v. Les autres mécanismes pèsent 0 à 7 %, et le recalage par cellule est indispensable.
- **Correction passe-haut RETENUE pour la phase 2** : elle réduit v de 62 à 67 % à 300 et 800 m, et u baisse aussi
  (−55 à −64 %) au lieu d'augmenter.
- L'asymétrie entre v et u vient du générateur : le rapport v/u de l'écart passe de 1,20–1,23 avec le générateur
  actuel à 1,04–1,10 avec la correction.
- Ces chiffres confirment sur la validation ceux mesurés plus tôt sur des jours d'entraînement (85 à 92 %).

Le même nombre de modes sur un axe deux fois plus long fait que la partie spectrale est **deux
fois plus grossière du nord au sud que d'est en ouest**. Entre 2 et 27 km dans la direction
nord-sud, seule la convolution locale reconstruit le champ.

C'était une cause possible de l'écart entre v et u. **Le diagnostic 2 l'écarte (10/11)** : seulement 3 à 5 %
de l'erreur sur v se trouve dans la bande 13–26 km le long de y.

**Forçage.** Les 7 scalaires (`scripts/forcage_era5.py`) sont des **moyennes sur tout le
domaine** (828 × 388 km plus une marge de 0,15°) :
- ug et vg à 850 hPa ;
- cisaillement entre 850 et 700 hPa ;
- rotation du vent entre 10 m et 850 hPa ;
- stabilité (θ 1000–925 hPa contre θ à 2 m) ;
- contraste terre-mer (température à 2 m sur terre moins sur mer).

Le modèle reçoit **aussi** le vent ERA5 à chaque nœud, sur les 16 niveaux. Le générateur utilise
les **mêmes** scalaires de domaine pour la rotation, le cisaillement au-dessus de la couche
limite, la stabilité et la brise ; seul le recalage de grande échelle sur ERA5 se fait par cellule.

**Forçage local contre scalaire, et erreur du modèle**
(`scripts/diagnostic_forcage_regional.py`, 180 instants de test, national v2) :

| Région | RMSE v du modèle à 800 m au-dessus du sol | RMSE u | Écart de cisaillement local − domaine | Écart de rotation local − domaine |
|---|---|---|---|---|
| Nord (> 35,5° N) | 0,926 | 0,671 | 1,1·10⁻³ s⁻¹ | 23,4° |
| Centre | 0,822 | 0,630 | 0,9·10⁻³ s⁻¹ | 22,8° |
| Sud (< 33,5° N) | 0,875 | 0,512 | 0,8·10⁻³ s⁻¹ | 20,0° |

Corrélation de Spearman entre l'écart local − domaine et le RMSE de v :

| | Sur tous les couples (instant, région) | Moyenne à l'intérieur des régions |
|---|---|---|
| Cisaillement | +0,08 | +0,07 |
| Rotation | −0,07 | −0,07 |

**Conclusion.** La rotation du vent varie bien de 20 à 23° entre une région et la moyenne du
domaine. Mais l'erreur du modèle sur v **n'augmente pas** là où cet écart est grand. L'hypothèse
du forçage unique comme cause principale de l'erreur sur v n'est **pas confirmée** par ce
diagnostic.

L'explication la plus probable : le modèle reçoit déjà le profil ERA5 local à chaque nœud, et le
générateur utilise lui aussi les scalaires du domaine. La stabilité n'est pas encore testée de la
même manière ; c'est à ajouter si l'on veut l'exclure.

Carte : `MyDrive/pcfno3d_national/runs/national_1km_v2/diagnostic_forcage/carte_rmse_800m.png`.

**Résultats déjà mesurés** (`docs/diagnostic_v_national.md`) :
- l'amortissement de w du générateur explique **85 à 92 %** de l'écart entre la vérité et ERA5 à
  300 et 800 m au-dessus du sol ;
- les perturbations aléatoires du générateur n'expliquent que **1,5 à 7 %** de l'erreur du modèle
  sur v ;
- ERA5 seule a un v **1,4 à 1,6 fois** pire que u face à la vérité ;
- aux aéroports, le générateur surestime surtout la vitesse à Gafsa et Remada (environ
  +0,9 m/s par rapport à ERA5). Les autres gros biais face aux mesures (Tozeur, Sfax, Kairouan)
  sont déjà dans ERA5.

**Le premier levier n'est donc pas le forçage par cellule : c'est l'amortissement de w dans le
générateur** (correction déjà écrite, `ANEO_GEN_AMORT_PASSE_HAUT`, commit 7b05b3d).

Test rapide des facteurs de perturbation (20 instants de validation) : voir l'annexe B1.

## 2. Plan des phases 2 à 8

Les durées sont estimées sur A100 pour le GPU et sur un Colab CPU de 8 cœurs pour la génération.
« Accord » signifie un accord écrit d'Ines avant de lancer.

### Phase 2 : nouvelle vérité nationale (dépend de la phase 1)
- **Correction** : amortissement de w filtré au-dessus de 12 km (déjà écrit). Recalibration aux
  12 aéroports sur les jours de **validation** seulement : amplitude de la brise, accélération par
  le relief (Gafsa, Remada), z0 de terrain dégagé aux aérodromes.
- **Fichiers** :
  - `scripts/generate_aneo_3d.py` : options, désactivées par défaut ;
  - `scripts/calibrer_generateur_stations.py` : passer à la validation 2021 au lieu de
    2023–2025 ;
  - `scripts/test_facteurs_perturbation.py`, à étendre aux nouvelles options ;
  - `configs/national/regle_verite_v3.json`, à écrire avant le test.
- **Critère d'acceptation** (écrit avant) : sur 20 instants de validation, RMSE vectoriel moyen
  de la vérité aux 12 aéroports ≤ celui d'ERA5 seule. Aujourd'hui : 3,11 contre 2,86 m/s.
- **Durée** : test sur 20 instants, environ 1 h en local par variante. Régénération complète
  après accord : environ 8 à 10 h de CPU sur Colab, en 3 parties, avec environ 50 Go sur le Drive.
- **Risque** : même corrigé, le générateur pourrait rester moins bon qu'ERA5 aux stations. La
  calibration d'α montre que le détail local n'est pas corrélé à l'erreur réelle. Si le critère
  n'est pas atteint, l'option de repli est une vérité dont le détail local est atténué.

**Résultats de la phase 2 (09/11/2026, en local, validation 2021 seulement)**

Règle `configs/national/regle_verite_v3.json` (sha256 4792413…), écrite avant le calcul. Les 12 variantes
utilisent toutes l'amortissement passe-haut. Erreur vectorielle aux 12 aéroports, sur les 20 instants de
calibration (ERA5 seule : 2,700 m/s) :

| Perturbations | Brise | Relief 0 | Relief 0,5 | Relief 1 |
|---|---|---|---|---|
| 0 | 0,5 | 2,716 | 2,719 | 2,721 |
| 0 | 1 | 2,748 | 2,751 | 2,753 |
| 0,4 | 0,5 | 2,736 | 2,739 | 2,741 |
| 0,4 | 1 | 2,768 | 2,770 | 2,773 |

Variante choisie par la règle : perturbations 0, brise 0,5, relief 1. Sur les 20 instants d'acceptation
(jours disjoints), elle donne 2,948 m/s contre 2,940 pour ERA5 seule : **refusée**, de 0,3 %.

**Repli** (règle `configs/national/regle_verite_v3_repli.json`, sha256 7683491…, écrite avant le calcul) :
garder une part β du détail local, u, v = ERA5 + β (généré − ERA5), w par la continuité.

| β | Aéroports, calibration (m/s) | Écart v 300 / 800 m |
|---|---|---|
| 0,25 | 2,687 | 0,236 / 0,201 |
| **0,5** | **2,685** | 0,449 / 0,355 |
| 0,75 | 2,696 | 0,669 / 0,519 |

**β = 0,5 ACCEPTÉ** : 2,925 m/s contre 2,940 pour ERA5 seule sur les instants d'acceptation, meilleure à
8 aéroports sur 12. C'est la première vérité d'entraînement nationale qui fait mieux qu'ERA5 aux stations.
L'écart vérité − ERA5 sur v à 300 m tombe de 1,46 (v2) à 0,43 m/s.

Réglages de la vérité v3 : passe-haut, perturbations 0, brise 0,5, relief 1, β = 0,5. Carnet de
régénération : `notebooks/current/pcfno3d_national_generation_v3_colab.ipynb`
(`scripts/make_national_nb.py --verite-v3`), fichiers `cache_v3_partie_*` distincts de la v2.
**Régénération complète : accord écrit d'Ines le 10/11/2026**, après lecture des 4 points demandés
(`docs/resultats/national/reponse_ines_avant_regeneration.md`). L'accord porte sur la **régénération de la vérité v3
seulement**. L'entraînement de la v4 demandera un accord distinct.
- **Carnet :** `notebooks/current/pcfno3d_national_generation_v3_colab.ipynb`. Le générateur vient du commit
  `effc49c`, inchangé depuis la phase 2.
- **Déroulement :** les 3 parties s'enchaînent dans une seule session Colab CPU, avec un point de reprise sur le
  Drive après chacune ; la session se libère à la fin.
- **Sorties :** `cache_v3_partie_<k>.zip` et `jeu_v3.zip`. Rien de la v2 n'est écrasé.

### Phase 3 : modèle national v4 (dépend de la phase 2)
- **Fichiers** :
  - `src/pcfno3d/forcage.py`, `geometry.py`, `dataset.py` : scalaires par cellule (grille ERA5 de
    0,25°) au lieu des moyennes de domaine, avec les mêmes formules que le générateur ;
  - `src/pcfno3d/dataset.py` : dégradation réaliste de l'entrée ERA5 (écart et longueur de
    corrélation estimés sur ERA5 − observations, jours d'entraînement seulement) et masquage par
    zones ;
  - `configs/national_v4_*.yaml` : une configuration par ablation A à D ;
  - `configs/national/regle_v4_national.json`, écrit avant l'entraînement ;
  - tests associés.
- **Ablations** (validation seulement) :
  - A = v3 actuelle ;
  - B = A + forçage par cellule ;
  - C = B + nouvelle vérité ;
  - D = C + entrée ANEO.
  Puis 3 graines de la variante retenue.
- **Durée** : environ 1 h 45 à 2 h par entraînement de 30 époques. 4 ablations plus 3 graines,
  soit environ 7 entraînements et 14 h d'A100. Une seule évaluation du test : environ 1 h (DSO
  au-dessus du sol, aéroports, radiosondages, comparaison à ERA5).
- **Remarque** : la phase 1 ne montre pas de lien entre forçage unique et erreur sur v. B pourrait
  donc ne rien apporter ; l'ablation le dira sur la validation.

### Phase 4 : OS7 national, prévision (dépend de la phase 3)
- **Données** : archive GFS NOAA (`noaa-gfs-bdp-pds`), échéances f24 et f48. La boîte nationale
  (30–38° N, 7–12° E) est **déjà** écrite par `pcfno3d.gfs` lors du téléchargement OS7 du
  07/11 (`fichier_national`, `--assembler-national`), mais seulement pour les instants du jeu du
  Grand Tunis. Il faut compléter pour les instants du jeu national (1000 instants, 2 échéances,
  9 niveaux), environ 6 à 8 h de téléchargement.
- **Fichiers** :
  - `configs/national_v5_gfs_colab.yaml` ;
  - `scripts/make_national_nb.py` : carnet OS7 ;
  - `scripts/evaluer_prevision.py` et `verifier_decisions.py`, passés aux 12 aéroports.
- **Durée** : affinage d'environ 12 époques (1 h) et évaluation PREV1 à PREV3 (1 h).

### Phase 5 : OS6 national, réseau ANEO (dépend de la phase 3)
- **Fichiers** :
  - `scripts/os6_reseau.py`, étendu au national : candidats sur terre, pente ≤ 20°, 1 km des
    frontières, des stations et des autres candidats, zones exclues en paramètre (GeoJSON) ;
  - capteur virtuel = vérité à la hauteur du mât, avec un bruit réaliste ;
  - placement glouton par 5 capteurs, sur la validation ;
  - courbe de dimensionnement sur le test, une seule fois, de 10 à au moins 400 capteurs ;
  - sorties CSV et GeoJSON.
- **Paramètres attendus de Tukhnanutha** : coût unitaire, hauteur du mât, zones exclues.
- **Durée** : environ 4 à 6 h d'A100 (le glouton fait une inférence par étape).
- **Condition** : comme au Grand Tunis, le résultat sera négatif si le modèle n'apprend pas à
  exploiter les capteurs. C'est l'objet de l'ablation D (entrée ANEO) en phase 3.

**État de la phase 5 (11/11/2026) : code, règle PROPOSÉE et carnet écrits ; rien n'est calculé.**
- **Règle** `configs/national/regle_os6_national.json` (+ `.sha256`), statut **PROPOSITION** : à valider par
  Ines avant tout calcul. Points à trancher listés dans la règle (`a_trancher_par_ines`) : condition principale
  « propre » ou « ERA5 dégradée », lissage du glouton (10 km proposés), altitude ≤ 650 m (celle de D),
  hauteur de mât, zones exclues et coût (Tukhnanutha), seuil et tailles.
- **Code** : `pcfno3d.os6` généralisé (pas de copie) et `scripts/os6_reseau_aneo.py --regle`. Capteur simulé
  comme à l'entraînement de D (`os6.CapteursSimules`, qui réutilise `aneo_entree.EntreeANEO.valeur_a_hauteur` et
  `SIGMA_CAPTEUR` : vérité d'entraînement à 25 m au-dessus du sol + 0,1 m/s + 2 %), entré comme une station
  (prioritaire sur le pseudo-capteur ERA5) ; critère 0–1 km au-dessus du sol (tranches de `dso_hauteur_sol.py`) ;
  lat, lon par la projection de la grille (`os6.GeoRefGrille`) ; zones exclues en GeoJSON (`--zones-exclues`) ;
  graine de D choisie sur la validation (`os6.meilleure_graine`). Sans `--regle`, le Grand Tunis donne les mêmes
  fichiers et les mêmes colonnes qu'avant (vérifié sur le mini-jeu : mêmes candidats, même ordre, écarts
  numériques sous 1e-6 m/s).
- **Carnet** `notebooks/current/pcfno3d_national_os6_colab.ipynb` (`scripts/make_national_os6_nb.py`), A100,
  sans entraînement, code au commit figé, **uniquement avec l'accord écrit d'Ines**. Sorties :
  `MyDrive/pcfno3d_national/runs/os6_national_D_s<graine>/`, mêmes fichiers que le Grand Tunis (plan CSV et
  GeoJSON, `courbe_test.csv`…) plus `courbe_test_par_tranche.csv`, la courbe d'information ERA5 dégradée,
  `choix_modele.json` et la copie de la règle.
- **Ressources estimées (non mesurées)** : environ 3 h 30 sur A100, 20 à 24 Go de GPU par lot de 4 dispositions.

### Phase 6 : agriculture nationale (dépend de la phase 4)
- **Données** : SYNOP INM via NOAA ISD (`scripts/isd_synop.py`). Il y a 26 stations tunisiennes
  actives en 2021–2022. 12 sont déjà couvertes par les METAR ; **14 stations INM à télécharger**
  pour 2021 à 2025 : Kelibia 607200, Béja 607230, Jendouba 607250, Le Kef 607320, Siliana 607340,
  Thala 607380, Kasserine 607390, Mahdia 607420, Sidi Bouzid 607480, Kebili 607640,
  Gabès INM 607670, Médenine 607700, Tataouine 607720, El Borma 607800. Nabeul et Zaghouan sont
  déjà présents. Environ 30 minutes de téléchargement, gratuit et sans compte.
- **Fichiers** :
  - `src/pcfno3d/atom_wind.py` et `scripts/verifier_decisions.py` : vent de 2 à 10 m, fenêtres
    J0 à J+2 et FAO-56 sur la grille nationale ;
  - AGRI1 et AGRI2 contre les stations INM 2022.
- **Durée** : environ 1 jour de développement et 2 h de calcul.

### Phase 7 : éolien (dépend de la phase 3)
- Carte nationale du potentiel de 50 à 150 m (1 km), meilleures zones, zoom Grand Tunis à 100 m,
  production J0 à J+2, alertes du parc, EOL2 contre les radiosondages de 0 à 250 m.
- **Fichiers** : `src/pcfno3d/windfarm.py`, `atlas.py`, `scripts/` (atlas national).
- **Durée** : environ 1 jour de développement et 3 h de calcul (atlas sur l'ensemble des
  instants).

### Phase 8 : ATOM Wind national (dépend des phases 5 à 7)
- **Fichiers** :
  - `webapp/atom-wind/` : carte nationale, vue 3D, onglets Agriculture et Éolien, zoom
    automatique sur le modèle 100 m du Grand Tunis ;
  - nouvel onglet admin « Réseau ANEO », d'après la maquette `reseau_aneo.html` **(à fournir)** ;
  - `service/atom_outils.py` et `src/pcfno3d/atom_wind.py` : route `/api/atom-wind/reseau-aneo`,
    avec le bandeau « Démonstration : plan d'exemple » tant que les résultats OS6 manquent, et le
    statut « installé » qui relance avec `--installes` ;
  - `scripts/build_atom_wind.py`, ancres, i18n (fr, en, ar), accès admin seulement ;
  - tests web.
- **Durée** : environ 2 à 3 jours de développement.

### Phase 9 : documents
Fiche du modèle national, historique des versions des DSO, DSO1 par niveau et par tranche
au-dessus du sol, résultats des ablations. Environ une demi-journée.

## 3. Données à télécharger

| Donnée | Source | Volume ou durée | Phase |
|---|---|---|---|
| GFS f24/f48, boîte nationale, instants du jeu national | NOAA AWS `noaa-gfs-bdp-pds`, lecture par plages d'octets via `.idx` | environ 6 à 8 h | 4 |
| SYNOP INM, 14 stations, 2021–2025 | NOAA NCEI ISD (`scripts/isd_synop.py`) | environ 30 min, quelques Mo | 6 |
| Zones exclues, coût, hauteur de mât | Tukhnanutha | — | 5 |
| Maquette `reseau_aneo.html` | Tukhnanutha / Ines | — | 8 |

## 4. Ordre proposé et chemin critique
1. Phase 2 : environ 1 jour de test, puis 10 h de génération après accord.
2. Phase 3 : environ 2 jours, dont 14 h de GPU.
3. Phases 4, 5 et 7 en parallèle : chacune environ 1 jour.
4. Phase 6 : après la phase 4.
5. Phase 8 : après les phases 5 à 7.
6. Phase 9 : à la fin.

Total : environ 8 à 10 jours ouvrés, si les accords arrivent sans délai.
