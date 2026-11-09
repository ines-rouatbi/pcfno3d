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

## 1. Phase 1 : diagnostics nationaux (faite, sans entraînement)

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
