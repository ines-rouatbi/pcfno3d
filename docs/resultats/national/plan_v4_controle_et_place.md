# Avant l'accord pour la v4 : contrôle du jeu v3, règle v4 corrigée, place sur le Drive (10/11/2026)

Demande d'Ines du 10/11, en trois points. Je n'ai rien supprimé ni entraîné.

## 1. Contrôle qualité du jeu v3 complet (après la régénération, sans entraînement)

Le protocole est fixé avant que le jeu existe. Il tourne sur Colab CPU, sur les fichiers `cache_v3_partie_<k>.zip`.

- **a) Conservation de la masse.**
  - Résidu r(k+½) de `meta.json`, calculé avec l'opérateur exact du générateur, sur les demi-niveaux dont les deux
    nœuds sont valides.
  - Instants couverts : les 1000, ou à défaut tous ceux de validation et de test plus 100 d'entraînement tirés avec
    une graine fixe.
  - On publie la RMS et le maximum, le même calcul sur la v2 pour référence, et le verdict « résidu au niveau de
    l'arrondi float16 de Y ».
- **b) Manifeste du cache.**
  - `pcfno3d manifeste-cache`, puis `--verifier`.
  - Le calendrier doit être celui de la v2, relu tel quel : 250 jours, 1000 instants, avec 700 d'entraînement,
    120 de validation et 180 de test.
  - Pour chaque instant de l'index, les trois fichiers `_S`, `_Y` et `_E` doivent exister.
  - Ni NaN ni infini dans `_S`, `_E`, ni dans u, v, w et ρ de `_Y` aux nœuds valides.
  - Les jours de chaque partie doivent être ceux du calendrier.
- **c) 10 instants de validation qui redonnent les chiffres du rapport.**
  - Les 10 instants sont tirés parmi les instants de validation du cache v3, avec la graine 2031.
  - Pour chacun, on recalcule la vérité avec le code validé en phase 2 (`compute_fields`, mêmes réglages v3, même
    jour, même heure, même graine et même secteur que le générateur).
  - **Égalité attendue :** l'écart maximal entre le `_Y` du cache et le recalcul ne doit pas dépasser l'arrondi
    float16.
  - **Recalcul des chiffres du rapport sur ces 10 instants :** écart rms vérité − ERA5 sur u et v à 300 et 800 m au-
    dessus du sol, et RMSE aux aéroports de la vérité et d'ERA5 seule. On les compare à ceux de l'acceptation
    (v : 0,434 et 0,320 ; u : 0,370 et 0,274 ; aéroports : 2,925 contre 2,940).
  - **Limite :** ces 10 instants ne sont pas les 20 de l'acceptation. Seuls les ordres de grandeur sont comparables ;
    l'égalité exacte est vérifiée par le recalcul.

Le résultat te sera envoyé avant toute demande d'accord pour la v4.

## 2. Règle v4 corrigée : `configs/national/regle_v4_national.json` (sha256 `eed8fd49…` ; une première version `81cf4b45…` est remplacée : les critères y sont reliés aux clés exactes que mesurent les scripts, la rugosité étant celle du domaine 0–1 km)

Elle remplace la proposition du 09/11 (sha256 `ebeb761d…`).
- **Même vérité.**
  - La v4 (graine choisie sur la validation) et le **modèle livré** (v2, poids inchangés, copiés dans
    `runs/national_1km_v2_sur_verite_v3`) sont évalués sur la **même vérité v3**.
  - Conditions identiques : mêmes 180 instants de test 2022, mêmes entrées, w par continuité, mêmes scripts. Le test
    est lu une fois par modèle.
- **Conditions pour que la v4 soit au moins aussi bonne** (tranches 0–1 km au-dessus du sol) :
  - **DSO1 :** RMSE max ≤ 1,01 × celui du modèle livré.
  - **DSO3 :** |rugosité − 1| ≤ celui du modèle livré + 0,02.
  - **DSO5 :** RMSE à 25 % des points d'entrée ≤ 1,01 × celui du modèle livré.
  - **DSO2 et DSO4 :** ils restent conformes.
- **Comparaison indépendante sur les mesures réelles** :
  - **Aéroports :** les 12, chacun retiré de l'entrée quand il est jugé. On publie l'écart v4 − livré avec son IC
    95 % (bootstrap sur les instants). Il faut v4 ≤ livré.
  - **Radiosondages de Tunis-Carthage, 0–1 km :** il faut v4 ≤ 1,03 × livré.
- **Décision :** la v4 remplace le modèle livré seulement si les **six conditions** sont remplies. Sinon la v2 reste
  livrée et la v4 est publiée comme essai.
- **Conformité :** les DSO sont rapportés avec les seuils inchangés pour **les deux modèles sur la vérité v3**. Les
  chiffres de la v2 sur la vérité v2 restent publiés à part, sans mélange.

## 3. Place sur le Drive pour les 3 graines

**Mesures du 10/11, 15 h** : 327,9 Go utilisés sur 400 Go, soit 72,1 Go libres avant la v3.

| Élément | Taille |
|---|---|
| Vérité v3 en cours : `cache_v3_partie_1/2/3.zip` (mêmes tailles que la v2 : 17,3 + 23,0 + 8,8) et `jeu_v3.zip` | ≈ 50 Go |
| **Libre après la v3** | **≈ 22 Go** |
| `best.pt` d'une graine (poids EMA, comme `national_1km_v2/best.pt`) | 1,69 Go |
| `last.pt` d'une graine (modèle, EMA, état d'Adam, pour la reprise après coupure) | 6,75 Go |
| Copie des poids du modèle livré pour l'évaluer sur la v3 (`runs/national_1km_v2_sur_verite_v3/best.pt`) | 1,69 Go |
| Sorties d'évaluation, DSO, journaux (par modèle évalué) | < 0,5 Go |

**Option A, proposée : rien à supprimer.**
- Les 3 graines s'entraînent avec `train.sync_last: false`. `last.pt` reste sur le disque local de Colab, et seul
  `best.pt` est recopié sur le Drive (toutes les 5 époques et à la fin).
- Besoin : 3 × 1,69 + 1,69 + ≈ 1 ≈ **7,8 Go**, d'où **≈ 14 Go libres à la fin**.
- Prix : si Colab coupe une session, la graine en cours repart de zéro, soit au plus environ 2 h perdues sur A100. Une
  coupure à l'époque 6 est arrivée le 08/11 pour la v2.

**Option B : reprise après coupure (`last.pt` sur le Drive).**
- Besoin : 7,8 + 3 × 6,75 ≈ **28 Go**, au-dessus des 22 Go libres. Il faut libérer au moins 6 Go, avec ton accord
  explicite.
- Attention : la corbeille du Drive compte dans le quota. La place n'est rendue qu'**après avoir vidé la corbeille**,
  c'est-à-dire par une suppression définitive.
- Candidats, du plus sûr au moins sûr (**rien n'est supprimé sans ton accord écrit**) :
  1. `pcfno3d_national/cache_partie_1/2/3.zip` (7/11), **49,1 Go**.
     - C'est le premier cache national. Il avait un vent nul à 0,5 % des nœuds de 2500 m (profils ERA5 vides
       au-dessus de 700 hPa) et a été remplacé par `cache_v2_partie_*` le 7/11 (commit `ac6a522`).
     - Plus aucun carnet ne le lit, et aucun résultat publié n'en dépend.
     - Il est impossible de l'archiver sur ines-pc, qui n'a que 49 Go libres.
  2. `runs/national_1km_v3_rugosite/last.pt`, **6,75 Go**.
     - Il vient de l'affinage rugosité v3, refusé par sa règle. Ce fichier ne sert qu'à reprendre un entraînement.
     - Son `best.pt` et ses résultats sont gardés.
  3. Les `last.pt` du Grand Tunis (`grand_tunis_v2_long`, `v8`), 6,75 Go chacun. Ce projet est figé, mais je ne
     propose d'y toucher que si tu le demandes.
- Après le choix de la graine, les `last.pt` des 3 graines (option B) ne serviront plus. Je te demanderai avant de
  les supprimer.

**Ce que je recommande :**
- **Option A,** qui ne supprime rien.
- **En plus, à ta décision,** la suppression définitive du cache v1 obsolète (49 Go), qui redonne de la marge pour la
  suite (phases 4 à 9).
