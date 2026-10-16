# M1 — Protocole de la dernière tentative DSO1 (650–1000 m)

*Protocole du lot M1, rédigé le 2026-10-14. La règle de décision ci-dessous a été versionnée
seule, avant toute mesure (commit `0b64978`), et complétée par l'Avenant n° 1 (commit `33da965`,
en fin de document) ; les sections intermédiaires ont été ajoutées ensuite et ne modifient ni l'une
ni l'autre. Rien n'a été exécuté sur GPU pour ce document : les chiffres nouveaux
viennent d'un diagnostic CPU (§ 3), reproductible avec `scripts/diag_geostrophique.py`.*

**En bref.**

- **Piste unique** : confirmer, sans réentraîner, la correction post-modèle de la v7
  (estimateur géostrophique v2 + ridge par niveau qui prédit le décalage moyen du niveau).
  Le réentraînement n'est qu'un repli, soumis à l'accord explicite de Rayen.
- **Étape 0 (CPU, faite)** : sur 608 instants d'entraînement et de validation, comparés au
  géostrophique **vrai** du générateur, l'estimateur v2 ramène l'erreur de 2,28 à 0,028 m/s
  (−98,8 %). Le principe de la correction est donc fondé. Sa transposition au réel ne l'est
  pas : avec de vrais baromètres (30 Pa) sur un réseau de 1 km, l'erreur remonte à 1,5 m/s.
- **Mesure décisive** : cellule BLOC 10i figée du notebook `fno3d_v7.ipynb`, une session GPU
  d'environ 1 h 30 (au plus 2 h).
- **Décision** : la règle pré-enregistrée ci-dessous. Si la correction est retenue, on gèle
  « v7 + correction » ; sinon, on gèle la v7 seule, avec le critère OMM.

## Règle de décision pré-enregistrée (validée par Rayen le 2026-10-14, salon #325)

**Objet.** Décider si la correction post-modèle de la v7 (estimateur géostrophique v2 + ridge
par niveau prédisant le décalage moyen du niveau) est retenue. La règle est écrite et versionnée
**avant toute mesure** ; elle ne sera pas modifiée après coup.

- **C1 — test synoptique** (protocole DSO1, 120 instants, 50 % des capteurs) : pire niveau
  évalué u ET v après correction ≤ 0,5 m/s. IC 95 % bootstrap sur les instants (1000 tirages)
  RAPPORTÉ, borne haute publiée mais non utilisée comme seuil (le test a servi à la conception
  de la correction).
- **C2 — épisodes** (46 instants jamais utilisés pour le choix), Δ = RMSE après − RMSE avant,
  bootstrap APPARIÉ sur les instants (1000 tirages), pour u et v. Sur chaque niveau 650, 800
  et 1000 m : borne haute de l'IC 95 % de Δ ≤ 0 (amélioration significative). Sur chaque
  niveau évalué de 100 à 500 m : borne haute de l'IC 95 % de Δ ≤ +0,02 m/s (pas de
  dégradation notable).
- **C3** — DSO2 (moyenne ≤ 1e-4), DSO3 (rugosité dans [0,9 ; 1,1]), DSO5 (RMSE ≤ 0,7 à 25 %
  des capteurs) et DSO4 (temps total modèle + correction < 1 s par instant, GPU) restent
  conformes.
- **Pour information, hors règle** : cas 100 % capteurs, valeurs absolues avant/après avec IC.
- **Décision** : correction RETENUE ⇔ C1 ET C2 ET C3. Sinon : gel de v7 seule, DSO1 rapporté
  honnêtement (critère OMM). Méthode figée : ridge par niveau 10h, alpha = 0,01 choisi sur la
  validation, paramètres dans couche_limite_v7.json (sha256 imprimé par la cellule 10i). Pas
  de v8 sans accord de Rayen.
- **Mesure** : cellule BLOC 10i du notebook Colab fno3d_v7.ipynb, la règle étant imprimée
  avant tout calcul.

> **Lecture de la règle** (note ajoutée après les commits `0b64978` et `33da965`, sans rien
> y changer). La règle est complétée par l'**Avenant n° 1**, en fin de document : le C2 se
> calcule par bootstrap par blocs, et l'IC de C1 par jours. Deux limites restent à connaître
> (§ 5.3) :
> (1) Les épisodes tombent sur des jours de **test**, donc sur les journées synoptiques qui
> ont servi à concevoir la correction. Il n'y a qu'environ 3 situations indépendantes.
> (2) Le C1 est mesuré sur l'ensemble qui a servi à la conception. Le C2 est la seule partie
> de la règle qui vaut confirmation, et sa puissance est limitée.

---

## 1. Point de départ (chiffres vérifiés dans les sources)

| Source | Ce qu'elle établit |
|---|---|
| `runs/v7/eval_best_dso_v7/test_summary.json` (notebook v7, BLOC 10) | 120 instants synoptiques de test, 50 % des capteurs. DSO1 non conforme uniquement à 650 m (u 0,564), 800 m (u 0,632) et 1000 m (u 0,665, v 0,508). Global : u 0,542, v 0,437. DSO2 2,58e-5, DSO3 0,951, DSO4 91 ms, DSO5 0,646. Épisodes (46 instants) : u entre 0,65 et 1,15 à tous les niveaux. |
| `tukhnanutha.md` § 5 | La v8 (moyenne de 8 passes, poids par niveau, 15 % d'exemples avec tous les capteurs, affinage sélectionné sur le DSO) ne gagne rien : la RMSE vectorielle à 1000 m vaut 0,837 pour la v7 comme pour la v8. Score DSO de validation de la v7 : 1,193 ; en test, le pire niveau donne 0,665 / 0,5 = 1,33. L'écart entre validation et test est donc d'environ +11 %. |
| Diagnostic Colab du 2026-10-12 (transmis, non archivé dans le dépôt) | À 650–1000 m, 80 à 90 % de l'erreur quadratique sur u et v est un décalage uniforme du niveau, différent à chaque instant. Ce décalage est corrélé à l'erreur du géostrophique estimé (u 0,60, v 0,48). La part « plan » ne pèse que 1 à 2 %. Une ridge sur des caractéristiques de surface ne fait pas mieux que le modèle. |
| Cellules 10g et 10h de `fno3d_v7.ipynb` (rapportées par une autre session, **non vérifiées ici**) | Estimateur v2 (Tv = p/(Rd·ρ), z en régresseur, ρ_sl = 1,2) : le résidu du plan passe de 5,34 à 0,10 Pa. Ridge par niveau, alpha = 0,01 choisi sur la validation. Sur le test à 50 % : pire niveau u 0,419 / v 0,382, DSO5 de 0,646 à 0,508. **Réserves** : 4 variantes ont été regardées sur le test ; le DSO4 avec la correction, les épisodes et le cas 100 % n'ont pas été mesurés. |

**Ce que montre le code du générateur** (`scripts/generate_aneo_3d.py`, `compute_fields`) :

- La pression au niveau de la mer est un **plan exact**, sans bruit :
  `p_sl = p0 + 1,2·f·(vg·x − ug·y)`. Le vent géostrophique est donc **uniforme** sur le
  domaine. Le vent thermique l'est aussi, car le gradient de température est constant sur la
  journée.
- La pression au capteur suit la formule hypsométrique, avec la température **virtuelle
  moyenne de la colonne** : `p = p_sl·exp(−g·z / (Rd·½(T_sl + T)·(1 + 0,608 q)))`. Ici `T`
  inclut le terme de stabilité de la couche limite, jusqu'à +2,4 K la nuit à 25 m et
  −0,85 K le jour.
- L'estimateur du pipeline (BLOC 6, `features.geostrophic_from_surface`) ramène la pression
  au niveau de la mer avec `T + ½·0,0065·z`. Il ignore l'humidité (≈ +1,8 K pour q ≈ 0,01)
  et la stabilité, et divise par la densité moyenne **des capteurs** au lieu de 1,2.
- Conséquence : une erreur de température devient une erreur sur `p_sl` proportionnelle à
  l'altitude du capteur. Le relief synthétique monte vers l'ouest, donc cette erreur se
  projette sur ∂p/∂x, c'est-à-dire sur `vg`. Le biais sur G varie d'un instant à l'autre
  (humidité, stabilité), et le modèle ne peut pas l'apprendre comme une constante.

## 2. Hypothèse unique, et pourquoi pas les autres

**Hypothèse.** Le décalage de niveau à 650–1000 m vient d'abord d'une erreur d'**estimation**
du vent géostrophique, et non d'un manque d'information dans les mesures. Un estimateur
cohérent avec la réduction hypsométrique (v2), associé à une correction du décalage niveau
par niveau, ramène le DSO1 sous 0,5 m/s **sans réentraîner**. Il reste à le **confirmer**
sans contamination, selon la règle ci-dessus.

Rayen a formulé le lot ainsi : « au plus 2 jours de GPU : raffinement par niveau OU ERA5 en
entrée ». Le tableau confronte ces deux options et les autres pistes.

| Piste | Verdict | Raison |
|---|---|---|
| **Raffinement par niveau, côté entraînement** (poids par niveau, sélection sur le DSO, affinage) | écartée (déjà testée) | La v8 l'a fait : gain ≤ 0,016 m/s sur u, perte jusqu'à 0,023 m/s sur v. Optimiser l'entraînement n'apporte pas l'information qui manque. |
| **Raffinement par niveau, après le modèle** (ridge qui prédit le décalage de chaque niveau) | **retenue**, avec l'estimateur v2 | C'est la forme la moins chère de l'option 1 de Rayen. Avec l'estimateur v1, une ridge ne battait pas le modèle (diagnostic du 12/10). Avec la v2, la variable qui porte l'information devient exacte (§ 3). Pas de GPU pour l'ajuster, quelques millisecondes à l'inférence, et la v7 reste intacte. |
| **ERA5 en entrée** | écartée : sans objet ici | Le monde synthétique **n'est pas piloté par la météo réelle** (`scripts/compare_synth_era5.py` : « comparer instant par instant n'a pas de sens »). Les dates 2025 ne sont que des étiquettes, et l'ancrage géographique (36,0 N ; 9,7 E) est **supposé**. ERA5 à l'instant t n'a donc aucun lien avec le G synthétique au même instant : en entrée, ce serait du bruit. Remplacer ERA5 par une « haute atmosphère » synthétique reviendrait à donner au modèle le G vrai du générateur, c'est-à-dire la cible. ERA5 a sa place dans un projet sur données **réelles**, avec un autre jeu et un autre protocole, pas dans le DSO1 synthétique. |
| Entrée géostrophique variable dans l'espace | écartée | Le G du générateur est uniforme **par construction** (plan exact) et la part « plan » de l'erreur ne pèse que 1 à 2 % : une estimation variable ne ferait que modéliser du bruit. Elle exigerait en plus un réentraînement complet. |
| Réentraînement avec l'estimateur v2 en entrée | repli, seulement sur accord de Rayen | Ce serait la confirmation « propre » de l'hypothèse (§ 6), mais elle coûte 6 à 10 h de GPU. Elle n'a d'intérêt que si la correction post-modèle échoue au C2 alors que l'étape 0 reste favorable. |

## 3. Étape 0 — diagnostic CPU go / no-go (exécuté le 2026-10-14)

**Question.** De combien l'estimateur v2 réduit-il l'erreur sur le géostrophique, par
rapport au G **vrai** ? Cette réduction rend-elle le vent moyen des niveaux hauts plus
prévisible ?

**Vérité de référence.** Le dataset ne contient pas `ug, vg` : ni `meta.json` ni les
colonnes des CSV journaliers ne les donnent, et `qc/qc_day_XXX.csv` ne fournit que la norme
`g`. On les recalcule donc en **rejouant les tirages du générateur** : graine
2026·1000 + numéro du jour, mêmes tirages dans le même ordre que `DayState`. Le secteur de
direction s'obtient en rejouant le tirage du relief synthétique puis le mélange des
secteurs.

Le rejeu est **vérifié** sur chaque instant : |G| retrouve la colonne `g` du QC à
4·10⁻¹⁵ près, et le script s'arrête dès qu'un écart dépasse 10⁻⁶. L'hypothèse sur le
secteur est vérifiée sur les 100 jours : le vent à 100 m est tourné de −20° à −38° par
rapport à G, ce qui correspond au `veer` de 20–40°. L'autre hypothèse donne des écarts
répartis au hasard.

**Données.** 76 jours d'entraînement et de validation sans épisode rapide, soit
**608 instants synoptiques**. Aucun jour de test : le script les refuse. La surface est
poolée 2×2 comme dans le pipeline. Tout est lu sans écriture : 3 jours viennent du cache
local, les autres sont lus en flux depuis les CSV. Durée : 37 min de CPU.

```
python scripts/diag_geostrophique.py --data-dir <aneo_dataset_3d_v2> --cache-dir <cache> \
    --days trainval --out diag.json
```

**Résultat 1 — erreur sur le G estimé (m/s), 50 % des capteurs.** Les résultats à 100 % et
à 25 % des capteurs sont identiques à ±0,005 près.

| Estimateur | RMSE ug | RMSE vg | RMSE vect. | P95 vect. | réduction |
|---|---:|---:|---:|---:|---:|
| actuel (v7, BLOC 6) | 1,642 | 1,579 | **2,278** | 3,836 | — |
| ρ = 1,2 seul | 1,561 | 1,491 | 2,159 | 3,400 | 5 % |
| Tv seul | 0,441 | 0,438 | 0,622 | 1,228 | 73 % |
| z en régresseur seul | 0,604 | 0,486 | 0,775 | 1,510 | 66 % |
| Tv + z | 0,283 | 0,276 | 0,395 | 0,904 | 83 % |
| Tv + ρ = 1,2 | 0,334 | 0,336 | 0,474 | 1,030 | 79 % |
| **v2 : Tv + z + ρ = 1,2** | **0,024** | **0,015** | **0,028** | **0,051** | **98,8 %** |

- Les trois corrections ne marchent **qu'ensemble** : sans ρ = 1,2, l'erreur reste à
  0,40 m/s ; sans z, à 0,47 m/s.
- L'erreur de la v7 (2,3 m/s RMS, 3,8 m/s au P95) vaut **4,6 fois** le seuil du DSO1.
- Le vent thermique garde une erreur de 3,8·10⁻⁴ s⁻¹, pour une norme RMS de 2,9·10⁻³ s⁻¹.
  Ajouter z en régresseur n'y change rien (vérifié). Ce plancher de bruit vaut 0,1 à
  0,3 m/s à 1000 m quand la couche limite est basse, et il subsiste après correction.

**Résultat 2 — prévisibilité du vent moyen des niveaux hauts.** On régresse le vent moyen
vrai de chaque niveau sur les nombres synoptiques que reçoit le modèle : G, vent thermique,
heure, et les produits G × heure (11 termes). La régression est validée en laissant de côté
un jour entier à chaque fois. Ce n'est pas le modèle, mais un plafond linéaire de ce que
l'information synoptique permet de prévoir.

| RMSE (m/s) | 650 m u / v | 800 m u / v | 1000 m u / v |
|---|---|---|---|
| A — caractéristiques v7 | 0,809 / 0,669 | 0,851 / 0,763 | 0,870 / 0,767 |
| B — G v2 | **0,237 / 0,285** | **0,295 / 0,348** | **0,328 / 0,376** |
| C — oracle (G et vent thermique vrais) | 0,235 / 0,286 | 0,292 / 0,347 | 0,317 / 0,371 |

Avec l'estimateur v2, l'information synoptique équivaut presque à celle de l'oracle
(B ≈ C). L'erreur de prévision à 1000 m baisse de 62 % sur u et de 51 % sur v.

**Résultat 3 — transposition au réel (bruit des baromètres).** On ajoute un bruit gaussien
indépendant à la pression de chaque capteur, puis on applique l'estimateur v2.
Échantillon : 56 instants train/val (jours 2, 3, 14, 20, 47, 74, 94), option
`--bruit-pa 0,10,30,50`. RMSE vectorielle de G, en m/s :

| Réseau | capteurs | 0 Pa | 10 Pa | 30 Pa | 50 Pa |
|---|---:|---:|---:|---:|---:|
| 50 % de la couche synthétique | 62 478 | 0,028 | 0,068 | 0,188 | 0,312 |
| un capteur tous les 500 m | 5 000 | 0,070 | 0,223 | 0,641 | 1,065 |
| un capteur tous les 1 km | 1 250 | 0,142 | 0,507 | **1,491** | **2,486** |

Le signal à mesurer est faible : pour G = 10 m/s, l'écart de pression vaut
1,2·f·G·L ≈ 26 Pa sur les 25 km du petit côté du domaine. Une erreur de 1 m sur l'altitude
d'une station équivaut à environ 12 Pa. Avec la densité d'un vrai réseau ANEO (500 m à
1 km) et des baromètres à ±30–50 Pa, l'estimateur v2 perd l'essentiel de son avance : 1,5 à
2,5 m/s d'erreur, contre 2,3 m/s pour l'estimateur v7 sur le synthétique exact. **Le gain
mesuré sur le synthétique ne se transpose pas tel quel au réel.** C'est la réserve
« baromètres réels », à rapporter **telle quelle** avec le verdict.

**Seuil go / no-go, et raisonnement chiffré.**

- *Borne basse (attribution linéaire).* Notons σ_O² la variance du décalage (80 à 90 % de
  la MSE) et ρ sa corrélation avec l'erreur sur G (u 0,60, v 0,48). Retirer une fraction r
  de l'erreur sur G retire au plus ρ²·σ_O²·(1 − (1 − r)²) de MSE, si le reste du décalage
  est indépendant de G. Avec r = 0,988 :

  | niveau | v7 | estimation |
  |---|---:|---:|
  | 1000 m, u | 0,665 | **0,547–0,561** |
  | 800 m, u | 0,632 | 0,520–0,533 |
  | 650 m, u | 0,564 | 0,464–0,476 |
  | 1000 m, v | 0,508 | 0,452–0,459 |

- *Estimation proportionnelle.* On met à l'échelle le décalage de la v7 (≈ 0,61 m/s RMS à
  1000 m sur u) par le rapport B / A du résultat 2 (0,328 / 0,870) : on obtient ≈ 0,23 m/s.
  En ajoutant la part de l'erreur qui n'est pas un décalage (0,21 à 0,30 m/s), la RMSE
  totale vaut **≈ 0,31–0,38 m/s**.
- Le résultat annoncé pour la correction (pire niveau u 0,419) tombe **entre** ces deux
  bornes. Il est donc physiquement plausible, et pas seulement un artefact de la
  contamination du test. Il dépasse toutefois nettement la borne basse ; or un résultat
  « trop beau » est aussi la signature d'un choix fait sur le test. La confirmation par le
  C2 est indispensable.
- **Seuil**, fixé après lecture des 24 premiers instants et avant le passage complet : GO
  si l'estimateur v2 réduit l'erreur vectorielle sur G d'au moins **50 %** à 25, 50 et
  100 % des capteurs. L'attribution linéaire prévoit alors un gain d'au moins **0,08 m/s à
  1000 m sur u** (0,665 → 0,58), soit cinq fois l'effet total de la v8. Il faut aussi que
  la RMSE validée du vent moyen à 800 et 1000 m baisse d'au moins **0,2 m/s** (B contre A).
- **Mesuré : réduction de 98,8 %, baisse de 0,39 à 0,56 m/s. Verdict : GO** pour la mesure
  BLOC 10i. Ce GO vaut pour le synthétique ; il ne dit rien du réel (résultat 3).

## 4. Méthode figée (ce qui est mesuré, et rien d'autre)

- **Modèle** : inchangé.
  - `best_dso_v7.pt`, époque 140, sha256
    `b2529442ee6fcdb18d3f35fe176663022604395743c161a25541ccd4e365e7de` ;
  - `norm_stats.json` de la v7, sha256
    `7bae6748f270fa5e3f08a69a7f8e919f01d641d29b125981168cc15a03ee25f7`.
- **Correction** : estimateur v2 (10g) et ridge par niveau (10h), alpha = 0,01 choisi sur
  la validation. Les paramètres sont dans les fichiers Drive `v7/correction_moyenne_v7.json`,
  `v7/geostrophique_v2_v7.json` et `v7/couche_limite_v7.json`. La cellule 10i imprime le
  **sha256 des trois** avant tout calcul ; ce sont ces valeurs qui entrent dans le gel (§ 8).
- **Inchangé par rapport à la v7** : poids, normalisation, architecture, masques (graine
  d'évaluation de la v7), niveaux évalués (`min_level_cover` = 0,05, soit 100 à 1000 m),
  définition des métriques (sommes de carrés cumulées sur les instants).
- **Une seule variante** passe en 10i : celle choisie sur la validation. Les trois autres
  variantes regardées sur le test ne sont ni mesurées ni présentées comme candidates.
- **À vérifier dans 10h avant d'exécuter 10i** : si les coefficients de la ridge ont été
  ajustés sur la validation, il faut l'écrire, et l'alpha doit alors avoir été choisi par
  validation croisée par jour. Si ce n'est pas le cas, le dire dans le rapport.

## 5. Mesure décisive — cellule BLOC 10i (GPU, exécutée par Ines)

### 5.1 Contenu de la cellule (dans cet ordre, sans retouche entre deux exécutions)

1. **Imprimer la règle** : le texte de la section pré-enregistrée, la date, le commit
   `0b64978`, puis les sha256 des poids, de `norm_stats.json` et des trois JSON de
   correction.
2. **Charger la correction figée.** La cellule ne fait aucun ajustement et aucun choix.
3. **Test synoptique** : 120 instants, 50 % des capteurs, avec les mêmes masques que la v7.
   - Mesures avant et après correction : RMSE et R² de u, v, w par niveau, et RMSE globale.
   - Verdict **C1**.
   - Garder, **pour chaque instant et chaque niveau**, la somme des carrés et le nombre de
     nœuds. Le bootstrap pourra ainsi recalculer des RMSE cumulées, et non des moyennes de
     RMSE.
4. **Épisodes** : 46 instants, 50 % des capteurs, mêmes mesures avant et après, puis
   verdict **C2** (Δ apparié).
5. **C3**, sur les champs **corrigés** :
   - DSO2 : résidu de masse moyen.
   - DSO3 : rugosité moyenne dans [0,9 ; 1,1]. Un décalage uniforme ne change pas les
     gradients, mais c'est à confirmer par la mesure, pas à supposer.
   - DSO5 : une passe à 25 % des capteurs, avec correction.
   - DSO4 : chaîne complète (estimateur v2, caractéristiques de couche limite, modèle,
     ridge, ajout du décalage), avec `torch.cuda.synchronize()`. On fait 10 instants de
     chauffe puis 50 mesurés, et on rapporte la médiane et le P95. Sur A100 seulement, pour
     rester comparable aux 91–104 ms de la v7.
6. **Pour information** : 100 % des capteurs avant et après, avec IC ; robustesse à 10, 25,
   50 et 100 % des capteurs.
7. **Bootstrap** (graine fixe, 1000 tirages), selon l'**Avenant n° 1** :
   - C1 : IC 95 % du pire niveau, pour u et pour v, en rééchantillonnant le test
     synoptique par **jours entiers** (15 jours) ;
   - C2 : bootstrap **apparié par blocs** de 6 instants consécutifs, tirés à l'intérieur de
     chaque épisode (tailles 18 / 18 / 10 conservées) ; IC 95 % de Δ par niveau et par
     composante ;
   - **pour information, hors règle** : les IC par instants, et la valeur de Δ **par
     épisode** (3 valeurs). L'avenant le dit : aucun bootstrap ne mesure la variabilité
     d'un épisode à l'autre.
8. **Verdict et archivage.** Imprimer le verdict C1 / C2 / C3 et la décision. Archiver sur
   le Drive, dans `aneo_fno3d_runs/v7/eval_m1_bloc10i/` : `summary.json`,
   `par_niveau_{synoptique,episodes}_{avant,apres}.csv`, `bootstrap.csv`, `dso4.json`,
   `sha256.txt`, et une copie du texte de la règle.

### 5.2 Règle d'exécution

- La cellule ne s'exécute **qu'une fois**.
- Une erreur de code (exception, fichier introuvable) peut être corrigée. La correction est
  alors commitée et décrite, puis **toute** la cellule est relancée.
- Il est interdit de corriger quoi que ce soit qui changerait un **chiffre** déjà affiché.
- Tout ce qui s'affiche est rapporté, y compris un résultat défavorable.

### 5.3 Limites connues de la mesure (à rapporter avec le verdict)

- **Les épisodes ne sont pas des données indépendantes.** Les 46 instants forment trois
  blocs :
  - rafales : jour 4, 18 instants à 1 min d'écart ;
  - lever du soleil : jour 97, 18 instants à 1 min d'écart ;
  - rafales 10 s : jour 11, 10 instants à 10 s d'écart.

  Ce sont des jours de **test**, dont les instants synoptiques ont servi à choisir la
  correction : ces instants n'ont jamais été vus, mais leurs situations météo, si. Le QC
  donne une corrélation temporelle ≥ 0,95 à 1 min. L'Avenant n° 1 traite
  l'autocorrélation **à l'intérieur** d'un épisode (bootstrap par blocs), mais pas la
  dépendance entre épisodes et test synoptique.
- **La physique des épisodes diffère.** Dans les rafales, l'amplitude des perturbations
  rotationnelles est doublée (0,25 au lieu de 0,12) : l'erreur y est moins un décalage de
  niveau. C'est pourquoi le C2 demande une amélioration à 650–1000 m, et seulement une
  non-dégradation en dessous, plutôt qu'un DSO1 absolu.
- **Les instants du test synoptique sont corrélés** (8 instants par jour) : c'est pourquoi
  l'IC de C1 se calcule par jours (Avenant n° 1).
- **Confirmation complémentaire proposée** (hors règle, à accepter par Rayen ; elle ne
  peut ni ajouter ni retirer de condition) : un **jeu vierge**, produit par le même
  générateur avec une autre graine et sur la même grille :
  `generate_aneo_3d.py --input-grid <grille de surface> --seed 2027 --days 3,13,23,...,93`.
  - Contenu : 10 jours, soit 80 instants synoptiques.
  - Coût : environ 30 min de CPU, ~3 Go de disque, et ~5 min de GPU dans la même session.
  - Le générateur du dépôt est bien celui qui a produit le dataset : le rejeu du § 3
    retrouve le QC à 10⁻¹⁵ près.
  - C'est le seul ensemble que personne n'aurait vu.

## 6. Repli : réentraînement (uniquement avec l'accord explicite de Rayen)

Le réentraînement n'est lancé **que si** Rayen le décide après le verdict, dans la limite
de **2 jours de GPU**. Plan, du moins cher au plus cher :

| Étape | Contenu | Durée A100 |
|---|---|---|
| préparation | montage du Drive, archive du code, `pip install -e .`, reconstruction du cache local (19 min mesurées le 12/10) | ~30 min par session |
| G1 | affinage depuis la v7 avec l'estimateur v2 en entrée (`train.init_from`, 30 époques, lr 1e-4, `select_by: dso`, la v7 étant le premier candidat) | 30 × 147 s ≈ 1 h 15 |
| G2 | réentraînement complet, recette v7 à l'identique sauf l'estimateur (140 époques, OneCycle, lr 1e-3) | 140 × 147 s ≈ 5 h 45 |
| évaluation | protocole de la v7, avec DSO3 dans [0,9 ; 1,1] | ~20 min par candidat |

- **L'estimateur v2 n'impose pas de reconstruire le cache.** Les caractéristiques sont
  recalculées à la volée depuis S (`ANEO3D.__getitem__`), et `norm_stats.json` ne dépend
  que de S et Y. On recopie celui de la v7 et on vérifie son sha256.
- **Arrêt anticipé de G2** :
  - à l'époque 100, arrêt si la perte de validation dépasse 1,05 × 0,0062 (la valeur de la
    v7 au même point) ;
  - à l'époque 120, arrêt si le score DSO de validation dépasse 1,00. Compte tenu de
    l'écart de +11 % entre validation et test, il faut ≤ 0,90 en validation pour espérer
    ≤ 1,0 en test.
- **Checkpoints** : mécanisme existant. `last.pt` (~7 Go) est écrit en local à chaque époque
  et copié sur le Drive toutes les 5 époques ; prévoir au moins 20 Go libres. À la reprise,
  ne pas changer `train.epochs`, à cause du calendrier OneCycle.
- **Code à modifier (lot séparé)** :
  - dans `config.py`, une option `geo_estimator: plan_v7 | tv_z`, avec `plan_v7` par défaut
    pour que la v7 reste reproductible ;
  - dans `features.geostrophic_from_surface`, la branche correspondante ;
  - la valeur est écrite dans le `config` du checkpoint et relue par `WindFieldModel` : le
    checkpoint fait foi, comme pour l'architecture ;
  - le BLOC 6 de `fno3d_v7_1.ipynb` est mis à jour à l'identique.

Au pire, le repli prend 2 sessions et environ 10 h de GPU.

## 7. Portage de la correction dans le paquet (lot séparé, une fois les fichiers disponibles)

Ce lot ne commence qu'après le verdict, et seulement si la correction est retenue.

- `src/pcfno3d/features.py` : `geostrophic_from_surface(..., method="v2")` avec
  Tv = p/(Rd·ρ), un régresseur z − z̄ (retiré si l'écart-type des altitudes visibles est
  < 1 m) et ρ = 1,2. `plan_v7` reste le défaut.
- `src/pcfno3d/correction.py` (nouveau) : lit `correction_moyenne_v7.json`,
  `geostrophique_v2_v7.json` et `couche_limite_v7.json`, vérifie leurs sha256, puis ajoute
  le décalage prédit par niveau à u et v (pas à w).
- `config.py` : `inference.post_correction: <chemin> | null`, null par défaut, pour que la
  v7 seule reste reproductible bit à bit. `WindFieldModel.predict`, l'`Evaluator` et la
  moyenne de passes appliquent la correction au même endroit.
- **Tests CPU** (`tests/test_correction.py`) :
  1. Sur un champ construit avec les formules du générateur (relief, humidité, stabilité),
     la v2 retrouve (ug, vg) à moins de 0,05 m/s, alors que `plan_v7` s'en écarte de plus
     de 0,5 m/s.
  2. Avec `post_correction: null`, les sorties sont identiques à celles de la v7.
  3. Sur terrain plat : pas de NaN, et le résultat égale celui de l'ajustement à 3
     régresseurs.
  4. Avec moins de 100 capteurs, le résultat reste (0, 0) comme aujourd'hui.
  5. Un sha256 qui ne correspond pas lève une erreur explicite.
  6. La correction ne touche ni w ni les nœuds sous le relief.
  7. Sur le mini-dataset, les cumuls de l'évaluation avec correction égalent un calcul
     direct.
  8. `scripts/smoke.ps1` passe avec la correction activée.

## 8. Gel du modèle après M1

| Cas | Tag git (annoté) | Modèle gelé |
|---|---|---|
| correction retenue | `modele-fige-v7-correction` | v7 + correction post-modèle (option du paquet, testée) |
| correction rejetée | `modele-fige-v7` | v7 seule ; DSO1 rapporté non conforme à 650–1000 m, critère OMM conforme |

- **Ce que le tag désigne** : le commit qui contient
  - le code qui sert et évalue le modèle gelé (et, si la correction est retenue, le
    portage du § 7) ;
  - les configurations ;
  - ce document **complété par les résultats de 10i** ;
  - le notebook, dans l'état exact où il a été exécuté ;
  - le dossier `docs/resultats/modele_fige/` : `summary.json` de 10i, le tableau final
    DSO1–DSO5 sur 120 instants, et `SHA256SUMS`.
- **Message du tag** : le modèle retenu et la décision (C1 / C2 / C3), l'époque 140, le
  score de validation 1,19523, et les sha256 :
  - `best_dso_v7.pt` : `b2529442ee6fcdb18d3f35fe176663022604395743c161a25541ccd4e365e7de` ;
  - `norm_stats.json` : `7bae6748f270fa5e3f08a69a7f8e919f01d641d29b125981168cc15a03ee25f7` ;
  - `test_summary.json` du notebook v7 : `7944ba0aa0311a08cd1e1555c1a225239b3deabc5c9c2b55dcf4406b5da17425` ;
  - si la correction est retenue, ceux des trois JSON de correction, tels qu'imprimés par
    10i.
- **Archives** (les poids, 1,8 Go, ne vont pas dans git) :
  - sur le Drive, `aneo_fno3d_runs/fige/modele_fige_2026-10.zip`, qui contient les poids,
    `norm_stats.json`, les JSON de correction, la configuration, `summary.json` et
    `SHA256SUMS` ;
  - en local, `AppData/Local/pcfno3d/runs/v7/` (déjà installé, sha256 vérifié).
- **Tableau final.** Si la v7 seule est gelée, son DSO3 (0,951) est relu dans [0,9 ; 1,1] et
  reste conforme. Aucun nouveau calcul n'est nécessaire ; 10i en donne de toute façon la
  mesure.
- Après le gel, le modèle ne change plus dans le cadre du PFE.

## 9. Créneau GPU à proposer à Rayen

- **État au 2026-10-14 au soir** : Rayen a lancé une session GPU et la mesure est en cours.
  Les points ci-dessous décrivent ce qu'elle doit respecter, et la session de réserve en
  cas de coupure ou de relance.
- **Une session de 2 h, sur A100 obligatoirement.** Le DSO4 doit rester comparable aux
  91–104 ms de la v7 ; sur un L4, ce chiffre serait faussé.
  - Proposition : **jeudi 15/10/2026, 10 h–12 h** ; réserve le vendredi 16/10, même
    horaire.
  - Durée attendue, environ 1 h 15 plus une marge : 10 min d'installation, ~20 min pour le
    cache s'il n'est pas sur le Drive, 20 à 30 min pour 10i, et ~5 min pour le jeu vierge
    s'il est accepté.
- **Avant la session** :
  - Ines vérifie dans 10h comment l'alpha a été choisi (§ 4) ;
  - les trois JSON et la cellule 10i sont déposés sur le Drive ;
  - Rayen accepte ou refuse le jeu vierge (§ 5.3) ; s'il l'accepte, Ines le génère sur CPU
    (~30 min), puis le mesure dans une session courte (~5 min de GPU, plus le cache).
- **Ce qu'un humain doit cliquer** :
  1. Rayen ouvre la session Colab Pro, puis *Exécution → Modifier le type d'exécution →
     GPU A100*.
  2. Exécuter la cellule `drive.mount` et accepter la fenêtre d'autorisation Google Drive,
     avec le compte qui contient `aneo_fno3d_runs`.
  3. Ines exécute les cellules de préparation, puis **10i une seule fois**, et garde
     l'onglet ouvert jusqu'au message d'archivage.
  4. Vérifier que `eval_m1_bloc10i/` est bien sur le Drive.
- **Secrets : aucun.** La mesure n'utilise ni clé d'API, ni `TUNNEL_CRED_CONTENTS`, ni
  jeton. Ne pas lancer le notebook atom sur le même environnement d'exécution pendant la
  mesure : il fausserait le DSO4.
- **Repli** (§ 6), seulement sur accord de Rayen : deux sessions la semaine suivante, de
  3 h puis de 8 h.

## 10. Risques et parades

| Risque | Parade |
|---|---|
| Le résultat annoncé sur le test (u 0,419) est gonflé, car la variante a été choisie parmi 4 regardées sur le test | Variante figée sur la validation (§ 4) ; le C1 ne décide pas seul ; confirmation par le C2 sur les épisodes ; jeu vierge proposé (§ 5.3) |
| IC bootstrap trop étroits (instants corrélés, ~3 épisodes indépendants) | Avenant n° 1 : blocs de 6 instants pour C2, jours pour C1. Δ par épisode rapporté pour information. Puissance de C2 limitée : à écrire avec le verdict |
| Le C2 échoue alors que la correction aide aux niveaux hauts (dans les rafales, l'erreur est peu un décalage) | La règle tranche : on garde la v7 seule. Le résultat est rapporté honnêtement, avec le détail par épisode |
| La chaîne complète dépasse le DSO4 | Coût attendu de quelques ms (un ajustement à 4 colonnes et 12 produits scalaires). Mesure sur A100 obligatoire ; à ≥ 1 s, le C3 échoue |
| La correction modifie le DSO2 (ρ varie à altitude fixe) | Mesuré sur les champs corrigés (C3) ; la v7 a une marge d'un facteur 4 (2,58e-5 pour un seuil de 1e-4) |
| Les JSON de correction ne sont pas ceux qui ont donné 0,419 | sha256 imprimés au début de 10i et repris dans le tag |
| Transposition au réel : baromètres à ±10–50 Pa, altitudes des stations à ±1 m | Réserve écrite telle quelle : sur un réseau de 1 km, l'erreur sur G est de 1,5 à 2,5 m/s (§ 3). La correction est un résultat **synthétique** : sur données réelles, il faudra des pressions recalées (QFE contrôlé, altitudes GNSS) et un nouvel ajustement de la ridge |
| La ridge a appris la physique propre au générateur (h_bl, veer, profils) | La correction ne vaut que pour ce générateur ; l'écrire dans le rapport, et dans `docs/INTEGRATION.md` si elle est retenue |
| La session est coupée pendant 10i | 10i ne fait que de l'inférence (~20 min) : on la relance entièrement, sans toucher au code |
| Dépassement du budget | 10i tient dans une session de 2 h ; le repli ne démarre que sur décision de Rayen, dans la limite de 2 jours |

## Annexe — reproduire le diagnostic CPU

- Script : `scripts/diag_geostrophique.py`. Il ne modifie aucune donnée ; il écrit seulement
  ses JSON et CSV de sortie, plus une copie temporaire de la géométrie.
  - `--days trainval` : 76 jours, 608 instants, 37 min.
  - `--days 2,3,14,20,47,74,94 --keeps 0.5 --bruit-pa 0,10,30,50` : sensibilité au bruit,
    ~3 min.
- Contrôle intégré : une assertion vérifie, sur chaque instant, que la variante « actuel
  (v7) » égale `pcfno3d.features.geostrophic_from_surface` à 10⁻⁶ près.
- Données lues : `meta.json`, `split_days.json`, `qc/qc_day_XXX.csv`, le cache local (jours
  2, 3 et 14) et les CSV journaliers des autres jours train/val. Aucun jour de test.

---

## Avenant n° 1 (2026-10-14, décision de Rayen, salon #332) — avant toute mesure

Objet : le mode de calcul de l'intervalle de confiance de **C2** (épisodes). Rien d'autre ne change :
seuils, niveaux, C1, C3 et la règle de décision restent ceux de la section précédente.

- **Constat** : les 46 instants d'épisodes ne sont pas indépendants. Ils tombent sur les jours de test
  4, 97 et 11, soit environ **3 situations** : 18 instants « rafales » à 1 min d'écart,
  18 « lever du soleil » et 10 « rafales 10 s ». Un bootstrap sur les instants sous-estime l'IC.
- **C2 utilise un bootstrap PAR BLOCS** : blocs de **6 instants consécutifs**, rééchantillonnés
  **à l'intérieur de chaque épisode**, en conservant la taille de chaque épisode (18 / 18 / 10).
  Le bootstrap reste apparié (même tirage avant / après), avec 1000 tirages. Pour C1, l'IC
  rapporté rééchantillonne le test synoptique par jours entiers.
- **Pour information, hors règle** : l'IC par instants est calculé et publié à côté.
- **Réserve, à reproduire telle quelle dans le rapport** : ce bootstrap stratifié est
  **conditionnel aux 3 situations**. Il corrige l'autocorrélation à l'intérieur d'un épisode,
  mais ne mesure pas la variabilité d'un épisode à l'autre. Avec 3 épisodes, aucune méthode ne
  le peut, et C2 a donc une **puissance limitée**.
- **Mise en œuvre** : cellule BLOC 10i, avec `BOOTSTRAP = "blocs"` et `BLOC_LEN = 6`, imprimés
  avec la règle avant tout calcul.

## Avenant n° 2 (2026-10-14, accepté par Rayen, salon #341) — confirmation sur un jeu vierge, HORS RÈGLE

Cette mesure **n'entre pas** dans la décision C1/C2/C3. Elle est rapportée pour information, à côté
de cette décision, quelle qu'en soit l'issue. Le protocole est écrit avant la génération du jeu.

- **Jeu** : `scripts/generate_aneo_3d.py` (générateur Tukhnanutha v2, inchangé), `--seed 2027` (le jeu
  d'origine utilise 2026), même grille statique (`--input-grid` = `aneo_grid_points.csv` du jeu
  v2 : même relief, mêmes capteurs), **15 jours** (`--n-days 15`, soit 120 instants synoptiques, comme le
  test). Génération sur le CPU d'ines-pc, sortie hors dépôt. Les sha256 de `meta.json` et du fichier de
  grille sont consignés. Les 15 jours sont **tous** traités comme du test : rien n'y est ajusté, ni le
  modèle, ni les statistiques de normalisation (celles de v7, figées), ni la correction (le JSON figé).
- **Mesure**, dans la prochaine session GPU prévue (M2 ou M6), environ 5 min, en une seule exécution :
  protocole DSO1 identique (50 % des capteurs, graine 2000, mêmes niveaux évalués, `min_level_cover` 0,05),
  sur les instants synoptiques. RMSE et R² par niveau pour u et v, avec **v7 seul** et **v7 + correction**,
  plus DSO2, DSO3 et DSO5. IC 95 % par bootstrap sur les **jours entiers** (15 grappes).
- **Rapporté** : le tableau par niveau, avant et après correction, avec les IC ; le pire niveau ; la
  différence appariée après − avant à 650–1000 m. Les épisodes du jeu vierge sont rapportés à part s'il y
  en a. Aucun seuil n'est appliqué, et aucune décision n'en dépend. Le résultat est publié tel quel,
  même s'il contredit 10i.
- **Réserve** : un jeu vierge du même générateur reste du synthétique. Il teste la généralisation à
  de nouvelles situations, pas le transfert au réel (voir la section sur les baromètres réels).

*Note technique de l'avenant n° 2, ajoutée avant la génération* : le jeu est produit par
`scripts/generate_jeu_vierge.py`, qui exécute le générateur **sans le modifier** et retire seulement les
instants d'épisodes du calendrier. On obtient 120 instants synoptiques, environ 4,4 Go. Avec les épisodes,
il y aurait 320 instants et environ 13 Go, plus que les 12 Go libres sur ines-pc. Le jeu ne contient
donc pas d'épisodes.

## Avenant n° 3 (2026-10-16) — lancement de la session 10i, sans changement de règle

Cet avenant décrit seulement comment la session est lancée. La règle de décision (C1, C2, C3), les
avenants n° 1 et n° 2 et le code de la cellule 10i ne changent pas.

- **Notebook** : `notebooks/current/m1_bloc10i_colab.ipynb`, autonome, lancé par « Tout exécuter ».
  Il est généré par `scripts/make_m1_colab_nb.py` à partir de `fno3d_v7.ipynb` (BLOCS 0 à 8 et 10)
  et de `cellule_10i.py` (fichier entier). Le code de ces cellules est **copié sans modification**,
  sorties vidées (`--check` le vérifie, ainsi que le sha256 du code 10i,
  `a1fcc878ed3c6c6047538e5b48df25c6633c972a413d6bd42f415926599fbd6f`).
- **Substitution du BLOC 9** : il n'est pas exécuté. S'il ne trouve pas `last.pt` sur le Drive, il
  réentraîne 140 époques et réécrit `best.pt`. À sa place, la cellule M1-b reprend à l'identique
  les deux lignes du BLOC 9 dont le BLOC 10 et 10i ont besoin (`amp_dtype`, `use_amp`). Le BLOC 10
  charge lui-même le checkpoint, comme dans la v7 : `best_dso_v7.pt`, dont le sha256 est vérifié.
- **BLOCS 10c à 10h non exécutés.** Ils ne figurent pas dans `fno3d_v7.ipynb` du dépôt. Les
  BLOCS 10f à 10h réajusteraient la correction et réécriraient les trois JSON figés. 10i n'a
  besoin que des globales des BLOCS 0 à 10 et de `couche_limite_v7.json` (§ 4 : aucun ajustement).
- **Écritures** :
  - Le BLOC 10 écrit dans `RUN_DIR/eval_<modèle>/`, c'est-à-dire sur les résultats figés de la v7.
    Pendant le BLOC 10 seulement, `RUN_DIR` pointe donc vers un dossier local qui contient une
    copie vérifiée de `best_dso_v7.pt` (cellule M1-c).
  - Avant 10i, `RUN_DIR` revient au Drive et `EVAL_DIR` pointe vers `eval_m1_bloc10i/` (cellule
    M1-d) : 10i y écrit `decision_10i.json` et `decision_10i_par_niveau.csv`.
  - M1-a contrôle les fichiers figés et imprime la règle avant tout calcul. Elle refuse de démarrer
    si une mesure 10i complète existe déjà (§ 5.2).
  - M1-e ajoute `sha256.txt`, `regle.txt` et les résultats rejoués du BLOC 10 (`bloc10_rejoue/`).
    Elle vérifie aussi le sha256 du code 10i réellement exécuté, et que les fichiers figés n'ont
    pas changé.
- **Jeu vierge (avenant n° 2)** : la dernière cellule (M1-f) ne s'exécute que si le jeu complet est
  sur le Drive, dans `aneo_dataset_3d_seed2027/` (15 jours). Ses résultats vont dans
  `eval_m1_bloc10i/jeu_vierge_seed2027/`. Sinon, elle affiche un message et ne fait rien.

---

## Résultat de la mesure décisive (2026-10-16)

*Section ajoutée après l'exécution de 10i. La règle pré-enregistrée et les avenants n° 1 à 3
ci-dessus sont inchangés. Chiffres recopiés des sorties de la session, sans recalcul.*

**Session.**

- Colab, **A100-SXM4-80GB** ; notebook `notebooks/current/m1_bloc10i_colab.ipynb` (avenant n° 3),
  « Tout exécuter ».
- sha256 du code 10i : `a1fcc878ed3c6c6047538e5b48df25c6633c972a413d6bd42f415926599fbd6f`
  (conforme à l'en-tête de `cellule_10i.py`).
- Correction : `couche_limite_v7.json`, sha256 `c50928ae2a93dfb3...` (préfixe imprimé par la
  session), alpha = 0,01, ρ_sl = 1,2.
- Instants : validation 64 ; test synoptique 120 ; épisodes 46 (jour 4 : 18, jour 11 : 10,
  jour 97 : 18) ; 100 % des capteurs 120 ; 25 % des capteurs 40.
- Résultats archivés sur le Drive : `aneo_fno3d_runs/v7/eval_m1_bloc10i/decision_10i.json` et
  `decision_10i_par_niveau.csv`.

### C1 — test synoptique (120 instants, 50 % des capteurs) : CONFORME

RMSE avant → après correction (m/s). Entre crochets : IC 95 % après correction, quand la
session l'a imprimé.

| Niveau | u avant → après | v avant → après |
|---|---|---|
| 100 m | 0,458 → 0,416 [0,261 ; 0,552] | 0,440 → 0,382 |
| 150 m | 0,476 → 0,418 | 0,456 → 0,381 |
| 200 m | 0,462 → 0,393 | 0,436 → 0,368 |
| 300 m | 0,457 → 0,365 | 0,424 → 0,344 |
| 400 m | 0,468 → 0,368 | 0,401 → 0,347 |
| 500 m | 0,481 → 0,380 | 0,394 → 0,351 |
| 650 m | 0,564 → 0,413 [0,305 ; 0,505] | 0,408 → 0,339 |
| 800 m | 0,632 → 0,404 [0,292 ; 0,503] | 0,466 → 0,362 |
| 1000 m | 0,665 → 0,419 [0,295 ; 0,521] | 0,508 → 0,355 |

- **Pire niveau**, IC bootstrap par blocs (jours entiers, avenant n° 1) :
  - u : 0,665 [0,482 ; 0,803] → **0,419** [0,320 ; 0,559] ;
  - v : 0,508 [0,422 ; 0,606] → **0,382** [0,290 ; 0,496].
- Pour information, IC du pire niveau après correction en mode « instants » : u [0,383 ; 0,495],
  v [0,346 ; 0,430].
- R² minimal après correction : 0,9933.
- Les deux pires niveaux sont ≤ 0,5 m/s et le R² ≥ 0,95 partout : **C1 conforme**. La borne haute
  de l'IC dépasse 0,5 pour u ; la règle la rapporte sans l'utiliser comme seuil, et le test a
  servi à la conception de la correction (limite de la règle, § 5.3).

### C2 — épisodes (46 instants jamais utilisés) : NON CONFORME

IC 95 % de Δ = RMSE après − RMSE avant, bootstrap apparié par blocs (avenant n° 1). Seuils : borne
haute ≤ +0,02 m/s de 100 à 500 m ; borne haute ≤ 0 (strict) à 650, 800 et 1000 m.

| Niveau | IC 95 % de Δ u | IC 95 % de Δ v | Verdict du niveau |
|---|---|---|---|
| 100 m | [−0,035 ; −0,021] | [+0,011 ; +0,016] | conforme |
| 150 m | [−0,120 ; −0,079] | [+0,031 ; +0,048] | **non conforme** (v) |
| 200 m | [−0,130 ; −0,093] | [+0,010 ; +0,023] | **non conforme** (v) |
| 300 m | — | — | conforme |
| 400 m | — | — | conforme |
| 500 m | — | — | conforme |
| 650 m | [+0,013 ; +0,025] | [−0,048 ; −0,019] | **non conforme** (u, seuil strict 0) |
| 800 m | — | — | conforme |
| 1000 m | [−0,171 ; −0,148] | [+0,041 ; +0,047] | **non conforme** (v, seuil strict 0) |

- Pire niveau sur les épisodes : u 1,148 → 1,056 ; v 1,068 → 1,104.
- La règle n'est pas tenue à quatre niveaux : v à 150, 200 et 1000 m, u à 650 m.
  **C2 non conforme**.

### C3 — critères inchangés : CONFORME

- DSO2 : 2,58e-5 (≤ 1e-4).
- DSO3 : 0,951 (dans [0,9 ; 1,1]).
- DSO5 : 0,508 (≤ 0,7 à 25 % des capteurs).
- DSO4 : 121 ms pour la chaîne complète, dont 92 ms pour le modèle seul (< 1 s, A100).

### Pour information, hors règle

- 100 % des capteurs : u 0,786 → 0,508 ; v 0,556 → 0,426.
- **Jeu vierge, graine 2027 (avenant n° 2)** : non mesuré, car le jeu n'est pas sur le Drive
  (la cellule M1-f n'a rien fait). Hors règle, sans effet sur la décision.

### Décision

**Correction NON RETENUE** : C1 et C3 sont conformes, C2 ne l'est pas (C1 ET C2 ET C3 exigés).

- **La v7 est gelée seule** (§ 8). Le DSO1 reste non atteint à 650–1000 m ; il est rapporté
  honnêtement, avec le critère OMM (DSO1 révisé proposé, conforme ; `docs/rapport_dso1.md`).
- La correction reste sur sa branche (`agent/ines/m1-correction-post-modele`), documentée,
  non fusionnée. Pas de v8 ni de réentraînement sans accord de Rayen (§ 6).
- **Réserve imprimée par 10i** : les coefficients sont appris sur du synthétique. Sur données
  réelles, l'erreur des baromètres (±10 à 50 Pa) est du même ordre que le signal géostrophique
  (26 Pa pour 10 m/s) : il faudrait recalibrer.
- Les réserves des § 5.3 et de l'avenant n° 1 s'appliquent : environ 3 situations
  indépendantes dans les épisodes, C2 de puissance limitée.

### Point à signaler (contrôle M1-a)

- Le sha256 de `eval_best_dso_v7/test_summary.json` **sur le Drive** vaut
  `28a5bbb8fb87b2c61f3c948760b899bb9c81590dd725c13b0550e6475c123894`. Il **diffère** de la valeur
  publiée au § 8 (`7944ba0a...`).
- La session ne l'a pas réécrit : `RUN_DIR` était redirigé vers un dossier local pendant le
  BLOC 10 (avenant n° 3). Le fichier a donc été modifié **avant** la session.
- La copie locale installée sur ines-pc (`AppData/Local/pcfno3d/runs/v7/eval_best_dso_v7/test_summary.json`)
  a bien le sha256 publié au § 8 (vérifié le 2026-10-16).
- À investiguer. **Sans effet sur le verdict** : 10i recalcule tout à partir des poids, dont le
  sha256 est vérifié.

### Gel

Le gel est consigné dans `docs/resultats/modele_fige/README.md`. Le tag annoté est **`v7-fige`**
(nom retenu par l'utilisatrice à la place de `modele-fige-v7`, prévu au § 8).
