# Réponse aux 4 points demandés avant l'accord de régénération (10/11/2026)

Rien n'a été régénéré ni entraîné. Toutes les mesures portent sur la **validation 2021** et n'utilisent aucun
instant du test 2022. ERA5 et la vérité générée sont des **vérités d'entraînement** ; les METAR des aéroports sont
les **mesures**.

## 1. Solution de repli β = 0,25 / 0,5 / 0,75

Règle `regle_repli_complements.json` (sha256 78a1c44…, écrite avant le calcul). Script `scripts/repli_complements.py`.
Sorties dans `docs/resultats/national/repli_complements/`.
- **Instants :** les 40 de la phase 2, soit 20 de calibration (graine 2026) et 20 d'acceptation (graine 2027, jours
  disjoints).
- **Mesure :** vent à 10 m, bloc 2 × 2, METAR à 30 min au plus. Écart = RMSE vectoriel de la vérité − RMSE d'ERA5
  seule, sur les mêmes paires.
- **IC 95 % :** bootstrap sur les **instants** (toutes les paires d'un instant tiré sont gardées), 5000 tirages.
- **Contrôle :** un seul passage du générateur à β = 1 ; les trois β s'en déduisent par linéarité. Les RMSE déjà
  publiés de la phase 2 sont retrouvés **exactement** (écart 0,0 m/s).

**Écart aux 12 aéroports (m/s ; négatif = meilleure qu'ERA5)**

| Vérité | Acceptation (226 paires) : écart [IC 95 %] | Calibration (223 paires) | Les 40 instants (449 paires) | Verdict de la règle (IC) | Règle d'origine (acceptation ≤ ERA5) |
|---|---|---|---|---|---|
| β = 0,25 | −0,011 [−0,031 ; +0,009] | −0,012 [−0,035 ; +0,012] | −0,012 [−0,027 ; +0,004] | équivalente à ERA5 | oui (2,929 ≤ 2,940) |
| **β = 0,5** | **−0,015 [−0,058 ; +0,027]** | −0,015 [−0,062 ; +0,035] | −0,015 [−0,047 ; +0,019] | **équivalente à ERA5** | **oui (2,925 ≤ 2,940)** |
| β = 0,75 | −0,009 [−0,076 ; +0,058] | −0,003 [−0,074 ; +0,073] | −0,006 [−0,055 ; +0,046] | équivalente à ERA5 | oui (2,931 ≤ 2,940) |
| v2 (référence) | **+0,174 [+0,075 ; +0,273]** | +0,282 [+0,161 ; +0,409] | +0,226 [+0,145 ; +0,310] | **moins bonne qu'ERA5** | non (3,114) |

**Validation croisée par station.**
- **Choix sans les 4 stations :** β est choisi sur les 8 autres aéroports et la calibration, avec le critère de la
  règle de repli. RMSE : 2,555 (β 0,25), 2,558 (β 0,5) et 2,578 (β 0,75). L'égalité à 0,005 près entraîne le plus
  grand β, soit **β = 0,5, le même que le choix sur les 12 stations**.
- **Jugement sur Tabarka, Gafsa, Sfax et Djerba, aux jours d'acceptation** (75 paires, stations et jours jamais vus
  par le choix) :

| Vérité | Écart à ERA5 [IC 95 %] | Verdict |
|---|---|---|
| β = 0,25 | −0,006 [−0,029 ; +0,018] | équivalente |
| **β = 0,5 (choisi)** | **−0,002 [−0,047 ; +0,043]** | **équivalente ; le choix généralise (écart ≤ 0)** |
| β = 0,75 | +0,013 [−0,051 ; +0,077] | équivalente |
| v2 | +0,269 [+0,129 ; +0,396] | moins bonne |

**Lecture honnête.**
- Aucun β n'est **significativement** meilleur qu'ERA5 aux aéroports : les IC contiennent tous 0, et le gain est
  au mieux de 0,015 m/s (0,5 %). Les aéroports ne permettent pas non plus de départager 0,25, 0,5 et 0,75.
- La vérité v2, elle, était **significativement moins bonne** qu'ERA5, de 0,17 à 0,28 m/s, et aussi sur les 4
  stations retenues.
- La correction fait donc passer la vérité d'entraînement de « moins bonne que la donnée d'entrée » à « aussi
  bonne qu'elle aux mesures ». Elle garde en plus un détail local que les mesures ne contredisent pas.

## 2. β = 0,5 : écart à ERA5 et détail gardé, comparés à la v2

Mêmes 20 instants d'acceptation (validation). L'écart rms est mesuré entre la vérité et ERA5 à la même hauteur au-dessus
du sol. La rugosité est la moyenne des |différences| entre nœuds voisins (1 km) en x et en y, u et v sommés, selon la
définition de `dso_tranches`.

| Vérité | Composante | Écart rms vérité − ERA5, 300 m | 800 m | Rugosité vérité / ERA5, 300 m | 800 m |
|---|---|---|---|---|---|
| v2 | u | 1,540 | 1,261 | 6,39 | 4,96 |
| v2 | v | 2,240 | 2,121 | 7,43 | 6,01 |
| **v3 β = 0,5** | u | **0,370** | **0,274** | **2,57** | **2,05** |
| **v3 β = 0,5** | v | **0,434** | **0,320** | **3,01** | **2,38** |
| v3 β = 0,25 (info) | v | 0,229 | 0,186 | 1,95 | 1,67 |
| v3 β = 0,75 (info) | v | 0,646 | 0,466 | 4,17 | 3,16 |
| v3 β = 1, correction seule (info) | v | 0,861 | 0,615 | 5,36 | 3,98 |

**Ce qu'on garde du détail à 1 km** (information, hors règle). Gradient moyen entre nœuds voisins du détail
vérité − ERA5, en m/s par km, sur v à 300 / 800 m :

| Vérité | Gradient du détail, 300 m | 800 m | Part du détail de la v2 |
|---|---|---|---|
| v2 | 0,310 | 0,224 | 100 % |
| v3, correction seule (β = 1) | 0,215 | 0,137 | 69 % / 61 % |
| **v3 β = 0,5** | **0,110** | **0,072** | **35 % / 32 %** |

Pour comparaison, ERA5 seule a un gradient de 0,041 / 0,037 m/s par km à 1 km.

- **Ce que garde la vérité v3 β = 0,5 :** elle reste **2,4 à 3 fois plus rugueuse qu'ERA5** à 1 km. Elle conserve
  environ **un tiers du détail local de la v2** et la moitié de celui du générateur corrigé.
- **Ce qu'elle perd :** l'écart à grande échelle, puisque l'écart rms vérité − ERA5 est divisé par 4 à 7 par rapport
  à la v2.
- **Ce qu'en dit le point 1 :** la part du détail retirée n'était pas confirmée par les mesures, puisque la v2 était
  moins bonne qu'ERA5. La part gardée est compatible avec les mesures (équivalente à ERA5).

## 3. Diagnostic 5 : cohérence de l'ERA5 du modèle et du générateur — aucun bug

Règle `regle_phase1_complements.json` (sha256 752e16c…, écrite avant le calcul). Script
`scripts/diagnostic_coherence_era5.py`, 20 instants de validation 2021 (graine 2028). Sorties dans
`docs/resultats/national/phase1_diag5/`.

| Contrôle | Résultat |
|---|---|
| RMSE modèle − générateur, u et v, sur les 16 niveaux et à 300 / 800 m au-dessus du sol | ≤ 1,4·10⁻⁶ m/s (arrondi float32) |
| Biais u et v | ≤ 7·10⁻⁹ m/s |
| Ordre des latitudes | le fichier ERA5 brut est du nord au sud ; `forcage_era5.py` le retrie (`argsort`) du sud au nord dans le npz ; le modèle et le générateur lisent ce même npz (latitudes, longitudes et temps identiques) |
| Valeurs du npz contre le fichier brut `era5_sl_2021_m01-06.nc` (u10, v10 ; 3 mailles × 3 instants) | écart 0,000 m/s, **même signe de v** |
| Instant UTC | temps du npz = `valid_time` du fichier brut ; horodatage du cache = jour du générateur + heure UTC |
| Interpolation verticale | les deux côtés interpolent **au-dessus du sol**, avec le même relief aux nœuds (écart 0 m), la même hauteur (écart 0 m) et le même plancher de 5 m |
| Position des nœuds | ≤ 0,07 m d'écart entre la projection du modèle et la grille du générateur |

Verdict de la règle : **COHÉRENT**. Il n'y a rien à corriger avant la régénération.

## 4. Ce que la correction change dans le générateur, et pourquoi c'est physique

**Comment le générateur construit la vérité** (`scripts/generate_aneo_3d.py`, `compute_fields`) :
1. u et v sont d'abord construits par les règles du générateur : profil de couche limite, accélération par le
   relief, vents de pente, brise, perturbations.
2. **Recalage sur ERA5.** À chaque niveau, la partie lisse du champ (au-delà de 12 km) est remplacée par ERA5 à la
   même hauteur au-dessus du sol. Le générateur ne garde que le détail de moins de 12 km.
3. **w par la continuité**, intégrée de bas en haut à partir de la condition au sol w = u·∇z_t.
4. **Amortissement de w.** Sans lui, le mouvement vertical créé par le relief remonterait jusqu'au sommet du
   domaine. À chaque niveau, le générateur ajoute donc un petit écoulement horizontal potentiel grad(χ)/ρ. Sa
   divergence retire une fraction 1 − exp(−Δz/300 m) du flux vertical hérité du niveau inférieur. χ est solution
   de ∇²χ = 2·excès/Δz, résolue par FFT sur **tout le domaine**. w est ensuite réintégré, et la continuité reste
   exacte.

**Le défaut.** L'étape 4 vient **après** le recalage de l'étape 2, et sa solution n'est pas locale :
- En Fourier, χ̂ = source / (−4π²k²). Le vent induit, grad χ, varie donc comme source / k : plus l'échelle est
  grande, plus il est amplifié.
- Un excès de flux vertical sur l'Atlas ou la Dorsale crée ainsi un vent irrotationnel jusqu'à l'échelle du pays.
  Ce vent s'ajoute à un champ déjà recalé sur ERA5 et éloigne la grande échelle de la vérité d'ERA5 de 1,4 à 2 m/s.
- Le domaine mesure 828 km du nord au sud et 388 km d'est en ouest. Le plus petit nombre d'onde non nul en y
  (1/828 km) est donc deux fois plus petit qu'en x (1/388 km). Les modes les plus longs, les plus amplifiés,
  existent surtout le long de y : l'erreur porte davantage sur v que sur u.
- Le diagnostic 4 le confirme sur la validation. Couper l'amortissement retire 79 à 89 % de l'écart vérité −
  ERA5. Les autres mécanismes pèsent 0 à 7 %.

**La correction** (`ANEO_GEN_AMORT_PASSE_HAUT=1`). Seule la partie de moins de 12 km du terme d'amortissement est
gardée (`gx -= passe_bas(gx, 12 km)`, de même pour gy), puis w est réintégré par la continuité.

**Pourquoi c'est physique :**
- **Échelle.** L'amortissement du mouvement vertical forcé par le relief est un ajustement **local**, à l'échelle
  des reliefs, de la dizaine de kilomètres. Une montagne ne crée pas un vent uniforme à 400 km. La circulation
  divergente de grande échelle, y compris celle due aux grands massifs, est déjà dans ERA5, qui la résout à ses
  propres échelles.
- **Cohérence avec le recalage.** C'est la même séparation d'échelles que le recalage de l'étape 2 : au-delà de
  12 km, ERA5 ; en dessous, le détail du générateur. Le terme d'amortissement était le seul à ne pas la respecter.
- **Masse.** w est réintégré à partir des u, v définitifs, donc la conservation de la masse reste exacte (DSO2 par
  construction). L'amortissement de w continue d'agir autour de chaque relief.
- **Artefact numérique supprimé.** La périodicité de la FFT, qui relie le nord au sud du pays, ne crée plus de vent
  à grande échelle.

**Les autres réglages, et ce qu'ils ne sont pas :**
- **Perturbations 0 et brise 0,5.** Ils ont été choisis sur la calibration (règle `regle_verite_v3.json`). Ils ne
  déplacent presque pas l'écart sur v (0,918 → 0,891, puis 0,8917 → 0,8914). Les perturbations aléatoires sont des
  structures inventées que les stations ne peuvent pas confirmer. Il est donc cohérent de les retirer d'une vérité
  d'entraînement.
- **β = 0,5 (repli).** u, v = ERA5 + 0,5 × (généré − ERA5), puis w par la continuité. Ce n'est **pas** une loi
  physique : c'est un **étalonnage de l'amplitude du détail local sur des mesures**, les METAR de la validation. Les
  règles du générateur donnent la forme du détail (relief, côte, pente), mais leur amplitude, paramétrée, est trop
  forte face aux aéroports. La masse reste conservée. Le point 1 montre si cet étalonnage tient sur des stations et
  des jours qui n'ont pas servi au choix.
