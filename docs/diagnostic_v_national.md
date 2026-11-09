# Diagnostic de v, modèle national (9 novembre 2026)

**Question.** Au-dessus de 200 m, le RMSE de v du modèle national est 1,4 à 1,5 fois celui de u.
Pourquoi ?

**Réponse.** Le défaut vient de la **vérité** (le générateur), pas du modèle ni du chargement des
données. La correction d'amortissement de w du générateur est appliquée **après** le recalage sur
ERA5. Elle introduit dans la vérité un écoulement potentiel **de grande échelle**, plus fort sur v,
que rien ne relie à ERA5. Le modèle doit l'apprendre.

## 1. Vérifications de code : aucun défaut

| Point | Résultat |
|---|---|
| Orientation de la grille | `iy` croît vers le nord, `ix` vers l'est (Bizerte iy = 1574, Remada iy = 480). Le nœud (0, 0) est à 30,15 N, 7,45 E. |
| Distance entre nœuds | Nord-sud 1002 m partout. Est-ouest 1039 m au sud, 998 m au centre, 952 m au nord (plan local, ±4,5 %). Le même plan sert à la vérité et au modèle : cette approximation ne crée pas d'écart entre u et v. |
| Ordre des latitudes ERA5 | `forcage_era5.py` remet les latitudes en ordre croissant **avec** les champs (`lat[o]`, `champs[:, :, o]`). |
| Signe de v | Convention météo (v > 0 vers le nord) dans ERA5, le générateur et les METAR (`v = −vitesse·cos(direction)`). |
| Coordonnées des nœuds pour ERA5 | Identiques au mètre près entre le générateur (`grille_reelle.json`) et le modèle (`InterpERA5`). |
| Normalisation (`norm_stats.json`) | y_mu u = 0,57, v = −0,87 ; y_sd u = 5,92, v = 4,82. Pas d'anomalie. |

## 2. ERA5 seule contre la vérité (180 instants de test, `scripts/diagnostic_v.py`)

| Niveau (m) | RMSE ERA5 u | RMSE ERA5 v | v / u | corr. u–u | corr. v–v | corr. croisées |
|---|---|---|---|---|---|---|
| 650 | 1,57 | 2,18 | 1,39 | 0,964 | 0,907 | −0,23 / −0,20 |
| 800 | 1,53 | 2,21 | 1,44 | 0,965 | 0,908 | −0,22 / −0,20 |
| 1000 | 1,46 | 2,00 | 1,37 | 0,965 | 0,924 | −0,21 / −0,18 |
| 1500 | 1,27 | 1,87 | 1,47 | 0,970 | 0,932 | −0,11 / −0,07 |
| 2000 | 0,89 | 1,42 | 1,60 | 0,987 | 0,965 | +0,05 / +0,07 |
| 2500 | 1,16 | 1,88 | 1,62 | 0,981 | 0,954 | +0,17 / +0,18 |

- **ERA5 seule a le même rapport v/u (1,4 à 1,6) que le modèle.** Le défaut est donc dans les données.
- Les corrélations directes sont élevées et les corrélations croisées faibles : u et v ne sont ni échangés ni de signe inversé.
- Le biais moyen est nul : le recalage sur ERA5 fonctionne en moyenne.
- Le modèle réduit l'erreur d'ERA5 sur u comme sur v. À 800 m : ERA5 1,53 / 2,21, national v2 0,65 / 0,97 m/s.

## 3. Ce n'est pas un effet de régime de vent

- **Pas une rotation du vent.** Le détail (vérité − ERA5) est plutôt parallèle au vent ERA5 que perpendiculaire (à 800 m : 2,30 contre 1,92).
- **Pas une saison.** v est pire que u en hiver comme en été, même quand |u| ≈ |v|. Hiver 2500 m : RMSE u 1,15, v 2,05, avec |u| 5,9 et |v| 6,0 m/s.

## 4. Un écart de grande échelle, créé par l'amortissement de w

- Le détail varie peu d'un nœud à l'autre (0,2 à 0,3 m/s) alors que son rms vaut 1,5 à 2 m/s.
- 87 à 92 % de son rms passe un filtre passe-bas de 12 km. Exemples : 800 m, v 2,33 sur 2,52 ; 2500 m, v 1,66 sur 1,84.
- Le générateur est pourtant conçu pour que la grande échelle soit celle d'ERA5 (recalage spatial, `compute_fields`).

Test direct, même instant d'entraînement (2024-03-12, 12 h UTC), avec et sans l'étape 2 de
`compute_fields` (amortissement vertical de w par un écoulement potentiel, Poisson par FFT) :

| | 800 m u | 800 m v | 2500 m u | 2500 m v |
|---|---|---|---|---|
| avec amortissement (jeu actuel) | 1,39 (> 12 km : 1,03) | 1,36 (0,94) | 0,90 (0,71) | **1,98 (1,84)** |
| sans amortissement | 0,20 (0,11) | 0,20 (0,11) | 0,11 (0,01) | 0,11 (0,01) |

Valeurs : écart RMS vérité − ERA5 (m/s), et entre parenthèses sa part au-dessus de 12 km.

**Cause.** Dans `compute_fields`, le recalage remplace la grande échelle (lissée à 12 km) par
ERA5. Ensuite seulement, l'étape d'amortissement ajoute à chaque niveau `grad(χ)/ρ`, où χ est
obtenu par une résolution de Poisson par FFT sur tout le domaine, périodique. Cette correction a
une composante de grande échelle, à l'échelle du domaine, plus forte dans l'axe long nord-sud,
donc sur v. Rien ne la ramène vers ERA5.

## 5. Conséquences et corrections possibles

- Une grande partie de l'échec du DSO1 au-dessus de 400 m mesure la capacité du modèle à
  reproduire un artefact du générateur, et non le vent réel.
- **Corrections possibles du générateur** : appliquer l'amortissement **avant** le recalage, de
  sorte que le recalage en retire la grande échelle ; ou filtrer en passe-haut la correction
  (> 12 km retirés). Il faut ensuite régénérer le jeu national et réentraîner : **à décider par
  Ines (accord écrit requis avant tout entraînement)**.
- La limite des modes de Fourier dans l'axe nord-sud (26 km, contre 12 km est-ouest) reste une
  anisotropie du modèle. Elle n'est pas nécessaire pour expliquer le rapport v/u, déjà présent
  dans ERA5 seule contre la vérité.

## 6. Attribution par composante, à 300 et 800 m au-dessus du sol (9 novembre 2026)

`scripts/attribution_v_generateur.py`. Diagnostic, pas une régénération du jeu : 4 instants
d'entraînement (2023-02-14 et 2024-08-06, à 3 h et 15 h UTC), calculés en coupant une composante
à la fois, avec les mêmes graines. Valeurs : écart RMS vérité − ERA5 (m/s), et entre parenthèses
sa part au-dessus de 12 km.

| Configuration | 300 m u | 300 m v | 800 m u | 800 m v |
|---|---|---|---|---|
| actuel (jeu v2) | 1,05 (0,93) | 1,28 (1,15) | 0,90 (0,83) | 1,35 (1,28) |
| sans amortissement de w | **0,17 (0,12)** | **0,15 (0,09)** | **0,10 (0,06)** | **0,11 (0,07)** |
| amortissement filtré < 12 km (correction) | 0,38 (0,19) | 0,45 (0,21) | 0,25 (0,14) | 0,31 (0,18) |
| sans brise | 1,05 | 1,27 | 0,90 | 1,35 |
| sans vents de pente | 1,04 | 1,24 | 0,93 | 1,39 |
| sans accélération par le relief | 1,05 | 1,28 | 0,90 | 1,35 |
| sans perturbations aléatoires | 1,04 | 1,27 | 0,90 | 1,34 |

À ces hauteurs, **l'amortissement de w explique 85 à 92 % de l'écart**, et presque tout l'excès
sur v. La brise, les vents de pente, le relief et les perturbations aléatoires y changent l'écart
de 0,01 à 0,05 m/s seulement. Au sol (10 m), aux aéroports, c'est en revanche la brise qui pèse le
plus (docs/diagnostic_generateur_national.md).

Sur tout le test 2022 (`scripts/carte_v_generateur.py`, 180 instants), l'écart RMS vérité − ERA5
vaut u 1,449 et v 2,083 m/s à 300 m au-dessus du sol, et u 1,353 et v 1,909 m/s à 800 m, avec un
biais moyen nul. Cartes sur le Drive : `runs/national_1km_v2/diagnostic_v/carte_v_300m.png` et
`carte_v_800m.png`.
