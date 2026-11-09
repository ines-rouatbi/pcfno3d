# Diagnostic du générateur national aux 12 aéroports (9 novembre 2026)

**Données.** `runs/national_1km_v2/validation_reelle/test/stations_paires.csv` : test 2022,
instants synoptiques, 1938 paires (station retirée de l'entrée). On compare, à 10 m :
- l'observation METAR ;
- ERA5 interpolé ;
- le générateur, c'est-à-dire la vérité d'entraînement (canal surface du cache, moyenne de la
  cellule de 1 km).

Les seuils et réglages qui en découleront devront être choisis sur les années
d'entraînement et de validation, jamais sur ce test 2022.

## 1. Ce que le générateur ajoute à ERA5 ne correspond pas à ce qu'ERA5 rate

Notations :
- d = générateur − ERA5 : le détail local ajouté par le générateur (relief, côte, brise, rugosité) ;
- e = observation − ERA5 : ce qu'ERA5 rate réellement ;
- α = ⟨d, e⟩ / ⟨d, d⟩ : la fraction du détail qui va dans le bon sens.

Lecture de α : 1 = détail juste ; 0 = détail sans rapport avec la réalité ; < 0 = détail à
contresens.

| | valeur |
|---|---|
| α, toutes stations | **−0,08** |
| RMSE vectoriel avec le détail (générateur) | 3,15 m/s |
| RMSE vectoriel sans le détail (ERA5 seul) | 2,88 m/s |

**À l'échelle des stations, les détails du générateur sont du bruit.** Ils dégradent ERA5
d'environ 9 %, et le modèle, qui les apprend fidèlement, hérite de cette dégradation.

## 2. Par station

| Aéroport | RMSE gén. | RMSE ERA5 | détail ajouté (rms) | α | vitesse obs. | vitesse ERA5 | vitesse gén. | cellule : sol, z0 | relief ±5 km (m) | mer (km) |
|---|---|---|---|---|---|---|---|---|---|---|
| Bizerte | 2,97 | 2,97 | 0,48 | +0,50 | 4,34 | 4,48 | 4,28 | agricole, 0,08 | 196 | 1,0 |
| Djerba-Zarzis | 2,42 | 2,59 | 1,22 | +0,79 | 4,51 | 4,76 | 4,40 | urbain, 0,20 | 13 | 2,5 |
| Enfidha | 3,05 | 3,14 | 0,48 | +1,77 | 4,18 | 3,48 | 3,71 | plaine, 0,01 | 21 | 3,6 |
| Gabès | 3,00 | 2,93 | 1,10 | +0,33 | 4,80 | 4,47 | 4,54 | urbain, 0,69 | 66 | 2,5 |
| Gafsa | 3,28 | 3,16 | 1,56 | +0,34 | 4,38 | 2,97 | 3,89 | plaine, 0,09 | 305 | 80 |
| Kairouan | 2,42 | 2,36 | 0,46 | −0,24 | 2,00 | 2,93 | 2,99 | urbain, 0,59 | 45 | 26 |
| Monastir | 3,08 | 2,68 | 1,20 | −0,31 | 4,28 | 3,68 | 3,46 | urbain, 0,26 | 57 | 0,5 |
| Remada | 3,37 | 2,35 | 1,67 | −0,55 | 2,78 | 4,18 | 5,08 | urbain, 0,51 | 212 | 123 |
| Sfax | 2,97 | 2,95 | 0,58 | +0,31 | 3,64 | 4,69 | 4,84 | plaine, 0,03 | 55 | 4,9 |
| Tabarka | 4,32 | 3,24 | 1,90 | −0,62 | 3,87 | 4,17 | 4,55 | plaine, 0,38 | 305 | 1,6 |
| Tozeur | 3,29 | 3,14 | 1,08 | +0,07 | 5,37 | 3,63 | 3,72 | plaine, 0,07 | 139 | 36 |
| Tunis-Carthage | 3,12 | 2,78 | 1,21 | −0,21 | 3,53 | 4,01 | 3,65 | urbain, 0,20 | 79 | 1,0 |

- **Le relief accélère le vent à tort.** À Remada, le générateur passe de 4,18 (ERA5) à
  5,08 m/s alors que l'observation est à 2,78. Tabarka, deuxième relief le plus marqué
  (305 m de dénivelé à ±5 km), montre la même chose. Ce sont les deux plus gros détails
  ajoutés (1,7 et 1,9 m/s rms), avec un α négatif.
- **Côte et brise à contresens.** Monastir (0,5 km de la mer), Tunis-Carthage et Tabarka ont
  un α négatif.
- **Bon sens, mais trop fort.** Bizerte, Gabès, Gafsa et Sfax ont un α entre 0,3 et 0,5 :
  le détail va dans la bonne direction mais il est 2 à 3 fois trop grand.
- **Six aéroports sur douze sont en cellule « urbain »** (z0 de 0,2 à 0,69 m dans
  WorldCover). Un anémomètre d'aéroport est sur terrain dégagé (z0 ≈ 0,03).

## 3. Par heure (UTC)

| heure | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 |
|---|---|---|---|---|---|---|---|---|
| RMSE générateur | 3,17 | 2,77 | 2,99 | 2,75 | 3,58 | 3,50 | 3,43 | 2,81 |
| RMSE ERA5 | 3,09 | 2,63 | 2,86 | 2,54 | 3,24 | 3,03 | 2,97 | 2,60 |
| α | +0,24 | −0,01 | +0,01 | +0,08 | 0,00 | **−0,25** | **−0,28** | −0,11 |
| vitesse obs. / ERA5 / gén. | 3,38 / 3,60 / 3,71 | 3,13 / 3,43 / 3,61 | 3,02 / 3,21 / 3,30 | 4,36 / 4,07 / 4,22 | 4,86 / 4,31 / 4,39 | 5,19 / 4,66 / 4,83 | 4,45 / 4,44 / 4,59 | 3,56 / 3,91 / 4,02 |

L'écart est le plus grand l'après-midi (15 h et 18 h UTC), quand la brise de mer et le
mélange convectif dominent : α y vaut environ −0,25, donc le détail ajouté est à contresens.

## 4. Leviers

1. **Côté modèle, sans régénérer le jeu.** Affiner le modèle national sur les vraies mesures
   METAR (terme de perte à 10 m, une station retirée à chaque fois). Le code existe déjà
   (`stations_reelles`, test `test_affinage_sur_les_stations`). Il faut l'entraîner sur les
   années d'entraînement, jamais sur 2022. C'est le levier le plus direct pour DSO7.
2. **Côté générateur, régénération d'environ 50 Go.** Réduire l'accélération par le relief
   à 1 km ; revoir la phase et la direction de la brise ; rugosité de terrain dégagé aux
   cellules d'aérodrome ; amplitude des détails réglée sur les stations des années
   d'entraînement (α visé > 0).
3. **Minimal.** Diminuer l'amplitude des détails (`ANEO_GEN_PERTURB`, brise, relief) : la
   vérité se rapproche d'ERA5. Le modèle ne dégradera plus ERA5 aux stations, mais il n'y
   apportera rien non plus.
