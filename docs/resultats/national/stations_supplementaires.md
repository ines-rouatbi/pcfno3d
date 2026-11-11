# Stations tunisiennes NOAA ISD et OGIMET, 2015–2022 (modèle national)

Écrit par `scripts/stations_supplementaires.py` (demandes d'Ines du 11/11/2026). Sources : NOAA NCEI ISD (`isd-history.csv` et fichiers annuels 2015–2022), liste OGIMET des indicatifs SYNOP du bloc 607, ERA5 10 m locale (2021, indicateur d'abri), Copernicus DEM GLO-30/GLO-90, ESA WorldCover 2021, Natural Earth (trait de côte).

**Deux usages.** (1) Stations de validation **supplémentaires** du modèle national : jugées seulement, **jamais en entrée** (C' et D gardent les 12 aéroports METAR de `stations.json`, la règle v4 `60671e84…` aussi) ; bilan à part du DSO7 (`validation_reelle/<jeu>/stations_supp_*.csv`, `bilan_supp.json`), hors décision v4. Les mettre en entrée serait une décision d'Ines. (2) Stations de la règle MOS proposée (`configs/national/regle_mos_era5.json`), aéroports compris.

- stations tunisiennes actives en 2015–2022 : **29** (ISD : 29 ; OGIMET : 28 indicatifs SYNOP ouverts) ;
- enregistrements des 12 aéroports METAR (code OACI ou moins de 3 km) : **13** (aéroports couverts : 12 ; Monastir a deux enregistrements, le SYNOP INM 60740 et l'identifiant ISD 607403) ; hors du domaine national : **0** ;
- **retenues** comme stations de validation supplémentaires : **16** (dans le domaine, pas un aéroport, au moins 50% des créneaux de 3 h de 2022 valides après contrôle) : Kelibia (INM60720), Beja (INM60723), Jendouba (INM60725), Nabeul (INM60728), Zaghouan (INM60729), El Kef (INM60732), Siliana (INM60734), Thala (INM60738), Kasserine (INM60739), Mahdia (INM60742), Sidi Bouzid (INM60748), Kebili (INM60764), Gabes Airport (INM60767), Medenine (INM60770), Tataouine (INM60772), El Borma (INM60780) ;
- téléchargé : ISD 647 Mo, sites (plages d'octets des COG + trait de côte) 0 Mo ; dossiers ignorés par git, jamais versionnés.

## Liste et part de messages valides par année

Part (%) des créneaux de 3 h (00, 03, …, 21 UTC ; 2 920 par an) avec un message de vent valide (qualité ISD non suspecte, vitesse, et direction ou calme) à 30 min au plus, **après contrôle qualité** (avant contrôle : colonnes `taux_brut_*` du CSV).

| Code | Nom | OACI | lat | lon | alt. (m) | aéroport METAR | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | retenue (validation) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| INM60710 | Tabarka | DTKA | 36.9758 | 8.8758 | 20 | DTKA | 97 | 99 | 98 | 97 | 97 | 99 | 99 | 99 | non |
| INM60714 | Bizerte | DTTB | 37.2500 | 9.8000 | 5 | DTTB | 93 | 96 | 97 | 98 | 96 | 98 | 96 | 94 | non |
| INM60715 | Tunis-Carthage | DTTA | 36.8506 | 10.2144 | 4 | DTTA | 100 | 100 | 100 | 99 | 100 | 100 | 99 | 99 | non |
| INM60720 | Kelibia | DTTL | 36.8500 | 11.0831 | 29 | — | 53 | 78 | 96 | 93 | 83 | 99 | 97 | 98 | **oui** |
| INM60723 | Beja | — | 36.7356 | 9.2258 | 158 | — | 54 | 45 | 72 | 55 | 68 | 95 | 73 | 94 | **oui** |
| INM60725 | Jendouba | DTTN | 36.4878 | 8.7908 | 143 | — | 90 | 97 | 96 | 65 | 94 | 96 | 95 | 97 | **oui** |
| INM60728 | Nabeul | — | 36.4500 | 10.7333 | 1 | — | 43 | 49 | 8 | 74 | 89 | 89 | 66 | 83 | **oui** |
| INM60729 | Zaghouan | — | 36.4331 | 10.0831 | 156 | — | 83 | 94 | 88 | 68 | 93 | 82 | 82 | 93 | **oui** |
| INM60731 | Enfidha | DTNH | 36.0667 | 10.4497 | 6 | DTNH | 100 | 99 | 99 | 99 | 99 | 100 | 98 | 100 | non |
| INM60732 | El Kef | — | 36.1228 | 8.7239 | 518 | — | 65 | 85 | 86 | 95 | 97 | 94 | 92 | 80 | **oui** |
| INM60734 | Siliana | — | 36.0539 | 9.4039 | 443 | — | 59 | 82 | 85 | 55 | 84 | 87 | 87 | 86 | **oui** |
| INM60735 | Kairouan | DTTK | 35.6667 | 10.0997 | 60 | DTTK | 98 | 99 | 99 | 99 | 99 | 99 | 98 | 95 | non |
| INM60738 | Thala | — | 35.5608 | 8.6878 | 1091 | — | 85 | 92 | 94 | 93 | 85 | 92 | 93 | 91 | **oui** |
| INM60739 | Kasserine | — | 35.1681 | 8.7914 | 707 | — | 73 | 94 | 41 | 1 | 58 | 97 | 97 | 96 | **oui** |
| INM60740 | Monastir-Skanes | DTTM | 35.7622 | 10.7575 | 2 | DTMB | 95 | 97 | 92 | 88 | 94 | 99 | 97 | 98 | non |
| INM60742 | Mahdia | — | 35.5000 | 11.0667 | 12 | — | 62 | 72 | 88 | 70 | 93 | 93 | 90 | 94 | **oui** |
| INM60745 | Gafsa | DTTF | 34.4167 | 8.8167 | 313 | DTTF | 88 | 89 | 95 | 93 | 96 | 97 | 95 | 96 | non |
| INM60748 | Sidi Bouzid | — | 35.0119 | 9.4856 | 354 | — | 78 | 92 | 68 | 63 | 62 | 96 | 91 | 91 | **oui** |
| INM60750 | Sfax El-Maou | DTTX | 34.7167 | 10.6831 | 21 | DTTX | 97 | 98 | 98 | 96 | 97 | 99 | 97 | 97 | non |
| INM60760 | Tozeur | DTTZ | 33.9167 | 8.1667 | 86 | DTTZ | 95 | 97 | 98 | 97 | 99 | 99 | 97 | 96 | non |
| INM60764 | Kebili | — | 33.7000 | 8.9667 | 44 | — | 68 | 96 | 96 | 71 | 95 | 97 | 96 | 97 | **oui** |
| INM60765 | Gabes | DTTG | 33.8833 | 10.1067 | 4 | DTTG | 83 | 92 | 92 | 92 | 93 | 100 | 98 | 99 | non |
| INM60767 | Gabes Airport | — | 33.7392 | 9.9164 | 120 | — | 83 | 93 | 94 | 93 | 95 | 94 | 95 | 95 | **oui** |
| INM60769 | Djerba Mellita | DTTJ | 33.8667 | 10.7667 | 3 | DTTJ | 99 | 100 | 100 | 99 | 100 | 99 | 98 | 100 | non |
| INM60770 | Medenine | — | 33.3294 | 10.4881 | 116 | — | 75 | 75 | 94 | 89 | 96 | 97 | 96 | 76 | **oui** |
| INM60772 | Tataouine | — | 32.9575 | 10.4669 | 215 | — | 48 | 93 | 92 | 75 | 94 | 96 | 96 | 96 | **oui** |
| INM60775 | Remada | DTTD | 32.3167 | 10.4000 | 300 | DTTD | 77 | 95 | 99 | 91 | 98 | 99 | 96 | 97 | non |
| INM60780 | El Borma | DTTR | 31.6831 | 9.3000 | 258 | — | 60 | 58 | 48 | 50 | 53 | 80 | 76 | 70 | **oui** |
| ISD60740399999 | Habib Bourguiba Intl | DTMB | 35.7580 | 10.7550 | 3 | DTMB | 98 | 99 | 98 | 98 | 99 | 100 | 98 | 99 | non |

## Site de chaque station

Maille de 500 m de la grille nationale, recalculée autour de la station avec le code de `scripts/grille_reelle.py` (`pcfno3d.sites_stations`) : altitude et pente du relief GLO-30 lissé comme la grille, z0 et occupation majoritaire de 5 × 5 sous-points WorldCover (moyenne logarithmique) ; TPI = altitude GLO-90 de la station moins l'altitude moyenne dans un disque de 5 et 20 km (mer à 0 m) ; distance au trait de côte Natural Earth 1:10 M (lagunes et sebkhas non comprises).

| Code | Nom | alt. officielle | alt. maille | pente (°) | z0 (m) | occupation | TPI 5 km | TPI 20 km | côte (km) |
|---|---|---|---|---|---|---|---|---|---|
| INM60710 | Tabarka | 20 | 65 | 0.4 | 0.247 | plaine | 9 | -94 | 3.6 |
| INM60714 | Bizerte | 5 | 6 | 0.4 | 0.141 | agricole | -28 | -31 | 1.5 |
| INM60715 | Tunis-Carthage | 4 | 6 | 0.2 | 0.459 | urbain | -4 | -18 | 1.5 |
| INM60720 | Kelibia | 29 | 21 | 0.9 | 0.828 | urbain | -5 | -2 | 2.8 |
| INM60723 | Beja | 158 | 153 | 1.1 | 0.233 | agricole | -67 | -132 | 42.8 |
| INM60725 | Jendouba | 143 | 144 | 0.1 | 0.208 | agricole | -1 | -114 | 53.0 |
| INM60728 | Nabeul | 1 | 9 | 0.4 | 1.500 | urbain | -10 | -35 | 2.2 |
| INM60729 | Zaghouan | 156 | 141 | 0.4 | 0.095 | agricole | -20 | -79 | 40.5 |
| INM60731 | Enfidha | 6 | 2 | 0.0 | 0.035 | plaine | -0 | -20 | 3.0 |
| INM60732 | El Kef | 518 | 518 | 0.6 | 0.156 | agricole | -20 | -29 | 89.2 |
| INM60734 | Siliana | 443 | 443 | 0.6 | 0.159 | agricole | -22 | -190 | 96.4 |
| INM60735 | Kairouan | 60 | 62 | 0.2 | 1.500 | urbain | 3 | -4 | 50.5 |
| INM60738 | Thala | 1091 | 1068 | 1.9 | 0.062 | agricole | 89 | 208 | 150.3 |
| INM60739 | Kasserine | 707 | 699 | 1.5 | 0.092 | plaine | -4 | -139 | 153.5 |
| INM60740 | Monastir-Skanes | 2 | 2 | 0.1 | 0.301 | urbain | -2 | -18 | 1.5 |
| INM60742 | Mahdia | 12 | 2 | 0.3 | 0.004 | eau | -0 | -4 | 1.7 |
| INM60745 | Gafsa | 313 | 312 | 0.3 | 0.055 | plaine | 3 | -32 | 112.6 |
| INM60748 | Sidi Bouzid | 354 | 354 | 0.9 | 0.098 | plaine | -12 | -15 | 96.2 |
| INM60750 | Sfax El-Maou | 21 | 23 | 0.3 | 0.029 | plaine | -1 | -14 | 5.9 |
| INM60760 | Tozeur | 86 | 28 | 0.3 | 0.256 | foret | -7 | -2 | 171.7 |
| INM60764 | Kebili | 44 | 36 | 0.8 | 1.228 | urbain | -0 | -11 | 104.8 |
| INM60765 | Gabes | 4 | 5 | 0.3 | 0.736 | urbain | -9 | -29 | 7.4 |
| INM60767 | Gabes Airport | 120 | 120 | 0.1 | 0.054 | plaine | -3 | -13 | 25.2 |
| INM60769 | Djerba Mellita | 3 | 1 | 0.1 | 0.019 | plaine | -2 | -4 | 3.2 |
| INM60770 | Medenine | 116 | 115 | 0.2 | 1.500 | urbain | 6 | 2 | 27.3 |
| INM60772 | Tataouine | 215 | 215 | 0.3 | 0.061 | plaine | -36 | -61 | 63.0 |
| INM60775 | Remada | 300 | 289 | 1.1 | 0.688 | urbain | 5 | -28 | 123.3 |
| INM60780 | El Borma | 258 | 249 | 0.4 | 0.010 | plaine | -11 | -8 | 240.2 |
| ISD60740399999 | Habib Bourguiba Intl | 3 | 1 | 0.2 | 0.058 | urbain | -3 | -20 | 1.7 |

## Contrôle qualité

Règles existantes de `pcfno3d.stations_reelles` (mêmes que pour Nabeul 2021) et deux contrôles du contrat ANEO adaptés au SYNOP (`controle_qualite`) :
- **exclusions manuelles** (`EXCLUSIONS` : Nabeul 2023–2024, hors des années étudiées ici) ;
- **mois en panne** : vent non nul dans moins de 10 % des messages du mois (au moins 20 messages) ;
- **calme** : 0 m/s gardé comme mesure (u = v = 0) ; une vitesse sans direction (et non calme) est écartée (u, v inconnus) ;
- **plages** : vitesse hors de 0–60 m/s, direction hors de 0–360°, |u| ou |v| > 60 m/s ;
- **valeur figée** : même vitesse (≥ 0,5 m/s) et même direction sur au moins 3 messages consécutifs couvrant au moins 24 h (le contrat dit 30 min pour des données à 10 s ; un SYNOP toutes les 3 h, quantifié à 1 m/s et 10°, répète banalement deux ou trois messages par vent établi) ;
- doublons (station, instant) : le premier message est gardé.

Indicateurs **signalés, jamais exclus d'office** (décision d'Ines, station par station) :
- **calmes suspects** : part annuelle de calmes > 2 × la médiane du réseau cette année-là et > 25% ;
- **abri probable** : vitesse moyenne < 70% de celle des voisines à moins de 100 km (mêmes créneaux, 2015–2021) **et** < 70% de celle d'ERA5 10 m à la station (2021, fichiers locaux ; 2022 = test, jamais lu).

| Code | Nom | écartés (exclusion / panne / plage / figée) | mois écartés | calmes 2021 | années à calmes suspects | v / voisines | v / ERA5 2021 | abri probable |
|---|---|---|---|---|---|---|---|---|
| INM60710 | Tabarka | 0 / 0 / 0 / 0 | aucun | 4 % | — | 1.36 | 0.97 | non |
| INM60714 | Bizerte | 0 / 0 / 0 / 0 | aucun | 10 % | — | 1.24 | 0.90 | non |
| INM60715 | Tunis-Carthage | 0 / 0 / 0 / 0 | aucun | 4 % | — | 1.09 | 0.84 | non |
| INM60720 | Kelibia | 0 / 1150 / 0 / 0 | 2015-12 (panne (vent nul)), 2016-01 (panne (vent nul)), 2016-02 (panne (vent nul)), 2019-02 (panne (vent nul)), 2019-03 (panne (vent nul)) | 4 % | — | 0.64 | 0.47 | **oui** |
| INM60723 | Beja | 0 / 0 / 0 / 0 | aucun | 35 % | 2019, 2020, 2021, 2022 | 0.58 | 0.55 | **oui** |
| INM60725 | Jendouba | 0 / 272 / 0 / 0 | 2018-09 (panne (vent nul)), 2018-10 (panne (vent nul)), 2018-11 (panne (vent nul)) | 8 % | — | 0.94 | 1.03 | non |
| INM60728 | Nabeul | 0 / 1842 / 0 / 65 | 2017-04 (panne (vent nul)), 2017-05 (panne (vent nul)), 2017-06 (panne (vent nul)), 2017-07 (panne (vent nul)), 2017-08 (panne (vent nul)), 2017-09 (panne (vent nul)), 2017-10 (panne (vent nul)), 2017-11 (panne (vent nul)), 2021-06 (panne (vent nul)), 2021-07 (panne (vent nul)), 2021-09 (panne (vent nul)) | 16 % | 2015, 2017 | 1.03 | 0.79 | non |
| INM60729 | Zaghouan | 0 / 0 / 0 / 0 | aucun | 9 % | — | 0.90 | 1.01 | non |
| INM60731 | Enfidha | 0 / 0 / 0 / 0 | aucun | 1 % | — | 1.29 | 1.11 | non |
| INM60732 | El Kef | 0 / 0 / 0 / 18 | aucun | 12 % | — | 0.79 | 0.98 | non |
| INM60734 | Siliana | 0 / 0 / 0 / 3 | aucun | 23 % | — | 0.80 | 0.88 | non |
| INM60735 | Kairouan | 0 / 0 / 0 / 0 | aucun | 10 % | 2015, 2016, 2017 | 0.42 | 0.65 | **oui** |
| INM60738 | Thala | 0 / 0 / 0 / 0 | aucun | 2 % | 2016 | 1.55 | 2.16 | non |
| INM60739 | Kasserine | 0 / 929 / 0 / 0 | 2018-01 (panne (vent nul)), 2018-05 (panne (vent nul)), 2018-06 (panne (vent nul)), 2018-07 (panne (vent nul)), 2018-08 (panne (vent nul)), 2018-09 (panne (vent nul)), 2018-10 (panne (vent nul)), 2018-11 (panne (vent nul)), 2018-12 (panne (vent nul)), 2019-01 (panne (vent nul)), 2019-07 (panne (vent nul)), 2019-08 (panne (vent nul)) | 1 % | 2018, 2019 | 0.99 | 1.33 | non |
| INM60740 | Monastir-Skanes | 0 / 0 / 0 / 0 | aucun | 2 % | — | 1.31 | 1.18 | non |
| INM60742 | Mahdia | 0 / 0 / 0 / 0 | aucun | 1 % | — | 0.90 | 0.90 | non |
| INM60745 | Gafsa | 0 / 0 / 0 / 0 | aucun | 1 % | — | 1.07 | 1.36 | non |
| INM60748 | Sidi Bouzid | 0 / 636 / 0 / 0 | 2017-08 (panne (vent nul)), 2019-06 (panne (vent nul)), 2019-07 (panne (vent nul)), 2019-08 (panne (vent nul)) | 2 % | — | 0.74 | 0.98 | non |
| INM60750 | Sfax El-Maou | 0 / 0 / 0 / 0 | aucun | 3 % | — | 0.90 | 0.77 | non |
| INM60760 | Tozeur | 0 / 0 / 0 / 0 | aucun | 1 % | — | 1.19 | 1.36 | non |
| INM60764 | Kebili | 0 / 0 / 0 / 8 | aucun | 18 % | — | 0.89 | 1.04 | non |
| INM60765 | Gabes | 0 / 0 / 0 / 0 | aucun | 4 % | — | 0.93 | 0.39 | non |
| INM60767 | Gabes Airport | 0 / 0 / 0 / 0 | aucun | 1 % | — | 1.23 | 1.11 | non |
| INM60769 | Djerba Mellita | 0 / 0 / 0 / 0 | aucun | 1 % | — | 1.32 | 0.97 | non |
| INM60770 | Medenine | 0 / 0 / 0 / 0 | aucun | 42 % | 2021, 2022 | 0.57 | 0.48 | **oui** |
| INM60772 | Tataouine | 0 / 0 / 0 / 0 | aucun | 6 % | — | 1.20 | 0.76 | non |
| INM60775 | Remada | 0 / 0 / 0 / 0 | aucun | 5 % | — | 0.90 | 0.64 | non |
| INM60780 | El Borma | 0 / 0 / 0 / 0 | aucun | 8 % | — | — | 0.95 | non |
| ISD60740399999 | Habib Bourguiba Intl | 0 / 0 / 0 / 0 | aucun | 4 % | — | 1.32 | 1.18 | non |

Détail par station et par mois : `stations_supplementaires_qc.csv`.

## Remarques

- INM60723 Beja : coordonnées ISD et OGIMET distantes de 4 km (ISD 36.733, 9.183 ; OGIMET 36.736, 9.226) ; les coordonnées OGIMET (WIGOS) sont retenues.
- INM60731 Enfidha : coordonnées ISD et OGIMET distantes de 2 km (ISD 36.083, 10.433 ; OGIMET 36.067, 10.450) ; les coordonnées OGIMET (WIGOS) sont retenues.
- INM60732 El Kef : coordonnées ISD et OGIMET distantes de 2 km (ISD 36.133, 8.700 ; OGIMET 36.123, 8.724) ; les coordonnées OGIMET (WIGOS) sont retenues.
- INM60734 Siliana : coordonnées ISD et OGIMET distantes de 4 km (ISD 36.067, 9.367 ; OGIMET 36.054, 9.404) ; les coordonnées OGIMET (WIGOS) sont retenues.
- INM60739 Kasserine : coordonnées ISD et OGIMET distantes de 4 km (ISD 35.167, 8.833 ; OGIMET 35.168, 8.791) ; les coordonnées OGIMET (WIGOS) sont retenues.
- INM60740 Monastir-Skanes : coordonnées ISD et OGIMET distantes de 11 km (ISD 35.667, 10.750 ; OGIMET 35.762, 10.758) ; les coordonnées OGIMET (WIGOS) sont retenues.
- INM60760 Tozeur : coordonnées ISD et OGIMET distantes de 6 km (ISD 33.940, 8.111 ; OGIMET 33.917, 8.167) ; les coordonnées OGIMET (WIGOS) sont retenues.
- INM60765 Gabes : coordonnées ISD et OGIMET distantes de 24 km (ISD 33.734, 9.919 ; OGIMET 33.883, 10.107) ; les coordonnées OGIMET (WIGOS) sont retenues.
- INM60770 Medenine : coordonnées ISD et OGIMET distantes de 2 km (ISD 33.350, 10.483 ; OGIMET 33.329, 10.488) ; les coordonnées OGIMET (WIGOS) sont retenues.
- INM60772 Tataouine : coordonnées ISD et OGIMET distantes de 5 km (ISD 32.917, 10.450 ; OGIMET 32.958, 10.467) ; les coordonnées OGIMET (WIGOS) sont retenues.
- INM60775 Remada : coordonnées ISD et OGIMET distantes de 2 km (ISD 32.306, 10.382 ; OGIMET 32.317, 10.400) ; les coordonnées OGIMET (WIGOS) sont retenues.
- INM60780 El Borma : coordonnées ISD et OGIMET distantes de 5 km (ISD 31.704, 9.255 ; OGIMET 31.683, 9.300) ; les coordonnées OGIMET (WIGOS) sont retenues.
- INM60728 : coordonnées 36.45, 10.7333 retenues (catalogue INM (grille_reelle.STATIONS) : 9 m pour 1 m officiel ; OGIMET et ISD (36,467 N 10,700 E) donnent 45 a 55 m, douteux).
- INM60710 Tabarka : altitude de la maille 65 m pour 20 m officiels (position ou altitude officielle à vérifier).
- INM60760 Tozeur : altitude de la maille 28 m pour 86 m officiels (position ou altitude officielle à vérifier).

## Utilisation

```bash
python scripts/stations_supplementaires.py --dossier artifacts/national/obs_supp --telecharger --descripteurs
python scripts/valider_grand_tunis_reel.py -c <config nationale> --obs <obs> --jeu test \
    --stations-supp configs/national/stations_supplementaires.csv --obs-supp <dossier des synop_*.csv>
# run deja valide (bilan.json des aeroports present) : ajouter --supp-seul, rien d'autre n'est reecrit
```
