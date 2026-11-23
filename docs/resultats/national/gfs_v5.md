# Champ 3D national nourri par GFS : règles v5.1 puis v5.2 (22-23/11/2026)

Objet : rendre valides, par des corrections étudiées et jugées honnêtement, les trois cases refusées le 18/11 par
`regle_national_gfs_v4` (champ 3D national nourri par GFS à l'analyse, à +24 h et à +48 h ; Tabarka dégradé de
13 à 14 %). Résultat : **les trois cases restent refusées**, à cause de Tabarka seul (+6,0 à +6,3 % contre GFS brut,
limite +5 %), alors que le candidat bat GFS brut d'environ 10 % en moyenne et aux 11 autres aéroports.

## Chronologie

1. **Règle v5** (`regle_national_gfs_v5.json`, 519ebd64…) : approuvée le 22/11, jamais exécutée (contrôle d'entrée
   de l'étape 4 impossible à satisfaire, budget). Gardée telle quelle.
2. **Règle v5.1** (`regle_national_gfs_v5_1.json`, 57071384…) : approuvée le 22/11 au soir.
   - Partie A (Colab CPU) : GFS f0 / f24 / f48 téléchargés ; entraînement + validation : 3, 9 et 4 échéances absentes
     sur 820 instants ; échantillon 2026 complet.
   - Partie B1 (Colab A100, 23/11 06:46-08:04) : affinage C1 (`runs/national_1km_v5_gfs`), 12 époques, époque 2
     retenue par le critère DSO sur 2021 (score 4,278 contre 4,298 au départ).
   - **Arrêt sans verdict** à l'étape 3, en 6 s, avant tout calcul aux aéroports : 111 instants de validation sur 120
     (9 sans échéance GFS : le run du 2020-12-31 n'existe pas dans l'archive NOAA AWS, qui commence le 2021-01-01,
     plus des trous de l'archive le 2021-02-03 et le 2021-02-07). Le contrôle de 2 % avait été appliqué aux seuls
     instants de validation au lieu du téléchargement : erreur de rédaction de l'agent, non vue avant l'accord.
     Sortie gardée : `MyDrive/pcfno3d_national/gfs_v5_regle/validation_2021/arret.json`.
3. **Règle v5.2** (`regle_national_gfs_v5_2.json`, 542e676c…) : approuvée le 23/11, AVANT tout score aux aéroports.
   Seul changement : 2 % au plus d'échéances absentes au téléchargement ; instants incomplets retirés pour toutes les
   sources et publiés. C1 de la v5.1 réutilisé tel quel (vérifié : marqueur de la v5.1, époque 2).

## Choix sur la validation 2021 (v5.2, carnet C, Colab L4, 23/11 09:30-10:35)

111 instants sur 120 (9 retirés, liste ci-dessus) ; 12 aéroports, chacun retiré de l'entrée quand il est jugé.

| Variante | analyse | +24 h | +48 h | RMSE moyen (3 cases) | passe les trois |
|---|---|---|---|---:|---|
| c1 | refusé | accepté | accepté | 2,916 | non |
| c2 a0 0,25 H 100 | accepté | accepté | accepté | 2,969 | oui |
| c2 a0 0,25 H 300 | accepté | accepté | accepté | 2,966 | oui |
| c2 a0 0,5 H 100 | refusé | refusé | accepté | 2,973 | non |
| c2 a0 0,5 H 300 | refusé | accepté | accepté | 2,971 | non |
| c2 a0 0,75 H 100 | refusé | refusé | refusé | 2,981 | non |
| c2 a0 0,75 H 300 | refusé | refusé | refusé | 2,979 | non |
| c3 a0 0,25 H 100 | accepté | accepté | accepté | 2,936 | oui |
| c3 a0 0,25 H 300 | accepté | accepté | accepté | 2,942 | oui |
| c3 a0 0,5 H 100 | accepté | accepté | accepté | 2,927 | oui |
| c3 a0 0,5 H 300 | accepté | accepté | accepté | 2,930 | oui |
| **c3 a0 0,75 H 100** | accepté | accepté | accepté | **2,920** | **oui — retenue** |
| c3 a0 0,75 H 300 | accepté | accepté | accepté | 2,921 | oui |

Diagnostic de Tabarka (C' s44 + GFS, analyse, 2021) : 3,179 contre 2,847 m/s pour GFS brut (+11,7 %) ; biais de
vitesse +0,36 m/s (GFS −0,13) ; erreur de direction 43° (GFS 39°) ; plus mauvais dans tous les secteurs de vent, de
jour comme de nuit, été comme hiver. Détails : `validation_2021_v5_2/validation_gfs_v5.md` sur le Drive.

## Jugement 2026 (v5.2, carnet B2, Colab L4, 23/11 11:34-17:18)

Contrôle préalable passé (parties vent de l'entrée de production identiques au chemin du cache : écart max
9,8e-6 ≤ 1e-5 ; canaux de surface différents, attendus, rapportés). METAR 2026 (archive IEM) : 105 626 mesures.
Sortie du carnet, recopiée telle quelle :

# Règle `regle_national_gfs_v5_2` : jugement 2026 aux 12 aéroports

Règle `configs/national/regle_national_gfs_v5_2.json` (sha256 `542e676c90da…`), approuvée par Ines le 23/11/2026, vérifiée avant le calcul. Candidat retenu sur la validation 2021 seulement : `c3_a0.75_H100`. Échantillon fixé par la règle : 2026-01-01 → 2026-10-31, jours impairs, 00 / 06 / 12 / 18 UTC (analyse toujours f000 ; les heures 03 / 09 / 15 / 21 du service ne sont pas jugées), jamais utilisée, **lue une seule fois**. Entrée construite comme le service temps réel (archive GFS NOAA, METAR IEM des 11 autres aéroports à l'analyse, aucun aux échéances). Les seules mesures sont les METAR.

Instants : 608 sur 608 (0 écartés pour toutes les sources faute d'échéance GFS) ; 152 jours ; IC 95 % par 2000 tirages des jours (graine 2030).

| Case | Moyenne RMSE candidat | Moyenne RMSE GFS brut | Écart | IC 95 % | Aéroports > +5 % | Verdict |
|---|---:|---:|---:|---|---|---|
| national 3D / analyse | 2.958 | 3.276 | -0.318 | [-0.352 ; -0.286] | Tabarka | **refuse** |
| national 3D / +24 h | 3.125 | 3.473 | -0.348 | [-0.386 ; -0.312] | Tabarka | **refuse** |
| national 3D / +48 h | 3.261 | 3.610 | -0.349 | [-0.390 ; -0.312] | Tabarka | **refuse** |

Rapporté sans verdict : C' s44 + GFS sans correction.

| Case | Moyenne RMSE C' s44 + GFS | Écart à GFS brut | IC 95 % |
|---|---:|---:|---|
| national 3D / analyse | 3.060 | -0.216 | [-0.240 ; -0.193] |
| national 3D / +24 h | 3.239 | -0.233 | [-0.262 ; -0.207] |
| national 3D / +48 h | 3.379 | -0.232 | [-0.262 ; -0.203] |

## Par aéroport (m/s)

| echeance   | station          |   candidat |   cp_s44_gfs |   gfs_brut |
|:-----------|:-----------------|-----------:|-------------:|-----------:|
| +24 h      | Bizerte          |      3.027 |        3.086 |      3.524 |
| +24 h      | Djerba-Zarzis    |      3.026 |        3.029 |      3.375 |
| +24 h      | Enfidha-Hammamet |      3.45  |        3.55  |      3.669 |
| +24 h      | Gabes            |      3.042 |        3.114 |      3.354 |
| +24 h      | Gafsa            |      3.872 |        3.951 |      3.915 |
| +24 h      | Kairouan         |      3.447 |        3.746 |      4.045 |
| +24 h      | Monastir         |      3.002 |        3.033 |      3.284 |
| +24 h      | Remada           |      2.201 |        2.385 |      3.208 |
| +24 h      | Sfax             |      3.142 |        3.272 |      3.77  |
| +24 h      | Tabarka          |      3.437 |        3.799 |      3.24  |
| +24 h      | Tozeur           |      3.086 |        3.105 |      3.185 |
| +24 h      | Tunis-Carthage   |      2.767 |        2.803 |      3.105 |
| +48 h      | Bizerte          |      3.251 |        3.308 |      3.726 |
| +48 h      | Djerba-Zarzis    |      3.181 |        3.178 |      3.533 |
| +48 h      | Enfidha-Hammamet |      3.684 |        3.787 |      3.918 |
| +48 h      | Gabes            |      3.16  |        3.236 |      3.447 |
| +48 h      | Gafsa            |      3.974 |        4.06  |      4.04  |
| +48 h      | Kairouan         |      3.425 |        3.719 |      4     |
| +48 h      | Monastir         |      3.373 |        3.408 |      3.698 |
| +48 h      | Remada           |      2.308 |        2.498 |      3.324 |
| +48 h      | Sfax             |      3.153 |        3.279 |      3.765 |
| +48 h      | Tabarka          |      3.547 |        3.888 |      3.345 |
| +48 h      | Tozeur           |      3.201 |        3.273 |      3.316 |
| +48 h      | Tunis-Carthage   |      2.874 |        2.908 |      3.212 |
| analyse    | Bizerte          |      2.846 |        2.91  |      3.343 |
| analyse    | Djerba-Zarzis    |      2.663 |        2.654 |      2.935 |
| analyse    | Enfidha-Hammamet |      3.274 |        3.363 |      3.472 |
| analyse    | Gabes            |      2.858 |        2.921 |      3.111 |
| analyse    | Gafsa            |      3.847 |        3.913 |      3.872 |
| analyse    | Kairouan         |      3.302 |        3.593 |      3.895 |
| analyse    | Monastir         |      2.9   |        2.938 |      3.159 |
| analyse    | Remada           |      2.076 |        2.241 |      3.002 |
| analyse    | Sfax             |      2.897 |        3.021 |      3.477 |
| analyse    | Tabarka          |      3.286 |        3.609 |      3.092 |
| analyse    | Tozeur           |      3.013 |        2.992 |      3.097 |
| analyse    | Tunis-Carthage   |      2.533 |        2.566 |      2.858 |

## Lecture

- Les trois cases sont **refusées** par la règle (Tabarka +6,3 / +6,1 / +6,0 % contre GFS brut, limite +5 %, soit
  3,247 m/s à l'analyse pour 3,286). Le seuil n'est pas changé après lecture ; 2026 ne sera pas relu.
- En moyenne sur les 12 aéroports, le candidat bat GFS brut d'environ 10 % aux trois échéances (IC 95 % entièrement
  négatifs) et fait mieux que C' s44 + GFS sans correction (−0,32 à −0,35 contre −0,22 à −0,23 m/s).
- Il est meilleur que GFS brut aux 11 autres aéroports, aux trois échéances ; à Tabarka, la correction ramène
  l'excès de +17 % (C' s44) à +6 %, sans passer sous 5 %. L'ancrage retenu sur 2021 (a0 = 0,75 : 25 % de GFS à
  10 m) est le plus léger des variantes qui passaient ; un ancrage plus fort aurait davantage rapproché Tabarka de
  GFS, au prix d'une erreur moyenne un peu plus haute sur 2021.
- Conséquence (règle, « consequences ») : statut « expérimental » gardé pour ces trois cases ; aucun fichier de
  verdicts n'est à renseigner dans `PCFNO3D_VERDICTS_GFS_V5`. Servir le candidat à la place de C' s44 + GFS reste
  une décision d'Ines.
- Coût Colab : 22,8 unités sur le plafond de 90 (solde 120,92 → 98,13 : parties A, B1, C et B2).
- GFS est une prévision et ERA5 une réanalyse ; les seules mesures sont les METAR.
