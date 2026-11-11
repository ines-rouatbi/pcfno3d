# Contrôle du branchement des capteurs — modèle national (C' et D graine 42), 11/11/2026

Demande d'Ines du 11/11 : test absurde, chemin des capteurs, lots d'entraînement de D, **avant de lancer D graine 44**.
**Aucun entraînement, aucune lecture du test.** Carnet `notebooks/current/pcfno3d_controle_capteurs_national_colab.ipynb`
(commit `8331b6d`, code figé `77a058b`), Colab A100, 861 s. Valeurs de **validation** (vérité d'entraînement v3,
jamais appelée « mesures »).

Sorties sur le Drive : `MyDrive/pcfno3d_national/runs/national_1km_v4_{Cp,D}_s42/controle_capteurs.json`,
journaux `MyDrive/pcfno3d_national/journaux/controle_capteurs_{Cp,D}_s42.log`.

## 1. Chemin des capteurs (3 instants de validation, 50 capteurs simulés)

| Contrôle | C' s42 | D s42 |
|---|---|---|
| Capteurs dans le masque | 50 / 50 (masque 80 325 → 80 363 à 80 366 ; quelques capteurs tombent sur une maille déjà masquée) | idem |
| Écart entre la valeur écrite dans S et la valeur du capteur | 0,0 | 0,0 |
| Masque de l'entrée du réseau aux capteurs (minimum) | 1,0 | 1,0 |
| Écart de la valeur normalisée dans l'entrée du réseau | 0,0 | 0,0 |
| Chemin de l'OS6 contre chemin de l'évaluation | chiffres identiques | chiffres identiques |

## 2. Test absurde : +15 m/s sur u aux 50 capteurs — max |Δ| en sortie (m/s)

| Instant | C' : u < 5 km / ailleurs ; v < 5 km | D : u < 5 km / ailleurs ; v < 5 km |
|---|---|---|
| 0001_000000 | 0,323 / 0,034 ; 0,375 | 0,536 / 0,049 ; 0,484 |
| 0001_060000 | 0,124 / 0,033 ; 0,166 | 0,530 / 0,044 ; 0,305 |
| 0001_180000 | 0,218 / 0,040 ; 0,272 | 0,500 / 0,059 ; 0,353 |

Témoin, +15 m/s aux aéroports : C' 0,14 à 0,18 m/s, D 0,34 à 0,38 m/s à moins de 5 km.
Ajout des 50 capteurs à leur valeur vraie : C' 0,04 à 0,08 m/s (u), D 0,10 à 0,15 m/s (u) à moins de 5 km.

## 3. Lots d'entraînement de D (mêmes réglages que l'entraînement)

| Lot | Capteurs tirés | Cellules capteurs dans l'entrée du réseau | Masque total |
|---|---|---|---|
| 1 | 161 | 161 | 70 361 |
| 2 | 76 | 76 | 70 894 |
| 3 | 155 | 155 | 80 445 |
| 4 | 129 | 129 | 68 589 |
| 5 | 96 | 96 | 80 396 |

## Conclusion

**Pas de bug de branchement** sur le modèle national : les capteurs entrent dans le masque et dans le tenseur
d'entrée avec leur valeur exacte, par les deux chemins, et D en voit 76 à 161 par lot d'entraînement. La sortie
répond localement (< 5 km) mais faiblement : +15 m/s donnent 0,12 à 0,32 m/s pour C' (≈ 1 à 2 %) et 0,50 à
0,54 m/s pour D (≈ 3,5 %), moins de 0,06 m/s ailleurs. D répond environ deux fois plus que C', sans exploiter
vraiment les capteurs : même constat qu'au Grand Tunis, qui motive D2.

Conformément à la consigne d'Ines, **D graine 44 peut être lancé** (le contrôle ne trouve pas de bug).
