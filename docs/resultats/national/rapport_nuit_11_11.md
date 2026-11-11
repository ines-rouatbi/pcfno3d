# Rapport de la nuit du 11/11 (accord écrit d'Ines donné à l'avance, cadre fixé par elle)

Tenu au fil de la nuit ; état final complété le matin. Toutes les valeurs de validation sont sur la vérité
d'entraînement v3 (générateur forcé par ERA5) — jamais appelée « mesures » ; les METAR et radiosondages sont les
mesures. Ligne « ERA5 seule » à côté de chaque résultat quand elle existe.

## 1. Contrôle du branchement des capteurs (demande d'Ines du 11/11)

Script `scripts/controle_capteurs.py` (commit `3f2e442`), carnets `pcfno3d_controle_capteurs_*_colab.ipynb`
(`8331b6d`), test unitaire `tests/test_forcage_era5.py::test_capteurs_arrivent_au_reseau`.

### Grand Tunis v4 (modèle livré, chemin de l'OS6) — 3 instants de validation, Colab CPU

| Instant | Masque avant → après 50 capteurs | Ajout des 50 capteurs (vraie valeur) : max \|Δu\| < 5 km | +15 m/s sur u aux 50 capteurs : max \|Δu\| < 5 km / ailleurs, max \|Δv\| < 5 km | +15 m/s aux 3 stations : max \|Δu\| < 5 km | Chemin OS6 (même test) | Masque seul ajouté (chemin OS6 Grand Tunis) |
|---|---|---|---|---|---|---|
| 0001_000000 | 597 → 647 | 0,134 | 0,218 / 0,007, 0,179 | 0,032 | 0,218 | 0,108 |
| 0001_060000 | 597 → 647 | 0,195 | 0,259 / 0,007, 0,151 | 0,030 | 0,259 | 0,108 |
| 0001_180000 | 597 → 647 | 0,194 | 0,199 / 0,009, 0,127 | 0,031 | 0,199 | 0,163 |

(m/s.) Chemin : les 50 capteurs sont dans le masque (+50 cellules), leur valeur est écrite exactement dans S
(écart 0,0), le tenseur d'entrée du réseau a le masque à 1 et la bonne valeur normalisée à ces cellules (écart 0,0).

**Conclusion Grand Tunis : pas de bug de branchement.** Les capteurs arrivent au réseau ; la sortie répond
localement (à moins de 5 km) mais très faiblement : +15 m/s en entrée → 0,20 à 0,26 m/s en sortie (≈ 1,5 %), moins
de 0,01 m/s ailleurs ; les chemins de l'évaluation et de l'OS6 donnent exactement les mêmes chiffres. Le modèle a
appris à presque ignorer les mesures ponctuelles (il se fie à l'ERA5 dense) : c'est ce qui rend la courbe de l'OS6
du Grand Tunis plate au 4e chiffre (0,5326), pas un défaut de code. Rien à refaire côté OS6 pour un bug ; un modèle
qui exploite les capteurs (D2) est nécessaire.

### National (C' et D graine 42) — à venir (carnet `pcfno3d_controle_capteurs_national_colab.ipynb`)

## 2. Entraînements v4 (règle `60671e84…`)

Voir `docs/resultats/national/v4_entrainements_validation.md` (graines 42) ; à compléter.

## 3. Ce qui reste à décider (au fil de la nuit)

- (à compléter)
