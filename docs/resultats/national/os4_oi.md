# OS4 national — dimensionner le réseau ANEO par interpolation optimale 3D (test 2022, lecture unique)

Date du calcul : 2026-11-15, 11:50 → 13:59 (heure de Tunis), Colab CPU High-RAM, carnet
`notebooks/current/pcfno3d_national_os4_oi_v4_colab.ipynb` épinglé sur le commit `42b4f7a`.
La session Colab s'est terminée seule à la fin (`runtime.unassign()`, vérifié : « Reconnecter », 0 session).

Règle : `configs/national/regle_os4_oi_v4_2.json`, sha256
`6f2f4eee159541a87325b7f00c44b1a817c9c2fd14f80e4b75f2dace57f6024a`, approuvée par Ines avant le calcul
(« règle A OK »). Aucun paramètre n'a changé entre l'essai du code (6 instants de validation) et la
lecture du test. **Le test (180 instants synoptiques 2022) a été lu une seule fois.**

## Nature de l'expérience

Simulation OSSE, **distincte du modèle C'** : le fond est l'ERA5 dégradée (σu 1,91, σv 2,02 m/s,
L 55 km), les capteurs ANEO sont simulés à 25 m (vérité générée + bruit 0,5 m/s), l'analyse est
fond + OI 3D (L 150 km, λ 4, décroissance verticale exp(−|dz|/300 m)). La vérité est la vérité
générée du jeu national : ce ne sont **pas des mesures**. Candidats : 220 557 cellules (critères
de la règle : hors chotts, ≥ 8 km des aéroports METAR). Placements : glouton sur la variance
d'erreur du fond (jours d'entraînement), grille régulière (point le plus éloigné), aléatoire × 3.
Critère fixé avant le calcul : N minimal = plus petit N dont la **borne basse de l'IC 95 %** du
gain sur la pire tranche 0–1 km au-dessus du sol est ≥ 10 % (bootstrap sur les jours, 2000 tirages,
graine 2030).

## Résultat

Sans réseau : pire tranche 0–1 km **1,807 m/s**, tranche 0–100 m 1,802 m/s, vent 10 m 2,083 m/s.

**N minimal : non atteint jusqu'à N = 400, pour toutes les stratégies** (au sens IC 95 % comme au
sens ponctuel). Conformément à A4, le résultat est publié tel quel, sans changer la règle.

| Stratégie | N | Pire tranche 0–1 km (m/s) | Gain | IC 95 % | Gain 0–100 m | Gain 10 m |
|---|---:|---:|---:|---|---:|---:|
| sans réseau | 0 | 1,8071 | — | — | — | — |
| glouton | 10 | 1,9147 | −5,95 % | [−11,99 ; −0,60] | −6,28 % | −1,16 % |
| glouton | 25 | 1,8546 | −2,63 % | [−9,12 ; +1,75] | −2,94 % | +2,52 % |
| glouton | 50 | 1,8519 | −2,48 % | [−9,51 ; +2,18] | −2,79 % | +4,72 % |
| glouton | 100 | 1,8092 | −0,11 % | [−6,56 ; +2,71] | −0,42 % | +7,60 % |
| glouton | 200 | 1,7574 | +2,75 % | [+1,98 ; +3,29] | +6,61 % | +10,55 % |
| glouton | 400 | 1,7516 | +3,07 % | [+2,74 ; +3,59] | +11,48 % | +12,50 % |
| grille régulière | 10 | 1,7908 | +0,90 % | [+0,61 ; +1,39] | +4,00 % | +4,33 % |
| grille régulière | 25 | 1,7810 | +1,44 % | [+1,12 ; +1,94] | +8,62 % | +9,30 % |
| grille régulière | 50 | 1,7727 | +1,90 % | [+1,59 ; +2,40] | +12,10 % | +13,86 % |
| grille régulière | 100 | 1,7640 | +2,39 % | [+2,06 ; +2,89] | +15,85 % | +17,81 % |
| grille régulière | 200 | 1,7558 | +2,84 % | [+2,52 ; +3,35] | +18,53 % | +21,43 % |
| grille régulière | 400 | 1,7488 | +3,23 % | [+2,91 ; +3,74] | +20,75 % | +24,28 % |
| aléatoire 1 | 10 | 1,7892 | +0,99 % | [+0,68 ; +1,47] | +4,48 % | +5,08 % |
| aléatoire 1 | 25 | 1,7819 | +1,40 % | [+1,09 ; +1,89] | +7,60 % | +8,79 % |
| aléatoire 1 | 50 | 1,7747 | +1,80 % | [+1,48 ; +2,30] | +10,02 % | +12,48 % |
| aléatoire 1 | 100 | 1,7657 | +2,30 % | [+1,97 ; +2,80] | +13,54 % | +16,72 % |
| aléatoire 1 | 200 | 1,7579 | +2,72 % | [+2,40 ; +3,24] | +15,91 % | +20,42 % |
| aléatoire 1 | 400 | 1,7501 | +3,15 % | [+2,83 ; +3,66] | +18,42 % | +23,55 % |
| aléatoire 2 | 10 | 1,7919 | +0,84 % | [+0,54 ; +1,32] | +2,66 % | +4,43 % |
| aléatoire 2 | 25 | 1,7819 | +1,40 % | [+1,09 ; +1,89] | +7,04 % | +9,11 % |
| aléatoire 2 | 50 | 1,7746 | +1,80 % | [+1,48 ; +2,30] | +9,76 % | +12,97 % |
| aléatoire 2 | 100 | 1,7663 | +2,26 % | [+1,94 ; +2,76] | +12,99 % | +17,08 % |
| aléatoire 2 | 200 | 1,7587 | +2,68 % | [+2,36 ; +3,19] | +15,69 % | +20,64 % |
| aléatoire 2 | 400 | 1,7503 | +3,15 % | [+2,82 ; +3,66] | +18,71 % | +23,81 % |
| aléatoire 3 | 10 | 1,7897 | +0,97 % | [+0,67 ; +1,47] | +4,20 % | +5,37 % |
| aléatoire 3 | 25 | 1,7813 | +1,43 % | [+1,13 ; +1,93] | +7,42 % | +9,49 % |
| aléatoire 3 | 50 | 1,7734 | +1,86 % | [+1,56 ; +2,36] | +10,16 % | +13,03 % |
| aléatoire 3 | 100 | 1,7655 | +2,31 % | [+2,00 ; +2,79] | +12,87 % | +17,03 % |
| aléatoire 3 | 200 | 1,7574 | +2,75 % | [+2,45 ; +3,24] | +15,53 % | +20,62 % |
| aléatoire 3 | 400 | 1,7502 | +3,15 % | [+2,84 ; +3,64] | +17,62 % | +23,78 % |

## Lecture

- **Critère de la règle (pire tranche 0–1 km, −10 %) : non atteint.** Le meilleur gain à N = 400
  est +3,2 % (grille régulière, IC 95 % [+2,9 ; +3,7]). Des capteurs à 25 m, propagés par l'OI
  avec H = 300 m, corrigent peu les tranches hautes (300–1000 m), qui portent la pire tranche.
- **Près du sol, le réseau aide nettement** (lecture descriptive, hors critère de la règle) :
  la grille régulière réduit l'erreur de la tranche 0–100 m de 12 % dès 50 capteurs et de 21 %
  à 400, et celle du vent à 10 m de 14 % à 50 capteurs et 24 % à 400.
- **La grille régulière est la meilleure stratégie à chaque N** ; l'aléatoire la suit de près.
- **Le glouton est la plus mauvaise stratégie** et **dégrade** la pire tranche jusqu'à 100 capteurs
  (−6,0 % à N = 10). Hypothèse, non testée : il place les premiers capteurs aux sites de plus forte
  variance d'erreur du fond, où le modèle de corrélation isotrope (L 150 km) de l'OI étale des
  incréments locaux trop loin. La règle étant figée, aucun paramètre n'a été réglé après coup.

## Fichiers (Drive, `MyDrive/pcfno3d_national/os4_oi_v4_2/`)

`courbe.csv`, `courbe.png`, `n_minimal.json`, `plan_glouton.geojson` (id, priorité, phase 1–6,
lat, lon, gain marginal, hauteur, source), `bilan.json`, `bilan.md` ; journal
`MyDrive/pcfno3d_national/journaux/os4_oi_v4_test.log`. Essai du code (validation) dans
`os4_oi_v4_2_essai/`.

L'onglet « Réseau ANEO » de la page ATOM Wind lit ce plan (`atom_wind._plan_os6`,
`reseau_aneo` avec `?domaine=national`) et affiche la courbe avec le seuil −10 % ; aucun nombre
recommandé n'est affiché puisque le critère n'est pas atteint.
