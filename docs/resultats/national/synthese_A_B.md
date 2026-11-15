# Synthèse finale A (OS4) et B (OS5) — produit national ATOM Wind

État au 2026-11-15, 14:10 (heure de Tunis). Dépôt `ines-rouatbi/pcfno3d`, branche `agent/ines/os7-national`.
Ce document ne contient que des chiffres écrits par les calculs ; ce qui n'a pas tourné est marqué
« non exécuté ».

## Le modèle servi (rappel, livré le 15/11)

National v4 **C' s44**, w servi = w_continuité + 0,75 (w_réseau − w_continuité)
(`regle_choix_v4_Cp_s44` `19cee4e5…`, `regle_w_servi_v4` `4e545345…`). Test 2022 lu une seule fois
(commit `35728f9`, `v4_test_2022.md`) :

| Critère | Valeur | Verdict |
|---|---|---|
| DSO1 RMSE max par niveau | 0,285 m/s | conforme |
| DSO1 R² min | 0,952 (w, 2000 m) | conforme |
| DSO2 résidu de masse | 2,4e-5 | conforme |
| DSO3 rugosité | 0,995 | conforme |
| DSO4 temps (A100, carnet de test) | 264 ms | conforme |
| DSO5 (capteurs retirés) | 0,286 | conforme |
| DSO6 couverture 80 % | 0,86 | conforme |
| 0–1 km contre la vérité générée | 0,235 m/s (ERA5 0,299) | meilleur que l'ERA5 |
| 12 aéroports réels, vent 10 m | 2,844 m/s (ERA5 2,857) | au niveau de l'ERA5 |

Produit 10 m servi (option i d'Ines, `regle_produit_vent10m_v4` `f07973e1…`) : ERA5 + OI des
mesures d'aéroport, jugé sur 2023–2024 jamais utilisées : **2,639 contre 2,739 m/s pour l'ERA5
(−3,7 %, IC [−0,108 ; −0,092]), meilleur à 9 aéroports sur 12** (commit `32923d2`).
La vérité générée et l'ERA5 ne sont pas des mesures ; seuls les aéroports le sont.

## A — OS4 : dimensionner le réseau ANEO (OI 3D, CPU)

| Étape | Commit | Contenu |
|---|---|---|
| A1 | `b50dbf2` | règle datée `regle_os4_oi_v4_2` (sha `6f2f4eee159541a8…`), approuvée par Ines |
| A2 | `348bdf3` (+ carnet `42b4f7a`) | code, tests, essai sur 6 instants de validation (aucun paramètre changé ensuite) |
| A3 | `ba022fc` | lecture unique des 180 instants de test 2022 → `os4_oi.md` |
| A4 | `ba022fc` | −10 % non atteint à N = 400 : publié tel quel, règle inchangée |

Résultat (pire tranche 0–1 km, sans réseau 1,807 m/s) : **N minimal non atteint jusqu'à 400 pour
toutes les stratégies**. Meilleur gain : grille régulière N = 400, +3,2 % (IC 95 % 2,9–3,7).
Près du sol, le réseau aide : grille régulière −12 % (0–100 m) et −14 % (10 m) dès 50 capteurs,
−21 % et −24 % à 400. Le glouton dégrade la pire tranche jusqu'à 100 capteurs. Fichiers :
`MyDrive/pcfno3d_national/os4_oi_v4_2/` (`courbe.csv/.png`, `n_minimal.json`,
`plan_glouton.geojson`, `bilan.json/.md`).

## B — OS5 : servir le national à côté du Grand Tunis

| Étape | Commit | Contenu | État |
|---|---|---|---|
| B1 | `31ea249` | plusieurs domaines dans un service, choix du plus petit domaine contenant la position (`atom_outils.modele_pour`), `/atom-wind/domaines`, `/atom-wind/point` | fait, testé (`tests/test_domaines_os5.py`) |
| B2 | `ff86ed8`, `31ea249` | couche 10 m ERA5 + OI servie ; chaque réponse indique `domaine` et `source_vent` | fait, testé |
| B3 | `9019f8d` | MCP (`prevision_point` par position, outil `domaines_servis`, 23 outils), `INTEGRATION.md`, OpenAPI | fait, testé |
| B4 | `f541553`, `fd48cc1` | vue nationale (`?domaine=national`, bascule Grand Tunis / Tunisie), onglet Réseau ANEO lit le plan OS4 | fait, testé |
| B5 | carnet `668c9c0` | DSO4 sur le GPU du service (médiane de 20 passes, p95, mémoire max) | **non exécuté** (attend le go d'Ines) |
| B6 | — | dernier état connu conservé (cache du Worker, aucun changement nécessaire) | inchangé |
| B7 | carnet `668c9c0` | démo Sfax, Gafsa, Tozeur, Gabès + Tunis, captures dans `docs/demo/` | **non exécuté** (attend le go d'Ines) |

Le carnet `pcfno3d_atom_deux_domaines_colab.ipynb` (L4, ~2 unités/h, < 10 unités) écrit
`NAT/os5/dso4_service.json` et `NAT/os5/demo_points.json` ; ces chiffres compléteront ce tableau.
