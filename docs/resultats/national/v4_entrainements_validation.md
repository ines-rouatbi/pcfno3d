# v4 nationale — entraînements (validation seulement, règle `60671e84…`)

Lu sur le Drive (`runs/national_1km_v4_<cand>_s<g>/history.json`, `journaux/*.log`). Chiffres de **validation**
(vérité d'entraînement v3, 64 instants), pas des verdicts DSO : le test n'est lu qu'avec l'accord séparé d'Ines.
Départ commun (modèle livré v2 sur la validation v3) : score DSO 3,695, RMSE max 1,848 m/s à 650 m ASL.

## Score de sélection (`val_score`, niveaux ASL) et époque retenue

| Entraînement | Époques | Meilleure époque | Score | RMSE max (niveau) |
|---|---|---|---|---|
| C' s42 | 30 | à lire (≥ 18) | ≤ 0,614 | ≤ 0,307 m/s (2500 m) à l'ép. 18 |
| D s42 | 30 | 27 | 2,317 | 1,159 m/s (25 m) |
| C' s43 | en cours | | | |
| D s43 | en cours | | | |
| C' s44 | à lancer | | | |
| D s44 | à lancer | | | |

D s42 : score 2,317 à l'ép. 27, puis 2,325 / 2,328 / 2,330 aux ép. 28–30 (plateau, légère remontée ; l'ép. 27 est
gardée). R² final u 0,983, v 0,979, w 0,937. Durée : ~207 s par époque sur A100 40 Go.

## Évaluation d'information (sans effet sur le choix) — `scripts/info_capteurs_v4.py`

`rmse_max_uv` = pire tranche 0–100 / 100–300 / 300–600 / 600–1000 m au-dessus du sol, u ou v ; 64 instants de
validation ; mêmes tirages pour tous (graine 1000 + instant) ; ERA5 dégradée = champ corrélé du candidat D
(σu 1,91, σv 2,02 m/s, L 55 km), sans zones.

| Entrée | C' s42 | D s42 | ERA5 seule |
|---|---|---|---|
| ERA5 propre (production) | **0,2533** | 0,9395 | 0,3233 |
| ERA5 dégradée | 1,8975 | **1,3544** | 1,8929 |
| ERA5 dégradée + 50 capteurs simulés | 1,8975 | 1,3545 | 1,8929 |
| Gain des 50 capteurs | 7,9e-8 | −8,7e-5 | — |

Lecture :
- C' fait mieux que l'ERA5 seule sur l'entrée de production (0,253 contre 0,323, sous 0,5 m/s) mais recopie une
  ERA5 dégradée.
- D corrige l'ERA5 dégradée (−28 %) mais perd sur l'ERA5 propre (0,940).
- **Aucun des deux n'exploite les capteurs simulés** (gain nul), même D qui en voit 20 à 200 par exemple à
  l'entraînement : même constat que l'OS6 du Grand Tunis (un point de mesure n'est qu'une correction locale pour
  ce réseau). Conséquence pour la phase 5 : il faut un modèle qui sache exploiter les capteurs (par exemple un canal
  de hauteur du capteur, une correction locale autour des capteurs), sous une nouvelle règle.
