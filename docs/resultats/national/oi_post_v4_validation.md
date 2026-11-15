# C' s44 + OI des innovations des aéroports — validation 2021

Règle `configs/national/regle_oi_post_v4.json` (sha256 `a1fa8921…`, écrite avant le calcul), carnet
`pcfno3d_national_v4_oi_post_colab.ipynb` (commit `f76ca17`, Colab A100, 15–16/11/2026, lancé sur « lance le carnet OI »
d'Ines). Validation de la vérité v3 : 120 instants synoptiques de 2021 ; chaque aéroport jugé est **retiré** de
l'entrée du modèle et des innovations. **2022 non lue, aucun entraînement, aucun bruit ajouté.** L 150 km, λ 4 fixés
par le diagnostic 2015–2020. Sorties : `MyDrive/pcfno3d_national/runs/national_1km_v4_Cp_s44/validation_reelle/val/`.

RMSE vectoriel à 10 m (m/s), METAR = mesures :

| Aéroport | ERA5 | ERA5 + OI | modèle | modèle + OI |
|---|---|---|---|---|
| Bizerte | 3,360 | 3,135 | 3,278 | 3,093 |
| Djerba-Zarzis | 2,916 | 2,682 | 2,835 | 2,664 |
| Enfidha-Hammamet | 3,247 | 3,150 | 3,314 | 3,171 |
| Gabès | 3,021 | 2,841 | 3,007 | 2,852 |
| Gafsa | 2,856 | 2,855 | 3,123 | 3,079 |
| Kairouan | 1,796 | 1,802 | 1,706 | 1,830 |
| Monastir | 2,363 | 2,277 | 2,695 | 2,621 |
| Remada | 2,596 | 2,536 | 2,350 | 2,294 |
| Sfax | 2,890 | 2,650 | 2,643 | 2,464 |
| Tabarka | 3,197 | 3,094 | 3,568 | 3,491 |
| Tozeur | 2,843 | 2,915 | 2,757 | 2,821 |
| Tunis-Carthage | 2,863 | 2,616 | 2,706 | 2,519 |
| **Moyenne** | **2,829** | **2,713** | **2,832** | **2,742** |

| Source | écart à l'ERA5 | IC 95 % (jours) | aéroports améliorés | pire dégradation |
|---|---|---|---|---|
| modèle + OI | −0,087 (−3,1 %) | [−0,139 ; −0,032] | 8 / 12 | +10,9 % |
| ERA5 + OI (référence) | −0,116 (−4,1 %) | [−0,167 ; −0,066] | 10 / 12 | +2,5 % |

**Verdict de la règle (écrite avant) : modèle + OI n'est PAS prometteur** : conditions (1) et (2) remplies, condition (3)
(aucun aéroport dégradé de plus de 5 %) non remplie. Le modèle seul est déjà moins bon que l'ERA5 à Tabarka (+12 %),
Monastir (+14 %) et Gafsa (+9 %), comme sur le test 2022 (+14 %, +8 %, +4 %) : un défaut systématique du vent à 10 m
de la vérité d'entraînement près des côtes et du relief, que l'OI des autres aéroports ne corrige pas. La règle
n'est pas modifiée après coup ; toute variante (par ex. vent à 10 m = ERA5 + OI, ou mélange modèle / ERA5 aux
sites) aura sa propre règle datée et devra être jugée sur des données indépendantes (2023–2024), 2022 ayant été lue.
