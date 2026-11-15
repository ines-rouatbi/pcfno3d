# Jugement indépendant 2023–2024 aux 12 aéroports

Règle `configs/national/regle_independant_2023_2024.json` (sha256 `98cdbeac…`, écrite avant le téléchargement, accord
d'Ines « ok télécharge 2023-2024 »), `scripts/independant_2023_2024.py`, lecture unique le 15/11/2026. Période
**jamais utilisée** (ni ajustement, ni validation, ni test). METAR IEM (mesures, 251 965 messages exploitables) ;
ERA5 10 m Open-Meteo (réanalyse). 5 848 instants synoptiques, 64 169 paires. Chaque aéroport jugé est **retiré**
des innovations ; L 150 km, λ 4 et les biais 2015–2020 sont ceux fixés avant (aucun réglage sur 2023–2024).

| Aéroport | n | RMSE vect. ERA5 | RMSE vect. ERA5 + OI | écart |
|---|---|---|---|---|
| Tabarka | 5319 | 3,130 | 3,021 | −3,5 % |
| Monastir | 5817 | 2,570 | 2,524 | −1,8 % |
| Enfidha-Hammamet | 5791 | 3,090 | 2,864 | −7,3 % |
| Tunis-Carthage | 5774 | 2,509 | 2,250 | −10,3 % |
| Bizerte | 5194 | 2,866 | 2,659 | −7,2 % |
| Remada | 4935 | 2,423 | 2,429 | +0,3 % |
| Gafsa | 5245 | 2,901 | 2,962 | +2,1 % |
| Gabès | 5321 | 2,837 | 2,737 | −3,5 % |
| Djerba-Zarzis | 5807 | 2,649 | 2,497 | −5,7 % |
| Kairouan | 3983 | 1,938 | 1,898 | −2,1 % |
| Sfax | 5492 | 2,980 | 2,788 | −6,4 % |
| Tozeur | 5491 | 2,978 | 3,040 | +2,1 % |
| **Moyenne** | | **2,739** | **2,639** | **−3,7 %** |

IC 95 % de l'écart (bootstrap par jours) : [−0,108 ; −0,092] m/s ; 9 aéroports sur 12 améliorés ; pire dégradation
+2,1 %. **Les trois conditions de la règle sont remplies : l'OI des mesures des aéroports bat l'ERA5 sur une période
indépendante.**

Portée : c'est la couche « mesures des aéroports » (ERA5 + OI) qui est jugée ici. Le modèle C' s44 lui-même n'est pas
jugé sur 2023–2024 (il faudrait le forçage ERA5 national de ces instants) ; sur la validation 2021, « modèle + OI »
gagnait −3,1 % mais restait dégradé de +10,9 % à Monastir (vent à 10 m de la vérité d'entraînement près des côtes),
voir `oi_post_v4_validation.md`.
