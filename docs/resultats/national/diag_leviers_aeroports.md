# Leviers pour battre l'ERA5 aux aéroports — diagnostic (validation 2021)

Règle `configs/national/regle_diag_leviers_aeroports.json` (sha256 `fc37901393b7…`, 15/11/2026, écrite avant le calcul),
`scripts/diag_leviers_aeroports.py`. Ajustement 2015–2020, jugement 2021 ; **2022 (test) non lue**. Chaque aéroport est
jugé **retiré** (ni dans l'ajustement, ni en entrée), comme le DSO7. Instants synoptiques (3 h). ERA5 = réanalyse,
jamais des mesures ; METAR = mesures. Diagnostic sans effet sur un modèle livré.

Contexte : sur le test 2022 (lecture unique), C' s44 fait 2,844 m/s aux 12 aéroports contre 2,857 pour l'ERA5 et
2,857 pour la vérité d'entraînement (générateur) : le réseau reproduit sa vérité, dont la précision aux aéroports est
celle de l'ERA5. Pour battre l'ERA5, il faut une information que l'ERA5 n'a pas : les mesures.

| Levier (validation 2021, aéroport retiré) | RMSE vectoriel moyen | écart à l'ERA5 | IC 95 % (jours) | aéroports améliorés | pire dégradation |
|---|---|---|---|---|---|
| ERA5 seule | 2,793 | — | — | — | — |
| **ERA5 + OI des 11 autres aéroports** (L 150 km, λ 4, choisis sur 2015–2020) | **2,696** | **−0,097 (−3,5 %)** | [−0,111 ; −0,083] | 8 / 12 | **+5,3 %** (Tozeur) |
| MOS générique (règle MOS, station retirée) | 2,629 | −0,164 (−5,9 %) | [−0,183 ; −0,143] | 8 / 12 | +24,7 % (Tozeur) |
| MOS + OI des résidus | 2,677 | −0,116 | [−0,137 ; −0,094] | 9 / 12 | +26,8 % |

Lecture : l'OI des innovations des autres aéroports est le levier **sûr** (gain net, IC entièrement négatif, aucun
aéroport dégradé de plus de 5,3 %) ; le MOS gagne plus en moyenne mais dégrade fortement les sites venteux de
l'intérieur et du sud (Tozeur, Gafsa, Djerba), comme dans la règle MOS (condition 3 non remplie).

Par aéroport (RMSE vectoriel, m/s) :

```
levier                 era5  mos_generique  mos_plus_oi  oi_aeroports
aeroport                                                             
Bizerte               2.911          2.364        2.411         2.654
Djerba Mellita        2.655          3.030        3.125         2.656
Enfidha               3.103          3.067        3.019         2.958
Gabes                 3.558          2.459        2.615         3.405
Gafsa                 2.973          3.302        3.373         3.043
Habib Bourguiba Intl  2.590          2.622        2.560         2.599
Kairouan              1.863          1.453        1.519         1.765
Remada                2.512          2.472        2.476         2.408
Sfax El-Maou          2.905          2.408        2.494         2.657
Tabarka               2.885          2.652        2.682         2.731
Tozeur                2.919          3.639        3.700         3.074
Tunis-Carthage        2.645          2.085        2.155         2.404```
