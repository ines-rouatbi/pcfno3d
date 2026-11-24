# Règle `regle_national_gfs_v6` : résultat (ARRÊT à l'étape 1)

Règle `configs/national/regle_national_gfs_v6.json` (sha256 `9babee638cf8…`), approuvée par Ines le 23/11/2026.
Carnet `notebooks/current/pcfno3d_national_gfs_v6_l4.ipynb` (commit 39b086b, code bb94ead), Colab L4, 24/11/2026
vers 07:12-07:20 (heure de Tunis), après deux tentatives sans GPU convenable (processeur seul, puis T4) libérées par
les gardes du carnet sans rien lancer.

**Résultat : ARRÊT.** Aucune variante n'est admissible sur la validation 2021 avec la marge de la règle (borne haute
de l'IC 95 % ≤ 0 et chaque aéroport ≤ +3 % face à GFS brut, dans les trois cases). Conformément à la règle,
**novembre 2026 n'est pas lu** : ni GFS ni METAR de novembre n'ont été téléchargés, aucun jugement n'a été calculé.
`MyDrive/pcfno3d_national/gfs_v6_regle/selection.json` contient l'arrêt ; la règle v6 ne peut pas être relancée.

Sortie du carnet, recopiée telle quelle :

| Variante | RMSE moyen (3 cases) | pire aéroport (analyse / +24 h / +48 h) | IC haut (3 cases) | admissible |
|---|---:|---|---|---|
| c1 | 2.916 | Tabarka +6.2% / Tabarka +4.8% / Tabarka +4.0% | -0.171 / -0.196 / -0.198 | non |
| c3_a0.75_H100 | 2.920 | Gafsa +4.4% / Gafsa +3.4% / Gafsa +2.9% | -0.175 / -0.197 / -0.197 | non |
| c3_a0.75_H300 | 2.921 | Gafsa +4.2% / Gafsa +3.3% / Gafsa +2.9% | -0.174 / -0.197 / -0.195 | non |
| c3_a0.5_H100 | 2.927 | Gafsa +4.3% / Gafsa +3.4% / Gafsa +2.9% | -0.175 / -0.196 / -0.195 | non |
| c3_a0.5_H300 | 2.930 | Gafsa +4.1% / Gafsa +3.3% / Gafsa +2.9% | -0.173 / -0.194 / -0.191 | non |
| c3_a0.25_H100 | 2.936 | Gafsa +4.3% / Gafsa +3.4% / Gafsa +2.9% | -0.174 / -0.191 / -0.191 | non |
| c3_a0.25_H300 | 2.942 | Gafsa +3.9% / Gafsa +3.2% / Gafsa +2.9% | -0.170 / -0.188 / -0.183 | non |
| c2_a0.25_H300 | 2.966 | Gafsa +4.1% / Gafsa +3.8% / Gafsa +3.2% | -0.156 / -0.170 / -0.166 | non |
| c2_a0.25_H100 | 2.969 | Gafsa +4.6% / Gafsa +4.4% / Gafsa +3.6% | -0.152 / -0.166 / -0.164 | non |
| c2_a0.5_H300 | 2.971 | Tabarka +5.3% / Tabarka +5.0% / Tabarka +4.0% | -0.149 / -0.164 / -0.161 | non |
| c2_a0.5_H100 | 2.973 | Tabarka +5.4% / Tabarka +5.1% / Tabarka +4.1% | -0.145 / -0.160 / -0.159 | non |
| c2_a0.75_H300 | 2.979 | Tabarka +8.3% / Tabarka +7.6% / Tabarka +6.3% | -0.139 / -0.153 / -0.153 | non |
| c2_a0.75_H100 | 2.981 | Tabarka +8.4% / Tabarka +7.7% / Tabarka +6.3% | -0.137 / -0.151 / -0.152 | non |

**ARRÊT** : aucune variante admissible sur 2021 (IC <= 0 et chaque aeroport <= +3 % dans les trois cases) : novembre
2026 n'est pas lu

## Lecture

- Toutes les variantes battent GFS brut en moyenne en 2021 (IC entièrement négatifs). Ce qui bloque, c'est la marge
  de +3 % par aéroport, et pour les variantes C3 ce n'est pas Tabarka mais **Gafsa** (+3,9 à +4,4 % à l'analyse),
  sous la limite de jugement (+5 %) mais au-dessus de la marge.
- Erreur de conception de l'agent : la marge de +3 % a été écrite sans vérifier, sur les résultats 2021 déjà connus,
  qu'au moins une variante pouvait la tenir. Elle ne le pouvait pas. Le seuil n'est pas changé après lecture.
- Coût Colab : environ 8 minutes de L4 (≈ 0,3 unité), plus deux allocations libérées aussitôt.
- Statut du service inchangé : `regle_service_zones_v5` (« validé » à plus de 50 km de Tabarka, « expérimental
  (zone Tabarka) » dedans), à côté du verdict refusé de `regle_national_gfs_v5_2`. Aucun fichier de verdicts v6.
- Novembre 2026 (1er au 23, toutes les 3 h) reste **jamais lu**.
