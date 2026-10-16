# Lot M8 — preuves : méthodes de référence, réseau (E1, E2, E3) et sobriété (E5)

*Modules : `src/pcfno3d/baselines.py`, `src/pcfno3d/preuves.py`, `src/pcfno3d/sobriete.py`.
Script GPU : `scripts/m8_preuves_colab.py`. Commande : `pcfno3d baselines`. Tests :
`tests/test_baselines.py`. Livrables L6 (rapport d'évaluation : comparaison avec une méthode
de référence) et L9 (études d'aide au dimensionnement du réseau) ; tâches 7, 29 à 33.*

Diagrammes UML du lot (tâche 14, PlantUML comme le prévoit le cahier p. 23) : `docs/uml/cas_utilisation.puml`, `docs/uml/classes.puml`, `docs/uml/sequence_reconstruction.puml`, `docs/uml/sequence_gpu_eteint.puml` (rendu : `plantuml docs/uml/*.puml`).

## Statut : chiffres v7 **mesurés** (Colab L4, 16/10/2026) ; E2 est un résultat négatif

Session `scripts/m8_preuves_colab.py` : `aneo_fno3d_runs/v7/preuves_pcfno3d/best_dso_v7/session_20261016T125440Z`. Résultats dans
« Résultats v7 (GPU) » ci-dessous. Les tests CPU vérifient le **mécanisme** sur le mini-jeu
(40 × 20 points, 4 niveaux, 6 jours) ; leurs chiffres ne disent rien de la v7.

## Définitions retenues (cahier des charges v9)

| Code | Définition du cahier | Page | Ce que le code mesure |
|---|---|---|---|
| L6 | « Rapport d'évaluation : critères, robustesse, comparaison avec une méthode de référence » | p. 21 (§ 9.1) | `comparer` : PC-FNO3D et références sur le test synoptique |
| Tâche 7 | « Comparaison avec une méthode de référence (profil logarithmique et krigeage) » | p. 44 (§ 20.2) | idem |
| — | lot 5 : « Comparaison avec les méthodes de référence — Profil logarithmique, PyKrige — Preuve de supériorité » | p. 27 | idem (krigeage écrit en numpy / scipy, voir plus bas) |
| E1 | « Étude de densité du réseau : erreur par niveau selon la part de stations (5 à 100 %) ; densité recommandée » (BF35, S8) | p. 42 (§ 19.1), tâche 31 p. 45 | `etude_densite` : 5, 10, 25, 50, 75, 100 % des capteurs + réseaux réguliers 300 m, 500 m, 1 km (§ 18.2, p. 41 ; « à confirmer sur les 120 instants de test », § 18.4, p. 42) |
| E2 | « Placement optimal des stations : ajout itératif d'une station là où l'incertitude est la plus forte ; carte des emplacements prioritaires ; gain par rapport à un placement au hasard » (BF36, S9) | p. 42, tâche 32 p. 45 | `placement_optimal` |
| E3 | « Capteur virtuel de secours : valeur d'une station en panne donnée par son nœud virtuel ; erreur de remplacement chiffrée » (BF37, S10) | p. 42, tâche 30 p. 45 | `capteur_secours` |
| E5 | « Empreinte carbone : énergie et CO₂ de l'entraînement et d'une reconstruction (CodeCarbon), comparés à une simulation numérique » (S11) | p. 42, tâche 33 p. 45 | `Sobriete`, `historique_entrainement` |
| L9 | « Études d'aide au dimensionnement du réseau : densité, placement, capteur de secours » | p. 21 | E1 + E2 + E3 |

E4 (gain de l'ensemble par masquage, tâche 29) relève du lot M2 (`docs/FIABILITE.md`) et
de l'étape A de la campagne v8 (`docs/rapport_dso1.md`, § 3) ; il n'est pas refait ici.

## Protocole commun

- **Instants** : les instants synoptiques du **test** (120 sur le jeu complet,
  `docs/rapport_dso1.md` § 1), graine **2000**, comme DSO1 (`Evaluator.run`).
- **Capteurs** : `mask.keep_eval` (50 %) pour la comparaison ; E1 fait varier la densité.
  Les masques d'une même graine sont **emboîtés** d'une fraction à l'autre : les capteurs
  visibles à 25 % font partie de ceux visibles à 50 %.
- **Mêmes instants, mêmes capteurs, mêmes nœuds pour toutes les méthodes** :
  `evaluate.metriques_loader` lit chaque instant **une seule fois** et évalue tous les
  prédicteurs dessus. C'est **le code de métriques de l'`Evaluator`** (il en a été extrait,
  sans changement de formule) : RMSE et R² par niveau **cumulés** sur les instants, nœuds
  valides et connus, niveaux sous 5 % de couverture exclus des verdicts, mêmes verdicts DSO1
  (`evaluate.verdicts_dso1`), résidu de masse (DSO2) et rugosité (DSO3) par instant.
  PC-FNO3D passe par `Evaluator.predire`, le chemin exact de DSO1.
- **Contrôle intégré** : sur tout le test, la ligne `pcfno3d` doit retrouver la table du
  notebook v7 (`run_dir/eval_best_dso_v7/test_par_niveau_synoptique.csv`) ; l'écart
  maximal est publié dans `resume.json` (`controle_v7`).
- **Séparation validation / test** : tout ce qui se règle (variogramme, α, emplacements
  E2) se règle sur la **validation** (sélection synoptique fixe, graine 1000).
  `charger_ajustement` refuse un fichier dont `"jeu"` n'est pas `"validation"`, et un
  ajustement n'est jamais écrasé.

## Méthodes de référence (L6)

Interpolation **horizontale** des capteurs visibles (vent à hauteur de capteur), puis loi
**verticale** qui l'extrapole aux nœuds 3D :

| Nom | Horizontal | Vertical | Paramètre ajusté (validation) |
|---|---|---|---|
| `log_idw` | inverse des distances², 8 voisins | profil logarithmique `U(h) = U_s · ln(h/z0) / ln(h_s/z0)` | aucun (z0 de la grille du générateur) |
| `puissance_idw` | idem | loi de puissance `U(h) = U_s · (h/h_s)^α` | α (grille 0 à 0,6, pas 0,01) |
| `log_krigeage` | krigeage ordinaire, 16 voisins | profil logarithmique | variogramme |
| `puissance_krigeage` | krigeage ordinaire, 16 voisins | loi de puissance | variogramme, α |

- `h` : hauteur du nœud au-dessus du sol (niveau ASL − relief) ; `h_s` : hauteur du
  capteur au-dessus du sol (`geom.z_sensor − geom.zt`, la même altitude de capteur que la
  détection de dérive M4).
- Ces lois ne font pas tourner le vent avec la hauteur et donnent **w = 0** : c'est une de
  leurs limites, celle que cite le cahier (p. 8 : « ne respectent pas les lois physiques ;
  fiabilité qui se dégrade en altitude ; pas d'incertitude par point »). Le résidu de masse
  (DSO2) des références est donc publié à côté, au même titre que leur RMSE.
- **Variogramme** : modèle exponentiel `γ(h) = pépite + palier · (1 − exp(−3h / portée))`,
  ajusté par moindres carrés pondérés sur le variogramme empirique de u et v
  **standardisés par instant** (les poids du krigeage ne dépendent pas du palier). Paires :
  8 plus proches voisins de 2 000 capteurs (petits écarts) + 20 000 paires au hasard (grands
  écarts), classes de distance géométriques jusqu'à la moitié du petit côté du domaine. Si
  la portée atteint la borne haute, `portee_a_la_borne` le signale.
- **α** : minimise la somme des carrés des erreurs de u et v sur les nœuds valides des
  niveaux évalués, séparément pour chaque interpolation horizontale ; la courbe RMSE(α) est
  publiée (`rmse_alpha`).
- **PyKrige** n'est pas installé et n'est pas ajouté comme dépendance : le krigeage
  ordinaire est écrit en numpy / scipy, en voisinage local (k plus proches capteurs), sous
  sa forme covariance (`C(h) = C(0) − γ(h)`, système symétrique défini positif, résolu par
  une élimination vectorisée sur le lot de cibles). La forme est équivalente au système en
  variogramme avec multiplicateur de Lagrange ; le test
  `test_krigeage_ordinaire_exact_poids_de_somme_un_et_systeme_complet` le vérifie contre le
  système complet écrit à la main. Mesure sur ce poste (CPU) : 10 000 cibles en 0,5 s, soit
  de l'ordre de 6 s par instant pour les 125 000 colonnes du domaine (extrapolation, à
  confirmer sur Colab). C'est pourquoi E1 n'évalue par défaut que `log_idw` et
  `puissance_idw` (`--references-e1`).

## E1, E2, E3

**E1 — densité.** Pour chaque densité, la table par niveau de chaque méthode, la RMSE
max de u, v sur les niveaux évalués, la RMSE globale et le nombre moyen de capteurs
visibles. **Règle de recommandation, fixée avant la mesure** : plus petite fraction (ou plus
grand espacement) dont la RMSE max de u, v est ≤ 0,5 m/s (seuil DSO1) ou ≤ 0,7 m/s (seuil
DSO5) ; `null` si aucune densité ne la satisfait.

Repère existant, **indicatif** (8 instants d'un seul jour, CPU, `tukhnanutha.md` § 5,
« Sensor density check ») : à 1 capteur par km, u / v valent 0,490 / 0,258 m/s à 100 m et
0,867 / 0,595 m/s à 1000 m. E1 sur les 120 instants doit confirmer ou infirmer ce constat.

**E2 — placement.** Réseau de base **fixe dans l'espace** (par défaut un capteur tous les
1 000 m, décalage tiré une fois), puis `n_etapes` ajouts de `par_etape` stations (par
défaut 4 × 5 % de la base). À chaque étape : σ **brut** de la vitesse par l'ensemble M2
(`reliability.members` passes masquées), moyen sur les niveaux évalués et sur 8 instants
de **validation** ; les colonnes de σ maximal sont retenues, deux à deux à au moins 2
mailles. Le gain se mesure sur le **test** contre 3 placements au hasard du même nombre de
stations, parmi les mêmes candidats : `gain_vs_hasard_m_s` = RMSE(hasard, moyenne) −
RMSE(optimal). Carte : `e2_placement.json` (emplacements, étape, σ) et `e2_cartes.npz`
(σ par étape, réseau de base, réseau final). Le σ n'est pas calibré : seul son classement
compte ici.

**E3 — capteur de secours.** À chaque instant de test, 50 stations visibles, deux à deux
à au moins 3 mailles, tombent en panne. Leur valeur est remplacée par :

- `pcfno3d` : la reconstruction faite **sans elles** (caractéristiques géostrophiques
  recalculées, comme en production), interpolée à l'altitude du capteur
  (`alerts.interp_colonne`, même règle que M4) ;
- `plus_proche`, `idw`, `krigeage` : interpolation horizontale des stations restantes.

Erreur de remplacement = valeur de remplacement − mesure : RMSE u, v, vectorielle, biais,
erreur de vitesse, 95ᵉ centile ; une ligne par panne dans `e3_secours.csv`. **Limite
connue** (`docs/ALERTES.md`) : le capteur est à 25–30 m du sol et, en relief, le premier
nœud valide de la colonne est parfois plus haut ; sous ce nœud, PC-FNO3D prend sa valeur
(pas d'extrapolation). Cet écart de représentativité pénalise PC-FNO3D face aux
interpolations entre capteurs, qui restent à hauteur de capteur.

## E5 — sobriété

Trois sources, de la plus directe à la moins directe, chaque chiffre portant sa `source` :

1. **CodeCarbon** s'il est installé (il ne l'est pas sur ce poste, et n'est pas une
   dépendance obligatoire) : énergie CPU + GPU + RAM et CO₂.
2. **nvidia-smi** sinon : puissance du GPU échantillonnée chaque seconde et intégrée —
   énergie du **GPU seul**, mesurée.
3. **`--gpu-watts`** sinon : temps × puissance donnée par l'utilisateur, une
   **estimation**.

Sans aucune des trois, seule la durée est publiée et `energie_kwh` vaut `null`, avec la
note « à mesurer sur GPU ». Hors CodeCarbon, le CO₂ n'est calculé que si l'intensité
carbone est fournie (`--intensite-gco2-kwh`) : aucun facteur n'est supposé.

**Entraînement v7 (déjà fait).** Seule la durée peut être reconstruite honnêtement, depuis
le journal du BLOC 9 de `notebooks/current/fno3d_v7.ipynb` (`historique_entrainement`) :

| Grandeur | Valeur | Source |
|---|---|---|
| GPU | NVIDIA A100-SXM4-80GB | sortie du notebook (cellule du BLOC 0) |
| Époques prévues | 140 | lignes « Ep … /140 » |
| Époques journalisées | 80 (61 à 140 ; reprise à l'époque 60) | idem |
| Durée journalisée | 9 582 s, soit 119,8 s par époque en moyenne | somme des durées par époque |
| Époques 1 à 60 | **non journalisées** (session précédente) : non estimées | — |
| Énergie, CO₂ | **non mesurés** : la puissance n'a pas été journalisée | — |

**Comparaison à une simulation numérique.** Le cahier (p. 8) indique « plusieurs heures de
calcul par instant » pour CFD / WRF. Le dépôt ne contient aucune mesure d'énergie d'une
simulation WRF : la comparaison chiffrée reste à sourcer.

## Résultats v7 (GPU) — Colab L4, 16/10/2026

Session `aneo_fno3d_runs/v7/preuves_pcfno3d/best_dso_v7/session_20261016T125440Z`. Test synoptique : 120 instants, 50 % des capteurs visibles, mêmes
instants, capteurs et nœuds pour toutes les méthodes.

**Ajustement (validation, 64 instants).** Variogramme exponentiel, pépite ≈ 0, palier
1,105, portée 7 741 m ; α = 0,06.

**L6 — comparaison aux références.**

| Méthode | RMSE max u, v (m/s) | RMSE global u / v (m/s) | R² min |
|---|---|---|---|
| **PC-FNO3D v7** | **0,665** (1 000 m) | **0,542 / 0,437** | **0,991** |
| loi de puissance + IDW | 5,027 | 2,974 / 3,474 | 0,569 |
| loi de puissance + krigeage | 5,031 | 2,979 / 3,478 | 0,568 |
| profil log + IDW | 6,481 | 4,171 / 3,269 | 0,156 |
| profil log + krigeage | 6,487 | 4,176 / 3,275 | 0,155 |

**E1 — densité.** PC-FNO3D tient DSO5 (RMSE ≤ 0,7 m/s) jusqu'à **50 % des capteurs**.
Aucun réseau régulier testé (un capteur tous les 300 m, 500 m ou 1 000 m) ne le tient, et
aucune référence ne le tient, quelle que soit la densité.

**E2 — placement guidé par σ : RÉSULTAT NÉGATIF.** RMSE max 2,387 m/s avec le placement
guidé par σ, contre 1,278 m/s en moyenne au hasard (écart-type 0,021) : gain **−1,109 m/s**.
Le σ de M2 ne guide pas le placement des stations.

**E3 — capteur virtuel de secours.** Erreur vectorielle RMS (p95) : krigeage 0,192 (0,38),
IDW 0,268 (0,545), plus proche voisin 0,446 (0,899), PC-FNO3D 1,495 m/s (3,216). Sur ce
treillis très dense, les voisins valent mieux que la reconstruction 3D.

**E5 — sobriété.** Inférence : **9,3 µWh GPU** par reconstruction (0,465 s ; nvidia-smi,
GPU seul). CO₂ **non calculé** (intensité carbone non fournie). Entraînement **non
mesuré**. CodeCarbon absent de la session.

## Résultats CPU du mini-jeu (mécanisme seulement)

Modèle de test de 2 époques sur le mini-jeu, 4 instants de test, CPU (session du
2026-10-15). **Ces chiffres ne disent rien de la v7.** De plus, le mini-jeu fabrique son
vent avec un **profil logarithmique** (z0 = 0,1 m, référence à 30 m :
`scripts/make_mini_dataset.py`) : la référence `log_idw` y est quasi exacte par
construction.

| Vérification | Observé |
|---|---|
| Même table pour PC-FNO3D via `metriques_loader` et via `Evaluator.evaluate_loader` | identique (test) |
| Prédicteur « vérité » | RMSE 0, R² 1 (test) |
| Mêmes nœuds pour toutes les méthodes | même vitesse moyenne par niveau (test) |
| Ajustement (validation) | α = 0,16 (idw et krigeage) ; portée à la borne (5 000 m) sur ce petit domaine |
| Comparaison (RMSE max u, v) | modèle jouet 3,79 m/s ; `log_idw` 0,062 ; `puissance_idw` 0,183 ; `log_krigeage` 0,063 ; `puissance_krigeage` 0,182 |
| E1 | masques emboîtés (moins de capteurs à 25 % qu'à 100 %) ; recommandation calculée |
| E2 | 2 étapes de 3 stations ; gain contre le hasard ≈ 0 (± 1·10⁻⁴ m/s) : le modèle jouet ne tire aucun parti des stations ajoutées |
| E3 | RMSE vectorielle : `plus_proche` 0,125, `idw` 0,069, `krigeage` 0,069, modèle jouet 3,13 m/s |
| E5 | durées seules, `energie_kwh = null` (ni CodeCarbon ni nvidia-smi sur ce poste) |
| Script | dossier de session neuf, aucun fichier existant réécrit, JSON en fins de ligne LF (test) |

## Session GPU (faite le 16/10/2026 ; commandes pour la rejouer)

```python
!pip install codecarbon            # facultatif : sinon nvidia-smi, sinon durées seules
!python scripts/m8_preuves_colab.py -c colab
# reprise d'un ajustement déjà fait, sans le refaire :
!python scripts/m8_preuves_colab.py -c colab --ajustement <session>/ajustement_validation.json --etapes comparaison,e1
```

L'archive Colab (`scripts/package_for_colab.py`) n'embarque pas `notebooks/` : pour la
durée d'entraînement (E5), passer `--journal-entrainement <chemin de fno3d_v7.ipynb>` ;
sinon `e5_sobriete.json` le dit (« journal introuvable »).

Tout s'écrit dans `run_dir/preuves_pcfno3d/<modèle>/session_<horodatage>/` (dossier neuf,
rien n'est écrasé) : `ajustement_validation.json`, `comparaison_test.json` /
`comparaison_par_niveau.csv`, `e1_densite.json` / `.csv`, `e2_placement.json`,
`e2_cartes.npz`, `e3_secours.json` / `.csv`, `e5_sobriete.json`, `resume.json`.

Ordre de grandeur des passes du modèle (estimation, à 91 ms par passe, DSO4 de
`tukhnanutha.md` § 5) : comparaison 120 ; E1 9 × 120 = 1 080 ; E2 4 étapes × 8 instants ×
9 passes + 5 réseaux × 120 = 888 ; E3 120 ; E5 22. Le krigeage, sur CPU, domine
(de l'ordre de 6 s par instant, voir plus haut).

À reporter dans le rapport après la session : la table L6 (PC-FNO3D contre chaque
référence, par niveau), le contrôle `controle_v7`, la densité recommandée (E1), le gain du
placement (E2), l'erreur de remplacement (E3), et l'énergie avec sa source (E5).

Sans poids (références seules, CPU) :

```bash
pcfno3d baselines --ajuster                 # ajuste sur la validation, compare sur le test
pcfno3d baselines --ajustement <json> --espacement 1000
```
