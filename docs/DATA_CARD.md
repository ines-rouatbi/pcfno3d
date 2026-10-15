# Data card — jeu de données synthétique `aneo_dataset_3d_v2`

*Livrable L7 (cahier des charges v9, § 9.1 p. 21 ; tâche 13 p. 44). Chaque chiffre est
suivi de sa source dans le dépôt. Le jeu lui-même n'est pas versionné (environ 35 Go,
`.gitignore`) : cette fiche décrit ce qu'en disent les fichiers du dépôt.*

## En bref

| Élément | Valeur | Source |
|---|---|---|
| Nature | **synthétique** : sortie d'un générateur physique, pas de mesures réelles | `tukhnanutha.md` § 3 et § 5 ; cahier p. 10 (§ 4.7), p. 23 (§ 10.3) |
| Générateur | `aneo_physical_db_multitime_v3_progress/` (et ses schémas MySQL / PostGIS) | `CLAUDE.md`, « Data and artifacts » |
| Date de génération | 2026-09-28 | `tukhnanutha.md` § 5 |
| Taille | ~35 Go | `CLAUDE.md` |
| Durée couverte | 100 jours, 1 000 instants | `tukhnanutha.md` § 5 ; `CLAUDE.md` |
| Domaine | 50 × 25 km, 12 niveaux de 25 à 1 000 m | cahier p. 7 (§ 1.3) |
| Grille de surface | 1 000 × 500 points au pas de 50 m | `tukhnanutha.md` § 5 |
| Nœuds 3D (cible) | 500 × 250 au pas de 100 m, niveaux 25 / 50 / 75 / 100 / 150 / 200 / 300 / 400 / 500 / 650 / 800 / 1 000 m **au-dessus du niveau de la mer** | `tukhnanutha.md` § 5 |
| Nœuds 3D valides (au-dessus du relief) | 849 184 | `tukhnanutha.md` § 3 |
| Points de surface pouvant porter un capteur | 494 006 par instant (≈ 62 000 visibles à 50 %) | `tukhnanutha.md` § 5, « Sensor density check » |
| Relief | jusqu'à 677 m | `tukhnanutha.md` § 5, « v7 served locally » |
| Hauteur des capteurs | 25 à 30 m au-dessus du sol | `tukhnanutha.md` § 2 |
| Ancrage géographique | **SUPPOSÉ** : coin sud-ouest à 36,00 N, 9,70 E (latitude cohérente avec f = 8,57·10⁻⁵ s⁻¹ du générateur) ; le relief reste celui du générateur, pas celui de ce lieu | `configs/colab.yaml` (`real.origin_assumed: true`) ; `docs/INTEGRATION.md` (`"suppose": true`) |

## Contenu

- `meta.json` : géométrie de la grille et des nœuds (lue par `Geometry`, jamais codée en
  dur) ; `split_days.json` : découpage par jour ; `aneo_grid_points.csv` (~127 Mo) : grille
  statique (relief `z_terrain_m`, rugosité `z0_m`, pente, orientation, hauteur des points) ;
  CSV journaliers au format ANEO ; `qc/qc_summary.json` : contrôle qualité du générateur.
  Source : `CLAUDE.md`, « Data and artifacts » ; `src/pcfno3d/geometry.py`.
- **Entrées** (surface) : 7 variables — vent u, v, w, température, densité de l'air,
  humidité relative, pression (`SURF_VARS`, `src/pcfno3d/geometry.py`) — plus `mask_i_obs`
  (1 = mesure, 0 = valeur interpolée). Le schéma et les plages du contrat ANEO sont au
  cahier p. 18–19 (§ 8.1–8.3) et dans `docs/CONTRAT_ANEO.md`.
- **Cible** : u, v, w et ρ aux nœuds 3D. Valeur 0 sous le relief dans le cache ; la
  densité ρ sert au résidu de continuité (DSO2). Source : `CLAUDE.md`, « Conventions ».
- **Conventions** : u vers l'est, v vers le nord, w vers le haut ; direction météorologique
  `(270 − atan2(v, u)) mod 360` (d'où vient le vent) ; coordonnées métriques locales.
- **Cache** (`pcfno3d build-cache`) : par instant, `{stem}_S.npy` (8, ny, nx), surface
  agrégée par blocs 2 × 2 sur la grille du modèle, et `{stem}_Y.npy` (4, nz, ny, nx).

## Découpage et usage en évaluation

| Élément | Valeur | Source |
|---|---|---|
| Découpage | **par jour entier**, 70 / 15 / 15 % (entraînement / validation / test) | `tukhnanutha.md` § 5 ; `split_days.json` |
| Instants de test | 166, dont **120 synoptiques** (heure pleine, multiple de 3 h) et 46 d'épisodes rapides (rafales, front, brise : test de stress, hors DSO1) | `docs/rapport_dso1.md` § 1 |
| Normalisation | statistiques calculées sur les **jours d'entraînement seulement** (`norm_stats.json`, partie du modèle) | `CLAUDE.md`, `normalize.py` |
| Masques d'évaluation | graine 1 000 (validation), 2 000 (test synoptique), 2 500 (épisodes), 3 000 (robustesse) | `docs/REPRODUCTIBILITE.md` ; `src/pcfno3d/dataset.py` |
| Niveaux exclus des verdicts | couverture < 5 % du domaine : 25, 50 et 75 m (presque entièrement sous le relief) | `docs/rapport_dso1.md` § 1 |

Tout ce qui se règle (poids de la perte, sélection du modèle, calibration M2, seuils M4,
ajustements M8) se règle sur la **validation** ; le test ne sert qu'aux verdicts.

## Contrôle qualité du générateur

`qc/qc_summary.json` passe toutes les vérifications — résidu de masse maximal 6,4·10⁻¹²,
corrélations spatiales et temporelles, erreur hydrostatique au 99ᵉ centile — **sauf**
l'équilibre des secteurs de direction (0,095 pour un seuil de 0,10 : le secteur 0° est un
peu sous-représenté). Source : `tukhnanutha.md` § 5. À mentionner dans le rapport ; non
bloquant.

## Limites

- **Synthétique.** Les cibles en altitude sont des sorties du générateur, pas des mesures
  (`tukhnanutha.md` § 3) : aucune vérité terrain réelle au-dessus des capteurs.
- **Densité irréaliste.** La couche de surface est quasi continue (≈ 62 000 points visibles
  à 50 %), alors qu'un réseau ANEO réel compte un capteur tous les 500 m à 1 km
  (`tukhnanutha.md` § 5 ; cahier p. 41, § 18.2). Premier test indicatif (8 instants) :
  l'erreur des niveaux bas est multipliée par 2 à 3 à densité réaliste. E1 (lot M8,
  `docs/PREUVES.md`) doit le mesurer sur les 120 instants.
- **Ancrage supposé** : aucune mesure réelle ne peut être placée tant que la position réelle
  du domaine n'est pas fournie (cahier p. 41, § 18.3).
- **Transférabilité au réel non démontrée** : le passage aux données réelles est prévu
  (`docs/REAL_DATA.md`, cahier § 18) mais rien de réel n'est encore importé.

## Usages

- **Prévus** : entraîner et évaluer PC-FNO3D selon les critères DSO ; tester la chaîne de
  données, le contrat ANEO, la fiabilité, les alertes et l'espace éolien ; démonstrations.
- **Déconseillés** : toute conclusion sur le vent réel d'un site tunisien ; tout
  dimensionnement d'installation ou décision d'investissement sans campagne de mesure
  (cahier p. 35, § 16.5 : « La production estimée reste une aide à la décision et doit être
  confirmée par une campagne de mesure avant tout investissement »).

## Accès et confidentialité

Données et documents confidentiels, réservés à Tukhnanutha (cahier p. 18, § 7.13) : dépôt
privé, aucun partage public des données.
