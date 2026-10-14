# Entraîner et servir PC-FNO3D sur des données réelles

*État au 14/10/2026. Ce qui est prêt dans le code, ce qu'il faut fournir, dans quel ordre.*

## 1. Le point de départ, sans complaisance

- **La v7 n'a jamais vu une seule mesure réelle.** Sa vérité 3D vient du générateur
  Tukhnanutha v2 : champs paramétriques, pression = un plan exact, vent géostrophique
  uniforme. Ses « capteurs » sont une couche de surface synthétique **quasi continue**
  (494 006 points à 50 m par instant ; ~62 000 visibles à 50 %).
- **Un vrai réseau est bien plus épars** : points ANEO espacés de ~500 m à 1 km. Premier
  test (`scripts/eval_reseau.py`, 8 instants, à confirmer sur les 120) : à densité réaliste,
  l'erreur aux niveaux bas est **multipliée par 2 à 3**, et v se dégrade à tous les niveaux.
- **Le domaine n'a pas de position géographique** : coordonnées métriques locales
  uniquement. Rien de réel ne peut s'y placer sans cet ancrage.

## 2. Ce qui est prêt dans le code

Principe : des **adaptateurs** convertissent les sources réelles dans le **format de cache
existant** (`S` surface, `Y` volume, `index.csv`). Entraînement, évaluation, API et MCP les
utilisent sans modification.

| Brique | Où | Rôle |
|---|---|---|
| Ancrage géographique | `realdata.GeoRef`, config `real.origin_lat/lon`, `rotation_deg` | lat/lon ↔ repère du domaine |
| Stations → entrée `S` | `normalize_stations`, `stations_to_surface` | contrôle qualité, (vitesse, direction) → (u, v), maille la plus proche, vent ramené à la hauteur des capteurs d'entraînement (loi log), T/p/HR manquants → atmosphère standard, densité de l'air humide |
| Champ maillé → cible `Y` dense | `grid_to_target`, `open_gridded` | réanalyse ou WRF (NetCDF), interpolation bilinéaire puis verticale aux 12 niveaux |
| Observations ponctuelles → `Y` partielle | `points_to_target` | radiosondes, mâts, lidar : seuls les nœuds observés sont connus (NaN ailleurs) |
| Apprentissage sur cible partielle | `losses.py`, `dso.py`, `evaluate.py`, `normalize.py`, `train.py` | NaN = non observé : la loss et les métriques ne comptent que les nœuds connus. Sans NaN (synthétique), **résultats identiques à la v7** |
| Import en une commande | `pcfno3d import-real --stations … [--points … \| --grid …]` | écrit le cache, découpe par jours entiers (1 jour sur 7 en validation, 1 sur 7 en test) |
| Reconstruction en direct | `POST /api/reconstruct/stations` | JSON de stations (lat/lon) → volume 3D, réutilisable ensuite via `/field`, `/point`, `/profile` (`stem` = `live_…`) |
| Test de densité réaliste | `scripts/eval_reseau.py` | évalue le modèle avec 1 capteur tous les 300 / 500 / 1000 m |
| Téléchargement ERA5 | `scripts/download_reanalysis.py` (`pip install -e .[real]`) | niveaux de pression 1000–850 hPa et niveaux simples, un fichier par mois, sur le domaine + marge ; `--test` vérifie le compte |
| Lecture ERA5 → cible | `open_era5_pressure_levels`, `import-real --grid … --grid-format era5-pl` | altitude = géopotentiel / g, densité, w depuis omega ; fond d'altitude réel (~31 km), pas une vérité à 100 m |

Tout cela est couvert par des tests (`tests/test_realdata.py` + l'API) sur des données
fabriquées à valeurs connues. **Aucune vraie donnée n'a encore été importée.**

### Formats attendus

**Stations (entrée)**, CSV ou Parquet, une ligne par station et par instant :

| colonne | obligatoire | note |
|---|---|---|
| `station_id`, `timestamp_utc`, `lat`, `lon`, `height_agl_m` | oui | hauteur du capteur au-dessus du sol |
| `wind_u_m_s`, `wind_v_m_s` **ou** `wind_speed_m_s`, `wind_dir_deg` | oui | direction météo : d'où vient le vent |
| `temperature_c`, `pressure_pa` | **fortement recommandé** | le vent géostrophique, seule information sur le haut de colonne, est estimé à partir de la pression et de la température des capteurs |
| `relative_humidity_pct`, `wind_w_m_s`, `air_density_kg_m3`, `quality` (0–1) | non | |

**Cible ponctuelle** (radiosondes, mâts) : `timestamp_utc`, `lat`, `lon`, `z_asl_m` ou
`z_agl_m`, puis le vent en (u, v) ou en (vitesse, direction).

**Cible maillée** : NetCDF avec u, v (w et la densité facultatifs), latitude, longitude,
temps et une hauteur au-dessus de la mer (coordonnée ou champ 3D). Les noms se donnent
avec `--grid-vars`.

## 3. Ce qu'il faut fournir (bloquant, dans cet ordre)

1. **La position réelle du domaine** : lat/lon du coin sud-ouest et orientation
   (Tukhnanutha). Sans elle, rien ne se place.
2. **Le relief et l'occupation du sol réels de ce domaine.** `aneo_grid_points.csv` vient
   du générateur. S'il n'est pas extrait d'un vrai modèle numérique de terrain, il faut le
   refaire à partir d'un MNT (ex. Copernicus DEM, 30 m) et d'une carte d'occupation du sol
   (pour la rugosité z0). L'outil n'existe pas encore : c'est un chantier de 1 à 2 jours
   une fois les sources choisies.
3. **La liste réelle des capteurs ANEO** : coordonnées et hauteurs, même sans mesures. Elle
   suffit pour évaluer la v7 à la disposition réelle (étape A ci-dessous).
4. **Les mesures ANEO réelles** si elles existent : période (au moins un an, pour couvrir
   les saisons), fréquence, format, qualité, et surtout **T et p**.
5. **Une cible pour l'altitude** : radiosondes de Tunis-Carthage (en général 00 et 12 UTC ;
   station et archive à vérifier auprès de l'INM ou d'une archive publique), et/ou
   simulations WRF.
6. **Des comptes** : Copernicus CDS (gratuit, pour ERA5 / CERRA) — **créé le 14/10**. La clé se met dans `~/.cdsapirc` par l'utilisateur, jamais dans le dépôt ni le salon. La couverture de la
   Tunisie par CERRA est à vérifier sur sa grille.
7. **Du calcul hors Colab** pour WRF et pour un service permanent. À chiffrer par l'agent
   de Rayen, avant tout lancement.

## 4. Plan par étapes, avec décision à chaque marche

| Étape | Prérequis | Ce qu'on obtient | Ordre de grandeur |
|---|---|---|---|
| **A. v7 à densité réaliste** sur les 120 instants | rien (déjà prêt) ; liste réelle des capteurs pour la vraie disposition | combien la v7 perd sur un vrai réseau | 30 min de GPU |
| **B. v7 face au réel** | A + mesures ANEO réelles + radiosondes | premier score sur des données réelles (cible partielle) | 1 à 2 jours, une fois les données reçues |
| **C. Entraînement à masques réalistes** (synthétique) | A | un modèle robuste à un réseau épars (v9) | ~5–6 h de GPU A100 |
| **D. Campagne WRF à ~100 m**, forcée ERA5/CERRA, sur dates représentatives | 1, 2, CDS, calcul cloud | vérité 3D dense **réelle-physique** pour le pré-entraînement | à chiffrer : simulations lourdes, multi-cœurs |
| **E. Pré-entraînement WRF puis affinage sur observations réelles** | D + B | modèle ancré dans le réel (cible partielle, déjà prise en charge) | quelques heures de GPU par cycle |
| **F. Temps réel** | B, un flux de mesures, un hôte permanent | reconstruction en continu via `/api/reconstruct/stations` | collecteur à écrire (petit) ; hôte GPU permanent à chiffrer |

**Changements de modèle à prévoir**, que le code ne fait pas encore :

- **Entrée éparse et irrégulière** : entraîner avec la disposition réelle des capteurs (ou
  des treillis), pas avec des fractions d'une couche dense.
- **Entrée en altitude** : un canal « profil vertical » (radiosonde) ; le haut de colonne est
  aujourd'hui limité par l'information disponible depuis la surface.
- **Nouvelles statistiques de normalisation** sur les données réelles, donc un nouveau
  modèle. La v7 sert de point de départ, pas de modèle final.
- **Protocole DSO** à redéfinir avec Tukhnanutha sur une vérité réelle. Le seuil de 0,5 m/s
  a été mesuré sur du synthétique.

## 5. Le « temps réel », concrètement

Côté code, c'est prêt : `POST /api/reconstruct/stations` reçoit un instant de mesures et
renvoie le volume en ~0,1 s sur GPU. Il manque :

- **un flux de mesures** : API, FTP ou MQTT du réseau ANEO (à obtenir), et un petit
  collecteur qui pousse les mesures toutes les N minutes ;
- **un hôte permanent**. Colab s'arrête après inactivité ou au bout de sa durée de vie
  (atom.zelqane.com est tombé ainsi le 13/10), et chaque relance demande une autorisation
  Google manuelle. Une petite machine GPU louée, ou ce poste en CPU (15–25 s par instant),
  tiendrait en continu ;
- **le domaine géoréférencé** (point 1 de la section 3), sans lequel l'API refuse avec un
  message explicite.
