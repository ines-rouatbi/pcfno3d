# Lot M6 — publication (critères HIS1 et DE3)

*Modules : `src/pcfno3d/publish.py` (instantané du contrat, dépôt local, client de
publication) et `src/pcfno3d/export_noeuds.py` (NetCDF et CSV des nœuds virtuels).
Script GPU : `scripts/m6_rejeu_colab.py`. CLI : `pcfno3d publish`. Contrat ATOM v1
**révision 2** (`contract/DECISIONS-rev2.md`, `contract/README.md`, `openapi.yaml`,
`schemas/*.json`, `tools/publish.mjs`, `tools/validate.mjs`) et Worker
`agent/rayen-audit-atom/f1-r2` (`src/v1.js`). Copie des schémas et de `validate.mjs` :
`tests/data/atom_rev2/` (branche `contrat-v1`, commit `b7b13b8`).*

## Statut : prêt, **publication réelle en attente du jeton, de R2 et d'une session GPU**

Les tests CPU (`tests/test_publication.py`) construisent des instantanés du mini-jeu, les
valident avec les schémas Python **et** avec `validate.mjs` du contrat, puis les publient
vers un Worker simulé (`tests/worker_simule.py`, la partie publication de `v1.js` en
Python) en coupant la connexion à différents moments. Aucun appel réseau hors
`127.0.0.1`, aucun jeton réel.

## Ce qui est publié

Un instantané = un dossier de **disposition identique aux fixtures** du contrat
(`contract/fixtures/<cas>/`) :

```
snapshots/<id>/
  fields/u.f32 v.f32 w.f32 speed.f32            Float32 LE, nz·ny·nx, NaN sous min_agl_m
  fields/sigma_speed.f32 sigma_u/v/w.f32         (M2 actif)
  fields/coverage.u8 reliability.u8 status.u8    (M2 actif) 0..254 → 0..1, 255 = absent
  snapshot.json quality.json alerts.json stations.json wind-farm.json data-quality.json
  profiles/<eolienne>.json windrose/<eolienne>.json   (parc M5 ; profils si M2 actif)
  models/<sha8>/meta.json terrain.f32 model-manifest.json   fichiers PAR MODÈLE (D3)
  publish-manifest.json latest.json times.json   jamais publiés dans l'instantané
```

| Décision | Application |
|---|---|
| B1 | `grid.min_agl_m` lu dans `meta.json` du générateur (`noeuds_3d.min_z_rel_m`, 5 m). Le masque publié est **recalculé** sur le relief publié (float32) : `z_asl − relief < min_agl_m` ⇔ NaN / 255. Un nœud valide pour le contrat sans valeur du modèle est une **erreur** (rien n'est inventé) ; un nœud du générateur sous `min_agl_m` (arrondi) devient absent et est compté (`info["noeuds_retires_min_agl"]`). |
| B2 | `x0_m = y0_m = 0`, nœud (i, j) en `(i·dx, j·dy)`, `domain.x_max_m = (nx − 1)·dx`. |
| B4 | `network.kind = alerts.reseau` (treillis). `stations.json` = capteurs **signalés** par une alerte ouverte (dérive, muette, exclue), identifiant `c<i>_<j>`. `sensors_visible` = mailles avec une mesure utilisée ; `sensors_total = visible + muettes + exclues`. |
| B5 | Lot inactif ⇒ `null`, jamais 0 : sans M2, `uncertainty = {aucune, 1, false}`, parts et médianes `null`, `share_high_reliability: null`, ni σ ni statut publiés, pas de profil ; sans M4, `physics`, `stations_summary` et `kpi.open_alerts` `null`. |
| D1 | `model_sha256` = sha256 des octets du fichier de poids (lu une fois par processus) ; `norm` = sha8 de `norm_stats.json` ; `snapshot_id = <stem>_<sha8>`, stem du cache `JJJ_HHMMSS` pour le rejeu, `live_AAAAMMJJ_HHMMSS` pour le direct (CSV). |
| D3 | `models/<sha8>/meta.json` et `terrain.f32` publiés **une fois** par modèle et par processus ; le manifeste d'instantané ne les contient pas. |
| D4 | `georef` : plan tangent équirectangulaire au nœud (0, 0), `m_per_deg_lat = 110574`, `m_per_deg_lon = 111320·cos(lat0)`, `bounds` des 4 coins, `crs = plan-tangent-local`. Sans `real.origin_*`, ancrage **supposé** 36,0 N 9,7 E (celui de `configs/colab.yaml`), `suppose: true`. |
| D5, D6 | Profils : nœud (i, j) de l'éolienne, `z_terrain_m = terrain[j·nx+i]`, valeurs = champs publiés arrondis, `p50 = speed`, `p10/p90 = speed ∓ 1,2816·sigma_speed` (p10 ≥ 0). |
| D7 | `status` de M2 (capteur visible à ≤ 15 m), y compris sur données synthétiques. |
| D8 | Alertes de M4 (journal JSONL : `raised_at` = première levée, `ouverte`/`close`). |
| D10 | `kpi.hub_wind_*` = `windfarm.hub_wind_kpi` sur le domaine à `wind_farm.hub_height_agl_m`. |
| D11 | `physics` = bloc de M4 (`alerts.physique`), `null` si une valeur est indéfinie. |
| D12 | `data-quality.json` : rapport du contrat M3 pour un CSV (période, verdict, muettes, hors plage, interpolées) ; pour un instant du cache, l'instant lui-même est le « fichier reçu ». |
| D13 | Un fichier par variable, tous niveaux, codage décrit dans `meta.encoding`. |
| D15 | `versions.data_kind` : `synthetique` (défaut), `reel` ou `test`, choisi explicitement. |
| D16 | `data_status = degrade` si un capteur est muet, exclu ou en dérive, ou si les capteurs disponibles sont insuffisants ; sinon `complete`. |
| m4 | Directions arrondies **avant** le modulo (`round_direction`), `null` sous 0,2 m/s. |

Les JSON sont écrits en UTF-8, LF, indentation 2, avec `allow_nan=False` : un NaN dans un
document fait échouer la construction au lieu de produire un JSON invalide.

## Publication (protocole de `publish.mjs`)

`ClientPublication(url).publier(dossier)` :

1. vérifie que `/v1/meta/<sha8>` n'existe pas déjà **avec un contenu différent** (un modèle
   est immuable : même poids ⇒ mêmes champs, même `data_kind`) ;
2. `PUT /v1/publish/model/<sha8>/files/<path>` (meta.json, terrain.f32) puis
   `POST /v1/publish/model/<sha8>/commit` — une fois par modèle et par processus ;
3. `PUT /v1/publish/<id>/files/<path>` pour chaque fichier du manifeste, en-tête
   `X-Content-SHA256`, chemin encodé (`/` → `%2F`), 8 Mo maximum (vérifié avant l'envoi) ;
   chaque fichier est relu et comparé au manifeste avant l'envoi ;
4. `POST /v1/publish/<id>/commit` avec le manifeste ; `latest` ne bascule **qu'ici**, après
   vérification par le Worker de la présence, de la taille et du sha256 de chaque fichier ;
5. `POST /v1/heartbeat`.

Reprises : 3 tentatives sur erreur réseau ou 5xx (PUT et commits, idempotents côté
Worker) ; un 4xx arrête tout. Refus avant toute requête : `http://` vers un hôte autre que
la boucle locale (le jeton passerait en clair), instantané `data_kind = test` vers
`atom.zelqane.com` (même règle que `publish.mjs`).

### HIS1 — atomicité

| Coupure | Effet (testé) |
|---|---|
| au milieu des PUT de l'instantané | exception, aucun commit, `latest` et `index/times.json` inchangés ; relancer termine la publication |
| sur le commit, avant traitement | `latest` inchangé |
| commit traité, réponse perdue | la reprise reçoit `unchanged: true`, `latest` = le nouvel instantané |
| fichier modifié après construction | refusé par le client avant l'envoi ; un manifeste qui annonce un fichier non déposé est refusé par le Worker (422) |
| écriture locale interrompue | ni dossier d'instantané ni dossier temporaire ; `latest.json` local inchangé |

Le dépôt local suit la même règle que le Worker : dossier écrit sous un nom temporaire puis
renommé, `index/times.json` ensuite, `latest.json` **en dernier** (`os.replace`), et
`latest` ne recule jamais (un instantané plus ancien publié après n'y touche pas).

### Jeton

Jamais en argument, jamais imprimé, jamais écrit : variable d'environnement
`ATOM_PUBLISH_TOKEN` (la même que `publish.mjs`), ou secret Colab du même nom pour le
script de rejeu. Le `repr` du client et ses exceptions ne le contiennent pas (testé). Il
est remis de personne à personne par Rayen.

## Historique et rejeu (`/v1/times`)

`DepotLocal(racine)` tient `index/times.json` (du plus récent au plus ancien, comme le
Worker), `index/vent_points.json` (vent au moyeu des éoliennes, pour les roses) et
`latest.json`. Les roses des vents d'un instantané sont calculées sur les instants **déjà
publiés jusqu'à lui** : jamais sur le futur, `n_valid` croît de 1 à N (l'interface signale
< 100).

```bash
# local (défaut), les 120 instants synoptiques de test dans l'ordre chronologique
python scripts/m6_rejeu_colab.py -c colab
# puis publication vers le Worker (jeton en secret Colab ou variable d'environnement)
python scripts/m6_rejeu_colab.py -c colab --remote https://atom.zelqane.com [--purger]
```

Reprise : un instant déjà écrit (local) ou déjà commité sur ce Worker
(`index/publies_distant.json`) est sauté sans recalcul. Au redémarrage, le suivi de
dérive de M4 repart de zéro (le journal, lui, est relu).

## CLI

```bash
pcfno3d publish --stem 004_120000 --data-kind synthetique [--exports] [--dso6 resume.json]
pcfno3d publish --csv mesures.csv                       # direct : contrat M3, stem live_…
ATOM_PUBLISH_TOKEN=… pcfno3d publish --stem … --remote https://atom.zelqane.com
```

Dossier par défaut : `run_dir/publication`. M2, M4 et M5 sont pris dans la configuration
(`inference.reliability`, `alerts.enabled`, `wind_farm.turbines`).

## DE3 — exports NetCDF et CSV (spec 8.4)

`export_noeuds.exporter(inst, geom, dossier)` (ou `--exports`) écrit, **hors** du dossier
publié, `exports/<id>/<id>.nc` et `exports/<id>/<id>_noeuds.csv`, avec exactement les
valeurs publiées (même masque B1, mêmes float32).

- **NetCDF** : dimensions `(altitude, y, x)` ; `u, v, w, vitesse, direction,
  hauteur_sol` (+ `sigma_u/v/w/speed, couverture, fiabilite, statut, classe_fiabilite`
  avec M2) ; coordonnées `x, y, altitude, lat(y, x), lon(y, x)` (projection D4),
  `z_terrain` ; unités et `standard_name` CF ; versions (`version_modele`,
  `model_sha256`, `version_donnees`, `data_kind`, `version_normalisation`, contrat),
  projection et ancrage en attributs globaux. netCDF4 (compressé) s'il est installé,
  sinon scipy (NetCDF3, dépendance du paquet).
- **CSV** : colonnes de la spec 8.4 dans leur ordre (`node_id, x_m, y_m, altitude_m,
  hauteur_sol_m, timestamp_utc, u, v, w, vitesse, direction, statut, source, sigma_u,
  sigma_v, sigma_w, couverture, fiabilite, classe_fiabilite, version_modele,
  version_donnees, version_normalisation`). `node_id` = point_id du nœud 3D du générateur
  (unique) ; `source` = capteur `c<i>_<j>` pour un nœud mesuré, « modèle PC-FNO3D »
  sinon ; `version_modele = <nom>@<sha8>`. Inconnu = cellule vide.
- « Chaque export est vérifié par un schéma de sortie avant d'être livré » : le CSV passe
  un schéma Pandera (`schema_csv` : colonnes exactes et ordonnées, plages, valeurs
  permises), le NetCDF est relu et contrôlé (`verifier_netcdf`) ; le fichier n'apparaît
  sous son nom final qu'après la vérification.

Taille v7 (12 × 250 × 500) : un champ Float32 fait 6,0 Mo (< 8 Mo), un instantané complet
avec M2 ≈ 52,5 Mo (24 Mo sans M2) ; le CSV ≈ 849 000 lignes (~130 Mo). Les exports ne
passent pas par R2 (> 8 Mo, pas de route Worker) : `assets.export_netcdf/csv` restent
`null`.

## Écarts restants avec le contrat rév. 2

1. **Profil sans M2.** Le schéma veut un intervalle « tout null si inconnu », mais
   `validate.mjs` exige `p50 = speed` et `p10/p90 = speed ∓ 1,2816·sigma_speed`, ce qui,
   avec `sigma_speed = null`, impose un intervalle de largeur nulle. Sans M2, M6 ne publie
   donc **aucun profil** (`profile: false`). À trancher : le validateur devrait accepter un
   intervalle null quand `sigma_speed` est null.
2. **Fichiers obligatoires même lot inactif.** `validate.mjs` exige `alerts.json`,
   `wind-farm.json` et `data-quality.json`. Sans M4 : `alerts.json` = alertes brutes du
   parc M5 ou `[]` et `kpi.open_alerts: null`, mais le Worker calcule
   `times.open_alerts` depuis `alerts.json` et y met 0 au lieu de null. Sans parc :
   `wind-farm.json` vide (`turbines: []`, `rated_kw: 0`, puissance null).
3. **Réseau.** Unité « capteur » = maille du modèle portant une mesure (convention M4), pas
   le capteur à 50 m ; `stations.json` : `quality_score` toujours null, `last_seen` null pour
   un capteur muet ou exclu. Mode `stations_aneo` (liste complète) **non produit** : refus
   explicite.
4. **data-quality d'un instant du cache** : pas de rapport M3 (le cache ne passe pas par le
   contrat) ; « fichier reçu » = l'instant, interpolées = mailles sans mesure réelle.
5. **Pas de simulation** : `scenario` toujours null (BF45, BF46 non publiés).
6. **Roses des vents** seulement pour les éoliennes déclarées (pas de point `site`).
7. **Battement** : un `POST /v1/heartbeat` après chaque publication, pas de battement
   périodique de 60 s (à ajouter au service en direct).
8. **Géoréférencement** : `realdata.GeoRef` (import de stations réelles, éoliennes
   déclarées en lat/lon) utilise une sphère, 111 195 m par degré pour la latitude **et**
   la longitude (× cos lat0), contre 110 574 et 111 320·cos(lat0) pour D4. Sur le domaine
   v7 (24,9 × 49,9 km à 36 N), l'écart atteint ≈ 139 m vers le nord et ≈ 56 m vers l'est
   au coin opposé, soit jusqu'à 1,5 maille de 100 m. Les lat/lon **publiés** (meta,
   NetCDF) suivent D4 ; `realdata` n'est pas modifié (son alignement sur D4 est à faire
   dans un lot qui retestera l'import réel).
9. **Règle B1 sur v7** : vérifiée sur le mini-jeu (masque publié = masque du générateur) ;
   sur la géométrie v7, elle est vérifiée au premier instant de la session GPU (erreur
   immédiate et explicite si un nœud au-dessus de 5 m n'a pas de valeur).
10. **Stockage** : 120 instants v7 ≈ 6,3 Go dans R2 avec M2 (offre gratuite R2 : 10 Go).

## Pour publier pour de vrai

- **Humain (Rayen)** : déployer le Worker `f1-r2` avec le binding R2 `SNAPSHOTS` et le
  secret `PUBLISH_TOKEN` ; remettre le jeton hors salon ; le poser en secret Colab
  `ATOM_PUBLISH_TOKEN` (accès autorisé au notebook) ; valider le budget R2.
- **GPU (Colab)** : poids v7 + `norm_stats.json` + cache des instants de test ;
  `python scripts/m6_rejeu_colab.py -c colab` (local), `node tools/validate.mjs` sur
  quelques dossiers, puis la même commande avec `--remote https://atom.zelqane.com`.
  Facultatif : `inference.reliability` (JSON de calibration M2), `alerts.enabled` (+ seuils
  AN1), `wind_farm.turbines`, `--dso6` (verdict DSO6). Un modèle garde la même
  configuration (M2, `data_kind`) pendant toute sa publication.
