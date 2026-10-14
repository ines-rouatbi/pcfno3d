# Contrat de données ANEO (cahier des charges §8.1–8.3, critère DE1)

Le contrôle est branché à **deux** endroits : la commande `pcfno3d validate` (avec
quarantaine optionnelle) et l'upload de l'API. **`build-cache` ne l'appelle pas** : valider
les fichiers journaliers avant de construire le cache reste une étape manuelle
(`pcfno3d validate … --quarantine`). Le rapport n'est **joint à une reconstruction qu'à
l'upload**. Code : `src/pcfno3d/contract.py`. Seuils : constantes en tête du module (classe
`Seuils` pour les surcharger ; `Seuils.upload()` pour le profil de l'API).

```bash
pcfno3d validate fichier.csv [autres...] [--meta meta.json] [--quarantine DIR] [--copier]
                 [--report out.json] [--pas 10] [--tuile 1000] [--surface-seule]
```

Code de retour : **0** conforme (avertissements compris), **1** quarantaine, **2** rejet.
Sans `--meta`, la grille déclarée est `paths.data_dir/meta.json` de la configuration.
`--report` écrit le rapport (un objet pour un fichier, une liste pour plusieurs).
`--surface-seule` ignore les nœuds 3D (fichiers journaliers complets : ~1,35 M de nœuds
par instant). La lecture se fait **en flux** (lots de 16 Mo, `pyarrow.csv.open_csv`), chaque
lot filtré avant le suivant ; si un lot contredit les types inférés, relecture en flux tout
en texte. La relecture tolérante pandas (lignes au mauvais nombre de champs) n'est tentée
que sous 64 Mo (8 Mo compressé) ; au-delà, `fichier_illisible` avec le message de pyarrow.

**API** : `POST /api/reconstruct/upload` valide le CSV avec le **profil upload**
(`Seuils.upload()`) : une station **absente** du CSV (aucune ligne) est une panne connue,
que le modèle traite (capteur indisponible, DSO5) — elle n'est ni muette ni comptée dans
le seuil de rejet des stations valides, calculé alors sur les stations présentes.

| verdict | réponse |
|---|---|
| `rejet` | `422`, corps `{"detail": ..., "qualite": <rapport>}` |
| `quarantaine` | `200` : reconstruction + `statut_qualite: "quarantaine"`, `avertissements` (motifs compris), `qualite` |
| `conforme` | `200` : reconstruction + `statut_qualite: "conforme"`, `avertissements`, `qualite` |

Mesuré sur un instant du mini-dataset (800 stations) : 100 % et 90 % des stations →
conforme (avertissements) ; 50 % et 25 % → quarantaine (moins de 70 % des stations
déclarées valides), reconstruction servie ; ~90 % des stations présentes restent valides.

## Verdicts

| verdict | effet | déclenché par |
|---|---|---|
| `rejet` | n'entre pas | une erreur **bloquante** ; > 1 % des lignes **stations** hors plage ou de densité incohérente (les nœuds 3D, ~73 % des lignes d'un fichier journalier, ne diluent pas le seuil) ; < 30 % de couples (station, instant) valides (profil upload : des couples présents) |
| `quarantaine` | un humain décide | toute anomalie de mesure (stations ≤ 1 %, ou sur un nœud 3D) ; < 70 % de stations valides au pire instant ; > 25 % de valeurs interpolées ou signalées ; > 5 % de stations muettes ; > 20 % de tuiles sans station valide |
| `conforme` | entre | le reste ; **avertissements** si > 5 % d'interpolées, une station muette, figée ou absente, une tuile vide |

**Quarantaine** (`controler`, `mettre_en_quarantaine`, option `--quarantine DIR`) : le
fichier non conforme est **déplacé** (ou copié avec `--copier`) dans `DIR`, son rapport
écrit à côté (`<nom>.rapport.json`). Rien n'est supprimé ni écrasé : un nom déjà pris
devient `<nom>_1.csv`, `<nom>_2.csv`… Un fichier conforme reste où il est.

## Schéma (§8.1) — Pandera, validation paresseuse

Deux schémas Pandera validés en `lazy=True` : **toutes** les erreurs sont collectées.

1. **brut** (valeurs telles que lues) : les 16 colonnes présentes ; `point_id`,
   `timestamp_utc`, `day_index`, `hour_utc`, `mask_i_obs`, `source_type` non vides ;
   types lisibles (entier, réel, ISO 8601). Une valeur illisible n'interrompt rien : elle
   devient NaN pour la suite et est signalée une fois.
2. **typé** : plages ci-dessous, `mask_i_obs ∈ {0, 1}`, `source_type` dans la liste
   déclarée, puis les règles §8.2 ligne à ligne.

| colonne | type | plage |
|---|---|---|
| point_id | entier | grille déclarée |
| timestamp_utc | ISO 8601, UTC | multiple du pas attendu |
| day_index, hour_utc | entier | cohérents avec timestamp_utc |
| temperature_c | réel | −20 … 55 °C |
| potential_temperature_k | réel | 250 … 340 K |
| air_density_kg_m3 | réel | 0,9 … 1,4 kg/m³ |
| relative_humidity_pct | réel | 0 … 100 % |
| water_vapor_pressure_pa | réel | 0 … 7 000 Pa |
| pressure_pa | réel | 80 000 … 106 000 Pa |
| wind_u_m_s, wind_v_m_s | réel | −60 … 60 m/s |
| wind_w_m_s | réel | −10 … 10 m/s |
| aneo_quality_score | réel | 0 … 1 |
| mask_i_obs | entier | 0 ou 1 |
| source_type | texte | `meta.json["source_type"]` (défaut : les 3 codes du générateur) |

Les variables mesurées peuvent être vides (comptées dans `hors_plage.*.manquantes`) ;
sans u, v ou T, la station est « sans mesure » à cet instant.

## Règles de cohérence (§8.2)

| règle (`regle`) | gravité | définition |
|---|---|---|
| `fichier_vide` | bloquant | fichier vide ou en-tête seul |
| `separateur_invalide` | bloquant | `;`, tabulation ou barre verticale plus fréquent que `,` dans l'en-tête |
| `doublon` | bloquant | couple (point_id, instant) déjà vu ; instant comparé après lecture (`T`/espace, fuseau) |
| `point_id_inconnu` | bloquant | hors de 1…nx·ny (stations) et de `point_id_debut`…+`nombre` (nœuds) |
| `source_incoherente` | bloquant | source « nœud 3D » sur une station, ou l'inverse |
| `horodatage_decroissant` | bloquant | instant antérieur au maximum des lignes précédentes (égalité permise : un instant = un bloc) |
| `pas_incompatible` | bloquant | instant non multiple de **10 s** depuis minuit UTC (`--pas`) |
| `hour_utc_incoherent` | bloquant | `hour_utc` ≠ heure de `timestamp_utc` |
| `day_index_incoherent` | bloquant | un `day_index` pour deux dates, ou l'inverse (le couple majoritaire fait foi) |
| `jour_multiple` | bloquant | ligne d'une autre date UTC que la date majoritaire : un fichier = un jour, quel que soit son nom |
| `jour_du_fichier` | bloquant | nom `aneo_observations_day_XXX_YYYYMMDD_…` : date ≠ YYYYMMDD ou day_index ≠ XXX |
| `densite_incoherente` | mesure | \|ρ − p/(R_d·T_v)\| / (p/(R_d·T_v)) > **2 %** |
| `hors_plage` | mesure | valeur hors de la plage §8.1 |

Les stations **absentes** du fichier sont listées (`qualite.fraicheur.stations_absentes`).

**Densité.** T_v = T / (1 − (e/p)(1 − R_d/R_v)), R_d = 287,05, R_v = 461,5. `e` =
`water_vapor_pressure_pa` s'il est plausible (0 ≤ e < 0,2 p), sinon HR × e_s(T) (Magnus),
sinon air sec ; le rapport compte les lignes de chaque cas. **Tolérance 2 %** : ≈ 6 K sur T
ou 2 kPa sur p. Mesuré : < 0,04 % sur `conforme.csv` (même formule T_v que le générateur,
aux arrondis près), < 0,35 % sur le mini-dataset (densité d'air sec), alors que l'écart air sec / air humide atteint ~2 % à
35 °C saturé — d'où T_v.

## Indicateurs qualité (§8.3)

Calculés sur les **stations** (point_id < `point_id_debut`), couples (station, instant),
lignes exploitables seulement (point_id connu, instant lisible et aligné, sans doublon).

| clé `qualite.…` | définition |
|---|---|
| `stations_valides` | présente, mesurée, ni interpolée ni signalée, dans les plages, densité cohérente, non figée — part globale, au pire instant, par instant |
| `interpolees_ou_signalees` | `mask_i_obs = 0` **ou** source interpolée **ou** `aneo_quality_score < 0,5` ; `interpolees_non_declarees` = mask 1 mais source interpolée |
| `hors_plage.<variable>` | nombre, part, nombre de stations touchées, min, max, manquantes |
| `fraicheur` | `absences` (station sans ligne ou sans u/v/T à un instant) ; **station muette** = au moins une absence **ou** une valeur figée ; **valeur figée** = même valeur exacte de u, v ou T sur ≥ **3** instants consécutifs couvrant ≥ **30 min** (à 10 s, trois valeurs égales sont banales), le **calme** (vitesse < 0,5 m/s) ne comptant pas pour le vent ; profil upload : une station sans aucune ligne est « absente », pas muette ; retard de la dernière mesure |
| `couverture` | tuiles de **1 000 m** (`--tuile`) : part de tuiles sans station valide (pire instant), densité de stations valides par km² (moyenne, min, médiane, max), grille `part_valide_par_tuile` (lignes = y) pour l'indice de couverture des nœuds virtuels |

## Rapport JSON

```json
{
  "contrat": "aneo-8/1", "fichier": "panne.csv", "controle_utc": "…Z",
  "verdict": "quarantaine",
  "motifs": ["stations muettes 25.0 % (> 5.0 %)", "…"],
  "avertissements": ["2 station(s) a valeur figee"],
  "resume": {"lignes": 108, "lignes_stations": 84, "lignes_noeuds_3d": 24, "instants": 4,
             "premier_instant": "…", "dernier_instant": "…", "jour_du_fichier": null,
             "lignes_en_erreur_bloquante": 0, "anomalies_de_mesure": 0},
  "erreurs": [{"regle": "type_reel", "section": "8.1", "gravite": "bloquant",
               "message": "reel attendu", "colonne": "temperature_c", "nombre": 1,
               "lignes": [2], "exemples": ["abc"]}],
  "qualite": {"stations_valides": {}, "interpolees_ou_signalees": {}, "hors_plage": {},
              "densite": {}, "anomalies": {}, "fraicheur": {}, "couverture": {}},
  "seuils": {"tol_densite_rel": 0.02, "…": "…"},
  "taille_octets": 14466
}
```

`lignes` = numéros de ligne du fichier (en-tête = 1), 10 au plus ; `exemples` = 5 au plus ;
listes d'identifiants tronquées à 100 (`tronque`).

## Jeux de non-régression

`tests/data/contract/` (régénérés par `python scripts/make_contract_files.py`, grille
6 × 4 stations + 6 nœuds, 4 instants synoptiques) : `conforme.csv` → conforme ;
`corrompu.csv` → rejet (13 règles d'un coup) ; `panne.csv` → quarantaine (4 stations
muettes 3 instants, 2 anémomètres figés, 33 % d'interpolées). Tests : `tests/test_contract.py`.

## Écarts cahier ↔ fichiers réels (le contrat suit les fichiers)

1. **Noms de colonnes** : identiques au cahier et au générateur (`OBS_COLUMNS`), aucun renommage.
2. **Horodatage** : le cahier dit ISO 8601 ; les fichiers écrivent `2025-01-07 03:00:00`
   (espace, sans fuseau). Accepté, lu comme UTC ; `…T…Z` et les décalages aussi.
3. **Nœuds 3D dans le même fichier** : les fichiers journaliers mêlent stations et nœuds
   virtuels (`ANEO_VIRTUAL_NODE_3D`, `mask_i_obs = 0`, `point_id ≥ point_id_debut`). Le
   schéma s'applique à toutes les lignes ; les indicateurs §8.3, aux stations seules.
4. **source_type** : le cahier nomme « station physique / valeur interpolée / nœud virtuel
   3D » ; les fichiers portent `ANEO_PHYSICAL_SYNTHETIC_MULTITIME`,
   `ANEO_PHYSICAL_INTERPOLATED_MULTITIME`, `ANEO_VIRTUAL_NODE_3D`. Liste déclarée =
   `meta.json["source_type"]`. Le mini-dataset n'en déclare que deux et marque ses capteurs
   manquants par `mask_i_obs = 0` avec la source capteur.
5. **day_index** n'est ni le jour de l'année ni un décalage de dates (le générateur
   répartit ses jours sur l'année) : seule la bijection day_index ↔ date est vérifiée, plus
   le nom de fichier quand il suit le motif journalier.
6. **hour_utc** est l'heure tronquée ; minutes et secondes (épisodes minute / 10 s) sont
   dans `timestamp_utc`.
7. **Pas de temps** : le cahier ne le chiffre pas ; 3 h (synoptique), 60 s et 10 s
   (épisodes) dans le générateur, 6 h dans le mini-dataset → contrôle d'alignement sur 10 s.
8. **Densité** : le générateur utilise T_v, le mini-dataset l'air sec, et sa
   `water_vapor_pressure_pa` vaut 12 × HR (non physique) ; la tolérance de 2 % couvre les deux.
9. **aneo_quality_score** : le cahier ne donne qu'une plage ; « signalée » sous 0,5
   (même seuil que `real.min_quality`).
