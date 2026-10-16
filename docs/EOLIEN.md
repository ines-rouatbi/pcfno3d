# Lot M5 — éolien (calcul) et critère EOL1

*Module : `src/pcfno3d/windfarm.py`. Contrat ATOM **révision 2** (`DECISIONS-rev2.md` : D8,
D9, D10, B3, m8 ; `wind-farm.schema.json`, `windrose.schema.json`, `alerts.schema.json`,
`snapshot.schema.json` pour `kpi.hub_wind_*`). Copie des schémas utilisés par les tests :
`tests/data/atom_rev2/` (branche `agent/rayen-audit-atom/contrat-v1`, commit `b7b13b8`).*

## Statut : prêt, **verdict EOL1 en attente de la session GPU**

Les tests CPU (`tests/test_windfarm.py`, mini-jeu) vérifient les formules sur des cas
connus et le mécanisme EOL1 de bout en bout. Ils ne disent rien de la qualité de v7 pour
la production : le chiffre EOL1 vient de `scripts/m5_eol1_colab.py`, à lancer sur GPU.

## Activation

Strictement **opt-in** : `wind_farm.turbines: null` par défaut. Dans ce cas, rien ne change :
ni la sortie, ni le `.npz`, ni le temps d'inférence (`Volume.farm_context` vaut `None`,
testé). `/api/wind-farm` répond alors `404` et l'outil MCP renvoie une erreur explicite.

```yaml
wind_farm:
  turbines: configs/eolien/parc_demo.yaml     # ou --set wind_farm.turbines=...
```

| Commande | Effet |
|---|---|
| `GET /api/wind-farm?stem=…` (portée `read`) | le parc à un instant |
| outil MCP `production_eolienne(stem)` | même réponse |
| `pcfno3d wind-farm --stem … [--parc F] [--out F]` | même réponse, en JSON (n'écrase jamais `--out`) |
| `pcfno3d windrose [--parc F] [--split test] [--instants N]` | roses + Weibull + AEP par éolienne, dans un dossier neuf `run_dir/eolien_pcfno3d/<modèle>/rose_<horodatage>/` |
| `scripts/m5_eol1_colab.py` | mesure EOL1, dossier neuf `run_dir/eolien_pcfno3d/<modèle>/eol1_<horodatage>/` |

## Déclaration du parc

Fichier YAML (ou JSON). L'exemple `configs/eolien/parc_demo.yaml` est **marqué comme
exemple** (`exemple: true`, noms « (exemple) ») : trois éoliennes virtuelles aux positions
des fixtures du contrat, à remplacer par le parc réel.

| Champ | Contenu |
|---|---|
| `id` | `[A-Za-z0-9_-]+`, unique |
| `name`, `virtual` | libellé ; `true` par défaut |
| `x_m`, `y_m` **ou** `lat`, `lon` | position ; lat/lon exigent un domaine géoréférencé (`real.origin_*`) |
| `hub_agl_m` | hauteur de moyeu **au-dessus du sol** |
| `rotor_d_m` | diamètre du rotor ; le bas du rotor doit rester au-dessus du sol |
| `model` | clé de `power_curves` |
| `air_density` | facultatif : densité mesurée au moyeu (source `mesure`) |

Une déclaration ambiguë est **refusée**, sans correction silencieuse : identifiant invalide
ou en double, modèle inconnu, position hors du domaine, rotor qui toucherait le sol, lat/lon
sans géoréférencement.

## Conventions

- **Sol et mer.** Les niveaux du modèle sont en altitude au-dessus de la mer (ASL). Toutes les
  hauteurs éoliennes sont **au-dessus du sol** : `h_k = z_k − relief(j, i)`. Un moyeu à 90 m
  sur une colline à 400 m est à 490 m ASL, et c'est à 90 m au-dessus du sol qu'on lit le vent
  (test `test_hauteur_au_dessus_du_sol_et_pas_de_la_mer`).
- **Colonne.** Le nœud le plus proche de l'éolienne (même règle que `Volume.at_point`). Il est
  rendu dans `details[].noeud` (`i`, `j`, `x_m`, `y_m`, comme D5). Avec une maille de 100 m,
  un rotor de 90 à 126 m tient dans une ou deux mailles.
- **Nœuds valides seulement.** Seuls les nœuds au-dessus du relief (`geom.valid`) comptent.
  Une valeur sous le sol n'est jamais lue, même finie (testé avec 999 sous le relief).
- **Direction météo** : d'où vient le vent, `(270 − atan2(v, u)) mod 360`. `u > 0` donne 270
  (vent d'ouest) et `v < 0` donne 0 (vent du nord). Elle est indéfinie sous 0,2 m/s
  (`wind_farm.calm_ms`, comme `dir_from_deg` du contrat).
- **Arrondi avant le modulo** (règle m4 du contrat). Une direction publiée est
  `round(d, 1) mod 360` (`round_direction`) : 359,96 donne 0,0, jamais 360,0. Un veer publié
  est `((round(x, 1) + 180) mod 360) − 180` (`round_veer`) : 179,96 et −179,96 donnent −180,0,
  jamais 180,0 (`exclusiveMaximum: 180`). Les tests couvrent ces bornes. La même règle est
  appliquée aux directions arrondies hors du lot M5 : `/api/profile`, l'outil MCP
  `wind_profile`, `Volume.to_nodes_frame` et les cartes CSV de `export.py`.

## Formules

**Profil vertical.** Interpolation linéaire en hauteur au-dessus du sol entre deux nœuds
valides. La vitesse est interpolée pour elle-même, comme dans D10 ; `u` et `v` ne servent
qu'à la direction. Sous le premier nœud valide (le bas du rotor peut y tomber), on applique
une loi de puissance ajustée sur les deux premiers nœuds :

    v(h) = v_0 · (h / h_0)^α,   α = ln(v_1 / v_0) / ln(h_1 / h_0), borné à [0 ; 1]

Si α est incalculable, on prend 1/7. Dans la couche de surface, le vent croît avec la
hauteur. `details[].extrapolation_sous_premier_noeud` le signale. Au-dessus du dernier
niveau, l'éolienne est **indisponible**.

**REWS** (vent équivalent rotor, IEC 61400-12-1:2017, pondéré par l'aire, D9). Le disque est
découpé en N tranches horizontales de même hauteur (`wind_farm.rews_segments`, 20 par
défaut) :

    REWS = ( Σ_i (A_i / A) · v_i³ )^(1/3)
    A_i = F(y_{i+1}) − F(y_i),   F(y) = y·√(R² − y²) + R²·arcsin(y / R),   A = π R²

v_i est la vitesse au centre de la tranche. Le terme de veer de l'IEC 2017,
`cos(φ_i)`, n'est **pas** appliqué : le contrat demande la moyenne du cube. Tests :
profil uniforme (REWS = v) ; profil linéaire, où la formule exacte est

    (a³ + ¾ a b² R²)^(1/3)

puisque E[y²] = R²/4 sur un disque ; loi de puissance comparée à une intégration à 4 000
tranches.

**Cisaillement** (D9), entre le bas et le haut du rotor (h ∓ R) :

    shear_alpha = ln(v_haut / v_bas) / ln(h_haut / h_bas)

Il vaut `null` si une des vitesses est nulle.

**Veer** (D9) :

    veer_deg = ((dir_haut − dir_bas + 180) mod 360) − 180   ∈ [−180, 180)

Il est positif quand la direction tourne dans le sens horaire avec la hauteur (350° → 10°
donne +20°), comme la spirale d'Ekman de l'hémisphère nord. Il vaut `null` si une des
vitesses est un calme. Le calcul garde la valeur brute ; la publication arrondit avant le
modulo (`round_veer`, voir Conventions).

**Densité de l'air au moyeu** (`air_density`, `air_density_source`, D9) :

| Source | Règle |
|---|---|
| `mesure` | `air_density` fourni dans la déclaration |
| `extrapolation_surface` | ρ du capteur **visible** le plus proche (≤ `wind_farm.rho_radius_m` = 2 km), canal `air_density_kg_m3` de la surface, ramené au moyeu par ρ_ISA(z_moyeu) / ρ_ISA(z_capteur) (altitudes ASL, atmosphère standard de `realdata.isa`) |
| `atmosphere_standard` | ρ_ISA(z_moyeu ASL) si aucun capteur visible n'est dans le rayon |

Une densité de surface hors de [0,5 ; 1,6] kg/m³ est ignorée. Une éolienne indisponible a
`air_density: null` et `air_density_source: null`.

**Puissance** (IEC 61400-12-1, normalisation en densité des éoliennes à pas variable) :

    v_n = REWS · (ρ / ρ_ref)^(1/3),   P = courbe(v_n)

La courbe est interpolée linéairement, avec ρ_ref = 1,225 kg/m³. Test exact : ρ = 0,729 ρ_ref
et v = 10 m/s donnent v_n = 9 m/s.

**État : une seule règle, la même vitesse que la puissance.** L'état et la puissance se
lisent tous deux sur v_n, le REWS normalisé en densité :

| État | Condition | Puissance |
|---|---|---|
| `arret_vent_faible` | v_n ≤ cut-in | 0 |
| `arret_vent_fort` | v_n > cut-out | 0 |
| `production` | cut-in < v_n ≤ cut-out | courbe(v_n) **> 0** |
| `indisponible` | colonne sans nœud valide autour du rotor | `null` |

`PowerCurve` refuse une table qui ne va pas jusqu'au cut-out, ou qui a une puissance nulle
entre cut-in et cut-out. On a donc toujours « production ⇔ P > 0 », ce qui est testé sur
4 000 tirages et sur le cas qui était incohérent : moyeu à 24,8 m/s et ρ = 1,30 kg/m³ donnent
v_n = 25,3 m/s, donc `arret_vent_fort` à 0 kW, et non plus « production » à 0 kW.

Le choix assumé : un vrai contrôleur décide le cut-in et le cut-out sur l'anémomètre de
nacelle, sans normalisation. Ici, état et puissance suivent la même grandeur, pour qu'ils
ne se contredisent jamais. Le vent brut au moyeu reste publié (`hub_speed`) et sert au
niveau `critique` de `vent_fort_moyeu`. La valeur de v_n est dans
`details[].rews_normalise_ms`. Le parc donne `n_production` et `n_arret` (arrêts vent
faible et vent fort ; les indisponibles ne comptent ni dans l'un ni dans l'autre).

## Intervalles (D6, D9)

Ils ne sont disponibles que si la fiabilité M2 a tourné (`inference.reliability`). Sinon,
`p10` et `p90` valent `null`, `p50` est la passe servie et `uncertainty` vaut
`{method: "aucune", members: 1, calibrated: false}`.

- **Membres.** Ce sont les N passes masquées de M2, les mêmes que pour σ (même graine,
  mêmes masques). On ne les garde qu'aux colonnes des éoliennes, soit quelques Ko
  (`Volume.farm_context.members_uv`) : **aucune passe du modèle en plus**.
- **Vitesses** (`hub_speed`, `rews`). On applique la formule du contrat : p50 = passe
  servie, p10 = max(0, x − 1,2816 σ), p90 = x + 1,2816 σ. On calcule

      σ = k · √( (1/N) Σ (x_m − x_0)² )

  pour la valeur au moyeu ou le REWS de chaque membre, où k est le facteur calibré de la
  vitesse de M2, interpolé à l'altitude ASL du moyeu (k = 1 en `non_calibre`).
- **Puissance d'une éolienne.** C'est l'image de l'intervalle de vitesse par la courbe :
  min et max de P sur l'intervalle [p10, p90] du REWS, avec la même règle d'état. Comme la
  courbe n'est pas linéaire (plateau, cut-out), un intervalle symétrique serait faux.
- **Parc** (D9). Ce sont les quantiles 10 % et 90 % de la puissance du parc **par membre**,
  pas la somme des quantiles des éoliennes : les erreurs d'éoliennes voisines sont
  corrélées. On ramène d'abord chaque membre à la dispersion calibrée,
  x_0 + k·(x_m − x_0). p50 = somme des puissances servies, et p10 ≤ p50 ≤ p90 est imposé.
  Avec 8 membres, ces quantiles restent grossiers.

## KPI du domaine (D10)

`kpi` = `{hub_wind_mean_ms, hub_wind_max_ms, hub_height_agl_m}`. C'est une statistique sur
**tout le domaine**, à `wind_farm.hub_height_agl_m` (100 m par défaut) au-dessus du sol.
`speed` est interpolée linéairement par colonne entre deux nœuds valides. Les colonnes où
cette hauteur n'est pas encadrée sont exclues, **sans extrapolation**. La valeur est `null`
si aucune colonne n'est utilisable. `hub_wind_kpi(volume, h)` est l'entrée de M6 pour
`snapshot.kpi`.

## Alertes (D8)

Elles suivent le format `alerts.schema.json`, avec `id` = sha8 de `kind|cible` (cible =
identifiant de l'éolienne ; la fixture `cisaillement|T-03` → `bdec1bd1` est testée),
`turbine_id` = cible, `location` = position déclarée, `status: "ouverte"` et
`raised_at` = horodatage du volume. **L'agrégation, la première levée et la clôture sont
le travail de M4**, qui remplace `raised_at` par la date de son journal.

`alert_id` et `iso_utc` sont **importés** de `pcfno3d.alerts` (M4) : il n'en existe
qu'une définition. Toutes les alertes passent par `_alert_meta`. `iso_utc(None)` lève une
erreur, et ce module ne lui passe jamais `None` : un volume sans horodatage ne produit
aucune alerte (`alerts.items` vide et `alertes_non_produites` dans la réponse).

**Alertes du parc et journal.** Les deux cas se distinguent ainsi :

- **Alertes activées** (`alerts.enabled`). Avec un parc déclaré (`wf.farm`),
  `MoteurAlertes.evaluer_modele` ajoute d'office les alertes `vent_fort_moyeu` /
  `cisaillement` du volume (`windfarm.farm_alerts`), qui passent par le journal M4 :
  - `raised_at` est la première levée ;
  - le statut passe d'`ouverte` à `close` ;
  - elles sont comptées dans `kpi.open_alerts`.

  Ce chemin sert `/api/alerts`, l'upload, l'outil MCP `alertes_actives` et
  `pcfno3d alertes`. `/api/wind-farm` et l'outil `production_eolienne` publient alors
  dans `alerts.items` **les items du journal** filtrés sur ces deux types
  (`alerts.items_externes`).
- **Alertes désactivées.** `/api/wind-farm` et `production_eolienne` gardent les alertes
  **brutes** de M5 : `status: "ouverte"` et `raised_at` = l'instant du volume, sans
  journal ni clôture.

La commande `pcfno3d wind-farm` (un instant isolé, sans historique) publie toujours les
alertes brutes.

| `kind` | Niveau | Condition (défauts, section `wind_farm`) |
|---|---|---|
| `vent_fort_moyeu` | `critique` | p50 au moyeu ≥ 35 m/s (`high_wind_critical_ms`) |
| | `important` | état `arret_vent_fort` (v_n > cut-out, même règle que l'état) |
| | `avertissement` | p90 de v_n (p50 sans M2) ≥ 0,9 × cut-out (`high_wind_warn_frac`) |
| `cisaillement` | `important` | α > 0,6 (`shear_important`) |
| | `avertissement` | α > 0,45 (`shear_warn`) |
| | `information` | α > 0,3 (`shear_info`, comme la fixture) |

Ces seuils sont des **choix d'ingénierie**, à régler avec l'exploitant. Le repère est le
cisaillement de conception de l'IEC 61400-1, α = 0,2 (profil de vent normal) ; 35 m/s est
de l'ordre du vent de référence des classes III (37,5 m/s). Le niveau `critique` n'existe
pas pour `cisaillement`.

## Weibull, AEP, rose des vents (B3, m8)

Ils sont calculés sur une **série d'instants**, une passe servie par instant
(`farm_windroses`, commande `windrose`), à la hauteur du moyeu de chaque éolienne.

- **Weibull** : maximum de vraisemblance sur les vitesses > 0. k est la racine (méthode de
  Brent) de

      Σ x^k ln x / Σ x^k − 1/k − moyenne(ln x) = 0

  et c = (moyenne x^k)^(1/k). Test : k et c retrouvés à 3 % et 2 % près sur 20 000 tirages
  (k, c) = (2,1 ; 9,4), (1,6 ; 6), (3 ; 12).
- **AEP** :

      AEP = 8 760 h × ∫ P(v_n) f_Weibull(v) dv   (pas de 0,01 m/s jusqu'à 60 m/s)

  v_n est normalisé à la densité moyenne du site sur la série. Le facteur de capacité vaut
  AEP / (8 760 × P_nominale). Test analytique : une courbe plate entre 0 et 25 m/s donne
  1 − exp(−(25/c)^k).
- **Rose.** Les secteurs sont de largeur égale (`wind_farm.sectors`, 12 par défaut), le
  premier centré sur 0, et `dir_center_deg` est la **provenance**. Classes de vitesse :
  `speed_bin_edges_ms`, la dernière ouverte. La direction est arrondie à 0,1° **avant**
  le classement, comme elle est publiée. `n_valid` compte **tous** les instants de
  vitesse connue, calmes compris (< 0,2 m/s, direction indéfinie). Le schéma n'a pas de
  champ « calme » (`additionalProperties: false`) : chaque calme est donc réparti **à
  parts égales entre les secteurs**, dans sa classe de vitesse (la première). Σ
  `frequency` = 1 (à l'arrondi 4 décimales près), même quand tous les instants sont
  calmes, et la somme des `speed_bins` d'un secteur vaut sa fréquence.
- **Sans instant valide**, il n'y a **pas de fichier**. `n_valid = 0` est autorisé par le
  schéma, mais la règle m8 (Σ = 1 ± 0,01) ne peut pas tenir, et `frequency` n'admet pas
  `null`. `farm_windroses` renvoie l'éolienne dans ses écartées, et `pcfno3d windrose`
  imprime pourquoi (« aucun instant valide… pas de fichier »). Cela arrive à une éolienne
  indisponible sur toute la série.
- `period` va du premier au dernier instant de la série (`…Z`). Sous 100 instants,
  l'interface doit signaler une statistique faible (contrat). Une AEP tirée de quelques
  jours **n'est pas** une production annuelle représentative : c'est le mécanisme qui est
  livré, la représentativité dépend de la série fournie.

## Courbes de puissance et sources

| Modèle | Source | Forme |
|---|---|---|
| `nrel-5mw-cp` | Jonkman J., Butterfield S., Musial W., Scott G. (2009), *Definition of a 5-MW Reference Wind Turbine for Offshore System Development*, NREL/TP-500-38060. Paramètres publiés : 5 MW, rotor 126 m, moyeu 90 m, cut-in 3 m/s, nominal 11,4 m/s, cut-out 25 m/s | **Approchée**, pas la table officielle : P = ½ ρ_ref A Cp v³ avec Cp = 0,4419 constant, calé pour atteindre 5 MW à 11,4 m/s ; P = 0 au cut-in |
| `generique-2MW` | table de démonstration des fixtures du contrat ATOM rév. 2 (`wind-farm.json`), rotor 90 m | Cp électrique de 0,28 à 4 m/s à 0,42 vers 8 m/s |

Les deux courbes sont testées sous la limite de Betz (Cp ≤ 16/27).

Autres références :

- IEC 61400-12-1:2017 (REWS, normalisation en densité) ;
- IEC 61400-1 (profil de vent normal α = 0,2, classes d'éoliennes) ;
- ISO 2533:1975, atmosphère type, via `realdata.isa`.

## EOL1 — définition

> **EOL1 : écart < 5 % entre la production du parc calculée depuis le champ reconstruit et
> celle calculée depuis la vérité 3D du générateur, sur les instants de test.**

    écart = |E_rec − E_vrai| / E_vrai,    E = Σ_t P_parc(t)

Le calcul est **le même** de part et d'autre :

- même parc, mêmes courbes ;
- même densité, tirée de la surface de l'instant ;
- même extrapolation.

Seul le champ de vent diffère : la reconstruction d'un côté, le cache `Y` (u, v) de
l'autre. EOL1 mesure donc l'effet de l'erreur de reconstruction sur la production, et non
la justesse de la courbe de puissance.

- **Instants.** Les instants synoptiques de **test**, ceux de DSO1, avec un pas constant :
  E est proportionnelle à l'énergie.
- **Capteurs.** Instants **et masques** sont exactement ceux de l'évaluation DSO1
  (`eol1_dataset`) : `ANEO3D` en évaluation sur la sélection synoptique de test, graine
  2000. Le masque de l'instant en position p est tiré par `default_rng(2000 + p)`, et
  cette position est conservée quand `--instants` sous-échantillonne. Le test le vérifie
  contre `eval_loader`, à l'identique. La reconstruction est la passe de production
  (`predict`), avec ce masque comme seul jeu de capteurs. Le verdict se lit à **100 %** des
  capteurs, le cas de production ; le script mesure aussi 50 %, soit `mask.keep_eval`, les
  masques mêmes de DSO1.
- **Seuil.** `wind_farm.eol1_max_rel` vaut 0,05.
- **Verdict indéfini.** Si la production vraie est nulle sur toute la série (calme), le
  verdict vaut `null`.
- **Diagnostics publiés à côté**, ils ne changent pas le verdict :
  - nMAE = moyenne |P_rec − P_vrai| / puissance installée ;
  - RMSE du vent au moyeu par éolienne ;
  - part des états identiques.

  Un écart d'énergie faible peut cacher des erreurs qui se compensent d'un instant à
  l'autre ; la nMAE le montre.

**Parc mesuré.** EOL1 dépend des positions : il se mesure sur le parc déclaré
(`--parc`, par défaut `configs/eolien/parc_demo.yaml`, qui est un **exemple**). Le verdict
officiel demande la déclaration réelle du parc (positions, hauteurs, modèles) par
Tukhnanutha.

**Sur CPU.** Le mécanisme est mesuré sur le mini-jeu :

- `eol1` donne une structure complète, sérialisable en JSON strict ;
- si la reconstruction est la vérité, l'écart est nul et le verdict conforme ;
- le script écrit dans un dossier neuf sans toucher aucun fichier existant.

### Session GPU (reste à faire)

```bash
!python scripts/m5_eol1_colab.py -c colab --parc <declaration_du_parc.yaml>
```

Le script écrit `eol1.json` (verdict) et `eol1_keep1.json` / `eol1_keep0.5.json` (détail
par instant) dans `run_dir/eolien_pcfno3d/<modèle>/eol1_<horodatage>/`. Il compte
120 instants × 2 fractions de capteurs, soit quelques minutes sur A100.

## Écarts restants avec le contrat rév. 2

- **Routes.** `/api/wind-farm` n'est pas la route du contrat (`/v1/wind-farm/{id}`, publiée
  par M6). La réponse **enveloppe** le document. `wind_farm` respecte exactement
  `wind-farm.schema.json` et `alerts` respecte `alerts.schema.json` (tous deux validés par
  les tests). Le reste (`kpi`, `details`, `uncertainty`, `membres`, `parc`,
  `timestamp_utc`) est hors schéma. M6 publie `wind_farm` tel quel.
- **Rose des vents.** Elle est produite en fichiers par `pcfno3d windrose`, sans route
  locale ; `/v1/windrose/{site_id}` est le travail de M6. Le schéma n'a pas de champ pour
  les calmes : ils comptent dans `n_valid` et sont répartis à parts égales entre les
  secteurs. Sans instant valide, aucun fichier n'est produit (voir plus haut).
- **Position.** `turbines[].x_m/y_m` est la position **déclarée**. Le calcul utilise le
  nœud le plus proche, donné dans `details[].noeud`.
- **`power_kw` d'une éolienne.** C'est l'image de l'intervalle de vitesse. Les quantiles
  sur membres ne valent que pour le parc, qui est ce que demande D9.
- **`power_curves`.** Elles sont publiées à la densité de référence 1,225 kg/m³. `rho_ref`,
  `rated_kw`, `cut_in_ms` et `source` restent dans la déclaration : le schéma des courbes
  n'admet que `speed_ms` et `power_kw`.
- **Extrapolation.** `hub_wind_*` (D10) n'extrapole pas sous le premier nœud, alors que
  les éoliennes le font (loi de puissance). Au même point, les deux valeurs ne diffèrent
  que si le moyeu est sous le premier nœud valide.
- **lat/lon.** La conversion utilise `realdata.GeoRef` : sphère de rayon moyen, origine au
  coin sud-ouest. D4 fixe 110 574 m/degré de latitude, contre 111 195 ici, soit 0,56 %,
  environ 140 m sur 25 km. L'alignement est à faire dans `realdata` (hors M5) avant
  d'accepter des positions en lat/lon pour le parc réel.
- **`zone_mal_couverte` et les autres `kind`.** Ils ne sont pas produits ici ; seuls
  `vent_fort_moyeu` et `cisaillement` sont du lot M5.
