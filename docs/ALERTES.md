# Lot M4 — alertes (critère AN1)

*Modules : `src/pcfno3d/alerts.py` (détection, agrégation, journal, moteur) et
`src/pcfno3d/an1.py` (simulation de pannes et de dérives). Besoins BF20 à BF23 ; critère
AN1 ; contrat ATOM **révision 2** (`DECISIONS-rev2.md` : D8, D11, B5, m9 ;
`alerts.schema.json`, `common.schema.json` : `alert_level`, `station_state` ;
`snapshot.schema.json` : `physics`, `kpi.open_alerts`, `stations_summary`).*

## Statut : v1 mesurée **non conforme** ; v2 prête, **en attente d'une seconde lecture du test**

La v1 (statistique « innovation ») a été mesurée sur GPU le 16/10/2026 : AN1 **non
conforme** (voir l'historique ci-dessous). La v2 (détecteur de cohérence spatiale, règle des
entrées sans le vent) a été conçue **après** cette première lecture du test. Son verdict
sera donc une **seconde mesure sur le même test**, pas une mesure indépendante : le choix
de la méthode a été influencé par ce que la première lecture a montré, même si aucun seuil
n'est réglé sur le test.

Les tests CPU vérifient le mécanisme sur le mini-jeu (modèle de 2 époques) : ils ne disent
rien de la qualité de détection sur v7. Le chiffre AN1 vient de
`scripts/m4_an1_colab.py`, à lancer sur GPU (voir plus bas).

## Historique des mesures AN1

Cette section n'est jamais supprimée ni réécrite : chaque mesure s'y ajoute, telle quelle.

### 1. Première lecture — v1, Colab L4, modèle v7, 16/10/2026

- **Protocole officiel** : 40 dérives et 10 pannes par jour, `e_ref` = 1 m/s.
- **Réglage** : statistique « innovation », κ = 12,75, réglé sur la validation (FDR cible
  2,5 %).
- **Validation** : détection des dérives 5,9 % (393 dérives), 0 fausse alerte sur
  1 874 250 capteurs sains.
- **Test (verdict)** : dérives détectées 5,2 % (386), fausses alertes 0 %. Par type :
  biais 2 m/s 6,7 %, biais 3 m/s ~19 %, gain et girouette ~0. Pannes (valeurs figées, hors
  verdict) 100 %. Dérive des entrées (unités) : détection 100 %, mais fausses alertes
  6,7 %.
- **Verdict : AN1 NON CONFORME** (seuils : détection ≥ 90 %, fausses alertes < 5 %).
- **Diagnostic** : avec 500 000 capteurs à 50 m (agrégés en mailles de 100 m), une station
  en dérive ne déplace presque pas la reconstruction. L'innovation (mesure − reconstruction
  sans le capteur) est noyée, et κ doit monter à 12,75 pour tenir le FDR.

### 2. Seconde lecture — v2 : à venir

Commande et protocole plus bas (« Session GPU »). Le résultat sera ajouté ici, marqué
« seconde mesure sur le test, après une première lecture (16/10/2026, non conforme) ».

## Un document par instant

`GET /api/alerts` renvoie exactement `alerts.schema.json` : `{"items": [...]}`. Chaque
alerte porte les 15 clés du contrat (`additionalProperties: false`), avec un niveau, une
cause et une recommandation en français. Le moteur agrège **toutes** les alertes, quel que
soit le lot qui les détecte :

| `kind` | Lot | Détection | Cible | Niveau |
|---|---|---|---|---|
| `hors_plage` | M3 | règles du contrat ANEO (`PLAGES`, densité), réappliquées par `stations_hors_plage` au fichier reçu | capteur `c<i>_<j>` | avertissement |
| `station_muette` | M3 | rapport du contrat (absences, valeurs figées) | capteur | avertissement |
| `stations_insuffisantes` | M3 | rapport du contrat ; sans rapport : part de mailles disponibles (treillis) ou de stations **attendues** (stations réelles, voir plus bas) | `domaine` | important (< 70 %), critique (< 30 %) |
| `capteur_derive` | M4 | v2 : mesure comparée à la médiane de ses voisins, corrigée du site (aucune passe) ; v1 : capteur masqué comparé à la reconstruction faite sans lui | capteur | important ; critique si l'écart ≥ 2 × seuil |
| `incoherence_physique` | M4 | résidu de continuité de l'instant > seuil DSO2 ; condition au sol (seuil à régler) | `domaine:masse`, `domaine:sol` | important ; critique si ≥ 10 × seuil ; sol : avertissement |
| `derive_entrees` | M4 | statistiques des entrées vs entraînement (`norm_stats.json`) | `domaine:<variable>` | avertissement ; important si ≥ 2 × seuil |
| `zone_mal_couverte` | M2 | indice de couverture moyen < 0,2 sur une maille de 2 km | maille `m<ix>_<iy>` | information |
| `vent_fort_moyeu`, `cisaillement` | M5 | **entrée externe** (`alertes_externes`), alimentée par `windfarm.farm_alerts` quand un parc est déclaré | éolienne | donné par M5 |

**Identifiant (D8).** `id = sha8("kind|cible")` (`alert_id`). Il est stable d'un instant
à l'autre, et c'est lui qui relie les instants dans le journal. Si deux sources donnent la
même cible, `agreger` garde la plus grave.

**Ordre.** Les ouvertes d'abord, puis par niveau décroissant, par type et par cible.

**Plafond.** Au plus `alerts.max_par_type` (100) alertes d'un même type par instant, les plus
fortes d'abord. Le rapport M3 tronque déjà ses listes d'identifiants à 100.

## Capteur en dérive (`capteur_derive`)

Quatre statistiques de décision sont disponibles (`alerts.derive_statistique`). Le choix et
les seuils se font sur la **validation** (`scripts/m4_an1_colab.py`) :

| Statistique | Version | Compare | Passes du modèle |
|---|---|---|---|
| `spatiale` (défaut) | v2 | la mesure à la médiane de ses voisins, corrigée du site | **aucune** |
| `innovation` | v1 | la mesure à la reconstruction faite sans le capteur | `derive_cote²` = 9 |
| `influence` | v1 | la reconstruction servie à celle faite sans le capteur | 9 |
| `combinee` | v2 | moyenne des z de `spatiale` et d'`innovation` | 9 |

### v2 : cohérence spatiale (`ecarts_spatiaux`)

**Pourquoi.** Un capteur à 50 m parmi 500 000 ne déplace presque pas la reconstruction :
l'écart entre la mesure et la reconstruction faite sans lui est surtout fait de l'erreur du
modèle, pas de la dérive (première lecture, ci-dessus). Les voisins immédiats d'un capteur,
eux, mesurent presque le même vent : c'est contre eux qu'une dérive ressort.

**Estimation par les voisins.** Pour chaque maille-capteur visible, sur les composantes u et
v :

    d = mesure − médiane des voisins          (carré de 2R + 1 mailles, centre exclu)
        [− écart de site attendu]             (si un modèle de sites est chargé)

Il faut au moins `derive_min_voisins` = 3 voisins disponibles ; sinon, la maille n'est pas
évaluée (NaN), jamais inventée. La médiane résiste à un voisin lui-même en dérive.

**Correction de site (`ModeleSites`).** L'écart naturel d'un capteur à ses voisins n'est
pas un bruit. Il vient de son site (relief, rugosité, vents de pente qui s'inversent entre
le jour et la nuit) et se répète d'un jour à l'autre. Par maille, une régression linéaire
(ridge, λ = 1) de cet écart sur `[1, u voisins, v voisins, sin h, cos h]` l'apprend sur les
**jours d'entraînement seulement** :

- `ajuster_sites_entrainement` refuse un instant de validation ou de test ;
- le fichier `sites_R<r>.npz` porte `jeu: "entrainement"`, et `ModeleSites.charger` refuse
  tout autre jeu ou une grille différente.

Le modèle fait partie du réglage, comme `norm_stats.json`.

**Échelle locale.** `s` = médiane des `|d|` des voisines (carré de rayon
`derive_rayon_echelle`) / 1,1774, avec un plancher de 0,05 m/s. Le facteur vient de la
médiane de `|d|` pour un bruit gaussien 2D d'écart-type 1. Puis `z = |d| / s`. Le seuil suit
ainsi la difficulté locale : une crête ou une vallée s'écarte naturellement plus de ses
voisines qu'une plaine.

**Persistance vectorielle.** `SuiviDerive(mode="vecteur")` cumule les écarts **en vecteur**
sur les `derive_fenetre` derniers instants du jour :

    a = |moyenne des d (u, v)|        z = a / moyenne de s

Un biais, un gain ou une girouette décalent la mesure dans une direction qui persiste, alors
que le bruit se compense en partie. On lève l'alerte si `n ≥ derive_min_instants`,
`z ≥ κ` et `a ≥ derive_min_m_s` = 0,5 m/s. La valeur publiée est `a` (m/s) ; le seuil
publié est `max(κ · s̄ locale, 0,5)`, ce qui garantit `value ≥ threshold`.

**Coût.** Aucune passe du modèle : deux médianes de voisinage par instant (tri numpy
vectorisé), environ 1 s par instant sur les 125 000 mailles de v7, sur le CPU d'ines-pc.

**Exploration avant réglage (ines-pc, CPU).** Les choix de la grille viennent d'une
exploration sur le cache local de v7 (jours 2 et 3 d'entraînement, jour 14 de validation),
avant toute mesure GPU de la v2 :

- l'écart à la médiane des voisins (R = 2) est fortement corrélé au relief relatif
  (corrélation 0,77 un jour de vent fort) ;
- un modèle de sites appris sur seulement deux jours d'entraînement divise par 2,4 le
  quantile 99,99 % de l'écart sur le jour de validation (1,38 → 0,53 m/s) ;
- avec 40 dérives injectées sur ce jour de validation (4 répétitions, 94 dérives d'au moins
  1 m/s), la détection au seuil qui laisse une fausse alerte par jour passe d'environ 98 %
  (R = 2, sans site) à 100 % (avec site).

Ce ne sont pas des mesures AN1 (un seul jour, pas la chaîne complète). Pour être exact : la
toute première exploration du bruit spatial a aussi affiché les quantiles du jour 17 (test),
avant que l'exploration ne soit limitée à l'entraînement et à la validation. Aucun paramètre
n'en a été tiré.

### v1 : capteur masqué (`ecarts_capteurs`)

**Principe (PLAN.md).** On masque le capteur, on reconstruit sans lui, puis on compare.

**Groupes en treillis.** Un leave-one-out par capteur coûterait une passe par capteur, soit
125 000 sur v7. On masque donc des **groupes** : la maille (j, i) appartient au groupe
`(j mod c)·c + (i mod c)`, avec `c = alerts.derive_cote = 3`, ce qui fait **9 passes** par
instant. Chaque passe retire 1/9 des capteurs, tous distants d'au moins 3 mailles, si bien
que leurs voisins restent visibles. C'est dans la distribution d'entraînement
(`mask.keep_train` = [0,2 ; 1,0]). Les caractéristiques géostrophiques et thermiques sont
**recalculées** sur les capteurs gardés (`WindFieldModel._member`, comme pour l'ensemble M2).

**Statistique.** On compare à l'altitude du capteur (`interp_colonne` : interpolation
linéaire entre niveaux valides ; sous le premier nœud valide, on prend la valeur de ce nœud,
sans extrapolation) :

- **innovation** : `d = mesure − reconstruction sans le capteur` ;
- **influence** : `d = reconstruction servie (avec) − reconstruction sans le capteur`.

L'influence annule l'écart de représentativité, c'est-à-dire un capteur à ~25 m du sol
alors que le premier nœud valide est parfois 100 m plus haut en relief. Mais elle suppose
que le modèle suive assez ses entrées localement.

**Normalisation robuste, par instant.**

    d_c = d − médiane(d)          (une dérive commune à tous n'est pas un capteur)
    s   = max(1,4826 · MAD(d_c), 0,05 m/s)
    z   = |d_c| / s               (écarts robustes)      a = |d_c|  (m/s)

**Persistance (mode « moyenne »).** z moyen et écart moyen sur la fenêtre, mêmes conditions
d'alerte que la v2.

### Commun aux quatre statistiques

La persistance n'est **mise à jour que pour un instant postérieur au dernier suivi**. Un
instant antérieur est évalué sans dérive (`derive.evaluee: false`), pour qu'une consultation
ne fausse pas la fenêtre. La fenêtre vit en mémoire : un redémarrage repart d'une fenêtre
vide, alors que le journal, lui, est relu. Sans `membre` (pas de passe possible), les
statistiques v1 ne sont pas évaluées ; la v2 n'en a pas besoin.

**Granularité.** Le modèle voit la surface par mailles de 100 m (`pool_surface` : moyenne
des 2 × 2 capteurs à 50 m). Le « capteur » détecté est donc la **maille-capteur**, désignée
par son capteur sud-ouest `c<i·stride>_<j·stride>` ; la cause le précise. Un seul capteur à
50 m qui dérive de b décale la maille de b/4.

## Dérive des entrées (`derive_entrees`)

Pour chaque variable de surface, on compare la moyenne de l'instant aux statistiques
d'**entraînement** (`norm_stats.json`, `surf_mu` / `surf_sd`) :

    z_moyen = (moyenne − mu) / sd          part_extreme = part des capteurs à plus de 5 sd

Toutes les variables sont décrites, mais seules celles de `entrees_variables` peuvent être
en dérive : température, densité, humidité relative, pression. Il y a dérive si
`|z_moyen| ≥ entrees_z_max` ou `part_extreme ≥ entrees_part_extreme`. Cette règle vise les
erreurs d'unité ou d'étalonnage d'un flux entier : températures en K, pressions en hPa,
humidité en fraction, densité en g/m³.

**D'où venaient les 6,7 % de fausses alertes de la v1.** La règle v1 (`|z_moyen| ≥ 3` ou
`part_extreme ≥ 2 %`) portait sur **les sept variables, vent compris**, et ses seuils
n'avaient jamais été réglés sur la validation. Une fausse alerte était comptée par instant
propre dès qu'**une** variable dépassait. Or le vent suit la météo du jour, pas les unités.
Sur le cache local de v7 (jours d'entraînement 2 et 3, validation 14), un jour de vent fort
(jour 2, 15 m/s de moyenne) donne déjà :

- `wind_w_m_s` : 1,1 à 1,9 % des capteurs à plus de 5 sd (w de surface à queue lourde en
  relief), pour un seuil de 2 % ;
- `wind_u_m_s` : `z_moyen` de 1,8 à 2,7, pour un seuil de 3.

Aux mêmes instants, température, densité, humidité et pression restent sous 2,1 sd, sans
aucune valeur extrême. 6,7 % des 120 instants de test, c'est 8 instants, autant qu'un
jour entier (8 instants synoptiques). L'hypothèse la plus probable est donc un jour de
météo, pas une erreur d'unité ; le diagnostic par variable de la session GPU le dira. La
v2 :

- **exclut le vent** de la décision. Une erreur d'unité du vent n'était de toute façon pas
  détectable ainsi (limite déjà connue), et une dérive d'anémomètre relève de
  `capteur_derive` ;
- **règle les seuils sur la validation** (`an1.choisir_regle_entrees`), par une grille
  fixée d'avance (`z_max` ∈ {3, 4, 5, 6, 8}, `part_extreme` ∈ {2, 5, 10, 25, 50 %}). On garde
  les règles dont les fausses alertes restent ≤ 2,5 %, puis la meilleure détection des
  quatre erreurs connues ; à égalité, les seuils les plus hauts, c'est-à-dire la plus
  grande marge contre la météo.

La détection ne se perd pas. Les quatre erreurs déplacent tous les capteurs bien au-delà de
5 sd avec les statistiques de v7 : K → +45 sd, hPa → −82 sd, g/m³ → +44 000 sd, et
fraction → −5,7 sd pour **chaque** capteur, donc `part_extreme` = 100 %. Le script rejoue
aussi la règle v1 sur la validation et sur le test (`entrees_regle_v1_diagnostic`, avec les
fausses alertes **par variable**) : ce diagnostic sera confirmé ou infirmé par les données
de la session GPU.

Défauts de la config, en attendant ce réglage : `entrees_z_max` = 5, `entrees_part_extreme`
= 25 %.

## Cohérence physique (D11, `physics`)

`physique()` produit le bloc `physics` du snapshot, avec ses clés exactes :

| Clé | Contenu |
|---|---|
| `mass_residual_rms`, `mass_residual_p95` | RMS et 95e centile de \|r\| sur `mass_mask`, avec le **même opérateur** que DSO2 (`Geometry.mass_residual`) |
| `mass_residual_ok` | `rms ≤ dso.mass_rms` (1e-4) pour **cet instant** ; DSO2 reste un agrégat sur 120 instants |
| `ground_condition_rms_ms` | RMS de `w − (u·∂zs/∂x + v·∂zs/∂y)` au premier nœud valide, relief lissé à 300 m |
| `rho_source` | `atmosphere_standard`, `extrapolation_surface` ou `verite_synthetique` |

**Densité.** La source par défaut est `extrapolation_surface` : la densité ISA multipliée par
un **facteur unique**, `médiane(ρ mesurée / ρ ISA)` aux capteurs visibles. Un champ ρ
horizontal construit à partir des mesures injecterait leur bruit dans `u·∇ρ`, soit ~1e-4,
de l'ordre du seuil. Sans densité mesurée, on se replie sur `atmosphere_standard`.
`verite_synthetique` prend Y[3] du cache, pour comparer à DSO2 lors d'un rejeu.

**Alertes.** `incoherence_physique` est levée sur le résidu de masse. La condition au sol
n'a **pas de seuil par défaut** (`sol_rms_max_m_s: null`) : sa distribution sur la
validation est mesurée par le script GPU (`resume.json` → `physique_validation`) avant tout
réglage.

## Couverture (M2) et contrat (M3)

**`zone_mal_couverte`.** On prend l'indice de couverture M2 au premier nœud valide de chaque
colonne, on le moyenne par maille de 2 km, et on lève l'alerte si la moyenne est sous 0,2.
Avec la fiabilité active, on prend `Volume.reliability.coverage`, et la cause cite σ
vitesse ; sinon, la couverture est recalculée (`coverage_index`, sans l'ensemble).

**Alertes M3.** Elles viennent du rapport de `pcfno3d.contract` (upload) et suivent **les
seuils du rapport** (`rapport["seuils"]`). Les `point_id` deviennent `c<ix>_<iy>`, avec
`ix = (point_id − 1) mod nx`.

Le rapport M3 reste **exactement** celui de main, sans aucune clé ajoutée (opt-in strict,
testé sur le corps complet de l'upload, en 200 comme en 422). Le rapport ne liste pas les
stations hors plage. Quand les alertes sont actives, `stations_hors_plage()` relit donc le
fichier reçu avec les **règles** du contrat (`PLAGES`, `_ecart_densite`, tolérance du
rapport).

**`stations_insuffisantes` sans rapport (`alerts.reseau`).** Cette alerte dépend du réseau :

- **`treillis_synthetique`** (défaut, un capteur par maille) : part = mailles disponibles /
  mailles.
- **`stations_aneo`** (stations réelles : quelques dizaines pour ~125 000 mailles) : part =
  stations disponibles / stations **attendues**, déclarées (`alerts.stations_attendues`) ou
  reçues (argument `stations_disponibles`). À défaut de ce dernier, on compte les mailles
  qui portent une station. Si le nombre attendu est inconnu, **aucune alerte** n'est émise
  plutôt qu'une alerte fausse ; sans cette règle, l'alerte serait critique en permanence.

**Externes (M5).** `alertes_externes(items)` accepte `vent_fort_moyeu` et `cisaillement` :
kind, target, level, title, cause, recommendation, plus les champs facultatifs. L'id est
recalculé, et un id fourni différent est refusé. Les alertes passent ensuite par le même
journal.

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

## Journal et cycle de vie (D8)

`JournalAlertes` écrit `run_dir/logs/alertes.jsonl` (`alerts.journal` ; `off` = en
mémoire), une ligne LF par **transition** :

    {"evenement": "levee" | "niveau" | "close", "instant": "2025-01-06T03:00:00Z",
     "id": "b4a95d28", "kind": "capteur_derive", "target": "c340_200",
     "level": "important", "alerte": {...}}

Le cycle de vie suit ces règles :

- **`raised_at`** est la première levée de l'**épisode en cours**, à l'heure de l'**instant**
  (pas l'heure murale), ce qui rend un rejeu reproductible. Le journal est relu au
  démarrage : une alerte ouverte garde son `raised_at` après un redémarrage.
- Une alerte absente de l'instant suivant passe **close**. Elle reste publiée avec
  `status: "close"` pendant `retention_close_h` (24 h d'instants), puis disparaît. Si elle
  se lève à nouveau, c'est un nouvel épisode avec un nouveau `raised_at`.
- **Ordre chronologique** : seul un instant postérieur au dernier journalisé écrit. Un
  instant antérieur est évalué en consultation, sans écrire.
- **Ligne illisible** : une ligne tronquée (arrêt brutal, synchro Drive) est ignorée à la
  relecture, avec un avertissement, et comptée (`lignes_illisibles`). Le service démarre
  quand même.

`kpi.open_alerts` = `compter_ouvertes(items)`. `stations_summary` = `resume_stations(items,
n_capteurs)` : `derive`, `muette`, `exclue` (hors plage) d'après les alertes ouvertes.

## Exposition

Les alertes sont strictement **opt-in** :

```yaml
alerts:
  enabled: true
  seuils: <run_dir>/alertes_pcfno3d/<modele>/session_v2_.../seuils_validation.json
  # le modele de sites retenu (sites_R<r>.npz) est relu a cote du JSON
```

Quand `enabled` vaut `false` (défaut), rien ne change, ce qui est testé :

- la valeur servie est identique bit à bit, et les alertes ne la modifient jamais, même
  activées ;
- le corps de l'upload (200 et 422) et le rapport M3 sont identiques à ceux de main,
  comparés récursivement, hors temps mesuré et horloge ;
- aucun journal n'est écrit ;
- `GET /api/alerts` répond `409` et l'outil MCP renvoie `disponible: false`.

| Interface | Effet |
|---|---|
| `GET /api/alerts?stem=…` | document `alerts.schema.json` de l'instant ; sans `stem`, les dernières alertes calculées (sinon le premier instant de test) ; portée `read`, clé requise comme toutes les routes |
| `POST /api/reconstruct/upload` | avec les alertes activées : `alertes` (avec les alertes M3 du rapport) et `physique` en plus |
| MCP `alertes_actives(stem)` | alertes **ouvertes**, comptes par niveau, `physique`, état de la détection de dérive |
| `pcfno3d alertes --split test [--limit N] [--out f.json]` | instants du cache dans l'ordre, journal, document du dernier instant (jamais écrasé) |
| `MoteurAlertes.evaluer_modele(...)` | appel direct, pour la publication M6 (`document()`, `physics`, `ouvertes()`) |

**Coût.** Le détecteur `spatiale` (v2, défaut) ne fait **aucune passe** : ~1 s par
instant sur CPU pour les 125 000 mailles de v7. Les statistiques v1 et `combinee` comptent
9 passes par instant (`derive_cote²`), et seulement pour `/api/alerts` : `predict` et DSO4
ne sont pas touchés. Sur CPU (15 à 25 s par passe pour v7), une évaluation v1 prend
plusieurs minutes ; c'est un usage GPU.

Le calcul ne bloque pas le service :

- l'upload lit le fichier en `async`, puis exécute contrat, inférence et alertes dans un
  thread du pool (`run_in_threadpool`) ; `/api/health` répond pendant ce temps (testé) ;
- `/api/alerts` est une route synchrone, que FastAPI exécute aussi dans le pool ;
- les passes restent sérialisées par le verrou de `WindFieldModel`, et l'état des alertes
  par le verrou du moteur.

**Densité « vérité ».** Avec `rho_source: verite_synthetique`, l'API, l'outil MCP et la
commande `alertes` chargent Y de la même façon (`MoteurAlertes.verite`) : le résultat mis en
cache pour un instant ne dépend pas de l'interface qui l'a calculé. Sans Y (upload, direct),
on se replie sur `extrapolation_surface`.

## AN1 : simulation de pannes et de dérives

**Critère.** AN1 demande au moins **90 % de détection** et **moins de 5 % de fausses
alertes**.

**Protocole (`an1.py`).** Un scénario = un jour, avec ses instants synoptiques dans l'ordre.
On injecte des défauts sur des mailles-capteurs disponibles à tous les instants du jour.
Elles sont tirées à graine fixe et espacées d'au moins `cote + 1` mailles :

- **dérives** : biais (vecteur constant de 0,5 à 3 m/s, direction aléatoire), gain (vitesse
  × 1,15 ou × 1,30), girouette (rotation de 15° ou 30°) ;
- **pannes** : valeur figée (u, v, T constants dès le premier instant), détectée par la
  **règle M3** (`contract.valeurs_figees`).

On déroule ensuite la chaîne complète : passe servie, détecteurs (les 9 passes sans capteur
seulement pour les statistiques v1 et `combinee`) et persistance. Le protocole (nombres,
espacement de `cote + 1` mailles, graines) est **inchangé** depuis la v1.

**Comptage.** L'unité est le couple (scénario, maille-capteur). Une maille est **signalée**
si elle reçoit au moins une alerte `capteur_derive` dans la journée.

| Mesure | Définition | Verdict |
|---|---|---|
| détection | signalés parmi les **dérives** d'amplitude effective ≥ `e_ref` = 1 m/s (le seuil de la fixture du contrat) | **oui**, ≥ 90 % |
| fausses alertes | **part des signalés qui sont sains** (FDR) : la question de l'opérateur, « cette alerte est-elle fausse ? » | **oui**, < 5 % |
| sains signalés / sains | taux de faux positifs par capteur | publié à côté |
| fausses alertes par instant | nombre moyen de capteurs sains en alerte à un instant | publié à côté |
| pannes (valeurs figées) | détection par la règle M3 ou par une alerte, et fausses alertes de la règle M3 | **hors verdict** : leur détection par égalité exacte est quasi certaine et gonflerait le taux |
| détail | détection par type et par classe d'amplitude, de 0 à 0,5, 0,5 à 1, 1 à 2, 2 à 3, plus de 3 m/s | publié à côté |

**Protocole fixé (`an1.PROTOCOLE`).** La part de fausses alertes dépend de la **prévalence**
des défauts : plus on injecte de dérives, plus elle baisse. Le verdict officiel se mesure
donc avec un protocole constant :

- **40 dérives et 10 pannes par jour** ;
- `e_ref` = 1 m/s ;
- les 8 types de dérive du plan `PLAN_DERIVES`, en rotation.

Avec 125 000 capteurs, 40 dérives par jour donnent une prévalence de ~3·10⁻⁴. Une part de
fausses alertes de 5 % correspond alors à ~2 fausses alertes par jour.

Si l'on passe d'autres valeurs (`--derives-par-jour`, `--pannes-par-jour`, `--e-ref`), le
script les accepte, mais il marque tout le résultat `statut: "exploratoire"`, avec
`an1_conforme: null`, et **ne produit aucun verdict officiel**.

**Entrées.** Chaque instant **propre** sert à mesurer les fausses alertes. Le même instant,
avec une erreur d'unité connue (K, hPa, fraction, g/m³), sert à mesurer la détection.

**Verdict.** AN1 est vert si les deux familles, dérives de capteurs et entrées, passent
toutes les deux, et seulement sous le protocole fixé.

**Réglage : validation seulement (v2).** Tout ce qui se règle l'est sur la validation, par
des grilles **fixées d'avance** (`an1.GRILLE_DETECTEURS`, `GRILLE_KAPPA`,
`GRILLE_ENTREES`, écrites le 16/10/2026 avant toute mesure GPU de la v2) :

- **détecteurs candidats** (`an1.detecteurs`) :
  - `spatiale` : rayon R ∈ {1, 2, 3} mailles (100 à 300 m), échelle locale L ∈ {2, 4},
    fenêtre W ∈ {3, 4, 8} instants, avec ou sans correction de site ;
  - `innovation`, `influence` : W ∈ {3, 4, 8} ;
  - `combinee` : la même grille que `spatiale`.

  Au total 78 candidats. Les passes du modèle sont communes à tous, et la médiane des
  voisins est calculée une fois par rayon et par instant ;
- **κ** : de 2 à 30 par pas de 0,25. Pour chaque détecteur, le plus petit κ dont la part de
  fausses alertes reste ≤ `--fdr-cible` = 2,5 % sur la validation ;
- **choix** (`an1.choisir_detecteur`) : la meilleure détection ; à égalité, la plus faible
  part de fausses alertes, puis l'ordre de la grille (spatiale d'abord, correction de site
  d'abord, fenêtre courte d'abord) ;
- **persistance** : `derive_min_instants` = W (fenêtre complète) ;
- **entrées** : `an1.choisir_regle_entrees` (voir plus haut).

Le JSON porte `jeu: "validation"` et `methode: "alertes-m4-v2"`, et `charger_seuils` refuse
tout autre jeu (les JSON v1 restent relisibles). Les modèles de sites viennent des jours
d'entraînement. Le test ne sert qu'au verdict.

Choisir parmi 78 candidats sur ~400 dérives de validation laisse un optimisme de sélection.
La marge (2,5 % visés pour 5 % permis) et le test le mesurent ; rien n'est réajusté après.

**Mesure CPU (mini-jeu, mécanisme seulement, exploratoire).** Les conditions sont un modèle
de 2 époques, 4 jours d'entraînement (modèles de sites), 1 jour de validation et 1 jour de
test, 8 dérives et 2 pannes par jour, toute la grille (`--derives-par-jour 8
--pannes-par-jour 2 --deuxieme-lecture`, 18 s sur ines-pc).

| Détecteur (validation) | κ retenu | Détection | Fausses alertes |
|---|---|---|---|
| `spatiale:R1:L2:W3:site` (**retenu**) | 2,0 | 1,00 | 0,00 |
| `spatiale:R2:L2:W4` (sans site) | 2,0 | 1,00 | 0,00 |
| `innovation:W3` | 3,5 | 1,00 | 0,00 |
| `influence:W3` | 14,5 | 0,00 | 0,00 |
| `*:W8` | — | 0,00 | 0,00 (jour de 4 instants : jamais 8 observations) |

Sur le test, avec `spatiale:R1:L2:W3:site` :

- détection des dérives 1,00 (biais 0,5 m/s : 0, hors verdict car sous `e_ref`) ;
- fausses alertes 0,00, aucune par instant ;
- pannes (hors verdict) 1,00 ;
- entrées : règle retenue `z_max` 8, `part_extreme` 50 % ; détection 1,00, aucune fausse
  alerte.

Le champ du mini-jeu est lisse : tout y est facile, et κ tombe au bas de la grille. Ces
chiffres vérifient la chaîne, pas la v2 sur v7. Les tests vérifient le mécanisme :

- dérives injectées connues → signalées ;
- capteurs sains → aucune alerte ;
- crête apprise par le modèle de sites → plus suspecte ;
- déterminisme à graine égale ;
- refus du test pour le réglage, du JSON comme du modèle de sites.

**Session GPU (`scripts/m4_an1_colab.py`, v2).**

```bash
!python scripts/m4_an1_colab.py -c colab --deuxieme-lecture   # protocole : 40 dérives, 10 pannes par jour
```

`--deuxieme-lecture` marque `an1_test.json` et `resume.json` : `lecture: "deuxieme"`, et la
note « seconde mesure sur le test, après une première lecture (16/10/2026, non conforme) ».
Le résultat ne doit **pas** être présenté comme une mesure indépendante : la v2 a été
conçue après avoir vu la première. Sans l'option, le script avertit, et le fichier porte
`lecture: "non declaree"`.

Tout s'écrit dans un dossier **neuf**,
`run_dir/alertes_pcfno3d/<modèle>/session_v2_<horodatage>/`, sans rien écraser :

- `sites_R1.npz`, `sites_R2.npz`, `sites_R3.npz` : modèles de sites (entraînement) ;
- `seuils_validation.json` : détecteur retenu, κ, règle des entrées, courbe κ → détection
  et fausses alertes, tous les candidats (`alternatives`) ;
- `an1_test.json` : verdict (ou `exploratoire` hors protocole), détection par type et par
  amplitude, fausses alertes par variable des entrées, règle v1 rejouée (diagnostic) ;
- `resume.json` : avec la distribution du résidu de masse et de la condition au sol sur la
  validation.

**Coût estimé.** Les modèles de sites demandent ~1 s par instant d'entraînement et par
rayon (~560 instants ; `--jours-entrainement N` pour en prendre moins). Chaque jour de
validation coûte ~2 min de CPU pour les détecteurs spatiaux, plus les 10 passes GPU par
instant de la v1. Il faut compter **environ 1 h à 1 h 30** en tout. `--statistiques
spatiale` saute les passes sans capteur. Le seuil retenu se sert ensuite par
`alerts.seuils` : le modèle de sites retenu est relu à côté du JSON.

## Écarts au contrat (rév. 2) et limites

- **Détecteur spatial et réseau clairsemé.** La cohérence spatiale suppose des voisins à
  100–300 m : c'est le cas du treillis (un capteur par maille). Avec quelques dizaines de
  stations réelles (`stations_aneo`), presque aucune n'a 3 voisins dans le rayon : elles
  ne sont pas évaluées (jamais une alerte inventée). Il faut alors revenir à `innovation`
  (`alerts.derive_statistique`), ou régler un rayon adapté sur des données réelles.
- **Modèle de sites.** Il est appris sur les jours d'entraînement d'un réseau donné. Un
  capteur déplacé, ou un site nouveau, garde l'ancien écart attendu tant que le modèle
  n'est pas réappris ; sans modèle (`derive_sites: null`), le détecteur reste valable,
  sans la correction.

- **Cibles « domaine ».** Les alertes qui portent sur tout l'instant prennent pour cible
  `domaine`, `domaine:masse`, `domaine:sol` ou `domaine:<variable>`. Le schéma accepte
  toute chaîne, mais sa description ne cite que capteur, éolienne ou maille. Leur id reste
  stable.
- **Capteur = maille de 100 m.** En treillis, `station_id` désigne le capteur sud-ouest de
  la maille du modèle, et la cause le dit. Les alertes M3 (upload), elles, désignent le
  vrai capteur à 50 m.
- **`station_muette`.** `value`, `threshold` et `unit` sont `null` : le rapport M3 ne donne
  pas de durée de silence par station, alors que la fixture affiche 140 min / 30 min.
- **Textes sans accents.** `title`, `cause` et `recommendation` sont écrits sans accents,
  comme tout le code du dépôt. L'interface peut les afficher tels quels.
- **`zone_mal_couverte`.** La règle porte sur la couverture seule ; σ vitesse est cité dans
  la cause, mais il ne conditionne pas l'alerte. La fixture parle de « couverture < 0,2 et
  incertitude élevée », mais ne fixe aucun seuil de σ.
- **Acquittement.** Il est hors contrat v1 (D8 : géré par le Worker). Seuls `ouverte` et
  `close` sont produits.
- **`stations.json` (`bias_ms`).** Il n'est pas produit ici. Il est publié par M6, qui peut
  reprendre la valeur de `capteur_derive` (l'écart en m/s) et `resume_stations`.
- **MCP.** L'outil `alertes_actives` est toujours listé : il y a 9 outils au lieu de 8, et
  l'outil répond `disponible: false` quand les alertes sont désactivées.
