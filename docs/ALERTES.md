# Lot M4 — alertes (critère AN1)

*Modules : `src/pcfno3d/alerts.py` (détection, agrégation, journal, moteur) et
`src/pcfno3d/an1.py` (simulation de pannes et de dérives). Besoins BF20 à BF23 ; critère
AN1 ; contrat ATOM **révision 2** (`DECISIONS-rev2.md` : D8, D11, B5, m9 ;
`alerts.schema.json`, `common.schema.json` : `alert_level`, `station_state` ;
`snapshot.schema.json` : `physics`, `kpi.open_alerts`, `stations_summary`).*

## Statut : prêt, **verdict AN1 en attente de la session GPU**

Les tests CPU vérifient le mécanisme sur le mini-jeu (modèle de 2 époques) : ils ne disent
rien de la qualité de détection sur v7. Le chiffre AN1 vient de
`scripts/m4_an1_colab.py`, à lancer sur GPU (voir plus bas).

## Un document par instant

`GET /api/alerts` renvoie exactement `alerts.schema.json` : `{"items": [...]}`. Chaque
alerte porte les 15 clés du contrat (`additionalProperties: false`), avec un niveau, une
cause et une recommandation en français. Le moteur agrège **toutes** les alertes, quel que
soit le lot qui les détecte :

| `kind` | Lot | Détection | Cible | Niveau |
|---|---|---|---|---|
| `hors_plage` | M3 | rapport du contrat ANEO (`qualite.stations_hors_plage`) | capteur `c<i>_<j>` | avertissement |
| `station_muette` | M3 | rapport du contrat (absences, valeurs figées) | capteur | avertissement |
| `stations_insuffisantes` | M3 | rapport du contrat ; sans rapport (instants du cache), part de mailles-capteurs disponibles | `domaine` | important (< 70 %), critique (< 30 %) |
| `capteur_derive` | M4 | capteur masqué comparé à la reconstruction faite sans lui | capteur | important ; critique si l'écart ≥ 2 × seuil |
| `incoherence_physique` | M4 | résidu de continuité de l'instant > seuil DSO2 ; condition au sol (seuil à régler) | `domaine:masse`, `domaine:sol` | important ; critique si ≥ 10 × seuil ; sol : avertissement |
| `derive_entrees` | M4 | statistiques des entrées vs entraînement (`norm_stats.json`) | `domaine:<variable>` | avertissement ; important si ≥ 2 × seuil |
| `zone_mal_couverte` | M2 | indice de couverture moyen < 0,2 sur une maille de 2 km | maille `m<ix>_<iy>` | information |
| `vent_fort_moyeu`, `cisaillement` | M5 | **entrée externe** (`alertes_externes`) : le calcul est fait par M5 | éolienne | donné par M5 |

**Identifiant (D8).** `id = sha8("kind|cible")` (`alert_id`). Il est stable d'un instant
à l'autre, et c'est lui qui relie les instants dans le journal. Si deux sources donnent la
même cible, `agreger` garde la plus grave.

**Ordre.** Les ouvertes d'abord, puis par niveau décroissant, par type et par cible.

**Plafond.** Au plus `alerts.max_par_type` (100) alertes d'un même type par instant, les plus
fortes d'abord. Le rapport M3 tronque déjà ses listes d'identifiants à 100.

## Capteur en dérive (`capteur_derive`)

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
sans extrapolation). Deux statistiques sont disponibles :

- **innovation** (défaut) : `d = mesure − reconstruction sans le capteur` ;
- **influence** : `d = reconstruction servie (avec) − reconstruction sans le capteur`.

L'influence annule l'écart de représentativité, c'est-à-dire un capteur à ~25 m du sol
alors que le premier nœud valide est parfois 100 m plus haut en relief. Mais elle suppose
que le modèle suive assez ses entrées localement. Sur le mini-jeu, le petit modèle ne le
fait pas : l'innovation détecte, l'influence presque pas (voir AN1). Les deux sont calculées
avec **les mêmes passes**, et le script GPU choisit sur la **validation**.

**Normalisation robuste, par instant.**

    d_c = d − médiane(d)          (une dérive commune à tous n'est pas un capteur)
    s   = max(1,4826 · MAD(d_c), 0,05 m/s)
    z   = |d_c| / s               (écarts robustes)      a = |d_c|  (m/s)

Le seuil suit ainsi la difficulté du jour : un vent fort élargit tous les écarts.

**Persistance.** `SuiviDerive` garde les `derive_fenetre` = 4 derniers instants. Une alerte
est levée si les trois conditions suivantes sont réunies :

- au moins `derive_min_instants` = 3 observations ;
- z moyen ≥ `derive_kappa` ;
- écart moyen ≥ `derive_min_m_s` = 0,5 m/s (plancher : la tolérance OMM sur le vent).

La valeur publiée (`value`) est l'écart moyen en m/s. Le seuil publié (`threshold`) vaut
`max(kappa · s̄, 0,5)` m/s. La persistance n'est **mise à jour que pour un instant postérieur
au dernier suivi**. Un instant antérieur est évalué sans dérive (`derive.evaluee: false`),
pour qu'une consultation ne fausse pas la fenêtre. La fenêtre vit en mémoire : un
redémarrage repart d'une fenêtre vide, alors que le journal, lui, est relu.

**Granularité.** Le modèle voit la surface par mailles de 100 m (`pool_surface` : moyenne
des 2 × 2 capteurs à 50 m). Le « capteur » détecté est donc la **maille-capteur**, désignée
par son capteur sud-ouest `c<i·stride>_<j·stride>` ; la cause le précise. Un seul capteur à
50 m qui dérive de b décale la maille de b/4.

## Dérive des entrées (`derive_entrees`)

Pour chaque variable de surface, on compare la moyenne de l'instant aux statistiques
d'**entraînement** (`norm_stats.json`, `surf_mu` / `surf_sd`) :

    z_moyen = (moyenne − mu) / sd          part_extreme = part des capteurs à plus de 5 sd

Il y a dérive si `|z_moyen| ≥ 3` ou `part_extreme ≥ 2 %` (`entrees_*`). Cette règle vise
les erreurs d'unité ou d'étalonnage d'un flux entier : températures en K, pressions en hPa,
humidité en fraction, densité en g/m³.

Une erreur d'unité sur le vent (nœuds au lieu de m/s) n'est **pas** détectable ainsi sans
fausses alertes les jours de vent fort. C'est une limite connue.

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
`ix = (point_id − 1) mod nx`. Le lot M4 a ajouté au rapport
`qualite.stations_hors_plage`, un ajout additif.

**Externes (M5).** `alertes_externes(items)` accepte `vent_fort_moyeu` et `cisaillement` :
kind, target, level, title, cause, recommendation, plus les champs facultatifs. L'id est
recalculé, et un id fourni différent est refusé. Les alertes passent ensuite par le même
journal.

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

`kpi.open_alerts` = `compter_ouvertes(items)`. `stations_summary` = `resume_stations(items,
n_capteurs)` : `derive`, `muette`, `exclue` (hors plage) d'après les alertes ouvertes.

## Exposition

Les alertes sont strictement **opt-in** :

```yaml
alerts:
  enabled: true
  seuils: <run_dir>/alertes_pcfno3d/<modele>/session_.../seuils_validation.json
```

Quand `enabled` vaut `false` (défaut), rien ne change, ce qui est testé :

- la valeur servie est identique bit à bit, et les alertes ne la modifient jamais, même
  activées ;
- la réponse de l'upload garde exactement ses clés ;
- aucun journal n'est écrit ;
- `GET /api/alerts` répond `409` et l'outil MCP renvoie `disponible: false`.

| Interface | Effet |
|---|---|
| `GET /api/alerts?stem=…` | document `alerts.schema.json` de l'instant ; sans `stem`, les dernières alertes calculées (sinon le premier instant de test) ; portée `read`, clé requise comme toutes les routes |
| `POST /api/reconstruct/upload` | avec les alertes activées : `alertes` (avec les alertes M3 du rapport) et `physique` en plus |
| MCP `alertes_actives(stem)` | alertes **ouvertes**, comptes par niveau, `physique`, état de la détection de dérive |
| `pcfno3d alertes --split test [--limit N] [--out f.json]` | instants du cache dans l'ordre, journal, document du dernier instant (jamais écrasé) |
| `MoteurAlertes.evaluer_modele(...)` | appel direct, pour la publication M6 (`document()`, `physics`, `ouvertes()`) |

**Coût.** On compte 9 passes par instant (`derive_cote²`), et seulement pour
`/api/alerts` : `predict` et DSO4 ne sont pas touchés. Sur CPU (15 à 25 s par passe pour
v7), une évaluation de dérive prend plusieurs minutes ; c'est un usage GPU.

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

On déroule ensuite la chaîne complète : passe servie, 9 passes et persistance.

**Comptage.** L'unité est le couple (scénario, maille-capteur). Une maille est **signalée**
si elle reçoit au moins une alerte capteur dans la journée.

| Mesure | Définition |
|---|---|
| détection | signalés parmi les défaillants : dérives d'**amplitude effective** ≥ `e_ref` = 1 m/s (le seuil de la fixture du contrat), et pannes |
| fausses alertes | **part des signalés qui sont sains**. C'est la question de l'opérateur : « cette alerte est-elle fausse ? ». Rapportée aux 125 000 capteurs sains, une proportion de 5 % ferait 6 000 fausses alertes, ce qui n'aurait pas de sens. |
| taux de faux positifs | signalés parmi les sains : publié, hors verdict |
| détail | détection par type et par classe d'amplitude, de 0 à 0,5, 0,5 à 1, 1 à 2, 2 à 3, plus de 3 m/s |

**Entrées.** Chaque instant **propre** sert à mesurer les fausses alertes. Le même instant,
avec une erreur d'unité connue (K, hPa, fraction, g/m³), sert à mesurer la détection.

**Verdict.** AN1 est vert si les deux familles, capteurs et entrées, passent toutes les deux.

**Seuils : validation seulement.** `choisir_kappa` prend le plus petit κ (donc la meilleure
détection) dont la part de fausses alertes reste sous `--fdr-cible` = 2,5 % **sur la
validation**, ce qui laisse une marge sous 5 %. On le fait pour chaque statistique, et l'on
retient celle qui détecte le mieux. Le JSON porte `jeu: "validation"`, et `charger_seuils`
refuse tout autre jeu. Le test ne sert qu'au verdict.

**Mesure CPU (mini-jeu, mécanisme seulement).** Les conditions sont un modèle de 2 époques,
un jour de validation et un jour de test, 8 dérives et 2 pannes par jour.

| Statistique | κ retenu | Détection (validation) | Fausses alertes (validation) |
|---|---|---|---|
| innovation | 3,5 | 1,00 | 0,00 |
| influence | 13,75 | 0,25 | 0,00 |

Sur le test, avec l'innovation, la détection vaut 1,00 et la part de fausses alertes 0,10.
Pour les entrées, la détection vaut 1,00 et il n'y a aucune fausse alerte. Le verdict est
donc **non conforme sur ce mini-jeu**, avec un seul jour de réglage et 200 capteurs : c'est
attendu. Les tests vérifient le mécanisme, pas ce chiffre.

**Session GPU (`scripts/m4_an1_colab.py`).**

```bash
!python scripts/m4_an1_colab.py -c colab            # --derives-par-jour 40 --figees-par-jour 10
```

Tout s'écrit dans un dossier **neuf**, `run_dir/alertes_pcfno3d/<modèle>/session_<horodatage>/`,
sans rien écraser :

- `seuils_validation.json` : κ, statistique, courbe κ → détection et fausses alertes ;
- `an1_test.json` : verdict ;
- `resume.json` : avec la distribution du résidu de masse et de la condition au sol sur la
  validation.

Le coût est d'environ 10 passes par instant, soit ~1 s sur GPU. Il faut compter ~4 min pour
les 120 instants de validation et ~2 min pour le test. Le seuil retenu se sert ensuite par
`alerts.seuils`.

## Écarts au contrat (rév. 2) et limites

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
