# Consommer le champ de vent PC-FNO3D depuis une autre application

Trois façons d'accéder aux sorties du modèle, du plus simple au plus volumineux :

| Besoin | Interface |
|---|---|
| Un agent / LLM qui interroge le vent | **serveur MCP** sur `/mcp` |
| Une application qui affiche ou calcule | **API JSON** sur `/api/...` |
| Un volume 3D complet à charger en mémoire | **export binaire** (`manifest.json` + Float32) |

Tout passe par le même processus, lancé par :

```bash
pip install -e ".[serve]"
pcfno3d apikey create app-externe --scope read     # affiche le secret UNE fois
pcfno3d serve --port 8000                          # ou l'image Docker
```

- Application web : `http://127.0.0.1:8000/app/`
- API interactive : `http://127.0.0.1:8000/api/docs`
- Serveur MCP : `http://127.0.0.1:8000/mcp`

## Où tourne le modèle : poste local ou Colab

Les poids entraînés et le GPU vivent sur le Google Drive / Colab de Rayen. Deux montages
possibles, selon ce dont l'application a besoin.

**A — Colab, c'est le plus direct.** Le notebook
`notebooks/current/pcfno3d_api_colab.ipynb` ouvre le service là où le modèle est déjà,
derrière une URL publique :

```python
from service.colab import serve_public
info = serve_public(config="colab", key_name="app-externe", scope="read")
print(info["public_url"], info["mcp_url"], info["api_key"])
```

L'application reçoit alors une base `https://xxx.trycloudflare.com` et une clé, et
interroge le vrai modèle sans que rien soit copié.

> **Ce montage est fait pour une démonstration ou une intégration, pas pour de la
> production.** La session Colab est éphémère (inactivité ~90 min, 12 h au maximum), donc
> **l'URL change à chaque relance** : une application qui la code en dur se cassera.
> Prévois la base d'API comme une configuration, pas comme une constante. Le tunnel est
> public : seule la clé protège les données.

**B — poste local ou serveur.** `pcfno3d serve` avec les checkpoints copiés dans
`run_dir`. URL stable, mais il faut descendre `best_dso_v7.pt` + `norm_stats.json` du
Drive — et, si l'application demande des instants du dataset, construire leur cache.

**C — Colab, avec une adresse stable devant.** C'est le montage recommandé dès qu'une
application doit tenir plusieurs jours. Le service local se met en relais vers Colab :

```bash
pcfno3d link set https://xxx.trycloudflare.com --key pcf_...
pcfno3d serve --port 8000
```

L'application et les clients MCP visent alors `http://127.0.0.1:8000` **et n'en changent
plus** ; quand la session Colab redémarre, seule la commande `link set` est à refaire.
Deux clés distinctes, volontairement :

```
application --(clé LOCALE)--> service local --(clé AMONT)--> Colab
```

La clé de l'instance Colab ne quitte pas la machine du relais, et révoquer une clé locale
ne touche pas à l'amont. Si l'amont est tombé, `/api/health` passe en `degraded` et les
requêtes répondent `502` en indiquant la commande de réparation — un signal exploitable,
pas un silence.

**D — Colab derrière une URL fixe publique : `atom.zelqane.com` (en place, octobre 2026).**
C'est le montage retenu avec Rayen pour la visualisation 3D :

```
navigateur --> atom.zelqane.com          Worker de Rayen : page 3D, injecte la clé côté serveur
           --> atom-origin.zelqane.com   tunnel Cloudflare NOMMÉ « atom » (URL fixe)
           --> Colab                     cloudflared --> service sur 127.0.0.1:8000 (GPU)
```

Le notebook `notebooks/current/pcfno3d_atom_colab.ipynb` le démarre :
`serve_public(..., named_tunnel={...}, key_hash=...)`. Deux secrets, remis de la main à la
main, jamais par le salon ni par git :

- les **identifiants du tunnel** (Rayen → Ines), en secret Colab `TUNNEL_CRED_CONTENTS`. Ils
  ne vont qu'au processus cloudflared, jamais sur disque ;
- la **clé de lecture** du Worker (Ines → Rayen). Colab ne connaît que son **empreinte**
  SHA-256, écrite dans le notebook : le serveur vérifie la clé sans jamais l'avoir eue.

L'URL survit aux redémarrages de Colab, mais le service ne vit que pendant la session :
hors session, la page de Rayen affiche « service endormi ». Le contrôle d'hôte du MCP est
réglé sur `atom-origin.zelqane.com` (sans quoi `/mcp` répond `421`).

La clé du Worker sert **tous** les visiteurs de la page : elle est soumise aux quotas
comme les autres (voir plus bas). Si la page 3D enchaîne beaucoup de requêtes, lui donner
un quota propre au démarrage : `serve_public(..., key_hash=..., quota_minute=1200)`
(rejouable, l'empreinte est mise à jour sans changer de clé).

Dans tous les cas, les interfaces décrites ci-dessous sont identiques : seule la base
d'URL change.

## Authentification

Une clé a la forme `pcf_<32 caractères>`. Seul son **hash SHA-256** est conservé
(`run_dir/api_keys.json`) : le secret s'affiche à la création et ne peut plus être
retrouvé. Deux portées :

| Portée | Droits |
|---|---|
| `read` | lire les instants, champs, points, profils, DSO ; lancer une reconstruction |
| `admin` | en plus : écrire des exports sur le disque du serveur, lister les clés |

La clé se transmet au choix par `X-API-Key: pcf_…`, `Authorization: Bearer pcf_…`, ou
`?api_key=pcf_…` (dernier recours pour un lien de navigateur).

```bash
pcfno3d apikey list                       # noms, portées, empreintes, quotas — jamais de secret
pcfno3d apikey revoke app-externe         # effet à la requête suivante, sans redémarrage
pcfno3d apikey quota app-externe --per-minute 600 --per-day 100000
```

Des clés peuvent aussi venir de l'environnement, pratique en conteneur :
`PCFNO3D_API_KEYS="app-externe:pcf_...,autre:pcf_..."` (portée `read`).

**Fermé par défaut.** Toute requête sans clé valide est refusée (`401`), sur `/api/...`
comme sur `/mcp` et les exports. **Si aucune clé n'existe, tout est refusé** sauf
`GET /api/health`, qui l'annonce : `"auth": "aucune_cle"` et un `message` « aucune cle
configuree ». Restent publiques, car elles ne contiennent aucune donnée : `/api/health`,
la documentation (`/api/docs`, `/api/openapi.json`) et l'application web `/app/` (qui
demande la clé).

Pour le développement local seulement, un mode ouvert existe sur **opt-in explicite** :
`pcfno3d serve --insecure-open` (ou `PCFNO3D_INSECURE_OPEN=1`). Il est annoncé par un
avertissement au démarrage et dans `/api/health` (`"auth": "ouvert"`), et n'a d'effet que
tant qu'aucune clé n'existe. Même actif, **il ne sert sans clé que les requêtes directes
de la boucle locale** : adresse `127.0.0.1` / `::1` et aucun en-tête de proxy
(`X-Forwarded-For`, `Forwarded`, `CF-Connecting-IP`, `X-Real-IP`) ; toute autre requête
reçoit `401`. Un `uvicorn --host 0.0.0.0` ou un proxy HTTP local qui pose ces en-têtes
(nginx, cloudflared, ngrok) ne l'ouvrent donc pas au réseau ; un relais TCP sans en-tête
(`ssh -R`, socat, proxy userland de Docker) arrive, lui, depuis la boucle locale : ne
l'utilisez jamais avec le mode ouvert. `pcfno3d serve` refuse de démarrer avec `--host` autre que
`127.0.0.1`/`localhost` dès que l'opt-in est demandé (option ou variable) ; il est ignoré
si `PCFNO3D_ALLOWED_HOSTS` est renseigné (proxy, domaine), et **jamais transmis** au
service lancé par `service.colab.serve_public` (Colab, tunnels).

**Révocation immédiate.** Le service relit `api_keys.json` dès que le fichier change (un
`stat` par requête) : une clé révoquée par `pcfno3d apikey revoke` — même lancé depuis un
autre terminal, service en marche — est refusée **à la requête suivante**. Le service
n'écrit jamais ce fichier en vérifiant une clé (la date de dernière utilisation va dans
`api_keys_usage.json`), il ne peut donc pas écraser une révocation.

**Quotas par clé.** Par défaut 300 requêtes par minute glissante et 50 000 par jour UTC
(`PCFNO3D_QUOTA_MINUTE`, `PCFNO3D_QUOTA_JOUR`, `0` = illimité) ; une clé peut avoir les
siens (`apikey create --per-minute/--per-day`, `apikey quota`). Au-delà : `429 Too Many
Requests` avec l'en-tête **`Retry-After`** (secondes). Une requête refusée n'est pas
comptée. Limite assumée : les compteurs sont **en mémoire du processus** — avec plusieurs
processus uvicorn, chacun compte de son côté (limite effective × N), et un redémarrage
les remet à zéro. Le service tourne aujourd'hui en un seul processus.

**Journal d'accès.** Une ligne JSON par requête authentifiée ou refusée, dans
`run_dir/logs/acces.jsonl` (`PCFNO3D_ACCESS_LOG` pour un autre chemin, relatif à
`run_dir` ; `off` pour couper) : horodatage UTC, nom de la clé, empreinte courte (8
caractères du hash, jamais le hash complet ni le secret), portée, méthode, chemin
(sans query string ; les paramètres sont gardés à part, **`api_key` retiré**), statut,
durée en ms, octets, IP (et l'IP déclarée par le proxy, `ip_transmise`, à part), motif
d'un refus. Rotation par taille : 10 Mo (`PCFNO3D_ACCESS_LOG_MAX_MB`), 5 fichiers gardés.
Les `401` sont regroupés : une ligne par IP et par minute au plus, la suivante portant
`refus_regroupes` (nombre de refus non écrits entre-temps) ; un flot de clés fausses ne
peut donc pas purger l'historique par la rotation.

En mode lien, le paramètre `?api_key=` (clé **locale**) est retiré avant le relais : seule
la clé amont, en en-tête, part vers l'instance distante.

## Serveur MCP

Transport *streamable HTTP*, réponses JSON. Déclaration côté client :

```json
{
  "mcpServers": {
    "pcfno3d": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "X-API-Key": "pcf_..." }
    }
  }
}
```

Pour un client local en *stdio* (sans HTTP ni clé) : `pcfno3d mcp`.

| Outil | Rôle |
|---|---|
| `grid_meta` | grille, altitudes disponibles, domaine, modèle chargé — **à appeler en premier** |
| `list_instants` | instants reconstructibles (`split` : train / val / test) |
| `wind_at_point` | u, v, w, vitesse, direction en un point (x, y, altitude) |
| `wind_profile` | profil vertical complet en un point, un niveau par ligne |
| `wind_field_at_altitude` | carte décimée d'une variable à une altitude |
| `instant_summary` | vitesses par niveau, secteur dominant, temps d'inférence |
| `dso_report` | dernier verdict DSO1–DSO5 archivé |
| `production_eolienne` | parc éolien déclaré (lot M5) : vent au moyeu, REWS, puissance, alertes ; erreur sans parc déclaré — voir `docs/EOLIEN.md` |
| `export_instant` | écrit un volume sur le disque du serveur — **portée `admin`** |

Appel direct, sans SDK :

```bash
curl -s http://127.0.0.1:8000/mcp \
  -H "X-API-Key: pcf_..." \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

Les résultats arrivent dans `result.content[0].text`, en JSON à parser.

> **Derrière un proxy ou un nom de domaine**, le transport MCP refuse un en-tête `Host`
> non déclaré (protection anti-DNS-rebinding) et répond `421 Misdirected Request`. Il faut
> lister les hôtes : `PCFNO3D_ALLOWED_HOSTS=mon-domaine.tn,mon-domaine.tn:443`.

## API JSON

| Route | Description |
|---|---|
| `GET /api/health` | état du service — **publique** ; `auth` : `cle_api`, `aucune_cle` (tout est refusé) ou `ouvert` (opt-in local) |
| `GET /api/meta` | grille, niveaux ASL, conventions, domaine en mètres, `georef` |
| `GET /api/instants?split=test&limit=100` | instants disponibles |
| `POST /api/reconstruct` | `{stem, keep, export, stride}` → résumé de l'instant |
| `POST /api/reconstruct/upload` | CSV de mesures de surface → reconstruction ; le CSV passe d'abord le contrat de données ANEO (`docs/CONTRAT_ANEO.md`, profil upload : stations absentes, en `mask_i_obs = 0` ou sans u/v/T = panne connue ; stations fausses masquées avant l'inférence) : rejet → `422` avec le rapport sous `qualite` ; conforme ou quarantaine → reconstruction + `statut_qualite`, `avertissements`, `qualite` |
| `GET /api/field?z=120&variable=speed&stride=4` | carte à une altitude |
| `GET /api/point?x=5000&y=5000&z=120` | vent en un point |
| `GET /api/profile?x=5000&y=5000` | profil vertical |
| `GET /api/dso` | dernier verdict DSO archivé |
| `GET /api/wind-farm?stem=…` | parc éolien déclaré (lot M5, opt-in `wind_farm.turbines`) : `wind_farm` (wind-farm.schema.json), `alerts`, `kpi` hub_wind_* ; `404` sans parc — voir `docs/EOLIEN.md` |
| `GET /api/export/{stem}/manifest.json` | manifeste d'un export |
| `GET /api/export/{stem}/{u,v,w,speed,terrain}.bin` | binaires Float32 |
| `POST /api/reconstruct/stations` | mesures **réelles** de stations (JSON, lat/lon) → volume 3D, mémorisé sous `stem` = `live_…` ; exige un domaine géoréférencé (`409` sinon), voir `docs/REAL_DATA.md` |
| `GET /api/keys` | liste des clés — **portée `admin`** |

```bash
curl -H "X-API-Key: pcf_..." "http://127.0.0.1:8000/api/point?x=5000&y=5000&z=120"
```

`/api/meta` porte l'ancrage géographique du domaine, lu dans la configuration `real` :

```json
"georef": {"lat0": 36.0, "lon0": 9.7, "rotation_deg": 0.0, "suppose": true, "coin": "sud-ouest"}
```

`(lat0, lon0)` est le coin **sud-ouest**, origine de `x` et `y`. `suppose: true` veut dire
que l'ancrage a été **choisi**, pas mesuré (domaine synthétique). `georef` vaut `null` si
la configuration n'en a pas.

```json
{"ix": 100, "iy": 100, "x_m": 5000.0, "y_m": 5000.0, "z_asl_m": 120.0,
 "z_terrain_m": 41.58, "wind_u_m_s": -6.064, "wind_v_m_s": -6.671,
 "wind_w_m_s": -0.174, "wind_speed_m_s": 9.016, "wind_direction_deg": 42.27,
 "timestamp_utc": "2025-01-06 00:00:00"}
```

## Export binaire (volume complet)

Un instant complet fait 12 × 250 × 500 × 4 octets par variable (~6 Mo), trop pour du
JSON. `POST /api/reconstruct` avec `{"export": true, "stride": 1}` (portée `admin`) écrit
dans `run_dir/web/recon_<stem>/` :

```
manifest.json   grille, niveaux, bornes, liste des champs, valeur de remplacement
terrain.bin     (ny, nx)        float32 — altitude du relief en m
u.bin v.bin w.bin speed.bin     (nz, ny, nx) float32
```

- Ordre mémoire : **z, y, x**, ligne par ligne, du sud vers le nord et de l'ouest vers l'est.
- Sous le relief, la valeur est **`-9999.0`** (ni NaN ni null : les deux sont illisibles en
  Float32Array et en JSON).
- `stride` décime horizontalement : `2` divise le poids par 4.

Lecture côté navigateur :

```js
const m = await (await fetch(`${base}/manifest.json`, {headers})).json();
const speed = new Float32Array(await (await fetch(`${base}/speed.bin`, {headers})).arrayBuffer());
const {nz, ny, nx} = m.grille;
const at = (k, j, i) => {
  const v = speed[(k * ny + j) * nx + i];
  return v === m.nodata ? null : v;      // null = sous le relief
};
```

## Conventions à ne pas se tromper

- `u` positif vers l'**est**, `v` vers le **nord**, `w` vers le **haut**, en m/s.
- La direction est en **convention météo** : `(270 − atan2(v, u)) mod 360`, c'est-à-dire
  **d'où vient** le vent. 0 = vent du nord, 90 = vent d'est. L'inverser est une erreur de
  180° qui passe inaperçue.
- Les altitudes sont en **mètres au-dessus du niveau de la mer** (ASL), pas au-dessus du
  sol. `z_agl_m` donne la hauteur sol quand elle est utile.
- `x` et `y` sont en mètres depuis le coin **sud-ouest** du domaine (repère métrique
  local, pas des degrés).
- Les altitudes intermédiaires sont **interpolées linéairement** entre les niveaux du
  modèle ; en dehors de `[25, 1000]` m, l'API répond `400`.
- Un point sous le relief vaut `null` en JSON et `-9999.0` en binaire. Ne jamais faire de
  moyenne sans écarter ces valeurs.

## Limites actuelles, à connaître avant de brancher une application

- **Les instants servis sont ceux du cache local**, construit depuis le dataset ANEO. Pour
  un instant nouveau, il faut passer `POST /api/reconstruct/upload` avec un CSV de mesures
  de surface — ce chemin ne demande aucun cache, seulement les poids. Il n'y a pas
  d'ingestion temps réel.
- **Sur Colab, l'URL n'est pas stable** (voir plus haut) : à traiter comme une
  configuration rechargeable, jamais comme une constante compilée.
- **Un seul modèle est chargé par processus**, choisi au démarrage
  (`best_dso_v7.pt`, sinon `best.pt`). Changer de modèle demande un redémarrage.
- Les volumes sont mis en cache en mémoire (4 instants au plus) : un balayage de
  nombreux instants recalcule, à environ une inférence par instant.
- `GET /api/dso` et `dso_report` renvoient ce qui a été **archivé** par
  `pcfno3d evaluate`. Si rien n'a été archivé, ils le disent — aucun chiffre n'est estimé.
- Pas de HTTPS dans le service : à placer derrière un reverse proxy ou un tunnel pour
  une exposition publique. Quotas et journal d'accès existent, mais les quotas sont
  comptés en mémoire d'un seul processus (voir Authentification).
