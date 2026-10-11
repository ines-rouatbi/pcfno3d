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
pcfno3d apikey list                       # noms, portées, empreintes — jamais de secret
pcfno3d apikey revoke app-externe
```

Des clés peuvent aussi venir de l'environnement, pratique en conteneur :
`PCFNO3D_API_KEYS="app-externe:pcf_...,autre:pcf_..."` (portée `read`).

**Si aucune clé n'existe, le service répond sans authentification** et l'annonce dans
`/api/health` (`"auth": "ouvert"`). Confortable en local, à ne pas exposer au réseau.

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
| `GET /api/health` | état du service — **publique**, dit si une clé est exigée |
| `GET /api/meta` | grille, niveaux ASL, conventions, domaine en mètres |
| `GET /api/instants?split=test&limit=100` | instants disponibles |
| `POST /api/reconstruct` | `{stem, keep, export, stride}` → résumé de l'instant |
| `POST /api/reconstruct/upload` | CSV de mesures de surface → reconstruction |
| `GET /api/field?z=120&variable=speed&stride=4` | carte à une altitude |
| `GET /api/point?x=5000&y=5000&z=120` | vent en un point |
| `GET /api/profile?x=5000&y=5000` | profil vertical |
| `GET /api/dso` | dernier verdict DSO archivé |
| `GET /api/export/{stem}/manifest.json` | manifeste d'un export |
| `GET /api/export/{stem}/{u,v,w,speed,terrain}.bin` | binaires Float32 |
| `GET /api/keys` | liste des clés — **portée `admin`** |

```bash
curl -H "X-API-Key: pcf_..." "http://127.0.0.1:8000/api/point?x=5000&y=5000&z=120"
```

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
- Pas de HTTPS, pas de limitation de débit, pas de journal d'audit : à placer derrière un
  reverse proxy pour une exposition publique.
