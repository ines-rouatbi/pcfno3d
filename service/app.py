"""Application web PC-FNO3D : interface, API d'inference et serveur MCP.

Un seul processus expose trois choses :

    /                 l'application web (carte, profil vertical, point, DSO)
    /api/...          l'API JSON, protegee par cle d'API
    /mcp              le serveur MCP (streamable HTTP), meme cle d'API

Le modele est charge une fois au demarrage et partage par les trois.

    pip install -e ".[serve]"
    python -m pcfno3d apikey create app-web --scope read
    python -m pcfno3d serve --port 8000

Variables d'environnement :
    PCFNO3D_CONFIG      nom ou chemin du YAML (defaut : default)
    PCFNO3D_CHECKPOINT  checkpoint a charger (defaut : best_dso_v7.pt sinon best.pt)
    PCFNO3D_DEVICE      cuda / cpu
    PCFNO3D_API_KEYS    cles supplementaires, "nom:cle,nom:cle" (portee read)
    PCFNO3D_ALLOWED_HOSTS  hotes autorises pour MCP (defaut : localhost, 127.0.0.1)
    PCFNO3D_QUOTA_MINUTE / PCFNO3D_QUOTA_JOUR   quotas par cle (service/quotas.py)
    PCFNO3D_ACCESS_LOG  journal d'acces JSONL (defaut run_dir/logs/acces.jsonl)
    PCFNO3D_INSECURE_OPEN  "1" : mode ouvert sans cle, developpement local seulement

Ferme par defaut : sans aucune cle active, tout est refuse (401) sauf /api/health, qui
dit « aucune cle configuree ». Le mode ouvert demande un opt-in explicite
(`pcfno3d serve --insecure-open`) et n'est jamais accepte sur le chemin Colab / tunnel.

Un seul garde-fou (`make_guard`) controle /api et /mcp : cle, quota, journal d'acces.
Les dependances `require_read` / `require_admin` des routes ne verifient plus que la
portee de la cle deja authentifiee.

`create_app()` est une fabrique : le gestionnaire de sessions MCP ne peut tourner qu'une
fois par instance, donc chaque demarrage (uvicorn, test) construit sa propre application.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import (APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, Security,
                     UploadFile)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

from pcfno3d.config import Config
from pcfno3d.export import export_web
from pcfno3d.inference import (FIABILITE_DESACTIVEE, ReliabilityError, Volume, WindFieldModel,
                               noeud_fiabilite_nul)
from pcfno3d.windfarm import round_direction

from . import mcp_server
from .auth import (NO_KEY_MESSAGE, OPEN_WARNING, ApiKeyRecord, ApiKeyStore, default_store_path,
                   extract_key, insecure_open_policy, insecure_open_requested, scope_allows,
                   summarize)
from .journal import JournalAcces, params_sans_secret
from .link import Upstream, install as install_link
from .quotas import Quotas

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
MAX_UPLOAD_MB = 200
MAX_CACHED_VOLUMES = 4
EXPORT_FILES = {"u.bin", "v.bin", "w.bin", "speed.bin", "direction.bin", "terrain.bin"}
# routes sans cle : la sante (la webapp l'appelle avant de demander une cle) et la
# documentation, qui ne contient aucune donnee. Tout le reste de /api, et /mcp, exige une cle.
API_PUBLIQUES = frozenset({"/api/health", "/api/docs", "/api/docs/oauth2-redirect",
                           "/api/openapi.json"})
LOG = logging.getLogger("pcfno3d.service")
# mode ouvert : seules ces adresses, sans aucun en-tete de proxy, sont servies sans cle
LOOPBACK_IPS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})
PROXY_HEADERS = ("x-forwarded-for", "forwarded", "cf-connecting-ip", "x-real-ip")
OPEN_REMOTE_MESSAGE = ("mode ouvert reserve aux requetes directes de la boucle locale "
                       "(127.0.0.1 / ::1, sans proxy) : cle d'API requise")

# etat du processus : une seule application est active a la fois
STATE: dict = {}
_VOLUME_LOCK = threading.Lock()


# ---------------------------------------------------------------------------- contexte
def _load_config() -> Config:
    return Config.load(os.environ.get("PCFNO3D_CONFIG") or "default").apply_service_env()


def _allowed_hosts() -> list[str]:
    """Garde-fou anti-DNS-rebinding du transport MCP : seuls ces hotes sont acceptes.

    L'en-tete Host porte le port, donc il faut declarer `127.0.0.1:<port>` et pas
    seulement `127.0.0.1` : `pcfno3d serve` renseigne PCFNO3D_PORT, et tout autre nom
    d'hote (nom de machine, domaine derriere un proxy) s'ajoute par
    PCFNO3D_ALLOWED_HOSTS.
    """
    port = (os.environ.get("PCFNO3D_PORT") or "8000").strip()
    base = ["testserver"]
    for host in ("localhost", "127.0.0.1"):
        base += [host, f"{host}:{port}"]
    extra = [h.strip() for h in (os.environ.get("PCFNO3D_ALLOWED_HOSTS") or "").split(",")
             if h.strip()]
    return base + extra


def model() -> WindFieldModel:
    wf = STATE.get("model")
    if wf is None:
        raise HTTPException(503, "modele non charge")
    return wf


def keys() -> ApiKeyStore:
    store = STATE.get("keys")
    if store is None:
        raise HTTPException(503, "trousseau non charge")
    return store


# ---------------------------------------------------------------------------- acces
def _check(request: Request, required: str) -> ApiKeyRecord | None:
    """Portee de la cle deja authentifiee par `make_guard`.

    Si le garde-fou n'est pas passe (routeur monte ailleurs), on authentifie ici : le
    defaut reste ferme.
    """
    if getattr(request.state, "acces_ouvert", False):
        return None                       # mode ouvert explicite, aucune cle active
    record = getattr(request.state, "cle", None)
    if record is None:
        store = keys()
        if store.open_mode and requete_locale(request):
            return None
        record = store.verify(extract_key(request.headers, request.query_params))
    if record is None:
        raise HTTPException(401, "cle d'API absente ou invalide (en-tete X-API-Key)",
                            headers={"WWW-Authenticate": "Bearer"})
    if not scope_allows(record, required):
        raise HTTPException(403, f"cle '{record.name}' : portee '{record.scope}' "
                                 f"insuffisante (requis : {required})")
    return record


# Declaration seulement, pour la documentation OpenAPI (bouton « Authorize » de Swagger) :
# la verification reste celle de `_check` / `make_guard` (X-API-Key, Bearer ou ?api_key=).
CLE_API = APIKeyHeader(name="X-API-Key", auto_error=False, scheme_name="cle_api",
                       description="Cle d'API du service (pcf_...), creee par `pcfno3d apikey create`.")


def require_read(request: Request, _cle: str | None = Security(CLE_API)) -> ApiKeyRecord | None:
    return _check(request, "read")


def require_admin(request: Request, _cle: str | None = Security(CLE_API)) -> ApiKeyRecord | None:
    return _check(request, "admin")


class MCPPathNormalizer:
    """Fait repondre /mcp directement, sans redirection 307 vers /mcp/.

    Starlette redirige /mcp vers /mcp/ parce que le serveur MCP est monte comme
    sous-application. C'est l'URL /mcp qui est publiee aux clients : certains ne
    rejouent pas le corps d'un POST apres une redirection, donc on reecrit le chemin
    avant le routage plutot que de renvoyer un 307.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and scope.get("path") == "/mcp":
            scope = dict(scope, path="/mcp/", raw_path=b"/mcp/")
        await self.app(scope, receive, send)


def requete_locale(request: Request) -> bool:
    """Requete DIRECTE de la boucle locale : adresse 127.0.0.1 / ::1 et aucun en-tete de
    proxy. Un reverse proxy local (nginx, cloudflared) se connecte bien depuis 127.0.0.1,
    mais pour le compte d'un client distant : il pose un de ces en-tetes, et la requete
    n'est alors plus consideree comme locale."""
    hote = request.client.host if request.client else None
    if hote not in LOOPBACK_IPS:
        return False
    return not any(request.headers.get(h) for h in PROXY_HEADERS)


def _protege(chemin: str, publiques: frozenset[str]) -> bool:
    if chemin == "/mcp" or chemin.startswith("/mcp/"):
        return True
    return (chemin == "/api" or chemin.startswith("/api/")) and chemin not in publiques


def _refus(chemin: str, statut: int, message: str, code_rpc: int,
           headers: dict | None = None) -> JSONResponse:
    """Refus au format de l'interface : JSON-RPC pour /mcp, {"detail"} pour /api."""
    if chemin.startswith("/mcp"):
        body = {"jsonrpc": "2.0", "id": None, "error": {"code": code_rpc, "message": message}}
    else:
        body = {"detail": message}
    return JSONResponse(body, status_code=statut, headers=headers)


def _minute() -> int:
    return int(time.time() // 60)


def _refus_a_journaliser(ip: str | None) -> int | None:
    """Regroupement des 401 : une ligne par IP et par minute au plus.

    Renvoie None si la ligne doit etre sautee, sinon le nombre de 401 de cette IP qui
    n'ont pas ete ecrits depuis sa ligne precedente (`refus_regroupes`). Un flot de cles
    fausses ne purge donc pas l'historique par la rotation.
    """
    etat: dict = STATE.setdefault("refus", {})
    minute = _minute()
    with _REFUS_LOCK:
        courant = etat.get(ip)
        if courant is not None and courant[0] == minute:
            courant[1] += 1
            return None
        regroupes = courant[1] if courant is not None else 0
        if len(etat) >= 10_000:                      # memoire bornee, cout O(1) amorti :
            etat.clear()                             # on oublie tout plutot que de parcourir
        etat[ip] = [minute, 0]
    return regroupes


_REFUS_LOCK = threading.Lock()


def _journaliser(request: Request, record: ApiKeyRecord | None, portee: str | None,
                 statut: int, debut: float, motif: str | None = None, reponse=None) -> None:
    """Une ligne du journal d'acces : jamais le secret, ni le hash complet."""
    journal = STATE.get("journal")
    if journal is None:
        return
    regroupes = None
    if int(statut) == 401:
        regroupes = _refus_a_journaliser(request.client.host if request.client else None)
        if regroupes is None:
            return
    champs: dict = {"cle": record.name if record else None,
                    "empreinte": record.empreinte if record else None,
                    "portee": portee, "methode": request.method,
                    "chemin": request.url.path,             # sans query string
                    "statut": int(statut),
                    "duree_ms": round((time.perf_counter() - debut) * 1000.0, 1),
                    "ip": request.client.host if request.client else None}
    params = params_sans_secret(request.query_params)
    if params:
        champs["params"] = params
    # derriere cloudflared ou un proxy, l'IP vue est 127.0.0.1 : on garde aussi l'IP
    # DECLAREE par le proxy, a part (sans proxy, ce champ peut etre forge)
    transmise = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
    if transmise:
        champs["ip_transmise"] = transmise[:200]
    if reponse is not None and reponse.headers.get("content-length", "").isdigit():
        champs["octets"] = int(reponse.headers["content-length"])
    if motif:
        champs["motif"] = motif
    if regroupes:
        champs["refus_regroupes"] = regroupes
    journal.ecrire(**champs)


def make_guard(publiques: frozenset[str] = API_PUBLIQUES, locale: bool = False):
    """Le garde-fou unique de /api et /mcp : cle valide, quota, puis journal d'acces.

    Le MCP est monte en ASGI et ne passe pas par Depends : il est filtre ici, comme /api.
    Une requete OPTIONS (pre-vol CORS, sans cle par construction) passe : le middleware
    CORS y repond sans atteindre aucune route.
    """
    invalide = ("cle d'API locale absente ou invalide" if locale
                else "cle d'API absente ou invalide (en-tete X-API-Key)")

    async def guard(request: Request, call_next):
        chemin = request.url.path
        if request.method == "OPTIONS" or not _protege(chemin, publiques):
            return await call_next(request)
        debut = time.perf_counter()
        store: ApiKeyStore | None = STATE.get("keys")
        record: ApiKeyRecord | None = None
        portee: str | None = None
        motif: str | None = None
        reponse = None
        if store is None:
            motif = "trousseau_absent"
            reponse = _refus(chemin, 503, "trousseau non charge", -32003)
        elif store.open_mode:
            if requete_locale(request):
                request.state.acces_ouvert = True
                portee = "ouvert"
            else:
                motif = "ouvert_hors_boucle_locale"
                reponse = _refus(chemin, 401, OPEN_REMOTE_MESSAGE, -32001,
                                 {"WWW-Authenticate": "Bearer"})
        else:
            presentee = extract_key(request.headers, request.query_params)
            record = store.verify(presentee)
            if record is None:
                if store.empty:
                    motif = "aucune_cle_configuree"
                    message = (f"{store.erreur} ; " if store.erreur else "") + NO_KEY_MESSAGE
                else:
                    motif = "cle_invalide" if presentee else "cle_absente"
                    message = invalide
                reponse = _refus(chemin, 401, message, -32001, {"WWW-Authenticate": "Bearer"})
            else:
                portee = record.scope
                quotas: Quotas | None = STATE.get("quotas")
                depasse = quotas.consommer(record) if quotas is not None else None
                if depasse is not None:
                    motif = f"quota_{depasse.limite}"
                    reponse = _refus(chemin, 429, depasse.message(record.name), -32004,
                                     {"Retry-After": str(depasse.retry_after)})
                elif chemin.startswith("/mcp") and not scope_allows(record, "admin"):
                    # un outil qui ecrit sur le disque demande la portee admin
                    body = await request.body()
                    if any(name.encode() in body for name in mcp_server.ADMIN_TOOLS):
                        motif = "portee"
                        reponse = _refus(chemin, 403, "outil reserve a la portee admin",
                                         -32002)
                if reponse is None:
                    request.state.cle = record
        if reponse is None:
            try:
                reponse = await call_next(request)
            except Exception:
                _journaliser(request, record, portee, 500, debut, "exception")
                raise
        _journaliser(request, record, portee, reponse.status_code, debut, motif, reponse)
        return reponse

    return guard


def _securite(cfg: Config) -> dict:
    """Trousseau, quotas et journal d'un demarrage, avec les avertissements qui comptent."""
    ouvert, raison = insecure_open_policy()
    store = ApiKeyStore(default_store_path(cfg.paths.run_dir), allow_open=ouvert)
    if ouvert:
        LOG.warning(OPEN_WARNING + ("" if store.empty else
                                    " Inactif tant que des cles existent : elles sont exigees."))
    elif insecure_open_requested():
        LOG.warning(raison)
    if store.empty and not ouvert:
        LOG.warning("aucune cle d'API active : toutes les requetes seront refusees (401) "
                    "sauf /api/health. `pcfno3d apikey create <nom>` pour en creer une.")
    return {"keys": store, "quotas": Quotas.from_env(),
            "journal": JournalAcces.from_env(cfg.paths.run_dir)}


# ---------------------------------------------------------------------------- helpers
def _stem_par_defaut(wf: WindFieldModel) -> str:
    """Premier instant de test du cache (sinon le premier instant)."""
    from pcfno3d.cache import load_index
    index = load_index(wf.cfg)
    test = index[index["split"] == "test"]
    rows = test if len(test) else index
    if rows.empty:
        raise HTTPException(400, "aucun instant en cache : precise 'stem'")
    return str(rows["stem"].iloc[0])


def _differee(wf: WindFieldModel) -> bool:
    """Fiabilite configuree : le service la DIFFERE (valeur servie d'abord, fiabilite dans
    un thread d'arriere-plan, rangee dans le meme volume du cache)."""
    return wf.reliability is not None


def _kw_differee(wf: WindFieldModel) -> dict:
    """Argument `defer` des appels a predict : present seulement si la fiabilite est active
    (desactivee, les appels restent exactement ceux d'avant)."""
    return {"defer": True} if _differee(wf) else {}


def _memoriser(cache_key: str, volume: Volume) -> None:
    """Range un volume dans le cache des volumes (le plus ancien sort au-dela de 4)."""
    with _VOLUME_LOCK:
        cache = STATE.setdefault("volumes", {})
        cache[cache_key] = volume
        if len(cache) > MAX_CACHED_VOLUMES:
            cache.pop(next(iter(cache)))


def _fiabilite_prete(volume: Volume) -> Volume:
    """Attend la fiabilite differee (alertes M4, parc M5, export) : jamais un etat partiel.
    Fiabilite en echec : 503 avec la cause."""
    try:
        volume.wait_reliability()
    except ReliabilityError as exc:
        raise HTTPException(503, str(exc)) from exc
    return volume


def _etat_fiabilite(wf: WindFieldModel, volume: Volume, stem: str | None,
                    keep: float = 1.0) -> dict:
    """Cles de fiabilite d'une reponse de reconstruction ; {} si elle est desactivee (la
    reponse garde alors exactement ses cles d'avant)."""
    if not _differee(wf) or volume.reliability_state == FIABILITE_DESACTIVEE:
        return {}
    st = volume.reliability_status()
    out = {"fiabilite": st["fiabilite"], "fiabilite_s": st["fiabilite_s"]}
    if stem is not None:
        out["fiabilite_route"] = f"/api/reliability/{stem}" + (
            f"?keep={keep:.3f}" if abs(keep - 1.0) > 1e-9 else "")
    return out


def _volume_for(stem: str | None, keep: float = 1.0) -> Volume:
    """Volume d'un instant, memorise (un instant complet pese ~18 Mo en memoire).

    Fiabilite active : la valeur est servie sans l'attendre ; la fiabilite part dans un
    thread d'arriere-plan (un seul par volume) et se range dans ce meme volume du cache."""
    wf = model()
    if stem is None:
        stem = _stem_par_defaut(wf)
    cache = STATE.setdefault("volumes", {})
    cache_key = f"{stem}:{keep:.3f}"
    volume = cache.get(cache_key)
    if volume is not None:
        wf.assess_reliability_async(volume)      # sans effet si deja lancee ou prete
        return volume
    # double verification sous verrou : l'application web demande la carte, le resume et
    # les fleches EN MEME TEMPS pour un nouvel instant. La premiere requete calcule, les
    # autres attendent et relisent le cache — sinon chacune relancerait la meme passe
    # (13 s et 5 Go chacune pour la v7 sur CPU).
    with _VOLUME_LOCK:
        volume = cache.get(cache_key)
        if volume is None:
            try:
                volume = wf.predict_stem(stem, keep=keep, **_kw_differee(wf))
            except KeyError as exc:
                raise HTTPException(404, f"instant inconnu : {stem}") from exc
            cache[cache_key] = volume
            if len(cache) > MAX_CACHED_VOLUMES:
                cache.pop(next(iter(cache)))
    wf.assess_reliability_async(volume)
    return volume


class ReconstructRequest(BaseModel):
    stem: str | None = Field(None, description='instant en cache, ex. "004_120000"')
    keep: float = Field(1.0, ge=0.0, le=1.0, description="fraction de capteurs utilises")
    export: bool = Field(False, description="ecrire aussi les binaires du visualiseur "
                                            "(portee admin)")
    stride: int = Field(1, ge=1, le=8, description="decimation horizontale de l'export")


# ---------------------------------------------------------------------------- routes
router = APIRouter(prefix="/api")


@router.get("/health", tags=["service"])
def health() -> dict:
    """Etat du service. Seule route publique : elle dit si une cle est exigee."""
    wf = STATE.get("model")
    store = STATE.get("keys")
    out = {"status": "ok" if wf else "degraded",
           "modele": wf.checkpoint_path.name if wf else None,
           "device": str(wf.device) if wf else None,
           "mcp": "/mcp"}
    if store is not None:
        out.update(summarize(store))
    return out


@router.get("/meta", tags=["service"])
def meta(_: ApiKeyRecord | None = Depends(require_read)) -> dict:
    wf = model()
    out = wf.meta()
    g = wf.geom
    out["domaine_m"] = {"x_max": float((g.nx - 1) * g.dx), "y_max": float((g.ny - 1) * g.dy)}
    out["georef"] = georef(wf.cfg)
    return out


def georef(cfg: Config) -> dict | None:
    """Ancrage geographique du domaine (config `real`), ou None s'il n'est pas configure.

    (lat0, lon0) est le coin SUD-OUEST, origine de x et y ; `suppose` = True quand
    l'ancrage est choisi et non mesure (cas du domaine synthetique).
    """
    real = cfg.real
    if real.origin_lat is None or real.origin_lon is None:
        return None
    return {"lat0": float(real.origin_lat), "lon0": float(real.origin_lon),
            "rotation_deg": float(real.rotation_deg), "suppose": bool(real.origin_assumed),
            "coin": "sud-ouest"}


@router.get("/instants", tags=["donnees"])
def instants(split: str | None = Query(None, pattern="^(train|val|test)$"),
             limit: int = Query(100, ge=1, le=5000),
             _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    from pcfno3d.cache import load_index

    wf = model()
    try:
        index = load_index(wf.cfg)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    if split:
        index = index[index["split"] == split]
    rows = index[["stem", "timestamp", "split"]].head(limit)
    return {"total": int(len(index)), "instants": rows.to_dict(orient="records")}


@router.post("/reconstruct", tags=["donnees"])
def reconstruct(req: ReconstructRequest,
                record: ApiKeyRecord | None = Depends(require_read)) -> dict:
    wf = model()
    if req.export and not (keys().open_mode or scope_allows(record, "admin")):
        raise HTTPException(403, "l'export sur disque demande la portee admin")
    volume = _volume_for(req.stem, keep=req.keep)
    # fiabilite differee : DSO4 juge la valeur seule (inference_s) ; la reponse n'attend
    # pas la fiabilite, elle dit son etat (en_cours / prete) et ou le suivre
    valid = wf.geom.valid
    out: dict = {
        "stem": req.stem, "timestamp_utc": volume.timestamp,
        "capteurs_utilises": volume.n_sensors,
        "inference_s": round(volume.inference_s, 4),
        "dso4_conforme": volume.inference_s < wf.cfg.dso.time_s,
        "grille": {"nz": volume.u.shape[0], "ny": volume.u.shape[1], "nx": volume.u.shape[2]},
        "noeuds_3d": int(valid.sum()),
        "vitesse_max_m_s": round(float(volume.speed[valid].max()), 3),
        "vitesse_moyenne_m_s": round(float(volume.speed[valid].mean()), 3),
    }
    if _differee(wf):
        out.update(_etat_fiabilite(wf, volume, req.stem or _stem_par_defaut(wf), req.keep))
    if req.export:
        _fiabilite_prete(volume)          # un export sur disque porte la fiabilite complete
        name = f"recon_{req.stem or 'latest'}"
        out["export"] = str(export_web(volume, wf.cfg.paths.web_root / name,
                                       stride=req.stride))
    return out


@router.post("/reconstruct/upload", tags=["donnees"])
async def reconstruct_upload(file: UploadFile = File(...), keep: float = 1.0,
                             _: ApiKeyRecord | None = Depends(require_read)):
    """Reconstruction depuis un CSV de mesures de surface d'un seul instant.

    Le fichier passe d'abord par le contrat de donnees ANEO (pcfno3d.contract), profil
    upload (`Seuils.upload()`) : une station absente, en mask_i_obs = 0, interpolee ou sans
    u/v/T est une panne CONNUE, que le modele traite (capteur indisponible, DSO5). Seul un
    REJET (erreur bloquante ; anomalies > 1 % des lignes stations ET > 5 lignes ; moins de
    30 % de lignes declarees mesurees valides) renvoie 422 avec le rapport sous `qualite`.
    Conforme ou en quarantaine -> la reconstruction, avec `statut_qualite`,
    `avertissements` (motifs de quarantaine compris) et `qualite`. Les stations que le
    contrat juge fausses (hors plage, densite, figees, sans u/v/T) sont MASQUEES avant
    l'inference : le modele ne recoit jamais une valeur que le rapport signale.
    """
    wf = model()
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"fichier trop gros (> {MAX_UPLOAD_MB} Mo)")
    # contrat, inference et alertes : des secondes (minutes sur CPU avec les alertes) de
    # calcul bloquant, hors de la boucle d'evenements (sinon /api/health attend aussi)
    return await run_in_threadpool(_upload_calcul, wf, data, file.filename, keep)


def _upload_calcul(wf: WindFieldModel, data: bytes, filename: str | None, keep: float):
    """Partie bloquante de l'upload, executee dans un thread du pool."""
    from pcfno3d.contract import Grille, Seuils, valider_fichier

    # gzip reconnu a ses octets magiques : lecteurs (pyarrow, pandas) decompressent
    # d'apres l'extension, quel que soit le nom envoye par le client
    suffix = ".csv.gz" if data[:2] == b"\x1f\x8b" else ".csv"
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        masque: dict = {}
        grille = Grille.depuis_meta(wf.geom.meta.raw)
        rapport = valider_fichier(tmp_path, grille, Seuils.upload(),
                                  nom=filename or tmp_path.name, masque=masque)
        if rapport["verdict"] == "rejet":
            return JSONResponse(status_code=422, content={
                "detail": f"CSV non conforme au contrat ANEO ({rapport['verdict']}) : "
                          + " ; ".join(rapport["motifs"]),
                "qualite": rapport})
        moteur = STATE.get("alertes")
        differee = _differee(wf)
        from service import atom_outils
        if atom_outils.entree_production_active(wf):
            # modele reel : capteurs du CSV comme stations + GFS de l'heure (entree de production)
            S0, hour, doy, stamp = wf.surface_from_csv(tmp_path, masque)
            S, vec = _entree_csv_production(wf, S0, stamp)
            volume = wf.predict(S, hour, doy, timestamp=stamp, keep=keep, forcage=vec, **_kw_differee(wf))
        elif moteur is None:
            volume = wf.predict_csv(tmp_path, keep=keep, exclure=masque, **_kw_differee(wf))
        else:
            # meme passe que predict_csv, en gardant S pour les alertes (lot M4)
            S, hour, doy, stamp = wf.surface_from_csv(tmp_path, masque)
            volume = wf.predict(S, hour, doy, timestamp=stamp, keep=keep, **_kw_differee(wf))
        if differee:
            # fiabilite en arriere-plan, volume range dans le cache (lisible ensuite par
            # /profile, /reliability... sous `stem`)
            stem = _stem_upload(volume.timestamp, keep)
            _memoriser(f"{stem}:1.000", volume)
            wf.assess_reliability_async(volume)
        if moteur is not None:
            from pcfno3d.alerts import stations_hors_plage
            hp = stations_hors_plage(tmp_path, grille.n_stations, grille.node_id_start,
                                     rapport["seuils"]["tol_densite_rel"])
            # les alertes ATTENDENT la fiabilite (couverture, sigma, membres du parc)
            _fiabilite_prete(volume)
            alertes = moteur.evaluer_modele(wf, S, hour, doy, volume, rapport=rapport,
                                            hors_plage=hp)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, f"CSV invalide : {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    valid = wf.geom.valid
    out = {"timestamp_utc": volume.timestamp, "capteurs_utilises": volume.n_sensors,
           "inference_s": round(volume.inference_s, 4),
           "vitesse_max_m_s": round(float(volume.speed[valid].max()), 3),
           "statut_qualite": rapport["verdict"],
           "avertissements": rapport["motifs"] + rapport["avertissements"],
           "qualite": rapport}
    if differee:
        out["stem"] = stem
        out.update(_etat_fiabilite(wf, volume, stem))
    if moteur is not None:
        # alertes activees seulement : sinon la reponse garde exactement ses cles
        out["alertes"] = alertes.document()
        out["physique"] = alertes.physics
    return out


def _stem_upload(timestamp: str | None, keep: float) -> str:
    """Cle de cache d'un volume televerse (fiabilite active seulement)."""
    import pandas as pd

    t = pd.Timestamp(timestamp) if timestamp else pd.Timestamp.now(tz="UTC")
    suffixe = "" if abs(keep - 1.0) < 1e-9 else f"_keep{keep:.2f}"
    return f"upload_{t.strftime('%Y%m%d_%H%M%S')}{suffixe}"


class StationObs(BaseModel):
    station_id: str
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    height_agl_m: float = Field(10.0, description="hauteur du capteur au-dessus du sol")
    wind_u_m_s: float | None = None
    wind_v_m_s: float | None = None
    wind_speed_m_s: float | None = None
    wind_dir_deg: float | None = Field(None, description="direction meteo, d'ou vient le vent")
    wind_w_m_s: float | None = None
    temperature_c: float | None = None
    pressure_pa: float | None = None
    relative_humidity_pct: float | None = None
    quality: float | None = Field(None, ge=0, le=1)


class StationsRequest(BaseModel):
    timestamp_utc: str = Field(..., description="ex. 2026-10-14T12:00:00Z")
    stations: list[StationObs] = Field(..., min_length=1, max_length=5000)


def _entree_stations_production(wf, geo, clean, ts):
    """Stations (lat, lon) -> entree de production du modele reel : vent ramene a 10 m par le profil
    logarithmique de la maille, une mesure par maille (moyenne), GFS de l'heure en grande echelle."""
    import numpy as np

    from service import atom_outils
    g = wf.geom
    x, y = geo.to_xy(clean["lat"].to_numpy(), clean["lon"].to_numpy())
    i, j = np.rint(np.asarray(x) / g.dx).astype(int), np.rint(np.asarray(y) / g.dy).astype(int)
    dedans = (i >= 0) & (i < g.nx) & (j >= 0) & (j < g.ny)
    if not dedans.any():
        raise HTTPException(400, "aucune station dans le domaine")
    par_maille: dict = {}
    for k in np.flatnonzero(dedans):
        z0 = max(float(g.z0[j[k], i[k]]), 1e-4)
        h = max(float(clean["height_agl_m"].iloc[k]), 1.0)
        f = np.log(10.0 / z0) / np.log(h / z0) if abs(h - 10.0) > 0.5 and h > z0 * 1.5 else 1.0
        par_maille.setdefault((j[k], i[k]), []).append((clean["wind_u_m_s"].iloc[k] * f,
                                                        clean["wind_v_m_s"].iloc[k] * f))
    pts = [(jj, ii, float(np.mean([m[0] for m in ms])), float(np.mean([m[1] for m in ms])))
           for (jj, ii), ms in par_maille.items()]
    try:
        S, vec, t = atom_outils.entree_production(wf, ts, pts)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:                                   # noqa: BLE001 — reseau, Open-Meteo
        raise HTTPException(503, f"grande echelle GFS indisponible : {exc}") from exc
    return S, vec, {"stations": int(dedans.sum()), "mailles": len(pts), "hors_domaine": int((~dedans).sum()),
                    "entree": "production : GFS de l'heure + stations a 10 m", "heure_gfs_utc": str(t)}


def _entree_csv_production(wf, S0, stamp):
    """Capteurs d'un CSV ANEO -> entree de production du modele reel (capteur = station a sa maille)."""
    import numpy as np

    from service import atom_outils
    jj, ii = np.nonzero(S0[7] > 0.5)
    pts = [(a, b, float(S0[0, a, b]), float(S0[1, a, b])) for a, b in zip(jj, ii)]
    try:
        S, vec, _ = atom_outils.entree_production(wf, stamp, pts)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(503, f"grande echelle GFS indisponible : {exc}") from exc
    return S, vec


@router.post("/reconstruct/stations", tags=["donnees"])
def reconstruct_stations(req: StationsRequest,
                         _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Reconstruction depuis des mesures REELLES de stations (lat/lon), un instant.

    Le volume obtenu est memorise sous `stem` ("live_..."), reutilisable ensuite par
    /field, /point, /profile. Exige un domaine georeference (config `real.origin_*`).
    """
    import pandas as pd

    from pcfno3d.realdata import GeoRef, normalize_stations, stations_to_surface

    wf = model()
    try:
        geo = GeoRef.from_cfg(wf.cfg)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    df = pd.DataFrame([s.model_dump() for s in req.stations])
    df["timestamp_utc"] = req.timestamp_utc
    try:
        clean, rapport = normalize_stations(df, wf.cfg)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"mesures invalides : {exc}") from exc
    ts = pd.Timestamp(req.timestamp_utc).tz_localize(None) \
        if pd.Timestamp(req.timestamp_utc).tzinfo else pd.Timestamp(req.timestamp_utc)
    hour = ts.hour + ts.minute / 60 + ts.second / 3600
    from service import atom_outils
    if atom_outils.entree_production_active(wf):
        # modele reel : l'entree de production (GFS de l'heure + stations a 10 m), pas la couche dense
        S, vec, info = _entree_stations_production(wf, geo, clean, ts)
        volume = wf.predict(S, hour, int(ts.dayofyear), timestamp=str(ts), real_sensors=True,
                            forcage=vec, **_kw_differee(wf))
    else:
        S, info = stations_to_surface(clean, wf.geom, geo, wf.cfg)
        if info["mailles"] == 0:
            raise HTTPException(400, f"aucune station exploitable dans le domaine ({info}, "
                                     f"controle : {rapport})")
        # mesures de stations REELLES (information de fiabilite ; le statut n'en depend pas)
        volume = wf.predict(S, hour, int(ts.dayofyear), timestamp=str(ts), real_sensors=True,
                            **_kw_differee(wf))
    stem = f"live_{ts.strftime('%Y%m%d_%H%M%S')}"
    _memoriser(f"{stem}:1.000", volume)
    wf.assess_reliability_async(volume)          # fiabilite en arriere-plan, si active
    valid = wf.geom.valid
    return {"stem": stem, "timestamp_utc": str(ts), "stations": info, "controle": rapport,
            "georeferencement": "suppose (domaine synthetique, position choisie)"
            if wf.cfg.real.origin_assumed else "mesure",
            "inference_s": round(volume.inference_s, 4),
            "vitesse_max_m_s": round(float(volume.speed[valid].max()), 3),
            **_etat_fiabilite(wf, volume, stem)}


@router.get("/field", tags=["donnees"])
def field(z: float = Query(..., description="altitude ASL en m"),
          stem: str | None = None,
          variable: str = Query("speed", pattern="^(speed|u|v|w|direction)$"),
          stride: int = Query(4, ge=1, le=16),
          _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Carte d'une variable a une altitude, decimee pour rester legere en JSON."""
    volume = _volume_for(stem)
    source = {"speed": volume.speed, "u": volume.u, "v": volume.v, "w": volume.w,
              "direction": volume.direction}[variable]
    try:
        plane = volume.at_altitude(z, source)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    plane = plane[::stride, ::stride]
    g = volume.geom
    terrain = g.zt[::stride, ::stride]
    finite = plane[np.isfinite(plane)]
    return {
        "timestamp_utc": volume.timestamp, "variable": variable, "z_asl_m": z,
        "stride": stride, "dx_m": g.dx * stride, "dy_m": g.dy * stride,
        "ny": int(plane.shape[0]), "nx": int(plane.shape[1]),
        "min": round(float(finite.min()), 3) if finite.size else None,
        "max": round(float(finite.max()), 3) if finite.size else None,
        "terrain_min_m": round(float(terrain.min()), 1),
        "terrain_max_m": round(float(terrain.max()), 1),
        # None la ou le point est sous le relief : JSON n'admet pas NaN
        "valeurs": [[None if not np.isfinite(v) else round(float(v), 3) for v in row]
                    for row in plane],
    }


@router.get("/point", tags=["donnees"])
def point(x: float = Query(..., description="x en m depuis le coin sud-ouest"),
          y: float = Query(..., description="y en m"),
          z: float = Query(..., description="altitude ASL en m"),
          stem: str | None = None,
          _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    volume = _volume_for(stem)
    try:
        return volume.at_point(x, y, z)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/profile", tags=["donnees"])
def profile(x: float = Query(...), y: float = Query(...), stem: str | None = None,
            _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Profil vertical complet en un point : une entree par niveau du modele.

    Avec la fiabilite active (`inference.reliability`, lot M2), chaque niveau porte aussi
    sigma_speed, sigma_u, sigma_v, sigma_w, speed_interval (p10, p50, p90), status,
    coverage, reliability, reliability_class — noms du contrat ATOM rev. 2 — et la reponse
    le bloc `uncertainty` (methode, membres, calibration) et `fiabilite`, l'etat de la
    fiabilite differee : tant qu'elle vaut `en_cours` (ou `echec`), ces champs sont null ;
    `prete` : les valeurs. Desactivee : aucune de ces cles.
    """
    volume = _volume_for(stem)
    g = volume.geom
    i = int(np.clip(round(x / g.dx), 0, g.nx - 1))
    j = int(np.clip(round(y / g.dy), 0, g.ny - 1))
    # etat lu UNE fois : un profil ne melange pas deux etats de la fiabilite
    etat = volume.reliability_state
    active = etat != FIABILITE_DESACTIVEE
    prete = active and volume.reliability is not None and etat == "prete"
    if active and not prete:
        etat = "echec" if etat == "echec" else "en_cours"
    niveaux = []
    for k, zl in enumerate(g.z_levels):
        u, v, w = volume.u[k, j, i], volume.v[k, j, i], volume.w[k, j, i]
        ok = bool(np.isfinite(u))
        niveaux.append({
            "z_asl_m": float(zl), "z_agl_m": round(float(zl - g.zt[j, i]), 2),
            "sous_le_relief": not ok,
            "wind_speed_m_s": round(float(np.hypot(u, v)), 3) if ok else None,
            # arrondi AVANT le modulo (regle m4 du contrat) : jamais 360,0
            "wind_direction_deg": float(round_direction(270.0 - np.degrees(np.arctan2(v, u))))
            if ok else None,
            "wind_w_m_s": round(float(w), 4) if ok else None,
        })
        if active:
            niveaux[-1].update(volume.node_reliability(k, j, i) if prete
                               else noeud_fiabilite_nul())
    out = {"timestamp_utc": volume.timestamp, "x_m": i * g.dx, "y_m": j * g.dy,
           "z_terrain_m": float(g.zt[j, i]), "niveaux": niveaux}
    if active:
        out["uncertainty"] = volume.reliability.uncertainty if prete else None
        out["fiabilite"] = etat
    return out


@router.get("/reliability/{stem}", tags=["qualite"])
def reliability_state(stem: str, keep: float = Query(1.0, ge=0.0, le=1.0),
                      _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Etat de la fiabilite differee d'un volume deja reconstruit : `fiabilite`
    (desactivee, en_cours, prete, echec), `inference_s` (la valeur, DSO4), `fiabilite_s`
    (delai de la fiabilite, null tant qu'elle n'est pas prete), `erreur`. Route legere :
    ne lance aucun calcul ; 404 si l'instant n'est pas dans le cache des volumes."""
    model()
    volume = STATE.get("volumes", {}).get(f"{stem}:{keep:.3f}")
    if volume is None:
        raise HTTPException(404, f"aucun volume en cache pour {stem} (keep {keep:.3f}) : "
                                 "POST /api/reconstruct d'abord")
    return {"stem": stem, **volume.reliability_status()}


def _alertes_stem(stem: str | None):
    """Alertes d'un instant du cache (lot M4), memorisees par instant dans le moteur."""
    moteur = STATE.get("alertes")
    if moteur is None:
        raise HTTPException(409, "alertes desactivees : alerts.enabled = false dans la "
                                 "configuration du service")
    wf = model()
    stem = stem or _stem_par_defaut(wf)
    volume = _fiabilite_prete(_volume_for(stem))     # couverture, sigma : fiabilite prete
    try:
        S, hour, doy, _ = wf.surface_from_cache(stem)
    except KeyError as exc:
        raise HTTPException(404, f"instant inconnu : {stem}") from exc
    return moteur.evaluer_modele(wf, S, hour, doy, volume, Y=moteur.verite(wf, stem),
                                 cle=f"stem:{stem}")


@router.get("/alerts", tags=["qualite"])
def alerts(stem: str | None = Query(None, description='instant en cache, ex. "004_120000" ; '
                                                       "absent : dernieres alertes calculees"),
           _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Alertes de l'instant (lot M4, AN1), au format `alerts.schema.json` du contrat ATOM
    rev. 2 : `{"items": [...]}`, ouvertes et closes recentes, avec niveau, cause et
    recommandation. Sans `stem` : les dernieres alertes calculees (upload compris), sinon
    le premier instant de test. 409 si les alertes sont desactivees (`alerts.enabled`)."""
    moteur = STATE.get("alertes")
    if moteur is not None and stem is None and moteur.dernier is not None:
        return moteur.dernier.document()
    return _alertes_stem(stem).document()


@router.get("/wind-farm", tags=["eolien"])
def wind_farm(stem: str | None = None,
              _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Parc eolien a un instant (lot M5) : vent au moyeu et REWS (p10/p50/p90), cisaillement
    (shear_alpha), veer, densite, puissance et etat par eolienne, production du parc,
    alertes vent_fort_moyeu / cisaillement et KPI hub_wind_* du domaine.

    `wind_farm` suit wind-farm.schema.json et `alerts` alerts.schema.json (contrat ATOM
    rev. 2). Les hauteurs sont au-dessus du SOL. 404 si aucun parc n'est declare
    (`wind_farm.turbines`, opt-in). Les intervalles demandent la fiabilite M2 active.
    """
    from pcfno3d.windfarm import farm_report

    wf = model()
    if wf.farm is None:
        raise HTTPException(404, "aucun parc eolien declare (configuration "
                                 "wind_farm.turbines, voir docs/EOLIEN.md)")
    stem = stem or _stem_par_defaut(wf)
    # quantiles du parc sur les membres de l'ensemble : fiabilite prete, jamais partielle
    out = farm_report(_fiabilite_prete(_volume_for(stem)), wf)
    if STATE.get("alertes") is not None:
        # alertes actives : les items du journal M4 (premiere levee, ouverte / close) ;
        # sinon, alertes brutes de M5 (raised_at = l'instant)
        from pcfno3d.alerts import items_externes
        out["alerts"]["items"] = items_externes(_alertes_stem(stem).items)
    return out


@router.get("/dso", tags=["qualite"])
def dso(_: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Dernier verdict DSO1-DSO5 archive, ou l'absence d'evaluation, sans rien inventer."""
    return mcp_server.dso_report()


# ---------------------------------------------------------------- front-end ATOM Wind
_ATOM_LOCK = threading.Lock()
_ATOM_JOURS_MAX = 6                     # journees gardees en memoire (quelques Mo chacune)


def _gzip_json(data: dict) -> Response:
    import gzip
    import json as _json
    corps = gzip.compress(_json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8"), compresslevel=5)
    return Response(corps, media_type="application/json",
                    headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})


def _atom_jours(wf: WindFieldModel) -> list[dict]:
    from pcfno3d import atom_wind
    from pcfno3d.cache import load_index
    j = STATE.get("atom_jours")
    if j is None:
        try:
            j = atom_wind.jours(load_index(wf.cfg), split="test")
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        STATE["atom_jours"] = j
    return j


@router.get("/atom-wind/grille", tags=["atom-wind"])
def atom_grille(_: ApiKeyRecord | None = Depends(require_read)) -> Response:
    """Relief, eau, z0 et stations sur la grille de 500 m du front-end ATOM Wind."""
    from service import atom_outils
    return _gzip_json(atom_outils.grille(model()))


@router.get("/atom-wind/jours", tags=["atom-wind"])
def atom_jours(_: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Les journees servies (instants de test du cache), en heure locale de Tunis (UTC+1), et
    la journee en temps reel (`direct`) quand le modele recoit le forcage ERA5."""
    from pcfno3d import atom_wind
    wf = model()
    out = {"modele": wf.checkpoint_path.name, "jours": _atom_jours(wf)}
    if wf.geom.forcage is not None and wf.cfg.real.origin_lat is not None:
        out["direct"] = {"date": atom_wind.date_locale_aujourdhui(),
                         "route": "/api/atom-wind/direct",
                         "previsions": [atom_wind.date_locale_aujourdhui(j) for j in (1, 2)]}
    return out


@router.get("/atom-wind/direct", tags=["atom-wind"])
def atom_direct(j: int = Query(0, ge=0, le=2, description="0 = aujourd'hui (temps reel), "
                               "1 = demain, 2 = apres-demain (prevision)"),
                _: ApiKeyRecord | None = Depends(require_read)) -> Response:
    """Le vent d'AUJOURD'HUI (heure locale de Tunis), heure par heure : grande echelle GFS
    (Open-Meteo) aux mailles et par les formules du forcage ERA5, vraies mesures des stations
    (METAR, SYNOP) pour les heures passees, prevision pour les heures a venir. Meme format que
    /atom-wind/jour, avec `temps_reel: true` et la source de chaque heure. Cache de 30 min
    (service.atom_outils, partage avec le MCP et l'assistant)."""
    wf = model()
    if wf.geom.forcage is None:
        raise HTTPException(409, "temps reel : modele sans forcage ERA5 (paths.forcage_era5)")
    from service import atom_outils
    try:
        return _gzip_json(atom_outils.jour(wf, j))
    except Exception as exc:                    # source externe muette, echeance absente...
        raise HTTPException(503, f"temps reel indisponible : {exc}") from exc


@router.get("/atom-wind/jour", tags=["atom-wind"])
def atom_jour(date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
              _: ApiKeyRecord | None = Depends(require_read)) -> Response:
    """Vent u, v aux hauteurs AGL de chaque instant d'une journee (grille de 500 m, int16 en
    cm/s, base64), + sigma de la vitesse si la fiabilite (M2) est active."""
    from pcfno3d import atom_wind
    wf = model()
    j = next((d for d in _atom_jours(wf) if d["date"] == date), None)
    if j is None:
        raise HTTPException(404, f"aucun instant servi le {date}")
    cache = STATE.setdefault("atom_jour", {})
    out = cache.get(date)
    if out is None:
        with _ATOM_LOCK:
            out = cache.get(date)
            if out is None:
                fiab = None
                if wf.cfg.inference.reliability is not None:
                    fiab = lambda vol: atom_wind.sigma_agl(vol, wf.geom)  # noqa: E731
                out = atom_wind.jour(wf.predict_stem, wf.geom, j, fiab)
                out["modele"] = wf.checkpoint_path.name
                while len(cache) >= _ATOM_JOURS_MAX:
                    cache.pop(next(iter(cache)))
                cache[date] = out
    return _gzip_json(out)


@router.get("/atom-wind/ensemble", tags=["atom-wind"])
def atom_ensemble(j: int = Query(0, ge=0, le=2, description="0 = aujourd'hui, 1 = demain, 2 = apres-demain"),
                  _: ApiKeyRecord | None = Depends(require_read)) -> Response:
    """Prevision PROBABILISTE (cahier v12, E19) : le modele sur les 31 membres GEFS (ensemble partiel),
    probabilite de pouvoir traiter et quantiles 10 / 50 / 90 % du vent a 10 m et 100 m, aux heures de
    traitement (6, 9, 12, 15, 18 h), grille de 500 m. Cache de 6 h (rythme de GEFS)."""
    wf = model()
    if wf.geom.forcage is None:
        raise HTTPException(409, "ensemble : modele sans forcage (paths.forcage_era5)")
    from service import atom_outils
    try:
        return _gzip_json(atom_outils.ensemble(wf, j))
    except Exception as exc:
        raise HTTPException(503, f"ensemble indisponible : {exc}") from exc


@router.get("/atom-wind/maintenance", tags=["atom-wind"])
def atom_maintenance(j: int = Query(0, ge=0, le=2, description="0 = aujourd'hui, 1 = demain, 2 = apres-demain"),
                     lieu: str | None = Query(None, max_length=80), lat: float | None = None, lon: float | None = None,
                     hauteur_m: float = Query(80.0, ge=2, le=1000), seuil_m_s: float = Query(12.0, gt=0, le=40),
                     duree_min_h: int = Query(2, ge=1, le=12),
                     _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Creneaux surs pour le travail en hauteur (cahier v12, E7) : vent moyen a la nacelle ou au mat sous
    le seuil du site, en plages d'au moins `duree_min_h` heures (parc virtuel par defaut)."""
    from service import atom_outils
    try:
        return atom_outils.fenetres_maintenance(model(), j, hauteur_m, seuil_m_s, duree_min_h, lat=lat, lon=lon, lieu=lieu)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/atom-wind/variations", tags=["atom-wind"])
def atom_variations(j: int = Query(0, ge=0, le=2, description="0 = aujourd'hui, 1 = demain, 2 = apres-demain"),
                    moyeu_m: float = Query(80.0, ge=30, le=200), seuil_pct: float = Query(20.0, gt=0, le=100),
                    _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Variations brusques de production du parc virtuel (cahier v12, E13) : puissance heure par heure et
    rampes d'au moins `seuil_pct` % de la puissance installee par heure."""
    from service import atom_outils
    try:
        return atom_outils.variations_production(model(), j, moyeu_m, seuil_pct)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/atom-wind/valeur-stations", tags=["atom-wind"])
def atom_valeur_stations(_: ApiKeyRecord | None = Depends(require_read)) -> Response:
    """Valeur de chaque station et son influence (cahier v12, E10 et E11), aujourd'hui : le modele est relance
    sans chaque station aux heures mesurees (toutes les 3 h). Cache de 30 min."""
    wf = model()
    if wf.geom.forcage is None:
        raise HTTPException(409, "valeur des stations : modele sans forcage (paths.forcage_era5)")
    from service import atom_outils
    try:
        return _gzip_json(atom_outils.valeur_stations(wf))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/atom-wind/explication", tags=["atom-wind"])
def atom_explication(lieu: str | None = Query(None, max_length=80), lat: float | None = None,
                     lon: float | None = None, _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Pourquoi ce vent ici (cahier v12, E10) : influence de chaque station sur le point aujourd'hui."""
    from service import atom_outils
    try:
        return atom_outils.explication_point(model(), lat=lat, lon=lon, lieu=lieu)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/atom-wind/dimensionnement", tags=["atom-wind"])
def atom_dimensionnement(lieu: str | None = Query(None, max_length=80), lat: float | None = None,
                         lon: float | None = None, hauteur_m: float = Query(30.0, ge=2, le=150),
                         conso_w: float = Query(5.0, gt=0, le=2000), autonomie_j: float = Query(3.0, ge=0, le=30),
                         _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Dimensionnement energetique d'un mat de mesure (cahier v12, E18) : micro-eolienne a la hauteur du mat,
    bilan avec la consommation de la station et taille de batterie, sur aujourd'hui a apres-demain."""
    from service import atom_outils
    try:
        return atom_outils.dimensionnement_mat(model(), lat=lat, lon=lon, lieu=lieu, hauteur_m=hauteur_m,
                                               conso_w=conso_w, autonomie_j=autonomie_j)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/atom-wind/rapport.pdf", tags=["atom-wind"])
def atom_rapport(type: str = Query("parcelle", pattern="^(parcelle|site|parc)$"),
                 lieu: str | None = Query(None, max_length=80), lat: float | None = None, lon: float | None = None,
                 moyeu_m: float = Query(80.0, ge=30, le=200),
                 _: ApiKeyRecord | None = Depends(require_read)) -> Response:
    """Rapport PDF automatique (cahier v12, E9) d'une parcelle, d'un site ou du parc virtuel : vent, rose des
    vents, production, alertes, recommandations et fiabilite, d'aujourd'hui a apres-demain."""
    from service import rapport_pdf
    try:
        pdf = rapport_pdf.construire(model(), type, lat=lat, lon=lon, lieu=lieu, moyeu_m=moyeu_m)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    nom = rapport_pdf.nom_fichier(type, lieu, lat, lon)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{nom}"'})


@router.get("/atom-wind/qualite", tags=["atom-wind"])
def atom_qualite(_: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Qualite du modele servi (cahier v12, TB5) : verdicts DSO1 a DSO6 sur le test, ecart aux vraies
    stations et decision de livraison, lus dans son run_dir (rien n'est recalcule)."""
    wf = model()
    from pcfno3d import atom_wind
    try:
        return atom_wind.qualite(wf.cfg.paths.run_dir, wf.cfg.dso, bool(wf.cfg.inference.w_continuite))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/atom-wind/agricole", tags=["atom-wind"])
def atom_agricole(j: int = Query(0, ge=0, le=2, description="0 = aujourd'hui, 1 = demain, 2 = apres-demain"),
                  lieu: str | None = Query(None, max_length=80), lat: float | None = None, lon: float | None = None,
                  _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Decision agricole d'une parcelle (cahier v12, S16) : creneaux de traitement et d'aspersion heure par
    heure (regle de l'application, vent a 10 m) et probabilite GEFS si elle est calculee."""
    from service import atom_outils
    try:
        return atom_outils.decision_agricole(model(), j, lat=lat, lon=lon, lieu=lieu)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/atom-wind/production", tags=["atom-wind"])
def atom_production(j: int = Query(0, ge=0, le=2, description="0 = aujourd'hui, 1 = demain, 2 = apres-demain"),
                    moyeu_m: float = Query(80.0, ge=30, le=200), lieu: str | None = Query(None, max_length=80),
                    lat: float | None = None, lon: float | None = None,
                    _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Production prevue d'une eolienne de reference en un point (cahier v12, S16), heure par heure."""
    from service import atom_outils
    if lieu is None and (lat is None or lon is None):
        lieu = "djebel ressas"
    try:
        return atom_outils.production_eolienne(model(), j, moyeu_m, lat=lat, lon=lon, lieu=lieu)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/atom-wind/historique", tags=["atom-wind"])
def atom_historique(_: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Historique (cahier v12, BF40, HIS1) : journees temps reel et previsions deja calculees, gardees par
    le service avec leurs indicateurs (une prevision se compare ensuite au temps reel du meme jour)."""
    from service import atom_outils
    return {"journees": atom_outils.historique(model())}


@router.get("/atom-wind/historique/{nom}", tags=["atom-wind"])
def atom_historique_jour(nom: str, _: ApiKeyRecord | None = Depends(require_read)) -> Response:
    """Une journee archivee, au format de /atom-wind/direct (gzip)."""
    from service import atom_outils
    try:
        corps = atom_outils.lire_historique(model(), nom)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, f"archive absente : {exc}") from exc
    return Response(corps, media_type="application/json", headers={"Content-Encoding": "gzip"})


class QuestionAssistant(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    historique: list[dict] = Field(default_factory=list, max_length=20,
                                   description='[{"role": "user"|"assistant", "contenu": "..."}]')


@router.post("/assistant", tags=["atom-wind"])
def assistant(q: QuestionAssistant, _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Assistant conversationnel (cahier v12, E6) : Claude repond a une question en langage naturel en
    appelant les outils du service (vent en un point, profil, decision agricole, production eolienne,
    alertes). Demande ANTHROPIC_API_KEY sur le service ; 503 sinon."""
    from service import assistant as asst
    wf = model()
    try:
        return asst.repondre(wf, q.question, q.historique)
    except asst.AssistantIndisponible as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:                    # erreur de l'API Claude (quota, reseau...)
        LOG.warning("assistant : %s", exc)
        raise HTTPException(502, f"assistant indisponible : {type(exc).__name__}") from exc


class AbonneSms(BaseModel):
    numero: str = Field(..., description="numero E.164, ex. +21612345678")
    langue: str = Field("fr", pattern="^(fr|ar)$")
    lieu: str | None = None
    lat: float | None = None
    lon: float | None = None


def _dossier_sms() -> Path:
    return model().cfg.paths.run_dir / "sms"


@router.get("/sms/abonnes", tags=["sms"])
def sms_abonnes(_: ApiKeyRecord | None = Depends(require_admin)) -> dict:
    """Abonnes aux alertes SMS (E12), numeros masques. Portee administrateur."""
    from service import sms
    ab = sms.Abonnes(_dossier_sms()).lire()
    return {"abonnes": [{**a, "numero": sms.masquer(a["numero"])} for a in ab]}


@router.post("/sms/abonnes", tags=["sms"])
def sms_abonner(a: AbonneSms, _: ApiKeyRecord | None = Depends(require_admin)) -> dict:
    from service import sms
    try:
        r = sms.Abonnes(_dossier_sms()).ajouter(a.numero, a.langue, a.lieu, a.lat, a.lon)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {**r, "numero": sms.masquer(r["numero"])}


@router.delete("/sms/abonnes/{numero}", tags=["sms"])
def sms_desabonner(numero: str, _: ApiKeyRecord | None = Depends(require_admin)) -> dict:
    from service import sms
    if not sms.Abonnes(_dossier_sms()).retirer(numero):
        raise HTTPException(404, "abonne inconnu")
    return {"retire": sms.masquer(numero)}


@router.post("/sms/envoyer", tags=["sms"])
def sms_envoyer(jour: int = Query(1, ge=0, le=2), simulation: bool = Query(True),
                _: ApiKeyRecord | None = Depends(require_admin)) -> dict:
    """Compose et envoie le SMS du jour `jour` a chaque abonne. simulation=true (defaut) : journal
    seulement ; false : fournisseur configure (PCFNO3D_SMS=twilio et ses identifiants)."""
    from service import sms
    try:
        return sms.envoyer_tous(model(), _dossier_sms(), jour=jour, simulation=simulation)
    except KeyError as exc:
        raise HTTPException(503, f"fournisseur SMS non configure : {exc}") from exc


def _sms_quotidien(stop: threading.Event) -> None:
    """Envoi automatique des SMS du lendemain a PCFNO3D_SMS_HEURE (heure de Tunis), une fois par jour."""
    from service import sms
    heure = int(os.environ["PCFNO3D_SMS_HEURE"])
    dernier = None
    while not stop.wait(60):
        t = time.gmtime(time.time() + 3600)                     # heure de Tunis (UTC+1)
        jour = time.strftime("%Y-%m-%d", t)
        if t.tm_hour == heure and dernier != jour and STATE.get("model") is not None:
            dernier = jour
            try:
                r = sms.envoyer_tous(STATE["model"], STATE["model"].cfg.paths.run_dir / "sms", jour=1)
                LOG.info("SMS quotidiens : %s abonnes (%s)", r["abonnes"], r["fournisseur"])
            except Exception as exc:                            # noqa: BLE001
                LOG.warning("SMS quotidiens : %s", exc)


def _atlas_dossier(wf: WindFieldModel) -> Path | None:
    for d in (wf.cfg.paths.run_dir / "atlas_eolien", wf.cfg.paths.run_dir.parent / "atlas_eolien"):
        if (d / "atlas.json").exists() and (d / "atlas.npz").exists():
            return d
    return None


@router.get("/atom-wind/atlas", tags=["atom-wind"])
def atom_atlas(_: ApiKeyRecord | None = Depends(require_read)) -> Response:
    """Atlas eolien (cahier v12, E8 ; scripts/atlas_eolien.py) : resume, trois meilleurs sites, et la
    vitesse moyenne et le facteur de charge au moyeu sur la grille de 500 m du front-end."""
    import json as _json

    from pcfno3d import atom_wind
    wf = model()
    d = _atlas_dossier(wf)
    if d is None:
        raise HTTPException(404, "atlas eolien non calcule (scripts/atlas_eolien.py)")
    out = _json.loads((d / "atlas.json").read_text(encoding="utf-8"))
    b = atom_wind.facteur(wf.geom)
    with np.load(d / "atlas.npz") as z:
        for cle in ("moyenne", "facteur_charge"):
            bloc = atom_wind.blocs(z[cle].astype(float), b)
            out[cle] = atom_wind.encoder(bloc)
            out["forme"] = list(bloc.shape)
    out["unite"] = "moyenne en m/s, facteur_charge en fraction ; int16 x100, -32768 = sans valeur"
    out["couches_geotiff"] = sorted(p.stem.removeprefix("atlas_") for p in d.glob("atlas_*.tif"))
    return _gzip_json(out)


@router.get("/atom-wind/atlas/{nom}", tags=["atom-wind"])
def atom_atlas_fichier(nom: str, _: ApiKeyRecord | None = Depends(require_read)):
    """atlas.png, ou une couche GeoTIFF (export SIG, E15) : atlas_<couche>.tif."""
    wf = model()
    d = _atlas_dossier(wf)
    if d is None:
        raise HTTPException(404, "atlas eolien non calcule")
    if nom == "atlas.png":
        return FileResponse(d / "atlas.png", media_type="image/png")
    f = d / nom
    if not (nom.startswith("atlas_") and nom.endswith(".tif") and f.exists() and f.parent == d):
        raise HTTPException(404, "fichier inconnu")
    return FileResponse(f, media_type="image/tiff", filename=nom)


@router.get("/atom-wind/reseau-aneo", tags=["atom-wind"])
def atom_reseau(_: ApiKeyRecord | None = Depends(require_read)):
    """Plan de deploiement ANEO (OS6, scripts/os6_reseau_aneo.py) en GeoJSON, s'il existe."""
    from pcfno3d import atom_wind
    f = atom_wind.plan_aneo(model().cfg.paths.run_dir)
    if f is None:
        raise HTTPException(404, "plan OS6 non calcule (scripts/os6_reseau_aneo.py)")
    return FileResponse(f, media_type="application/geo+json")


@router.get("/exports/{stem}.{fmt}", tags=["donnees"])
def export_instant_fichier(stem: str, fmt: str, _: ApiKeyRecord | None = Depends(require_read)) -> Response:
    """Export d'un instant du cache (cahier v12, BF47, S6 ; contrat DE3) : NetCDF (.nc, CF) ou CSV des
    noeuds (.csv), avec la version du modele, les statistiques de normalisation et l'horodatage source.
    Les memes valeurs que l'instantane publie (pcfno3d.export_noeuds)."""
    import tempfile

    from pcfno3d.cache import load_index
    from pcfno3d.export_noeuds import exporter_csv, exporter_netcdf
    from pcfno3d.publish import instantane_du_cache

    if fmt not in ("nc", "csv"):
        raise HTTPException(422, "format : nc ou csv")
    wf = model()
    if stem not in set(load_index(wf.cfg)["stem"].astype(str)):
        raise HTTPException(404, f"instant inconnu : {stem}")
    genre = "reel" if wf.cfg.mask.entree == "era5_stations" else "synthetique"
    try:
        inst = instantane_du_cache(wf, stem, data_kind=genre, garder_champs=True)
    except ValueError as exc:                    # domaine non georeference, contrat
        raise HTTPException(409, str(exc)) from exc
    with tempfile.TemporaryDirectory() as d:
        f = (exporter_netcdf(inst, wf.geom, Path(d) / f"{inst.snapshot_id}.nc") if fmt == "nc"
             else exporter_csv(inst, wf.geom, Path(d) / f"{inst.snapshot_id}_noeuds.csv"))
        corps = Path(f).read_bytes()
        nom = Path(f).name
    return Response(corps, media_type="application/x-netcdf" if fmt == "nc" else "text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{nom}"'})


@router.get("/export/{stem}/manifest.json", tags=["donnees"])
def export_manifest(stem: str, _: ApiKeyRecord | None = Depends(require_read)):
    """Manifeste d'un export deja produit (POST /api/reconstruct avec export=true)."""
    path = model().cfg.paths.web_root / f"recon_{stem}" / "manifest.json"
    if not path.exists():
        raise HTTPException(404, f"aucun export pour {stem}")
    return FileResponse(path, media_type="application/json")


@router.get("/export/{stem}/{name}", tags=["donnees"])
def export_file(stem: str, name: str, _: ApiKeyRecord | None = Depends(require_read)):
    """Binaire Float32 d'un export (u.bin, v.bin, w.bin, speed.bin, terrain.bin)."""
    if name not in EXPORT_FILES:
        raise HTTPException(404, "fichier inconnu")
    path = model().cfg.paths.web_root / f"recon_{stem}" / name
    if not path.exists():
        raise HTTPException(404, f"{name} absent pour {stem}")
    return FileResponse(path, media_type="application/octet-stream")


@router.get("/keys", tags=["service"])
def list_keys(_: ApiKeyRecord | None = Depends(require_admin)) -> dict:
    """Liste des cles, sans secret : seule une empreinte est renvoyee."""
    return {"cles": keys().list_public()}


class NouvelleCle(BaseModel):
    nom: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    portee: str = Field("read", pattern="^(read|admin)$")
    quota_minute: int | None = Field(None, ge=1)
    quota_jour: int | None = Field(None, ge=1)


class QuotasCle(BaseModel):
    quota_minute: int | None = Field(None, ge=1)
    quota_jour: int | None = Field(None, ge=1)
    defaut: bool = Field(False, description="revenir aux quotas par defaut")


@router.post("/keys", tags=["service"])
def create_key(c: NouvelleCle, _: ApiKeyRecord | None = Depends(require_admin)) -> dict:
    """Cree une cle (cahier v12, BF42) ; le secret n'est renvoye qu'UNE fois, seule son empreinte est
    gardee. Portee admin seulement."""
    try:
        secret, rec = keys().create(c.nom, scope=c.portee, quota_minute=c.quota_minute, quota_jour=c.quota_jour)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"cle": secret, "avertissement": "secret affiche une seule fois : le noter maintenant", **rec.public()}


@router.delete("/keys/{nom}", tags=["service"])
def revoke_key(nom: str, _: ApiKeyRecord | None = Depends(require_admin)) -> dict:
    """Revoque une cle : elle cesse de fonctionner a la requete suivante (critere API1)."""
    if not keys().revoke(nom):
        raise HTTPException(404, f"aucune cle active nommee {nom}")
    return {"revoquee": nom}


@router.patch("/keys/{nom}", tags=["service"])
def quota_key(nom: str, q: QuotasCle, _: ApiKeyRecord | None = Depends(require_admin)) -> dict:
    """Quotas d'une cle active (par minute, par jour) ou retour aux valeurs par defaut."""
    try:
        rec = keys().set_quota(nom, q.quota_minute, q.quota_jour, reset=q.defaut)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if rec is None:
        raise HTTPException(404, f"aucune cle active nommee {nom}")
    return rec.public()


# ---------------------------------------------------------------------------- fabrique
def create_app() -> FastAPI:
    """Construit une application complete : API + MCP + interface web.

    Deux modes :
      - normal : le modele est charge ici et sert les requetes ;
      - lien   : aucun modele local, tout `/api` et `/mcp` est transmis a une instance
                 distante (typiquement Colab). Voir service/link.py.

    Chaque appel cree son propre gestionnaire de sessions MCP, qui ne peut etre demarre
    qu'une fois ; c'est pour cela que l'application est produite par une fabrique plutot
    que definie au niveau du module.
    """
    upstream = Upstream.resolve(_load_config().paths.run_dir)
    if upstream is not None:
        return _create_linked_app(upstream)

    mcp_app = mcp_server.mcp.streamable_http_app(
        streamable_http_path="/", json_response=True, stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=_allowed_hosts(),
            allowed_origins=["*"]),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        cfg = _load_config()
        wf = WindFieldModel(cfg, checkpoint=os.environ.get("PCFNO3D_CHECKPOINT") or None,
                            device=os.environ.get("PCFNO3D_DEVICE") or None)
        STATE.update({"model": wf, "cfg": cfg, "volumes": {}, **_securite(cfg)})
        mcp_server.set_model(wf)          # le MCP reutilise le modele deja charge
        # alertes (lot M4) : strictement opt-in ; un seul moteur (un seul journal) pour
        # l'API et le MCP
        if cfg.alerts.enabled:
            from pcfno3d.alerts import MoteurAlertes
            STATE["alertes"] = MoteurAlertes(cfg)
        mcp_server.set_alertes(STATE.get("alertes"))
        # SMS quotidiens (E12) : seulement si PCFNO3D_SMS_HEURE est donnee
        arret_sms = threading.Event()
        if os.environ.get("PCFNO3D_SMS_HEURE"):
            threading.Thread(target=_sms_quotidien, args=(arret_sms,), daemon=True, name="sms").start()
        async with mcp_app.router.lifespan_context(mcp_app):
            yield
        arret_sms.set()
        STATE.clear()

    app = FastAPI(
        title="PC-FNO3D — champ de vent 3D",
        version="0.1.0",
        description="Reconstruction du vent (u, v, w) de 25 a 1000 m a partir du reseau "
                    "ANEO. Authentification par cle d'API : en-tete `X-API-Key`.",
        docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    # le dernier ajoute est le plus exterieur : normalisation de /mcp, puis CORS (pre-vol,
    # et en-tetes CORS meme sur un 401 / 429), puis le garde-fou
    app.middleware("http")(make_guard(API_PUBLIQUES))
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    app.add_middleware(MCPPathNormalizer)
    app.include_router(router)
    app.mount("/mcp", mcp_app)

    if WEBAPP_DIR.exists():
        app.mount("/app", StaticFiles(directory=WEBAPP_DIR, html=True), name="webapp")

        @app.get("/", include_in_schema=False)
        def index():
            return RedirectResponse("/app/")
    else:                                                   # pragma: no cover
        @app.get("/", include_in_schema=False)
        def index():
            return {"service": "pcfno3d", "api": "/api/docs", "mcp": "/mcp"}

    return app


def _create_linked_app(upstream: Upstream) -> FastAPI:
    """Mode lien : pas de modele ici, tout part vers l'amont.

    L'authentification locale est conservee : le navigateur presente une cle LOCALE, et
    c'est le service qui ajoute la cle de l'amont. Ainsi la cle de l'instance Colab ne
    quitte jamais cette machine.
    """
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        cfg = _load_config()
        STATE.update({"cfg": cfg, "upstream": upstream, "volumes": {}, **_securite(cfg)})
        print(f"mode lien : les requetes partent vers {upstream.url} ({upstream.label})")
        yield
        STATE.clear()

    app = FastAPI(
        title="PC-FNO3D — champ de vent 3D (lien)",
        version="0.1.0",
        description=f"Relais vers une instance distante ({upstream.label}). Le modele "
                    "tourne a distance ; cette adresse reste stable quand celle de "
                    "l'amont change.",
        docs_url=None, redoc_url=None, openapi_url=None,
        lifespan=lifespan,
    )
    # meme garde-fou qu'en mode normal, avec la cle LOCALE, avant toute transmission ;
    # seule /api/health est publique (rien de l'amont n'est relaye sans cle)
    app.middleware("http")(make_guard(frozenset({"/api/health"}), locale=True))
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.get("/api/health", tags=["service"])
    async def health_lien() -> dict:
        from .link import check

        store = STATE.get("keys")
        amont = await check(upstream)
        out = {"status": "ok" if amont.get("joignable") else "degraded",
               "mode": "lien",
               "amont": upstream.public(),
               "modele": amont.get("modele"),
               "device": amont.get("device"),
               "mcp": "/mcp",
               "sante_amont": amont}
        if store is not None:
            out.update(summarize(store))
        return out

    install_link(app, upstream)

    if WEBAPP_DIR.exists():
        app.mount("/app", StaticFiles(directory=WEBAPP_DIR, html=True), name="webapp")

        @app.get("/", include_in_schema=False)
        def index():
            return RedirectResponse("/app/")

    return app


app = create_app()
