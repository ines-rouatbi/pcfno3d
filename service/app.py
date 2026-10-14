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
from fastapi import (APIRouter, Depends, FastAPI, File, HTTPException, Query, Request,
                     UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

from pcfno3d.config import Config
from pcfno3d.export import export_web
from pcfno3d.inference import Volume, WindFieldModel

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
    cfg = Config.load(os.environ.get("PCFNO3D_CONFIG") or "default")
    for var, key in (("PCFNO3D_RUN_DIR", "paths.run_dir"),
                     ("PCFNO3D_DATA_DIR", "paths.data_dir"),
                     ("PCFNO3D_CACHE_DIR", "paths.cache_dir")):
        if os.environ.get(var):
            cfg.apply_overrides({key: Path(os.environ[var])})
    return cfg


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


def require_read(request: Request) -> ApiKeyRecord | None:
    return _check(request, "read")


def require_admin(request: Request) -> ApiKeyRecord | None:
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
        if len(etat) > 10_000:                       # memoire bornee
            for cle in [k for k, v in etat.items() if v[0] != minute]:
                del etat[cle]
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
def _volume_for(stem: str | None, keep: float = 1.0) -> Volume:
    """Volume d'un instant, memorise (un instant complet pese ~18 Mo en memoire)."""
    wf = model()
    if stem is None:
        from pcfno3d.cache import load_index
        index = load_index(wf.cfg)
        test = index[index["split"] == "test"]
        rows = test if len(test) else index
        if rows.empty:
            raise HTTPException(400, "aucun instant en cache : precise 'stem'")
        stem = str(rows["stem"].iloc[0])
    cache = STATE.setdefault("volumes", {})
    cache_key = f"{stem}:{keep:.3f}"
    volume = cache.get(cache_key)
    if volume is not None:
        return volume
    # double verification sous verrou : l'application web demande la carte, le resume et
    # les fleches EN MEME TEMPS pour un nouvel instant. La premiere requete calcule, les
    # autres attendent et relisent le cache — sinon chacune relancerait la meme passe
    # (13 s et 5 Go chacune pour la v7 sur CPU).
    with _VOLUME_LOCK:
        volume = cache.get(cache_key)
        if volume is None:
            try:
                volume = wf.predict_stem(stem, keep=keep)
            except KeyError as exc:
                raise HTTPException(404, f"instant inconnu : {stem}") from exc
            cache[cache_key] = volume
            if len(cache) > MAX_CACHED_VOLUMES:
                cache.pop(next(iter(cache)))
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
    if req.export:
        name = f"recon_{req.stem or 'latest'}"
        out["export"] = str(export_web(volume, wf.cfg.paths.web_root / name,
                                       stride=req.stride))
    return out


@router.post("/reconstruct/upload", tags=["donnees"])
async def reconstruct_upload(file: UploadFile = File(...), keep: float = 1.0,
                             _: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Reconstruction depuis un CSV de mesures de surface d'un seul instant."""
    wf = model()
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"fichier trop gros (> {MAX_UPLOAD_MB} Mo)")
    with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        volume = wf.predict_csv(tmp_path, keep=keep)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, f"CSV invalide : {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    valid = wf.geom.valid
    return {"timestamp_utc": volume.timestamp, "capteurs_utilises": volume.n_sensors,
            "inference_s": round(volume.inference_s, 4),
            "vitesse_max_m_s": round(float(volume.speed[valid].max()), 3)}


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
    S, info = stations_to_surface(clean, wf.geom, geo, wf.cfg)
    if info["mailles"] == 0:
        raise HTTPException(400, f"aucune station exploitable dans le domaine ({info}, "
                                 f"controle : {rapport})")
    ts = pd.Timestamp(req.timestamp_utc).tz_localize(None) \
        if pd.Timestamp(req.timestamp_utc).tzinfo else pd.Timestamp(req.timestamp_utc)
    hour = ts.hour + ts.minute / 60 + ts.second / 3600
    volume = wf.predict(S, hour, int(ts.dayofyear), timestamp=str(ts))
    stem = f"live_{ts.strftime('%Y%m%d_%H%M%S')}"
    with _VOLUME_LOCK:
        cache = STATE.setdefault("volumes", {})
        cache[f"{stem}:1.000"] = volume
        if len(cache) > MAX_CACHED_VOLUMES:
            cache.pop(next(iter(cache)))
    valid = wf.geom.valid
    return {"stem": stem, "timestamp_utc": str(ts), "stations": info, "controle": rapport,
            "georeferencement": "suppose (domaine synthetique, position choisie)"
            if wf.cfg.real.origin_assumed else "mesure",
            "inference_s": round(volume.inference_s, 4),
            "vitesse_max_m_s": round(float(volume.speed[valid].max()), 3)}


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
    """Profil vertical complet en un point : une entree par niveau du modele."""
    volume = _volume_for(stem)
    g = volume.geom
    i = int(np.clip(round(x / g.dx), 0, g.nx - 1))
    j = int(np.clip(round(y / g.dy), 0, g.ny - 1))
    niveaux = []
    for k, zl in enumerate(g.z_levels):
        u, v, w = volume.u[k, j, i], volume.v[k, j, i], volume.w[k, j, i]
        ok = bool(np.isfinite(u))
        niveaux.append({
            "z_asl_m": float(zl), "z_agl_m": round(float(zl - g.zt[j, i]), 2),
            "sous_le_relief": not ok,
            "wind_speed_m_s": round(float(np.hypot(u, v)), 3) if ok else None,
            "wind_direction_deg": round(float((270.0 - np.degrees(np.arctan2(v, u))) % 360.0), 1)
            if ok else None,
            "wind_w_m_s": round(float(w), 4) if ok else None,
        })
    return {"timestamp_utc": volume.timestamp, "x_m": i * g.dx, "y_m": j * g.dy,
            "z_terrain_m": float(g.zt[j, i]), "niveaux": niveaux}


@router.get("/dso", tags=["qualite"])
def dso(_: ApiKeyRecord | None = Depends(require_read)) -> dict:
    """Dernier verdict DSO1-DSO5 archive, ou l'absence d'evaluation, sans rien inventer."""
    return mcp_server.dso_report()


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
        async with mcp_app.router.lifespan_context(mcp_app):
            yield
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
