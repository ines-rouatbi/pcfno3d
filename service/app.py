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

Si aucune cle n'existe, l'API repond sans authentification et le signale dans
/api/health : pratique en local, a ne pas exposer au reseau.

`create_app()` est une fabrique : le gestionnaire de sessions MCP ne peut tourner qu'une
fois par instance, donc chaque demarrage (uvicorn, test) construit sa propre application.
"""

from __future__ import annotations

import os
import tempfile
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
from .auth import (ApiKeyRecord, ApiKeyStore, default_store_path, extract_key, scope_allows,
                   summarize)

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
MAX_UPLOAD_MB = 200
MAX_CACHED_VOLUMES = 4
EXPORT_FILES = {"u.bin", "v.bin", "w.bin", "speed.bin", "direction.bin", "terrain.bin"}

# etat du processus : une seule application est active a la fois
STATE: dict = {}


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
    store = keys()
    if store.open_mode:
        return None                       # aucune cle definie : mode ouvert
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


async def protect_mcp(request: Request, call_next):
    """Le MCP est monte en ASGI : il ne passe pas par Depends, on le filtre ici."""
    if request.url.path.startswith("/mcp"):
        store = STATE.get("keys")
        if store is not None and not store.open_mode:
            record = store.verify(extract_key(request.headers, request.query_params))
            if record is None:
                return JSONResponse(
                    {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32001, "message": "cle d'API absente ou invalide"}},
                    status_code=401, headers={"WWW-Authenticate": "Bearer"})
            if not scope_allows(record, "admin"):
                # un outil qui ecrit sur le disque demande la portee admin
                body = await request.body()
                if any(name.encode() in body for name in mcp_server.ADMIN_TOOLS):
                    return JSONResponse(
                        {"jsonrpc": "2.0", "id": None,
                         "error": {"code": -32002,
                                   "message": "outil reserve a la portee admin"}},
                        status_code=403)
    return await call_next(request)


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
    if cache_key not in cache:
        try:
            cache[cache_key] = wf.predict_stem(stem, keep=keep)
        except KeyError as exc:
            raise HTTPException(404, f"instant inconnu : {stem}") from exc
        if len(cache) > MAX_CACHED_VOLUMES:
            cache.pop(next(iter(cache)))
    return cache[cache_key]


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
    return out


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
        out["export"] = str(export_web(volume, wf.cfg.paths.run_dir / "web" / name,
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
    path = model().cfg.paths.run_dir / "web" / f"recon_{stem}" / "manifest.json"
    if not path.exists():
        raise HTTPException(404, f"aucun export pour {stem}")
    return FileResponse(path, media_type="application/json")


@router.get("/export/{stem}/{name}", tags=["donnees"])
def export_file(stem: str, name: str, _: ApiKeyRecord | None = Depends(require_read)):
    """Binaire Float32 d'un export (u.bin, v.bin, w.bin, speed.bin, terrain.bin)."""
    if name not in EXPORT_FILES:
        raise HTTPException(404, "fichier inconnu")
    path = model().cfg.paths.run_dir / "web" / f"recon_{stem}" / name
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

    Chaque appel cree son propre gestionnaire de sessions MCP, qui ne peut etre demarre
    qu'une fois ; c'est pour cela que l'application est produite par une fabrique plutot
    que definie au niveau du module.
    """
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
        STATE.update({"model": wf, "cfg": cfg, "volumes": {},
                      "keys": ApiKeyStore(default_store_path(cfg.paths.run_dir))})
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
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    app.add_middleware(MCPPathNormalizer)
    app.middleware("http")(protect_mcp)
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


app = create_app()
