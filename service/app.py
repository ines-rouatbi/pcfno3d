"""API HTTP d'inference PC-FNO3D (tache T9 — deploiement).

Le modele est charge une seule fois au demarrage : chaque requete ne fait qu'une passe
avant. Lancement :

    pip install -r service/requirements.txt
    python -m pcfno3d -c default serve --port 8000
    # ou directement :
    uvicorn service.app:app --port 8000

Variables d'environnement lues au demarrage :
    PCFNO3D_CONFIG      nom ou chemin du YAML de configuration (defaut : default)
    PCFNO3D_CHECKPOINT  checkpoint a charger (defaut : best_dso_v7.pt sinon best.pt)
    PCFNO3D_DEVICE      cuda / cpu (defaut : cuda si disponible)

Routes :
    GET  /health                   etat du service
    GET  /meta                     grille, niveaux, modele charge
    GET  /instants                 instants disponibles en cache
    POST /reconstruct              {stem} ou CSV de surface -> volume 3D (resume + liens)
    GET  /field?z=120&stem=...     carte du vent a une altitude (JSON)
    GET  /point?x=&y=&z=&stem=     vent en un point
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pcfno3d.config import Config
from pcfno3d.export import export_web
from pcfno3d.inference import Volume, WindFieldModel

STATE: dict = {}
MAX_UPLOAD_MB = 200


def _load_model() -> WindFieldModel:
    cfg = Config.load(os.environ.get("PCFNO3D_CONFIG") or "default")
    for var, key in (("PCFNO3D_RUN_DIR", "paths.run_dir"),
                     ("PCFNO3D_DATA_DIR", "paths.data_dir"),
                     ("PCFNO3D_CACHE_DIR", "paths.cache_dir")):
        if os.environ.get(var):
            cfg.apply_overrides({key: Path(os.environ[var])})
    return WindFieldModel(cfg, checkpoint=os.environ.get("PCFNO3D_CHECKPOINT") or None,
                          device=os.environ.get("PCFNO3D_DEVICE") or None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # charge le modele une fois pour toutes : l'inference doit tenir sous 1 s (DSO4)
    STATE["model"] = _load_model()
    STATE["cache"] = {}
    yield
    STATE.clear()


app = FastAPI(title="PC-FNO3D — reconstruction du vent 3D", version="0.1.0",
              lifespan=lifespan)
# le visualiseur peut etre servi depuis un autre port
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def model() -> WindFieldModel:
    wf = STATE.get("model")
    if wf is None:
        raise HTTPException(503, "modele non charge")
    return wf


def _volume_for(stem: str | None) -> Volume:
    """Volume d'un instant en cache, memorise (un instant = ~18 Mo en memoire)."""
    wf = model()
    if stem is None:
        from pcfno3d.cache import load_index
        test = load_index(wf.cfg)
        test = test[test["split"] == "test"]
        if test.empty:
            raise HTTPException(400, "aucun instant de test en cache : precise 'stem'")
        stem = str(test["stem"].iloc[0])
    cache = STATE.setdefault("cache", {})
    if stem not in cache:
        try:
            cache[stem] = wf.predict_stem(stem)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if len(cache) > 4:                      # garde les 4 derniers instants
            cache.pop(next(iter(cache)))
    return cache[stem]


class ReconstructRequest(BaseModel):
    stem: str | None = Field(None, description='instant en cache, ex. "004_120000"')
    keep: float = Field(1.0, ge=0.0, le=1.0, description="fraction de capteurs utilises")
    export: bool = Field(False, description="ecrire aussi les binaires du visualiseur")
    stride: int = Field(1, ge=1, le=8, description="decimation horizontale de l'export")


@app.get("/health")
def health() -> dict:
    wf = STATE.get("model")
    return {"status": "ok" if wf else "degraded",
            "modele": wf.checkpoint_path.name if wf else None,
            "device": str(wf.device) if wf else None}


@app.get("/meta")
def meta() -> dict:
    return model().meta()


@app.get("/instants")
def instants(split: str | None = Query(None, pattern="^(train|val|test)$"),
             limit: int = Query(100, ge=1, le=5000)) -> dict:
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


@app.post("/reconstruct")
def reconstruct(req: ReconstructRequest) -> dict:
    wf = model()
    try:
        volume = wf.predict_stem(req.stem, keep=req.keep) if req.stem \
            else _volume_for(None)
    except KeyError as exc:
        raise HTTPException(404, f"instant inconnu : {req.stem}") from exc
    out: dict = {
        "timestamp_utc": volume.timestamp,
        "capteurs_utilises": volume.n_sensors,
        "inference_s": round(volume.inference_s, 4),
        "dso4_conforme": volume.inference_s < wf.cfg.dso.time_s,
        "grille": {"nz": volume.u.shape[0], "ny": volume.u.shape[1],
                   "nx": volume.u.shape[2]},
        "vitesse_max_m_s": float(np.nanmax(volume.speed)),
        "vitesse_moyenne_m_s": float(np.nanmean(volume.speed)),
    }
    if req.export:
        name = f"recon_{(req.stem or 'latest')}"
        out["export"] = str(export_web(volume, wf.cfg.paths.run_dir / "web" / name,
                                       stride=req.stride))
    return out


@app.post("/reconstruct/upload")
async def reconstruct_upload(file: UploadFile = File(...), keep: float = 1.0) -> dict:
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
    return {
        "timestamp_utc": volume.timestamp,
        "capteurs_utilises": volume.n_sensors,
        "inference_s": round(volume.inference_s, 4),
        "vitesse_max_m_s": float(np.nanmax(volume.speed)),
    }


@app.get("/field")
def field(z: float = Query(..., description="altitude ASL en m"),
          stem: str | None = None,
          variable: str = Query("speed", pattern="^(speed|u|v|w|direction)$"),
          stride: int = Query(4, ge=1, le=16)) -> dict:
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
    return {
        "timestamp_utc": volume.timestamp, "variable": variable, "z_asl_m": z,
        "stride": stride, "dx_m": g.dx * stride, "dy_m": g.dy * stride,
        "ny": plane.shape[0], "nx": plane.shape[1],
        # None la ou le point est sous le relief : JSON n'admet pas NaN
        "valeurs": [[None if not np.isfinite(v) else round(float(v), 3) for v in row]
                    for row in plane],
    }


@app.get("/point")
def point(x: float = Query(..., description="x en m depuis le coin sud-ouest"),
          y: float = Query(..., description="y en m"),
          z: float = Query(..., description="altitude ASL en m"),
          stem: str | None = None) -> dict:
    volume = _volume_for(stem)
    try:
        return volume.at_point(x, y, z)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
