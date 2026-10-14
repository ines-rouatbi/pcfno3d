"""Adaptateur explicite : sortie actuelle du service -> contrat v1 du tableau de bord.

Le contrat (`atom-dashboard/contract/`, recopie dans `tests/contract/`) fixe le format que le
Worker sert depuis R2. Ce module convertit ce que le service sait produire AUJOURD'HUI, sans
rien inventer :

    meta         to_meta(geom, cfg)            georef obligatoire, refuse de le deviner
    terrain      terrain_bytes(geom)           Float32 LE (ny, nx), m ASL, sans NaN
    champs       field_bytes(volume, k, var)   Float32 LE (ny, nx), NaN sous le relief
    profil       to_profile(volume, x, y)      champs de fiabilite a null (lot M2)
    resume       to_snapshot_partial(...)      sans les cles des lots M2 / M3 / M4 / M5
    manifeste    to_publish_manifest(...)      sha256 + taille (la publication R2 est M6)

Ce qui manque est liste dans `SNAPSHOT_PENDING` avec le lot qui le livrera ; les tests du
contrat s'en servent pour se resserrer lot apres lot. Les fonctions sont pures (sauf
`file_sha256`, qui lit un fichier) : elles ne chargent pas de modele et n'ecrivent rien.

Conventions (identiques a celles du service, cf. CLAUDE.md) :
  - index d'un binaire de niveau = j*nx + i, j = 0 au SUD, i = 0 a l'OUEST ;
  - noeud (i, j) en x = i*dx, y = j*dy depuis le coin sud-ouest (pas au centre de maille) ;
  - altitudes en m au-dessus de la mer ; hauteur sol = altitude - relief ;
  - direction meteo (d'ou vient le vent) = (270 - atan2(v, u)) mod 360, null sous 0,2 m/s ;
  - sous le relief (= noeud absent du generateur, a moins de `min_z_rel_m` du sol) : NaN dans
    les binaires, null en JSON. Jamais -9999 (c'est le format de `export_web`, pas celui-ci).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_VERSION = "1.0"
ORDER = "z,y,x ; j=0 au sud, i=0 à l ouest"
CONVENTIONS = {
    "u": "positif vers l Est",
    "v": "positif vers le Nord",
    "w": "positif vers le haut",
    "direction": "meteo : d ou vient le vent, (270 - atan2(v,u)) mod 360",
    "nodata": "null en JSON, NaN dans les binaires",
}
# variables que le service produit aujourd'hui ; sigma_*, coverage, reliability, status = M2
FIELDS_TODAY = ("u", "v", "w", "speed")
UNITS = {"u": "m/s", "v": "m/s", "w": "m/s", "speed": "m/s", "altitude": "m au-dessus de la mer"}
CALM_MS = 0.2   # sous ce seuil, la direction est indefinie (null)
# une seule passe deterministe : pas d'incertitude estimee tant que M2 n'est pas livre
NO_UNCERTAINTY = {"method": "aucune (passe unique, fiabilite = lot M2)", "members": 1,
                  "calibrated": False}

# Cles REQUISES par snapshot.schema.json que le service ne sait pas encore produire, et le lot
# qui les livrera. Un lot qui en ajoute une doit la retirer d'ici (le test le verifie).
SNAPSHOT_PENDING = {
    "kpi.stations_active": "M3 (definition d'une station, stations muettes)",
    "kpi.stations_total": "M3 (definition d'une station, stations muettes)",
    "kpi.share_high_reliability": "M2 (classes de fiabilite)",
    "kpi.mass_residual_rms": "M4 (residu de masse en production)",
    "kpi.open_alerts": "M4 (alertes)",
    "kpi.hub_wind_mean_ms": "M5 (vent au moyeu)",
    "kpi.hub_wind_max_ms": "M5 (vent au moyeu)",
    "kpi.hub_height_agl_m": "M5 (vent au moyeu)",
    "physics": "M4 (residu de masse, condition au sol en production)",
    "stations_summary": "M3 + M4 (etats muette / derive)",
}


# ---------------------------------------------------------------------------- identites
def file_sha256(path: str | Path, chunk: int = 8 << 20) -> str:
    """sha256 d'un fichier (les poids servis : 1,8 Go pour v7, lus par blocs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def snapshot_id(stem: str, model_sha256: str) -> str:
    """`<stem>_<sha8 du modele>` ; le stem est ramene a [0-9A-Za-z_]."""
    if not re.fullmatch(r"[0-9a-f]{64}", model_sha256):
        raise ValueError("model_sha256 : 64 caracteres hexadecimaux minuscules attendus")
    clean = re.sub(r"[^0-9A-Za-z_]", "_", str(stem))
    return f"{clean}_{model_sha256[:8]}"


def iso_utc(stamp) -> str:
    """Horodatage du service -> ISO 8601 UTC suffixe Z, a la seconde.

    Les horodatages du cache et du generateur sont `timestamp_utc` SANS fuseau : ils sont
    deja en UTC, on ne les decale pas.
    """
    ts = pd.Timestamp(stamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_versions(model: str, norm: str, data: str, model_sha256: str | None = None) -> dict:
    out = {"model": str(model), "norm": str(norm), "data": str(data),
           "contract": CONTRACT_VERSION}
    if model_sha256 is not None:
        out["model_sha256"] = model_sha256
    return out


# ---------------------------------------------------------------------------- calculs
def direction_from(u: float, v: float) -> float | None:
    """Direction meteo arrondie a 0,01 deg, dans [0, 360) ; null si calme ou non fini."""
    if not (np.isfinite(u) and np.isfinite(v)) or np.hypot(u, v) < CALM_MS:
        return None
    d = round(float((270.0 - np.degrees(np.arctan2(v, u))) % 360.0), 2)
    return 0.0 if d >= 360.0 else d


def _num(x: float, nd: int) -> float | None:
    return round(float(x), nd) if np.isfinite(x) else None


# ---------------------------------------------------------------------------- meta
def to_meta(geom, cfg, terrain_href: str = "/v1/terrain") -> dict:
    """GET /v1/meta. Refuse un domaine sans ancrage : le contrat rend `georef` obligatoire."""
    real = cfg.real
    if real.origin_lat is None or real.origin_lon is None:
        raise ValueError("georef obligatoire dans le contrat v1 : renseigne real.origin_lat "
                         "et real.origin_lon (coin sud-ouest) ; le service ne le devine pas")
    zt = np.asarray(geom.zt, dtype=np.float32)
    producer = geom.meta.raw.get("producteur", "generateur inconnu")
    return {
        "contract_version": CONTRACT_VERSION,
        "grid": {
            "nx": int(geom.nx), "ny": int(geom.ny), "nz": int(geom.nz),
            "dx_m": float(geom.dx), "dy_m": float(geom.dy),
            "levels_asl_m": [float(z) for z in geom.z_levels],
            "order": ORDER,
            "valid_nodes": int(geom.valid.sum()),
            "valid_fraction_by_level": [round(float(c), 4) for c in geom.cover],
        },
        "domain": {
            # noeud (i, j) en (i*dx, j*dy) : le dernier noeud est en (nx-1)*dx
            "x_max_m": float((geom.nx - 1) * geom.dx),
            "y_max_m": float((geom.ny - 1) * geom.dy),
            "origin": "coin_sud_ouest",
            "crs": "local-metrique",
        },
        "georef": {"lat0": float(real.origin_lat), "lon0": float(real.origin_lon),
                   "rotation_deg": float(real.rotation_deg),
                   "suppose": bool(real.origin_assumed)},
        "terrain": {"min_m": round(float(zt.min()), 2), "max_m": round(float(zt.max()), 2),
                    "href": terrain_href,
                    "source": f"{producer} (donnees synthetiques)"},
        "fields": list(FIELDS_TODAY),
        "units": dict(UNITS),
        "conventions": dict(CONVENTIONS),
    }


# ---------------------------------------------------------------------------- binaires
def terrain_bytes(geom) -> bytes:
    """Relief au noeud (non lisse), Float32 little-endian, ny*nx, index j*nx + i."""
    zt = np.ascontiguousarray(geom.zt, dtype="<f4")
    if not np.isfinite(zt).all():
        raise ValueError("le relief contient des valeurs non finies")
    return zt.tobytes(order="C")


def _field(volume, var: str) -> np.ndarray:
    data = {"u": volume.u, "v": volume.v, "w": volume.w, "speed": volume.speed}
    if var not in data:
        raise KeyError(f"variable {var!r} non produite aujourd'hui (disponibles : "
                       f"{', '.join(FIELDS_TODAY)})")
    return data[var]


def field_bytes(volume, level: int, var: str) -> bytes:
    """Un niveau d'une variable : Float32 LE (ny, nx), NaN la ou le noeud n'existe pas."""
    valid = volume.geom.valid[level]
    a = np.where(valid, _field(volume, var)[level], np.nan)
    return np.ascontiguousarray(a, dtype="<f4").tobytes(order="C")


def fields_files(volume, fields: tuple[str, ...] = FIELDS_TODAY) -> dict[str, bytes]:
    """Tous les binaires d'un instant, sous leur chemin de publication `fields/{k}_{var}.f32`."""
    return {f"fields/{k}_{var}.f32": field_bytes(volume, k, var)
            for k in range(volume.geom.nz) for var in fields}


# ---------------------------------------------------------------------------- profil
def to_profile(volume, x_m: float, y_m: float) -> dict:
    """GET /v1/profiles/{id}?x_m&y_m au noeud le plus proche (meme regle que /api/profile).

    `x_m`, `y_m` renvoyes sont ceux du NOEUD utilise, pas ceux demandes. Statut, couverture,
    fiabilite et intervalle restent a null jusqu'au lot M2 (avec `below_terrain = false`,
    null veut dire « pas encore calcule », pas « sous le relief »).
    """
    g = volume.geom
    i = int(np.clip(round(x_m / g.dx), 0, g.nx - 1))
    j = int(np.clip(round(y_m / g.dy), 0, g.ny - 1))
    z_terrain = float(g.zt[j, i])
    levels = []
    for k, z in enumerate(g.z_levels):
        ok = bool(g.valid[k, j, i])
        u, v, w = (float(a[k, j, i]) if ok else float("nan")
                   for a in (volume.u, volume.v, volume.w))
        speed = float(np.hypot(u, v))
        levels.append({
            "z_asl_m": float(z),
            "agl_m": round(float(z) - z_terrain, 2),
            "below_terrain": not ok,
            "u": _num(u, 3), "v": _num(v, 3), "w": _num(w, 4),
            "speed": _num(speed, 3),
            "dir_from_deg": direction_from(u, v),
            "speed_interval": {"p10": None, "p50": None, "p90": None},   # M2
            "status": None, "coverage": None, "reliability": None,       # M2
            "reliability_class": None,                                    # M2
        })
    return {
        "x_m": float(i * g.dx), "y_m": float(j * g.dy),
        "z_terrain_m": round(z_terrain, 2),
        "nearest_station": None,   # M3 : pas de liste de stations dans le jeu synthetique
        "levels": levels,
        "uncertainty": dict(NO_UNCERTAINTY),
    }


# ---------------------------------------------------------------------------- resume
def to_snapshot_partial(volume, stem: str, versions: dict) -> dict:
    """GET /v1/snapshots/{id} SANS les cles de `SNAPSHOT_PENDING` (non conforme tel quel)."""
    sid = snapshot_id(stem, versions["model_sha256"])
    return {
        "snapshot_id": sid,
        "valid_time": iso_utc(volume.timestamp),
        "versions": dict(versions),
        "kpi": {"reconstruction_ms": round(float(volume.inference_s) * 1000.0, 1)},
        "scenario": None,
        "assets": {"fields_template": f"/v1/fields/{sid}/{{level}}/{{var}}",
                   "profile": f"/v1/profiles/{sid}"},
    }


# ---------------------------------------------------------------------------- publication
def to_publish_manifest(sid: str, valid_time: str, versions: dict, files: dict[str, bytes],
                        generated_at: str | None = None,
                        last_observation_at: str | None = None,
                        data_status: str | None = None) -> dict:
    """Manifeste du POST /v1/publish/{id}/commit : sha256 et taille de chaque fichier.

    Seulement le manifeste : l'envoi atomique vers R2 est le lot M6.
    """
    out = {
        "snapshot_id": sid,
        "valid_time": valid_time,
        "generated_at": generated_at or now_utc(),
        "last_observation_at": last_observation_at,
        "versions": dict(versions),
        "files": [{"path": p, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
                  for p, b in files.items()],
    }
    if data_status is not None:
        out["data_status"] = data_status
    return out
