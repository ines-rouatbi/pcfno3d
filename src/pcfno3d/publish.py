"""Lot M6 — publication des instantanes au format du contrat ATOM v1 revision 2 (HIS1, DE3).

Trois etapes, chacune utilisable seule :

    construire_instantane(...)  Volume (+ fiabilite M2, alertes M4, parc M5 quand ils sont
                                actifs) -> Instantane : tous les fichiers en memoire, avec le
                                manifeste d'instantane et le manifeste du modele (sha256).
    DepotLocal.ecrire(inst)     dossier local de disposition IDENTIQUE aux fixtures du contrat
                                (contract/fixtures/<cas>/), ecrit de facon atomique ; tient
                                l'historique (index/times.json) et latest.json.
    ClientPublication.publier() protocole de contract/tools/publish.mjs, vers le Worker :
                                fichiers du modele (une fois), commit du modele, fichiers de
                                l'instantane, commit de l'instantane, battement de coeur.

Atomicite (HIS1) : le Worker ne bascule `latest` qu'apres un commit dont il a verifie chaque
fichier (presence, taille, sha256). Le client ne commite qu'apres avoir depose TOUS les
fichiers ; une coupure au milieu laisse `latest` sur l'instantane precedent, et relancer la
meme publication la termine (PUT idempotents, commit idempotent). Le depot local suit la
meme regle : dossier ecrit sous un nom temporaire puis renomme, latest.json remplace en
dernier par os.replace.

Decisions du contrat appliquees (contract/DECISIONS-rev2.md) : B1 (min_agl_m, NaN <=>
z_asl - relief < min_agl_m, verifie a la publication), B2 (x0 = y0 = 0, noeud en i.dx),
B4 (network, stations.json = capteurs signales c<i>_<j>), B5 (null, jamais 0 pour
l'inconnu ; uncertainty « aucune » sans M2), D1 (model_sha256 = sha256 du fichier de poids,
norm = sha8 de norm_stats.json, snapshot_id = <stem>[_<scenario>]_<sha8>), D3 (fichiers par
modele), D4 (georef plan tangent), D11 (physics), D12 (data-quality), D13 (un fichier par
variable), D15 (data_kind), D16 (data_status). Ecarts restants : docs/PUBLICATION.md.

Le jeton de publication ne passe JAMAIS en argument : il est lu dans l'environnement
(ATOM_PUBLISH_TOKEN), n'est ni imprime, ni journalise, ni ecrit dans un fichier.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import math
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .alerts import compter_ouvertes, iso_utc, resume_stations
from .reliability import (CLASSES, NODATA_U8, STATUT_MESURE, U8_SCALE, UNCERTAINTY_AUCUNE,
                          Z80, calibration_block, quality_by_level, to_u8_unit)
from .windfarm import round_direction

CONTRACT_VERSION = "1.0"
MAX_FILE_BYTES = 8 * 1024 * 1024          # limite du Worker par fichier (413 au-dela)
TOKEN_ENV = "ATOM_PUBLISH_TOKEN"          # meme variable que contract/tools/publish.mjs
DATA_KINDS = ("synthetique", "reel", "test")
# D4 : plan tangent equirectangulaire au noeud (0, 0)
M_PER_DEG_LAT = 110574
M_PER_DEG_LON_EQUATEUR = 111320.0
MIN_AGL_DEFAUT = 5.0                      # B1 : 5 m pour v7 (meta.json du generateur)
CALME_MS = 0.2                            # direction null en dessous (dir_from_deg)
# variables du contrat (meta.fields), dans l'ordre des fixtures
CHAMPS_VENT = ("u", "v", "w", "speed")
CHAMPS_M2 = ("sigma_speed", "sigma_u", "sigma_v", "sigma_w", "coverage", "reliability",
             "status")
_ID_RE = re.compile(r"^[0-9A-Za-z_]+_[0-9a-f]{8}$")
_SHA8_RE = re.compile(r"^[0-9a-f]{8}$")
_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)*$")
_CAPTEUR_RE = re.compile(r"^c(\d+)_(\d+)$")
# fichiers du dossier local jamais publies dans l'instantane (comme validate.mjs)
NON_PUBLIES = ("latest.json", "times.json", "publish-manifest.json")

_SHA_CACHE: dict[tuple[str, int, int], str] = {}


# ---------------------------------------------------------------------- briques
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_fichier(path: str | Path) -> str:
    """sha256 d'un fichier lu par blocs (1,81 Go pour v7), memorise par (chemin, taille,
    date) : une seule lecture par processus et par version du fichier."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for bloc in iter(lambda: f.read(1 << 22), b""):
                h.update(bloc)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def json_bytes(doc) -> bytes:
    """JSON du contrat : UTF-8, LF, indentation 2, et JAMAIS de NaN (allow_nan=False)."""
    return (json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8")


def maintenant_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stem_direct(instant) -> str:
    """Stem du direct (D1) : live_AAAAMMJJ_HHMMSS (UTC)."""
    t = iso_utc(instant)
    return "live_" + t[:10].replace("-", "") + "_" + t[11:19].replace(":", "")


def snapshot_id(stem: str, sha8: str, scenario: str | None = None) -> str:
    """<stem>[_<scenario>]_<sha8> (D1) ; leve si l'identifiant sort du motif du contrat."""
    sid = f"{stem}_{scenario}_{sha8}" if scenario else f"{stem}_{sha8}"
    if not _ID_RE.match(sid) or not _SHA8_RE.match(sha8):
        raise ValueError(f"snapshot_id invalide : {sid!r} (motif <stem>[_<scenario>]_<sha8>)")
    return sid


def _r(x, nd: int = 3) -> float | None:
    """Arrondi JSON ; None pour un inconnu (B5 : jamais NaN, jamais 0 a la place)."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _nombre_entier(x: float):
    """25.0 -> 25 (les niveaux et le pas s'ecrivent comme dans les fixtures)."""
    x = float(x)
    return int(x) if x.is_integer() else x


# ---------------------------------------------------------------------- versions (D1, D15)
def versions_modele(wf, data_kind: str, data: str | None = None,
                    model: str | None = None) -> dict:
    """Bloc `versions` : poids servis (sha256 des octets), norm_stats.json (sha8), jeu de
    donnees, nature des donnees et version du contrat."""
    if data_kind not in DATA_KINDS:
        raise ValueError(f"data_kind doit valoir {' | '.join(DATA_KINDS)}, recu {data_kind!r}")
    cfg = wf.cfg
    return {"model": model or str(cfg.name),
            "model_sha256": sha256_fichier(wf.checkpoint_path),
            "norm": sha256_fichier(cfg.paths.norm_stats)[:8],
            "data": data or Path(cfg.paths.data_dir).name,
            "data_kind": data_kind,
            "contract": CONTRACT_VERSION}


# ---------------------------------------------------------------------- georef (D4)
def georef_contrat(cfg, geom) -> dict:
    """georef de meta.json (D4) : lat0/lon0 = noeud (0, 0), plan tangent equirectangulaire
    (110574 m par degre de latitude, 111320 cos(lat0) par degre de longitude), emprise des
    4 coins. L'ancrage vient UNIQUEMENT de la configuration (`real.origin_lat/lon`,
    `origin_assumed` -> `suppose`) : M6 n'invente pas de position et refuse de publier sans
    ancrage. (realdata.GeoRef utilise une sphere, 111195 m par degre : ecart documente dans
    docs/PUBLICATION.md, realdata n'est pas modifie.)"""
    real = cfg.real
    if real.origin_lat is None or real.origin_lon is None:
        raise ValueError(
            "publication refusee : domaine sans ancrage geographique. Le contrat exige "
            "meta.georef (lat0/lon0 du noeud (0, 0)) : renseigner real.origin_lat, "
            "real.origin_lon et real.origin_assumed (true si la position est choisie et non "
            "mesuree) dans la configuration, comme configs/colab.yaml, colab_v8.yaml et "
            "local_v7.yaml (ancrage suppose 36,0 N 9,7 E).")
    lat0, lon0 = float(real.origin_lat), float(real.origin_lon)
    suppose = bool(real.origin_assumed)
    rot = float(real.rotation_deg)
    m_lon = round(M_PER_DEG_LON_EQUATEUR * math.cos(math.radians(lat0)), 3)
    g = {"lat0": lat0, "lon0": lon0, "rotation_deg": rot,
         "projection": "equirectangulaire_tangent_noeud_0_0",
         "m_per_deg_lat": M_PER_DEG_LAT, "m_per_deg_lon": m_lon}
    xm, ym = (geom.nx - 1) * geom.dx, (geom.ny - 1) * geom.dy
    lat, lon = latlon(g, np.array([0.0, xm, 0.0, xm]), np.array([0.0, 0.0, ym, ym]))
    g["bounds"] = {"lat_min": round(float(lat.min()), 6), "lat_max": round(float(lat.max()), 6),
                   "lon_min": round(float(lon.min()), 6), "lon_max": round(float(lon.max()), 6)}
    g["suppose"] = suppose
    return g


def latlon(georef: dict, x, y) -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) d'un point du domaine selon D4 : Est = x cos t - y sin t,
    Nord = x sin t + y cos t ; lat = lat0 + Nord / m_lat ; lon = lon0 + Est / m_lon."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    a = math.radians(georef["rotation_deg"])
    est = x * math.cos(a) - y * math.sin(a)
    nord = x * math.sin(a) + y * math.cos(a)
    return (georef["lat0"] + nord / georef["m_per_deg_lat"],
            georef["lon0"] + est / georef["m_per_deg_lon"])


# ---------------------------------------------------------------------- champs publies
def min_agl_m(geom) -> float:
    """B1 : hauteur minimale d'un noeud au-dessus du sol, lue dans meta.json du generateur
    (`noeuds_3d.min_z_rel_m`), 5 m sinon."""
    v = (geom.meta.raw.get("noeuds_3d") or {}).get("min_z_rel_m")
    return float(v) if v is not None else MIN_AGL_DEFAUT


@dataclass
class ChampsPublies:
    """Le volume tel qu'il est publie : masque du contrat applique (B1), float32 et uint8
    du contrat. Sert aussi aux exports NetCDF / CSV (DE3) : memes valeurs partout."""

    terrain: np.ndarray                 # (ny, nx) float32, relief publie (sans NaN)
    valide: np.ndarray                  # (nz, ny, nx) bool : z - relief >= min_agl_m
    min_agl_m: float
    f32: dict                           # u, v, w, speed (+ sigma_*) : float32, NaN absent
    u8: dict                            # coverage, reliability, status : uint8, 255 absent
    coverage: np.ndarray | None = None  # float32 (exports), NaN absent
    reliability: np.ndarray | None = None
    reliability_class: np.ndarray | None = None   # uint8 0 faible 1 moyenne 2 elevee, 255
    retires: int = 0                    # noeuds valides du generateur sous min_agl_m


def champs_publies(volume, geom) -> ChampsPublies:
    """Applique la regle B1 sur le relief PUBLIE (float32) : NaN / 255 <=> z_asl - relief
    < min_agl_m, exactement comme validate.mjs la recalcule. Un noeud valide pour le
    contrat mais sans valeur du modele est une erreur (on n'invente pas de valeur) ; un
    noeud du generateur sous min_agl_m (ecart d'arrondi) devient absent et est compte."""
    terrain = np.asarray(geom.zt, np.float32)
    if not np.isfinite(terrain).all():
        raise ValueError("relief avec des NaN : le contrat l'interdit (terrain.f32)")
    hmin = min_agl_m(geom)
    z = np.asarray(geom.z_levels, np.float64)[:, None, None]
    valide = (z - terrain.astype(np.float64)[None]) >= hmin
    sans_valeur = valide & ~(geom.valid & np.isfinite(volume.u) & np.isfinite(volume.v)
                             & np.isfinite(volume.w))
    if sans_valeur.any():
        raise ValueError(f"{int(sans_valeur.sum())} noeud(s) au-dessus de min_agl_m = {hmin} m "
                         "sans valeur du modele : la regle B1 (NaN <=> z - relief < "
                         "min_agl_m) ne peut pas etre tenue avec ce relief")
    retires = int((geom.valid & ~valide).sum())

    def f(a) -> np.ndarray:
        return np.where(valide, np.asarray(a, np.float32), np.float32(np.nan)).astype(
            np.float32)

    u, v, w = f(volume.u), f(volume.v), f(volume.w)
    out = ChampsPublies(terrain, valide, hmin, {"u": u, "v": v, "w": w,
                                                 "speed": f(np.hypot(u, v))}, {},
                        retires=retires)
    rel = volume.reliability
    if rel is not None:
        for nom, a in (("sigma_speed", rel.sigma_speed), ("sigma_u", rel.sigma_u),
                       ("sigma_v", rel.sigma_v), ("sigma_w", rel.sigma_w)):
            out.f32[nom] = f(a)
        out.coverage, out.reliability = f(rel.coverage), f(rel.reliability)
        out.u8["coverage"] = to_u8_unit(out.coverage)
        out.u8["reliability"] = to_u8_unit(out.reliability)
        st = np.asarray(rel.status, np.uint8).copy()
        st[~valide] = NODATA_U8
        out.u8["status"] = st
        cl = np.asarray(rel.reliability_class, np.uint8).copy()
        cl[~valide] = NODATA_U8
        out.reliability_class = cl
        if (out.u8["status"][valide] > 1).any():
            raise ValueError("status : valeur hors {0, 1} sur un noeud valide")
    _verifier_bornes(out)
    return out


def _verifier_bornes(ch: ChampsPublies) -> None:
    """Plages de common.schema.json (component_ms, speed_ms, sigma_ms) : le publieur
    verifie, il ne tronque jamais une valeur."""
    for nom, lim in (("u", 80.0), ("v", 80.0), ("w", 80.0), ("speed", 80.0),
                     ("sigma_speed", 40.0), ("sigma_u", 40.0), ("sigma_v", 40.0),
                     ("sigma_w", 40.0)):
        a = ch.f32.get(nom)
        if a is None:
            continue
        m = np.nanmax(np.abs(a)) if np.isfinite(a).any() else 0.0
        if m > lim:
            raise ValueError(f"{nom} : |valeur| max {m:.2f} > {lim} (plage du contrat)")


# ---------------------------------------------------------------------- l'instantane
@dataclass
class Instantane:
    """Un instantane complet, en memoire. `fichiers` : chemins relatifs a snapshots/{id}/ ;
    `modele` : chemins relatifs a models/{sha8}/."""

    snapshot_id: str
    sha8: str
    valid_time: str
    generated_at: str
    last_observation_at: str | None
    data_status: str
    versions: dict
    fichiers: dict[str, bytes]
    modele: dict[str, bytes]
    manifeste: dict
    manifeste_modele: dict
    open_alerts: int | None
    champs: ChampsPublies | None = None
    info: dict = field(default_factory=dict)
    vent_points: dict = field(default_factory=dict)     # {id: (vitesse, direction, rho)}


def _entree(path: str, data: bytes) -> dict:
    if not _PATH_RE.match(path):
        raise ValueError(f"chemin publie invalide : {path!r}")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{path} : {len(data)} octets > {MAX_FILE_BYTES} (limite du Worker)")
    if not data:
        raise ValueError(f"{path} : fichier vide (le manifeste exige bytes >= 1)")
    return {"path": path, "sha256": sha256_bytes(data), "bytes": len(data)}


def meta_contrat(geom, ch: ChampsPublies, versions: dict, georef: dict, sha8: str,
                 terrain_sha: str, terrain_bytes: int, source: str) -> dict:
    """models/{sha8}/meta.json (meta.schema.json) : grille, domaine, georef, relief,
    codage de chaque variable publiee (D13), unites et conventions."""
    nz, ny, nx = geom.shape
    champs = list(CHAMPS_VENT) + [c for c in CHAMPS_M2 if c in ch.f32 or c in ch.u8]
    encoding = {}
    for c in champs:
        if c in ch.u8:
            encoding[c] = {"dtype": "uint8", "file": f"fields/{c}.u8", "nodata": NODATA_U8,
                           "scale": 1 if c == "status" else U8_SCALE}
        else:
            encoding[c] = {"dtype": "float32", "file": f"fields/{c}.f32", "nodata": "NaN",
                           "scale": None}
    units = {c: "m/s" for c in champs if c not in ("coverage", "reliability", "status")}
    units.update({c: "0-1" for c in ("coverage", "reliability") if c in champs})
    if "status" in champs:
        units["status"] = "1 mesur\u00e9 / 0 estim\u00e9"
    units["altitude"] = "m au-dessus de la mer"
    frac = ch.valide.reshape(nz, -1).mean(axis=1)
    return {
        "versions": versions,
        "grid": {"nx": nx, "ny": ny, "nz": nz, "dx_m": _nombre_entier(geom.dx),
                 "dy_m": _nombre_entier(geom.dy), "x0_m": 0, "y0_m": 0,
                 "min_agl_m": _nombre_entier(ch.min_agl_m),
                 "levels_asl_m": [_nombre_entier(z) for z in geom.z_levels],
                 "order": "zyx_sud_ouest", "valid_nodes": int(ch.valide.sum()),
                 "valid_fraction_by_level": [round(float(x), 4) for x in frac]},
        "domain": {"x_max_m": _nombre_entier((nx - 1) * geom.dx),
                   "y_max_m": _nombre_entier((ny - 1) * geom.dy),
                   "origin": "noeud_0_0_sud_ouest", "crs": "plan-tangent-local"},
        "georef": georef,
        "terrain": {"min_m": round(float(ch.terrain.min()), 2),
                    "max_m": round(float(ch.terrain.max()), 2),
                    "href": f"/v1/terrain/{sha8}", "sha256": terrain_sha,
                    "bytes": terrain_bytes, "source": source},
        "fields": champs,
        "encoding": encoding,
        "units": units,
        "conventions": {"u": "positif_est", "v": "positif_nord", "w": "positif_haut",
                        "direction": "meteo_provenance", "altitude": "asl",
                        "nodata": "null_json_nan_f32_255_u8"},
    }


def _direction(u: float, v: float) -> float | None:
    """Provenance meteo arrondie AVANT le modulo (m4) ; null si vitesse < 0,2 m/s."""
    if not (math.isfinite(u) and math.isfinite(v)) or math.hypot(u, v) < CALME_MS:
        return None
    return float(round_direction(270.0 - math.degrees(math.atan2(v, u)), 2))


def _capteur_proche(sensor: np.ndarray, geom, i: int, j: int) -> dict | None:
    jj, ii = np.nonzero(sensor)
    if ii.size == 0:
        return None
    d = np.hypot((ii - i) * geom.dx, (jj - j) * geom.dy)
    k = int(np.argmin(d))
    return {"id": f"c{int(ii[k]) * geom.stride}_{int(jj[k]) * geom.stride}",
            "distance_m": round(float(d[k]), 1)}


def profil_point(point_id: str, i: int, j: int, geom, ch: ChampsPublies, sensor: np.ndarray,
                 uncertainty: dict) -> dict:
    """profiles/{point_id}.json (profile.schema.json, D5) : la colonne (i, j) du modele,
    valeurs identiques aux champs publies (arrondies), intervalle de vitesse D6."""
    zt = float(ch.terrain[j, i])
    niveaux = []
    for k, z in enumerate(geom.z_levels):
        below = not bool(ch.valide[k, j, i])
        lv = {"z_asl_m": _nombre_entier(z), "z_agl_m": round(float(z) - zt, 3),
              "below_terrain": below}
        if below:
            lv.update({c: None for c in ("u", "v", "w", "speed", "sigma_speed",
                                         "dir_from_deg")})
            lv.update({"speed_interval": {"p10": None, "p50": None, "p90": None},
                       "status": None, "coverage": None, "reliability": None,
                       "reliability_class": None})
        else:
            u, v = _r(ch.f32["u"][k, j, i]), _r(ch.f32["v"][k, j, i])
            speed = _r(ch.f32["speed"][k, j, i])
            sig = _r(ch.f32["sigma_speed"][k, j, i])
            st = int(ch.u8["status"][k, j, i])
            lv.update({"u": u, "v": v, "w": _r(ch.f32["w"][k, j, i], 4), "speed": speed,
                       "sigma_speed": sig, "dir_from_deg": _direction(u, v),
                       "speed_interval": {"p10": _r(max(0.0, speed - Z80 * sig)),
                                          "p50": speed, "p90": _r(speed + Z80 * sig)},
                       "status": "mesure" if st == STATUT_MESURE else "estime",
                       "coverage": _r(ch.coverage[k, j, i]),
                       "reliability": _r(ch.reliability[k, j, i]),
                       "reliability_class": CLASSES[int(ch.reliability_class[k, j, i])]})
        niveaux.append(lv)
    return {"point_id": point_id, "i": int(i), "j": int(j),
            "x_m": _nombre_entier(i * geom.dx), "y_m": _nombre_entier(j * geom.dy),
            "z_terrain_m": round(zt, 3), "nearest_sensor": _capteur_proche(sensor, geom, i, j),
            "levels": niveaux, "uncertainty": uncertainty}


def quality_doc(volume, geom, ch: ChampsPublies, dso6: dict | None) -> dict:
    """quality.json : M2 actif -> uncertainty, by_level, calibration (resume DSO6 ou null) ;
    sinon uncertainty « aucune » et des parts null (B5)."""
    nz = geom.nz
    vue = SimpleNamespace(nz=nz, valid=ch.valide, z_levels=geom.z_levels)
    rel = volume.reliability
    if rel is not None:
        by = quality_by_level(rel, vue)
        for b in by:
            for k in ("share_elevee", "share_moyenne", "share_faible", "sigma_speed_median",
                      "coverage_median"):
                b[k] = _r(b[k])
            b["z_asl_m"] = _nombre_entier(b["z_asl_m"])
        return {"uncertainty": dict(rel.uncertainty), "by_level": by,
                "calibration": calibration_block(dso6)}
    by = [{"z_asl_m": _nombre_entier(geom.z_levels[k]),
           "valid_nodes": int(ch.valide[k].sum()), "share_elevee": None,
           "share_moyenne": None, "share_faible": None, "sigma_speed_median": None,
           "coverage_median": None} for k in range(nz)]
    return {"uncertainty": dict(UNCERTAINTY_AUCUNE), "by_level": by, "calibration": None}


def _maille_capteur(sid: str, geom) -> tuple[int, int] | None:
    m = _CAPTEUR_RE.match(sid)
    if not m:
        return None
    i = min(int(m.group(1)) // geom.stride, geom.nx - 1)
    j = min(int(m.group(2)) // geom.stride, geom.ny - 1)
    return i, j


def stations_doc(items: list[dict], geom, S: np.ndarray, valid_time: str) -> list[dict]:
    """stations.json en treillis (B4) : seulement les capteurs SIGNALES par une alerte
    ouverte (derive, muette, exclue), identifiant c<i>_<j>. Meme priorite d'etat que
    alerts.resume_stations (muette / exclue l'emportent sur derive)."""
    etat: dict[str, tuple[str, dict]] = {}
    for a in items:
        if a["status"] != "ouverte" or a["station_id"] is None:
            continue
        e = {"capteur_derive": "derive", "station_muette": "muette",
             "hors_plage": "exclue"}.get(a["kind"])
        if e and (a["station_id"] not in etat or etat[a["station_id"]][0] == "derive"):
            etat[a["station_id"]] = (e, a)
    out = []
    for sid in sorted(etat):
        e, a = etat[sid]
        ij = _maille_capteur(sid, geom)
        loc = a["location"] or ({"x_m": ij[0] * geom.dx, "y_m": ij[1] * geom.dy}
                                if ij else {"x_m": 0.0, "y_m": 0.0})
        # identifiant hors treillis (non c<i>_<j>) : hauteur inconnue -> null (B5), jamais 0
        h = float(geom.zrel_sensor[ij[1], ij[0]]) if ij else None
        speed = direction = None
        if e == "derive" and ij is not None:
            u, v = float(S[0][ij[1], ij[0]]), float(S[1][ij[1], ij[0]])
            speed, direction = _r(math.hypot(u, v), 2), _direction(u, v)
        out.append({"id": sid, "x_m": loc["x_m"], "y_m": loc["y_m"],
                    "height_agl_m": _r(h, 2) if h is not None else None,
                    "state": e, "last_seen": valid_time if e == "derive" else None,
                    "speed": speed, "dir_from_deg": direction,
                    "bias_ms": _r(a["value"], 3) if e == "derive" else None,
                    "quality_score": None})
    return out


def data_quality_doc(valid_time: str, rapport: dict | None, n_total: int, n_observes: int,
                     n_interpoles: int, n_presents: int, n_muettes: int,
                     n_exclues: int, estime: bool = False) -> dict:
    """data-quality.json (D12), comptes en capteurs a 50 m. Avec un rapport du contrat M3
    (upload CSV) : periode, verdict du fichier, muettes, interpolees, hors plage. Instant du
    cache : l'instant lui-meme est le fichier recu, et les capteurs sont ESTIMES depuis la
    disponibilite par maille (`comptage_depuis_surface`), ce que dit le detail."""
    files = {"received": 1, "accepted": 1, "rejected": 0, "quarantined": 0}
    period = {"from": valid_time, "to": valid_time}
    issues = []
    if rapport is not None:
        res = rapport.get("resume") or {}
        if res.get("premier_instant"):
            period = {"from": iso_utc(res["premier_instant"]),
                      "to": iso_utc(res["dernier_instant"])}
        nom = rapport.get("fichier") or "fichier"
        if rapport.get("verdict") == "quarantaine":
            files = {"received": 1, "accepted": 0, "rejected": 0, "quarantined": 1}
            issues.append({"kind": "quarantaine", "target": nom, "count": 1,
                           "detail": "; ".join(rapport.get("motifs") or []) or "quarantaine"})
        elif rapport.get("verdict") == "rejet":
            raise ValueError("fichier rejete par le contrat ANEO : aucun instantane publie")
        q = rapport.get("qualite") or {}
        mu = ((q.get("fraicheur") or {}).get("stations_muettes") or {}).get("nombre", 0)
        if mu:
            issues.append({"kind": "capteur_muet", "target": nom, "count": int(mu),
                           "detail": "Capteurs sans mesure sur au moins un instant"})
        hp = sum(int(v.get("nombre", 0)) for v in (q.get("hors_plage") or {}).values())
        if hp:
            issues.append({"kind": "hors_plage", "target": nom, "count": hp,
                           "detail": "Valeurs hors de la plage physique (contrat 8.1-8.2)"})
    if n_interpoles:
        issues.append({"kind": "valeur_interpolee", "target": "treillis",
                       "count": int(n_interpoles),
                       "detail": "Capteurs sans mesure reelle (mask_i_obs = 0), interpoles"
                       + (" ; compte ESTIME depuis la disponibilite par maille" if estime
                          else "")})
    return {"period": period, "files": files,
            "sensors": {"total": int(n_total), "observed": int(n_observes),
                        "interpolated": int(n_interpoles), "muette": int(n_muettes),
                        "exclue": int(n_exclues)},
            "interpolated_fraction": round(n_interpoles / max(n_presents, 1), 4),
            "issues": issues}


def comptage_depuis_surface(S: np.ndarray, geom) -> dict:
    """Capteurs a 50 m d'un instant dont on n'a que la disponibilite AGREGEE par maille
    (cache : S[7] = au moins un capteur disponible dans le bloc stride x stride). ESTIMATION :
    round(fraction x stride^2) capteurs utilises par maille (1 seul capteur par maille si
    le modele n'agrege pas : les autres n'entrent pas dans l'instant)."""
    k = geom.stride ** 2 if getattr(geom, "pool_surface_enabled", True) else 1
    frac = np.clip(np.nan_to_num(np.asarray(S[7], np.float64)), 0.0, 1.0)
    utilises = int(np.rint(frac * k).sum())
    presents = int(frac.size * k)
    return {"presents": presents, "utilises": utilises, "interpoles": presents - utilises,
            "estime": True}


def windrose_points(farm, wcfg, serie: dict, actuel: dict, valid_time: str) -> dict:
    """Roses des vents des eoliennes (windrose.schema.json) sur l'historique publie
    jusqu'a l'instant courant compris — jamais un instant futur (rejeu « comme en
    direct »). serie : {id: [(instant, vitesse, direction, rho), ...]}."""
    from .windfarm import windrose_document

    out = {}
    for t in farm.turbines:
        rows = [r for r in serie.get(t.id, []) if r[0] < valid_time]
        if t.id in actuel:
            rows.append((valid_time, *actuel[t.id]))
        if not rows:
            continue
        a = np.array([[np.nan if x is None else x for x in r[1:]] for r in rows], float)
        rho = float(np.nanmean(a[:, 2])) if np.isfinite(a[:, 2]).any() else None
        doc = windrose_document(t.id, t.x_m, t.y_m, t.hub_agl_m, [r[0] for r in rows],
                                a[:, 0], a[:, 1], wcfg, farm.curves[t.model], rho)
        if doc is not None:
            out[t.id] = doc
    return out


def construire_instantane(wf, volume, S: np.ndarray, *, stem: str, data_kind: str,
                          alertes=None, rapport: dict | None = None, dso6: dict | None = None,
                          data: str | None = None, model: str | None = None,
                          serie_vent: dict | None = None, historique: list[dict] | None = None,
                          generated_at: str | None = None,
                          last_observation_at: str | None = "valid_time",
                          capteurs: dict | None = None,
                          garder_champs: bool = False) -> Instantane:
    """Construit un instantane complet (tous les fichiers du contrat) pour un Volume.

    wf : WindFieldModel (geometrie, poids, configuration, parc M5, fiabilite M2).
    S : surface (8, ny, nx) de l'instant (capteurs visibles = S[7] > 0,5).
    alertes : ResultatAlertes de M4 (None = M4 inactif : physics, open_alerts,
        stations_summary null ; alerts.json = alertes brutes du parc M5 s'il y en a).
    rapport : rapport du contrat ANEO (M3) pour data-quality ; None pour un instant du cache.
    dso6 : resume DSO6 (quality.calibration), None tant qu'aucun verdict n'existe.
    serie_vent / historique : etat du DepotLocal (roses des vents, times.json).
    capteurs : comptage des capteurs a 50 m (B4 : unite « capteur ») ; None = estimation
        depuis la disponibilite par maille (`comptage_depuis_surface`).
    """
    geom, cfg = wf.geom, wf.cfg
    if volume.timestamp is None:
        raise ValueError("volume sans horodatage : valid_time est obligatoire")
    valid_time = iso_utc(volume.timestamp)
    generated_at = generated_at or maintenant_utc()
    if last_observation_at == "valid_time":
        last_observation_at = valid_time
    versions = versions_modele(wf, data_kind, data, model)
    sha8 = versions["model_sha256"][:8]
    sid = snapshot_id(stem, sha8)
    ch = champs_publies(volume, geom)
    sensor = np.asarray(S[7], float) > 0.5

    # ---- fichiers par modele (D3) : meta.json et terrain.f32
    tb = ch.terrain.astype("<f4").tobytes()
    georef = georef_contrat(cfg, geom)
    source = str(geom.meta.raw.get("producteur", "generateur")) + " (meta.json)"
    meta = meta_contrat(geom, ch, versions, georef, sha8, sha256_bytes(tb), len(tb), source)
    modele = {"meta.json": json_bytes(meta), "terrain.f32": tb}

    # ---- champs (D13)
    fichiers: dict[str, bytes] = {}
    for c in meta["fields"]:
        enc = meta["encoding"][c]
        a = ch.u8[c] if enc["dtype"] == "uint8" else ch.f32[c].astype("<f4")
        fichiers[enc["file"]] = np.ascontiguousarray(a).tobytes()

    # ---- alertes (M4) et parc (M5)
    farm_calc = None
    if wf.farm is not None:
        from .windfarm import compute_farm
        farm_calc = compute_farm(volume, wf.farm, cfg.wind_farm, calibration=wf.reliability)
    if alertes is not None:
        items = list(alertes.items)
    elif farm_calc is not None:
        items = farm_calc.alerts(cfg.wind_farm)            # alertes brutes de M5
    else:
        items = []
    # times.open_alerts : null quand M4 est inactif (jamais 0, ni les seules alertes de M5)
    open_alerts = sum(a["status"] == "ouverte" for a in items) if alertes is not None \
        else None

    # ---- reseau (B4) et stations
    if cfg.alerts.reseau != "treillis_synthetique":
        raise ValueError("reseau stations_aneo : stations.json doit lister TOUTES les stations "
                         "(B4) ; M6 ne sait publier que le treillis synthetique pour l'instant")
    stations = stations_doc(items, geom, S, valid_time) if alertes is not None else []
    # unite « capteur » = capteur du treillis a 50 m (B4, fixtures), la meme pour le total,
    # les visibles et les perdus (muets ou exclus, listes dans stations.json)
    capteurs = dict(capteurs) if capteurs is not None else comptage_depuis_surface(S, geom)
    perdus = sum(s["state"] in ("muette", "exclue") for s in stations)
    visibles = int(capteurs["utilises"])
    network = {"kind": cfg.alerts.reseau, "sensors_total": visibles + perdus,
               "sensors_visible": visibles, "unit": "capteur"}
    if network["sensors_total"] - network["sensors_visible"] != perdus:
        raise ValueError("reseau incoherent : total - visibles != capteurs perdus")
    if not capteurs["estime"] and visibles + perdus > capteurs["presents"]:
        raise ValueError(f"reseau incoherent : {visibles} capteurs utilises + {perdus} perdus "
                         f"> {capteurs['presents']} capteurs presents dans le fichier")

    # ---- points declares, profils (M2) et roses des vents (M5)
    rel = volume.reliability
    points, profils, roses, vent = [], {}, {}, {}
    if wf.farm is not None:
        for c in farm_calc.turbines:
            if c.state != "indisponible" and math.isfinite(c.hub_speed):
                vent[c.turbine.id] = (float(c.hub_speed), c.hub_direction, c.rho)
        roses = windrose_points(wf.farm, cfg.wind_farm, serie_vent or {}, vent, valid_time)
        for t in wf.farm.turbines:
            if rel is not None:
                profils[t.id] = profil_point(t.id, t.i, t.j, geom, ch, sensor, rel.uncertainty)
            points.append({"id": t.id, "kind": "eolienne", "name": t.name,
                           "x_m": _nombre_entier(t.x_m), "y_m": _nombre_entier(t.y_m),
                           "profile": t.id in profils, "windrose": t.id in roses})

    # ---- KPI, physics (D11), resume des stations
    from .windfarm import hub_wind_kpi
    kpi = {"share_high_reliability": None,
           "reconstruction_ms": _r(volume.inference_s * 1000.0, 1),
           "open_alerts": compter_ouvertes(items) if alertes is not None else None,
           **hub_wind_kpi(volume, cfg.wind_farm.hub_height_agl_m)}
    if rel is not None and ch.valide.any():
        kpi["share_high_reliability"] = _r(
            (ch.reliability_class[ch.valide] == 2).mean(), 4)
    physics = None
    if alertes is not None and alertes.physics is not None:
        p = alertes.physics
        if all(p.get(k) is not None for k in ("mass_residual_rms", "mass_residual_p95",
                                              "ground_condition_rms_ms")):
            physics = dict(p)
    summary = resume_stations(items, network["sensors_total"]) if alertes is not None \
        else None

    # ---- data_status (D16) : degrade si des capteurs manquent ou sont en derive
    degrade_kinds = {"station_muette", "capteur_derive", "stations_insuffisantes",
                     "hors_plage"}
    degrade = perdus > 0 or any(a["status"] == "ouverte" and a["kind"] in degrade_kinds
                                for a in items)
    if alertes is None and sensor.mean() < cfg.alerts.capteurs_part_min:
        degrade = True
    data_status = "degrade" if degrade else "complete"

    base = f"/v1/{{}}/{sid}"
    snapshot = {
        "snapshot_id": sid, "valid_time": valid_time, "versions": versions,
        "network": network, "kpi": kpi, "physics": physics, "stations_summary": summary,
        "scenario": None, "points": points,
        "assets": {"meta": f"/v1/meta/{sha8}", "terrain": f"/v1/terrain/{sha8}",
                   "fields_template": f"/v1/fields/{sid}/{{level}}/{{var}}",
                   "column_template": f"/v1/column/{sid}?i={{i}}&j={{j}}",
                   "profile_template": f"/v1/profiles/{sid}/{{point_id}}" if profils else None,
                   "windrose_template": f"/v1/windrose/{{site_id}}?snapshot={sid}"
                   if roses else None,
                   "quality": base.format("quality"), "alerts": base.format("alerts"),
                   "stations": base.format("stations"), "wind_farm": base.format("wind-farm"),
                   "data_quality": base.format("data-quality"),
                   "export_netcdf": None, "export_csv": None},
    }

    # ---- documents JSON de l'instantane
    if farm_calc is not None:
        wind_farm = farm_calc.document()
    else:
        # aucun parc declare : document vide, valide (0 eolienne, puissance inconnue)
        wind_farm = {"turbines": [], "farm": {"power_kw": {"p10": None, "p50": None,
                                                           "p90": None},
                                              "rated_kw": 0, "n_production": 0, "n_arret": 0},
                     "power_curves": {}}
    dq = data_quality_doc(valid_time, rapport, network["sensors_total"], visibles,
                          int(capteurs["interpoles"]), int(capteurs["presents"]),
                          sum(s["state"] == "muette" for s in stations),
                          sum(s["state"] == "exclue" for s in stations),
                          estime=bool(capteurs["estime"]))
    fichiers["stations.json"] = json_bytes({"items": stations})
    fichiers["alerts.json"] = json_bytes({"items": items})
    fichiers["wind-farm.json"] = json_bytes(wind_farm)
    for pid, doc in profils.items():
        fichiers[f"profiles/{pid}.json"] = json_bytes(doc)
    for pid, doc in roses.items():
        fichiers[f"windrose/{pid}.json"] = json_bytes(doc)
    fichiers["quality.json"] = json_bytes(quality_doc(volume, geom, ch, dso6))
    fichiers["data-quality.json"] = json_bytes(dq)
    fichiers["snapshot.json"] = json_bytes(snapshot)

    manifeste = {"snapshot_id": sid, "valid_time": valid_time, "generated_at": generated_at,
                 "last_observation_at": last_observation_at, "data_status": data_status,
                 "versions": versions,
                 "files": [_entree(p, b) for p, b in fichiers.items()]}
    manifeste_modele = {"model_sha8": sha8, "versions": versions,
                        "files": [_entree(p, b) for p, b in modele.items()]}
    info = {"noeuds_retires_min_agl": ch.retires, "fiabilite": rel is not None,
            "alertes": alertes is not None, "parc": wf.farm is not None,
            "historique": len(historique or []), "capteurs": capteurs}
    return Instantane(sid, sha8, valid_time, generated_at, last_observation_at, data_status,
                      versions, fichiers, modele, manifeste, manifeste_modele, open_alerts,
                      ch if garder_champs else None, info, vent)


def instantane_du_cache(wf, stem: str, *, data_kind: str, depot: "DepotLocal | None" = None,
                        moteur=None, dso6: dict | None = None,
                        generated_at: str | None = None, garder_champs: bool = False
                        ) -> Instantane:
    """Un instant du cache -> Instantane : passe servie (fiabilite M2 et parc M5 s'ils
    sont configures dans `wf`), alertes M4 si `moteur` est fourni (journal chronologique),
    roses des vents et historique lus dans `depot`."""
    S, hour, doy, stamp = wf.surface_from_cache(stem)
    vol = wf.predict(S, hour, doy, timestamp=stamp)
    res = None
    if moteur is not None:
        res = moteur.evaluer_modele(wf, S, hour, doy, vol, Y=moteur.verite(wf, stem),
                                    cle=f"stem:{stem}")
    return construire_instantane(
        wf, vol, S, stem=stem, data_kind=data_kind, alertes=res, dso6=dso6,
        serie_vent=depot.serie_vent() if depot is not None else None,
        historique=depot.times() if depot is not None else None,
        generated_at=generated_at, garder_champs=garder_champs)


def instantane_du_csv(wf, path: str | Path, *, data_kind: str,
                      depot: "DepotLocal | None" = None, moteur=None,
                      dso6: dict | None = None, generated_at: str | None = None,
                      garder_champs: bool = False) -> Instantane:
    """Un fichier de mesures ANEO (premier instant) -> Instantane du DIRECT (stem
    live_AAAAMMJJ_HHMMSS), par le meme chemin que POST /api/reconstruct/upload : contrat
    M3 (Seuils.upload), stations fausses masquees, rapport repris dans data-quality.json
    (D12) et, avec `moteur`, dans les alertes M4. Un fichier rejete ne publie rien."""
    from .contract import Grille, Seuils, valider_fichier

    masque: dict = {}
    grille = Grille.depuis_meta(wf.geom.meta.raw)
    rapport = valider_fichier(path, grille, Seuils.upload(), nom=Path(path).name,
                              masque=masque)
    if rapport["verdict"] == "rejet":
        raise ValueError("CSV rejete par le contrat ANEO : " + " ; ".join(rapport["motifs"]))
    comptage: dict = {}
    S, hour, doy, stamp = wf.surface_from_csv(path, masque, comptage=comptage)
    vol = wf.predict(S, hour, doy, timestamp=stamp)
    res = None
    if moteur is not None:
        from .alerts import stations_hors_plage
        hp = stations_hors_plage(path, grille.n_stations, grille.node_id_start,
                                 rapport["seuils"]["tol_densite_rel"])
        res = moteur.evaluer_modele(wf, S, hour, doy, vol, rapport=rapport, hors_plage=hp)
    # instant reconstruit = premier instant du fichier : c'est la derniere observation
    # entree dans CET instantane (last_observation_at = valid_time)
    return construire_instantane(
        wf, vol, S, stem=stem_direct(stamp), data_kind=data_kind, alertes=res,
        rapport=rapport, dso6=dso6, capteurs=comptage,
        serie_vent=depot.serie_vent() if depot is not None else None,
        historique=depot.times() if depot is not None else None,
        generated_at=generated_at, garder_champs=garder_champs)


# ---------------------------------------------------------------------- depot local
def _ecrire_atomique(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _lire_json(path: Path, defaut):
    return json.loads(path.read_bytes().decode("utf-8")) if path.exists() else defaut


class DepotLocal:
    """Publication locale, disposition des fixtures du contrat :

        <racine>/snapshots/<id>/      un dossier par instantane (fields/, *.json, profiles/,
                                      windrose/, models/<sha8>/, publish-manifest.json,
                                      latest.json, times.json) : validate.mjs s'y applique
        <racine>/index/times.json     historique (times.schema.json), du plus recent au plus
                                      ancien — comme index/times.json du Worker
        <racine>/index/vent_points.json  series de vent au moyeu des eoliennes (roses)
        <racine>/latest.json          dernier instantane complet (bascule en DERNIER)
        <racine>/exports/<id>/        NetCDF et CSV (DE3), hors de l'instantane publie
    """

    def __init__(self, racine: str | Path) -> None:
        self.racine = Path(racine)
        self.snapshots = self.racine / "snapshots"
        self.index = self.racine / "index"
        self.repares: list[str] = []
        if self.snapshots.is_dir():
            self.reparer()

    def reparer(self) -> list[str]:
        """Complete index/times.json et latest.json depuis les dossiers d'instantanes
        presents (une coupure entre le renommage du dossier et l'index ne laisse pas de trou
        definitif) et supprime les dossiers temporaires abandonnes. Les entrees de l'index
        sans dossier (purge apres publication) sont gardees. Renvoie les ids ajoutes."""
        for tmp in self.snapshots.glob(".*.partiel*"):
            shutil.rmtree(tmp, ignore_errors=True)
        items = {t["snapshot_id"]: t for t in self.times()}
        ajoutes, plus_recent = [], None
        for d in sorted(p for p in self.snapshots.iterdir() if p.is_dir()):
            if not (d / "publish-manifest.json").exists():
                continue
            man = _lire_json(d / "publish-manifest.json", None)
            snap = _lire_json(d / "snapshot.json", {})
            sid, vt = man["snapshot_id"], man["valid_time"]
            if sid not in items:
                oa = (snap.get("kpi") or {}).get("open_alerts")
                items[sid] = {"snapshot_id": sid, "valid_time": vt,
                              "open_alerts": sum(oa.values()) if oa is not None else None}
                ajoutes.append(sid)
            if plus_recent is None or vt > plus_recent[1]:
                plus_recent = (d, vt)
        if ajoutes:
            liste = sorted(items.values(), key=lambda t: t["valid_time"], reverse=True)
            _ecrire_atomique(self.index / "times.json",
                             json_bytes({"total": len(liste), "next": None, "items": liste}))
        courant = self.latest()
        if plus_recent is not None and (courant is None
                                        or courant["valid_time"] < plus_recent[1]):
            _ecrire_atomique(self.racine / "latest.json",
                             (plus_recent[0] / "latest.json").read_bytes())
            ajoutes.append("latest.json")
        self.repares = ajoutes
        return ajoutes

    # -------------------------------------------------------------- lecture
    def times(self) -> list[dict]:
        return _lire_json(self.index / "times.json", {"items": []})["items"]

    def serie_vent(self) -> dict:
        return _lire_json(self.index / "vent_points.json", {})

    def latest(self) -> dict | None:
        return _lire_json(self.racine / "latest.json", None)

    def dossier(self, sid: str) -> Path:
        return self.snapshots / sid

    def existe(self, sid: str) -> bool:
        return (self.dossier(sid) / "publish-manifest.json").exists()

    # -------------------------------------------------------------- ecriture
    @staticmethod
    def _latest_doc(inst: Instantane, published_at: str) -> dict:
        """latest.json au format du Worker ; en local, pas de battement : compute_status
        « unknown », heartbeat_at null (jamais une valeur inventee)."""
        return {"snapshot_id": inst.snapshot_id, "valid_time": inst.valid_time,
                "generated_at": inst.generated_at, "last_observation_at": inst.last_observation_at,
                "published_at": published_at, "compute_status": "unknown", "heartbeat_at": None,
                "data_status": inst.data_status, "versions": inst.versions,
                "href": f"/v1/snapshots/{inst.snapshot_id}"}

    def ecrire(self, inst: Instantane, published_at: str | None = None) -> Path:
        """Ecrit l'instantane sous un nom temporaire, le renomme, puis met a jour l'index et
        enfin latest.json (seulement si valid_time >= celui du latest courant, comme le
        Worker). Un instantane deja ecrit n'est jamais reecrit (immuable)."""
        dest = self.dossier(inst.snapshot_id)
        if dest.exists():
            raise FileExistsError(f"{dest} existe deja : un instantane est immuable")
        published_at = published_at or inst.generated_at
        items = [t for t in self.times() if t["snapshot_id"] != inst.snapshot_id]
        items.append({"snapshot_id": inst.snapshot_id, "valid_time": inst.valid_time,
                      "open_alerts": inst.open_alerts})
        items.sort(key=lambda t: t["valid_time"], reverse=True)
        times_doc = {"total": len(items), "next": None, "items": items}
        latest_doc = self._latest_doc(inst, published_at)

        self.snapshots.mkdir(parents=True, exist_ok=True)
        tmp = self.snapshots / f".{inst.snapshot_id}.partiel{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            for rel, data in inst.fichiers.items():
                p = tmp / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            mdir = tmp / "models" / inst.sha8
            for rel, data in inst.modele.items():
                p = mdir / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            (mdir / "model-manifest.json").write_bytes(json_bytes(inst.manifeste_modele))
            (tmp / "times.json").write_bytes(json_bytes(times_doc))
            (tmp / "latest.json").write_bytes(json_bytes(latest_doc))
            # le manifeste en dernier : un dossier sans manifeste n'est jamais lu
            (tmp / "publish-manifest.json").write_bytes(json_bytes(inst.manifeste))
            os.replace(tmp, dest)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        # historique, series de vent, puis latest en dernier
        _ecrire_atomique(self.index / "times.json", json_bytes(times_doc))
        if inst.vent_points:
            serie = self.serie_vent()
            for pid, (s, d, rho) in inst.vent_points.items():
                rows = [r for r in serie.get(pid, []) if r[0] != inst.valid_time]
                rows.append([inst.valid_time, s, d, rho])
                rows.sort(key=lambda r: r[0])
                serie[pid] = rows
            _ecrire_atomique(self.index / "vent_points.json", json_bytes(serie))
        courant = self.latest()
        if courant is None or courant["valid_time"] <= inst.valid_time:
            _ecrire_atomique(self.racine / "latest.json", json_bytes(latest_doc))
        return dest


# ---------------------------------------------------------------------- client HTTP
class PublicationRefusee(RuntimeError):
    """Le Worker a refuse une requete (4xx), ou elle a echoue apres les reprises."""

    def __init__(self, message: str, status: int | None = None, corps: str | None = None):
        super().__init__(message)
        self.status, self.corps = status, corps


class PublicationIncomplete(PublicationRefusee):
    """Commit accepte (ou deja present) mais l'index ou latest du Worker ne le montrent
    pas : l'instantane n'est PAS publie. Reparation cote Worker necessaire."""


def jeton_publication(env: str = TOKEN_ENV) -> str | None:
    """Jeton depuis l'environnement (jamais un argument de ligne de commande)."""
    tok = os.environ.get(env, "").strip()
    return tok or None


def normaliser_hote(host: str) -> str:
    """Hote compare en minuscules, sans point final (ATOM.zelqane.com. == atom.zelqane.com)."""
    return host.strip().lower().rstrip(".")


def _boucle_locale(host: str) -> bool:
    host = normaliser_hote(host)
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class _SansRedirection(urllib.request.HTTPRedirectHandler):
    """Aucune redirection suivie : une 3xx devient une erreur. Sinon urllib renverrait la
    requete, en-tete Authorization compris, vers l'hote de la redirection."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ClientPublication:
    """Client du protocole de contract/tools/publish.mjs (meme ordre, memes en-tetes).

        1. PUT /v1/publish/model/{sha8}/files/{path} (meta.json, terrain.f32) puis
           POST /v1/publish/model/{sha8}/commit - une seule fois par modele et par
           processus (le Worker rend de toute facon ces appels idempotents) ;
        2. PUT /v1/publish/{id}/files/{path} pour chaque fichier du manifeste, en-tete
           X-Content-SHA256, chemin encode (/ -> %2F) ;
        3. POST /v1/publish/{id}/commit avec le manifeste, puis RELECTURE : le commit n'est
           reussi que si GET /v1/times?from=vt&to=vt contient l'id ET GET /v1/latest a un
           valid_time >= vt (latest ne recule jamais). Sinon le commit est rejoue une fois
           (le Worker corrige refait l'index et latest), puis PublicationIncomplete ;
        4. POST /v1/heartbeat.

    Reprises : 3 tentatives sur erreur reseau ou 5xx (comme publish.mjs pour les PUT ;
    ici aussi pour les commits, que le Worker rend idempotents). Un 4xx arrete tout.
    Aucune redirection n'est suivie (le jeton ne part jamais vers un autre hote). Le jeton
    n'apparait dans aucun message, aucune exception, aucun repr.
    """

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 120.0,
                 tentatives: int = 3, pause_s: float = 1.0, journal=print) -> None:
        url = urllib.parse.urlsplit(base_url.rstrip("/"))
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError(f"URL de publication invalide : {base_url!r}")
        self.host = normaliser_hote(url.hostname)
        self.local = _boucle_locale(self.host)
        if url.scheme == "http" and not self.local:
            raise ValueError("publication en http vers un hote distant refusee : le jeton "
                             "passerait en clair (https obligatoire hors 127.0.0.1)")
        self.base = base_url.rstrip("/")
        self._token = token if token is not None else jeton_publication()
        if not self._token:
            raise ValueError(f"jeton de publication absent : definir la variable "
                             f"d'environnement {TOKEN_ENV} (jamais en argument)")
        self.timeout, self.tentatives, self.pause_s = timeout, int(tentatives), pause_s
        self.journal = journal or (lambda *_: None)
        self.modeles_commites: set[str] = set()
        self.requetes = 0
        self._opener = urllib.request.build_opener(_SansRedirection())

    def __repr__(self) -> str:                      # jamais le jeton
        return f"ClientPublication({self.base!r})"

    # -------------------------------------------------------------- transport
    def _requete(self, methode: str, chemin: str, corps: bytes | None = None,
                 entetes: dict | None = None, auth: bool = True) -> tuple[int, bytes]:
        h = dict(entetes or {})
        if auth:
            h["authorization"] = f"Bearer {self._token}"
        dernier = None
        for essai in range(1, self.tentatives + 1):
            req = urllib.request.Request(self.base + chemin, data=corps, method=methode,
                                         headers=h)
            self.requetes += 1
            try:
                with self._opener.open(req, timeout=self.timeout) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                texte = e.read().decode("utf-8", "replace")[:2000]
                if 300 <= e.code < 400:
                    raise PublicationRefusee(f"{methode} {chemin} : redirection {e.code} "
                                             "refusee (le jeton ne suit jamais une "
                                             "redirection) : verifier l'URL du Worker",
                                             e.code, texte) from None
                if e.code < 500 or essai >= self.tentatives:
                    raise PublicationRefusee(f"{methode} {chemin} : {e.code} {texte}",
                                             e.code, texte) from None
                dernier = f"{e.code} {texte}"
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                    TimeoutError, OSError) as e:
                dernier = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
                if essai >= self.tentatives:
                    raise PublicationRefusee(f"{methode} {chemin} : echec reseau apres "
                                             f"{essai} tentatives ({dernier})") from None
            time.sleep(self.pause_s * essai)
        raise PublicationRefusee(f"{methode} {chemin} : {dernier}")

    def _put_tous(self, prefixe: str, racine: Path, fichiers: list[dict]) -> None:
        for n, f in enumerate(fichiers, 1):
            data = (racine / f["path"]).read_bytes()
            if len(data) != f["bytes"] or sha256_bytes(data) != f["sha256"]:
                raise PublicationRefusee(f"{f['path']} : contenu local different du manifeste "
                                         "(fichier modifie apres construction ?)")
            self._requete("PUT", f"{prefixe}/files/{urllib.parse.quote(f['path'], safe='')}",
                          data, {"x-content-sha256": f["sha256"],
                                 "content-type": "application/octet-stream"})
            if n % 10 == 0:
                self.journal(f"  {prefixe} {n}/{len(fichiers)}")

    def _commit(self, prefixe: str, manifeste: dict) -> dict:
        corps = json.dumps(manifeste, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        _, rep = self._requete("POST", f"{prefixe}/commit", corps,
                               {"content-type": "application/json"})
        doc = json.loads(rep.decode("utf-8") or "{}")
        self.journal(f"commit {prefixe} : {json.dumps(doc, ensure_ascii=False)}")
        return doc

    def _lire(self, chemin: str):
        _, rep = self._requete("GET", chemin, auth=False)
        return json.loads(rep.decode("utf-8"))

    def est_publie(self, sid: str, valid_time: str) -> tuple[bool, str]:
        """(publie, raison) : l'id est dans /v1/times a son valid_time ET /v1/latest n'est
        pas plus ancien que lui. Lectures publiques, sans jeton."""
        q = urllib.parse.urlencode({"from": valid_time, "to": valid_time})
        try:
            times = self._lire(f"/v1/times?{q}")
        except PublicationRefusee as e:
            return False, f"/v1/times illisible ({e.status})"
        if not any(t.get("snapshot_id") == sid for t in times.get("items", [])):
            return False, "absent de /v1/times"
        try:
            latest = self._lire("/v1/latest")
        except PublicationRefusee as e:
            return False, f"/v1/latest illisible ({e.status})"
        if not latest.get("valid_time") or latest["valid_time"] < valid_time:
            return False, f"/v1/latest en retard ({latest.get('snapshot_id')})"
        return True, "ok"

    # -------------------------------------------------------------- protocole
    def verifier_meta_publiee(self, sha8: str, meta: dict) -> None:
        """meta d'un modele est IMMUABLE : si /v1/meta/{sha8} existe deja et differe
        (fiabilite M2 activee d'un cote seulement, autre data_kind...), on s'arrete avant de
        deposer un instantane qui ne correspondrait pas a son meta."""
        try:
            status, rep = self._requete("GET", f"/v1/meta/{sha8}", auth=False)
        except PublicationRefusee as e:
            if e.status == 404:
                return
            raise
        if status == 200 and json.loads(rep.decode("utf-8")) != meta:
            raise PublicationRefusee(f"meta du modele {sha8} deja publie et different : un "
                                     "modele garde les memes champs (fiabilite M2) et les "
                                     "memes versions pour toute sa duree de vie")

    def publier(self, dossier: str | Path, battement: bool = True) -> dict:
        """Publie un dossier d'instantane (disposition DepotLocal / fixtures). Renvoie la
        reponse du commit de l'instantane ; PublicationIncomplete si le Worker ne montre
        pas l'instantane apres commit (jamais a marquer publie, jamais a purger)."""
        d = Path(dossier)
        manifeste = json.loads((d / "publish-manifest.json").read_bytes().decode("utf-8"))
        sid = manifeste["snapshot_id"]
        if not _ID_RE.match(sid):
            raise ValueError(f"snapshot_id invalide : {sid!r}")
        sha8 = sid[-8:]
        test = manifeste["versions"].get("data_kind") == "test" \
            or "fixtures" in d.resolve().parts
        if test and not self.local:
            raise PublicationRefusee("refus : un instantane de test ne se publie que vers un "
                                     "hote de bouclage (localhost, 127.0.0.1, ::1)")
        mdir = d / "models" / sha8
        mman = json.loads((mdir / "model-manifest.json").read_bytes().decode("utf-8"))
        if sha8 not in self.modeles_commites:
            meta = json.loads((mdir / "meta.json").read_bytes().decode("utf-8"))
            self.verifier_meta_publiee(sha8, meta)
            self._put_tous(f"/v1/publish/model/{sha8}", mdir, mman["files"])
            self._commit(f"/v1/publish/model/{sha8}", mman)
            self.modeles_commites.add(sha8)
        self._put_tous(f"/v1/publish/{sid}", d, manifeste["files"])
        vt = manifeste["valid_time"]
        rep = self._commit(f"/v1/publish/{sid}", manifeste)
        ok, raison = self.est_publie(sid, vt)
        if not ok:
            # commit present mais index / latest non ecrits (echec du Worker apres
            # _commit.json) : un seul rejeu, que le Worker corrige rend reparateur
            self.journal(f"commit {sid} non visible ({raison}) : rejeu du commit")
            rep = self._commit(f"/v1/publish/{sid}", manifeste)
            ok, raison = self.est_publie(sid, vt)
        if not ok:
            raise PublicationIncomplete(
                f"{sid} commite mais non publie ({raison}) apres rejeu du commit : "
                "reparation necessaire cote Worker (index/times.json et latest.json a "
                "refaire depuis snapshots/{id}/_commit.json) ; ne pas marquer publie, ne pas "
                "purger le dossier local")
        if battement:
            self.battement()
        return rep

    def battement(self) -> None:
        """POST /v1/heartbeat : compute_status reste « online » tant qu'il arrive."""
        self._requete("POST", "/v1/heartbeat", b"")


# ---------------------------------------------------------------------- reprise
def champs_attendus(wf) -> list[str]:
    """meta.fields que produirait ce WindFieldModel (M2 actif ou non)."""
    return list(CHAMPS_VENT) + (list(CHAMPS_M2) if wf.reliability is not None else [])


def differences_configuration(dossier: str | Path, versions: dict,
                              champs: list[str]) -> list[str]:
    """Ecarts entre un instantane deja publie et la configuration courante (versions,
    champs publies). Vide = compatible : on peut reutiliser le dossier tel quel."""
    d = Path(dossier)
    man = json.loads((d / "publish-manifest.json").read_bytes().decode("utf-8"))
    meta = json.loads((d / "models" / man["snapshot_id"][-8:] / "meta.json").read_bytes()
                      .decode("utf-8"))
    out = [f"versions.{k} : publie {man['versions'].get(k)!r}, configuration {v!r}"
           for k, v in versions.items() if man["versions"].get(k) != v]
    if list(meta["fields"]) != list(champs):
        out.append(f"champs publies {meta['fields']} != configuration {champs} "
                   "(fiabilite M2 activee d'un cote seulement ?)")
    return out
