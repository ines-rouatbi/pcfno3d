"""Donnees du front-end ATOM Wind (webapp/atom-wind/) : le champ du modele, compact.

Le front-end raisonne en hauteur AU-DESSUS DU SOL (une grue a 40 m, un drone a 100 m...) sur une
grille de 500 m (68 x 56 km pour le Grand Tunis). Un volume complet (12 x 560 x 680, 100 m) pese
~55 Mo : on envoie a la place, pour chaque instant d'une journee, u et v a quelques hauteurs AGL,
moyennes par blocs de 500 m, en centimetres par seconde (int16) — ~0,6 Mo par instant avant gzip.

    grille(...)  relief, eau, z0, stations, georeferencement (une fois)
    jours(...)   les journees servies (heure locale de Tunis, UTC+1, sans heure d'ete)
    jour(...)    les champs d'une journee
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

HAUTEURS_AGL = (10, 25, 50, 80, 100, 150, 200, 300, 500, 800, 1000)
PAS_M = 500.0
UTC_LOCAL_H = 1                      # Tunisie : UTC+1 toute l'annee
NODATA_I16 = -32768


def champ_agl(u: np.ndarray, v: np.ndarray, geom, h: float) -> tuple[np.ndarray, np.ndarray]:
    """u, v (nz, ny, nx ; NaN sous le relief) a h m au-dessus du sol, colonne par colonne :
    interpolation entre deux noeuds valides, ou profil log depuis le noeud valide le plus bas
    (z0 de la maille) si h est sous ce noeud — comme pcfno3d.stations_reelles.poids_10m.
    Au-dessus du dernier noeud valide : valeur de ce noeud."""
    valid = geom.valid
    hk = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    uo = np.full(u.shape[1:], np.nan)
    vo = np.full(u.shape[1:], np.nan)
    for k in range(len(geom.z_levels) - 1):
        m = valid[k] & valid[k + 1] & (hk[k] <= h) & (h <= hk[k + 1]) & np.isnan(uo)
        a = np.where(m, (h - hk[k]) / np.maximum(hk[k + 1] - hk[k], 1e-6), 0.0)
        uo = np.where(m, (1 - a) * u[k] + a * u[k + 1], uo)
        vo = np.where(m, (1 - a) * v[k] + a * v[k + 1], vo)
    k0 = np.argmax(valid, axis=0)
    a_un = valid.any(axis=0)
    h0 = np.take_along_axis(hk, k0[None], 0)[0]
    bas = a_un & (h < h0) & np.isnan(uo)
    z0 = np.maximum(geom.z0, 1e-4)
    f = np.where(bas, np.log(h / z0) / np.log(np.maximum(h0, 1.01 * z0) / z0), 0.0)
    u0 = np.take_along_axis(u, k0[None], 0)[0]
    v0 = np.take_along_axis(v, k0[None], 0)[0]
    uo = np.where(bas, f * u0, uo)
    vo = np.where(bas, f * v0, vo)
    # au-dessus du dernier noeud valide de la colonne
    k1 = valid.shape[0] - 1 - np.argmax(valid[::-1], axis=0)
    haut = a_un & np.isnan(uo) & (h > np.take_along_axis(hk, k1[None], 0)[0])
    uo = np.where(haut, np.take_along_axis(u, k1[None], 0)[0], uo)
    vo = np.where(haut, np.take_along_axis(v, k1[None], 0)[0], vo)
    return uo, vo


def blocs(a: np.ndarray, b: int, fn=np.nanmean) -> np.ndarray:
    """Moyenne (ou autre) par blocs b x b ; le bord incomplet est garde."""
    ny, nx = a.shape
    py, px = (-ny) % b, (-nx) % b
    p = np.pad(a.astype(float), ((0, py), (0, px)), constant_values=np.nan)
    p = p.reshape((ny + py) // b, b, (nx + px) // b, b)
    with np.errstate(invalid="ignore"), _sans_alerte():
        return fn(p, axis=(1, 3))


class _sans_alerte:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


def facteur(geom) -> int:
    return max(int(round(PAS_M / geom.dx)), 1)


def grille(geom, georef, cfg) -> dict:
    b = facteur(geom)
    terrain = blocs(geom.zt, b)
    eau = blocs((geom.z0 < 0.001).astype(float), b) > 0.5
    z0 = np.exp(blocs(np.log(np.maximum(geom.z0, 1e-4)), b))
    stations = []
    f = cfg.paths.data_dir / "stations.json"
    if f.exists():
        for s in json.loads(f.read_text(encoding="utf-8")):
            x, y = s["ix"] * geom.dx, s["iy"] * geom.dy
            lat, lon = georef.to_latlon(x, y) if georef is not None else (None, None)
            stations.append({"nom": s.get("nom", "station"), "x_km": x / 1000, "y_km": y / 1000,
                             "lat": None if lat is None else round(float(lat), 5),
                             "lon": None if lon is None else round(float(lon), 5)})
    return {
        "format": "atom-wind/1", "nx": int(terrain.shape[1]), "ny": int(terrain.shape[0]),
        "pas_km": geom.dx * b / 1000, "largeur_km": geom.nx * geom.dx / 1000,
        "hauteur_km": geom.ny * geom.dy / 1000,
        "lat0": None if georef is None else georef.lat0,
        "lon0": None if georef is None else georef.lon0,
        "rotation_deg": None if georef is None else georef.rotation_deg,
        "hauteurs_agl_m": list(HAUTEURS_AGL),
        "terrain_m": np.round(np.nan_to_num(terrain), 1).tolist(),
        "eau": eau.astype(int).tolist(),
        "z0_m": np.round(z0, 5).tolist(),
        "stations": stations,
    }


def heure_locale(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t + pd.Timedelta(hours=UTC_LOCAL_H)


def jours(index: pd.DataFrame, split: str | None = "test") -> list[dict]:
    idx = index if split is None else index[index["split"] == split]
    out: dict[str, list] = {}
    for stem, ts in zip(idx["stem"], idx["timestamp"]):
        loc = heure_locale(ts)
        out.setdefault(loc.strftime("%Y-%m-%d"), []).append(
            {"stem": str(stem), "timestamp_utc": pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "heure_locale": round(loc.hour + loc.minute / 60, 3)})
    return [{"date": d, "instants": sorted(v, key=lambda r: r["heure_locale"])}
            for d, v in sorted(out.items())]


def encoder(champs: np.ndarray) -> str:
    """float (m/s, NaN) -> int16 en cm/s, petit-boutiste, base64."""
    q = np.where(np.isfinite(champs), np.clip(np.round(champs * 100), -32767, 32767), NODATA_I16)
    return base64.b64encode(q.astype("<i2").tobytes()).decode("ascii")


def instant(volume, geom) -> np.ndarray:
    """(n_hauteurs, 2, ny, nx) : u, v aux hauteurs AGL, moyennes par blocs de 500 m."""
    b = facteur(geom)
    out = []
    for h in HAUTEURS_AGL:
        u, v = champ_agl(volume.u, volume.v, geom, float(h))
        out.append([blocs(u, b), blocs(v, b)])
    return np.asarray(out, dtype=np.float32)


def jour(predire, geom, j: dict, fiabilite=None) -> dict:
    """Les champs d'une journee. predire(stem) -> Volume ; fiabilite(volume) -> (n_h, ny, nx)
    sigma de la vitesse (m/s) ou None."""
    instants, blocs_vent, blocs_sigma = [], [], []
    for it in j["instants"]:
        vol = predire(it["stem"])
        blocs_vent.append(instant(vol, geom))
        if fiabilite is not None:
            s = fiabilite(vol)
            if s is not None:
                blocs_sigma.append(s)
        instants.append({**it, "inference_s": round(float(vol.inference_s), 3),
                         "capteurs_utilises": int(vol.n_sensors)})
    a = np.stack(blocs_vent)
    out = {"format": "atom-wind/1", "date": j["date"], "instants": instants,
           "hauteurs_agl_m": list(HAUTEURS_AGL), "forme": list(a.shape),
           "unite": "cm/s, int16 petit-boutiste, -32768 = sans valeur",
           "ordre": "instant, hauteur, (u, v), y, x", "vent": encoder(a)}
    if blocs_sigma and len(blocs_sigma) == len(instants):
        s = np.stack(blocs_sigma)
        out["sigma_vitesse"] = encoder(s)
        out["forme_sigma"] = list(s.shape)
    return out


def sigma_agl(volume, geom) -> np.ndarray | None:
    """sigma de la vitesse aux hauteurs AGL (meme interpolation que le vent), si la
    fiabilite (lot M2) du volume est prete."""
    rel = getattr(volume, "reliability", None)
    if rel is None:
        return None
    b = facteur(geom)
    s = rel.sigma_speed
    return np.asarray([blocs(champ_agl(s, s, geom, float(h))[0], b) for h in HAUTEURS_AGL],
                      dtype=np.float32)


def plan_aneo(run_dir: Path) -> Path | None:
    """Le plan OS6 (scripts/os6_reseau_aneo.py), s'il a ete calcule."""
    for d in (run_dir / "os6_reseau_aneo", run_dir.parent / "os6_reseau_aneo"):
        f = d / "plan_aneo.geojson"
        if f.exists():
            return f
    return None
