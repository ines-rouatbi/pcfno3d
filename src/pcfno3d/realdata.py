"""Donnees REELLES -> le cache du pipeline (S, Y, index.csv), sans toucher au reste.

Le pipeline (entrainement, evaluation, API) lit un cache : par instant, une surface
`S` (8, ny, nx) et une cible `Y` (4, nz, ny, nx), plus `index.csv`. Le jeu synthetique le
remplit depuis les CSV du generateur (`cache.py`). Ce module le remplit depuis des
sources reelles, dans le MEME format :

    entree  : un tableau de stations (lat, lon, hauteur, vent, T, p, HR...) -> S
    cible   : un champ maille 3D (reanalyse CERRA/ERA5, sortie WRF...)    -> Y dense
              ou des observations ponctuelles (radiosondes, mats, lidar)    -> Y partiel
              ou rien (inference seule)                                     -> Y vide

Une cible PARTIELLE porte NaN la ou rien n'est observe : la loss et les metriques ne
supervisent que les noeuds connus (voir losses.py, dso.py, evaluate.py). La densite
`rho` (canal 3 de Y) est toujours remplie : le residu de masse en a besoin partout ; a
defaut d'observation, atmosphere standard (ISA).

Tout passe par un ancrage geographique du domaine (section `real` de la configuration :
coin sud-ouest en lat/lon). Le domaine synthetique n'en a pas : sans lui, rien ne se
place, et ce module refuse de deviner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import Config
from .geometry import SURF_VARS, Geometry

R_EARTH = 6_371_008.8          # m, rayon moyen
RD, RV = 287.05, 461.5         # J/(kg.K), air sec et vapeur d'eau

STATION_REQUIRED = ["station_id", "timestamp_utc", "lat", "lon", "height_agl_m"]
STATION_OPTIONAL = ["wind_u_m_s", "wind_v_m_s", "wind_speed_m_s", "wind_dir_deg",
                    "wind_w_m_s", "temperature_c", "pressure_pa", "relative_humidity_pct",
                    "air_density_kg_m3", "quality"]
POINT_REQUIRED = ["timestamp_utc", "lat", "lon"]          # + z_asl_m ou z_agl_m, + vent


# ---------------------------------------------------------------------- geographie
@dataclass
class GeoRef:
    """Plan tangent local ancre au coin SUD-OUEST du domaine (x vers l'est, y vers le nord).

    Projection equirectangulaire autour du coin, plus une rotation eventuelle du domaine.
    Sur 50 km, l'ecart a une projection conforme reste de l'ordre du metre, bien en
    dessous de la maille de 100 m.
    """

    lat0: float
    lon0: float
    rotation_deg: float = 0.0       # angle de l'axe x du domaine depuis l'est (antihoraire)

    @classmethod
    def from_cfg(cls, cfg: Config) -> "GeoRef":
        r = cfg.real
        if r.origin_lat is None or r.origin_lon is None:
            raise ValueError(
                "domaine non georeference : renseigne real.origin_lat et real.origin_lon "
                "(coin SUD-OUEST du domaine, en degres) dans la configuration. Le jeu "
                "synthetique n'a pas d'ancrage geographique : il faut les coordonnees "
                "reelles du domaine (Tukhnanutha).")
        return cls(float(r.origin_lat), float(r.origin_lon), float(r.rotation_deg))

    def to_xy(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        e = R_EARTH * math.cos(math.radians(self.lat0)) * np.radians(lon - self.lon0)
        n = R_EARTH * np.radians(lat - self.lat0)
        a = math.radians(self.rotation_deg)
        return e * math.cos(a) + n * math.sin(a), -e * math.sin(a) + n * math.cos(a)

    def to_latlon(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.asarray(x, float), np.asarray(y, float)
        a = math.radians(self.rotation_deg)
        e = x * math.cos(a) - y * math.sin(a)
        n = x * math.sin(a) + y * math.cos(a)
        lat = self.lat0 + np.degrees(n / R_EARTH)
        lon = self.lon0 + np.degrees(e / (R_EARTH * math.cos(math.radians(self.lat0))))
        return lat, lon


@dataclass
class PlanGrille:
    """Projection des grilles REELLES (scripts/grille_reelle.py, forcage.InterpERA5) : plan local a pas
    metrique constant, pris a la latitude CENTRALE de la grille fine (lat = lat0 + y / m_lat,
    lon = lon0 + x / m_lon). Meme interface que GeoRef (to_xy, to_latlon, lat0, lon0, rotation_deg).

    Sur le Grand Tunis, l'ecart a GeoRef (pas pris au coin sud-ouest, sphere) reste sous 0,15 km ; sur la
    Tunisie entiere (828 x 388 km), GeoRef deplacerait l'est du domaine d'environ 15 km : les couches du
    front-end ATOM Wind (stations, lieux, plan ANEO) utilisent donc ce plan, celui du relief et du modele."""

    lat0: float
    lon0: float
    m_lon: float
    m_lat: float
    lat_centrale: float
    rotation_deg: float = 0.0

    @classmethod
    def from_geom(cls, cfg: Config, geom) -> "PlanGrille":
        from .forcage import metres_par_degre
        r = cfg.real
        if r.origin_lat is None or r.origin_lon is None:
            raise ValueError("domaine non georeference : real.origin_lat / origin_lon")
        if abs(float(r.rotation_deg or 0.0)) > 1e-9:
            raise ValueError("PlanGrille : grille tournee non prise en charge")
        lat0, lon0 = float(r.origin_lat), float(r.origin_lon)
        lat_c = lat0 + geom.ny_s * geom.dy_s / 2 / metres_par_degre(lat0)[1]
        m_lon, m_lat = metres_par_degre(lat_c)
        return cls(lat0, lon0, float(m_lon), float(m_lat), float(lat_c))

    def to_xy(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        return (lon - self.lon0) * self.m_lon, (lat - self.lat0) * self.m_lat

    def to_latlon(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.asarray(x, float), np.asarray(y, float)
        return self.lat0 + y / self.m_lat, self.lon0 + x / self.m_lon


def georef_grille(cfg: Config, geom) -> "PlanGrille | GeoRef | None":
    """La projection du domaine servi : PlanGrille (celle du relief) si la grille est georeferencee et
    non tournee, GeoRef si elle est tournee, None sans ancrage geographique."""
    try:
        return PlanGrille.from_geom(cfg, geom)
    except ValueError:
        try:
            return GeoRef.from_cfg(cfg)
        except ValueError:
            return None


# ---------------------------------------------------------------------- physique
def saturation_vapour_pressure(t_c):
    """Magnus (Pa), T en degres C."""
    t_c = np.asarray(t_c, float)
    return 611.2 * np.exp(17.62 * t_c / (243.12 + t_c))


def isa(z_asl_m):
    """Atmosphere standard : (pression Pa, temperature C, densite kg/m3) a l'altitude z."""
    z = np.asarray(z_asl_m, float)
    t_c = 15.0 - 0.0065 * z
    p = 101325.0 * (1.0 - 2.25577e-5 * z) ** 5.25588
    return p, t_c, p / (RD * (t_c + 273.15))


def moist_density(p_pa, t_c, rh_pct):
    """Densite de l'air humide (gaz parfait, melange air sec + vapeur)."""
    tk = np.asarray(t_c, float) + 273.15
    e = np.clip(np.asarray(rh_pct, float), 0, 100) / 100.0 * saturation_vapour_pressure(t_c)
    return (np.asarray(p_pa, float) - e) / (RD * tk) + e / (RV * tk)


def wind_components(speed, direction_deg):
    """Convention meteo (direction d'OU vient le vent) -> u (est), v (nord)."""
    d = np.radians(np.asarray(direction_deg, float))
    s = np.asarray(speed, float)
    return -s * np.sin(d), -s * np.cos(d)


# ---------------------------------------------------------------------- stations -> S
def normalize_stations(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Controle et mise en forme d'un tableau de stations. Renvoie (propre, rapport).

    Colonnes : voir STATION_REQUIRED / STATION_OPTIONAL. Vent en (u, v) OU en
    (vitesse, direction meteo). Les mesures douteuses sont ECARTEES, jamais corrigees.
    """
    missing = [c for c in STATION_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"colonnes obligatoires absentes du tableau de stations : {missing}")
    d = df.copy()
    for c in STATION_OPTIONAL:
        if c not in d.columns:
            d[c] = np.nan
    # JSON : les champs absents arrivent en None (colonnes « object ») -> nombres ou NaN
    for c in STATION_OPTIONAL + ["lat", "lon", "height_agl_m"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True).dt.tz_localize(None)
    need_uv = d["wind_u_m_s"].isna() | d["wind_v_m_s"].isna()
    u, v = wind_components(d.loc[need_uv, "wind_speed_m_s"], d.loc[need_uv, "wind_dir_deg"])
    d.loc[need_uv, "wind_u_m_s"], d.loc[need_uv, "wind_v_m_s"] = u, v
    d["wind_w_m_s"] = d["wind_w_m_s"].fillna(0.0)

    rapport = {"lignes": int(len(d))}
    r = cfg.real
    bad_wind = d["wind_u_m_s"].isna() | d["wind_v_m_s"].isna()
    too_fast = np.hypot(d["wind_u_m_s"], d["wind_v_m_s"]) > r.max_speed_m_s
    low_q = d["quality"].fillna(1.0) < r.min_quality
    bad_h = ~(d["height_agl_m"].between(0.5, 300.0))
    for nom, m in (("vent_absent", bad_wind), ("vent_aberrant", too_fast),
                   ("qualite_faible", low_q), ("hauteur_hors_bornes", bad_h)):
        rapport[f"ecartees_{nom}"] = int(m.sum())
    d = d[~(bad_wind | too_fast | low_q | bad_h)].reset_index(drop=True)
    rapport["gardees"] = int(len(d))
    return d, rapport


def stations_to_surface(d: pd.DataFrame, geom: Geometry, geo: GeoRef,
                        cfg: Config) -> tuple[np.ndarray, dict]:
    """Les stations d'UN instant -> S (8, ny, nx) sur la grille du modele.

    Chaque station tombe dans la maille la plus proche ; plusieurs stations d'une meme
    maille sont moyennees. Le vent est ramene a la hauteur des capteurs d'entrainement
    (loi logarithmique, rugosite de la maille) : le modele n'a vu que des capteurs a
    ~25-30 m. T, p, HR manquants -> atmosphere standard a l'altitude de la station ; la
    densite est recalculee si absente.
    """
    ny, nx = geom.ny, geom.nx
    x, y = geo.to_xy(d["lat"].values, d["lon"].values)
    i = np.rint(x / geom.dx).astype(int)
    j = np.rint(y / geom.dy).astype(int)
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    d, i, j = d[inside], i[inside], j[inside]

    zt = geom.zt[j, i]
    z_st = zt + d["height_agl_m"].values
    p_isa, t_isa, _ = isa(z_st)
    t_c = d["temperature_c"].fillna(pd.Series(t_isa, index=d.index)).values
    p = d["pressure_pa"].fillna(pd.Series(p_isa, index=d.index)).values
    rh = d["relative_humidity_pct"].fillna(50.0).values
    rho = d["air_density_kg_m3"].fillna(pd.Series(moist_density(p, t_c, rh),
                                                  index=d.index)).values

    z_ref = cfg.real.ref_height_m or float(np.nanmedian(geom.zrel_sensor))
    z0 = np.maximum(geom.z0[j, i], 1e-3)
    h = np.maximum(d["height_agl_m"].values, 2 * z0)
    f = np.log(z_ref / z0) / np.log(h / z0)                  # loi log vers z_ref
    vals = np.stack([d["wind_u_m_s"].values * f, d["wind_v_m_s"].values * f,
                     d["wind_w_m_s"].values, t_c, rho, rh, p])  # ordre de SURF_VARS

    S = np.zeros((8, ny, nx), dtype=np.float64)
    cnt = np.zeros((ny, nx))
    np.add.at(cnt, (j, i), 1.0)
    for c in range(len(SURF_VARS)):
        np.add.at(S[c], (j, i), vals[c])
    has = cnt > 0
    S[:7, has] /= cnt[has]
    S[7] = has.astype(np.float64)
    info = {"stations": int(len(d)), "hors_domaine": int((~inside).sum()),
            "mailles": int(has.sum()), "hauteur_ref_m": round(z_ref, 1)}
    return S.astype(np.float32), info


# ---------------------------------------------------------------------- cibles -> Y
def _node_latlon(geom: Geometry, geo: GeoRef) -> tuple[np.ndarray, np.ndarray]:
    jj, ii = np.meshgrid(np.arange(geom.ny), np.arange(geom.nx), indexing="ij")
    return geo.to_latlon(ii * geom.dx, jj * geom.dy)


def _rho_background(geom: Geometry) -> np.ndarray:
    """Densite ISA aux noeuds (nz, ny, nx) : le residu de masse en a besoin partout."""
    z = np.asarray(geom.z_levels, float)[:, None, None] * np.ones((1, geom.ny, geom.nx))
    return isa(z)[2]


def grid_to_target(lat, lon, z_asl, fields: dict[str, np.ndarray], geom: Geometry,
                   geo: GeoRef) -> np.ndarray:
    """Champ maille (reanalyse / WRF) -> Y (4, nz, ny, nx) aux noeuds du modele.

    lat (nlat,), lon (nlon,) croissants ; z_asl (nzs,) ou (nzs, nlat, nlon), en m au-dessus
    de la mer, croissant ; fields : 'u', 'v' obligatoires, 'w' et 'rho' facultatifs, de
    forme (nzs, nlat, nlon). Bilineaire a l'horizontale, lineaire a la verticale. Hors de
    la source : NaN (inconnu). Sous le relief : 0, comme le cache synthetique.
    """
    from scipy.interpolate import RegularGridInterpolator

    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if lat[0] > lat[-1]:                       # beaucoup de produits stockent nord -> sud
        lat = lat[::-1]
        fields = {k: v[:, ::-1] for k, v in fields.items()}
        if np.ndim(z_asl) == 3:
            z_asl = np.asarray(z_asl)[:, ::-1]
    for k in ("u", "v"):
        if k not in fields:
            raise ValueError(f"champ '{k}' absent de la cible maillee")
    nlat_t, nlon_t = _node_latlon(geom, geo)
    pts = np.stack([nlat_t.ravel(), nlon_t.ravel()], axis=1)

    def horiz(a3):                             # (nzs, nlat, nlon) -> (nzs, ny*nx)
        return np.stack([RegularGridInterpolator((lat, lon), lev, bounds_error=False,
                                                  fill_value=np.nan)(pts) for lev in a3])

    zs = horiz(np.asarray(z_asl, float)) if np.ndim(z_asl) == 3 \
        else np.repeat(np.asarray(z_asl, float)[:, None], pts.shape[0], axis=1)
    targets = np.asarray(geom.z_levels, float)
    nz, ny, nx = geom.shape
    Y = np.full((4, nz, ny * nx), np.nan)
    for c, name in enumerate(("u", "v", "w", "rho")):
        if name not in fields or fields[name] is None:
            continue
        src = horiz(np.asarray(fields[name], float))
        for k, zt in enumerate(targets):
            # tolerance d'1 mm : un niveau du modele confondu avec le premier ou le dernier
            # niveau source ne doit pas tomber « hors source » pour une erreur d'arrondi
            eps = 1e-3
            below = (zs <= zt + eps).sum(axis=0) - 1           # indice du niveau source sous zt
            ok = (zt >= zs[0] - eps) & (zt <= zs[-1] + eps)
            b = np.clip(below, 0, zs.shape[0] - 2)
            cols = np.arange(zs.shape[1])
            z0_, z1_ = zs[b, cols], zs[b + 1, cols]
            wgt = np.clip((zt - z0_) / np.where(z1_ > z0_, z1_ - z0_, np.nan), 0, 1)
            val = src[b, cols] * (1 - wgt) + src[b + 1, cols] * wgt
            Y[c, k] = np.where(ok, val, np.nan)
    Y = Y.reshape(4, nz, ny, nx)
    rho_bg = _rho_background(geom)
    Y[3] = np.where(np.isfinite(Y[3]), Y[3], rho_bg)
    Y[:, ~geom.valid] = 0.0
    return Y.astype(np.float32)


def points_to_target(d: pd.DataFrame, geom: Geometry, geo: GeoRef) -> np.ndarray:
    """Observations ponctuelles d'UN instant (radiosonde, mat, lidar) -> Y PARTIEL.

    Chaque mesure va au noeud le plus proche (horizontalement, et au niveau du modele le
    plus proche s'il est a moins d'une demi-maille verticale). Le reste vaut NaN : la
    loss ne supervise que ces noeuds. rho : observe si fourni, sinon ISA.
    """
    d = d.copy()
    if "z_asl_m" not in d.columns:
        if "z_agl_m" not in d.columns:
            raise ValueError("observations ponctuelles : il faut z_asl_m ou z_agl_m")
        d["z_asl_m"] = np.nan
    for c in ("wind_u_m_s", "wind_v_m_s", "wind_w_m_s", "wind_speed_m_s", "wind_dir_deg",
              "air_density_kg_m3", "z_agl_m"):
        if c not in d.columns:
            d[c] = np.nan
    need = d["wind_u_m_s"].isna() | d["wind_v_m_s"].isna()
    u, v = wind_components(d.loc[need, "wind_speed_m_s"], d.loc[need, "wind_dir_deg"])
    d.loc[need, "wind_u_m_s"], d.loc[need, "wind_v_m_s"] = u, v

    nz, ny, nx = geom.shape
    x, y = geo.to_xy(d["lat"].values, d["lon"].values)
    i, j = np.rint(x / geom.dx).astype(int), np.rint(y / geom.dy).astype(int)
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    d, i, j = d[inside], i[inside], j[inside]
    z = d["z_asl_m"].values.copy()
    agl = np.isnan(z)
    z[agl] = geom.zt[j[agl], i[agl]] + d["z_agl_m"].values[agl]
    levels = np.asarray(geom.z_levels, float)
    k = np.abs(z[:, None] - levels[None]).argmin(axis=1)
    gaps = np.diff(levels)
    half = np.concatenate([[gaps[0]], np.minimum(gaps[:-1], gaps[1:]), [gaps[-1]]]) / 2
    close = np.abs(z - levels[k]) <= half[k]

    Y = np.full((4, nz, ny, nx), np.nan)
    sums = np.zeros((4, nz, ny, nx)); cnt = np.zeros((4, nz, ny, nx))
    cols = ("wind_u_m_s", "wind_v_m_s", "wind_w_m_s", "air_density_kg_m3")
    for c, name in enumerate(cols):
        val = d[name].values
        m = close & np.isfinite(val)
        np.add.at(sums[c], (k[m], j[m], i[m]), val[m])
        np.add.at(cnt[c], (k[m], j[m], i[m]), 1.0)
    have = cnt > 0
    Y[have] = sums[have] / cnt[have]
    Y[3] = np.where(np.isfinite(Y[3]), Y[3], _rho_background(geom))
    Y[:, ~geom.valid] = 0.0
    return Y.astype(np.float32)


def empty_target(geom: Geometry) -> np.ndarray:
    """Aucune cible (inference seule, ou entrainement purement physique) : NaN + rho ISA."""
    nz, ny, nx = geom.shape
    Y = np.full((4, nz, ny, nx), np.nan)
    Y[3] = _rho_background(geom)
    Y[:, ~geom.valid] = 0.0
    return Y.astype(np.float32)


def open_gridded(path: str | Path, var_map: dict[str, str]) -> Callable:
    """Fichier NetCDF/GRIB maille -> fonction timestamp -> (lat, lon, z_asl, fields).

    var_map nomme les variables du fichier : {'u': ..., 'v': ..., 'w': ..., 'rho': ...,
    'lat': ..., 'lon': ..., 'z': ..., 'time': ...}. 'z' : une coordonnee de hauteur au-dessus
    de la mer, ou une variable 3D. Lecture paresseuse via xarray (dependance facultative).
    """
    try:
        import xarray as xr
    except ImportError as exc:                 # pragma: no cover - selon l'environnement
        raise ImportError("lecture NetCDF/GRIB : pip install xarray netCDF4 (et cfgrib "
                          "pour du GRIB)") from exc
    ds = xr.open_dataset(path)
    t_name = var_map.get("time", "time")

    def at(ts: pd.Timestamp):
        snap = ds.sel({t_name: np.datetime64(ts)}, method="nearest")
        if abs(pd.Timestamp(snap[t_name].values) - ts) > pd.Timedelta("31min"):
            return None
        lat = snap[var_map.get("lat", "latitude")].values
        lon = snap[var_map.get("lon", "longitude")].values
        z = snap[var_map["z"]].values
        fields = {k: snap[var_map[k]].values for k in ("u", "v", "w", "rho") if var_map.get(k)}
        return lat, lon, z, fields
    return at


G0 = 9.80665                   # m/s2, pour passer du geopotentiel a l'altitude


def open_era5_pressure_levels(paths) -> Callable:
    """Fichiers ERA5 « niveaux de pression » (scripts/download_reanalysis.py) -> fonction
    timestamp -> (lat, lon, z_asl, fields), a passer a `grid_to_target`.

    Altitude = geopotentiel / g ; densite = p / (Rd T) ; w = -omega / (rho g). Les niveaux
    sont tries du bas vers le haut. ERA5 (~31 km) donne un FOND d'altitude reel, pas une
    verite a 100 m. Noms acceptes : ceux du CDS actuel (valid_time, pressure_level) et
    les anciens (time, level).
    """
    import glob

    try:
        import xarray as xr
    except ImportError as exc:                 # pragma: no cover - selon l'environnement
        raise ImportError("pip install xarray netCDF4") from exc
    files = sorted(glob.glob(str(paths))) if isinstance(paths, (str, Path)) else list(paths)
    if not files:
        raise FileNotFoundError(f"aucun fichier ERA5 pour {paths}")
    datasets = [xr.open_dataset(f) for f in files]

    def dim(ds, *names):
        return next(n for n in names if n in ds.dims or n in ds.coords)

    def at(ts: pd.Timestamp):
        for ds in datasets:
            t = dim(ds, "valid_time", "time")
            times = pd.to_datetime(ds[t].values)
            k = int(np.abs(times - ts).argmin())
            if abs(times[k] - ts) > pd.Timedelta("31min"):
                continue
            snap = ds.isel({t: k})
            lev = dim(snap, "pressure_level", "level")
            p = snap[lev].values.astype(float) * 100.0                 # hPa -> Pa
            z = snap["z"].values / G0                                  # (nlev, nlat, nlon)
            T = snap["t"].values
            rho = p[:, None, None] / (RD * T)
            fields = {"u": snap["u"].values, "v": snap["v"].values, "rho": rho}
            if "w" in snap:
                fields["w"] = -snap["w"].values / (rho * G0)           # Pa/s -> m/s
            order = np.argsort(np.nanmean(z, axis=(1, 2)))             # bas -> haut
            fields = {kk: np.asarray(vv)[order] for kk, vv in fields.items()}
            return (snap["latitude"].values, snap["longitude"].values, z[order], fields)
        return None
    return at


# ---------------------------------------------------------------------- ecriture du cache
def split_of_day(day: int, period: int) -> str:
    """Decoupage par jours entiers, entrelace : val et test couvrent toute la periode."""
    r = day % period
    return "val" if r == period // 2 else ("test" if r == period - 1 else "train")


def import_real(cfg: Config, stations: pd.DataFrame,
                target: Callable[[pd.Timestamp], np.ndarray | None] | None = None,
                verbose: bool = True) -> dict:
    """Stations (+ cible eventuelle) -> cache S/Y + index.csv, au format du pipeline.

    target : fonction timestamp -> Y (4, nz, ny, nx) ou None (pas de cible a cet instant ;
    l'instant est alors ecrit sans supervision, utile pour l'evaluation qualitative).
    """
    geom = Geometry(cfg, device="cpu")
    geo = GeoRef.from_cfg(cfg)
    clean, rapport = normalize_stations(stations, cfg)
    clean["bin"] = clean["timestamp_utc"].dt.floor(cfg.real.time_bin)
    agg = (clean.groupby(["bin", "station_id"], as_index=False)
                .agg({c: "mean" for c in ["lat", "lon", "height_agl_m", "wind_u_m_s",
                                          "wind_v_m_s", "wind_w_m_s", "temperature_c",
                                          "pressure_pa", "relative_humidity_pct",
                                          "air_density_kg_m3"]}))
    first = agg["bin"].min().normalize()
    cache = cfg.paths.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    rows, kinds = [], {"dense": 0, "partielle": 0, "aucune": 0}
    for ts, group in agg.groupby("bin"):
        S, info = stations_to_surface(group, geom, geo, cfg)
        if info["mailles"] == 0:
            continue
        Y = target(ts) if target is not None else None
        if Y is None:
            Y, kind = empty_target(geom), "aucune"
        else:
            known = np.isfinite(Y[:2][:, geom.valid])
            kind = "dense" if known.all() else ("partielle" if known.any() else "aucune")
        kinds[kind] += 1
        day = (ts.normalize() - first).days + 1
        stem = f"{day:03d}_{ts.strftime('%H%M%S')}"
        np.save(cache / f"{stem}_S.npy", S)
        np.save(cache / f"{stem}_Y.npy", Y.astype(np.float32))
        rows.append({"stem": stem, "day": day, "timestamp": str(ts),
                     "split": split_of_day(day, cfg.real.split_period_days),
                     "hour": ts.hour + ts.minute / 60 + ts.second / 3600,
                     "doy": ts.dayofyear, "cible": kind, "capteurs": info["mailles"]})
    new = pd.DataFrame(rows)
    idx_path = cfg.paths.index_csv
    if idx_path.exists() and len(new):
        old = pd.read_csv(idx_path)
        new = pd.concat([old[~old["stem"].isin(new["stem"])], new], ignore_index=True)
    if len(new):
        new.sort_values("stem").to_csv(idx_path, index=False)
    rapport.update({"instants": len(rows), "cibles": kinds, "cache": str(cache),
                    "debut": str(first.date()) if len(rows) else None})
    if verbose:
        print(f"import reel : {len(rows)} instants -> {cache} | cibles {kinds} | "
              f"stations gardees {rapport['gardees']}/{rapport['lignes']}")
    return rapport
