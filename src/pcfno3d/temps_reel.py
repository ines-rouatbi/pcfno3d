"""Entree TEMPS REEL du modele reel (Grand Tunis) : le vent d'aujourd'hui, heure par heure.

ERA5 arrive avec ~5 jours de retard : en temps reel, la grande echelle vient de la prevision
mondiale GFS (NOAA) a 0,25 deg servie par Open-Meteo (cahier v12, OS7 : GFS, archive depuis 2021,
la ou ECMWF n'est archive que depuis 2024), sur les MEMES
mailles que le forcage d'entrainement (`geom.forcage.lat/lon`) et calcule par les MEMES formules
que scripts/forcage_era5.py :

    ug, vg = vent a 850 hPa ; cisaillement (V700 - V850) / dz ; veer 10 m / 850 hPa borne a
    [0, 45] ; stabilite = tanh((theta 1000-925 hPa - 1,5 - theta 2 m) / 3) ; contraste terre-mer
    = t2m terres - t2m mer (la regle « sans slx » du script) ; profil par maille aux hauteurs
    au-dessus du sol (10 m, 100 m et niveaux de pression, interpolation lineaire).

Differences assumees, dites dans les reponses : relief des mailles = altitude Open-Meteo du
point (MNT 90 m) au lieu du geopotentiel de surface ERA5 ; heures passees = previsions GFS a
courte echeance, heures a venir = prevision. Les 11 niveaux de pression d'ERA5 (1000 a 700 hPa)
existent dans GFS. Le modele a appris avec ERA5 : l'ecart ERA5 / GFS n'est pas corrige tant que
l'affinage OS7 sur les previsions GFS archivees n'est pas fait.

Stations : METAR de Tunis-Carthage (aviationweather.gov, comme pcfno3d.metar) et SYNOP de
Zaghouan (60729) et Nabeul (60728) (OGIMET), la mesure la plus proche a 30 min au plus, comme
`stations_reelles.obs_a`. La surface hors stations (T, p, HR, rho) vient des champs GFS ramenes
au relief de la grille ; seul le vent des points d'entree compte pour le modele (les autres
cellules sont masquees).
"""
from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .forcage import SCALAIRES, ForcageERA5
from .geometry import N_SURF

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
MODELE_GFS = "gfs_global"
OGIMET = "https://www.ogimet.com/cgi-bin/getsynop"
URL_METAR = "https://aviationweather.gov/api/data/metar"
# les niveaux du fichier GFS principal de la NOAA (pgrb2.0p25), ceux de l'affinage OS7 (pcfno3d.gfs)
NIVEAUX_P = (1000, 975, 950, 925, 900, 850, 800, 750, 700)
G, R_AIR, KAPPA, LAPSE = 9.80665, 287.05, 0.2857, 0.0065
NOEUD = 0.514444
SYNOP_CODES = {"60729": "INM60729", "60728": "INM60728"}
UA = {"User-Agent": "pcfno3d-atom-wind/1.0 (PFE Tukhnanutha)"}
FENETRE = pd.Timedelta(minutes=30)


def _contexte_tls():
    """Verification TLS avec le magasin de certificats du SYSTEME (truststore) si disponible,
    sinon certifi, sinon celui de Python : un Python au magasin perime (ines-pc) echouait."""
    import ssl
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_TLS = None


def _get(url: str, timeout: float = 30.0) -> bytes:
    global _TLS
    if _TLS is None:
        _TLS = _contexte_tls()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=_TLS) as r:
        return r.read()


# ---------------------------------------------------------------------- GFS (Open-Meteo)
def _vars_gfs() -> list[str]:
    v = ["wind_speed_10m", "wind_direction_10m", "wind_speed_100m", "wind_direction_100m",
         "temperature_2m", "relative_humidity_2m", "surface_pressure", "pressure_msl",
         "shortwave_radiation"]
    for p in NIVEAUX_P:
        v += [f"wind_speed_{p}hPa", f"wind_direction_{p}hPa", f"geopotential_height_{p}hPa"]
    return v


def telecharger_gfs(lat: np.ndarray, lon: np.ndarray, debut: str, fin: str,
                    lire=None) -> dict:
    """Champs horaires GFS aux mailles (lat x lon) de `debut` a `fin` (AAAA-MM-JJ, UTC).
    Renvoie {"temps": int64 s, "elevation": (ny, nx), <variable>: (T, ny, nx)}."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    q = {"latitude": ",".join(f"{x:.4f}" for x in la.ravel()),
         "longitude": ",".join(f"{x:.4f}" for x in lo.ravel()),
         "hourly": ",".join(_vars_gfs()), "models": MODELE_GFS, "wind_speed_unit": "ms",
         "timezone": "GMT", "cell_selection": "nearest", "start_date": debut, "end_date": fin}
    d = json.loads((lire or _get)(f"{OPEN_METEO}?{urllib.parse.urlencode(q)}"))
    d = d if isinstance(d, list) else [d]
    if len(d) != la.size:
        raise ValueError(f"Open-Meteo : {len(d)} points recus pour {la.size} mailles")
    temps = pd.to_datetime(d[0]["hourly"]["time"]).values.astype("datetime64[s]").astype(np.int64)
    out = {"temps": temps,
           "elevation": np.array([p.get("elevation", 0.0) for p in d], float).reshape(la.shape)}
    for v in _vars_gfs():
        a = np.array([[np.nan if x is None else x for x in p["hourly"][v]] for p in d], float)
        out[v] = a.T.reshape((len(temps),) + la.shape)
    return out


def _uv(vitesse, direction):
    r = np.radians(direction)
    return -vitesse * np.sin(r), -vitesse * np.cos(r)


def forcage_gfs(c: dict, hauteurs_sol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalaires (T, 7) MIS A L'ECHELLE comme ForcageERA5, et u, v (T, nh, ny, nx) par maille aux
    hauteurs au-dessus du sol : les formules de scripts/forcage_era5.py."""
    from .tunisie import interpoler_hauteurs

    def niv(nom, p):
        return c[f"{nom}_{p}hPa"]

    niveaux = [p for p in NIVEAUX_P if f"wind_speed_{p}hPa" in c]

    u850, v850 = _uv(niv("wind_speed", 850), niv("wind_direction", 850))
    u700, v700 = _uv(niv("wind_speed", 700), niv("wind_direction", 700))
    u10, v10 = _uv(c["wind_speed_10m"], c["wind_direction_10m"])
    u100, v100 = _uv(c["wind_speed_100m"], c["wind_direction_100m"])
    m = lambda a: np.nanmean(a, axis=(1, 2))                         # noqa: E731
    dz = m(niv("geopotential_height", 700) - niv("geopotential_height", 850))
    ug, vg = m(u850), m(v850)
    u10m, v10m = m(u10), m(v10)
    veer = (np.degrees(np.arctan2(v10m, u10m)) - np.degrees(np.arctan2(vg, ug)) + 180.0) % 360.0 - 180.0
    veer = np.where(np.hypot(u10m, v10m) > 1.0, np.clip(veer, 0.0, 45.0), 25.0)
    tv = (niv("geopotential_height", 925) - niv("geopotential_height", 1000)) * G \
        / (R_AIR * math.log(1000.0 / 925.0))
    th_couche = m(tv * (1000.0 / 962.5) ** KAPPA)
    t2k = c["temperature_2m"] + 273.15
    th_2m = m(t2k * (1000.0 / c["surface_pressure"]) ** KAPPA)       # hPa / hPa
    stab = np.tanh((th_couche - 1.5 - th_2m) / 3.0)
    # masque terre-mer du modele (GFS : LAND) s'il est fourni, sinon l'altitude du point
    terre = np.asarray(c["terre"], bool) if "terre" in c else c["elevation"] > 1.0
    if terre.any() and (~terre).any():
        dt_tm = np.nanmean(c["temperature_2m"][:, terre], 1) - np.nanmean(c["temperature_2m"][:, ~terre], 1)
    else:
        dt_tm = np.zeros(len(ug))
    brut = {"ug": ug, "vg": vg, "shear_u": (u700 - u850).mean((1, 2)) / dz,
            "shear_v": (v700 - v850).mean((1, 2)) / dz, "veer_deg": veer, "stabilite": stab,
            "dt_terre_mer_k": dt_tm}
    scal = np.stack([brut[k] * f for k, f in SCALAIRES], 1)

    oro = c["elevation"]
    T = len(ug)
    zl = np.stack([niv("geopotential_height", p) for p in niveaux])        # (P, T, y, x) ASL
    h = np.concatenate([np.broadcast_to(oro + 10.0, (T,) + oro.shape)[None],
                        np.broadcast_to(oro + 100.0, (T,) + oro.shape)[None],
                        np.where(zl > oro[None, None] + 100.0, zl, np.nan)])
    champs = []
    for a10, a100, comp in ((u10, u100, 0), (v10, v100, 1)):
        lv = [_uv(niv("wind_speed", p), niv("wind_direction", p))[comp] for p in niveaux]
        val = np.concatenate([a10[None], a100[None], np.stack(lv)])
        champs.append(interpoler_hauteurs(h, val, [oro + hh for hh in hauteurs_sol])
                      .transpose(1, 0, 2, 3))                               # (T, nh, y, x)
    return c["temps"], scal, champs[0], champs[1]


class ForcageDirect(ForcageERA5):
    """Meme interface que ForcageERA5 (vecteur, forme_cellules, n_extra), sur les champs GFS."""

    def __init__(self, temps, scal, u, v, modele: ForcageERA5) -> None:      # noqa: D107
        self.t_csv = self.temps = np.asarray(temps, np.int64)
        self.scal = np.asarray(scal, np.float64)
        self.lat, self.lon, self.hauteurs = modele.lat, modele.lon, modele.hauteurs
        # une maille sans valeur (hauteur hors du profil) : la valeur de la hauteur voisine
        self.u = _boucher(np.asarray(u, np.float32))
        self.v = _boucher(np.asarray(v, np.float32))
        if self.u.shape[1:] != modele.u.shape[1:]:
            raise ValueError(f"forcage GFS {self.u.shape[1:]} != mailles du modele {modele.u.shape[1:]}")
        if not np.isfinite(self.scal).all():
            raise ValueError("forcage GFS : scalaires incomplets")
        self.forme_cellules = (2,) + self.u.shape[1:]
        self.n_cellules = int(np.prod(self.forme_cellules))


def _boucher(a: np.ndarray) -> np.ndarray:
    """NaN en hauteur (T, nh, y, x) : valeur valide la plus proche dans la colonne."""
    a = a.copy()
    for k in range(1, a.shape[1]):
        a[:, k] = np.where(np.isfinite(a[:, k]), a[:, k], a[:, k - 1])
    for k in range(a.shape[1] - 2, -1, -1):
        a[:, k] = np.where(np.isfinite(a[:, k]), a[:, k], a[:, k + 1])
    return np.nan_to_num(a)


# ---------------------------------------------------------------------- stations
def metar_dtta(heures: int = 30, lire=None) -> pd.DataFrame:
    corps = (lire or _get)(f"{URL_METAR}?ids=DTTA&format=json&hours={int(heures)}")
    lignes = []
    for o in (json.loads(corps) if corps.strip() else []):
        t = o.get("reportTime") or o.get("obsTime")
        if t is None or o.get("wspd") is None:
            continue
        t = (pd.Timestamp(t, unit="s") if isinstance(t, (int, float)) else
             pd.Timestamp(t).tz_convert("UTC").tz_localize(None))
        ff = float(o["wspd"]) * NOEUD
        if ff == 0.0:
            u = v = 0.0
        elif isinstance(o.get("wdir"), (int, float)):
            u, v = _uv(ff, float(o["wdir"]))
        else:
            continue                                   # direction variable : inconnue
        lignes.append({"station_id": "DTTA", "t": t, "wind_u_m_s": float(u), "wind_v_m_s": float(v)})
    return pd.DataFrame(lignes, columns=["station_id", "t", "wind_u_m_s", "wind_v_m_s"])


def lire_synop(texte: str) -> pd.DataFrame:
    """Lignes OGIMET « IIiii,AAAA,MM,JJ,HH,mm,AAXX ... » -> vent (u, v) des stations connues.
    Groupe Nddff (et 00fff si ff >= 99) ; iw (dernier chiffre de YYGGi) : 0-1 m/s, 3-4 noeuds."""
    lignes = []
    for ligne in texte.splitlines():
        champs = ligne.strip().split(",")
        if len(champs) < 7 or champs[0] not in SYNOP_CODES:
            continue
        try:
            t = pd.Timestamp(int(champs[1]), int(champs[2]), int(champs[3]), int(champs[4]), int(champs[5]))
            g = champs[6].replace("=", "").split()
            i = g.index("AAXX")
            iw = g[i + 1][4]
            n_ddff = g[i + 4]                     # AAXX YYGGi IIiii iRiXhVV Nddff
            dd, ff = n_ddff[1:3], n_ddff[3:5]
            if "/" in dd or "/" in ff:
                continue
            ff = int(ff)
            if ff == 99:
                ff = int(g[i + 5][2:5])
        except (ValueError, IndexError):
            continue
        if iw not in "0134":
            continue
        vitesse = ff * (NOEUD if iw in "34" else 1.0)
        dd = int(dd)
        if dd == 0 or vitesse == 0:
            u = v = 0.0
        elif dd == 99 or dd > 36:
            continue                               # variable : direction inconnue
        else:
            u, v = _uv(vitesse, dd * 10.0)
        lignes.append({"station_id": SYNOP_CODES[champs[0]], "t": t,
                       "wind_u_m_s": float(u), "wind_v_m_s": float(v)})
    return pd.DataFrame(lignes, columns=["station_id", "t", "wind_u_m_s", "wind_v_m_s"])


def synop_ogimet(debut: datetime, fin: datetime, lire=None) -> pd.DataFrame:
    lire = lire or _get
    tables = []
    for bloc in SYNOP_CODES:
        url = (f"{OGIMET}?block={bloc}&begin={debut:%Y%m%d%H%M}&end={fin:%Y%m%d%H%M}")
        try:
            tables.append(lire_synop(lire(url).decode("utf-8", "replace")))
        except Exception:                          # noqa: BLE001  (une station muette n'arrete rien)
            continue
    return pd.concat(tables, ignore_index=True) if tables else \
        pd.DataFrame(columns=["station_id", "t", "wind_u_m_s", "wind_v_m_s"])


def mesures_a(obs: pd.DataFrame, t: pd.Timestamp) -> dict[str, tuple[float, float]]:
    """{code: (u, v)} : la mesure la plus proche de t, a 30 min au plus (comme obs_a)."""
    if obs is None or obs.empty:
        return {}
    fen = obs[(obs["t"] >= t - FENETRE) & (obs["t"] <= t + FENETRE)]
    out = {}
    for code, g in fen.groupby("station_id"):
        r = g.iloc[int(np.argmin(np.abs((g["t"] - t).to_numpy())))]
        out[code] = (float(r["wind_u_m_s"]), float(r["wind_v_m_s"]))
    return out


# ---------------------------------------------------------------------- entree du modele
def _bil(champ: np.ndarray, interp) -> np.ndarray:
    """(ny_e, nx_e) -> (ny, nx) aux colonnes du modele (memes poids 2D que InterpERA5)."""
    iy, ix, a, b = interp.iy, interp.ix, interp.a, interp.b
    return ((1 - a) * (1 - b) * champ[iy, ix] + (1 - a) * b * champ[iy, ix + 1]
            + a * (1 - b) * champ[iy + 1, ix] + a * b * champ[iy + 1, ix + 1])


def entree(geom, c: dict, f: ForcageDirect, it: int, mesures: dict) -> np.ndarray:
    """S (8, ny, nx) de l'instant it des champs GFS : GFS 10 m (a la place d'ERA5) sur le reseau regulier, les
    stations mesurees, la surface (T, rho, HR, p) ramenee au relief de la grille."""
    from .stations_reelles import cellules_codes, entree_reelle

    interp = geom.interp_era5
    bil = lambda ch: _bil(ch, interp)             # noqa: E731
    E = np.stack([bil(f.u[it, 0]), bil(f.v[it, 0])]).astype(np.float32)     # 10 m sol
    zt = geom.zt
    t_mer = bil(c["temperature_2m"][it] + LAPSE * c["elevation"])
    t_c = t_mer - LAPSE * zt
    tk = t_c + 273.15
    p = bil(c["pressure_msl"][it] * 100.0) * np.exp(-G * zt / (R_AIR * tk))
    S = np.zeros((N_SURF + 1,) + zt.shape, np.float32)
    S[0], S[1] = E
    S[3] = t_c
    S[4] = p / (R_AIR * tk)
    S[5] = np.clip(bil(c["relative_humidity_2m"][it]), 1.0, 100.0)
    S[6] = p
    S_in, capteur = entree_reelle(S, E, geom, cellules_codes(geom), mesures)
    S_in[7] = capteur
    return S_in


class SourceDirect:
    """Les champs GFS et les mesures d'une journee, rafraichis au plus toutes les `ttl` s."""

    def __init__(self, geom, ttl_s: float = 1800.0, lire=None) -> None:
        if geom.forcage is None:
            raise ValueError("temps reel : modele sans forcage ERA5 (paths.forcage_era5)")
        self.geom, self.ttl, self.lire = geom, ttl_s, lire
        self._cache: dict = {}
        self._cache_gefs: dict = {}

    def journee(self, jour_utc: str) -> dict:
        """{"forcage": ForcageDirect, "champs": c, "obs": DataFrame, "obtenu": epoch}."""
        e = self._cache.get(jour_utc)
        if e is not None and time.time() - e["obtenu"] < self.ttl:
            return e
        j = pd.Timestamp(jour_utc)
        debut, fin = (j - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), (j + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        g = self.geom
        c = telecharger_gfs(g.forcage.lat, g.forcage.lon, debut, fin, lire=self.lire)
        temps, scal, u, v = forcage_gfs(c, g.forcage.hauteurs)
        f = ForcageDirect(temps, scal, u, v, g.forcage)
        maintenant = datetime.now(timezone.utc).replace(tzinfo=None)
        d0 = j.to_pydatetime() - timedelta(hours=3)
        d1 = min(j.to_pydatetime() + timedelta(days=1, hours=1), maintenant + timedelta(hours=1))
        obs = []
        try:
            obs.append(metar_dtta(heures=int((maintenant - d0).total_seconds() // 3600) + 2,
                                  lire=self.lire))
        except Exception:                          # noqa: BLE001
            pass
        if d1 > d0:
            obs.append(synop_ogimet(d0, d1, lire=self.lire))
        obs = pd.concat(obs, ignore_index=True) if obs else pd.DataFrame(
            columns=["station_id", "t", "wind_u_m_s", "wind_v_m_s"])
        e = {"forcage": f, "champs": c, "obs": obs, "obtenu": time.time()}
        self._cache[jour_utc] = e
        while len(self._cache) > 3:                # aujourd'hui, demain, apres-demain
            self._cache.pop(next(iter(self._cache)))
        return e

    def ensemble(self, jour_utc: str) -> dict:
        """Les membres GEFS (vent a 10 m) autour de la journee, rafraichis toutes les 6 h."""
        e = self._cache_gefs.get(jour_utc)
        if e is not None and time.time() - e["obtenu"] < max(self.ttl, 6 * 3600):
            return e
        j = pd.Timestamp(jour_utc)
        g = self.geom
        e = telecharger_gefs(g.forcage.lat, g.forcage.lon, (j - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                             (j + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), lire=self.lire)
        e["obtenu"] = time.time()
        self._cache_gefs = {jour_utc: e, **{k: v for k, v in self._cache_gefs.items() if k != jour_utc}}
        while len(self._cache_gefs) > 3:
            self._cache_gefs.pop(list(self._cache_gefs)[-1])
        return e


# ---------------------------------------------------------------------- ensemble GEFS (E19)
URL_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODELE_GEFS = "gfs025"
N_MEMBRES_GEFS = 31                     # controle + 30 membres perturbes


def telecharger_gefs(lat, lon, debut: str, fin: str, lire=None) -> dict:
    """Vent a 10 m des 31 membres GEFS 0,25 deg (Open-Meteo, seules les variables de surface y sont)
    aux mailles : {"temps", "u10": (M, T, ny, nx), "v10": (M, T, ny, nx)}."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    q = {"latitude": ",".join(f"{x:.4f}" for x in la.ravel()),
         "longitude": ",".join(f"{x:.4f}" for x in lo.ravel()),
         "hourly": "wind_speed_10m,wind_direction_10m", "models": MODELE_GEFS, "wind_speed_unit": "ms",
         "timezone": "GMT", "cell_selection": "nearest", "start_date": debut, "end_date": fin}
    d = json.loads((lire or _get)(f"{URL_ENSEMBLE}?{urllib.parse.urlencode(q)}"))
    d = d if isinstance(d, list) else [d]
    if len(d) != la.size:
        raise ValueError(f"Open-Meteo ensemble : {len(d)} points recus pour {la.size} mailles")
    temps = pd.to_datetime(d[0]["hourly"]["time"]).values.astype("datetime64[s]").astype(np.int64)
    suffixes = [""] + [f"_member{m:02d}" for m in range(1, N_MEMBRES_GEFS)]
    u = np.full((len(suffixes), len(temps)) + la.shape, np.nan)
    v = np.full_like(u, np.nan)
    for m, suf in enumerate(suffixes):
        for k, p in enumerate(d):
            h = p["hourly"]
            if f"wind_speed_10m{suf}" not in h:
                continue
            s = np.array([np.nan if x is None else x for x in h[f"wind_speed_10m{suf}"]], float)
            di = np.array([np.nan if x is None else x for x in h[f"wind_direction_10m{suf}"]], float)
            uu, vv = _uv(s, di)
            iy, ix = np.unravel_index(k, la.shape)
            u[m, :, iy, ix], v[m, :, iy, ix] = uu, vv
    garde = np.isfinite(u).all(axis=(1, 2, 3))
    if not garde.any():
        raise ValueError("Open-Meteo ensemble : aucun membre complet")
    return {"temps": temps, "u10": u[garde], "v10": v[garde]}


def forcage_membre(f: ForcageDirect, it: int, du10: np.ndarray, dv10: np.ndarray,
                   echelle_m: float = 300.0) -> ForcageDirect:
    """Le forcage deterministe GFS dont le profil par maille, a l'instant it, est decale de l'ecart
    du membre au vent a 10 m, attenue avec la hauteur (exp(-(h - 10) / echelle)) : ensemble PARTIEL
    (Open-Meteo ne fournit pas les niveaux de pression des membres GEFS)."""
    g = ForcageDirect.__new__(ForcageDirect)
    g.__dict__.update(f.__dict__)
    poids = np.exp(-np.maximum(np.asarray(f.hauteurs, float) - 10.0, 0.0) / echelle_m)[:, None, None]
    g.u, g.v = f.u.copy(), f.v.copy()
    g.u[it] = f.u[it] + poids * du10[None]
    g.v[it] = f.v[it] + poids * dv10[None]
    return g
