"""Modele national « Tunisie ERA5 » : METAR des aeroports -> vent 3D sur toute la Tunisie.

Un modele DISTINCT de PC-FNO3D v7 (domaine de 50 x 25 km au pas de 100 m, relief synthetique) :
    entree  : les METAR des 12 aeroports tunisiens actifs a un instant (vent u, v, temperature,
              pression a la station, humidite, presence), plus l'heure et le jour de l'annee ;
    sortie  : le vent horizontal (u, v) sur la grille ERA5 de la boite Tunisie (0,25 deg,
              ~28 km ; 30,0-37,75 N, 7,25-11,75 E), a 10, 100, 250, 500, 750 et 1000 m
              au-dessus du sol ;
    verite  : ERA5 (Copernicus) : vent a 10 et 100 m des niveaux simples, niveaux de pression
              1000-700 hPa (fichiers era5_pl_* : 1000-850 ; era5_plh_* : 800-700, ~3 000 m)
              interpoles lineairement en hauteur au-dessus du sol (geopotentiel, orographie
              ERA5). Au-dessus du plus haut niveau disponible la cible est ABSENTE (NaN),
              jamais extrapolee ; avec 700 hPa, 1000 m au-dessus du sol est couvert partout.
    horizon : 0 h = reconstruction de l'instant des mesures ; 3 h, 6 h... = prevision du champ
              a l'instant des mesures + horizon (un modele par horizon).
C'est le champ ERA5 qu'on apprend a reconstruire depuis les seules mesures disponibles en temps
reel : sa resolution (~28 km) est celle d'ERA5, pas celle de PC-FNO3D. Decoupage temporel :
2021-2023 entrainement, 2024 validation, 2025 test.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

G = 9.80665
HAUTEURS_AGL = (10.0, 100.0, 250.0, 500.0, 750.0, 1000.0)
VARIABLES_STATION = ("u", "v", "temperature_c", "pression_hpa", "humidite_pct", "presente")
ANNEES = {"entrainement": (2021, 2022, 2023), "validation": (2024,), "test": (2025,)}


# --------------------------------------------------------------------------- ERA5 -> cible
def interpoler_hauteurs(h: np.ndarray, val: np.ndarray, cibles) -> np.ndarray:
    """Interpolation lineaire en hauteur, colonne par colonne, sans extrapolation.

    h, val : (n, ...) hauteurs (m au-dessus du sol) et valeurs ; NaN = point absent. Les
    hauteurs valides n'ont pas a etre consecutives (niveaux sous le relief au milieu).
    Renvoie (len(cibles), ...) ; NaN hors de [plus bas, plus haut] point valide."""
    h = np.asarray(h, np.float64)
    val = np.asarray(val, np.float64)
    ok = np.isfinite(h) & np.isfinite(val)
    out = []
    for c in cibles:
        bas = np.where(ok & (h <= c), h, -np.inf)
        haut = np.where(ok & (h >= c), h, np.inf)
        ib, ih = bas.argmax(0)[None], haut.argmin(0)[None]
        hb = np.take_along_axis(bas, ib, 0)[0]
        hh = np.take_along_axis(haut, ih, 0)[0]
        vb = np.take_along_axis(val, ib, 0)[0]
        vh = np.take_along_axis(val, ih, 0)[0]
        poids = np.where(hh > hb, (c - hb) / np.where(hh > hb, hh - hb, 1.0), 0.0)
        r = vb + poids * (vh - vb)
        out.append(np.where(np.isfinite(hb) & np.isfinite(hh), r, np.nan))
    return np.stack(out).astype(np.float32)


def _paires_mensuelles(dossier: Path) -> list[tuple[Path, Path, Path | None]]:
    """Fichiers era5_sl_* / era5_pl_* (+ era5_plh_*, niveaux hauts, s'il existe) de
    download_reanalysis.py, apparies par periode."""
    paires = []
    for sl in sorted(Path(dossier).glob("era5_sl_*.nc")):
        pl = sl.with_name(sl.name.replace("era5_sl_", "era5_pl_"))
        plh = sl.with_name(sl.name.replace("era5_sl_", "era5_plh_"))
        if pl.exists():
            paires.append((sl, pl, plh if plh.exists() else None))
    return paires


def cible_era5(sl, pl) -> dict:
    """Deux jeux xarray (niveaux simples, niveaux de pression) d'une meme periode -> cible :
    temps, lat, lon, orographie (m), U et V (t, h, y, x) aux HAUTEURS_AGL."""
    tdim = "valid_time" if "valid_time" in sl.dims else "time"
    pdim = "pressure_level" if "pressure_level" in pl.dims else "level"
    pl = pl.sortby(pdim, ascending=False)                    # 1000 hPa (bas) -> 850 hPa (haut)
    orog = sl["z"].isel({tdim: 0}).values / G if tdim in sl["z"].dims else sl["z"].values / G
    zpl = pl["z"].values / G                                 # (t, p, y, x), m au-dessus de la mer
    agl = zpl - orog[None, None]
    agl = np.where(agl > 100.0, agl, np.nan)                 # 10 et 100 m viennent des niveaux simples
    t = sl[tdim].values.shape[0]
    hs = np.concatenate([np.full((t, 1) + orog.shape, 10.0), np.full((t, 1) + orog.shape, 100.0),
                         agl], axis=1)
    sortie = {}
    for comp, (c10, c100) in (("u", ("u10", "u100")), ("v", ("v10", "v100"))):
        vals = np.concatenate([sl[c10].values[:, None], sl[c100].values[:, None],
                               pl[comp].values], axis=1)
        sortie[comp] = np.moveaxis(interpoler_hauteurs(np.moveaxis(hs, 1, 0),
                                                       np.moveaxis(vals, 1, 0), HAUTEURS_AGL), 0, 1)
    return {"temps": sl[tdim].values.astype("datetime64[s]"), "lat": sl["latitude"].values,
            "lon": sl["longitude"].values, "orographie_m": orog.astype(np.float32),
            "U": sortie["u"], "V": sortie["v"]}


def lire_era5(dossier: Path) -> dict:
    """Toutes les periodes telechargees -> cible concatenee dans le temps (sans doublon)."""
    import xarray as xr

    morceaux = []
    for sl_f, pl_f, plh_f in _paires_mensuelles(dossier):
        with xr.open_dataset(sl_f) as sl, xr.open_dataset(pl_f) as pl:
            pl = pl[["u", "v", "z"]].load()
            if plh_f is not None:
                with xr.open_dataset(plh_f) as plh:
                    pdim = "pressure_level" if "pressure_level" in pl.dims else "level"
                    pl = xr.concat([pl, plh[["u", "v", "z"]].load()], dim=pdim)
            morceaux.append(cible_era5(sl.load(), pl))
    if not morceaux:
        raise FileNotFoundError(f"aucune paire era5_sl_* / era5_pl_* dans {dossier}")
    temps = np.concatenate([m["temps"] for m in morceaux])
    ordre = np.unique(temps, return_index=True)[1]
    return {"temps": temps[ordre], "lat": morceaux[0]["lat"], "lon": morceaux[0]["lon"],
            "orographie_m": morceaux[0]["orographie_m"],
            "U": np.concatenate([m["U"] for m in morceaux])[ordre],
            "V": np.concatenate([m["V"] for m in morceaux])[ordre]}


# --------------------------------------------------------------------------- METAR -> entree
def lire_metar(fichiers) -> "pd.DataFrame":
    """Archives CSV de pcfno3d.metar -> u, v (direction meteo convertie), T, p (hPa), HR."""
    import pandas as pd

    df = pd.concat([pd.read_csv(f) for f in fichiers], ignore_index=True)
    df["t"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_localize(None)
    s, d = df["wind_speed_m_s"], np.radians(df["wind_dir_deg"])
    u = df["wind_u_m_s"].where(df["wind_u_m_s"].notna(), -s * np.sin(d))
    v = df["wind_v_m_s"].where(df["wind_v_m_s"].notna(), -s * np.cos(d))
    return pd.DataFrame({"station": df["station_id"], "t": df["t"], "u": u, "v": v,
                         "temperature_c": df["temperature_c"],
                         "pression_hpa": df["pressure_pa"] / 100.0,
                         "humidite_pct": df["relative_humidity_pct"]}).dropna(subset=["u", "v"])


def entree_stations(metar, temps, stations, tolerance_min: int = 30) -> np.ndarray:
    """(t, S, F) : pour chaque instant et chaque station, la mesure la plus proche a +/-
    `tolerance_min` minutes ; absente -> NaN et presente = 0. F = VARIABLES_STATION."""
    import pandas as pd

    t_ns = np.asarray(temps).astype("datetime64[ns]")
    cible = pd.DataFrame({"t": t_ns}).sort_values("t")
    X = np.full((len(temps), len(stations), len(VARIABLES_STATION)), np.nan, np.float32)
    rang = np.argsort(np.argsort(t_ns, kind="stable"), kind="stable")
    metar = metar.assign(t=metar["t"].values.astype("datetime64[ns]"))
    for k, st in enumerate(stations):
        m = metar[metar["station"] == st].sort_values("t")
        if m.empty:
            X[:, k, -1] = 0.0
            continue
        j = pd.merge_asof(cible, m, on="t", direction="nearest",
                          tolerance=pd.Timedelta(minutes=tolerance_min))
        vals = j[list(VARIABLES_STATION[:-1])].to_numpy(np.float32)[rang]
        X[:, k, :-1] = vals
        X[:, k, -1] = np.isfinite(vals[:, 0]).astype(np.float32)
    return X


def variables_temps(temps) -> np.ndarray:
    """(t, 4) : heure et jour de l'annee en sinus / cosinus."""
    import pandas as pd

    t = pd.to_datetime(np.asarray(temps))
    h = 2 * np.pi * (t.hour + t.minute / 60.0) / 24.0
    d = 2 * np.pi * (t.dayofyear - 1) / 365.25
    return np.stack([np.sin(h), np.cos(h), np.sin(d), np.cos(d)], 1).astype(np.float32)


# --------------------------------------------------------------------------- modele
@dataclass
class Normalisation:
    x_moy: np.ndarray       # (F-1,) variables mesurees (sans « presente »)
    x_ect: np.ndarray
    y_moy: np.ndarray       # (C,) par canal de sortie (u, v par hauteur)
    y_ect: np.ndarray

    @classmethod
    def depuis(cls, X: np.ndarray, Y: np.ndarray) -> "Normalisation":
        xm = X[..., :-1].reshape(-1, X.shape[-1] - 1)
        ym = np.moveaxis(Y, 1, -1).reshape(-1, Y.shape[1])
        return cls(np.nanmean(xm, 0), np.nanstd(xm, 0) + 1e-6,
                   np.nanmean(ym, 0), np.nanstd(ym, 0) + 1e-6)

    def entree(self, X: np.ndarray) -> np.ndarray:
        """NaN -> 0 apres normalisation (= valeur moyenne) ; « presente » dit ce qui manque."""
        z = (X[..., :-1] - self.x_moy) / self.x_ect
        z = np.where(np.isfinite(z) & (X[..., -1:] > 0), z, 0.0)
        return np.concatenate([z, X[..., -1:]], -1).astype(np.float32)

    def etat(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("x_moy", "x_ect", "y_moy", "y_ect")}

    @classmethod
    def charger(cls, d: dict) -> "Normalisation":
        return cls(*(np.asarray(d[k], np.float32) for k in ("x_moy", "x_ect", "y_moy", "y_ect")))


def construire_modele(n_stations: int, n_var: int, ny: int, nx: int, n_sortie: int,
                      latent: int = 256, base: int = 64):
    """Encodeur MLP des stations (+ temps) -> decodeur convolutif vers la grille, plus un champ
    de biais appris (le « fond » climatologique)."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    h0, w0 = math.ceil(ny / 4), math.ceil(nx / 4)

    class StationsVersChamp(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(n_stations * n_var + 4, 512), nn.GELU(),
                                     nn.Linear(512, latent), nn.GELU())
            self.proj = nn.Linear(latent, base * h0 * w0)
            self.dec = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(base, base, 3, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(base, base, 3, padding=1),
                nn.GELU())
            self.tete = nn.Conv2d(base, n_sortie, 3, padding=1)
            self.fond = nn.Parameter(torch.zeros(n_sortie, ny, nx))

        def forward(self, x, tv):
            z = self.enc(torch.cat([x.flatten(1), tv], 1))
            g = self.dec(self.proj(z).view(-1, base, h0, w0))
            g = F.interpolate(g, size=(ny, nx), mode="bilinear", align_corners=False)
            return self.tete(g) + self.fond

    return StationsVersChamp()


def perte_masquee(pred, cible):
    """MSE sur les seules cibles connues (NaN = au-dessus du plus haut niveau ERA5)."""
    import torch

    ok = torch.isfinite(cible)
    return ((pred - torch.nan_to_num(cible)) ** 2)[ok].mean()


def masquer_stations(x, p_max: float = 0.5, generateur=None):
    """Augmentation : chaque exemple perd une part tiree dans [0, p_max] de ses stations
    (capteur en panne) : variables a 0, presente = 0."""
    import torch

    b, s, _ = x.shape
    p = torch.rand(b, 1, generator=generateur) * p_max
    garde = (torch.rand(b, s, generator=generateur) >= p).float().unsqueeze(-1)
    x = x * garde
    return x


# --------------------------------------------------------------------------- references
def rmse_par_niveau(pred: np.ndarray, Y: np.ndarray, masque: np.ndarray | None = None) -> dict:
    """RMSE (m/s) de u, v et de la vitesse par hauteur ; pred, Y : (t, 2H, y, x) avec les
    canaux [u_h0..u_h5, v_h0..v_h5] ; masque (y, x) des mailles comptees."""
    H = len(HAUTEURS_AGL)
    m = np.ones(Y.shape[-2:], bool) if masque is None else masque
    out = {}
    for k, h in enumerate(HAUTEURS_AGL):
        pu, pv, yu, yv = pred[:, k], pred[:, H + k], Y[:, k], Y[:, H + k]
        ok = np.isfinite(yu) & np.isfinite(yv) & m[None]
        eu, ev = (pu - yu)[ok], (pv - yv)[ok]
        es = (np.hypot(pu, pv) - np.hypot(yu, yv))[ok]
        out[f"{int(h)} m"] = {"u": float(np.sqrt(np.mean(eu ** 2))), "v": float(np.sqrt(np.mean(ev ** 2))),
                              "vitesse": float(np.sqrt(np.mean(es ** 2))), "n": int(ok.sum())}
    return out


def reference_climatologie(Y_train, temps_train, temps) -> np.ndarray:
    """Moyenne ERA5 de l'entrainement par heure UTC et par maille (aucune mesure utilisee)."""
    import pandas as pd

    ht = pd.to_datetime(np.asarray(temps_train)).hour.values
    hq = pd.to_datetime(np.asarray(temps)).hour.values
    moy = {h: np.nanmean(Y_train[ht == h], 0) for h in np.unique(ht)}
    return np.stack([moy[h] for h in hq]).astype(np.float32)


def reference_idw(X: np.ndarray, st_lat, st_lon, lat, lon, alpha: float = 1 / 7,
                  puissance: float = 2.0) -> np.ndarray:
    """Vent des METAR (10 m) interpole en distance inverse sur la grille, puis loi puissance
    (exposant alpha) vers chaque hauteur : la reference « sans apprentissage » du lot M8."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    cosm = math.cos(math.radians(float(np.mean(lat))))
    d2 = ((la[None] - np.asarray(st_lat)[:, None, None]) ** 2
          + ((lo[None] - np.asarray(st_lon)[:, None, None]) * cosm) ** 2)    # (S, y, x)
    w = 1.0 / np.maximum(d2, 1e-6) ** (puissance / 2)
    pres = (X[..., -1] > 0) & np.isfinite(X[..., 0])                          # (t, S)
    u = np.nan_to_num(X[..., 0]) * pres
    v = np.nan_to_num(X[..., 1]) * pres
    ws = np.einsum("ts,syx->tyx", pres.astype(np.float64), w)
    u10 = np.einsum("ts,syx->tyx", u, w) / np.where(ws > 0, ws, np.nan)
    v10 = np.einsum("ts,syx->tyx", v, w) / np.where(ws > 0, ws, np.nan)
    f = np.array([(h / 10.0) ** alpha for h in HAUTEURS_AGL])
    return np.concatenate([u10[:, None] * f[None, :, None, None],
                           v10[:, None] * f[None, :, None, None]], 1).astype(np.float32)


def paires_horizon(temps, horizon_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices (source, cible) tels que temps[cible] = temps[source] + horizon : mesures a
    l'instant source, champ a l'instant cible (horizon 0 : les memes)."""
    t = np.asarray(temps).astype("datetime64[s]")
    pos = {v: i for i, v in enumerate(t.astype("int64"))}
    dt = int(horizon_h) * 3600
    src = [i for i, v in enumerate(t.astype("int64")) if v + dt in pos]
    return np.array(src, int), np.array([pos[t.astype("int64")[i] + dt] for i in src], int)


def annee_de(temps) -> np.ndarray:
    return np.asarray(temps).astype("datetime64[Y]").astype(int) + 1970


def nom_periode(f: Path) -> str | None:
    m = re.search(r"era5_(?:sl|pl)_(\d{6})", f.name)
    return m.group(1) if m else None
