#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tukhnanutha - Generateur du dataset ANEO 3D (version 2)
=======================================================

Produit un dataset synthetique physiquement coherent pour l'entrainement du PC-FNO3D :

  - aneo_grid_points.csv
        500 000 points de surface (grille 1000 x 500 a 50 m)
        + noeuds virtuels 3D (une colonne sur deux, 12 niveaux ASL de 25 a 1000 m)
  - aneo_observations_day_XXX_YYYYMMDD_multitime.csv.gz (ou .csv avec --csv-brut)
        un fichier par jour, memes colonnes que le dataset actuel
  - split_days.json, meta.json, qc/ (controle qualite par instant + synthese)

Les colonnes des deux fichiers sont identiques a celles du dataset existant.
Les noeuds 3D se distinguent par source_type = ANEO_VIRTUAL_NODE_3D.

Principes physiques :
  - profil de couche limite (log pres du sol, melange, jet nocturne, rotation d'Ekman)
  - vent thermique au-dessus de la couche limite, coherent avec le gradient de temperature
  - hauteur de couche limite variable jour / nuit, stabilite continue dans le temps
  - vents de pente (anabatiques le jour, catabatiques la nuit)
  - perturbations mesoechelle advectees par le vent (coherence minute par minute)
  - w calcule par la continuite : le residu de masse discret est nul par construction
  - thermodynamique coherente (hydrostatique, gaz parfait, Magnus, temperature potentielle)

Utilisation (Colab) :
  python generate_aneo_3d.py --input-grid /content/drive/MyDrive/data_4x/aneo_grid_points.csv \
                             --output-dir /content/aneo_dataset_3d \
                             --drive-dir /content/drive/MyDrive/aneo_dataset_3d \
                             --workers 8

Test rapide (1 jour, 3 instants) :
  python generate_aneo_3d.py --output-dir ./test_3d --days 1 --max-instants 3 --workers 1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

try:
    import pyarrow as pa
    import pyarrow.csv as pacsv
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# grille de surface : 1000 x 500 (50 x 25 km, v7) par defaut ; une grille reelle d'une autre taille
# passe par --nx/--ny, qui fixent aussi ces variables d'environnement pour les processus de calcul
NX_S = int(os.environ.get("ANEO_GEN_NX", 1000))
NY_S = int(os.environ.get("ANEO_GEN_NY", 500))
# pas de surface (m) : 50 m (v7, Grand Tunis) ; --dx pour une autre grille (national : 500 m),
# qui fixe aussi ANEO_GEN_DX pour les processus de calcul
DX_S = float(os.environ.get("ANEO_GEN_DX", 50.0))
STRIDE = 2                      # noeuds 3D : une colonne sur deux
NX_V, NY_V = NX_S // STRIDE, NY_S // STRIDE
DX_V = DX_S * STRIDE            # pas des noeuds 3D (m)

# niveaux des noeuds (m au-dessus de la mer) : ceux de la v7 par defaut ; --niveaux (ANEO_GEN_NIVEAUX)
# pour un relief plus haut (national : jusqu'a 2500 m) ; 100 et 200 m doivent y etre (controles)
Z_LEVELS_V7 = (25., 50., 75., 100., 150., 200., 300., 400., 500., 650., 800., 1000.)
Z_LEVELS = np.array([float(z) for z in os.environ.get("ANEO_GEN_NIVEAUX", "").split(",") if z.strip()]
                    or Z_LEVELS_V7)
NZ = len(Z_LEVELS)
MIN_ZREL_NODE = 5.0             # pas de noeud a moins de 5 m au-dessus du terrain
TERRAIN_DECAY_M = 300.0         # decroissance verticale de l'influence du relief sur le vent
TERRAIN_SMOOTH_M = 300.0        # lissage du relief vu par l'ecoulement
W_DAMPING_M = 300.0             # echelle d'amortissement vertical de w (m)
A_ROT, A_DIV, A_SLOPE = 0.12, 0.03, 1.0     # amplitudes des perturbations (rotationnelle, divergente, pente)
# forcage ERA5 : les perturbations aleatoires de petite echelle sont reduites (facteur), pour que la
# verite ne contienne pas de structures inventees que les stations reelles ne peuvent pas confirmer
PERTURB = float(os.environ.get("ANEO_GEN_PERTURB", 1.0))
A_ROT, A_DIV = A_ROT * PERTURB, A_DIV * PERTURB
# brise de mer / de terre (forcage ERA5 seulement) : profondeur, penetration terre et mer
H_BRISE_M, L_BRISE_TERRE_M, L_BRISE_MER_M = 400.0, 15000.0, 10000.0

N_DAYS_DEFAULT = 100
SYNOPTIC_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]
YEAR = 2025

# Constantes physiques
G_ACC = 9.80665
RD = 287.05
KAPPA = 0.2857
P_REF = 100000.0
F_CORIOLIS = float(os.environ.get("ANEO_GEN_F", 8.57e-5))   # latitude ~36 N (grille reelle : la sienne)
LAPSE = 0.0065

OBS_COLUMNS = [
    "point_id", "timestamp_utc", "day_index", "hour_utc",
    "temperature_c", "potential_temperature_k", "air_density_kg_m3",
    "relative_humidity_pct", "water_vapor_pressure_pa", "pressure_pa",
    "wind_u_m_s", "wind_v_m_s", "wind_w_m_s",
    "aneo_quality_score", "mask_i_obs", "source_type",
]

SRC_SURFACE = "ANEO_PHYSICAL_SYNTHETIC_MULTITIME"
SRC_INTERP = "ANEO_PHYSICAL_INTERPOLATED_MULTITIME"
SRC_NODE = "ANEO_VIRTUAL_NODE_3D"

DECIMALS = {
    "temperature_c": 3, "potential_temperature_k": 3, "air_density_kg_m3": 5,
    "relative_humidity_pct": 2, "water_vapor_pressure_pa": 1, "pressure_pa": 1,
    "wind_u_m_s": 3, "wind_v_m_s": 3, "wind_w_m_s": 4, "aneo_quality_score": 4,
}

# Seuils du controle qualite (cahier des charges)
QC_THRESHOLDS = {
    "mass_residual_mean_sq_max": 1e-6,
    "spatial_corr_surface_min": 0.85,
    "spatial_corr_volume_min": 0.80,
    "temporal_corr_1min_min": 0.95,
    "temporal_corr_3h_min": 0.60,
    "direction_sector_min_share": 0.10,
    "hydrostatic_rel_err_max": 0.03,
}


# =============================================================================
# OUTILS GENERIQUES
# =============================================================================

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gaussian_spectrum_field(shape, dx, dy, corr_len, rng):
    """Coefficients de Fourier d'un champ aleatoire lisse, periodique, de variance ~1."""
    ny, nx = shape
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    k2 = kx[None, :] ** 2 + ky[:, None] ** 2
    amp = np.exp(-0.25 * k2 * (2.0 * np.pi * corr_len) ** 2)
    coeffs = np.fft.fft2(rng.standard_normal(shape)) * amp
    field = np.fft.ifft2(coeffs).real
    coeffs /= max(field.std(), 1e-12)
    coeffs[0, 0] = 0.0
    return coeffs


def spectral_eval(coeffs, kx, ky, shift_x, shift_y, deriv=None):
    """Evalue un champ spectral translate de (shift_x, shift_y) metres, avec derivee optionnelle."""
    phase = np.exp(-2j * np.pi * (kx[None, :] * shift_x + ky[:, None] * shift_y))
    c = coeffs * phase
    if deriv == "x":
        c = c * (2j * np.pi * kx[None, :])
    elif deriv == "y":
        c = c * (2j * np.pi * ky[:, None])
    return np.fft.ifft2(c).real


def magnus_es(t_c):
    """Pression de vapeur saturante (Pa), formule de Magnus."""
    return 611.2 * np.exp(17.62 * t_c / (243.12 + t_c))


def ddx(f, dx, axis):
    """Derivee centree a l'interieur, decentree aux bords (np.gradient)."""
    return np.gradient(f, dx, axis=axis)


def horizontal_divergence(rho_u, rho_v, dx, dy):
    """D = d(rho u)/dx + d(rho v)/dy sur un volume (NZ, NY, NX)."""
    return ddx(rho_u, dx, axis=2) + ddx(rho_v, dy, axis=1)


def mass_residual(rho, u, v, w, z_levels, dx, dy):
    """
    Residu de conservation de masse aux demi-niveaux (k+1/2) :
        r = (rho w[k+1] - rho w[k]) / dz_k + 0.5 * (D[k] + D[k+1])
    C'est l'operateur exact utilise pour construire w : a reprendre tel quel dans la loss.
    """
    d = horizontal_divergence(rho * u, rho * v, dx, dy)
    rw = rho * w
    dz = np.diff(z_levels)[:, None, None]
    return (rw[1:] - rw[:-1]) / dz + 0.5 * (d[1:] + d[:-1])


def lag1_corr(a, axis):
    a = np.asarray(a, dtype=np.float64)
    if axis == 1:
        x, y = a[:, 1:], a[:, :-1]
    else:
        x, y = a[1:, :], a[:-1, :]
    x = x.ravel()
    y = y.ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or x[ok].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def masked_lag1_corr(a, valid):
    """Correlation entre cellules voisines (x et y) en ne gardant que les paires valides."""
    vals = []
    for axis in (1, 0):
        if axis == 1:
            x, y = a[:, 1:], a[:, :-1]
            m = valid[:, 1:] & valid[:, :-1]
        else:
            x, y = a[1:, :], a[:-1, :]
            m = valid[1:, :] & valid[:-1, :]
        if m.sum() > 10 and x[m].std() > 1e-12:
            vals.append(np.corrcoef(x[m], y[m])[0, 1])
    return float(np.mean(vals)) if vals else float("nan")


def met_direction(u, v):
    """Direction meteo (d'ou vient le vent), degres, 0 = Nord, sens horaire."""
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


# =============================================================================
# GRILLE STATIQUE
# =============================================================================

def synthetic_terrain(rng):
    """Relief synthetique (utilise uniquement si aucune grille n'est fournie)."""
    shape = (NY_S, NX_S)
    big = np.fft.ifft2(gaussian_spectrum_field(shape, DX_S, DX_S, 6000.0, rng)).real
    mid = np.fft.ifft2(gaussian_spectrum_field(shape, DX_S, DX_S, 1500.0, rng)).real
    small = np.fft.ifft2(gaussian_spectrum_field(shape, DX_S, DX_S, 400.0, rng)).real
    x = np.linspace(0.0, 1.0, NX_S)[None, :]
    ramp = np.clip(1.15 - 1.1 * x, 0.0, 1.0)            # relief a l'ouest, cote a l'est
    base = 0.55 * big + 0.3 * mid + 0.15 * small
    base = (base - base.min()) / (base.max() - base.min())
    zt = 850.0 * (base ** 1.6) * (0.25 + 0.75 * ramp) + 40.0 * ramp
    sea = (x > 0.93) & (zt < 25.0)
    zt = np.where(sea, 0.0, zt)
    return zt.astype(np.float64), np.broadcast_to(sea, shape).copy()


def build_synthetic_grid(rng):
    zt, sea = synthetic_terrain(rng)
    iy, ix = np.meshgrid(np.arange(NY_S), np.arange(NX_S), indexing="ij")
    x = ix * DX_S
    y = iy * DX_S
    gy, gx = np.gradient(zt, DX_S, DX_S)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.degrees(np.arctan2(-gx, -gy)) % 360.0

    lu_noise = np.fft.ifft2(gaussian_spectrum_field((NY_S, NX_S), DX_S, DX_S, 1200.0, rng)).real
    land_use = np.full((NY_S, NX_S), 2, dtype=np.int64)               # plaine
    land_use = np.where(lu_noise > 0.6, 3, land_use)                    # agricole
    land_use = np.where((lu_noise < -0.9) & (zt > 150), 4, land_use)    # foret
    land_use = np.where((lu_noise > 1.6) & (zt < 120), 5, land_use)     # urbain
    land_use = np.where(sea, 1, land_use)                               # eau
    names = {1: "eau", 2: "plaine", 3: "agricole", 4: "foret", 5: "urbain"}
    z0_by_lu = {1: 0.0002, 2: 0.03, 3: 0.1, 4: 1.0, 5: 1.5}
    z0 = np.vectorize(z0_by_lu.get)(land_use).astype(np.float64)
    zrel = rng.uniform(20.0, 30.0, size=(NY_S, NX_S))

    df = pd.DataFrame({
        "point_id": (iy * NX_S + ix + 1).ravel(),
        "ix": ix.ravel(), "iy": iy.ravel(),
        "x_m": x.ravel().astype(np.float64), "y_m": y.ravel().astype(np.float64),
        "z_terrain_m": np.round(zt.ravel(), 2), "z_rel_m": np.round(zrel.ravel(), 2),
        "z0_m": z0.ravel(), "slope_deg": np.round(slope.ravel(), 3),
        "aspect_deg": np.round(aspect.ravel(), 3),
        "land_use_code": land_use.ravel(),
        "terrain_type": pd.Series(land_use.ravel()).map(names).values,
    })
    df["geom_wkt"] = "POINT (" + df["x_m"].map("{:.1f}".format) + " " + df["y_m"].map("{:.1f}".format) + ")"
    return df


def load_or_build_grid(input_grid: Path | None, rng):
    if input_grid is not None:
        log(f"Lecture de la grille existante : {input_grid}")
        df = pd.read_csv(input_grid)
        df = df[df["point_id"] <= NX_S * NY_S] if "point_id" in df.columns else df
        need = ["point_id", "ix", "iy", "x_m", "y_m", "z_terrain_m", "z_rel_m", "z0_m"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            sys.exit(f"Colonnes manquantes dans la grille : {missing}")
        df = df.sort_values(["iy", "ix"]).reset_index(drop=True)
        if len(df) != NX_S * NY_S:
            sys.exit(f"La grille doit contenir {NX_S * NY_S} points de surface, trouve {len(df)}")
    else:
        log("Aucune grille fournie : generation d'un relief synthetique")
        df = build_synthetic_grid(rng)
    return df


def build_static(df_surface: pd.DataFrame):
    """Tableaux 2D de surface et 3D des noeuds, dans l'ordre (iy, ix)."""
    def g(col):
        return df_surface[col].values.reshape(NY_S, NX_S).astype(np.float64)

    zt = g("z_terrain_m")
    z0 = np.maximum(g("z0_m"), 1e-4)
    zrel = g("z_rel_m")
    x = g("x_m")
    y = g("y_m")
    dx_s = float(x[0, 1] - x[0, 0]) if abs(x[0, 1] - x[0, 0]) > 0 else DX_S
    dy_s = float(y[1, 0] - y[0, 0]) if abs(y[1, 0] - y[0, 0]) > 0 else DX_S
    # les lissages en m (relief vu par l'ecoulement, brise) passent par DX_S : une grille d'un
    # autre pas sans --dx les rendrait faux sans erreur visible
    if abs(dx_s - DX_S) > 1e-6 * DX_S:
        raise ValueError(f"grille au pas {dx_s:g} m, generateur configure a {DX_S:g} m : passer --dx {dx_s:g}")
    point_id = df_surface["point_id"].values.reshape(NY_S, NX_S).astype(np.int64)

    zt_v = zt[::STRIDE, ::STRIDE]
    node_valid = (Z_LEVELS[:, None, None] - zt_v[None, :, :]) >= MIN_ZREL_NODE
    kk, jj, ii = np.nonzero(node_valid)

    geo = {}
    if "lat" in df_surface.columns and "lon" in df_surface.columns:
        geo = {"lat": g("lat"), "lon": g("lon")}
    return {
        **geo,
        "zt": zt, "z0": z0, "zrel": zrel, "x": x, "y": y,
        "dx_s": dx_s, "dy_s": dy_s, "point_id": point_id,
        "node_k": kk.astype(np.int32), "node_j": jj.astype(np.int32), "node_i": ii.astype(np.int32),
        "node_valid": node_valid,
    }


def write_grid_file(df_surface: pd.DataFrame, static: dict, out_path: Path):
    """Grille = points de surface inchanges + noeuds virtuels (memes colonnes)."""
    kk, jj, ii = static["node_k"], static["node_j"], static["node_i"]
    parent_flat = (jj * STRIDE) * NX_S + (ii * STRIDE)
    nodes = df_surface.iloc[parent_flat].copy().reset_index(drop=True)
    n_nodes = len(nodes)
    start_id = int(df_surface["point_id"].max()) + 1
    nodes["point_id"] = np.arange(start_id, start_id + n_nodes, dtype=np.int64)
    z_asl = Z_LEVELS[kk]
    nodes["z_rel_m"] = np.round(z_asl - nodes["z_terrain_m"].values, 2)
    for col in ("z_abs_m", "z_asl_m"):
        if col in nodes.columns:
            nodes[col] = z_asl
    if "geom_wkt" in nodes.columns:
        first = str(df_surface["geom_wkt"].iloc[0]).upper()
        if first.startswith("POINT Z"):
            nodes["geom_wkt"] = ("POINT Z (" + nodes["x_m"].map("{:.2f}".format) + " "
                                 + nodes["y_m"].map("{:.2f}".format) + " "
                                 + pd.Series(z_asl).map("{:.2f}".format) + ")")
    grid_all = pd.concat([df_surface, nodes[df_surface.columns]], ignore_index=True)
    grid_all.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    return start_id, n_nodes


# =============================================================================
# CALENDRIER : JOURS, INSTANTS, EPISODES, SPLIT
# =============================================================================

def build_split(n_days):
    test, val, train = [], [], []
    for i in range(1, n_days + 1):
        r = (i - 1) % 20
        if r in (3, 10, 16):
            test.append(i)
        elif r in (6, 13, 19):
            val.append(i)
        else:
            train.append(i)
    return {"train_days": train, "val_days": val, "test_days": test}


def build_calendar(n_days, split, rng):
    """Jours repartis sur l'annee + episodes minute / 10 s places dans les trois ensembles."""
    days = []
    for i in range(n_days):
        date = dt.date(YEAR, 1, 1) + dt.timedelta(days=int(round(i * 365.0 / n_days)))
        days.append({"day_index": i + 1, "date": date})

    def spread(pool, n):
        if n <= 0 or not pool:
            return []
        idx = np.linspace(0, len(pool) - 1, n).round().astype(int)
        return [pool[j] for j in idx]

    n_min = min(10, n_days)
    minute_days = (spread(split["train_days"], round(n_min * 0.7))
                   + spread(split["test_days"], round(n_min * 0.2))
                   + spread(split["val_days"], n_min - round(n_min * 0.7) - round(n_min * 0.2)))
    sec_days = spread(split["train_days"][3:], 1) + spread(split["test_days"][1:], 1)

    episode_types = ["lever_soleil", "brise", "front", "rafales"]
    episode_start = {"lever_soleil": (5, 30), "brise": (13, 30), "front": (10, 30), "rafales": (16, 30)}

    schedule = {}
    for d in days:
        idx = d["day_index"]
        inst = [(h * 3600, None) for h in SYNOPTIC_HOURS]
        if idx in minute_days:
            etype = episode_types[minute_days.index(idx) % len(episode_types)]
            h0, m0 = episode_start[etype]
            t0 = h0 * 3600 + m0 * 60
            inst += [(t0 + 60 * n, etype) for n in range(18)]
        if idx in sec_days:
            t0 = 11 * 3600 + 15 * 60
            inst += [(t0 + 10 * n, "rafales_10s") for n in range(10)]
        inst = sorted(set(inst), key=lambda t: t[0])
        schedule[idx] = inst
    return days, schedule, minute_days, sec_days


# =============================================================================
# ETAT METEOROLOGIQUE D'UNE JOURNEE
# =============================================================================

class DayState:
    """Parametres synoptiques et champs spectraux d'une journee (deterministes par graine)."""

    def __init__(self, day_index, date, sector, static, seed):
        rng = np.random.default_rng(seed)
        self.rng_obs = np.random.default_rng(seed + 7919)
        self.day_index = day_index
        self.date = date
        self.doy = date.timetuple().tm_yday

        # Vent geostrophique
        cat = rng.random()
        if cat < 0.12:
            self.g_mean = rng.uniform(1.0, 2.5)
        elif cat < 0.80:
            self.g_mean = rng.uniform(4.0, 12.0)
        else:
            self.g_mean = rng.uniform(15.0, 26.0)
        self.dir_mean = (sector * 45.0 + rng.uniform(-15.0, 15.0)) % 360.0
        self.g_phase = rng.uniform(0, 2 * np.pi)
        self.dir_phase = rng.uniform(0, 2 * np.pi)
        self.dir_swing = rng.uniform(10.0, 30.0)

        self.cloud = rng.uniform(0.0, 1.0)
        self.veer = rng.uniform(20.0, 40.0)
        self.jet_day = rng.random() < 0.25
        self.jet_amp = rng.uniform(0.2, 0.5)
        self.jet_z = rng.uniform(150.0, 400.0)
        # Gradient horizontal de temperature de grande echelle (K/m) : il fixe le vent thermique,
        # c'est-a-dire la variation du vent geostrophique avec l'altitude (au lieu d'un tirage
        # arbitraire). Il est visible en surface, donc reconstructible.
        grad_mag = rng.uniform(0.0, 1.2e-5)
        grad_dir = rng.uniform(0.0, 2.0 * math.pi)
        self.dtdx = grad_mag * math.cos(grad_dir)
        self.dtdy = grad_mag * math.sin(grad_dir)

        # Thermodynamique
        self.t_mean = 19.0 + 8.0 * math.sin(2 * math.pi * (self.doy - 110) / 365.0) + rng.normal(0, 1.5)
        # Vent thermique : d(ug, vg)/dz = (g / (f T)) * (-dT/dy, dT/dx)
        coef = G_ACC / (F_CORIOLIS * (self.t_mean + 273.15))
        self.shear_u = -coef * self.dtdy
        self.shear_v = coef * self.dtdx
        self.rh_mean = 65.0 + 10.0 * math.cos(2 * math.pi * (self.doy - 15) / 365.0) + rng.normal(0, 5)
        self.p_mean = 101300.0 + 500.0 * math.sin(2 * math.pi * self.doy / 365.0 * 3.0) + rng.normal(0, 300)

        # Grille spectrale (surface 50 m)
        dx_s, dy_s = static["dx_s"], static["dy_s"]
        shape = (NY_S, NX_S)
        self.kx = np.fft.fftfreq(NX_S, d=dx_s)
        self.ky = np.fft.fftfreq(NY_S, d=dy_s)
        self.fields = {}
        for name, corr in (("psi", 3000.0), ("div", 4000.0), ("temp", 1000.0), ("rh", 5000.0)):
            self.fields[name] = (gaussian_spectrum_field(shape, dx_s, dy_s, corr, rng),
                                 gaussian_spectrum_field(shape, dx_s, dy_s, corr, rng))
        # normalisation de la perturbation rotationnelle
        fa = self.fields["psi"][0]
        up = spectral_eval(fa, self.kx, self.ky, 0, 0, "y")
        vp = spectral_eval(fa, self.kx, self.ky, 0, 0, "x")
        self.psi_norm = float(np.sqrt(np.mean(up ** 2 + vp ** 2)))

        # Front (episode) : orientation aleatoire
        ang = rng.uniform(0, 2 * np.pi)
        self.front_n = (math.cos(ang), math.sin(ang))
        self.front_turn = rng.choice([-1.0, 1.0]) * rng.uniform(60.0, 90.0)

    # --- evolution temporelle -------------------------------------------------
    def geostrophic(self, t_sec):
        h = t_sec / 3600.0
        g = self.g_mean * (1.0 + 0.15 * math.sin(2 * math.pi * (h - 3.0) / 24.0 + self.g_phase))
        d = (self.dir_mean + self.dir_swing * math.sin(2 * math.pi * h / 24.0 + self.dir_phase)) % 360.0
        rad = math.radians(d)
        return g, d, -g * math.sin(rad), -g * math.cos(rad)

    def stability(self, t_sec, g):
        """s > 0 stable, s < 0 instable, continu dans le temps."""
        h = t_sec / 3600.0
        s_raw = -math.sin(2 * math.pi * (h - 6.5) / 24.0)
        return s_raw * (1.0 - 0.8 * min(g / 14.0, 1.0)) * (1.0 - 0.5 * self.cloud)

    # crochets : le forcage ERA5 (DayStateERA5) les remplace ; ici, les formules d'origine
    def au_temps(self, t_sec):
        """Met a jour les parametres qui varient dans la journee (rien en synthetique)."""

    def hauteur_cl(self, t_sec, s):
        return 700.0 - 500.0 * s if s > 0 else 700.0 + 800.0 * (-s)

    def t_mer(self, t_sec, g, h):
        amp_d = 5.0 * (1.0 - 0.6 * min(g / 15.0, 1.0)) * (1.0 - 0.5 * self.cloud)
        return self.t_mean + amp_d * math.sin(2 * math.pi * (h - 8.0) / 24.0)

    def p_mer(self, t_sec, h):
        return self.p_mean + 120.0 * math.sin(2 * math.pi * h / 12.0)

    def brise(self, t_sec, g):
        """Amplitude de la brise (m/s) : > 0 brise de mer (vers la terre), < 0 brise de terre."""
        return 0.0

    def profil_cible(self, t_sec):
        """Vent moyen impose a chaque niveau (u, v) ou None : rien en synthetique."""
        return None

    def blend(self, name, t_sec):
        fa, fb = self.fields[name]
        phi = (math.pi / 2.0) * (t_sec / 3600.0) / 6.0
        return math.cos(phi) * fa + math.sin(phi) * fb


class DayStateERA5(DayState):
    """Journee pilotee par ERA5 (scripts/forcage_era5.py) au lieu de tirages au hasard.

    Seuls ERA5, le relief et le sol entrent ici : aucune station. Restent aleatoires les
    perturbations de petite echelle (reduites par ANEO_GEN_PERTURB) et les champs de bruit de
    temperature et d'humidite ; le jet nocturne synthetique est coupe."""

    def __init__(self, day_index, date, sector, static, seed, forcage: pd.DataFrame,
                 champs: dict | None = None):
        super().__init__(day_index, date, sector, static, seed)
        self.champs = champs
        t0 = pd.Timestamp(date)
        f = forcage[(forcage.index >= t0 - pd.Timedelta(hours=3))
                    & (forcage.index <= t0 + pd.Timedelta(hours=27))]
        if f.empty:
            raise ValueError(f"pas de forcage ERA5 pour le {date}")
        self.f = f
        self.f_s = (f.index - t0).total_seconds().to_numpy()
        self.jet_day = False
        self.cloud = 0.5
        jour = f[(f.index >= t0) & (f.index < t0 + pd.Timedelta(days=1))]
        self.t_mean = float(jour["t_mer_c"].mean())
        self.p_mean = float(jour["p_mer_pa"].mean())
        self.au_temps(0.0)

    def _val(self, col, t_sec):
        return float(np.interp(t_sec, self.f_s, self.f[col].to_numpy()))

    def au_temps(self, t_sec):
        self.veer = self._val("veer_deg", t_sec)
        self.shear_u = self._val("shear_u", t_sec)
        self.shear_v = self._val("shear_v", t_sec)
        # gradient de temperature coherent avec le cisaillement (vent thermique inverse)
        coef = G_ACC / (F_CORIOLIS * (self.t_mean + 273.15))
        self.dtdy = -self.shear_u / coef
        self.dtdx = self.shear_v / coef

    def geostrophic(self, t_sec):
        ug, vg = self._val("ug", t_sec), self._val("vg", t_sec)
        g = math.hypot(ug, vg)
        d = (270.0 - math.degrees(math.atan2(vg, ug))) % 360.0
        return g, d, ug, vg

    def stability(self, t_sec, g):
        return float(np.clip(self._val("stabilite", t_sec), -1.0, 1.0))

    def hauteur_cl(self, t_sec, s):
        if "h_cl_m" in self.f and self.f["h_cl_m"].notna().all():
            return float(np.clip(self._val("h_cl_m", t_sec), 150.0, 2500.0))
        return super().hauteur_cl(t_sec, s)

    def t_mer(self, t_sec, g, h):
        return self._val("t_mer_c", t_sec)

    def p_mer(self, t_sec, h):
        return self._val("p_mer_pa", t_sec)

    def brise(self, t_sec, g):
        """Amplitude ~ 1,5 sqrt(|dT terre-mer|), au plus 6 m/s, etouffee par le vent synoptique ;
        la brise de terre (nuit, dT < 0) est deux fois plus faible."""
        dt_tm = self._val("dt_terre_mer_k", t_sec)
        a = min(6.0, 1.5 * math.sqrt(abs(dt_tm))) * math.exp(-(g / 8.0) ** 2)
        return a if dt_tm > 0 else -0.5 * a

    def champ_cible(self, t_sec):
        """Vent ERA5 maille par maille (u, v : niveau, lat, lon) a l'instant, interpole dans le
        temps ; None sans fichier de champs."""
        c = self.champs
        if c is None:
            return None
        t = (pd.Timestamp(self.date) + pd.Timedelta(seconds=t_sec)).value // 10 ** 9
        i = int(np.clip(np.searchsorted(c["temps"], t), 1, len(c["temps"]) - 1))
        t0, t1 = c["temps"][i - 1], c["temps"][i]
        a = float(np.clip((t - t0) / max(t1 - t0, 1), 0.0, 1.0))
        if abs(t - t0) > 6 * 3600 and abs(t1 - t) > 6 * 3600:
            raise ValueError(f"pas de champ ERA5 pres du {pd.Timestamp(t, unit='s')}")
        return ((1 - a) * c["u"][i - 1] + a * c["u"][i], (1 - a) * c["v"][i - 1] + a * c["v"][i])

    def profil_cible(self, t_sec):
        """Vent ERA5 moyen sur la zone a chaque niveau du modele : la grande echelle du champ
        genere est celle d'ERA5, le generateur n'en ajoute que les details (relief, cote)."""
        u = np.array([self._val(f"u_era5_{int(z)}", t_sec) for z in Z_LEVELS])
        v = np.array([self._val(f"v_era5_{int(z)}", t_sec) for z in Z_LEVELS])
        return u, v


def lire_champs(chemin_forcage) -> dict | None:
    """Champs ERA5 par maille ecrits a cote du forcage (forcage_era5.npz), s'ils existent."""
    f = Path(chemin_forcage).with_suffix(".npz")
    if not f.exists():
        return None
    with np.load(f) as d:
        return {k: d[k] for k in d.files}


def _bilineaire(lat_e, lon_e, champ, lat, lon):
    """Champ (..., ny_e, nx_e) sur une grille reguliere croissante -> points (lat, lon)."""
    fy = np.clip((lat - lat_e[0]) / (lat_e[1] - lat_e[0]), 0, len(lat_e) - 1.000001)
    fx = np.clip((lon - lon_e[0]) / (lon_e[1] - lon_e[0]), 0, len(lon_e) - 1.000001)
    i, j = np.floor(fy).astype(int), np.floor(fx).astype(int)
    a, b = fy - i, fx - j
    return ((1 - a) * (1 - b) * champ[..., i, j] + (1 - a) * b * champ[..., i, j + 1]
            + a * (1 - b) * champ[..., i + 1, j] + a * b * champ[..., i + 1, j + 1])


def _era5_au_sol(champs, champ, lat, lon, h_sol):
    """Vent ERA5 aux points (lat, lon) a la hauteur h_sol au-dessus du sol (meme forme) :
    bilineaire dans le plan, lineaire en hauteur entre les hauteurs du fichier."""
    hs = champs["hauteurs_sol"]
    h = np.clip(h_sol, hs[0], hs[-1])
    k = np.clip(np.searchsorted(hs, h) - 1, 0, len(hs) - 2)
    t = (h - hs[k]) / (hs[k + 1] - hs[k])
    plan = _bilineaire(champs["lat"], champs["lon"], champ, lat, lon)      # (nh, ...)
    lo = np.take_along_axis(plan, k[None], axis=0)[0]
    hi = np.take_along_axis(plan, (k + 1)[None], axis=0)[0]
    return (1.0 - t) * lo + t * hi


def _passe_bas(a, sigma_px, facteur=8):
    """Lissage gaussien d'un grand champ 2D par sous-echantillonnage (rapide)."""
    from scipy.ndimage import zoom
    petit = a[::facteur, ::facteur]
    petit = gaussian_filter(petit, sigma=sigma_px / facteur, mode="nearest")
    return zoom(petit, (a.shape[0] / petit.shape[0], a.shape[1] / petit.shape[1]), order=1)[:a.shape[0], :a.shape[1]]


def lire_forcage(chemin) -> pd.DataFrame:
    f = pd.read_csv(chemin)
    f.index = pd.to_datetime(f["timestamp_utc"]).dt.tz_localize(None)
    return f.sort_index()


def _periodes(texte):
    out = []
    for morceau in (texte or "").split(","):
        if morceau.strip():
            a, b = morceau.split(":")
            out.append((pd.Timestamp(a), pd.Timestamp(b)))
    return out


JOUR_0_ERA5 = pd.Timestamp("2021-01-01")
SAISONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
HEURES_A, HEURES_B = (0, 6, 12, 18), (3, 9, 15, 21)


def numero_jour(d) -> int:
    """Numero de jour stable : jours depuis le 01/01/2021, + 1 (nom des instants en cache)."""
    return int((pd.Timestamp(d).normalize() - JOUR_0_ERA5).days) + 1


def _regulier(liste: list, n: int) -> list:
    """n elements regulierement espaces dans une liste triee (tous si n >= len)."""
    if n >= len(liste):
        return list(liste)
    if n <= 0:
        return []
    return [liste[int(i)] for i in np.unique(np.linspace(0, len(liste) - 1, n).round().astype(int))]


def calendrier_equilibre(forcage: pd.DataFrame, test: str, val: str, marge_jours: int,
                         instants: int, heures_par_jour: int = 4, part_test: float = 0.18,
                         part_val: float = 0.12):
    """`instants` repartis par saison et par heure (generation reelle, Grand Tunis).

    Chaque jour retenu porte `heures_par_jour` instants : 4 = alternance (0, 6, 12, 18) /
    (3, 9, 15, 21) h UTC, d'un jour retenu au suivant dans chaque groupe -> les 8 heures
    synoptiques autant representees ; 8 = toutes.
      test / validation : part_test / part_val des jours, a parts egales entre les blocs de
        dates, places regulierement sur TOUT le bloc (pas seulement ses jours deja forces) :
        les jours d'un bloc dont le forcage manque encore attendent ses donnees, le calendrier
        ne change pas quand elles arrivent ;
      entrainement : le reste, a parts egales entre les 4 saisons (DJF, MAM, JJA, SON), parmi
        les jours forces a plus de `marge_jours` des blocs.
    Le calendrier est fige dans calendrier.json par l'appelant (main) des le premier passage."""
    heures = forcage.index[forcage.index.hour.isin(SYNOPTIC_HOURS)]
    compte = pd.Series(1, index=heures.normalize()).groupby(level=0).sum()
    forces = {d for d, n in compte.items() if n == len(SYNOPTIC_HOURS)}
    blocs = {"test": _periodes(test), "val": _periodes(val)}
    marge = pd.Timedelta(days=marge_jours)
    tous_blocs = blocs["test"] + blocs["val"]

    n_jours = int(round(instants / heures_par_jour))
    n_test, n_val = int(round(n_jours * part_test)), int(round(n_jours * part_val))
    groupes = []                                    # (jeu, nom du groupe, dates)
    for jeu, n in (("test", n_test), ("val", n_val)):
        for k, (a, b) in enumerate(blocs[jeu]):
            dates = list(pd.date_range(a, b, freq="D"))
            groupes.append((jeu, f"{jeu}_{a:%Y%m%d}", _regulier(dates, n // len(blocs[jeu])
                                                                  + (k < n % len(blocs[jeu])))))
    libres = sorted(d for d in forces if not any(a - marge <= d <= b + marge for a, b in tous_blocs))
    n_train = n_jours - n_test - n_val
    for k, saison in enumerate(("DJF", "MAM", "JJA", "SON")):
        dates = [d for d in libres if SAISONS[d.month] == saison]
        groupes.append(("train", saison, _regulier(dates, n_train // 4 + (k < n_train % 4))))

    days, split, schedule = [], {"train_days": [], "val_days": [], "test_days": []}, {}
    for jeu, _, dates in groupes:
        for j, d in enumerate(dates):
            idx = numero_jour(d)
            if heures_par_jour >= 8:
                h = SYNOPTIC_HOURS
            else:
                h = (HEURES_A if j % 2 == 0 else HEURES_B)[:heures_par_jour]
            days.append({"day_index": idx, "date": d.date()})
            split[f"{jeu}_days"].append(idx)
            schedule[idx] = [(hh * 3600, None) for hh in h]
    days.sort(key=lambda x: x["day_index"])
    for k in split:
        split[k].sort()
    return days, split, schedule


def calendrier_era5(forcage: pd.DataFrame, test: str, val: str, marge_jours: int,
                    pas_train: int, debut: str | None = None, fin: str | None = None,
                    pas_test: int = 1, pas_val: int = 1):
    """Jours reels ayant les 8 heures synoptiques dans le forcage ; test et validation par blocs
    de dates, entrainement = le reste, a plus de `marge_jours` des blocs ; un jour sur
    `pas_train` (`pas_test`, `pas_val`) garde dans chaque ensemble (volume du jeu)."""
    heures = forcage.index[forcage.index.hour.isin(SYNOPTIC_HOURS)]
    compte = pd.Series(1, index=heures.normalize()).groupby(level=0).sum()
    dates = [d for d, n in compte.items() if n == len(SYNOPTIC_HOURS)]
    if debut:
        dates = [d for d in dates if d >= pd.Timestamp(debut)]
    if fin:
        dates = [d for d in dates if d <= pd.Timestamp(fin)]
    blocs = {"test": _periodes(test), "val": _periodes(val)}
    marge = pd.Timedelta(days=marge_jours)

    def dans(d, liste, m=pd.Timedelta(0)):
        return any(a - m <= d <= b + m for a, b in liste)

    days, split = [], {"train_days": [], "val_days": [], "test_days": []}
    vus = {"train_days": 0, "val_days": 0, "test_days": 0}
    pas = {"train_days": pas_train, "val_days": pas_val, "test_days": pas_test}
    for d in dates:
        if dans(d, blocs["test"]):
            jeu = "test_days"
        elif dans(d, blocs["val"]):
            jeu = "val_days"
        elif dans(d, blocs["test"] + blocs["val"], marge):
            continue                                   # tampon autour des blocs
        else:
            jeu = "train_days"
        vus[jeu] += 1
        if (vus[jeu] - 1) % pas[jeu]:
            continue
        idx = numero_jour(d)
        days.append({"day_index": idx, "date": d.date()})
        split[jeu].append(idx)
    schedule = {d["day_index"]: [(h * 3600, None) for h in SYNOPTIC_HOURS] for d in days}
    return days, split, schedule


# =============================================================================
# CHAMPS PHYSIQUES D'UN INSTANT
# =============================================================================

def wind_profile_factor(z_agl, z0, h_bl, s, st: DayState, g):
    """Facteur de vitesse S(z) (fraction du vent geostrophique) et rotation (degres)."""
    za = np.maximum(z_agl, 0.0)
    z_sl = max(0.1 * h_bl, 20.0)
    log_part = np.log1p(za / z0) / np.log1p(z_sl / z0)
    log_part = np.minimum(log_part, 1.0)
    zeta = np.minimum(za / h_bl, 1.0)
    m0 = float(np.clip(0.72 - 0.22 * s, 0.5, 0.92))
    m = m0 + (1.0 - m0) * zeta
    jet = 0.0
    if st.jet_day and s > 0.25:
        jet = st.jet_amp * min(s / 0.6, 1.0) * np.exp(-((za - st.jet_z) / (0.35 * st.jet_z)) ** 2)
    speed = log_part * m * (1.0 + jet)
    speed = np.where(z_agl <= 0.0, 0.0, speed)
    rot = st.veer * (1.0 - zeta)
    return speed, rot


def temperature_profile(t_ground, z_agl, s):
    za = np.maximum(z_agl, 0.0)
    t = t_ground - LAPSE * za
    if s > 0:
        t = t + 8.0 * min(s, 1.0) * (1.0 - np.exp(-za / 70.0))
    else:
        t = t - 1.5 * min(-s, 1.0) * (1.0 - np.exp(-za / 30.0))
    return t


def compute_fields(st: DayState, static: dict, pre: dict, t_sec: float, episode: str | None):
    """
    Calcule tous les champs a l'instant t.
    Retourne un dict avec les champs de surface (NY_S, NX_S) et de volume (NZ, NY_V, NX_V).
    """
    st.au_temps(t_sec)
    g, gdir, ug, vg = st.geostrophic(t_sec)
    s = st.stability(t_sec, g)
    h_bl = st.hauteur_cl(t_sec, s)
    h = t_sec / 3600.0
    a_brise = st.brise(t_sec, g)

    # Advection des perturbations par le vent moyen de couche limite
    shift_x = 0.8 * ug * t_sec
    shift_y = 0.8 * vg * t_sec

    a_rot, a_div, a_slope = A_ROT, A_DIV, A_SLOPE
    if episode in ("rafales", "rafales_10s"):
        a_rot, a_div = 0.25, 0.06
    if episode == "brise":
        a_slope = 2.0

    psi_c = st.blend("psi", t_sec)
    up_s = -spectral_eval(psi_c, st.kx, st.ky, shift_x, shift_y, "y") / st.psi_norm
    vp_s = spectral_eval(psi_c, st.kx, st.ky, shift_x, shift_y, "x") / st.psi_norm
    div_s = spectral_eval(st.blend("div", t_sec), st.kx, st.ky, shift_x, shift_y)
    tn_s = spectral_eval(st.blend("temp", t_sec), st.kx, st.ky, shift_x, shift_y)
    rn_s = spectral_eval(st.blend("rh", t_sec), st.kx, st.ky, shift_x, shift_y)

    # Front : rotation de la direction de part et d'autre d'une ligne mobile
    front_s = 0.0
    if episode == "front":
        nx, ny = st.front_n
        t_ep = (t_sec % 3600.0) - 30 * 60.0
        pos = (static["x"] - static["x"].mean()) * nx + (static["y"] - static["y"].mean()) * ny
        front_s = st.front_turn * 0.5 * (1.0 + np.tanh((pos - 12.0 * t_ep) / 2000.0))

    # Temperature et humidite de reference (niveau mer puis sol)
    t_sl_mean = st.t_mer(t_sec, g, h)
    xc = static["x"] - static["x"].mean()
    yc = static["y"] - static["y"].mean()
    t_sl_s = t_sl_mean + 0.1 * tn_s + st.dtdx * xc + st.dtdy * yc
    rh_s_ref = st.rh_mean - 1.8 * (t_sl_mean - st.t_mean) + 8.0 * rn_s

    # Pression au niveau de la mer : maree + equilibre geostrophique
    p0 = st.p_mer(t_sec, h)
    p_sl_s = p0 + 1.2 * F_CORIOLIS * (vg * xc - ug * yc)

    def state_at(z_asl, z_agl_true, sl):
        """
        Champs a une altitude donnee pour une tranche horizontale (indices sl).
        Sous le terrain, le vent est prolonge par sa valeur a MIN_ZREL_NODE : ces cellules
        sont masquees, mais le prolongement lisse evite les marches d'escalier dans w.
        """
        zt = static["zt"][sl]
        z0 = static["z0"][sl]
        # Hauteur effective pour le profil de vent : suit le relief pres du sol, puis les
        # lignes de courant s'aplatissent avec l'altitude (la perturbation du relief decroit).
        zt_s = pre["zt_smooth"][sl]
        z_agl_prof = z_asl - zt_s * np.exp(-np.maximum(z_asl - zt_s, 0.0) / TERRAIN_DECAY_M)
        z_agl = np.maximum(np.minimum(z_agl_prof, z_asl), MIN_ZREL_NODE)
        speed, rot = wind_profile_factor(z_agl, z0, h_bl, s, st, g)
        env = 1.0 - 0.5 * np.minimum(np.maximum(z_agl, 0) / h_bl, 1.0)

        rel = pre["relief"][sl]
        speedup = np.clip(1.0 + 0.15 * rel / 100.0 * np.exp(-np.maximum(z_agl, 0) / 300.0), 0.6, 1.6)
        mod = speedup * (1.0 + a_div * div_s[sl] * env)

        ang = np.radians(rot + (front_s[sl] if isinstance(front_s, np.ndarray) else front_s))
        ub = ug * np.cos(ang) - vg * np.sin(ang)
        vb = ug * np.sin(ang) + vg * np.cos(ang)
        amp = speed * g
        u = ub * speed * mod + a_rot * amp * env * up_s[sl]
        v = vb * speed * mod + a_rot * amp * env * vp_s[sl]

        # Vent thermique au-dessus de la couche limite (lie au gradient de temperature de surface)
        above = np.maximum(z_agl - h_bl, 0.0)
        u = u + st.shear_u * above
        v = v + st.shear_v * above

        # Vents de pente : montants le jour, descendants la nuit
        sw = a_slope * (1.5 * max(-s, 0.0) - 1.0 * max(s, 0.0))
        decay = np.where(z_agl > 0, np.exp(-np.maximum(z_agl, 0) / 150.0), 0.0)
        u = u + sw * decay * pre["upslope_x"][sl]
        v = v + sw * decay * pre["upslope_y"][sl]

        # Brise de mer (jour) / de terre (nuit), perpendiculaire a la cote : courant de basses
        # couches vers la terre, retour plus faible au-dessus ; w suit par la continuite
        if a_brise != 0.0:
            zb = np.maximum(z_agl, 0.0) / H_BRISE_M
            prof = np.where(zb < 1.0, 1.0 - zb,
                            np.where(zb < 2.0, -0.35 * np.sin(np.pi * (zb - 1.0)), 0.0))
            u = u + a_brise * prof * pre["brise_x"][sl]
            v = v + a_brise * prof * pre["brise_y"][sl]

        t_ground = t_sl_s[sl] - LAPSE * zt
        # temperature et humidite : hauteur reelle au-dessus du sol (gradient adiabatique local)
        z_th = np.maximum(z_agl_true, 0.0)
        t_c = temperature_profile(t_ground, z_th, s)
        zeta = np.minimum(z_th / h_bl, 1.0)
        rh = rh_s_ref[sl] + 12.0 * zeta - 20.0 * np.maximum(z_th - h_bl, 0.0) / 1000.0
        rh = np.clip(rh, 5.0, 100.0)

        t_k = t_c + 273.15
        t_mean_col = 0.5 * (t_sl_s[sl] + 273.15 + t_k)
        e_tmp = rh / 100.0 * magnus_es(t_c)
        q_tmp = 0.622 * e_tmp / (p_sl_s[sl] - 0.378 * e_tmp)
        p = p_sl_s[sl] * np.exp(-G_ACC * z_asl / (RD * t_mean_col * (1.0 + 0.608 * q_tmp)))
        e = rh / 100.0 * magnus_es(t_c)
        q = 0.622 * e / (p - 0.378 * e)
        t_v = t_k * (1.0 + 0.608 * q)
        rho = p / (RD * t_v)
        theta = t_k * (P_REF / p) ** KAPPA
        return {"u": u, "v": v, "t": t_c, "rh": rh, "p": p, "rho": rho, "theta": theta, "e": e}

    # ---------------------------------------------------------------- volume
    sl_v = (slice(None, None, STRIDE), slice(None, None, STRIDE))
    zt_v = static["zt"][sl_v]
    vol = {k: np.empty((NZ, NY_V, NX_V)) for k in ("u", "v", "t", "rh", "p", "rho", "theta", "e")}
    for k, z in enumerate(Z_LEVELS):
        f = state_at(z, z - zt_v, sl_v)
        for key in vol:
            vol[key][k] = f[key]

    # Recalage de la grande echelle (forcage ERA5) : a chaque niveau, un vent UNIFORME ramene la
    # moyenne sur les noeuds valides a celle d'ERA5. Uniforme : la divergence horizontale ne
    # change pas ; w est integre ensuite, la conservation de la masse reste exacte.
    champ = st.champ_cible(t_sec) if hasattr(st, "champ_cible") else None
    cible = st.profil_cible(t_sec)
    du = dv = None
    cu = cv = None
    if champ is not None and "lat" in static:
        # recalage SPATIAL : la grande echelle (lissee a ~12 km) du champ genere est remplacee,
        # niveau par niveau, par ERA5 interpole a chaque noeud ; les details restent
        # (meme hauteur au-dessus du sol que dans ERA5 : le recalage suit le relief)
        lat_v, lon_v = static["lat"][sl_v], static["lon"][sl_v]
        sig = 12000.0 / (static["dx_s"] * STRIDE)
        cu, cv = np.zeros_like(vol["u"]), np.zeros_like(vol["v"])
        for k in range(NZ):
            h_sol = np.maximum(Z_LEVELS[k] - zt_v, MIN_ZREL_NODE)
            cu[k] = _era5_au_sol(st.champs, champ[0], lat_v, lon_v, h_sol) - _passe_bas(vol["u"][k], sig)
            cv[k] = _era5_au_sol(st.champs, champ[1], lat_v, lon_v, h_sol) - _passe_bas(vol["v"][k], sig)
        vol["u"] += cu
        vol["v"] += cv
    elif cible is not None:
        du, dv = np.zeros(NZ), np.zeros(NZ)
        for k in range(NZ):
            ok = static["node_valid"][k]
            if ok.any() and np.isfinite(cible[0][k]) and np.isfinite(cible[1][k]):
                du[k] = cible[0][k] - float(vol["u"][k][ok].mean())
                dv[k] = cible[1][k] - float(vol["v"][k][ok].mean())
        for k in range(NZ):                     # niveaux sans noeud : correction du plus proche
            if not static["node_valid"][k].any():
                j = int(np.argmax(static["node_valid"].reshape(NZ, -1).any(1)))
                du[k], dv[k] = du[j], dv[j]
        vol["u"] += du[:, None, None]
        vol["v"] += dv[:, None, None]

    # w par la continuite (integration de bas en haut, schema aux demi-niveaux)
    dx_v = static["dx_s"] * STRIDE
    dy_v = static["dy_s"] * STRIDE
    # Condition cinematique au sol (w = u.grad(z_t)) imposee au premier noeud valide de chaque
    # colonne, puis integration de la continuite vers le haut. Le residu est nul sur tous les
    # demi-niveaux dont les deux noeuds sont valides.
    nv = static["node_valid"]
    dz = np.diff(Z_LEVELS)

    def integrate_w():
        dd = horizontal_divergence(vol["rho"] * vol["u"], vol["rho"] * vol["v"], dx_v, dy_v)
        w_bc = vol["u"] * pre["dzt_dx_v"][None] + vol["v"] * pre["dzt_dy_v"][None]
        rw = np.zeros_like(dd)
        rw[0] = np.where(nv[0], vol["rho"][0] * w_bc[0], 0.0)
        for k in range(NZ - 1):
            cont = rw[k] - 0.5 * (dd[k] + dd[k + 1]) * dz[k]
            first = nv[k + 1] & ~nv[k]
            rw[k + 1] = np.where(first, vol["rho"][k + 1] * w_bc[k + 1], np.where(nv[k + 1], cont, 0.0))
        return rw

    # 1) premiere integration
    rho_w = integrate_w()

    # 2) amortissement vertical de w : l'influence du relief doit decroitre avec l'altitude.
    #    A chaque niveau, un petit ecoulement potentiel horizontal (grad(chi) / rho, lisse)
    #    retire une fraction du flux vertical herite du niveau inferieur. La continuite reste
    #    exacte car w est reintegre a partir des u, v definitifs.
    lap = -(2 * np.pi) ** 2 * (pre["kx_v"][None, :] ** 2 + pre["ky_v"][:, None] ** 2)
    lap[0, 0] = 1.0
    corr_gx = np.zeros_like(vol["u"])
    corr_gy = np.zeros_like(vol["v"])
    for k in range(1, NZ):
        pair = nv[k] & nv[k - 1]
        excess = np.where(pair, rho_w[k - 1] * (1.0 - math.exp(-dz[k - 1] / W_DAMPING_M)), 0.0)
        src = 2.0 * excess / dz[k - 1]
        src = src - src.mean()
        chi_hat = np.fft.fft2(src) / lap
        chi_hat[0, 0] = 0.0
        gx = np.fft.ifft2(chi_hat * (2j * np.pi * pre["kx_v"][None, :])).real
        gy = np.fft.ifft2(chi_hat * (2j * np.pi * pre["ky_v"][:, None])).real
        corr_gx[k], corr_gy[k] = gx, gy
        vol["u"][k] += gx / vol["rho"][k]
        vol["v"][k] += gy / vol["rho"][k]
        rho_w = integrate_w()

    # 3) integration finale (w exactement coherent avec u, v definitifs)
    rho_w = integrate_w()
    vol["w"] = rho_w / vol["rho"]

    # --------------------------------------------------------------- surface
    sl_s = (slice(None), slice(None))
    z_abs = static["zt"] + static["zrel"]
    surf = state_at(z_abs, static["zrel"], sl_s)
    if cu is not None:                          # recalage spatial, a la hauteur du capteur
        sig_s = 12000.0 / static["dx_s"]
        for i_c, cle in ((0, "u"), (1, "v")):
            cible_s = _era5_au_sol(st.champs, champ[i_c], static["lat"], static["lon"], static["zrel"])
            surf[cle] = surf[cle] + cible_s - _passe_bas(surf[cle], sig_s, facteur=16)
    if du is not None:                          # meme recalage, a l'altitude du capteur
        surf["u"] = surf["u"] + np.interp(z_abs, Z_LEVELS, du)
        surf["v"] = surf["v"] + np.interp(z_abs, Z_LEVELS, dv)

    idx = np.clip(np.searchsorted(Z_LEVELS, z_abs) - 1, 0, NZ - 2)
    z_lo = Z_LEVELS[idx]
    z_hi = Z_LEVELS[idx + 1]

    def up(a):
        return np.repeat(np.repeat(a, STRIDE, axis=1), STRIDE, axis=2)[:, :NY_S, :NX_S]

    def at_zabs(a3):
        lo = np.take_along_axis(a3, idx[None], axis=0)[0]
        hi = np.take_along_axis(a3, (idx + 1)[None], axis=0)[0]
        t = np.clip((z_abs - z_lo) / (z_hi - z_lo), 0.0, 1.0)
        return (1.0 - t) * lo + t * hi

    # meme correction potentielle que le volume, interpolee a l'altitude du capteur
    surf["u"] = surf["u"] + at_zabs(up(corr_gx)) / surf["rho"]
    surf["v"] = surf["v"] + at_zabs(up(corr_gy)) / surf["rho"]

    # w de surface : interpolation entre le sol (condition cinematique) et le noeud valide au-dessus
    w_up = up(vol["w"])
    w_lo = np.take_along_axis(w_up, idx[None], axis=0)[0]
    w_hi = np.take_along_axis(w_up, (idx + 1)[None], axis=0)[0]
    w_ground = surf["u"] * pre["dzt_dx_s"] + surf["v"] * pre["dzt_dy_s"]
    below = (z_lo - static["zt"]) < MIN_ZREL_NODE
    z_lo = np.where(below, static["zt"], z_lo)
    w_lo = np.where(below, w_ground, w_lo)
    wt = np.clip((z_abs - z_lo) / np.maximum(z_hi - z_lo, 1e-6), 0.0, 1.0)
    surf["w"] = (1.0 - wt) * w_lo + wt * w_hi

    meta = {"g": g, "gdir": gdir, "s": s, "h_bl": h_bl, "brise": a_brise}
    return surf, vol, meta


def mer_ouverte(eau: np.ndarray, largeur_chenal_m: float = 400.0, dx: float = DX_S) -> np.ndarray:
    """Eau marine : la partie de l'eau reliee au bord du domaine, une fois coupes les chenaux
    plus etroits que `largeur_chenal_m` (canal de La Goulette...). Lacs, lagunes et sebkhas
    interieurs ne comptent pas comme mer pour la brise. Regle generale, sans exception locale."""
    from scipy.ndimage import binary_dilation, binary_erosion, label
    r = max(int(round(largeur_chenal_m / (2.0 * dx))), 1)
    coeur = binary_erosion(eau, iterations=r, border_value=1)
    lab, _ = label(coeur)
    bord = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    mer = np.isin(lab, bord[bord > 0])
    return binary_dilation(mer, iterations=r) & eau      # le trait de cote est restaure


def precompute_static_derived(static: dict):
    zt = static["zt"]
    rel = zt - gaussian_filter(zt, sigma=1500.0 / DX_S, mode="nearest")
    zs = gaussian_filter(zt, sigma=300.0 / DX_S, mode="nearest")
    gy, gx = np.gradient(zs, static["dy_s"], static["dx_s"])
    norm = np.hypot(gx, gy)
    strength = np.clip(norm / 0.05, 0.0, 1.0)
    ux = np.where(norm > 1e-6, gx / np.maximum(norm, 1e-6), 0.0) * strength
    uy = np.where(norm > 1e-6, gy / np.maximum(norm, 1e-6), 0.0) * strength
    # l'ecoulement voit un relief lisse (echelle ~300 m) : pentes et condition au sol
    zt_flow = gaussian_filter(zt, sigma=TERRAIN_SMOOTH_M / DX_S, mode="nearest")
    zt_v = zt_flow[::STRIDE, ::STRIDE]
    gyv, gxv = np.gradient(zt_v, static["dy_s"] * STRIDE, static["dx_s"] * STRIDE)
    gys, gxs = np.gradient(zt_flow, static["dy_s"], static["dx_s"])
    # profil vertical de la correction divergente : integrale (trapezes) egale a 1
    prof = Z_LEVELS / Z_LEVELS[-1]
    integ = np.sum(0.5 * (prof[1:] + prof[:-1]) * np.diff(Z_LEVELS))
    # brise : distance signee a la cote MARINE (> 0 sur terre ; lacs et sebkhas = terre),
    # normale vers l'interieur des terres, poids maximal a la cote et decroissant vers la terre
    # (15 km) et vers la mer (10 km)
    from scipy.ndimage import distance_transform_edt
    eau = mer_ouverte(static["z0"] <= 3e-4, dx=static["dx_s"])
    if eau.any() and (~eau).any():
        dist = (distance_transform_edt(~eau) - distance_transform_edt(eau)) * static["dx_s"]
        ds = gaussian_filter(dist, sigma=500.0 / DX_S, mode="nearest")
        nyc, nxc = np.gradient(ds, static["dy_s"], static["dx_s"])
        nn = np.maximum(np.hypot(nxc, nyc), 1e-9)
        poids = np.where(dist >= 0, np.exp(-dist / L_BRISE_TERRE_M), np.exp(dist / L_BRISE_MER_M))
        bx, by = poids * nxc / nn, poids * nyc / nn
    else:
        bx = by = np.zeros_like(zt)
    return {"relief": rel, "upslope_x": ux, "upslope_y": uy, "dzt_dx_v": gxv, "dzt_dy_v": gyv,
            "brise_x": bx, "brise_y": by,
            "dzt_dx_s": gxs, "dzt_dy_s": gys, "zt_smooth": zt_flow,
            "corr_profile": prof / integ,
            "kx_v": np.fft.fftfreq(NX_V, d=static["dx_s"] * STRIDE),
            "ky_v": np.fft.fftfreq(NY_V, d=static["dy_s"] * STRIDE)}


# =============================================================================
# ECRITURE
# =============================================================================

def instant_rows(st: DayState, static: dict, surf: dict, vol: dict, t_sec: float,
                 date: dt.date, quality: np.ndarray, interp_mask: np.ndarray, node_id_start: int):
    """Construit les colonnes d'un instant : surface (ordre point_id) puis noeuds 3D."""
    ts = dt.datetime.combine(date, dt.time()) + dt.timedelta(seconds=float(t_sec))
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    hour = ts.hour

    rng = st.rng_obs
    speed = math.hypot(*st.geostrophic(t_sec)[2:])
    sig_uv = min(0.1, 0.02 + 0.006 * speed)

    order = np.argsort(static["point_id"].ravel(), kind="stable")
    n_s = order.size

    def s_flat(a):
        return np.asarray(a).ravel()[order]

    shape = (NY_S, NX_S)
    noisy_grid = ~interp_mask
    su_g = surf["u"] + noisy_grid * rng.normal(0.0, sig_uv, shape)
    sv_g = surf["v"] + noisy_grid * rng.normal(0.0, sig_uv, shape)
    sw_g = surf["w"] + noisy_grid * rng.normal(0.0, 0.003, shape)
    st_g = surf["t"] + noisy_grid * rng.normal(0.0, 0.05, shape)
    interp = interp_mask.ravel()[order]
    su, sv, sw, stc = s_flat(su_g), s_flat(sv_g), s_flat(sw_g), s_flat(st_g)

    kk, jj, ii = static["node_k"], static["node_j"], static["node_i"]

    def v_flat(a):
        return a[kk, jj, ii]

    n_v = kk.size
    cols = {
        "point_id": np.concatenate([static["point_id"].ravel()[order],
                                    np.arange(node_id_start, node_id_start + n_v, dtype=np.int64)]),
        "timestamp_utc": np.full(n_s + n_v, ts_str, dtype=object),
        "day_index": np.full(n_s + n_v, st.day_index, dtype=np.int64),
        "hour_utc": np.full(n_s + n_v, hour, dtype=np.int64),
        "temperature_c": np.concatenate([stc, v_flat(vol["t"])]),
        "potential_temperature_k": np.concatenate([s_flat(surf["theta"]), v_flat(vol["theta"])]),
        "air_density_kg_m3": np.concatenate([s_flat(surf["rho"]), v_flat(vol["rho"])]),
        "relative_humidity_pct": np.concatenate([s_flat(surf["rh"]), v_flat(vol["rh"])]),
        "water_vapor_pressure_pa": np.concatenate([s_flat(surf["e"]), v_flat(vol["e"])]),
        "pressure_pa": np.concatenate([s_flat(surf["p"]), v_flat(vol["p"])]),
        "wind_u_m_s": np.concatenate([su, v_flat(vol["u"])]),
        "wind_v_m_s": np.concatenate([sv, v_flat(vol["v"])]),
        "wind_w_m_s": np.concatenate([sw, v_flat(vol["w"])]),
        "aneo_quality_score": np.concatenate([quality.ravel()[order], np.ones(n_v)]),
        "mask_i_obs": np.concatenate([(~interp).astype(np.int64), np.zeros(n_v, dtype=np.int64)]),
        "source_type": np.concatenate([np.where(interp, SRC_INTERP, SRC_SURFACE).astype(object),
                                       np.full(n_v, SRC_NODE, dtype=object)]),
    }
    for c, dec in DECIMALS.items():
        cols[c] = np.round(cols[c].astype(np.float64), dec)
    return {c: cols[c] for c in OBS_COLUMNS}, su_g, sv_g, st_g


def instant_tenseurs(st: DayState, static: dict, surf: dict, vol: dict, t_sec: float,
                     interp_mask: np.ndarray, pool: bool = True):
    """Un instant directement au format du cache pcfno3d (sans CSV) :
        S (8, ny, nx) float32 : u, v, w, T, rho, HR, p, capteur_dispo (surface, bruit de mesure
          compris, agregee par bloc STRIDE x STRIDE comme Geometry.pool_surface) ;
        Y (4, nz, ny, nx) float16 : u, v, w, rho aux noeuds 3D, 0 sous le relief.
    float16 suffit a la cible (1e-3 relatif) et divise le cache par deux ; S reste en float32
    (la pression depasse la plage du float16). Renvoie aussi les champs bruites pour le QC."""
    rng = st.rng_obs
    speed = math.hypot(*st.geostrophic(t_sec)[2:])
    sig_uv = min(0.1, 0.02 + 0.006 * speed)
    shape = (NY_S, NX_S)
    noisy = ~interp_mask
    su = surf["u"] + noisy * rng.normal(0.0, sig_uv, shape)
    sv = surf["v"] + noisy * rng.normal(0.0, sig_uv, shape)
    sw = surf["w"] + noisy * rng.normal(0.0, 0.003, shape)
    stc = surf["t"] + noisy * rng.normal(0.0, 0.05, shape)
    canaux = (su, sv, sw, stc, surf["rho"], surf["rh"], surf["p"])

    k = STRIDE
    S = np.zeros((8, NY_V, NX_V), np.float32)
    if pool:
        m = noisy[:NY_V * k, :NX_V * k].reshape(NY_V, k, NX_V, k).astype(np.float64)
        cnt = m.sum(axis=(1, 3))
        for c, a in enumerate(canaux):
            v = np.asarray(a, np.float64)[:NY_V * k, :NX_V * k].reshape(NY_V, k, NX_V, k)
            S[c] = np.where(cnt > 0, (v * m).sum(axis=(1, 3)) / np.maximum(cnt, 1.0), 0.0)
        S[7] = cnt > 0
    else:
        for c, a in enumerate(canaux):
            S[c] = np.asarray(a)[::k, ::k][:NY_V, :NX_V]
        S[7] = noisy[::k, ::k][:NY_V, :NX_V]

    nv = static["node_valid"]
    Y = np.zeros((4, NZ, NY_V, NX_V), np.float16)
    for c, cle in enumerate(("u", "v", "w", "rho")):
        Y[c] = np.where(nv, np.nan_to_num(vol[cle]), 0.0)
    return S, Y, su, sv, stc


def _arrow_csv_options():
    """Options CSV pyarrow sans guillemets ; None si la version installee ne les supporte pas."""
    if not HAS_ARROW:
        return None
    try:
        return pacsv.WriteOptions(include_header=False, quoting_style="none")
    except TypeError:
        return None


class DayWriter:
    """
    Ecriture incrementale d'un fichier par jour (.csv ou .csv.gz, entete unique).
    L'entete est ecrite a la main (compatible avec toutes les versions de pyarrow).
    Si pyarrow est absent ou trop ancien, l'ecriture passe par pandas (plus lente).
    """

    def __init__(self, path: Path, compress: bool = True):
        self.path = path
        self.tmp = path.with_suffix(path.suffix + ".part")
        self.writer = None
        self.options = _arrow_csv_options()
        self.use_arrow = self.options is not None
        header = ",".join(OBS_COLUMNS) + "\n"
        if self.use_arrow:
            if compress:
                self.stream = pa.CompressedOutputStream(str(self.tmp), "gzip")
            else:
                self.stream = pa.OSFile(str(self.tmp), "wb")
            self.stream.write(header.encode("utf-8"))
        else:
            if compress:
                import gzip
                self.stream = gzip.open(self.tmp, "wt", encoding="utf-8", newline="\n")
            else:
                self.stream = open(self.tmp, "w", encoding="utf-8", newline="\n")
            self.stream.write(header)

    def write(self, cols: dict):
        if self.use_arrow:
            table = pa.table(cols)
            if self.writer is None:
                self.writer = pacsv.CSVWriter(self.stream, table.schema, write_options=self.options)
            self.writer.write_table(table)
        else:
            pd.DataFrame(cols).to_csv(self.stream, index=False, header=False, lineterminator="\n")

    def close(self):
        if self.use_arrow and self.writer is not None:
            self.writer.close()
        self.stream.close()
        os.replace(self.tmp, self.path)


# =============================================================================
# CONTROLE QUALITE PAR INSTANT
# =============================================================================

def qc_instant(static, vol, surf_u, surf_v, surf_t, meta, prev, t_sec):
    nv = static["node_valid"]
    dx_v = static["dx_s"] * STRIDE
    dy_v = static["dy_s"] * STRIDE

    # Residu de masse sur les valeurs arrondies (ce que le modele lira)
    rho = np.round(vol["rho"], DECIMALS["air_density_kg_m3"])
    u = np.round(vol["u"], DECIMALS["wind_u_m_s"])
    v = np.round(vol["v"], DECIMALS["wind_v_m_s"])
    w = np.round(vol["w"], DECIMALS["wind_w_m_s"])
    r = mass_residual(rho, u, v, w, Z_LEVELS, dx_v, dy_v)
    half_valid = nv[1:] & nv[:-1]
    mass_ms = float(np.mean(r[half_valid] ** 2)) if half_valid.any() else float("nan")

    # Vent a 100 m ASL (noeuds valides)
    k100 = int(np.where(Z_LEVELS == 100.0)[0][0])
    m100 = nv[k100]
    u100, v100 = vol["u"][k100][m100], vol["v"][k100][m100]
    spd100 = float(np.mean(np.hypot(u100, v100))) if m100.any() else float("nan")
    dir100 = float(met_direction(u100.mean(), v100.mean())) if m100.any() else float("nan")

    # Correlations spatiales
    corr_s = {
        "corr_surf_u": 0.5 * (lag1_corr(surf_u, 1) + lag1_corr(surf_u, 0)),
        "corr_surf_v": 0.5 * (lag1_corr(surf_v, 1) + lag1_corr(surf_v, 0)),
        "corr_surf_t": 0.5 * (lag1_corr(surf_t, 1) + lag1_corr(surf_t, 0)),
    }
    k200 = int(np.where(Z_LEVELS == 200.0)[0][0])
    corr_v = {
        "corr_vol_u": masked_lag1_corr(vol["u"][k200], nv[k200]),
        "corr_vol_v": masked_lag1_corr(vol["v"][k200], nv[k200]),
        "corr_vol_t": masked_lag1_corr(vol["t"][k200], nv[k200]),
    }

    # Hydrostatique : dp/dz = -rho g
    p = vol["p"]
    dpdz = (p[1:] - p[:-1]) / np.diff(Z_LEVELS)[:, None, None]
    rho_mid = 0.5 * (vol["rho"][1:] + vol["rho"][:-1])
    rel_err = np.abs(dpdz + rho_mid * G_ACC) / (rho_mid * G_ACC)
    hydro = float(np.percentile(rel_err[half_valid], 99)) if half_valid.any() else float("nan")

    w_std = float(np.std(vol["w"][nv]))

    # Coherence temporelle (surface u, v)
    t_corr = float("nan")
    dt_prev = float("nan")
    if prev is not None:
        dt_prev = t_sec - prev["t"]
        a = np.concatenate([surf_u.ravel(), surf_v.ravel()])
        b = np.concatenate([prev["u"].ravel(), prev["v"].ravel()])
        t_corr = float(np.corrcoef(a, b)[0, 1])

    s = meta["s"]
    stab = "stable" if s > 0.25 else ("instable" if s < -0.25 else "neutre")
    row = {
        "mass_residual_mean_sq": mass_ms, "speed_100m": spd100, "dir_100m": dir100,
        "w_std": w_std, "hydrostatic_rel_err_p99": hydro,
        "temporal_corr": t_corr, "dt_prev_s": dt_prev,
        "stability": stab, "s": s, "h_bl": meta["h_bl"], "g": meta["g"],
    }
    row.update(corr_s)
    row.update(corr_v)
    return row


# =============================================================================
# GENERATION D'UNE JOURNEE (processus parallele)
# =============================================================================

_STATIC = None
_PRE = None
_FORCAGE = None
_CHAMPS = None


def _worker_init(static_path: str, forcage_path: str | None = None):
    global _STATIC, _PRE, _FORCAGE
    global _CHAMPS
    _FORCAGE = lire_forcage(forcage_path) if forcage_path else None
    _CHAMPS = lire_champs(forcage_path) if forcage_path else None
    data = np.load(static_path, allow_pickle=False)
    _STATIC = {k: data[k] for k in data.files}
    for k in ("dx_s", "dy_s"):
        _STATIC[k] = float(_STATIC[k])
    _PRE = precompute_static_derived(_STATIC)


def generate_day(day: dict, instants: list, sector: int, seed: int, out_dir: str,
                 node_id_start: int, drive_dir: str | None, compress: bool = True):
    static, pre = _STATIC, _PRE
    out_dir = Path(out_dir)
    date = day["date"]
    idx = day["day_index"]
    ext = ".csv.gz" if compress else ".csv"
    fname = f"aneo_observations_day_{idx:03d}_{date.strftime('%Y%m%d')}_multitime{ext}"
    final_path = (Path(drive_dir) if drive_dir else out_dir) / fname
    qc_path = out_dir / "qc" / f"qc_day_{idx:03d}.csv"
    if final_path.exists() and qc_path.exists():
        return idx, "deja present", 0.0

    t_start = time.time()
    st = (DayStateERA5(idx, date, sector, static, seed, _FORCAGE, _CHAMPS) if _FORCAGE is not None
          else DayState(idx, date, sector, static, seed))
    rng_day = np.random.default_rng(seed + 104729)
    quality = rng_day.uniform(0.85, 0.95, size=(NY_S, NX_S))
    interp_mask = rng_day.random((NY_S, NX_S)) < 0.012
    quality = np.where(interp_mask, rng_day.uniform(0.68, 0.80, size=(NY_S, NX_S)), quality)

    writer = DayWriter(out_dir / fname, compress)
    qc_rows = []
    prev = None
    for t_sec, episode in instants:
        surf, vol, meta = compute_fields(st, static, pre, t_sec, episode)
        cols, su, sv, stc = instant_rows(st, static, surf, vol, t_sec, date, quality,
                                         interp_mask, node_id_start)
        writer.write(cols)
        row = qc_instant(static, vol, su, sv, stc, meta, prev, t_sec)
        row.update({"day_index": idx, "timestamp_utc": cols["timestamp_utc"][0],
                    "episode": episode or "synoptique"})
        qc_rows.append(row)
        prev = {"t": t_sec, "u": su, "v": sv}
    writer.close()

    if drive_dir:
        Path(drive_dir).mkdir(parents=True, exist_ok=True)
        shutil.move(str(out_dir / fname), str(final_path))
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False, encoding="utf-8")
    return idx, "ok", time.time() - t_start


def generate_day_cache(day: dict, instants: list, sector: int, seed: int, out_dir: str,
                       cache_dir: str, pool: bool = True):
    """Comme generate_day, mais ecrit directement <jour>_<HHMMSS>_S.npy / _Y.npy dans le
    cache du modele (pas de CSV : ~3 fois moins de disque, pas de conversion sur Colab).
    Renvoie aussi les lignes d'index (stem, day, timestamp, hour, doy)."""
    static, pre = _STATIC, _PRE
    out_dir, cache_dir = Path(out_dir), Path(cache_dir)
    date = day["date"]
    idx = day["day_index"]
    qc_path = out_dir / "qc" / f"qc_day_{idx:04d}.csv"

    t_start = time.time()
    st = (DayStateERA5(idx, date, sector, static, seed, _FORCAGE, _CHAMPS) if _FORCAGE is not None
          else DayState(idx, date, sector, static, seed))
    rng_day = np.random.default_rng(seed + 104729)
    interp_mask = rng_day.random((NY_S, NX_S)) < 0.012

    qc_rows, rows = [], []
    prev = None
    for t_sec, episode in instants:
        surf, vol, meta = compute_fields(st, static, pre, t_sec, episode)
        S, Y, su, sv, stc = instant_tenseurs(st, static, surf, vol, t_sec, interp_mask, pool)
        ts = dt.datetime.combine(date, dt.time()) + dt.timedelta(seconds=float(t_sec))
        stem = f"{idx:04d}_{ts.strftime('%H%M%S')}"
        tenseurs = [("_S", S), ("_Y", Y)]
        champ = st.champ_cible(t_sec) if hasattr(st, "champ_cible") else None
        if champ is not None and "lat" in static:
            # entree de production : vent ERA5 a 10 m au-dessus du sol aux noeuds (100 m)
            lat_v = static["lat"][::STRIDE, ::STRIDE][:NY_V, :NX_V]
            lon_v = static["lon"][::STRIDE, ::STRIDE][:NY_V, :NX_V]
            dix = np.full(lat_v.shape, 10.0)
            E = np.stack([_era5_au_sol(st.champs, champ[c], lat_v, lon_v, dix) for c in (0, 1)])
            tenseurs.append(("_E", E.astype(np.float16)))
        for suffixe, a in tenseurs:
            tmp = cache_dir / f"{stem}{suffixe}.tmp.npy"
            np.save(tmp, a)
            os.replace(tmp, cache_dir / f"{stem}{suffixe}.npy")
        rows.append({"stem": stem, "day": idx, "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                     "hour": ts.hour + ts.minute / 60 + ts.second / 3600,
                     "doy": ts.timetuple().tm_yday})
        row = qc_instant(static, vol, su, sv, stc, meta, prev, t_sec)
        row.update({"day_index": idx, "timestamp_utc": rows[-1]["timestamp"],
                    "episode": episode or "synoptique"})
        qc_rows.append(row)
        prev = {"t": t_sec, "u": su, "v": sv}
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False, encoding="utf-8")
    return idx, "ok", time.time() - t_start, rows


# =============================================================================
# SYNTHESE DU CONTROLE QUALITE
# =============================================================================

def qc_summary(out_dir: Path, split: dict):
    files = sorted((out_dir / "qc").glob("qc_day_*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(out_dir / "qc" / "qc_instants.csv", index=False, encoding="utf-8")
    th = QC_THRESHOLDS

    sectors = ((df["dir_100m"] + 22.5) % 360 // 45).astype(int)
    share = sectors.value_counts(normalize=True).reindex(range(8), fill_value=0.0)
    one_min = df[(df["dt_prev_s"] > 0) & (df["dt_prev_s"] <= 60)]
    three_h = df[df["dt_prev_s"] == 3 * 3600]

    checks = {
        "residu_masse_max": [float(df["mass_residual_mean_sq"].max()), th["mass_residual_mean_sq_max"], "<="],
        "corr_spatiale_surface_u_min": [float(df["corr_surf_u"].min()), th["spatial_corr_surface_min"], ">="],
        "corr_spatiale_surface_v_min": [float(df["corr_surf_v"].min()), th["spatial_corr_surface_min"], ">="],
        "corr_spatiale_surface_t_min": [float(df["corr_surf_t"].min()), th["spatial_corr_surface_min"], ">="],
        "corr_spatiale_volume_u_min": [float(df["corr_vol_u"].min()), th["spatial_corr_volume_min"], ">="],
        "corr_spatiale_volume_v_min": [float(df["corr_vol_v"].min()), th["spatial_corr_volume_min"], ">="],
        "corr_temporelle_1min_min": [float(one_min["temporal_corr"].min()) if len(one_min) else float("nan"),
                                     th["temporal_corr_1min_min"], ">="],
        "corr_temporelle_3h_moyenne": [float(three_h["temporal_corr"].mean()) if len(three_h) else float("nan"),
                                       th["temporal_corr_3h_min"], ">="],
        "part_min_secteur_direction": [float(share.min()), th["direction_sector_min_share"], ">="],
        "erreur_hydrostatique_p99_max": [float(df["hydrostatic_rel_err_p99"].max()),
                                         th["hydrostatic_rel_err_max"], "<="],
    }
    result = {}
    for name, (val, lim, op) in checks.items():
        ok = (val <= lim) if op == "<=" else (val >= lim)
        result[name] = {"valeur": val, "seuil": lim, "conforme": bool(ok) if np.isfinite(val) else None}

    info = {
        "instants": int(len(df)),
        "part_vitesse_100m_sup_12": float((df["speed_100m"] > 12).mean()),
        "part_vitesse_100m_inf_2": float((df["speed_100m"] < 2).mean()),
        "repartition_stabilite": df["stability"].value_counts(normalize=True).round(3).to_dict(),
        "repartition_secteurs": {f"{i * 45}deg": round(float(x), 3) for i, x in share.items()},
        "w_std_mediane": float(df["w_std"].median()),
        "w_std_p95": float(df["w_std"].quantile(0.95)),
    }
    summary = {"controles": result, "informations": info}
    with open(out_dir / "qc" / "qc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def _configurer(args) -> None:
    """Taille de grille, Coriolis et perturbations : variables globales du processus principal
    ET variables d'environnement, lues a l'import par les processus de calcul."""
    global NX_S, NY_S, NX_V, NY_V, F_CORIOLIS, A_ROT, A_DIV, PERTURB, DX_S, DX_V, Z_LEVELS, NZ
    if args.dx:
        os.environ["ANEO_GEN_DX"] = str(float(args.dx))
        DX_S = float(args.dx)
        DX_V = DX_S * STRIDE
    if args.niveaux:
        niv = sorted(float(z) for z in args.niveaux.split(","))
        if 100.0 not in niv or 200.0 not in niv:
            raise SystemExit("--niveaux doit contenir 100 et 200 m (controles de qualite)")
        os.environ["ANEO_GEN_NIVEAUX"] = ",".join(f"{z:g}" for z in niv)
        Z_LEVELS = np.array(niv)
        NZ = len(Z_LEVELS)
    if args.nx and args.ny:
        os.environ["ANEO_GEN_NX"], os.environ["ANEO_GEN_NY"] = str(args.nx), str(args.ny)
        NX_S, NY_S = args.nx, args.ny
        NX_V, NY_V = NX_S // STRIDE, NY_S // STRIDE
    if args.f_coriolis:
        os.environ["ANEO_GEN_F"] = str(args.f_coriolis)
        F_CORIOLIS = args.f_coriolis
    if args.perturbations is not None:
        os.environ["ANEO_GEN_PERTURB"] = str(args.perturbations)
        A_ROT, A_DIV = A_ROT / PERTURB * args.perturbations, A_DIV / PERTURB * args.perturbations
        PERTURB = args.perturbations


def generer_cache(args, tasks, split, static_path: Path, out_dir: Path) -> None:
    """Generation parallele vers le cache (--cache-dir). Reprise : un jour dont tous les
    instants sont deja dans index.csv est saute. L'index porte aussi le vent geostrophique et
    le cisaillement ERA5 de l'instant (entree grande echelle possible du modele reel)."""
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.csv"
    # stations reelles du domaine (grille_reelle.json a cote de la grille) -> cellules du
    # modele, pour l'entree « era5_stations » (pcfno3d.dataset)
    info = Path(args.input_grid).parent / "grille_reelle.json" if args.input_grid else None
    if info is not None and info.exists():
        st = json.loads(info.read_text(encoding="utf-8")).get("stations", {})
        cellules = [{"nom": n, "iy": int(v["iy"]) // STRIDE, "ix": int(v["ix"]) // STRIDE,
                     "lat": v["lat"], "lon": v["lon"], **({"code": v["code"]} if "code" in v else {})}
                    for n, v in st.items() if v.get("dans_la_zone")]
        (out_dir / "stations.json").write_text(json.dumps(cellules, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
        log(f"stations du domaine : {', '.join(c['nom'] for c in cellules)}")
    faits = set(pd.read_csv(index_path)["day"]) if index_path.exists() else set()
    jeu = {d: n for n, cle in (("train", "train_days"), ("val", "val_days"), ("test", "test_days"))
           for d in split[cle]}
    forcage = lire_forcage(args.forcage) if args.forcage else None
    restant = [t for t in tasks if t[0]["day_index"] not in faits]
    log(f"Cache {cache_dir} : {len(tasks) - len(restant)} jours deja faits, {len(restant)} a generer")

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                             initargs=(str(static_path),
                                       str(args.forcage) if args.forcage else None)) as ex:
        futs = [ex.submit(generate_day_cache, d, inst, sec, seed, str(out_dir), str(cache_dir),
                          not args.sans_pool)
                for d, inst, sec, seed in restant]
        for fut in as_completed(futs):
            idx, status, dur, rows = fut.result()
            df = pd.DataFrame(rows)
            df["split"] = jeu.get(idx, "test")
            if forcage is not None:
                f = forcage.reindex(pd.to_datetime(df["timestamp"]))
                for c in ("ug", "vg", "shear_u", "shear_v"):
                    df[f"era5_{c}"] = f[c].values if c in f else np.nan
            df.to_csv(index_path, mode="a", header=not index_path.exists(), index=False)
            done += 1
            eta = (time.time() - t0) / done * (len(restant) - done)
            log(f"Jour {idx:04d} : {status} ({dur:.0f} s, {len(rows)} instants) | "
                f"{done}/{len(restant)} | reste ~{eta / 60:.0f} min")

    summary = qc_summary(out_dir, split)
    if summary:
        for name, r in summary["controles"].items():
            flag = {True: "CONFORME", False: "NON CONFORME", None: "n/a"}[r["conforme"]]
            log(f"  {name:32s} {r['valeur']:.4g} (seuil {r['seuil']}) -> {flag}")
    log("Termine.")


def main():
    ap = argparse.ArgumentParser(description="Generateur du dataset ANEO 3D (Tukhnanutha)")
    ap.add_argument("--input-grid", type=Path, default=None,
                    help="aneo_grid_points.csv existant (500 000 points). Recommande.")
    ap.add_argument("--output-dir", type=Path, default=Path("aneo_dataset_3d"))
    ap.add_argument("--drive-dir", type=Path, default=None,
                    help="Dossier Drive ou deplacer chaque jour termine (libere le disque local)")
    ap.add_argument("--n-days", type=int, default=N_DAYS_DEFAULT)
    ap.add_argument("--days", type=str, default=None,
                    help="Sous-ensemble de jours, ex. '1-5' ou '1,7,12' (tests)")
    ap.add_argument("--max-instants", type=int, default=None, help="Limite d'instants par jour (tests)")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--nx", type=int, default=None, help="taille de la grille de surface (grille reelle)")
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--forcage", type=Path, default=None,
                    help="forcage ERA5 (scripts/forcage_era5.py) : jours reels au lieu de tirages")
    ap.add_argument("--test", default="", help="blocs de test, ex. 2022-01-01:2022-03-31,2022-07-01:2022-08-31")
    ap.add_argument("--val", default="", help="blocs de validation, meme format")
    ap.add_argument("--marge-jours", type=int, default=7, help="jours ecartes autour des blocs")
    ap.add_argument("--pas-train", type=int, default=1, help="un jour d'entrainement sur N")
    ap.add_argument("--pas-test", type=int, default=1, help="un jour de test sur N")
    ap.add_argument("--pas-val", type=int, default=1, help="un jour de validation sur N")
    ap.add_argument("--instants", type=int, default=None,
                    help="forcage : nombre d'instants repartis par saison et par heure "
                         "(calendrier_equilibre, fige dans calendrier.json)")
    ap.add_argument("--heures-par-jour", type=int, default=4)
    ap.add_argument("--jeux", default="train,val,test",
                    help="ensembles a generer dans ce passage (points de reprise)")
    ap.add_argument("--partie", default=None,
                    help="i/n : la i-eme des n parts (par numero de jour) des jours selectionnes")
    ap.add_argument("--debut", default=None)
    ap.add_argument("--fin", default=None)
    ap.add_argument("--perturbations", type=float, default=None,
                    help="facteur des perturbations aleatoires de petite echelle (forcage : 0,4)")
    ap.add_argument("--f-coriolis", type=float, default=None)
    ap.add_argument("--dx", type=float, default=None,
                    help="pas de la grille de surface reelle (m) ; defaut 50 ; national : 500")
    ap.add_argument("--niveaux", default=None,
                    help="niveaux des noeuds (m au-dessus de la mer), ex. 25,50,...,2500 ; defaut v7")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="ecrit directement le cache du modele (_S.npy / _Y.npy + index.csv) au "
                         "lieu des CSV journaliers ; la grille, meta.json et split_days.json "
                         "restent dans --output-dir")
    ap.add_argument("--sans-pool", action="store_true",
                    help="cache : un capteur sur STRIDE**2 au lieu de la moyenne par bloc "
                         "(doit suivre model.pool_surface de la config d'entrainement)")
    ap.add_argument("--csv-brut", action="store_true",
                    help="Fichiers d'observations en .csv non compresse (environ 4 fois plus volumineux)")
    args = ap.parse_args()
    _configurer(args)

    out_dir = args.output_dir
    (out_dir / "qc").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if _arrow_csv_options() is None:
        log("pyarrow absent ou trop ancien : ecriture pandas (plus lente). "
            "Pour accelerer : pip install -U pyarrow")

    # 1. Grille
    df_surface = load_or_build_grid(args.input_grid, rng)
    static = build_static(df_surface)
    grid_path = (args.drive_dir or out_dir) / "aneo_grid_points.csv"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    node_id_start = int(df_surface["point_id"].max()) + 1
    n_nodes = int(static["node_k"].size)
    if not grid_path.exists():
        log("Ecriture de la grille (surface + noeuds 3D)...")
        write_grid_file(df_surface, static, grid_path)
    log(f"Grille : {NX_S * NY_S:,} points de surface + {n_nodes:,} noeuds 3D -> {grid_path}")

    static_path = out_dir / "static_cache.npz"
    np.savez(static_path, **{k: v for k, v in static.items()})

    # 2. Calendrier et split
    if args.forcage and args.instants:
        cal = out_dir / "calendrier.json"
        if cal.exists():
            c = json.loads(cal.read_text(encoding="utf-8"))
            days = [{"day_index": d["day_index"], "date": dt.date.fromisoformat(d["date"])}
                    for d in c["days"]]
            split = c["split"]
            schedule = {int(k): [(t, None) for t in v] for k, v in c["schedule"].items()}
            log(f"Calendrier fige relu : {cal}")
        else:
            days, split, schedule = calendrier_equilibre(
                lire_forcage(args.forcage), args.test, args.val, args.marge_jours, args.instants,
                args.heures_par_jour)
            cal.write_text(json.dumps({
                "days": [{"day_index": d["day_index"], "date": d["date"].isoformat()} for d in days],
                "split": split,
                "schedule": {str(k): [t for t, _ in v] for k, v in schedule.items()},
                "parametres": {"instants": args.instants, "heures_par_jour": args.heures_par_jour,
                               "test": args.test, "val": args.val, "marge_jours": args.marge_jours},
            }, indent=1), encoding="utf-8")
            log(f"Calendrier equilibre fige : {cal}")
        forces = set(lire_forcage(args.forcage).index.normalize())
        manque = [d for d in days if pd.Timestamp(d["date"]) not in forces]
        minute_days, sec_days = [], []
        args.n_days = len(days)
        log(f"{len(days)} jours, {sum(len(v) for v in schedule.values())} instants (test "
            f"{len(split['test_days'])}, validation {len(split['val_days'])}, entrainement "
            f"{len(split['train_days'])} jours) ; {len(manque)} jours sans forcage pour l'instant")
        days = [d for d in days if pd.Timestamp(d["date"]) in forces]
    elif args.forcage:
        days, split, schedule = calendrier_era5(lire_forcage(args.forcage), args.test, args.val,
                                                args.marge_jours, args.pas_train, args.debut, args.fin,
                                                args.pas_test, args.pas_val)
        minute_days, sec_days = [], []
        args.n_days = len(days)
        log(f"Forcage ERA5 : {len(days)} jours reels (test {len(split['test_days'])}, "
            f"validation {len(split['val_days'])}, entrainement {len(split['train_days'])})")
    else:
        split = build_split(args.n_days)
        days, schedule, minute_days, sec_days = build_calendar(args.n_days, split, rng)
    n_sect = max([d["day_index"] for d in days] + [args.n_days])
    sectors = np.tile(np.arange(8), n_sect // 8 + 1)[: n_sect]
    rng.shuffle(sectors)

    selected = [d["day_index"] for d in days]
    if args.forcage:
        jeux = {f"{j.strip()}_days" for j in args.jeux.split(",") if j.strip()}
        garde = {i for k in jeux for i in split[k]}
        selected = [i for i in selected if i in garde]
        if args.partie:
            a, n = (int(x) for x in args.partie.split("/"))
            selected = [x for k, x in enumerate(selected) if k * n // len(selected) == a - 1]
    if args.days:
        sel = set()
        for part in args.days.split(","):
            if "-" in part:
                a, b = part.split("-")
                sel.update(range(int(a), int(b) + 1))
            else:
                sel.add(int(part))
        selected = [i for i in selected if i in sel]

    n_inst_total = sum(len(schedule[i]) for i in schedule)
    rows_per_inst = NX_S * NY_S + n_nodes
    meta = {
        "producteur": "Tukhnanutha",
        "version_generateur": 2,
        "grille_surface": {"nx": NX_S, "ny": NY_S, "dx_m": static["dx_s"], "dy_m": static["dy_s"]},
        "noeuds_3d": {"stride": STRIDE, "nx": NX_V, "ny": NY_V, "dx_m": static["dx_s"] * STRIDE,
                      "z_levels_asl_m": Z_LEVELS.tolist(), "min_z_rel_m": MIN_ZREL_NODE,
                      "point_id_debut": node_id_start, "nombre": n_nodes},
        "source_type": {"capteur": SRC_SURFACE, "interpole": SRC_INTERP, "noeud_3d": SRC_NODE},
        "conventions": {
            "u": "positif vers l'Est", "v": "positif vers le Nord", "w": "positif vers le haut",
            "direction_meteo": "(270 - atan2(v, u) en degres) mod 360, d'ou vient le vent",
            "hour_utc": "heure entiere ; minutes et secondes dans timestamp_utc",
        },
        "residu_de_masse": ("r(k+1/2) = (rho w[k+1] - rho w[k]) / dz_k + 0.5 (D[k] + D[k+1]), "
                            "D = d(rho u)/dx + d(rho v)/dy en differences centrees (np.gradient), "
                            "sur les noeuds 3D (pas horizontal = dx_m des noeuds, sans correction de pente)"),
        "calendrier": {"jours": args.n_days, "instants": n_inst_total,
                       "jours_episodes_minute": minute_days, "jours_episodes_10s": sec_days},
        "estimation_lignes_par_instant": rows_per_inst,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    with open(out_dir / "split_days.json", "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    if args.drive_dir:
        for name in ("meta.json", "split_days.json"):
            shutil.copy2(out_dir / name, args.drive_dir / name)

    compress = not args.csv_brut
    est_gb = n_inst_total * rows_per_inst * (30e-9 if compress else 130e-9)
    log(f"Calendrier : {args.n_days} jours, {n_inst_total} instants, ~{est_gb:.0f} Go estimes "
        f"({'csv.gz' if compress else 'csv brut'})")
    log(f"Jours a generer : {len(selected)} | workers : {args.workers}")

    # 3. Generation parallele par jour
    tasks = []
    for d in days:
        i = d["day_index"]
        if i not in selected:
            continue
        inst = schedule[i][: args.max_instants] if args.max_instants else schedule[i]
        tasks.append((d, inst, int(sectors[i - 1]), args.seed * 1000 + i))

    if args.cache_dir:
        generer_cache(args, tasks, split, static_path, out_dir)
        return

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                             initargs=(str(static_path),
                                       str(args.forcage) if args.forcage else None)) as ex:
        futs = [ex.submit(generate_day, d, inst, sec, seed, str(out_dir), node_id_start,
                          str(args.drive_dir) if args.drive_dir else None, compress)
                for d, inst, sec, seed in tasks]
        for fut in as_completed(futs):
            idx, status, dur = fut.result()
            done += 1
            eta = (time.time() - t0) / done * (len(tasks) - done)
            log(f"Jour {idx:03d} : {status} ({dur:.0f} s) | {done}/{len(tasks)} | reste ~{eta / 60:.0f} min")

    # 4. Controle qualite global
    summary = qc_summary(out_dir, split)
    if args.drive_dir and (out_dir / "qc").exists():
        shutil.copytree(out_dir / "qc", args.drive_dir / "qc", dirs_exist_ok=True)
    if summary:
        log("Controle qualite :")
        for name, r in summary["controles"].items():
            flag = {True: "CONFORME", False: "NON CONFORME", None: "n/a"}[r["conforme"]]
            log(f"  {name:32s} {r['valeur']:.4g} (seuil {r['seuil']}) -> {flag}")
        log(f"  details : {out_dir / 'qc' / 'qc_summary.json'}")
    log("Termine.")


if __name__ == "__main__":
    main()
