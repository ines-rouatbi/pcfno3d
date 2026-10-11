"""BLOC 6 (1re partie) — caracteristiques temps / physique estimees depuis la surface.

Le modele ne voit que les capteurs visibles. Pour lui donner l'etat synoptique a grande
echelle, on ajuste un plan sur la pression et la temperature des capteurs visibles et on
en deduit le vent geostrophique et le vent thermique. Ces 8 nombres sont recopies en
chaque point du volume.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import Geometry

G = 9.80665          # m/s2
R_AIR = 287.05       # J/(kg.K)
LAPSE = 0.0065       # K/m, gradient standard
MIN_SENSORS = 100    # en dessous, l'ajustement du plan n'est pas fiable


def geostrophic_from_surface(S: np.ndarray, sensor: np.ndarray,
                             geom: Geometry) -> tuple[float, float]:
    """Vent geostrophique : pression reduite au niveau de la mer, plan ajuste, equilibre.

    ug = -(1 / rho f) dp/dy ; vg = (1 / rho f) dp/dx
    """
    m = sensor
    n = int(m.sum())
    if n < MIN_SENSORS:
        return 0.0, 0.0
    p = S[6][m].astype(np.float64)
    t_k = S[3][m].astype(np.float64) + 273.15
    z = geom.z_sensor[m]
    # formule barometrique : supprime l'effet de l'altitude du capteur
    p_sl = p * np.exp(G * z / (R_AIR * (t_k + 0.5 * LAPSE * z)))
    A = np.stack([np.ones(n), geom.xc[m], geom.yc[m]], axis=1)
    coef, *_ = np.linalg.lstsq(A, p_sl, rcond=None)     # [a, dp/dx, dp/dy]
    rho = float(S[4][m].mean())
    f = geom.cfg.f_coriolis
    return -coef[2] / (rho * f), coef[1] / (rho * f)


def thermal_wind_from_surface(S: np.ndarray, sensor: np.ndarray,
                              geom: Geometry) -> tuple[float, float]:
    """Vent thermique (variation du vent geostrophique avec l'altitude, en s-1).

    d(ug, vg)/dz = (g / (f T)) * (-dT/dy, dT/dx), avec T ramenee au niveau de la mer.
    """
    m = sensor
    n = int(m.sum())
    if n < MIN_SENSORS:
        return 0.0, 0.0
    t_red = S[3][m].astype(np.float64) + LAPSE * geom.z_sensor[m]
    A = np.stack([np.ones(n), geom.xc[m], geom.yc[m]], axis=1)
    coef, *_ = np.linalg.lstsq(A, t_red, rcond=None)    # [a, dT/dx, dT/dy]
    k = G / (geom.cfg.f_coriolis * (float(t_red.mean()) + 273.15))
    return -k * coef[2], k * coef[1]


def time_geo_features(S: np.ndarray, sensor: np.ndarray, hour: float, doy: int,
                      geom: Geometry) -> np.ndarray:
    """Les 8 canaux : sin/cos de l'heure, sin/cos du jour, vent geostrophique, vent thermique."""
    ug, vg = geostrophic_from_surface(S, sensor, geom)
    su, sv = thermal_wind_from_surface(S, sensor, geom)
    return np.array([
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365),
        ug / 10.0, vg / 10.0, su * 200.0, sv * 200.0,
    ], dtype=np.float32)
