"""Sillage des eoliennes, modele de Jensen (cahier v12, BF72).

Chaque eolienne laisse derriere elle un sillage qui s'elargit lineairement (pente k) et dont le deficit de
vitesse decroit avec la distance (Jensen 1983, Katic et al. 1986) :

    deficit_j(s) = (1 - sqrt(1 - Ct_j)) / (1 + 2 k s / D_j)^2

ou s est la distance sous le vent. Une eolienne n'en recoit que la part de son disque rotor couverte par
le sillage (recouvrement de deux disques), et les deficits de plusieurs sillages se combinent en somme
quadratique. k = 0,075 a terre (0,04 en mer) ; le coefficient de poussee Ct vaut ~0,8 sous la vitesse
nominale et baisse au-dela (la pale se met en drapeau).

Les positions sont en metres (x vers l'est, y vers le nord), la direction est celle d'ou VIENT le vent
(convention meteo), comme partout dans pcfno3d.
"""
from __future__ import annotations

import math

import numpy as np

K_TERRE = 0.075
K_MER = 0.04


def ct_generique(v: np.ndarray, v_nominale: float = 11.0, cut_in: float = 3.0, cut_out: float = 25.0) -> np.ndarray:
    """Coefficient de poussee generique : 0,8 jusqu'a la vitesse nominale, puis 0,8 (v_n / v)^3 (puissance
    constante), 0 hors de la plage de production."""
    v = np.asarray(v, dtype=float)
    ct = np.where(v <= v_nominale, 0.8, 0.8 * (v_nominale / np.maximum(v, 1e-6)) ** 3)
    return np.where((v < cut_in) | (v > cut_out), 0.0, np.clip(ct, 0.0, 0.95))


def _recouvrement(d: float, r_rotor: float, r_sillage: float) -> float:
    """Part de l'aire du disque rotor (rayon r_rotor) couverte par le disque du sillage (rayon r_sillage),
    centres a la distance d."""
    if d >= r_rotor + r_sillage:
        return 0.0
    if d <= abs(r_sillage - r_rotor):
        return 1.0 if r_sillage >= r_rotor else (r_sillage / r_rotor) ** 2
    a = r_rotor ** 2 * math.acos((d * d + r_rotor ** 2 - r_sillage ** 2) / (2 * d * r_rotor))
    b = r_sillage ** 2 * math.acos((d * d + r_sillage ** 2 - r_rotor ** 2) / (2 * d * r_sillage))
    c = 0.5 * math.sqrt(max((-d + r_rotor + r_sillage) * (d + r_rotor - r_sillage)
                            * (d - r_rotor + r_sillage) * (d + r_rotor + r_sillage), 0.0))
    return (a + b - c) / (math.pi * r_rotor ** 2)


def jensen(x_m, y_m, diametres_m, vitesses_libres, direction_deg: float, ct=None, k: float = K_TERRE) -> np.ndarray:
    """Vitesses au moyeu AVEC sillages (m/s), pour un vent venant de `direction_deg`.

    x_m, y_m, diametres_m, vitesses_libres : une valeur par eolienne (vitesse sans sillage, au moyeu).
    ct : coefficients de poussee (par defaut ct_generique des vitesses libres)."""
    x, y = np.asarray(x_m, float), np.asarray(y_m, float)
    D, u = np.asarray(diametres_m, float), np.asarray(vitesses_libres, float)
    ct = ct_generique(u) if ct is None else np.asarray(ct, float)
    # vecteur unitaire dans le sens ou le vent VA (direction meteo + 180)
    a = math.radians((direction_deg + 180.0) % 360.0)
    ex, ey = math.sin(a), math.cos(a)
    n = len(x)
    deficit2 = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx, dy = x[i] - x[j], y[i] - y[j]
            s = dx * ex + dy * ey                         # distance sous le vent de j vers i
            if s <= 0.0:
                continue
            r = abs(dx * ey - dy * ex)                    # ecart lateral a l'axe du sillage
            rw = D[j] / 2.0 + k * s
            f = _recouvrement(r, D[i] / 2.0, rw)
            if f > 0.0 and ct[j] > 0.0:
                d = (1.0 - math.sqrt(1.0 - ct[j])) / (1.0 + 2.0 * k * s / D[j]) ** 2
                deficit2[i] += (d * f) ** 2
    return u * (1.0 - np.sqrt(deficit2))
