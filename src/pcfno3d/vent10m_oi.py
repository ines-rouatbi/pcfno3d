"""Couche de vent a 10 m « fond + OI des mesures des aeroports » du produit national v4
(configs/national/regle_produit_vent10m_v4.json, option i d'Ines du 15/11/2026).

    vent_10m(x) = fond(x) + sum_j alpha_j rho(d(x, j)),   alpha = (rho(D) + lambda I)^-1 (d - b)

d_j = mesure a l'aeroport j - fond a sa cellule, b_j = biais 2015-2020 de l'aeroport j, rho gaussienne de longueur
L. L = 150 km et lambda = 4 ont ete choisis sur 2015-2020 (aeroport retire) et juges sur 2023-2024, periode jamais
utilisee : -3,7 % sur l'ERA5 aux 12 aeroports (docs/resultats/national/independant_2023_2024.md). Le champ 3D du
modele n'est pas touche. Distances dans le plan local de la grille (pas constant dx, dy).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

L_KM = 150.0
LAMBDA = 4.0
BIAIS = Path(__file__).resolve().parents[2] / "configs" / "national" / "biais_innovations_aeroports_2015_2020.json"


def charger_biais(chemin: str | Path = BIAIS) -> dict[str, tuple[float, float]]:
    b = json.loads(Path(chemin).read_text(encoding="utf-8"))["biais"]
    return {c: (float(v["du_moyen"]), float(v["dv_moyen"])) for c, v in b.items()}


def poids_oi(iy, ix, dx_m: float, dy_m: float, L_km: float = L_KM, lam: float = LAMBDA) -> np.ndarray:
    """(rho(D) + lambda I)^-1 pour les aeroports mesures (indices de cellules)."""
    iy, ix = np.asarray(iy, float), np.asarray(ix, float)
    d = np.hypot((iy[:, None] - iy[None]) * dy_m, (ix[:, None] - ix[None]) * dx_m) / 1000.0
    return np.linalg.inv(np.exp(-d ** 2 / (2 * L_km ** 2)) + lam * np.eye(len(iy)))


def couche_10m(fond: np.ndarray, cellules: dict[str, tuple[int, int]], mesures: dict[str, tuple[float, float]],
               dx_m: float, dy_m: float, biais: dict[str, tuple[float, float]] | None = None,
               L_km: float = L_KM, lam: float = LAMBDA) -> tuple[np.ndarray, np.ndarray]:
    """fond (2, ny, nx) u, v a 10 m ; cellules {code: (iy, ix)} ; mesures {code: (u, v)} de l'instant (absentes
    ou non finies ignorees). -> (vent 10 m corrige (2, ny, nx), increment (2, ny, nx)). NaN du fond conserves."""
    fond = np.asarray(fond, np.float64)
    biais = charger_biais() if biais is None else biais
    codes = [c for c in cellules if c in mesures and np.all(np.isfinite(mesures[c]))]
    inc = np.zeros_like(fond)
    if not codes:
        return fond.copy(), inc
    iy = np.array([cellules[c][0] for c in codes])
    ix = np.array([cellules[c][1] for c in codes])
    d = np.array([[mesures[c][k] - fond[k, cellules[c][0], cellules[c][1]] - biais.get(c, (0.0, 0.0))[k]
                   for k in (0, 1)] for c in codes])                                   # (n, 2)
    ok = np.isfinite(d).all(axis=1)
    if not ok.any():
        return fond.copy(), inc
    iy, ix, d = iy[ok], ix[ok], d[ok]
    alpha = poids_oi(iy, ix, dx_m, dy_m, L_km, lam) @ d                                # (n, 2)
    ny, nx = fond.shape[1:]
    gy = (np.arange(ny)[:, None] * dy_m)
    gx = (np.arange(nx)[None, :] * dx_m)
    for j in range(len(iy)):
        r2 = ((gy - iy[j] * dy_m) ** 2 + (gx - ix[j] * dx_m) ** 2) / 1e6
        rho = np.exp(-r2 / (2 * L_km ** 2))
        inc[0] += alpha[j, 0] * rho
        inc[1] += alpha[j, 1] * rho
    return fond + inc, inc


def couche_service(geom, fond: np.ndarray, mesures: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Couche 10 m d'un instant pour le service : fond (2, ny, nx) + OI des stations du domaine mesurees
    (cellules de stations.json). -> (u10, v10), NaN hors du domaine valide comme le fond."""
    from .stations_reelles import cellules_codes
    v, _ = couche_10m(fond, cellules_codes(geom), mesures, float(geom.dx), float(geom.dy))
    return v[0], v[1]
