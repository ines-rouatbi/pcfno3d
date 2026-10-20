"""Validation du profil vertical sur de VRAIS radiosondages (IGRA : Tunis-Carthage, Tozeur).

Pour chaque radiosondage dont l'heure nominale est un instant du cache (00 ou 12 UTC), on
compare au ballon, a la meme hauteur au-dessus du sol, le vent reconstruit par le modele (a
partir des seuls anemometres) et la verite d'entrainement (ERA5-Land + ERA5) de la maille du
site. Les deux sont juges contre la mesure : l'ecart ERA5 - ballon est le plancher que le
modele, entraine sur ERA5, ne peut pas descendre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SITES = {"TSM00060715": ("Tunis-Carthage", 36.8507, 10.2147),
         "TSM00060760": ("Tozeur", 33.9167, 8.1667)}
TRANCHES = (0.0, 250.0, 500.0, 750.0, 1000.0)          # m au-dessus du sol


def maille(meta: dict, lat: float, lon: float, ny: int, nx: int) -> tuple[int, int]:
    """(iy, ix) de la maille la plus proche, grille reguliere de meta['georeference']."""
    g = meta["georeference"]
    pas = g["pas_deg"]
    iy = int(round((lat - g["lat_sud"]) / pas))
    ix = int(round((lon - g["lon_ouest"]) / pas))
    if not (0 <= iy < ny and 0 <= ix < nx):
        raise ValueError(f"({lat}, {lon}) hors de la grille")
    return iy, ix


def colonne_aux_hauteurs(h_noeuds: np.ndarray, u: np.ndarray, v: np.ndarray,
                         h_cibles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u, v d'une colonne (noeuds valides, hauteurs croissantes au-dessus du sol) interpoles
    lineairement aux hauteurs du ballon ; NaN hors de la colonne (pas d'extrapolation)."""
    ok = np.isfinite(h_noeuds) & np.isfinite(u) & np.isfinite(v)
    h, uu, vv = h_noeuds[ok], u[ok], v[ok]
    if len(h) < 2:
        return np.full(len(h_cibles), np.nan), np.full(len(h_cibles), np.nan)
    o = np.argsort(h)
    h, uu, vv = h[o], uu[o], vv[o]
    return (np.interp(h_cibles, h, uu, left=np.nan, right=np.nan),
            np.interp(h_cibles, h, vv, left=np.nan, right=np.nan))


def bilan(paires: pd.DataFrame) -> pd.DataFrame:
    """Par site et tranche de hauteur : n, erreur vectorielle RMS et biais de vitesse du
    modele et d'ERA5, tous deux contre le ballon."""
    p = paires.copy()
    p["tranche"] = pd.cut(p["h_agl_m"], TRANCHES)
    lignes = []
    for (site, tr), g in p.groupby(["site", "tranche"], observed=True):
        ligne = {"site": site, "tranche_m": str(tr), "n": len(g),
                 "radiosondages": g["temps"].nunique()}
        for src in ("modele", "era5"):
            du, dv = g[f"u_{src}"] - g["u_obs"], g[f"v_{src}"] - g["v_obs"]
            ligne[f"rmse_vect_{src}"] = float(np.sqrt(np.mean(du ** 2 + dv ** 2)))
            ligne[f"biais_vitesse_{src}"] = float(np.mean(
                np.hypot(g[f"u_{src}"], g[f"v_{src}"]) - np.hypot(g["u_obs"], g["v_obs"])))
        lignes.append(ligne)
    return pd.DataFrame(lignes)
