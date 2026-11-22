"""Candidat C2 de la regle configs/national/regle_national_gfs_v5_1.json : « ancrage GFS », un post-traitement du
champ du modele, sans entrainement.

A une hauteur h au-dessus du sol :

    u = u_gfs + a(h) (u_modele - u_gfs)        (meme chose pour v)

a(h) = a0 a 10 m, croit lineairement jusqu'a 1 a la hauteur H, vaut 1 au-dessus (le champ 3D n'est pas touche
au-dessus de H). Sous 10 m (noeuds bas des colonnes, noeuds sous le relief) a(h) = a0 : la regle ne definit a(h)
qu'a partir de 10 m, il n'est pas extrapole (choix ecrit ici, signale dans le compte rendu). u_gfs est le vent du
forcage GFS de l'instant AU NOEUD (`InterpERA5` : bilineaire dans le plan, lineaire en hauteur au-dessus du sol),
c'est-a-dire la base residuelle que le modele recoit ; a 10 m il vaut le « GFS brut » de la regle.

w est recalcule par continuite comme le service (`Geometry.w_servi`) : w_servi = w_c(u, v) + beta (w_reseau -
w_c(u, v)) avec w_c lineaire en (u, v) (densite standard), donc, pour le champ ancre,
w' = w + (1 - beta) w_c(u' - u, v' - v) ; sans `inference.w_continuite`, w (celui du reseau) est garde.

Grille de la regle : a0 dans {0,25 ; 0,5 ; 0,75}, H dans {100 m ; 300 m} (6 reglages, choisis sur la validation
2021 seulement, par scripts/choisir_national_gfs_v5.py). Aucun facteur propre a un aeroport.
"""
from __future__ import annotations

import numpy as np

H_BAS = 10.0
GRILLE_A0 = (0.25, 0.5, 0.75)
GRILLE_H = (100.0, 300.0)


def reglages() -> list[tuple[float, float]]:
    """Les 6 reglages (a0, H) de la regle, dans l'ordre de la grille."""
    return [(a0, h) for a0 in GRILLE_A0 for h in GRILLE_H]


def nom_reglage(famille: str, a0: float, H: float) -> str:
    """« c2_a0.25_H100 » (famille c2 ou c3)."""
    return f"{famille}_a{a0:g}_H{H:g}"


def coefficient(h, a0: float, H: float) -> np.ndarray:
    """a(h) : a0 jusqu'a 10 m, lineaire de a0 (10 m) a 1 (H), 1 au-dessus de H."""
    if not 0.0 <= float(a0) <= 1.0:
        raise ValueError(f"a0 doit etre dans [0, 1], recu {a0}")
    if not float(H) > H_BAS:
        raise ValueError(f"H doit depasser {H_BAS} m, recu {H}")
    x = np.clip((np.asarray(h, float) - H_BAS) / (float(H) - H_BAS), 0.0, 1.0)
    return float(a0) + (1.0 - float(a0)) * x


def ancrer(u_modele, v_modele, u_gfs, v_gfs, h, a0: float, H: float) -> tuple[np.ndarray, np.ndarray]:
    """u, v ancres (memes formes, diffusion numpy ; NaN propages)."""
    a = coefficient(h, a0, H)
    u_g, v_g = np.asarray(u_gfs, float), np.asarray(v_gfs, float)
    return u_g + a * (np.asarray(u_modele, float) - u_g), v_g + a * (np.asarray(v_modele, float) - v_g)


def hauteurs_noeuds(geom) -> np.ndarray:
    """(nz, ny, nx) hauteur au-dessus du sol de chaque noeud (z niveau - relief de la colonne)."""
    return np.asarray(geom.z_levels, float)[:, None, None] - np.asarray(geom.zt, float)[None]


def cellules_du_vecteur(geom, vecteur) -> np.ndarray:
    """Les cellules u, v du forcage dans le vecteur de l'instant (`ForcageERA5.vecteur` / `ForcageDirect`)."""
    from .forcage import N_SCAL
    n = geom.forcage.n_cellules
    v = np.asarray(vecteur, np.float32)
    return v[N_SCAL:N_SCAL + n]


def gfs_aux_noeuds(geom, vecteur) -> np.ndarray:
    """(2, nz, ny, nx) vent du forcage de l'instant a chaque noeud (la base residuelle du modele)."""
    return geom.interp_era5.dense_np(cellules_du_vecteur(geom, vecteur))


def colonne_gfs(geom, vecteur, iy: int, ix: int) -> np.ndarray:
    """(2, nz) vent du forcage aux noeuds de la colonne (iy, ix) : les memes poids que `gfs_aux_noeuds`, sans
    calculer tout le volume (evaluation aux aeroports)."""
    it = geom.interp_era5
    c = np.asarray(cellules_du_vecteur(geom, vecteur), np.float64).reshape(it.forme_cellules)
    jy, jx, a, b = int(it.iy[iy, ix]), int(it.ix[iy, ix]), float(it.a[iy, ix]), float(it.b[iy, ix])
    plan = ((1 - a) * (1 - b) * c[:, :, jy, jx] + (1 - a) * b * c[:, :, jy, jx + 1]
            + a * (1 - b) * c[:, :, jy + 1, jx] + a * b * c[:, :, jy + 1, jx + 1])        # (2, nh)
    k, t = it.k[:, iy, ix], it.t[:, iy, ix]                                                   # (nz,)
    return ((1 - t) * plan[:, k] + t * plan[:, k + 1]).astype(np.float32)


def ancrer_volume(geom, u, v, w, vecteur, a0: float, H: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Le champ 3D ancre (nz, ny, nx) : u, v par la formule, w par continuite comme le service. Passer le champ
    SERVI sans masque NaN (`predict(mask_nan=False)`) ; des NaN eventuels (noeuds sous le relief) comptent pour 0
    dans la continuite et restent NaN en sortie."""
    import torch

    g = gfs_aux_noeuds(geom, vecteur)
    h = hauteurs_noeuds(geom)
    u0, v0, w0 = (np.asarray(x, np.float64) for x in (u, v, w))
    u1, v1 = ancrer(u0, v0, g[0], g[1], h, a0, H)
    inf = geom.cfg.inference
    if not inf.w_continuite:
        return u1, v1, w0.copy()
    beta = float(getattr(inf, "w_beta", 0.0))
    nan = ~(np.isfinite(u0) & np.isfinite(v0) & np.isfinite(w0))
    du = np.nan_to_num(u1 - u0)
    dv = np.nan_to_num(v1 - v0)
    dt = geom.dz_t.dtype if hasattr(geom, "dz_t") else torch.float32
    with torch.no_grad():
        wc = geom.w_continuite(torch.as_tensor(du[None], dtype=dt, device=geom.device),
                               torch.as_tensor(dv[None], dtype=dt, device=geom.device))[0]
    w1 = w0 + (1.0 - beta) * wc.double().cpu().numpy()
    w1[nan] = np.nan
    return u1, v1, w1
