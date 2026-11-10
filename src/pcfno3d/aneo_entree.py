"""Entree « ANEO » du candidat D national (configs/national/regle_v4_national.json), ENTRAINEMENT seulement.

Deux transformations des ENTREES (jamais des predictions ni de la cible) :

1. degradation d'ERA5 : un champ gaussien (u, v independants), d'ecart-type sigma_u, sigma_v et de
   correlation spatiale exp(-d / L), est ajoute aux mailles ERA5 de l'instant, a toutes les hauteurs
   (entree dense ET base du residuel, qui en sont interpolees), et au meme point au vent ERA5 a 10 m du
   treillis d'entree (interpolation bilineaire des memes poids qu'InterpERA5). sigma_u, sigma_v et L
   viennent de scripts/degradation_era5.py : ecart ERA5 10 m - METAR aux 12 aeroports sur les jours
   d'ENTRAINEMENT, correlation entre paires d'aeroports en fonction de la distance.
2. masquage par zones : avec la probabilite p_zone, un disque (rayon tire dans rayon_km, centre tire sur
   un noeud de terre) ou les points ERA5 a 10 m du treillis sont retires et ou les mailles ERA5 dont le
   centre est dans le disque prennent le profil moyen du domaine a l'instant (la transition au bord du
   disque est donc celle de l'interpolation bilineaire, une maille). Les cellules des stations restent.
3. capteurs ANEO simules (ajout d'Ines, 10/11) : a chaque exemple, n tire uniformement dans n_capteurs
   (20 a 200) mailles terrestres parmi les candidats de l'OS6 (`os6.candidats`, contraintes par defaut :
   pente <= 20 deg, 1 km du bord du domaine et des aeroports, terre z0 >= 0,005 m, altitude <= 650 m).
   Valeur = verite d'entrainement interpolee a `hauteur_capteur_m` (25 m) au-dessus du sol dans la colonne
   (couche de surface du cache a z_rel du capteur + noeuds valides, interpolation lineaire en ln h), plus
   un bruit de mesure gaussien par composante d'ecart-type 0,1 m/s + 2 % de la vitesse. Entree comme
   une station (cellule du treillis, prioritaire sur le pseudo-capteur ERA5, gardee dans une zone masquee).

Validation, test et production : rien de tout cela (le Dataset ne l'appelle qu'a l'entrainement), sauf
l'evaluation d'information de la regle v4 (scripts/info_capteurs_v4.py), qui passe sa propre EntreeANEO.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class EntreeANEO:
    SIGMA_CAPTEUR = (0.1, 0.02)                    # m/s + part de la vitesse, par composante

    def __init__(self, geom, chemin_json: str | Path | None, p_zone: float,
                 rayon_km: tuple[float, float], n_capteurs: tuple[int, int] = (0, 0),
                 hauteur_capteur_m: float = 25.0) -> None:
        from .forcage import metres_par_degre
        fo, itp = geom.forcage, geom.interp_era5
        if fo is None or itp is None:
            raise ValueError("entree ANEO : paths.forcage_era5 requis")
        self.forme = fo.forme_cellules                      # (2, nh, ny_e, nx_e)
        self.p_zone = float(p_zone)
        self.rayon_km = tuple(float(r) for r in rayon_km)
        self.itp = itp
        lat0 = float(np.mean(itp.lat))
        m_lon, m_lat = metres_par_degre(lat0)
        # coordonnees planes (km) des centres de mailles et des noeuds
        LA, LO = np.meshgrid(fo.lat, fo.lon, indexing="ij")
        self.ye, self.xe = (LA - lat0) * m_lat / 1000.0, (LO - fo.lon[0]) * m_lon / 1000.0
        self.yn, self.xn = (itp.lat - lat0) * m_lat / 1000.0, (itp.lon - fo.lon[0]) * m_lon / 1000.0
        self.terre = np.argwhere(np.asarray(geom.zt) > 1.0)
        self.n_capteurs = tuple(int(n) for n in n_capteurs)
        self.h_capteur = float(hauteur_capteur_m)
        self.geom = geom
        self._cand = None
        self.sigma = None
        if chemin_json:
            p = json.loads(Path(chemin_json).read_text(encoding="utf-8"))
            self.sigma = (float(p["sigma_u"]), float(p["sigma_v"]))
            self.L_km = float(p["L_km"])
            pts = np.stack([self.ye.ravel(), self.xe.ravel()], 1)
            d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
            self.chol = np.linalg.cholesky(np.exp(-d / self.L_km) + 1e-6 * np.eye(len(pts)))

    def _aux_noeuds(self, champ: np.ndarray) -> np.ndarray:
        it = self.itp
        return ((1 - it.a) * (1 - it.b) * champ[it.iy, it.ix] + (1 - it.a) * it.b * champ[it.iy, it.ix + 1]
                + it.a * (1 - it.b) * champ[it.iy + 1, it.ix] + it.a * it.b * champ[it.iy + 1, it.ix + 1])

    def appliquer(self, cellules: np.ndarray, rng: np.random.Generator):
        """cellules (n_cellules,) de l'instant -> (cellules modifiees, delta du vent a 10 m aux noeuds
        (2, ny, nx), zone retiree du treillis (ny, nx) bool ou None)."""
        c = np.asarray(cellules, np.float64).reshape(self.forme).copy()
        ny_e, nx_e = self.forme[2:]
        delta_n = np.zeros((2,) + self.yn.shape)
        if self.sigma is not None:
            for k in (0, 1):
                d = (self.chol @ rng.standard_normal(ny_e * nx_e)).reshape(ny_e, nx_e) * self.sigma[k]
                c[k] += d[None]                             # meme perturbation a toutes les hauteurs
                delta_n[k] = self._aux_noeuds(d)
        zone = None
        if self.p_zone > 0 and len(self.terre) and rng.random() < self.p_zone:
            iy, ix = self.terre[rng.integers(len(self.terre))]
            r = rng.uniform(*self.rayon_km)
            cy, cx = self.yn[iy, ix], self.xn[iy, ix]
            dedans = np.hypot(self.ye - cy, self.xe - cx) <= r
            moyen = c.mean(axis=(2, 3), keepdims=True)      # profil moyen du domaine (2, nh, 1, 1)
            c = np.where(dedans[None, None], moyen, c)
            zone = np.hypot(self.yn - cy, self.xn - cx) <= r
        return c.ravel().astype(np.float32), delta_n.astype(np.float32), zone


    def candidats(self) -> np.ndarray:
        if self._cand is None:
            from .os6 import Contraintes, candidats
            self._cand = candidats(self.geom, Contraintes())
            if not len(self._cand):
                raise ValueError("capteurs ANEO simules : aucun candidat (contraintes de l'OS6)")
        return self._cand

    def valeur_a_hauteur(self, S: np.ndarray, Y: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
        """u, v de la verite a h_capteur au-dessus du sol dans la colonne (iy, ix) : couche de surface
        du cache (a z_rel du capteur) et noeuds valides plus hauts, lineaire en ln h."""
        g = self.geom
        h = [float(g.zrel_sensor[iy, ix])]
        uv = [(float(S[0, iy, ix]), float(S[1, iy, ix]))]
        zt = float(g.zt[iy, ix])
        for k in range(g.nz):
            hk = float(g.z_levels[k]) - zt
            if g.valid[k, iy, ix] and hk > h[-1]:
                h.append(hk)
                uv.append((float(Y[0, k, iy, ix]), float(Y[1, k, iy, ix])))
        lh, a = np.log(np.maximum(h, 0.1)), np.asarray(uv)
        x = np.log(self.h_capteur)
        return float(np.interp(x, lh, a[:, 0])), float(np.interp(x, lh, a[:, 1]))

    def capteurs(self, S: np.ndarray, Y: np.ndarray, rng: np.random.Generator, n: int | None = None):
        """(cellules (n, 2), u (n,), v (n,)) des capteurs simules de l'exemple, ou None."""
        if n is None:
            if self.n_capteurs[1] <= 0:
                return None
            n = int(rng.integers(self.n_capteurs[0], self.n_capteurs[1] + 1))
        if n <= 0:
            return None
        cand = self.candidats()
        cel = cand[rng.choice(len(cand), size=min(n, len(cand)), replace=False)]
        u, v = np.empty(len(cel)), np.empty(len(cel))
        a, b = self.SIGMA_CAPTEUR
        for i, (iy, ix) in enumerate(cel):
            uu, vv = self.valeur_a_hauteur(S, Y, int(iy), int(ix))
            sd = a + b * float(np.hypot(uu, vv))
            u[i], v[i] = uu + rng.normal(0.0, sd), vv + rng.normal(0.0, sd)
        return cel, u.astype(np.float32), v.astype(np.float32)


def entree_aneo(geom):
    """L'EntreeANEO de la config, construite une fois par geometrie (None si desactivee)."""
    m = geom.cfg.mask
    if not m.aneo_degradation and m.aneo_p_zone <= 0 and m.aneo_capteurs[1] <= 0:
        return None
    e = getattr(geom, "_entree_aneo", None)
    if e is None:
        e = EntreeANEO(geom, m.aneo_degradation, m.aneo_p_zone, m.aneo_rayon_km,
                       m.aneo_capteurs, m.aneo_hauteur_capteur_m)
        geom._entree_aneo = e
    return e
