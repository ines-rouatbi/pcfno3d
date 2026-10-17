"""PC-FNO3D Tunisie : jeu ERA5 + METAR au format v7 (pcfno3d.tunisie_fno3d)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcfno3d import tunisie_fno3d as T

P_HPA = np.array([1000.0, 925.0, 850.0, 700.0])


def _isotherme(tv: float, t: int, ny: int, nx: int, z_1000: float = 100.0) -> np.ndarray:
    """Geopotentiel d'une atmosphere isotherme : phi = g z_1000 + R Tv ln(1000 / p)."""
    phi = T.G * z_1000 + T.R_AIR * tv * np.log(1000.0 / P_HPA)
    return np.broadcast_to(phi[None, :, None, None], (t, len(P_HPA), ny, nx)).copy()


def test_densite_hypsometrique_isotherme():
    rho = T.densite_niveaux(_isotherme(280.0, 2, 3, 4), P_HPA)
    attendu = P_HPA * 100.0 / (T.R_AIR * 280.0)
    assert np.allclose(rho[0, :, 1, 2], attendu, rtol=1e-9)


def _cas(t=2, ny=4, nx=5):
    zt = np.zeros((ny, nx))
    zt[2, 3] = 600.0
    lsm = np.ones((ny, nx))
    sl = {k: np.full((t, ny, nx), v) for k, v in
          (("u10", 3.0), ("v10", -1.0), ("u100", 5.0), ("v100", -2.0), ("t2m", 290.0),
           ("sp", 101000.0))}
    pl = {"p_hpa": P_HPA, "u": np.full((t, 4, ny, nx), 8.0), "v": np.full((t, 4, ny, nx), -3.0),
          "z": _isotherme(285.0, t, ny, nx, z_1000=120.0),
          "w": np.full((t, 4, ny, nx), -0.1)}                         # omega < 0 : ascendance
    return sl, pl, zt, lsm


def test_verite_noeuds_valides_et_conventions():
    sl, pl, zt, lsm = _cas()
    Y = T.verite_era5(sl, pl, zt, lsm, dx=25000.0, dy=28000.0)
    valid = T.noeuds_valides(zt)
    assert Y.shape == (2, 4, len(T.NIVEAUX_ASL), 4, 5)
    assert np.isfinite(Y).all() and (Y[:, :, ~valid] == 0).all()
    k10 = T.NIVEAUX_ASL.index(10.0)
    assert np.allclose(Y[:, 0, k10, 0, 0], 3.0) and np.allclose(Y[:, 1, k10, 0, 0], -1.0)
    k1000 = T.NIVEAUX_ASL.index(1000.0)
    assert np.allclose(Y[:, 0, k1000, 0, 0], 8.0)                    # niveaux de pression
    assert (Y[:, 2, k1000, 0, 0] > 0).all()                           # w = -omega / (rho g) > 0
    assert not valid[k10, 2, 3] and valid[T.NIVEAUX_ASL.index(1150.0), 2, 3]
    rho = Y[:, 3][:, valid]
    assert rho.min() > 1.0 and rho.max() < 1.3


def test_capteurs_anemometre_et_pression_ramenee():
    stations = pd.DataFrame({"station": ["A", "B"], "iy": [1, 1], "ix": [2, 2],
                             "elevation_m": [0.0, 0.0], "hauteur_anemometre_m": [10.0, 10.0],
                             "z_capteur_modele_m": [110.0, 110.0]})
    X = np.full((3, 2, 6), np.nan, np.float32)
    X[:, 0, :5] = [2.0, 4.0, 20.0, 1010.0, 60.0]
    X[:2, 1, :5] = [4.0, 0.0, 20.0, 1010.0, np.nan]                    # HR absente : 50 %
    S = T.surface_capteurs(X, stations, 4, 5)
    assert S.shape == (3, 8, 4, 5)
    assert S[:, 7, 1, 2].tolist() == [1.0, 1.0, 1.0] and S[:, 7].sum() == 3
    assert np.isclose(S[0, 0, 1, 2], 3.0) and np.isclose(S[2, 0, 1, 2], 2.0)   # moyenne puis seule
    assert (S[:, 2] == 0).all()                                        # l'anemometre ne mesure pas w
    assert S[0, 6, 1, 2] < 101000.0                                    # 100 m plus haut que la station
    assert 1.1 < S[0, 4, 1, 2] < 1.25


def test_jeu_lu_par_geometry(tmp_path):
    from pcfno3d.config import Config
    from pcfno3d.geometry import Geometry

    ny, nx = 6, 5
    lat = 30.0 + 0.25 * np.arange(ny)
    lon = 8.0 + 0.25 * np.arange(nx)
    zt = np.linspace(0, 900, ny * nx).reshape(ny, nx)
    meta = T.ecrire_jeu(tmp_path / "jeu", lat, lon, zt, np.ones((ny, nx)), np.full((ny, nx), 10.0))
    cfg = Config.load("tunisie_fno3d")
    cfg.apply_overrides({"paths.data_dir": tmp_path / "jeu", "paths.cache_dir": tmp_path / "c"})
    geom = Geometry(cfg, use_cache=False)
    assert geom.shape == (len(T.NIVEAUX_ASL), ny, nx)
    assert geom.valid.sum() == meta["noeuds_3d"]["nombre"] == T.noeuds_valides(zt).sum()
    assert np.allclose(geom.z_sensor, zt + 10.0)
    dx, dy = T.pas_metriques(lat, lon)                               # a 30,6 N
    assert np.isclose(geom.dx, dx) and np.isclose(geom.dy, dy) and 23900 < dx < 24050
