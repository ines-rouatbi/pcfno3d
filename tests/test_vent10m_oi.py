"""Couche 10 m « fond + OI des aeroports » (regle_produit_vent10m_v4)."""
import numpy as np

from pcfno3d.vent10m_oi import couche_10m, poids_oi


def _fond():
    return np.zeros((2, 40, 60))


def test_sans_mesure_le_fond_est_rendu():
    fond = _fond() + 3.0
    v, inc = couche_10m(fond, {"A": (10, 10)}, {}, 1000.0, 1000.0, biais={})
    assert np.array_equal(v, fond) and not inc.any()


def test_un_point_rend_d_sur_1_plus_lambda_et_decroit():
    v, inc = couche_10m(_fond(), {"A": (20, 30)}, {"A": (5.0, -5.0)}, 1000.0, 1000.0, biais={})
    assert np.isclose(inc[0, 20, 30], 5.0 / 5.0) and np.isclose(inc[1, 20, 30], -1.0)
    assert abs(inc[0, 0, 0]) < inc[0, 20, 30]


def test_biais_retire_et_points_absents_ignores():
    _, inc = couche_10m(_fond(), {"A": (20, 30), "B": (5, 5)}, {"A": (2.0, 0.0), "B": (np.nan, 1.0)},
                        1000.0, 1000.0, biais={"A": (2.0, 0.0)})
    assert np.allclose(inc, 0.0)


def test_identique_au_jugement_hors_grille():
    # meme formule que scripts/oi_post_aeroports.oi_autres (L 150 km, lambda 4) aux points eux-memes
    iy, ix = np.array([0, 50, 120]), np.array([0, 80, 10])
    W = poids_oi(iy, ix, 1000.0, 1000.0)
    d = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
    fond = np.zeros((2, 130, 100))
    _, inc = couche_10m(fond, {k: (int(a), int(b)) for k, a, b in zip("abc", iy, ix)},
                        {k: tuple(x) for k, x in zip("abc", d)}, 1000.0, 1000.0, biais={})
    dist = np.hypot(iy[:, None] - iy[None], ix[:, None] - ix[None])
    rho = np.exp(-dist ** 2 / (2 * 150.0 ** 2))
    attendu = rho @ (W @ d)
    assert np.allclose(inc[:, iy, ix].T, attendu)


def test_atom_wind_instant_remplace_le_niveau_10m(geom):
    from types import SimpleNamespace

    from pcfno3d import atom_wind

    nz, ny, nx = geom.nz, geom.ny, geom.nx
    vol = SimpleNamespace(u=np.where(geom.valid, 4.0, np.nan), v=np.where(geom.valid, -2.0, np.nan))
    sans = atom_wind.instant(vol, geom)
    c10 = (np.full((ny, nx), 7.0), np.full((ny, nx), 1.0))
    avec = atom_wind.instant(vol, geom, c10)
    i10 = atom_wind.HAUTEURS_AGL.index(10)
    colonne = geom.valid.any(axis=0).any()
    assert colonne
    assert np.nanmax(np.abs(avec[i10, 0] - 7.0)) < 1e-6 and np.nanmax(np.abs(avec[i10, 1] - 1.0)) < 1e-6
    autres = [k for k in range(len(atom_wind.HAUTEURS_AGL)) if k != i10]
    assert np.array_equal(np.nan_to_num(avec[autres]), np.nan_to_num(sans[autres]))


def test_config_couche_10m_oi(cfg):
    from pcfno3d.config import InferenceCfg
    assert InferenceCfg().couche_10m_oi is False
    assert InferenceCfg(couche_10m_oi=True).couche_10m_oi is True
