"""OI 3D de l'OS4 (regle_os4_oi_v4) : fonctions de calcul, sans donnees."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("os4v4", Path(__file__).resolve().parents[1] / "scripts" / "os4_oi_v4.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_un_point_au_sol_et_decroissance_verticale():
    al = m.alpha_oi(np.array([[0.0, 0.0]]), np.array([25.0]), np.array([[5.0, -5.0]]), 150.0, 4.0, 300.0)
    assert np.allclose(al, [[1.0, -1.0]])                                   # d / (1 + lambda)
    h = np.array([25.0, 325.0])[:, None]                                    # 2 niveaux, 1 colonne
    vert = {25.0: np.exp(-np.abs(h - 25.0) / 300.0)}
    inc = m.increment_3d(al, np.ones((1, 1)), np.array([25.0]), vert)
    assert np.allclose(inc[0, :, 0], [1.0, np.exp(-1.0)]) and np.allclose(inc[1, :, 0], [-1.0, -np.exp(-1.0)])


def test_points_a_plusieurs_hauteurs_somme_par_hauteur():
    yx = np.array([[0.0, 0.0], [0.0, 0.0], [100.0, 0.0]])
    z = np.array([25.0, 500.0, 25.0])
    d = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    al = m.alpha_oi(yx, z, d, 150.0, 4.0, 300.0)
    rho = np.array([[1.0, 0.5], [1.0, 0.5], [0.8, 0.9]])
    h = np.array([25.0, 400.0])[:, None] * np.ones((1, 2))
    vert = {zz: np.exp(-np.abs(h - zz) / 300.0) for zz in (25.0, 500.0)}
    inc = m.increment_3d(al, rho, z, vert)
    attendu = sum(al[j][:, None, None] * rho[j][None, None] * vert[z[j]][None] for j in range(3))
    assert np.allclose(inc, attendu)


def test_glouton_et_point_le_plus_eloigne():
    cand = np.array([[0, 0], [0, 400], [400, 0], [400, 400], [200, 200]], float)
    var = np.array([1.0, 0.5, 0.4, 0.3, 2.0])
    o = m.glouton(var, cand, 5, 150.0, 4.0)
    assert o[0] == 4 and len(set(o)) == 5
    assert list(m.plus_eloigne(cand, 3, 4))[0] == 4


def test_gains_marginaux_et_phases():
    cand = np.array([[0, 0], [0, 400], [400, 0], [400, 400], [200, 200]], float)
    o, g = m.glouton(np.array([1.0, 0.5, 0.4, 0.3, 2.0]), cand, 5, 150.0, 4.0, gains=True)
    assert len(o) == len(g) == 5 and (g >= 0).all() and g[0] >= g[-1]
    assert [m.phase(r) for r in (1, 10, 11, 25, 26, 100, 101, 400)] == [1, 1, 2, 2, 3, 4, 5, 6]


def test_metrique_pire_tranche():
    sse = np.array([[[4.0, 1.0], [9.0, 0.0]]])                   # 1 cle, 2 tranches, u v
    cnt = np.ones_like(sse)
    assert np.allclose(m.metrique(sse, cnt), [3.0])
