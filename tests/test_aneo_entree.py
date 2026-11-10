"""Entree ANEO du candidat D national (pcfno3d.aneo_entree) : degradation correlee et masquage par zones."""

import json
from types import SimpleNamespace

import numpy as np

from pcfno3d.aneo_entree import EntreeANEO


def _geom(tmp_path):
    lat_e, lon_e = np.arange(30.0, 33.01, 0.25), np.arange(8.0, 10.01, 0.25)
    nh, ny, nx = 3, 40, 30
    la = np.linspace(30.2, 32.8, ny)[:, None] + 0 * np.zeros((ny, nx))
    lo = np.linspace(8.2, 9.8, nx)[None, :] + 0 * np.zeros((ny, nx))
    fy = (la - lat_e[0]) / 0.25
    fx = (lo - lon_e[0]) / 0.25
    iy, ix = np.floor(fy).astype(int), np.floor(fx).astype(int)
    itp = SimpleNamespace(lat=la, lon=lo, iy=iy, ix=ix, a=fy - iy, b=fx - ix)
    fo = SimpleNamespace(lat=lat_e, lon=lon_e, forme_cellules=(2, nh, len(lat_e), len(lon_e)))
    zt = np.full((ny, nx), 100.0)
    zt[:, :5] = 0.0                                   # mer a l'ouest
    f = tmp_path / "deg.json"
    f.write_text(json.dumps({"sigma_u": 2.0, "sigma_v": 1.0, "L_km": 100.0}))
    return SimpleNamespace(forcage=fo, interp_era5=itp, zt=zt), f


def test_degradation_et_zone(tmp_path):
    g, f = _geom(tmp_path)
    e = EntreeANEO(g, f, p_zone=1.0, rayon_km=(50.0, 60.0))
    cel = np.ones(int(np.prod(g.forcage.forme_cellules)), np.float32)
    c, dn, zone = e.appliquer(cel, np.random.default_rng(0))
    c = c.reshape(g.forcage.forme_cellules)
    assert dn.shape == (2,) + g.zt.shape and zone.shape == g.zt.shape and zone.any()
    # meme perturbation a toutes les hauteurs hors zone ; profil moyen dans la zone
    assert np.allclose(c[0, 0] - c[0, 1], 0.0, atol=1e-5)
    # reproductible avec la meme graine
    c2, dn2, z2 = e.appliquer(cel, np.random.default_rng(0))
    assert np.array_equal(c.ravel(), c2) and np.array_equal(zone, z2)


def test_ecart_type_et_correlation(tmp_path):
    g, f = _geom(tmp_path)
    e = EntreeANEO(g, f, p_zone=0.0, rayon_km=(50.0, 150.0))
    cel = np.zeros(int(np.prod(g.forcage.forme_cellules)), np.float32)
    d = np.stack([e.appliquer(cel, np.random.default_rng(k))[0].reshape(g.forcage.forme_cellules)[:, 0]
                  for k in range(400)])                   # (n, 2, ny_e, nx_e)
    assert abs(d[:, 0].std() - 2.0) < 0.15 and abs(d[:, 1].std() - 1.0) < 0.08
    # deux mailles voisines (~25 km) plus correlees que deux mailles eloignees (~200 km)
    proche = np.corrcoef(d[:, 0, 5, 3], d[:, 0, 5, 4])[0, 1]
    loin = np.corrcoef(d[:, 0, 0, 0], d[:, 0, 8, 0])[0, 1]
    assert proche > 0.6 and loin < 0.35


def test_sans_degradation_rien_ne_change(tmp_path):
    g, _ = _geom(tmp_path)
    e = EntreeANEO(g, None, p_zone=0.0, rayon_km=(50.0, 150.0))
    cel = np.random.default_rng(3).normal(size=int(np.prod(g.forcage.forme_cellules))).astype(np.float32)
    c, dn, zone = e.appliquer(cel, np.random.default_rng(0))
    assert np.allclose(c, cel) and not dn.any() and zone is None


def test_capteurs_simules_a_25_m(tmp_path):
    g, _ = _geom(tmp_path)
    ny, nx = g.zt.shape
    g.nz, g.z_levels = 3, np.array([25.0, 150.0, 400.0])
    g.zrel_sensor = np.full((ny, nx), 10.0)
    g.valid = np.ones((3, ny, nx), bool)
    g.valid[0] = False                                 # 25 m ASL sous un relief de 100 m
    e = EntreeANEO(g, None, p_zone=0.0, rayon_km=(50.0, 150.0), n_capteurs=(20, 200))
    S = np.zeros((8, ny, nx), np.float32)
    S[0], S[1] = 5.0, -2.0                             # couche de surface a 10 m
    Y = np.zeros((4, 3, ny, nx), np.float32)
    Y[0, 1], Y[1, 1] = 10.0, -4.0                      # noeud a 150 m ASL = 50 m au-dessus du sol
    u, v = e.valeur_a_hauteur(S, Y, 10, 10)
    t = (np.log(25.0) - np.log(10.0)) / (np.log(50.0) - np.log(10.0))
    assert abs(u - (5 + 5 * t)) < 1e-5 and abs(v - (-2 - 2 * t)) < 1e-5
    e._cand = np.argwhere(g.zt > 1.0)                  # candidats de l'OS6 remplaces (geometrie factice)
    for k in range(20):
        cel, cu, cv = e.capteurs(S, Y, np.random.default_rng(k))
        assert 20 <= len(cel) <= 200 and len(np.unique(cel, axis=0)) == len(cel)
        assert (g.zt[cel[:, 0], cel[:, 1]] > 1.0).all()
    cel, cu, cv = e.capteurs(S, Y, np.random.default_rng(0), n=2000)
    sd = 0.1 + 0.02 * np.hypot(5 + 5 * t, 2 + 2 * t)
    assert abs((cu - (5 + 5 * t)).std() - sd) < 0.03 and abs(cu.mean() - (5 + 5 * t)) < 0.03
    assert e.capteurs(S, Y, np.random.default_rng(0), n=0) is None
    assert EntreeANEO(g, None, 0.0, (50.0, 150.0)).capteurs(S, Y, np.random.default_rng(0)) is None
