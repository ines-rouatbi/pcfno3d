"""Modele national « Tunisie ERA5 » : cible ERA5, entree METAR, modele, references.
Donnees fabriquees a valeurs connues ; aucun reseau."""

import numpy as np
import pandas as pd
import pytest

from pcfno3d import tunisie

H = len(tunisie.HAUTEURS_AGL)


def test_interpolation_en_hauteur_sans_extrapolation():
    h = np.array([10.0, 100.0, np.nan, 400.0, 1200.0])[:, None]     # un niveau sous le relief
    v = np.array([1.0, 2.0, 99.0, 5.0, 13.0])[:, None]
    r = tunisie.interpoler_hauteurs(h, v, tunisie.HAUTEURS_AGL)[:, 0]
    assert r[0] == pytest.approx(1.0) and r[1] == pytest.approx(2.0)
    assert r[2] == pytest.approx(2.0 + 150 / 300 * 3.0)               # 250 m entre 100 et 400 m
    assert r[3] == pytest.approx(5.0 + 100 / 800 * 8.0)               # 500 m
    assert r[5] == pytest.approx(5.0 + 600 / 800 * 8.0)               # 1000 m
    r = tunisie.interpoler_hauteurs(np.array([10.0, 100.0, 600.0])[:, None],
                                    np.array([1.0, 2.0, 3.0])[:, None], tunisie.HAUTEURS_AGL)[:, 0]
    assert np.isnan(r[4]) and np.isnan(r[5])                           # au-dessus du plus haut : absent


def _era5(nt=3):
    import xarray as xr

    t = pd.date_range("2025-01-01", periods=nt, freq="3h")
    lat, lon = [37.0, 36.75], [10.0, 10.25]
    orog = np.array([[0.0, 0.0], [0.0, 900.0]]) * tunisie.G             # une maille a 900 m
    pls = [1000, 975, 950, 925, 900, 875, 850]
    zpl = np.array([110, 320, 540, 760, 990, 1220, 1460], float)       # m au-dessus de la mer
    z = np.broadcast_to(zpl[None, :, None, None] * tunisie.G, (nt, 7, 2, 2)).copy()
    upl = np.broadcast_to((zpl / 100.0)[None, :, None, None], (nt, 7, 2, 2)).copy()
    sl = xr.Dataset({k: (("valid_time", "latitude", "longitude"), np.full((nt, 2, 2), val, float))
                     for k, val in (("u10", 1.0), ("v10", 0.5), ("u100", 1.0), ("v100", 0.5))},
                    coords={"valid_time": t, "latitude": lat, "longitude": lon})
    sl["z"] = (("valid_time", "latitude", "longitude"), np.broadcast_to(orog, (nt, 2, 2)).copy())
    pl = xr.Dataset({"u": (("valid_time", "pressure_level", "latitude", "longitude"), upl),
                     "v": (("valid_time", "pressure_level", "latitude", "longitude"), upl * 0),
                     "z": (("valid_time", "pressure_level", "latitude", "longitude"), z)},
                    coords={"valid_time": t, "pressure_level": pls[::-1], "latitude": lat,
                            "longitude": lon}).sortby("pressure_level")
    return sl, pl


def test_cible_era5_hauteurs_au_dessus_du_sol():
    c = tunisie.cible_era5(*_era5())
    assert c["U"].shape == (3, H, 2, 2) and c["orographie_m"][1, 1] == pytest.approx(900.0)
    plaine = c["U"][0, :, 0, 0]
    assert plaine[0] == pytest.approx(1.0) and plaine[1] == pytest.approx(1.0)      # 10 et 100 m
    # 250 m au-dessus d'un sol a 0 m : entre 1000 hPa (110 m) exclu (< 100 m ? non : 110 > 100)
    assert plaine[2] == pytest.approx(np.interp(250, [100, 110, 320], [1.0, 1.1, 3.2]))
    montagne = c["U"][0, :, 1, 1]                                       # sol a 900 m
    assert montagne[2] == pytest.approx(np.interp(250, [100, 320], [1.0, 12.2]))    # 1220 m asl
    assert np.isnan(montagne[4]) and np.isnan(montagne[5])             # 850 hPa = 560 m sol : rien au-dessus


def test_entree_stations_alignement_et_absence():
    obs = pd.DataFrame({"station": ["A", "A", "B"],
                        "t": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 03:20", "2025-01-01 01:10"]),
                        "u": [1.0, 2.0, 3.0], "v": [0.0, 0.0, 0.0], "temperature_c": [10.0, 11.0, 12.0],
                        "pression_hpa": [1010.0, 1011.0, 1012.0], "humidite_pct": [50.0, 60.0, 70.0]})
    temps = np.array(["2025-01-01T03:00", "2025-01-01T00:00"], dtype="datetime64[s]")
    X = tunisie.entree_stations(obs, temps, ["A", "B", "C"])
    assert X.shape == (2, 3, len(tunisie.VARIABLES_STATION))
    assert X[0, 0, 0] == 2.0 and X[1, 0, 0] == 1.0                     # ordre des instants respecte
    assert X[0, 1, -1] == 0.0 and X[1, 1, -1] == 0.0                    # B a 01:10 : hors tolerance
    assert (X[:, 2, -1] == 0).all()                                     # C jamais mesuree


def test_normalisation_absent_vaut_zero():
    X = np.zeros((4, 2, 6), np.float32)
    X[..., 0] = [[1, 3], [1, 3], [1, 3], [1, 3]]
    X[..., -1] = 1.0
    X[0, 1, :-1] = np.nan
    X[0, 1, -1] = 0.0
    Y = np.random.default_rng(0).normal(size=(4, 2 * H, 3, 3)).astype(np.float32)
    n = tunisie.Normalisation.depuis(X, Y)
    z = n.entree(X)
    assert np.isfinite(z).all() and (z[0, 1, :-1] == 0).all() and z[0, 1, -1] == 0


def test_modele_forme_perte_masquee_et_pannes():
    import torch

    m = tunisie.construire_modele(12, 6, 32, 19, 2 * H)
    x, tv = torch.randn(5, 12, 6), torch.randn(5, 4)
    p = m(x, tv)
    assert p.shape == (5, 2 * H, 32, 19)
    cible = torch.randn(5, 2 * H, 32, 19)
    cible[:, -1] = float("nan")
    assert torch.isfinite(tunisie.perte_masquee(p, cible))
    xm = tunisie.masquer_stations(torch.ones(200, 12, 6), 0.5, torch.Generator().manual_seed(0))
    part = 1 - xm[..., -1].mean().item()
    assert 0.15 < part < 0.35                                           # ~25 % en moyenne


def test_apprend_un_champ_lie_aux_stations():
    import torch

    rng = np.random.default_rng(1)
    n, S = 256, 3
    X = np.zeros((n, S, 6), np.float32)
    X[..., 0] = rng.normal(size=(n, S))
    X[..., -1] = 1.0
    Y = np.repeat(X[:, :1, 0][:, :, None, None], 2 * H, 1) * np.ones((1, 2 * H, 8, 6), np.float32)
    norm = tunisie.Normalisation.depuis(X, Y)
    torch.manual_seed(0)
    m = tunisie.construire_modele(S, 6, 8, 6, 2 * H, latent=32, base=8)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    xn = torch.from_numpy(norm.entree(X))
    yn = torch.from_numpy((Y - norm.y_moy[None, :, None, None]) / norm.y_ect[None, :, None, None])
    tv = torch.zeros(n, 4)
    debut = float(tunisie.perte_masquee(m(xn, tv), yn))
    for _ in range(150):
        loss = tunisie.perte_masquee(m(xn, tv), yn)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.1 * debut


def test_references_climatologie_et_idw():
    temps_tr = np.array(["2025-01-01T00", "2025-01-02T00", "2025-01-01T03"], dtype="datetime64[s]")
    Ytr = np.zeros((3, 2 * H, 2, 2), np.float32)
    Ytr[:2] = 2.0
    Ytr[2] = 5.0
    c = tunisie.reference_climatologie(Ytr, temps_tr, np.array(["2025-06-01T03", "2025-06-01T00"],
                                                                dtype="datetime64[s]"))
    assert c[0].mean() == 5.0 and c[1].mean() == 2.0
    X = np.zeros((1, 2, 6), np.float32)
    X[0, 0, :2] = [3.0, 0.0]
    X[0, 0, -1] = 1.0                                                   # B absente : ignoree
    X[0, 1, :2] = [100.0, 0.0]
    p = tunisie.reference_idw(X, [36.0, 35.0], [10.0, 9.0], np.array([36.0, 35.5]), np.array([10.0, 10.5]))
    assert p[0, 0, 0, 0] == pytest.approx(3.0, abs=1e-3)                 # 10 m sur la station
    assert p[0, H - 1, 0, 0] == pytest.approx(3.0 * 100 ** (1 / 7), rel=1e-3)   # 1000 m, loi puissance
    r = tunisie.rmse_par_niveau(p, p)
    assert all(v["vitesse"] == 0 for v in r.values())


def test_paires_horizon_prevision():
    t = np.array(["2025-01-01T00", "2025-01-01T03", "2025-01-01T06", "2025-01-01T12"],
                 dtype="datetime64[s]")
    s, c = tunisie.paires_horizon(t, 0)
    assert (s == c).all() and len(s) == 4
    s, c = tunisie.paires_horizon(t, 3)
    assert s.tolist() == [0, 1] and c.tolist() == [1, 2]            # 06 h -> 09 h absent
    s, c = tunisie.paires_horizon(t, 6)
    assert s.tolist() == [0, 2] and c.tolist() == [2, 3]


def test_niveaux_hauts_couvrent_1000_m_sur_le_relief():
    import xarray as xr

    sl, pl = _era5()
    z_haut = np.array([1960.0, 2200.0, 2460.0, 3010.0])                # 800 -> 700 hPa
    plh = xr.Dataset({k: (("valid_time", "pressure_level", "latitude", "longitude"),
                          np.broadcast_to(v[None, :, None, None], (3, 4, 2, 2)).copy())
                      for k, v in (("u", z_haut / 100.0), ("v", z_haut * 0), ("z", z_haut * tunisie.G))},
                     coords={"valid_time": sl["valid_time"], "pressure_level": [800, 775, 750, 700],
                             "latitude": sl["latitude"], "longitude": sl["longitude"]})
    c = tunisie.cible_era5(sl, xr.concat([pl, plh], dim="pressure_level"))
    montagne = c["U"][0, :, 1, 1]                                       # sol a 900 m
    assert np.isfinite(montagne).all()                                  # 1000 m sol = 1900 m mer
    assert montagne[5] == pytest.approx(np.interp(1900, [1460, 1960], [14.6, 19.6]))
