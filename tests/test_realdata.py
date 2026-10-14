"""Donnees reelles -> cache du pipeline (pcfno3d.realdata).

Aucune vraie donnee n'est disponible a ce jour : on fabrique des stations, un champ maille
et une radiosonde a valeurs CONNUES sur la geometrie du mini-dataset, et on verifie que
chaque valeur arrive au bon noeud, avec la bonne valeur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

LAT0, LON0 = 36.40, 10.10          # ancrage arbitraire (le domaine reel reste a fournir)


@pytest.fixture()
def geo():
    from pcfno3d.realdata import GeoRef
    return GeoRef(LAT0, LON0)


@pytest.fixture()
def rcfg(cfg, tmp_path):
    """Meme geometrie que le mini-dataset, cache SEPARE (ne pas melanger avec le synthetique)."""
    from pcfno3d.config import Config
    c = Config.load("test")
    c.apply_overrides({"paths.data_dir": cfg.paths.data_dir,
                       "paths.cache_dir": tmp_path / "cache_reel",
                       "paths.run_dir": tmp_path / "run_reel",
                       "real.origin_lat": LAT0, "real.origin_lon": LON0})
    c.paths.mkdirs()
    return c


def _node_ll(geom, geo, j, i):
    lat, lon = geo.to_latlon(i * geom.dx, j * geom.dy)
    return float(lat), float(lon)


# ------------------------------------------------------------------ geographie, physique
def test_georef_aller_retour_et_distances(geo):
    from pcfno3d.realdata import GeoRef
    lat, lon = geo.to_latlon(np.array([0.0, 1000.0, 0.0]), np.array([0.0, 0.0, 1000.0]))
    x, y = geo.to_xy(lat, lon)
    np.testing.assert_allclose(x, [0, 1000, 0], atol=1e-6)
    np.testing.assert_allclose(y, [0, 0, 1000], atol=1e-6)
    assert lon[1] > LON0 and lat[2] > LAT0                  # x vers l'est, y vers le nord
    rot = GeoRef(LAT0, LON0, rotation_deg=90.0)             # axe x du domaine vers le nord
    lat_r, lon_r = rot.to_latlon(1000.0, 0.0)
    assert abs(float(lon_r) - LON0) < 1e-9 and float(lat_r) > LAT0


def test_domaine_non_georeference_refuse(cfg):
    from pcfno3d.realdata import GeoRef
    with pytest.raises(ValueError, match="origin_lat"):
        GeoRef.from_cfg(cfg)


def test_conventions_physiques():
    from pcfno3d.realdata import isa, moist_density, wind_components
    u, v = wind_components([10.0, 10.0], [0.0, 270.0])     # vent DU nord, vent D'ouest
    np.testing.assert_allclose(u, [0.0, 10.0], atol=1e-9)
    np.testing.assert_allclose(v, [-10.0, 0.0], atol=1e-9)
    p, t, rho = isa(0.0)
    assert abs(p - 101325) < 1 and abs(t - 15) < 1e-9 and abs(rho - 1.225) < 0.001
    assert moist_density(101325, 15, 80) < moist_density(101325, 15, 0)   # l'eau allege


# ------------------------------------------------------------------ stations -> S
def test_stations_dans_la_bonne_maille_moyennees_et_controlees(rcfg, geom, geo):
    from pcfno3d.realdata import normalize_stations, stations_to_surface
    z_ref = float(np.nanmedian(geom.zrel_sensor))
    j, i = geom.ny // 2, geom.nx // 2
    lat, lon = _node_ll(geom, geo, j, i)
    lat_out, lon_out = _node_ll(geom, geo, -5, -5)
    df = pd.DataFrame({
        "station_id": ["A", "B", "C", "D", "E"],
        "timestamp_utc": ["2026-01-01T12:00:00Z"] * 5,
        "lat": [lat, lat + 1e-5, lat_out, lat, lat],
        "lon": [lon, lon, lon_out, lon, lon],
        "height_agl_m": [z_ref] * 5,
        "wind_u_m_s": [4.0, 6.0, 1.0, np.nan, 200.0],
        "wind_v_m_s": [1.0, 3.0, 1.0, np.nan, 0.0],
        "temperature_c": [20.0, 22.0, 20.0, 20.0, 20.0],
        "quality": [0.9, 0.9, 0.9, 0.9, 0.9],
    })
    clean, rapport = normalize_stations(df, rcfg)
    assert rapport["ecartees_vent_absent"] == 1 and rapport["ecartees_vent_aberrant"] == 1
    S, info = stations_to_surface(clean, geom, geo, rcfg)
    assert info == {"stations": 2, "hors_domaine": 1, "mailles": 1,
                    "hauteur_ref_m": round(z_ref, 1)}
    assert S[7].sum() == 1 and S[7, j, i] == 1
    np.testing.assert_allclose(S[:2, j, i], [5.0, 2.0], rtol=1e-5)      # A et B moyennees
    assert abs(S[3, j, i] - 21.0) < 1e-4 and 1.0 < S[4, j, i] < 1.3    # T, rho calculee


def test_vent_ramene_a_la_hauteur_des_capteurs(rcfg, geom, geo):
    from pcfno3d.realdata import normalize_stations, stations_to_surface
    j, i = 1, 1
    lat, lon = _node_ll(geom, geo, j, i)
    df = pd.DataFrame({"station_id": ["A"], "timestamp_utc": ["2026-01-01T00:00:00"],
                       "lat": [lat], "lon": [lon], "height_agl_m": [10.0],
                       "wind_speed_m_s": [5.0], "wind_dir_deg": [270.0]})
    S, info = stations_to_surface(normalize_stations(df, rcfg)[0], geom, geo, rcfg)
    z0 = max(float(geom.z0[j, i]), 1e-3)
    attendu = 5.0 * np.log(info["hauteur_ref_m"] / z0) / np.log(10.0 / z0)
    assert abs(S[0, j, i] - attendu) < 0.02 and attendu > 5.0       # plus fort plus haut


# ------------------------------------------------------------------ cibles -> Y
def test_champ_maille_interpole_exactement_un_champ_lineaire(geom, geo):
    from pcfno3d.realdata import grid_to_target
    lat_max, lon_max = geo.to_latlon(geom.nx * geom.dx * 1.5, geom.ny * geom.dy * 1.5)
    lat = np.linspace(LAT0 - 0.05, float(lat_max), 12)[::-1]       # stocke nord -> sud
    lon = np.linspace(LON0 - 0.05, float(lon_max), 15)
    z = np.linspace(0.0, 1500.0, 16)
    Z, LA, LO = np.meshgrid(z, lat, lon, indexing="ij")
    u = 2.0 + 0.004 * Z
    v = -1.0 + 30.0 * (LA - LAT0)
    Y = grid_to_target(lat, lon, z, {"u": u, "v": v}, geom, geo)
    jj, ii = np.meshgrid(np.arange(geom.ny), np.arange(geom.nx), indexing="ij")
    nlat, _ = geo.to_latlon(ii * geom.dx, jj * geom.dy)
    for k, zk in enumerate(geom.z_levels):
        m = geom.valid[k]
        if m.any():
            np.testing.assert_allclose(Y[0, k][m], 2.0 + 0.004 * zk, rtol=1e-5)
            np.testing.assert_allclose(Y[1, k][m], (-1.0 + 30.0 * (nlat - LAT0))[m],
                                       atol=1e-4)
    assert np.all(Y[:, ~geom.valid] == 0)                      # sous le relief, comme le cache
    assert np.isnan(Y[2][geom.valid]).all()                    # w non fourni : inconnu
    assert np.isfinite(Y[3][geom.valid]).all()                 # rho toujours rempli (ISA)


def test_radiosonde_cible_partielle(geom, geo):
    from pcfno3d.realdata import points_to_target
    j, i = geom.ny // 2, geom.nx // 2
    lat, lon = _node_ll(geom, geo, j, i)
    k_ok = [k for k in range(geom.nz) if geom.valid[k, j, i]]
    sonde = pd.DataFrame({"timestamp_utc": "2026-01-01T12:00:00", "lat": lat, "lon": lon,
                          "z_asl_m": [geom.z_levels[k] + 1.0 for k in k_ok],
                          "wind_speed_m_s": 8.0, "wind_dir_deg": 180.0})   # vent du sud
    Y = points_to_target(sonde, geom, geo)
    known = np.isfinite(Y[0]) & geom.valid
    assert known.sum() == len(k_ok) and all(known[k, j, i] for k in k_ok)
    np.testing.assert_allclose(Y[1, k_ok, j, i], 8.0, atol=1e-5)    # v = +8 (vers le nord)
    np.testing.assert_allclose(Y[0, k_ok, j, i], 0.0, atol=1e-5)


# ------------------------------------------------------------------ bout en bout
def test_import_reel_ecrit_un_cache_lisible_par_le_pipeline(rcfg, geom, geo):
    from pcfno3d.cache import load_index
    from pcfno3d.dataset import make_splits
    from pcfno3d.realdata import empty_target, import_real, points_to_target

    j, i = geom.ny // 2, geom.nx // 2
    lat, lon = _node_ll(geom, geo, j, i)
    lat2, lon2 = _node_ll(geom, geo, 1, 2)
    stamps = pd.date_range("2026-03-01", periods=14 * 4, freq="6h")
    st = pd.DataFrame([{"station_id": s, "timestamp_utc": t, "lat": la, "lon": lo,
                        "height_agl_m": 25.0, "wind_u_m_s": 3.0, "wind_v_m_s": -2.0,
                        "temperature_c": 18.0, "pressure_pa": 100500.0}
                       for t in stamps for s, la, lo in (("A", lat, lon), ("B", lat2, lon2))])
    k_ok = [k for k in range(geom.nz) if geom.valid[k, j, i]]

    def cible(ts):
        if ts.hour != 12:
            return None                                    # radiosonde a 12 h seulement
        sonde = pd.DataFrame({"timestamp_utc": ts, "lat": lat, "lon": lon,
                              "z_asl_m": [geom.z_levels[k] for k in k_ok],
                              "wind_u_m_s": 5.0, "wind_v_m_s": 0.0})
        return points_to_target(sonde, geom, geo)

    rapport = import_real(rcfg, st, cible, verbose=False)
    assert rapport["instants"] == len(stamps)
    assert rapport["cibles"] == {"dense": 0, "partielle": 14, "aucune": 42}
    index = load_index(rcfg)
    assert set(index["split"]) == {"train", "val", "test"}
    splits = make_splits(rcfg, index)
    assert len(splits.train[0]) > len(splits.val[0]) > 0
    S = np.load(rcfg.paths.cache_dir / f"{index['stem'].iloc[0]}_S.npy")
    assert S.shape == (8, geom.ny, geom.nx) and S[7].sum() == 2
    Y = np.load(rcfg.paths.cache_dir / f"{index['stem'].iloc[0]}_Y.npy")
    assert Y.shape == (4, *geom.shape)
    assert np.isfinite(empty_target(geom)[3][geom.valid]).all()


def test_loss_ne_supervise_que_les_noeuds_connus(cfg, geom, norm):
    from pcfno3d.losses import PhysicsLoss
    torch.manual_seed(0)
    loss = PhysicsLoss(cfg, geom, norm)
    nz, ny, nx = geom.shape
    y = torch.zeros(1, 4, nz, ny, nx)
    y[:, 3] = 1.2
    y[:, :3] = torch.randn(1, 3, nz, ny, nx)
    pred = torch.randn(1, 3, nz, ny, nx)
    _, full = loss(pred, y, 1.0)

    partial = y.clone()
    hole = torch.rand(nz, ny, nx) < 0.7
    partial[:, :3, hole] = float("nan")
    _, part = loss(pred, partial, 1.0)

    y_n = norm.normalize_target(y)
    keep = geom.valid_t & ~hole
    err = (pred - y_n) ** 2
    err[:, 2] *= cfg.loss.w_weight * norm.w_level_w[:, None, None]
    attendu = err[:, :, keep].mean().item()
    assert abs(part["data"] - attendu) < 1e-5
    assert part["grad"] == 0.0 and full["grad"] > 0.0          # pas de rugosite sans verite dense
    assert np.isfinite(part["mass"])                           # rho reste connu partout
