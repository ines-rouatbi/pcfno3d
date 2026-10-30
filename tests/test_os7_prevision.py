"""OS7 (prevision GFS) : choix du run, forcage GFS relu par le modele, reseau a 10 m tire du forcage,
pas de station en entree, evaluation PREV aux stations (scripts/evaluer_prevision.py)."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_forcage_era5 import HAUTEURS, LAT_E, LON_E, _jeu_reel, cfg_era5, forcage, vent  # noqa: F401

REPO = Path(__file__).resolve().parents[1]


def test_run_pour():
    from pcfno3d.gfs import run_pour

    assert run_pour(pd.Timestamp("2022-01-11 00:00"), 24) == (pd.Timestamp("2022-01-10 00:00"), 24)
    assert run_pour(pd.Timestamp("2022-01-11 03:00"), 24) == (pd.Timestamp("2022-01-10 00:00"), 27)
    assert run_pour(pd.Timestamp("2022-01-11 21:00"), 48) == (pd.Timestamp("2022-01-09 18:00"), 51)


def _champs_gfs(temps: pd.DatetimeIndex) -> dict:
    """Champs « Open-Meteo » synthetiques aux mailles du forcage de test : vent lineaire (comme la
    fonction `vent` du forcage ERA5 de test), niveaux GFS, relief 0."""
    from pcfno3d.gfs import NIVEAUX_GFS

    la, lo = np.meshgrid(LAT_E, LON_E, indexing="ij")
    T = len(temps)
    c = {"temps": np.array([int(t.value // 10 ** 9) for t in temps], np.int64),
         "elevation": np.zeros(la.shape), "terre": la < 36.05}
    def uvdir(h):
        u, v = vent(h, la, lo)
        s = np.hypot(u, v)
        d = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        return np.broadcast_to(s, (T,) + la.shape).copy(), np.broadcast_to(d, (T,) + la.shape).copy()
    c["wind_speed_10m"], c["wind_direction_10m"] = uvdir(10.0)
    c["wind_speed_100m"], c["wind_direction_100m"] = uvdir(100.0)
    for p in NIVEAUX_GFS:
        z = 110.0 + (1000 - p) * 9.0
        c[f"wind_speed_{p}hPa"], c[f"wind_direction_{p}hPa"] = uvdir(z)
        c[f"geopotential_height_{p}hPa"] = np.full((T,) + la.shape, z)
    for k, val in (("temperature_2m", 20.0), ("relative_humidity_2m", 60.0), ("surface_pressure", 1010.0),
                   ("pressure_msl", 1012.0), ("shortwave_radiation", 200.0)):
        c[k] = np.full((T,) + la.shape, val)
    return c


@pytest.fixture(scope="module")
def jeu_gfs(cfg_era5, index, tmp_path_factory):  # noqa: F811
    """Le mini-jeu reel entraine (ERA5), plus un forcage « GFS » ecrit par pcfno3d.gfs, et la
    configuration de prevision qui le lit."""
    from pcfno3d.gfs import ecrire_forcage
    from pcfno3d.train import Trainer

    d = tmp_path_factory.mktemp("os7")
    c, obs = _jeu_reel(cfg_era5, index, d)
    Trainer(c, device="cpu", verbose=False).fit()
    t = pd.to_datetime(index["timestamp"], utc=True).dt.tz_localize(None)
    temps = pd.date_range(t.min().floor("D") - pd.Timedelta(hours=3), t.max().ceil("D") + pd.Timedelta(hours=3), freq="3h")
    champs = _champs_gfs(temps)
    ech = {int(x.value // 10 ** 9): ("run", 24) for x in temps}
    df = ecrire_forcage(champs, HAUTEURS, LAT_E, LON_E, d / "forcage_gfs_f24.csv", ech)
    p = copy.deepcopy(c)
    p.apply_overrides({"paths.forcage_era5": d / "forcage_gfs_f24.csv", "mask.e_source": "forcage",
                       "mask.stations_entree": False, "mask.p_station": 0.0})
    return c, p, obs, df


def test_forcage_gfs_relu(jeu_gfs):
    from pcfno3d.forcage import ForcageERA5
    from pcfno3d.geometry import Geometry

    c, p, _, df = jeu_gfs
    assert {"run_utc", "echeance_h"} <= set(df.columns)
    f = ForcageERA5(p.paths.forcage_era5)
    g_era5 = Geometry(c)
    assert f.n_extra == g_era5.forcage.n_extra           # meme forme d'entree que le modele ERA5
    assert np.isfinite(f.scal).all() and np.isfinite(f.u).all()


def test_reseau_10m_tire_du_forcage_et_sans_station(jeu_gfs):
    from pcfno3d.dataset import entree_era5_stations
    from pcfno3d.forcage import charger_e, horodatage
    from pcfno3d.geometry import Geometry
    from pcfno3d.stations_reelles import cellules_codes

    _, p, _, _ = jeu_gfs
    g = Geometry(p)
    stem = pd.read_csv(p.paths.cache_dir / "index.csv")["stem"].iloc[0]
    E = charger_e(g, p.paths.cache_dir / f"{stem}_E.npy")
    # vent synthetique lineaire en lat / lon : la bilineaire le restitue a 10 m
    u_ref, v_ref = vent(10.0, g.interp_era5.lat if hasattr(g.interp_era5, "lat") else None, g.interp_era5.lon)
    np.testing.assert_allclose(E[0], u_ref, atol=1e-3)
    np.testing.assert_allclose(E[1], v_ref, atol=1e-3)
    assert horodatage(g, stem)
    S = np.load(p.paths.cache_dir / f"{stem}_S.npy")
    _, capteur = entree_era5_stations(S, p.paths.cache_dir / f"{stem}_E.npy", g, np.random.default_rng(0), train=False)
    for iy, ix in cellules_codes(g).values():
        k = max(int(round(p.mask.pas_era5_m / g.dx)), 1)
        sur_reseau = (iy - k // 2) % k == 0 and (ix - k // 2) % k == 0
        assert capteur[iy, ix] == sur_reseau            # la station n'entre pas comme station


def test_evaluer_prevision(jeu_gfs, tmp_path):
    spec = importlib.util.spec_from_file_location("ep", REPO / "scripts" / "evaluer_prevision.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    c, p, obs, _ = jeu_gfs
    paires, bilan, resume = m.evaluer(p, obs, 24, device="cpu", verbose=False)
    assert len(paires) > 0 and resume["critere"] == "PREV1"
    assert set(bilan["source"]) == {"prevision", "gfs", "persistance"}
    assert paires[["u_prevision", "v_prevision", "u_gfs", "v_gfs"]].notna().all().all()
    t = resume["verdicts"]["toutes"]
    assert set(t["rmse_vect"]) == {"prevision", "gfs", "persistance"}
    json.dumps(resume, default=str)
    with pytest.raises(ValueError, match="prevision"):
        m.evaluer(c, obs, 24, device="cpu", verbose=False)       # configuration ERA5 refusee
