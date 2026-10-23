"""scripts/valider_grand_tunis_reel.py : entree avec les vraies mesures, vent a 10 m, bilans."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("valider", REPO / "scripts" / "valider_grand_tunis_reel.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def test_entree_reelle_mesures_era5_et_station_exclue(geom, monkeypatch):
    monkeypatch.setattr(geom.cfg.mask, "pas_era5_m", 3 * geom.dx)   # reseau tous les 3 noeuds
    S = np.random.default_rng(0).normal(size=(8,) + geom.shape[1:]).astype(np.float32)
    E = np.full((2,) + geom.shape[1:], 7.0, np.float32)
    k = max(int(round(geom.cfg.mask.pas_era5_m / geom.dx)), 1)
    hors = (k // 2 + 1, k // 2 + 1)          # cellule hors du reseau
    sur = (k // 2, k // 2)                                          # cellule du reseau
    cellules = {"DTTA": hors, "INM60729": sur}
    mesures = {"DTTA": (1.0, 2.0), "INM60729": (3.0, 4.0)}

    Si, cap = V.entree_reelle(S, E, geom, cellules, mesures)
    assert cap[hors] and cap[sur]
    assert tuple(Si[:2, hors[0], hors[1]]) == (1.0, 2.0)
    assert tuple(Si[:2, sur[0], sur[1]]) == (3.0, 4.0)              # la station prime
    assert (Si[2] == 0).all()

    Si, cap = V.entree_reelle(S, E, geom, cellules, mesures, exclue="DTTA")
    assert not cap[hors]                                            # hors reseau : sans capteur
    Si, cap = V.entree_reelle(S, E, geom, cellules, mesures, exclue="INM60729")
    assert cap[sur] and tuple(Si[:2, sur[0], sur[1]]) == (7.0, 7.0)  # sur le reseau : ERA5


def test_vent_a_10m_interpole_ou_profil_log():
    h = np.array([-20.0, 5.0, 30.0, 80.0])
    u = np.array([np.nan, 2.0, 4.0, 5.0])
    v = np.array([np.nan, 0.0, 0.0, 0.0])
    um, vm = V.vent_a_hauteur(u, v, h, z0=0.1)
    assert um == pytest.approx(2.0 + 2.0 * (10 - 5) / 25)           # entre 5 et 30 m
    um, _ = V.vent_a_hauteur(u[2:], v[2:], h[2:], z0=0.1)           # plus bas noeud a 30 m
    assert um == pytest.approx(4.0 * math.log(100) / math.log(300))
    assert math.isnan(V.vent_a_hauteur(np.full(3, np.nan), np.full(3, np.nan), h[:3], 0.1)[0])


def test_bilan_stations_parfait_et_direction():
    p = pd.DataFrame({"station": "X", "u_obs": [3.0, 0.0], "v_obs": [0.0, -3.0],
                      "u_parfait": [3.0, 0.0], "v_parfait": [0.0, -3.0],
                      "u_tourne": [0.0, 3.0], "v_tourne": [3.0, 0.0]})
    b = V.bilan_stations(p, ("parfait", "tourne")).set_index("source")
    assert b.loc["parfait", "rmse_vect"] == 0 and b.loc["parfait", "mae_direction_deg"] == 0
    assert b.loc["tourne", "mae_direction_deg"] == pytest.approx(90.0)


def test_observations_mois_en_panne_ecartes(tmp_path):
    t = pd.date_range("2021-06-01", periods=120, freq="12h")
    lignes = [{"station_id": "INM60728", "timestamp_utc": x.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "wind_speed_m_s": 0.0 if x.month == 6 else 3.0, "wind_dir_deg": 90.0,
               "wind_u_m_s": np.nan, "wind_v_m_s": np.nan} for x in t]
    pd.DataFrame(lignes).to_csv(tmp_path / "synop_2021.csv", index=False)
    obs = V.lire_observations(tmp_path)
    assert (obs["t"].dt.month != 6).all() and len(obs) > 0
    assert obs["wind_u_m_s"].iloc[0] == pytest.approx(-3.0)        # vent d'est : u < 0
    m = V.obs_a(obs, pd.Timestamp("2021-07-05 00:10"))
    assert m["INM60728"][0] == pytest.approx(-3.0)
