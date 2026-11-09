"""Modele national : les stations (12 aeroports METAR) suivent stations.json (code et nom) ; le
Grand Tunis garde ses trois stations et ses noms (fichiers deja sur le Drive, sans code)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd


def _geom(dossier, stations):
    (dossier / "stations.json").write_text(json.dumps(stations), encoding="utf-8")
    return SimpleNamespace(cfg=SimpleNamespace(paths=SimpleNamespace(data_dir=dossier)))


def test_codes_et_noms_nationaux(tmp_path):
    from pcfno3d.stations_reelles import cellules_codes, noms_stations

    g = _geom(tmp_path, [{"nom": "DTTX Sfax", "code": "DTTX", "iy": 506, "ix": 299, "lat": 34.7, "lon": 10.7},
                         {"nom": "DTTA Tunis-Carthage", "code": "DTTA", "iy": 870, "ix": 237, "lat": 36.8, "lon": 10.2}])
    assert cellules_codes(g) == {"DTTX": (506, 299), "DTTA": (870, 237)}
    noms = noms_stations(g)
    assert noms["DTTX"] == "Sfax" and noms["DTTA"] == "Tunis-Carthage"


def test_grand_tunis_inchange(tmp_path):
    from pcfno3d.stations_reelles import CODES, cellules_codes, noms_stations

    g = _geom(tmp_path, [{"nom": "DTTA Tunis-Carthage (METAR, INM 60715)", "iy": 1, "ix": 2, "lat": 0, "lon": 0},
                         {"nom": "Zaghouan-Mograne (synop INM 60729, NOAA ISD 607290)", "iy": 3, "ix": 4,
                          "lat": 0, "lon": 0}])
    assert cellules_codes(g) == {"DTTA": (1, 2), "INM60729": (3, 4)}
    assert noms_stations(g) == CODES


def test_observations_filtrees_par_codes(tmp_path):
    from pcfno3d.stations_reelles import lire_observations

    lignes = [{"station_id": c, "timestamp_utc": f"2022-01-0{j + 1}T12:00:00Z", "wind_speed_m_s": 5.0,
               "wind_dir_deg": 270.0, "wind_u_m_s": None, "wind_v_m_s": None}
              for c in ("DTTX", "DTTZ", "DTTA") for j in range(3)]
    pd.DataFrame(lignes).to_csv(tmp_path / "metar_2022.csv", index=False)
    obs = lire_observations(tmp_path, verbose=False, codes=["DTTX", "DTTZ"])
    assert set(obs["station_id"]) == {"DTTX", "DTTZ"}
    assert abs(obs["wind_u_m_s"].iloc[0] - 5.0) < 1e-9                     # vent d'ouest : u > 0
    assert set(lire_observations(tmp_path, verbose=False)["station_id"]) == {"DTTA"}   # defaut : Grand Tunis


def test_affinage_lit_toutes_les_stations_du_domaine(tmp_path):
    """Affinage sur les stations (table_stations) : les mesures lues sont celles des stations
    de stations.json (12 aeroports au national), pas seulement les 3 codes du Grand Tunis."""
    from pcfno3d.stations_reelles import table_stations

    lignes = [{"station_id": c, "timestamp_utc": "2023-05-01T12:00:00Z", "wind_speed_m_s": 4.0,
               "wind_dir_deg": 180.0, "wind_u_m_s": None, "wind_v_m_s": None}
              for c in ("DTTX", "DTTZ", "DTTA")]
    obs = tmp_path / "obs"
    obs.mkdir()
    pd.DataFrame(lignes).to_csv(obs / "metar_2023.csv", index=False)
    pd.DataFrame({"stem": ["0001_120000"], "timestamp": ["2023-05-01 12:00:00"]}).to_csv(
        tmp_path / "index.csv", index=False)
    g = _geom(tmp_path, [{"nom": "DTTX Sfax", "code": "DTTX", "iy": 5, "ix": 6},
                         {"nom": "DTTZ Tozeur", "code": "DTTZ", "iy": 7, "ix": 8}])
    g.cfg.paths.stations_obs = obs
    g.cfg.paths.index_csv = tmp_path / "index.csv"
    t = table_stations(g)["0001_120000"]
    assert set(t) == {"DTTX", "DTTZ"}
    assert abs(t["DTTX"][1] - 4.0) < 1e-9                                   # vent du sud : v > 0
