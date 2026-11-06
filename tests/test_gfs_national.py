"""OS7 + modele national : un seul telechargement GFS garde les mailles du Grand Tunis ET la boite
nationale ; reprise par instant (une archive d'avant, mailles seules, est completee sans toucher
aux mailles) ; assemblage du cube national. Aucun acces reseau : le telechargement est simule."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from test_forcage_era5 import HAUTEURS, LAT_E, LON_E
from test_os7_prevision import _champs_gfs


def test_grille_tunisie_couvre_le_pays_et_le_grand_tunis():
    from pcfno3d.gfs import grille_tunisie

    lat, lon = grille_tunisie()
    assert lat.shape == (33,) and lon.shape == (21,)
    assert np.all(np.diff(lat) > 0) and np.allclose(np.diff(lon), 0.25)
    # territoire tunisien (30,2-37,6 N, 7,5-11,6 E) entierement dedans
    assert lat[0] <= 30.2 and lat[-1] >= 37.6 and lon[0] <= 7.5 and lon[-1] >= 11.6
    # les mailles du forcage Grand Tunis (grille 0,25 deg) sont des noeuds de la boite
    for la in (36.25, 36.5, 36.75, 37.0):
        assert np.isclose(lat, la).any()
    for lo in (9.75, 10.0, 10.25, 10.5, 10.75, 11.0):
        assert np.isclose(lon, lo).any()


def _faux_telechargement(appels):
    def telecharger_zones(run, echeance_h, zones, dossier):
        appels.append((run, echeance_h, tuple(zones)))
        t = pd.Timestamp(run) + pd.Timedelta(hours=echeance_h)
        c = {k: v for k, v in _champs_gfs(pd.DatetimeIndex([t])).items() if k != "temps"}
        out = {"mailles": dict(c, source="test")}
        if "tunisie" in zones:
            la, lo = zones["tunisie"]
            forme = (len(la), len(lo))
            tn = {k: np.full((1,) + forme, float(np.mean(v)) + t.hour) for k, v in c.items()
                  if k not in ("elevation", "terre")}
            tn["elevation"] = np.zeros(forme)
            tn["terre"] = np.ones(forme, bool)
            out["tunisie"] = dict(tn, source="test")
        return out
    return telecharger_zones


def test_un_telechargement_deux_zones_et_reprise(tmp_path, monkeypatch):
    from pcfno3d import gfs

    appels = []
    monkeypatch.setattr(gfs, "telecharger_zones", _faux_telechargement(appels))
    instants = list(pd.date_range("2022-01-10 00:00", periods=4, freq="3h"))
    reprise = tmp_path / "gfs_f24_instants"
    reprise.mkdir()
    # archive d'avant : le premier instant n'a que le fichier des mailles
    ancien = reprise / f"{instants[0]:%Y%m%dT%H%M}_f24.npz"
    run0, l0 = gfs.run_pour(instants[0], 24)
    c0 = _faux_telechargement([])(run0, l0, {"mailles": None}, None)["mailles"]
    np.savez_compressed(ancien, **{k: v for k, v in c0.items() if k != "source"})
    os.utime(ancien, ns=(10 ** 18, 10 ** 18))

    df = gfs.construire(instants, 24, LAT_E, LON_E, HAUTEURS, tmp_path / "forcage_gfs_f24.csv", reprise,
                        travailleurs=1, verbose=False)
    assert len(appels) == 4 and all(z == ("mailles", "tunisie") for *_, z in appels)
    for t in instants:
        assert gfs.fichier_national(reprise, t, 24).exists()
    # le fichier des mailles deja la n'est pas reecrit
    assert ancien.stat().st_mtime_ns == 10 ** 18
    assert {"run_utc", "echeance_h"} <= set(df.columns)

    # relance : rien a telecharger
    appels.clear()
    gfs.construire(instants[1:], 24, LAT_E, LON_E, HAUTEURS, tmp_path / "f.csv", reprise,
                   travailleurs=1, verbose=False)
    assert appels == []

    cube = gfs.assembler_national(reprise, 24, tmp_path / "gfs_tunisie_f24.npz", verbose=False)
    lat, lon = gfs.grille_tunisie()
    assert cube["wind_speed_10m"].shape == (4, len(lat), len(lon))
    assert cube["wind_speed_10m"].dtype == np.float32
    assert list(cube["temps"]) == [int(t.value // 10 ** 9) for t in instants]
    assert list(cube["echeance_h"]) == [gfs.run_pour(t, 24)[1] for t in instants]
    assert cube["elevation"].shape == (len(lat), len(lon))
    with np.load(tmp_path / "gfs_tunisie_f24.npz") as d:
        assert {"lat", "lon", "run_utc", "wind_speed_850hPa", "geopotential_height_700hPa"} <= set(d.files)


def test_sans_national(tmp_path, monkeypatch):
    from pcfno3d import gfs

    appels = []
    monkeypatch.setattr(gfs, "telecharger_zones", _faux_telechargement(appels))
    instants = list(pd.date_range("2022-01-10 00:00", periods=2, freq="3h"))
    gfs.construire(instants, 24, LAT_E, LON_E, HAUTEURS, tmp_path / "f.csv", tmp_path / "r",
                   travailleurs=1, verbose=False, national=False)
    assert all(z == ("mailles",) for *_, z in appels)
    assert not list((tmp_path / "r").glob("*_tn.npz"))
