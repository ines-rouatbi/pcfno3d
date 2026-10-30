"""Temps reel (pcfno3d.temps_reel, atom_wind.jour_direct, /api/atom-wind/direct) : decodage SYNOP,
formules du forcage sur les champs GFS, journee complete avec des reponses reseau simulees."""

from __future__ import annotations

import base64
import json
import math
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from test_forcage_era5 import _jeu_reel, cfg_era5, forcage  # noqa: F401  (fixtures)


# ---------------------------------------------------------------------- SYNOP
def test_lire_synop():
    from pcfno3d.temps_reel import lire_synop

    texte = "\n".join([
        # noeuds (iw = 4), 100 deg, 5 kt
        "60729,2026,10,26,09,00,AAXX 26094 60729 26/// /1005 10228 20147 3//// 4//// 5//// 222// 2//// 333 60007 910// 91116=",
        # m/s (iw = 1), calme
        "60728,2026,10,26,06,00,AAXX 26061 60728 32970 80000 10155=",
        # direction variable : ignoree
        "60728,2026,10,26,09,00,AAXX 26091 60728 32970 89903 10155=",
        # ff >= 99 : groupe 00fff (m/s), 270 deg, 102 m/s (fictif, pour le format)
        "60728,2026,10,26,12,00,AAXX 26121 60728 32970 82799 00102 10155=",
        # station inconnue
        "60715,2026,10,26,09,00,AAXX 26094 60715 26/// /1005 10228=",
    ])
    t = lire_synop(texte)
    assert list(t["station_id"]) == ["INM60729", "INM60728", "INM60728"]
    u, v = t.iloc[0][["wind_u_m_s", "wind_v_m_s"]]
    ff = 5 * 0.514444
    assert u == pytest.approx(-ff * math.sin(math.radians(100))) and v == pytest.approx(-ff * math.cos(math.radians(100)))
    assert (t.iloc[1]["wind_u_m_s"], t.iloc[1]["wind_v_m_s"]) == (0.0, 0.0)
    assert t.iloc[2]["wind_u_m_s"] == pytest.approx(102.0) and abs(t.iloc[2]["wind_v_m_s"]) < 1e-9


# ---------------------------------------------------------------------- reponses simulees
NIV = {1000: (8.0, 270.0, 110.0), 975: (8.5, 271.0, 330.0), 950: (9.0, 272.0, 560.0),
       925: (10.0, 275.0, 800.0), 900: (10.5, 276.0, 1030.0), 850: (12.0, 280.0, 1500.0),
       800: (13.0, 282.0, 1970.0), 750: (14.0, 284.0, 2460.0), 700: (15.0, 285.0, 3100.0)}


def fausse_source(url: str) -> bytes:
    u = urllib.parse.urlparse(url)
    q = dict(urllib.parse.parse_qsl(u.query))
    if u.netloc == "api.open-meteo.com":
        lats = q["latitude"].split(",")
        temps = pd.date_range(q["start_date"], pd.Timestamp(q["end_date"]) + pd.Timedelta(hours=23), freq="h")
        n = len(temps)
        pts = []
        for i, _ in enumerate(lats):
            h = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in temps],
                 "wind_speed_10m": [5.0] * n, "wind_direction_10m": [260.0] * n,
                 "wind_speed_100m": [7.0] * n, "wind_direction_100m": [265.0] * n,
                 "temperature_2m": [20.0 + (i % 2)] * n, "relative_humidity_2m": [60.0] * n,
                 "surface_pressure": [1010.0] * n, "pressure_msl": [1015.0] * n,
                 "shortwave_radiation": [300.0] * n}
            for p, (s, d, z) in NIV.items():
                h[f"wind_speed_{p}hPa"] = [s] * n
                h[f"wind_direction_{p}hPa"] = [d] * n
                h[f"geopotential_height_{p}hPa"] = [z] * n
            pts.append({"elevation": 0.0 if i % 2 else 50.0, "hourly": h})
        return json.dumps(pts).encode()
    if "ensemble-api" in u.netloc:
        lats = q["latitude"].split(",")
        temps = pd.date_range(q["start_date"], pd.Timestamp(q["end_date"]) + pd.Timedelta(hours=23), freq="h")
        n = len(temps)
        pts = []
        for i, _ in enumerate(lats):
            h = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in temps]}
            for m in range(31):
                suf = "" if m == 0 else f"_member{m:02d}"
                h[f"wind_speed_10m{suf}"] = [1.0 + 0.15 * m] * n      # de 1 a 5,5 m/s selon le membre
                h[f"wind_direction_10m{suf}"] = [260.0] * n
            pts.append({"hourly": h})
        return json.dumps(pts).encode()
    if "aviationweather" in u.netloc:
        t0 = pd.Timestamp("2026-10-26T00:00:00Z")
        return json.dumps([{"icaoId": "DTTA", "reportTime": (t0 + pd.Timedelta(hours=k)).isoformat(),
                            "wspd": 10, "wdir": 290} for k in range(0, 12)]).encode()
    if "ogimet" in u.netloc:
        return b"60729,2026,10,26,06,00,AAXX 26064 60729 16/// /0604 10174 20112=\n"
    raise AssertionError(url)


def test_forcage_gfs_formules():
    from pcfno3d.forcage import SCALAIRES
    from pcfno3d.temps_reel import forcage_gfs, telecharger_gfs

    c = telecharger_gfs(np.array([36.0, 36.25]), np.array([10.0, 10.25]), "2026-10-26", "2026-10-26",
                        lire=fausse_source)
    assert c["wind_speed_850hPa"].shape == (24, 2, 2)
    temps, scal, u, v = forcage_gfs(c, (10.0, 100.0, 500.0, 1200.0))
    k = dict(SCALAIRES)
    u850 = -12 * math.sin(math.radians(280))
    assert scal[0, 0] == pytest.approx(u850 * k["ug"], rel=1e-6)
    dz = 3100.0 - 1500.0
    u700 = -15 * math.sin(math.radians(285))
    assert scal[0, 2] == pytest.approx((u700 - u850) / dz * k["shear_u"], rel=1e-6)
    assert 0.0 <= scal[0, 4] / k["veer_deg"] <= 45.0
    assert scal[0, 6] != 0.0                       # terres a 21 deg, mer a 20 (fausse source)
    assert u.shape == (24, 4, 2, 2)
    # 10 m au-dessus du sol : exactement le vent a 10 m
    assert u[0, 0, 0, 0] == pytest.approx(-5 * math.sin(math.radians(260)), rel=1e-5)


@pytest.fixture(scope="module")
def jeu(cfg_era5, index, tmp_path_factory):  # noqa: F811
    from pcfno3d.train import Trainer

    c, _ = _jeu_reel(cfg_era5, index, tmp_path_factory.mktemp("direct"))
    Trainer(c, device="cpu", verbose=False).fit()
    return c


def test_journee_temps_reel(jeu):
    from pcfno3d import atom_wind, temps_reel
    from pcfno3d.inference import WindFieldModel

    wf = WindFieldModel(jeu, device="cpu", verbose=False)
    src = temps_reel.SourceDirect(wf.geom, lire=fausse_source)
    maintenant = pd.Timestamp("2026-10-26T07:20:00")
    d = atom_wind.jour_direct(wf, src, "2026-10-26", maintenant=maintenant)
    assert d["temps_reel"] is True and len(d["instants"]) == 24
    hl = [i["heure_locale"] for i in d["instants"]]
    assert hl == list(range(24))
    # 07 h locale = 06 h UTC : METAR DTTA et SYNOP 60729 ; apres « maintenant » : prevision
    i7 = d["instants"][7]
    assert i7["source"] == "temps reel" and set(i7["stations"]) == {"DTTA", "INM60729"}
    assert d["instants"][12]["source"] == "prevision" and d["instants"][12]["stations"] == []
    a = np.frombuffer(base64.b64decode(d["vent"]), dtype="<i2").reshape(d["forme"])
    assert (a != -32768).mean() > 0.5
    # espace agriculteur : temperature et humidite a 2 m de GFS (20-21 deg C, 60 % dans la fausse source)
    t2m = np.frombuffer(base64.b64decode(d["t2m"]), dtype="<i2").astype(float) / 100
    hr = np.frombuffer(base64.b64decode(d["hr2m"]), dtype="<i2").astype(float) / 100
    assert 10.0 < np.median(t2m) < 25.0 and np.median(hr) == pytest.approx(60.0, abs=0.5)
    assert all(i["rayonnement_w_m2"] == pytest.approx(300.0) for i in d["instants"])


def test_valeur_des_stations(jeu):
    """E10 / E11 : deni de donnees sur une heure passee ou deux stations mesurent."""
    from pcfno3d import atom_wind, temps_reel
    from pcfno3d.inference import WindFieldModel

    wf = WindFieldModel(jeu, device="cpu", verbose=False)
    src = temps_reel.SourceDirect(wf.geom, lire=fausse_source)
    d = atom_wind.valeur_stations(wf, src, "2026-10-26", maintenant=pd.Timestamp("2026-10-26T07:20:00"), heures=(7, 12))
    assert d["heures_locales"] == [7.0]                     # 12 h est dans le futur : pas rejouee
    assert d["codes"] == ["DTTA", "INM60729"] and d["forme"][0] == 2
    for st in d["stations"]:
        assert st["instants"] == 1 and st["erreur_sans_elle_m_s"] >= 0 and st["aire_influence_km2"] >= 0
    cartes = np.frombuffer(base64.b64decode(d["influence"]), dtype="<i2").reshape(d["forme"])
    assert (cartes > 0).any()                               # retirer une station change le champ
    with pytest.raises(ValueError, match="aucune heure mesuree"):
        atom_wind.valeur_stations(wf, src, "2026-10-26", maintenant=pd.Timestamp("2026-10-26T07:20:00"), heures=(20,))


def test_forcage_direct_meme_forme(jeu):
    from pcfno3d import temps_reel
    from pcfno3d.geometry import Geometry

    g = Geometry(jeu)
    src = temps_reel.SourceDirect(g, lire=fausse_source)
    e = src.journee("2026-10-26")
    vec = e["forcage"].vecteur(pd.Timestamp("2026-10-26T06:00:00"))
    assert vec.shape == (g.forcage.n_extra,) and np.isfinite(vec).all()
    S = temps_reel.entree(g, e["champs"], e["forcage"], 6, {"DTTA": (3.0, -1.0)})
    assert S.shape[0] == 8 and S[7].sum() > 0
    from pcfno3d.stations_reelles import cellules_codes
    iy, ix = cellules_codes(g)["DTTA"]
    assert (S[0, iy, ix], S[1, iy, ix], S[7, iy, ix]) == (3.0, -1.0, 1.0)
    assert np.isfinite(S[:7][:, S[7] > 0]).all()


def test_route_direct(jeu, monkeypatch):
    from fastapi.testclient import TestClient

    from pcfno3d import temps_reel
    monkeypatch.setattr(temps_reel, "_get", fausse_source)
    for var, val in (("PCFNO3D_DATA_DIR", jeu.paths.data_dir), ("PCFNO3D_CACHE_DIR", jeu.paths.cache_dir),
                     ("PCFNO3D_RUN_DIR", jeu.paths.run_dir), ("PCFNO3D_DEVICE", "cpu")):
        monkeypatch.setenv(var, str(val))
    import pcfno3d.config as cm
    monkeypatch.setattr(cm.Config, "load", classmethod(lambda cls, _n: jeu))
    for var in ("PCFNO3D_API_KEYS", "PCFNO3D_INSECURE_OPEN"):
        monkeypatch.delenv(var, raising=False)
    from service.auth import ApiKeyStore, default_store_path
    chemin = default_store_path(jeu.paths.run_dir)
    chemin.unlink(missing_ok=True)
    cle, _ = ApiKeyStore(chemin).create("tests-direct", scope="read")
    from service.app import create_app
    with TestClient(create_app(), headers={"X-API-Key": cle}) as c:
        j = c.get("/api/atom-wind/jours").json()
        assert j["direct"]["route"] == "/api/atom-wind/direct"
        r = c.get("/api/atom-wind/direct")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["temps_reel"] is True and d["date"] == j["direct"]["date"]
        assert c.get("/api/atom-wind/direct").json()["mis_a_jour_utc"] == d["mis_a_jour_utc"]   # cache
        # prevision : demain, toutes les heures en prevision, sans station
        assert len(j["direct"]["previsions"]) == 2
        p1 = c.get("/api/atom-wind/direct", params={"j": 1}).json()
        assert p1["prevision"] is True and p1["date"] == j["direct"]["previsions"][0]
        assert all(i["source"] == "prevision" and i["stations"] == [] for i in p1["instants"])
        assert c.get("/api/atom-wind/direct", params={"j": 5}).status_code == 422
        # prevision probabiliste (E19) : demain, 31 scenarios GEFS
        en = c.get("/api/atom-wind/ensemble", params={"j": 1})
        assert en.status_code == 200, en.text
        assert en.json()["ensemble"] is True and en.json()["instants"][0]["membres"] == 31


def test_ensemble_gefs(jeu):
    from pcfno3d import atom_wind, temps_reel
    from pcfno3d.inference import WindFieldModel

    wf = WindFieldModel(jeu, device="cpu", verbose=False)
    src = temps_reel.SourceDirect(wf.geom, lire=fausse_source)
    d = atom_wind.jour_ensemble(wf, src, "2026-10-27", maintenant=pd.Timestamp("2026-10-26T12:00"))
    assert d["ensemble"] is True and [i["heure_locale"] for i in d["instants"]] == list(atom_wind.HEURES_ENSEMBLE)
    assert all(i["membres"] == 31 and i["source"] == "prevision" for i in d["instants"])
    P = np.frombuffer(base64.b64decode(d["proba_traitement"]), dtype="<i2").astype(float).reshape(d["forme_proba"])
    P = np.where(P == -32768, np.nan, P / 100)
    assert np.nanmin(P) >= 0.0 and np.nanmax(P) <= 1.0 and np.isfinite(P).mean() > 0.5
    Q = np.frombuffer(base64.b64decode(d["vitesse_quantiles"]), dtype="<i2").astype(float).reshape(d["forme_quantiles"])
    Q = np.where(Q == -32768, np.nan, Q / 100)
    assert Q.shape[1:3] == (2, 3)
    ok = np.isfinite(Q).all(axis=2)
    assert (np.diff(Q, axis=2)[np.broadcast_to(ok[:, :, None], np.diff(Q, axis=2).shape)] >= -1e-6).all()   # p10 <= p50 <= p90


def test_traitement_possible():
    from pcfno3d.atom_wind import traitement_possible
    k = np.array([2.9, 3.0, 15.0, 15.1]) / 3.6
    assert traitement_possible(k).tolist() == [False, True, True, False]
