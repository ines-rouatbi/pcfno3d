"""Collecteur METAR : conversions, instant commun, archive, position, corps de requete.
Hors ligne : aucun appel reseau."""

import csv
import math

import pytest

from pcfno3d import metar
from pcfno3d.realdata import GeoRef, wind_components


def obs(**k):
    base = {"icaoId": "DTNH", "reportTime": "2026-10-17T13:30:00.000Z", "lat": 36.074,
            "lon": 10.436, "elev": 1, "wdir": 80, "wspd": 10, "temp": 26, "dewp": 17,
            "altim": 1015}
    base.update(k)
    return base


def test_conversion_complete():
    l, raison = metar.metar_vers_station(obs())
    assert raison is None
    assert l["station_id"] == "DTNH" and l["timestamp_utc"] == "2026-10-17T13:30:00Z"
    assert l["wind_speed_m_s"] == pytest.approx(5.144, abs=1e-3)       # 10 kt
    assert l["wind_dir_deg"] == 80.0 and l["height_agl_m"] == 10.0
    assert l["temperature_c"] == 26.0
    assert 50 < l["relative_humidity_pct"] < 60                         # 26 / 17 deg C : ~57 %
    assert l["pressure_pa"] == pytest.approx(101488.0, abs=2.0)         # QNH 1015 a 1 m
    assert l["quality"] == 1.0 and l["source"] == "metar"


def test_pression_et_humidite():
    assert metar.pression_station(1013.25, 0.0) == pytest.approx(101325.0)
    # ~ -12 hPa par 100 m pres du sol
    assert metar.pression_station(1013.25, 100.0) == pytest.approx(100129.0, abs=5.0)
    assert metar.humidite_relative(20.0, 20.0) == pytest.approx(100.0)
    assert metar.humidite_relative(30.0, 10.0) == pytest.approx(28.7, abs=0.5)


def test_calme_et_vent_variable():
    l, _ = metar.metar_vers_station(obs(wspd=0, wdir=0))
    assert (l["wind_u_m_s"], l["wind_v_m_s"], l["wind_speed_m_s"]) == (0.0, 0.0, 0.0)
    assert l["wind_dir_deg"] is None
    l, _ = metar.metar_vers_station(obs(wspd=0, wdir="VRB"))
    assert l["wind_speed_m_s"] == 0.0                                    # calme variable
    l, raison = metar.metar_vers_station(obs(wspd=3, wdir="VRB"))
    assert l is None and "variable" in raison                            # jamais une direction inventee


def test_mesures_inexploitables():
    assert metar.metar_vers_station(obs(wspd=None))[0] is None
    assert metar.metar_vers_station(obs(lat=None))[0] is None
    l, _ = metar.metar_vers_station(obs(temp=None, dewp=None, altim=None))
    assert l["temperature_c"] is None and l["pressure_pa"] is None
    assert l["relative_humidity_pct"] is None


def test_direction_meteo_coherente_avec_le_pipeline():
    # vent d'est (80 deg) : u negatif (vers l'ouest), comme realdata.wind_components
    l, _ = metar.metar_vers_station(obs())
    u, v = wind_components(l["wind_speed_m_s"], l["wind_dir_deg"])
    assert float(u) < 0 and abs(float(u)) > abs(float(v))


def test_instant_commun():
    lignes = [metar.metar_vers_station(obs(icaoId=i, reportTime=t))[0] for i, t in [
        ("DTNH", "2026-10-17T13:30:00Z"), ("DTNH", "2026-10-17T13:00:00Z"),
        ("DTKA", "2026-10-17T13:00:00Z"), ("DTTA", "2026-10-17T11:00:00Z")]]
    instant, choix = metar.instant_commun(lignes)
    assert instant == "2026-10-17T13:00:00Z"
    assert [(l["station_id"], l["timestamp_utc"]) for l in choix] == [
        ("DTKA", "2026-10-17T13:00:00Z"), ("DTNH", "2026-10-17T13:00:00Z")]   # DTTA trop ancien
    assert metar.instant_commun([]) == (None, [])


def test_archive_sans_doublon(tmp_path):
    a = tmp_path / "metar.csv"
    l1, _ = metar.metar_vers_station(obs())
    l2, _ = metar.metar_vers_station(obs(icaoId="DTKA"))
    assert metar.ajouter_archive(a, [l1]) == 1
    assert metar.ajouter_archive(a, [l1, l2]) == 1
    with open(a, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["station_id"] for r in rows] == ["DTNH", "DTKA"]
    assert list(rows[0]) == metar.COLONNES


def test_position_domaine():
    geo = GeoRef(36.0, 9.7)
    dedans, _ = metar.metar_vers_station(obs(icaoId="IN", lat=36.1, lon=9.9))
    dehors, _ = metar.metar_vers_station(obs())                         # Enfidha, a l'est
    pos = {p["station_id"]: p for p in metar.position_domaine([dedans, dehors], geo, 49_900, 24_900)}
    assert pos["IN"]["distance_domaine_km"] == 0
    x_enfidha = 6_371_008.8 * math.cos(math.radians(36.0)) * math.radians(10.436 - 9.7)
    assert pos["DTNH"]["distance_domaine_km"] == pytest.approx((x_enfidha - 49_900) / 1000, abs=0.2)


def test_corps_reconstruction():
    l, _ = metar.metar_vers_station(obs(wspd=0))
    corps = metar.corps_reconstruction("2026-10-17T13:00:00Z", [l])
    st = corps["stations"][0]
    assert corps["timestamp_utc"] == "2026-10-17T13:00:00Z"
    assert st["wind_u_m_s"] == 0.0 and "wind_dir_deg" not in st and "source" not in st
