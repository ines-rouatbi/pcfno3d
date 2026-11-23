"""Mode, date et statut des donnees servies (demande d'Ines du 16/11/2026)."""
import json

from pcfno3d import statut_service as st


def test_analyse_validee_et_temps_reel_national_experimental():
    assert st.statut(True, st.ANALYSE)["statut"] == st.VALIDE
    assert st.statut(True, st.ANALYSE, couche_10m=True)["statut"] == st.VALIDE
    assert st.statut(False, st.ANALYSE)["statut"] == st.VALIDE
    for mode in (st.TEMPS_REEL, st.PREVISION):
        for c in (False, True):
            s = st.statut(True, mode, couche_10m=c, j=1)
            assert s["statut"] == st.EXPERIMENTAL, (mode, c)
    assert "regle_national_gfs_v4" in st.statut(True, st.TEMPS_REEL)["preuve"]


def test_grand_tunis_temps_reel_et_prevision_v4_experimentaux_v5_valide():
    assert st.statut(False, st.TEMPS_REEL)["statut"] == st.EXPERIMENTAL
    assert st.statut(False, st.PREVISION, j=1)["statut"] == st.EXPERIMENTAL
    assert st.statut(False, st.PREVISION, j=1, prevision_os7=True)["statut"] == st.VALIDE
    assert st.statut(False, st.PREVISION, j=0, prevision_os7=True)["statut"] == st.EXPERIMENTAL


def test_verdicts_ignores_sans_empreinte_de_la_regle(tmp_path, monkeypatch):
    regle = tmp_path / "regle_national_gfs_v4.json"
    regle.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(st, "REGLE_GFS", regle)
    v = tmp_path / "verdicts.json"
    v.write_text(json.dumps({"regle_sha256": "abc", "cases": {"a / analyse": "accepte"}}), encoding="utf-8")
    assert st.verdicts_gfs_v4(v) == {}                       # regle pas encore hachee
    (tmp_path / "regle_national_gfs_v4.json.sha256").write_text("def  regle_national_gfs_v4.json\n")
    assert st.verdicts_gfs_v4(v) == {}                       # empreinte differente
    (tmp_path / "regle_national_gfs_v4.json.sha256").write_text("abc  regle_national_gfs_v4.json\n")
    ver = st.verdicts_gfs_v4(v)
    assert ver == {"a / analyse": True}
    assert st.statut(True, st.TEMPS_REEL, verdicts=ver)["statut"] == st.VALIDE
    assert st.statut(True, st.TEMPS_REEL, couche_10m=True, verdicts=ver)["statut"] == st.EXPERIMENTAL


def test_prevision_couche_10m_dit_gfs_brut():
    assert "GFS brut" in st.statut(True, st.PREVISION, couche_10m=True)["preuve"]


def test_la_regle_proposee_n_est_pas_encore_hachee():
    assert st.REGLE_GFS.exists()
    assert "PROPOSITION" in json.loads(st.REGLE_GFS.read_text(encoding="utf-8"))["statut"]


# ---------------------------------------------------------------------- regle_service_zones_v5 (23/11/2026)
def test_regle_zones_hachee_et_valeurs_recopiees():
    z = st.regle_zones()
    assert z is not None and z["sha256"] == "b239c5c6e1db7561def70fa6720c0fda5967d8358c0a35f1194ea28b213e8b6b"
    r = json.loads(st.REGLE_ZONES.read_text(encoding="utf-8"))
    assert "disque de 50 km" in r["zone"]["definition"] and "6371 km" in r["zone"]["definition"]
    assert "36.95 N, 8.75 E" in r["zone"]["centre"] and "0,01 deg" in r["zone"]["centre"]
    assert st.ZONE_TABARKA == {"code": "DTKA", "nom": "Tabarka", "lat": 36.95, "lon": 8.75, "rayon_km": 50.0}
    assert (st.RAYON_TERRE_KM, st.TOLERANCE_CENTRE_DEG) == (6371.0, 0.01)
    assert st.LIBELLE_ZONE in r["statut_par_point"]["dans_la_zone"]
    assert "REFUSEES" in r["base"]["verdict_publie"] and "ce n'est pas une acceptation par test" in \
        r["ce_que_cette_regle_n_est_pas"][0]


def test_regle_zones_modifiee_ignoree(tmp_path, monkeypatch):
    f = tmp_path / "regle_service_zones_v5.json"
    f.write_bytes(st.REGLE_ZONES.read_bytes().replace(b"50 km", b"80 km"))
    (tmp_path / "regle_service_zones_v5.json.sha256").write_text(
        st.REGLE_ZONES.with_name(st.REGLE_ZONES.name + ".sha256").read_text())
    monkeypatch.setattr(st, "REGLE_ZONES", f)
    assert st.regle_zones() is None


def test_statut_par_distance_a_tabarka():
    z = st.regle_zones()
    assert round(st.distance_km(36.95, 8.75, 37.245, 9.791), 0) == 98        # Bizerte, hors zone
    for mode, j in ((st.TEMPS_REEL, 0), (st.PREVISION, 1), (st.PREVISION, 2)):
        tab = st.statut(True, mode, j=j, zones=z, lat=36.95, lon=8.75)
        assert tab["statut"] == st.EXPERIMENTAL and tab["libelle"] == st.LIBELLE_ZONE
        bord = st.statut(True, mode, j=j, zones=z, lat=36.95 + 49.0 / 111.2, lon=8.75)       # ~49 km au nord
        assert bord["statut"] == st.EXPERIMENTAL
        loin = st.statut(True, mode, j=j, zones=z, lat=36.95 + 52.0 / 111.2, lon=8.75)       # ~52 km
        assert loin["statut"] == st.VALIDE and "regle_service_zones_v5" in loin["preuve"]
        tunis = st.statut(True, mode, j=j, zones=z, lat=36.851, lon=10.227)
        assert tunis["statut"] == st.VALIDE
        jour = st.statut(True, mode, j=j, zones=z)                  # journee entiere : hors zone, zone fournie
        assert jour["statut"] == st.VALIDE and jour["libelle"] == "valide (hors zone Tabarka)"
        assert jour["zone_experimentale"]["rayon_km"] == 50.0
    # la couche 10 m, l'analyse et le Grand Tunis ne sont pas touches par la regle
    assert st.statut(True, st.TEMPS_REEL, couche_10m=True, zones=z, lat=37.5, lon=10.0)["statut"] == st.EXPERIMENTAL
    assert "zone_experimentale" not in st.statut(True, st.ANALYSE, zones=z, lat=36.95, lon=8.75)
    assert st.statut(False, st.TEMPS_REEL, zones=z, lat=36.0, lon=10.0)["statut"] == st.EXPERIMENTAL


def test_cas_refuse_jamais_dit_pas_encore_passe():
    for mode, j in ((st.TEMPS_REEL, 0), (st.PREVISION, 1), (st.PREVISION, 2)):
        p = st.statut(True, mode, j=j)["preuve"]
        assert "pas encore" not in p and "refuse" in p and "regle_national_gfs_v5_2" in p
    assert "pas encore" not in st.statut(True, st.TEMPS_REEL, couche_10m=True)["preuve"]
