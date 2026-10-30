"""Routes REST du cahier v12 sur un vrai client HTTP (mini-jeu, fausse source temps reel, cle de lecture) :
decision agricole et production (S16), maintenance (E7), rampes (E13), dimensionnement (E18), rapport PDF
(E9), qualite du modele servi (TB5)."""

from __future__ import annotations

import pytest

from test_temps_reel import _jeu_reel, cfg_era5, fausse_source, forcage, jeu  # noqa: F401
from test_assistant import LAT, LON  # noqa: F401


@pytest.fixture
def client(jeu, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient

    from pcfno3d import temps_reel
    from service import atom_outils
    monkeypatch.setattr(temps_reel, "_get", fausse_source)
    monkeypatch.setattr(atom_outils, "PARC_VIRTUEL", (("E1", LAT, LON),))
    for c in atom_outils._CACHES.values():
        c.clear()
    for var, val in (("PCFNO3D_DATA_DIR", jeu.paths.data_dir), ("PCFNO3D_CACHE_DIR", jeu.paths.cache_dir),
                     ("PCFNO3D_RUN_DIR", jeu.paths.run_dir), ("PCFNO3D_DEVICE", "cpu")):
        monkeypatch.setenv(var, str(val))
    for var in ("PCFNO3D_API_KEYS", "PCFNO3D_INSECURE_OPEN"):
        monkeypatch.delenv(var, raising=False)
    import pcfno3d.config as cm
    monkeypatch.setattr(cm.Config, "load", classmethod(lambda cls, _n: jeu))
    from service.auth import ApiKeyStore, default_store_path
    chemin = default_store_path(jeu.paths.run_dir)
    chemin.unlink(missing_ok=True)
    cle, _ = ApiKeyStore(chemin).create("tests-v12", scope="read")
    from service.app import create_app
    with TestClient(create_app(), headers={"X-API-Key": cle}) as c:
        yield c


def test_routes_metier(client):
    pt = {"lat": LAT, "lon": LON}
    a = client.get("/api/atom-wind/agricole", params={"j": 1, **pt})
    assert a.status_code == 200 and len(a.json()["heures"]) == 24
    p = client.get("/api/atom-wind/production", params={"j": 1, "moyeu_m": 80, **pt})
    assert p.status_code == 200 and 0 <= p.json()["facteur_charge"] <= 1
    m = client.get("/api/atom-wind/maintenance", params={"j": 1, "hauteur_m": 80, **pt})
    assert m.status_code == 200 and m.json()["seuil_m_s"] == 12.0
    assert client.get("/api/atom-wind/variations", params={"j": 1}).json()["puissance_installee_kw"] == 2000
    d = client.get("/api/atom-wind/dimensionnement", params=pt)
    assert d.status_code == 200 and d.json()["batterie_wh"] > 0
    r = client.get("/api/atom-wind/rapport.pdf", params={"type": "parcelle", **pt})
    assert r.status_code == 200 and r.content.startswith(b"%PDF") and "attachment" in r.headers["content-disposition"]
    assert client.get("/api/atom-wind/agricole", params={"lieu": "Paris"}).status_code == 422
    assert client.get("/api/atom-wind/rapport.pdf", params={"type": "autre"}).status_code == 422


def test_qualite_sans_evaluation(client, jeu):  # noqa: F811
    r = client.get("/api/atom-wind/qualite")
    ev = jeu.paths.run_dir / "eval_pcfno3d"
    if not any(ev.glob("*/test_summary.json")):
        assert r.status_code == 404 and "evaluation absente" in r.json()["detail"]
    else:
        assert r.status_code == 200 and [l["code"] for l in r.json()["dso"]][:5] == ["DSO1", "DSO2", "DSO3", "DSO4", "DSO5"]


def test_sans_cle_refuse(client):
    from fastapi.testclient import TestClient
    nu = TestClient(client.app)
    assert nu.get("/api/atom-wind/agricole").status_code == 401


def test_historique(client, jeu, monkeypatch):  # noqa: F811
    """BF40 / HIS1 : chaque journee calculee est archivee et se relit par l'API, nom verifie."""
    from pcfno3d import atom_wind
    monkeypatch.setenv("PCFNO3D_HISTORIQUE", "1")
    assert client.get("/api/atom-wind/agricole", params={"j": 1, "lat": LAT, "lon": LON}).status_code == 200
    h = client.get("/api/atom-wind/historique").json()["journees"]
    j1 = [r for r in h if r["type"] == "prevision" and r["echeance_j"] == 1]       # d'autres tests en ont ecrit
    assert j1 and j1[0]["emise_le"] == atom_wind.date_locale_aujourdhui(0)
    assert j1[0]["date"] == atom_wind.date_locale_aujourdhui(1)
    d = client.get(f"/api/atom-wind/historique/{j1[0]['nom']}")
    assert d.status_code == 200 and d.json()["archive"] == j1[0]["nom"] and d.json()["date"] == j1[0]["date"]
    assert client.get("/api/atom-wind/historique/x.json.gz").status_code == 422
    assert client.get("/api/atom-wind/historique/2000-01-01_temps_reel.json.gz").status_code == 404


def test_stations_entree_de_production(client, jeu, monkeypatch):  # noqa: F811
    """BF1 / BF4 : pour le modele reel, des stations envoyees a l'API entrent comme en production (GFS de
    l'heure + stations a 10 m dans leur maille), pas dans la couche dense du modele synthetique."""
    from service import atom_outils
    vus = {}
    vrai = atom_outils.entree_production

    def espion(wf, ts, pts):
        vus["pts"] = pts
        return vrai(wf, ts, pts)
    monkeypatch.setattr(atom_outils, "entree_production", espion)
    corps = {"timestamp_utc": "2026-10-26T06:00:00Z", "stations": [
        {"station_id": "A", "lat": LAT, "lon": LON, "height_agl_m": 10, "wind_u_m_s": 3.0, "wind_v_m_s": -1.0},
        {"station_id": "B", "lat": LAT, "lon": LON, "height_agl_m": 30, "wind_u_m_s": 6.0, "wind_v_m_s": 0.0},
        {"station_id": "loin", "lat": 40.0, "lon": 5.0, "wind_u_m_s": 1.0, "wind_v_m_s": 1.0}]}
    r = client.post("/api/reconstruct/stations", json=corps)
    assert r.status_code == 200, r.text
    d = r.json()["stations"]
    assert d["entree"].startswith("production") and d["mailles"] == 1 and d["hors_domaine"] >= 1
    (iy, ix, u, v), = vus["pts"]                   # A et B dans la meme maille : moyenne, B ramene a 10 m
    assert 3.0 < u < 4.5 and v == -0.5
    # Open-Meteo injoignable : 503 avec la cause, jamais une reconstruction sur une mauvaise entree
    from pcfno3d import temps_reel

    def panne(_url, **_kw):
        raise OSError("reseau coupe")
    monkeypatch.setattr(temps_reel, "_get", panne)
    corps["timestamp_utc"] = "2026-09-01T00:00:00Z"
    r = client.post("/api/reconstruct/stations", json=corps)
    assert r.status_code == 503 and "GFS" in r.json()["detail"]
