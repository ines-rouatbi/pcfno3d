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
