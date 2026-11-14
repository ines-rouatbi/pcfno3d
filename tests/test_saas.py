"""Service SaaS sans modele livre : il demarre, sert les comptes et les alertes de chaque utilisateur, et
les routes de donnees disent clairement que le modele n'est pas installe."""

from __future__ import annotations

import copy

import pytest


@pytest.fixture
def client_sans_modele(cfg, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    c = copy.deepcopy(cfg)
    c.apply_overrides({"paths.run_dir": tmp_path / "run_vide"})       # aucun best.pt ici
    c.paths.mkdirs()
    for var, val in (("PCFNO3D_DATA_DIR", c.paths.data_dir), ("PCFNO3D_CACHE_DIR", c.paths.cache_dir),
                     ("PCFNO3D_RUN_DIR", c.paths.run_dir), ("PCFNO3D_DEVICE", "cpu")):
        monkeypatch.setenv(var, str(val))
    for var in ("PCFNO3D_API_KEYS", "PCFNO3D_INSECURE_OPEN", "PCFNO3D_CHECKPOINT"):
        monkeypatch.delenv(var, raising=False)
    import pcfno3d.config as cm
    monkeypatch.setattr(cm.Config, "load", classmethod(lambda cls, _n: c))
    from service.auth import ApiKeyStore, default_store_path
    cle, _ = ApiKeyStore(default_store_path(c.paths.run_dir)).create("worker", scope="read")
    from service.app import create_app
    with TestClient(create_app(), headers={"X-API-Key": cle}) as client:
        yield client


def test_service_sans_modele(client_sans_modele):
    c = client_sans_modele
    h = c.get("/api/health").json()
    assert h["modele_installe"] is False and h["status"] == "degraded"
    r = c.get("/api/atom-wind/grille")
    assert r.status_code == 503 and "non installe" in r.json()["detail"]


def test_alertes_de_l_utilisateur(client_sans_modele):
    c = client_sans_modele
    assert c.post("/api/comptes/moi/abonnements", json={"numero": "+21612345678", "lieu": "Mornag"}).status_code == 401
    for email in ("a@exemple.tn", "b@exemple.tn"):
        assert c.post("/api/comptes", json={"email": email, "mot_de_passe": "motdepasse1", "nom": email,
                                             "role": "agri"}).status_code == 200
    sa = c.post("/api/comptes/connexion", json={"email": "a@exemple.tn", "mot_de_passe": "motdepasse1"}).json()["cle"]
    sb = c.post("/api/comptes/connexion", json={"email": "b@exemple.tn", "mot_de_passe": "motdepasse1"}).json()["cle"]
    ha, hb = {"X-API-Key": sa}, {"X-API-Key": sb}

    r = c.post("/api/comptes/moi/abonnements", headers=ha,
               json={"numero": "+21612345678", "lieu": "Mornag", "activite": "traiter"})
    assert r.status_code == 200 and r.json()["numero"] != "+21612345678" and r.json()["compte"] == "a@exemple.tn"
    assert c.post("/api/comptes/moi/abonnements", headers=ha, json={"numero": "123", "lieu": "Mornag"}).status_code == 400
    # le numero d'un autre compte ne peut pas etre repris
    assert c.post("/api/comptes/moi/abonnements", headers=hb,
                  json={"numero": "+21612345678", "lieu": "Nabeul"}).status_code == 409
    la = c.get("/api/comptes/moi/abonnements", headers=ha).json()["abonnements"]
    assert len(la) == 1 and la[0]["activite"] == "traiter"
    assert c.get("/api/comptes/moi/abonnements", headers=hb).json()["abonnements"] == []
    assert c.delete(f"/api/comptes/moi/abonnements/{la[0]['numero']}", headers=hb).status_code == 404
    assert c.delete(f"/api/comptes/moi/abonnements/{la[0]['numero']}", headers=ha).status_code == 200
    assert c.get("/api/comptes/moi/abonnements", headers=ha).json()["abonnements"] == []
