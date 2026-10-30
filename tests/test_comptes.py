"""Comptes et roles cote serveur (BF19, BF43) : inscription, connexion par cle de session, espace eolien
reserve aux comptes valides, desactivation immediate, blocage apres 5 echecs, mot de passe jamais garde."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from test_temps_reel import _jeu_reel, cfg_era5, fausse_source, forcage, jeu  # noqa: F401
from test_routes_v12 import client  # noqa: F401
from test_assistant import LAT, LON  # noqa: F401


def test_hachage_et_regles(tmp_path):
    from service.auth import ApiKeyStore
    from service.comptes import Comptes
    c = Comptes(tmp_path, ApiKeyStore(tmp_path / "cles.json"))
    with pytest.raises(ValueError, match="e-mail"):
        c.creer("pas-une-adresse", "motdepasse1")
    with pytest.raises(ValueError, match="chiffre"):
        c.creer("a@exemple.tn", "sanschiffre")
    with pytest.raises(ValueError, match="administrateur"):
        c.creer("a@exemple.tn", "motdepasse1", role="admin")
    c.creer("A@Exemple.tn", "motdepasse1", role="agri")
    with pytest.raises(ValueError, match="existe deja"):
        c.creer("a@exemple.tn", "autre1234")
    brut = (tmp_path / "comptes.json").read_text(encoding="utf-8")
    assert "motdepasse1" not in brut and json.loads(brut)[0]["email"] == "a@exemple.tn"
    for _ in range(5):
        with pytest.raises(ValueError):
            c.connecter("a@exemple.tn", "mauvais")
    with pytest.raises(PermissionError):                       # bloque, meme avec le bon mot de passe
        c.connecter("a@exemple.tn", "motdepasse1")


def test_parcours_http(client, jeu):  # noqa: F811
    nu = TestClient(client.app)
    r = client.post("/api/comptes", json={"email": "agri@exemple.tn", "mot_de_passe": "champ2026", "role": "agri"})
    assert r.status_code == 200 and r.json()["role"] == "agri" and "hachage" not in r.text
    r = client.post("/api/comptes", json={"email": "eol@exemple.tn", "mot_de_passe": "rotor2026", "role": "eolien"})
    assert r.json()["role"] == "public" and r.json()["demande_role"] == "eolien"
    assert client.post("/api/comptes/connexion", json={"email": "agri@exemple.tn", "mot_de_passe": "x"}).status_code == 401
    cle_agri = client.post("/api/comptes/connexion", json={"email": "agri@exemple.tn", "mot_de_passe": "champ2026"}).json()["cle"]
    h = {"X-API-Key": cle_agri}
    assert nu.get("/api/comptes/moi", headers=h).json()["compte"]["role"] == "agri"
    pt = {"lat": LAT, "lon": LON}
    assert nu.get("/api/atom-wind/agricole", params={"j": 1, **pt}, headers=h).status_code == 200
    assert nu.get("/api/atom-wind/production", params={"j": 1, **pt}, headers=h).status_code == 403
    assert nu.get("/api/atom-wind/rapport.pdf", params={"type": "site", **pt}, headers=h).status_code == 403
    # la cle de service (celle du client) garde l'espace eolien
    assert client.get("/api/atom-wind/production", params={"j": 1, **pt}).status_code == 200
    # l'administrateur valide l'exploitant eolien
    from service.auth import ApiKeyStore, default_store_path
    admin, _ = ApiKeyStore(default_store_path(jeu.paths.run_dir)).create("comptes-admin", scope="admin")
    ha = {"X-API-Key": admin}
    assert client.get("/api/comptes").status_code == 403
    assert {c["email"] for c in nu.get("/api/comptes", headers=ha).json()["comptes"]} >= {"agri@exemple.tn", "eol@exemple.tn"}
    cle_eol = client.post("/api/comptes/connexion", json={"email": "eol@exemple.tn", "mot_de_passe": "rotor2026"}).json()["cle"]
    assert nu.get("/api/atom-wind/production", params={"j": 1, **pt}, headers={"X-API-Key": cle_eol}).status_code == 403
    assert nu.post("/api/comptes/eol@exemple.tn/validation", headers=ha).json()["role"] == "eolien"
    assert nu.get("/api/atom-wind/production", params={"j": 1, **pt}, headers={"X-API-Key": cle_eol}).status_code == 200
    # une nouvelle connexion remplace la session ; un compte desactive perd la sienne aussitot
    cle2 = client.post("/api/comptes/connexion", json={"email": "agri@exemple.tn", "mot_de_passe": "champ2026"}).json()["cle"]
    assert nu.get("/api/comptes/moi", headers=h).status_code == 401
    assert nu.post("/api/comptes/agri@exemple.tn/statut", params={"actif": False}, headers=ha).json()["statut"] == "desactive"
    assert nu.get("/api/comptes/moi", headers={"X-API-Key": cle2}).status_code == 401
    assert client.post("/api/comptes/connexion", json={"email": "agri@exemple.tn", "mot_de_passe": "champ2026"}).status_code == 401
    # deconnexion
    assert nu.post("/api/comptes/deconnexion", headers={"X-API-Key": cle_eol}).json() == {"deconnecte": True}
    assert nu.get("/api/comptes/moi", headers={"X-API-Key": cle_eol}).status_code == 401


def test_cli_compte_admin(jeu, monkeypatch, tmp_path):  # noqa: F811
    from pcfno3d.cli import main
    import pcfno3d.config as cm
    monkeypatch.setattr(cm.Config, "load", classmethod(lambda cls, _n: jeu))
    monkeypatch.delenv("PCFNO3D_MOT_DE_PASSE_ADMIN", raising=False)
    assert main(["-c", "x", "compte-admin", "chef@exemple.tn"]) == 2
    monkeypatch.setenv("PCFNO3D_MOT_DE_PASSE_ADMIN", "admin2026!")
    assert main(["-c", "x", "compte-admin", "chef@exemple.tn"]) == 0
    from service.auth import ApiKeyStore, default_store_path
    from service.comptes import Comptes
    cle, compte = Comptes(jeu.paths.run_dir, ApiKeyStore(default_store_path(jeu.paths.run_dir))).connecter(
        "chef@exemple.tn", "admin2026!")
    assert compte["role"] == "admin" and cle.startswith("pcf_")
