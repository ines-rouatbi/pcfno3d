"""Comptes et roles cote serveur (BF19, BF43) : inscription, adresse e-mail verifiee par un lien qui expire,
connexion par cle de session qui expire, mot de passe oublie, chaque espace reserve a SON role (agriculteur,
exploitant eolien, administration), desactivation immediate, blocage apres 5 echecs, mot de passe jamais garde."""

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


def _lien(dossier, sorte):
    """Le jeton du dernier e-mail ecrit dans la boite locale (PCFNO3D_COURRIELS_LOCAUX=1)."""
    import email
    import re
    f = sorted((dossier / "courriels_locaux").glob("*.eml"))[-1]
    corps = email.message_from_bytes(f.read_bytes()).get_payload(decode=True).decode("utf-8")
    return re.search(sorte + r"=([A-Za-z0-9_\-]+)", corps).group(1)


def test_liens_verification_et_reinitialisation(tmp_path, monkeypatch):
    import service.comptes as cpt
    from service.auth import ApiKeyStore
    c = cpt.Comptes(tmp_path, ApiKeyStore(tmp_path / "cles.json"))
    c.creer("b@exemple.tn", "motdepasse1", role="agri")
    with pytest.raises(cpt.EmailNonVerifie):                    # bon mot de passe, adresse pas encore verifiee
        c.connecter("b@exemple.tn", "motdepasse1")
    with pytest.raises(ValueError, match="incorrect"):           # mauvais mot de passe : rien n'est revele
        c.connecter("b@exemple.tn", "mauvais123")
    j1 = c.jeton_verification("b@exemple.tn")
    assert j1 and c.jeton_verification("b@exemple.tn") is None  # au plus un lien par minute
    assert c.jeton_verification("inconnu@exemple.tn") is None
    assert j1 not in (tmp_path / "comptes.json").read_text(encoding="utf-8")   # seule l'empreinte est gardee
    # un nouveau lien annule l'ancien ; un lien expire est refuse
    monkeypatch.setattr(cpt, "RENVOI_MIN_S", 0)
    j2 = c.jeton_verification("b@exemple.tn")
    with pytest.raises(cpt.LienInvalide, match="invalide"):
        c.verifier_email(j1)
    t = cpt.time.time()
    monkeypatch.setattr(cpt.time, "time", lambda: t + cpt.VERIFICATION_S + 5)
    with pytest.raises(cpt.LienInvalide, match="expire"):
        c.verifier_email(j2)
    monkeypatch.undo()
    monkeypatch.setattr(cpt, "RENVOI_MIN_S", 0)
    j3 = c.jeton_verification("b@exemple.tn")
    assert c.verifier_email(j3)["email_verifie"] is True
    with pytest.raises(cpt.LienInvalide):                        # usage unique
        c.verifier_email(j3)
    cle, compte = c.connecter("b@exemple.tn", "motdepasse1")
    assert compte["role"] == "agri"
    # la session expire : il faut redonner son mot de passe
    rec = c.store.verify(cle)
    assert rec.expires_at and not rec.expired() and rec.expired(cpt.time.time() + cpt.SESSION_S + 5)
    # mot de passe oublie : lien d'1 h, la session ouverte tombe, l'ancien mot de passe ne vaut plus rien
    r = c.jeton_reinitialisation("b@exemple.tn")
    with pytest.raises(ValueError, match="chiffre"):
        c.reinitialiser(r, "court")
    c.reinitialiser(r, "nouveau2026")
    assert c.store.verify(cle) is None
    with pytest.raises(ValueError):
        c.connecter("b@exemple.tn", "motdepasse1")
    assert c.connecter("b@exemple.tn", "nouveau2026")[1]["email"] == "b@exemple.tn"
    with pytest.raises(cpt.LienInvalide):
        c.reinitialiser(r, "encore2026")
    # un compte d'avant la verification d'e-mail reste utilisable
    liste = json.loads((tmp_path / "comptes.json").read_text(encoding="utf-8"))
    liste[0].pop("email_verifie")
    (tmp_path / "comptes.json").write_text(json.dumps(liste), encoding="utf-8")
    assert c.lister()[0]["email_verifie"] is True
    # administration : role (jamais admin), adresse confirmee a la main
    c.creer("c@exemple.tn", "motdepasse1")
    assert c.changer_role("c@exemple.tn", "eolien")["role"] == "eolien"
    with pytest.raises(ValueError):
        c.changer_role("c@exemple.tn", "admin")
    assert c.marquer_verifie("c@exemple.tn")["email_verifie"] is True
    # suppression : l'entree disparait, la session tombe, un administrateur ne se supprime pas ici
    cle_c, _ = c.connecter("c@exemple.tn", "motdepasse1")
    assert c.supprimer("C@exemple.tn")["email"] == "c@exemple.tn"
    assert "c@exemple.tn" not in (tmp_path / "comptes.json").read_text(encoding="utf-8")
    assert c.store.verify(cle_c) is None
    with pytest.raises(KeyError):
        c.supprimer("c@exemple.tn")
    c.creer_admin("chef@exemple.tn", "admin2026!")
    with pytest.raises(ValueError, match="administrateur"):
        c.supprimer("chef@exemple.tn")


def test_parcours_http(client, jeu, monkeypatch):  # noqa: F811
    import service.comptes as cpt
    monkeypatch.setattr(cpt, "RENVOI_MIN_S", 0)
    monkeypatch.setenv("PCFNO3D_COURRIELS_LOCAUX", "1")
    monkeypatch.delenv("PCFNO3D_SMTP_HOTE", raising=False)
    monkeypatch.setenv("PCFNO3D_URL_APP", "https://atom.exemple.tn/app/atom-wind/")
    run = jeu.paths.run_dir
    nu = TestClient(client.app)
    r = client.post("/api/comptes", json={"email": "agri@exemple.tn", "mot_de_passe": "champ2026", "role": "agri"})
    assert r.status_code == 200 and r.json()["role"] == "agri" and "hachage" not in r.text
    assert r.json()["email_verifie"] is False and r.json()["verification"]["envoye"] is True
    # pas de connexion tant que l'adresse n'est pas verifiee, meme avec le bon mot de passe
    assert client.post("/api/comptes/connexion", json={"email": "agri@exemple.tn", "mot_de_passe": "x"}).status_code == 401
    r = client.post("/api/comptes/connexion", json={"email": "agri@exemple.tn", "mot_de_passe": "champ2026"})
    assert r.status_code == 403 and "verifiee" in r.json()["detail"]
    jeton = _lien(run, "verifier-email")
    assert "https://atom.exemple.tn/app/atom-wind/#verifier-email=" in next(
        (run / "courriels_locaux").glob("*.eml")).read_text(encoding="utf-8")
    assert client.post("/api/comptes/verification", json={"jeton": "x" * 43}).status_code == 400
    assert client.post("/api/comptes/verification", json={"jeton": jeton}).json()["email_verifie"] is True
    assert client.post("/api/comptes/verification", json={"jeton": jeton}).status_code == 400   # usage unique
    # meme reponse pour une adresse inconnue (pas d'enumeration)
    assert client.post("/api/comptes/verification/renvoi", json={"email": "personne@exemple.tn"}).json()["ok"]
    assert client.post("/api/comptes/mot-de-passe/oubli", json={"email": "personne@exemple.tn"}).json()["ok"]
    r = client.post("/api/comptes", json={"email": "eol@exemple.tn", "mot_de_passe": "rotor2026", "role": "eolien"})
    assert r.json()["role"] == "public" and r.json()["demande_role"] == "eolien"
    client.post("/api/comptes/verification", json={"jeton": _lien(run, "verifier-email")})
    r = client.post("/api/comptes/connexion", json={"email": "agri@exemple.tn", "mot_de_passe": "champ2026"})
    assert r.json()["expire_dans_s"] == 8 * 3600
    h = {"X-API-Key": r.json()["cle"]}
    assert nu.get("/api/comptes/moi", headers=h).json()["compte"]["role"] == "agri"
    pt = {"lat": LAT, "lon": LON}
    # espace agriculteur : l'agriculteur seulement
    assert nu.get("/api/atom-wind/agricole", params={"j": 1, **pt}, headers=h).status_code == 200
    assert nu.get("/api/atom-wind/production", params={"j": 1, **pt}, headers=h).status_code == 403
    assert nu.get("/api/atom-wind/rapport.pdf", params={"type": "site", **pt}, headers=h).status_code == 403
    assert nu.get("/api/atom-wind/scenario", params={"heure": 1, "retirer": "x"}, headers=h).status_code == 403
    assert nu.get("/api/comptes", headers=h).status_code == 403
    # la cle de lecture (celle que le Worker ajoute pour un visiteur sans compte) n'ouvre aucun espace reserve
    assert client.get("/api/atom-wind/production", params={"j": 1, **pt}).status_code == 403
    r = client.get("/api/atom-wind/agricole", params={"j": 1, **pt})
    assert r.status_code == 403 and r.headers.get("cache-control") == "private, no-store"
    # une cle de service de portee admin (ligne de commande) garde l'acces technique
    from service.auth import ApiKeyStore, default_store_path
    admin, _ = ApiKeyStore(default_store_path(run)).create("comptes-admin", scope="admin")
    ha = {"X-API-Key": admin}
    assert nu.get("/api/atom-wind/production", params={"j": 1, **pt}, headers=ha).status_code == 200
    assert client.get("/api/comptes").status_code == 403
    assert {c["email"] for c in nu.get("/api/comptes", headers=ha).json()["comptes"]} >= {"agri@exemple.tn", "eol@exemple.tn"}
    cle_eol = client.post("/api/comptes/connexion", json={"email": "eol@exemple.tn", "mot_de_passe": "rotor2026"}).json()["cle"]
    he = {"X-API-Key": cle_eol}
    assert nu.get("/api/atom-wind/production", params={"j": 1, **pt}, headers=he).status_code == 403
    # la page encode l'adresse dans le chemin (%40) : le serveur la decode
    assert nu.post("/api/comptes/eol%40exemple.tn/validation", headers=ha).json()["role"] == "eolien"
    assert nu.get("/api/atom-wind/production", params={"j": 1, **pt}, headers=he).status_code == 200
    assert nu.get("/api/atom-wind/agricole", params={"j": 1, **pt}, headers=he).status_code == 403
    # un compte administrateur gere les comptes, mais n'entre ni dans l'espace agriculteur ni dans l'espace eolien
    from service.comptes import Comptes
    Comptes(run, ApiKeyStore(default_store_path(run))).creer_admin("chef@exemple.tn", "admin2026!")
    r = client.post("/api/comptes/connexion", json={"email": "chef@exemple.tn", "mot_de_passe": "admin2026!"})
    hadm = {"X-API-Key": r.json()["cle"]}
    assert nu.get("/api/comptes", headers=hadm).status_code == 200
    assert nu.get("/api/atom-wind/production", params={"j": 1, **pt}, headers=hadm).status_code == 403
    assert nu.get("/api/atom-wind/agricole", params={"j": 1, **pt}, headers=hadm).status_code == 403
    # gestion des comptes : changer le role, confirmer une adresse
    assert nu.post("/api/comptes/eol@exemple.tn/role", params={"role": "agri"}, headers=hadm).json()["role"] == "agri"
    assert nu.post("/api/comptes/eol@exemple.tn/role", params={"role": "admin"}, headers=hadm).status_code == 422
    assert nu.post("/api/comptes/eol@exemple.tn/role", params={"role": "eolien"}, headers=hadm).json()["role"] == "eolien"
    client.post("/api/comptes", json={"email": "x@exemple.tn", "mot_de_passe": "xavier2026"})
    assert nu.post("/api/comptes/x@exemple.tn/email-verifie", headers=hadm).json()["email_verifie"] is True
    # mot de passe oublie : lien envoye, nouveau mot de passe, la session ouverte tombe
    client.post("/api/comptes/mot-de-passe/oubli", json={"email": "agri@exemple.tn"})
    jr = _lien(run, "nouveau-mot-de-passe")
    assert client.post("/api/comptes/mot-de-passe/reinitialisation",
                       json={"jeton": jr, "mot_de_passe": "recolte2026"}).json()["ok"] is True
    assert nu.get("/api/comptes/moi", headers=h).status_code == 401
    assert client.post("/api/comptes/connexion", json={"email": "agri@exemple.tn", "mot_de_passe": "champ2026"}).status_code == 401
    cle2 = client.post("/api/comptes/connexion", json={"email": "agri@exemple.tn", "mot_de_passe": "recolte2026"}).json()["cle"]
    # un compte desactive perd sa session aussitot
    assert nu.post("/api/comptes/agri@exemple.tn/statut", params={"actif": False}, headers=ha).json()["statut"] == "desactive"
    assert nu.get("/api/comptes/moi", headers={"X-API-Key": cle2}).status_code == 401
    assert client.post("/api/comptes/connexion", json={"email": "agri@exemple.tn", "mot_de_passe": "recolte2026"}).status_code == 401
    # deconnexion
    assert nu.post("/api/comptes/deconnexion", headers=he).json() == {"deconnecte": True}
    assert nu.get("/api/comptes/moi", headers=he).status_code == 401
    # suppression par l'administrateur : session fermee, alertes SMS du compte retirees, connexion impossible
    cle_x = client.post("/api/comptes/connexion", json={"email": "x@exemple.tn", "mot_de_passe": "xavier2026"}).json()["cle"]
    hx = {"X-API-Key": cle_x}
    assert nu.post("/api/comptes/moi/abonnements", headers=hx, json={"numero": "+21698765432", "lieu": "Mornag"}).status_code == 200
    assert nu.delete("/api/comptes/x@exemple.tn", headers=hx).status_code == 403            # pas un admin
    r = nu.delete("/api/comptes/x%40exemple.tn", headers=hadm)
    assert r.json() == {"supprime": True, "email": "x@exemple.tn", "alertes_sms_retirees": 1}
    assert nu.get("/api/comptes/moi", headers=hx).status_code == 401
    assert client.post("/api/comptes/connexion", json={"email": "x@exemple.tn", "mot_de_passe": "xavier2026"}).status_code == 401
    assert "x@exemple.tn" not in {c["email"] for c in nu.get("/api/comptes", headers=hadm).json()["comptes"]}
    assert nu.delete("/api/comptes/x@exemple.tn", headers=hadm).status_code == 404
    assert nu.delete("/api/comptes/chef@exemple.tn", headers=hadm).status_code == 422       # un admin : non


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


def test_assistant_respecte_le_role_du_serveur():
    """Les outils agricoles et eoliens de l'assistant suivent le role de la SESSION, pas celui que dit la page."""
    from service import assistant as asst
    for role, outil, refuse in (("agri", "production_eolienne", True), ("visiteur", "decision_agricole", True),
                                ("admin", "plan_haies", True), ("public", "variations_production", True),
                                ("eolien", "decision_agricole", True)):
        jeton = asst.ROLE_SERVEUR.set(role)
        try:
            assert asst.executer(None, outil, {}).get("refuse") is refuse, (role, outil)
        finally:
            asst.ROLE_SERVEUR.reset(jeton)
