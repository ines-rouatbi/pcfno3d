"""Alertes SMS (E12) : abonnes, message FR / AR, journal, Twilio simule, routes administrateur."""

from __future__ import annotations

import json
import urllib.parse

import pytest

from test_temps_reel import _jeu_reel, cfg_era5, fausse_source, forcage, jeu  # noqa: F401
from test_assistant import LAT, LON, wf  # noqa: F401


def test_abonnes(tmp_path):
    from service import sms
    ab = sms.Abonnes(tmp_path)
    ab.ajouter("+21612345678", "ar", lieu="Mornag")
    ab.ajouter("+21612345678", "fr", lieu="Mornag")          # remplace, pas de doublon
    assert [a["langue"] for a in ab.lire()] == ["fr"]
    with pytest.raises(ValueError, match="E.164"):
        ab.ajouter("0612345678")
    with pytest.raises(ValueError, match="lieu"):
        ab.ajouter("+21612345679")
    assert ab.retirer("+21612345678") and not ab.retirer("+21612345678")
    assert sms.masquer("+21612345678") == "+216*****678"


def test_messages_fr_ar_et_journal(wf, tmp_path):  # noqa: F811
    from service import sms
    for langue, mot in (("fr", "demain"), ("ar", "غدًا")):
        m = sms.message(wf, {"numero": "+21612345678", "langue": langue, "lat": LAT, "lon": LON, "lieu": "Ma parcelle"})
        assert mot in m and len(m) < 320
    sms.Abonnes(tmp_path).ajouter("+21612345678", "fr", lat=LAT, lon=LON, lieu="Ma parcelle")
    r = sms.envoyer_tous(wf, tmp_path, jour=1, simulation=True)
    assert r["fournisseur"] == "journal" and r["resultats"][0]["envoye"] is False
    ligne = json.loads((tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert ligne["a"] == "+216*****678" and "12345" not in json.dumps(ligne)


def test_twilio_simule(monkeypatch):
    from service import sms
    for k, v in (("TWILIO_ACCOUNT_SID", "AC123"), ("TWILIO_AUTH_TOKEN", "secret"), ("TWILIO_FROM", "+15550000000")):
        monkeypatch.setenv(k, v)
    vu = {}

    class Reponse:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b'{"status": "queued", "sid": "SM1"}'

    def ouvrir(req, timeout):
        vu["url"], vu["corps"], vu["auth"] = req.full_url, req.data, req.headers["Authorization"]
        return Reponse()

    r = sms.FournisseurTwilio(ouvrir=ouvrir).envoyer("+21612345678", "test")
    assert r == {"envoye": True, "mode": "twilio", "statut": "queued", "sid": "SM1"}
    assert vu["url"].endswith("/Accounts/AC123/Messages.json")
    assert urllib.parse.parse_qs(vu["corps"].decode())["To"] == ["+21612345678"]
    assert vu["auth"].startswith("Basic ")


def test_routes_sms_admin(jeu, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient

    from pcfno3d import temps_reel
    monkeypatch.setattr(temps_reel, "_get", fausse_source)
    for var, val in (("PCFNO3D_DATA_DIR", jeu.paths.data_dir), ("PCFNO3D_CACHE_DIR", jeu.paths.cache_dir),
                     ("PCFNO3D_RUN_DIR", jeu.paths.run_dir), ("PCFNO3D_DEVICE", "cpu")):
        monkeypatch.setenv(var, str(val))
    for var in ("PCFNO3D_API_KEYS", "PCFNO3D_INSECURE_OPEN", "PCFNO3D_SMS", "PCFNO3D_SMS_HEURE"):
        monkeypatch.delenv(var, raising=False)
    import pcfno3d.config as cm
    monkeypatch.setattr(cm.Config, "load", classmethod(lambda cls, _n: jeu))
    from service.auth import ApiKeyStore, default_store_path
    chemin = default_store_path(jeu.paths.run_dir)
    chemin.unlink(missing_ok=True)
    store = ApiKeyStore(chemin)
    lecture, _ = store.create("lecture", scope="read")
    admin, _ = store.create("admin", scope="admin")
    from service.app import create_app
    with TestClient(create_app()) as c:
        corps = {"numero": "+21612345678", "langue": "ar", "lat": LAT, "lon": LON, "lieu": "Ma parcelle"}
        assert c.post("/api/sms/abonnes", json=corps, headers={"X-API-Key": lecture}).status_code == 403
        r = c.post("/api/sms/abonnes", json=corps, headers={"X-API-Key": admin})
        assert r.status_code == 200 and r.json()["numero"] == "+216*****678"
        assert c.get("/api/sms/abonnes", headers={"X-API-Key": admin}).json()["abonnes"][0]["numero"] == "+216*****678"
        e = c.post("/api/sms/envoyer", params={"jour": 1}, headers={"X-API-Key": admin}).json()
        assert e["fournisseur"] == "journal" and e["abonnes"] == 1 and "غدًا" in e["resultats"][0]["texte"]
        assert c.delete("/api/sms/abonnes/+21612345678", headers={"X-API-Key": admin}).status_code == 200
