"""Assistant integre (E6) et lecture en un point (service.atom_outils), avec un client Claude simule :
aucun appel reseau, aucune cle."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from test_temps_reel import _jeu_reel, cfg_era5, fausse_source, forcage, jeu  # noqa: F401

LAT, LON = 36.005, 10.005          # dans le mini-domaine (coin sud-ouest 36,0 N / 10,0 E)


@pytest.fixture
def wf(jeu, monkeypatch):  # noqa: F811
    from pcfno3d import temps_reel
    from pcfno3d.inference import WindFieldModel
    from service import atom_outils
    monkeypatch.setattr(temps_reel, "_get", fausse_source)
    for c in atom_outils._CACHES.values():
        c.clear()
    return WindFieldModel(jeu, device="cpu", verbose=False)


def test_lecture_en_un_point(wf):
    from service import atom_outils as ao
    p = ao.lire_point(wf, 1, 10.0, 9.0, lat=LAT, lon=LON)
    assert p["source"] == "prevision" and p["vitesse_m_s"] >= 0 and p["temperature_2m_c"] is not None
    assert 0 <= p["direction_provenance_deg"] < 360
    pr = ao.profil(wf, 1, 9.0, lat=LAT, lon=LON)
    assert [n["hauteur_sol_m"] for n in pr["niveaux"]][:2] == [10, 25]
    d = ao.decision_agricole(wf, 1, lat=LAT, lon=LON)
    assert len(d["heures"]) == 24 and {h["traitement"] for h in d["heures"]} <= {"oui", "attention", "non"}
    e = ao.production_eolienne(wf, 1, 80.0, lat=LAT, lon=LON)
    assert 0 <= e["facteur_charge"] <= 1 and len(e["heures"]) == 24
    a = ao.alertes_vent(wf, 1)
    assert "max_10m" in a and "max_100m" in a
    with pytest.raises(ValueError, match="hors du domaine"):
        ao.lire_point(wf, 1, 10.0, 9.0, lieu="Nabeul")          # hors du mini-domaine
    with pytest.raises(ValueError, match="lieu inconnu"):
        ao.position(wf, lieu="Paris")


def test_maintenance_et_variations(wf, monkeypatch):
    from service import atom_outils as ao
    m = ao.fenetres_maintenance(wf, 1, 80.0, 12.0, 2, lat=LAT, lon=LON)
    assert len(m["heures"]) == 24 and m["seuil_m_s"] == 12.0
    sures = {h["heure"] for h in m["heures"] if h["sur"]}
    for c in m["creneaux_surs"]:                       # chaque plage : heures sures et au moins 2 h
        a, b = (int(t.split()[0]) for t in c.split(" - "))
        assert b - a >= 2 and set(range(a, b)) <= sures
    assert ao.fenetres_maintenance(wf, 1, 80.0, 0.01, 2, lat=LAT, lon=LON)["creneaux_surs"] == []
    monkeypatch.setattr(ao, "PARC_VIRTUEL", (("E1", LAT, LON), ("E2", LAT + 0.002, LON + 0.002)))
    v = ao.variations_production(wf, 1, 80.0, seuil_pct=0.001)
    assert len(v["puissance_kw"]) == 24 and v["puissance_installee_kw"] == 4000
    for r in v["rampes"]:
        assert r["variation_kw"] == v["puissance_kw"][r["a"]] - v["puissance_kw"][r["de"]]
    assert ao.variations_production(wf, 1, 80.0, seuil_pct=100)["rampes"] == []


class FauxClaude:
    """Premier tour : appelle vent_au_point ; second tour : repond par du texte."""

    def __init__(self):
        self.requetes = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.requetes.append(kw)
        if len(self.requetes) == 1:
            return NS(stop_reason="tool_use", model=kw["model"], content=[
                NS(type="text", text="Je regarde."),
                NS(type="tool_use", id="t1", name="vent_au_point",
                   input={"lieu": None, "lat": LAT, "lon": LON, "jour": 1, "heure": 9, "hauteur_m": 10})])
        return NS(stop_reason="end_turn", model=kw["model"], content=[NS(type="text", text="Oui, vous pouvez traiter.")])


def test_assistant_boucle_outils(wf):
    from service import assistant
    fc = FauxClaude()
    r = assistant.repondre(wf, "Peut-on traiter demain a 9 h ?", cl=fc)
    assert r["reponse"] == "Oui, vous pouvez traiter." and r["arret"] == "end_turn"
    assert r["outils"][0]["nom"] == "vent_au_point"
    q1, q2 = fc.requetes
    assert q1["model"] == "claude-opus-5-5" and q1["fallbacks"] == "default" and q1["betas"] == [assistant.BETA_REPLI]
    assert all(t["strict"] for t in q1["tools"]) and q1["tools"][-1]["cache_control"]["type"] == "ephemeral"
    resultat = q2["messages"][-1]["content"][0]
    assert resultat["type"] == "tool_result" and "is_error" not in resultat
    assert json.loads(resultat["content"])["source"] == "prevision"


def test_assistant_erreur_d_outil_et_refus(wf):
    from service import assistant

    class Hors(FauxClaude):
        def create(self, **kw):
            self.requetes.append(kw)
            if len(self.requetes) == 1:
                return NS(stop_reason="tool_use", model=kw["model"], content=[
                    NS(type="tool_use", id="t1", name="vent_au_point",
                       input={"lieu": "Paris", "lat": None, "lon": None, "jour": 0, "heure": 9, "hauteur_m": 10})])
            return NS(stop_reason="refusal", model=kw["model"], content=[])
    h = Hors()
    r = assistant.repondre(wf, "et a Paris ?", cl=h)
    assert r["arret"] == "refus"
    err = h.requetes[1]["messages"][-1]["content"][0]
    assert err["is_error"] is True and "lieu inconnu" in err["content"]


def test_route_assistant_sans_cle(jeu, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient
    for var, val in (("PCFNO3D_DATA_DIR", jeu.paths.data_dir), ("PCFNO3D_CACHE_DIR", jeu.paths.cache_dir),
                     ("PCFNO3D_RUN_DIR", jeu.paths.run_dir), ("PCFNO3D_DEVICE", "cpu")):
        monkeypatch.setenv(var, str(val))
    for var in ("PCFNO3D_API_KEYS", "PCFNO3D_INSECURE_OPEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    import pcfno3d.config as cm
    monkeypatch.setattr(cm.Config, "load", classmethod(lambda cls, _n: jeu))
    from service.auth import ApiKeyStore, default_store_path
    chemin = default_store_path(jeu.paths.run_dir)
    chemin.unlink(missing_ok=True)
    cle, _ = ApiKeyStore(chemin).create("tests-assistant", scope="read")
    from service.app import create_app
    with TestClient(create_app(), headers={"X-API-Key": cle}) as c:
        r = c.post("/api/assistant", json={"question": "Quel vent demain ?"})
        assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
        assert c.post("/api/assistant", json={"question": ""}).status_code == 422
