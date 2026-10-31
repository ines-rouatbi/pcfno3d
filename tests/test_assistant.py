"""Assistant integre (E6) et lecture en un point (service.atom_outils), avec un client Claude simule :
aucun appel reseau, aucune cle."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pandas as pd
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
    # sillages (BF72) : jamais plus de production qu'a vent libre
    assert all(a <= b + 1 for a, b in zip(v["puissance_kw"], v["puissance_sans_sillage_kw"]))
    assert 0.0 <= v["perte_sillage_pct"] <= 100.0


def test_explication_point(wf, monkeypatch):
    from pcfno3d import atom_wind
    from service import atom_outils as ao
    vrai = atom_wind.valeur_stations
    monkeypatch.setattr(atom_wind, "valeur_stations", lambda wf_, src, date, **kw: vrai(
        wf_, src, "2026-10-26", maintenant=pd.Timestamp("2026-10-26T07:20:00"), heures=(7,)))
    e = ao.explication_point(wf, lat=LAT, lon=LON)
    assert {s["code"] for s in e["stations"]} == {"DTTA", "INM60729"}
    assert sum(s["part_pct"] for s in e["stations"]) in (0, 99, 100, 101) and e["texte"]


def test_dimensionnement_mat(wf):
    from service import atom_outils as ao
    d = ao.dimensionnement_mat(wf, lat=LAT, lon=LON, hauteur_m=30.0, conso_w=5.0, autonomie_j=3.0)
    assert d["heures"] % 24 == 0 and len(d["production_w"]) == d["heures"]
    # au moins 3 jours de consommation, ramenes a la profondeur de decharge et au rendement
    assert d["batterie_wh"] >= round(5.0 * 24 * 3 / (0.8 * 0.9)) - 1
    assert d["plus_grand_deficit_wh"] >= 0 and 0 <= max(d["production_w"]) <= 400
    gros = ao.dimensionnement_mat(wf, lat=LAT, lon=LON, conso_w=500.0, autonomie_j=0.0)
    assert gros["plus_grand_deficit_wh"] > 0 and gros["batterie_wh"] >= gros["plus_grand_deficit_wh"]


@pytest.mark.parametrize("type_", ["parcelle", "site", "parc"])
def test_rapport_pdf(wf, monkeypatch, type_):
    from service import atom_outils as ao
    from service import rapport_pdf
    monkeypatch.setattr(ao, "PARC_VIRTUEL", (("E1", LAT, LON),))
    pdf = rapport_pdf.construire(wf, type_, lat=LAT, lon=LON)
    assert pdf.startswith(b"%PDF") and len(pdf) > 5000
    assert rapport_pdf.nom_fichier(type_, "Mornag (vignes)", None, None).startswith(f"atom_wind_{type_}_mornag_vignes_")
    with pytest.raises(ValueError, match="type de rapport"):
        rapport_pdf.construire(wf, "inconnu", lat=LAT, lon=LON)


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


def test_espaces_perspective(wf):
    """Holding (BF85), chantier (BF86), environnement (BF87) sur les journees disponibles."""
    import json as js
    import math
    from service import atom_outils as ao
    with pytest.raises(ValueError, match="Porteur"):
        ao.sites_porteurs(wf, hauteur_m=18)
    s = ao.sites_porteurs(wf, hauteur_m=20, n=4, ecart_km=0.5, marge_km=0.0)
    prod = [x["production_wh_par_jour"] for x in s["sites"]]
    assert 1 <= len(prod) <= 4 and prod == sorted(prod, reverse=True) and s["plan_aneo"] is None
    for a in s["sites"]:
        for b in s["sites"]:
            if a is not b:
                assert math.hypot(a["x_km"] - b["x_km"], a["y_km"] - b["y_km"]) >= 0.5
    # un plan OS6 present : ses capteurs sont classes et chaque site dit sa distance au plus proche
    d = wf.cfg.paths.run_dir / "os6_reseau_aneo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan_aneo.geojson").write_text(js.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"ordre": 1}, "geometry": {"type": "Point", "coordinates": [LON, LAT]}}]}))
    try:
        s = ao.sites_porteurs(wf, hauteur_m=20, n=2, ecart_km=0.5, marge_km=0.0)
        assert s["plan_aneo"][0]["ordre_os6"] == 1 and "capteur_aneo_prevu_km" in s["sites"][0]
    finally:
        (d / "plan_aneo.geojson").unlink()

    g = ao.vent_grue(wf, 1, 40.0, 20.0, 10.0, lat=LAT, lon=LON)
    assert len(g["heures"]) == 24
    for h in g["heures"]:
        assert h["rafale_m_s"] >= h["vent_m_s"] - 0.05
        assert (h["statut"] == "arret") == (h["rafale_m_s"] >= 20.0 - 0.05) or abs(h["rafale_m_s"] - 20.0) < 0.1
        assert h["effort_charge_n"] >= 0
    assert ao.vent_grue(wf, 1, 40.0, 0.01, lat=LAT, lon=LON)["creneaux_levage"] == []
    assert all(h["effort_charge_n"] == 0 for h in ao.vent_grue(wf, 1, 40.0, lat=LAT, lon=LON)["heures"])
    assert ao.facteur_rafale_hauteur(1.5, 0.1, 10.0) == pytest.approx(1.5)
    assert 1.0 < ao.facteur_rafale_hauteur(1.5, 0.1, 200.0) < 1.5
    assert ao.facteur_rafale_hauteur(None, 0.01, 10.0) == pytest.approx(1.22)

    hz = ao.plan_haies(wf, 3.0, 200.0, 0.0, lat=LAT, lon=LON)
    assert hz["azimut_haie_deg"] == round((hz["direction_dominante_deg"] + 90) % 180) % 180 or \
        abs(hz["azimut_haie_deg"] - (hz["direction_dominante_deg"] + 90) % 180) <= 1
    assert sum(r["frequence_pct"] for r in hz["rose"]) == pytest.approx(100, abs=1)
    assert sum(r["part_vents_forts_pct"] for r in hz["rose"]) == pytest.approx(100, abs=1)
    feats = hz["geojson"]["features"]
    assert [f["properties"]["type"] for f in feats] == ["haie", "abri_fort", "abri", "au_vent"]
    for f in feats[1:]:
        anneau = f["geometry"]["coordinates"][0]
        assert anneau[0] == anneau[-1] and len(anneau) == 5
    assert hz["zone_protegee_m"] == {"abri_fort": 24.0, "abri": 45.0, "au_vent": 9.0}
    assert "energie" in ao.plan_haies(wf, 3.0, 200.0, 1000.0, lat=LAT, lon=LON)["critere"]
