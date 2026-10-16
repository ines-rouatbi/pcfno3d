"""Integration des lots M4 (alertes) et M5 (eolien) : cablage des alertes du parc dans le
moteur M4, journal, /api/wind-farm, ensemble M2, opt-in combine.

Le parc est celui de la demonstration (courbes de configs/eolien/parc_demo.yaml), place sur
le mini-domaine : les positions de la demo (25 x 50 km) n'y tiennent pas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from pcfno3d.alerts import (KINDS_EXTERNES, MoteurAlertes, alert_id, compter_ouvertes,
                            items_externes, verifier_alerte)
from pcfno3d.config import ReliabilityCfg

REPO = Path(__file__).resolve().parents[1]
ATOM = REPO / "tests" / "data" / "atom_rev2"
DEMO = REPO / "configs" / "eolien" / "parc_demo.yaml"


@pytest.fixture(scope="module")
def schema_alertes():
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    docs = [json.loads((ATOM / f"{n}.schema.json").read_text(encoding="utf-8"))
            for n in ("alerts", "common")]
    reg = referencing.Registry().with_resources(
        [(d["$id"], referencing.Resource.from_contents(d)) for d in docs])
    return jsonschema.Draft202012Validator(docs[0], registry=reg)


def _valide(validator, doc: dict) -> None:
    erreurs = sorted(validator.iter_errors(doc), key=str)
    assert not erreurs, [e.message for e in erreurs[:5]]
    for a in doc["items"]:
        verifier_alerte(a)


@pytest.fixture(scope="module")
def trained(cfg, index, norm):
    from pcfno3d.train import Trainer
    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg


@pytest.fixture
def parc(geom, tmp_path) -> Path:
    """Courbe generique-2MW de la demo, 3 eoliennes ou le rotor [40, 80] m au-dessus du sol
    est encadre par des noeuds valides du mini-domaine."""
    demo = yaml.safe_load(DEMO.read_text(encoding="utf-8"))
    h = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    hv = np.where(geom.valid, h, np.nan)
    ok = (np.nanmin(hv, axis=0) <= 39.0) & (np.nanmax(hv, axis=0) >= 81.0)
    jj, ii = np.nonzero(ok)
    pick = np.linspace(0, len(jj) - 1, 3).astype(int)
    data = {"exemple": True,
            "power_curves": {"generique-2MW": demo["power_curves"]["generique-2MW"]},
            "turbines": [{"id": f"T-0{n + 1}", "name": f"T-0{n + 1} (exemple)",
                          "x_m": float(ii[k] * geom.dx), "y_m": float(jj[k] * geom.dy),
                          "hub_agl_m": 60.0, "rotor_d_m": 40.0, "model": "generique-2MW"}
                         for n, k in enumerate(pick)]}
    path = tmp_path / "parc_mini.yaml"
    path.write_bytes(yaml.safe_dump(data, allow_unicode=True).encode("utf-8"))
    return path


def _jour(cfg, split: str = "test") -> list[str]:
    from pcfno3d.cache import load_index
    idx = load_index(cfg)
    return idx[idx["split"] == split].sort_values("timestamp")["stem"].tolist()


# ------------------------------------------------------------------ ensemble M2
def test_m4_m5_attendent_la_fiabilite_differee(trained, parc, tmp_path):
    """Fiabilite differee : le parc (quantiles sur les membres captures) et les alertes
    (zone_mal_couverte : couverture et sigma) d'un volume en_cours attendent la fiabilite
    (ou la calculent) et donnent EXACTEMENT le resultat du calcul synchrone ; jamais un
    etat partiel (membres absents, couverture sans sigma)."""
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.windfarm import compute_farm
    trained.wind_farm.turbines = parc
    trained.inference.reliability = "non_calibre"
    trained.reliability.members = 3
    try:
        wf = WindFieldModel(trained, device="cpu", verbose=False)
        stem = _jour(trained)[0]
        S, hour, doy, stamp = wf.surface_from_cache(stem)
        sync = wf.predict(S, hour, doy, timestamp=stamp)
        wcfg = trained.wind_farm
        ref = compute_farm(sync, wf.farm, wcfg, calibration=wf.reliability).to_json(wcfg)
        assert ref["membres"] == 3 and ref["wind_farm"]["farm"]["power_kw"]["p90"] is not None

        dif = wf.predict(S, hour, doy, timestamp=stamp, defer=True)
        assert dif.reliability_state == "en_cours" and dif.farm_context.members_uv is None
        out = compute_farm(dif, wf.farm, wcfg, calibration=wf.reliability).to_json(wcfg)
        assert dif.reliability_state == "prete" and out == ref
        np.testing.assert_array_equal(dif.farm_context.members_uv,
                                      sync.farm_context.members_uv)

        def doc(vol, nom):
            m = MoteurAlertes(trained, journal=tmp_path / nom)
            return m.evaluer_modele(wf, S, hour, doy, vol).document()

        dif2 = wf.predict(S, hour, doy, timestamp=stamp, defer=True)
        d_dif = doc(dif2, "b.jsonl")
        assert dif2.reliability_state == "prete"
        assert d_dif == doc(sync, "a.jsonl")
    finally:
        trained.wind_farm.turbines = None
        trained.inference.reliability = None
        trained.reliability.members = ReliabilityCfg().members


def test_passes_de_derive_non_capturees_par_le_parc(trained, parc, tmp_path):
    """Les cote^2 passes de derive M4 (wf._member) n'entrent PAS dans les membres de
    l'ensemble M2 captures pour le parc (Volume.farm_context.members_uv)."""
    from pcfno3d.inference import WindFieldModel
    trained.wind_farm.turbines = parc
    trained.inference.reliability = "non_calibre"
    trained.reliability.members = 3
    try:
        wf = WindFieldModel(trained, device="cpu", verbose=False)
        appels = []
        vrai = wf._capture
        wf._capture = lambda uvw, captured: (appels.append(1), vrai(uvw, captured))[1]
        stem = _jour(trained)[0]
        S, hour, doy, stamp = wf.surface_from_cache(stem)
        vol = wf.predict(S, hour, doy, timestamp=stamp)
        assert len(appels) == 3 and vol.farm_context.members_uv.shape[0] == 3
        avant = vol.farm_context.members_uv.copy()
        membres = []
        vrai_membre = wf._member
        wf._member = lambda *a: (membres.append(1), vrai_membre(*a))[1]
        res = MoteurAlertes(trained, journal=tmp_path / "j.jsonl").evaluer_modele(
            wf, S, hour, doy, vol)
        assert res.derive["evaluee"] and len(membres) == res.derive["passes"] == 9
        assert len(appels) == 3                          # aucune passe M4 capturee
        np.testing.assert_array_equal(vol.farm_context.members_uv, avant)
    finally:
        trained.wind_farm.turbines = None
        trained.inference.reliability = None
        trained.reliability.members = ReliabilityCfg().members


# ------------------------------------------------------------------ service
@pytest.fixture
def service(trained, monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp")

    def lancer(alertes: bool, parc: Path | None):
        conf = tmp_path / f"svc_{int(alertes)}_{int(parc is not None)}.yaml"
        txt = (REPO / "configs" / "test.yaml").read_text(encoding="utf-8")
        if alertes:
            txt += f"\nalerts:\n  enabled: true\n  journal: {(tmp_path / 'a.jsonl').as_posix()}\n"
        if parc is not None:
            txt += f"\nwind_farm:\n  turbines: {parc.as_posix()}\n"
        conf.write_bytes(txt.encode("utf-8"))
        monkeypatch.setenv("PCFNO3D_CONFIG", str(conf))
        for var, p in (("PCFNO3D_DATA_DIR", trained.paths.data_dir),
                       ("PCFNO3D_CACHE_DIR", trained.paths.cache_dir),
                       ("PCFNO3D_RUN_DIR", trained.paths.run_dir)):
            monkeypatch.setenv(var, str(p))
        monkeypatch.setenv("PCFNO3D_DEVICE", "cpu")
        for var in ("PCFNO3D_API_KEYS", "PCFNO3D_INSECURE_OPEN", "PCFNO3D_QUOTA_MINUTE",
                    "PCFNO3D_QUOTA_JOUR", "PCFNO3D_UPSTREAM"):
            monkeypatch.delenv(var, raising=False)
        from service.app import create_app
        return create_app()

    from service.auth import ApiKeyStore, default_store_path
    path = default_store_path(trained.paths.run_dir)
    backup = path.read_bytes() if path.exists() else None
    path.unlink(missing_ok=True)
    cle, _ = ApiKeyStore(path).create("tests-m45", scope="read")
    try:
        yield lancer, cle
    finally:
        path.unlink(missing_ok=True)
        if backup is not None:
            path.write_bytes(backup)
        from service import mcp_server
        mcp_server.set_alertes(None)


def test_alertes_du_parc_dans_le_journal_m4(service, trained, parc, tmp_path,
                                            schema_alertes):
    """Parc de demo + alertes actives : /api/alerts contient les alertes M5 (id M4 = sha8 de
    kind|cible), valide contre le schema rev. 2 ; elles sont ouvertes puis closes dans le
    journal ; /api/wind-farm reprend les items du journal ; kpi.open_alerts les compte."""
    from fastapi.testclient import TestClient

    from service import app as svc
    lancer, cle = service
    t0, t1 = _jour(trained)[:2]
    with TestClient(lancer(True, parc), headers={"X-API-Key": cle}) as c:
        wcfg = svc.STATE["model"].cfg.wind_farm
        wcfg.shear_info = -1.0                # tout cisaillement mesure leve une alerte
        r = c.get("/api/alerts", params={"stem": t0})
        assert r.status_code == 200, r.text
        doc = r.json()
        _valide(schema_alertes, doc)
        m5 = items_externes(doc["items"])
        assert m5 and {a["kind"] for a in m5} <= set(KINDS_EXTERNES)
        cis = [a for a in m5 if a["kind"] == "cisaillement"]
        assert cis
        for a in m5:
            assert a["id"] == alert_id(a["kind"], a["target"])
            assert a["turbine_id"] == a["target"] and a["status"] == "ouverte"
        raised = {a["id"]: a["raised_at"] for a in m5}
        moteur = svc.STATE["alertes"]
        assert moteur.dernier.ouvertes() == compter_ouvertes(doc["items"])
        assert moteur.dernier.ouvertes()["information"] >= len(
            [a for a in cis if a["level"] == "information"])
        # /api/wind-farm : items du journal (meme raised_at), pas les alertes brutes
        wfr = c.get("/api/wind-farm", params={"stem": t0}).json()
        assert wfr["alerts"]["items"] == m5
        # instant suivant sans cisaillement au-dessus du seuil : les alertes M5 se closent
        wcfg.shear_info = wcfg.shear_warn = wcfg.shear_important = 100.0
        doc1 = c.get("/api/alerts", params={"stem": t1}).json()
        _valide(schema_alertes, doc1)
        closes = {a["id"]: a for a in items_externes(doc1["items"])}
        for aid, t in raised.items():
            assert closes[aid]["status"] == "close" and closes[aid]["raised_at"] == t
        assert c.get("/api/wind-farm", params={"stem": t1}).json()["alerts"]["items"] \
            == items_externes(doc1["items"])
    lignes = [json.loads(x) for x in
              (tmp_path / "a.jsonl").read_bytes().decode("utf-8").splitlines()]
    for aid in raised:
        ev = [e["evenement"] for e in lignes if e["id"] == aid]
        assert ev[0] == "levee" and ev[-1] == "close"


def test_parc_sans_alertes_garde_les_alertes_brutes(service, trained, parc):
    """Alertes desactivees : /api/wind-farm garde le comportement M5 (raised_at = instant)."""
    from fastapi.testclient import TestClient

    from pcfno3d.alerts import iso_utc
    from service import app as svc
    lancer, cle = service
    t0 = _jour(trained)[0]
    with TestClient(lancer(False, parc), headers={"X-API-Key": cle}) as c:
        svc.STATE["model"].cfg.wind_farm.shear_info = -1.0
        assert c.get("/api/alerts").status_code == 409
        body = c.get("/api/wind-farm", params={"stem": t0}).json()
        stamp = iso_utc(svc.STATE["model"].surface_from_cache(t0)[3])
    items = body["alerts"]["items"]
    assert items and all(a["raised_at"] == stamp and a["status"] == "ouverte" for a in items)


def test_tout_desactive_comme_main(service, trained):
    """Ni alertes, ni parc, ni fiabilite : pas de cle ajoutee, routes des lots fermees."""
    from fastapi.testclient import TestClient
    lancer, cle = service
    with TestClient(lancer(False, None), headers={"X-API-Key": cle}) as c:
        from service import app as svc
        wf = svc.STATE["model"]
        assert wf.farm is None and wf.reliability is None and svc.STATE.get("alertes") is None
        assert "fiabilite" not in c.get("/api/meta").json()
        stem = _jour(trained)[0]
        rec = c.post("/api/reconstruct", json={"stem": stem}).json()
        assert "fiabilite_s" not in rec
        assert c.get("/api/alerts").status_code == 409
        assert c.get("/api/wind-farm").status_code == 404
        vol = svc._volume_for(stem)
        assert vol.farm_context is None and vol.reliability is None
