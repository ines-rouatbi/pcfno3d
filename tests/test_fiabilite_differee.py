"""Fiabilite differee (decision du 16/10/2026, docs/FIABILITE.md « Fiabilite differee »).

La valeur est servie seule (DSO4 = son temps) ; la fiabilite (ensemble + calibration + statut
+ couverture + classe) est un produit a part, calcule en synchrone (scripts, publication,
evaluation) ou en arriere-plan (service). Verifie : equivalence numerique EXACTE avec le
calcul synchrone d'avant, etats et temps, echec, verrou prioritaire, et le service
(/api/reconstruct, /api/profile en_cours puis prete, /api/reliability, /api/health pendant le
calcul, MCP), fiabilite desactivee inchangee.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from pcfno3d.config import ReliabilityCfg
from pcfno3d.reliability import assess

REPO = Path(__file__).resolve().parents[1]
CHAMPS = ("sigma_u", "sigma_v", "sigma_w", "sigma_speed", "status", "coverage", "reliability",
          "reliability_class")
CLES_NOEUD = {"sigma_speed", "sigma_u", "sigma_v", "sigma_w", "speed_interval", "status",
              "coverage", "reliability", "reliability_class"}
# cles de /api/reconstruct AVANT la fiabilite differee (fiabilite desactivee : inchangees)
CLES_RECONSTRUCT = {"stem", "timestamp_utc", "capteurs_utilises", "inference_s",
                    "dso4_conforme", "grille", "noeuds_3d", "vitesse_max_m_s",
                    "vitesse_moyenne_m_s"}


@pytest.fixture(scope="module")
def trained(cfg, index, norm):
    from pcfno3d.train import Trainer
    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg


@pytest.fixture(scope="module")
def instant(trained):
    from pcfno3d.cache import load_index
    idx = load_index(trained)
    return idx[idx["split"] == "test"].sort_values("timestamp")["stem"].tolist()[:2]


@pytest.fixture
def wf_fiable(trained):
    """Modele avec la fiabilite active (sigma bruts, 4 membres, lots de 3)."""
    from pcfno3d.inference import WindFieldModel
    trained.inference.reliability = "non_calibre"
    trained.reliability.members = 4
    trained.reliability.batch = 3
    try:
        yield WindFieldModel(trained, device="cpu", verbose=False)
    finally:
        trained.inference.reliability = None
        trained.reliability.members = ReliabilityCfg().members
        trained.reliability.batch = ReliabilityCfg().batch


def _bloquer(wf):
    """Remplace wf._members par une version qui attend un feu : la fiabilite reste en_cours
    tant que le feu n'est pas donne. Renvoie (feu, appels)."""
    feu, appels = threading.Event(), []
    vrai = wf._members

    def bloque(*a, **k):
        appels.append(k.get("secondaire"))
        assert feu.wait(60), "feu jamais donne"
        return vrai(*a, **k)

    wf._members = bloque
    return feu, appels


def _meme_fiabilite(a, b) -> None:
    for name in CHAMPS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert a.uncertainty == b.uncertainty and a.info == b.info


# ------------------------------------------------------------------ coeur
def test_differee_identique_au_calcul_synchrone(wf_fiable, instant):
    """Equivalence EXACTE (bit a bit) : fiabilite differee == calcul synchrone d'avant
    (reliability.assess sur la passe brute, ensemble par wf._members), quel que soit le
    chemin : predict synchrone, predict(defer=True) + assess_reliability, thread
    d'arriere-plan, ou volume sans fiabilite + assess_reliability(volume, S, hour, doy)."""
    wf = wf_fiable
    S, hour, doy, stamp = wf.surface_from_cache(instant[0])
    brut = wf.predict(S, hour, doy, timestamp=stamp, mask_nan=False, reliability=False)
    sensor = S[7] > 0.5
    ref = assess((brut.u, brut.v, brut.w),
                 lambda masks: wf._members(S, masks, hour, doy), sensor, wf.geom,
                 wf.cfg.reliability, wf.reliability, False)

    sync = wf.predict(S, hour, doy, timestamp=stamp)
    assert sync.reliability_state == "prete"
    _meme_fiabilite(sync.reliability, ref)

    dif = wf.predict(S, hour, doy, timestamp=stamp, defer=True)
    assert dif.reliability_state == "en_cours" and dif.reliability is None
    assert dif.reliability_s == 0.0 and dif.inference_s > 0
    for x, y in ((dif.u, sync.u), (dif.v, sync.v), (dif.w, sync.w)):
        np.testing.assert_array_equal(x, y)          # la valeur ne depend de rien
    assert wf.assess_reliability(dif) is dif and dif.reliability_state == "prete"
    _meme_fiabilite(dif.reliability, ref)
    assert dif.reliability_s > 0

    fond = wf.predict(S, hour, doy, timestamp=stamp, defer=True)
    t = wf.assess_reliability_async(fond)
    assert t is not None and wf.assess_reliability_async(fond) is None   # un seul calcul
    t.join(60)
    assert fond.reliability_state == "prete"
    _meme_fiabilite(fond.reliability, ref)
    assert wf.assess_reliability_async(fond) is None                     # deja prete

    # volume calcule SANS fiabilite (NaN sous le relief) : meme resultat, entrees explicites
    sans = wf.predict(S, hour, doy, timestamp=stamp, reliability=False)
    assert sans.reliability_state == "desactivee"
    with pytest.raises(ValueError, match="S, hour et doy"):
        wf.assess_reliability(sans)
    wf.assess_reliability(sans, S, hour, doy)
    _meme_fiabilite(sans.reliability, ref)


def test_etat_temps_et_echec(wf_fiable, instant, tmp_path):
    from pcfno3d.inference import ReliabilityError
    wf = wf_fiable
    S, hour, doy, stamp = wf.surface_from_cache(instant[0])
    vol = wf.predict(S, hour, doy, timestamp=stamp, defer=True)
    st = vol.reliability_status()
    assert st == {"timestamp_utc": stamp, "fiabilite": "en_cours",
                  "inference_s": round(vol.inference_s, 4), "fiabilite_s": None, "erreur": None}
    # un lecteur de la fiabilite (npz) ne voit jamais un etat partiel : il la calcule ici
    files = set(np.load(vol.save_npz(tmp_path / "v.npz")).files)
    assert vol.reliability_state == "prete" and {"sigma_speed", "status"} <= files
    st = vol.reliability_status()
    assert st["fiabilite"] == "prete" and st["fiabilite_s"] > 0

    # echec : etat echec, cause gardee, wait leve ReliabilityError ; synchrone : l'erreur
    # d'origine remonte telle quelle (ex. memoire GPU dans le script Colab)
    def casse(*a, **k):
        raise MemoryError("lot trop gros")

    wf._members = casse
    try:
        bad = wf.predict(S, hour, doy, timestamp=stamp, defer=True)
        wf.assess_reliability_async(bad).join(60)
        assert bad.reliability_state == "echec" and bad.reliability is None
        assert "lot trop gros" in bad.reliability_status()["erreur"]
        with pytest.raises(ReliabilityError, match="lot trop gros"):
            bad.wait_reliability()
        assert bad.wait_reliability(raise_on_error=False) == "echec"
        with pytest.raises(MemoryError):
            wf.predict(S, hour, doy, timestamp=stamp)
    finally:
        del wf._members


def test_benchmark_dso4_valeur_seule(wf_fiable, instant):
    """DSO4 = temps de la valeur ; delai de la fiabilite a cote, sans verdict."""
    wf = wf_fiable
    S, hour, doy, _ = wf.surface_from_cache(instant[0])
    b = wf.benchmark(S, hour, doy, n=2, warmup=0)
    assert b["mean_s"] > 0 and isinstance(b["conforme"], bool)
    assert set(b["fiabilite"]) == {"n", "mean_s", "p95_s", "max_s", "membres", "lot"}
    assert b["fiabilite"]["membres"] == 4 and b["fiabilite"]["lot"] == 3
    assert "conforme" not in b["fiabilite"]
    sans = wf.benchmark(S, hour, doy, n=2, warmup=0, reliability=False)
    assert "fiabilite" not in sans


def test_verrou_la_passe_servie_passe_avant_les_lots(wf_fiable, instant):
    """ModelLock : un lot de l'ensemble (secondaire) attend qu'aucune passe servie
    n'attende ; la passe servie n'attend que la fin du lot en cours."""
    from pcfno3d.inference import ModelLock
    lock, ordre = ModelLock(), []
    lock.__enter__()                          # une passe en cours
    lots = threading.Thread(target=lambda: (lock.secondaire().__enter__(),
                                            ordre.append("lot"), lock.__exit__()))
    lots.start()
    time.sleep(0.05)
    servie = threading.Thread(target=lambda: (lock.__enter__(), ordre.append("servie"),
                                              lock.__exit__()))
    servie.start()
    time.sleep(0.05)
    lock.__exit__()                           # fin de la passe en cours
    lots.join(5)
    servie.join(5)
    assert ordre == ["servie", "lot"] and not lock.locked()

    # sur le modele : une passe servie pendant que la fiabilite d'un autre volume attend
    wf = wf_fiable
    S, hour, doy, stamp = wf.surface_from_cache(instant[0])
    feu, appels = _bloquer(wf)
    try:
        vol = wf.predict(S, hour, doy, timestamp=stamp, defer=True)
        wf.assess_reliability_async(vol)
        t0 = time.perf_counter()
        autre = wf.predict(*wf.surface_from_cache(instant[1])[:3], reliability=False)
        assert autre.reliability_state == "desactivee" and time.perf_counter() - t0 < 30
        assert vol.reliability_state == "en_cours"
        assert appels == [True]               # l'ensemble differe passe en secondaire
    finally:
        feu.set()
        vol.wait_reliability(60)
        del wf._members
    assert vol.reliability_state == "prete"


# ------------------------------------------------------------------ service
@pytest.fixture
def service(trained, monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp")

    def lancer(fiabilite: bool):
        conf = tmp_path / f"svc_fiab_{int(fiabilite)}.yaml"
        txt = (REPO / "configs" / "test.yaml").read_text(encoding="utf-8")
        if fiabilite:
            txt += "\ninference:\n  reliability: non_calibre\n" \
                   "\nreliability:\n  members: 3\n  batch: 2\n"
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
    cle, _ = ApiKeyStore(path).create("tests-fiabilite", scope="read")
    try:
        yield lancer, cle
    finally:
        path.unlink(missing_ok=True)
        if backup is not None:
            path.write_bytes(backup)
        from service import mcp_server
        mcp_server.set_alertes(None)


def test_service_profile_en_cours_puis_prete(service, instant):
    """/api/reconstruct repond sans attendre (fiabilite en_cours) ; /api/health et une autre
    reconstruction repondent pendant le calcul ; /api/profile donne les champs a null avec
    fiabilite en_cours, puis les valeurs quand elle est prete ; /api/reliability suit l'etat.
    MCP : meme etat explicite."""
    from fastapi.testclient import TestClient

    from service import app as svc
    from service import mcp_server
    lancer, cle = service
    t0, t1 = instant
    with TestClient(lancer(True), headers={"X-API-Key": cle}) as c:
        wf = svc.STATE["model"]
        assert wf.reliability is not None
        feu, appels = _bloquer(wf)
        try:
            rec = c.post("/api/reconstruct", json={"stem": t0})
            assert rec.status_code == 200, rec.text
            rec = rec.json()
            assert rec["fiabilite"] == "en_cours" and rec["fiabilite_s"] is None
            assert rec["fiabilite_route"] == f"/api/reliability/{t0}"
            assert rec["inference_s"] > 0 and rec["dso4_conforme"] == (rec["inference_s"] < 1.0)
            assert CLES_RECONSTRUCT <= set(rec)
            # pendant le calcul : la sante, et la VALEUR d'un autre instant
            h = c.get("/api/health")
            assert h.status_code == 200 and h.json()["status"] == "ok"
            autre = c.post("/api/reconstruct", json={"stem": t1}).json()
            assert autre["fiabilite"] == "en_cours"
            r = c.get(f"/api/reliability/{t0}").json()
            assert r["fiabilite"] == "en_cours" and r["fiabilite_s"] is None
            assert r["stem"] == t0 and r["inference_s"] == rec["inference_s"]
            prof = c.get("/api/profile", params={"x": 0, "y": 0, "stem": t0})
            assert prof.status_code == 200
            body = json.loads(prof.text)
            assert body["fiabilite"] == "en_cours" and body["uncertainty"] is None
            for niv in body["niveaux"]:
                assert CLES_NOEUD <= set(niv)
                assert all(niv[k] is None for k in CLES_NOEUD - {"speed_interval"})
                assert set(niv["speed_interval"].values()) == {None}
            assert any(n["wind_speed_m_s"] is not None for n in body["niveaux"])
            # MCP : l'etat explicite, champs de fiabilite a null
            mcp_server._VOLUMES.clear()
            res = mcp_server.instant_summary(t0)
            assert res["fiabilite"] == "en_cours" and res["fiabilite_s"] is None
            prof_mcp = mcp_server.wind_profile(0.0, 0.0, t0)
            assert prof_mcp["fiabilite"] == "en_cours" and prof_mcp["uncertainty"] is None
            assert c.get("/api/reliability/inconnu").status_code == 404
        finally:
            feu.set()
        vol = svc.STATE["volumes"][f"{t0}:1.000"]
        assert vol.wait_reliability(60) == "prete"
        assert all(appels)                    # tous les ensembles en secondaire
        r = c.get(f"/api/reliability/{t0}").json()
        assert r["fiabilite"] == "prete" and r["fiabilite_s"] > 0 and r["erreur"] is None
        body = json.loads(c.get("/api/profile", params={"x": 0, "y": 0, "stem": t0}).text)
        assert body["fiabilite"] == "prete" and body["uncertainty"]["members"] == 3
        g = wf.geom
        for k, niv in enumerate(body["niveaux"]):
            assert {x: niv[x] for x in CLES_NOEUD} == vol.node_reliability(k, 0, 0)
            assert (niv["status"] is None) == niv["sous_le_relief"] == (not g.valid[k, 0, 0])
        rec2 = c.post("/api/reconstruct", json={"stem": t0}).json()
        assert rec2["fiabilite"] == "prete" and rec2["fiabilite_s"] == r["fiabilite_s"]
        # MCP : prete
        mvol = mcp_server._VOLUMES[t0]
        assert mvol.wait_reliability(60) == "prete"
        assert mcp_server.instant_summary(t0)["fiabilite"] == "prete"
        prof_mcp = mcp_server.wind_profile(0.0, 0.0, t0)
        assert prof_mcp["fiabilite"] == "prete" and prof_mcp["uncertainty"]["members"] == 3
        # le volume du MCP et celui de l'API ont la meme fiabilite (meme calcul)
        _meme_fiabilite(mvol.reliability, vol.reliability)
        svc.STATE["volumes"][f"{t1}:1.000"].wait_reliability(60)


def test_service_desactivee_inchange(service, instant):
    """Fiabilite desactivee : reponses inchangees (aucune cle ajoutee), valeur bit a bit
    celle du modele, etat `desactivee` sur la route legere ; aucun thread lance."""
    from fastapi.testclient import TestClient

    from service import app as svc
    from service import mcp_server
    lancer, cle = service
    t0 = instant[0]
    with TestClient(lancer(False), headers={"X-API-Key": cle}) as c:
        wf = svc.STATE["model"]
        assert wf.reliability is None
        rec = c.post("/api/reconstruct", json={"stem": t0}).json()
        assert set(rec) == CLES_RECONSTRUCT
        body = c.get("/api/profile", params={"x": 0, "y": 0, "stem": t0}).json()
        assert "fiabilite" not in body and "uncertainty" not in body
        assert all(not (CLES_NOEUD & set(n)) for n in body["niveaux"])
        vol = svc.STATE["volumes"][f"{t0}:1.000"]
        assert vol.reliability_state == "desactivee" and vol.reliability is None
        assert wf.assess_reliability_async(vol) is None
        ref = wf.predict_stem(t0)
        for x, y in ((vol.u, ref.u), (vol.v, ref.v), (vol.w, ref.w)):
            np.testing.assert_array_equal(x, y)
        r = c.get(f"/api/reliability/{t0}").json()
        assert r["fiabilite"] == "desactivee" and r["fiabilite_s"] is None
        mcp_server._VOLUMES.clear()
        assert "fiabilite" not in mcp_server.instant_summary(t0)
        prof = mcp_server.wind_profile(0.0, 0.0, t0)
        assert "fiabilite" not in prof and "uncertainty" not in prof
