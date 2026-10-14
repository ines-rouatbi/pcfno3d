"""Le mode lien : l'application locale sert les donnees d'une instance distante.

Deux services reels sont lances : l'un joue l'instance Colab (il charge le modele du
mini-jeu de donnees), l'autre est en mode lien et lui transmet tout. On verifie que les
donnees traversent, que la cle de l'amont ne fuit jamais vers le client, et que la panne
de l'amont donne un message exploitable plutot qu'une erreur opaque.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")
pytest.importorskip("httpx")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _get(url: str, key: str | None = None, timeout: float = 90.0):
    req = urllib.request.Request(url, headers={"X-API-Key": key} if key else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:        # noqa: S310
        return json.loads(r.read().decode("utf-8"))


def _launch(port: int, env: dict, log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        return subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "service.app:app",
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(REPO_ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)


def _base_env(cfg) -> dict:
    return {**os.environ,
            "PCFNO3D_CONFIG": "test",
            "PCFNO3D_DATA_DIR": str(cfg.paths.data_dir),
            "PCFNO3D_CACHE_DIR": str(cfg.paths.cache_dir),
            "PCFNO3D_DEVICE": "cpu",
            "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")])}


@pytest.fixture(scope="module")
def paire(cfg, index, norm, tmp_path_factory, free_ports):
    """(base du lien, cle locale, cle amont, processus) — deux services qui tournent."""
    port_amont, port_lien = free_ports(2)
    from pcfno3d.train import Trainer
    from service.auth import ApiKeyStore, default_store_path
    from service.colab import wait_for_health
    from service.link import Upstream

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()

    # --- amont : le run existant (modele + statistiques), avec sa propre cle
    amont_store = ApiKeyStore(default_store_path(cfg.paths.run_dir))
    cle_amont, _ = amont_store.create("test-amont", scope="read")
    env_amont = {**_base_env(cfg), "PCFNO3D_RUN_DIR": str(cfg.paths.run_dir),
                 "PCFNO3D_PORT": str(port_amont)}
    log_amont = cfg.paths.run_dir / "amont.log"
    p_amont = _launch(port_amont, env_amont, log_amont)

    # --- lien : un run_dir separe, sans modele, qui ne contient que le lien et sa cle
    run_lien = tmp_path_factory.mktemp("run_lien")
    lien_store = ApiKeyStore(default_store_path(run_lien))
    cle_locale, _ = lien_store.create("test-navigateur", scope="read")
    Upstream(url=f"http://127.0.0.1:{port_amont}", key=cle_amont,
             label="faux-colab").save(run_lien)
    env_lien = {**_base_env(cfg), "PCFNO3D_RUN_DIR": str(run_lien),
                "PCFNO3D_PORT": str(port_lien)}
    log_lien = run_lien / "lien.log"
    p_lien = _launch(port_lien, env_lien, log_lien)

    try:
        wait_for_health(f"http://127.0.0.1:{port_amont}", timeout=240, log_path=log_amont)
        wait_for_health(f"http://127.0.0.1:{port_lien}", timeout=240, log_path=log_lien)
        yield (f"http://127.0.0.1:{port_lien}", cle_locale, cle_amont,
               f"http://127.0.0.1:{port_amont}", run_lien)
    finally:
        for p in (p_lien, p_amont):
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()


# ---------------------------------------------------------------------------- stockage
def test_enregistrement_et_oubli_du_lien(tmp_path):
    from service.link import Upstream

    assert Upstream.load(tmp_path) is None
    Upstream(url="https://abc.trycloudflare.com/", key="pcf_amont").save(tmp_path)

    relu = Upstream.load(tmp_path)
    assert relu.url == "https://abc.trycloudflare.com"      # la barre finale est retiree
    assert relu.key == "pcf_amont"
    assert relu.public() == {"url": "https://abc.trycloudflare.com", "label": "colab",
                             "cle_amont": True}
    assert "pcf_amont" not in json.dumps(relu.public())      # la cle amont ne sort pas

    assert Upstream.clear(tmp_path) is True
    assert Upstream.load(tmp_path) is None
    assert Upstream.clear(tmp_path) is False


def test_url_invalide_refusee():
    from service.link import Upstream

    with pytest.raises(ValueError, match="URL amont invalide"):
        Upstream(url="xxx.trycloudflare.com")


def test_environnement_prime_sur_le_fichier(tmp_path, monkeypatch):
    from service.link import Upstream

    Upstream(url="https://du-fichier.example", key="k1").save(tmp_path)
    assert Upstream.resolve(tmp_path).url == "https://du-fichier.example"

    monkeypatch.setenv("PCFNO3D_UPSTREAM", "https://de-l-env.example")
    monkeypatch.setenv("PCFNO3D_UPSTREAM_KEY", "k2")
    resolu = Upstream.resolve(tmp_path)
    assert resolu.url == "https://de-l-env.example" and resolu.key == "k2"


def test_entetes_transmis_sans_le_saut_local():
    from service.link import forward_headers

    entrants = {"host": "127.0.0.1:8000", "x-api-key": "cle-locale",
                "authorization": "Bearer cle-locale", "content-length": "42",
                "accept": "application/json", "content-type": "application/json"}
    sortants = forward_headers(entrants, "cle-amont")
    assert sortants["X-API-Key"] == "cle-amont"          # la cle amont remplace la locale
    assert "cle-locale" not in json.dumps(sortants)
    for interdit in ("host", "authorization", "content-length"):
        assert interdit not in {k.lower() for k in sortants}
    assert sortants["accept"] == "application/json"      # le reste passe


# ---------------------------------------------------------------------------- bout en bout
def test_health_decrit_le_lien(paire):
    base, cle_locale, cle_amont, _, _ = paire
    health = _get(f"{base}/api/health")
    assert health["mode"] == "lien"
    assert health["status"] == "ok"
    assert health["amont"]["label"] == "faux-colab"
    assert health["amont"]["cle_amont"] is True
    assert health["modele"].endswith(".pt")              # vient de l'amont
    # la cle de l'amont ne doit apparaitre nulle part dans la reponse
    assert cle_amont not in json.dumps(health)


def test_les_donnees_traversent_le_lien(paire):
    base, cle_locale, _, _, _ = paire

    relaye = _get(f"{base}/api/meta", key=cle_locale)
    assert relaye["grille"]["nz"] == len(relaye["z_levels_asl_m"])

    point = _get(f"{base}/api/point?x=400&y=400&z=120", key=cle_locale)
    assert point["wind_speed_m_s"] > 0
    assert 0.0 <= point["wind_direction_deg"] <= 360.0

    profil = _get(f"{base}/api/profile?x=400&y=400", key=cle_locale)
    assert len(profil["niveaux"]) == relaye["grille"]["nz"]

    instants = _get(f"{base}/api/instants?split=test", key=cle_locale)
    assert instants["total"] > 0


def test_la_cle_locale_est_exigee_et_suffit(paire):
    base, cle_locale, cle_amont, _, _ = paire

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{base}/api/meta")                         # aucune cle
    assert exc.value.code == 401

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{base}/api/meta", key="pcf_inconnue")
    assert exc.value.code == 401

    # la cle de l'amont n'est PAS valable cote local : les trousseaux sont separes
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{base}/api/meta", key=cle_amont)
    assert exc.value.code == 401

    assert _get(f"{base}/api/meta", key=cle_locale)["grille"]["nz"] > 0


def test_le_mcp_traverse_aussi(paire):
    base, cle_locale, _, _, _ = paire
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list",
                          "params": {}}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/mcp", data=payload,
        headers={"X-API-Key": cle_locale, "Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"})
    with urllib.request.urlopen(req, timeout=120) as r:               # noqa: S310
        tools = json.loads(r.read().decode("utf-8"))["result"]["tools"]
    assert {t["name"] for t in tools} >= {"grid_meta", "wind_at_point", "wind_profile"}


def test_application_web_servie_en_mode_lien(paire):
    base = paire[0]
    with urllib.request.urlopen(f"{base}/app/", timeout=30) as r:     # noqa: S310
        page = r.read().decode("utf-8")
    assert "PC-FNO3D" in page


def test_amont_injoignable_donne_un_message_actionnable(cfg, tmp_path_factory, index, norm,
                                                       free_ports):
    """Une URL de tunnel Colab morte doit produire un 502 qui dit quoi faire."""
    from service.auth import ApiKeyStore, default_store_path
    from service.colab import wait_for_health
    from service.link import Upstream

    run_lien = tmp_path_factory.mktemp("run_lien_mort")
    store = ApiKeyStore(default_store_path(run_lien))
    cle, _ = store.create("navigateur", scope="read")
    # port libre et laisse ferme : simule un tunnel dont la session Colab est terminee
    port_mort, port_lien = free_ports(2)
    Upstream(url=f"http://127.0.0.1:{port_mort}", key="pcf_amont",
             label="colab-mort").save(run_lien)

    env = {**_base_env(cfg), "PCFNO3D_RUN_DIR": str(run_lien), "PCFNO3D_PORT": str(port_lien)}
    log = run_lien / "lien.log"
    p = _launch(port_lien, env, log)
    try:
        health = wait_for_health(f"http://127.0.0.1:{port_lien}", timeout=180, log_path=log)
        assert health["mode"] == "lien"
        assert health["status"] == "degraded"             # l'amont est injoignable
        assert health["sante_amont"]["joignable"] is False

        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(f"http://127.0.0.1:{port_lien}/api/meta", key=cle)
        assert exc.value.code == 502
        detail = json.loads(exc.value.read().decode("utf-8"))["detail"]
        assert "injoignable" in detail
        assert "link set" in detail                       # dit comment reparer
    finally:
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
