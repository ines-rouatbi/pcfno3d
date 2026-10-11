"""Le chemin « servir depuis Colab » : cle, verification du modele, service en
sous-processus, attente de /api/health.

Tout est teste SANS tunnel (`tunnel=False`) : cloudflared n'est pas installe sur un poste
de developpement et une URL publique n'a rien a faire dans une suite de tests. Seule
l'extraction de l'URL dans la sortie de cloudflared est verifiee, sur un echantillon
reel.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

CLOUDFLARED_SAMPLE = """\
2026-10-11T18:04:12Z INF Thank you for trying Cloudflare Tunnel.
2026-10-11T18:04:12Z INF Requests will be sent to your local webserver
2026-10-11T18:04:14Z INF +--------------------------------------------------------+
2026-10-11T18:04:14Z INF |  Your quick Tunnel has been created! Visit it at       |
2026-10-11T18:04:14Z INF |  https://brave-wind-field-tunisia.trycloudflare.com    |
2026-10-11T18:04:14Z INF +--------------------------------------------------------+
"""


# ---------------------------------------------------------------------------- tunnel
def test_extraction_de_l_url_publique():
    from service.tunnel import find_url

    assert find_url(CLOUDFLARED_SAMPLE) == "https://brave-wind-field-tunisia.trycloudflare.com"
    assert find_url("rien d'interessant ici") is None
    assert find_url("") is None
    assert find_url(None) is None


def test_binaire_cloudflared_inconnu_explique_pourquoi(monkeypatch, tmp_path):
    """Sur une plateforme sans binaire connu, le message doit etre actionnable."""
    import service.tunnel as t

    monkeypatch.setattr(t.shutil, "which", lambda _: None)
    monkeypatch.setattr(t.platform, "system", lambda: "Plan9")
    monkeypatch.setattr(t.platform, "machine", lambda: "sparc")
    with pytest.raises(RuntimeError, match="cloudflared"):
        t.ensure_cloudflared(tmp_path)


# ---------------------------------------------------------------------------- garde-fous
def test_refus_si_aucun_checkpoint(cfg, tmp_path):
    """Mieux vaut s'arreter net que demarrer un service qui n'a rien a servir."""
    from pcfno3d.config import Config
    from service.colab import check_model_files

    vide = Config.load("test")
    vide.apply_overrides({"paths.run_dir": tmp_path / "vide"})
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        check_model_files(vide)


def test_refus_si_statistiques_absentes(cfg, tmp_path, index, norm):
    """Le checkpoint seul ne suffit pas : sans norm_stats.json, rien n'est denormalisable."""
    import shutil

    from pcfno3d.config import Config
    from pcfno3d.train import Trainer
    from service.colab import check_model_files

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()

    partiel = tmp_path / "partiel"
    partiel.mkdir()
    shutil.copy2(cfg.paths.ckpt_best, partiel / "best.pt")      # poids seuls, sans stats
    autre = Config.load("test")
    autre.apply_overrides({"paths.run_dir": partiel})
    with pytest.raises(FileNotFoundError, match="norm_stats"):
        check_model_files(autre)


def test_cle_dupliquee_est_renommee_jamais_relue(cfg, tmp_path):
    """Un secret ne peut pas etre retrouve : un nom deja pris donne une nouvelle cle."""
    from pcfno3d.config import Config
    from service.colab import ensure_api_key

    local = Config.load("test")
    local.apply_overrides({"paths.run_dir": tmp_path / "cles"})
    (tmp_path / "cles").mkdir(parents=True, exist_ok=True)

    first, renamed = ensure_api_key(local, name="colab")
    assert first.startswith("pcf_") and renamed is False
    second, renamed = ensure_api_key(local, name="colab")
    assert second != first and renamed is True

    from service.auth import ApiKeyStore, default_store_path
    store = ApiKeyStore(default_store_path(local.paths.run_dir))
    assert store.verify(first) is not None and store.verify(second) is not None
    assert {r.name for r in store.records} == {"colab", "colab-2"}


def test_attente_de_health_signale_le_journal(tmp_path):
    from service.colab import wait_for_health

    log = tmp_path / "service.log"
    log.write_text("ImportError: torch introuvable", encoding="utf-8")
    with pytest.raises(TimeoutError, match="torch introuvable"):
        wait_for_health("http://127.0.0.1:59999", timeout=1.0, log_path=log)


# ---------------------------------------------------------------------------- bout en bout
def test_service_demarre_en_sous_processus_et_repond(cfg, index, norm, monkeypatch):
    """serve_public sans tunnel : un vrai uvicorn, interroge par le reseau."""
    import urllib.request

    from pcfno3d.train import Trainer
    from service.colab import serve_public, stop, summary

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()

    monkeypatch.setenv("PCFNO3D_DATA_DIR", str(cfg.paths.data_dir))
    monkeypatch.setenv("PCFNO3D_CACHE_DIR", str(cfg.paths.cache_dir))
    monkeypatch.setenv("PCFNO3D_RUN_DIR", str(cfg.paths.run_dir))

    info = serve_public(config="test", port=8911, key_name="pytest-colab", device="cpu",
                        tunnel=False, wait=240.0,
                        log_path=cfg.paths.run_dir / "colab_service.log")
    try:
        assert info["health"]["status"] == "ok"
        assert info["health"]["auth"] == "cle_api"      # la cle creee protege le service
        assert info["api_key"].startswith("pcf_")
        assert info["mcp_url"].endswith("/mcp")

        # le resume est ce qu'on transmet a l'autre application : il doit porter l'essentiel
        text = summary(info)
        assert info["api_key"] in text and info["mcp_url"] in text

        base = info["public_url"]

        req = urllib.request.Request(f"{base}/api/meta")
        with pytest.raises(urllib.error.HTTPError) as exc:      # sans cle : refuse
            urllib.request.urlopen(req, timeout=30)
        assert exc.value.code == 401

        req = urllib.request.Request(f"{base}/api/meta",
                                     headers={"X-API-Key": info["api_key"]})
        with urllib.request.urlopen(req, timeout=60) as r:
            meta = json.loads(r.read().decode("utf-8"))
        assert meta["grille"]["nz"] == len(meta["z_levels_asl_m"])

        # le MCP passe par le meme port et la meme cle
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list",
                              "params": {}}).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/mcp", data=payload,
            headers={"X-API-Key": info["api_key"], "Content-Type": "application/json",
                     "Accept": "application/json, text/event-stream"})
        with urllib.request.urlopen(req, timeout=60) as r:
            tools = json.loads(r.read().decode("utf-8"))["result"]["tools"]
        assert len(tools) == 8
    finally:
        stop(info)
    assert info["process"].poll() is not None
