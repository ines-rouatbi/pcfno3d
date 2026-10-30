"""Lot M7 — critere API1 : toute requete sans cle valide est refusee ; une cle revoquee
cesse de fonctionner immediatement.

Ferme par defaut, mode ouvert seulement sur opt-in (jamais sur le chemin Colab),
revocation prise en compte sans redemarrage (y compris par la CLI d'un autre processus),
quotas par cle (429 + Retry-After), journal d'acces JSONL sans aucun secret, et bloc
`georef` de /api/meta.

Aucune vraie cle ici : toutes sont generees pendant le test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")
from fastapi.testclient import TestClient      # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL = ("127.0.0.1", 50000)             # TestClient se presente sinon comme "testclient"
MCP_HEADERS = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream",
               "MCP-Protocol-Version": "2025-06-18"}


def _rpc(client, method, key=None, params=None):
    headers = dict(MCP_HEADERS)
    if key:
        headers["X-API-Key"] = key
    return client.post("/mcp", headers=headers, json={
        "jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})


# ---------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def trained(cfg, index, norm):
    from pcfno3d.train import Trainer

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg


@pytest.fixture
def env(trained, monkeypatch, tmp_path):
    """Service sur le run du mini-jeu ; trousseau vide, journal dans tmp_path."""
    for var, val in (("PCFNO3D_CONFIG", "test"), ("PCFNO3D_DEVICE", "cpu"),
                     ("PCFNO3D_DATA_DIR", str(trained.paths.data_dir)),
                     ("PCFNO3D_CACHE_DIR", str(trained.paths.cache_dir)),
                     ("PCFNO3D_RUN_DIR", str(trained.paths.run_dir)),
                     ("PCFNO3D_ACCESS_LOG", str(tmp_path / "acces.jsonl"))):
        monkeypatch.setenv(var, val)
    for var in ("PCFNO3D_API_KEYS", "PCFNO3D_INSECURE_OPEN", "PCFNO3D_FORBID_OPEN",
                "PCFNO3D_ALLOWED_HOSTS", "PCFNO3D_QUOTA_MINUTE", "PCFNO3D_QUOTA_JOUR"):
        monkeypatch.delenv(var, raising=False)
    store_path = trained.paths.run_dir / "api_keys.json"
    sauvegarde = store_path.read_text(encoding="utf-8") if store_path.exists() else None
    store_path.unlink(missing_ok=True)
    yield trained
    store_path.unlink(missing_ok=True)
    if sauvegarde is not None:
        store_path.write_text(sauvegarde, encoding="utf-8")


def _store(cfg):
    from service.auth import ApiKeyStore, default_store_path

    return ApiKeyStore(default_store_path(cfg.paths.run_dir))


def _journal(tmp_path) -> list[dict]:
    path = tmp_path / "acces.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------- ferme par defaut
def test_sans_aucune_cle_tout_est_refuse_sauf_health(env):
    from service.app import create_app

    with TestClient(create_app()) as client:
        health = client.get("/api/health").json()
        assert health["auth"] == "aucune_cle"
        assert "aucune cle configuree" in health["message"]
        assert health["cles_actives"] == 0

        for chemin in ("/api/meta", "/api/instants", "/api/dso", "/api/keys",
                       "/api/field?z=500", "/api/point?x=1&y=1&z=500",
                       "/api/export/004_120000/manifest.json",
                       "/api/export/004_120000/u.bin"):
            r = client.get(chemin)
            assert r.status_code == 401, (chemin, r.status_code)
            assert "aucune cle configuree" in r.json()["detail"]
        assert client.post("/api/reconstruct", json={"export": True}).status_code == 401
        r = _rpc(client, "tools/list")
        assert r.status_code == 401
        assert r.json()["error"]["code"] == -32001
        # une cle quelconque n'ouvre rien : il n'y a rien contre quoi la verifier
        from service.auth import generate_key
        assert client.get("/api/meta", headers={"X-API-Key": generate_key()}).status_code == 401

        # l'interface et la documentation restent servies : aucune donnee dedans
        assert client.get("/app/").status_code == 200
        assert client.get("/api/openapi.json").status_code == 200


def test_mode_ouvert_seulement_sur_opt_in_explicite(env, monkeypatch):
    from service.app import create_app

    monkeypatch.setenv("PCFNO3D_INSECURE_OPEN", "1")
    with TestClient(create_app(), client=LOCAL) as client:
        health = client.get("/api/health").json()
        assert health["auth"] == "ouvert"
        assert "developpement local" in health["avertissement"].lower()
        assert client.get("/api/meta").status_code == 200
        assert _rpc(client, "tools/list").status_code == 200

        # des qu'une cle existe, elle est exigee, opt-in ou pas
        secret, _ = _store(env).create("dev", scope="read")
        assert client.get("/api/meta").status_code == 401
        assert client.get("/api/meta", headers={"X-API-Key": secret}).status_code == 200
        assert client.get("/api/health").json()["auth"] == "cle_api"


@pytest.mark.parametrize("ip, entetes, attendu", [
    (("127.0.0.1", 50000), {}, 200),
    (("::1", 50000), {}, 200),
    (("8.8.8.8", 1), {}, 401),
    (("8.8.8.8", 1), {"X-Forwarded-For": "8.8.8.8", "Host": "evil.example"}, 401),
    (("127.0.0.1", 50000), {"X-Forwarded-For": "8.8.8.8"}, 401),
    (("127.0.0.1", 50000), {"Forwarded": "for=8.8.8.8"}, 401),
    (("127.0.0.1", 50000), {"CF-Connecting-IP": "8.8.8.8"}, 401),
    (("127.0.0.1", 50000), {"X-Real-IP": "8.8.8.8"}, 401),
    (("testclient", 50000), {}, 401),
])
def test_mode_ouvert_ne_sert_que_la_boucle_locale_directe(env, monkeypatch, ip, entetes,
                                                          attendu):
    """Ouvert ne veut pas dire expose : uvicorn --host 0.0.0.0 ou un proxy local
    (qui pose X-Forwarded-For...) ne doivent rien obtenir sans cle."""
    from service.app import create_app

    monkeypatch.setenv("PCFNO3D_INSECURE_OPEN", "1")
    with TestClient(create_app(), client=ip) as client:
        r = client.get("/api/meta", headers=entetes)
        assert r.status_code == attendu, r.text
        rpc = client.post("/mcp", headers={**MCP_HEADERS, **entetes}, json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        assert rpc.status_code == attendu
        if attendu == 401:
            assert "boucle locale" in r.json()["detail"]


@pytest.mark.parametrize("var, valeur", [("PCFNO3D_FORBID_OPEN", "1"),
                                          ("PCFNO3D_ALLOWED_HOSTS", "atom-origin.zelqane.com")])
def test_opt_in_ignore_derriere_tunnel_ou_proxy(env, monkeypatch, var, valeur):
    from service.app import create_app

    monkeypatch.setenv("PCFNO3D_INSECURE_OPEN", "1")
    monkeypatch.setenv(var, valeur)
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["auth"] == "aucune_cle"
        assert client.get("/api/meta").status_code == 401


def test_chemin_colab_ne_transmet_jamais_le_mode_ouvert(cfg, monkeypatch):
    from service.auth import ENV_FORBID_OPEN, ENV_INSECURE_OPEN
    from service.colab import service_env

    monkeypatch.setenv(ENV_INSECURE_OPEN, "1")
    env = service_env(cfg, "test", 8999, device="cpu")
    assert ENV_INSECURE_OPEN not in env
    assert env[ENV_FORBID_OPEN] == "1"
    assert env["PCFNO3D_PORT"] == "8999" and env["PCFNO3D_DEVICE"] == "cpu"


def test_cli_serve_refuse_le_mode_ouvert_hors_boucle_locale(monkeypatch):
    pytest.importorskip("uvicorn")
    from pcfno3d.cli import main

    monkeypatch.delenv("PCFNO3D_INSECURE_OPEN", raising=False)
    assert main(["-c", "test", "serve", "--host", "0.0.0.0", "--insecure-open"]) == 2
    assert "PCFNO3D_INSECURE_OPEN" not in os.environ
    # meme refus quand c'est la variable d'environnement qui demande le mode ouvert
    monkeypatch.setenv("PCFNO3D_INSECURE_OPEN", "1")
    for hote in ("0.0.0.0", "192.168.1.10", "::"):
        assert main(["-c", "test", "serve", "--host", hote]) == 2


# ---------------------------------------------------------------------------- revocation
def test_revocation_immediate_http_et_mcp_sans_redemarrage(env):
    """Revoquee par un AUTRE processus (la CLI), la cle tombe a la requete suivante."""
    from service.app import create_app

    secret, _ = _store(env).create("a-revoquer", scope="read")
    autre, _ = _store(env).create("reste-active", scope="read")
    with TestClient(create_app()) as client:
        assert client.get("/api/meta", headers={"X-API-Key": secret}).status_code == 200
        assert _rpc(client, "tools/list", key=secret).status_code == 200

        # la vraie CLI, dans un processus separe, sur le meme trousseau
        out = subprocess.run(
            [sys.executable, "-m", "pcfno3d", "-c", "test", "--run-dir",
             str(env.paths.run_dir), "apikey", "revoke", "a-revoquer"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(
                [str(REPO_ROOT / "src"), str(REPO_ROOT)])})
        assert out.returncode == 0, out.stderr
        assert secret not in out.stdout + out.stderr

        assert client.get("/api/meta", headers={"X-API-Key": secret}).status_code == 401
        assert client.get(f"/api/meta?api_key={secret}").status_code == 401
        assert _rpc(client, "tools/list", key=secret).status_code == 401
        # les autres cles ne sont pas touchees
        assert client.get("/api/meta", headers={"X-API-Key": autre}).status_code == 200
        assert _rpc(client, "tools/list", key=autre).status_code == 200


def test_revoquer_la_derniere_cle_ferme_tout(env):
    from service.app import create_app

    secret, _ = _store(env).create("unique", scope="admin")
    with TestClient(create_app()) as client:
        assert client.get("/api/keys", headers={"X-API-Key": secret}).status_code == 200
        assert _store(env).revoke("unique")
        r = client.get("/api/keys", headers={"X-API-Key": secret})
        assert r.status_code == 401 and "aucune cle configuree" in r.json()["detail"]
        assert client.get("/api/health").json()["auth"] == "aucune_cle"


def test_verifier_une_cle_n_ecrit_jamais_le_trousseau(tmp_path):
    """Le serveur ne reecrit pas api_keys.json : il ne peut pas ecraser une revocation."""
    from service.auth import ApiKeyStore

    store = ApiKeyStore(tmp_path / "api_keys.json")
    secret, _ = store.create("k", scope="read")
    avant = (tmp_path / "api_keys.json").read_bytes()
    st = (tmp_path / "api_keys.json").stat()
    for _ in range(3):
        assert store.verify(secret) is not None
    assert (tmp_path / "api_keys.json").read_bytes() == avant
    assert (tmp_path / "api_keys.json").stat().st_mtime_ns == st.st_mtime_ns
    # la derniere utilisation est gardee a part, et relue par `apikey list`
    assert store.usage_path.exists()
    assert secret not in store.usage_path.read_text(encoding="utf-8")
    assert ApiKeyStore(tmp_path / "api_keys.json").list_public()[0]["last_used_at"]


def test_erreur_de_lecture_passagere_ne_bloque_pas_le_trousseau(tmp_path, monkeypatch):
    """Un verrou d'antivirus (PermissionError) ne doit pas fermer le trousseau jusqu'a sa
    prochaine modification : la lecture est retentee a la requete suivante."""
    from service.auth import ApiKeyStore

    store = ApiKeyStore(tmp_path / "api_keys.json")
    secret, _ = store.create("k", scope="read")
    original = Path.read_text
    echecs = {"n": 2}                                # load() puis la relecture suivante

    def verrouille(self, *a, **kw):
        if self.name == "api_keys.json" and echecs["n"] > 0:
            echecs["n"] -= 1
            raise PermissionError("fichier verrouille")
        return original(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", verrouille)
    store.load()                                     # lecture ratee : ferme, pas ouvert
    assert store.erreur and not store.open_mode
    assert store.verify(secret) is None             # toujours verrouille : refuse
    assert store.verify(secret).name == "k"         # relu, sans modification du fichier
    assert store.erreur is None


def test_trousseau_illisible_ferme_meme_avec_opt_in(tmp_path):
    from service.auth import ApiKeyStore, summarize

    (tmp_path / "api_keys.json").write_text("{pas du json", encoding="utf-8")
    store = ApiKeyStore(tmp_path / "api_keys.json", allow_open=True)
    assert store.erreur and not store.open_mode
    assert summarize(store)["auth"] == "aucune_cle"


# ---------------------------------------------------------------------------- quotas
class _Horloge:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_quotas_minute_et_jour_avec_horloge():
    from service.auth import ApiKeyRecord
    from service.quotas import Quotas

    h = _Horloge(10 * 86_400 + 3600.0)          # 01:00 UTC
    q = Quotas(par_minute=3, par_jour=5, horloge=h)
    cle = ApiKeyRecord(name="k", key_hash="a" * 64)
    assert [q.consommer(cle) for _ in range(3)] == [None] * 3
    dep = q.consommer(cle)
    assert dep.limite == "minute" and dep.quota == 3 and 1 <= dep.retry_after <= 60
    h.t += 61                                    # la fenetre glisse
    assert q.consommer(cle) is None and q.consommer(cle) is None
    dep = q.consommer(cle)                       # 5 requetes acceptees ce jour
    assert dep.limite == "jour"
    assert dep.retry_after == 23 * 3600 - 61     # jusqu'a minuit UTC
    h.t = 11 * 86_400 + 1                        # jour suivant
    assert q.consommer(cle) is None

    # surcharge par cle : 0 = illimite, et une autre cle a son propre compteur
    libre = ApiKeyRecord(name="libre", key_hash="b" * 64, quota_minute=0, quota_jour=0)
    assert all(q.consommer(libre) is None for _ in range(50))


def test_quota_depasse_donne_429_et_retry_after(env, monkeypatch, tmp_path):
    from service.app import create_app

    monkeypatch.setenv("PCFNO3D_QUOTA_MINUTE", "1000")
    serree, _ = _store(env).create("serree", scope="read", quota_minute=2)
    large, _ = _store(env).create("large", scope="read")
    with TestClient(create_app()) as client:
        h = {"X-API-Key": serree}
        assert client.get("/api/meta", headers=h).status_code == 200
        assert _rpc(client, "tools/list", key=serree).status_code == 200
        r = client.get("/api/meta", headers=h)
        assert r.status_code == 429
        assert 1 <= int(r.headers["Retry-After"]) <= 60
        assert "quota depasse" in r.json()["detail"] and "serree" in r.json()["detail"]
        r = _rpc(client, "tools/list", key=serree)
        assert r.status_code == 429 and "Retry-After" in r.headers
        # le quota est par cle : l'autre passe
        assert client.get("/api/meta", headers={"X-API-Key": large}).status_code == 200
        # /api/health n'est pas compte
        assert client.get("/api/health").status_code == 200
    motifs = [ligne.get("motif") for ligne in _journal(tmp_path)]
    assert motifs.count("quota_minute") == 2


def test_quota_par_defaut_vient_de_l_environnement(env, monkeypatch):
    from service.app import create_app

    monkeypatch.setenv("PCFNO3D_QUOTA_MINUTE", "0")
    monkeypatch.setenv("PCFNO3D_QUOTA_JOUR", "2")
    secret, _ = _store(env).create("jour", scope="read")
    with TestClient(create_app()) as client:
        h = {"X-API-Key": secret}
        assert [client.get("/api/meta", headers=h).status_code for _ in range(3)] \
            == [200, 200, 429]
        assert int(client.get("/api/meta", headers=h).headers["Retry-After"]) <= 86_400


def test_quota_modifiable_par_la_cli(env):
    from pcfno3d.cli import main

    _store(env).create("cli-quota", scope="read")
    base = ["-c", "test", "--run-dir", str(env.paths.run_dir), "apikey"]
    assert main(base + ["quota", "cli-quota", "--per-minute", "7", "--per-day", "0"]) == 0
    rec = next(r for r in _store(env).records if r.name == "cli-quota")
    assert (rec.quota_minute, rec.quota_jour) == (7, 0)
    assert main(base + ["quota", "cli-quota", "--reset"]) == 0
    rec = next(r for r in _store(env).records if r.name == "cli-quota")
    assert (rec.quota_minute, rec.quota_jour) == (None, None)
    assert main(base + ["quota", "inconnue", "--per-minute", "1"]) == 1


def test_empreinte_atom_toujours_acceptee_avec_quota(tmp_path):
    """Chemin du notebook atom : cle enregistree par empreinte, rejouable, quota reglable."""
    from service.auth import ApiKeyStore, generate_key, hash_key

    secret = generate_key()
    store = ApiKeyStore(tmp_path / "api_keys.json")
    store.register_hash("atom-worker", hash_key(secret), scope="read")
    relu = ApiKeyStore(tmp_path / "api_keys.json")
    assert relu.verify(secret).name == "atom-worker"
    rec = relu.register_hash("atom-worker", hash_key(secret), scope="read",
                             quota_minute=1200)
    assert rec.quota_minute == 1200 and len(relu.records) == 1
    assert ApiKeyStore(tmp_path / "api_keys.json").records[0].quota_minute == 1200


# ---------------------------------------------------------------------------- journal
def test_journal_d_acces_sans_jamais_le_secret(env, tmp_path):
    from service.app import create_app
    from service.auth import generate_key, hash_key

    secret, record = _store(env).create("journalisee", scope="read")
    faux = generate_key()
    with TestClient(create_app()) as client:
        assert client.get("/api/meta", headers={"X-API-Key": secret}).status_code == 200
        assert client.get(f"/api/instants?split=test&api_key={secret}").status_code == 200
        assert client.get("/api/meta", headers={"Authorization": f"Bearer {secret}"}) \
            .status_code == 200
        assert client.get(f"/api/meta?api_key={faux}").status_code == 401
        assert client.get("/api/meta").status_code == 401      # meme IP, meme minute
        assert _rpc(client, "tools/list", key=secret).status_code == 200
        client.get("/api/health")                         # publique : pas journalisee

    brut = (tmp_path / "acces.jsonl").read_text(encoding="utf-8")
    for interdit in (secret, faux, hash_key(secret), hash_key(faux), hash_key(secret)[:12]):
        assert interdit not in brut
    lignes = _journal(tmp_path)
    assert len(lignes) == 5                    # le 2e refus de la minute est regroupe
    assert all(set(ligne) >= {"ts", "cle", "empreinte", "portee", "methode", "chemin",
                              "statut", "duree_ms", "ip"} for ligne in lignes)
    assert all(ligne["ts"].endswith("Z") for ligne in lignes)
    assert all("?" not in ligne["chemin"] for ligne in lignes)

    ok = lignes[0]
    assert ok["cle"] == "journalisee" and ok["empreinte"] == hash_key(secret)[:8]
    assert ok["portee"] == "read" and ok["methode"] == "GET" and ok["statut"] == 200
    assert ok["chemin"] == "/api/meta" and ok["duree_ms"] >= 0 and ok["ip"]
    par_query = lignes[1]
    assert par_query["chemin"] == "/api/instants" and par_query["params"] == "split=test"
    assert par_query["cle"] == "journalisee"
    refus = [ligne for ligne in lignes if ligne["statut"] == 401]
    assert [r["motif"] for r in refus] == ["cle_invalide"]
    assert all(r["cle"] is None and r["empreinte"] is None for r in refus)
    assert lignes[-1]["chemin"].startswith("/mcp") and lignes[-1]["cle"] == "journalisee"


def test_refus_regroupes_une_ligne_par_ip_et_par_minute(env, tmp_path, monkeypatch):
    """Un flot de 401 ne doit pas purger l'historique par la rotation."""
    import service.app as app_mod
    from service.auth import generate_key

    _store(env).create("existe", scope="read")
    minute = {"m": 1000}
    monkeypatch.setattr(app_mod, "_minute", lambda: minute["m"])
    app = app_mod.create_app()
    with TestClient(app) as client:
        for _ in range(50):
            assert client.get("/api/meta", headers={"X-API-Key": generate_key()}) \
                .status_code == 401
        # une autre IP a sa propre ligne (meme application, sans relancer le cycle de vie)
        assert TestClient(app, client=("10.0.0.2", 1)).get("/api/meta").status_code == 401
        minute["m"] += 1
        assert client.get("/api/meta").status_code == 401
    refus = [ligne for ligne in _journal(tmp_path) if ligne["statut"] == 401]
    assert [(r["ip"], r.get("refus_regroupes")) for r in refus] == [
        ("testclient", None), ("10.0.0.2", None), ("testclient", 49)]


def test_lien_ne_relaie_jamais_la_cle_locale_en_query(tmp_path, monkeypatch):
    """Mode lien : `?api_key=` (cle LOCALE) est retire avant le relais vers l'amont."""
    import httpx

    from service.auth import ApiKeyStore, default_store_path
    from service.link import Upstream, forward_params

    capture: dict = {}

    async def faux_amont(self, method, url, params=None, headers=None, content=None, **_):
        capture.update(method=method, url=url, params=params, headers=headers)
        return httpx.Response(200, json={"relaye": True})

    monkeypatch.setattr(httpx.AsyncClient, "request", faux_amont)
    for var, val in (("PCFNO3D_CONFIG", "test"), ("PCFNO3D_RUN_DIR", str(tmp_path)),
                     ("PCFNO3D_ACCESS_LOG", "off")):
        monkeypatch.setenv(var, val)
    for var in ("PCFNO3D_UPSTREAM", "PCFNO3D_API_KEYS", "PCFNO3D_INSECURE_OPEN"):
        monkeypatch.delenv(var, raising=False)
    locale, _ = ApiKeyStore(default_store_path(tmp_path)).create("nav", scope="read")
    Upstream(url="http://amont.invalid", key="cle-amont-de-test").save(tmp_path)

    from service.app import create_app
    with TestClient(create_app()) as client:
        r = client.get(f"/api/point?x=1&api_key={locale}&y=2&API_KEY={locale}")
    assert r.status_code == 200 and r.json() == {"relaye": True}
    assert capture["url"] == "http://amont.invalid/api/point"
    assert list(capture["params"]) == [("x", "1"), ("y", "2")]
    assert locale not in repr(capture)                 # ni en query, ni en en-tete
    assert capture["headers"]["X-API-Key"] == "cle-amont-de-test"

    from starlette.datastructures import QueryParams
    assert forward_params(QueryParams("a=1&api_key=s&a=2")) == [("a", "1"), ("a", "2")]


def test_journal_rotation_par_taille(tmp_path):
    from service.journal import JournalAcces

    j = JournalAcces(tmp_path / "acces.jsonl", max_octets=300, garder=2)
    for i in range(30):
        j.ecrire(i=i, remplissage="x" * 40)
    assert (tmp_path / "acces.jsonl").stat().st_size <= 300
    assert (tmp_path / "acces.jsonl.1").exists() and (tmp_path / "acces.jsonl.2").exists()
    assert not (tmp_path / "acces.jsonl.3").exists()
    derniere = (tmp_path / "acces.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    assert json.loads(derniere)["i"] == 29


def test_params_masques():
    from starlette.datastructures import QueryParams

    from service.journal import params_sans_secret

    q = QueryParams("x=1&api_key=pcf_secret&API_KEY=autre&note=pcf_colle&z=3")
    assert params_sans_secret(q) == "x=1&note=%2A%2A%2A&z=3"


# ---------------------------------------------------------------------------- georef
def test_meta_porte_le_georeferencement(env):
    from service.app import STATE, create_app

    secret, _ = _store(env).create("geo", scope="read")
    with TestClient(create_app()) as client:
        h = {"X-API-Key": secret}
        assert client.get("/api/meta", headers=h).json()["georef"] is None   # mini-jeu

        real = STATE["model"].cfg.real
        sauve = (real.origin_lat, real.origin_lon, real.rotation_deg, real.origin_assumed)
        real.origin_lat, real.origin_lon, real.rotation_deg, real.origin_assumed = \
            36.0, 9.7, 0.0, True
        try:
            geo = client.get("/api/meta", headers=h).json()["georef"]
        finally:
            real.origin_lat, real.origin_lon, real.rotation_deg, real.origin_assumed = sauve
    assert geo == {"lat0": 36.0, "lon0": 9.7, "rotation_deg": 0.0, "suppose": True,
                   "coin": "sud-ouest"}


def test_cles_gerees_par_l_api_admin(env, tmp_path):
    """BF42 : creation (secret montre une fois), quotas et revocation par l'API, portee admin seulement ;
    la cle revoquee tombe aussitot ; aucun secret dans le journal d'acces."""
    from service.app import create_app

    admin, _ = _store(env).create("admin-api", scope="admin")
    lecture, _ = _store(env).create("lecteur", scope="read")
    with TestClient(create_app()) as client:
        h = {"X-API-Key": admin}
        assert client.post("/api/keys", json={"nom": "x"}, headers={"X-API-Key": lecture}).status_code == 403
        r = client.post("/api/keys", json={"nom": "appli-agri", "portee": "read", "quota_minute": 30}, headers=h)
        assert r.status_code == 200
        nouvelle = r.json()["cle"]
        assert nouvelle.startswith("pcf_") and r.json()["quota_minute"] == 30
        assert client.post("/api/keys", json={"nom": "appli-agri"}, headers=h).status_code == 409
        assert client.post("/api/keys", json={"nom": "mauvais nom!"}, headers=h).status_code == 422
        assert client.get("/api/meta", headers={"X-API-Key": nouvelle}).status_code == 200
        q = client.patch("/api/keys/appli-agri", json={"quota_jour": 500}, headers=h)
        assert q.status_code == 200 and q.json()["quota_jour"] == 500
        assert client.delete("/api/keys/appli-agri", headers=h).status_code == 200
        assert client.get("/api/meta", headers={"X-API-Key": nouvelle}).status_code == 401
        assert client.delete("/api/keys/appli-agri", headers=h).status_code == 404
        noms = [c["name"] for c in client.get("/api/keys", headers=h).json()["cles"]]
        assert "appli-agri" in noms
    journal = (tmp_path / "acces.jsonl").read_text(encoding="utf-8")
    assert nouvelle not in journal and admin not in journal
