"""L'application web, l'API protegee par cle, et le serveur MCP.

Tout est teste en memoire (pas d'uvicorn) sur le mini-dataset. Les tests sont ignores si
fastapi ou le SDK MCP ne sont pas installes : le paquet reste utilisable sans le service.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")
from fastapi.testclient import TestClient      # noqa: E402

MCP_HEADERS = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream",
               "MCP-Protocol-Version": "2025-06-18"}


def _rpc(client, method, params=None, key=None, rid=1):
    headers = dict(MCP_HEADERS)
    if key:
        headers["X-API-Key"] = key
    return client.post("/mcp", headers=headers, json={
        "jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})


def _tool(client, name, arguments=None, key=None):
    r = _rpc(client, "tools/call", {"name": name, "arguments": arguments or {}}, key=key)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body["error"]
    result = body["result"]
    assert result.get("isError") is not True, result
    # le contenu structure est le plus pratique a verifier ; sinon on relit le texte
    if "structuredContent" in result:
        return result["structuredContent"]
    return json.loads(result["content"][0]["text"])


# ---------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def trained(cfg, index, norm):
    """Un checkpoint entraine 2 epoques, suffisant pour servir des predictions."""
    from pcfno3d.train import Trainer

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg


@pytest.fixture
def env(trained, monkeypatch):
    monkeypatch.setenv("PCFNO3D_CONFIG", "test")
    monkeypatch.setenv("PCFNO3D_DATA_DIR", str(trained.paths.data_dir))
    monkeypatch.setenv("PCFNO3D_CACHE_DIR", str(trained.paths.cache_dir))
    monkeypatch.setenv("PCFNO3D_RUN_DIR", str(trained.paths.run_dir))
    monkeypatch.setenv("PCFNO3D_DEVICE", "cpu")
    for var in ("PCFNO3D_API_KEYS", "PCFNO3D_INSECURE_OPEN", "PCFNO3D_QUOTA_MINUTE",
                "PCFNO3D_QUOTA_JOUR"):
        monkeypatch.delenv(var, raising=False)
    return trained


@pytest.fixture
def client(env):
    """Service protege, client muni d'une cle `read` envoyee a chaque requete.

    Le service est ferme par defaut (lot M7) : il n'y a plus de mode ouvert implicite.
    Les tests fonctionnels passent donc par une vraie cle, de la portee la plus faible.
    """
    from service.auth import ApiKeyStore, default_store_path

    store_path = default_store_path(env.paths.run_dir)
    backup = store_path.read_text(encoding="utf-8") if store_path.exists() else None
    store_path.unlink(missing_ok=True)
    read_key, _ = ApiKeyStore(store_path).create("tests-lecture", scope="read")
    from service.app import create_app
    app = create_app()
    with TestClient(app, headers={"X-API-Key": read_key}) as c:
        yield c
    store_path.unlink(missing_ok=True)
    if backup is not None:
        store_path.write_text(backup, encoding="utf-8")


@pytest.fixture
def secured(env):
    """Service avec une cle `read` et une cle `admin`."""
    from service.auth import ApiKeyStore, default_store_path

    path = default_store_path(env.paths.run_dir)
    path.unlink(missing_ok=True)
    store = ApiKeyStore(path)
    read_key, _ = store.create("app-web", scope="read")
    admin_key, _ = store.create("pipeline", scope="admin")

    from service.app import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c, read_key, admin_key
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------- trousseau
def test_cle_stockee_en_hash_uniquement(tmp_path):
    from service.auth import ApiKeyStore, hash_key

    store = ApiKeyStore(tmp_path / "keys.json")
    assert store.empty and not store.open_mode       # vide = ferme, jamais ouvert
    secret, record = store.create("demo", scope="read")

    assert secret.startswith("pcf_") and len(secret) > 20
    raw = (tmp_path / "keys.json").read_text(encoding="utf-8")
    assert secret not in raw, "le secret ne doit jamais etre ecrit en clair"
    assert hash_key(secret) in raw
    assert not store.empty

    assert store.verify(secret) is not None
    assert store.verify(secret + "x") is None
    assert store.verify(None) is None
    assert store.verify("") is None

    # relecture depuis le disque : la cle reste valable
    assert ApiKeyStore(tmp_path / "keys.json").verify(secret).name == "demo"

    assert store.revoke("demo") is True
    assert store.verify(secret) is None
    assert store.revoke("demo") is False
    assert record.name == "demo"


def test_portees(tmp_path):
    from service.auth import ApiKeyStore, scope_allows

    store = ApiKeyStore(tmp_path / "keys.json")
    read, _ = store.create("r", scope="read")
    admin, _ = store.create("a", scope="admin")
    assert scope_allows(store.verify(read), "read") is True
    assert scope_allows(store.verify(read), "admin") is False
    assert scope_allows(store.verify(admin), "read") is True       # admin couvre read
    assert scope_allows(store.verify(admin), "admin") is True
    assert scope_allows(None, "read") is False
    with pytest.raises(ValueError):
        store.create("x", scope="root")
    with pytest.raises(ValueError):
        store.create("r")                                          # nom deja pris


def test_cles_par_variable_d_environnement(tmp_path, monkeypatch):
    from service.auth import ApiKeyStore

    monkeypatch.setenv("PCFNO3D_API_KEYS", "autre-app:pcf_depuis_env")
    store = ApiKeyStore(tmp_path / "keys.json")
    assert not store.empty
    assert store.verify("pcf_depuis_env").name == "autre-app"


# ---------------------------------------------------------------------------- API
def test_health_sans_cle_dit_que_tout_est_refuse(env):
    """Sans aucune cle : /health le dit, et le reste refuse (ferme par defaut)."""
    store_path = env.paths.run_dir / "api_keys.json"
    backup = store_path.read_text(encoding="utf-8") if store_path.exists() else None
    store_path.unlink(missing_ok=True)
    from service.app import create_app
    try:
        with TestClient(create_app()) as c:
            body = c.get("/api/health").json()
            assert body["status"] == "ok"
            assert body["auth"] == "aucune_cle"
            assert "aucune cle configuree" in body["message"]
            assert body["mcp"] == "/mcp"
            assert c.get("/api/meta").status_code == 401
            assert _rpc(c, "tools/list").status_code == 401
    finally:
        if backup is not None:
            store_path.write_text(backup, encoding="utf-8")


def test_api_repond_avec_la_cle(client):
    assert client.get("/api/health").json()["auth"] == "cle_api"
    meta = client.get("/api/meta").json()
    assert meta["grille"]["nz"] == len(meta["z_levels_asl_m"])


def test_api_protegee_refuse_sans_cle(secured):
    client, read_key, _ = secured
    assert client.get("/api/health").status_code == 200        # /health reste public
    assert client.get("/api/health").json()["auth"] == "cle_api"

    assert client.get("/api/meta").status_code == 401
    assert client.get("/api/meta", headers={"X-API-Key": "pcf_faux"}).status_code == 401
    assert client.get("/api/meta", headers={"X-API-Key": read_key}).status_code == 200
    # Authorization: Bearer et ?api_key= fonctionnent aussi
    assert client.get("/api/meta",
                      headers={"Authorization": f"Bearer {read_key}"}).status_code == 200
    assert client.get(f"/api/meta?api_key={read_key}").status_code == 200


def test_portee_read_ne_peut_pas_exporter(secured):
    client, read_key, admin_key = secured
    stem = client.get("/api/instants", headers={"X-API-Key": read_key}).json()["instants"][0]["stem"]

    r = client.post("/api/reconstruct", headers={"X-API-Key": read_key},
                    json={"stem": stem, "export": True})
    assert r.status_code == 403

    r = client.post("/api/reconstruct", headers={"X-API-Key": admin_key},
                    json={"stem": stem, "export": True, "stride": 2})
    assert r.status_code == 200, r.text
    assert "export" in r.json()

    assert client.get("/api/keys", headers={"X-API-Key": read_key}).status_code == 403
    keys = client.get("/api/keys", headers={"X-API-Key": admin_key}).json()["cles"]
    assert {k["name"] for k in keys} == {"app-web", "pipeline"}
    assert all("key_hash" not in k for k in keys)      # jamais le hash complet


def test_profil_vertical(client):
    meta = client.get("/api/meta").json()
    body = client.get("/api/profile", params={"x": 200, "y": 200}).json()
    assert len(body["niveaux"]) == meta["grille"]["nz"]
    above = [n for n in body["niveaux"] if not n["sous_le_relief"]]
    assert above, "aucun niveau au-dessus du relief"
    assert all(n["wind_speed_m_s"] is not None for n in above)
    assert all(n["wind_speed_m_s"] is None for n in body["niveaux"] if n["sous_le_relief"])


def test_carte_et_point(client):
    meta = client.get("/api/meta").json()
    z = meta["z_levels_asl_m"][-1]
    field = client.get("/api/field", params={"z": z, "stride": 1}).json()
    assert len(field["valeurs"]) == field["ny"]
    assert field["terrain_max_m"] >= field["terrain_min_m"]

    point = client.get("/api/point", params={"x": 100, "y": 100, "z": z}).json()
    assert 0.0 <= point["wind_direction_deg"] <= 360.0


def test_application_web_servie(client):
    page = client.get("/app/")
    assert page.status_code == 200
    assert "PC-FNO3D" in page.text
    assert client.get("/app/app.js").status_code == 200
    assert client.get("/app/style.css").status_code == 200
    assert client.get("/", follow_redirects=False).status_code in (307, 302)


# ---------------------------------------------------------------------------- MCP
def test_mcp_initialise_et_liste_les_outils(client):
    r = _rpc(client, "initialize", {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"}})
    assert r.status_code == 200, r.text
    info = r.json()["result"]
    assert info["serverInfo"]["name"] == "pcfno3d-vent-3d"

    tools = {t["name"] for t in _rpc(client, "tools/list").json()["result"]["tools"]}
    assert tools == {"grid_meta", "list_instants", "wind_at_point", "wind_profile",
                     "wind_field_at_altitude", "instant_summary", "dso_report",
                     "export_instant", "alertes_actives", "production_eolienne",
                     "prevision_point", "decision_agricole", "fenetres_maintenance",
                     "variations_production"}


def test_mcp_outils_de_lecture(client):
    meta = _tool(client, "grid_meta")
    assert meta["grille"]["nz"] == len(meta["z_levels_asl_m"])
    z = meta["z_levels_asl_m"][-1]

    instants = _tool(client, "list_instants", {"split": "test", "limit": 5})
    assert instants["total"] > 0
    stem = instants["instants"][0]["stem"]

    point = _tool(client, "wind_at_point",
                  {"x_m": 200, "y_m": 200, "z_asl_m": z, "stem": stem})
    assert point["stem"] == stem
    assert 0.0 <= point["wind_direction_deg"] <= 360.0

    profile = _tool(client, "wind_profile", {"x_m": 200, "y_m": 200, "stem": stem})
    assert len(profile["niveaux"]) == meta["grille"]["nz"]

    field = _tool(client, "wind_field_at_altitude",
                  {"z_asl_m": z, "stem": stem, "stride": 1})
    assert field["ny"] > 0 and len(field["valeurs"]) == field["ny"]

    summary = _tool(client, "instant_summary", {"stem": stem})
    assert summary["noeuds_3d"] > 0
    assert summary["inference_s"] > 0
    assert summary["secteur_dominant_deg"] in {"0", "45", "90", "135", "180", "225",
                                               "270", "315"}


def test_mcp_dso_report_ne_devine_rien(client, cfg):
    body = _tool(client, "dso_report")
    # une evaluation a deja tourne dans la suite de tests, ou alors elle est annoncee absente
    if body["disponible"]:
        assert "dso1_rmse" in body["resume"]
    else:
        assert "evaluate" in body["message"]


def test_mcp_argument_invalide_est_rejete(client):
    r = _rpc(client, "tools/call",
             {"name": "wind_field_at_altitude", "arguments": {"z_asl_m": 50, "variable": "xyz"}})
    body = r.json()
    assert "error" in body or body["result"].get("isError") is True


def test_mcp_exige_la_cle_quand_elle_est_active(secured):
    client, read_key, admin_key = secured
    assert _rpc(client, "tools/list").status_code == 401
    assert _rpc(client, "tools/list", key="pcf_faux").status_code == 401
    assert _rpc(client, "tools/list", key=read_key).status_code == 200

    meta = _tool(client, "grid_meta", key=read_key)
    assert meta["noeuds_valides"] > 0

    # export_instant ecrit sur le disque : portee admin obligatoire
    r = _rpc(client, "tools/call", {"name": "export_instant", "arguments": {"stride": 2}},
             key=read_key)
    assert r.status_code == 403

    out = _tool(client, "export_instant", {"stride": 2}, key=admin_key)
    assert out["manifeste"]["format"] == "pcfno3d-web/1"
    assert os.path.isdir(out["dossier"])


# ---------------------------------------------------------------------------- donnees reelles
def test_reconstruction_depuis_des_stations_reelles(client):
    """POST /api/reconstruct/stations : lat/lon -> volume, reutilisable par /point."""
    from service.app import STATE

    r = client.post("/api/reconstruct/stations", json={
        "timestamp_utc": "2026-10-14T12:00:00Z",
        "stations": [{"station_id": "A", "lat": 36.40, "lon": 10.10, "wind_u_m_s": 3.0,
                      "wind_v_m_s": 1.0}]})
    assert r.status_code == 409 and "origin_lat" in r.json()["detail"]   # pas d'ancrage

    cfg = STATE["model"].cfg
    cfg.real.origin_lat, cfg.real.origin_lon = 36.40, 10.10
    try:
        geom = STATE["model"].geom
        from pcfno3d.realdata import GeoRef
        lat, lon = GeoRef(36.40, 10.10).to_latlon(geom.nx // 2 * geom.dx, geom.ny // 2 * geom.dy)
        r = client.post("/api/reconstruct/stations", json={
            "timestamp_utc": "2026-10-14T12:00:00Z",
            "stations": [{"station_id": "A", "lat": float(lat), "lon": float(lon),
                          "height_agl_m": 25, "wind_speed_m_s": 6.0, "wind_dir_deg": 250,
                          "temperature_c": 21.0, "pressure_pa": 100800.0},
                         {"station_id": "Loin", "lat": 40.0, "lon": 15.0, "wind_u_m_s": 1.0,
                          "wind_v_m_s": 1.0}]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["stem"] == "live_20261014_120000"
        assert body["stations"]["stations"] == 1 and body["stations"]["hors_domaine"] == 1
        p = client.get("/api/profile", params={"x": 0, "y": 0, "stem": body["stem"]})
        assert p.status_code == 200 and len(p.json()["niveaux"]) == geom.nz
    finally:
        cfg.real.origin_lat = cfg.real.origin_lon = None


# ---------------------------------------------------------------------------- contrat ANEO
def _instant_csv(cfg, corrompre: bool = False, part: float = 1.0, modifier=None) -> bytes:
    """Les stations du premier instant d'un jour du mini-dataset, en CSV ; `part` garde
    une station sur 1/part (panne partielle repartie sur le domaine)."""
    import pandas as pd

    meta = json.loads((cfg.paths.data_dir / "meta.json").read_text(encoding="utf-8"))
    day = sorted(cfg.paths.data_dir.glob("aneo_observations_day_*"))[0]
    df = pd.read_csv(day)
    df = df[(df["timestamp_utc"] == df["timestamp_utc"].iloc[0])
            & (df["point_id"] < meta["noeuds_3d"]["point_id_debut"])]
    if part < 1.0:
        rng = np.random.default_rng(0)
        df = df[rng.random(len(df)) < part]
    if corrompre:
        df = df.astype({"temperature_c": object})
        df.iloc[0, df.columns.get_loc("temperature_c")] = "abc"
        df = pd.concat([df, df.iloc[[5]]])
    if modifier is not None:
        df = modifier(df.copy())
    return df.to_csv(index=False).encode("utf-8")


def _upload(client, data: bytes, name: str = "instant.csv"):
    r = client.post("/api/reconstruct/upload", files={"file": (name, data, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()                     # JSON strict : un NaN aurait deja donne un 500
    assert math.isfinite(body["vitesse_max_m_s"]) and body["capteurs_utilises"] > 0
    return body


def test_upload_conforme_joint_le_rapport_qualite(client, env):
    r = client.post("/api/reconstruct/upload",
                         files={"file": ("instant.csv", _instant_csv(env), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["capteurs_utilises"] > 0
    assert body["statut_qualite"] == "conforme"
    q = body["qualite"]
    assert q["verdict"] == "conforme" and q["erreurs"] == []
    # ~10 % de capteurs manquants dans le mini-dataset : un avertissement, pas un refus
    assert any("interpolees" in a for a in body["avertissements"])


@pytest.mark.parametrize("part", [0.9, 0.5, 0.25])
def test_upload_panne_partielle_reconstruit(client, env, part):
    """DSO5 : un sous-ensemble de stations est reconstruit (200), jamais refuse (422)."""
    r = client.post("/api/reconstruct/upload",
                         files={"file": ("instant.csv", _instant_csv(env, part=part),
                                         "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["capteurs_utilises"] > 0 and body["vitesse_max_m_s"] > 0
    assert body["statut_qualite"] in ("conforme", "quarantaine")
    q = body["qualite"]
    assert q["qualite"]["fraicheur"]["stations_muettes"]["nombre"] == 0
    assert q["qualite"]["fraicheur"]["stations_absentes"]["nombre"] > 0
    assert any("absente" in a for a in body["avertissements"])
    if body["statut_qualite"] == "quarantaine":
        assert set(q["motifs"]) <= set(body["avertissements"])


def test_upload_non_conforme_renvoie_422_et_le_rapport(client, env):
    r = client.post("/api/reconstruct/upload",
                         files={"file": ("instant.csv", _instant_csv(env, corrompre=True),
                                         "text/csv")})
    assert r.status_code == 422, r.text
    body = r.json()
    assert "contrat ANEO" in body["detail"]
    q = body["qualite"]
    assert q["verdict"] == "rejet"
    regles = {e["regle"] for e in q["erreurs"]}
    assert {"type_reel", "doublon"} <= regles


def test_upload_25_pct_visibles_par_le_masque(client, env):
    """Encodage du dataset : capteur manquant = ligne mask_i_obs = 0. Panne connue."""
    def masquer(df):
        rng = np.random.default_rng(1)
        df.loc[rng.random(len(df)) >= 0.25, "mask_i_obs"] = 0
        return df

    body = _upload(client, _instant_csv(env, modifier=masquer))
    q = body["qualite"]
    assert body["statut_qualite"] == "quarantaine"            # 75 % interpolees ou signalees
    assert q["qualite"]["stations_valides"]["part_mesurees"] > 0.9
    assert q["qualite"]["fraicheur"]["stations_muettes"]["nombre"] == 0


def test_upload_station_a_99_degres_masquee(client, env):
    def chauffer(df):
        df.iloc[3, df.columns.get_loc("temperature_c")] = 99.0
        return df

    body = _upload(client, _instant_csv(env, modifier=chauffer))
    assert body["statut_qualite"] == "quarantaine"
    masquees = body["qualite"]["qualite"]["stations_a_masquer"]
    assert masquees["nombre"] == 1
    assert any("masquer" in a for a in body["avertissements"])


def test_upload_station_sans_vent_ni_temperature(client, env):
    def vider(df):
        for c in ("wind_u_m_s", "wind_v_m_s", "temperature_c"):
            df[c] = df[c].astype(object)
            df.iloc[2, df.columns.get_loc(c)] = None
        return df

    body = _upload(client, _instant_csv(env, modifier=vider))
    assert body["statut_qualite"] in ("conforme", "quarantaine")
    assert body["qualite"]["qualite"]["stations_a_masquer"]["nombre"] == 1


def test_upload_gzip_reconnu_aux_octets(client, env):
    import gzip

    data = gzip.compress(_instant_csv(env))
    body = _upload(client, data, name="instant.csv")          # nom sans .gz
    assert body["statut_qualite"] == "conforme"
    assert _upload(client, data, name="instant.csv.gz")["statut_qualite"] == "conforme"
