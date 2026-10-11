"""L'API d'inference, testee en memoire (sans lancer uvicorn).

Les tests sont ignores si fastapi n'est pas installe : le paquet reste utilisable sans le
service (`pip install -e .[serve]` pour l'activer).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient      # noqa: E402


@pytest.fixture(scope="module")
def client(cfg, index, norm):
    """Entraine 2 epoques puis monte l'application sur ce run."""
    from pcfno3d.train import Trainer

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()

    os.environ["PCFNO3D_CONFIG"] = "test"
    os.environ["PCFNO3D_DATA_DIR"] = str(cfg.paths.data_dir)
    os.environ["PCFNO3D_CACHE_DIR"] = str(cfg.paths.cache_dir)
    os.environ["PCFNO3D_RUN_DIR"] = str(cfg.paths.run_dir)
    os.environ["PCFNO3D_DEVICE"] = "cpu"

    from service.app import app
    with TestClient(app) as c:
        yield c


def test_health_et_meta(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["modele"].endswith(".pt")

    meta = client.get("/meta").json()
    assert meta["grille"]["nz"] == len(meta["z_levels_asl_m"])
    assert "wind_speed_m_s" in meta["variables"]


def test_liste_des_instants(client):
    body = client.get("/instants", params={"split": "test", "limit": 5}).json()
    assert body["total"] > 0
    assert all(r["split"] == "test" for r in body["instants"])


def test_reconstruction_dun_instant(client):
    stem = client.get("/instants", params={"split": "test"}).json()["instants"][0]["stem"]
    body = client.post("/reconstruct", json={"stem": stem, "keep": 1.0}).json()
    assert body["capteurs_utilises"] > 0
    assert body["inference_s"] > 0
    assert body["dso4_conforme"] is True          # DSO4 : < 1 s par instant
    assert body["grille"]["nz"] >= 1


def test_carte_a_une_altitude(client):
    meta = client.get("/meta").json()
    z = meta["z_levels_asl_m"][-1]
    body = client.get("/field", params={"z": z, "variable": "speed", "stride": 1}).json()
    assert body["ny"] > 0 and body["nx"] > 0
    assert len(body["valeurs"]) == body["ny"]
    # sous le relief, la valeur est None (JSON n'admet pas NaN)
    assert any(v is None or isinstance(v, float) for row in body["valeurs"] for v in row)


def test_altitude_hors_domaine_renvoie_400(client):
    assert client.get("/field", params={"z": 99999}).status_code == 400


def test_vent_en_un_point(client):
    meta = client.get("/meta").json()
    z = meta["z_levels_asl_m"][-1]
    body = client.get("/point", params={"x": 100, "y": 100, "z": z}).json()
    assert 0.0 <= body["wind_direction_deg"] <= 360.0
    assert body["z_asl_m"] == z


def test_instant_inconnu_renvoie_404(client):
    assert client.post("/reconstruct", json={"stem": "999_999999"}).status_code in (404, 422)
