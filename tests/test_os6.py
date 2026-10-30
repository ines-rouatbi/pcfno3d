"""OS6 (pcfno3d.os6, scripts/os6_reseau_aneo.py) : candidats sous contraintes, placement
glouton sur la validation, courbe de dimensionnement sur le test, plan CSV / GeoJSON."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_forcage_era5 import _jeu_reel, cfg_era5, forcage  # noqa: F401  (fixtures)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def jeu(cfg_era5, index, tmp_path_factory):  # noqa: F811
    from pcfno3d.train import Trainer

    c, _ = _jeu_reel(cfg_era5, index, tmp_path_factory.mktemp("os6"))
    Trainer(c, device="cpu", verbose=False).fit()
    return c


def _contraintes(tmp_path, **kw) -> Path:
    d = {"pas_candidats_m": 100.0, "distance_min_m": 200.0, "marge_bord_m": 0.0,
         "altitude_max_m": 1e4, "pente_max_deg": 90.0, "z0_min_m": 0.0, "n_max": 4, **kw}
    f = tmp_path / "contraintes.json"
    f.write_text(json.dumps(d), encoding="utf-8")
    return f


def test_candidats_respectent_les_contraintes(jeu, tmp_path):
    from pcfno3d import os6
    from pcfno3d.dataset import cellules_stations
    from pcfno3d.geometry import Geometry
    from pcfno3d.realdata import GeoRef

    g = Geometry(jeu)
    c = os6.Contraintes.charger(_contraintes(tmp_path))
    cand = os6.candidats(g, c, GeoRef.from_cfg(jeu))
    assert len(cand) > 0
    assert g.valid.any(axis=0)[cand[:, 0], cand[:, 1]].all()
    st = cellules_stations(g)
    d = np.hypot((cand[:, None, 0] - st[None, :, 0]) * g.dy,
                 (cand[:, None, 1] - st[None, :, 1]) * g.dx)
    assert d.min() >= c.distance_min_m
    # une zone exclue (lat/lon) retire les candidats qu'elle couvre
    geo = GeoRef.from_cfg(jeu)
    lat, lon = geo.to_latlon(np.array([0.0, g.nx * g.dx]), np.array([0.0, g.ny * g.dy]))
    tout = [[lat[0] - 1, lon[0] - 1], [lat[0] - 1, lon[1] + 1], [lat[1] + 1, lon[1] + 1],
            [lat[1] + 1, lon[0] - 1]]
    c2 = os6.Contraintes.charger(_contraintes(tmp_path, exclusions=[tout]))
    assert len(os6.candidats(g, c2, geo)) == 0
    with pytest.raises(ValueError, match="contraintes inconnues"):
        os6.Contraintes(**{}).charger(_contraintes(tmp_path, budget_euros=1))


def test_grille_et_hasard(jeu, tmp_path):
    from pcfno3d import os6
    from pcfno3d.geometry import Geometry

    g = Geometry(jeu)
    c = os6.Contraintes.charger(_contraintes(tmp_path))
    cand = os6.candidats(g, c)
    for n in (3, 6):
        gr = os6.grille_reguliere(cand, n, g)
        assert len(gr) <= n and len(gr) >= min(n, len(cand)) // 2
        h = os6.au_hasard(cand, n, g, c.distance_min_m, 0)
        assert len(h) <= n
        for i in range(len(h)):
            for k in range(i):
                assert math.hypot((h[i, 0] - h[k, 0]) * g.dy, (h[i, 1] - h[k, 1]) * g.dx) \
                    >= c.distance_min_m


def test_os6_de_bout_en_bout(jeu, tmp_path, monkeypatch):
    """Placement (validation), courbe (test), plan : fichiers ecrits, ordre et gains."""
    import pcfno3d.config as config_mod

    spec = importlib.util.spec_from_file_location("os6_script", REPO / "scripts" / "os6_reseau_aneo.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(config_mod.Config, "load", classmethod(lambda cls, _n: jeu))
    sortie = tmp_path / "os6"
    monkeypatch.setattr(sys, "argv", [
        "os6", "-c", "x", "--device", "cpu", "--sortie", str(sortie),
        "--contraintes", str(_contraintes(tmp_path)), "--n-val", "2", "--n-test", "2",
        "--par-etape", "2", "--tailles", "2,4"])
    assert m.main() == 0
    for f in ("candidats.csv", "ordre_validation.csv", "plan_aneo.csv", "plan_aneo.geojson",
              "courbe_test.csv", "courbe_test_par_niveau.csv", "os6.json", "courbe.png",
              "carte.png"):
        assert (sortie / f).exists(), f
    p = pd.read_csv(sortie / "plan_aneo.csv")
    assert list(p["ordre"]) == list(range(1, len(p) + 1)) and len(p) == 4
    assert p[["lat", "lon", "hauteur_mat_m", "gain_rmse_validation_m_s"]].notna().all().all()
    gj = json.loads((sortie / "plan_aneo.geojson").read_text(encoding="utf-8"))
    types_ = {f["properties"]["type"] for f in gj["features"]}
    assert types_ == {"capteur ANEO propose", "station existante"}
    r = pd.read_csv(sortie / "courbe_test.csv")
    assert set(r["strategie"]) == {"actuel", "optimise", "grille", "hasard"}
    assert r["rmse_vect"].notna().all()
    resume = json.loads((sortie / "os6.json").read_text(encoding="utf-8"))
    assert "nombre_recommande" in resume and "limites" in resume
    # reprise : un second passage relit placement et courbe sans recalcul
    assert m.main() == 0


def test_plan_repris_depuis_les_capteurs_installes(jeu, tmp_path, monkeypatch):
    """BF78 : avec des capteurs deja poses, le plan repart d'eux (pas recomptes ni reproposes)."""
    import pcfno3d.config as config_mod

    spec = importlib.util.spec_from_file_location("os6_script", REPO / "scripts" / "os6_reseau_aneo.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(config_mod.Config, "load", classmethod(lambda cls, _n: jeu))
    base = ["os6", "-c", "x", "--device", "cpu", "--contraintes", str(_contraintes(tmp_path)),
            "--n-val", "2", "--n-test", "2", "--par-etape", "2", "--tailles", "2"]
    monkeypatch.setattr(sys, "argv", base + ["--sortie", str(tmp_path / "a")])
    assert m.main() == 0
    premier = pd.read_csv(tmp_path / "a" / "plan_aneo.csv").iloc[0]
    pd.DataFrame([{"lat": premier["lat"], "lon": premier["lon"]}]).to_csv(tmp_path / "poses.csv", index=False)
    monkeypatch.setattr(sys, "argv", base + ["--sortie", str(tmp_path / "b"), "--installes", str(tmp_path / "poses.csv")])
    assert m.main() == 0
    p = pd.read_csv(tmp_path / "b" / "plan_aneo.csv")
    assert len(p) == 4                                         # budget entier pour les nouveaux
    assert not ((p["iy"] == premier["iy"]) & (p["ix"] == premier["ix"])).any()
