"""Atlas eolien (E8), export SIG (E15) : moments, Weibull, facteur de charge, sites, GeoTIFF, route."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from test_os6 import jeu  # noqa: F401  (fixture : mini-jeu reel entraine, georeference)
from test_forcage_era5 import cfg_era5, forcage  # noqa: F401


def test_weibull_par_les_moments():
    from pcfno3d.atlas import Cumul, weibull_moments

    rng = np.random.default_rng(0)
    k0, c0 = 2.0, 7.0
    cum = Cumul((1,))
    for v in c0 * rng.weibull(k0, 20000):
        cum.ajouter(np.array([v]))
    m = cum.moments()
    k, c = weibull_moments(m["moyenne"], m["ecart_type"])
    assert k[0] == pytest.approx(k0, rel=0.05) and c[0] == pytest.approx(c0, rel=0.03)


def test_facteur_de_charge_croit_avec_le_vent():
    from pathlib import Path

    from pcfno3d.atlas import charger_courbe, facteur_charge, table_aep
    courbe = charger_courbe(Path(__file__).resolve().parents[1] / "configs" / "eolien" / "parc_demo.yaml")
    t = table_aep(courbe, 1.2)
    fc = facteur_charge(np.array([2.0, 2.0, 2.0]), np.array([4.0, 7.0, 10.0]), t)
    assert (np.diff(fc) > 0).all() and 0 < fc[0] < fc[-1] < 1


def test_atlas_de_bout_en_bout(jeu, tmp_path):  # noqa: F811
    from PIL import Image

    from pcfno3d import atlas
    from pcfno3d.cache import load_index
    from pcfno3d.inference import WindFieldModel

    wf = WindFieldModel(jeu, device="cpu", verbose=False)
    stems = list(load_index(jeu)["stem"])[:6]
    a = atlas.construire(wf, stems, hub_agl_m=100.0, verbose=False)
    assert a["instants"] == 6 and np.isfinite(a["moyenne"]).mean() > 0.5
    fc = a["facteur_charge"]
    assert np.nanmin(fc) >= 0 and np.nanmax(fc) <= 1
    g = wf.geom
    for i, (iy, ix) in enumerate(a["sites"]):
        for jy, jx in a["sites"][:i]:
            assert math.hypot((iy - jy) * g.dy, (ix - jx) * g.dx) >= atlas.DISTANCE_SITES_M
    resume = atlas.ecrire(a, g, tmp_path / "atlas_eolien")
    assert (tmp_path / "atlas_eolien" / "atlas.png").exists()
    tif = tmp_path / "atlas_eolien" / "atlas_moyenne.tif"
    im = np.array(Image.open(tif))
    assert im.shape == g.zt.shape
    np.testing.assert_allclose(im[::-1][np.isfinite(a["moyenne"])], a["moyenne"][np.isfinite(a["moyenne"])], rtol=1e-6)
    assert all("lat" in s for s in resume["sites"])
    json.dumps(resume)


def test_route_atlas(jeu, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient

    from pcfno3d import atlas
    from pcfno3d.cache import load_index
    from pcfno3d.inference import WindFieldModel

    wf = WindFieldModel(jeu, device="cpu", verbose=False)
    a = atlas.construire(wf, list(load_index(jeu)["stem"])[:3], verbose=False)
    atlas.ecrire(a, wf.geom, jeu.paths.run_dir / "atlas_eolien")
    for var, val in (("PCFNO3D_DATA_DIR", jeu.paths.data_dir), ("PCFNO3D_CACHE_DIR", jeu.paths.cache_dir),
                     ("PCFNO3D_RUN_DIR", jeu.paths.run_dir), ("PCFNO3D_DEVICE", "cpu")):
        monkeypatch.setenv(var, str(val))
    for var in ("PCFNO3D_API_KEYS", "PCFNO3D_INSECURE_OPEN"):
        monkeypatch.delenv(var, raising=False)
    import pcfno3d.config as cm
    monkeypatch.setattr(cm.Config, "load", classmethod(lambda cls, _n: jeu))
    from service.auth import ApiKeyStore, default_store_path
    chemin = default_store_path(jeu.paths.run_dir)
    chemin.unlink(missing_ok=True)
    cle, _ = ApiKeyStore(chemin).create("tests-atlas", scope="read")
    from service.app import create_app
    with TestClient(create_app(), headers={"X-API-Key": cle}) as c:
        r = c.get("/api/atom-wind/atlas")
        assert r.status_code == 200, r.text
        d = r.json()
        assert "facteur_charge" in d and d["sites"] and "moyenne" in d["couches_geotiff"]
        assert c.get("/api/atom-wind/atlas/atlas.png").headers["content-type"] == "image/png"
        assert c.get("/api/atom-wind/atlas/atlas_moyenne.tif").status_code == 200
        assert c.get("/api/atom-wind/atlas/..%2Fsecret.tif").status_code == 404
        assert c.get("/api/atom-wind/atlas/atlas.json").status_code == 404
