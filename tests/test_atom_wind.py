"""Front-end ATOM Wind branche sur l'API : routes /api/atom-wind/*, champ AGL compact,
page construite depuis la maquette (scripts/build_atom_wind.py)."""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from test_webapp_and_mcp import client, env, trained  # noqa: F401  (fixtures)

REPO = Path(__file__).resolve().parents[1]


def _decoder(b64: str, forme) -> np.ndarray:
    a = np.frombuffer(base64.b64decode(b64), dtype="<i2").astype(float)
    a[a == -32768] = np.nan
    return (a / 100).reshape(forme)


def test_champ_agl_interpolation_profil_log_et_sommet(geom):
    from pcfno3d.atom_wind import champ_agl

    u = np.where(geom.valid, 1.0 + 0.01 * np.asarray(geom.z_levels, float)[:, None, None], np.nan)
    v = np.zeros_like(u)
    hk = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    # entre deux noeuds valides : interpolation lineaire en hauteur AGL (u lineaire en z)
    k = next(k for k in range(geom.nz - 1) if (geom.valid[k] & geom.valid[k + 1]).any())
    cols = geom.valid[k] & geom.valid[k + 1] & ~geom.valid[:k].any(axis=0) if k else \
        geom.valid[0] & geom.valid[1]
    h = float(np.median(((hk[k] + hk[k + 1]) / 2)[cols]))
    uo, _ = champ_agl(u, v, geom, h)
    m = cols & (hk[k] <= h) & (h <= hk[k + 1])
    assert m.any()
    np.testing.assert_allclose(uo[m], (1.0 + 0.01 * (geom.zt + h))[m], rtol=1e-6)
    # tres haut : valeur du dernier noeud valide, jamais NaN sur une colonne valide
    uo, _ = champ_agl(u, v, geom, 1e5)
    assert np.isfinite(uo[geom.valid.any(axis=0)]).all()
    # tres bas : profil log, plus faible que le premier noeud
    uo, _ = champ_agl(u, v, geom, 1.0)
    k0 = np.argmax(geom.valid, axis=0)
    u0 = np.take_along_axis(u, k0[None], 0)[0]
    ok = geom.valid.any(axis=0)
    assert (np.abs(uo[ok]) <= np.abs(u0[ok]) + 1e-9).all()


def test_routes_atom_wind(client, env):  # noqa: F811
    from pcfno3d import atom_wind
    from pcfno3d.geometry import Geometry

    g = client.get("/api/atom-wind/grille")
    assert g.status_code == 200, g.text
    grille = g.json()
    geom = Geometry(env)
    b = atom_wind.facteur(geom)
    assert grille["nx"] == -(-geom.nx // b) and grille["ny"] == -(-geom.ny // b)
    assert np.array(grille["terrain_m"]).shape == (grille["ny"], grille["nx"])
    assert grille["hauteurs_agl_m"] == list(atom_wind.HAUTEURS_AGL)

    jours = client.get("/api/atom-wind/jours").json()["jours"]
    assert jours and all(j["instants"] for j in jours)
    j = jours[0]
    hl = [i["heure_locale"] for i in j["instants"]]
    assert hl == sorted(hl)

    r = client.get("/api/atom-wind/jour", params={"date": j["date"]})
    assert r.status_code == 200, r.text
    d = r.json()
    a = _decoder(d["vent"], d["forme"])
    assert a.shape == (len(j["instants"]), len(atom_wind.HAUTEURS_AGL), 2, grille["ny"], grille["nx"])
    assert np.isfinite(a).mean() > 0.5
    # meme valeur que le calcul direct (premier instant, 100 m, u), au centimetre pres
    from pcfno3d.inference import WindFieldModel
    wf = WindFieldModel(env, device="cpu", verbose=False)
    vol = wf.predict_stem(j["instants"][0]["stem"])
    u100, _ = atom_wind.champ_agl(vol.u, vol.v, geom, 100.0)
    ref = atom_wind.blocs(u100, b)
    ih = atom_wind.HAUTEURS_AGL.index(100)
    m = np.isfinite(ref) & np.isfinite(a[0, ih, 0])
    np.testing.assert_allclose(a[0, ih, 0][m], ref[m], atol=0.006)

    assert client.get("/api/atom-wind/jour", params={"date": "1999-01-01"}).status_code == 404
    assert client.get("/api/atom-wind/jour", params={"date": "pas-une-date"}).status_code == 422


def test_reseau_aneo_et_cle(client, env):  # noqa: F811
    from fastapi.testclient import TestClient

    assert client.get("/api/atom-wind/reseau-aneo").status_code == 404
    d = env.paths.run_dir / "os6_reseau_aneo"
    d.mkdir(parents=True, exist_ok=True)
    plan = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": "ANEO-001", "type": "capteur ANEO propose", "ordre": 1},
         "geometry": {"type": "Point", "coordinates": [10.2, 36.6]}}]}
    (d / "plan_aneo.geojson").write_text(json.dumps(plan), encoding="utf-8")
    try:
        r = client.get("/api/atom-wind/reseau-aneo")
        assert r.status_code == 200 and r.json()["features"][0]["properties"]["id"] == "ANEO-001"
    finally:
        (d / "plan_aneo.geojson").unlink()
    # sans cle : refuse (le service est ferme par defaut)
    sans = TestClient(client.app)
    assert sans.get("/api/atom-wind/grille").status_code == 401


def test_page_construite_depuis_la_maquette(client):  # noqa: F811
    spec = importlib.util.spec_from_file_location("build_aw", REPO / "scripts" / "build_atom_wind.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    source = (REPO / "docs" / "atom_wind" / "atom_wind_maquette.html").read_text(encoding="utf-8")
    html = m.construire(source)
    servi = (REPO / "webapp" / "atom-wind" / "index.html").read_text(encoding="utf-8")
    assert html == servi, "webapp/atom-wind/index.html n'est pas a jour : python scripts/build_atom_wind.py"
    for a in ('src="atom-data.js"', 'id="atom-main"', "ATOM.appliquerGrille", "ATOM.vent(",
              "ATOM.planXY(xy)", "ATOM.aSigma()", 'id="aw-jour"', "ATOM.rafraichir",
              'src="atom-3d.js"', "AW3=ATOM.vue3d(", "AW3.particules()", "AW3.fiche(x,y)", "setEx:v=>{EX=v}"):
        assert a in html, a
    assert html.startswith("<!doctype html>")
    with pytest.raises(SystemExit, match="accroche"):
        m.construire(source.replace("function wind(x, y, z, h){", "function vent(x, y, z, h){"))
    r = client.get("/app/atom-wind/")
    assert r.status_code == 200 and "atom-data.js" in r.text
    assert client.get("/app/atom-wind/atom-data.js").status_code == 200
    js3d = client.get("/app/atom-wind/atom-3d.js")
    assert js3d.status_code == 200 and "A.vue3d" in js3d.text
    for f in ("atom-i18n.js", "i18n/ar.json", "i18n/en.json"):
        assert client.get(f"/app/atom-wind/{f}").status_code == 200, f


def test_dictionnaires_de_traduction():
    """E17 / TB7 : la liste des textes suit la page, chaque dictionnaire couvre au moins 90 % des textes
    (le reste : du code que l'extraction ne sait pas ecarter) et garde les memes {n} que la cle."""
    import re
    spec = importlib.util.spec_from_file_location("extraire", REPO / "scripts" / "extraire_textes_atom_wind.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    web = REPO / "webapp" / "atom-wind"
    textes = m.avec_supplements(m.extraire((web / "index.html").read_text(encoding="utf-8"),
                                           tuple((web / f).read_text(encoding="utf-8") for f in m.MODULES)))
    stocke = json.loads((web / "i18n" / "textes_fr.json").read_text(encoding="utf-8"))
    assert textes == stocke, "textes a traduire perimes : python scripts/extraire_textes_atom_wind.py"
    for langue in ("en", "ar"):
        d = json.loads((web / "i18n" / f"{langue}.json").read_text(encoding="utf-8"))
        for k, v in d.items():
            assert sorted(re.findall(r"\{\d+\}", k)) == sorted(re.findall(r"\{\d+\}", v)), (langue, k)
        assert sum(t in d for t in textes) >= 0.9 * len(textes), langue
    html = (web / "index.html").read_text(encoding="utf-8")
    assert 'src="atom-i18n.js"' in html and 'data-lang="ar"' in html and 'disabled title="Prévu"' not in html
