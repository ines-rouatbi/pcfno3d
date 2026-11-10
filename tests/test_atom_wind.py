"""Front-end ATOM Wind branche sur l'API : routes /api/atom-wind/*, champ AGL compact,
page construite depuis la maquette (scripts/build_atom_wind.py)."""

from __future__ import annotations

import base64
import importlib.util
import json
import math
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


def test_blocs_selon_le_domaine_servi(monkeypatch):
    """Grand Tunis : blocs de 500 m (inchanges) ; national 828 x 388 km a 1 km : blocs de 4 km (97 x 207)."""
    from types import SimpleNamespace

    from pcfno3d import atom_wind

    monkeypatch.delenv("PCFNO3D_ATOM_PAS_M", raising=False)
    gt = SimpleNamespace(nx=680, ny=560, dx=100.0, dy=100.0)
    nat = SimpleNamespace(nx=388, ny=828, dx=1000.0, dy=1000.0)
    assert atom_wind.facteur(gt) == 5
    b = atom_wind.facteur(nat)
    assert b == 4 and -(-nat.nx // b) * -(-nat.ny // b) == 97 * 207 <= atom_wind.MAX_BLOCS
    monkeypatch.setenv("PCFNO3D_ATOM_PAS_M", "2000")
    assert atom_wind.facteur(nat) == 2


def test_plan_grille_projection():
    """PlanGrille = la projection de scripts/grille_reelle.py ; ecart a GeoRef < 0,15 km au Grand Tunis,
    ~15 km a l'est du domaine national."""
    from types import SimpleNamespace

    from pcfno3d.realdata import GeoRef, PlanGrille

    def plan(lat0, lon0, ny_s, dy_s):
        cfg = SimpleNamespace(real=SimpleNamespace(origin_lat=lat0, origin_lon=lon0, rotation_deg=0.0))
        return PlanGrille.from_geom(cfg, SimpleNamespace(ny_s=ny_s, dy_s=dy_s))

    gt = plan(36.395, 10.04, 1120, 50.0)
    assert abs(gt.m_lon / 1000 - 89.42) < 0.01 and abs(gt.m_lat / 1000 - 110.97) < 0.01   # constantes de la page
    x, y = gt.to_xy(*gt.to_latlon(12_345.0, 6_789.0))
    assert abs(x - 12_345.0) < 1e-6 and abs(y - 6_789.0) < 1e-6
    la, lo = gt.to_latlon(68_000.0, 56_000.0)
    xg, yg = GeoRef(36.395, 10.04).to_xy(la, lo)
    assert math.hypot(xg - 68_000.0, yg - 56_000.0) < 150.0
    nat = plan(30.15, 7.45, 1656, 500.0)
    la, lo = nat.to_latlon(388_000.0, 0.0)
    xg, _ = GeoRef(30.15, 7.45).to_xy(la, lo)
    assert 10_000.0 < xg - 388_000.0 < 20_000.0


def test_grille_decrit_le_domaine(client):  # noqa: F811
    g = client.get("/api/atom-wind/grille").json()
    assert g["domaine"]["etendue"] == "local" and g["noeuds_par_bloc"] >= 1
    assert g["pas_km"] == pytest.approx(g["pas_noeuds_m"] * g["noeuds_par_bloc"] / 1000)
    for s in g["stations"]:
        assert {"nom", "code", "x_km", "y_km"} <= set(s)
    if g["projection"] is not None:
        assert g["projection"]["km_par_deg_lon"] > 0 and g["emprise"]["lat_max"] > g["emprise"]["lat_min"]


def _os6(d: Path) -> None:
    """Sorties OS6 au format de scripts/os6_reseau_aneo.py (4 capteurs, courbe a 2 et 4 capteurs)."""
    d.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame([{"id": f"ANEO-{k:03d}", "ordre": k, "etape": (k + 1) // 2, "iy": 2 + k, "ix": 3 + 2 * k,
                   "score_erreur_m2_s2": 1.0, "gain_rmse_validation_m_s": 0.01 * k,
                   "rmse_validation_apres_m_s": 0.6 - 0.01 * k, "x_m": (3 + 2 * k) * 100.0, "y_m": (2 + k) * 100.0,
                   "lat": 36.40 + 0.001 * k, "lon": 10.05 + 0.002 * k, "altitude_sol_m": 20.0 + k,
                   "hauteur_mat_m": 10.0, "gain_cumule_m_s": 0.01 * k * (k + 1) / 2} for k in range(1, 5)]
                 ).to_csv(d / "plan_aneo.csv", index=False)
    lignes = [("actuel", 0, 0.53, 0.55)]
    for s, dec in (("optimise", 0.03), ("grille", 0.01), ("hasard", 0.005)):
        lignes += [(s, 2, 0.53 - dec, 0.55 - dec), (s, 4, 0.53 - 2 * dec, 0.55 - 2 * dec)]
    t = pd.DataFrame(lignes, columns=["strategie", "n_capteurs", "rmse_vect", "pire_rmse_niveau"])
    t["seuil_respecte"] = t["pire_rmse_niveau"] <= 0.5
    t.to_csv(d / "courbe_test.csv", index=False)
    (d / "os6.json").write_text(json.dumps({"modele": "best.pt", "configuration": "test", "seuil_rmse_m_s": 0.5,
                                            "nombre_recommande": None,
                                            "reseau_actuel": {"rmse_vect": 0.53, "pire_rmse_niveau": 0.55},
                                            "limites": "OSSE"}), encoding="utf-8")


def test_onglet_reseau_aneo(client, env):  # noqa: F811
    """Onglet admin « Reseau ANEO » : plan d'exemple signale sans OS6 ; avec les sorties OS6, phases aux paliers
    de la courbe, cout a renseigner puis fourni, statut installe ecrit dans os6_installes.csv et commande
    --installes rendue (jamais lancee) ; portee admin."""
    import shutil

    from service.auth import ApiKeyStore, default_store_path

    admin, _ = ApiKeyStore(default_store_path(env.paths.run_dir)).create("tests-admin-reseau", scope="admin")
    A = {"X-API-Key": admin}
    run = env.paths.run_dir
    d = run / "os6_reseau_aneo"
    poses, param = run / "os6_installes.csv", run / "os6_parametres.json"
    for f in (poses, param):
        f.unlink(missing_ok=True)
    shutil.rmtree(d, ignore_errors=True)
    try:
        # lecture seule : la vue complete est reservee a l'administrateur
        assert client.get("/api/atom-wind/reseau-aneo", params={"detail": "true"}).status_code == 403
        r = client.get("/api/atom-wind/reseau-aneo", params={"detail": "true"}, headers=A)
        assert r.status_code == 200, r.text
        v = r.json()
        assert v["demonstration"] is True and v["bandeau"] == "Démonstration : plan d'exemple"
        assert v["courbe"] is None and all(c["gain_m_s"] is None for c in v["capteurs"])
        assert v["cout"]["unitaire"] is None and "à renseigner" in v["cout"]["note"]
        assert client.post("/api/atom-wind/reseau-aneo/installes", json={"installes": []},
                           headers=A).status_code == 409

        _os6(d)
        v = client.get("/api/atom-wind/reseau-aneo", params={"detail": "true"}, headers=A).json()
        assert v["demonstration"] is False and [c["id"] for c in v["capteurs"]] == [f"ANEO-{k:03d}" for k in range(1, 5)]
        assert [(p["de"], p["a"]) for p in v["phases"]] == [(1, 2), (3, 4)]
        assert [c["phase"] for c in v["capteurs"]] == [1, 1, 2, 2]
        assert v["phases"][0]["pire_rmse_niveau_m_s"] == pytest.approx(0.52)
        assert v["phases"][0]["gain_pire_niveau_m_s"] == pytest.approx(0.03)
        assert v["phases"][1]["gain_pire_niveau_m_s"] == pytest.approx(0.03)
        assert v["phases"][0]["seuil_dso1_respecte"] is False and v["phases"][1]["seuil_dso1_respecte"] is True
        assert v["phases"][0]["cout"] is None
        assert set(v["courbe"]["strategies"]) == {"optimise", "grille", "hasard"}
        assert v["courbe"]["actuel"]["pire_rmse_niveau_m_s"] == pytest.approx(0.55)
        assert v["capteurs"][0]["x_km"] == pytest.approx(0.5) and v["seuil_dso1_m_s"] == 0.5
        assert v["relance"]["lancee_par_le_service"] is False and "--installes" not in v["relance"]["commande"]

        # cout unitaire fourni par l'exploitant
        param.write_text(json.dumps({"cout_unitaire": 1200, "devise": "TND"}), encoding="utf-8")
        v = client.get("/api/atom-wind/reseau-aneo", params={"detail": "true"}, headers=A).json()
        assert v["cout"]["unitaire"] == 1200 and v["phases"][1]["cout_cumule"] == 4800

        # statut installe : fichier ecrit, commande rendue
        assert client.post("/api/atom-wind/reseau-aneo/installes", json={"installes": ["ANEO-002"]}).status_code == 403
        assert client.post("/api/atom-wind/reseau-aneo/installes", json={"installes": ["ANEO-999"]},
                           headers=A).status_code == 422
        r = client.post("/api/atom-wind/reseau-aneo/installes", json={"installes": ["ANEO-002"]}, headers=A)
        assert r.status_code == 200, r.text
        o = r.json()
        assert o["installes"] == 1 and Path(o["fichier"]) == poses and poses.exists()
        assert "--installes" in o["relance"]["commande"] and str(poses) in o["relance"]["commande"]
        assert "--tailles 2,4" in o["relance"]["commande"] and "os6_reseau_aneo_installes_" in o["relance"]["sortie"]
        v = client.get("/api/atom-wind/reseau-aneo", params={"detail": "true"}, headers=A).json()
        st = {c["id"]: c["statut"] for c in v["capteurs"]}
        assert st["ANEO-002"] == "installé" and st["ANEO-001"] == "prévu" and v["phases"][0]["installes"] == 1
        # exports : les fichiers OS6 tels quels (admin), noms connus seulement
        assert client.get("/api/atom-wind/reseau-aneo/fichier/plan_aneo.csv").status_code == 403
        r = client.get("/api/atom-wind/reseau-aneo/fichier/plan_aneo.csv", headers=A)
        assert r.status_code == 200 and r.text.startswith("id,ordre")
        assert client.get("/api/atom-wind/reseau-aneo/fichier/api_keys.json", headers=A).status_code == 404
        # la couche GeoJSON de la page reste en lecture (404 sans plan_aneo.geojson)
        assert client.get("/api/atom-wind/reseau-aneo").status_code == 404
    finally:
        for f in (poses, param):
            f.unlink(missing_ok=True)
        shutil.rmtree(d, ignore_errors=True)


def test_installes_reconnus_par_coordonnees():
    """os6_installes.csv sans identifiant (format de --installes : lat, lon) : le capteur prevu le plus proche
    a moins de 1 km devient « installe », un capteur plus loin reste hors du plan ; phases sans courbe."""
    from pcfno3d import atom_wind
    from pcfno3d.realdata import PlanGrille

    geo = PlanGrille(36.395, 10.04, 89_420.0, 110_970.0, 36.65)
    capteurs = [{"id": f"ANEO-{k}", "x_km": 10.0 * k, "y_km": 5.0, "statut": "prévu"} for k in range(1, 4)]
    la, lo = geo.to_latlon(20_400.0, 5_300.0)                    # a 0,5 km de ANEO-2
    lb, lob = geo.to_latlon(45_000.0, 30_000.0)                  # loin de tout
    hors = atom_wind._associer_installes(capteurs, [
        {"id": None, "lat": float(la), "lon": float(lo), "installe_le": "2026-11-01"},
        {"id": None, "lat": float(lb), "lon": float(lob), "installe_le": None}], geo)
    assert [c["statut"] for c in capteurs] == ["prévu", "installé", "prévu"] and len(hors) == 1
    assert atom_wind.phases([10, 20, 30, 50], 36) == [(1, 10), (11, 20), (21, 30), (31, 36)]
    assert atom_wind.phases([2, 4], 4) == [(1, 2), (3, 4)] and atom_wind.phases([10], 0) == []


def test_page_construite_depuis_la_maquette(client):  # noqa: F811
    spec = importlib.util.spec_from_file_location("build_aw", REPO / "scripts" / "build_atom_wind.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    source = (REPO / "docs" / "atom_wind" / "atom_wind_maquette.html").read_text(encoding="utf-8")
    html = m.construire(source)
    servi = (REPO / "webapp" / "atom-wind" / "index.html").read_text(encoding="utf-8")
    assert 'id="root"' in servi and 'type="module"' in servi
    for a in ('src="atom-data.js"', 'id="atom-main"', "ATOM.appliquerGrille", "ATOM.vent(",
              "ATOM.planXY(xy)", "ATOM.aSigma()", 'id="aw-jour"', "ATOM.rafraichir",
              'src="atom-3d.js"', "AW3=ATOM.vue3d(", "AW3.particules()", "AW3.fiche(x,y)", "setEx:v=>{EX=v}",
              # accroche 21 : onglet d'administration « Reseau ANEO »
              'data-v="reseau"', 'id="aneo-reseau"', 'reseau:["admin"]', "ATOM.reseauAneo.afficher()",
              'src="reseau-aneo.js"'):
        assert a in html, a
    assert html.startswith("<!doctype html>")
    with pytest.raises(SystemExit, match="accroche"):
        m.construire(source.replace("function wind(x, y, z, h){", "function vent(x, y, z, h){"))
    runtime = (REPO / "frontend/src/engine/atom-runtime.js").read_text(encoding="utf-8")
    for hook in ("ATOM.vent(", "ATOM.aSigma()", "AW3=ATOM.vue3d(", "AW3.fiche(x,y)",
                 'reseau:["admin"]', "ATOM.reseauAneo.afficher()", "ATOM.domaine()"):
        assert hook in runtime, hook
    # onglet Reseau ANEO : vue React (point de montage), navigation, chargement du module, habillage isole
    vue = (REPO / "frontend/src/views/ViewReseauAneo.tsx").read_text(encoding="utf-8")
    assert 'id="v-reseau"' in vue and 'id="aneo-reseau"' in vue and "reseau-aneo.css" in vue
    assert "id: 'reseau'" in (REPO / "frontend/src/components.tsx").read_text(encoding="utf-8")
    assert "engine/reseau-aneo.js" in (REPO / "frontend/src/workspace.ts").read_text(encoding="utf-8")
    moteur = (REPO / "frontend/src/engine/reseau-aneo.js").read_text(encoding="utf-8")
    assert "/atom-wind/reseau-aneo?detail=true" in moteur and "A.reseauAneo" in moteur
    for f in ("i18n/ar.json", "i18n/en.json"):
        assert client.get(f"/app/atom-wind/{f}").status_code == 200, f
    r = client.get("/app/atom-wind/")
    assert r.status_code == 200 and 'id="root"' in r.text
    import re
    assets = re.findall(r'(?:src|href)="(/app/assets/[^\"]+)"', r.text)
    assert assets, "npm run build must produce served React/CSS assets"
    if not (REPO / "webapp" / "assets").is_dir():
        pytest.skip("bundles React non construits (webapp/assets/ est ignore par git) : npm ci && npm run build")
    for asset in assets:
        assert client.get(asset).status_code == 200, asset


def test_dictionnaires_de_traduction():
    """E17 / TB7 : la liste des textes suit la page, chaque dictionnaire couvre au moins 90 % des textes
    (le reste : du code que l'extraction ne sait pas ecarter) et garde les memes {n} que la cle."""
    import re
    spec = importlib.util.spec_from_file_location("extraire", REPO / "scripts" / "extraire_textes_atom_wind.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    web = REPO / "webapp" / "atom-wind"
    textes = m.textes_frontend()
    stocke = json.loads((web / "i18n" / "textes_fr.json").read_text(encoding="utf-8"))
    assert textes == stocke, "textes a traduire perimes : python scripts/extraire_textes_atom_wind.py"
    for langue in ("en", "ar"):
        d = json.loads((web / "i18n" / f"{langue}.json").read_text(encoding="utf-8"))
        for k, v in d.items():
            assert sorted(re.findall(r"\{\d+\}", k)) == sorted(re.findall(r"\{\d+\}", v)), (langue, k)
        assert sum(t in d for t in textes) >= 0.9 * len(textes), langue
    shell = (REPO / "frontend/src/components.tsx").read_text(encoding="utf-8")
    bootstrap = (REPO / "frontend/src/workspace.ts").read_text(encoding="utf-8")
    assert 'atom-i18n.js' in bootstrap and 'data-lang="ar"' in shell
