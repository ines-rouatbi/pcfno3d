"""OS5 : plusieurs domaines dans le meme service (Grand Tunis + national)."""
from types import SimpleNamespace

import pytest

from service import atom_outils


def _wf(nom, nx, dx, couche=False):
    geom = SimpleNamespace(nx=nx, ny=nx, dx=dx, dy=dx)
    cfg = SimpleNamespace(name=nom, inference=SimpleNamespace(couche_10m_oi=couche))
    return SimpleNamespace(geom=geom, cfg=cfg)


def test_modele_pour_choisit_le_plus_petit_domaine(monkeypatch):
    gt, nat = _wf("gt", 680, 100.0), _wf("nat", 828, 1000.0, couche=True)
    dedans = {id(gt): lambda lat, lon: 36.4 <= lat <= 36.9 and 10.0 <= lon <= 10.8,
              id(nat): lambda lat, lon: 30.1 <= lat <= 37.6 and 7.4 <= lon <= 11.7}

    def position(wf, lat=None, lon=None, lieu=None):
        if not dedans[id(wf)](lat, lon):
            raise ValueError("hors du domaine")
        return 0.0, 0.0, "x"
    monkeypatch.setattr(atom_outils, "position", position)
    atom_outils.enregistrer_domaines({"nat": nat, "gt": gt})
    try:
        assert atom_outils.modele_pour(lat=36.85, lon=10.23) is gt          # Tunis : la maille fine
        assert atom_outils.modele_pour(lat=34.74, lon=10.76) is nat         # Sfax
        assert atom_outils.modele_pour(lat=33.92, lon=8.13) is nat          # Tozeur
        assert atom_outils.modele_pour(lat=36.85, lon=10.23, domaine="nat") is nat
        with pytest.raises(ValueError):
            atom_outils.modele_pour(lat=45.0, lon=2.0)
        with pytest.raises(KeyError):
            atom_outils.modele_pour(domaine="inconnu")
    finally:
        atom_outils.enregistrer_domaines({})
    assert atom_outils.modele_pour(lat=1.0, lon=1.0, defaut=gt) is gt     # un seul domaine : le principal


def test_source_du_vent_selon_la_hauteur():
    nat = _wf("nat", 828, 1000.0, couche=True)
    assert "corrigee par les aeroports" in atom_outils.source_vent(nat, 10.0)
    assert atom_outils.source_vent(nat, 10.0, temps_reel=True).startswith("GFS")
    assert atom_outils.source_vent(nat, 100.0).startswith("modele national")
    assert atom_outils.source_vent(_wf("gt", 680, 100.0), 10.0).startswith("modele Grand Tunis")


from test_webapp_and_mcp import client, env, trained  # noqa: E402,F401  (fixtures)


def test_routes_domaines_un_seul_domaine(client):  # noqa: F811
    r = client.get("/api/atom-wind/domaines")
    assert r.status_code == 200, r.text
    d = r.json()["domaines"]
    assert len(d) == 1 and d[0]["principal"] is True
    assert client.get("/api/atom-wind/grille?domaine=inconnu").status_code == 404
    assert client.get("/api/atom-wind/jours").status_code == 200


def test_onglet_aneo_lit_le_plan_os4(tmp_path, cfg, geom):
    import json

    import pandas as pd

    from pcfno3d import atom_wind
    run = tmp_path / "runs" / "national"
    run.mkdir(parents=True)
    d = tmp_path / atom_wind.NOM_OS4
    d.mkdir()
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [10.0 + k / 10, 35.0]},
              "properties": {"id": f"ANEO-{k:03d}", "priorite": k, "phase": 1 if k <= 10 else 2, "lat": 35.0,
                             "lon": 10.0 + k / 10, "gain_marginal": 0.1 / k, "hauteur_m": 25.0}} for k in range(1, 13)]
    (d / "plan_glouton.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    lignes = [{"strategie": "sans_reseau", "N": 0, "pire_tranche_0_1km": 1.8, "rmse_vect_10m": 2.0, "gain": 0.0,
               "gain_ic95_bas": 0.0, "gain_ic95_haut": 0.0}]
    lignes += [{"strategie": s, "N": n, "pire_tranche_0_1km": 1.8 - n / 1000, "rmse_vect_10m": 2.0 - n / 500,
                "gain": n / 1800, "gain_ic95_bas": n / 2000, "gain_ic95_haut": n / 1600}
               for s in ("glouton", "grille_reguliere") for n in (10, 25)]
    pd.DataFrame(lignes).to_csv(d / "courbe.csv", index=False)
    (d / "bilan.json").write_text(json.dumps({"sans_reseau": {"pire_tranche_0_1km": 1.8},
                                              "N_minimal": {"glouton": {"N_minimal_ic95": "non atteint jusqu'a N = 400"}}}))
    assert atom_wind.plan_aneo(run) == d / "plan_glouton.geojson"
    r = atom_wind.reseau_aneo(run, cfg, geom, None)
    assert r["demonstration"] is False and r["bandeau"].startswith("Plan OS4")
    assert len(r["capteurs"]) == 12 and r["capteurs"][0]["id"] == "ANEO-001" and r["capteurs"][0]["gain_marginal"] == 0.1
    assert r["courbe"]["actuel"]["pire_rmse_niveau_m_s"] == 1.8 and "optimise" in r["courbe"]["strategies"]
    assert abs(r["seuil_dso1_m_s"] - 1.62) < 1e-9 and r["nombre_recommande"] is None


def test_annoter_mode_date_statut_par_heure():
    """Chaque heure servie dit son mode, sa date et son statut ; hors du Grand Tunis le temps reel est
    « experimental » tant que regle_national_gfs_v4 n'est pas passee (demande d'Ines du 16/11)."""
    nat = _wf("nat", 828, 1000.0, couche=True)
    nat.cfg.paths = SimpleNamespace(run_dir="runs/national_1km_v4_Cp_s44")
    p = {"date": "2026-11-16", "instants": [
        {"heure_locale": 9.0, "source": "temps reel", "timestamp_utc": "2026-11-16T08:00:00Z"},
        {"heure_locale": 15.0, "source": "prevision", "timestamp_utc": "2026-11-16T14:00:00Z"}]}
    atom_outils.annoter(nat, p, j=0)
    assert [i["mode"] for i in p["instants"]] == ["temps reel", "prevision"]
    assert all(i["statut"] == "experimental" for i in p["instants"])
    assert p["instants"][0]["date_donnee"] == "2026-11-16T08:00:00Z"
    assert p["statut"] == "experimental" and p["statut_couche_10m"] == "experimental"
    a = {"date": "2022-07-01", "instants": [{"heure_locale": 13.0, "timestamp_utc": "2022-07-01T12:00:00Z"}]}
    atom_outils.annoter(nat, a, analyse=True)
    assert a["mode"] == "analyse" and a["statut"] == "valide" and a["statut_couche_10m"] == "valide"
    gt = _wf("gt", 680, 100.0)
    gt.cfg.paths = SimpleNamespace(run_dir="runs/grand_tunis_v9_production")
    g = {"date": "2026-11-17", "instants": [{"heure_locale": 9.0, "source": "prevision"}]}
    assert atom_outils.annoter(gt, g, j=1)["statut"] == "experimental"      # v4 nourri par GFS
    gt.cfg.paths = SimpleNamespace(run_dir="runs/grand_tunis_v5_gfs")
    g = {"date": "2026-11-17", "instants": [{"heure_locale": 9.0, "source": "prevision"}]}
    assert atom_outils.annoter(gt, g, j=1)["statut"] == "valide"            # v5 : PREV1 conforme


def test_source_10m_en_prevision_dit_gfs_brut():
    nat = _wf("nat", 828, 1000.0, couche=True)
    assert "GFS brut" in atom_outils.source_vent(nat, 10.0, temps_reel=True, mode="prevision")


def test_toutes_les_routes_et_outils_choisissent_le_domaine(monkeypatch):
    """Agriculture, éolien, outils, alertes, historique… : chaque route et chaque outil MCP prend le modèle du
    domaine demandé (?domaine=national), sinon celui qui contient la position, sinon le principal."""
    from service import app as appmod, mcp_server
    gt, nat = _wf("gt", 680, 100.0), _wf("national", 828, 1000.0, couche=True)
    dedans = {id(gt): lambda lat, lon: 36.4 <= lat <= 36.9 and 10.0 <= lon <= 10.8,
              id(nat): lambda lat, lon: 30.1 <= lat <= 37.6 and 7.4 <= lon <= 11.7}

    def position(wf, lat=None, lon=None, lieu=None):
        if lieu:
            lat, lon = {"sfax": (34.72, 10.69), "tunis": (36.80, 10.18)}[lieu.lower()]
        if not dedans[id(wf)](lat, lon):
            raise ValueError("hors du domaine")
        return 0.0, 0.0, "x"
    monkeypatch.setattr(atom_outils, "position", position)
    monkeypatch.setattr(appmod, "model", lambda: gt)
    monkeypatch.setattr(mcp_server, "get_model", lambda: gt)
    atom_outils.enregistrer_domaines({"gt": gt, "national": nat})
    try:
        for choisir in (appmod.modele_requete, lambda d=None, la=None, lo=None, li=None: mcp_server._modele_requete(la, lo, li, d)):
            assert choisir() is gt                                   # rien : le principal
            assert choisir("national") is nat                        # ?domaine=national
            assert choisir(None, 34.74, 10.76) is nat                # Sfax : seul le national la contient
            assert choisir(None, 36.85, 10.23) is gt                 # Tunis : la maille fine
            assert choisir(None, None, None, "Sfax") is nat          # un lieu
            assert choisir(None, 45.0, 2.0) is gt                    # hors de tout : le principal le dira
    finally:
        atom_outils.enregistrer_domaines({})
    # chaque route ATOM Wind qui sert du vent accepte ?domaine=
    import inspect
    for nom in ("atom_agricole", "atom_production", "atom_maintenance", "atom_variations", "atom_grue", "atom_haies",
                "atom_porteurs", "atom_dimensionnement", "atom_explication", "atom_alertes", "atom_historique",
                "atom_qualite", "atom_ensemble", "atom_scenario", "atom_valeur_stations", "atom_rapport", "atom_atlas"):
        assert "domaine" in inspect.signature(getattr(appmod, nom)).parameters, nom


def test_les_24_gouvernorats_sont_dans_le_domaine_national():
    """Les chefs-lieux des 24 gouvernorats (lieux de l'application en national) tombent tous dans l'emprise du
    domaine national (30.15-37.615 N, 7.45-11.644 E), et un nom du Grand Tunis garde son lieu du Grand Tunis."""
    from service import atom_outils
    G = atom_outils.GOUVERNORATS
    assert len(G) == 24 and {"sfax", "gafsa", "tozeur", "bizerte", "tataouine", "le kef"} <= set(G)
    assert all(30.15 <= la <= 37.615 and 7.45 <= lo <= 11.644 for la, lo in G.values())
    assert atom_outils.LIEUX["nabeul"] == (36.455, 10.715)


def test_onglet_aneo_lit_la_regle_os4_v4_3(tmp_path, cfg, geom):
    """regle_os4_oi_v4_3 : prioritaire sur v4_2 une fois jugee ; plan en grille ; couche 0-100 m ; 0-1 km en information."""
    import json

    import pandas as pd

    from pcfno3d import atom_wind
    run = tmp_path / "runs" / "national"
    run.mkdir(parents=True)
    v42 = tmp_path / atom_wind.NOM_OS4
    v42.mkdir()
    pt = lambda k: {"type": "Feature", "geometry": {"type": "Point", "coordinates": [10.0 + k / 10, 35.0]},
                    "properties": {"id": f"ANEO-{k:03d}", "priorite": k, "phase": 1, "lat": 35.0, "lon": 10.0 + k / 10,
                                   "hauteur_m": 25.0}}
    (v42 / "plan_glouton.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [pt(1)]}))
    d = tmp_path / atom_wind.NOM_OS4_V43
    d.mkdir()
    (d / "plan_grille.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [pt(k) for k in range(1, 6)]}))
    assert atom_wind.dossier_os4(run) == v42                     # pas encore de bilan.json : la v4_2 reste servie

    def ligne(s, n, g):
        r0 = 1.8
        return {"strategie": s, "N": n, "rmse_vect_0_100m": r0 * (1 - g), "rmse_vect_10m": 2.0 * (1 - g),
                "rmse_vect_0_1km": 1.7 * (1 - g / 5), "pire_tranche_0_1km": 1.81 * (1 - g / 6),
                "gain_rmse_vect_0_100m": g, "gain_rmse_vect_0_100m_ic95_bas": g - 0.02,
                "gain_rmse_vect_0_100m_ic95_haut": g + 0.02, "gain_rmse_vect_0_1km": g / 5}
    lignes = [ligne("sans_reseau", 0, 0.0)] + [ligne("grille_reguliere", n, n / 400) for n in (10, 50)] \
        + [ligne("glouton", n, n / 500) for n in (10, 50)] + [ligne(f"aleatoire_{k}", 50, 0.10 + k / 100) for k in (1, 2, 3)]
    pd.DataFrame(lignes).to_csv(d / "courbe.csv", index=False)
    (d / "bilan.json").write_text(json.dumps({
        "sans_reseau": {"rmse_vect_0_100m": 1.8}, "verdict_N50_0_100m": {"grille_reguliere": {"verdict": "CONFORME"}},
        "N_minimal": {"grille_reguliere": {"N_minimal_ic95_0_100m": 50}}}))
    assert atom_wind.dossier_os4(run) == d and atom_wind.plan_aneo(run) == d / "plan_grille.geojson"
    r = atom_wind.reseau_aneo(run, cfg, geom, None)
    c = r["courbe"]
    assert c["cle"] == "rmse_0_100m_m_s" and c["couche"] == "0-100 m" and c["strategie_plan"] == "grille"
    assert c["actuel"]["rmse_0_100m_m_s"] == 1.8 and set(c["strategies"]) == {"grille", "optimise", "hasard"}
    g50 = next(p for p in c["strategies"]["grille"] if p["n"] == 50)
    assert g50["seuil_respecte"] is True and abs(g50["gain"] - 0.125) < 1e-9 and g50["rmse_0_1km_m_s"] < 1.7
    h = c["strategies"]["hasard"][0]
    assert h["tirages"] == 3 and abs(h["gain"] - 0.12) < 1e-9 and h["gain_ic95"] is None
    assert len(r["capteurs"]) == 5 and abs(r["seuil_dso1_m_s"] - 1.62) < 1e-9 and r["nombre_recommande"] == 50
    assert r["os4"]["libelle_hauteur"] == "Mât ANEO · anémomètre à 25 m" and r["os4"]["couche_jugee"] == "0-100 m"
    assert "0-1 km" in r["os4"]["information"]
