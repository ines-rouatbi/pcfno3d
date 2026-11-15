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
