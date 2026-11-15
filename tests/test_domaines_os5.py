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
