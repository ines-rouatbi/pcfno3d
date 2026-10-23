"""scripts/valider_grand_tunis_reel.py : entree avec les vraies mesures, vent a 10 m, bilans."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("valider", REPO / "scripts" / "valider_grand_tunis_reel.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def test_entree_reelle_mesures_era5_et_station_exclue(geom, monkeypatch):
    monkeypatch.setattr(geom.cfg.mask, "pas_era5_m", 3 * geom.dx)   # reseau tous les 3 noeuds
    S = np.random.default_rng(0).normal(size=(8,) + geom.shape[1:]).astype(np.float32)
    E = np.full((2,) + geom.shape[1:], 7.0, np.float32)
    k = max(int(round(geom.cfg.mask.pas_era5_m / geom.dx)), 1)
    hors = (k // 2 + 1, k // 2 + 1)          # cellule hors du reseau
    sur = (k // 2, k // 2)                                          # cellule du reseau
    cellules = {"DTTA": hors, "INM60729": sur}
    mesures = {"DTTA": (1.0, 2.0), "INM60729": (3.0, 4.0)}

    Si, cap = V.entree_reelle(S, E, geom, cellules, mesures)
    assert cap[hors] and cap[sur]
    assert tuple(Si[:2, hors[0], hors[1]]) == (1.0, 2.0)
    assert tuple(Si[:2, sur[0], sur[1]]) == (3.0, 4.0)              # la station prime
    assert (Si[2] == 0).all()

    Si, cap = V.entree_reelle(S, E, geom, cellules, mesures, exclue="DTTA")
    assert not cap[hors]                                            # hors reseau : sans capteur
    Si, cap = V.entree_reelle(S, E, geom, cellules, mesures, exclue="INM60729")
    assert cap[sur] and tuple(Si[:2, sur[0], sur[1]]) == (7.0, 7.0)  # sur le reseau : ERA5


def test_vent_a_10m_interpole_ou_profil_log():
    h = np.array([-20.0, 5.0, 30.0, 80.0])
    u = np.array([np.nan, 2.0, 4.0, 5.0])
    v = np.array([np.nan, 0.0, 0.0, 0.0])
    um, vm = V.vent_a_hauteur(u, v, h, z0=0.1)
    assert um == pytest.approx(2.0 + 2.0 * (10 - 5) / 25)           # entre 5 et 30 m
    um, _ = V.vent_a_hauteur(u[2:], v[2:], h[2:], z0=0.1)           # plus bas noeud a 30 m
    assert um == pytest.approx(4.0 * math.log(100) / math.log(300))
    assert math.isnan(V.vent_a_hauteur(np.full(3, np.nan), np.full(3, np.nan), h[:3], 0.1)[0])


def test_bilan_stations_parfait_et_direction():
    p = pd.DataFrame({"station": "X", "u_obs": [3.0, 0.0], "v_obs": [0.0, -3.0],
                      "u_parfait": [3.0, 0.0], "v_parfait": [0.0, -3.0],
                      "u_tourne": [0.0, 3.0], "v_tourne": [3.0, 0.0]})
    b = V.bilan_stations(p, ("parfait", "tourne")).set_index("source")
    assert b.loc["parfait", "rmse_vect"] == 0 and b.loc["parfait", "mae_direction_deg"] == 0
    assert b.loc["tourne", "mae_direction_deg"] == pytest.approx(90.0)


def test_observations_mois_en_panne_ecartes(tmp_path):
    t = pd.date_range("2021-06-01", periods=120, freq="12h")
    lignes = [{"station_id": "INM60728", "timestamp_utc": x.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "wind_speed_m_s": 0.0 if x.month == 6 else 3.0, "wind_dir_deg": 90.0,
               "wind_u_m_s": np.nan, "wind_v_m_s": np.nan} for x in t]
    pd.DataFrame(lignes).to_csv(tmp_path / "synop_2021.csv", index=False)
    obs = V.lire_observations(tmp_path)
    assert (obs["t"].dt.month != 6).all() and len(obs) > 0
    assert obs["wind_u_m_s"].iloc[0] == pytest.approx(-3.0)        # vent d'est : u < 0
    m = V.obs_a(obs, pd.Timestamp("2021-07-05 00:10"))
    assert m["INM60728"][0] == pytest.approx(-3.0)


def test_robustesse_dso5_retire_des_points_en_mode_reel(geom, monkeypatch, tmp_path):
    """era5_stations : keep < 1 retire des pseudo-capteurs ERA5 et des stations en evaluation ;
    keep = 1 donne exactement l'entree complete (DSO1 inchange) ; sans effet a l'entrainement."""
    from pcfno3d.dataset import entree_era5_stations

    monkeypatch.setattr(geom.cfg.mask, "pas_era5_m", geom.dx)          # un point par noeud
    monkeypatch.setattr(geom, "_cellules_stations", np.array([[1, 1], [2, 3]]), raising=False)
    S = np.zeros((8,) + geom.shape[1:], np.float32)
    f = tmp_path / "x_E.npy"
    np.save(f, np.ones((2,) + geom.shape[1:], np.float16))
    plein = entree_era5_stations(S, f, geom, np.random.default_rng(0), train=False)[1]
    idem = entree_era5_stations(S, f, geom, np.random.default_rng(0), train=False, keep=1.0)[1]
    assert (plein == idem).all() and plein.all()
    quart = entree_era5_stations(S, f, geom, np.random.default_rng(0), train=False, keep=0.25)[1]
    assert 0.1 < quart.mean() < 0.4


def test_poids_10m_identiques_a_l_interpolation(geom):
    """La combinaison lineaire de l'affinage donne le meme vent a 10 m que la validation."""
    import torch

    from pcfno3d.stations_reelles import perte_station, poids_10m, vent_a_hauteur

    rng = np.random.default_rng(1)
    u = rng.normal(size=geom.shape).astype(np.float32)
    jj, ii = np.nonzero(geom.valid.any(axis=0))
    h = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    for iy, ix in list(zip(jj, ii))[:: max(len(jj) // 15, 1)]:
        w = poids_10m(geom, iy, ix)
        col = np.where(geom.valid[:, iy, ix], u[:, iy, ix], np.nan)
        ref, _ = V.vent_a_hauteur(col, col, h[:, iy, ix], geom.z0[iy, ix])
        if w is None:
            assert math.isnan(ref)
            continue
        assert w[2] * u[w[0], iy, ix] + w[3] * u[w[1], iy, ix] == pytest.approx(ref, rel=1e-5, abs=1e-5)

    # perte nulle si la prediction (en m/s) vaut la mesure ; derivable
    class N:
        y_mu = torch.zeros(3)
        y_sd = torch.ones(3)
    iy, ix = int(jj[0]), int(ii[0])
    ka, kb, wa, wb = poids_10m(geom, iy, ix)
    pred = torch.zeros((1, 3) + geom.shape, requires_grad=True)
    cible = torch.tensor([[ka, kb, wa, wb, iy, ix, 0.0, 0.0, 1.0, 1.0]])
    assert float(perte_station(pred, cible, N)) == 0.0
    cible[0, 6] = 2.0
    l = perte_station(pred, cible, N)
    l.backward()
    assert float(l) > 0 and float(pred.grad.abs().sum()) > 0

    # direction seule (pv = 0) : bonne direction, vitesse double -> aucune erreur ; avec la
    # vitesse (pv = 1) -> erreur
    with torch.no_grad():
        p = torch.zeros((1, 3) + geom.shape)
        p[0, 0] = 4.0 / max(wa + wb, 1e-6)          # u predit a 10 m = 4 m/s partout
    cible = torch.tensor([[ka, kb, wa, wb, iy, ix, 2.0, 0.0, 1.0, 0.0]])
    assert float(perte_station(p, cible, N)) == pytest.approx(0.0, abs=1e-6)
    cible[0, 9] = 1.0
    assert float(perte_station(p, cible, N)) > 0.5
    cible[0, 9] = 0.0
    cible[0, 6], cible[0, 7] = 0.0, 2.0              # direction fausse (90 deg) -> erreur
    assert float(perte_station(p, cible, N)) > 0.5


def test_champ_agl_identique_a_poids_10m(geom):
    """Carte a 10 m de scripts/comparer_modeles_grand_tunis.py = combinaison de l'affinage."""
    from pcfno3d.stations_reelles import poids_10m

    spec = importlib.util.spec_from_file_location("comparer", REPO / "scripts" / "comparer_modeles_grand_tunis.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    u = np.where(geom.valid, np.random.default_rng(2).normal(size=geom.shape), np.nan)
    carte, _ = C.champ_agl(u, u, geom, 10.0)
    for iy in range(geom.ny):
        for ix in range(geom.nx):
            w = poids_10m(geom, iy, ix)
            if w is None:
                assert np.isnan(carte[iy, ix])
            else:
                assert carte[iy, ix] == pytest.approx(w[2] * u[w[0], iy, ix] + w[3] * u[w[1], iy, ix],
                                                      rel=1e-6, abs=1e-6)


def test_bilan_plis_et_regle_de_livraison(tmp_path):
    spec = importlib.util.spec_from_file_location("plis", REPO / "scripts" / "bilan_plis_grand_tunis.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)

    def ecrire(run, valeurs):
        d = tmp_path / run / "validation_reelle" / "test"
        d.mkdir(parents=True)
        lignes = [{"station": s, "source": src, "rmse_vect": x, "biais_vitesse": 0.0}
                  for s, par_src in valeurs.items() for src, x in par_src.items()]
        pd.DataFrame(lignes).to_csv(d / "stations_bilan.csv", index=False)

    stations = ("Tunis-Carthage", "Zaghouan", "Nabeul")
    ecrire(B.REFERENCE, {s: {"modele": 2.5, "era5": 2.6} for s in stations})
    with pytest.raises(FileNotFoundError):
        B.bilan(tmp_path / "rien")
    t, d = B.bilan(tmp_path)
    assert not d["plis_complets"]
    for pli, s in B.PLIS.items():
        ecrire(pli, {s: {"modele": 2.0 if s != "Nabeul" else 2.7}})
    t, d = B.bilan(tmp_path)
    assert d["plis_complets"] and d["stations_en_progres"] == 2 and d["gain"]
    assert d["modele_livre"].startswith("v3")
    # gains dans le bruit (-0,4 % et -1,5 %) : pas de gain, la v2 reste livree
    for pli, s in B.PLIS.items():
        ecrire_bis = tmp_path / pli / "validation_reelle" / "test" / "stations_bilan.csv"
        ecrire_bis.unlink()
        (tmp_path / pli / "validation_reelle" / "test").rmdir()
        (tmp_path / pli / "validation_reelle").rmdir()
        (tmp_path / pli).rmdir()
        ecrire(pli, {s: {"modele": {"Tunis-Carthage": 2.46, "Zaghouan": 2.49, "Nabeul": 2.8}[s]}})
    t, d = B.bilan(tmp_path)
    assert not d["gain"] and d["modele_livre"].startswith("v2")
