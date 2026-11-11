"""scripts/mos_era5.py : regle hachee, paires, validation croisee par site, garde-fou de la validation,
test lu une seule fois. Donnees synthetiques, aucun acces reseau."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mos_era5", REPO / "scripts" / "mos_era5.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_regle_hachee_et_lue():
    r, h = M.regle()
    assert len(h) == 64 and "accord d'Ines" in r["statut"]
    assert r["decoupage"] == {**r["decoupage"], "entrainement": [2015, 2016, 2017, 2018, 2019, 2020],
                              "validation": [2021], "test": [2022]}
    g = M.grille(r)
    assert len(g) == 5 + 4 and len({M.nom_config(c) for c in g}) == len(g)
    simple = sorted(g, key=M.cle_simplicite)
    assert simple[0] == {"famille": "ridge", "alpha": 1000.0} and simple[-1]["famille"] == "gradient_boosting"
    assert "models=era5" in M.url_open_meteo(36.8, 10.2, "2015-01-01", "2015-12-31", M.VARS_OM)
    assert M.poids_requete(8, 365) == pytest.approx(365 / 14)


def test_sites_a_moins_de_10_km():
    st = pd.DataFrame({"code": ["INM60740", "ISD60740399999", "INM60742"], "lat": [35.7622, 35.758, 35.5],
                       "lon": [10.7575, 10.755, 11.0667]})
    s = M.sites(st)
    assert s["INM60740"] == s["ISD60740399999"] == "INM60740" and s["INM60742"] == "INM60742"


def _om(t, ws, wd, t2m=20.0, lat=36.75, lon=10.25):
    n = len(t)
    return pd.DataFrame({"temps": t, "wind_speed_10m": ws, "wind_direction_10m": wd, "wind_speed_100m": 2 * np.asarray(ws),
                         "wind_direction_100m": wd, "temperature_2m": t2m, "boundary_layer_height": 500.0,
                         "shortwave_radiation": 0.0, "cloud_cover": 10.0, "lat_maille": lat, "lon_maille": lon}).iloc[:n]


def test_paires_et_variables():
    t = pd.date_range("2018-03-01", periods=8, freq="3h")
    e = _om(t, 4.0, 270.0)                                    # vent d'ouest : u10 = +4
    obs = pd.DataFrame({"creneau": t[:6], "wind_u_m_s": 5.0, "wind_v_m_s": 1.0, "wind_speed_m_s": np.hypot(5, 1)})
    site = {"z_maille_500m_m": 100.0, "pente_maille_deg": 2.0, "z0_maille_m": 0.1, "tpi_5km_m": -5.0,
            "tpi_20km_m": -20.0, "distance_cote_km": 3.0}
    p = M.construire_paires(obs, e, pd.Series(15.0, index=t), site, z_era5=40.0)
    r, _ = M.regle()
    assert len(p) == 6 and set(r["variables"]) <= set(p.columns)
    assert p["u10_era5"].iloc[0] == pytest.approx(4.0) and abs(p["v10_era5"].iloc[0]) < 1e-9
    assert p["du"].iloc[0] == pytest.approx(1.0) and p["dv"].iloc[0] == pytest.approx(1.0)
    assert p["rapport_v100_v10_era5"].iloc[0] == pytest.approx(2.0)
    assert p["dt_terre_mer_k"].iloc[0] == pytest.approx(5.0) and p["dz_maille_moins_era5_m"].iloc[0] == 60.0
    assert p["log10_z0_maille"].iloc[0] == pytest.approx(-1.0)


def test_telechargement_reprise_et_deux_echecs(tmp_path):
    t = pd.date_range("2015-01-01", periods=3, freq="h")
    js = {"latitude": 36.75, "longitude": 10.25,
          "hourly": {"time": t.strftime("%Y-%m-%dT%H:%M").tolist(), **{v: [1.0, 2.0, 3.0] for v in M.VARS_OM}}}
    appels = []

    def lire(url):
        appels.append(url)
        return js
    f = tmp_path / "X_2015.csv"
    assert M.telecharger_point(36.8, 10.2, 2015, M.VARS_OM, f, lire=lire) and len(appels) == 1
    assert M.telecharger_point(36.8, 10.2, 2015, M.VARS_OM, f, lire=lire) and len(appels) == 1   # reprise
    assert pd.read_csv(f)["lat_maille"].iloc[0] == 36.75

    def panne(url):
        raise ValueError("reponse invalide")
    assert not M.telecharger_point(36.8, 10.2, 2016, M.VARS_OM, tmp_path / "X_2016.csv", lire=panne)
    st = pd.DataFrame({"code": ["A", "B"], "lat": [36.0, 35.0], "lon": [10.0, 9.0]})
    res = M.etape_era5(st, tmp_path / "e", [2015, 2016], [[37.0, 11.0]], lire=panne)
    assert len(res["echecs"]) == 2                                   # arret apres deux echecs


def test_bootstrap_ecart_nul_si_mos_egal_era5():
    t = pd.date_range("2021-01-01", periods=200, freq="3h")
    p = pd.DataFrame({"site": np.repeat(["a", "b"], 100), "temps": np.tile(t[:100], 2), "u_obs": 1.0, "v_obs": 0.0,
                      "u10_era5": np.linspace(0, 2, 200), "v10_era5": 0.5})
    p["u_mos"], p["v_mos"] = p["u10_era5"], p["v10_era5"]
    e, lo, hi = M.bootstrap_ecart(p, 200, 1)
    assert e == pytest.approx(0.0, abs=1e-12) and lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------- de bout en bout, synthetique
def _jeu(dossier: Path, r: dict, signal: bool, annees) -> None:
    """10 stations (dont 2 a moins de 10 km : un site), 2015-2022 ; erreur ERA5 = fonction des
    variables (signal) ou bruit pur."""
    rng = np.random.default_rng(3)
    n_st = 10
    lat = 33.0 + 0.4 * np.arange(n_st)
    lon = np.full(n_st, 9.5)
    lat[1], lon[1] = lat[0] + 0.02, lon[0]                    # meme site que la station 0
    st = pd.DataFrame({"code": [f"S{k:02d}" for k in range(n_st)], "lat": lat, "lon": lon})
    st["site"] = st["code"].map(M.sites(st))
    for a in annees:
        st[f"taux_qc_{a}"] = 0.9
    st.to_csv(dossier / "stations.csv", index=False)
    tpi = rng.normal(0, 30, n_st)
    lignes = []
    for k, s in st.iterrows():
        t = pd.date_range(f"{annees[0]}-01-01", f"{annees[-1]}-12-31 21:00", freq="9h")   # leger
        n = len(t)
        p = pd.DataFrame({"code": s["code"], "site": s["site"], "temps": t, "annee": t.year, "aeroport": k < 3,
                          "abri_probable": False})
        for v in r["variables"]:
            p[v] = rng.normal(0, 1, n)
        p["tpi_5km_m"] = tpi[k]
        p["u10_era5"], p["v10_era5"] = rng.normal(2, 3, n), rng.normal(0, 3, n)
        p["vitesse10_era5"] = np.hypot(p["u10_era5"], p["v10_era5"])
        if signal:
            du = -0.4 * p["u10_era5"] + 0.02 * p["tpi_5km_m"] + 0.3 * p["sin_heure"]
            dv = -0.4 * p["v10_era5"] + 0.3 * p["cos_jour"]
        else:
            du, dv = pd.Series(0.0, index=p.index), pd.Series(0.0, index=p.index)
        p["du"], p["dv"] = du + rng.normal(0, 1.2, n), dv + rng.normal(0, 1.2, n)
        p["u_obs"], p["v_obs"] = p["u10_era5"] + p["du"], p["v10_era5"] + p["dv"]
        p["vitesse_obs"] = np.hypot(p["u_obs"], p["v_obs"])
        p["dvitesse"] = p["vitesse_obs"] - p["vitesse10_era5"]
        lignes.append(p)
    paires = pd.concat(lignes, ignore_index=True)
    for a, g in paires.groupby("annee"):
        g.to_csv(dossier / f"paires_{a}.csv.gz", index=False)


def _regle_legere():
    r, _ = M.regle()
    r = copy.deepcopy(r)
    r["modeles"]["ridge"]["alpha"] = [1.0, 100.0]
    r["modeles"]["gradient_boosting"].update({"max_depth": [3], "max_iter": [30]})
    r["jugement_test"]["bootstrap"]["tirages"] = 300
    return r


def test_mos_bat_era5_puis_test_lu_une_fois(tmp_path):
    r = _regle_legere()
    annees = r["decoupage"]["entrainement"] + r["decoupage"]["validation"] + r["decoupage"]["test"]
    _jeu(tmp_path, r, signal=True, annees=annees)
    test = tmp_path / "paires_2022.csv.gz"
    cache = test.with_name("cache_2022.csv.gz")
    test.rename(cache)                                         # choisir ne doit jamais lire 2022
    sortie = tmp_path / "sortie"
    choix = M.etape_choisir(tmp_path, sortie, r, "h" * 64, log=lambda *a: None)
    assert choix["test_autorise"] and choix["validation"]["bat_era5"]
    v = choix["validation"]
    assert v["moyenne_rmse_vect_mos"] < v["moyenne_rmse_vect_era5"] and v["ic95_ecart"][1] < 0
    assert set(choix["sites_juges_validation"]) == set(pd.read_csv(tmp_path / "stations.csv")["site"])
    assert len(choix["sites_juges_validation"]) == 9           # 10 stations, 9 sites
    with pytest.raises(SystemExit):
        M.etape_choisir(tmp_path, sortie, r, "h" * 64, log=lambda *a: None)     # jamais ecrase
    cache.rename(test)
    with pytest.raises(SystemExit):
        M.etape_tester(tmp_path, sortie, r, "x" * 64, log=lambda *a: None)      # regle changee
    res = M.etape_tester(tmp_path, sortie, r, "h" * 64, log=lambda *a: None)
    assert res["test"]["bat_era5"] and not res["anomalies"]
    t = pd.read_csv(sortie / "test_sites.csv")
    assert (t["rmse_vect_mos"] < t["rmse_vect_era5"]).all() and "rmse_vitesse_mos_dvitesse" in t
    with pytest.raises(SystemExit):
        M.etape_tester(tmp_path, sortie, r, "h" * 64, log=lambda *a: None)      # le test ne se relit pas


def test_sans_signal_le_test_n_est_pas_lu(tmp_path):
    r = _regle_legere()
    annees = r["decoupage"]["entrainement"] + r["decoupage"]["validation"]
    _jeu(tmp_path, r, signal=False, annees=annees)
    sortie = tmp_path / "sortie"
    choix = M.etape_choisir(tmp_path, sortie, r, "h" * 64, log=lambda *a: None)
    assert not choix["test_autorise"]
    with pytest.raises(SystemExit):
        M.etape_tester(tmp_path, sortie, r, "h" * 64, log=lambda *a: None)
    assert not (sortie / "test.json").exists()
    assert json.loads((sortie / "choix.json").read_text(encoding="utf-8"))["config_choisie"]


def test_validation_croisee_retire_tout_le_site(tmp_path, monkeypatch):
    """Quand un site est juge, aucune de ses stations n'entre dans l'ajustement."""
    r = _regle_legere()
    r["modeles"]["gradient_boosting"]["max_iter"] = []
    r["modeles"]["ridge"]["alpha"] = [1.0]
    annees = r["decoupage"]["entrainement"] + r["decoupage"]["validation"]
    _jeu(tmp_path, r, signal=True, annees=annees)
    vus = []
    vrai = M.ajuster

    def espion(c, train, variables, graine, cibles=("du", "dv")):
        vus.append(set(train["site"]))
        return vrai(c, train, variables, graine, cibles)
    monkeypatch.setattr(M, "ajuster", espion)
    choix = M.etape_choisir(tmp_path, tmp_path / "s", r, "h" * 64, log=lambda *a: None)
    sites_ = sorted(choix["sites_juges_validation"])
    for s, vu in zip(sites_, vus[:len(sites_)]):
        assert s not in vu and len(vu) == len(sites_) - 1
