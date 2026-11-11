"""Stations supplementaires du modele national (scripts/stations_supplementaires.py, stations_reelles) :
inventaire ISD + OGIMET, controle qualite, placement sur la grille, validation reelle a part du DSO7.
Fichiers synthetiques, aucun acces reseau."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_forcage_era5 import LAT0, LON0, _jeu_reel, cfg_era5, forcage  # noqa: F401

REPO = Path(__file__).resolve().parents[1]


def _script(nom: str):
    spec = importlib.util.spec_from_file_location(nom, REPO / "scripts" / f"{nom}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _script("stations_supplementaires")


def _obs(code, t, vit, dirn):
    return pd.DataFrame({"station_id": code, "timestamp_utc": pd.DatetimeIndex(t).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "wind_speed_m_s": vit, "wind_dir_deg": dirn, "wind_u_m_s": np.nan, "wind_v_m_s": np.nan})


# ---------------------------------------------------------------------- controle qualite
def test_controle_qualite_motifs():
    from pcfno3d.stations_reelles import controle_qualite, preparer_observations

    rng = np.random.default_rng(0)
    t = pd.date_range("2021-05-01", "2021-07-31 21:00", freq="3h")
    vit = rng.integers(1, 9, len(t)).astype(float)
    dirn = rng.integers(0, 36, len(t)) * 10.0
    mai = t.month == 5
    vit[(t.month == 6)] = 0.0                                  # juin : anemometre en panne
    vit[5], dirn[5] = 75.0, 90.0                               # hors plage (mai)
    vit[20:30], dirn[20:30] = 4.0, 120.0                       # 10 messages egaux (27 h) : figes
    vit[40:42], dirn[40:42] = 3.0, 200.0                       # 2 messages egaux : banal, garde
    vit[50], dirn[50] = 0.0, np.nan                            # calme : garde, u = v = 0
    o = _obs("INM60720", t, vit, dirn)
    o = pd.concat([o, o.iloc[[60]]], ignore_index=True)        # doublon
    gardees, rap = controle_qualite(preparer_observations(o))
    rap = rap.set_index("mois")
    assert rap.loc["2021-06", "mois_ecarte"] and rap.loc["2021-06", "motif_mois"] == "panne (vent nul)"
    assert rap.loc["2021-05", "hors plage"] == 1 and rap.loc["2021-05", "valeur figee"] == 10
    assert not rap.loc["2021-05", "mois_ecarte"] and not rap.loc["2021-07", "mois_ecarte"]
    assert (gardees["t"].dt.month != 6).all()
    assert len(gardees) == mai.sum() - 11 + (t.month == 7).sum()       # doublon retire
    calme = gardees[gardees["t"] == t[50]]
    assert len(calme) == 1 and calme["wind_u_m_s"].iloc[0] == 0 and calme["wind_v_m_s"].iloc[0] == 0
    assert (gardees["t"] == t[40]).any()


def test_lire_observations_inchange_pour_les_aeroports(tmp_path):
    """Le refactorisation garde exactement la lecture des 12 aeroports (entree de C' et D)."""
    from pcfno3d.stations_reelles import lire_observations

    t = pd.date_range("2021-06-01", periods=400, freq="3h")
    o = pd.concat([_obs("DTTA", t, np.where(t.month == 6, 0.0, 3.0), 90.0),
                   _obs("DTTX", t, 5.0, 270.0), _obs("INM60720", t, 4.0, 0.0)])
    o.to_csv(tmp_path / "metar_2021.csv", index=False)
    obs = lire_observations(tmp_path, verbose=False, codes=["DTTA", "DTTX"])
    assert set(obs["station_id"]) == {"DTTA", "DTTX"}                       # jamais une station supp.
    assert not ((obs["station_id"] == "DTTA") & (obs["t"].dt.month == 6)).any()
    assert len(obs) == len(t) + (t.month != 6).sum()


# ---------------------------------------------------------------------- inventaire
HIST = '''"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"
"607150","99999","CARTHAGE","TS","","DTTA","+36.851","+010.227","+0006.7","19430608","20250824"
"607200","99999","KELIBIA","TS","","DTTL","+36.850","+011.083","+0030.0","19430524","20250824"
"607403","99999","HABIB BOURGUIBA INTL","TS","","DTMB","+35.758","+010.755","+0002.7","19890908","20250824"
"607470","99999","KASSERINE NORTH","TS","","","+35.183","+008.800","+0689.0","19490108","19520529"
"607690","99999","ZARZIS / DJERBA","TN","","DTTJ","+33.875","+010.775","+0005.8","19660722","20250824"
"723270","13897","NASHVILLE","US","TN","KBNA","+36.111","-086.688","+0179.0","19510101","20250827"
'''

OGI = """<table><tr><td>0-20000-0-60720 WIGOS ID</td><td>60720</td><td>DTTL</td><td>Kelibia</td><td>Tunisia</td>
<td>36-51-00N</td><td>011-04-59E</td><td>29</td><td>1968-01-01</td><td>----</td></tr>
<tr><td>x</td><td>60740</td><td>DTTM</td><td>Monastir-Skanes</td><td>Tunisia</td><td>35-45-44N</td><td>010-45-27E</td>
<td>2</td><td>1968-06-01</td><td>----</td></tr>
<tr><td>x</td><td>60780</td><td>DTTR</td><td>El Borma</td><td>Tunisia</td><td>31-40-59N</td><td>009-18-00E</td>
<td>258</td><td>1978-01-01</td><td>2014-11-01</td></tr>
<tr><td>x</td><td>02607</td><td>ESDB</td><td>Angelholm</td><td>Sweden</td><td>56-17-52N</td><td>012-50-39E</td>
<td>18</td><td>2002-12-07</td><td>----</td></tr></table>"""


def test_inventaire_isd_ogimet_et_aeroports(tmp_path):
    (tmp_path / "h.csv").write_text(HIST, encoding="utf-8")
    (tmp_path / "o.html").write_text(OGI, encoding="latin-1")
    isd = S.inventaire_isd(tmp_path / "h.csv", (2015, 2022))
    assert set(isd["isd"]) == {"60715099999", "60720099999", "60740399999", "60769099999"}  # ni 1952 ni US
    assert isd.set_index("isd").loc["60740399999", "wmo"] is np.nan or pd.isna(isd.set_index("isd").loc["60740399999", "wmo"])
    ogi = S.lire_ogimet(tmp_path / "o.html")
    assert set(ogi["wmo"]) == {"60720", "60740"}                    # El Borma ferme, Suede exclue
    assert abs(ogi.set_index("wmo").loc["60720", "lon_ogimet"] - (11 + 4 / 60 + 59 / 3600)) < 1e-9
    aer = pd.read_csv(REPO / "configs" / "national" / "stations_metar.csv")
    inv = S.fusionner(isd, ogi, aer).set_index("code")
    assert inv.loc["INM60715", "aeroport_metar"] == "DTTA"             # OACI
    assert inv.loc["INM60769", "aeroport_metar"] == "DTTJ"
    assert inv.loc["INM60740", "aeroport_metar"] == "DTMB"             # OGIMET seul, a 0,5 km de DTMB
    assert inv.loc["INM60740", "source"] == "OGIMET"
    assert inv.loc["INM60720", "aeroport_metar"] == "" and inv.loc["INM60720", "dans_domaine"]
    assert inv.loc["INM60720", "lat"] == pytest.approx(36.85)          # coordonnees OGIMET retenues


def test_taux_valides_et_creneaux():
    t = pd.Series(pd.to_datetime(["2021-01-01 00:10", "2021-01-01 02:35", "2021-01-01 03:20",
                                  "2021-01-01 07:00", "2020-12-31 23:50"]))
    assert S.taux_valides(t, 2021) == pytest.approx(2 / 2920)          # 00 h et 03 h ; 07 h hors fenetre
    o = pd.DataFrame({"t": t, "wind_speed_m_s": [1.0, 2.0, 3.0, 4.0, 5.0], "wind_u_m_s": 0.0, "wind_v_m_s": 0.0})
    c = S.au_creneau(o).set_index("creneau")["wind_speed_m_s"]
    assert c[pd.Timestamp("2021-01-01 03:00")] == 3.0                  # le plus proche (20 min) garde
    assert c[pd.Timestamp("2021-01-01 00:00")] == 1.0 and len(c) == 2       # 23 h 50 et 00 h 10 : meme creneau


def test_indicateur_abri_sans_era5():
    t = pd.date_range("2018-01-01", periods=400, freq="3h")
    inv = pd.DataFrame({"code": ["A", "B", "C"], "nom": ["a", "b", "c"], "lat": [36.0, 36.2, 36.4],
                        "lon": [10.0, 10.1, 10.0]})
    series = {c: pd.DataFrame({"creneau": t, "wind_speed_m_s": v, "wind_u_m_s": 0.0, "wind_v_m_s": 0.0})
              for c, v in (("A", 1.5), ("B", 5.0), ("C", 5.0))}
    out = S.indicateurs_abri(inv, series, None).set_index("code")
    assert out.loc["A", "rapport_vitesse_voisines"] == pytest.approx(0.3)
    assert out.loc["B", "rapport_vitesse_voisines"] == pytest.approx(5.0 / 3.25, abs=1e-3)
    assert not out["abri_probable"].any()                    # ERA5 absente : jamais signale sur un seul indice


def test_distance_cote():
    from pcfno3d.sites_stations import distance_cote_km

    cote = np.array([[36.0, 10.0], [37.0, 11.0]])
    assert distance_cote_km(36.0, 10.0, cote) == 0.0
    assert distance_cote_km(36.1, 10.0, cote) == pytest.approx(11.1, abs=0.1)


def test_liste_versionnee_coherente():
    """configs/national/stations_supplementaires.csv : stations dans le domaine, aucun aeroport METAR."""
    from pcfno3d.stations_reelles import stations_supplementaires

    f = REPO / "configs" / "national" / "stations_supplementaires.csv"
    t = stations_supplementaires(f)
    aer = pd.read_csv(REPO / "configs" / "national" / "stations_metar.csv")
    assert len(t) and not set(t["code"]) & set(aer["code"])
    assert t["lat"].between(30.15, 37.615).all() and t["lon"].between(7.45, 11.644).all()
    for a in aer.itertuples():
        assert all(S.distance_km(a.lat, a.lon, r.lat, r.lon) >= S.DIST_AEROPORT_KM for r in t.itertuples())


# ---------------------------------------------------------------------- placement et validation
def test_cellules_latlon_comme_le_generateur(geom_supp):
    from pcfno3d.realdata import PlanGrille
    from pcfno3d.stations_reelles import cellules_latlon

    g = geom_supp
    plan = PlanGrille.from_geom(g.cfg, g)
    jj, ii = np.nonzero(g.valid.any(axis=0))
    iy, ix = int(jj[len(jj) // 3]), int(ii[len(jj) // 3])
    # centre d'une cellule fine de la maille (iy, ix) -> la meme maille
    la, lo = plan.to_latlon((ix * g.stride + 0.5) * g.dx_s, (iy * g.stride + 0.5) * g.dy_s)
    t = pd.DataFrame({"code": ["X", "Y"], "lat": [float(la), 10.0], "lon": [float(lo), 10.0]})
    cel, ec = cellules_latlon(g, t)
    assert cel == {"X": (iy, ix)} and ec == {"Y": "hors du domaine"}


@pytest.fixture(scope="module")
def geom_supp(cfg_era5):  # noqa: F811
    from pcfno3d.geometry import Geometry
    return Geometry(cfg_era5, device="cpu")


@pytest.fixture(scope="module")
def reel_supp(cfg_era5, index, tmp_path_factory):  # noqa: F811
    from pcfno3d.train import Trainer

    c, obs = _jeu_reel(cfg_era5, index, tmp_path_factory.mktemp("supp"))
    Trainer(c, device="cpu", verbose=False).fit()
    return c, obs


def test_validation_supplementaire_a_part_du_dso7(reel_supp, tmp_path):
    """Stations supplementaires : jugees a leur cellule, jamais en entree ; les sorties des
    aeroports sont IDENTIQUES avec ou sans elles ; --supp-seul ne reecrit rien d'autre."""
    from pcfno3d.geometry import Geometry
    from pcfno3d.realdata import PlanGrille
    from pcfno3d.stations_reelles import cellules_codes

    c, obs = reel_supp
    V = _script("valider_grand_tunis_reel")
    sortie = c.paths.run_dir / "validation_reelle" / "test"
    V.valider(c, obs, "test", device="cpu")
    ref = pd.read_csv(sortie / "stations_bilan.csv")
    ref_paires = pd.read_csv(sortie / "stations_paires.csv")

    g = Geometry(c)
    occupees = set(cellules_codes(g).values())
    jj, ii = np.nonzero(g.valid.any(axis=0))
    libres = [(int(a), int(b)) for a, b in zip(jj, ii) if (int(a), int(b)) not in occupees]
    plan = PlanGrille.from_geom(c, g)
    lignes = []
    for k, (iy, ix) in enumerate([libres[0], libres[len(libres) // 2]]):
        la, lo = plan.to_latlon((ix * g.stride + 0.5) * g.dx_s, (iy * g.stride + 0.5) * g.dy_s)
        lignes.append({"code": f"INM6079{k}", "nom": f"Supp{k}", "lat": float(la), "lon": float(lo), "retenue": 1})
    lignes.append({"code": "INM60799", "nom": "Loin", "lat": 10.0, "lon": 10.0, "retenue": 1})
    lignes.append({"code": "INM60798", "nom": "Non retenue", "lat": lignes[0]["lat"], "lon": lignes[0]["lon"],
                   "retenue": 0})
    liste = tmp_path / "stations_supplementaires.csv"
    pd.DataFrame(lignes).to_csv(liste, index=False)
    supp = tmp_path / "obs_supp"
    supp.mkdir()
    idx = pd.read_csv(c.paths.cache_dir / "index.csv")
    t = pd.to_datetime(idx["timestamp"])
    vit = 3.0 + np.arange(len(t)) % 5                          # vitesses variables : jamais figees
    pd.concat([_obs("INM60790", t, vit, 180.0), _obs("INM60791", t, vit / 2, 90.0)]).to_csv(
        supp / "synop_2025.csv", index=False)

    res = V.valider(c, obs, "test", device="cpu", stations_supp=liste, obs_supp=supp)
    pd.testing.assert_frame_equal(pd.read_csv(sortie / "stations_bilan.csv"), ref)          # DSO7 intact
    pd.testing.assert_frame_equal(pd.read_csv(sortie / "stations_paires.csv"), ref_paires)
    b = pd.read_csv(sortie / "stations_supp_bilan.csv")
    assert set(b["station"]) == {"Supp0", "Supp1"} and b["rmse_vect"].notna().all()
    assert set(b["source"]) == {"modele", "era5", "generateur"}
    p = pd.read_csv(sortie / "stations_supp_paires.csv")
    assert set(p["code"]) == {"INM60790", "INM60791"}
    q = p[p["code"] == "INM60790"]
    assert (q["v_obs"] >= 3.0).all() and np.allclose(q["u_obs"], 0.0, atol=1e-9)   # vent du sud
    j = json.loads((sortie / "bilan_supp.json").read_text(encoding="utf-8"))
    assert j["ecartees"] == {"INM60799": "hors du domaine"} and j["stations"] == 2
    assert "hors DSO7" in j["note"] and "stations_supplementaires" in res
    assert (sortie / "stations_supp_qc.csv").exists()
    assert "stations_supplementaires" not in json.loads((sortie / "bilan.json").read_text(encoding="utf-8"))

    # --supp-seul : seulement les sorties supplementaires ; celles des aeroports ne sont pas touchees
    (sortie / "stations_bilan.csv").write_text("temoin", encoding="utf-8")
    V.valider(c, obs, "test", device="cpu", stations_supp=liste, obs_supp=supp, supp_seul=True)
    assert (sortie / "stations_bilan.csv").read_text(encoding="utf-8") == "temoin"
    assert len(pd.read_csv(sortie / "stations_supp_paires.csv")) == len(p)
