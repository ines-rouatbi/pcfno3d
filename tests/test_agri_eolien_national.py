"""Phases 6 et 7 nationales : agriculture (regle figee, creneaux, FAO-56, GFS sur la grille, AGRI1 / AGRI2,
produit J0-J+2) et eolien (atlas 50-150 m, zoom Grand Tunis, EOL2), sur le mini-jeu reel ; regles
PROPOSITION hachees ; carnet. Aucun acces reseau, aucun GPU."""

from __future__ import annotations

import importlib.util
import json
import math
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


@pytest.fixture(scope="module")
def reel(cfg_era5, index, tmp_path_factory):  # noqa: F811
    """Mini-jeu reel entraine (3 stations, ERA5, observations et radiosondages synthetiques)."""
    from pcfno3d.train import Trainer

    c, obs = _jeu_reel(cfg_era5, index, tmp_path_factory.mktemp("agri_eol"))
    Trainer(c, device="cpu", verbose=False).fit()
    return c, obs


@pytest.fixture()
def config_fixe(reel, monkeypatch):
    """Config.load(<n'importe quel nom>) -> la config du mini-jeu (comme tests/test_atlas.py)."""
    import pcfno3d.config as cm
    c, _ = reel
    monkeypatch.setattr(cm.Config, "load", classmethod(lambda cls, _n: c))
    return c


# ---------------------------------------------------------------------- regle figee
def test_regle_identique_au_grand_tunis(monkeypatch):
    from pcfno3d import agriculture as ag
    from pcfno3d.atom_wind import traitement_possible

    vd = _script("verifier_decisions")
    kmh = np.concatenate([np.arange(0, 40, 0.05), [2.999999, 3, 15, 15.000001, 19, 19.000001, 25, 25.000001]])
    v = kmh / 3.6
    for nom in ("traitement", "aspersion"):
        np.testing.assert_array_equal(ag.REGLES[nom](v), vd.REGLES[nom](v))
    np.testing.assert_array_equal(ag.decision_traitement(v) == "oui", traitement_possible(v))

    # service/atom_outils.decision_agricole : meme regle, heure par heure
    import service.atom_outils as ao
    for debut in range(0, len(v), 24):
        bloc = v[debut:debut + 24]
        if len(bloc) < 24:
            break
        monkeypatch.setattr(ao, "position", lambda wf, lat, lon, lieu: (0.0, 0.0, "x"))
        monkeypatch.setattr(ao, "jour", lambda wf, j: {"date": "2022-01-01"})
        monkeypatch.setattr(ao, "vent", lambda wf, p, x, y, z, h, b=bloc: (float(b[int(h)]), 0.0))
        r = ao.decision_agricole(object(), 0)
        assert [x["traitement"] for x in r["heures"]] == list(ag.decision_traitement(bloc))
        assert [x["aspersion"] for x in r["heures"]] == list(ag.decision_aspersion(bloc))


def test_creneaux_et_interpolation_en_temps():
    from pcfno3d import agriculture as ag

    assert ag.creneaux([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    assert ag.creneaux([1, 1], heures=[6, 7]) == [(6, 8)]
    assert ag.creneaux([1, 1], heures=[6, 9]) == [(6, 7), (9, 10)]      # trou dans les heures
    assert ag.texte_creneaux([]) == "aucun creneau"
    t = np.array([1.0, 7.0, 13.0])
    c = np.array([[0.0], [6.0], [12.0]])
    out = ag.interpoler_heures(t, c, np.arange(24))
    np.testing.assert_allclose(out[1:14, 0], np.arange(0, 13))
    assert np.isnan(out[0, 0]) and np.isnan(out[14:, 0]).all()           # pas d'extrapolation
    code = ag.coder(np.array(["oui", "non", "attention"]), np.array([True, True, False]))
    assert code.tolist() == [0, 2, -1]


def test_fao56_exemple_et_portage_de_la_page():
    """FAO-56 exemple 18 (Bruxelles, 6 juillet : ET0 = 3,9 mm/j) et le portage exact de et0Day."""
    from pcfno3d import agriculture as ag

    es = (ag.esat(21.5) + ag.esat(12.3)) / 2
    r = ag.et0_journalier(21.5, 12.3, 1.409 / es * 100, 2.078, 100.0, 50.8, 187, rs_mj=22.07)
    assert float(r["et0_mm"]) == pytest.approx(3.9, abs=0.1) and not r["rayonnement_repli"]
    assert ag.U2_DEPUIS_U10 == pytest.approx(0.748, abs=5e-4)

    def page(tmax, tmin, rh, u2, z, lat, j):           # et0Day de la maquette, ligne a ligne
        t = (tmax + tmin) / 2
        es_ = lambda x: 0.6108 * math.exp(17.27 * x / (x + 237.3))  # noqa: E731
        d = 4098 * es_(t) / ((t + 237.3) ** 2)
        p = 101.3 * ((293 - .0065 * z) / 293) ** 5.26
        g = .000665 * p
        e_s = (es_(tmax) + es_(tmin)) / 2
        ea = e_s * rh / 100
        phi = lat * math.pi / 180
        dr = 1 + .033 * math.cos(2 * math.pi * j / 365)
        dl = .409 * math.sin(2 * math.pi * j / 365 - 1.39)
        ws = math.acos(-math.tan(phi) * math.tan(dl))
        ra = 24 * 60 / math.pi * .082 * dr * (ws * math.sin(phi) * math.sin(dl) + math.cos(phi) * math.cos(dl) * math.sin(ws))
        rs, rso = .72 * ra, (.75 + 2e-5 * z) * ra
        rnl = 4.903e-9 * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2) * (.34 - .14 * math.sqrt(ea)) * (1.35 * rs / rso - .35)
        rn = .77 * rs - rnl
        return (.408 * d * rn + g * 900 / (t + 273) * u2 * (e_s - ea)) / (d + g * (1 + .34 * u2))

    for args in ((33.0, 21.0, 45.0, 3.1, 10.0, 36.65, 200), (18.0, 6.0, 70.0, 1.2, 600.0, 33.0, 30)):
        r = ag.et0_journalier(*args)
        assert r["rayonnement_repli"] and float(r["et0_mm"]) == pytest.approx(page(*args), rel=1e-9)


def test_gfs_bilineaire_sur_la_grille(reel, tmp_path):
    from pcfno3d import agriculture as ag
    from pcfno3d.geometry import Geometry

    c, _ = reel
    g = Geometry(c)
    lat_g, lon_g = np.array([35.75, 36.0, 36.25]), np.array([9.75, 10.0, 10.25])
    la, lo = np.meshgrid(lat_g, lon_g, indexing="ij")
    temps = np.array([1_640_995_200, 1_641_016_800])
    t2m = np.stack([20 + 4 * (la - 36) + 2 * (lo - 10)] * 2)
    np.savez(tmp_path / "cube.npz", temps=temps, lat=lat_g, lon=lon_g, elevation=np.zeros_like(la),
             echeance_h=np.array([24, 24], np.int16), temperature_2m=t2m.astype(np.float32),
             relative_humidity_2m=np.full_like(t2m, 55.0, dtype=np.float32),
             shortwave_radiation=np.full_like(t2m, 300.0, dtype=np.float32))
    m = ag.MeteoGFS(tmp_path / "cube.npz", g)
    a = m.a(int(temps[1]))
    lat, lon = ag.latlon_noeuds(g)
    attendu = 20 + 4 * (lat - 36) + 2 * (lon - 10) - 0.0065 * g.zt         # lineaire + relief
    np.testing.assert_allclose(a["t2m"], attendu, atol=1e-4)
    assert np.allclose(a["hr2m"], 55.0) and np.allclose(a["ray"], 300.0)
    assert m.a(123) is None


def test_geotiff_aller_retour(tmp_path):
    from pcfno3d.sig import ecrire_geotiff, lire_geotiff

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    a[1, 2] = np.nan
    ecrire_geotiff(tmp_path / "a.tif", a, 10.0, 36.0, 0.01, 0.009)
    b, g = lire_geotiff(tmp_path / "a.tif")
    np.testing.assert_array_equal(np.isnan(b), np.isnan(a))
    np.testing.assert_allclose(b[np.isfinite(a)], a[np.isfinite(a)])
    assert g["lon0"] == pytest.approx(10.0) and g["lat0"] == pytest.approx(36.0)
    assert g["dlon"] == pytest.approx(0.01) and g["dlat"] == pytest.approx(0.009)


def test_entree_de_production_avec_tous_les_aeroports(reel, tmp_path):
    """paths.stations_obs : chaque station de stations.json (codes nationaux) recoit sa mesure ; avant,
    seuls les 3 codes du Grand Tunis etaient lus."""
    import copy
    import shutil

    from pcfno3d.cache import load_index
    from pcfno3d.inference import WindFieldModel

    c0, _ = reel
    c = copy.deepcopy(c0)
    data = tmp_path / "jeu"
    shutil.copytree(c0.paths.data_dir, data)
    st = json.loads((data / "stations.json").read_text(encoding="utf-8"))
    codes = ["DTTX", "DTTZ", "DTKA"]
    for s, k in zip(st, codes):
        s["code"], s["nom"] = k, f"{k} aeroport"
    (data / "stations.json").write_text(json.dumps(st), encoding="utf-8")
    obs = tmp_path / "obs"
    obs.mkdir()
    stem, t = load_index(c0).iloc[0][["stem", "timestamp"]]
    t = pd.Timestamp(t)
    pd.DataFrame([{"station_id": k, "timestamp_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "wind_speed_m_s": 1.0 + i,
                   "wind_dir_deg": 270.0, "wind_u_m_s": 1.0 + i, "wind_v_m_s": 0.0}
                  for i, k in enumerate(codes)]).to_csv(obs / "metar_2022.csv", index=False)
    c.apply_overrides({"paths.data_dir": data, "paths.stations_obs": obs})
    wf = WindFieldModel(c, device="cpu", verbose=False)
    S, *_ = wf.surface_from_cache(str(stem))
    for i, s in enumerate(st):
        assert S[0, s["iy"], s["ix"]] == pytest.approx(1.0 + i) and S[7, s["iy"], s["ix"]] == 1.0


# ---------------------------------------------------------------------- regles hachees
def test_regles_proposition_hachees():
    from pcfno3d.regles import verifier

    for f in ("regle_agri_national.json", "regle_eolien_national.json"):
        r, h = verifier(REPO / "configs" / "national" / f)
        assert r["statut"].startswith("PROPOSITION") and len(h) == 64
    r, _ = verifier(REPO / "configs" / "national" / "regle_agri_national.json")
    from pcfno3d.agriculture import TEXTE_REGLE
    assert r["regle_de_decision"]["texte"] == TEXTE_REGLE


def test_regle_modifiee_refusee(tmp_path):
    from pcfno3d.regles import hacher, sortie_neuve, verifier

    f = tmp_path / "regle.json"
    f.write_text('{"a": 1}', encoding="utf-8")
    hacher(f)
    verifier(f)
    f.write_text('{"a": 2}', encoding="utf-8")
    with pytest.raises(SystemExit):
        verifier(f)
    with pytest.raises(SystemExit):
        hacher(f)                                     # une empreinte differente existe deja
    d = sortie_neuve(tmp_path / "s", "x.json")
    (d / "x.json").write_text("{}")
    with pytest.raises(SystemExit):
        sortie_neuve(tmp_path / "s", "x.json")


# ---------------------------------------------------------------------- phase 6 de bout en bout
def test_agri1_agri2_sur_les_paires_station_exclue(reel, tmp_path):
    c, obs = reel
    V = _script("valider_grand_tunis_reel")
    V.valider(c, obs, "test", device="cpu")
    A = _script("agri_national")
    run = c.paths.run_dir
    assert A.main(["decisions", "--run", str(run), "--sortie", str(tmp_path / "agri")]) == 0
    v = json.loads((tmp_path / "agri" / "verdict_agri_national.json").read_text(encoding="utf-8"))
    assert v["statut_regle"] == "PROPOSITION" and len(v["sha256_regle"]) == 64
    assert set(v["stations"]) == {"Tunis-Carthage", "Zaghouan", "Nabeul"}
    assert 0 <= v["agri1"]["accord_modele"] <= 1 and isinstance(v["agri2"]["conforme"], bool)
    assert v["agri1"]["conforme"] == (v["agri1"]["accord_modele"] >= v["agri1"]["accord_era5"])
    # format de verifier_decisions : lu par scripts/bilan_conformite.py
    d = json.loads((tmp_path / "agri" / "decisions_agricoles.json").read_text(encoding="utf-8"))
    assert "modele / toutes heures" in d["agri1"] and "modele / toutes heures" in d["agri2"]
    assert (tmp_path / "agri" / "verdict_agri_national.md").exists()
    with pytest.raises(SystemExit):                    # jamais recalcule
        A.main(["decisions", "--run", str(run), "--sortie", str(tmp_path / "agri")])


def test_produit_agricole_j0_j2(reel, config_fixe, tmp_path):
    from pcfno3d.cache import load_index
    from pcfno3d.sig import lire_geotiff

    c, _ = reel
    A = _script("agri_national")
    idx = load_index(c)
    assert A.jours_de_base(idx, 1)                      # un jour de validation
    # cube GFS couvrant le mini-domaine et tous les instants
    t = pd.to_datetime(idx["timestamp"], utc=True).dt.tz_localize(None)
    temps = (t.astype("int64") // 10 ** 9).to_numpy()
    lat_g, lon_g = np.array([35.75, 36.0, 36.25]), np.array([9.75, 10.0, 10.25])
    n = len(temps)
    heures = t.dt.hour.to_numpy()[:, None, None]
    np.savez(tmp_path / "gfs.npz", temps=temps, lat=lat_g, lon=lon_g, elevation=np.zeros((3, 3)),
             temperature_2m=(15 + 8 * np.sin((heures - 9) * np.pi / 12) * np.ones((n, 3, 3))).astype(np.float32),
             relative_humidity_2m=np.full((n, 3, 3), 60.0, np.float32),
             shortwave_radiation=np.full((n, 3, 3), 250.0, np.float32))
    assert A.main(["produit", "-c", "x", "--device", "cpu", "--gfs", str(tmp_path / "gfs.npz"),
                   "--n-jours", "1", "--sortie", str(tmp_path / "p")]) == 0
    r = json.loads((tmp_path / "p" / "produit_agricole.json").read_text(encoding="utf-8"))
    base = r["jours_de_base"][0]
    presents = [j for j in base["jours"] if not j.get("absent")]
    assert presents and all(j["meteo"] == ["GFS"] and j["rayonnement_repli"] is False for j in presents)
    d = tmp_path / "p" / f"agri_{base['base']}"
    j0 = presents[0]
    z = np.load(d / f"agri_{j0['libelle'].replace('+', 'p')}_{j0['date']}.npz")
    tr = z["traitement"]
    assert tr.shape[0] == 24 and set(np.unique(tr)) <= {-1, 0, 1, 2}
    assert (tr[:, :, :] != -1).any() and (tr[0] == -1).all()       # 0 h locale : avant la 1re sortie (1 h)
    et0, _ = lire_geotiff(d / f"et0_mm_{j0['libelle'].replace('+', 'p')}_{j0['date']}.tif")
    assert np.isfinite(et0).any() and np.nanmin(et0) > 0
    a = pd.read_csv(d / "aeroports.csv")
    assert set(a["code"]) == {"DTTA", "INM60729", "INM60728"} and (d / "carte.png").exists()
    # sans cube GFS : repli signale
    assert A.main(["produit", "-c", "x", "--device", "cpu", "--n-jours", "1", "--sortie", str(tmp_path / "q")]) == 0
    r = json.loads((tmp_path / "q" / "produit_agricole.json").read_text(encoding="utf-8"))
    j = [x for x in r["jours_de_base"][0]["jours"] if not x.get("absent")][0]
    assert j["meteo"] == ["verite d'entrainement (ERA5)"] and j["rayonnement_repli"] is True


# ---------------------------------------------------------------------- phase 7 de bout en bout
def test_atlas_national_et_zoom(reel, config_fixe, tmp_path):
    from pcfno3d import atlas
    from pcfno3d.sig import ecrire_geotiff, lire_geotiff

    A = _script("atlas_eolien_national")
    assert A.main(["carte", "-c", "x", "--device", "cpu", "--max-instants", "2",
                   "--sortie", str(tmp_path / "carte")]) == 0
    s = tmp_path / "carte"
    out = json.loads((s / "atlas_national.json").read_text(encoding="utf-8"))
    assert out["hauteurs_agl_m"] == [50.0, 100.0, 150.0] and out["carte_principale"] == ["val", "test"]
    res = pd.read_csv(s / "resume.csv")
    assert set(res["source"]) == {"modele", "era5"} and set(res["carte"]) == {"principal", "information"}
    z = np.load(s / "atlas_national.npz")
    for h in (50, 100, 150):
        m, _ = lire_geotiff(s / f"moyenne_{h}m.tif")
        np.testing.assert_allclose(m, z[f"modele_principal_{h}m_moyenne"], rtol=1e-6)
        fc = z[f"modele_principal_{h}m_facteur_charge"]
        assert np.nanmin(fc) >= 0 and np.nanmax(fc) <= 1
    for f in ("atlas_national.png", "ecart_era5.png", "zones.csv", "blocs_5km.csv"):
        assert (s / f).exists()
    zones = pd.read_csv(s / "zones.csv")
    assert (zones["territoire"] == "territoire non verifie").all() if len(zones) else True
    with pytest.raises(SystemExit):
        A.main(["carte", "-c", "x", "--device", "cpu", "--max-instants", "2", "--sortie", str(s)])

    # densite par maille : c_n = c (rho / rho_ref)^(1/3) donne la production de weibull_aep a rho
    from pcfno3d.windfarm import weibull_aep
    courbe = atlas.charger_courbe(REPO / "configs" / "eolien" / "parc_demo.yaml")
    t = atlas.table_aep(courbe, courbe.rho_ref)
    rho = 1.05
    f = atlas.facteur_charge(np.array([2.2]), np.array([8.0 * (rho / courbe.rho_ref) ** (1 / 3)]), t)[0]
    assert f == pytest.approx(weibull_aep(courbe, 2.2, 8.0, rho)[1], rel=2e-3)

    # zoom : une carte fine (100 m) constante dans la boite d'une carte grossiere (1 km) constante
    nat = tmp_path / "nat"
    ecrire_geotiff(nat / "moyenne_100m.tif", np.full((20, 20), 7.0), 10.0, 36.0, 0.01, 0.009)
    gt = tmp_path / "gt"
    ecrire_geotiff(gt / "atlas_moyenne.tif", np.full((30, 40), 6.0), 10.05, 36.05, 0.001, 0.0009)
    (gt / "atlas.json").write_text(json.dumps({"hub_agl_m": 100.0, "instants": 12, "sites": []}))
    assert A.main(["zoom", "--grand-tunis", str(gt), "--national", str(nat), "--sortie", str(tmp_path / "z")]) == 0
    zj = json.loads((tmp_path / "z" / "zoom.json").read_text(encoding="utf-8"))
    assert zj["statut"] == "calcule" and zj["mailles_communes_1km"] > 0
    assert zj["ecart_moyen_national_moins_gt_m_s"] == pytest.approx(1.0) and zj["ecart_rms_m_s"] == pytest.approx(1.0)
    (gt / "atlas.json").write_text(json.dumps({"hub_agl_m": 80.0}))
    assert A.main(["zoom", "--grand-tunis", str(gt), "--national", str(nat), "--sortie", str(tmp_path / "z2")]) == 0
    assert json.loads((tmp_path / "z2" / "zoom.json").read_text(encoding="utf-8"))["statut"] == "absent"


def test_eol2_national(reel, config_fixe, tmp_path, monkeypatch):
    from pcfno3d import radiosondages
    from pcfno3d.geometry import Geometry

    c, obs = reel
    g = Geometry(c)
    # le site de Tunis-Carthage place dans le mini-domaine (les vraies coordonnees en sont loin)
    lat = LAT0 + 0.5 * g.ny * g.dy / 111_000
    lon = LON0 + 0.5 * g.nx * g.dx / (111_000 * math.cos(math.radians(LAT0)))
    monkeypatch.setitem(radiosondages.SITES, "TSM00060715", ("Tunis-Carthage", lat, lon))
    E = _script("eol2_national")
    assert E.main(["-c", "x", "--device", "cpu", "--obs", str(obs), "--sortie", str(tmp_path / "e")]) == 0
    p = pd.read_csv(tmp_path / "e" / "eol2_paires.csv")
    assert len(p) and set(p["site"]) == {"Tunis-Carthage"}
    assert p[["u_modele", "u_era5"]].notna().any().all()
    v = json.loads((tmp_path / "e" / "eol2.json").read_text(encoding="utf-8"))["EOL2"]
    assert v["tranche_m"] == [0.0, 250.0] and v["statut"] == "à mesurer"     # moins de 10 radiosondages
    assert v["radiosondages"] >= 1 and np.isfinite(v["rmse_vect_modele"])
    assert (tmp_path / "e" / "eol2_bilan.csv").exists()
    with pytest.raises(SystemExit):
        E.main(["-c", "x", "--device", "cpu", "--obs", str(obs), "--sortie", str(tmp_path / "e")])


# ---------------------------------------------------------------------- carnet
def test_carnet_phases_6_7():
    N = _script("make_national_agri_eol_nb")
    nb = N.construire("0123456789abcdef0123456789abcdef01234567")
    texte = "".join("".join(c["source"]) for c in nb["cells"])
    for f, h in N.empreintes().items():
        assert f in texte and h in texte
    assert "accord écrit d'Ines" in texte and "0123456789abcdef0123456789abcdef01234567" in texte
    assert "decision_v4_national.json" in texte and "choix_v4.json" in texte
    for s in ("agri_national.py decisions", "agri_national.py produit", "atlas_eolien_national.py carte",
              "atlas_eolien_national.py zoom", "eol2_national.py"):
        assert s in texte
    assert "train" not in [l.split()[-1] for l in texte.splitlines() if "pcfno3d -c" in l and l.split()]
    assert nb["metadata"]["accelerator"] == "GPU"
