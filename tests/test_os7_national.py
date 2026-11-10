"""OS7 NATIONAL (phase 4 de la feuille de route nationale) : forcage GFS du modele national decoupe dans la
boite nationale (memes mailles, memes hauteurs, memes formules que scripts/forcage_era5.py), reprise sur
l'archive du Grand Tunis sans y ecrire, PREV aux aeroports nationaux, regle hachee, verdicts, carnet.

Aucun acces reseau : le telechargement GFS est simule, l'ERA5 est synthetique (meme atmosphere des deux
cotes : le forcage GFS doit retrouver EXACTEMENT le forcage ERA5)."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_forcage_era5 import cfg_era5, forcage  # noqa: F401  (fixtures)
from test_os7_prevision import jeu_gfs  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]
G = 9.80665
NIVEAUX_NATIONAUX = (25., 50., 75., 100., 150., 200., 300., 400., 500., 650., 800., 1000., 1250., 1500., 2000., 2500.)
# petite zone au sud-ouest de la boite nationale : avec la marge de 0,15 deg de forcage_era5.py, 5 x 5 mailles
ZONE = {"lat_sud": 30.15, "lat_nord": 30.85, "lon_ouest": 7.65, "lon_est": 8.35}
INSTANTS = list(pd.date_range("2022-01-10 00:00", periods=3, freq="3h"))


def _module(nom: str, fichier: str):
    spec = importlib.util.spec_from_file_location(nom, REPO / "scripts" / fichier)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------- une atmosphere synthetique
def _atmosphere(t: pd.Timestamp, lat: np.ndarray, lon: np.ndarray) -> dict:
    """Champs physiques (unites ERA5) a l'instant t sur la grille (lat, lon) : relief de 200 a 1000 m (les
    hauteurs nationales de 2000-2500 m au-dessus du sol passent au-dessus de 700 hPa), mer a l'est."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    k = (t - INSTANTS[0]) / pd.Timedelta(hours=1)
    oro = np.clip(200.0 + 800.0 * (la - 30.0), 0.0, None)
    a = {"oro": oro, "lsm": (lo < 8.2).astype(float),
         "t2m": 285.0 + 2.0 * (lo < 8.2) - 0.006 * oro + 0.1 * k, "sp": 101300.0 - 11.0 * oro,
         "msl": 101500.0 + 10.0 * k + 0 * la,
         "u10": 3.0 + 2.0 * (la - 30.0) + 0.1 * k, "v10": -1.0 + 1.5 * (lo - 7.0),
         "u100": 5.0 + 2.0 * (la - 30.0) + 0.1 * k, "v100": -0.5 + 1.5 * (lo - 7.0)}
    for p in (1000, 975, 950, 925, 900, 850, 800, 750, 700):
        dp = 1000 - p
        a[f"z{p}"] = 110.0 + 9.5 * dp + 5.0 * (la - 30.0) + 0.2 * k     # m au-dessus de la mer
        a[f"u{p}"] = 6.0 + 0.04 * dp + 1.5 * (la - 30.0) - 0.05 * k
        a[f"v{p}"] = 1.0 + 0.02 * dp + 2.0 * (lo - 7.0) + 0.03 * k
    return a


def _era5(dossier: Path) -> Path:
    """Fichiers ERA5 synthetiques (latitudes decroissantes, comme le CDS) de la petite zone avec sa marge."""
    import xarray as xr

    lat = np.round(np.arange(31.0, 29.99, -0.25), 4)
    lon = np.round(np.arange(7.5, 8.51, 0.25), 4)
    niveaux = [1000, 975, 950, 925, 900, 850, 800, 750, 700]
    A = [_atmosphere(t, lat, lon) for t in INSTANTS]
    co = {"valid_time": INSTANTS, "latitude": lat, "longitude": lon}
    pl = xr.Dataset({n: (("valid_time", "pressure_level", "latitude", "longitude"),
                         np.stack([np.stack([a[f"{n}{p}"] * (G if n == "z" else 1.0) for p in niveaux]) for a in A]))
                     for n in ("u", "v", "z")}, coords=dict(co, pressure_level=niveaux))
    sl = xr.Dataset({n: (("valid_time", "latitude", "longitude"), np.stack([a[n] for a in A]))
                     for n in ("u10", "v10", "u100", "v100", "t2m", "sp", "msl")}, coords=co)
    sl["z"] = (("valid_time", "latitude", "longitude"), np.stack([a["oro"] * G for a in A]))
    dossier.mkdir(parents=True, exist_ok=True)
    pl.to_netcdf(dossier / "era5_plx_2022_01.nc")
    sl.to_netcdf(dossier / "era5_sl_2022_01.nc")
    xr.Dataset({"lsm": (("latitude", "longitude"), A[0]["lsm"])},
               coords={"latitude": lat, "longitude": lon}).to_netcdf(dossier / "masque_terre_mer.nc")
    return dossier


def _faux_gfs(appels: list):
    """telecharger_zones simule : la MEME atmosphere, au format « Open-Meteo » de pcfno3d.gfs.extraire."""
    from pcfno3d.gfs import NIVEAUX_GFS

    def telecharger_zones(run, echeance_h, zones, dossier):
        appels.append((run, echeance_h, tuple(zones)))
        t = pd.Timestamp(run) + pd.Timedelta(hours=echeance_h)
        out = {}
        for nom, (la, lo) in zones.items():
            a = _atmosphere(t, np.asarray(la), np.asarray(lo))

            def sd(u, v):
                return (np.hypot(u, v)[None], ((270.0 - np.degrees(np.arctan2(v, u))) % 360.0)[None])
            c = {"elevation": a["oro"], "terre": a["lsm"] >= 0.5,
                 "temperature_2m": (a["t2m"] - 273.15)[None], "relative_humidity_2m": np.full((1,) + a["oro"].shape, 50.0),
                 "surface_pressure": (a["sp"] / 100.0)[None], "pressure_msl": (a["msl"] / 100.0)[None],
                 "shortwave_radiation": np.zeros((1,) + a["oro"].shape), "source": "test"}
            c["wind_speed_10m"], c["wind_direction_10m"] = sd(a["u10"], a["v10"])
            c["wind_speed_100m"], c["wind_direction_100m"] = sd(a["u100"], a["v100"])
            for p in NIVEAUX_GFS:
                c[f"wind_speed_{p}hPa"], c[f"wind_direction_{p}hPa"] = sd(a[f"u{p}"], a[f"v{p}"])
                c[f"geopotential_height_{p}hPa"] = a[f"z{p}"][None]
            out[nom] = c
        return out
    return telecharger_zones


@pytest.fixture(scope="module")
def era5_national(tmp_path_factory):
    """Le forcage ERA5 « national » (16 niveaux jusqu'a 2500 m) calcule par scripts/forcage_era5.py."""
    d = tmp_path_factory.mktemp("era5_national")
    fe = _module("forcage_era5", "forcage_era5.py")
    df = fe.forcage(_era5(d / "era5"), {"zone": ZONE}, slx=False, niveaux=NIVEAUX_NATIONAUX)
    df.to_csv(d / "forcage_era5.csv", index=False, float_format="%.6g")
    np.savez_compressed(d / "forcage_era5.npz", **df.attrs["champs"])
    return d / "forcage_era5.csv", df


# ---------------------------------------------------------------------- forcage GFS national
def test_indices_boite():
    from pcfno3d.gfs import grille_tunisie, indices_boite

    lat_tn, lon_tn = grille_tunisie()
    # mailles du forcage ERA5 national : 30,15-37,615 N et 7,45-11,644 E avec 0,15 deg de marge
    lat = np.round(np.arange(30.0, 37.76, 0.25), 4)
    lon = np.round(np.arange(7.5, 11.76, 0.25), 4)
    iy, ix = indices_boite(lat, lon)
    np.testing.assert_allclose(lat_tn[iy], lat)
    np.testing.assert_allclose(lon_tn[ix], lon)
    with pytest.raises(ValueError, match="hors des noeuds"):
        indices_boite(np.array([30.1]), np.array([8.0]))            # pas sur la grille 0,25 deg
    with pytest.raises(ValueError, match="hors des noeuds"):
        indices_boite(np.array([30.0]), np.array([12.5]))           # hors de la boite


def test_forcage_gfs_national_egal_au_forcage_era5(era5_national, tmp_path, monkeypatch):
    """Meme atmosphere des deux cotes : le forcage GFS national (decoupe dans la boite, formules de
    temps_reel.forcage_gfs) redonne le forcage ERA5 de scripts/forcage_era5.py : memes mailles, memes 17
    hauteurs au-dessus du sol (comblement au-dessus de 700 hPa compris), memes 7 scalaires."""
    from pcfno3d import gfs
    from pcfno3d.forcage import SCALAIRES, ForcageERA5

    csv_era5, df_era5 = era5_national
    ref = ForcageERA5(csv_era5)
    assert ref.hauteurs[-1] == 2500.0 and len(ref.hauteurs) == 17
    appels = []
    monkeypatch.setattr(gfs, "telecharger_zones", _faux_gfs(appels))

    # archive du Grand Tunis : le premier instant y a deja sa boite nationale (lue seulement)
    gt = tmp_path / "gt_os7" / "gfs_f24_instants"
    gt.mkdir(parents=True)
    run0, l0 = gfs.run_pour(INSTANTS[0], 24)
    la_tn, lo_tn = gfs.grille_tunisie()
    c0 = _faux_gfs([])(run0, l0, {"tunisie": (la_tn, lo_tn)}, None)["tunisie"]
    gfs._ecrire_national(gfs.fichier_national(gt, INSTANTS[0], 24), c0, la_tn, lo_tn, run0, l0)
    avant = {f.name: f.stat().st_mtime_ns for f in gt.iterdir()}

    # un instant de validite = instant ERA5 : on reprend ici t = run + echeance (le faux GFS est l'atmosphere a t)
    sortie = tmp_path / "nat" / "forcage_gfs_f24.csv"
    df = gfs.construire_national(INSTANTS, 24, ref.lat, ref.lon, ref.hauteurs, sortie, tmp_path / "nat" / "tn",
                                 lecture=[gt], travailleurs=1, verbose=False)
    assert len(appels) == 2 and all(z == ("tunisie",) for *_, z in appels)   # boite seule, pas l'instant du GT
    assert {f.name: f.stat().st_mtime_ns for f in gt.iterdir()} == avant     # rien ecrit dans l'archive GT
    assert len(list((tmp_path / "nat" / "tn").glob("*_tn.npz"))) == 2

    f = ForcageERA5(sortie)
    assert f.forme_cellules == ref.forme_cellules
    np.testing.assert_allclose(f.lat, ref.lat)
    np.testing.assert_allclose(f.lon, ref.lon)
    np.testing.assert_array_equal(f.temps, ref.temps)
    np.testing.assert_allclose(f.u, ref.u, atol=2e-4)
    np.testing.assert_allclose(f.v, ref.v, atol=2e-4)
    for c, _ in SCALAIRES:
        np.testing.assert_allclose(df[c].to_numpy(float), df_era5[c].to_numpy(float), rtol=1e-4, atol=1e-6,
                                   err_msg=c)
    assert {"run_utc", "echeance_h"} <= set(df.columns)
    assert list(df["echeance_h"]) == [gfs.run_pour(t, 24)[1] for t in INSTANTS]          # 24, 27, 24

    # relance : tout est la, rien n'est telecharge
    appels.clear()
    gfs.construire_national(INSTANTS, 24, ref.lat, ref.lon, ref.hauteurs, tmp_path / "nat" / "b.csv",
                            tmp_path / "nat" / "tn", lecture=[gt], travailleurs=1, verbose=False)
    assert appels == []


def test_forcage_national_trous_ecrits(era5_national, tmp_path, monkeypatch):
    """Une echeance absente de l'archive : les autres instants sont ecrits, l'absente est listee."""
    from pcfno3d import gfs
    from pcfno3d.forcage import ForcageERA5

    ref = ForcageERA5(era5_national[0])
    vrai = _faux_gfs([])

    def avec_trou(run, echeance_h, zones, dossier):
        if pd.Timestamp(run) + pd.Timedelta(hours=echeance_h) == INSTANTS[1]:
            raise FileNotFoundError("GFS introuvable")
        return vrai(run, echeance_h, zones, dossier)
    monkeypatch.setattr(gfs, "telecharger_zones", avec_trou)
    sortie = tmp_path / "f.csv"
    df = gfs.construire_national(INSTANTS, 24, ref.lat, ref.lon, ref.hauteurs, sortie, tmp_path / "tn",
                                 travailleurs=1, verbose=False)
    assert len(df) == 2
    assert len(json.loads(sortie.with_suffix(".erreurs.json").read_text(encoding="utf-8"))) == 1


def test_main_national(era5_national, tmp_path, monkeypatch):
    """python -m pcfno3d.gfs --national : instants de l'index du cache, mailles et hauteurs du forcage ERA5
    de la configuration, controle de forme, aucun cube (la boite reste par instant)."""
    import yaml

    from pcfno3d import gfs

    csv_era5, _ = era5_national
    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame({"stem": [f"s{k}" for k in range(3)], "day": [0, 0, 0],
                  "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in INSTANTS],
                  "split": ["train", "val", "test"]}).to_csv(cache / "index.csv", index=False)
    conf = tmp_path / "national_test.yaml"
    conf.write_text(yaml.safe_dump({"name": "national_test", "paths": {
        "data_dir": str(tmp_path / "jeu"), "cache_dir": str(cache), "run_dir": str(tmp_path / "run"),
        "forcage_era5": str(csv_era5)}}), encoding="utf-8")
    appels = []
    monkeypatch.setattr(gfs, "telecharger_zones", _faux_gfs(appels))
    sortie = tmp_path / "os7" / "forcage_gfs_f48.csv"
    monkeypatch.setattr(sys, "argv", ["gfs", "--config", str(conf), "--national", "--echeance", "48",
                                      "--sortie", str(sortie), "--travailleurs", "1"])
    assert gfs.main() == 0
    assert len(appels) == 3 and all(z == ("tunisie",) for *_, z in appels)
    assert len(pd.read_csv(sortie)) == 3 and sortie.with_suffix(".npz").exists()
    assert not list(sortie.parent.glob("gfs_tunisie_*.npz"))


# ---------------------------------------------------------------------- configuration et regle
def test_configuration_v5_nationale():
    import dataclasses

    from pcfno3d.config import Config

    v5, nat = Config.load("national_v5_gfs_colab"), Config.load("national_1km_colab")
    for sec in ("model", "dso", "real"):                   # meme modele et meme domaine que le national
        assert dataclasses.asdict(getattr(v5, sec)) == dataclasses.asdict(getattr(nat, sec)), sec
    assert v5.mask.e_source == "forcage" and v5.mask.stations_entree is False
    assert v5.mask.entree == "era5_stations" and v5.mask.pas_era5_m == nat.mask.pas_era5_m
    assert v5.mask.aneo_capteurs == (0, 0) and v5.mask.aneo_degradation is None
    assert Path(v5.paths.run_dir).name == "national_1km_v5_gfs"
    assert Path(v5.paths.forcage_era5).name == "forcage_gfs_f24.csv"
    assert v5.paths.data_dir == nat.paths.data_dir and v5.paths.cache_dir == nat.paths.cache_dir
    assert v5.train.select_by == "dso" and v5.train.epochs == 12 and v5.loss.lambda_rough == 1.0


def test_regle_os7_hachee_et_proposee():
    f = REPO / "configs" / "national" / "regle_os7_national.json"
    attendu = (f.parent / (f.name + ".sha256")).read_text(encoding="ascii").split()
    assert attendu[1] == f.name
    assert hashlib.sha256(f.read_bytes()).hexdigest() == attendu[0]
    r = json.loads(f.read_text(encoding="utf-8"))
    assert "PROPOSITION soumise a l'accord ecrit d'Ines" in r["statut"]
    assert r["fixee_le"].startswith("2026-11-11")
    assert {"PREV1", "PREV2", "PREV3"} <= set(r["criteres"])
    # les seuils du juge sont ceux ecrits dans la regle
    j = _module("juger", "juger_os7_national.py")
    p3 = json.dumps(r["criteres"]["PREV3"])
    assert f"au moins {j.PREV3_MIN_PAIRES} paires" in p3 and f"au moins {j.PREV3_MIN_STATIONS} aeroports" in p3
    assert f"<= {round(j.SEUIL_FAUX_OUI * 100)} %" in p3
    assert j.verifier_regle() == attendu[0]


def test_regle_modifiee_refusee(tmp_path):
    j = _module("juger", "juger_os7_national.py")
    f = tmp_path / "regle_os7_national.json"
    shutil.copy2(REPO / "configs" / "national" / "regle_os7_national.json", f)
    shutil.copy2(REPO / "configs" / "national" / "regle_os7_national.json.sha256", tmp_path / (f.name + ".sha256"))
    assert j.verifier_regle(f)
    f.write_text(f.read_text(encoding="utf-8").replace("12 epoques", "20 epoques"), encoding="utf-8")
    with pytest.raises(SystemExit, match="modifiee"):
        j.verifier_regle(f)


# ---------------------------------------------------------------------- verdicts
def _bilan(e_prev, e_gfs, e_pers, code):
    return {"critere": code, "paires": 300, "instants": 30, "verdicts": {"toutes": {
        "rmse_vect": {"prevision": e_prev, "gfs": e_gfs, "persistance": e_pers},
        "mieux_que_gfs": e_prev < e_gfs, "mieux_que_persistance": e_prev < e_pers,
        "stations": 12, "stations_mieux_que_gfs": 9, "stations_mieux_que_persistance": 12}}}


def test_juger_os7_national(tmp_path):
    j = _module("juger", "juger_os7_national.py")
    os7 = tmp_path / "os7"
    v = j.juger(os7)
    assert v["PREV1"]["verdict"] == v["PREV2"]["verdict"] == v["PREV3"]["verdict"] == "non mesure"

    for d, b in (("v5_f24", _bilan(2.6, 2.9, 4.1, "PREV1")), ("v5_f48", _bilan(3.2, 3.1, 4.5, "PREV2")),
                 ("sans_affinage_f24", _bilan(2.8, 2.9, 4.1, "PREV1"))):
        (os7 / d).mkdir(parents=True)
        (os7 / d / "prevision_bilan.json").write_text(json.dumps(b), encoding="utf-8")
    stations = [f"A{k}" for k in range(8)]
    pd.DataFrame({"station": np.repeat(stations, 15), "u_obs": 1.0, "v_obs": 0.0}).to_csv(
        os7 / "v5_f24" / "prevision_paires.csv", index=False)
    (os7 / "v5_f24" / "decisions").mkdir()
    dec = {"agri1": {"prevision / toutes heures": {"accord": 0.66, "kappa": 0.36},
                     "gfs / toutes heures": {"accord": 0.63, "kappa": 0.30},
                     "persistance / toutes heures": {"accord": 0.61, "kappa": 0.25}},
           "agri2": {"prevision / toutes heures": {"faux_oui": 0.03, "conforme": True}}}
    (os7 / "v5_f24" / "decisions" / "decisions_agricoles.json").write_text(json.dumps(dec), encoding="utf-8")
    v = j.juger(os7)
    assert v["PREV1"]["verdict"] == "conforme" and v["PREV2"]["verdict"] == "non conforme"
    assert v["PREV3"]["verdict"] == "conforme" and v["PREV3"]["aeroports"] == 8
    assert v["rapporte"]["reference_sans_affinage_f24"]["verdict"] == "conforme"
    assert v["rapporte"]["reference_sans_affinage_f48"]["verdict"] == "non mesure"
    # faux oui au-dessus de 5 % : non conforme meme avec un meilleur accord
    dec["agri2"]["prevision / toutes heures"]["faux_oui"] = 0.06
    (os7 / "v5_f24" / "decisions" / "decisions_agricoles.json").write_text(json.dumps(dec), encoding="utf-8")
    assert j.juger(os7)["PREV3"]["verdict"] == "non conforme"
    # trop peu d'aeroports : PREV3 non mesurable
    pd.DataFrame({"station": np.repeat(stations[:5], 30), "u_obs": 1.0, "v_obs": 0.0}).to_csv(
        os7 / "v5_f24" / "prevision_paires.csv", index=False)
    assert j.juger(os7)["PREV3"]["verdict"] == "non mesurable"
    assert "PREV1" in j.en_markdown(j.juger(os7))


# ---------------------------------------------------------------------- PREV aux aeroports nationaux
def test_evaluer_prevision_aeroports_nationaux(jeu_gfs, tmp_path):  # noqa: F811
    """evaluer_prevision.py lit les stations de stations.json (codes des aeroports nationaux), pas les 3 du
    Grand Tunis seulement ; noms tires de stations.json ; IC bootstrap des ecarts."""
    _, p, _, _ = jeu_gfs
    c = copy.deepcopy(p)
    data = tmp_path / "jeu"
    shutil.copytree(c.paths.data_dir, data)
    c.apply_overrides({"paths.data_dir": data})
    ancien = json.loads((p.paths.data_dir / "stations.json").read_text(encoding="utf-8"))
    codes = [("DTTX", "DTTX Sfax"), ("DTTZ", "DTTZ Tozeur"), ("DTKA", "DTKA Tabarka")]
    (data / "stations.json").write_text(json.dumps(
        [{"nom": n, "code": k, "iy": s["iy"], "ix": s["ix"]} for (k, n), s in zip(codes, ancien)]), encoding="utf-8")
    index = pd.read_csv(c.paths.cache_dir / "index.csv")
    obs = tmp_path / "obs"
    obs.mkdir()
    lignes = []
    for t in pd.to_datetime(index["timestamp"]):
        for t2 in (t, t - pd.Timedelta(hours=24)):
            for k, (code, _) in enumerate(codes + [("DTMB", "")]):          # DTMB : hors du domaine, ignore
                lignes.append({"station_id": code, "timestamp_utc": t2.strftime("%Y-%m-%dT%H:%M:%SZ"),
                               "wind_speed_m_s": 3.0 + k, "wind_dir_deg": 250.0, "wind_u_m_s": np.nan,
                               "wind_v_m_s": np.nan})
    pd.DataFrame(lignes).drop_duplicates(["station_id", "timestamp_utc"]).to_csv(obs / "metar_2022.csv", index=False)

    m = _module("ep_nat", "evaluer_prevision.py")
    paires, bilan, resume = m.evaluer(c, obs, 24, device="cpu", verbose=False, bootstrap=200)
    assert set(paires["station"]) == {"Sfax", "Tozeur", "Tabarka"}
    assert paires[["u_persistance", "v_persistance"]].notna().all().all()
    t = resume["verdicts"]["toutes"]
    assert t["stations"] == 3 and 0 <= t["stations_mieux_que_gfs"] <= 3
    ic = t["ic95_ecart"]["gfs"]
    assert ic["tirages"] == 200 and ic["ic95"][0] <= ic["ecart"] <= ic["ic95"][1]
    assert ic["ecart"] == pytest.approx(t["rmse_vect"]["prevision"] - t["rmse_vect"]["gfs"], abs=1e-9)
    json.dumps(resume, default=str)


# ---------------------------------------------------------------------- carnet
def test_carnet_os7_national():
    nb_mod = _module("nb_os7", "make_national_os7_nb.py")
    commit = "0123456789abcdef0123456789abcdef01234567"
    nb = nb_mod.os7_national(commit)
    texte = "".join("".join(c["source"]) for c in nb["cells"])
    tete = "".join(nb["cells"][0]["source"])
    assert "NE PAS LANCER sans l'accord écrit d'Ines" in tete and "PROPOSITION soumise à l'accord écrit" in tete
    for nom in ("regle_v4_national.json", "regle_os7_national.json"):
        h = hashlib.sha256((REPO / "configs" / "national" / nom).read_bytes()).hexdigest()
        assert f'("{nom}", "{h}")' in texte                               # empreinte verifiee par le carnet
    assert f'COMMIT = "{commit}"' in texte and "@" + "SHA" not in texte and "@COMMIT@" not in texte
    assert 'ACCORD_ECRIT_INES = ""' in texte and "raise SystemExit(\"accord ecrit d'Ines absent" in texte
    for attendu in ("--national", "--lecture", "national_v5_gfs_colab", "choix_v4.json", "cache_v3_partie",
                    "evaluer_prevision.py", "verifier_decisions.py", "juger_os7_national.py", "runtime.unassign()"):
        assert attendu in texte, attendu
    assert nb["metadata"]["colab"]["gpuType"] == "A100"
    # chaque cellule de code est du Python valide (hors commandes IPython « ! » et « % »)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(l for l in c["source"] if not l.lstrip().startswith(("!", "%")))
            compile(src, "<cellule>", "exec")


def test_carnet_os7_national_a_jour():
    """Le carnet du depot est celui que le generateur produit pour son commit (regles et code inchanges)."""
    f = REPO / "notebooks" / "current" / "pcfno3d_national_os7_colab.ipynb"
    if not f.exists():
        pytest.skip("carnet pas encore genere")
    commit = re.search(r'COMMIT = \\"([0-9a-f]{40})\\"', f.read_text(encoding="utf-8"))
    assert commit, "commit fige absent du carnet"
    nb = _module("nb_os7b", "make_national_os7_nb.py").os7_national(commit[1])
    assert json.loads(f.read_text(encoding="utf-8")) == nb
