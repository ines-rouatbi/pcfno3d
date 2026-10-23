"""Entree ERA5 du modele reel (pcfno3d.forcage) : profil aux noeuds, scalaires, sortie residuelle.

Forcage synthetique LINEAIRE en latitude, longitude, hauteur et temps : l'interpolation
(bilineaire dans le plan, lineaire en hauteur et en temps) doit le restituer exactement.
"""

from __future__ import annotations

import copy
import types

import numpy as np
import pandas as pd
import pytest
import torch

from pcfno3d.dataset import ANEO3D, assemble_input
from pcfno3d.forcage import N_SCAL, SCALAIRES, ForcageERA5
from pcfno3d.geometry import N_TFEAT, Geometry
from pcfno3d.model import build_model, save_inference_checkpoint
from pcfno3d.train import Trainer

LAT0, LON0 = 36.0, 10.0
HAUTEURS = np.array([5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 650, 800, 3000], float)
LAT_E = np.array([35.9, 36.0, 36.1, 36.2])
LON_E = np.array([9.9, 10.0, 10.1, 10.2])


def vent(h, lat, lon):
    """Champ de reference, lineaire : exactement interpolable."""
    return 2.0 + 0.01 * h + 30.0 * (lat - LAT0), -1.0 + 0.005 * h + 20.0 * (lon - LON0)


@pytest.fixture(scope="module")
def forcage(tmp_path_factory, index):
    d = tmp_path_factory.mktemp("forcage")
    t = pd.to_datetime(index["timestamp"])
    temps = pd.date_range(t.min().floor("D") - pd.Timedelta(hours=3),
                          t.max().ceil("D") + pd.Timedelta(hours=3), freq="3h")
    heures = ((temps - temps[0]) / pd.Timedelta(hours=1)).to_numpy()
    f = pd.DataFrame({"timestamp_utc": temps.strftime("%Y-%m-%dT%H:%M:%SZ")})
    for c, _ in SCALAIRES:
        f[c] = heures * 0.1          # lineaire en temps
    f.to_csv(d / "forcage_era5.csv", index=False)
    h, la, lo = np.meshgrid(HAUTEURS, LAT_E, LON_E, indexing="ij")
    u, v = vent(h, la, lo)
    n = len(temps)
    np.savez(d / "forcage_era5.npz", temps=(temps.astype("int64") // 10 ** 9).to_numpy(),
             lat=LAT_E, lon=LON_E, hauteurs_sol=HAUTEURS,
             u=np.broadcast_to(u, (n,) + u.shape).astype(np.float32),
             v=np.broadcast_to(v, (n,) + v.shape).astype(np.float32))
    return d / "forcage_era5.csv"


@pytest.fixture(scope="module")
def cfg_era5(cfg, forcage):
    c = copy.deepcopy(cfg)
    c.apply_overrides({"paths.forcage_era5": forcage, "real.origin_lat": LAT0,
                       "real.origin_lon": LON0})
    return c


@pytest.fixture(scope="module")
def geom_era5(cfg_era5):
    return Geometry(cfg_era5, device="cpu")


def _item(geom, index):
    train = index[index["split"] == "train"]
    ds = ANEO3D(train["stem"].values, train["hour"].values, train["doy"].values,
                geom, train=False, seed=0)
    return ds[0], train.iloc[0]


def test_canaux_et_vecteur_tfeat(geom, geom_era5, index):
    assert geom_era5.c_in == geom.c_in + N_SCAL + 2
    (_, _, tfeat, _, _), row = _item(geom_era5, index)
    assert tfeat.shape == (N_TFEAT + geom_era5.forcage.n_extra,)
    # scalaires : interpolation lineaire en temps, puis facteur d'echelle
    t0 = pd.Timestamp(pd.read_csv(geom_era5.cfg.paths.forcage_era5)["timestamp_utc"][0])
    heures = (pd.Timestamp(row["timestamp"], tz="UTC") - t0) / pd.Timedelta(hours=1)
    attendu = np.array([heures * 0.1 * k for _, k in SCALAIRES], np.float32)
    np.testing.assert_allclose(tfeat[N_TFEAT:N_TFEAT + N_SCAL].numpy(), attendu, rtol=1e-5)


def test_vent_era5_exact_aux_noeuds(geom_era5, index, norm):
    (s_raw, sensor, tfeat, _, _), _ = _item(geom_era5, index)
    g = geom_era5
    x = assemble_input(s_raw[None], sensor[None], tfeat[None], g, norm)
    assert x.shape == (1, g.c_in, *g.shape)
    it = g.interp_era5
    h = np.maximum(np.asarray(g.z_levels)[:, None, None] - g.zt[None], 5.0)
    u_ref, v_ref = vent(h, it.lat[None], it.lon[None])
    dense = it.dense(tfeat[None, N_TFEAT + N_SCAL:])[0].numpy()
    np.testing.assert_allclose(dense[0], u_ref, atol=1e-4)
    np.testing.assert_allclose(dense[1], v_ref, atol=1e-4)
    np.testing.assert_allclose(it.dense_np(tfeat[N_TFEAT + N_SCAL:].numpy()), dense, atol=1e-4)
    # canaux d'entree : normalises comme la cible, 0 sous le relief
    mu, sd = norm.y_mu[:2].numpy(), norm.y_sd[:2].numpy()
    attendu = (dense - mu[:, None, None, None]) / sd[:, None, None, None] * g.valid
    np.testing.assert_allclose(x[0, -2:].numpy(), attendu, atol=1e-5)


def test_instant_sans_forcage_refuse(geom_era5):
    with pytest.raises(ValueError, match="3 h"):
        geom_era5.forcage.vecteur("2019-06-01 12:00:00")


def test_residuel_exige_le_forcage(cfg):
    c = copy.deepcopy(cfg)
    c.model.residuel_era5 = True
    with pytest.raises(ValueError, match="forcage_era5"):
        build_model(c, 25)


def test_affinage_depuis_un_modele_25_canaux(cfg, cfg_era5, geom, geom_era5, index, norm,
                                             tmp_path):
    """init_from : colonnes existantes gardees, nouvelles a 0, u, v de depart = ERA5."""
    torch.manual_seed(0)
    ancien = build_model(cfg, geom.c_in)
    ck = save_inference_checkpoint(tmp_path / "ancien.pt", ancien, cfg, geom.c_in)

    c = copy.deepcopy(cfg_era5)
    c.model.residuel_era5 = True
    hote = types.SimpleNamespace(cfg=c, device=torch.device("cpu"), verbose=False,
                                 model=build_model(c, geom_era5.c_in))
    Trainer._init_from(hote, ck)
    w = hote.model.state_dict()["lift.0.weight"]
    np.testing.assert_array_equal(w[:, :geom.c_in].numpy(),
                                  ancien.state_dict()["lift.0.weight"].numpy())
    assert (w[:, geom.c_in:] == 0).all()

    (s_raw, sensor, tfeat, _, _), _ = _item(geom_era5, index)
    with torch.no_grad():
        x = assemble_input(s_raw[None], sensor[None], tfeat[None], geom_era5, norm)
        u, v, w_ = norm.denorm(hote.model.eval()(x))
        u_old, v_old, w_old = norm.denorm(ancien.eval()(x[:, :geom.c_in]))
    dense = geom_era5.interp_era5.dense(tfeat[None, N_TFEAT + N_SCAL:])[0]
    valid = torch.from_numpy(geom_era5.valid)
    torch.testing.assert_close(u[0][valid], dense[0][valid], atol=1e-4, rtol=0)
    torch.testing.assert_close(v[0][valid], dense[1][valid], atol=1e-4, rtol=0)
    torch.testing.assert_close(w_, w_old, atol=1e-5, rtol=0)     # w : inchange au depart


def test_entrainement_puis_service_de_bout_en_bout(cfg_era5, index, tmp_path):
    """Entrainement court (entree ERA5 + sortie residuelle), puis le service relit le
    checkpoint : architecture residuelle restituee, l'instant donne le forcage."""
    from pcfno3d.inference import WindFieldModel

    c = copy.deepcopy(cfg_era5)
    c.apply_overrides({"paths.run_dir": tmp_path / "run", "model.residuel_era5": True})
    Trainer(c, device="cpu", verbose=False).fit()
    ck = torch.load(c.paths.ckpt_best, map_location="cpu", weights_only=False)
    assert ck["config"]["RESIDUEL_ERA5"] is True
    assert ck["config"]["C_IN"] == Geometry(c).c_in

    c2 = copy.deepcopy(c)
    c2.model.residuel_era5 = False               # le checkpoint fait foi
    wf = WindFieldModel(c2, device="cpu", verbose=False)
    assert wf.model.residuel
    stem = index[index["split"] == "test"]["stem"].iloc[0]
    vol = wf.predict_stem(stem)
    assert np.isfinite(vol.u[wf.geom.valid]).all()
    S, hour, doy, _ = wf.surface_from_cache(stem)
    with pytest.raises(ValueError, match="timestamp"):
        wf.predict(S, hour, doy)                  # sans horodatage : pas de forcage


    # l'evaluation complete (DSO1-DSO5, dont la mesure de temps DSO4) passe le forcage partout
    from pcfno3d.evaluate import Evaluator
    summary = Evaluator(c2, device="cpu", verbose=False).run()
    assert summary


def test_validation_reelle_de_bout_en_bout(cfg_era5, index, tmp_path):
    """scripts/valider_grand_tunis_reel.py sur le mini-jeu : entree era5_stations avec des
    mesures synthetiques aux 3 stations, un radiosondage ; bilans ecrits, sans NaN."""
    import importlib.util
    import json
    import shutil
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("valider", repo / "scripts" / "valider_grand_tunis_reel.py")
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)

    c = copy.deepcopy(cfg_era5)
    data = tmp_path / "jeu"
    shutil.copytree(c.paths.data_dir, data)
    c.apply_overrides({"paths.data_dir": data, "paths.run_dir": tmp_path / "run",
                       "mask.entree": "era5_stations", "mask.pas_era5_m": 3 * Geometry(c).dx,
                       "model.residuel_era5": True})
    g = Geometry(c)
    jj, ii = np.nonzero(g.valid.any(axis=0))
    cells = [(int(jj[k]), int(ii[k])) for k in (0, len(jj) // 2, len(jj) - 1)]
    noms = ["DTTA Tunis-Carthage", "Zaghouan (INM 60729)", "Nabeul (INM 60728)"]
    (data / "stations.json").write_text(json.dumps(
        [{"nom": n, "iy": a, "ix": b} for n, (a, b) in zip(noms, cells)]), encoding="utf-8")
    for stem in index["stem"]:
        S = np.load(c.paths.cache_dir / f"{stem}_S.npy")
        np.save(c.paths.cache_dir / f"{stem}_E.npy", S[:2].astype(np.float16))
    Trainer(c, device="cpu", verbose=False).fit()

    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    test = index[index["split"] == "test"]
    lignes, sondes = [], []
    for t in pd.to_datetime(test["timestamp"]):
        for code in ("DTTA", "INM60729", "INM60728"):
            lignes.append({"station_id": code, "timestamp_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "wind_speed_m_s": 4.0, "wind_dir_deg": 270.0,
                           "wind_u_m_s": 4.0, "wind_v_m_s": 0.0})
        if t.hour in (0, 12):
            for h in (50.0, 200.0, 600.0):
                sondes.append({"temps": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "h_agl_m": h,
                               "u": 5.0, "v": 1.0})
    pd.DataFrame(lignes).to_csv(obs_dir / "metar_2025.csv", index=False)
    pd.DataFrame(sondes).to_csv(obs_dir / "radiosondages_TSM00060715.csv", index=False)

    resume = V.valider(c, obs_dir, "test", device="cpu")
    sortie = c.paths.run_dir / "validation_reelle" / "test"
    b = pd.read_csv(sortie / "stations_bilan.csv")
    assert set(b["station"]) == {"Tunis-Carthage", "Zaghouan", "Nabeul"}
    assert set(b["source"]) == {"modele", "era5", "generateur"}
    assert b["rmse_vect"].notna().all()
    assert "stations" in resume
    if sondes:
        assert (sortie / "radiosondages_bilan.csv").exists()
