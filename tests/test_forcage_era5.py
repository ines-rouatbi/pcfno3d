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


def test_predicteur_era5_et_verdict_national(cfg_era5, index, norm, tmp_path):
    """La reference ERA5 interpolee rend exactement le vent ERA5 aux noeuds valides ; le
    verdict national s'ecrit apres l'evaluation, avec un gain sur ERA5 par niveau."""
    import json

    from pcfno3d.dso_national import predicteur_era5, verdict_national
    from pcfno3d.evaluate import Evaluator

    g = Geometry(cfg_era5, device="cpu")
    (s_raw, sensor, tfeat, y, _), _ = _item(g, index)
    u, v, w = predicteur_era5(g, norm)(s_raw[None], sensor[None], tfeat[None], y[None], 0, "x")
    dense = g.interp_era5.dense(tfeat[None, N_TFEAT + N_SCAL:])[0]
    valid = torch.from_numpy(g.valid)
    torch.testing.assert_close(u[0][valid], dense[0][valid], atol=1e-4, rtol=0)
    torch.testing.assert_close(v[0][valid], dense[1][valid], atol=1e-4, rtol=0)
    assert (u[0][~valid] == 0).all() and w.shape == u.shape

    c = copy.deepcopy(cfg_era5)
    c.apply_overrides({"paths.run_dir": tmp_path / "run", "model.residuel_era5": True})
    Trainer(c, device="cpu", verbose=False).fit()
    Evaluator(c, device="cpu", verbose=False).run()
    v_nat = verdict_national(c, device="cpu", verbose=False)
    out = next((tmp_path / "run" / "eval_pcfno3d").glob("*/verdict_national.json"))
    lu = json.loads(out.read_text(encoding="utf-8"))
    assert lu["conformes"] == v_nat["conformes"]
    couverts = [r for r in lu["par_niveau"] if r["couverture"] > 0]   # 25 m : sous le relief
    assert couverts and all(np.isfinite(r["rmse_vect_era5"]) and np.isfinite(r["gain_era5"])
                            for r in couverts)
    assert lu["dso7_conforme"] is None                       # pas de validation reelle ici
    assert (out.parent / "verdict_national.md").exists()

    # DSO par tranches au-dessus du sol : modele et ERA5 sur les memes noeuds
    from pcfno3d.dso_tranches import dso_tranches, masques_tranches
    bornes = (0.0, 60.0, 150.0, 400.0)
    m = masques_tranches(g, bornes)
    total = sum(x.sum() for _, x in m)
    assert total == g.valid.sum()                        # chaque noeud valide dans une tranche
    v_t = dso_tranches(c, device="cpu", bornes=bornes, verbose=False)
    assert len(v_t["modele"]["par_tranche"]) == len(bornes)       # 3 tranches + information
    assert "era5" in v_t and "domaine_0_1km" in v_t["era5"]
    r = [x for x in v_t["modele"]["par_tranche"] if x["noeuds_par_instant"] > 0]
    assert r and all(np.isfinite(x["rmse_u"]) and "degradation_25pct" in x for x in r)
    assert (out.parent / "dso_tranches.md").exists()

    # scripts/dso_hauteur_sol.py (memes tranches par defaut) : memes RMSE que dso-tranches, pour
    # le modele et pour ERA5 (deux chemins : canaux d'entree / cellules de tfeat apres N_SCAL)
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "dso_hauteur_sol", Path(__file__).resolve().parents[1] / "scripts" / "dso_hauteur_sol.py")
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    ev = Evaluator(c, device="cpu", verbose=False)
    tab, info = H.mesurer(ev, c, None, None)
    ref = dso_tranches(c, device="cpu", verbose=False)
    for src in ("modele", "era5"):
        for r in ref[src]["par_tranche"][:-1]:
            if r["noeuds_par_instant"] == 0:
                continue
            h = tab[(tab.predicteur == src) & (tab.tranche == r["tranche"])].iloc[0]
            assert abs(h.rmse_u - r["rmse_u"]) < 1e-4 and abs(h.rmse_v - r["rmse_v"]) < 1e-4
    vd = H.verdicts(tab, info["global"], c, None)
    assert set(vd) == {"modele", "era5"} and "dso1_conforme" in vd["modele"]


def _jeu_reel(cfg_era5, index, tmp_path):
    """Mini-jeu en mode era5_stations (3 stations, _E, forcage) + observations synthetiques
    aux instants de test et d'entrainement ; renvoie (config, dossier des observations)."""
    import json
    import shutil

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
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    lignes, sondes = [], []
    for t in pd.to_datetime(index["timestamp"]):
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
    return c, obs_dir


def test_validation_reelle_de_bout_en_bout(cfg_era5, index, tmp_path):
    """scripts/valider_grand_tunis_reel.py sur le mini-jeu : entree era5_stations avec des
    mesures synthetiques aux 3 stations, un radiosondage ; bilans ecrits, sans NaN."""
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("valider", repo / "scripts" / "valider_grand_tunis_reel.py")
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)

    c, obs_dir = _jeu_reel(cfg_era5, index, tmp_path)
    Trainer(c, device="cpu", verbose=False).fit()
    resume = V.valider(c, obs_dir, "test", device="cpu")
    sortie = c.paths.run_dir / "validation_reelle" / "test"
    b = pd.read_csv(sortie / "stations_bilan.csv")
    assert set(b["station"]) == {"Tunis-Carthage", "Zaghouan", "Nabeul"}
    assert set(b["source"]) == {"modele", "era5", "generateur"}
    assert b["rmse_vect"].notna().all()
    assert "stations" in resume
    assert (sortie / "radiosondages_bilan.csv").exists()


def test_affinage_sur_les_stations(cfg_era5, index, tmp_path):
    """Affinage : vraies mesures en entree, une station cible retiree de l'entree, terme de
    perte a 10 m ; last.pt n'est pas recopie dans run_dir avec train.sync_last = false."""
    from pcfno3d.dataset import ANEO3D
    from pcfno3d.stations_reelles import N_CIBLE

    c, obs_dir = _jeu_reel(cfg_era5, index, tmp_path)
    c.apply_overrides({"paths.stations_obs": obs_dir, "loss.lambda_station": 0.5,
                       "train.stations_cibles": ["DTTA", "INM60729"], "train.sync_last": False,
                       "paths.local_dir": tmp_path / "local"})
    g = Geometry(c)
    train = index[index["split"] == "train"]
    ds = ANEO3D(train["stem"].values, train["hour"].values, train["doy"].values, g, train=True,
                n_samples=8)
    item = ds[0]
    assert len(item) == 6 and item[5].shape == (N_CIBLE,) and float(item[5][8]) == 1.0
    cible = item[5].numpy()
    assert (int(cible[4]), int(cible[5])) != g._cellules_codes["INM60728"]   # jamais cible
    iy, ix = int(cible[4]), int(cible[5])
    # la MESURE de la cible n'est pas en entree (sa cellule peut etre un pseudo-capteur ERA5)
    assert not item[1].numpy()[iy, ix] or float(item[0][0, iy, ix]) != 4.0

    hist = Trainer(c, device="cpu", verbose=False).fit()
    assert hist and np.isfinite(hist[-1]["train"]["station"]) and hist[-1]["train"]["station"] > 0
    assert c.paths.ckpt_best.exists() and not c.paths.ckpt_last.exists()


def test_service_entree_de_production_en_mode_reel(cfg_era5, index, tmp_path):
    """era5_stations : surface_from_cache donne l'entree de production (ERA5 10 m sur le reseau
    + stations, vraies mesures si paths.stations_obs), pas la couche de surface generee."""
    from pcfno3d.dataset import entree_era5_stations
    from pcfno3d.inference import WindFieldModel

    c, obs_dir = _jeu_reel(cfg_era5, index, tmp_path)
    Trainer(c, device="cpu", verbose=False).fit()
    wf = WindFieldModel(c, device="cpu", verbose=False)
    stem = index[index["split"] == "test"]["stem"].iloc[0]
    S, hour, doy, ts = wf.surface_from_cache(stem)
    brut = np.load(c.paths.cache_dir / f"{stem}_S.npy")
    _, attendu = entree_era5_stations(brut, c.paths.cache_dir / f"{stem}_E.npy", wf.geom,
                                      np.random.default_rng(0), train=False)
    assert ((S[7] > 0.5) == attendu).all() and attendu.sum() < attendu.size
    assert np.isfinite(wf.predict_stem(stem).u[wf.geom.valid]).all()

    c2 = copy.deepcopy(c)
    c2.apply_overrides({"paths.stations_obs": obs_dir})
    wf2 = WindFieldModel(c2, device="cpu", verbose=False)
    S2, *_ = wf2.surface_from_cache(stem)
    from pcfno3d.stations_reelles import cellules_codes
    iy, ix = cellules_codes(wf2.geom)["DTTA"]
    assert S2[7, iy, ix] == 1.0 and S2[0, iy, ix] == pytest.approx(4.0)   # vraie mesure


def test_diagnostic_v(cfg_era5, index, tmp_path):
    """scripts/diagnostic_v.py : ERA5 seule contre la verite, u et v, par niveau ; le forcage
    synthetique etant lineaire, ERA5 aux noeuds suit la verite du cache (correlations finies)."""
    import subprocess
    import sys
    from pathlib import Path

    c = copy.deepcopy(cfg_era5)
    c.apply_overrides({"paths.run_dir": tmp_path / "run"})
    f = c.save(tmp_path / "c.yaml")
    repo = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, str(repo / "scripts" / "diagnostic_v.py"), "-c", str(f)],
                       capture_output=True, text=True, cwd=repo)
    assert p.returncode == 0, p.stderr[-2000:]
    t = pd.read_csv(tmp_path / "run" / "diagnostic_v" / "diagnostic_v.csv")
    ok = t[t["couverture"] > 0]
    assert len(ok) and np.isfinite(ok[["era5_rmse_u", "era5_rmse_v", "corr_u_u", "corr_v_v"]]).all().all()


def test_carte_v_generateur(cfg_era5, index, tmp_path):
    """scripts/carte_v_generateur.py : cartes de l'ecart verite - ERA5 a hauteur fixe au-dessus
    du sol (forcage lineaire du mini-jeu : cartes finies la ou la colonne monte assez haut)."""
    import subprocess
    import sys
    from pathlib import Path

    c = copy.deepcopy(cfg_era5)
    c.apply_overrides({"paths.run_dir": tmp_path / "run"})
    f = c.save(tmp_path / "c.yaml")
    repo = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, str(repo / "scripts" / "carte_v_generateur.py"), "-c", str(f)],
                       capture_output=True, text=True, cwd=repo)
    assert p.returncode == 0, p.stderr[-2000:]
    d = tmp_path / "run" / "diagnostic_v"
    assert (d / "carte_v_300m.png").exists() and (d / "cartes.npz").exists()
    z = np.load(d / "cartes.npz")
    assert {"rmse_v_300m", "rmse_u_800m", "biais_v_300m"} <= set(z.files)


def test_diagnostic_forcage_regional(cfg_era5, index, tmp_path):
    """scripts/diagnostic_forcage_regional.py tourne de bout en bout sur le mini-jeu (modele court)."""
    import subprocess
    import sys
    from pathlib import Path

    c = copy.deepcopy(cfg_era5)
    c.apply_overrides({"paths.run_dir": tmp_path / "run", "model.residuel_era5": True})
    Trainer(c, device="cpu", verbose=False).fit()
    f = c.save(tmp_path / "c.yaml")
    repo = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, str(repo / "scripts" / "diagnostic_forcage_regional.py"), "-c", str(f),
                        "--device", "cpu", "--hauteur", "60"], capture_output=True, text=True, cwd=repo,
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert p.returncode == 0, p.stderr[-2500:]
    assert (tmp_path / "run" / "diagnostic_forcage" / "resume.json").exists()


def test_diagnostic_v_phase1(cfg_era5, index, tmp_path):
    """scripts/diagnostic_v_phase1.py : modes, spectres et tranches entrainement / test, mini-jeu."""
    import subprocess
    import sys
    from pathlib import Path

    c = copy.deepcopy(cfg_era5)
    c.apply_overrides({"paths.run_dir": tmp_path / "run", "model.residuel_era5": True})
    Trainer(c, device="cpu", verbose=False).fit()
    f = c.save(tmp_path / "c.yaml")
    repo = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, str(repo / "scripts" / "diagnostic_v_phase1.py"), "-c", str(f),
                        "--device", "cpu", "--n-train", "3", "--hauteurs", "40,80"], capture_output=True, text=True, cwd=repo,
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert p.returncode == 0, p.stderr[-2500:]
    d = tmp_path / "run" / "diagnostic_v_phase1"
    import json as _json
    r = _json.loads((d / "resume.json").read_text(encoding="utf-8"))
    assert r["modes"]["lambda_min_y_km"] > 0 and (d / "tranches.csv").exists()


def test_capteurs_aneo_simules_en_entree(cfg_era5, index, tmp_path):
    """Candidat D national : capteurs ANEO simules en entree comme des stations (entrainement, et
    evaluation d'information seulement quand on passe aneo_eval) ; un entrainement court tourne."""
    from pcfno3d.aneo_entree import EntreeANEO, entree_aneo

    c, _ = _jeu_reel(cfg_era5, index, tmp_path)
    c.apply_overrides({"mask.aneo_capteurs": [2, 4], "mask.aneo_p_zone": 0.5})
    g = Geometry(c, device="cpu")
    e = entree_aneo(g)
    assert e is not None and e.n_capteurs == (2, 4)
    try:
        e.candidats()
    except ValueError:                            # mini-grille : contraintes de l'OS6 trop strictes
        e._cand = np.argwhere(g.valid.any(axis=0))
    val = index[index["split"] == "val"]
    args = (val["stem"].values, val["hour"].values, val["doy"].values, g)
    base = ANEO3D(*args, train=False, seed=1000, keep_eval=1.0)[0]
    ev = EntreeANEO(g, None, 0.0, (50.0, 150.0))
    ev._cand = e._cand
    ev.n_capteurs = (3, 3)
    avec = ANEO3D(*args, train=False, seed=1000, keep_eval=1.0, aneo_eval=ev)[0]
    nouveaux = (avec[1] > 0.5) & ~(base[1] > 0.5)
    assert 0 < int(nouveaux.sum()) <= 3
    assert torch.equal(ANEO3D(*args, train=False, seed=1000, keep_eval=1.0, aneo_eval=ev)[0][0], avec[0])   # reproductible
    item = ANEO3D(*args[:3], g, train=True, n_samples=2)[0]
    assert torch.isfinite(item[0]).all()


def test_info_capteurs_v4_de_bout_en_bout(cfg_era5, index, tmp_path, monkeypatch):
    """scripts/info_capteurs_v4.py sur le mini-jeu : trois conditions, gain des capteurs ecrit."""
    import importlib.util
    import json
    import sys
    from pathlib import Path

    import pcfno3d.aneo_entree as AE

    repo = Path(__file__).resolve().parents[1]
    c, _ = _jeu_reel(cfg_era5, index, tmp_path)
    Trainer(c, device="cpu", verbose=False).fit()
    deg = tmp_path / "deg.json"
    deg.write_text(json.dumps({"sigma_u": 1.0, "sigma_v": 1.0, "L_km": 100.0}))
    cfgf = tmp_path / "c.yaml"
    c.save(cfgf) if hasattr(c, "save") else pytest.skip("Config sans save")
    orig = AE.EntreeANEO.candidats

    def cand(self):                                   # mini-grille : contraintes de l'OS6 trop strictes
        try:
            return orig(self)
        except ValueError:
            self._cand = np.argwhere(self.geom.valid.any(axis=0))
            return self._cand
    monkeypatch.setattr(AE.EntreeANEO, "candidats", cand)
    spec = importlib.util.spec_from_file_location("info", repo / "scripts" / "info_capteurs_v4.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    monkeypatch.setattr(sys, "argv", ["info", "-c", str(cfgf), "--run", str(c.paths.run_dir),
                                      "--degradation", str(deg), "--n-capteurs", "3"])
    assert M.main() == 0
    out = json.loads((Path(c.paths.run_dir) / "info_capteurs_validation.json").read_text(encoding="utf-8"))
    assert set(out["conditions"]) == {"propre", "era5_degradee", "era5_degradee_3_capteurs"}
    assert np.isfinite(out["gain_des_capteurs_rmse_max_uv"])
    cp = out["conditions"]
    assert cp["era5_degradee"]["era5"]["rmse_max_uv"] != cp["propre"]["era5"]["rmse_max_uv"]


def test_capteurs_arrivent_au_reseau(cfg_era5, index, tmp_path, monkeypatch):
    """Demande d'Ines (11/11) : une entree modifiee aux capteurs (simules ou aeroports) doit changer la sortie
    pres des capteurs, par le chemin de l'evaluation ET par celui de l'OS6 ; scripts/controle_capteurs.py."""
    import importlib.util
    import json
    import sys
    from pathlib import Path

    import pcfno3d.aneo_entree as AE

    repo = Path(__file__).resolve().parents[1]
    c, _ = _jeu_reel(cfg_era5, index, tmp_path)
    Trainer(c, device="cpu", verbose=False).fit()
    cfgf = tmp_path / "c.yaml"
    c.save(cfgf)
    orig = AE.EntreeANEO.candidats

    def cand(self):                                   # mini-grille : contraintes de l'OS6 trop strictes
        try:
            return orig(self)
        except ValueError:
            self._cand = np.argwhere(self.geom.valid.any(axis=0))
            return self._cand
    monkeypatch.setattr(AE.EntreeANEO, "candidats", cand)
    spec = importlib.util.spec_from_file_location("ctrl", repo / "scripts" / "controle_capteurs.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    monkeypatch.setattr(sys, "argv", ["ctrl", "-c", str(cfgf), "--run", str(c.paths.run_dir), "--instants", "1",
                                      "--n-capteurs", "3", "--rayon-km", "3"])
    assert M.main() == 0
    out = json.loads((Path(c.paths.run_dir) / "controle_capteurs.json").read_text(encoding="utf-8"))
    r = out["controles"][0]
    assert r["capteurs_dans_le_masque"] == r["capteurs_tires"] == 3
    assert r["ecart_max_valeur_ecrite_S_u"] < 1e-5
    assert r["entree_reseau_masque_aux_capteurs_min"] > 0.5
    assert r["entree_reseau_u_aux_capteurs_ecart_max"] < 1e-3
    for cle in ("absurde_capteurs_evaluation", "absurde_capteurs_chemin_os6", "absurde_aeroports_evaluation"):
        # sortie IDENTIQUE (0) = capteurs non branches ; la sensibilite elle-meme depend du modele entraine
        assert r[cle]["max_u_proche"] > 1e-6, (cle, r[cle])
