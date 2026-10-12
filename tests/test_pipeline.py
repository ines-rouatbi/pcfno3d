"""Le pipeline complet sur le mini-dataset : cache -> entrainement -> evaluation ->
reconstruction -> export web. Ne verifie pas la qualite (2 epoques), mais que chaque
etape produit bien ses fichiers et que la reprise fonctionne."""

from __future__ import annotations

import json

import numpy as np


def test_cache_et_index(cfg, index):
    assert len(index) > 0
    assert set(index.columns) >= {"stem", "day", "timestamp", "split", "hour", "doy"}
    assert set(index["split"]) <= {"train", "val", "test"}
    # un jour n'appartient qu'a un seul ensemble
    assert index.groupby("day")["split"].nunique().max() == 1
    for stem in index["stem"].head(3):
        assert (cfg.paths.cache_dir / f"{stem}_S.npy").exists()
        assert (cfg.paths.cache_dir / f"{stem}_Y.npy").exists()


def test_verite_nulle_sous_le_relief(cfg, geom, index):
    from pcfno3d.cache import check_ground_truth

    for row in check_ground_truth(cfg, geom, n=2):
        assert row["max_hors_domaine"] == 0.0
        assert 0.8 < row["rho_min"] < 1.5, "densite de l'air hors du domaine physique"


def test_statistiques_ecrites_et_relues(cfg, norm):
    assert cfg.paths.norm_stats.exists()
    stats = json.loads(cfg.paths.norm_stats.read_text(encoding="utf-8"))
    assert {"surf_mu", "surf_sd", "y_mu", "y_sd", "w_level_sd"} <= set(stats)
    assert all(sd > 0 for sd in stats["y_sd"])


def test_entrainement_puis_reprise(cfg, index, norm):
    from pcfno3d.train import Trainer

    history = Trainer(cfg, device="cpu", verbose=False).fit()
    assert len(history) == cfg.train.epochs
    assert np.isfinite([h["val_score"] for h in history]).all()
    assert cfg.paths.ckpt_best.exists() and cfg.paths.ckpt_last.exists()

    # relancer ne refait pas d'epoque : la reprise part de la derniere terminee
    again = Trainer(cfg, device="cpu", verbose=False)
    assert again.start_epoch == cfg.train.epochs
    assert again.fit() == history


def test_evaluation_produit_le_verdict_dso(cfg):
    from pcfno3d.evaluate import Evaluator

    summary = Evaluator(cfg, device="cpu", verbose=False).run()
    for key in ("dso1_rmse", "dso2_conforme", "dso3_conforme", "dso4_conforme",
                "dso5_conforme", "mae_direction_deg", "par_niveau_synoptique"):
        assert key in summary
    assert summary["dso4_inference_s"] > 0
    # DSO1 revise propose : tolerance OMM par niveau, jamais en dessous du seuil officiel
    ev = [r for r in summary["par_niveau_synoptique"] if r["couverture"] >= cfg.dso.min_level_cover]
    for r in ev:
        assert r["seuil_omm"] == max(cfg.dso.omm_floor, cfg.dso.omm_rel * r["vitesse_moyenne"])
        assert r["seuil_omm"] >= cfg.dso.rmse
    assert summary["critere_omm"] == all(max(r["rmse_u"], r["rmse_v"]) <= r["seuil_omm"] for r in ev)
    out = cfg.paths.run_dir / "eval_pcfno3d" / "best"
    for name in ("test_par_niveau_synoptique.csv", "test_par_instant.csv",
                 "robustesse_capteurs.csv", "test_summary.json"):
        assert (out / name).exists(), name


def test_reconstruction_et_export_web(cfg):
    from pcfno3d.reconstruct import reconstruct

    result = reconstruct(cfg, device="cpu", figures=True, web=True, verbose=False)
    assert result["noeuds"] > 0
    assert result["inference_s"] > 0
    npz = np.load(result["npz"])
    assert {"u", "v", "w", "speed", "direction", "z_levels", "z_terrain"} <= set(npz.files)
    assert npz["u"].shape == npz["speed"].shape

    manifest_path = next((cfg.paths.run_dir / "web").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "pcfno3d-web/1"
    assert set(manifest["champs"]) == {"u", "v", "w", "speed"}


def test_inference_depuis_un_csv_de_surface(cfg, geom, index):
    """Le chemin de production : un CSV de mesures d'un instant, sans verite 3D."""
    import pandas as pd

    from pcfno3d.geometry import SURF_VARS
    from pcfno3d.inference import WindFieldModel

    # reconstruit un CSV de surface a partir du dataset brut d'un jour de test
    day_file = next(cfg.paths.data_dir.glob("aneo_observations_day_006_*_multitime.csv.gz"))
    df = pd.read_csv(day_file)
    df = df[df["timestamp_utc"] == df["timestamp_utc"].iloc[0]]
    df = df[df["point_id"] < geom.node_id_start]
    csv = cfg.paths.run_dir / "surface_instant.csv"
    df[["point_id", "timestamp_utc", "mask_i_obs"] + SURF_VARS].to_csv(csv, index=False)

    wf = WindFieldModel(cfg, device="cpu", verbose=False)
    volume = wf.predict_csv(csv)
    assert volume.u.shape == geom.shape
    assert volume.n_sensors > 0
    # NaN exactement sous le relief, valeurs finies ailleurs
    assert np.isnan(volume.u[~geom.valid]).all()
    assert np.isfinite(volume.u[geom.valid]).all()

    point = volume.at_point(x_m=geom.dx * 3, y_m=geom.dy * 3,
                            z_asl_m=float(geom.z_levels[-1]))
    assert set(point) >= {"wind_u_m_s", "wind_speed_m_s", "wind_direction_deg"}
    assert 0.0 <= point["wind_direction_deg"] <= 360.0


def test_meta_du_service(cfg):
    from pcfno3d.inference import WindFieldModel

    meta = WindFieldModel(cfg, device="cpu", verbose=False).meta()
    assert meta["grille"]["nz"] == len(meta["z_levels_asl_m"])
    assert meta["noeuds_valides"] > 0


def test_export_par_lot_des_instants_de_test(cfg, index, norm, tmp_path):
    """export-web --split test : un modele charge une fois, un dossier par instant, dans
    paths.web_dir (Colab : disque local), et une relance ne refait rien."""
    import json as _json

    from pcfno3d.cache import load_index
    from pcfno3d.export import export_split
    from pcfno3d.inference import WindFieldModel

    from pcfno3d.train import Trainer

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    cfg.paths.web_dir = tmp_path / "web"
    try:
        wf = WindFieldModel(cfg, device="cpu", verbose=False)
        n = int((load_index(cfg)["split"] == "test").sum())
        r = export_split(wf, split="test", stride=2, verbose=False)
        assert r == {"exportes": n, "deja_la": 0, "total": n, "dossier": str(tmp_path / "web")}
        dossiers = sorted((tmp_path / "web").glob("recon_*"))
        assert len(dossiers) == n
        m = _json.loads((dossiers[0] / "manifest.json").read_text(encoding="utf-8"))
        assert m["grille"]["stride"] == 2
        assert (dossiers[0] / "u.bin").stat().st_size == 4 * m["grille"]["nz"] *             m["grille"]["ny"] * m["grille"]["nx"]
        assert export_split(wf, split="test", stride=2, verbose=False)["exportes"] == 0
    finally:
        cfg.paths.web_dir = None
