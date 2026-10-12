"""Import de l'evaluation imprimee par le BLOC 10 du notebook v7.

Teste sur le VRAI notebook (notebooks/current/fno3d_v7.ipynb) : ces chiffres sont le
bilan de reference du projet, une regression du parseur les fausserait en silence.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_V7 = REPO_ROOT / "notebooks" / "current" / "fno3d_v7.ipynb"


def _mod():
    spec = importlib.util.spec_from_file_location(
        "import_notebook_eval", REPO_ROOT / "scripts" / "import_notebook_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def v7():
    if not NOTEBOOK_V7.exists():
        pytest.skip("notebook v7 absent")
    mod = _mod()
    return mod.parse(mod.bloc10_text(NOTEBOOK_V7))


def test_bilan_v7_tel_qu_imprime(v7):
    assert v7["modele"] == "best_dso_v7.pt" and v7["epoque"] == 140
    assert v7["instants_synoptiques"] == 120 and v7["capteurs_visibles"] == 0.5
    # le seul objectif manque : DSO1, a cause de la RMSE
    assert v7["dso1_rmse"] is False and v7["dso1_rmse_global_conforme"] is False
    assert v7["dso1_r2"] is True and v7["dso1_r2_uvw"] is True
    assert all(v7[k] for k in ("dso2_conforme", "dso3_conforme", "dso4_conforme",
                               "dso5_conforme", "critere_omm"))
    assert v7["dso1_rmse_global_u"] == 0.542 and v7["dso1_rmse_global_v"] == 0.437
    assert v7["dso2_rms_moyen"] == 2.58e-05 and v7["dso2_rms_max"] == 6.69e-05
    assert v7["dso3_rugosite"] == 0.951
    assert v7["dso4_inference_s"] == pytest.approx(0.091)
    assert v7["dso5_rmse_25pct"] == 0.646
    assert v7["mae_direction_deg"] == 3.64


def test_table_par_niveau(v7):
    lv = {r["z_m"]: r for r in v7["par_niveau_synoptique"]}
    assert sorted(lv) == [25, 50, 75, 100, 150, 200, 300, 400, 500, 650, 800, 1000]
    assert lv[1000]["rmse_u"] == 0.665 and lv[1000]["rmse_v"] == 0.508
    assert lv[25]["r2_v"] == -0.8695                   # valeur negative bien relue
    assert lv[25]["couverture"] == 0.0 and lv[1000]["couverture"] == 1.0
    # les niveaux hors seuil sont exactement 650, 800, 1000 m
    hors = [z for z, r in lv.items()
            if r["couverture"] >= 0.05 and max(r["rmse_u"], r["rmse_v"]) > 0.5]
    assert hors == [650, 800, 1000]


def test_robustesse_et_episodes(v7):
    rob = {r["capteurs_visibles"]: r for r in v7["robustesse"]}
    assert sorted(rob) == [0.1, 0.25, 0.5, 1.0]
    # l'anomalie que corrige mask.p_full : 100 % de capteurs fait PIRE que 50 %
    assert rob[1.0]["rmse_u_max"] > rob[0.5]["rmse_u_max"]
    assert v7["instants_episodes"] == 46 and len(v7["par_niveau_episodes"]) == 9


def test_bloc_10_incomplet_refuse(tmp_path):
    nb = {"cells": [{"cell_type": "code", "source": ["# BLOC 10 — Evaluation\n"],
                     "outputs": [{"output_type": "stream", "text": ["TEST SYNOPTIQUE\n"]}]}]}
    p = tmp_path / "coupe.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    with pytest.raises(ValueError, match="BLOC 10 OK"):
        _mod().bloc10_text(p)


def test_notebook_sans_bloc_10_refuse(tmp_path):
    p = tmp_path / "vide.ipynb"
    p.write_text(json.dumps({"cells": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="aucun BLOC 10"):
        _mod().bloc10_text(p)


def test_valeur_manquante_refusee():
    """Un resume partiel affiche comme complet serait pire que pas de resume."""
    with pytest.raises(ValueError, match="introuvable"):
        _mod().parse("BLOC 10 OK\nModele evalue : x.pt\n")


def test_le_resume_importe_est_servi_avec_sa_source(cfg, index, norm, tmp_path, monkeypatch):
    """dso_report doit trouver le resume du notebook et le renvoyer avec sa provenance."""
    if not NOTEBOOK_V7.exists():
        pytest.skip("notebook v7 absent")
    import shutil

    import yaml

    from pcfno3d.train import Trainer
    from service import mcp_server

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    run = tmp_path / "run"
    run.mkdir()
    shutil.copy2(cfg.paths.ckpt_best, run / "best_dso_v7.pt")    # meme nom que la v7
    shutil.copy2(cfg.paths.norm_stats, run / "norm_stats.json")
    raw = yaml.safe_load((REPO_ROOT / "configs" / "test.yaml").read_text(encoding="utf-8"))
    raw["paths"].update({"data_dir": str(cfg.paths.data_dir),
                         "cache_dir": str(cfg.paths.cache_dir), "run_dir": str(run)})
    conf = tmp_path / "c.yaml"
    conf.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["imp", str(NOTEBOOK_V7), "--config", str(conf)])
    assert _mod().main() == 0

    from pcfno3d.config import Config
    from pcfno3d.inference import WindFieldModel
    mcp_server.set_model(WindFieldModel(Config.load(conf), device="cpu", verbose=False))
    rep = mcp_server.dso_report()
    assert rep["disponible"] is True
    assert rep["resume"]["source"]["notebook"] == "fno3d_v7.ipynb"
    assert rep["resume"]["source"]["materiel"] == "GPU Colab"
    assert len(rep["resume"]["robustesse"]) == 4               # plus retiree du resume
