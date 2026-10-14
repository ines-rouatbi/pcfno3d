"""Servir la v7 en local : verrou de calcul et installation verifiee des poids.

Deux risques propres a un modele lent sur CPU (la v7 : 13,7 s et 5 Go par passe) :
  - l'application web lance plusieurs requetes en parallele pour un nouvel instant ; sans
    verrou, chacune relancerait la meme passe et la memoire saturerait ;
  - un telechargement de 1,8 Go peut etre tronque ; l'installer ne doit jamais remplacer
    un modele qui marche par un fichier casse.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _installer():
    spec = importlib.util.spec_from_file_location(
        "install_v7_weights", REPO_ROOT / "scripts" / "install_v7_weights.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trained(cfg, index, norm):
    from pcfno3d.train import Trainer

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg.paths.ckpt_best


# ---------------------------------------------------------------------------- verrou
def test_requetes_paralleles_ne_calculent_qu_une_fois(cfg, index, norm, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _trained(cfg, index, norm)
    for var, val in (("PCFNO3D_CONFIG", "test"), ("PCFNO3D_DEVICE", "cpu"),
                     ("PCFNO3D_DATA_DIR", str(cfg.paths.data_dir)),
                     ("PCFNO3D_CACHE_DIR", str(cfg.paths.cache_dir)),
                     ("PCFNO3D_RUN_DIR", str(cfg.paths.run_dir))):
        monkeypatch.setenv(var, val)
    monkeypatch.delenv("PCFNO3D_INSECURE_OPEN", raising=False)
    keys = cfg.paths.run_dir / "api_keys.json"
    sauvegarde = keys.read_text(encoding="utf-8") if keys.exists() else None
    keys.unlink(missing_ok=True)
    # service ferme par defaut : une cle de lecture, envoyee par le client
    from service.auth import ApiKeyStore
    cle, _ = ApiKeyStore(keys).create("test-verrou", scope="read")

    from pcfno3d.inference import WindFieldModel
    from service.app import create_app

    appels = []
    original = WindFieldModel.predict_stem

    def lent(self, stem, keep=1.0, seed=0):
        appels.append(stem)
        time.sleep(0.5)                                # comme une passe CPU, en court
        return original(self, stem, keep=keep, seed=seed)

    monkeypatch.setattr(WindFieldModel, "predict_stem", lent)
    try:
        with TestClient(create_app(), headers={"X-API-Key": cle}) as client:
            z = client.get("/api/meta").json()["z_levels_asl_m"][-1]
            stem = client.get("/api/instants?split=test").json()["instants"][1]["stem"]
            codes = []

            def requete(var):
                r = client.get("/api/field", params={"z": z, "stem": stem, "variable": var})
                codes.append(r.status_code)

            fils = [threading.Thread(target=requete, args=(v,))
                    for v in ("speed", "u", "v", "w")]
            for f in fils:
                f.start()
            for f in fils:
                f.join()
    finally:
        keys.unlink(missing_ok=True)
        if sauvegarde is not None:
            keys.write_text(sauvegarde, encoding="utf-8")

    assert codes == [200] * 4
    assert appels.count(stem) == 1, f"{appels.count(stem)} passes pour un seul instant"


# ---------------------------------------------------------------------------- installeur
def test_installeur_refuse_un_dossier_sans_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="best_dso_v7.pt"):
        _installer().find_files(tmp_path)


def test_installeur_refuse_sans_statistiques(tmp_path):
    (tmp_path / "best_dso_v7.pt").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="norm_stats"):
        _installer().find_files(tmp_path)


def test_installeur_controle_les_statistiques(tmp_path):
    inst = _installer()
    bonnes = {"surf_mu": [0.0] * 7, "surf_sd": [1.0] * 7, "y_mu": [0.0] * 3,
              "y_sd": [1.0] * 3, "w_level_sd": [0.1] * 12}
    p = tmp_path / "norm_stats.json"
    p.write_text(json.dumps(bonnes), encoding="utf-8")
    assert inst.check_stats(p, 12)["y_sd"] == [1.0] * 3

    for casse, motif in ((dict(bonnes, w_level_sd=[0.1] * 4), "w_level_sd"),
                         ({k: v for k, v in bonnes.items() if k != "y_mu"}, "y_mu"),
                         (dict(bonnes, y_sd=[1.0, 0.0, 1.0]), "ecart-type")):
        p.write_text(json.dumps(casse), encoding="utf-8")
        with pytest.raises(ValueError, match=motif):
            inst.check_stats(p, 12)


def test_installeur_refuse_un_telechargement_tronque(cfg, geom, tmp_path):
    inst = _installer()
    tronque = tmp_path / "best_dso_v7.pt"
    tronque.write_bytes(b"\0" * 1024)
    with pytest.raises(ValueError, match="incomplet"):
        inst.check_checkpoint(tronque, cfg, geom)


def test_installeur_refuse_une_autre_architecture(cfg, geom, index, norm, tmp_path,
                                                  monkeypatch):
    """Des poids d'une autre largeur ne doivent pas etre acceptes, meme partiellement."""
    from pcfno3d.config import Config
    from pcfno3d.geometry import Geometry

    inst = _installer()
    monkeypatch.setattr(inst, "MIN_CHECKPOINT_BYTES", 0)
    ckpt = _trained(cfg, index, norm)
    autre = Config.load("test")
    autre.apply_overrides({"paths.data_dir": cfg.paths.data_dir,
                           "paths.cache_dir": cfg.paths.cache_dir,
                           "model.width": cfg.model.width + 4})
    with pytest.raises(RuntimeError):                  # load_state_dict(strict=True)
        inst.check_checkpoint(ckpt, autre, Geometry(autre, device="cpu"))


def test_installeur_installe_apres_verification(cfg, index, norm, tmp_path, monkeypatch):
    """Chemin nominal, avec le modele du mini-jeu de donnees dans le role de la v7."""
    inst = _installer()
    monkeypatch.setattr(inst, "MIN_CHECKPOINT_BYTES", 0)
    ckpt = _trained(cfg, index, norm)

    telechargements = tmp_path / "Downloads"
    telechargements.mkdir()
    shutil.copy2(ckpt, telechargements / "best_dso_v7.pt")
    shutil.copy2(cfg.paths.norm_stats, telechargements / "norm_stats.json")

    cible = tmp_path / "runs_v7"
    raw = yaml.safe_load((REPO_ROOT / "configs" / "test.yaml").read_text(encoding="utf-8"))
    raw["paths"].update({"data_dir": str(cfg.paths.data_dir),
                         "cache_dir": str(cfg.paths.cache_dir), "run_dir": str(cible)})
    conf = tmp_path / "v7_local.yaml"
    conf.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["install", str(telechargements), "--config", str(conf)])
    assert inst.main() == 0
    assert (cible / "best_dso_v7.pt").exists()
    assert (cible / "norm_stats.json").exists()
    assert not list(cible.glob("*.part"))             # copie atomique terminee
    # l'installation est servable telle quelle
    from pcfno3d.config import Config
    from pcfno3d.inference import WindFieldModel

    wf = WindFieldModel(Config.load(conf), device="cpu", verbose=False)
    assert wf.checkpoint_path.name == "best_dso_v7.pt"
    assert os.path.samefile(wf.checkpoint_path, cible / "best_dso_v7.pt")


def test_configuration_locale_v7_reprend_l_architecture_v7():
    from pcfno3d.config import Config

    local, ref = Config.load("local_v7"), Config.load("default")
    assert local.model == ref.model
    assert local.inference.tta == 1                  # 8 passes = ~2 min par instant sur CPU
    assert "OneDrive" not in str(local.paths.run_dir)
