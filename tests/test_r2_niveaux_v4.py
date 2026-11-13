"""scripts/r2_niveaux_v4.py sur le mini-jeu reel : un petit entrainement, R2 par niveau avant / apres
calibration hors pli, DSO4 en mediane. Aucun GPU, aucun acces reseau, le test n'est pas lu."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_forcage_era5 import LAT0, LON0, _jeu_reel, cfg_era5, forcage  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
REGLE = REPO / "configs" / "national" / "regle_postraitement_r2_v4.json"


def _script():
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("r2_niveaux_v4", REPO / "scripts" / "r2_niveaux_v4.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def entraine(cfg_era5, index, tmp_path_factory):  # noqa: F811
    from pcfno3d.train import Trainer

    tmp = tmp_path_factory.mktemp("r2niv")
    c, _ = _jeu_reel(cfg_era5, index, tmp)
    c = copy.deepcopy(c)
    c.apply_overrides({"paths.run_dir": tmp / "runs" / "mini_Cp_s1", "train.seed": 1})
    c.paths.mkdirs()
    Trainer(c, device="cpu", verbose=False).fit()
    return c, tmp / "runs" / "mini_Cp_s1"


def test_ajuster_moindres_carres():
    M = _script()
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 0.5 + 2.0 * x
    st = np.array([4, x.sum(), y.sum(), (x * x).sum(), (x * y).sum()])
    a, b = M.ajuster(st)
    assert a == pytest.approx(0.5) and b == pytest.approx(2.0)
    assert M.ajuster(np.zeros(5)) == (0.0, 1.0)                 # niveau vide : identite


def test_script_de_bout_en_bout(entraine, tmp_path):
    c, run = entraine
    cfgf = c.save(tmp_path / "c.yaml")
    M = _script()
    argv = ["-c", str(cfgf), "--run", str(run), "--regle", str(REGLE), "--device", "cpu", "--n-dso4", "3",
            "--chauffe-dso4", "1"]
    assert M.main(argv) == 0
    out = run / "r2_niveaux_validation"
    bilan = json.loads((out / "bilan.json").read_text(encoding="utf-8"))
    assert bilan["jeu"] == "validation" and bilan["regle_sha256"]
    assert bilan["post_traitement_retenu_pour_le_test"] in ("w_continuite", "calibre_w_continuite")
    tab = pd.read_csv(out / "r2_par_niveau.csv")
    assert set(tab["predicteur"]) >= {"brut", "w_continuite", "calibre_w_continuite"}
    # u, v identiques entre brut et w_continuite (seul w change)
    b, w = (tab[tab.predicteur == p].reset_index(drop=True) for p in ("brut", "w_continuite"))
    np.testing.assert_allclose(b["r2_u"], w["r2_u"], equal_nan=True)
    d4 = json.loads((out / "dso4.json").read_text(encoding="utf-8"))
    assert d4["n"] == 3 and d4["mediane_s"] > 0
    co = json.loads((out / "coefficients_calibration.json").read_text(encoding="utf-8"))
    assert len(co["servis"]) == 2 and len(co["servis"][0]) == len(co["niveaux_m"])
    with pytest.raises(SystemExit):                               # jamais d'ecrasement
        M.main(argv)
