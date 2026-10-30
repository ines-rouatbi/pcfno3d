"""v4 Grand Tunis (regles d'Ines, 2026-10-24) : DSO3 par niveau dans l'evaluation, w par
continuite aussi dans l'ensemble de fiabilite (DSO6 de w = w servi), et regle de livraison
v4 / v2 de scripts/decision_v4_grand_tunis.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "decision_v4", REPO / "scripts" / "decision_v4_grand_tunis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trained(cfg, index, norm):
    from pcfno3d.train import Trainer
    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg


def test_rugosite_par_niveau(trained):
    """La table par niveau du test porte le DSO3 du niveau (rapport de rugosite moyen)."""
    from pcfno3d.evaluate import Evaluator

    ev = Evaluator(trained, device="cpu", verbose=False)
    ev.run()
    lv = pd.read_csv(ev.out_dir / "test_par_niveau_synoptique.csv")
    assert "rugosite" in lv.columns
    ev_lv = lv[lv["couverture"] >= trained.dso.min_level_cover]
    assert np.isfinite(ev_lv["rugosite"]).all() and (ev_lv["rugosite"] > 0).all()


def test_ensemble_w_par_continuite(trained):
    """inference.w_continuite : chaque passe de l'ensemble de fiabilite recalcule w par
    continuite, comme la passe servie (sinon DSO6 jugerait un autre w que celui servi)."""
    from pcfno3d.cache import load_index
    from pcfno3d.inference import WindFieldModel

    stem = load_index(trained).query("split == 'test'")["stem"].iloc[0]
    trained.inference.w_continuite = True
    try:
        wf = WindFieldModel(trained, device="cpu", verbose=False)
        S, hour, doy, _ = wf.surface_from_cache(stem)
        m = S[7] > 0.5
        rng = np.random.default_rng(0)
        masks = [m & (rng.random(m.shape) < 0.7) for _ in range(3)]
        for u, v, w in wf._members(S, masks, hour, doy, batch=2):
            attendu = wf.geom.w_continuite(torch.from_numpy(u)[None], torch.from_numpy(v)[None])
            np.testing.assert_allclose(w, attendu[0].numpy(), atol=1e-5)
    finally:
        trained.inference.w_continuite = False


# ---------------------------------------------------------------------------- regle
def _ecrire(runs: Path, run: str, s: dict, lv: pd.DataFrame, stations: dict) -> None:
    for dossier in ("best", "best_wcont"):
        d = runs / run / "eval_pcfno3d" / dossier
        d.mkdir(parents=True, exist_ok=True)
        (d / "test_summary.json").write_text(json.dumps(s), encoding="utf-8")
        lv.to_csv(d / "test_par_niveau_synoptique.csv", index=False)
    d = runs / run / "validation_reelle" / "test"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"station": k, "source": "modele", "rmse_vect": x} for k, x in stations.items()]
                 ).to_csv(d / "stations_bilan.csv", index=False)


def _jeu(rmse: float = 0.4, r2_w: float = 0.93, temps: float = 0.3, d5: float = 0.6):
    lv = pd.DataFrame({"z_m": [25, 50, 100], "couverture": [0.6, 0.9, 1.0],
                       "rmse_u": [rmse + 0.1, rmse, rmse], "rmse_v": [rmse, rmse, rmse],
                       "r2_u": [0.98] * 3, "r2_v": [0.98] * 3, "r2_w": [r2_w] * 3,
                       "rugosite": [0.85] * 3})
    s = {"dso1_rmse": rmse + 0.1 <= 0.5, "dso1_r2": True, "dso1_r2_uvw": r2_w >= 0.95,
         "dso1_rmse_global_u": rmse, "dso1_rmse_global_v": rmse,
         "dso1_rmse_global_conforme": rmse <= 0.5, "dso2_rms_moyen": 1e-6,
         "dso3_rugosite": 0.85, "dso4_inference_s": temps, "dso4_conforme": temps < 1.0,
         "dso5_rmse_25pct": d5, "dso5_conforme": d5 <= 0.7}
    return s, lv


@pytest.mark.parametrize("v4, stations_v4, attendu", [
    (dict(rmse=0.35, r2_w=0.96), {"A": 2.4, "B": 2.6}, 0),        # mieux partout : v4 livree
    (dict(rmse=0.40, r2_w=0.93), {"A": 2.5, "B": 2.7}, 0),        # egale : au moins aussi bonne
    (dict(rmse=0.35, r2_w=0.96), {"A": 2.4, "B": 2.9}, 3),        # une station recule de 7 %
    (dict(rmse=0.42, r2_w=0.96), {"A": 2.5, "B": 2.7}, 3),        # RMSE global +5 %
    (dict(rmse=0.35, d5=0.75), {"A": 2.5, "B": 2.7}, 3),          # DSO5 perdu
])
def test_regle_de_livraison(tmp_path, v4, stations_v4, attendu):
    mod = _script()
    s2, lv2 = _jeu()
    s4, lv4 = _jeu(**v4)
    _ecrire(tmp_path, mod.V2, s2, lv2, {"A": 2.5, "B": 2.7})
    _ecrire(tmp_path, mod.V4, s4, lv4, stations_v4)
    import sys
    argv, sys.argv = sys.argv, ["decision", "--runs", str(tmp_path)]
    try:
        assert mod.main() == attendu
    finally:
        sys.argv = argv
    dec = json.loads((tmp_path / "decision_v4.json").read_text(encoding="utf-8"))
    assert dec["modele_livre"] == (mod.V4 if attendu == 0 else mod.V2)
    assert dec["w_continuite"] is True
    assert dec["dso2_w_continuite"] == "garanti par construction"
    t = pd.read_csv(tmp_path / "comparaison_v2_v4_par_niveau.csv")
    assert {"r2_w_reseau_v2", "r2_w_continuite_v4", "rugosite_v4"} <= set(t.columns)
    texte = (tmp_path / "rapport_v4.md").read_text(encoding="utf-8")
    assert "garanti par construction" in texte and "NON utilisees pour decider" in texte
    assert "variantes_proposees" in dec and dec["variantes_proposees"]["v2"]["dso3_previsible"] is None


def test_evaluations_incompletes(tmp_path):
    mod = _script()
    s2, lv2 = _jeu()
    _ecrire(tmp_path, mod.V2, s2, lv2, {"A": 2.5})
    import sys
    argv, sys.argv = sys.argv, ["decision", "--runs", str(tmp_path)]
    try:
        assert mod.main() == 2
    finally:
        sys.argv = argv


@pytest.mark.parametrize("challenger, stations, attendu", [
    (dict(rmse=0.35, r2_w=0.96), {"A": 2.4, "B": 2.6}, 0),
    (dict(rmse=0.42, r2_w=0.96), {"A": 2.5, "B": 2.7}, 3),
])
def test_champion_challenger(tmp_path, challenger, stations, attendu):
    """E14 : la regle v4 appliquee a deux runs quelconques."""
    spec = importlib.util.spec_from_file_location("cc", REPO / "scripts" / "champion_challenger.py")
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    s1, l1 = _jeu()
    s2, l2 = _jeu(**challenger)
    _ecrire(tmp_path, "champion", s1, l1, {"A": 2.5, "B": 2.7})
    _ecrire(tmp_path, "nouveau", s2, l2, stations)
    assert cc.main(["--runs", str(tmp_path), "--champion", "champion", "--challenger", "nouveau"]) == attendu
    r = json.loads((tmp_path / "champion_challenger_nouveau.json").read_text(encoding="utf-8"))
    assert r["modele_en_service"] == ("nouveau" if attendu == 0 else "champion")
    assert "| DSO1 |" in (tmp_path / "champion_challenger_nouveau.md").read_text(encoding="utf-8")
    assert cc.main(["--runs", str(tmp_path), "--champion", "champion", "--challenger", "absent"]) == 2
