"""Regle de livraison de la v6 (scripts/decision_v6_grand_tunis.py), fixee avant le test : livree
seulement si DSO1 (chaque niveau) et DSO3 conformes, sans recul sur DSO2/4/5 ni sur les vraies
mesures ; sinon la v4 reste. decision_v4.json (lu par le service) n'est jamais modifie."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("decision_v6", REPO / "scripts" / "decision_v6_grand_tunis.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(runs: Path, nom: str, rmse25: float, rugosite: float, stations: dict, dso1_ok: bool, dso3_ok: bool):
    d = runs / nom / "eval_pcfno3d" / "best_wcont"
    d.mkdir(parents=True)
    pd.DataFrame({"z_m": [25, 100, 1000], "couverture": [0.26, 0.56, 1.0],
                  "rmse_u": [rmse25, 0.40, 0.32], "rmse_v": [0.50, 0.40, 0.31],
                  "r2_u": [0.98, 0.99, 0.997], "r2_v": [0.98, 0.99, 0.997],
                  "r2_w": [0.95, 0.5, -1.0], "rugosite": [rugosite - 0.3, rugosite, rugosite]}
                 ).to_csv(d / "test_par_niveau_synoptique.csv", index=False)
    (d / "test_summary.json").write_text(json.dumps({
        "dso1_rmse": dso1_ok, "dso1_r2": True, "dso1_r2_uvw": False, "dso1_rmse_global_conforme": True,
        "dso1_rmse_global_u": 0.38, "dso1_rmse_global_v": 0.37, "dso2_rms_moyen": 1.05e-5,
        "dso2_rms_tous_instants": 1.05e-5, "dso2_conforme": True, "dso3_rugosite": rugosite,
        "dso3_conforme": dso3_ok, "dso4_inference_s": 0.30, "dso4_conforme": True,
        "dso5_rmse_25pct": 0.535, "dso5_conforme": True}), encoding="utf-8")
    v = runs / nom / "validation_reelle" / "test"
    v.mkdir(parents=True)
    pd.DataFrame([{"source": "modele", "station": s, "rmse_vect": x} for s, x in stations.items()]
                 ).to_csv(v / "stations_bilan.csv", index=False)


V4_ST = {"Nabeul": 2.42, "Tunis-Carthage": 2.70, "Zaghouan": 2.29}


def test_v6_livree_si_dso1_et_dso3_conformes_sans_recul(tmp_path, monkeypatch):
    m = _module()
    _run(tmp_path, m.V4, 0.528, 0.805, V4_ST, False, False)
    _run(tmp_path, m.V6, 0.495, 0.93, {"Nabeul": 2.43, "Tunis-Carthage": 2.71, "Zaghouan": 2.29}, True, True)
    monkeypatch.setattr("sys.argv", ["x", "--runs", str(tmp_path)])
    assert m.main() == 0
    dec = json.loads((tmp_path / "decision_v6.json").read_text(encoding="utf-8"))
    assert dec["modele_livre"] == m.V6 and dec["dso2_w_continuite"] == "garanti par construction"
    assert dec["criteres"]["DSO1"]["pire_rmse_niveau"] == [0.528, 0.5]
    assert not (tmp_path / "decision_v4.json").exists()
    assert (tmp_path / "rapport_v6.md").exists()


def test_v4_reste_si_recul_aux_stations_ou_dso3_manque(tmp_path, monkeypatch):
    m = _module()
    _run(tmp_path, m.V4, 0.528, 0.805, V4_ST, False, False)
    _run(tmp_path, m.V6, 0.49, 0.95, {"Nabeul": 2.60, "Tunis-Carthage": 2.70, "Zaghouan": 2.29}, True, True)
    monkeypatch.setattr("sys.argv", ["x", "--runs", str(tmp_path)])
    assert m.main() == 3                                   # Nabeul +7 % : la v4 reste
    dec = json.loads((tmp_path / "decision_v6.json").read_text(encoding="utf-8"))
    assert dec["modele_livre"] == m.V4 and "Nabeul" in dec["criteres"]["mesures_reelles"]["stations_en_recul"]


def test_evaluation_absente(tmp_path, monkeypatch):
    m = _module()
    monkeypatch.setattr("sys.argv", ["x", "--runs", str(tmp_path)])
    assert m.main() == 2
