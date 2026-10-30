"""AGRI1 / AGRI2 : scripts/verifier_decisions.py (regle figee, accord, kappa, faux oui)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("vd", REPO / "scripts" / "verifier_decisions.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("kmh, attendu", [(0, "attention"), (2.9, "attention"), (3.0, "oui"), (15, "oui"),
                                          (15.1, "attention"), (19, "attention"), (19.1, "non"), (60, "non")])
def test_regle_traitement(kmh, attendu):
    assert _mod().decision_traitement(np.array([kmh / 3.6]))[0] == attendu


def test_accord_kappa_faux_oui(tmp_path, monkeypatch):
    m = _mod()
    # 4 heures : station oui, oui, non, non ; modele oui, non, oui, non
    p = pd.DataFrame({"station": ["A"] * 4, "temps": pd.date_range("2022-01-10 08:00", periods=4, freq="3h"),
                      "u_obs": [2.0, 2.0, 8.0, 8.0], "v_obs": 0.0,
                      "u_modele": [2.0, 8.0, 2.0, 8.0], "v_modele": 0.0,
                      "u_era5": [2.0, 2.0, 8.0, 8.0], "v_era5": 0.0})
    t = m.evaluer(p, ("modele", "era5"))
    r = t[(t.regle == "traitement") & (t.source == "modele") & (t.periode == "toutes heures") & (t.station == "toutes")].iloc[0]
    assert r.accord == pytest.approx(0.5) and r.faux_oui == pytest.approx(0.25)
    assert r.faux_oui_parmi_non == pytest.approx(0.5)
    e = t[(t.regle == "traitement") & (t.source == "era5") & (t.periode == "toutes heures") & (t.station == "toutes")].iloc[0]
    assert e.accord == 1.0 and e.kappa == pytest.approx(1.0) and e.faux_oui == 0.0
    # sortie complete
    f = tmp_path / "paires.csv"
    p.to_csv(f, index=False)
    monkeypatch.setattr(sys, "argv", ["vd", "--paires", str(f), "--sources", "modele,era5", "--sortie", str(tmp_path / "o")])
    assert m.main() == 0
    j = json.loads((tmp_path / "o" / "decisions_agricoles.json").read_text(encoding="utf-8"))
    assert j["agri2"]["era5 / toutes heures"]["conforme"] is True
    assert j["agri2"]["modele / toutes heures"]["conforme"] is False
