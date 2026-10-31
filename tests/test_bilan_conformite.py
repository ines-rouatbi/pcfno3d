"""Bilan de conformite au cahier v12 (scripts/bilan_conformite.py) : lecture seule des resultats, IDW de
REEL1 calcule sans la station jugee, statuts conforme / non conforme / a mesurer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from test_decision_v4 import _ecrire, _jeu

REPO = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("bilan_conformite", REPO / "scripts" / "bilan_conformite.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_idw_sans_la_station_jugee():
    m = _script()
    p = pd.DataFrame([{"station": s, "temps": "t0", "u_obs": u, "v_obs": 0.0}
                      for s, u in (("Tunis-Carthage", 10.0), ("Zaghouan", 2.0), ("Nabeul", 4.0))])
    q = m.idw_stations(p).set_index("station")
    # Tunis-Carthage : uniquement Zaghouan et Nabeul, ponderes par 1/d^2 ; jamais sa propre mesure (10)
    assert 2.0 < q.loc["Tunis-Carthage", "u_idw"] < 4.0
    p.loc[p["station"] == "Nabeul", "u_obs"] = np.nan
    assert m.idw_stations(p).set_index("station").loc["Tunis-Carthage", "u_idw"] == 2.0


def test_bilan(tmp_path):
    m = _script()
    s, lv = _jeu(rmse=0.4)
    s["par_niveau_synoptique"] = lv.to_dict(orient="records")
    _ecrire(tmp_path, "v4", s, lv, {"Nabeul": 2.4})
    (tmp_path / "decision_v4.json").write_text(json.dumps({"modele_livre": "v4", "texte": "v4"}), encoding="utf-8")
    reel = tmp_path / "v4" / "validation_reelle" / "test"
    lignes = []
    for k in range(6):
        for st, uo in (("Tunis-Carthage", 9.0), ("Zaghouan", 1.0), ("Nabeul", 5.0)):
            lignes.append({"station": st, "temps": f"t{k}", "u_obs": uo, "v_obs": 1.0, "u_modele": uo + 0.05,
                           "v_modele": 1.0, "u_era5": uo + 1.5, "v_era5": 1.0})
    pd.DataFrame(lignes).to_csv(reel / "stations_paires.csv", index=False)
    pd.DataFrame([{"site": "Tunis-Carthage", "temps": f"t{k}", "h_agl_m": h, "u_obs": 5.0, "v_obs": 0.0,
                   "u_modele": 5.5, "v_modele": 0.0, "u_era5": 5.0 + (2.0 if h < 300 else 0.1), "v_era5": 0.0}
                  for k in range(3) for h in (50, 200, 600)]).to_csv(reel / "radiosondages_paires.csv", index=False)
    (tmp_path / "os6").mkdir()
    (tmp_path / "os6" / "os6.json").write_text(json.dumps({
        "nombre_recommande": 40, "seuil_rmse_m_s": 0.5, "courbe": [{"n_capteurs": 10}],
        "comparaison_net1": {"10": {"optimise": 0.8, "grille": 1.0, "hasard": 1.1},
                             "40": {"optimise": 0.6, "grille": 0.75, "hasard": 0.8}}}), encoding="utf-8")
    b = m.bilan(tmp_path)
    c = {l["code"]: l for l in b["criteres"]}
    assert b["modele"] == "v4" and c["DSO1"]["statut"] == "conforme"
    # IDW des autres stations : 2,7 / 7,4 / 5,2 m/s (ecarts 6,3 / 6,4 / 0,2) ; modele a 0,05 -> conforme
    assert c["REEL1"]["statut"] == "conforme" and set(c["REEL1"]["detail"]) == {"Tunis-Carthage", "Zaghouan", "Nabeul"}
    assert c["EOL2"]["statut"] == "conforme"            # 0-250 m : ERA5 a 2 m/s, modele a 0,5
    assert c["REEL2"]["statut"] == "conforme"           # 25-1 000 m : 0,5 contre ~1,6
    assert c["NET1"]["statut"] == "conforme" and c["NET1"]["valeur"] == 0.2 and c["NET2"]["valeur"] == 40
    assert c["NET2"]["statut"] == "conforme"
    assert c["PREV1"]["statut"] == "à mesurer" and c["AGRI2"]["statut"] == "à mesurer"
    assert c["DSO7"]["statut"] == "proposé" and c["UX1"]["statut"] == "à mesurer"
    texte = m.rapport(b)
    assert "| REEL1 |" in texte and "Bilan de conformité" in texte
    assert m.main(["--runs", str(tmp_path)]) == 0 and (tmp_path / "bilan_conformite.md").exists()
    # aucun nombre de capteurs n'atteint le seuil : la courbe seule ne rend pas NET2 conforme
    o = json.loads((tmp_path / "os6" / "os6.json").read_text(encoding="utf-8"))
    (tmp_path / "os6" / "os6.json").write_text(json.dumps({**o, "nombre_recommande": None}), encoding="utf-8")
    c = {l["code"]: l for l in m.bilan(tmp_path)["criteres"]}
    assert c["NET2"]["statut"].startswith("non conforme")


def test_references_de_tests_existent():
    import re
    m = _script()
    for code, _, _, source in m.DOCUMENTES:
        for f in re.findall(r"tests/[\w*]+\.py", source):
            assert (REPO / f).exists(), (code, f)
