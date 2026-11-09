"""DSO du modele national (pcfno3d.dso_national) : verdicts sur les chiffres reels de la
national v2 (test 2022, 09/11/2026), puis le calcul de bout en bout sur le mini-jeu ERA5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcfno3d.config import Config
from pcfno3d.dso_national import juger, rapport_md

# national v2, epoque 30, test 2022 (eval_pcfno3d/best_wcont/test_par_niveau_synoptique.csv)
V2 = [  # z, couverture, vitesse, rmse_u, r2_u, rmse_v, r2_v, rmse_w, r2_w, rugosite
    (25, .236, 5.370, .416, .992, .440, .987, .004, .937, .495),
    (50, .282, 5.389, .395, .993, .426, .988, .008, .863, .542),
    (75, .318, 5.469, .370, .994, .393, .990, .012, .811, .587),
    (100, .353, 5.588, .365, .995, .396, .991, .016, .791, .614),
    (150, .436, 5.653, .360, .995, .396, .991, .020, .782, .646),
    (200, .527, 5.688, .390, .994, .484, .986, .024, .796, .686),
    (300, .675, 6.088, .420, .994, .557, .984, .030, .813, .747),
    (400, .768, 6.319, .531, .991, .739, .974, .034, .840, .783),
    (500, .830, 6.479, .616, .989, .886, .966, .037, .856, .817),
    (650, .917, 6.612, .628, .989, .918, .968, .043, .839, .829),
    (800, .946, 6.574, .654, .987, .970, .966, .047, .824, .839),
    (1000, .978, 6.396, .615, .988, .935, .967, .050, .803, .850),
    (1250, .998, 6.268, .585, .988, .865, .971, .055, .747, .853),
    (1500, 1.0, 6.299, .644, .985, .925, .966, .057, .552, .864),
    (2000, 1.0, 6.943, .508, .991, .756, .980, .067, .635, .841),
    (2500, 1.0, 8.100, .769, .983, 1.158, .965, .071, .458, .784),
]
COLS = ["z_m", "couverture", "vitesse_moyenne", "rmse_u", "r2_u", "rmse_v", "r2_v",
        "rmse_w", "r2_w", "rugosite"]
SUMMARY_V2 = {"dso2_conforme": True, "dso2_rms_moyen": 2.37e-6, "dso4_inference_s": 0.248,
              "dso4_conforme": True,
              "robustesse": [{"capteurs_visibles": k, "rmse_u_max": u, "rmse_v_max": v}
                             for k, u, v in ((0.1, .7418, 1.1364), (0.25, .7401, 1.1360),
                                             (0.5, .7380, 1.1363), (1.0, .7362, 1.1410))]}
# stations_bilan.csv de la v2 (rmse_vect, modele / era5)
STATIONS_V2 = {"Bizerte": (2.926, 2.968), "Djerba-Zarzis": (2.466, 2.593),
               "Enfidha-Hammamet": (3.145, 3.142), "Gabes": (3.066, 2.933),
               "Gafsa": (3.491, 3.160), "Kairouan": (2.558, 2.358), "Monastir": (2.835, 2.680),
               "Remada": (3.008, 2.351), "Sfax": (2.864, 2.948), "Tabarka": (4.258, 3.242),
               "Tozeur": (3.241, 3.138), "Tunis-Carthage": (2.687, 2.776)}


def _niveaux(facteur_era5: float = 1.0):
    m = pd.DataFrame(V2, columns=COLS)
    e = m.copy()
    e[["rmse_u", "rmse_v"]] *= facteur_era5
    return m, e


def _stations():
    return pd.DataFrame([{"station": s, "source": src, "rmse_vect": val}
                         for s, (mo, er) in STATIONS_V2.items()
                         for src, val in (("modele", mo), ("era5", er))])


def test_verdicts_de_la_national_v2():
    cfg = Config()
    m, e = _niveaux(facteur_era5=1.2)            # ERA5 fictif 20 % moins bon partout
    v = juger(m, e, SUMMARY_V2, {"dso6_conforme": True}, _stations(), cfg)
    # DSO1a : 0,557 m/s (v a 300 m) > max(0,5 ; 0,609) ? non ; 0,739 a 400 m > 0,632 : echec
    assert v["dso1a_rmse_relatif"] is False
    assert v["dso1b_gain_era5"] is True and v["dso1b_gain_min"] == pytest.approx(1 - 1 / 1.2, abs=1e-3)
    assert v["dso1c_r2_uv"] is True
    assert v["dso1_conforme"] is False
    # DSO3 : niveaux couvrant >= 50 % (200 m et au-dessus)
    attendu = np.mean([r[9] for r in V2 if r[1] >= 0.5])
    assert v["dso3_rugosite"] == pytest.approx(attendu) and v["dso3_conforme"] is False
    # DSO5 : 1,1360 / 1,1410 - 1 = -0,4 % de degradation : conforme
    assert v["dso5_degradation"] == pytest.approx(1.1360 / 1.1410 - 1)
    assert v["dso5_conforme"] is True
    # DSO7 : 3,045 contre 2,857 m/s, 4 stations sur 12 meilleures qu'ERA5
    assert v["dso7_rmse_vect_modele"] == pytest.approx(3.0453, abs=1e-3)
    assert v["dso7_rmse_vect_era5"] == pytest.approx(2.8574, abs=1e-3)
    assert v["dso7_stations_meilleures_que_era5"] == 4 and v["dso7_conforme"] is False
    assert v["conformes"] == "4/7"           # DSO2, DSO4, DSO5, DSO6
    md = rapport_md(v)
    assert "DSO7" in md and "2500" in md


def test_gain_negatif_sur_era5_echoue():
    m, e = _niveaux(facteur_era5=0.9)            # ERA5 meilleur que le modele
    v = juger(m, e, SUMMARY_V2, None, None, Config())
    assert v["dso1b_gain_era5"] is False and v["dso1b_gain_min"] < 0
    assert v["dso6_conforme"] is None and v["dso7_conforme"] is None
    assert v["conformes"].endswith("/5")


def test_seuils_dans_la_config(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("dso_national:\n  z_max_m: 2500\n  rel: 0.2\n", encoding="utf-8")
    c = Config.load(f)
    assert c.dso_national.z_max_m == 2500 and c.dso_national.rel == 0.2
    m, e = _niveaux(1.2)
    v = juger(m, e, SUMMARY_V2, None, None, c)
    assert v["dso1a_rmse_relatif"] is True     # 20 % de la vitesse : tous les niveaux passent
    assert v["au_dessus_de_z_max"]["erreur_relative_max"] is None
