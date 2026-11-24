"""Regle OS4 v4_3 (regle_os4_oi_v4_3) : fonctions de calcul, choix du reglage, metriques et bootstrap, sans donnees."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("os4v43", REPO / "scripts" / "os4_oi_v4_3.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_regle_hachee_et_valeurs():
    from pcfno3d.regles import verifier
    r, h = verifier(m.REGLE)
    assert h == "21cb71ba14afaeb574eeac5916343ee742236764000b7b8141a362144ff62541"
    g = r["etape_1_choix_sur_validation_2021"]["grille"]
    assert g["L_km"] == [55, 100, 150] and g["lambda"] == [0.065, 0.25, 1.0, 4.0] and g["H_m"] == [300, 1000, 3000]
    assert g["combinaisons"] == 72 == 2 * 3 * 4 * 3
    assert "25 m" in r["ce_qui_ne_change_pas"]["capteurs"] and r["ce_qui_ne_change_pas"]["N"] == [10, 25, 50, 100, 200, 400]
    assert "N = 50 et N = 200" in r["etape_1_choix_sur_validation_2021"]["critere_de_choix"]
    assert "borne BASSE" in r["critere_os4"]["verdict"] and ">= 10 %" in r["critere_os4"]["verdict"]
    assert "N = 50" in r["critere_os4"]["verdict"]
    t = r["transparence"]
    assert "erreur de formulation" in t["erreur_de_formulation"] and "NON parce que" in t["erreur_de_formulation"]
    assert "APRES la lecture du test 2022" in t["redaction_apres_lecture"] and "RELIT" in t["redaction_apres_lecture"]
    assert "verdict d'origine" in t["v4_2_reste_publiee"]
    assert any("profils verticaux" in x for x in r["interdits"])
    assert "10 unites" in r["cout_et_budget"]


def test_forme_gaussienne_identique_a_v4_2():
    yx = np.array([[0.0, 0.0], [30.0, 40.0], [200.0, 0.0]])
    z = np.full(3, 25.0)
    d = np.array([[1.0, -1.0], [0.5, 0.2], [-0.3, 0.4]])
    assert np.allclose(m.alpha_oi(yx, z, d, "gaussienne", 150.0, 4.0, 300.0), m.V2.alpha_oi(yx, z, d, 150.0, 4.0, 300.0))
    var = np.array([1.0, 0.5, 0.4, 0.3, 2.0])
    cand = np.array([[0, 0], [0, 400], [400, 0], [400, 400], [200, 200]], float)
    o, g = m.glouton(var, cand, 5, "gaussienne", 150.0, 4.0)
    o2, g2 = m.V2.glouton(var, cand, 5, 150.0, 4.0, gains=True)
    assert list(o) == list(o2) and np.allclose(g, g2)


def test_forme_exponentielle_et_un_capteur():
    assert np.isclose(m.rho_h(np.array(55.0), "exponentielle", 55.0), np.exp(-1))
    assert np.isclose(m.rho_h(np.array(55.0), "gaussienne", 55.0), np.exp(-0.5))
    with pytest.raises(ValueError):
        m.rho_h(np.array(1.0), "autre", 1.0)
    al = m.alpha_oi(np.zeros((1, 2)), np.array([25.0]), np.array([[2.0, -2.0]]), "exponentielle", 55.0, 0.065, 300.0)
    assert np.allclose(al, [[2.0 / 1.065, -2.0 / 1.065]])       # lambda petit : le capteur corrige presque tout


def test_choix_et_ex_aequo():
    g = pd.DataFrame([
        {"forme": "exponentielle", "L_km": 55.0, "lambda": 0.065, "H_m": 1000.0, "rmse_0_100m_N50": 1.40, "rmse_0_100m_N200": 1.20},
        {"forme": "gaussienne", "L_km": 100.0, "lambda": 0.25, "H_m": 300.0, "rmse_0_100m_N50": 1.401, "rmse_0_100m_N200": 1.200},
        {"forme": "gaussienne", "L_km": 150.0, "lambda": 4.0, "H_m": 300.0, "rmse_0_100m_N50": 1.60, "rmse_0_100m_N200": 1.50},
    ])
    c = m.choisir_reglage(g)
    assert c["forme"] == "gaussienne" and c["L_km"] == 100.0 and c["ex_aequo"] == 2      # ecart < 0,002 : la plus proche de v4_2
    g.loc[1, "rmse_0_100m_N50"] = 1.45
    c = m.choisir_reglage(g)
    assert (c["forme"], c["L_km"], c["lambda"], c["H_m"]) == ("exponentielle", 55.0, 0.065, 1000.0)


def test_metriques_d_une_couche():
    tu = np.array([1.0, 0.0, 3.0, 0.2])
    tv = np.array([0.0, 2.0, 0.0, 0.1])
    au, av = tu + np.array([0.0, 0.0, 0.0, 5.0]), tv.copy()            # une seule erreur, sur un vent faible
    b = m.stats_couche(au, av, tu, tv)
    S = np.zeros(len(m.CHAMPS))
    for k, v in b.items():
        S[m.IX[k]] = v
    r = m.metriques(S[None], np.array([b["b_linf"]]))
    assert np.isclose(r["rmse_vect_0_100m"][0], np.sqrt(25 / 4)) and np.isclose(r["linf_0_100m"][0], 5.0)
    assert np.isclose(r["biais_u_0_100m"][0], 5 / 4) and np.isclose(r["biais_v_0_100m"][0], 0.0)
    assert np.isclose(r["mae_direction_0_100m_deg"][0], 0.0)           # le seul ecart est sur un vent < 1 m/s
    sst = (tu ** 2).sum() - tu.sum() ** 2 / 4
    assert np.isclose(r["r2_u_0_100m"][0], 1 - 25 / sst) and np.isclose(r["r2_v_0_100m"][0], 1.0)
    assert np.isclose(m.direction(np.array([0.0]), np.array([-1.0]))[0], 0.0)    # vent du nord
    b2 = m.stats_couche(np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))
    assert np.isclose(b2["b_dir"], 90.0)                                          # ouest contre sud : 90 deg


def test_resume_bootstrap_et_verdict():
    rng = np.random.default_rng(0)
    n_inst, n_cles = 12, 3
    S = np.zeros((n_inst, n_cles, len(m.CHAMPS)))
    for i in range(n_inst):
        base = 10 + rng.random()
        for k, f in enumerate((1.0, 0.8, 0.95)):                        # cle 1 : 20 % d'erreur en moins
            S[i, k, m.IX["b_sse"]] = f ** 2 * base
            S[i, k, m.IX["b_n"]] = 10
            S[i, k, m.IX["m10_sse"]], S[i, k, m.IX["m10_n"]] = f ** 2 * 5, 5
            S[i, k, m.IX["c_sse"]], S[i, k, m.IX["c_n"]] = 3, 3
            for t in range(4):
                S[i, k, m.IX[f"t{t}_su"]], S[i, k, m.IX[f"t{t}_sv"]], S[i, k, m.IX[f"t{t}_n"]] = 1, 1, 1
    L = np.ones((n_inst, n_cles))
    jours = np.repeat(np.arange(4), 3)
    P = m.poids_bootstrap(jours, n=200)
    assert P.shape == (200, n_inst) and np.allclose(P.sum(1), n_inst)
    R = m.resumer(S, L, P)
    g = R["rmse_vect_0_100m"]
    assert np.isclose(g["gain"][0], 0.0) and np.isclose(g["gain"][1], 0.2, atol=1e-6)
    assert g["ic_gain"][0][1] <= 0.2 + 1e-9 <= g["ic_gain"][1][1] + 1e-9
    assert np.allclose(R["rmse_vect_0_1km"]["gain"], 0.0)
    assert m.verdict(0.10) == "CONFORME" and m.verdict(0.0999) == "NON CONFORME"


def test_le_choix_ne_lit_jamais_le_test():
    with pytest.raises(SystemExit, match="ne lit jamais le test"):
        m.main(["choisir", "-c", "test", "--degradation", "x.json", "--sortie", "y", "--lecture-test"])


def test_carnet_v4_3():
    s = importlib.util.spec_from_file_location("nb43", REPO / "scripts" / "make_os4_oi_v4_3_nb.py")
    g = importlib.util.module_from_spec(s)
    s.loader.exec_module(g)
    nb = g.carnet("a" * 40)
    x = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "a" * 40 in x and g.SHA in x and "os4_oi_v4_3.py" in x and "runtime.unassign()" in x
    assert x.index("--essai") < x.index('base.format(etape="choisir"))') < x.index('base.format(etape="juger"))') \
        < x.index("--lecture-test")
    assert '"os4_oi_v4_3"' in x and "gpuType" not in nb["metadata"].get("colab", {})
    calcul = "".join([c for c in nb["cells"] if c["cell_type"] == "code"][-1]["source"])
    for interdit in ("rmtree", "unlink(", "shutil.move(", "rename("):
        assert interdit not in calcul, interdit                    # rien n'est supprime ni touche sur le Drive
