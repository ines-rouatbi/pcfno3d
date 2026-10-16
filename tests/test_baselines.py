"""Lot M8 : methodes de reference (L6), E1 / E2 / E3 (L9), E5 et script GPU.

Le mini-jeu verifie le MECANISME (memes instants, memes noeuds, ajustement sur la
validation seulement, dossier neuf, degradation propre sans CodeCarbon), pas la qualite :
les chiffres de la v7 viennent de `scripts/m8_preuves_colab.py` sur GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pcfno3d.baselines import (IDW, Krigeage, Variogramme, VariogrammeEmpirique,
                               ajuster_variogramme, charger_ajustement, ecrire_ajustement,
                               facteur_vertical, interpoler, paires_variogramme, references)
from pcfno3d.sobriete import Sobriete, energie_entrainement, historique_entrainement


# ------------------------------------------------------------------ lois et interpolation
def test_lois_verticales_valeurs_connues():
    h_s, z0 = np.array(10.0), np.array(0.1)
    assert facteur_vertical("log", 10.0, h_s, z0) == pytest.approx(1.0)
    # ln(100 / 0,1) / ln(10 / 0,1) = 3 / 2
    assert facteur_vertical("log", 100.0, h_s, z0) == pytest.approx(1.5)
    assert facteur_vertical("puissance", 40.0, h_s, alpha=0.5) == pytest.approx(2.0)
    assert facteur_vertical("puissance", 500.0, h_s, alpha=0.0) == pytest.approx(1.0)
    f = facteur_vertical("log", np.array([20.0, 50.0, 200.0, 1000.0]), h_s, z0)
    assert np.all(np.diff(f) > 0)
    with pytest.raises(ValueError):
        facteur_vertical("puissance", 50.0, h_s)          # alpha manquant
    with pytest.raises(ValueError):
        facteur_vertical("spline", 50.0, h_s, z0)


def test_idw_exact_aux_capteurs_et_champ_constant():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 1000, (50, 2))
    dst = np.vstack([src[:5], rng.uniform(0, 1000, (20, 2))])
    vals = np.stack([rng.normal(size=50), np.full(50, 3.0)])
    out = interpoler(IDW(), src, vals, dst)
    np.testing.assert_allclose(out[0, :5], vals[0, :5], atol=1e-9)
    np.testing.assert_allclose(out[1], 3.0, atol=1e-9)
    assert interpoler(IDW(), src[:0], vals[:, :0], dst).shape == (2, 25)   # aucun capteur


def test_krigeage_ordinaire_exact_poids_de_somme_un_et_systeme_complet():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 2000, (12, 2))
    dst = np.vstack([src[:3], rng.uniform(0, 2000, (7, 2))])
    vg = Variogramme(pepite=0.1, palier=1.0, portee_m=800.0)
    kr = Krigeage(vg, voisins=12)
    idx, w = kr.poids(src, dst)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-9)
    vals = rng.normal(size=(1, 12))
    out = interpoler(kr, src, vals, dst)
    np.testing.assert_allclose(out[0, :3], vals[0, :3], atol=1e-8)       # exact au capteur
    # voisinage = tous les capteurs : meme solution que le systeme complet ecrit a la main
    d = np.linalg.norm(src[:, None] - src[None], axis=-1)
    A = np.ones((13, 13))
    A[:12, :12] = vg.gamma(d)
    A[12, 12] = 0.0
    for m in range(len(dst)):
        b = np.ones(13)
        b[:12] = vg.gamma(np.linalg.norm(src - dst[m], axis=1))
        lam = np.linalg.solve(A, b)[:12]
        assert float(lam @ vals[0]) == pytest.approx(float(out[0, m]), abs=1e-8)


def test_variogramme_retrouve_la_portee_d_un_champ_connu():
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 5000, (300, 2))
    vrai = Variogramme(pepite=0.0, palier=1.0, portee_m=1500.0)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    cov = 1.0 - vrai.gamma(d)
    L = np.linalg.cholesky(cov + 1e-8 * np.eye(len(xy)))
    emp = VariogrammeEmpirique(50.0, 2500.0)
    for _ in range(40):
        z = L @ rng.normal(size=len(xy))
        i, j = paires_variogramme(xy, rng, 2500.0, n_centres=300, n_aleatoires=5000)
        emp.ajouter(xy, z, i, j)
    h, g, n = emp.table()
    fit = ajuster_variogramme(h, g, n, 50.0, 2500.0)
    assert 1000.0 < fit.portee_m < 2300.0
    assert fit.pepite < 0.2


def test_ajustement_du_test_refuse_et_jamais_ecrase(tmp_path):
    bon = {"jeu": "validation", "alpha": {"idw": 0.1, "krigeage": 0.1},
           "variogramme": {"modele": "exponentiel", "pepite": 0.0, "palier": 1.0,
                           "portee_m": 500.0}}
    p = ecrire_ajustement(bon, tmp_path / "a.json")
    assert charger_ajustement(p)["jeu"] == "validation"
    with pytest.raises(FileExistsError):
        ecrire_ajustement(bon, p)
    (tmp_path / "t.json").write_text(json.dumps({**bon, "jeu": "test"}), encoding="utf-8")
    with pytest.raises(ValueError, match="VALIDATION"):
        charger_ajustement(tmp_path / "t.json")
    # sans ajustement, seule la reference sans parametre existe
    assert list(references(None)) == ["log_idw"]
    assert set(references(bon)) == {"log_idw", "puissance_idw", "log_krigeage",
                                    "puissance_krigeage"}
    with pytest.raises(ValueError):
        references(None, ["log_krigeage"])


# ------------------------------------------------------------------ E5
def test_sobriete_sans_codecarbon_ni_gpu_n_invente_rien():
    with Sobriete("x", utiliser_codecarbon=False, utiliser_nvidia_smi=False) as s:
        pass
    r = s.resultat
    assert r["duree_s"] >= 0 and r["energie_kwh"] is None and r["co2_kg"] is None
    assert "a mesurer" in r["note"]
    with Sobriete("y", gpu_watts=360.0, intensite_gco2_kwh=500.0,
                  utiliser_codecarbon=False, utiliser_nvidia_smi=False) as s:
        pass
    r = s.resultat
    assert r["source_energie"] == "puissance_declaree"
    assert r["energie_kwh"] * 3.6e6 / 360.0 == pytest.approx(r["duree_s"], abs=1e-3)
    assert r["co2_kg"] == pytest.approx(r["energie_kwh"] * 0.5)


def test_historique_entrainement_notebook_et_history(tmp_path):
    nb = {"cells": [{"cell_type": "code", "outputs": [{"text": [
        "GPU    : NVIDIA A100-SXM4-80GB\n", "Reprise a l'epoque 1 (meilleure val = 0.1)\n",
        "Ep   2/3 | phys=1.00 | train data=0.1 | R2 u=0.9 | 100 s *\n",
        "Ep   3/3 | phys=1.00 | train data=0.1 | R2 u=0.9 | 110 s\n"]}]}]}
    p = tmp_path / "v.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    h = historique_entrainement(p)
    assert (h["epoques_prevues"], h["epoques_journalisees"], h["epoques_non_journalisees"]) \
        == (3, 2, 1)
    assert h["duree_journalisee_s"] == 210.0 and h["gpu"] == "NVIDIA A100-SXM4-80GB"
    assert h["reprise_a_l_epoque"] == 1
    assert energie_entrainement(h)["energie_kwh"] is None          # pas de puissance : rien
    assert energie_entrainement(h, 360.0)["energie_kwh"] == pytest.approx(210 * 360 / 3.6e6)
    hj = tmp_path / "history.json"
    hj.write_text(json.dumps([{"epoch": -1, "val_score": 1.0},
                              {"epoch": 0, "time_s": 5.0}, {"epoch": 1, "time_s": 7.0}]),
                  encoding="utf-8")
    h2 = historique_entrainement(hj)
    assert h2["epoques_journalisees"] == 2 and h2["duree_journalisee_s"] == 12.0


# ------------------------------------------------------------------ mini-jeu
@pytest.fixture(scope="module")
def trained(cfg, index, norm):
    from pcfno3d.train import Trainer
    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg


@pytest.fixture(scope="module")
def ev(trained):
    from pcfno3d.evaluate import Evaluator
    return Evaluator(trained, device="cpu", verbose=False)


@pytest.fixture(scope="module")
def ajustement(trained, ev):
    from pcfno3d.baselines import ajuster_sur_validation
    return ajuster_sur_validation(trained, ev.geom, ev.splits)


def test_ajustement_sur_la_validation_du_mini_jeu(ajustement):
    assert ajustement["jeu"] == "validation" and ajustement["graine"] == 1000
    assert ajustement["instants"] >= 1
    vg = ajustement["variogramme"]
    assert vg["portee_m"] > 0 and vg["palier"] > 0 and vg["pepite"] >= 0
    assert set(ajustement["alpha"]) == {"idw", "krigeage"}
    assert all(0.0 <= a <= 0.6 for a in ajustement["alpha"].values())


def test_metriques_partagees_identiques_a_l_evaluateur(trained, ev):
    """Le code de metriques de M8 EST celui de l'Evaluator : meme table pour PC-FNO3D,
    verite -> RMSE 0 et R2 1."""
    from pcfno3d.dataset import eval_loader
    from pcfno3d.evaluate import metriques_loader

    idx = np.arange(min(2, len(ev.splits.test[0])))
    loader, stems = eval_loader(trained, ev.geom, ev.splits, idx, 2000)
    lv_ref, pi_ref = ev.evaluate_loader(loader, stems)

    def verite(s_raw, sensor, tfeat, y, j, stem):
        return [y[:, 0], y[:, 1], y[:, 2]]

    res = metriques_loader(loader, stems, ev.geom, ev.splits.stamp_of,
                           {"a": ev.predire, "verite": verite}, ev.loss_fn.roughness_ratio)
    assert res["a"][0].equals(lv_ref) and res["a"][1].equals(pi_ref)
    lv = res["verite"][0]
    assert np.allclose(lv["rmse_u"].dropna(), 0.0) and np.allclose(lv["rmse_v"].dropna(), 0.0)
    assert np.allclose(lv["r2_u"].dropna(), 1.0)


def test_comparaison_memes_instants_et_noeuds(trained, ev, ajustement):
    from pcfno3d.dataset import eval_loader
    from pcfno3d.preuves import Banc, comparer, tableau_niveaux

    banc = Banc(trained, ev, ajustement)
    comp = comparer(banc, n_instants=2)
    assert set(comp["methodes"]) == {"pcfno3d", "log_idw", "puissance_idw", "log_krigeage",
                                     "puissance_krigeage"}
    assert comp["graine"] == 2000 and comp["jeu"] == "test"
    # la ligne pcfno3d = l'evaluation DSO1 du modele sur les memes instants
    lv_ref, _ = ev.evaluate_loader(*eval_loader(trained, ev.geom, ev.splits,
                                                banc.instants(2), 2000))
    pk = comp["methodes"]["pcfno3d"]["par_niveau"]
    got = [(r["rmse_u"], ref) for r, ref in zip(pk, lv_ref["rmse_u"]) if np.isfinite(ref)]
    assert got and all(g == pytest.approx(ref) for g, ref in got)
    assert sum(r["rmse_u"] is None for r in pk) == int(lv_ref["rmse_u"].isna().sum())
    # memes noeuds pour toutes les methodes : meme couverture et meme vitesse moyenne
    t = tableau_niveaux(comp)
    for _, g in t.groupby("z_m"):
        assert g["vitesse_moyenne"].nunique() == 1
    for r in comp["methodes"].values():
        assert r["rmse_max_uv"] is not None and r["rmse_max_uv"] >= 0
    json.dumps(comp)                                 # JSON strict (aucun NaN brut)
    # references seules, sans poids : meme resultat pour les references
    seules = comparer(Banc(trained, None, ajustement), n_instants=2)
    assert "pcfno3d" not in seules["methodes"]
    assert seules["methodes"]["log_idw"]["rmse_max_uv"] == pytest.approx(
        comp["methodes"]["log_idw"]["rmse_max_uv"])


def test_e1_e2_e3_sur_le_mini_jeu(trained, ev, ajustement):
    from pcfno3d.preuves import Banc, capteur_secours, etude_densite, placement_optimal

    banc = Banc(trained, ev, ajustement, ["log_idw", "log_krigeage"])
    e1 = etude_densite(banc, fractions=(0.25, 1.0), espacements=(300.0,), n_instants=2)
    fr = [x for x in e1["lignes"] if x["methode"] == "pcfno3d" and x["reseau"] == "fraction"]
    assert [x["capteurs_visibles"] for x in fr] == [0.25, 1.0]
    assert fr[0]["capteurs_moyen"] < fr[1]["capteurs_moyen"]          # masques emboites
    assert set(e1["recommandation"]) == {"pcfno3d", "log_idw", "log_krigeage"}

    trained.reliability.members = 2
    try:
        e2 = placement_optimal(banc, base_espacement_m=300.0, n_etapes=2, par_etape=2,
                               n_validation=1, n_test=2, n_hasard=1)
    finally:
        from pcfno3d.config import ReliabilityCfg
        trained.reliability.members = ReliabilityCfg().members
    assert e2["stations_ajoutees"] == len(e2["emplacements"]) <= 4
    assert {x["etape"] for x in e2["emplacements"]} <= {1, 2}
    c = e2["_cartes"]
    assert c["reseau_final"].sum() == c["base"].sum() + e2["stations_ajoutees"]
    assert set(e2["resume"]["rmse_max_uv"]) >= {"base", "optimal", "hasard_moyen",
                                                "gain_vs_hasard_m_s"}

    e3 = capteur_secours(banc, n_instants=2, n_pannes=4, espacement=2)
    assert set(e3["resume"]) == {"plus_proche", "idw", "krigeage", "pcfno3d"}
    df = e3["_lignes"]
    assert len(df) == e3["resume"]["idw"]["n"] > 0
    assert {"u_mesure", "u_pcfno3d", "u_krigeage"} <= set(df.columns)


def test_script_m8_dossier_neuf(trained, monkeypatch):
    """scripts/m8_preuves_colab.py sur le mini-jeu : session neuve, rien d'ecrase."""
    import importlib.util

    from pcfno3d.config import REPO_ROOT
    for var, p in (("ANEO_DATA", trained.paths.data_dir), ("ANEO_CACHE", trained.paths.cache_dir),
                   ("ANEO_RUN", trained.paths.run_dir)):
        monkeypatch.setenv(var, str(p))
    spec = importlib.util.spec_from_file_location(
        "m8_script", REPO_ROOT / "scripts" / "m8_preuves_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    avant = {p: p.stat().st_mtime_ns for p in trained.paths.run_dir.rglob("*") if p.is_file()}
    rc = mod.main(["-c", "test", "--device", "cpu", "--instants", "2",
                   "--instants-validation", "1", "--fractions", "0.5,1", "--espacements", "300",
                   "--e2-etapes", "1", "--e2-par-etape", "2", "--e2-validation", "1",
                   "--e2-hasard", "1", "--membres", "2", "--e3-pannes", "3",
                   "--e5-instants", "1", "--sans-codecarbon"])
    assert rc == 0
    for p, t in avant.items():                       # aucun fichier existant reecrit
        assert p.stat().st_mtime_ns == t, p
    sessions = sorted((trained.paths.run_dir / "preuves_pcfno3d").glob("*/session_*"))
    assert sessions
    s = sessions[-1]
    for name in ("ajustement_validation.json", "comparaison_test.json", "e1_densite.json",
                 "e2_placement.json", "e2_cartes.npz", "e3_secours.json", "e5_sobriete.json",
                 "resume.json"):
        assert (s / name).exists(), name
    for name in ("resume.json", "comparaison_test.json"):
        assert b"\r\n" not in (s / name).read_bytes()      # fins de ligne LF
    resume = json.loads((s / "resume.json").read_text(encoding="utf-8"))
    assert {"ajustement", "comparaison", "e1", "e2", "e3", "e5"} <= set(resume)
    e5 = resume["e5"]
    assert e5["inference_pcfno3d"]["par_reconstruction"]["duree_s"] > 0
    # sans CodeCarbon ni nvidia-smi sur ce poste : energie None, jamais un chiffre invente
    if not resume["versions"]["nvidia_smi"]:
        assert e5["inference_pcfno3d"]["phase"]["energie_kwh"] is None
    assert e5["entrainement_v7"]["historique"]["epoques_journalisees"] > 0


def test_cli_baselines_sans_modele(trained, tmp_path):
    """`pcfno3d baselines` : references seules (aucun poids), JSON neuf, jamais ecrase."""
    from pcfno3d.cli import main

    out = tmp_path / "comp.json"
    base = ["-c", "test", "--set", f"paths.data_dir={json.dumps(str(trained.paths.data_dir))}",
            "--set", f"paths.cache_dir={json.dumps(str(trained.paths.cache_dir))}",
            "--set", f"paths.run_dir={json.dumps(str(trained.paths.run_dir))}"]
    assert main(base + ["baselines", "--instants", "2", "--out", str(out)]) == 0
    comp = json.loads(out.read_text(encoding="utf-8"))
    assert list(comp["methodes"]) == ["log_idw"]
    assert main(base + ["baselines", "--instants", "2", "--out", str(out)]) == 1
