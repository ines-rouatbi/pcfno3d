"""Lot M2 — fiabilite du champ reconstruit (pcfno3d.reliability, DSO6).

Tests unitaires sur des tableaux construits a la main (statut, couverture, ensemble,
calibration sur un bruit CONNU), puis integration sur le mini-dataset : desactivee = sortie
inchangee bit a bit, activee = deterministe et sans effet sur la valeur servie.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pcfno3d.config import ReliabilityCfg
from pcfno3d.reliability import (CHAMPS_CONTRAT, CLASSES, COMPOSANTES, METHOD, NODATA_U8,
                                 STATUT_ESTIME, STATUT_MESURE, U8_SCALE, UNCERTAINTY_AUCUNE, Z80,
                                 Calibration, DSO6Accumulator, RatioHistogram,
                                 calibration_block, coverage_index, ensemble_sigma,
                                 fit_calibration, member_masks, node_status, quality_by_level,
                                 reliability_class, reliability_index, speed_interval,
                                 to_u8_unit)

# cles EXACTES du bloc uncertainty_method du contrat ATOM (common.schema.json,
# additionalProperties: false)
CLES_UNCERTAINTY = {"method", "members", "calibrated", "scale_factor", "coverage_80_validation"}
CLES_NOEUD = {"sigma_speed", "sigma_u", "sigma_v", "sigma_w", "speed_interval", "status",
              "coverage", "reliability", "reliability_class"}
# variables de meta.fields (contrat rev. 2) produites par M2, et leur codage (D13)
CHAMPS_REV2 = {"sigma_speed": "float32", "sigma_u": "float32", "sigma_v": "float32",
               "sigma_w": "float32", "coverage": "uint8", "reliability": "uint8",
               "status": "uint8"}


def _sensor(geom, frac=0.3, seed=0):
    return np.random.default_rng(seed).random((geom.ny, geom.nx)) < frac


# ------------------------------------------------------------------ statut
def test_statut_regle_d7_capteur_visible_a_15_m(geom):
    """Rev. 2, D7 : mesure <=> capteur VISIBLE dans la colonne et |dz| <= 15 m, sur donnees
    synthetiques comme reelles ; 255 sous le relief."""
    sensor = _sensor(geom)
    st = node_status(geom, sensor, dz_m=15.0)
    assert st.dtype == np.uint8 and st.shape == geom.shape
    dz = np.abs(np.asarray(geom.z_levels, float)[:, None, None] - geom.z_sensor[None])
    attendu = sensor[None] & (dz <= 15.0) & geom.valid
    assert ((st == STATUT_MESURE) == attendu).all()
    assert (st[geom.valid & ~attendu] == STATUT_ESTIME).all()
    assert (st[~geom.valid] == NODATA_U8).all()
    # sans capteur visible, rien n'est « mesure »
    vide = node_status(geom, np.zeros_like(sensor), dz_m=15.0)
    assert not (vide == STATUT_MESURE).any()


def test_statut_mesure_seulement_meme_colonne_proche(geom):
    sensor = np.zeros((geom.ny, geom.nx), bool)
    j, i = geom.ny // 2, geom.nx // 2
    sensor[j, i] = True
    st = node_status(geom, sensor, dz_m=15.0)
    dz = np.abs(np.asarray(geom.z_levels, float) - geom.z_sensor[j, i])
    attendu = (dz <= 15.0) & geom.valid[:, j, i]
    assert ((st[:, j, i] == STATUT_MESURE) == attendu).all()
    # aucune autre colonne n'est « mesuree »
    autre = st.copy()
    autre[:, j, i] = STATUT_ESTIME
    assert not (autre == STATUT_MESURE).any()


# ------------------------------------------------------------------ couverture
def test_couverture_bornee_nan_sous_relief_et_decroissante(geom):
    sensor = _sensor(geom, 0.2)
    cov = coverage_index(geom, sensor, 1000.0, 500.0, 2.0)
    assert cov.dtype == np.float32
    assert np.isnan(cov[~geom.valid]).all()
    v = cov[geom.valid]
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
    # decroit avec la hauteur au-dessus du sol, a colonne fixee
    for j, i in ((0, 0), (geom.ny // 2, geom.nx // 2)):
        col = cov[:, j, i][geom.valid[:, j, i]]
        assert (np.diff(col) <= 1e-7).all()


def test_couverture_decroit_avec_la_distance_au_capteur(geom):
    sensor = np.zeros((geom.ny, geom.nx), bool)
    j = geom.ny // 2
    sensor[j, 0] = True
    # H immense : seul le facteur horizontal joue (le relief varie le long de la ligne)
    cov = coverage_index(geom, sensor, 1000.0, 1e12, 2.0)
    k = int(np.argmax(geom.valid[:, j, :].all(axis=1)))    # un niveau valide sur la ligne
    ligne = cov[k, j, :]
    assert (np.diff(ligne) <= 1e-7).all() and ligne[0] > ligne[-1]
    # sans aucun capteur : couverture nulle partout
    vide = coverage_index(geom, np.zeros_like(sensor), 1000.0, 500.0, 2.0)
    assert np.nanmax(vide) == 0.0


# ------------------------------------------------------------------ ensemble
def test_masques_deterministes_et_inclus(geom):
    sensor = _sensor(geom, 0.4)
    a = member_masks(sensor, 6, 0.7, seed=3)
    b = member_masks(sensor, 6, 0.7, seed=3)
    assert all((x == y).all() for x, y in zip(a, b))
    assert all(not (m & ~sensor).any() for m in a)        # jamais un capteur invente
    assert len({m.tobytes() for m in a}) > 1               # des masques differents
    frac = np.mean([m.sum() / sensor.sum() for m in a])
    assert 0.6 < frac < 0.8


def test_sigma_ensemble_valeurs_connues():
    z = np.zeros((1, 1, 2), np.float32)
    served = (z, z, z)
    # u : ecarts +1 et -1 -> sigma 1 ; v : +3 et +3 -> 3 ; w nul
    members = [(z + 1, z + 3, z), (z - 1, z + 3, z)]
    sig = ensemble_sigma(served, members)
    np.testing.assert_allclose(sig["u"], 1.0)
    np.testing.assert_allclose(sig["v"], 3.0)
    np.testing.assert_allclose(sig["w"], 0.0)
    np.testing.assert_allclose(sig["speed"], np.sqrt(((np.hypot(1, 3) - 0) ** 2) * 2 / 2))


# ------------------------------------------------------------------ calibration
def _rcfg(**kw):
    return SimpleNamespace(**{**vars(ReliabilityCfg()), **kw})


def _bruit_connu(n, c, seed):
    """sigma brut variable, erreur vraie ~ N(0, (c sigma)^2) : le bon k vaut c."""
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.05, 2.0, n)
    err = np.abs(rng.normal(0.0, c * sigma))
    return err, sigma


def test_calibration_atteint_80_pourcent_sur_bruit_connu(geom):
    levels = geom.eval_levels()
    c_vrai = {"u": 1.7, "v": 0.6, "w": 2.5, "speed": 1.1}
    hist = RatioHistogram(geom.nz)
    for c in COMPOSANTES:
        for k in levels:
            hist.add(c, k, *_bruit_connu(20000, c_vrai[c], seed=k), floor=0.01)
    data = fit_calibration(hist, geom, "validation", _rcfg(calibration_q=0.8), n_instants=1)
    assert "regle_quantile" not in data and data["quantile"] == 0.8
    for c in COMPOSANTES:
        assert data["k_global"][c] == pytest.approx(c_vrai[c], rel=0.03)
        assert data["couverture80_validation"][c] == pytest.approx(0.8, abs=0.01)
    assert data["scale_factor"] == data["k_global"]["speed"]
    assert data["coverage_80_validation"] == data["couverture80_validation"]["speed"]

    # sur un tirage INDEPENDANT, l'intervalle calibre couvre ~80 % et le DSO6 est conforme
    cal = Calibration(data)
    acc = DSO6Accumulator(geom.nz)
    for c in COMPOSANTES:
        for k in levels:
            err, sigma = _bruit_connu(20000, c_vrai[c], seed=1000 + k)
            acc.add(c, k, err, sigma * cal.factor(c, k))
    s = acc.summary(geom)
    for c in COMPOSANTES:
        assert s["global"][f"couverture80_{c}"] == pytest.approx(0.8, abs=0.01)
        assert s["global"][f"spearman_{c}"] > 0.3
    assert s["dso6_correlation_ok"]
    # calibre a 80 % pile : le verdict strict >= 0,80 depend du tirage (d'ou calibration_q)
    assert s["dso6_couverture_ok"] == all(s["global"][f"couverture80_{c}"] >= 0.8
                                          for c in ("u", "v", "w"))
    assert s["coverage_80"] == s["global"]["couverture80_speed"]

    # avec une marge decidee sur la validation (q = 0,85), le test passe nettement
    data85 = fit_calibration(hist, geom, "validation", _rcfg(calibration_q=0.85), 1)
    assert data85["quantile"] == 0.85
    cal85, acc85 = Calibration(data85), DSO6Accumulator(geom.nz)
    for c in COMPOSANTES:
        for k in levels:
            err, sigma = _bruit_connu(20000, c_vrai[c], seed=1000 + k)
            acc85.add(c, k, err, sigma * cal85.factor(c, k))
    assert acc85.summary(geom)["dso6_conforme"]


def test_echantillon_spearman_uniforme_sur_les_instants():
    """L'echantillon du Spearman (au plus `cap` couples) represente TOUS les instants a
    parts egales : un re-tirage parmi anciens + nouveaux ne gardait que les derniers."""
    geom1 = SimpleNamespace(nz=1)
    acc = DSO6Accumulator(geom1.nz, cap=2000, seed=0)
    for t in range(50):                          # 50 instants de 7000 couples
        err = np.full(7000, float(t))            # le numero d'instant, relu dans l'echantillon
        acc.add("u", 0, err, np.ones(7000))
    t = acc.pairs["u"][0][:, 0]
    assert t.size == 2000 and acc.seen["u"][0] == 350000
    parts = np.bincount(t.astype(int), minlength=50) / t.size
    assert parts[:25].sum() == pytest.approx(0.5, abs=0.05)   # premiers instants presents
    assert parts.max() < 0.05                                  # aucun instant ne domine
    # deterministe a graine fixe
    acc2 = DSO6Accumulator(1, cap=2000, seed=0)
    for tt in range(50):
        acc2.add("u", 0, np.full(7000, float(tt)), np.ones(7000))
    np.testing.assert_array_equal(acc.pairs["u"][0], acc2.pairs["u"][0])


def test_sigma_non_calibre_et_sous_dispersion_detectee(geom):
    """k = 1 (non calibre) sur un bruit 2x plus fort que sigma : couverture bien < 80 %,
    DSO6 non conforme — le verdict ne se contente pas de la forme de l'intervalle."""
    levels = geom.eval_levels()
    acc = DSO6Accumulator(geom.nz)
    cal = Calibration.load("non_calibre")
    for c in COMPOSANTES:
        for k in levels:
            err, sigma = _bruit_connu(5000, 2.0, seed=k)
            acc.add(c, k, err, sigma * cal.factor(c, k))
    s = acc.summary(geom)
    assert s["global"]["couverture80_u"] < 0.6
    assert not s["dso6_couverture_ok"] and not s["dso6_conforme"]


def _plis_connus(geom, c_a, c_b, n=20000):
    """Deux plis de jours : bruit d'echelle c_a (pli a) et c_b (pli b), meme pour u, v, w,
    speed. Renvoie (pli a, pli b, somme)."""
    plis = (RatioHistogram(geom.nz), RatioHistogram(geom.nz))
    for h, cc, s0 in ((plis[0], c_a, 0), (plis[1], c_b, 500)):
        for c in COMPOSANTES:
            for k in geom.eval_levels():
                h.add(c, k, *_bruit_connu(n, cc, seed=s0 + k), floor=0.01)
    return plis[0], plis[1], RatioHistogram.total(list(plis))


def test_regle_auto_plus_petit_q_de_la_grille(geom):
    """Regle B : k sur un pli, couverture sur l'autre (deux sens) ; q retenu = plus petit q de
    la grille dont la couverture hors-pli minimale u, v, w atteint 0,80 ; k finaux sur le
    total avec ce q ; tout est ecrit dans le JSON."""
    from pcfno3d.reliability import Q_GRILLE, REGLE_Q, choose_q
    assert Q_GRILLE == (0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9)
    levels = list(geom.eval_levels())
    # pli b plus disperse que le pli a : q = 0,80 ne couvre pas b depuis a
    a, b, tot = _plis_connus(geom, 1.0, 1.12)
    r = choose_q((a, b), levels, geom.nz, _rcfg())
    assert r["regle"] == REGLE_Q and r["grille"] == list(Q_GRILLE) and r["atteint"]
    hp = r["couverture_hors_pli"]
    assert list(hp) == [f"{q:.2f}" for q in Q_GRILLE]
    for d in hp.values():
        assert d["min_uvw"] == min(d[s][c] for s in ("a_vers_b", "b_vers_a")
                                   for c in ("u", "v", "w"))
    q = r["q_retenu"]
    assert 0.8 < q <= 0.9
    assert hp[f"{q:.2f}"]["min_uvw"] >= 0.8
    assert all(hp[f"{x:.2f}"]["min_uvw"] < 0.8 for x in Q_GRILLE if x < q)
    # plis identiques en loi : la marge reste petite
    a2, b2, _ = _plis_connus(geom, 1.0, 1.0)
    assert choose_q((a2, b2), levels, geom.nz, _rcfg())["q_retenu"] <= 0.82

    # fit_calibration « auto » : le q retenu, les k finaux sur TOUTE la validation
    info = {"plis": "jours", "jours_a": [1], "jours_b": [2]}
    data = fit_calibration(tot, geom, "validation", _rcfg(), 3, plis=(a, b), plis_info=info)
    assert data["calibration_q"] == "auto" and data["quantile"] == q
    regle = data["regle_quantile"]
    assert regle["q_retenu"] == q and regle["plis"] == "jours" and "couverture_hors_pli" in regle
    for c in COMPOSANTES:
        assert data["k_global"][c] == tot.k(c, None, q)
        assert data["couverture80_validation"][c] >= q - 0.01
    json.dumps(data, allow_nan=False)

    # aucun q de la grille ne suffit : 0,90, signale
    a3, b3, _ = _plis_connus(geom, 1.0, 3.0)
    r3 = choose_q((a3, b3), levels, geom.nz, _rcfg())
    assert r3["q_retenu"] == 0.9 and r3["atteint"] is False


def test_regle_auto_exige_les_plis_et_refuse_le_test(geom):
    a, b, tot = _plis_connus(geom, 1.0, 1.0, n=500)
    with pytest.raises(ValueError, match="plis"):
        fit_calibration(tot, geom, "validation", _rcfg(), 1)
    with pytest.raises(ValueError, match="VALIDATION"):
        fit_calibration(tot, geom, "test", _rcfg(), 1, plis=(a, b))
    with pytest.raises(ValueError, match="vide"):
        fit_calibration(tot, geom, "validation", _rcfg(), 1,
                        plis=(a, RatioHistogram(geom.nz)))


def test_config_calibration_q_et_lot():
    assert ReliabilityCfg().calibration_q == "auto" and ReliabilityCfg().batch == 4
    assert ReliabilityCfg(calibration_q="0.82").calibration_q == 0.82
    assert ReliabilityCfg(calibration_q=" AUTO ").calibration_q == "auto"
    with pytest.raises(ValueError):
        ReliabilityCfg(calibration_q=1.2)
    with pytest.raises(ValueError):
        ReliabilityCfg(batch=0)


def test_calibration_refusee_sur_le_test(geom, tmp_path):
    hist = RatioHistogram(geom.nz)
    with pytest.raises(ValueError, match="VALIDATION"):
        fit_calibration(hist, geom, "test", _rcfg(), n_instants=1)
    # un JSON qui pretend venir du test est refuse au chargement
    bad = tmp_path / "cal.json"
    bad.write_text(json.dumps({"method": METHOD, "jeu": "test"}), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        Calibration.load(bad)


def test_bloc_uncertainty_cles_du_contrat(geom, tmp_path):
    nc = Calibration.load("non_calibre").uncertainty_method(8)
    assert set(nc) == CLES_UNCERTAINTY
    assert nc["calibrated"] is False and nc["scale_factor"] is None
    hist = RatioHistogram(geom.nz)
    for k in geom.eval_levels():
        for c in COMPOSANTES:
            hist.add(c, k, *_bruit_connu(2000, 1.3, seed=k), floor=0.01)
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(fit_calibration(hist, geom, "validation",
                                               _rcfg(calibration_q=0.8), 1)),
                    encoding="utf-8")
    cal = Calibration.load(path)
    um = cal.uncertainty_method(8)
    assert set(um) == CLES_UNCERTAINTY
    assert um["method"] == METHOD and um["members"] == 8 and um["calibrated"] is True
    assert um["scale_factor"] > 0 and 0 <= um["coverage_80_validation"] <= 1
    assert len(cal.sha256) == 64


# ------------------------------------------------------------------ fiabilite, intervalle
def test_indice_classe_et_intervalle():
    r = reliability_index(np.array([0.0, 0.2, 3.0, np.nan]), np.array([1.0, 0.5, 0.0, 0.5]),
                          sigma_ref=1.0)
    assert r[0] == pytest.approx(1.0) and r[2] == pytest.approx(0.5 * np.exp(-3.0))
    rc = ReliabilityCfg()
    assert (rc.class_high, rc.class_mid) == (0.75, 0.5)    # seuils du tableau de bord
    cl = reliability_class(r, rc.class_high, rc.class_mid)
    assert [CLASSES[x] for x in cl[:3]] == ["elevee", "moyenne", "faible"]
    assert cl[3] == NODATA_U8
    p10, p90 = speed_interval(np.array([5.0, 0.2]), np.array([1.0, 1.0]))
    assert p10[0] == pytest.approx(5 - Z80) and p90[0] == pytest.approx(5 + Z80)
    assert p10[1] == 0.0                                   # une vitesse n'est jamais < 0
    # formule exacte du contrat rev. 2 (D6) : p10 = max(0, speed - 1,2816 sigma_speed)
    rng = np.random.default_rng(0)
    speed, sig = rng.uniform(0, 20, 1000), rng.uniform(0, 5, 1000)
    p10, p90 = speed_interval(speed, sig)
    np.testing.assert_allclose(p10, np.maximum(0, speed - 1.2815515655446004 * sig), rtol=1e-6)
    np.testing.assert_allclose(p90, speed + 1.2815515655446004 * sig, rtol=1e-6)
    assert round(Z80, 4) == 1.2816


def test_codage_uint8_du_contrat():
    """Rev. 2, D13 : coverage et reliability en uint8 quantifie (0..254 -> 0..1, 255 absent)."""
    assert CHAMPS_CONTRAT == CHAMPS_REV2
    a = np.array([0.0, 1.0, 0.5, np.nan, 1.2, -0.1, 1 / 254])
    b = to_u8_unit(a)
    assert b.dtype == np.uint8
    assert b.tolist() == [0, 254, 127, 255, 254, 0, 1]
    ok = np.isfinite(a) & (a >= 0) & (a <= 1)
    assert np.abs(b[ok] * U8_SCALE - a[ok]).max() <= 0.5 / 254 + 1e-12
    assert UNCERTAINTY_AUCUNE == {"method": "aucune", "members": 1, "calibrated": False}


def test_classe_prise_sur_la_valeur_publiee():
    """Le tableau de bord reclasse depuis `reliability` publie (octet / 254) avec 0,75 / 0,5
    (fixtures du contrat, make-fixtures.mjs) : la classe de M2 doit etre la meme partout."""
    from pcfno3d.reliability import classe_octet
    # pres des seuils : 0,4985 est publie 127/254 = 0,5 -> « moyenne » pour le front ;
    # 0,75 tout juste est publie 190/254 = 0,748 (190,5 arrondi au pair) -> « moyenne »
    r = np.array([0.4985, 0.4975, 0.749, 0.7505, 0.75, 0.5], np.float32)
    assert [CLASSES[x] for x in reliability_class(r, 0.75, 0.5)] == \
        ["moyenne", "faible", "moyenne", "elevee", "moyenne", "moyenne"]
    # balayage : classe de M2 == classe recalculee comme le front, sur l'octet publie
    r = np.linspace(0.0, 1.0, 20001).astype(np.float32)
    q = to_u8_unit(r)
    front = np.where(q / 254 >= 0.75, 2, np.where(q / 254 >= 0.5, 1, 0))
    np.testing.assert_array_equal(reliability_class(r, 0.75, 0.5), front)
    np.testing.assert_array_equal(classe_octet(q, 0.75, 0.5), front)
    assert classe_octet(np.array([255], np.uint8), 0.75, 0.5)[0] == NODATA_U8


def test_blocs_quality_du_contrat(geom):
    """quality.schema.json : by_level (parts par classe, medianes) et calibration."""
    from pcfno3d.reliability import Reliability
    shp = geom.shape
    rng = np.random.default_rng(1)
    sig = np.where(geom.valid, rng.uniform(0, 2, shp), np.nan).astype(np.float32)
    cov = np.where(geom.valid, rng.uniform(0, 1, shp), np.nan).astype(np.float32)
    r = reliability_index(sig, cov, 1.0)
    rel = Reliability(sig, sig, sig, sig, node_status(geom, _sensor(geom), 15.0), cov, r,
                      reliability_class(r, 0.75, 0.5), UNCERTAINTY_AUCUNE)
    rows = quality_by_level(rel, geom)
    assert len(rows) == geom.nz
    cles = {"z_asl_m", "valid_nodes", "share_elevee", "share_moyenne", "share_faible",
            "sigma_speed_median", "coverage_median"}
    for k, row in enumerate(rows):
        assert set(row) == cles and row["valid_nodes"] == int(geom.valid[k].sum())
        if row["valid_nodes"]:
            assert row["share_elevee"] + row["share_moyenne"] + row["share_faible"] == \
                pytest.approx(1.0)
            assert row["sigma_speed_median"] == pytest.approx(
                float(np.median(sig[k][geom.valid[k]])))
        else:
            assert row["share_elevee"] is None and row["sigma_speed_median"] is None
    assert calibration_block(None) is None
    blk = calibration_block({"instants_synoptiques": 120, "capteurs_visibles": 0.5,
                             "coverage_80": 0.81, "corr_error_sigma": float("nan"),
                             "dso6_conforme": False})
    assert blk == {"evaluated_on": "120 instants de test, 50% des capteurs",
                   "coverage_80": 0.81, "corr_error_sigma": None, "dso6_pass": False}


# ------------------------------------------------------------------ integration
@pytest.fixture(scope="module")
def trained(cfg, index, norm):
    from pcfno3d.train import Trainer
    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg


@pytest.fixture(scope="module")
def instant(trained):
    from pcfno3d.cache import load_index
    stem = str(load_index(trained)["stem"].iloc[0])
    return stem


@pytest.fixture
def wf_fiable(trained):
    """Modele avec la fiabilite active (sigma bruts) ; la config est restauree apres."""
    from pcfno3d.inference import WindFieldModel
    trained.inference.reliability = "non_calibre"
    trained.reliability.members = 4
    try:
        yield WindFieldModel(trained, device="cpu", verbose=False)
    finally:
        trained.inference.reliability = None
        trained.reliability.members = ReliabilityCfg().members


def test_desactivee_rien_ne_change(trained, instant, tmp_path):
    from pcfno3d.export import export_web
    from pcfno3d.inference import WindFieldModel
    assert trained.inference.reliability is None
    wf = WindFieldModel(trained, device="cpu", verbose=False)
    assert wf.reliability is None and "fiabilite" not in wf.meta()
    vol = wf.predict_stem(instant)
    assert vol.reliability is None and vol.reliability_s == 0.0
    # meme sortie que le modele seul (passe directe, sans le code de M2), bit a bit
    import torch

    from pcfno3d.dataset import assemble_input
    from pcfno3d.features import time_geo_features
    S, hour, doy, _ = wf.surface_from_cache(instant)
    sensor = S[7] > 0.5
    tfeat = time_geo_features(S, sensor, hour, doy, wf.geom)
    with torch.no_grad():
        pred = wf.model(assemble_input(torch.from_numpy(S[None, :7]),
                                       torch.from_numpy(sensor[None].astype(np.float32)),
                                       torch.from_numpy(tfeat[None]), wf.geom, wf.norm))
    ref = [np.where(wf.geom.valid, a[0].numpy(), np.nan)
           for a in wf.norm.denorm(pred.float())]
    for a, b in zip((vol.u, vol.v, vol.w), ref):
        np.testing.assert_array_equal(a, b)
    files = set(np.load(vol.save_npz(tmp_path / "v.npz")).files)
    assert files == {"u", "v", "w", "speed", "direction", "z_levels", "x_m", "y_m",
                     "z_terrain", "timestamp"}
    manifest = json.loads((export_web(vol, tmp_path / "web") / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert "fiabilite" not in manifest
    assert not (tmp_path / "web" / "sigma_u.bin").exists()
    with pytest.raises(ValueError, match="inference.reliability"):
        wf.predict(*wf.surface_from_cache(instant)[:3], reliability=True)


def test_activee_deterministe_et_valeur_servie_inchangee(wf_fiable, instant, tmp_path):
    from pcfno3d.export import export_web
    wf = wf_fiable
    S, hour, doy, stamp = wf.surface_from_cache(instant)
    a = wf.predict(S, hour, doy, timestamp=stamp)
    b = wf.predict(S, hour, doy, timestamp=stamp)
    sans = wf.predict(S, hour, doy, timestamp=stamp, reliability=False)
    rel = a.reliability
    assert rel is not None and sans.reliability is None
    # la valeur servie ne depend pas de la fiabilite (passe unique, D6)
    for x, y in ((a.u, sans.u), (a.v, sans.v), (a.w, sans.w)):
        np.testing.assert_array_equal(x, y)
    # deterministe : memes graines, memes sigma
    for name in ("sigma_u", "sigma_v", "sigma_w", "sigma_speed", "coverage", "reliability"):
        np.testing.assert_array_equal(getattr(rel, name), getattr(b.reliability, name))
    g = wf.geom
    for name in ("sigma_u", "sigma_v", "sigma_w", "sigma_speed"):
        x = getattr(rel, name)
        assert np.isnan(x[~g.valid]).all() and (x[g.valid] >= 0).all()
    assert (rel.sigma_u[g.valid] > 0).any()
    # plancher de sigma, le meme qu'a la calibration (k = 1 en non_calibre)
    floor = np.float32(ReliabilityCfg().sigma_floor_m_s)
    for name in ("sigma_u", "sigma_v", "sigma_w", "sigma_speed"):
        assert (getattr(rel, name)[g.valid] >= floor).all()
    # D7 : les capteurs simules comptent ; meme regle que node_status
    sensor = S[7] > 0.5
    np.testing.assert_array_equal(rel.status, node_status(g, sensor, 15.0))
    assert (rel.status[g.valid] == STATUT_MESURE).any()          # la demo de BF7
    assert (rel.status[~g.valid] == NODATA_U8).all()
    assert rel.info["donnees_reelles"] is False
    assert set(np.unique(rel.reliability_class[g.valid])) <= {0, 1, 2}
    assert rel.uncertainty == {"method": METHOD, "members": 4, "calibrated": False,
                               "scale_factor": None, "coverage_80_validation": None}
    # fiabilite differee : inference_s = la valeur seule (DSO4), reliability_s = son delai
    assert a.reliability_state == "prete" and sans.reliability_state == "desactivee"
    assert a.reliability_s > 0 and a.inference_s > 0
    assert wf.meta()["fiabilite"]["uncertainty"]["members"] == 4

    # exposition : npz et manifeste web
    npz = np.load(a.save_npz(tmp_path / "v.npz"))
    assert {"sigma_u", "sigma_v", "sigma_w", "status", "coverage", "reliability",
            "reliability_class", "speed_p10", "speed_p90", "uncertainty"} <= set(npz.files)
    assert npz["status"].dtype == np.uint8 and npz["sigma_u"].dtype == np.float32
    out = export_web(a, tmp_path / "web", stride=2)
    fia = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["fiabilite"]
    assert fia["uncertainty"]["method"] == METHOD
    nz, ny, nx = g.nz, len(range(0, g.ny, 2)), len(range(0, g.nx, 2))
    assert (out / "sigma_u.bin").stat().st_size == nz * ny * nx * 4
    st = np.frombuffer((out / "status.bin").read_bytes(), np.uint8)
    assert st.size == nz * ny * nx and set(np.unique(st)) <= {0, 1, 255}
    assert (out / "sigma_speed.bin").stat().st_size == nz * ny * nx * 4
    assert npz["sigma_speed"].dtype == np.float32
    assert fia["champs"]["status"]["codes"] == {"0": "estime", "1": "mesure"}


def test_noeud_noms_du_contrat(wf_fiable, instant):
    vol = wf_fiable.predict_stem(instant)
    g = vol.geom
    k, j, i = (int(x[0]) for x in np.nonzero(g.valid))
    n = vol.node_reliability(k, j, i)
    assert set(n) == CLES_NOEUD
    assert n["status"] in ("estime", "mesure") and n["reliability_class"] in CLASSES
    si = n["speed_interval"]
    assert set(si) == {"p10", "p50", "p90"} and si["p10"] <= si["p50"] <= si["p90"]
    sp, sg = float(vol.speed[k, j, i]), float(vol.reliability.sigma_speed[k, j, i])
    assert si["p50"] == round(sp, 3)                            # p50 = vitesse servie (D6)
    assert si["p10"] == round(max(0.0, sp - Z80 * sg), 3)
    assert si["p90"] == round(sp + Z80 * sg, 3)
    assert n["sigma_speed"] == round(sg, 3)
    assert 0 <= n["coverage"] <= 1 and 0 <= n["reliability"] <= 1
    k2, j2, i2 = (int(x[0]) for x in np.nonzero(~g.valid))
    sous = vol.node_reliability(k2, j2, i2)
    assert set(sous) == CLES_NOEUD and sous["status"] is None and sous["sigma_u"] is None
    assert sous["sigma_speed"] is None


def test_api_profile_bloc_incertitude(wf_fiable, instant):
    pytest.importorskip("fastapi")
    from service import app as svc
    wf = wf_fiable
    vol = wf.predict_stem(instant)
    old = dict(svc.STATE)
    try:
        svc.STATE["model"] = wf
        svc.STATE["volumes"] = {f"{instant}:1.000": vol}
        body = svc.profile(x=0.0, y=0.0, stem=instant, _=None)
    finally:
        svc.STATE.clear()
        svc.STATE.update(old)
    assert set(body["uncertainty"]) == CLES_UNCERTAINTY
    for niv in body["niveaux"]:
        assert CLES_NOEUD <= set(niv)
        assert niv["status"] in ("estime", "mesure", None)
        assert (niv["status"] is None) == niv["sous_le_relief"]
    json.dumps(body, allow_nan=False)                      # JSON strict : aucun NaN


def test_calibration_puis_dso6_sur_le_mini_jeu(trained, tmp_path):
    from pcfno3d.evaluate import Evaluator
    trained.reliability.members = 3
    try:
        ev = Evaluator(trained, device="cpu", verbose=False)
        cal = ev.calibrate_reliability(out=tmp_path / "cal.json", n_instants=2)
        assert cal["jeu"] == "validation" and cal["instants"] == 2
        assert set(cal["k"]) == set(COMPOSANTES) and len(cal["k"]["u"]) == ev.geom.nz
        # une calibration retenue n'est jamais ecrasee
        with pytest.raises(FileExistsError):
            ev.calibrate_reliability(out=tmp_path / "cal.json", n_instants=1)
        s = ev.dso6(calibration=tmp_path / "cal.json", n_instants=2)
        # un verdict DSO6 n'est jamais ecrase (verifie AVANT le calcul)
        with pytest.raises(FileExistsError):
            ev.dso6(calibration=tmp_path / "cal.json", n_instants=2)
        assert s["jeu"] == "test" and s["uncertainty"]["calibrated"] is True
        assert 0 <= s["coverage_80"] <= 1
        assert {"dso6_conforme", "dso6_couverture_ok", "dso6_correlation_ok",
                "corr_error_sigma", "par_niveau"} <= set(s)
        written = json.loads(open(s["fichier"], encoding="utf-8").read())
        assert written["calibration_sha256"] == Calibration.load(tmp_path / "cal.json").sha256
        # regle auto (defaut) : regle, grille, couverture hors-pli de chaque q, q retenu ;
        # un seul jour de validation dans le mini-jeu -> repli par instants, SIGNALE
        regle = cal["regle_quantile"]
        assert cal["calibration_q"] == "auto" and cal["quantile"] == regle["q_retenu"]
        assert regle["q_retenu"] in regle["grille"] and len(regle["couverture_hors_pli"]) == 11
        assert regle["plis"].startswith("instants (repli")
        assert regle["instants_a"] == 1 and regle["instants_b"] == 1
        assert "mesure" not in s and s["calibration_quantile"] == cal["quantile"]
        # seconde lecture du test : le verdict le dit
        s2 = ev.dso6(calibration=tmp_path / "cal.json", n_instants=1,
                     out=tmp_path / "dso6_2.json", deuxieme_lecture=True)
        assert s2["mesure"] == ("seconde mesure sur le test, apres une premiere lecture "
                                "(15/10/2026, non conforme)")
        # plis par JOURS des qu'il y en a deux : rang pair / impair des jours presents
        tr = [str(x) for x in ev.splits.train[0]]
        pli, info = ev._plis_jours(SimpleNamespace(stems=tr))
        jours = sorted({int(ev._day_of[x]) for x in tr})
        assert info["plis"] == "jours" and len(jours) >= 2
        assert info["jours_a"] == jours[0::2] and info["jours_b"] == jours[1::2]
        assert all(pli[x] == jours.index(int(ev._day_of[x])) % 2 for x in tr)
        assert info["instants_a"] + info["instants_b"] == len(tr)
    finally:
        trained.reliability.members = ReliabilityCfg().members


def test_ensemble_par_lots_identique_passe_par_passe(trained, instant):
    """Ensemble par lots (reliability.batch) : memes sorties que passe par passe. Lot de 1 :
    bit a bit le calcul d'avant (ensemble_forward k=1, caracteristiques recalculees)."""
    import torch

    from pcfno3d.geometry import N_SURF
    from pcfno3d.inference import WindFieldModel, ensemble_forward
    wf = WindFieldModel(trained, device="cpu", verbose=False)
    S, hour, doy, _ = wf.surface_from_cache(instant)
    masks = member_masks(S[7] > 0.5, 7, 0.7, seed=11)
    ref = []
    for m in masks:                                    # l'ancien _member, recopie
        with torch.no_grad():
            pred = ensemble_forward(wf.model, wf.geom, wf.norm,
                                    torch.from_numpy(np.ascontiguousarray(S[:N_SURF]))[None],
                                    torch.from_numpy(m.astype(np.float32))[None], hour, doy,
                                    k=1)
        ref.append(tuple(a[0].numpy() for a in wf.norm.denorm(pred.float())))
    un = wf._members(S, masks, hour, doy, batch=1)
    for a, b in zip(un, ref):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
    for x, y in zip(wf._member(S, masks[2], hour, doy), ref[2]):
        np.testing.assert_array_equal(x, y)
    for lot in (3, 4, 8):                              # lot partiel, plein, plus grand que N
        par_lots = wf._members(S, masks, hour, doy, batch=lot)
        assert len(par_lots) == len(masks)
        for a, b in zip(par_lots, ref):
            for x, y in zip(a, b):
                np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)

    # predict avec fiabilite : sigma identiques quel que soit le lot
    trained.inference.reliability = "non_calibre"
    trained.reliability.members = 5
    try:
        wf2 = WindFieldModel(trained, device="cpu", verbose=False)
        sig = {}
        for lot in (1, 4):
            trained.reliability.batch = lot
            sig[lot] = wf2.predict(S, hour, doy).reliability
        assert sig[4].info["lot"] == 4
        for name in ("sigma_u", "sigma_v", "sigma_w", "sigma_speed"):
            np.testing.assert_allclose(getattr(sig[1], name), getattr(sig[4], name),
                                       rtol=1e-4, atol=1e-5)
    finally:
        trained.inference.reliability = None
        trained.reliability.members = ReliabilityCfg().members
        trained.reliability.batch = ReliabilityCfg().batch


def test_script_colab_ecrit_dans_un_dossier_neuf(trained, monkeypatch):
    """scripts/m2_calibration_colab.py sur le mini-jeu : session neuve, rien d'ecrase."""
    import importlib.util

    from pcfno3d.config import REPO_ROOT
    for var, p in (("ANEO_DATA", trained.paths.data_dir), ("ANEO_CACHE", trained.paths.cache_dir),
                   ("ANEO_RUN", trained.paths.run_dir)):
        monkeypatch.setenv(var, str(p))
    spec = importlib.util.spec_from_file_location(
        "m2_script", REPO_ROOT / "scripts" / "m2_calibration_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    avant = {p: p.stat().st_mtime_ns for p in trained.paths.run_dir.rglob("*") if p.is_file()}
    # 2 instants de validation : la regle auto a besoin de deux plis non vides
    rc = mod.main(["-c", "test", "--device", "cpu", "--instants-validation", "2",
                   "--instants-test", "1", "--membres", "2", "--benchmark", "1",
                   "--deuxieme-lecture"])
    assert rc == 0
    for p, t in avant.items():                       # aucun fichier existant reecrit
        assert p.stat().st_mtime_ns == t, p
    sessions = sorted((trained.paths.run_dir / "fiabilite_pcfno3d").glob("*/session_*"))
    assert sessions
    resume = json.loads((sessions[-1] / "resume.json").read_text(encoding="utf-8"))
    assert {"calibration", "dso6", "dso4"} <= set(resume)
    assert resume["mesure"].startswith("seconde mesure sur le test")
    assert resume["calibration"]["calibration_q"] == "auto"
    assert resume["calibration"]["regle_quantile"]["q_retenu"] == resume["calibration"]["quantile"]
    # fiabilite differee : DSO4 juge la valeur seule ; le delai de la fiabilite est a cote,
    # sans verdict (aucune cle `conforme` dans le delai)
    dso4 = resume["dso4"]
    assert dso4["valeur"]["mean_s"] > 0 and "conforme" in dso4["valeur"]
    assert dso4["mesure"] == "valeur seule (fiabilite differee)"
    assert dso4["delai_fiabilite"]["mean_s"] > 0 and "conforme" not in dso4["delai_fiabilite"]
    assert dso4["delai_fiabilite"]["membres"] == 2
    assert dso4["lot"] == ReliabilityCfg().batch
    assert set(dso4["delai_fiabilite_par_lot"]) == {"1", "4", "8"}
    assert all(b["mean_s"] > 0 and b["lot"] == int(k)
               for k, b in dso4["delai_fiabilite_par_lot"].items())
    verdict = json.loads((sessions[-1] / "dso6.json").read_text(encoding="utf-8"))
    assert verdict["mesure"] == resume["mesure"]
    for name in ("calibration_validation.json", "dso6.json"):
        assert (sessions[-1] / name).exists()

    # DSO4 seul, avec la calibration retenue : ni calibration ni DSO6, session neuve
    cal = sessions[-1] / "calibration_validation.json"
    assert mod.main(["-c", "test", "--device", "cpu", "--dso4-seulement"]) == 1
    assert mod.main(["-c", "test", "--device", "cpu", "--dso4-seulement", "--calibration",
                     str(cal), "--membres", "2", "--benchmark", "1"]) == 0
    seul = sorted((trained.paths.run_dir / "fiabilite_pcfno3d").glob("*/session_*"))[-1]
    assert seul != sessions[-1]
    assert sorted(p.name for p in seul.iterdir()) == ["resume_dso4.json"]
    d4 = json.loads((seul / "resume_dso4.json").read_text(encoding="utf-8"))["dso4"]
    assert d4["valeur"]["mean_s"] > 0 and d4["delai_fiabilite"]["mean_s"] > 0


def test_avec_correction_m1_chaque_passe_est_corrigee(trained, instant, tmp_path):
    """inference.correction ET inference.reliability : la passe servie et chaque passe de
    l'ensemble recoivent le meme post-traitement (sinon sigma mesurerait la correction)."""
    pytest.importorskip("pcfno3d.correction")
    from pcfno3d.correction import FEATURES_10H
    from pcfno3d.inference import WindFieldModel
    geom_ = WindFieldModel(trained, device="cpu", verbose=False).geom
    rng = np.random.default_rng(5)
    niveaux = {f"{int(geom_.z_levels[k])}_{c}": {
        "utilise": True, "mu": rng.normal(0, 1, 30).tolist(),
        "sd": (np.abs(rng.normal(1, 0.3, 30)) + 0.1).tolist(),
        "coef": rng.normal(0, 0.2, 31).tolist()} for k in geom_.eval_levels() for c in "uv"}
    path = tmp_path / "couche_limite.json"
    path.write_text(json.dumps({"variante_rho_sl": "1.2", "ridge_alpha": 0.01,
                                "features": list(FEATURES_10H), "niveaux": niveaux}),
                    encoding="utf-8")
    trained.inference.correction, trained.inference.reliability = path, "non_calibre"
    trained.reliability.members = 3
    try:
        wf = WindFieldModel(trained, device="cpu", verbose=False)
        S, hour, doy, stamp = wf.surface_from_cache(instant)
        vol = wf.predict(S, hour, doy, timestamp=stamp)
        sans = wf.predict(S, hour, doy, timestamp=stamp, reliability=False)
        assert vol.correction is not None and vol.reliability is not None
        np.testing.assert_array_equal(vol.u, sans.u)            # servie : inchangee
        # une passe de l'ensemble avec TOUS les capteurs = la passe servie, corrigee
        u, v, _ = wf._member(S, S[7] > 0.5, hour, doy)
        ok = wf.geom.valid
        np.testing.assert_allclose(u[ok], vol.u[ok], atol=1e-5)
        np.testing.assert_allclose(v[ok], vol.v[ok], atol=1e-5)
    finally:
        trained.inference.correction = trained.inference.reliability = None
        trained.reliability.members = ReliabilityCfg().members
