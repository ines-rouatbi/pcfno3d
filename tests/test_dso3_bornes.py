"""DSO3 : rugosite prediction / verite ENTRE 0,9 et 1,1 (cahier des charges).

La v7 n'etait jugee que sur la borne haute (plage (0,0 ; 1,1) du notebook). Un champ trop
lisse (< 0,9) est pourtant une faute au meme titre qu'un champ bruite (> 1,1) : le verdict
et le score de selection DSO doivent penaliser les deux cotes.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _metrics(rough: float, rmse: float = 0.3, nz: int = 2):
    """Cumuls factices : RMSE `rmse` a chaque niveau, R2 proche de 1, residu faible."""
    from pcfno3d.dso import LevelMetrics

    m = LevelMetrics(nz)
    n = 1000.0
    m.n[:] = n
    m.sse[:] = rmse ** 2 * n
    m.s1[:] = 0.0
    m.s2[:] = 100.0 * n                     # variance de la verite 100 (m/s)^2 : R2 ~ 1
    m.mass.append(1e-5)
    m.rough.append(rough)
    return m


GEOM = SimpleNamespace(eval_levels=lambda: [0, 1], z_levels=np.array([500.0, 1000.0]))


# ---------------------------------------------------------------------------- config
def test_plage_par_defaut_est_celle_du_cahier_des_charges():
    from pcfno3d.config import Config, DSOCfg

    assert DSOCfg().roughness_range == (0.9, 1.1)
    assert Config.load("default").dso.roughness_range == (0.9, 1.1)


def test_aucun_yaml_ne_revient_a_l_ancienne_plage():
    """Un YAML qui surchargerait (0,0 ; 1,1) annulerait la borne basse sans bruit."""
    import yaml

    from pcfno3d.config import CONFIG_DIR

    vus = 0
    for f in sorted(CONFIG_DIR.glob("*.yaml")):
        raw = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        plage = (raw.get("dso") or {}).get("roughness_range")
        if plage is not None:
            vus += 1
            assert tuple(plage) == (0.9, 1.1), f.name
    assert vus > 0


@pytest.mark.parametrize("plage", [(0.9, 1.0), (1.0, 1.1), (-0.1, 1.1), (1.1, 0.9)])
def test_plage_incoherente_refusee(plage):
    from pcfno3d.config import DSOCfg

    with pytest.raises(ValueError, match="roughness_range"):
        DSOCfg(roughness_range=plage)


# ---------------------------------------------------------------------------- verdict
@pytest.mark.parametrize("rough, attendu", [
    (0.85, False), (0.8999, False), (0.9, True), (0.95, True), (0.951, True),
    (1.0, True), (1.1, True), (1.1001, False), (1.15, False), (float("nan"), False),
])
def test_verdict_dso3(rough, attendu):
    from pcfno3d.config import DSOCfg
    from pcfno3d.dso import dso3_conforme

    assert dso3_conforme(rough, DSOCfg().roughness_range) is attendu


def test_rapport_affiche_les_deux_bornes(cfg, index, norm, capsys):
    """Le rapport d'evaluation juge sur [0,9 ; 1,1] et l'affiche, pas « <= 1,1 » seul."""
    from pcfno3d.dso import dso3_conforme
    from pcfno3d.evaluate import Evaluator
    from pcfno3d.train import Trainer

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    summary = Evaluator(cfg, device="cpu", verbose=True).run()
    texte = capsys.readouterr().out
    assert summary["dso3_conforme"] == dso3_conforme(summary["dso3_rugosite"],
                                                     cfg.dso.roughness_range)
    assert "DSO3" in texte and "[0.9 ; 1.1]" in texte


# ---------------------------------------------------------------------------- score
@pytest.mark.parametrize("rough, ecart", [
    (1.0, 0.0), (0.95, 0.5), (1.05, 0.5), (0.9, 1.0), (1.1, 1.0), (0.85, 1.5), (1.15, 1.5),
])
def test_ecart_dso3_symetrique_et_normalise(rough, ecart):
    from pcfno3d.dso import dso3_ecart

    assert dso3_ecart(rough, (0.9, 1.1)) == pytest.approx(ecart)


def test_score_de_selection_penalise_un_champ_trop_lisse():
    """0,85 : hors DSO3 -> rapport 1,5 > 1, et c'est lui qui limite le score."""
    from pcfno3d import dso
    from pcfno3d.config import Config

    cfg = Config.load("default")
    sc = dso.score(_metrics(0.85), cfg, GEOM)
    assert sc["ratios"]["dso3_rugosite"] == pytest.approx(1.5)
    assert sc["limitant"] == "dso3_rugosite"
    assert sc["score"] == pytest.approx(1.5)

    # a l'interieur de la plage, le terme reste <= 1 et ne limite plus
    sc_ok = dso.score(_metrics(0.95), cfg, GEOM)
    assert sc_ok["ratios"]["dso3_rugosite"] == pytest.approx(0.5)
    assert sc_ok["score"] < sc["score"]
    # symetrie : trop bruite d'autant est penalise d'autant
    sc_haut = dso.score(_metrics(1.15), cfg, GEOM)
    assert sc_haut["ratios"]["dso3_rugosite"] == pytest.approx(1.5)


def test_v7_reste_conforme_et_son_score_ne_change_pas():
    """v7 : rugosite 0,951, RMSE max 0,665 m/s. Le verdict DSO3 reste conforme, et le
    score DSO (pire rapport) reste celui du DSO1 : 0,665 / 0,5 = 1,33."""
    from pcfno3d import dso
    from pcfno3d.config import Config

    cfg = Config.load("default")
    assert dso.dso3_conforme(0.951, cfg.dso.roughness_range)
    sc = dso.score(_metrics(0.951, rmse=0.665), cfg, GEOM)
    assert sc["limitant"] == "dso1_rmse"
    assert sc["score"] == pytest.approx(0.665 / 0.5)
    assert sc["ratios"]["dso3_rugosite"] == pytest.approx(0.49)
