"""Les corrections v8 visant le DSO1, verifiees sur le mini-jeu de donnees.

On ne verifie PAS ici qu'elles font passer le DSO1 sur la v7 : cela demande les poids
v7 et un GPU, donc Colab. On verifie qu'elles font exactement ce qu'elles annoncent :
  - la moyenne de passes recalcule bien les caracteristiques par passe ;
  - les poids par niveau changent la loss la ou ils doivent, et seulement la ;
  - les poids sont derives de la VALIDATION et refusent le test ;
  - l'affinage repart du modele d'origine et ne peut pas le degrader ;
  - p_full produit bien des exemples avec tous les capteurs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch


def _trained(cfg, index, norm):
    from pcfno3d.train import Trainer

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg.paths.ckpt_best


# ---------------------------------------------------------------------------- config
def test_valeurs_par_defaut_reproduisent_la_v7():
    """Sans option, rien ne change : la v7 reste reproductible a l'identique."""
    from pcfno3d.config import Config

    cfg = Config.load("default")
    assert cfg.loss.uv_level_weight is None
    assert cfg.mask.p_full == 0.0
    assert cfg.inference.tta == 1
    assert cfg.train.init_from is None
    assert cfg.train.select_by == "val_score"


def test_options_invalides_refusees():
    from pcfno3d.config import Config

    for key, value in (("mask.p_full", 1.5), ("inference.tta", 0),
                       ("inference.tta_keep", 0.0), ("train.select_by", "loss"),
                       ("loss.uv_level_weight", [1.0, -1.0])):
        with pytest.raises((ValueError, TypeError)):
            Config.load("default", **{key.replace(".", "__"): value})


def test_poids_calcules_sur_le_test_refuses(tmp_path):
    """Regler l'entrainement sur le test, c'est mesurer sa propre copie."""
    from pcfno3d.config import Config

    triche = tmp_path / "poids_test.json"
    triche.write_text(json.dumps({"uv_level_weight": [1.0] * 12, "jeu": "test"}),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="TEST"):
        Config.load("default", loss__uv_level_weight=str(triche))


def test_configuration_v8_se_charge_et_garde_l_architecture_v7(tmp_path):
    """colab_v8.yaml ne sert que sur Colab : une faute de frappe ne se verrait que la-bas."""
    import yaml

    from pcfno3d.config import CONFIG_DIR, Config

    raw = yaml.safe_load((CONFIG_DIR / "colab_v8.yaml").read_text(encoding="utf-8"))
    poids = tmp_path / "uv.json"
    poids.write_text(json.dumps({"uv_level_weight": [1.0] * 12, "jeu": "validation"}),
                     encoding="utf-8")
    raw["loss"]["uv_level_weight"] = str(poids)       # chemins Colab -> fichier local
    copie = tmp_path / "v8.yaml"
    copie.write_text(yaml.safe_dump(raw), encoding="utf-8")

    v8, v7 = Config.load(copie), Config.load("colab")
    assert v8.model == v7.model                       # les poids v7 doivent s'y charger
    assert v8.paths.run_dir != v7.paths.run_dir        # la v7 n'est jamais ecrasee
    assert v8.train.select_by == "dso" and v8.train.init_from.endswith("best_dso_v7.pt")
    assert v8.loss.lambda_grad == v7.loss.lambda_grad  # pas le confondant du BLOC 9b
    assert v8.mask.p_full > 0 and v8.loss.uv_level_weight == [1.0] * 12

    # sans le fichier de poids, la configuration refuse de se charger : entrainer sans
    # eux donnerait silencieusement autre chose que la v8
    raw["loss"]["uv_level_weight"] = str(tmp_path / "absent.json")
    copie.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="dso-weights"):
        Config.load(copie)


# ---------------------------------------------------------------------------- loss
def test_poids_par_niveau_n_agissent_que_sur_u_v(cfg, geom, index, norm):
    """Un poids double sur un niveau double exactement son erreur u, v — pas celle de w."""
    from pcfno3d.losses import PhysicsLoss

    k = geom.eval_levels()[-1]
    y = torch.from_numpy(np.load(cfg.paths.cache_dir / f"{index['stem'].iloc[0]}_Y.npy"))[None]
    target = norm.normalize_target(y)
    pred = target.clone()
    pred[:, :, k][:, :, geom.valid_t[k]] += 0.5         # erreur uniquement au niveau k

    base = PhysicsLoss(cfg, geom, norm)
    l0, p0 = base(pred, y, phys_w=0.0)

    weights = [1.0] * geom.nz
    weights[k] = 2.0
    cfg.loss.uv_level_weight = weights
    try:
        heavy = PhysicsLoss(cfg, geom, norm)
        l1, p1 = heavy(pred, y, phys_w=0.0)
    finally:
        cfg.loss.uv_level_weight = None

    # erreur de w ponderee par w_level_w, inchangee ; u et v doubles
    n_valid = int(geom.valid_t.sum())
    nk = int(geom.valid_t[k].sum())
    w_part = float((0.5 ** 2) * nk * norm.w_level_w[k] * cfg.loss.w_weight)
    uv_part = float(2 * (0.5 ** 2) * nk)
    attendu_0 = (uv_part + w_part) / (3 * n_valid)
    attendu_1 = (2 * uv_part + w_part) / (3 * n_valid)
    assert p0["data"] == pytest.approx(attendu_0, rel=1e-4)
    assert p1["data"] == pytest.approx(attendu_1, rel=1e-4)


def test_mauvais_nombre_de_poids_refuse(cfg, geom, norm):
    from pcfno3d.losses import PhysicsLoss

    cfg.loss.uv_level_weight = [1.0] * (geom.nz + 1)
    try:
        with pytest.raises(ValueError, match="niveaux"):
            PhysicsLoss(cfg, geom, norm)
    finally:
        cfg.loss.uv_level_weight = None


# ---------------------------------------------------------------------------- poids DSO
def test_poids_derives_de_la_validation(cfg, geom, index, norm):
    from pcfno3d import dso
    from pcfno3d.dataset import make_loaders, make_splits
    from pcfno3d.inference import WindFieldModel

    _trained(cfg, index, norm)
    wf = WindFieldModel(cfg, device="cpu", verbose=False)
    loader = make_loaders(cfg, wf.geom, make_splits(cfg, index))["val"]
    metrics = dso.measure(wf.model, loader, wf.geom, wf.norm, wf.device)

    weights = dso.level_weights(metrics, cfg, wf.geom, floor=0.5, cap=4.0)
    assert len(weights) == geom.nz
    assert all(w > 0 for w in weights)
    # normalises : la moyenne ponderee par le nombre de noeuds vaut 1, l'echelle de L_data
    # ne change pas
    n = metrics.n
    assert float((np.array(weights) * n).sum() / n.sum()) == pytest.approx(1.0, abs=1e-3)
    # le niveau le pire recoit le poids le plus fort parmi les niveaux evalues
    rmse = metrics.rmse()
    lev = geom.eval_levels()
    pire = lev[int(np.nanargmax([np.nanmax(rmse[:2, k]) for k in lev]))]
    assert weights[pire] == max(weights[k] for k in lev)

    sc = dso.score(metrics, cfg, wf.geom)
    assert sc["score"] == max(sc["ratios"].values())
    assert sc["limitant"] in sc["ratios"]
    assert "niveau" in dso.describe_levels(metrics, cfg, wf.geom)


def test_commande_dso_weights_ecrit_un_fichier_utilisable(cfg, index, norm, tmp_path):
    from pcfno3d.cli import main
    from pcfno3d.config import Config

    _trained(cfg, index, norm)
    out = tmp_path / "uv.json"
    rc = main(["-c", "test", "--data-dir", str(cfg.paths.data_dir),
               "--cache-dir", str(cfg.paths.cache_dir), "--run-dir", str(cfg.paths.run_dir),
               "dso-weights", "--device", "cpu", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["jeu"] == "validation"
    assert data["modele"] == "best.pt"
    # le fichier se recharge directement comme option de configuration
    relu = Config.load("test", loss__uv_level_weight=str(out))
    assert relu.loss.uv_level_weight == data["uv_level_weight"]


# ---------------------------------------------------------------------------- TTA
def test_tta_recalcule_les_caracteristiques_par_passe(cfg, geom, index, norm, monkeypatch):
    """Chaque passe doit voir le vent geostrophique de SES capteurs, pas celui du total."""
    import pcfno3d.inference as inf

    _trained(cfg, index, norm)
    wf = inf.WindFieldModel(cfg, device="cpu", verbose=False)
    stem = index[index["split"] == "test"]["stem"].iloc[0]
    S, hour, doy, _ = wf.surface_from_cache(stem)

    vus = []
    original = inf.time_geo_features

    def espion(S_, sensor, h, d, g):
        vus.append(int(np.asarray(sensor).sum()))
        return original(S_, sensor, h, d, g)

    monkeypatch.setattr(inf, "time_geo_features", espion)
    wf.predict(S, hour, doy, tta=4, tta_keep=0.6)
    # 1 appel pour l'entree de base, puis 1 par passe supplementaire, chacune avec moins
    # de capteurs que la base
    assert len(vus) == 4
    base = vus[0]
    assert all(n < base for n in vus[1:]), vus


def test_tta_une_passe_egale_la_prediction_simple(cfg, index, norm):
    from pcfno3d.inference import WindFieldModel

    _trained(cfg, index, norm)
    wf = WindFieldModel(cfg, device="cpu", verbose=False)
    stem = index[index["split"] == "test"]["stem"].iloc[0]
    a = wf.predict_stem(stem)
    S, hour, doy, ts = wf.surface_from_cache(stem)
    b = wf.predict(S, hour, doy, timestamp=ts, tta=1)
    assert np.allclose(np.nan_to_num(a.u), np.nan_to_num(b.u))


def test_tta_moyenne_reellement_les_passes(cfg, index, norm):
    from pcfno3d.inference import WindFieldModel

    _trained(cfg, index, norm)
    wf = WindFieldModel(cfg, device="cpu", verbose=False)
    stem = index[index["split"] == "test"]["stem"].iloc[0]
    S, hour, doy, _ = wf.surface_from_cache(stem)
    simple = wf.predict(S, hour, doy, tta=1)
    moyenne = wf.predict(S, hour, doy, tta=4, tta_keep=0.5)
    assert not np.allclose(np.nan_to_num(simple.u), np.nan_to_num(moyenne.u))
    assert moyenne.inference_s > 0


def test_evaluation_avec_tta(cfg, index, norm):
    from pcfno3d.evaluate import Evaluator

    _trained(cfg, index, norm)
    ev = Evaluator(cfg, device="cpu", verbose=False, tta=3, tta_keep=0.7)
    try:
        summary = ev.run()
    finally:
        cfg.inference.tta, cfg.inference.tta_keep = 1, 0.7      # pas de fuite entre tests
    assert summary["tta"] == 3
    assert ev.out_dir.name.endswith("_tta3")                    # n'ecrase pas l'eval simple
    assert (ev.out_dir / "test_summary.json").exists()


def test_evaluation_n_ecrase_jamais_les_resultats_du_notebook(cfg, index, norm):
    """Le notebook v7 ecrit run_dir/eval_<modele>/ : c'est la reference de comparaison."""
    from pcfno3d.evaluate import Evaluator

    _trained(cfg, index, norm)
    reference = cfg.paths.run_dir / "eval_best"
    reference.mkdir(parents=True, exist_ok=True)
    original = {"origine": "notebook v7", "rmse_u_1000m": 0.665}
    (reference / "test_summary.json").write_text(json.dumps(original), encoding="utf-8")

    ev = Evaluator(cfg, device="cpu", verbose=False)
    ev.run()
    assert ev.out_dir != reference
    assert ev.out_dir.parent.name == "eval_pcfno3d"
    assert json.loads((reference / "test_summary.json").read_text(encoding="utf-8")) == original


# ---------------------------------------------------------------------------- p_full
def test_p_full_donne_des_exemples_avec_tous_les_capteurs(cfg, geom, index):
    from pcfno3d.dataset import ANEO3D

    train = index[index["split"] == "train"]
    cfg.mask.p_full = 1.0
    try:
        ds = ANEO3D(train["stem"].values, train["hour"].values, train["doy"].values,
                    geom, train=True, n_samples=6)
        for i in range(6):
            s_raw, sensor, _, _, j = ds[i]
            S = np.load(cfg.paths.cache_dir / f"{ds.stems[j]}_S.npy")
            assert np.array_equal(sensor.numpy() > 0.5, S[7] > 0.5)   # rien de masque
    finally:
        cfg.mask.p_full = 0.0


# ---------------------------------------------------------------------------- affinage
def test_affinage_part_du_modele_d_origine_et_ne_le_degrade_pas(cfg, index, norm, tmp_path):
    """L'echec du BLOC 9b ne peut pas se reproduire : l'original est le premier candidat."""
    from pcfno3d.config import Config
    from pcfno3d.train import Trainer

    origine = _trained(cfg, index, norm)
    fin = Config.load("test")
    fin.apply_overrides({
        "paths.data_dir": cfg.paths.data_dir, "paths.cache_dir": cfg.paths.cache_dir,
        "paths.run_dir": tmp_path / "v8", "train.init_from": str(origine),
        "train.select_by": "dso", "train.epochs": 2, "train.lr": 1e-4,
        "mask.p_full": 0.15,
    })
    fin.paths.mkdirs()
    import shutil
    shutil.copy2(cfg.paths.norm_stats, fin.paths.norm_stats)   # meme normalisation

    trainer = Trainer(fin, device="cpu", verbose=False)
    history = trainer.fit()

    assert history[0]["epoch"] == -1                        # le depart est enregistre
    depart = history[0]["val_score"]
    ck = torch.load(fin.paths.ckpt_best, map_location="cpu", weights_only=False)
    # le meilleur retenu n'est jamais pire que le modele d'origine
    assert ck["val_score"] <= depart + 1e-9
    assert all("dso" in h for h in history)


def test_affinage_refuse_une_architecture_differente(cfg, index, norm, tmp_path):
    from pcfno3d.config import Config
    from pcfno3d.train import Trainer

    origine = _trained(cfg, index, norm)
    autre = Config.load("test")
    autre.apply_overrides({
        "paths.data_dir": cfg.paths.data_dir, "paths.cache_dir": cfg.paths.cache_dir,
        "paths.run_dir": tmp_path / "x", "train.init_from": str(origine),
        "model.width": cfg.model.width + 4,
    })
    autre.paths.mkdirs()
    import shutil
    shutil.copy2(cfg.paths.norm_stats, autre.paths.norm_stats)
    with pytest.raises(ValueError, match="architecture differente"):
        Trainer(autre, device="cpu", verbose=False)


# ---------------------------------------------------------------------------- rapport
def test_niveau_quasi_vide_exclu_de_la_comparaison(cfg, geom, index, norm):
    """Le R2 de -11 899 du BLOC 12 venait d'un niveau a 0 % de couverture."""
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.reconstruct import compare_to_truth

    _trained(cfg, index, norm)
    wf = WindFieldModel(cfg, device="cpu", verbose=False)
    stem = index[index["split"] == "test"]["stem"].iloc[0]
    table = compare_to_truth(wf.predict_stem(stem),
                             np.load(cfg.paths.cache_dir / f"{stem}_Y.npy"))
    gardes = {int(geom.z_levels[k]) for k in geom.eval_levels()}
    assert set(table["z_m"]) <= gardes


def test_selection_v6_sans_r2_uvw(cfg, geom):
    """v6 : train.select_criteres retire r2_uvw du score de selection (le R2 de w par continuite
    aux niveaux hauts l'ecraserait) ; tous les rapports restent publies. Defaut : inchange."""
    import copy

    from pcfno3d import dso

    m = dso.LevelMetrics(geom.nz)
    rng = np.random.default_rng(0)
    for k in range(geom.nz):
        for c in range(3):
            t = rng.normal(5, 1, 200)
            p = t + rng.normal(0, 0.45 if c < 2 else 3.0, 200)      # w tres mauvais : R2 < 0
            m.n[c, k] += 200
            m.sse[c, k] += float(((p - t) ** 2).sum())
            m.s1[c, k] += float(t.sum())
            m.s2[c, k] += float((t ** 2).sum())
    m.mass.append(1e-5)
    m.rough.append(0.85)
    tout = dso.score(m, cfg, geom)
    assert tout["limitant"] == "r2_uvw"
    c = copy.deepcopy(cfg)
    c.train.select_criteres = ["dso1_rmse", "dso2_masse", "dso3_rugosite"]
    v6 = dso.score(m, c, geom)
    assert v6["limitant"] in ("dso1_rmse", "dso3_rugosite") and v6["score"] < tout["score"]
    assert "r2_uvw" in v6["ratios"] and v6["criteres"] == c.train.select_criteres
    with pytest.raises(ValueError):
        from pcfno3d.config import TrainCfg
        TrainCfg(select_criteres=["inconnu"])
