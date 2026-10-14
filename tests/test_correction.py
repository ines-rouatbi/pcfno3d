"""Lot M1 — correction post-modele du vent moyen par niveau (pcfno3d.correction).

Trois garanties :
  1. desactivee (defaut), la sortie est celle du modele seul, bit a bit ;
  2. activee, le decalage est exactement celui des coefficients, aux bons niveaux et
     composantes, sans toucher aux noeuds sous le relief ;
  3. le portage calcule LA MEME CHOSE que le notebook : on execute les fonctions des copies
     exactes des cellules (tests/reference/notebook_v7_correction/) sur les memes tableaux.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REF = Path(__file__).parent / "reference" / "notebook_v7_correction"
TOL = 1e-6


# ------------------------------------------------------------------ outillage
def _nb_functions(fichier: str, noms: tuple[str, ...], ns: dict) -> dict:
    """Execute UNIQUEMENT les definitions de fonctions demandees d'une cellule du notebook,
    dans l'espace `ns` (qui joue les variables globales du notebook)."""
    tree = ast.parse((REF / fichier).read_text(encoding="utf-8"))
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in noms]
    assert {d.name for d in defs} == set(noms), f"fonctions absentes de {fichier}"
    exec(compile(ast.Module(body=defs, type_ignores=[]), fichier, "exec"), ns)   # noqa: S102
    return ns


def _nb_globals(geom) -> dict:
    """Les variables globales du notebook dont ces cellules dependent, prises du paquet."""
    from pcfno3d.correction import FEATURES_10F
    return {"np": np, "math": math, "torch": torch, "time": time, "json": json,
            "XC": geom.xc, "YC": geom.yc, "Z_SENSOR": geom.z_sensor,
            "F_CORIOLIS": geom.cfg.f_coriolis, "NZ": geom.nz, "Z_LEVELS": geom.z_levels,
            "VALID": geom.valid, "VALID_T": torch.from_numpy(geom.valid),
            "EVAL_LEVELS": geom.eval_levels(), "FEATURES": list(FEATURES_10F),
            "RD_GAS": 287.05, "G0": 9.80665, "LAPSE_STD": 0.0065,
            "K_TOP": geom.nz - 1, "RIDGE_ALPHA": 1.0}


@pytest.fixture(scope="module")
def instant(cfg, geom, index):
    """Une surface reelle du mini-jeu (tous ses capteurs) et une prediction plausible."""
    from pcfno3d.cache import load_index
    from pcfno3d.features import time_geo_features

    row = load_index(cfg).iloc[0]
    S = np.load(cfg.paths.cache_dir / f"{row['stem']}_S.npy")
    Y = np.load(cfg.paths.cache_dir / f"{row['stem']}_Y.npy")
    sensor = S[7] > 0.5
    assert sensor.sum() >= 100, "le mini-jeu doit offrir >= 100 capteurs (seuil du BLOC 10g)"
    tfeat = time_geo_features(S, sensor, float(row["hour"]), int(row["doy"]), geom)
    rng = np.random.default_rng(0)
    u = (Y[0] + rng.normal(0, 0.4, Y[0].shape)).astype(np.float32)
    v = (Y[1] + rng.normal(0, 0.4, Y[1].shape)).astype(np.float32)
    u[~geom.valid], v[~geom.valid] = np.nan, np.nan            # comme un Volume (mask_nan)
    return S, sensor, tfeat, u, v


def _json_aleatoire(geom, seed=1, variante="1.2", utilise=True) -> dict:
    from pcfno3d.correction import FEATURES_10H
    rng = np.random.default_rng(seed)
    niveaux = {}
    for k in geom.eval_levels():
        for c in "uv":
            niveaux[f"{int(geom.z_levels[k])}_{c}"] = {
                "utilise": utilise, "mu": rng.normal(0, 1, 30).tolist(),
                "sd": (np.abs(rng.normal(1, 0.3, 30)) + 0.1).tolist(),
                "coef": rng.normal(0, 0.2, 31).tolist()}
    return {"variante_rho_sl": variante, "ridge_alpha": 0.01, "features": list(FEATURES_10H),
            "niveaux": niveaux}


# ------------------------------------------------------------------ references intactes
def test_copies_du_notebook_intactes():
    attendu = dict(reversed(line.split("  ")) for line in
                   (REF / "SHA256SUMS").read_text(encoding="utf-8").splitlines())
    for fichier, h in attendu.items():
        assert hashlib.sha256((REF / fichier).read_bytes()).hexdigest() == h, fichier


# ------------------------------------------------------------------ equivalence avec le notebook
def test_geostrophique_v2_identique_au_bloc_10g(geom, instant):
    from pcfno3d.correction import geo_row, geostrophic_v2, plane_residual_v1
    S, sensor, *_ = instant
    ns = _nb_functions("bloc_06_dataset.py", ("geostrophic_from_surface",), _nb_globals(geom))
    _nb_functions("bloc_10g.py", ("geostrophic_v2", "plane_residual_v1", "geo_row"), ns)
    np.testing.assert_allclose(geostrophic_v2(S, sensor, geom), ns["geostrophic_v2"](S, sensor),
                               rtol=TOL, atol=TOL)
    assert abs(plane_residual_v1(S, sensor, geom) - ns["plane_residual_v1"](S, sensor)) < TOL
    np.testing.assert_allclose(geo_row(S, sensor, geom), ns["geo_row"](S, sensor),
                               rtol=TOL, atol=TOL)
    peu = np.zeros_like(sensor)
    peu.flat[:50] = True                                     # < 100 capteurs : zeros, comme 10g
    assert not geostrophic_v2(S, peu, geom).any()


def test_caracteristiques_identiques_aux_blocs_10f_10h(geom, instant):
    from pcfno3d.correction import geo_row, instant_features, x2, x3
    S, sensor, tfeat, u, v = instant
    ns = _nb_functions("bloc_10f.py", ("instant_features",), _nb_globals(geom))
    _nb_functions("bloc_10g.py", ("x2",), ns)
    _nb_functions("bloc_10h.py", ("rotate", "x3"), ns)
    nb_X = ns["instant_features"](torch.from_numpy(S[None, :7]),
                                  torch.from_numpy(sensor[None].astype(np.float32)),
                                  torch.from_numpy(tfeat[None]), torch.from_numpy(u[None]),
                                  torch.from_numpy(v[None]))
    X = instant_features(S, sensor, tfeat, u, v, geom)
    np.testing.assert_allclose(X, nb_X, rtol=TOL, atol=TOL)

    rng = np.random.default_rng(2)                            # plusieurs instants
    Xn = rng.normal(0, 3, (7, geom.nz, 13))
    Gn = rng.normal(0, 5, (7, 10))
    for variante in ("1.2", "calc"):
        ns.update(VARIANT=variante, I_GEO=3 if variante == "1.2" else 5)
        for k in geom.eval_levels():
            np.testing.assert_array_equal(x2(Xn, Gn, k, variante), ns["x2"](Xn, Gn, k, variante))
            got = x3(Xn, Gn, k, variante)
            assert got.shape == (7, 30)
            np.testing.assert_array_equal(got, ns["x3"](Xn, Gn, k))
    G = geo_row(S, sensor, geom)[None]
    assert x3(X[None], G, geom.nz - 1, "1.2").shape == (1, 30)


def test_application_identique_au_bloc_12_etape_2b(geom, instant, tmp_path):
    from pcfno3d.correction import MeanWindCorrection
    S, sensor, tfeat, u, v = instant
    for variante in ("1.2", "calc"):
        data = _json_aleatoire(geom, seed=3, variante=variante)
        (tmp_path / "couche_limite_v7.json").write_text(json.dumps(data), encoding="utf-8")
        ns = {**_nb_globals(geom), "RUN_DIR": tmp_path, "S": S, "sensor": sensor,
              "tfeat": tfeat, "u": u.copy(), "v": v.copy(), "t_rec": 0.0, "print": lambda *a, **k: None}
        exec(compile((REF / "bloc_12_etape_2b.py").read_text(encoding="utf-8"),   # noqa: S102
                     "bloc_12_etape_2b.py", "exec"), ns)
        assert ns["corr_applied"]
        uc, vc, n = MeanWindCorrection(data).apply(u, v, S, sensor, tfeat, geom)
        assert n == 2 * len(geom.eval_levels())
        np.testing.assert_allclose(uc, ns["u"], rtol=TOL, atol=TOL, equal_nan=True)
        np.testing.assert_allclose(vc, ns["v"], rtol=TOL, atol=TOL, equal_nan=True)


def test_ridge_identique_au_bloc_10h(geom):
    from pcfno3d.correction import ridge_fit, ridge_pred
    ns = _nb_functions("bloc_10h.py", ("ridge_fit_a",), _nb_globals(geom))
    _nb_functions("bloc_10f.py", ("ridge_pred",), ns)
    rng = np.random.default_rng(4)
    X, t = rng.normal(0, 2, (60, 30)), rng.normal(0, 1, 60)
    for alpha in (0.01, 1.0, 100.0):
        a, b = ridge_fit(X, t, alpha), ns["ridge_fit_a"](X, t, alpha)
        for key in ("mu", "sd", "coef"):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ridge_pred(a, X), ns["ridge_pred"](b, X), rtol=1e-12)


# ------------------------------------------------------------------ comportement
def test_coefficients_connus_decalage_exact(geom, instant):
    """Cible constante c a un seul (niveau, composante) : la moyenne devient c, rien d'autre ne
    bouge, le decalage est uniforme et le relief garde ses NaN."""
    from pcfno3d.correction import FEATURES_10H, MeanWindCorrection
    S, sensor, tfeat, u, v = instant
    k = geom.eval_levels()[-1]
    cible = 7.25
    zero = {"mu": [0.0] * 30, "sd": [1.0] * 30}
    data = {"variante_rho_sl": "1.2", "ridge_alpha": 0.01, "features": list(FEATURES_10H),
            "niveaux": {f"{int(geom.z_levels[k])}_u": {"utilise": True, **zero,
                                                        "coef": [cible] + [0.0] * 30},
                        f"{int(geom.z_levels[k])}_v": {"utilise": False, **zero,
                                                        "coef": [99.0] + [0.0] * 30}}}
    uc, vc, n = MeanWindCorrection(data).apply(u, v, S, sensor, tfeat, geom)
    assert n == 1
    m = geom.valid[k]
    assert abs(float(uc[k][m].mean()) - cible) < 1e-5
    delta = uc[k][m] - u[k][m]
    np.testing.assert_allclose(delta, delta[0], atol=1e-5)             # decalage uniforme
    np.testing.assert_array_equal(np.isnan(uc), np.isnan(u))            # relief : NaN preserves
    assert np.isnan(uc[~geom.valid]).all()
    autres = [kk for kk in range(geom.nz) if kk != k]
    np.testing.assert_array_equal(uc[autres], u[autres])                # autres niveaux intacts
    np.testing.assert_array_equal(vc, v)                                # v non utilise : intact


def test_json_incompatible_refuse(geom):
    from pcfno3d.correction import MeanWindCorrection
    data = _json_aleatoire(geom)
    data["features"] = data["features"][:-1]
    with pytest.raises(ValueError, match="30 caracteristiques"):
        MeanWindCorrection(data)


def test_ajustement_retrouve_un_decalage_connu(geom):
    """Moyenne vraie = combinaison lineaire connue des 30 caracteristiques : la ridge choisie
    sur la validation la retrouve, et corrige le niveau."""
    from pcfno3d.correction import fit_correction, x3
    rng = np.random.default_rng(5)
    n_tr, n_va = 400, 120

    def tirage(n):
        X = rng.normal(0, 2, (n, geom.nz, 13))
        G = rng.normal(0, 4, (n, 10))
        return X, G

    w_vrai = rng.normal(0, 0.3, 30)
    X_tr, G_tr = tirage(n_tr)
    X_va, G_va = tirage(n_va)

    def cible(X, G):
        T = np.zeros((len(X), geom.nz, 2))
        for k in geom.eval_levels():
            for ci in range(2):
                T[:, k, ci] = 0.5 + ci + x3(X, G, k, "1.2") @ w_vrai
        return T

    T_tr, T_va = cible(X_tr, G_tr), cible(X_va, G_va)
    data, rapport = fit_correction(X_tr, G_tr, T_tr, X_va, G_va, T_va, geom)
    assert data["ridge_alpha"] == 0.01                       # le moins regularise gagne
    assert set(data["niveaux"]) == {f"{int(geom.z_levels[k])}_{c}"
                                    for k in geom.eval_levels() for c in "uv"}
    for k in geom.eval_levels():
        for ci, c in enumerate("uv"):
            par = data["niveaux"][f"{int(geom.z_levels[k])}_{c}"]
            assert par["utilise"]                            # bat la moyenne du modele
            sd = np.asarray(par["sd"])
            w_hat = np.asarray(par["coef"])[1:] / sd
            if data["variante_rho_sl"] == "1.2":
                np.testing.assert_allclose(w_hat, w_vrai, atol=2e-3)
    assert rapport["variante"] in ("1.2", "calc")


# ------------------------------------------------------------------ integration
@pytest.fixture(scope="module")
def trained(cfg, index, norm):
    from pcfno3d.train import Trainer
    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg


def test_desactivee_sortie_du_modele_seul_bit_a_bit(trained, geom, instant):
    """correction: null (defaut) -> exactement la sortie du modele, sans aucun recalage."""
    from pcfno3d.dataset import assemble_input
    from pcfno3d.inference import WindFieldModel
    assert trained.inference.correction is None
    wf = WindFieldModel(trained, device="cpu", verbose=False)
    assert wf.correction is None and wf.meta()["correction"] is None
    from pcfno3d.features import time_geo_features
    S, sensor, _, _, _ = instant
    vol = wf.predict(S, 12.0, 100, keep=1.0)
    assert vol.correction is None
    tfeat = time_geo_features(S, sensor, 12.0, 100, wf.geom)     # memes heure et jour
    with torch.no_grad():
        pred = wf.model(assemble_input(torch.from_numpy(S[None, :7]),
                                       torch.from_numpy(sensor[None].astype(np.float32)),
                                       torch.from_numpy(tfeat[None]), wf.geom, wf.norm))
    u = wf.norm.denorm(pred.float())[0][0].numpy()
    np.testing.assert_array_equal(vol.u, np.where(wf.geom.valid, u, np.nan))


def test_activee_sha256_dans_meta_exports_et_evaluation(trained, geom, instant, tmp_path):
    import copy

    from pcfno3d.evaluate import Evaluator
    from pcfno3d.export import export_web
    from pcfno3d.inference import WindFieldModel

    path = tmp_path / "couche_limite_v7.json"
    path.write_text(json.dumps(_json_aleatoire(geom, seed=6)), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg = copy.deepcopy(trained)
    cfg.apply_overrides({"inference.correction": path})

    wf = WindFieldModel(cfg, device="cpu", verbose=False)
    assert wf.meta()["correction"]["sha256"] == sha
    S, sensor, *_ = instant
    vol = wf.predict(S, 12.0, 100, keep=1.0)
    assert vol.correction["sha256"] == sha and vol.correction["appliquee_a"] > 0
    assert np.isnan(vol.u[~geom.valid]).all()
    manifest = json.loads((export_web(vol, tmp_path / "web") / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["correction"]["sha256"] == sha
    npz = np.load(vol.save_npz(tmp_path / "v.npz"))
    assert json.loads(str(npz["correction"]))["sha256"] == sha

    ev = Evaluator(cfg, device="cpu", verbose=False)
    assert ev.out_dir.name.endswith(f"_corr{sha[:8]}")


def test_script_de_reproduction_bout_en_bout(trained, geom):
    """Le script de la session GPU tourne sur le mini-jeu (CPU) : il ecrit dans
    reproduction_paquet/ seulement, ne touche pas au JSON fige, et retrouve un JSON fige
    produit par le meme ajustement (comparaison a 1e-6)."""
    import importlib.util
    import shutil

    spec = importlib.util.spec_from_file_location(
        "m1_repro", Path(__file__).resolve().parents[1] / "scripts" / "m1_reproduction_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    run = trained.paths.run_dir
    sets = [f"paths.data_dir={trained.paths.data_dir}", f"paths.cache_dir={trained.paths.cache_dir}",
            f"paths.run_dir={run}"]
    argv = ["-c", "test", "--sans-attendu"] + [a for s in sets for a in ("--set", s)]
    fige = run / "couche_limite_v7.json"
    repro_dir = run / "reproduction_paquet"
    try:
        assert mod.main(argv) == 2                           # sans JSON fige : refus explicite
        # « figer » le JSON qu'un premier passage du paquet produit, puis rejouer le script
        shutil.rmtree(repro_dir, ignore_errors=True)
        from pcfno3d.cache import load_index
        from pcfno3d.config import Config
        from pcfno3d.correction import fit_correction
        from pcfno3d.dataset import make_splits
        from pcfno3d.inference import WindFieldModel
        cfg = Config.load("test")
        cfg.apply_overrides({"paths.data_dir": trained.paths.data_dir,
                             "paths.cache_dir": trained.paths.cache_dir, "paths.run_dir": run})
        wf = WindFieldModel(cfg, device="cpu", verbose=False)
        splits = make_splits(cfg, load_index(cfg))
        tr = mod.collecte(wf, cfg, splits, *splits.train, mod.SEED_TR)
        vs = splits.val_selection
        va = mod.collecte(wf, cfg, splits, splits.val[0][vs], splits.val[1][vs],
                          splits.val[2][vs], mod.SEED_VA)
        data, _ = fit_correction(*tr, *va, wf.geom)
        fige.write_text(json.dumps(data, indent=2), encoding="utf-8")
        avant = hashlib.sha256(fige.read_bytes()).hexdigest()

        assert mod.main(argv) == 0
        assert hashlib.sha256(fige.read_bytes()).hexdigest() == avant      # fige intact
        assert {p.name for p in repro_dir.iterdir()} == {
            "couche_limite_v7_paquet.json", "rapport_ajustement.json", "comparaison_test.json"}
    finally:
        fige.unlink(missing_ok=True)
        shutil.rmtree(repro_dir, ignore_errors=True)
