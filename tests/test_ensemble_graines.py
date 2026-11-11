"""Moyenne des graines (pcfno3d.moyenne, scripts/ensemble_graines.py, regle PROPOSITION hachee, carnet) sur le
mini-jeu reel : trois petits entrainements de graines differentes. Aucun GPU, aucun acces reseau."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from test_forcage_era5 import LAT0, LON0, _jeu_reel, cfg_era5, forcage  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
GRAINES = (1, 2, 3)
PREFIXE = "mini_Cp"


def _script(nom: str):
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(nom, REPO / "scripts" / f"{nom}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def graines(cfg_era5, index, tmp_path_factory):  # noqa: F811
    """(config du mini-jeu reel, dossier des runs, [run_dir par graine]) : 3 entrainements, graines 1, 2, 3."""
    from pcfno3d.train import Trainer

    tmp = tmp_path_factory.mktemp("ensemble")
    c, _ = _jeu_reel(cfg_era5, index, tmp)
    c.apply_overrides({"inference.w_continuite": True})
    runs = tmp / "runs"
    dirs = []
    for g in GRAINES:
        cg = copy.deepcopy(c)
        cg.apply_overrides({"paths.run_dir": runs / f"{PREFIXE}_s{g}", "train.seed": g})
        cg.paths.mkdirs()
        Trainer(cg, device="cpu", verbose=False).fit()
        dirs.append(runs / f"{PREFIXE}_s{g}")
    return c, runs, dirs


def _membre(c, run):
    from pcfno3d.inference import WindFieldModel

    cg = copy.deepcopy(c)
    cg.apply_overrides({"paths.run_dir": run})
    return cg, WindFieldModel(cg, device="cpu", verbose=False)


def test_moyenne_des_sorties_et_w_par_continuite(graines):
    from pcfno3d.cache import load_index
    from pcfno3d.moyenne import ModeleMoyen

    c, _, dirs = graines
    mm = ModeleMoyen(c, dirs, noms=[f"s{g}" for g in GRAINES], device="cpu", verbose=False)
    stem = str(load_index(c)["stem"].iloc[0])
    S, hour, doy, stamp = mm.surface_from_cache(stem)
    vol = mm.predict(S, hour, doy, timestamp=stamp)
    assert vol.inference_s > 0
    seuls = [_membre(c, r)[1].predict(S, hour, doy, timestamp=stamp, mask_nan=False) for r in dirs]
    assert not np.allclose(seuls[0].u, seuls[1].u)                 # graines differentes, sorties differentes
    valid = mm.geom.valid
    u = np.mean([s.u for s in seuls], axis=0)
    v = np.mean([s.v for s in seuls], axis=0)
    np.testing.assert_allclose(vol.u[valid], u[valid], atol=1e-5)
    np.testing.assert_allclose(vol.v[valid], v[valid], atol=1e-5)
    assert np.isnan(vol.u[~valid]).all()
    # w par continuite des (u, v) moyens ; lineaire, il egale la moyenne des w par continuite des membres
    w = mm.geom.w_continuite(torch.from_numpy(u)[None], torch.from_numpy(v)[None])[0].numpy()
    np.testing.assert_allclose(vol.w[valid], w[valid], atol=1e-6)
    np.testing.assert_allclose(vol.w[valid], np.mean([s.w for s in seuls], axis=0)[valid], atol=1e-6)

    # benchmark DSO4 : champs du DSO4 existant, plus le temps de chaque graine seule
    b = mm.benchmark(S, hour, doy, n=2, warmup=1, timestamp=stamp)
    assert {"n", "mean_s", "p95_s", "max_s", "conforme", "membres", "seuil_s"} <= set(b)
    assert b["mean_s"] > 0 and b["membres"] == 3 and set(b["par_membre"]) == {"s1", "s2", "s3"}


def test_predicteurs_identiques_a_l_evaluation_et_dispersion(graines):
    from torch.utils.data import DataLoader

    from pcfno3d.dataset import ANEO3D
    from pcfno3d.evaluate import Evaluator
    from pcfno3d.moyenne import ModeleMoyen

    c, _, dirs = graines
    mm = ModeleMoyen(c, dirs, noms=["a", "b", "c"], device="cpu", verbose=False)
    ev = Evaluator(_membre(c, dirs[1])[0], device="cpu", verbose=False)
    s, h, d = ev.splits.val
    ds = ANEO3D(s[:1], h[:1], d[:1], mm.geom, train=False, seed=1000)
    s_raw, sensor, tfeat, y, j = next(iter(DataLoader(ds, batch_size=1)))
    p = mm.predicteurs()
    assert list(p) == ["moyenne", "a", "b", "c"]
    moy = p["moyenne"](s_raw, sensor, tfeat, y, int(j), str(s[0]))
    membres = [p[n](s_raw, sensor, tfeat, y, int(j), str(s[0])) for n in ("a", "b", "c")]
    # le membre b relu du cache = la passe servie de l'evaluation de ce seul entrainement
    ref = ev.predire(s_raw, sensor, tfeat, y, int(j), str(s[0]))
    for a, r in zip(membres[1], ref):
        torch.testing.assert_close(a, r)
    for a, k in zip(moy[:2], range(2)):
        torch.testing.assert_close(a, sum(m[k] for m in membres) / 3)
    sig = mm.dispersion(moy, membres)
    assert set(sig) == {"u", "v", "w", "speed"}
    assert all((x >= 0).all() and np.isfinite(x).all() for x in sig.values())
    assert sig["u"][mm.geom.valid].max() > 0
    # trois fois le meme entrainement : dispersion nulle, moyenne = le membre
    mm1 = ModeleMoyen(c, [dirs[0]] * 3, noms=["x", "y", "z"], device="cpu", verbose=False)
    sorties = mm1.passes(s_raw, sensor, tfeat)
    m1 = mm1.moyenne(sorties)
    torch.testing.assert_close(m1[0], sorties[0][0])
    assert all(np.allclose(x, 0, atol=1e-6) for x in mm1.dispersion(m1, sorties).values())
    with pytest.raises(ValueError, match="double"):
        ModeleMoyen(c, dirs[:2], noms=["a", "a"], device="cpu", verbose=False)


def _regle(dossier: Path, c, **param) -> Path:
    from pcfno3d.cache import load_index
    from pcfno3d.dataset import make_splits
    from pcfno3d.regles import hacher

    r = json.loads((REPO / "configs" / "national" / "regle_ensemble_cp.json").read_text(encoding="utf-8"))
    n_val = len(make_splits(c, load_index(c)).val_selection)
    r["parametres"].update({"prefixe": PREFIXE, "graines": list(GRAINES), "sortie": "mini_Cp_ens",
                            "n_val": n_val, "dso4_n": 2, "dso4_chauffe": 1, **param})
    p = dossier / "regle_ensemble_cp.json"
    p.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    hacher(p)
    return p


def test_script_ensemble_graines_de_bout_en_bout(graines, tmp_path, monkeypatch):
    from pcfno3d.config import Config
    from pcfno3d.evaluate import Evaluator
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.reliability import METHOD_GRAINES, Calibration

    c, runs, dirs = graines
    cfgf = c.save(tmp_path / "c.yaml")
    regle = _regle(tmp_path, c)
    M = _script("ensemble_graines")
    argv = ["ensemble_graines", "-c", str(cfgf), "--runs", str(runs), "--regle", str(regle),
            "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    assert M.main() in (0, 3)
    out = runs / "mini_Cp_ens"
    b = json.loads((out / "bilan_validation.json").read_text(encoding="utf-8"))
    assert b["jeu"] == "validation" and "non lu" in b["test"]
    assert set(b["verdicts_validation"]) == {"moyenne", "s1", "s2", "s3", "era5"}
    assert set(b["verdicts_validation_keep25"]) == {"moyenne", "s1", "s2", "s3", "era5"}
    assert set(b["criteres"]) == set(M.CRITERES)
    assert b["moyenne_proposee_pour_le_test"] == all(x["rempli"] for x in b["criteres"].values())
    choix = json.loads((out / "membres.json").read_text(encoding="utf-8"))["meilleure_graine"]
    assert b["meilleure_graine"] == f"s{choix['graine']}"
    # DSO4 : temps de la passe servie de la moyenne, instant de validation
    d4 = json.loads((out / "dso4_moyenne.json").read_text(encoding="utf-8"))
    assert d4["jeu"] == "validation" and d4["mean_s"] > 0 and d4["n"] == 2 and "par_membre" in d4
    # DSO6 : dispersion >= 0, calibration de la VALIDATION, autre methode que la fiabilite servie
    d6 = json.loads((out / "dso6_graines_validation.json").read_text(encoding="utf-8"))
    assert d6["jeu"] == "validation" and "pas un verdict" in d6["nature"]
    assert all(x >= 0 for x in d6["sigma_brut_min_m_s"].values())
    assert all(x > 0 for x in d6["sigma_brut_moyen_m_s"].values())
    assert "dso6_conforme" not in d6 and {"couverture80_u", "spearman_u"} <= set(d6["global"])
    pc = out / "calibration_graines_validation.json"
    cal = Calibration.load(pc)
    assert cal.method == METHOD_GRAINES and cal.data["members"] == 3 and cal.data["jeu"] == "validation"
    assert d6["calibration_sha256"] == cal.sha256
    cr = copy.deepcopy(c)
    cr.apply_overrides({"paths.run_dir": dirs[0], "inference.reliability": str(pc)})
    with pytest.raises(ValueError, match="ensemble-graines"):
        WindFieldModel(cr, device="cpu", verbose=False)     # jamais pour la fiabilite servie

    # memes chiffres que dso_hauteur_sol.mesurer pour cette graine seule (pas de code de metrique duplique)
    import pandas as pd
    H = _script("dso_hauteur_sol")
    ev = Evaluator(_membre(c, dirs[0])[0], device="cpu", verbose=False)
    ref, _ = H.mesurer(ev, Config.load(str(cfgf)), None, None, jeu="validation")
    tab = pd.read_csv(out / "tranches_validation.csv")
    for _, r in ref[ref.predicteur == "modele"].iterrows():
        if r.noeuds_par_instant == 0:
            continue
        t = tab[(tab.predicteur == "s1") & (tab.tranche == r.tranche)].iloc[0]
        assert abs(t.rmse_u - r.rmse_u) < 1e-5 and abs(t.rmse_v - r.rmse_v) < 1e-5
    e_ref = ref[ref.predicteur == "era5"].set_index("tranche")["rmse_u"]
    e = tab[tab.predicteur == "era5"].set_index("tranche")["rmse_u"]
    pd.testing.assert_series_equal(e.dropna(), e_ref.dropna(), check_names=False)

    # jamais d'ecrasement ; une regle modifiee est refusee
    with pytest.raises(SystemExit, match="existe deja"):
        M.main()
    autre = tmp_path / "x"
    autre.mkdir()
    r2 = _regle(autre, c, tol_rugosite=0.05)
    r2.write_text(r2.read_text(encoding="utf-8").replace("0.05", "0.06"), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [a if a != str(regle) else str(r2) for a in argv])
    with pytest.raises(SystemExit, match="modifiee"):
        M.main()


def test_regle_ensemble_hachee_et_proposition():
    from pcfno3d.regles import verifier

    regle, sha = verifier(REPO / "configs" / "national" / "regle_ensemble_cp.json")
    assert "PROPOSITION" in regle["statut"] and "accord ecrit d'Ines" in regle["statut"]
    P = regle["parametres"]
    assert P["graines"] == [42, 43, 44] and P["prefixe"] == "national_1km_v4_Cp" and P["n_val"] == 64
    assert P["sortie"] == "national_1km_v4_Cp_ens"
    assert "NON lu" in regle["test"] and "accord ecrit SEPARE" in regle["test"]
    # la regle v4 n'est pas modifiee
    assert verifier(REPO / "configs" / "national" / "regle_v4_national.json")[1].startswith("60671e84")


def test_carnet_ensemble_cp():
    """Carnet : accord d'Ines en tete, commit fige, empreintes verifiees, A100, aucun entrainement ni test."""
    M = _script("make_ensemble_nb")
    sha = (REPO / "configs" / "national" / "regle_ensemble_cp.json.sha256").read_text(encoding="ascii").split()[0]
    nb = M.ensemble_cp("0123456789abcdef0123456789abcdef01234567")
    assert nb["metadata"]["colab"]["gpuType"] == "A100"
    texte = "\n".join("".join(cl["source"]) for cl in nb["cells"])
    tete = "".join(nb["cells"][0]["source"])
    assert "accord écrit d'Ines" in tete and "aucune lecture du test" in tete and "PROPOSITION" in tete
    assert "0123456789abcdef0123456789abcdef01234567" in texte and sha in texte and "60671e84" in texte
    assert "ensemble_graines.py" in texte and "choisir_poids" in texte and "runtime.unassign()" in texte
    for interdit in (" train\"", " train ", "evaluate", "dso_hauteur_sol.py", "--jeu test", " dso6 ",
                     "verdict-national"):
        assert interdit not in texte, interdit
    f = REPO / "notebooks" / "current" / M.NOM
    assert f.exists() and sha in f.read_text(encoding="utf-8")
