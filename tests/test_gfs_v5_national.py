"""Regle configs/national/regle_national_gfs_v5_2.json (remplace la v5.1, arretee sans verdict) : empreinte, constantes recopiees, ancrage GFS (C2), tirage de
l'echeance a l'entrainement (C1), critere identique a gfs_v4, choix des variantes (ex aequo, arret), controle de
l'entree de production, jugement 2026 (refus si deja lu) et lecture des verdicts par le service. CPU, mini-jeu
synthetique, aucun acces reseau, aucune donnee reelle."""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_forcage_era5 import HAUTEURS, cfg_era5, forcage  # noqa: F401  (fixtures)
from test_os7_prevision import jeu_gfs  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]


def _charger(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, REPO / "scripts" / fichier)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _charger("choisir_gfs_v5_t", "choisir_national_gfs_v5.py")
J = _charger("juger_gfs_v5_t", "juger_national_gfs_v5.py")
J4 = _charger("juger_gfs_v4_t", "juger_national_gfs_v4.py")
REGLE = json.loads(C.REGLE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- empreinte et constantes
def test_regle_hachee_et_constantes_recopiees():
    sha = C.verifier_regle()
    assert sha == "542e676c90da2a6440ef3e64f323f8ebc3cf41f9926777934db54f5827583cb0"
    assert J.REGLE == C.REGLE
    assert list(C.CASES) == REGLE["critere"]["cases_jugees"]
    assert "2000 tirages, graine 2030" in REGLE["metrique"] and (C.TIRAGES, C.GRAINE) == (2000, 2030)
    assert "plus de 5 %" in REGLE["critere"]["par_case"] and C.DEGRADATION_MAX == 0.05
    ctl = REGLE["critere"]["controle_d_echeances"]
    assert "au plus 2 % d'echeances absentes" in ctl and C.ABSENTS_MAX == 0.02
    assert "retire pour TOUTES les sources" in ctl and "pas d'arret pour ces instants retires" in ctl
    assert "0,005 m/s" in REGLE["etape_3_choix_sur_validation_2021"]["retenu"] and C.EX_AEQUO == 0.005
    assert "(C2, puis C1, puis C3)" in REGLE["etape_3_choix_sur_validation_2021"]["retenu"]
    assert C.ORDRE_SIMPLICITE == ("c2", "c1", "c3")
    assert "1 + 6 + 6 = 13" in REGLE["etape_3_choix_sur_validation_2021"]["calcul"] and len(C.variantes()) == 13
    jug = REGLE["periodes"]["jugement"]
    assert "du 2026-01-01 au 2026-10-31" in jug and "jours de l'annee impairs" in jug
    assert "00, 06, 12 et 18 UTC" in jug and "~608 instants" in jug and "toujours l'echeance f000" in jug
    assert (J.DEBUT_2026, J.FIN_2026, J.HEURES_2026) == ("2026-01-01T00:00", "2026-10-31T18:00", (0, 6, 12, 18))
    ctrl = REGLE["etape_4_jugement_2026"]["controle_prealable"]
    assert "ecart max <= 1e-5" in ctrl and J.TOLERANCE == 1e-5
    assert "vecteur de forcage (scalaires et cellules), reseau GFS 10 m (pseudo-capteurs), carte et valeurs des " \
           "stations" in ctrl and "sans arret" in ctrl
    assert set(J.VENT) == {"capteurs_differents", "forcage_scalaires", "forcage_cellules", "reseau_gfs_10m",
                           "stations_valeurs"}
    assert "non calcule faute de budget" in REGLE["etape_4_jugement_2026"]["sources"]
    assert J.NON_CALCULE == "non calcule faute de budget"
    assert "PLAFOND 90 unites" in REGLE["budget"]
    # la v5 (jamais executee) reste hachee et intacte, mais n'est plus la regle executee
    v5 = C.REGLE.with_name("regle_national_gfs_v5.json")
    assert C.verifier_regle(v5) == "519ebd64ba700df855f60ba26c625923f7da8677f4949f3dbda1207cf6688d73"
    # la v5.1 (arretee sans verdict le 23/11) reste hachee et intacte, elle non plus n'est plus executee
    v51 = C.REGLE.with_name("regle_national_gfs_v5_1.json")
    assert C.verifier_regle(v51) == "570713841039a5fda8e4d8727a047a4c6a3171a9c212b5402cdb2b7356be9cdd"
    assert C.REGLE.name == "regle_national_gfs_v5_2.json"
    # seul le controle d'echeances change : tout le reste est recopie de la v5.1 a l'identique
    r51 = json.loads(v51.read_text(encoding="utf-8"))
    for k in ("objet", "periodes", "etape_1_diagnostic_validation_2021", "metrique", "consequences", "interdits"):
        assert REGLE[k] == r51[k], k
    assert REGLE["critere"]["par_case"] == r51["critere"]["par_case"]
    assert REGLE["critere"]["cases_jugees"] == r51["critere"]["cases_jugees"]
    e2, e2_51 = dict(REGLE["etape_2_candidats"]), r51["etape_2_candidats"]
    c1 = dict(e2.pop("C1_v5_gfs"))
    assert "REUTILISE tel quel, PAS reentraine" in c1.pop("deja_fait")
    c1["nom"] = e2_51["C1_v5_gfs"]["nom"]
    assert c1 == e2_51["C1_v5_gfs"] and e2 == {k: v for k, v in e2_51.items() if k != "C1_v5_gfs"}
    e3 = {k: v for k, v in REGLE["etape_3_choix_sur_validation_2021"].items() if k not in ("calcul", "sortie", "gpu")}
    assert e3 == {k: v for k, v in r51["etape_3_choix_sur_validation_2021"].items() if k != "calcul"}
    assert REGLE["etape_4_jugement_2026"] == r51["etape_4_jugement_2026"]
    # le critere est recopie de gfs_v4 : memes constantes que son script
    assert (C.TIRAGES, C.GRAINE, C.DEGRADATION_MAX, C.ABSENTS_MAX) == \
        (J4.TIRAGES, J4.GRAINE, J4.DEGRADATION_MAX, J4.ABSENTS_MAX)


def test_regle_modifiee_refusee(tmp_path):
    f = tmp_path / C.REGLE.name
    shutil.copy2(C.REGLE, f)
    shutil.copy2(C.REGLE.with_name(C.REGLE.name + ".sha256"), f.with_name(f.name + ".sha256"))
    assert C.verifier_regle(f) == C.verifier_regle()
    f.write_bytes(f.read_bytes().replace(b"0,005", b"0,010"))
    with pytest.raises(SystemExit):
        C.verifier_regle(f)


def test_grille_c2_et_configuration_c1_recopiees():
    from pcfno3d.ancrage_gfs import GRILLE_A0, GRILLE_H
    from pcfno3d.config import Config

    c2 = REGLE["etape_2_candidats"]["C2_ancrage_gfs"]["grille"]
    assert "a0 dans {0,25 ; 0,5 ; 0,75}, H dans {100 m ; 300 m}" in c2
    assert GRILLE_A0 == (0.25, 0.5, 0.75) and GRILLE_H == (100.0, 300.0)
    c1 = REGLE["etape_2_candidats"]["C1_v5_gfs"]
    assert "12 epoques, lr 5e-5" in c1["entrainement"] and "runs/national_1km_v5_gfs" in c1["nom"]
    cfg = Config.load("national_v5_gfs_mixte_colab")
    assert cfg.name == "national_1km_v5_gfs" and cfg.paths.run_dir.name == "national_1km_v5_gfs"
    assert (cfg.train.epochs, cfg.train.lr, cfg.train.select_by) == (12, 5e-5, "dso")
    assert cfg.train.select_criteres == ["dso1_rmse", "dso2_masse", "dso3_rugosite"]
    assert "national_1km_v4_Cp_s44" in str(cfg.train.init_from)
    # reglages de perte de la v4 (national_1km_colab + lambda_rough 1,0, celui de C')
    v4 = Config.load("national_1km_colab")
    for k in ("lambda_mass", "mass_ref", "lambda_bc", "lambda_smooth", "lambda_grad", "w_weight", "w_level_max"):
        assert getattr(cfg.loss, k) == getattr(v4.loss, k), k
    assert cfg.loss.lambda_rough == 1.0
    assert sorted(cfg.paths.forcages_echeances) == [0, 24, 48]
    assert cfg.mask.p_station == v4.mask.p_station == 0.8 and cfg.mask.stations_entree
    # les configurations existantes ne tirent pas d'echeance (comportement inchange)
    for nom in ("national_1km_colab", "national_v5_gfs_colab", "grand_tunis_v5_gfs_colab", "test"):
        assert Config.load(nom).paths.forcages_echeances is None, nom


# ---------------------------------------------------------------------- C2 : ancrage GFS
def test_coefficient_a_h():
    from pcfno3d.ancrage_gfs import coefficient

    a = coefficient(np.array([0.0, 5.0, 10.0, 55.0, 100.0, 150.0, 1000.0]), 0.25, 100.0)
    np.testing.assert_allclose(a, [0.25, 0.25, 0.25, 0.625, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(coefficient(155.0, 0.5, 300.0), 0.5 + 0.5 * 145 / 290)
    for a0, H in ((1.2, 100.0), (0.5, 10.0), (-0.1, 300.0)):
        with pytest.raises(ValueError):
            coefficient(10.0, a0, H)


def test_ancrer_identite_au_dessus_de_H_et_a0_1():
    from pcfno3d.ancrage_gfs import ancrer

    h = np.array([10.0, 50.0, 100.0, 150.0, 300.0, 800.0])
    um, vm = np.array([5.0, 6, 7, 8, 9, 10]), np.array([-1.0, -2, -3, -4, -5, -6])
    ug, vg = um + 2.0, vm - 1.0
    u, v = ancrer(um, vm, ug, vg, h, 0.25, 100.0)
    np.testing.assert_allclose(u[2:], um[2:]) and np.testing.assert_allclose(v[2:], vm[2:])   # h >= H : modele
    np.testing.assert_allclose(u[0], ug[0] + 0.25 * (um[0] - ug[0]))                         # 10 m : a0
    u1, v1 = ancrer(um, vm, ug, vg, h, 1.0, 300.0)
    np.testing.assert_allclose(u1, um) and np.testing.assert_allclose(v1, vm)                 # a0 = 1 : modele
    u0, _ = ancrer(um, vm, ug, vg, h, 0.0, 300.0)
    np.testing.assert_allclose(u0[0], ug[0])                                                   # a0 = 0 a 10 m : GFS
    un, _ = ancrer(np.array([np.nan, 1.0]), np.array([0.0, 0.0]), np.zeros(2), np.zeros(2), np.array([10.0, 10.0]),
                   0.5, 100.0)
    assert np.isnan(un[0]) and un[1] == 0.5


def test_ancrer_volume_w_par_continuite(cfg_era5, index):  # noqa: F811
    import torch

    from pcfno3d.ancrage_gfs import ancrer_volume, colonne_gfs, gfs_aux_noeuds, hauteurs_noeuds
    from pcfno3d.forcage import horodatage
    from pcfno3d.geometry import Geometry

    g = Geometry(copy.deepcopy(cfg_era5), device="cpu")
    vec = g.forcage.vecteur(horodatage(g, index["stem"].iloc[0]))
    dense = gfs_aux_noeuds(g, vec)
    for iy, ix in ((0, 0), (g.ny // 2, g.nx // 3), (g.ny - 1, g.nx - 1)):
        np.testing.assert_allclose(colonne_gfs(g, vec, iy, ix), dense[:, :, iy, ix], atol=1e-5)
    rng = np.random.default_rng(0)
    u0 = dense[0] + rng.normal(0, 1, dense[0].shape)
    v0 = dense[1] + rng.normal(0, 1, dense[1].shape)
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32)[None])          # noqa: E731
    for beta in (0.0, 0.75):
        g.cfg.inference.w_continuite, g.cfg.inference.w_beta = True, beta
        w_res = rng.normal(0, 0.1, u0.shape)
        w0 = g.w_servi(t(u0), t(v0), t(w_res))[0].numpy()                  # w servi du modele
        u1, v1, w1 = ancrer_volume(g, u0, v0, w0, vec, 0.5, 100.0)
        attendu = g.w_servi(t(u1), t(v1), t(w_res))[0].numpy()              # w servi du champ ancre
        np.testing.assert_allclose(w1, attendu, atol=1e-3, rtol=1e-4)
        haut = hauteurs_noeuds(g) >= 100.0
        np.testing.assert_allclose(u1[haut], u0[haut]) and np.testing.assert_allclose(v1[haut], v0[haut])
    g.cfg.inference.w_continuite = False
    w0 = rng.normal(0, 0.1, u0.shape)
    assert np.array_equal(ancrer_volume(g, u0, v0, w0, vec, 0.5, 100.0)[2], w0)               # w du reseau garde


# ---------------------------------------------------------------------- C1 : echeance tiree par exemple
def _forcages_decales(src_csv: Path, d: Path, decalages=(0.0, 10.0, 20.0)) -> dict:
    """Trois forcages de memes mailles, u decale de 0 / 10 / 20 m/s : l'echeance d'un exemple se lit dans ses
    cellules."""
    out = {}
    for e, dec in zip((0, 24, 48), decalages):
        csv = d / f"forcage_gfs_f{e}.csv"
        shutil.copy2(src_csv, csv)
        with np.load(Path(src_csv).with_suffix(".npz")) as z:
            arr = {k: z[k] for k in z.files}
        arr["u"] = (arr["u"] + dec).astype(np.float32)
        np.savez(csv.with_suffix(".npz"), **arr)
        out[e] = csv
    return out


@pytest.fixture(scope="module")
def jeu_echeances(jeu_gfs, tmp_path_factory):  # noqa: F811
    c, p, _, _ = jeu_gfs
    d = tmp_path_factory.mktemp("echeances")
    f = _forcages_decales(Path(p.paths.forcage_era5), d)
    m = copy.deepcopy(c)
    m.apply_overrides({"paths.forcage_era5": f[0], "mask.e_source": "forcage",
                       "paths.forcages_echeances": f, "paths.run_dir": d / "run", "paths.local_dir": None})
    return c, m, f


def _echeance_des_cellules(geom, tfeat) -> int:
    from pcfno3d.forcage import N_SCAL
    from pcfno3d.geometry import N_TFEAT
    n = geom.forcage.n_cellules
    cel = np.asarray(tfeat)[N_TFEAT + N_SCAL:N_TFEAT + N_SCAL + n].reshape(geom.forcage.forme_cellules)
    ref = geom.forcages_echeances[0].u.mean()
    return {0: 0, 1: 24, 2: 48}[int(round((cel[0].mean() - ref) / 10.0))]


def test_tirage_des_echeances(jeu_echeances):
    from pcfno3d.dataset import echeance_exemple
    from pcfno3d.geometry import Geometry

    c, m, _ = jeu_echeances
    g = Geometry(m)
    assert sorted(g.forcages_echeances) == [0, 24, 48]
    rng = np.random.default_rng(1)
    tirees = [echeance_exemple(g, rng, 0)[1] for _ in range(600)]
    n = {e: tirees.count(e) for e in (0, 24, 48)}
    assert all(150 < x < 250 for x in n.values()), n                     # uniforme
    assert [echeance_exemple(g, None, i)[1] for i in range(6)] == [0, 24, 48, 0, 24, 48]      # evaluation : fixe
    fe = {e: echeance_exemple(g, None, i)[0] for i, e in enumerate((0, 24, 48))}
    assert all(fe[e] is g.forcages_echeances[e] for e in fe)
    g0 = Geometry(c)                                                     # sans l'option : rien ne change
    assert g0.forcages_echeances is None and echeance_exemple(g0, rng, 3) == (g0.forcage, None)


def test_stations_seulement_a_f0_et_forcage_de_l_echeance(jeu_echeances, monkeypatch):
    import pcfno3d.dataset as D
    from pcfno3d.cache import load_index
    from pcfno3d.dataset import ANEO3D
    from pcfno3d.geometry import Geometry

    _, m, _ = jeu_echeances
    g = Geometry(m)
    idx = load_index(m)
    tr = idx[idx["split"] == "train"]
    vus = []
    vrai = D.entree_era5_stations

    def espion(*a, **k):
        vus.append((k.get("forcage"), k.get("stations")))
        return vrai(*a, **k)

    monkeypatch.setattr(D, "entree_era5_stations", espion)
    ds = ANEO3D(tr["stem"].values, tr["hour"].values, tr["doy"].values, g, train=True, n_samples=60)
    for i in range(60):
        _, _, tfeat, _, _ = ds[i]
        e = _echeance_des_cellules(g, tfeat)
        fo, stations = vus[-1]
        assert fo is g.forcages_echeances[e]                       # cellules ET reseau a 10 m de l'echeance
        assert stations is (e == 0)                                     # aeroports seulement a f0
    assert {_echeance_des_cellules(g, ds[i][2]) for i in range(40)} == {0, 24, 48}
    # evaluation : echeance i modulo 3, stations (toutes) seulement a f0, hors reseau aucun capteur aux echeances
    ev = ANEO3D(tr["stem"].values[:6], tr["hour"].values[:6], tr["doy"].values[:6], g, train=False, keep_eval=1.0)
    from pcfno3d.dataset import cellules_stations
    k = max(int(round(m.mask.pas_era5_m / g.dx)), 1)
    hors = [(iy, ix) for iy, ix in cellules_stations(g) if (iy - k // 2) % k or (ix - k // 2) % k]
    for i in range(6):
        _, sensor, tfeat, _, _ = ev[i]
        e = _echeance_des_cellules(g, tfeat)
        assert e == (0, 24, 48)[i % 3]
        for iy, ix in hors:
            assert bool(sensor[iy, ix]) == (e == 0)


def test_index_sans_l_une_des_echeances_ecarte(jeu_echeances, tmp_path):
    from pcfno3d.cache import load_index

    c, m, f = jeu_echeances
    trou = tmp_path / "forcage_gfs_f48.csv"
    df = pd.read_csv(f[48])
    t = pd.to_datetime(df["timestamp_utc"])
    df[(t < t.iloc[len(t) // 2] - pd.Timedelta(hours=12)) | (t > t.iloc[len(t) // 2] + pd.Timedelta(hours=12))] \
        .to_csv(trou, index=False)
    shutil.copy2(f[48].with_suffix(".npz"), trou.with_suffix(".npz"))
    m2 = copy.deepcopy(m)
    m2.paths.forcages_echeances = {**f, 48: trou}
    assert len(load_index(m2)) < len(load_index(m)) == len(load_index(c))


def test_forcage_d_une_autre_grille_refuse(jeu_echeances, tmp_path):
    from pcfno3d.geometry import Geometry

    _, m, f = jeu_echeances
    autre = tmp_path / "forcage_gfs_f24.csv"
    shutil.copy2(f[24], autre)
    with np.load(f[24].with_suffix(".npz")) as z:
        arr = {k: z[k] for k in z.files}
    arr["lat"] = arr["lat"] + 0.05
    np.savez(autre.with_suffix(".npz"), **arr)
    m2 = copy.deepcopy(m)
    m2.paths.forcages_echeances = {**f, 24: autre}
    with pytest.raises(ValueError, match="mailles"):
        Geometry(m2)


def test_affinage_c1_tourne(jeu_echeances):
    from pcfno3d.train import Trainer

    c, m, _ = jeu_echeances
    m2 = copy.deepcopy(m)
    m2.apply_overrides({"train.epochs": 1, "train.init_from": str(c.paths.run_dir / "best.pt"),
                        "train.select_by": "dso"})
    Path(m2.paths.run_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(c.paths.norm_stats, Path(m2.paths.run_dir) / "norm_stats.json")      # normalisation de C'
    h = Trainer(m2, device="cpu", verbose=False).fit()
    assert [r["epoch"] for r in h] == [-1, 0]                           # depart C' (premier candidat), puis 1 epoque
    assert (Path(m2.paths.run_dir) / "best.pt").exists()


# ---------------------------------------------------------------------- critere : identique a gfs_v4
def _paires(gain: float, degrade: str | None = None, n_jours: int = 20, graine: int = 0, nom="a_national_gfs"):
    rng = np.random.default_rng(graine)
    lignes = []
    for e in C.ECHEANCES:
        for d in range(n_jours):
            for h in range(8):
                for a in range(12):
                    uo, vo = rng.normal(0, 4, 2)
                    eg = rng.normal(0, 2.5, 2)
                    f = 1.3 if degrade == f"A{a:02d}" else (1 - gain)
                    lignes.append({"station": f"A{a:02d}", "jour": f"2021-01-{d + 1:02d}", "echeance_h": e,
                                   "u_obs": uo, "v_obs": vo, "u_gfs_brut": uo + eg[0], "v_gfs_brut": vo + eg[1],
                                   f"u_{nom}": uo + f * eg[0], f"v_{nom}": vo + f * eg[1],
                                   "u_b_gfs_oi_10m": uo + f * eg[0], "v_b_gfs_oi_10m": vo + f * eg[1]})
    return pd.DataFrame(lignes)


@pytest.mark.parametrize("gain,degrade", [(0.10, None), (0.0, None), (0.15, "A03"), (-0.05, None)])
def test_critere_identique_a_gfs_v4(gain, degrade):
    p = _paires(gain, degrade)
    st = sorted(p["station"].unique())
    v4 = J4.juger(p, st, "x" * 64)["details"]
    v5 = C.juger_variantes(p, ["a_national_gfs"], st)["a_national_gfs"]["details"]
    for case5, case4 in zip(C.CASES, ("a / analyse", "a / +24 h", "a / +48 h")):
        a, b = v5[case5], v4[case4]
        assert a["verdict"] == b["verdict"]
        assert a["ecart"] == b["ecart"] and a["ic95"] == b["ic95"]
        assert a["aeroports_degrades_plus_5_pct"] == b["aeroports_degrades_plus_5_pct"]
        assert a["par_aeroport"] == b["par_aeroport"]
    if degrade:
        assert all(v5[c]["verdict"] == "refuse" for c in C.CASES)
    j = _charger("juger_gfs_v5_t2", "juger_national_gfs_v5.py")
    pp = p.rename(columns={"u_a_national_gfs": "u_candidat", "v_a_national_gfs": "v_candidat"}).assign(
        u_cp_s44_gfs=p["u_gfs_brut"], v_cp_s44_gfs=p["v_gfs_brut"])
    vj = j.juger_paires(pp, st, "y" * 64, {"variante": {"nom": "c1", "famille": "c1"}})
    assert vj["cases"] == {c5: v4[c4]["verdict"] for c5, c4 in zip(C.CASES, ("a / analyse", "a / +24 h", "a / +48 h"))}
    assert all(abs(d["ecart"]) < 1e-12 for d in vj["rapporte_sans_verdict_cp_s44_gfs"].values())


# ---------------------------------------------------------------------- choix des variantes
def _res(**moyennes):
    """{nom: (passe, rmse moyen)} -> resultats au format de juger_variantes."""
    return {n: {"passe_les_trois": ok, "rmse_moyen_trois_cases": r} for n, (ok, r) in moyennes.items()}


FAM = {v["nom"]: v["famille"] for v in C.variantes()}


def test_choix_plus_bas_rmse_et_ex_aequo():
    r = _res(c1=(True, 2.000), **{"c2_a0.5_H100": (True, 2.004), "c3_a0.25_H300": (True, 1.999)})
    assert C.choisir(r, FAM)["retenue"] == "c2_a0.5_H100"                # a 0,005 du meilleur : C2, la plus simple
    r = _res(c1=(True, 2.000), **{"c2_a0.5_H100": (True, 2.0060), "c3_a0.25_H300": (True, 2.0035)})
    assert C.choisir(r, FAM)["retenue"] == "c1"                          # C2 hors ex aequo ; C1 avant C3
    r = _res(c1=(True, 2.010), **{"c3_a0.25_H300": (True, 2.000), "c2_a0.5_H100": (False, 1.5)})
    s = C.choisir(r, FAM)
    assert s["retenue"] == "c3_a0.25_H300" and s["passent"] == ["c1", "c3_a0.25_H300"]   # C2 ne passe pas
    r = _res(**{"c2_a0.5_H100": (True, 2.003), "c2_a0.25_H300": (True, 2.001), "c1": (True, 2.000)})
    assert C.choisir(r, FAM)["retenue"] == "c2_a0.25_H300"                # deux C2 ex aequo : le plus bas RMSE
    assert C.choisir(_res(c1=(True, 2.0)), FAM)["famille"] == "c1"


def test_aucune_variante_arret():
    r = _res(c1=(False, 1.0), **{"c2_a0.5_H100": (False, 0.9), "c3_a0.75_H100": (False, 0.8)})
    s = C.choisir(r, FAM)
    assert s["arret"] is True and s["retenue"] is None and "2026 n'est pas lue" in s["raison"]


def test_variantes_de_la_regle():
    noms = [v["nom"] for v in C.variantes()]
    assert noms[0] == "c1" and len(set(noms)) == 13
    assert sum(n.startswith("c2_") for n in noms) == sum(n.startswith("c3_") for n in noms) == 6
    assert "c2_a0.25_H100" in noms and "c3_a0.75_H300" in noms


# ---------------------------------------------------------------------- refus : une seule lecture
def _selection(d: Path, sha: str, arret=False, variante=None) -> Path:
    f = d / "selection.json"
    f.write_text(json.dumps({"regle_sha256": sha, "arret": arret, "variante": variante}), encoding="utf-8")
    return f


def test_jugement_refuse_si_deja_lu_ou_sans_selection(tmp_path):
    sha = C.verifier_regle()
    sortie = tmp_path / "resultat"
    sortie.mkdir()
    var = {"nom": "c2_a0.5_H100", "famille": "c2", "a0": 0.5, "H": 100.0}
    for nom in J.SORTIES:
        (sortie / nom).write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit, match="une seule lecture"):
            J.juger(None, None, _selection(tmp_path, sha, variante=var), [tmp_path], tmp_path, sortie)
        (sortie / nom).unlink()
    with pytest.raises(SystemExit, match="ARRET"):
        J.juger(None, None, _selection(tmp_path, sha, arret=True), [tmp_path], tmp_path, sortie)
    with pytest.raises(SystemExit, match="empreinte"):
        J.juger(None, None, _selection(tmp_path, "0" * 64, variante=var), [tmp_path], tmp_path, sortie)
    with pytest.raises(SystemExit, match="controle"):
        J.juger(None, None, _selection(tmp_path, sha, variante=var), [tmp_path], tmp_path, sortie)
    (sortie / "controle_entree.json").write_text(json.dumps({"regle_sha256": sha, "passe": False}), encoding="utf-8")
    with pytest.raises(SystemExit, match="controle"):
        J.juger(None, None, _selection(tmp_path, sha, variante=var), [tmp_path], tmp_path, sortie)
    assert not (sortie / "paires.csv").exists()
    with pytest.raises(SystemExit, match="existe deja"):
        J.controler(None, tmp_path, [tmp_path], tmp_path, sortie)
    v = tmp_path / "validation"
    v.mkdir()
    (v / "selection.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="selection"):
        C.calculer(None, None, tmp_path, tmp_path, v)


# ---------------------------------------------------------------------- service : verdicts gfs_v5
def test_statut_service_lit_les_verdicts_v5(tmp_path, monkeypatch):
    from pcfno3d import statut_service as S

    sha = C.verifier_regle()
    f = tmp_path / "verdicts_gfs_v5.json"
    f.write_text(json.dumps({"regle_sha256": sha, "cases": {"national 3D / analyse": "accepte",
                                                          "national 3D / +24 h": "refuse",
                                                          "national 3D / +48 h": "accepte", "autre": "accepte"}}),
                 encoding="utf-8")
    v5 = S.verdicts_gfs_v5(f)
    assert v5 == {"national 3D / analyse": True, "national 3D / +24 h": False, "national 3D / +48 h": True}
    s = S.statut(True, S.TEMPS_REEL, verdicts_v5=v5)
    assert s["statut"] == S.VALIDE and "regle_national_gfs_v5_2" in s["preuve"]
    assert S.statut(True, S.PREVISION, j=1, verdicts_v5=v5)["statut"] == S.EXPERIMENTAL          # +24 h refuse
    assert S.statut(True, S.PREVISION, j=2, verdicts_v5=v5)["statut"] == S.VALIDE                # +48 h accepte
    assert S.statut(True, S.TEMPS_REEL, couche_10m=True, verdicts_v5=v5)["statut"] == S.EXPERIMENTAL   # couche 10 m
    assert S.statut(False, S.TEMPS_REEL, verdicts_v5=v5)["statut"] == S.EXPERIMENTAL              # Grand Tunis
    assert S.statut(True, S.TEMPS_REEL)["statut"] == S.EXPERIMENTAL                               # sans fichier
    # empreinte d'une autre regle (gfs_v4) : ignore
    f.write_text(json.dumps({"regle_sha256": J4.verifier_regle(), "cases": {"national 3D / analyse": "accepte"}}),
                 encoding="utf-8")
    assert S.verdicts_gfs_v5(f) == {}
    # empreintes de la v5 (jamais executee) et de la v5.1 (arretee sans verdict) : refusees, seule la v5.2 compte
    for vieux in ("519ebd64ba700df855f60ba26c625923f7da8677f4949f3dbda1207cf6688d73",
                  "570713841039a5fda8e4d8727a047a4c6a3171a9c212b5402cdb2b7356be9cdd"):
        f.write_text(json.dumps({"regle_sha256": vieux, "cases": {"national 3D / analyse": "accepte"}}),
                     encoding="utf-8")
        assert S.verdicts_gfs_v5(f) == {}
    assert S.REGLE_GFS_V5.name == "regle_national_gfs_v5_2.json"
    monkeypatch.setenv("PCFNO3D_VERDICTS_GFS_V5", str(tmp_path / "absent.json"))
    assert S.verdicts_gfs_v5() == {}
    # gfs_v4 inchange : ses cases gardent leur preuve
    s4 = S.statut(True, S.TEMPS_REEL, verdicts={"a / analyse": True}, verdicts_v5={})
    assert s4["statut"] == S.VALIDE and "regle_national_gfs_v4" in s4["preuve"]


# ---------------------------------------------------------------------- de bout en bout (mini-jeu, CPU)
LAT_N = np.array([35.75, 36.0, 36.25, 36.5])          # mailles du forcage : noeuds de la boite nationale (0,25 deg)
LON_N = np.array([9.75, 10.0, 10.25, 10.5])
CODES = [("DTKA", "DTKA Tabarka"), ("DTTX", "DTTX Sfax"), ("DTTZ", "DTTZ Tozeur")]


def _champs_boite(t: pd.Timestamp, decalage: float) -> dict:
    """Champs GFS synthetiques de la boite nationale a l'instant t (1, 33, 21) : vent lineaire en lat / lon /
    hauteur, variable dans le temps ; T, HR, p constants."""
    from pcfno3d.gfs import NIVEAUX_GFS, grille_tunisie

    lat, lon = grille_tunisie()
    la, lo = np.meshgrid(lat, lon, indexing="ij")

    def uvdir(h):
        u = 2.0 + 0.01 * h + 3.0 * (la - 36.0) + 0.1 * t.hour + decalage
        v = -1.0 + 0.005 * h + 2.0 * (lo - 10.0)
        return (np.hypot(u, v)[None].astype(np.float32),
                ((270.0 - np.degrees(np.arctan2(v, u))) % 360.0)[None].astype(np.float32))

    c = {"elevation": np.zeros(la.shape, np.float32), "terre": la < 36.1}
    c["wind_speed_10m"], c["wind_direction_10m"] = uvdir(10.0)
    c["wind_speed_100m"], c["wind_direction_100m"] = uvdir(100.0)
    for p in NIVEAUX_GFS:
        z = 110.0 + (1000 - p) * 9.0
        c[f"wind_speed_{p}hPa"], c[f"wind_direction_{p}hPa"] = uvdir(z)
        c[f"geopotential_height_{p}hPa"] = np.full((1,) + la.shape, z, np.float32)
    for k, val in (("temperature_2m", 20.0 + 0.1 * t.hour), ("relative_humidity_2m", 60.0),
                   ("surface_pressure", 1010.0), ("pressure_msl", 1012.0), ("shortwave_radiation", 200.0)):
        c[k] = np.full((1,) + la.shape, val, np.float32)
    return c


@pytest.fixture(scope="module")
def jeu_v5(jeu_gfs, tmp_path_factory):  # noqa: F811
    """Mini-jeu entraine + archive GFS synthetique (boite nationale, toutes les 3 h, f0 / f24 / f48), forcages par
    echeance ecrits par pcfno3d.gfs.construire_national, METAR synthetiques ; aucun telechargement."""
    from pcfno3d import gfs

    c, _, _, _ = jeu_gfs
    d = tmp_path_factory.mktemp("gfs_v5")
    data = d / "jeu"
    shutil.copytree(c.paths.data_dir, data)
    ancien = json.loads((c.paths.data_dir / "stations.json").read_text(encoding="utf-8"))
    (data / "stations.json").write_text(json.dumps(
        [{"nom": n, "code": k, "iy": s["iy"], "ix": s["ix"]} for (k, n), s in zip(CODES, ancien)]), encoding="utf-8")
    idx = pd.read_csv(c.paths.cache_dir / "index.csv")
    t_cache = sorted(pd.to_datetime(idx["timestamp"]).unique())
    tous = gfs.instants_periode(str(t_cache[0].floor("D")), str(t_cache[-1].floor("D") + pd.Timedelta(hours=21)))
    lat_tn, lon_tn = gfs.grille_tunisie()
    tn = {e: d / f"gfs_f{e}_tn" for e in C.ECHEANCES}
    forc = d / "forcages"
    for e in C.ECHEANCES:
        tn[e].mkdir()
        for t in tous:
            run, l_eff = gfs.run_pour(t, e)
            gfs._ecrire_national(gfs.fichier_national(tn[e], t, e), _champs_boite(t, e / 24.0), lat_tn, lon_tn,
                                 run, l_eff)
        gfs.construire_national(list(t_cache), e, LAT_N, LON_N, HAUTEURS, forc / f"forcage_gfs_f{e}.csv", tn[e],
                                travailleurs=1, verbose=False)
    base = copy.deepcopy(c)
    base.apply_overrides({"paths.data_dir": data, "paths.forcage_era5": forc / "forcage_gfs_f0.csv"})
    obs = d / "obs"
    obs.mkdir()
    rng = np.random.default_rng(3)
    lignes = [{"station_id": k, "timestamp_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "wind_speed_m_s": float(rng.uniform(1, 8)), "wind_dir_deg": float(rng.uniform(0, 360)),
               "wind_u_m_s": np.nan, "wind_v_m_s": np.nan} for t in tous for k, _ in CODES]
    pd.DataFrame(lignes).to_csv(obs / "metar_2025.csv", index=False)
    return base, forc, tn, obs, tous


def test_choix_de_bout_en_bout(jeu_v5, tmp_path):
    base, forc, _, obs, _ = jeu_v5
    sortie = tmp_path / "validation_2021"
    sel = C.calculer(base, copy.deepcopy(base), forc, obs, sortie, device="cpu", verbose=False)
    p = pd.read_csv(sortie / "paires_validation.csv")
    assert set(p["echeance_h"]) == set(C.ECHEANCES) and set(p["code"]) == {k for k, _ in CODES}
    for v in C.variantes():
        assert p[f"u_{v['nom']}"].notna().all(), v["nom"]
    # C1 = le meme modele ici : c1 == cp_s44 et c3_x == c2_x exactement
    np.testing.assert_allclose(p["u_c1"], p["u_cp_s44"])
    np.testing.assert_allclose(p["u_c3_a0.5_H100"], p["u_c2_a0.5_H100"])
    # ancrage : 1 - a(h) = (1 - a0)(1 - x(h)), l'ecart au modele est proportionnel a 1 - a0 (H fixe)
    for H in (100, 300):
        d25, d75 = p[f"u_c2_a0.25_H{H}"] - p["u_cp_s44"], p[f"u_c2_a0.75_H{H}"] - p["u_cp_s44"]
        np.testing.assert_allclose(3.0 * d75, d25, atol=1e-6)
        assert d25.abs().max() > 1e-3                                    # l'ancrage agit pres du sol
    for f in ("selection.json", "variantes.json", "diagnostic_tabarka.csv", "profil_tabarka.csv",
              "bilan_validation.csv", "validation_gfs_v5.md"):
        assert (sortie / f).exists(), f
    diag = pd.read_csv(sortie / "diagnostic_tabarka.csv")
    assert {"tout", "secteur", "periode", "saison"} <= set(diag["decoupage"])
    assert sel["regle_sha256"] == C.verifier_regle() and isinstance(sel["arret"], bool)
    with pytest.raises(SystemExit):
        C.calculer(base, copy.deepcopy(base), forc, obs, sortie, device="cpu", verbose=False)


def test_instant_sans_echeance_retire_sans_arret(jeu_v5, tmp_path, monkeypatch):
    """Regle v5.2, controle_d_echeances (b) : un instant de validation sans l'une des trois echeances est retire
    pour toutes les sources et les trois cases, publie, et le choix continue (la v5.1 s'arretait ici)."""
    base, forc, _, obs, _ = jeu_v5
    stems, row = C.stems_validation(base)
    retire = sorted(stems)[0]
    t_retire = C.instant_utc(row, retire)
    lire = C.temps_du_forcage
    monkeypatch.setattr(C, "temps_du_forcage",
                        lambda f: lire(f) - {t_retire} if Path(f).name == "forcage_gfs_f24.csv" else lire(f))
    sortie = tmp_path / "validation_2021_v5_2"
    sel = C.calculer(base, copy.deepcopy(base), forc, obs, sortie, device="cpu", verbose=False)
    info = sel["instants"]
    assert info["absents"] == 1 and info["absents_liste"] == [retire] and info["absents_utc"] == [str(t_retire)]
    assert info["absents_par_echeance"] == {"analyse": 0, "+24 h": 1, "+48 h": 0}
    assert not (sortie / "arret.json").exists() and (sortie / "selection.json").exists()
    p = pd.read_csv(sortie / "paires_validation.csv")
    assert retire not in set(p["stem"]) and set(p["echeance_h"]) == set(C.ECHEANCES)
    assert str(t_retire) in (sortie / "validation_gfs_v5.md").read_text(encoding="utf-8")


def test_controle_entree_de_production(jeu_v5, tmp_path):
    """Regle v5.1 : les parties VENT de l'entree de production (temps_reel) et du cache sont identiques (forcage,
    reseau GFS 10 m, stations, carte des capteurs) : le controle passe ; la surface T, rho, HR, p du cache
    (generateur) n'est pas celle de GFS : ecarts rapportes (moyenne, max), sans arret."""
    base, forc, tn, obs, _ = jeu_v5
    sortie = tmp_path / "resultat"
    r = J.controler(base, forc, list(tn.values()), obs, sortie, verbose=False)
    v, a = r["parties_vent"], r["rapporte_sans_arret"]
    assert r["entrees_comparees"] > 0 and set(v) == set(J.VENT)
    assert v["capteurs_differents"]["max"] == 0
    for k in ("forcage_scalaires", "forcage_cellules", "reseau_gfs_10m", "stations_valeurs"):
        assert v[k]["max"] <= 1e-5 and v[k]["moyenne"] <= v[k]["max"], k
    assert r["passe"] is True                                             # vent identique : 2026 peut etre lu
    assert a["surface_temperature_normalisee"]["max"] > 1e-5              # T : cache != GFS, rapporte sans arret
    assert a["surface_temperature_brute"]["moyenne"] > 0
    for k in ("densite", "humidite", "pression"):
        assert {"moyenne", "max"} <= set(a[f"surface_{k}_normalisee"])
    assert "traits_geostrophique_vent_thermique_4" in a and a["heure"]["max"] == 0
    lu = json.loads((sortie / "controle_entree.json").read_text(encoding="utf-8"))
    assert lu["passe"] is True and lu["regle_sha256"] == C.verifier_regle()
    with pytest.raises(SystemExit, match="existe deja"):
        J.controler(base, forc, list(tn.values()), obs, sortie, verbose=False)


def test_controle_vent_different_arrete(jeu_v5, tmp_path):
    """Un forcage du cache qui ne correspond pas a l'archive (vent decale) : parties vent differentes, ARRET."""
    base, forc, tn, obs, _ = jeu_v5
    autre = tmp_path / "forcages"
    autre.mkdir()
    for e in C.ECHEANCES:
        shutil.copy2(forc / f"forcage_gfs_f{e}.csv", autre / f"forcage_gfs_f{e}.csv")
        with np.load(forc / f"forcage_gfs_f{e}.npz") as z:
            arr = {k: z[k] for k in z.files}
        arr["u"] = (arr["u"] + 0.5).astype(np.float32)
        np.savez(autre / f"forcage_gfs_f{e}.npz", **arr)
    r = J.controler(base, autre, list(tn.values()), obs, tmp_path / "resultat", verbose=False)
    assert r["passe"] is False and r["parties_vent"]["forcage_cellules"]["max"] > 0.4
    assert "ARRET" in r["consequence"]


@pytest.mark.parametrize("variante", ["c2_a0.5_H100", "c3_a0.25_H300"])
def test_jugement_de_bout_en_bout(jeu_v5, tmp_path, monkeypatch, variante):
    from pcfno3d import statut_service as S

    base, _, tn, obs, tous = jeu_v5
    sha = C.verifier_regle()
    j = _charger(f"juger_gfs_v5_{variante}", "juger_national_gfs_v5.py")
    monkeypatch.setattr(j, "echantillon_2026", lambda: list(tous[8:]))   # mini echantillon (l'archive synthetique)
    sortie = tmp_path / "resultat"
    sortie.mkdir()
    (sortie / "controle_entree.json").write_text(json.dumps({"regle_sha256": sha, "passe": True}), encoding="utf-8")
    var = next(v for v in C.variantes() if v["nom"] == variante)
    sel = _selection(tmp_path, sha, variante=var)
    v = j.juger(base, copy.deepcopy(base), sel, list(tn.values()), obs, sortie, device="cpu", verbose=False)
    assert set(v["cases"]) == set(C.CASES) and v["candidat"]["nom"] == variante
    p = pd.read_csv(sortie / "paires.csv")
    assert set(p["echeance_h"]) == set(C.ECHEANCES)
    assert p[["u_candidat", "u_gfs_brut", "u_cp_s44_gfs"]].notna().all().all()
    assert v["instants"]["absents"] == 0 and v["instants"]["instants"] == len(tous) - 8
    for f in ("bilan.csv", "verdicts_gfs_v5.json", "gfs_v5.md"):
        assert (sortie / f).exists()
    assert set(S.verdicts_gfs_v5(sortie / "verdicts_gfs_v5.json")) == set(C.CASES)
    with pytest.raises(SystemExit, match="une seule lecture"):
        j.juger(base, base, sel, list(tn.values()), obs, sortie, device="cpu", verbose=False)


def test_jugement_sans_cp_s44_faute_de_budget(jeu_v5, tmp_path, monkeypatch):
    base, _, tn, obs, tous = jeu_v5
    sha = C.verifier_regle()
    j = _charger("juger_gfs_v5_sans_cp", "juger_national_gfs_v5.py")
    monkeypatch.setattr(j, "echantillon_2026", lambda: list(tous[-6:]))
    assert j.modeles_requis("c1", False) == ("c1",) and j.modeles_requis("c2", False) == ("cp",)
    sortie = tmp_path / "resultat"
    sortie.mkdir()
    (sortie / "controle_entree.json").write_text(json.dumps({"regle_sha256": sha, "passe": True}), encoding="utf-8")
    var = next(v for v in C.variantes() if v["nom"] == "c1")
    v = j.juger(base, copy.deepcopy(base), _selection(tmp_path, sha, variante=var), list(tn.values()), obs, sortie,
                device="cpu", verbose=False, avec_cp_s44=False)
    assert v["cp_s44_gfs_calcule"] is False
    assert all(x == "non calcule faute de budget" for x in v["rapporte_sans_verdict_cp_s44_gfs"].values())
    assert "non calcule faute de budget" in (sortie / "gfs_v5.md").read_text(encoding="utf-8")
    assert pd.read_csv(sortie / "paires.csv")["u_cp_s44_gfs"].isna().all()


# ---------------------------------------------------------------------- carnets
def test_carnets_commit_fige_empreintes_et_dossiers_neufs():
    sys_path = str(REPO / "scripts")
    import sys
    sys.path.insert(0, sys_path)
    try:
        g = _charger("make_gfs_v5_nb_t", "make_national_gfs_v5_nb.py")
    finally:
        sys.path.remove(sys_path)
    sha = C.verifier_regle()
    sha_v51 = C.verifier_regle(C.REGLE.with_name("regle_national_gfs_v5_1.json"))
    sha_v5 = C.verifier_regle(C.REGLE.with_name("regle_national_gfs_v5.json"))
    for commit in (g.COMMIT, "a" * 40):
        nbs = g.carnets(commit)
        assert set(nbs) == {"C", "B2"}
        src = {k: "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
               for k, nb in nbs.items()}
        for k, x in src.items():
            assert x.index("COMMIT_FIGE") < x.index("UNITES_DEJA_UTILISEES") < x.index("drive.mount")
            assert x.index("22 * 1024") < x.index("drive.mount")                 # GPU verifie avant le Drive
            assert sha in x and sha_v51 in x and J4.verifier_regle() in x      # v5.2, marqueur v5.1, gfs_v4
            assert sha_v5 not in x                                             # la v5 n'est pas executee
            assert f'SHA_REGLE = "{sha}"' in x and f'SHA_V5_1 = "{sha_v51}"' in x
            assert "cout ESTIME" in x and "plafond : 90 unites (v5.1 + v5.2)" in x and "~4.8 unites/h" in x
            assert f'COMMIT = "{commit}"' in x and "runtime.unassign()" in x
            assert 'GV5 / "validation_2021_v5_2"' in x and 'GV5 / "validation_2021"\n' not in x
            assert nbs[k]["metadata"]["colab"]["gpuType"] == "L4" and '("L4", "A100")' in x
            # C1 reutilise : verifie (marqueur v5.1, 12 epoques, epoque 2), jamais reentraine
            assert "verifier_c1()" in x and 'R5 / "regle_gfs_v5_1.json"' in x and "if ep != 2:" in x
            assert "pcfno3d -c national_v5_gfs_mixte_colab" not in x and "entrainer_c1" not in x
            assert "recalcul_activations" not in x
            for interdit in ("cdsapirc", "apikey", "rmtree(GV5", "rmtree(R5", "unlink(", "verdicts_gfs_v4.json",
                             "shutil.move(", "marqueur.write_text"):
                assert interdit not in x, (k, interdit)
        c, b2 = src["C"], src["B2"]
        # C : etapes 1 + 3 seulement ; refus si la selection v5.2 existe deja, AVANT l'extraction du cache
        assert "choisir_national_gfs_v5.py" in c and 'if sel["arret"]' in c
        assert "juger_national_gfs_v5.py" not in c and "historique_metar.py" not in c
        assert c.index('(VAL / "selection.json").exists()') < c.index("zf.extract(m, CACHE)")
        assert c.index("verifier_c1()", c.index("PARTIE C")) < c.index("choisir_national_gfs_v5.py")
        # B2 : refus sans selection v5.2 / si ARRET / si deja juge, AVANT l'extraction du cache
        assert "choisir_national_gfs_v5.py" not in b2
        assert b2.index('_sel.get("regle_sha256") != SHA_REGLE') < b2.index("zf.extract(m, CACHE)")
        assert b2.index('_sel.get("arret")') < b2.index("zf.extract(m, CACHE)")
        assert b2.index('"verdicts_gfs_v5.json", "paires.csv"') < b2.index("zf.extract(m, CACHE)")
        assert b2.index("juger_national_gfs_v5.py controle") < b2.index("historique_metar.py") \
            < b2.index("juger_national_gfs_v5.py juger")
    # les carnets de la v5.1 restent tels quels (histoire) et ne sont plus produits par le generateur
    for n in ("pcfno3d_national_gfs_v5_A_cpu.ipynb", "pcfno3d_national_gfs_v5_B1_a100.ipynb"):
        assert (REPO / "notebooks" / "current" / n).exists()
    assert set(g.NOMS.values()) == {"pcfno3d_national_gfs_v5_2_C_l4.ipynb", "pcfno3d_national_gfs_v5_2_B2_l4.ipynb"}


def test_echantillon_2026_et_lecture_de_l_archive(jeu_v5):
    from pcfno3d.gfs import instants_echantillon, lire_national, run_pour

    t = J.echantillon_2026()
    assert len(t) == 608 == 152 * 4 and len(set(t)) == 608 and t == sorted(t)
    assert t[0] == pd.Timestamp("2026-01-01 00:00") and t[-1] == pd.Timestamp("2026-10-30 18:00")
    assert all(pd.Timestamp("2026-01-01") <= x <= pd.Timestamp("2026-10-31 23:59") for x in t)
    assert all(x.dayofyear % 2 == 1 and x.hour in (0, 6, 12, 18) and x.minute == 0 for x in t)
    assert all(run_pour(x, 0)[1] == 0 for x in t)                       # analyse toujours f000
    assert {x.dayofyear for x in t} == set(range(1, 304, 2))
    assert len(instants_echantillon("2026-01-01T00:00", "2026-01-04T23:00", (0, 12), jours_impairs=False)) == 8
    _, _, tn, _, tous = jeu_v5
    manquant = tous[-1] + pd.Timedelta(hours=3)
    c, ech, absents = lire_national(tous[:3] + [manquant], 24, LAT_N, LON_N, [tn[24]])
    assert absents == [manquant] and len(c["temps"]) == 3 and c["wind_speed_10m"].shape == (3, 4, 4)
    assert all(v[1] >= 24 for v in ech.values())                         # echeance effective >= 24 h
