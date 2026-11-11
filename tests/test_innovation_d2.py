"""Candidat D2 national (pcfno3d.innovation, regle_d2_national.json PROPOSEE, definition d'Ines du 11/11) :
canal d'innovation (10 m, OI ou Cressman, decroissance verticale, distance au point le plus proche),
reference « interpolation optimale seule », extension init_from, statistiques, sur le mini-jeu."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pcfno3d.dataset import ANEO3D, assemble_input, entree_era5_stations
from pcfno3d.geometry import Geometry
from pcfno3d.innovation import N_OI, InnovationOI, ajuster_statistiques, decouper_tfeat, predicteur_oi
from pcfno3d.model import build_model, save_inference_checkpoint
from pcfno3d.train import Trainer
from test_forcage_era5 import _jeu_reel, cfg_era5, forcage, vent  # noqa: F401  (fixtures)

REPO = Path(__file__).resolve().parents[1]
L_M = 300.0                                             # L du mini-jeu (domaine 2 x 1 km)


def _module(nom):
    spec = importlib.util.spec_from_file_location(nom, REPO / "scripts" / f"{nom}.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    return M


def _d2(c, **inn):
    """Config D2 depuis une config C' (mini-jeu en mode era5_stations) ; capteurs simules a 10 m."""
    c2 = copy.deepcopy(c)
    c2.apply_overrides({"innovation.actif": True, "innovation.L_km": L_M / 1000.0,
                        "mask.aneo_hauteur_capteur_m": 10.0, **{f"innovation.{k}": v for k, v in inn.items()}})
    return c2


@pytest.fixture(scope="module")
def jeu_obs(cfg_era5, index, tmp_path_factory):
    return _jeu_reel(cfg_era5, index, tmp_path_factory.mktemp("d2"))


@pytest.fixture(scope="module")
def jeu(jeu_obs):
    return jeu_obs[0]


@pytest.fixture(scope="module")
def geom_c(jeu):
    return Geometry(jeu)


@pytest.fixture(scope="module")
def geom_d2(jeu):
    return Geometry(_d2(jeu))


def _norm(geom, index):
    from pcfno3d.normalize import Normalizer
    return Normalizer.load_or_compute(geom.cfg, geom, index, verbose=False)


def _val(index, geom):
    val = index[index["split"] == "val"]
    return val["stem"].values, val["hour"].values, val["doy"].values, geom


class _Candidats:
    """EntreeANEO de l'evaluation avec n capteurs a 10 m ; mini-grille : candidats = colonnes valides (les
    contraintes de l'OS6 sont trop strictes ici) ; `plus` m/s ajoutes a u du premier capteur."""

    def __init__(self, geom, n, plus=0.0):
        from pcfno3d.aneo_entree import EntreeANEO
        self.e = EntreeANEO(geom, None, 0.0, (50.0, 150.0), (n, n), 10.0)
        jj, ii = np.nonzero(geom.valid.any(axis=0))
        self.e._cand = np.stack([jj, ii], 1)
        self.plus, self.cel = plus, None

    def appliquer(self, cel, rng):
        return self.e.appliquer(cel, rng)

    def capteurs(self, S, Y, rng, n=None):
        cel, u, v = self.e.capteurs(S, Y, rng, n)
        u = u.copy()
        u[0] += self.plus
        self.cel = cel
        return cel, u, v


# ---------------------------------------------------------------------------------- (1) l'etalement
def test_oi_un_point_rend_l_innovation_et_decroit(geom_d2):
    oi = geom_d2.oi
    r2 = 0.5 ** 2                                           # rapport_o_b = sigma_o / sigma_b = 0,5
    iy, ix = geom_d2.ny // 2, geom_d2.nx // 2
    out = oi.analyse([iy], [ix], [3.0], [-2.0], [0.5])
    assert out.shape == (N_OI, geom_d2.ny, geom_d2.nx)
    assert abs(out[0, iy, ix] - 3.0 / (1 + r2)) < 1e-5 and abs(out[1, iy, ix] + 2.0 / (1 + r2)) < 1e-5
    ligne = out[0, iy, ix:]
    assert (np.diff(ligne) <= 1e-7).all() and ligne[-1] < ligne[0]          # decroit avec la distance
    d = np.arange(len(ligne)) * geom_d2.dx
    np.testing.assert_allclose(ligne, 3.0 / (1 + r2) * np.exp(-0.5 * (d / L_M) ** 2) * (d <= 4 * L_M), atol=1e-5)
    # canal distance : 0 au point, distance euclidienne (km) ailleurs
    yy, xx = np.meshgrid(np.arange(geom_d2.ny) * geom_d2.dy, np.arange(geom_d2.nx) * geom_d2.dx, indexing="ij")
    np.testing.assert_allclose(out[2], np.hypot(yy - iy * geom_d2.dy, xx - ix * geom_d2.dx) / 1000.0, atol=1e-5)
    # sans point : pas d'increment, distance = diagonale du domaine
    vide = oi.analyse([], [], [], [], [])
    assert not vide[:2].any() and np.allclose(vide[2], oi.distance_max_km)
    # exponentielle : meme valeur au point, decroissance exp(-d/L)
    e = InnovationOI(geom_d2, _d2(geom_d2.cfg, correlation="exponentielle").innovation)
    o2 = e.analyse([iy], [ix], [3.0], [0.0], [0.5])
    assert abs(o2[0, iy, ix + 2] - 3.0 / (1 + r2) * np.exp(-2 * geom_d2.dx / L_M)) < 1e-5
    # Cressman (version simple) : le point rend toute l'innovation, poids (R^2 - d^2) / (R^2 + d^2), R = 2 L
    c = InnovationOI(geom_d2, _d2(geom_d2.cfg, methode="cressman").innovation)
    o3 = c.analyse([iy], [ix], [3.0], [0.0], [0.5])
    R2, d2 = (2 * L_M) ** 2, (3 * geom_d2.dx) ** 2
    assert abs(o3[0, iy, ix] - 3.0) < 1e-5 and abs(o3[0, iy, ix + 3] - 3.0 * (R2 - d2) / (R2 + d2)) < 1e-5
    assert o3[0, iy, ix + 6] == 0.0


def test_era5_10m_au_point_et_decroissance_verticale(geom_d2):
    oi, it = geom_d2.oi, geom_d2.interp_era5
    vec = geom_d2.forcage.vecteur("2025-01-02 12:00:00")
    cel = vec[len(vec) - geom_d2.forcage.n_cellules:]
    iy, ix = np.array([1, 5, 8]), np.array([2, 10, 17])
    u, v = oi.era5_10m_aux_points(cel, iy, ix)
    uu, vv = vent(10.0, it.lat[iy, ix], it.lon[iy, ix])          # champ lineaire : exact
    np.testing.assert_allclose(u, uu, atol=1e-4)
    np.testing.assert_allclose(v, vv, atol=1e-4)
    hn = np.maximum(np.asarray(geom_d2.z_levels)[:, None, None] - geom_d2.zt[None], 5.0)
    f = np.exp(-np.maximum(hn - 10.0, 0) / 500.0) * geom_d2.valid
    np.testing.assert_allclose(oi.f3d, f, atol=1e-6)
    # capteurs a 25 m ramenes a 10 m par la loi log (option « loi_log »)
    c = _d2(geom_d2.cfg, capteurs_10m="loi_log")
    c.mask.aneo_hauteur_capteur_m = 25.0
    g = types.SimpleNamespace(**{k: getattr(geom_d2, k) for k in ("ny", "nx", "z0")}, cfg=c)
    o = InnovationOI.__new__(InnovationOI)
    o.geom, o.cfg, o.ny, o.nx = g, c.innovation, geom_d2.ny, geom_d2.nx
    S = np.zeros((8, geom_d2.ny, geom_d2.nx), np.float32)
    S[0, 3, 4] = 6.0
    p = o.points(S, None, np.array([[3, 4]]))
    z0 = max(float(geom_d2.z0[3, 4]), 1e-4)
    assert abs(p["u"][0] - 6.0 * math.log(10 / z0) / math.log(25 / z0)) < 1e-6
    # « verite_10m » avec des capteurs simules a 25 m : refuse (incoherent)
    c3 = _d2(geom_d2.cfg)
    c3.mask.aneo_hauteur_capteur_m = 25.0
    o.cfg, o.geom = c3.innovation, types.SimpleNamespace(z0=geom_d2.z0, cfg=c3)
    with pytest.raises(ValueError, match="aneo_hauteur_capteur_m"):
        o.points(S, None, np.array([[3, 4]]))


# ------------------------------------------------------------ (2) +15 m/s a un capteur : canaux et sortie
def test_plus_15_a_un_capteur_change_les_canaux_et_la_sortie(geom_d2, index):
    norm = _norm(geom_d2, index)
    args = _val(index, geom_d2)
    a = _Candidats(geom_d2, 3)
    b = _Candidats(geom_d2, 3, plus=15.0)
    xa = ANEO3D(*args, train=False, seed=1000, keep_eval=1.0, aneo_eval=a)[0]
    xb = ANEO3D(*args, train=False, seed=1000, keep_eval=1.0, aneo_eval=b)[0]
    assert np.array_equal(a.cel, b.cel)                     # memes tirages
    iy, ix = (int(x) for x in b.cel[0])
    _, qa = decouper_tfeat(xa[2][None], geom_d2)
    _, qb = decouper_tfeat(xb[2][None], geom_d2)
    dq = (qb - qa).reshape(N_OI, geom_d2.ny, geom_d2.nx).numpy()
    assert dq[0, iy, ix] > 5.0                               # ~15 / (1 + 0,25) si le capteur est isole
    assert not dq[2].any()                                   # memes points : meme distance
    assert qa.reshape(N_OI, geom_d2.ny, geom_d2.nx)[2, iy, ix] == 0.0
    # canaux d'entree du reseau : du (normalise, x f(h)) change dans la colonne du capteur
    x1 = assemble_input(*(t[None] for t in xa[:3]), geom_d2, norm)
    x2 = assemble_input(*(t[None] for t in xb[:3]), geom_d2, norm)
    c_du = geom_d2.c_in - 2 - N_OI
    assert x1.shape[1] == geom_d2.c_in
    col = geom_d2.valid[:, iy, ix]
    assert (x2 - x1)[0, c_du, col, iy, ix].min() > 2.0 / float(norm.y_sd[0])
    assert torch.equal(x1[:, -2:], x2[:, -2:])               # canaux ERA5 inchanges (base « era5 »)
    # modele neuf (colonnes du lift non nulles) : la sortie change pres du capteur
    torch.manual_seed(0)
    net = build_model(geom_d2.cfg, geom_d2.c_in).eval()
    with torch.no_grad():
        d = (norm.denorm(net(x2))[0] - norm.denorm(net(x1))[0])[0].abs().numpy()
    assert d[geom_d2.valid[:, iy, ix], iy, ix].max() > 1e-3


# ----------------------------------------------------------------- (3) init_from un modele C' : identique
def test_init_from_c_prime_sortie_identique_au_depart(jeu, geom_c, geom_d2, index, tmp_path):
    torch.manual_seed(1)
    parent = build_model(jeu, geom_c.c_in)
    with torch.no_grad():                                    # un C' « entraine » : tete u, v non nulle
        for p in parent.proj.parameters():
            p.add_(0.1 * torch.randn_like(p))
    ck = save_inference_checkpoint(tmp_path / "cp.pt", parent, jeu, geom_c.c_in)
    c2 = geom_d2.cfg
    hote = types.SimpleNamespace(cfg=c2, device=torch.device("cpu"), verbose=False,
                                 model=build_model(c2, geom_d2.c_in))
    Trainer._init_from(hote, ck)
    w, w0 = hote.model.state_dict()["lift.0.weight"], parent.state_dict()["lift.0.weight"]
    assert w.shape[1] == w0.shape[1] + N_OI
    np.testing.assert_array_equal(w[:, :-2 - N_OI].numpy(), w0[:, :-2].numpy())
    np.testing.assert_array_equal(w[:, -2:].numpy(), w0[:, -2:].numpy())
    assert (w[:, -2 - N_OI:-2] == 0).all()
    norm = _norm(geom_c, index)
    ic = ANEO3D(*_val(index, geom_c), train=False, seed=1000, keep_eval=1.0)[0]
    idd = ANEO3D(*_val(index, geom_d2), train=False, seed=1000, keep_eval=1.0)[0]
    _, q = decouper_tfeat(idd[2][None], geom_d2)
    assert q.reshape(N_OI, -1)[:2].abs().max() > 0          # innovation aux aeroports non nulle
    with torch.no_grad():
        oc = parent.eval()(assemble_input(*(t[None] for t in ic[:3]), geom_c, norm))
        od = hote.model.eval()(assemble_input(*(t[None] for t in idd[:3]), geom_d2, norm))
    torch.testing.assert_close(od, oc, atol=1e-5, rtol=0)


# ------------------------------------------------- (4) OI seule meilleure qu'ERA5 sur un cas synthetique
def test_oi_seule_bat_era5_sur_anomalie_lisse(geom_d2, index):
    g, oi = geom_d2, geom_d2.oi
    item = ANEO3D(*_val(index, g), train=False, seed=1000, keep_eval=1.0)[0]
    tfeat = item[2][None]
    cel, _ = decouper_tfeat(tfeat, g)
    era5 = g.interp_era5.dense(cel)[0].numpy()               # (2, nz, ny, nx)
    yy, xx = np.meshgrid(np.arange(g.ny) * g.dy, np.arange(g.nx) * g.dx, indexing="ij")
    A = (3.0 * np.exp(-((yy - 400) ** 2 + (xx - 600) ** 2) / (2 * 450.0 ** 2))
         - 2.0 * np.exp(-((yy - 600) ** 2 + (xx - 1500) ** 2) / (2 * 500.0 ** 2)))
    verite = era5 + np.stack([A, 0.5 * A])[:, None] * oi.f3d[None]
    rng = np.random.default_rng(3)
    jj, ii = np.nonzero(g.valid.any(axis=0))
    k = rng.choice(len(jj), 15, replace=False)
    iy, ix = jj[k], ii[k]
    q = oi.analyse(iy, ix, A[iy, ix], 0.5 * A[iy, ix], np.full(len(k), 0.25))   # mesure = verite au point
    n = tfeat.shape[1] - oi.n_queue
    t2 = torch.cat([tfeat[:, :n], torch.from_numpy(q.ravel())[None]], 1)
    u, v, w = predicteur_oi(g)(None, None, t2, None, 0, "x")
    m = g.valid
    e_oi = np.sqrt(np.mean((u[0].numpy()[m] - verite[0][m]) ** 2 + (v[0].numpy()[m] - verite[1][m]) ** 2))
    e_era5 = np.sqrt(np.mean((era5[0][m] - verite[0][m]) ** 2 + (era5[1][m] - verite[1][m]) ** 2))
    assert e_oi < 0.6 * e_era5, (e_oi, e_era5)
    assert torch.isfinite(w).all()


# ------------------------------------------------------------------------------ option et statistiques
def test_option_base_residuelle_era5_plus_innovation(jeu, index):
    from pcfno3d.dso_national import predicteur_era5
    gb = Geometry(_d2(jeu, base_residuel="era5_oi"))
    ga = Geometry(_d2(jeu))
    norm = _norm(gb, index)
    it = ANEO3D(*_val(index, gb), train=False, seed=1000, keep_eval=1.0)[0]
    xb = assemble_input(*(t[None] for t in it[:3]), gb, norm)
    xa = assemble_input(*(t[None] for t in it[:3]), ga, norm)
    _, q = decouper_tfeat(it[2][None], gb)
    inc = gb.oi.increments_3d(q)[0]                           # (2, nz, ny, nx) m/s
    attendu = xa[0, -2:] + inc / norm.y_sd[:2, None, None, None]
    torch.testing.assert_close(xb[0, -2:], attendu * gb.valid_t, atol=1e-5, rtol=0)
    assert torch.equal(xb[:, :-2], xa[:, :-2])
    # la reference « ERA5 seule » reste ERA5 seule
    u_a = predicteur_era5(ga, norm)(*(t[None] for t in it[:3]), None, 0, "x")[0]
    u_b = predicteur_era5(gb, norm)(*(t[None] for t in it[:3]), None, 0, "x")[0]
    torch.testing.assert_close(u_a, u_b, atol=1e-4, rtol=0)


def test_statistiques_hollingsworth_loennberg_synthetiques():
    """Innovations synthetiques : sigma_b 2, sigma_o 1, L 35 km gaussienne ; 14 stations a moins de 90 km."""
    S = _module("statistiques_innovation_d2")
    rng = np.random.default_rng(0)
    st = {f"S{k}": (34.0 + 0.6 * rng.random(), 9.0 + 0.6 * rng.random()) for k in range(14)}
    codes = sorted(st)
    D = np.array([[S.haversine_km(*st[a], *st[b]) for b in codes] for a in codes])
    B = 4.0 * np.exp(-0.5 * (D / 35.0) ** 2)
    C = np.linalg.cholesky(B + 1e-9 * np.eye(len(codes)))
    lignes = []
    for t in range(3000):
        xu = C @ rng.standard_normal(len(codes)) + rng.normal(0, 1.0, len(codes)) + 0.3
        xv = C @ rng.standard_normal(len(codes)) + rng.normal(0, 1.0, len(codes))
        lignes += [{"t": t, "code": c, "du": xu[i], "dv": xv[i]} for i, c in enumerate(codes)]
    paires, var, biais = S.paires_et_variances(pd.DataFrame(lignes), st)
    assert abs(np.mean([b[0] for b in biais.values()]) - 0.3) < 0.05
    r = ajuster_statistiques(paires, var, "gaussienne", (20.0, 50.0))
    assert abs(r["L_km"] - 35.0) <= 5.0 and abs(r["rapport_o_b"] - 0.5) < 0.1 and not r["L_en_butee"]
    r2 = ajuster_statistiques([p for p in paires if p["distance_km"] > 40], var, "gaussienne", (20.0, 50.0))
    assert r2["distance_min_km"] > 40                        # paires lointaines seulement : rapporte


# ------------------------------------------------------------------------- non-regression et service
def test_sans_innovation_rien_ne_change(jeu, geom_c, geom_d2, index):
    assert geom_c.oi is None and geom_d2.c_in == geom_c.c_in + N_OI
    S = np.load(jeu.paths.cache_dir / f"{index['stem'].iloc[0]}_S.npy")
    r = entree_era5_stations(S, jeu.paths.cache_dir / f"{index['stem'].iloc[0]}_E.npy", geom_c,
                             np.random.default_rng(0), False)
    assert len(r) == 2
    a = ANEO3D(*_val(index, geom_c), train=False, seed=1000, keep_eval=1.0)[0]
    b = ANEO3D(*_val(index, geom_d2), train=False, seed=1000, keep_eval=1.0)[0]
    n = len(a[2])
    assert torch.equal(a[2], b[2][:n]) and len(b[2]) == n + geom_d2.oi.n_queue
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_entrainement_service_et_scripts_d2(jeu_obs, index, tmp_path, monkeypatch):
    """Entrainement court D2 (entree D : ERA5 degradee, zones, capteurs simules + innovation), service qui relit
    les parametres dans le checkpoint, dso_hauteur_sol / verdict national / validation reelle avec la reference
    OI seule, info_capteurs_d2 (choix de H puis information)."""
    import pcfno3d.aneo_entree as AE
    from pcfno3d.dso_national import verdict_national
    from pcfno3d.evaluate import Evaluator
    from pcfno3d.inference import WindFieldModel

    orig = AE.EntreeANEO.candidats

    def cand(self):                                   # mini-grille : contraintes de l'OS6 trop strictes
        try:
            return orig(self)
        except ValueError:
            self._cand = np.argwhere(self.geom.valid.any(axis=0))
            return self._cand
    monkeypatch.setattr(AE.EntreeANEO, "candidats", cand)
    jeu, obs_dir = jeu_obs
    c = _d2(jeu)
    deg = tmp_path / "deg.json"
    deg.write_text(json.dumps({"sigma_u": 1.0, "sigma_v": 1.0, "L_km": 100.0}))
    c.apply_overrides({"paths.run_dir": tmp_path / "run", "mask.aneo_capteurs": [2, 4],
                       "mask.aneo_degradation": str(deg), "mask.aneo_p_zone": 0.5})
    Trainer(c, device="cpu", verbose=False).fit()
    ck = torch.load(c.paths.ckpt_best, map_location="cpu", weights_only=False)
    assert ck["config"]["INNOVATION"]["actif"] is True and ck["config"]["C_IN"] == Geometry(c).c_in
    c_serv = copy.deepcopy(c)
    c_serv.innovation.actif = False                    # le checkpoint fait foi
    wf = WindFieldModel(c_serv, device="cpu", verbose=False)
    assert wf.geom.oi is not None and wf.geom.oi.L == L_M
    stem = index[index["split"] == "test"]["stem"].iloc[0]
    vol = wf.predict_stem(stem)
    assert np.isfinite(vol.u[wf.geom.valid]).all()
    S, hour, doy, stamp = wf.surface_from_cache(stem)
    jj, ii = np.nonzero((S[7] > 0.5) & wf.geom.valid.any(axis=0))
    v2 = wf.predict(S, hour, doy, timestamp=stamp, capteurs_oi=np.array([[jj[0], ii[0]]]))
    assert np.isfinite(v2.u[wf.geom.valid]).all()

    # DSO en hauteur au-dessus du sol : trois predicteurs dont la reference « OI seule »
    H = _module("dso_hauteur_sol")
    ev = Evaluator(c, device="cpu", verbose=False)
    tab, info = H.mesurer(ev, c, None, None)
    assert set(tab.predicteur) == {"modele", "era5", "era5_oi"}
    assert set(H.verdicts(tab, info["global"], c, None)) == {"modele", "era5", "era5_oi"}
    # validation sur les mesures (aeroports exclus a tour de role) : source era5_oi
    V = _module("valider_grand_tunis_reel")
    V.valider(c, obs_dir, "test", device="cpu")
    b = pd.read_csv(c.paths.run_dir / "validation_reelle" / "test" / "stations_bilan.csv")
    assert set(b["source"]) == {"modele", "era5", "generateur", "era5_oi"} and b["rmse_vect"].notna().all()
    # verdict national : colonne OI seule a cote d'ERA5 seule
    ev.run()
    v = verdict_national(c, device="cpu", verbose=False)
    assert all("rmse_vect_era5_oi" in r for r in v["par_niveau"]) and "dso7_rmse_vect_era5_oi" in v

    # scripts/info_capteurs_d2.py : choix de H (sans modele) puis information
    cfgf = tmp_path / "c.yaml"
    c.save(cfgf)
    sys.path.insert(0, str(REPO / "scripts"))
    M = _module("info_capteurs_d2")
    reg = tmp_path / "reglage.json"
    monkeypatch.setattr(sys, "argv", ["x", "reglage", "-c", str(cfgf), "--degradation", str(deg),
                                      "--sortie", str(reg), "--H", "200,1000", "--n-capteurs", "3"])
    assert M.main() == 0
    r = json.loads(reg.read_text(encoding="utf-8"))
    assert r["H_retenu_m"] in (200.0, 1000.0) and set(r["resultats_par_H"]) == {"200", "1000"}
    with pytest.raises(SystemExit):
        M.main()                                      # jamais ecrase
    monkeypatch.setattr(sys, "argv", ["x", "info", "-c", str(cfgf), "--run", str(c.paths.run_dir),
                                      "--degradation", str(deg), "--n-capteurs", "3"])
    assert M.main() == 0
    out = json.loads((c.paths.run_dir / "info_capteurs_d2_validation.json").read_text(encoding="utf-8"))
    assert set(out["conditions"]) == {"propre", "era5_degradee", "era5_degradee_3_capteurs"}
    assert all(np.isfinite(x) for x in out["gain_des_capteurs_rmse_max_uv"].values())


def test_decision_d2_sur_des_bilans_synthetiques(tmp_path, monkeypatch):
    """scripts/decision_d2_national.py : neuf conditions, lignes ERA5 seule et OI seule, code 0 / 3."""
    def run(nom, rmse, oi=None, u_err=0.5):
        r = tmp_path / nom
        (r / "dso_hauteur_sol").mkdir(parents=True)
        v = {"modele": {"rmse_max_uv": rmse, "dso3_rugosite": 0.9, "dso2_conforme": True},
             "era5": {"rmse_max_uv": 2.0, "dso3_rugosite": 0.7}}
        if oi is not None:
            v["era5_oi"] = {"rmse_max_uv": oi, "dso3_rugosite": 0.7}
        for f in ("bilan.json", "bilan_keep25.json"):
            (r / "dso_hauteur_sol" / f).write_text(json.dumps({"verdicts": v}))
        (r / "eval_pcfno3d" / "best_wcont").mkdir(parents=True)
        (r / "eval_pcfno3d" / "best_wcont" / "verdict_national.json").write_text(json.dumps({"dso4_conforme": True}))
        vr = r / "validation_reelle" / "test"
        vr.mkdir(parents=True)
        t = [f"2022-01-0{k} 00:00:00" for k in range(1, 6)]
        p = pd.DataFrame([{"station": s, "temps": x, "u_obs": 1.0, "v_obs": 0.0, "u_modele": 1.0 + u_err,
                           "v_modele": 0.0, "u_era5": 2.0, "v_era5": 0.0, "u_era5_oi": 1.5, "v_era5_oi": 0.0}
                          for s in ("A", "B") for x in t])
        p.to_csv(vr / "stations_paires.csv", index=False)
        q = pd.DataFrame([{"temps": x, "h_agl_m": h, "u_obs": 1.0, "v_obs": 0.0, "u_modele": 1.0 + u_err,
                           "v_modele": 0.0, "u_era5": 2.0, "v_era5": 0.0} for x in t for h in (100.0, 500.0)])
        q.to_csv(vr / "radiosondages_paires.csv", index=False)
        return r
    livre = run("livre", 1.0, u_err=0.6)
    bon, faible = run("d2_bon", 0.9, oi=1.2, u_err=0.4), run("d2_faible", 0.9, oi=0.8, u_err=0.4)
    sys.path.insert(0, str(REPO / "scripts"))
    M = _module("decision_d2_national")
    monkeypatch.setattr(sys, "argv", ["x", "--d2", str(bon), "--livre", str(livre), "--sortie", str(tmp_path / "a")])
    assert M.main() == 0
    d = json.loads((tmp_path / "a" / "decision_d2_national.json").read_text(encoding="utf-8"))
    assert len(d["conditions"]) == 9 and d["le_candidat_remplace_le_modele_livre"]
    assert d["aeroports"]["oi_seule"] == pytest.approx(0.5)
    assert d["meme_verite_v3"]["dso1_rmse_max_uv"]["oi_seule"] == 1.2
    monkeypatch.setattr(sys, "argv", ["x", "--d2", str(faible), "--livre", str(livre), "--sortie", str(tmp_path / "b")])
    assert M.main() == 3                                     # pas mieux que l'OI seule : le livre reste
