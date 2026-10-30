"""DE2 — reproductibilite : meme entree + meme version -> meme sortie.

Tolerance documentee (docs/REPRODUCTIBILITE.md) : sur CPU, a version de torch, checkpoint,
norm_stats et nombre de threads identiques, la reconstruction est IDENTIQUE BIT A BIT
(ecart max 0,0 m/s). Ces tests verifient exactement cela, sur le mini-jeu de donnees :

  - deux modeles charges separement depuis le meme checkpoint donnent le meme volume ;
  - le meme modele appele deux fois donne le meme volume (aucun etat cache) ;
  - le masquage des capteurs (`keep`) et la moyenne de passes (`tta`) sont fixes par la
    graine : meme graine -> memes capteurs, graine differente -> autres capteurs ;
  - le jeu d'evaluation masque les memes capteurs a chaque lecture.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

# Ecart maximal admis entre deux reconstructions du meme instant, en m/s. 0 : egalite bit a
# bit, mesuree sur CPU (torch 2.2, mini-jeu). Ne pas relacher sans mesurer pourquoi.
TOLERANCE_M_S = 0.0


def _trained(cfg):
    from pcfno3d.train import Trainer

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg.paths.ckpt_best


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def _ecart_max(a, b) -> float:
    """Ecart max sur u, v, w ; les NaN (sous le relief) doivent tomber aux memes noeuds."""
    pire = 0.0
    for x, y in ((a.u, b.u), (a.v, b.v), (a.w, b.w)):
        assert np.array_equal(np.isnan(x), np.isnan(y))
        ok = ~np.isnan(x)
        pire = max(pire, float(np.max(np.abs(x[ok].astype(np.float64)
                                             - y[ok].astype(np.float64)), initial=0.0)))
    return pire


def _identiques(a, b) -> None:
    ecart = _ecart_max(a, b)
    assert ecart <= TOLERANCE_M_S, f"ecart max {ecart:.3e} m/s > {TOLERANCE_M_S} m/s"
    if TOLERANCE_M_S == 0.0:
        for x, y in ((a.u, b.u), (a.v, b.v), (a.w, b.w)):
            assert np.array_equal(x, y, equal_nan=True)
    assert a.n_sensors == b.n_sensors


def _stem_test(index) -> str:
    return index[index["split"] == "test"]["stem"].iloc[0]


# ---------------------------------------------------------------------------- inference
def test_deux_chargements_du_meme_checkpoint_donnent_la_meme_sortie(cfg, index, norm):
    from pcfno3d.inference import WindFieldModel

    ck = _trained(cfg)
    empreinte = _sha256(ck)
    stem = _stem_test(index)

    a = WindFieldModel(cfg, device="cpu", verbose=False).predict_stem(stem, keep=0.5, seed=7)
    b = WindFieldModel(cfg, device="cpu", verbose=False).predict_stem(stem, keep=0.5, seed=7)
    _identiques(a, b)
    assert np.isfinite(a.u).any()                       # un vrai volume, pas que des NaN
    assert _sha256(ck) == empreinte                     # le chargement ne modifie pas le fichier


@pytest.mark.parametrize("keep", [1.0, 0.5])
def test_meme_modele_appele_deux_fois_donne_la_meme_sortie(cfg, index, norm, keep):
    from pcfno3d.inference import WindFieldModel

    _trained(cfg)
    wf = WindFieldModel(cfg, device="cpu", verbose=False)
    stem = _stem_test(index)
    _identiques(wf.predict_stem(stem, keep=keep, seed=3), wf.predict_stem(stem, keep=keep, seed=3))


def test_moyenne_de_passes_reproductible_a_graine_fixee(cfg, index, norm):
    from pcfno3d.inference import WindFieldModel

    _trained(cfg)
    wf = WindFieldModel(cfg, device="cpu", verbose=False)
    S, hour, doy, _ = wf.surface_from_cache(_stem_test(index))
    a = wf.predict(S, hour, doy, seed=11, tta=3, tta_keep=0.6)
    b = wf.predict(S, hour, doy, seed=11, tta=3, tta_keep=0.6)
    _identiques(a, b)
    c = wf.predict(S, hour, doy, seed=12, tta=3, tta_keep=0.6)
    assert _ecart_max(a, c) > 0.0                       # la graine change bien les passes


# ---------------------------------------------------------------------------- masquage
def test_masquage_des_capteurs_fixe_par_la_graine(cfg, index, norm, monkeypatch):
    """Les capteurs gardes avec `keep` ne dependent que de la graine (defaut 0)."""
    import pcfno3d.inference as inf

    _trained(cfg)
    wf = inf.WindFieldModel(cfg, device="cpu", verbose=False)
    S, hour, doy, _ = wf.surface_from_cache(_stem_test(index))

    masques = []
    original = inf.time_geo_features

    def espion(S_, sensor, h, d, g):
        masques.append(np.array(sensor, copy=True))
        return original(S_, sensor, h, d, g)

    monkeypatch.setattr(inf, "time_geo_features", espion)
    wf.predict(S, hour, doy, keep=0.5, seed=5)
    wf.predict(S, hour, doy, keep=0.5, seed=5)
    wf.predict(S, hour, doy, keep=0.5, seed=6)
    wf.predict(S, hour, doy, keep=0.5)                  # graine par defaut : 0
    wf.predict(S, hour, doy, keep=0.5, seed=0)
    assert np.array_equal(masques[0], masques[1])
    assert not np.array_equal(masques[0], masques[2])
    assert np.array_equal(masques[3], masques[4])
    vus = int(masques[0].sum())
    assert 0 < vus < int((S[7] > 0.5).sum())            # keep = 0,5 masque vraiment


def test_jeu_d_evaluation_masque_toujours_les_memes_capteurs(cfg, geom, index):
    """Deux lectures (et deux jeux de meme graine) voient les memes capteurs."""
    from pcfno3d.dataset import ANEO3D

    te = index[index["split"] == "test"]
    args = (te["stem"].values, te["hour"].values, te["doy"].values, geom)
    d1 = ANEO3D(*args, train=False, seed=2000)
    d2 = ANEO3D(*args, train=False, seed=2000)
    d3 = ANEO3D(*args, train=False, seed=2001)
    for i in range(min(len(d1), 3)):
        s1, s2, s3 = d1[i][1].numpy(), d2[i][1].numpy(), d3[i][1].numpy()
        assert np.array_equal(s1, d1[i][1].numpy())
        assert np.array_equal(s1, s2)
        assert not np.array_equal(s1, s3)


def test_manifeste_du_cache(cfg, index):
    """BF15 / L2 : le cache porte l'empreinte de ce qui l'a construit ; un index modifie se voit."""
    from pcfno3d.cache import ecrire_manifeste, verifier_manifeste
    f = ecrire_manifeste(cfg)
    m = json.loads(f.read_text(encoding="utf-8"))
    assert m["instants"] == len(index) and m["meta_json_sha256"] and m["index_csv_sha256"]
    assert verifier_manifeste(cfg) == []
    sauve = cfg.paths.index_csv.read_bytes()
    try:
        cfg.paths.index_csv.write_bytes(sauve + b"\n")
        assert any(e.startswith("index_csv_sha256") for e in verifier_manifeste(cfg))
    finally:
        cfg.paths.index_csv.write_bytes(sauve)
    assert verifier_manifeste(cfg) == []
