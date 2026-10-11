"""Forme de la sortie, differentiabilite, et comportement attendu de chaque terme de loss."""

from __future__ import annotations

import numpy as np
import torch

from pcfno3d.dataset import ANEO3D, assemble_input
from pcfno3d.losses import PhysicsLoss, physics_weight
from pcfno3d.model import build_model, count_parameters


def _batch(cfg, geom, index):
    """Un instant du cache, mis en forme comme le DataLoader le ferait."""
    train = index[index["split"] == "train"]
    ds = ANEO3D(train["stem"].values, train["hour"].values, train["doy"].values,
                geom, train=False, seed=0)
    s_raw, sensor, tfeat, y, _ = ds[0]
    return (s_raw[None], sensor[None], tfeat[None], y[None])


def test_sortie_du_modele_a_la_bonne_forme(cfg, geom, index, norm):
    s_raw, sensor, tfeat, y = _batch(cfg, geom, index)
    x = assemble_input(s_raw, sensor, tfeat, geom, norm)
    assert x.shape == (1, geom.c_in, *geom.shape)
    model = build_model(cfg, geom.c_in)
    pred = model(x)
    assert pred.shape == (1, 3, *geom.shape)
    assert torch.isfinite(pred).all()
    assert count_parameters(model) > 0


def test_la_loss_est_differentiable(cfg, geom, index, norm):
    s_raw, sensor, tfeat, y = _batch(cfg, geom, index)
    model = build_model(cfg, geom.c_in)
    loss_fn = PhysicsLoss(cfg, geom, norm)
    pred = model(assemble_input(s_raw, sensor, tfeat, geom, norm))
    total, parts = loss_fn(pred, y, phys_w=1.0)
    total.backward()
    assert torch.isfinite(total)
    assert set(parts) == {"data", "mass", "bc", "smooth", "grad"}
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_loss_nulle_sur_une_prediction_parfaite(cfg, geom, index, norm):
    """Avec pred = verite normalisee, L_data et L_grad doivent s'annuler."""
    _, _, _, y = _batch(cfg, geom, index)
    loss_fn = PhysicsLoss(cfg, geom, norm)
    pred = norm.normalize_target(y)
    _, parts = loss_fn(pred, y, phys_w=0.0)
    assert parts["data"] < 1e-8
    assert parts["grad"] < 1e-6


def test_r2_parfait_sur_une_prediction_parfaite(cfg, geom, index, norm):
    _, _, _, y = _batch(cfg, geom, index)
    loss_fn = PhysicsLoss(cfg, geom, norm)
    r2 = loss_fn.r2_per_component(norm.normalize_target(y), y)
    assert all(v > 0.999 for v in r2), r2


def test_rugosite_relative_vaut_un_pour_la_verite(cfg, geom, index, norm):
    _, _, _, y = _batch(cfg, geom, index)
    loss_fn = PhysicsLoss(cfg, geom, norm)
    ratio = loss_fn.roughness_ratio(y[:, 0], y[:, 1], y).item()
    assert abs(ratio - 1.0) < 1e-5


def test_poids_de_w_par_niveau_borne(cfg, norm):
    w = norm.w_level_w
    assert float(w.min()) >= 1.0
    assert float(w.max()) <= cfg.loss.w_level_max + 1e-6


def test_montee_progressive_de_la_physique():
    assert physics_weight(0, 10) == 0.1
    assert physics_weight(9, 10) == 1.0
    assert physics_weight(50, 10) == 1.0
    assert physics_weight(0, 0) == 1.0        # pas de division par zero


def test_normalisation_est_inversible(cfg, geom, index, norm):
    _, _, _, y = _batch(cfg, geom, index)
    back = torch.stack(norm.denorm(norm.normalize_target(y)), dim=1)
    assert torch.allclose(back, y[:, :3], atol=1e-3)


def test_entree_masquee_la_ou_il_n_y_a_pas_de_capteur(cfg, geom, index, norm):
    s_raw, sensor, tfeat, _ = _batch(cfg, geom, index)
    x = assemble_input(s_raw, sensor, tfeat, geom, norm)
    hole = sensor[0] < 0.5
    if hole.any():
        # les 7 canaux de mesure sont a zero la ou le capteur est masque
        assert np.allclose(x[0, :7, 0][:, hole].numpy(), 0.0)
