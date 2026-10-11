"""La geometrie et les operateurs discrets sont la base de toute la physique du modele :
une erreur ici fausse silencieusement L_mass, L_bc et les DSO."""

from __future__ import annotations

import numpy as np
import torch


def test_grille_lue_depuis_meta(geom):
    assert geom.shape == (geom.nz, geom.ny, geom.nx)
    assert geom.nx == geom.nx_s // geom.stride
    assert geom.dx == geom.dx_s * geom.stride
    assert geom.c_in == 25, "25 canaux d'entree attendus (7+1+6+3+8)"


def test_masques_coherents(geom):
    # un noeud existe au-dessus du relief, jamais en dessous
    assert geom.valid.any() and not geom.valid.all()
    # l'interieur est inclus dans le valide, le masque de masse dans l'interieur
    assert (geom.valid_interior & geom.valid).sum() == geom.valid_interior.sum()
    assert geom.mass_mask.shape == (geom.nz - 1, geom.ny, geom.nx)
    # un seul premier noeud valide par colonne, au maximum
    assert geom.first_valid.sum(axis=0).max() <= 1
    # une colonne qui a des noeuds valides a exactement un premier noeud
    has = geom.valid.any(axis=0)
    assert (geom.first_valid.sum(axis=0)[has] == 1).all()


def test_canaux_statiques_ordre_de_grandeur(geom):
    assert geom.static_2d.shape == (6, geom.ny, geom.nx)
    assert geom.static_3d.shape == (3, geom.nz, geom.ny, geom.nx)
    assert np.abs(geom.static_2d).max() < 50.0, "canaux statiques non normalises"
    assert np.isfinite(geom.static_2d).all() and np.isfinite(geom.static_3d).all()


def test_grad1d_egale_numpy_gradient(geom):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1, geom.nz, geom.ny, geom.nx))
    t = torch.tensor(a, dtype=torch.float64)
    got = geom.grad1d(t, geom.dx, -1).numpy()[0]
    want = np.gradient(a[0], geom.dx, axis=-1)
    assert np.allclose(got, want, atol=1e-10)


def test_residu_de_masse_nul_sur_un_champ_conservatif(geom):
    """rho constant, u et v constants, w lineaire en z : la divergence doit s'annuler."""
    nz, ny, nx = geom.shape
    rho = torch.full((1, nz, ny, nx), 1.2)
    u = torch.full((1, nz, ny, nx), 5.0)
    v = torch.full((1, nz, ny, nx), -3.0)
    w = torch.zeros((1, nz, ny, nx))
    r = geom.mass_residual(rho, u, v, w)
    assert r.abs().max().item() < 1e-6


def test_cache_geometrie_relu(cfg, geom):
    """Le second chargement passe par le .npz et donne les memes tableaux."""
    from pcfno3d.geometry import Geometry

    assert cfg.paths.geometry_cache.exists()
    again = Geometry(cfg, device="cpu")
    assert np.array_equal(again.valid, geom.valid)
    assert np.allclose(again.zt, geom.zt)
    assert np.allclose(again.static_2d, geom.static_2d)
