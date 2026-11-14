"""inference.w_beta (regle_w_servi_v4) : w servi = w_continuite + beta (w_reseau - w_continuite)."""
import pytest
import torch


def test_w_servi_melange(geom):
    inf = geom.cfg.inference
    ancien = (inf.w_continuite, inf.w_beta)
    g = torch.Generator().manual_seed(0)
    forme = (1, geom.nz, *geom.shape[1:])
    u, v, w = (torch.randn(forme, generator=g) * geom.valid_t for _ in range(3))
    try:
        inf.w_continuite = False
        assert torch.equal(geom.w_servi(u, v, w), w)               # w du reseau, comme la v7
        inf.w_continuite, inf.w_beta = True, 0.0
        wc = geom.w_continuite(u, v)
        assert torch.equal(geom.w_servi(u, v, w), wc)              # defaut inchange
        inf.w_beta = 1.0
        assert torch.allclose(geom.w_servi(u, v, w), w)
        inf.w_beta = 0.75
        w75 = geom.w_servi(u, v, w)
        assert torch.allclose(w75, wc + 0.75 * (w - wc))
        # residu de masse lineaire en w : 0,75 x celui du w du reseau (a rho standard)
        rho = geom.rho_standard().expand_as(u)
        r75 = geom.mass_residual(rho, u, v, w75)[:, geom.mass_mask_t]
        r1 = geom.mass_residual(rho, u, v, w)[:, geom.mass_mask_t]
        r0 = geom.mass_residual(rho, u, v, wc)[:, geom.mass_mask_t]
        assert torch.allclose(r75 - r0, 0.75 * (r1 - r0), atol=1e-5)
    finally:
        inf.w_continuite, inf.w_beta = ancien


def test_w_beta_borne(cfg):
    from pcfno3d.config import InferenceCfg
    with pytest.raises(ValueError):
        InferenceCfg(w_beta=1.5)
    assert InferenceCfg().w_beta == 0.0
