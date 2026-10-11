"""BLOC 8 — loss physique et metriques.

L = L_data + lambda_grad . L_grad + phi(t) . [ lambda_mass . L_mass / mass_ref
                                               + lambda_bc . L_bc
                                               + lambda_smooth . L_smooth ]

L_data   MSE normalisee sur les noeuds valides (w pondere par niveau)
L_grad   erreur relative sur les differences entre voisins : empeche un champ trop lisse
L_mass   residu de continuite en unites physiques, meme operateur que le generateur
L_bc     w = u.dz/dx + v.dz/dy au premier noeud au-dessus du sol
phi(t)   montee progressive de la physique sur les premieres epoques
"""

from __future__ import annotations

import torch

from .config import Config
from .geometry import Geometry
from .normalize import Normalizer


class PhysicsLoss:
    """Porte la geometrie, les statistiques et les poids : appelable comme une fonction."""

    def __init__(self, cfg: Config, geom: Geometry, norm: Normalizer) -> None:
        self.cfg = cfg
        self.geom = geom
        self.norm = norm
        # poids de u, v par niveau (DSO1 est un maximum par niveau, pas une moyenne)
        weights = cfg.loss.uv_level_weight
        if weights is not None:
            if len(weights) != geom.nz:
                raise ValueError(f"loss.uv_level_weight : {len(weights)} valeurs pour "
                                 f"{geom.nz} niveaux")
            self.uv_level_w = torch.tensor(weights, dtype=torch.float32, device=geom.device)
        else:
            self.uv_level_w = None

    # ------------------------------------------------------------------ loss
    def __call__(self, pred: torch.Tensor, y: torch.Tensor,
                 phys_w: float) -> tuple[torch.Tensor, dict[str, float]]:
        """pred : (B, 3, nz, ny, nx) normalise ; y : (B, 4, nz, ny, nx) physique (u, v, w, rho)."""
        cfg, geom, norm = self.cfg.loss, self.geom, self.norm
        pred = pred.float()
        y_n = norm.normalize_target(y)

        err = (pred - y_n) ** 2
        uv = err[:, :2]
        if self.uv_level_w is not None:
            # niveaux hors seuil DSO1 plus lourds (voir LossCfg.uv_level_weight)
            uv = uv * self.uv_level_w[None, None, :, None, None]
        # w est tres faible en altitude : son erreur est remontee par le poids du niveau
        err = torch.cat([uv,
                         err[:, 2:3] * cfg.w_weight * norm.w_level_w[None, None, :, None, None]],
                        dim=1)
        l_data = err[:, :, geom.valid_t].mean()

        u, v, w = norm.denorm(pred)
        rho = y[:, 3]
        r = geom.mass_residual(rho, u, v, w)
        l_mass = (r[:, geom.mass_mask_t] ** 2).mean()

        # condition cinematique au sol : le vent suit le relief, il ne le traverse pas
        w_bc = u * geom.dzt_dx_t + v * geom.dzt_dy_t
        l_bc = (((w - w_bc) / norm.y_sd[2])[:, geom.first_valid_t] ** 2).mean()

        l_grad = torch.zeros((), device=pred.device)
        if cfg.lambda_grad > 0:
            gxp, gxt = pred[..., 1:] - pred[..., :-1], y_n[..., 1:] - y_n[..., :-1]
            gyp, gyt = pred[..., 1:, :] - pred[..., :-1, :], y_n[..., 1:, :] - y_n[..., :-1, :]
            num = ((gxp - gxt) ** 2)[:, :, geom.grad_mask_x_t].sum() \
                + ((gyp - gyt) ** 2)[:, :, geom.grad_mask_y_t].sum()
            den = (gxt ** 2)[:, :, geom.grad_mask_x_t].sum() \
                + (gyt ** 2)[:, :, geom.grad_mask_y_t].sum() + 1e-12
            l_grad = num / den

        l_smooth = torch.zeros((), device=pred.device)
        if cfg.lambda_smooth > 0:
            l_smooth = ((pred[:, :2, :, :, 1:] - pred[:, :2, :, :, :-1]) ** 2).mean() + \
                       ((pred[:, :2, :, 1:, :] - pred[:, :2, :, :-1, :]) ** 2).mean()

        total = l_data + cfg.lambda_grad * l_grad + phys_w * (
            cfg.lambda_mass * l_mass / cfg.mass_ref
            + cfg.lambda_bc * l_bc
            + cfg.lambda_smooth * l_smooth
        )
        return total, {"data": l_data.item(), "mass": l_mass.item(), "bc": l_bc.item(),
                       "smooth": l_smooth.item(), "grad": l_grad.item()}

    def val_score(self, parts: dict[str, float]) -> float:
        """Score de selection du meilleur modele (meme formule que le BLOC 9)."""
        cfg = self.cfg.loss
        return parts["data"] + cfg.lambda_mass * parts["mass"] / cfg.mass_ref \
            + cfg.lambda_bc * parts["bc"]

    # ------------------------------------------------------------------ metriques
    def r2_per_component(self, pred: torch.Tensor, y: torch.Tensor) -> list[float]:
        """R2 par composante sur les noeuds valides, en unites physiques."""
        valid = self.geom.valid_t
        out = []
        for c, p in enumerate(self.norm.denorm(pred.float())):
            t = y[:, c][:, valid]
            q = p[:, valid]
            ss_res = ((q - t) ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            out.append((1 - ss_res / (ss_tot + 1e-12)).item())
        return out

    def roughness(self, a: torch.Tensor) -> torch.Tensor:
        """DSO3 — rugosite |dA/dx| + |dA/dy| moyennee sur les paires de voisins valides."""
        gx = (a[..., :, 1:] - a[..., :, :-1]).abs()[:, self.geom.grad_mask_x_t]
        gy = (a[..., 1:, :] - a[..., :-1, :]).abs()[:, self.geom.grad_mask_y_t]
        return gx.mean() + gy.mean()

    def roughness_ratio(self, u: torch.Tensor, v: torch.Tensor,
                        y: torch.Tensor) -> torch.Tensor:
        """Rugosite de la prediction / rugosite de la verite (1 = ni lisse ni bruite)."""
        return (self.roughness(u) + self.roughness(v)) \
            / (self.roughness(y[:, 0]) + self.roughness(y[:, 1]))


def physics_weight(epoch: int, warmup_epochs: int) -> float:
    """phi(t) : 0 -> 1 sur les `warmup_epochs` premieres epoques."""
    return min(1.0, (epoch + 1) / max(warmup_epochs, 1))
