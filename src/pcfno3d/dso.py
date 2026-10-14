"""Mesure des DSO sur un jeu d'instants, score DSO, et poids par niveau qui en derivent.

Trois usages :

  - `train.select_by: dso` : choisir le meilleur checkpoint sur le critere de l'objectif
    (pire rapport mesure / seuil), et non sur la loss moyenne ;
  - `pcfno3d dso-weights` : deriver `loss.uv_level_weight` de la RMSE PAR NIVEAU ;
  - le diagnostic : quel niveau, quelle composante empeche le DSO1.

Regle stricte : tout ce qui sert a REGLER l'entrainement (poids, selection) se mesure sur
la VALIDATION. Le test ne sert qu'au verdict final ; regler sur le test, c'est mesurer sa
propre copie.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .config import Config
from .dataset import assemble_input
from .geometry import Geometry
from .normalize import Normalizer


@dataclass
class LevelMetrics:
    """Cumuls par niveau : RMSE et R2 GLOBAUX (sommes de carres), pas des moyennes."""

    nz: int
    n: np.ndarray = None                        # (3, nz) noeuds connus par niveau
    sse: np.ndarray = None                      # (3, nz) somme des carres des erreurs
    s1: np.ndarray = None                       # (3, nz) somme des verites
    s2: np.ndarray = None                       # (3, nz) somme des carres des verites
    mass: list[float] = field(default_factory=list)
    rough: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.n = np.zeros((3, self.nz))
        self.sse = np.zeros((3, self.nz))
        self.s1 = np.zeros((3, self.nz))
        self.s2 = np.zeros((3, self.nz))

    def rmse(self) -> np.ndarray:
        """(3, nz) ; NaN pour un niveau sans noeud."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(self.sse / self.n)

    def r2(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            sst = self.s2 - self.s1 ** 2 / self.n
            return 1.0 - self.sse / np.where(sst > 0, sst, np.nan)


@torch.no_grad()
def measure(net: torch.nn.Module, loader, geom: Geometry, norm: Normalizer,
            device: torch.device, amp_dtype=torch.float32, use_amp: bool = False) -> LevelMetrics:
    """Passe un loader (ANEO3D, evaluation) et cumule les metriques par niveau."""
    from .losses import PhysicsLoss

    rough_fn = PhysicsLoss(geom.cfg, geom, norm)
    net.eval()
    m = LevelMetrics(geom.nz)
    for s_raw, sensor, tfeat, y, _ in loader:
        s_raw, sensor, tfeat, y = (t.to(device) for t in (s_raw, sensor, tfeat, y))
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = net(assemble_input(s_raw, sensor, tfeat, geom, norm))
        u, v, w = norm.denorm(pred.float())
        for k in range(geom.nz):
            for c, p in enumerate((u, v, w)):
                # noeuds valides ET connus (cible partielle : NaN = pas d'observation)
                mk = geom.valid_t[k] & torch.isfinite(y[0, c, k])
                nk = int(mk.sum().item())
                if nk == 0:
                    continue
                m.n[c, k] += nk
                t = y[0, c, k][mk].double()
                m.sse[c, k] += ((p[0, k][mk].double() - t) ** 2).sum().item()
                m.s1[c, k] += t.sum().item()
                m.s2[c, k] += (t ** 2).sum().item()
        r = geom.mass_residual(y[:, 3], u, v, w)
        m.mass.append(torch.sqrt((r[:, geom.mass_mask_t] ** 2).mean()).item())
        m.rough.append(rough_fn.roughness_ratio(u, v, y).item())
    return m


def score(m: LevelMetrics, cfg: Config, geom: Geometry) -> dict:
    """Score DSO : le PIRE rapport mesure / seuil. <= 1 signifie que tout est respecte.

    Meme definition que le BLOC 9b du notebook, pour que les chiffres se comparent.
    """
    levels = geom.eval_levels()
    rmse = m.rmse()[:2, levels]                     # u, v aux niveaux evalues
    r2 = m.r2()[:, levels]
    mass = float(np.mean(m.mass)) if m.mass else float("nan")
    rough = float(np.mean(m.rough)) if m.rough else float("nan")
    d = cfg.dso
    ratios = {
        "dso1_rmse": float(np.nanmax(rmse)) / d.rmse,
        "dso2_masse": mass / d.mass_rms,
        "dso3_rugosite": max(rough - 1.0, 0.0) / (d.roughness_range[1] - 1.0),
        "r2_uvw": (1.0 - float(np.nanmin(r2))) / (1.0 - d.r2_all),
    }
    worst = max(ratios, key=ratios.get)
    k_worst = levels[int(np.nanargmax(np.nanmax(rmse, axis=0)))]
    return {
        "score": float(ratios[worst]),
        "score_sum": float(sum(ratios.values())),
        "limitant": worst,
        "ratios": ratios,
        "rmse_max": float(np.nanmax(rmse)),
        "niveau_le_pire_m": int(geom.z_levels[k_worst]),
        "masse_rms": mass,
        "rugosite": rough,
        "r2_min": float(np.nanmin(r2)),
    }


def level_weights(m: LevelMetrics, cfg: Config, geom: Geometry,
                  floor: float = 0.5, cap: float = 4.0) -> list[float]:
    """Poids de u, v par niveau pour viser le DSO1 (un maximum par niveau).

    w_k = (RMSE_k / seuil)^2 aux niveaux evalues, ou RMSE_k est la pire de u et v ;
    puis bornes [floor, cap] et normalisation pour que la MOYENNE PONDEREE PAR LE NOMBRE
    DE NOEUDS vaille 1 : l'echelle totale de L_data ne bouge pas, donc l'equilibre avec
    les termes de masse et de condition au sol reste celui de la v7.

    Les niveaux hors evaluation (couverture < 5 %) gardent un poids neutre de 1.
    """
    levels = geom.eval_levels()
    rmse = m.rmse()
    w = np.ones(geom.nz)
    for k in levels:
        worst = np.nanmax(rmse[:2, k])
        if np.isfinite(worst):
            w[k] = (worst / cfg.dso.rmse) ** 2
    w = np.clip(w, floor, cap)
    n = m.n[0]
    if n.sum() > 0:
        w = w / (float((w * n).sum()) / float(n.sum()))
    return [round(float(x), 4) for x in w]


def describe_levels(m: LevelMetrics, cfg: Config, geom: Geometry) -> str:
    """Tableau lisible : RMSE par niveau et marge au seuil DSO1."""
    rmse, r2 = m.rmse(), m.r2()
    lines = [" niveau | couverture | RMSE u  RMSE v | marge DSO1 |  R2 u    R2 v"]
    for k in range(geom.nz):
        if m.n[0, k] == 0:
            continue
        cover = geom.cover[k]
        worst = np.nanmax(rmse[:2, k])
        flag = "" if cover >= cfg.dso.min_level_cover else "  (hors DSO)"
        mark = "OK " if worst <= cfg.dso.rmse else "HORS"
        lines.append(f" {int(geom.z_levels[k]):5d} m |  {cover:6.1%}   | {rmse[0, k]:.3f}   "
                     f"{rmse[1, k]:.3f} | {mark} {worst / cfg.dso.rmse:4.2f}  | "
                     f"{r2[0, k]:.4f}  {r2[1, k]:.4f}{flag}")
    return "\n".join(lines)
