"""BLOC 5 — statistiques de normalisation, calculees sur les jours d'entrainement seuls.

Le fichier `norm_stats.json` fait partie du modele : le service d'inference en a besoin
pour normaliser l'entree et denormaliser la sortie. Il est donc ecrit dans `run_dir`, a
cote des checkpoints, et jamais recalcule si present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .geometry import N_SURF, SURF_VARS, Geometry

STATS_SAMPLES = 200      # instants d'entrainement tires pour estimer les statistiques


class Normalizer:
    """Moyennes / ecarts-types de la surface et de la cible, plus le poids de w par niveau."""

    def __init__(self, stats: dict, geom: Geometry, w_level_max: float) -> None:
        self.stats = stats
        d = geom.device
        self.surf_mu = torch.tensor(stats["surf_mu"], dtype=torch.float32, device=d)[:, None, None]
        self.surf_sd = torch.tensor(stats["surf_sd"], dtype=torch.float32, device=d)[:, None, None]
        self.y_mu = torch.tensor(stats["y_mu"], dtype=torch.float32, device=d)
        self.y_sd = torch.tensor(stats["y_sd"], dtype=torch.float32, device=d)
        self.w_level_sd = torch.tensor(stats["w_level_sd"], dtype=torch.float32, device=d)
        # w est tres faible en altitude : sans ce poids la loss l'ignore et son R2 chute
        self.w_level_w = torch.clamp((self.y_sd[2] / self.w_level_sd) ** 2, 1.0, w_level_max)

    # ------------------------------------------------------------------ chargement
    @classmethod
    def load_or_compute(cls, cfg: Config, geom: Geometry, index: pd.DataFrame | None = None,
                        verbose: bool = True) -> "Normalizer":
        path = cfg.paths.norm_stats
        stats = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        missing = [k for k in ("surf_mu", "surf_sd", "y_mu", "y_sd", "w_level_sd")
                   if k not in stats]
        if missing:
            if index is None:
                from .cache import load_index
                index = load_index(cfg)
            stats = compute_stats(cfg, geom, index, stats, verbose=verbose)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            if verbose:
                print(f"statistiques de normalisation ecrites dans {path}")
        return cls(stats, geom, cfg.loss.w_level_max)

    @classmethod
    def from_file(cls, path: str | Path, geom: Geometry, w_level_max: float) -> "Normalizer":
        stats = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(stats, geom, w_level_max)

    # ------------------------------------------------------------------ usage
    def normalize_surface(self, s_raw: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        """z-score des 7 variables, puis 0 la ou le capteur est masque."""
        return (s_raw - self.surf_mu) / self.surf_sd * sensor[:, None]

    def denorm(self, pred: torch.Tensor) -> list[torch.Tensor]:
        """(B, 3, ...) normalise -> [u, v, w] en m/s."""
        return [pred[:, c] * self.y_sd[c] + self.y_mu[c] for c in range(3)]

    def normalize_target(self, y: torch.Tensor) -> torch.Tensor:
        """(B, >=3, nz, ny, nx) en m/s -> u, v, w normalises."""
        return (y[:, :3] - self.y_mu[None, :, None, None, None]) \
            / self.y_sd[None, :, None, None, None]

    def to(self, device: torch.device | str) -> "Normalizer":
        for name in ("surf_mu", "surf_sd", "y_mu", "y_sd", "w_level_sd", "w_level_w"):
            setattr(self, name, getattr(self, name).to(device))
        return self


def compute_stats(cfg: Config, geom: Geometry, index: pd.DataFrame, stats: dict | None = None,
                  verbose: bool = True) -> dict:
    """Moyennes et ecarts-types sur un echantillon d'instants d'ENTRAINEMENT (pas de fuite)."""
    stats = dict(stats or {})
    cache = cfg.paths.cache_dir
    train = index[index["split"] == "train"]["stem"]
    if len(train) == 0:
        raise ValueError("Aucun instant d'entrainement dans l'index : cache incomplet ?")
    sample = train.sample(min(STATS_SAMPLES, len(train)), random_state=0)

    if any(k not in stats for k in ("surf_mu", "surf_sd", "y_mu", "y_sd")):
        s_sum = np.zeros(N_SURF); s_sq = np.zeros(N_SURF); s_n = 0
        y_sum = np.zeros(3); y_sq = np.zeros(3); y_n = 0
        for stem in sample:
            S = np.load(cache / f"{stem}_S.npy").astype(np.float64)
            m = S[7] > 0.5                       # seulement la ou un capteur existe
            s_sum += S[:N_SURF, m].sum(axis=1)
            s_sq += (S[:N_SURF, m] ** 2).sum(axis=1)
            s_n += int(m.sum())
            Y = np.load(cache / f"{stem}_Y.npy").astype(np.float64)
            y_sum += Y[:3, geom.valid].sum(axis=1)
            y_sq += (Y[:3, geom.valid] ** 2).sum(axis=1)
            y_n += int(geom.valid.sum())
        s_mu = s_sum / s_n
        y_mu = y_sum / y_n
        stats.update({
            "surf_mu": s_mu.tolist(),
            "surf_sd": np.sqrt(np.maximum(s_sq / s_n - s_mu ** 2, 1e-12)).tolist(),
            "y_mu": y_mu.tolist(),
            "y_sd": np.sqrt(np.maximum(y_sq / y_n - y_mu ** 2, 1e-12)).tolist(),
            "surf_vars": SURF_VARS,
        })

    if "w_level_sd" not in stats:
        nz = geom.nz
        wl_sum = np.zeros(nz); wl_sq = np.zeros(nz); wl_n = np.zeros(nz)
        for stem in sample:
            Yw = np.load(cache / f"{stem}_Y.npy")[2].astype(np.float64)
            for k in range(nz):
                vals = Yw[k][geom.valid[k]]
                wl_sum[k] += vals.sum(); wl_sq[k] += (vals ** 2).sum(); wl_n[k] += vals.size
        wl_n = np.maximum(wl_n, 1)
        stats["w_level_sd"] = np.sqrt(
            np.maximum(wl_sq / wl_n - (wl_sum / wl_n) ** 2, 1e-12)).tolist()

    stats["n_instants_echantillon"] = int(len(sample))
    if verbose:
        print(f"cible mu = {np.round(stats['y_mu'], 3)} sd = {np.round(stats['y_sd'], 3)}")
    return stats
