"""BLOC 6 (2e partie) — Dataset, DataLoaders et assemblage des 25 canaux d'entree."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .features import time_geo_features
from .geometry import N_SURF, Geometry
from .normalize import Normalizer


class ANEO3D(Dataset):
    """Un exemple = un instant : surface brute, capteurs visibles, canaux temps, cible.

    En entrainement, l'instant et la fraction de capteurs visibles sont tires au hasard
    (robustesse a un reseau incomplet). En evaluation, tout est fixe par la graine, donc
    deux evaluations du meme modele sont comparables.
    """

    def __init__(self, stems, hours, doys, geom: Geometry, train: bool,
                 n_samples: int | None = None, seed: int = 0,
                 keep_eval: float | None = None) -> None:
        self.stems = list(stems)
        self.hours = list(hours)
        self.doys = list(doys)
        self.geom = geom
        self.cache_dir = geom.cfg.paths.cache_dir
        self.train = train
        self.n_samples = n_samples
        self.seed = seed
        self.keep_train = geom.cfg.mask.keep_train
        self.p_full = geom.cfg.mask.p_full
        self.keep_eval = geom.cfg.mask.keep_eval if keep_eval is None else keep_eval

    def __len__(self) -> int:
        return self.n_samples if self.n_samples else len(self.stems)

    def __getitem__(self, i: int):
        if self.train:
            j = int(np.random.randint(len(self.stems)))
            rng = np.random.default_rng()
            # p_full : une part des exemples garde TOUS ses capteurs, le cas de production,
            # qui n'est sinon qu'un bord de la distribution uniforme (voir MaskCfg.p_full)
            keep = 1.0 if rng.random() < self.p_full else rng.uniform(*self.keep_train)
        else:
            j = i
            rng = np.random.default_rng(self.seed + i)   # memes capteurs masques a chaque fois
            keep = self.keep_eval
        stem = self.stems[j]
        S = np.load(self.cache_dir / f"{stem}_S.npy")
        Y = np.load(self.cache_dir / f"{stem}_Y.npy")
        sensor = (S[7] > 0.5) & (rng.random(S[7].shape) < keep)
        tfeat = time_geo_features(S, sensor, self.hours[j], self.doys[j], self.geom)
        return (torch.from_numpy(S[:N_SURF]), torch.from_numpy(sensor.astype(np.float32)),
                torch.from_numpy(tfeat), torch.from_numpy(Y), j)


def assemble_input(s_raw: torch.Tensor, sensor: torch.Tensor, tfeat: torch.Tensor,
                   geom: Geometry, norm: Normalizer) -> torch.Tensor:
    """Construit l'entree (B, c_in, nz, ny, nx) sur le device de la geometrie.

    7 mesures normalisees + 1 carte de disponibilite + 6 canaux de relief, recopies sur
    les nz niveaux ; + 3 canaux de geometrie par niveau ; + 8 canaux temps / physique.
    """
    B = s_raw.shape[0]
    nz, ny, nx = geom.shape
    s = norm.normalize_surface(s_raw, sensor)
    s2d = torch.cat([s, sensor[:, None], geom.static_2d_t.expand(B, -1, -1, -1)], dim=1)
    x2d = s2d[:, :, None].expand(-1, -1, nz, -1, -1)
    x3d = geom.static_3d_t.expand(B, -1, -1, -1, -1)
    xt = tfeat[:, :, None, None, None].expand(-1, -1, nz, ny, nx)
    return torch.cat([x2d, x3d, xt], dim=1)


@dataclass
class Splits:
    """Les trois ensembles d'instants, plus la selection fixe de validation."""

    index: pd.DataFrame
    train: tuple[np.ndarray, np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray, np.ndarray]
    val_selection: np.ndarray
    stamp_of: dict[str, str]

    def summary(self) -> str:
        return (f"instants : train={len(self.train[0])} | "
                f"val={len(self.val_selection)} (sur {len(self.val[0])}) | "
                f"test={len(self.test[0])}")


def is_synoptic(stamp) -> bool:
    """Instant synoptique : heure pleine, multiple de 3 h (les DSO portent sur ceux-la)."""
    t = pd.Timestamp(stamp)
    return t.minute == 0 and t.second == 0 and t.hour % 3 == 0


def make_splits(cfg: Config, index: pd.DataFrame) -> Splits:
    def subset(split: str):
        d = index[index["split"] == split]
        return d["stem"].values, d["hour"].values, d["doy"].values

    train, val, test = subset("train"), subset("val"), subset("test")

    # validation : instants synoptiques repartis regulierement sur les jours de validation
    stamps = index.set_index("stem").loc[val[0], "timestamp"] if len(val[0]) else []
    syn = np.array([is_synoptic(x) for x in stamps], dtype=bool) if len(val[0]) \
        else np.zeros(0, dtype=bool)
    candidates = np.nonzero(syn)[0] if syn.any() else np.arange(len(val[0]))
    n = min(cfg.train.val_samples, len(candidates))
    val_selection = candidates[np.unique(np.linspace(0, len(candidates) - 1, n).astype(int))] \
        if n > 0 else np.zeros(0, dtype=int)

    return Splits(index=index, train=train, val=val, test=test, val_selection=val_selection,
                  stamp_of=dict(zip(index["stem"], index["timestamp"])))


def make_loaders(cfg: Config, geom: Geometry, splits: Splits) -> dict[str, DataLoader]:
    """Les trois DataLoaders de l'entrainement (BLOC 6)."""
    nw = cfg.train.num_workers
    kw = dict(num_workers=nw, pin_memory=(geom.device.type == "cuda"),
              persistent_workers=nw > 0)
    tr_s, tr_h, tr_d = splits.train
    va_s, va_h, va_d = splits.val
    te_s, te_h, te_d = splits.test
    sel = splits.val_selection

    train_ds = ANEO3D(tr_s, tr_h, tr_d, geom, train=True,
                      n_samples=cfg.train.steps_per_epoch * cfg.train.batch_size)
    val_ds = ANEO3D(va_s[sel], va_h[sel], va_d[sel], geom, train=False, seed=1000)
    test_ds = ANEO3D(te_s, te_h, te_d, geom, train=False, seed=2000)
    return {
        # pas de shuffle : le tirage de l'instant est deja aleatoire dans __getitem__
        "train": DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=False, **kw),
        "val": DataLoader(val_ds, batch_size=1, shuffle=False, **kw),
        "test": DataLoader(test_ds, batch_size=1, shuffle=False, **kw),
    }


def eval_loader(cfg: Config, geom: Geometry, splits: Splits, idx: np.ndarray, seed: int,
                keep: float | None = None) -> tuple[DataLoader, np.ndarray]:
    """Un loader deterministe sur un sous-ensemble du test (DSO1, episodes, robustesse)."""
    te_s, te_h, te_d = splits.test
    ds = ANEO3D(te_s[idx], te_h[idx], te_d[idx], geom, train=False, seed=seed, keep_eval=keep)
    return DataLoader(ds, batch_size=1), te_s[idx]
