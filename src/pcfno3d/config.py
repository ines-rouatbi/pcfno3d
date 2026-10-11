"""BLOC 0/1 — configuration : chemins, hyperparametres, seuils DSO.

Toutes les constantes qui etaient des variables globales du notebook sont ici, dans des
dataclasses chargeables depuis un YAML (`configs/*.yaml`). Rien n'est lu sur le disque a
l'import : `Config.load()` ne touche qu'au fichier de configuration.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _resolve(path: str | Path) -> Path:
    """Chemin absolu tel quel ; chemin relatif resolu depuis la racine du depot."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


# --------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------
@dataclass
class Paths:
    """Les trois dossiers du pipeline.

    data_dir  : dataset brut du generateur (aneo_grid_points.csv, meta.json, CSV journaliers)
    cache_dir : tenseurs .npy par instant (BLOC 3) — disque local rapide, regenerable
    run_dir   : checkpoints, statistiques, figures, evaluations — a conserver
    """

    data_dir: Path = REPO_ROOT / "aneo_dataset_3d_v2"
    cache_dir: Path = REPO_ROOT / "artifacts" / "cache"
    run_dir: Path = REPO_ROOT / "artifacts" / "runs" / "v7"
    # copie de securite du cache (Google Drive sur Colab) ; None = pas de copie
    cache_mirror: Path | None = None

    def __post_init__(self) -> None:
        # un chemin relatif dans un YAML est relatif a la racine du depot, pas au cwd :
        # les commandes marchent donc depuis n'importe quel dossier
        self.data_dir = _resolve(self.data_dir)
        self.cache_dir = _resolve(self.cache_dir)
        self.run_dir = _resolve(self.run_dir)
        if self.cache_mirror is not None:
            self.cache_mirror = _resolve(self.cache_mirror)

    def mkdirs(self) -> None:
        for d in (self.cache_dir, self.run_dir, self.run_dir / "figures"):
            d.mkdir(parents=True, exist_ok=True)

    # fichiers nommes une seule fois, pour que tous les modules pointent au meme endroit
    @property
    def index_csv(self) -> Path:
        return self.cache_dir / "index.csv"

    @property
    def norm_stats(self) -> Path:
        return self.run_dir / "norm_stats.json"

    @property
    def geometry_cache(self) -> Path:
        return self.cache_dir / "geometry_v1.npz"

    @property
    def ckpt_best(self) -> Path:
        return self.run_dir / "best.pt"

    @property
    def ckpt_best_dso(self) -> Path:
        return self.run_dir / "best_dso_v7.pt"

    @property
    def ckpt_last(self) -> Path:
        return self.run_dir / "last.pt"

    @property
    def history(self) -> Path:
        return self.run_dir / "history.json"

    def checkpoint_for_inference(self) -> Path:
        """Le modele retenu : raffinement DSO s'il existe, sinon le meilleur de l'entrainement."""
        for p in (self.ckpt_best_dso, self.ckpt_best):
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Aucun checkpoint dans {self.run_dir} (attendu best_dso_v7.pt ou best.pt)"
        )


@dataclass
class ModelCfg:
    """BLOC 7 — architecture."""

    width: int = 48                             # canaux internes
    modes: tuple[int, int, int] = (6, 32, 32)   # modes de Fourier conserves (z, y, x)
    n_layers: int = 4
    pad: tuple[int, int, int] = (4, 8, 8)       # rembourrage avant FFT (z, y, x)
    proj_hidden: int = 128
    local_conv: bool = True                     # branche convolutive locale 3x3x3
    pool_surface: bool = True                   # moyenne des capteurs par bloc stride x stride

    def __post_init__(self) -> None:
        self.modes = tuple(int(m) for m in self.modes)      # type: ignore[assignment]
        self.pad = tuple(int(p) for p in self.pad)          # type: ignore[assignment]


@dataclass
class TrainCfg:
    """BLOC 9 — entrainement."""

    epochs: int = 140
    steps_per_epoch: int = 400
    val_samples: int = 64
    batch_size: int = 1
    accum_steps: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_phys_epochs: int = 10
    num_workers: int = 4
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    save_every: int = 5
    seed: int = 42


@dataclass
class LossCfg:
    """BLOC 8 — ponderation des termes de la loss."""

    lambda_mass: float = 1.0
    mass_ref: float = 5e-7        # echelle du terme de masse (L_mass / mass_ref)
    lambda_bc: float = 0.1
    lambda_smooth: float = 0.0    # donnees deja lisses : desactive
    lambda_grad: float = 0.15
    w_weight: float = 1.0
    w_level_max: float = 25.0     # plafond du poids de w par niveau


@dataclass
class MaskCfg:
    """Masquage des capteurs d'entree (robustesse a un reseau incomplet)."""

    keep_train: tuple[float, float] = (0.2, 1.0)
    keep_eval: float = 0.5

    def __post_init__(self) -> None:
        self.keep_train = tuple(float(k) for k in self.keep_train)   # type: ignore[assignment]


@dataclass
class DSOCfg:
    """Seuils des objectifs de l'entreprise (DSO1 a DSO5)."""

    rmse: float = 0.5                 # DSO1 : RMSE max de u, v par niveau (m/s)
    r2: float = 0.95                  # DSO1 : R2 min de u, v par niveau
    r2_all: float = 0.95              # objectif etendu : R2 min de u, v ET w
    mass_rms: float = 1e-4            # DSO2 : RMS du residu de continuite (kg.m-3.s-1)
    roughness_range: tuple[float, float] = (0.0, 1.1)   # DSO3
    time_s: float = 1.0               # DSO4 : temps d'inference par instant (s)
    rmse_25pct: float = 0.7           # DSO5 : RMSE max avec 25 % des capteurs
    min_level_cover: float = 0.05     # niveau ignore s'il couvre moins de 5 % du domaine

    def __post_init__(self) -> None:
        self.roughness_range = tuple(float(r) for r in self.roughness_range)  # type: ignore[assignment]


SECTIONS: dict[str, type] = {
    "paths": Paths, "model": ModelCfg, "train": TrainCfg,
    "loss": LossCfg, "mask": MaskCfg, "dso": DSOCfg,
}


# --------------------------------------------------------------------------------------
# configuration complete
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    name: str = "v7"
    paths: Paths = field(default_factory=Paths)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    dso: DSOCfg = field(default_factory=DSOCfg)
    terrain_smooth_m: float = 300.0   # meme lissage du relief que le generateur
    f_coriolis: float = 8.57e-5       # s-1, latitude ~36 N (vent geostrophique)

    # ---------------------------------------------------------------- chargement
    @classmethod
    def load(cls, source: str | Path | None = None, **overrides: Any) -> "Config":
        """Charge un YAML (nom court, chemin, ou None = defauts) puis applique les surcharges.

        `Config.load("test")` lit `configs/test.yaml`. Les surcharges sont pointees :
        `Config.load(train__epochs=2)`.
        """
        data: dict[str, Any] = {}
        if source is not None:
            path = Path(source)
            if not path.exists() and path.suffix == "":
                path = CONFIG_DIR / f"{source}.yaml"
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in SECTIONS:
                kwargs[key] = SECTIONS[key](**(value or {}))
            else:
                kwargs[key] = value
        cfg = cls(**kwargs)
        cfg.apply_overrides(overrides)
        cfg.apply_env()
        return cfg

    def apply_overrides(self, overrides: Mapping[str, Any]) -> None:
        """`{"train__epochs": 2}` ou `{"train.epochs": 2}` -> self.train.epochs = 2."""
        for key, value in overrides.items():
            if value is None:
                continue
            parts = key.replace("__", ".").split(".")
            target: Any = self
            for part in parts[:-1]:
                target = getattr(target, part)
            name = parts[-1]
            if not hasattr(target, name):
                raise KeyError(f"Option de configuration inconnue : {key}")
            setattr(target, name, value)
            post_init = getattr(target, "__post_init__", None)
            if post_init is not None:
                post_init()

    def apply_env(self) -> None:
        """Variables d'environnement du notebook (ANEO_DATA / ANEO_CACHE / ANEO_RUN)."""
        for attr, var in (("data_dir", "ANEO_DATA"), ("cache_dir", "ANEO_CACHE"),
                          ("run_dir", "ANEO_RUN")):
            if os.environ.get(var):
                setattr(self.paths, attr, Path(os.environ[var]))
        self.paths.__post_init__()

    # ---------------------------------------------------------------- utilitaires
    def to_dict(self) -> dict[str, Any]:
        return _unbuild(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
        return path

    def seed_everything(self) -> None:
        import torch

        random.seed(self.train.seed)
        np.random.seed(self.train.seed)
        torch.manual_seed(self.train.seed)


def _unbuild(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _unbuild(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (tuple, list)):
        return [_unbuild(v) for v in obj]
    return obj


def pick_device(prefer: str | None = None):
    """GPU si disponible, avec TF32 active comme dans le notebook."""
    import torch

    name = prefer or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device
