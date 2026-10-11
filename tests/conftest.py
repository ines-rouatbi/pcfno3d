"""Fixtures : un mini-dataset genere une fois, puis mis en cache pour la session.

Aucun test ne depend du dataset reel (35 Go) ni d'un GPU : tout tourne sur CPU en
quelques secondes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(scope="session")
def mini_dataset(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("aneo_mini")
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_mini_dataset.py"),
                    str(out)], check=True, capture_output=True)
    return out


@pytest.fixture(scope="session")
def cfg(mini_dataset, tmp_path_factory):
    from pcfno3d.config import Config

    root = tmp_path_factory.mktemp("run")
    cfg = Config.load("test")
    cfg.apply_overrides({
        "paths.data_dir": mini_dataset,
        "paths.cache_dir": root / "cache",
        "paths.run_dir": root / "run",
        "train.num_workers": 0,
    })
    cfg.paths.mkdirs()
    return cfg


@pytest.fixture(scope="session")
def geom(cfg):
    from pcfno3d.geometry import Geometry

    return Geometry(cfg, device="cpu")


@pytest.fixture(scope="session")
def index(cfg, geom):
    from pcfno3d.cache import build_cache

    return build_cache(cfg, geom, verbose=False)


@pytest.fixture(scope="session")
def norm(cfg, geom, index):
    from pcfno3d.normalize import Normalizer

    return Normalizer.load_or_compute(cfg, geom, index, verbose=False)
