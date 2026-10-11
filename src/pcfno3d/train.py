"""BLOC 9 — entrainement avec reprise automatique, EMA des poids et sauvegarde atomique.

Deux checkpoints :
    last.pt  etat complet (modele, EMA, optimiseur, scheduler, historique) -> reprise
    best.pt  poids EMA du meilleur score de validation + architecture -> evaluation, service

La boucle est interruptible : relancer `train` reprend a l'epoque suivante.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config, pick_device
from .dataset import assemble_input, make_loaders, make_splits
from .geometry import Geometry
from .losses import PhysicsLoss, physics_weight
from .model import build_model, count_parameters
from .normalize import Normalizer


class Trainer:
    def __init__(self, cfg: Config, device: torch.device | str | None = None,
                 verbose: bool = True) -> None:
        from .cache import load_index

        self.cfg = cfg
        self.verbose = verbose
        self.device = pick_device(str(device) if device else None)
        cfg.paths.mkdirs()
        cfg.seed_everything()

        self.geom = Geometry(cfg, device=self.device)
        self.index = load_index(cfg)
        self.norm = Normalizer.load_or_compute(cfg, self.geom, self.index, verbose=verbose)
        self.splits = make_splits(cfg, self.index)
        self.loaders = make_loaders(cfg, self.geom, self.splits)
        self.loss_fn = PhysicsLoss(cfg, self.geom, self.norm)

        self.model = build_model(cfg, self.geom.c_in).to(self.device)
        self.ema_model = copy.deepcopy(self.model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_steps = 0

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.lr,
                                           weight_decay=cfg.train.weight_decay)
        self.total_steps = cfg.train.epochs * math.ceil(
            cfg.train.steps_per_epoch / cfg.train.accum_steps)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=cfg.train.lr, total_steps=self.total_steps,
            pct_start=0.1, anneal_strategy="cos")

        self.amp_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.use_amp = self.device.type == "cuda"

        self.start_epoch, self.best_val, self.history = 0, float("inf"), []
        self._resume()

        if verbose:
            print(f"device : {self.device} | {self.geom.summary()}")
            print(f"parametres : {count_parameters(self.model) / 1e6:.2f} M")
            print(self.splits.summary())

    # ------------------------------------------------------------------ EMA
    def _ema_update(self) -> None:
        """Moyenne exponentielle des poids, avec decroissance progressive au demarrage."""
        self.ema_steps += 1
        d = min(self.cfg.train.ema_decay, (1 + self.ema_steps) / (10 + self.ema_steps))
        with torch.no_grad():
            for pe, p in zip(self.ema_model.parameters(), self.model.parameters()):
                pe.mul_(d).add_(p.detach(), alpha=1 - d)

    # ------------------------------------------------------------------ reprise
    def _resume(self) -> None:
        path = self.cfg.paths.ckpt_last
        if not path.exists():
            return
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck.get("model_raw", ck["model"]))
        self.ema_model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scheduler.load_state_dict(ck["scheduler"])
        self.ema_steps = ck.get("ema_steps", 0)
        self.start_epoch = ck["epoch"] + 1
        self.best_val = ck["best_val"]
        self.history = ck["history"]
        if self.verbose:
            print(f"reprise a l'epoque {self.start_epoch} "
                  f"(meilleure val = {self.best_val:.5f}) depuis {path.name}")

    def _save_atomic(self, path: Path, payload: dict) -> None:
        """Ecrit via un fichier temporaire : un checkpoint n'est jamais a moitie ecrit."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _mirror_cache_dir(self) -> None:
        """Copie du cache vers `cache_mirror` (Drive) si configure."""
        mirror = self.cfg.paths.cache_mirror
        if mirror is None:
            return
        mirror.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.cfg.paths.cache_dir, mirror, dirs_exist_ok=True)

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self, net: torch.nn.Module | None = None) -> tuple[dict[str, float], list[float]]:
        """Loss de validation et R2 GLOBAL cumule sur tous les instants (comme le BLOC 10)."""
        net = net or self.ema_model
        net.eval()
        acc = {"data": 0.0, "mass": 0.0, "bc": 0.0, "grad": 0.0}
        stats = np.zeros((3, 3))        # par composante : sse, somme t, somme t^2
        count = 0
        loader = self.loaders["val"]
        for s_raw, sensor, tfeat, y, _ in loader:
            s_raw, sensor, tfeat, y = (t.to(self.device, non_blocking=True)
                                       for t in (s_raw, sensor, tfeat, y))
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.use_amp):
                pred = net(assemble_input(s_raw, sensor, tfeat, self.geom, self.norm))
            _, parts = self.loss_fn(pred, y, 1.0)
            for k in acc:
                acc[k] += parts[k]
            valid = self.geom.valid_t
            for c, p in enumerate(self.norm.denorm(pred.float())):
                pv, tv = p[:, valid].double(), y[:, c][:, valid].double()
                stats[c] += [((pv - tv) ** 2).sum().item(), tv.sum().item(),
                             (tv ** 2).sum().item()]
            count += int(valid.sum().item()) * y.shape[0]
        n = max(len(loader), 1)
        r2 = [1 - stats[c, 0] / (stats[c, 2] - stats[c, 1] ** 2 / count) for c in range(3)] \
            if count else [float("nan")] * 3
        return {k: v / n for k, v in acc.items()}, r2

    # ------------------------------------------------------------------ boucle
    def fit(self) -> list[dict]:
        cfg = self.cfg
        for epoch in range(self.start_epoch, cfg.train.epochs):
            self.model.train()
            phys_w = physics_weight(epoch, cfg.train.warmup_phys_epochs)
            t0 = time.time()
            tr = {"total": 0.0, "data": 0.0, "mass": 0.0, "bc": 0.0}
            loader = self.loaders["train"]
            self.optimizer.zero_grad(set_to_none=True)

            for step, (s_raw, sensor, tfeat, y, _) in enumerate(loader):
                s_raw, sensor, tfeat, y = (t.to(self.device, non_blocking=True)
                                           for t in (s_raw, sensor, tfeat, y))
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                    enabled=self.use_amp):
                    pred = self.model(assemble_input(s_raw, sensor, tfeat, self.geom, self.norm))
                loss, parts = self.loss_fn(pred, y, phys_w)
                (loss / cfg.train.accum_steps).backward()

                if (step + 1) % cfg.train.accum_steps == 0 or step + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._ema_update()
                    if self.scheduler.last_epoch < self.total_steps - 1:
                        self.scheduler.step()

                tr["total"] += loss.item()
                for k in ("data", "mass", "bc"):
                    tr[k] += parts[k]

            n = max(len(loader), 1)
            tr = {k: v / n for k, v in tr.items()}
            va, r2 = self.validate()
            score = self.loss_fn.val_score(va)
            rec = {"epoch": epoch, "phys_w": phys_w,
                   "lr": self.optimizer.param_groups[0]["lr"], "train": tr, "val": va,
                   "val_r2_u": r2[0], "val_r2_v": r2[1], "val_r2_w": r2[2],
                   "val_score": score, "time_s": time.time() - t0}
            self.history.append(rec)

            improved = score < self.best_val
            if improved:
                self.best_val = score
                m = cfg.model
                self._save_atomic(cfg.paths.ckpt_best, {
                    "model": self.ema_model.state_dict(), "epoch": epoch, "val_score": score,
                    "config": {"WIDTH": m.width, "MODES": list(m.modes),
                               "N_LAYERS": m.n_layers, "PAD": list(m.pad),
                               "PROJ_HIDDEN": m.proj_hidden, "C_IN": self.geom.c_in,
                               "LOCAL_CONV": m.local_conv, "POOL_SURFACE": m.pool_surface},
                })
            self._save_atomic(cfg.paths.ckpt_last, {
                "model": self.ema_model.state_dict(), "model_raw": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "ema_steps": self.ema_steps,
                "scheduler": self.scheduler.state_dict(), "epoch": epoch,
                "best_val": self.best_val, "history": self.history,
            })
            cfg.paths.history.write_text(json.dumps(self.history, indent=1), encoding="utf-8")

            if self.verbose:
                print(f"Ep {epoch + 1:3d}/{cfg.train.epochs} | phys={phys_w:.2f} | "
                      f"train data={tr['data']:.4f} mass={tr['mass']:.2e} | "
                      f"val data={va['data']:.4f} mass={va['mass']:.2e} bc={va['bc']:.3f} "
                      f"grad={va['grad']:.3f} | R2 u={r2[0]:.3f} v={r2[1]:.3f} w={r2[2]:.3f} "
                      f"| {rec['time_s']:.0f} s{' *' if improved else ''}")
            if (epoch + 1) % cfg.train.save_every == 0 or epoch == cfg.train.epochs - 1:
                self._mirror_cache_dir()

        return self.history


def train(cfg: Config, device: str | None = None, verbose: bool = True) -> list[dict]:
    return Trainer(cfg, device=device, verbose=verbose).fit()
