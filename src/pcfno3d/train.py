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
        if cfg.train.init_from:
            self._init_from(Path(cfg.train.init_from))
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

        # bf16 seulement sur les GPU qui l'ont en materiel (Ampere et plus : L4, A100) ; sur
        # un T4 (Colab gratuit), bf16 est emule et lent : on reste en fp32
        bf16 = self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 8
        self.amp_dtype = torch.bfloat16 if bf16 else torch.float32
        self.use_amp = bf16

        self.start_epoch, self.best_val, self.history = 0, float("inf"), []
        self._best_dirty = False        # best.pt de travail plus recent que le durable
        self.syncs: list[dict] = []      # trace des copies vers le durable (tests, journal)
        self._resume()

        if verbose:
            print(f"device : {self.device} | {self.geom.summary()}")
            print(f"parametres : {count_parameters(self.model) / 1e6:.2f} M")
            print(self.splits.summary())

    # ------------------------------------------------------------------ affinage
    def _init_from(self, path: Path) -> None:
        """Reprend les POIDS d'un modele entraine (pas l'optimiseur) pour l'affiner.

        L'architecture doit etre identique : des poids de largeur 48 ne vont pas dans un
        modele de largeur 10. On le verifie avant de charger, pour echouer avec un message
        plutot qu'avec une erreur de forme de tenseur.
        """
        from .model import model_cfg_from_checkpoint

        if not path.exists():
            raise FileNotFoundError(f"train.init_from : {path} introuvable")
        ck = torch.load(path, map_location=self.device, weights_only=False)
        saved = model_cfg_from_checkpoint(ck)
        if saved is not None:
            mine = self.cfg.model
            diff = [name for name in ("width", "modes", "n_layers", "proj_hidden", "local_conv")
                    if getattr(saved, name) != getattr(mine, name)]
            if diff:
                raise ValueError(
                    f"train.init_from : architecture differente sur {diff} "
                    f"(checkpoint {[getattr(saved, n) for n in diff]}, "
                    f"config {[getattr(mine, n) for n in diff]})")
        sd = dict(ck["model"])
        notes = []
        # entree elargie (forcage ERA5 : 25 -> 34 canaux) : les nouveaux canaux sont AJOUTES
        # a la fin, les colonnes existantes de la premiere couche gardent leurs poids et les
        # nouvelles partent de 0 (au depart, le modele voit ce qu'il voyait)
        cle = "lift.0.weight"
        w_old, w_new = sd.get(cle), self.model.state_dict()[cle]
        if w_old is not None and w_old.shape != w_new.shape:
            if w_old.shape[0] != w_new.shape[0] or w_old.shape[1] > w_new.shape[1]:
                raise ValueError(f"train.init_from : {cle} {tuple(w_old.shape)} incompatible "
                                 f"avec {tuple(w_new.shape)}")
            neuf = torch.zeros_like(w_new)
            neuf[:, :w_old.shape[1]] = w_old.to(neuf)
            sd[cle] = neuf
            notes.append(f"entree {w_old.shape[1]} -> {w_new.shape[1]} canaux (nouveaux a 0)")
        # sortie residuelle depuis un modele qui ne l'etait pas : la derniere couche de u, v
        # repart de 0, donc u, v = ERA5 au depart ; w garde ce qu'il a appris
        avant = bool((ck.get("config") or {}).get("RESIDUEL_ERA5", False))
        if self.cfg.model.residuel_era5 and not avant:
            for nom in ("proj.2.weight", "proj.2.bias"):
                sd[nom] = sd[nom].clone()
                sd[nom][:2] = 0
            notes.append("u, v : derniere couche a 0 (depart = ERA5)")
        self.model.load_state_dict(sd)
        if self.verbose:
            print(f"affinage a partir de {path.name} "
                  f"(epoque {ck.get('epoch', -1) + 1} du modele d'origine)"
                  + (" ; " + " ; ".join(notes) if notes else ""))

    def _dso_selection(self) -> dict:
        """Score DSO sur la validation : le critere de selection de la v8."""
        from . import dso

        metrics = dso.measure(self.ema_model, self.loaders["val"], self.geom, self.norm,
                              self.device, self.amp_dtype, self.use_amp)
        return dso.score(metrics, self.cfg, self.geom)

    # ------------------------------------------------------------------ EMA
    def _ema_update(self) -> None:
        """Moyenne exponentielle des poids, avec decroissance progressive au demarrage."""
        self.ema_steps += 1
        d = min(self.cfg.train.ema_decay, (1 + self.ema_steps) / (10 + self.ema_steps))
        with torch.no_grad():
            for pe, p in zip(self.ema_model.parameters(), self.model.parameters()):
                pe.mul_(d).add_(p.detach(), alpha=1 - d)

    # ------------------------------------------------------------------ emplacements
    # Deux niveaux, comme le BLOC 9 du notebook v7 :
    #   « travail » : ou l'on ecrit a chaque epoque — paths.local_dir s'il est defini
    #                 (disque local de Colab), sinon run_dir directement ;
    #   « durable » : run_dir (le Drive sur Colab), alimente par _sync() toutes les
    #                 `save_every` epoques, a la fin, et sur interruption.
    @property
    def _work_dir(self) -> Path:
        return self.cfg.paths.local_dir or self.cfg.paths.run_dir

    @property
    def _split(self) -> bool:
        """Vrai si le travail et le durable sont deux dossiers differents."""
        local = self.cfg.paths.local_dir
        return local is not None and local.resolve() != self.cfg.paths.run_dir.resolve()

    @property
    def _last_work(self) -> Path:
        return self._work_dir / "last.pt"

    @property
    def _best_work(self) -> Path:
        return self._work_dir / "best.pt"

    # ------------------------------------------------------------------ reprise
    def _resume(self) -> None:
        """Reprend de last.pt : le local d'abord (meme session), sinon le durable.

        Une session Colab neuve n'a plus son disque local : on repart alors de la derniere
        copie durable (au plus `save_every` epoques de retard), et on ramene best.pt en local
        pour que la comparaison des epoques suivantes reste juste.
        """
        candidates = [self._last_work]
        if self._split:
            candidates.append(self.cfg.paths.ckpt_last)
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
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
        if self._split and path == self.cfg.paths.ckpt_last:
            if self.cfg.paths.ckpt_best.exists() and not self._best_work.exists():
                self._copy_atomic(self.cfg.paths.ckpt_best, self._best_work)
        if self.verbose:
            origine = "copie durable" if path == self.cfg.paths.ckpt_last and self._split \
                else "disque de travail"
            print(f"reprise a l'epoque {self.start_epoch} "
                  f"(meilleure val = {self.best_val:.5f}) depuis {path} ({origine})")

    # ------------------------------------------------------------------ ecritures
    def _save_atomic(self, path: Path, payload: dict) -> None:
        """Ecrit via un fichier temporaire : un checkpoint n'est jamais a moitie ecrit."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @staticmethod
    def _copy_atomic(src: Path, dst: Path) -> None:
        """Copie via un fichier temporaire : le Drive ne voit jamais un fichier a moitie copie."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

    def _sync(self, reason: str = "") -> None:
        """Copie le travail vers le durable. best.pt seulement s'il a change depuis.

        Sans local_dir, tout est deja ecrit dans run_dir : rien a faire.
        """
        if not self._split:
            return
        t0 = time.time()
        copied = []
        if self._best_dirty and self._best_work.exists():
            self._copy_atomic(self._best_work, self.cfg.paths.ckpt_best)
            self._best_dirty = False
            copied.append("best.pt")
        if self.cfg.train.sync_last and self._last_work.exists():
            self._copy_atomic(self._last_work, self.cfg.paths.ckpt_last)
            copied.append("last.pt")
        self.syncs.append({"raison": reason, "fichiers": copied})
        if self.verbose and copied:
            print(f"      -> {', '.join(copied)} recopie(s) vers {self.cfg.paths.run_dir} "
                  f"({time.time() - t0:.0f} s{', ' + reason if reason else ''})")

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self, net: torch.nn.Module | None = None) -> tuple[dict[str, float], list[float]]:
        """Loss de validation et R2 GLOBAL cumule sur tous les instants (comme le BLOC 10)."""
        net = net or self.ema_model
        net.eval()
        acc = {"data": 0.0, "mass": 0.0, "bc": 0.0, "grad": 0.0}
        stats = np.zeros((3, 3))        # par composante : sse, somme t, somme t^2
        counts = np.zeros(3)            # noeuds connus par composante
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
                known = valid[None] & torch.isfinite(y[:, c])      # NaN = pas d'observation
                pv, tv = p[known].double(), y[:, c][known].double()
                stats[c] += [((pv - tv) ** 2).sum().item(), tv.sum().item(),
                             (tv ** 2).sum().item()]
                counts[c] += int(known.sum().item())
        n = max(len(loader), 1)
        r2 = [1 - stats[c, 0] / (stats[c, 2] - stats[c, 1] ** 2 / counts[c])
              if counts[c] else float("nan") for c in range(3)]
        return {k: v / n for k, v in acc.items()}, r2

    # ------------------------------------------------------------------ boucle
    def _save_best(self, epoch: int, score: float, extra: dict | None = None) -> None:
        m = self.cfg.model
        payload = {
            "model": self.ema_model.state_dict(), "epoch": epoch, "val_score": score,
            "config": {"WIDTH": m.width, "MODES": list(m.modes), "N_LAYERS": m.n_layers,
                       "PAD": list(m.pad), "PROJ_HIDDEN": m.proj_hidden, "C_IN": self.geom.c_in,
                       "LOCAL_CONV": m.local_conv, "POOL_SURFACE": m.pool_surface,
                       "RESIDUEL_ERA5": m.residuel_era5},
        }
        payload.update(extra or {})
        self._save_atomic(self._best_work, payload)
        self._best_dirty = True

    def _baseline_for_finetune(self) -> None:
        """En affinage selectionne sur les DSO, le modele d'origine est le premier candidat.

        Sans cela, la premiere epoque d'affinage serait enregistree comme « meilleure » meme
        si elle degrade le modele de depart — ce qui est arrive au BLOC 9b de la v7. Ici,
        l'affinage ne peut que garder l'original ou faire mieux.
        """
        sel = self._dso_selection()
        self.best_val = sel["score"]
        self._save_best(-1, sel["score"], {"dso_val": sel, "origine": "init_from"})
        self.history.append({"epoch": -1, "val_score": sel["score"], "select_by": "dso",
                             "dso": sel, "note": "modele d'origine, avant affinage"})
        if self.verbose:
            print(f"depart (modele d'origine) : score DSO = {sel['score']:.3f} | limitant "
                  f"{sel['limitant']} | RMSE max {sel['rmse_max']:.3f} m/s a "
                  f"{sel['niveau_le_pire_m']} m")

    def fit(self) -> list[dict]:
        cfg = self.cfg
        if (cfg.train.init_from and cfg.train.select_by == "dso"
                and self.start_epoch == 0 and not self.history):
            self._baseline_for_finetune()
            # le modele d'origine est le premier candidat : il doit etre durable tout de suite
            self._sync("modele d'origine")
        termine = False
        try:
            self._epochs()
            termine = True
        finally:
            if not termine:
                # interruption (cellule arretee, erreur) : on sauve ce qui peut l'etre.
                # Une session Colab coupee net ne passe pas ici — d'ou les copies regulieres.
                try:
                    self._sync("interruption")
                except Exception as exc:                  # noqa: BLE001
                    print(f"      !! copie vers {cfg.paths.run_dir} impossible : {exc}")
        return self.history

    def _epochs(self) -> None:
        cfg = self.cfg
        for epoch in range(self.start_epoch, cfg.train.epochs):
            self.model.train()
            phys_w = physics_weight(epoch, cfg.train.warmup_phys_epochs)
            t0 = time.time()
            tr = {"total": 0.0, "data": 0.0, "mass": 0.0, "bc": 0.0, "station": 0.0}
            loader = self.loaders["train"]
            self.optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(loader):
                s_raw, sensor, tfeat, y = (t.to(self.device, non_blocking=True)
                                           for t in batch[:4])
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                    enabled=self.use_amp):
                    pred = self.model(assemble_input(s_raw, sensor, tfeat, self.geom, self.norm))
                loss, parts = self.loss_fn(pred, y, phys_w)
                if len(batch) > 5:
                    # affinage : vent a 10 m a la station cible contre la vraie mesure
                    from .stations_reelles import perte_station
                    l_st = perte_station(pred, batch[5].to(self.device), self.norm)
                    loss = loss + cfg.loss.lambda_station * l_st
                    tr["station"] += float(l_st.item())
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
            dso_sel = None
            if cfg.train.select_by == "dso":
                dso_sel = self._dso_selection()
                score = dso_sel["score"]
            rec = {"epoch": epoch, "phys_w": phys_w,
                   "lr": self.optimizer.param_groups[0]["lr"], "train": tr, "val": va,
                   "val_r2_u": r2[0], "val_r2_v": r2[1], "val_r2_w": r2[2],
                   "val_score": score, "select_by": cfg.train.select_by, "dso": dso_sel,
                   "time_s": time.time() - t0}
            self.history.append(rec)

            improved = score < self.best_val
            if improved:
                self.best_val = score
                self._save_best(epoch, score, {"dso_val": dso_sel} if dso_sel else None)
            # ~7 Go (modele, EMA, etat d'Adam) : sur le disque de travail, a chaque epoque
            self._save_atomic(self._last_work, {
                "model": self.ema_model.state_dict(), "model_raw": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "ema_steps": self.ema_steps,
                "scheduler": self.scheduler.state_dict(), "epoch": epoch,
                "best_val": self.best_val, "history": self.history,
            })
            # quelques ko : directement dans run_dir, pour suivre l'entrainement depuis le Drive
            cfg.paths.history.write_text(json.dumps(self.history, indent=1), encoding="utf-8")

            if self.verbose:
                print(f"Ep {epoch + 1:3d}/{cfg.train.epochs} | phys={phys_w:.2f} | "
                      f"train data={tr['data']:.4f} mass={tr['mass']:.2e} "
                      + (f"station={tr['station']:.3f} " if tr["station"] else "")
                      + f"| "
                      f"val data={va['data']:.4f} mass={va['mass']:.2e} bc={va['bc']:.3f} "
                      f"grad={va['grad']:.3f} | R2 u={r2[0]:.3f} v={r2[1]:.3f} w={r2[2]:.3f} "
                      + (f"| DSO {dso_sel['score']:.3f} ({dso_sel['limitant']}, RMSE max "
                         f"{dso_sel['rmse_max']:.3f} a {dso_sel['niveau_le_pire_m']} m) "
                         if dso_sel else "")
                      + f"| {rec['time_s']:.0f} s{' *' if improved else ''}")
            if (epoch + 1) % cfg.train.save_every == 0 or epoch == cfg.train.epochs - 1:
                self._sync(f"epoque {epoch + 1}")


def train(cfg: Config, device: str | None = None, verbose: bool = True) -> list[dict]:
    return Trainer(cfg, device=device, verbose=verbose).fit()
