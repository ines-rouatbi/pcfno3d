"""BLOC 7 — le modele PC-FNO3D (operateur neuronal de Fourier 3D)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config, ModelCfg


class SpectralConv3d(nn.Module):
    """Couche de Fourier 3D : poids complexes appris sur les modes conserves (z, y, x)."""

    def __init__(self, cin: int, cout: int, modes: tuple[int, int, int]) -> None:
        super().__init__()
        self.cout = cout
        self.m1, self.m2, self.m3 = modes
        scale = 1.0 / (cin * cout)
        # 4 blocs de poids : un par "coin" du spectre (signes de kz et ky)
        self.w = nn.ParameterList([
            nn.Parameter(scale * torch.randn(cin, cout, *modes, dtype=torch.cfloat))
            for _ in range(4)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # la FFT se calcule en fp32 : en bf16 le spectre est trop grossier
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B, _, D, H, W = x.shape
            xf = torch.fft.rfftn(x, dim=(-3, -2, -1))
            out = torch.zeros(B, self.cout, D, H, W // 2 + 1, dtype=torch.cfloat,
                              device=x.device)
            m1, m2, m3 = self.m1, self.m2, self.m3

            def mul(a, w):
                return torch.einsum("bixyz,ioxyz->boxyz", a, w)

            out[:, :, :m1, :m2, :m3] = mul(xf[:, :, :m1, :m2, :m3], self.w[0])
            out[:, :, -m1:, :m2, :m3] = mul(xf[:, :, -m1:, :m2, :m3], self.w[1])
            out[:, :, :m1, -m2:, :m3] = mul(xf[:, :, :m1, -m2:, :m3], self.w[2])
            out[:, :, -m1:, -m2:, :m3] = mul(xf[:, :, -m1:, -m2:, :m3], self.w[3])
            # retour sur la grille : chaque point depend de tout le domaine
            return torch.fft.irfftn(out, s=(D, H, W), dim=(-3, -2, -1))


class PCFNO3D(nn.Module):
    """Lift 1x1x1 -> n couches (Fourier + skip + conv locale) -> tete de projection u, v, w."""

    def __init__(self, cin: int, width: int, modes: tuple[int, int, int], n_layers: int,
                 pad: tuple[int, int, int], proj_hidden: int = 128,
                 local: bool = False, recalcul: bool = False) -> None:
        super().__init__()
        self.pad = pad
        # recalcul des activations couche par couche a la retropropagation (memoire GPU des
        # grandes grilles, ex. Grand Tunis 12 x 560 x 680) : memes resultats, ~30 % plus lent
        self.recalcul = recalcul
        self.lift = nn.Sequential(nn.Conv3d(cin, width, 1), nn.GELU(),
                                  nn.Conv3d(width, width, 1))
        self.spec = nn.ModuleList([SpectralConv3d(width, width, modes)
                                   for _ in range(n_layers)])
        self.skip = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(n_layers)])
        # la couche de Fourier ne garde que les grandes echelles (modes tronques) ; la
        # convolution 3x3x3 restitue les details fins (vents de pente, contrastes de rugosite)
        self.local = nn.ModuleList([nn.Conv3d(width, width, 3, padding=1)
                                    for _ in range(n_layers)]) if local else None
        self.proj = nn.Sequential(nn.Conv3d(width, proj_hidden, 1), nn.GELU(),
                                  nn.Conv3d(proj_hidden, 3, 1))

    def _couche(self, x: torch.Tensor, n: int) -> torch.Tensor:
        y = self.spec[n](x) + self.skip[n](x)
        if self.local is not None:
            y = y + self.local[n](x)
        return F.gelu(y) if n < len(self.spec) - 1 else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, H, W = x.shape[-3:]
        pz, py, px = self.pad
        x = self.lift(x)
        # zeros aux bords : limite la periodicite imposee par la FFT
        x = F.pad(x, (0, px, 0, py, 0, pz))
        for n in range(len(self.spec)):
            if self.recalcul and self.training and x.requires_grad:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(self._couche, x, n, use_reentrant=False)
            else:
                x = self._couche(x, n)
        x = x[..., :D, :H, :W]
        # tete en fp32 : en bf16 la sortie est arrondie a quelques cm/s, et ce bruit passe
        # dans les differences centrees, creant un plancher sur le residu de masse (DSO2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.proj(x.float())      # (B, 3, nz, ny, nx) : u, v, w normalises


def build_model(cfg: Config, c_in: int) -> PCFNO3D:
    m = cfg.model
    return PCFNO3D(c_in, m.width, m.modes, m.n_layers, m.pad, m.proj_hidden,
                   local=m.local_conv, recalcul=m.recalcul_activations)


def count_parameters(model: nn.Module) -> int:
    """Un parametre complexe compte pour deux reels."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


def model_cfg_from_checkpoint(ckpt: dict) -> ModelCfg | None:
    """Relit l'architecture enregistree dans le checkpoint (BLOC 9 ecrit `config`)."""
    saved = ckpt.get("config")
    if not saved:
        return None
    return ModelCfg(width=saved["WIDTH"], modes=tuple(saved["MODES"]),
                    n_layers=saved["N_LAYERS"], pad=tuple(saved["PAD"]),
                    proj_hidden=saved["PROJ_HIDDEN"], local_conv=saved["LOCAL_CONV"],
                    pool_surface=saved["POOL_SURFACE"])


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    return torch.load(Path(path), map_location=device, weights_only=False)


def save_inference_checkpoint(path: str | Path, model: nn.Module, cfg: Config,
                              c_in: int, extra: dict | None = None) -> Path:
    """Checkpoint auto-suffisant pour le service : poids + architecture."""
    m = cfg.model
    payload = {
        "model": model.state_dict(),
        "config": {"WIDTH": m.width, "MODES": list(m.modes), "N_LAYERS": m.n_layers,
                   "PAD": list(m.pad), "PROJ_HIDDEN": m.proj_hidden, "C_IN": c_in,
                   "LOCAL_CONV": m.local_conv, "POOL_SURFACE": m.pool_surface},
    }
    payload.update(extra or {})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path
