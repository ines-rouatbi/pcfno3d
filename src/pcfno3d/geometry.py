"""BLOC 2 — grille statique, masques, canaux statiques et operateurs discrets.

`Geometry` rassemble tout ce qui ne depend pas de l'instant : la grille du modele, le
relief, les masques de validite et les operateurs de derivation. C'est l'objet que le
notebook gardait dans des variables globales (VALID, MASS_MASK, STATIC_2D_T, ...).

La lecture de `aneo_grid_points.csv` (127 Mo) est mise en cache dans un .npz, parce que
l'entrainement, l'evaluation et le service la refont a chaque demarrage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, gaussian_filter

from .config import Config

# 7 variables mesurees en surface, dans l'ordre des canaux de S
SURF_VARS = ["wind_u_m_s", "wind_v_m_s", "wind_w_m_s", "temperature_c",
             "air_density_kg_m3", "relative_humidity_pct", "pressure_pa"]
N_SURF = len(SURF_VARS)
# heure (2), jour de l'annee (2), vent geostrophique (2), vent thermique (2)
N_TFEAT = 8
# colonnes de la grille utilisees
GRID_COLS = ["point_id", "ix", "iy", "z_terrain_m", "z_rel_m", "z0_m", "slope_deg", "aspect_deg"]
GEOMETRY_CACHE_VERSION = 1


@dataclass
class Meta:
    """Lecture de `meta.json` : la geometrie vient du generateur, jamais de constantes."""

    nx_s: int
    ny_s: int
    dx_s: float
    dy_s: float
    stride: int
    z_levels: np.ndarray
    node_id_start: int
    raw: dict

    @classmethod
    def load(cls, data_dir: Path) -> "Meta":
        with open(Path(data_dir) / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        surface, nodes = meta["grille_surface"], meta["noeuds_3d"]
        return cls(
            nx_s=int(surface["nx"]), ny_s=int(surface["ny"]),
            dx_s=float(surface["dx_m"]), dy_s=float(surface["dy_m"]),
            stride=int(nodes["stride"]),
            z_levels=np.asarray(nodes["z_levels_asl_m"], dtype=np.float64),
            node_id_start=int(nodes["point_id_debut"]),
            raw=meta,
        )


def load_split(data_dir: Path) -> dict[str, list[int]]:
    with open(Path(data_dir) / "split_days.json", encoding="utf-8") as f:
        return json.load(f)


class Geometry:
    """Grille du modele, masques et operateurs. Immuable apres construction."""

    def __init__(self, cfg: Config, device: torch.device | str = "cpu",
                 use_cache: bool = True) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.meta = Meta.load(cfg.paths.data_dir)
        self.split = load_split(cfg.paths.data_dir)
        self.pool_surface_enabled = bool(cfg.model.pool_surface)

        m = self.meta
        self.nx_s, self.ny_s = m.nx_s, m.ny_s
        self.dx_s, self.dy_s = m.dx_s, m.dy_s
        self.stride = m.stride
        self.nx, self.ny = m.nx_s // m.stride, m.ny_s // m.stride
        self.dx, self.dy = m.dx_s * m.stride, m.dy_s * m.stride
        self.z_levels = m.z_levels
        self.nz = len(m.z_levels)
        self.node_id_start = m.node_id_start

        arrays = self._load_cached() if use_cache else None
        if arrays is None:
            arrays = self._build_from_csv()
            if use_cache:
                self._save_cache(arrays)
        for key, value in arrays.items():
            setattr(self, key, value)

        self._derive()
        self._to_torch()

        # modele reel : vent ERA5 aux noeuds + forcage de l'instant (pcfno3d.forcage)
        self.forcage = self.interp_era5 = None
        if cfg.paths.forcage_era5 is not None:
            from .forcage import ForcageERA5, InterpERA5
            self.forcage = ForcageERA5(cfg.paths.forcage_era5)
            self.interp_era5 = InterpERA5(self.forcage, self)

    # ------------------------------------------------------------------ construction
    def _build_from_csv(self) -> dict[str, np.ndarray]:
        """Lit la grille et en deduit relief, masques et tables de correspondance."""
        grid = pd.read_csv(self.cfg.paths.data_dir / "aneo_grid_points.csv", usecols=GRID_COLS)
        surf = (grid[grid["point_id"] < self.node_id_start]
                .sort_values(["iy", "ix"]).reset_index(drop=True))
        if len(surf) != self.nx_s * self.ny_s:
            raise ValueError(
                f"Grille de surface incomplete : {len(surf)} points pour "
                f"{self.nx_s} x {self.ny_s} = {self.nx_s * self.ny_s} attendus"
            )
        nodes = grid[grid["point_id"] >= self.node_id_start].reset_index(drop=True)
        s = self.stride

        def s2d(col: str) -> np.ndarray:
            return surf[col].values.reshape(self.ny_s, self.nx_s).astype(np.float64)

        zt_50 = s2d("z_terrain_m")
        out: dict[str, np.ndarray] = {
            "zt_50": zt_50,
            "zt": zt_50[::s, ::s],
            "z0": np.maximum(s2d("z0_m")[::s, ::s], 1e-4),
            "slope": s2d("slope_deg")[::s, ::s],
            "aspect": np.radians(s2d("aspect_deg")[::s, ::s]),
            "zrel_sensor": s2d("z_rel_m")[::s, ::s],
        }

        # point_id de surface -> indice plat sur la grille fine
        pid_to_surf = np.full(int(surf["point_id"].max()) + 1, -1, dtype=np.int64)
        pid_to_surf[surf["point_id"].values] = np.arange(len(surf))
        out["pid_to_surf"] = pid_to_surf

        # noeud 3D -> indice plat dans le volume (nz, ny, nx)
        z_abs = nodes["z_terrain_m"].values + nodes["z_rel_m"].values
        k = np.abs(z_abs[:, None] - self.z_levels[None, :]).argmin(axis=1)
        j = nodes["iy"].values // s
        i = nodes["ix"].values // s
        node_flat = np.full(int(nodes["point_id"].max()) - self.node_id_start + 1, -1,
                            dtype=np.int64)
        node_flat[nodes["point_id"].values - self.node_id_start] = (k * self.ny + j) * self.nx + i
        out["node_flat"] = node_flat

        # masque des noeuds existants (au-dessus du relief)
        valid = np.zeros(self.nz * self.ny * self.nx, dtype=bool)
        valid[node_flat[node_flat >= 0]] = True
        out["valid"] = valid.reshape(self.nz, self.ny, self.nx)

        # relief lisse comme le generateur -> pentes vues par l'ecoulement (condition au sol)
        zt_flow = gaussian_filter(zt_50, sigma=self.cfg.terrain_smooth_m / self.dx_s,
                                  mode="nearest")[::s, ::s]
        dzt_dy, dzt_dx = np.gradient(zt_flow, self.dy, self.dx)
        out["dzt_dx"], out["dzt_dy"] = dzt_dx, dzt_dy

        # altitude absolue du capteur (moyenne du bloc stride x stride si pooling)
        if self.pool_surface_enabled:
            z_sensor = (zt_50 + s2d("z_rel_m")).reshape(
                self.ny, s, self.nx, s).mean(axis=(1, 3))
        else:
            z_sensor = out["zt"] + out["zrel_sensor"]
        out["z_sensor"] = z_sensor
        return out

    def _derive(self) -> None:
        """Masques derives et canaux statiques d'entree (ordre de grandeur O(1))."""
        valid = self.valid
        # residu de masse : il faut les 4 voisins horizontaux valides (differences centrees)
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        self.valid_interior = np.stack([
            binary_erosion(valid[k], structure=cross, border_value=1) for k in range(self.nz)
        ])
        self.mass_mask = self.valid_interior[1:] & self.valid_interior[:-1]

        # premier noeud au-dessus du sol de chaque colonne (condition cinematique)
        first_valid = valid.copy()
        first_valid[1:] &= ~valid[:-1]
        self.first_valid = first_valid

        # 6 canaux de relief
        logz0 = np.log(self.z0)
        self.static_2d = np.stack([
            self.zt / 500.0,
            (logz0 - logz0.mean()) / (logz0.std() + 1e-6),
            self.slope / 10.0,
            np.sin(self.aspect),
            np.cos(self.aspect),
            (self.zrel_sensor - 25.0) / 5.0,
        ]).astype(np.float32)

        # 3 canaux de geometrie par niveau
        z_asl = np.broadcast_to(self.z_levels[:, None, None], (self.nz, self.ny, self.nx))
        z_agl = np.clip(z_asl - self.zt[None], 0.0, None)
        self.static_3d = np.stack([z_asl / 1000.0, z_agl / 1000.0,
                                   valid.astype(np.float64)]).astype(np.float32)

        # coordonnees centrees (ajustement du plan de pression / temperature)
        self.xc = (np.arange(self.nx) * self.dx)[None, :].repeat(self.ny, 0) \
            - (self.nx - 1) * self.dx / 2
        self.yc = (np.arange(self.ny) * self.dy)[:, None].repeat(self.nx, 1) \
            - (self.ny - 1) * self.dy / 2

        # part du domaine au-dessus du relief, par niveau
        self.cover = valid.reshape(self.nz, -1).mean(axis=1)

    def _to_torch(self) -> None:
        d = self.device
        self.dz_t = torch.tensor(np.diff(self.z_levels), dtype=torch.float32, device=d)
        self.valid_t = torch.from_numpy(self.valid).to(d)
        self.grad_mask_x_t = torch.from_numpy(self.valid[:, :, 1:] & self.valid[:, :, :-1]).to(d)
        self.grad_mask_y_t = torch.from_numpy(self.valid[:, 1:, :] & self.valid[:, :-1, :]).to(d)
        self.mass_mask_t = torch.from_numpy(self.mass_mask).to(d)
        self.first_valid_t = torch.from_numpy(self.first_valid).to(d)
        self.dzt_dx_t = torch.tensor(self.dzt_dx, dtype=torch.float32, device=d)
        self.dzt_dy_t = torch.tensor(self.dzt_dy, dtype=torch.float32, device=d)
        self.static_2d_t = torch.from_numpy(self.static_2d).to(d)
        self.static_3d_t = torch.from_numpy(self.static_3d).to(d)

    # ------------------------------------------------------------------ cache disque
    def _cache_signature(self) -> np.ndarray:
        return np.array([GEOMETRY_CACHE_VERSION, self.nx_s, self.ny_s, self.stride, self.nz,
                         int(self.pool_surface_enabled), int(self.cfg.terrain_smooth_m)],
                        dtype=np.int64)

    def _load_cached(self) -> dict[str, np.ndarray] | None:
        path = self.cfg.paths.geometry_cache
        if not path.exists():
            return None
        try:
            with np.load(path) as z:
                if not np.array_equal(z["signature"], self._cache_signature()):
                    return None
                return {k: z[k] for k in z.files if k != "signature"}
        except Exception:
            return None   # cache illisible : on reconstruit

    def _save_cache(self, arrays: dict[str, np.ndarray]) -> None:
        path = self.cfg.paths.geometry_cache
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        # np.savez_compressed ajoute ".npz" a un nom de fichier, pas a un descripteur ouvert
        with open(tmp, "wb") as f:
            np.savez_compressed(f, signature=self._cache_signature(), **arrays)
        tmp.replace(path)

    # ------------------------------------------------------------------ proprietes
    @property
    def c_in(self) -> int:
        """7 mesures + 1 disponibilite + 6 relief + 3 geometrie + 8 temps = 25 canaux ;
        + 7 scalaires de forcage + u, v ERA5 aux noeuds = 34 avec paths.forcage_era5."""
        n = N_SURF + 1 + self.static_2d.shape[0] + self.static_3d.shape[0] + N_TFEAT
        if self.forcage is not None:
            from .forcage import N_DENSE, N_SCAL
            n += N_SCAL + N_DENSE
        return n

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.nz, self.ny, self.nx

    def split_of(self, day: int) -> str:
        if day in self.split["train_days"]:
            return "train"
        if day in self.split["val_days"]:
            return "val"
        return "test"

    def eval_levels(self) -> list[int]:
        """Niveaux retenus pour les DSO par niveau (couverture suffisante)."""
        return [k for k in range(self.nz) if self.cover[k] >= self.cfg.dso.min_level_cover]

    # ------------------------------------------------------------------ operateurs
    def pool_surface(self, S: np.ndarray) -> np.ndarray:
        """(8, ny_s*nx_s) ou (8, ny_s, nx_s) -> (8, ny, nx).

        Canaux 0-6 : variables ; canal 7 : capteur disponible. Avec pooling, moyenne des
        capteurs disponibles de chaque bloc stride x stride (tous les capteurs servent et
        le bruit de mesure diminue) ; sinon un capteur sur stride**2.
        """
        s = self.stride
        S = S.reshape(8, self.ny_s, self.nx_s)
        if not self.pool_surface_enabled:
            return S[:, ::s, ::s].copy()
        m = (S[7] > 0.5).astype(np.float32).reshape(self.ny, s, self.nx, s)
        cnt = m.sum(axis=(1, 3))
        out = np.zeros((8, self.ny, self.nx), dtype=np.float32)
        for c in range(7):
            v = S[c].reshape(self.ny, s, self.nx, s)
            out[c] = np.where(cnt > 0, (v * m).sum(axis=(1, 3)) / np.maximum(cnt, 1.0), 0.0)
        out[7] = (cnt > 0).astype(np.float32)
        return out

    @staticmethod
    def grad1d(f: torch.Tensor, h: float, dim: int) -> torch.Tensor:
        """Differences centrees a l'interieur, decentrees aux bords (equivalent np.gradient)."""
        n = f.shape[dim]
        first = (f.narrow(dim, 1, 1) - f.narrow(dim, 0, 1)) / h
        inner = (f.narrow(dim, 2, n - 2) - f.narrow(dim, 0, n - 2)) / (2.0 * h)
        last = (f.narrow(dim, n - 1, 1) - f.narrow(dim, n - 2, 1)) / h
        return torch.cat([first, inner, last], dim=dim)

    def mass_residual(self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                      w: torch.Tensor) -> torch.Tensor:
        """Residu de continuite, meme operateur que le generateur (cf. meta.json).

        r(k+1/2) = (rho w[k+1] - rho w[k]) / dz_k + 0.5 (D[k] + D[k+1])
        avec D = d(rho u)/dx + d(rho v)/dy. Tenseurs (B, nz, ny, nx) -> (B, nz-1, ny, nx).
        """
        d = self.grad1d(rho * u, self.dx, -1) + self.grad1d(rho * v, self.dy, -2)
        rw = rho * w
        return (rw[:, 1:] - rw[:, :-1]) / self.dz_t[None, :, None, None] \
            + 0.5 * (d[:, 1:] + d[:, :-1])

    def to(self, device: torch.device | str) -> "Geometry":
        self.device = torch.device(device)
        self._to_torch()
        return self

    def summary(self) -> str:
        return (f"grille {self.nz} x {self.ny} x {self.nx} (dx={self.dx:.0f} m) | "
                f"noeuds valides {self.valid.sum():,} / {self.valid.size:,} | "
                f"canaux d'entree {self.c_in}")
