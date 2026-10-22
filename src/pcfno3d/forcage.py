"""Entree ERA5 du modele reel (Grand Tunis) : vent ERA5 a chaque noeud et forcage de l'instant.

Diagnostic du 2026-10-22 (scripts/diag_grand_tunis.py, 64 instants de validation) : ERA5
interpole aux noeuds explique deja R2 u 0,95 / v 0,97 de la verite generee ; le modele, qui ne
voyait qu'ERA5 a 10 m sur un reseau de 2,5 km, plafonnait a 0,78 / 0,73. Avec
`paths.forcage_era5` (forcage_era5.csv, et forcage_era5.npz a cote), il recoit en plus :

  - 7 scalaires de l'instant (forcage_era5.csv) : ug, vg, cisaillement u / v, rotation du vent
    avec l'altitude, stabilite, contraste terre-mer (l'heure est deja dans les 8 canaux temps) ;
  - le vent ERA5 a chaque noeud (2 canaux 3D), calcule comme le generateur
    (generate_aneo_3d._era5_au_sol) : bilineaire dans le plan, lineaire en hauteur au-dessus
    du sol, h = max(z - relief, 5 m) bornee aux hauteurs du fichier.

Les cellules ERA5 de l'instant (2 x 14 hauteurs x 4 x 4) voyagent dans le vecteur `tfeat` du
lot, apres les 8 canaux temps et les 7 scalaires : les loaders, l'evaluation et le service
gardent la meme forme de lot, et assemble_input en tire le champ dense sur le peripherique.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# (colonne de forcage_era5.csv, facteur) : des valeurs d'ordre 1, comme les 8 canaux temps
SCALAIRES = (("ug", 0.1), ("vg", 0.1), ("shear_u", 200.0), ("shear_v", 200.0),
             ("veer_deg", 1.0 / 30.0), ("stabilite", 1.0), ("dt_terre_mer_k", 0.2))
N_SCAL = len(SCALAIRES)
N_DENSE = 2                     # u, v ERA5 aux noeuds
MIN_ZREL_NODE = 5.0             # comme le generateur (pas de noeud a moins de 5 m du sol)
ECART_MAX_S = 3 * 3600          # instant a plus de 3 h de toute echeance du forcage : refuse


def secondes_utc(stamp) -> int:
    """Horodatage (naif = UTC, ou avec fuseau) -> secondes depuis 1970 en UTC."""
    t = pd.Timestamp(stamp)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return int(t.value // 10 ** 9)


def metres_par_degre(lat_deg: float) -> tuple[float, float]:
    """Comme scripts/grille_reelle.py (plan local, pas metrique constant)."""
    lat0 = math.radians(lat_deg)
    m_lon = 111412.84 * math.cos(lat0) - 93.5 * math.cos(3 * lat0)
    m_lat = 111132.92 - 559.82 * math.cos(2 * lat0) + 1.175 * math.cos(4 * lat0)
    return m_lon, m_lat


class ForcageERA5:
    """forcage_era5.csv (scalaires) + forcage_era5.npz (u, v par maille ERA5 et hauteur)."""

    def __init__(self, chemin_csv: str | Path) -> None:
        chemin_csv = Path(chemin_csv)
        cols = [c for c, _ in SCALAIRES]
        f = pd.read_csv(chemin_csv, usecols=["timestamp_utc"] + cols)
        t = pd.to_datetime(f["timestamp_utc"], utc=True).dt.tz_localize(None)
        ordre = np.argsort(t.values)
        self.t_csv = (t.values[ordre].astype("datetime64[s]").astype(np.int64))
        self.scal = np.stack([f[c].to_numpy(np.float64)[ordre] * k for c, k in SCALAIRES], 1)
        if not np.isfinite(self.scal).all():
            raise ValueError(f"{chemin_csv.name} : valeurs vides dans {cols}")
        npz = chemin_csv.with_suffix(".npz")
        if not npz.exists():
            raise FileNotFoundError(f"{npz} absent : le profil ERA5 par maille est requis")
        with np.load(npz) as d:
            self.temps = d["temps"].astype(np.int64)
            self.lat = d["lat"].astype(np.float64)
            self.lon = d["lon"].astype(np.float64)
            self.hauteurs = d["hauteurs_sol"].astype(np.float64)
            self.u = d["u"].astype(np.float32)
            self.v = d["v"].astype(np.float32)
        self.forme_cellules = (2,) + self.u.shape[1:]           # (2, nh, ny_e, nx_e)
        self.n_cellules = int(np.prod(self.forme_cellules))

    @property
    def n_extra(self) -> int:
        """Longueur ajoutee au vecteur tfeat."""
        return N_SCAL + self.n_cellules

    @staticmethod
    def _encadrer(temps: np.ndarray, s: int, nom: str) -> tuple[int, float]:
        i = int(np.clip(np.searchsorted(temps, s), 1, len(temps) - 1))
        t0, t1 = temps[i - 1], temps[i]
        if min(abs(s - t0), abs(t1 - s)) > ECART_MAX_S:
            raise ValueError(f"{nom} : aucune echeance a moins de 3 h du "
                             f"{pd.Timestamp(s, unit='s')} UTC")
        return i, float(np.clip((s - t0) / max(t1 - t0, 1), 0.0, 1.0))

    def vecteur(self, stamp) -> np.ndarray:
        """Les 7 scalaires puis les cellules u, v de l'instant (interpolation lineaire en
        temps, comme le generateur), aplaties : (n_extra,) float32."""
        s = secondes_utc(stamp)
        i, a = self._encadrer(self.t_csv, s, "forcage_era5.csv")
        scal = (1 - a) * self.scal[i - 1] + a * self.scal[i]
        j, b = self._encadrer(self.temps, s, "forcage_era5.npz")
        cu = (1 - b) * self.u[j - 1] + b * self.u[j]
        cv = (1 - b) * self.v[j - 1] + b * self.v[j]
        return np.concatenate([scal, cu.ravel(), cv.ravel()]).astype(np.float32)


class InterpERA5:
    """Operateur fixe cellules ERA5 -> noeuds du modele (bilineaire + lineaire en hauteur)."""

    def __init__(self, forcage: ForcageERA5, geom) -> None:
        real = geom.cfg.real
        if real.origin_lat is None or real.origin_lon is None:
            raise ValueError("paths.forcage_era5 demande real.origin_lat / origin_lon "
                             "(coin sud-ouest de la grille)")
        if abs(float(real.rotation_deg)) > 1e-9:
            raise ValueError("paths.forcage_era5 : grille tournee non prise en charge")
        lat0, lon0 = float(real.origin_lat), float(real.origin_lon)
        # meme projection que scripts/grille_reelle.py : pas metrique pris a la latitude
        # centrale de la grille fine
        lat_c = lat0 + geom.ny_s * geom.dy_s / 2 / metres_par_degre(lat0)[1]
        m_lon, m_lat = metres_par_degre(lat_c)
        jj, ii = np.meshgrid(np.arange(geom.ny), np.arange(geom.nx), indexing="ij")
        self.lat = lat0 + jj * geom.dy / m_lat
        self.lon = lon0 + ii * geom.dx / m_lon

        le, oe = forcage.lat, forcage.lon
        fy = np.clip((self.lat - le[0]) / (le[1] - le[0]), 0, len(le) - 1.000001)
        fx = np.clip((self.lon - oe[0]) / (oe[1] - oe[0]), 0, len(oe) - 1.000001)
        self.iy, self.ix = np.floor(fy).astype(np.int64), np.floor(fx).astype(np.int64)
        self.a, self.b = fy - self.iy, fx - self.ix

        hs = forcage.hauteurs
        z = np.asarray(geom.z_levels, np.float64)[:, None, None]
        h = np.clip(np.maximum(z - geom.zt[None], MIN_ZREL_NODE), hs[0], hs[-1])
        self.k = np.clip(np.searchsorted(hs, h) - 1, 0, len(hs) - 2).astype(np.int64)
        self.t = (h - hs[self.k]) / (hs[self.k + 1] - hs[self.k])
        self.forme_cellules = forcage.forme_cellules
        self._torch: dict = {}

    def dense_np(self, cellules: np.ndarray) -> np.ndarray:
        """(2, nh, ny_e, nx_e) -> (2, nz, ny, nx), en numpy (controles)."""
        c = np.asarray(cellules, np.float64).reshape(self.forme_cellules)
        iy, ix, a, b = self.iy, self.ix, self.a, self.b
        plan = ((1 - a) * (1 - b) * c[:, :, iy, ix] + (1 - a) * b * c[:, :, iy, ix + 1]
                + a * (1 - b) * c[:, :, iy + 1, ix] + a * b * c[:, :, iy + 1, ix + 1])
        k = np.broadcast_to(self.k[None], (2,) + self.k.shape)
        lo = np.take_along_axis(plan, k, axis=1)
        hi = np.take_along_axis(plan, k + 1, axis=1)
        return ((1 - self.t) * lo + self.t * hi).astype(np.float32)

    def _tenseurs(self, device: torch.device) -> dict:
        cle = str(device)
        if cle not in self._torch:
            def t(a, dt=torch.float32):
                return torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
            self._torch[cle] = {
                "iy": t(self.iy, torch.long), "ix": t(self.ix, torch.long),
                "a": t(self.a), "b": t(self.b), "k": t(self.k, torch.long), "t": t(self.t)}
        return self._torch[cle]

    def dense(self, cellules: torch.Tensor) -> torch.Tensor:
        """(B, n_cellules) -> (B, 2, nz, ny, nx) en m/s, sur le peripherique de l'entree."""
        p = self._tenseurs(cellules.device)
        B = cellules.shape[0]
        c = cellules.float().reshape((B,) + tuple(self.forme_cellules))
        iy, ix, a, b = p["iy"], p["ix"], p["a"], p["b"]
        plan = ((1 - a) * (1 - b) * c[..., iy, ix] + (1 - a) * b * c[..., iy, ix + 1]
                + a * (1 - b) * c[..., iy + 1, ix] + a * b * c[..., iy + 1, ix + 1])
        k = p["k"][None, None].expand(B, 2, -1, -1, -1)
        lo = torch.gather(plan, 2, k)
        hi = torch.gather(plan, 2, k + 1)
        return (1 - p["t"]) * lo + p["t"] * hi


def horodatage(geom, stem: str) -> str:
    """Instant (UTC) d'un stem du cache, lu une fois dans index.csv."""
    table = getattr(geom, "_horodatages", None)
    if table is None:
        index = pd.read_csv(geom.cfg.paths.index_csv, usecols=["stem", "timestamp"])
        table = dict(zip(index["stem"].astype(str), index["timestamp"].astype(str)))
        geom._horodatages = table
    return table[str(stem)]
