"""BLOC 6 (2e partie) — Dataset, DataLoaders et assemblage des 25 canaux d'entree."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .features import time_geo_features
from .geometry import N_SURF, N_TFEAT, Geometry
from .normalize import Normalizer


class ANEO3D(Dataset):
    """Un exemple = un instant : surface brute, capteurs visibles, canaux temps, cible.

    En entrainement, l'instant et la fraction de capteurs visibles sont tires au hasard
    (robustesse a un reseau incomplet). En evaluation, tout est fixe par la graine, donc
    deux evaluations du meme modele sont comparables.
    """

    def __init__(self, stems, hours, doys, geom: Geometry, train: bool,
                 n_samples: int | None = None, seed: int = 0,
                 keep_eval: float | None = None, spacing_m: float | None = None,
                 reseau: np.ndarray | None = None, aneo_eval=None) -> None:
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
        # evaluation seulement : reseau REGULIER (un capteur tous les spacing_m), au lieu
        # d'une fraction aleatoire de la couche de surface synthetique, quasi continue
        self.spacing_m = spacing_m
        # evaluation seulement (lot M8, E2) : reseau FIXE dans l'espace (ny, nx), le meme a
        # chaque instant, au lieu d'un tirage par instant
        self.reseau = None if reseau is None else np.asarray(reseau, dtype=bool)
        # evaluation d'information de la regle v4 seulement : une EntreeANEO appliquee aussi hors
        # entrainement (ERA5 degradee, n capteurs simules fixes), memes tirages a chaque evaluation
        self.aneo_eval = aneo_eval

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
        # le cache du generateur reel stocke la cible en float16 (moitie du disque)
        Y = np.load(self.cache_dir / f"{stem}_Y.npy").astype(np.float32, copy=False)
        cible = None
        # forcage ERA5 de l'instant : 7 scalaires + cellules u, v (pcfno3d.forcage) ; candidat D
        # national : cellules degradees / masquees par zones a l'ENTRAINEMENT seulement (aneo_entree)
        vec = delta_e = zone = capt = None
        aneo = None
        if self.train:
            from .aneo_entree import entree_aneo
            aneo = entree_aneo(self.geom)
        elif self.aneo_eval is not None:
            aneo = self.aneo_eval
        if self.geom.forcage is not None:
            from .forcage import horodatage
            vec = self.geom.forcage.vecteur(horodatage(self.geom, stem))
            if aneo is not None:
                n_sc = len(vec) - self.geom.forcage.n_cellules
                cel, delta_e, zone = aneo.appliquer(vec[n_sc:], rng)
                vec = np.concatenate([vec[:n_sc], cel]).astype(np.float32)
        if aneo is not None:
            capt = aneo.capteurs(S, Y, rng)
        oi = getattr(self.geom, "oi", None)
        points = None
        if self.geom.cfg.mask.entree == "era5_stations" and self.train and affinage_stations(self.geom.cfg):
            if oi is not None:
                raise ValueError("innovation.actif : l'affinage sur les stations n'est pas pris en charge")
            # affinage sur les stations : vraies mesures en entree, une station en cible
            from .stations_reelles import entree_affinage
            S, sensor, cible = entree_affinage(S, self.cache_dir / f"{stem}_E.npy", self.geom,
                                               rng, stem)
        elif self.geom.cfg.mask.entree == "era5_stations":
            r = entree_era5_stations(S, self.cache_dir / f"{stem}_E.npy", self.geom,
                                     rng, self.train, keep=keep, delta_e=delta_e, zone=zone,
                                     capteurs=capt, avec_points=oi is not None)
            S, sensor = r[0], r[1]
            points = r[2] if oi is not None else None
        elif not self.train and self.reseau is not None:
            sensor = (S[7] > 0.5) & self.reseau
        elif not self.train and self.spacing_m:
            sensor = (S[7] > 0.5) & lattice_mask(S[7].shape, self.spacing_m, self.geom.dx, rng)
        else:
            sensor = (S[7] > 0.5) & (rng.random(S[7].shape) < keep)
        tfeat = time_geo_features(S, sensor, self.hours[j], self.doys[j], self.geom)
        if vec is not None:
            tfeat = np.concatenate([tfeat, vec])
        if oi is not None:
            # candidat D2 : innovations aux points de mesure (aeroports en entree + capteurs ANEO) contre
            # les cellules ERA5 de l'instant telles que le reseau les recoit (degradees a l'entrainement)
            n_sc = len(vec) - self.geom.forcage.n_cellules
            tfeat = np.concatenate([tfeat, oi.queue(S, vec[n_sc:], points["stations"],
                                                    points["capteurs"])]).astype(np.float32)
        out = (torch.from_numpy(S[:N_SURF]), torch.from_numpy(sensor.astype(np.float32)),
               torch.from_numpy(tfeat), torch.from_numpy(Y), j)
        # 6e element seulement a l'entrainement avec affinage (Trainer._epochs le lit)
        return out if cible is None else out + (torch.from_numpy(cible),)


def affinage_stations(cfg: Config) -> bool:
    """Affinage sur les stations actif : poids > 0 et dossier d'observations renseigne."""
    return cfg.loss.lambda_station > 0 and cfg.paths.stations_obs is not None


def cellules_stations(geom: Geometry) -> np.ndarray:
    """(n, 2) indices (iy, ix) des stations reelles sur la grille du modele
    (data_dir/stations.json, ecrit par le generateur) ; vide sans fichier."""
    cache = getattr(geom, "_cellules_stations", None)
    if cache is None:
        import json
        f = geom.cfg.paths.data_dir / "stations.json"
        cache = np.zeros((0, 2), dtype=int)
        if f.exists():
            st = json.loads(f.read_text(encoding="utf-8"))
            cache = np.array([[s["iy"], s["ix"]] for s in st], dtype=int).reshape(-1, 2)
        geom._cellules_stations = cache
    return cache


def entree_era5_stations(S: np.ndarray, chemin_e, geom: Geometry, rng: np.random.Generator,
                         train: bool, keep: float = 1.0, delta_e: np.ndarray | None = None,
                         zone: np.ndarray | None = None, capteurs=None,
                         avec_points: bool = False) -> tuple:
    """Entree de production du modele reel : vent ERA5 a 10 m sur un reseau regulier (tous
    les `mask.pas_era5_m`) + les stations reelles (valeur du cache = verite generee au point).

    T, rho, HR, p des pseudo-capteurs restent ceux de la couche de surface du cache (grande
    echelle) ; w = 0 partout (ni ERA5 a 10 m ni un anemometre ne le donnent).

    keep (evaluation seulement, robustesse DSO5) : part des points d'entree gardes, chaque
    pseudo-capteur ERA5 et chaque station tire independamment. 1 = entree complete (DSO1) ;
    sans effet a l'entrainement, ou l'entree est tiree comme ci-dessus.

    avec_points (candidat D2) : renvoie aussi {"stations": masque (ny, nx) des stations EN ENTREE,
    "capteurs": cellules (n, 2) des capteurs ANEO ou None}, les points de mesure de l'innovation."""
    m = geom.cfg.mask
    from .forcage import charger_e
    E = charger_e(geom, chemin_e)
    if delta_e is not None:                     # candidat D : ERA5 degradee (aneo_entree)
        E = E + delta_e
    k = max(int(round(m.pas_era5_m / geom.dx)), 1)
    o = (int(rng.integers(0, k)), int(rng.integers(0, k))) if train else (k // 2, k // 2)
    reseau = np.zeros(S.shape[1:], dtype=bool)
    reseau[o[0]::k, o[1]::k] = True
    if zone is not None:                        # candidat D : zone sans ERA5 (aneo_entree)
        reseau &= ~zone
    st = np.zeros_like(reseau)
    for iy, ix in (cellules_stations(geom) if m.stations_entree else ()):
        if not train or rng.random() < m.p_station:
            st[iy, ix] = True
    if not train and keep < 1.0:
        reseau &= rng.random(reseau.shape) < keep
        st &= rng.random(st.shape) < keep
    S = S.copy()
    st_stations = st.copy()                     # stations seules (avant les capteurs ANEO)
    if capteurs is not None:                    # candidat D : capteurs ANEO simules (aneo_entree)
        cel, cu, cv = capteurs
        st[cel[:, 0], cel[:, 1]] = True
        S[0, cel[:, 0], cel[:, 1]] = cu
        S[1, cel[:, 0], cel[:, 1]] = cv
    era5 = reseau & ~st                         # une station prime sur le pseudo-capteur
    S[0] = np.where(era5, E[0], S[0])
    S[1] = np.where(era5, E[1], S[1])
    S[2] = 0.0
    sensor = (reseau | st) & np.isfinite(S[:7]).all(axis=0)
    if not avec_points:
        return S, sensor
    return S, sensor, {"stations": st_stations & sensor,
                       "capteurs": None if capteurs is None else capteurs[0]}


def lattice_mask(shape: tuple[int, int], spacing_m: float, dx_m: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Un capteur tous les `spacing_m` metres, grille decalee au hasard (graine fixe).

    Approche la densite d'un vrai reseau (Rayen : points ANEO espaces de ~500 m a 1 km),
    bien plus epars que la couche synthetique d'entrainement. Ce n'est PAS la disposition
    reelle des capteurs, seulement sa densite.
    """
    k = max(int(round(spacing_m / dx_m)), 1)
    oy, ox = (int(o) for o in rng.integers(0, k, size=2))
    m = np.zeros(shape, dtype=bool)
    m[oy::k, ox::k] = True
    return m


def assemble_input(s_raw: torch.Tensor, sensor: torch.Tensor, tfeat: torch.Tensor,
                   geom: Geometry, norm: Normalizer) -> torch.Tensor:
    """Construit l'entree (B, c_in, nz, ny, nx) sur le device de la geometrie.

    7 mesures normalisees + 1 carte de disponibilite + 6 canaux de relief, recopies sur
    les nz niveaux ; + 3 canaux de geometrie par niveau ; + 8 canaux temps / physique.
    Avec le forcage ERA5 (paths.forcage_era5) : + 7 scalaires de forcage, puis u, v ERA5 a
    chaque noeud normalises comme la cible, en DERNIER (la sortie residuelle les lit la).
    Candidat D2 (innovation.actif) : + champ d'innovation etale u, v et distance au point de mesure
    le plus proche (pcfno3d.innovation), juste AVANT les 2 canaux ERA5 ; avec
    innovation.base_residuel = era5_oi, ces 2 derniers canaux sont ERA5 + innovation etalee.
    """
    B = s_raw.shape[0]
    nz, ny, nx = geom.shape
    s = norm.normalize_surface(s_raw, sensor)
    s2d = torch.cat([s, sensor[:, None], geom.static_2d_t.expand(B, -1, -1, -1)], dim=1)
    x2d = s2d[:, :, None].expand(-1, -1, nz, -1, -1)
    x3d = geom.static_3d_t.expand(B, -1, -1, -1, -1)
    if geom.forcage is None:
        xt = tfeat[:, :, None, None, None].expand(-1, -1, nz, ny, nx)
        return torch.cat([x2d, x3d, xt], dim=1)
    from .forcage import N_SCAL
    from .innovation import decouper_tfeat
    n = N_TFEAT + N_SCAL
    xt = tfeat[:, :n, None, None, None].expand(-1, -1, nz, ny, nx)
    cel, q = decouper_tfeat(tfeat, geom)
    era5 = geom.interp_era5.dense(cel)
    if q is not None and geom.cfg.innovation.base_residuel == "era5_oi":
        era5 = era5 + geom.oi.increments_3d(q)          # option D2 : base residuelle ERA5 + innovation
    era5 = (era5 - norm.y_mu[None, :2, None, None, None])         / norm.y_sd[None, :2, None, None, None] * geom.valid_t
    if q is None:
        return torch.cat([x2d, x3d, xt, era5.to(x2d.dtype)], dim=1)
    # candidat D2 : champ d'innovation u, v et distance au point le plus proche, AVANT les 2 canaux ERA5
    # (la sortie residuelle lit les deux derniers)
    oi = geom.oi.canaux(q, norm.y_sd, nz)
    return torch.cat([x2d, x3d, xt, oi.to(x2d.dtype), era5.to(x2d.dtype)], dim=1)


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
                keep: float | None = None, spacing_m: float | None = None,
                reseau: np.ndarray | None = None) -> tuple[DataLoader, np.ndarray]:
    """Un loader deterministe sur un sous-ensemble du test (DSO1, episodes, robustesse)."""
    te_s, te_h, te_d = splits.test
    ds = ANEO3D(te_s[idx], te_h[idx], te_d[idx], geom, train=False, seed=seed, keep_eval=keep,
                spacing_m=spacing_m, reseau=reseau)
    return DataLoader(ds, batch_size=1), te_s[idx]
