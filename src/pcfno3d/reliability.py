"""Lot M2 — fiabilite du champ reconstruit (critere DSO6 ; besoins BF7, BF8, BF9 ; L4).

Pour chaque noeud du volume servi : statut (mesure / estime), indice de couverture par les
capteurs visibles, incertitude calibree sigma_u, sigma_v, sigma_w (et sigma de la vitesse),
intervalle de vitesse p10 / p50 / p90, indice et classe de fiabilite. Les choix et les
formules sont justifies dans docs/FIABILITE.md ; resume :

    valeur servie   : la passe UNIQUE avec les capteurs fournis (inchangee : la moyenne de
                      plusieurs passes degrade DSO1, cf. etape A et D6). Avec la correction M1
                      active, la passe servie est corrigee.
    sigma brut      : dispersion de N passes MASQUEES autour de la passe servie,
                      sigma = sqrt( (1/N) * somme_i (x_i - x_0)^2 ),  i = 1..N
                      chaque passe garde chaque capteur visible avec la probabilite `keep`,
                      graines fixes, caracteristiques geostrophiques/thermiques RECALCULEES
                      sur les capteurs gardes (comme le TTA), correction M1 appliquee a chaque
                      passe si elle est active.
    sigma calibre   : k * sigma brut, k par composante et par niveau, ajuste sur la VALIDATION
                      pour que |erreur| <= 1,2816 * k * sigma dans 80 % des cas.
    intervalle      : p50 = vitesse servie ; p10, p90 = vitesse -/+ 1,2816 * sigma_vitesse
                      calibre (p10 tronque a 0) — approximation gaussienne, pas des quantiles
                      empiriques de l'ensemble.

Strictement opt-in : `inference.reliability` vaut None par defaut, et rien ne change alors
(ni la sortie, ni le temps d'inference).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter

from .geometry import Geometry

METHOD = "ensemble-masquage-v1"
Z80 = 1.2815515655446004                 # quantile 90 % de la loi normale : intervalle a 80 %
COMPOSANTES = ("u", "v", "w", "speed")
STATUT_ESTIME, STATUT_MESURE, NODATA_U8 = 0, 1, 255
CLASSES = ("faible", "moyenne", "elevee")  # codes 0, 1, 2 ; 255 sous le relief
NON_CALIBRE = "non_calibre"

# histogrammes de |erreur| / sigma pour la calibration : log10 dans [-4, 4], 4000 classes
# (largeur 0,002 decade, soit ~0,5 % sur k) — la memoire ne depend pas du nombre de noeuds
_H_LO, _H_HI, _H_N = -4.0, 4.0, 4000


# ---------------------------------------------------------------------- statut, couverture
def node_status(geom: Geometry, sensor: np.ndarray, real_sensors: bool,
                dz_m: float) -> np.ndarray:
    """(nz, ny, nx) uint8 : 1 mesure, 0 estime, 255 sous le relief.

    « mesure » : donnees REELLES, capteur visible dans la meme colonne (i, j), et
    |z_noeud - z_capteur| <= dz_m (altitudes au-dessus de la mer). Sur donnees synthetiques,
    tout est « estime » : aucun capteur n'y mesure quoi que ce soit (D7).
    """
    st = np.full(geom.shape, STATUT_ESTIME, dtype=np.uint8)
    if real_sensors:
        z = np.asarray(geom.z_levels, float)[:, None, None]
        near = np.abs(z - geom.z_sensor[None]) <= dz_m
        st[near & sensor[None] & geom.valid] = STATUT_MESURE
    st[~geom.valid] = NODATA_U8
    return st


def coverage_index(geom: Geometry, sensor: np.ndarray, length_m: float, height_m: float,
                   density_ref: float) -> np.ndarray:
    """(nz, ny, nx) float32 dans [0, 1] (NaN sous le relief).

    densite D(j, i) = somme sur les capteurs visibles de exp(-d^2 / (2 L^2))   (d horizontale)
    couverture      = (1 - exp(-D / D0)) * exp(-h / H)                          (h au-dessus du sol)

    L = length_m, D0 = density_ref, H = height_m. Croit avec le nombre de capteurs proches,
    decroit avec la distance horizontale et avec la hauteur au-dessus du sol.
    """
    s = length_m / geom.dx
    # gaussian_filter normalise ses poids (somme 1) ; x 2 pi s^2 rend la somme des noyaux
    dens = gaussian_filter(sensor.astype(np.float64), sigma=s, mode="constant",
                           truncate=4.0) * (2.0 * np.pi * s * s)
    horiz = 1.0 - np.exp(-dens / density_ref)
    z_agl = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    cov = horiz[None] * np.exp(-np.clip(z_agl, 0.0, None) / height_m)
    cov = np.clip(cov, 0.0, 1.0).astype(np.float32)
    cov[~geom.valid] = np.nan
    return cov


# ---------------------------------------------------------------------- ensemble
def member_masks(sensor: np.ndarray, n: int, keep: float, seed: int) -> list[np.ndarray]:
    """N masques deterministes : chaque capteur visible est garde avec la probabilite keep."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        kept = sensor & (rng.random(sensor.shape) < keep)
        out.append(kept if kept.any() else sensor.copy())
    return out


def ensemble_sigma(served: tuple[np.ndarray, np.ndarray, np.ndarray],
                   members: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    """sigma BRUT par composante : RMS des ecarts des passes masquees a la passe servie."""
    u0, v0, w0 = (np.asarray(a, np.float64) for a in served)
    s0 = np.hypot(u0, v0)
    acc = {c: np.zeros_like(u0) for c in COMPOSANTES}
    for u, v, w in members:
        acc["u"] += (u - u0) ** 2
        acc["v"] += (v - v0) ** 2
        acc["w"] += (w - w0) ** 2
        acc["speed"] += (np.hypot(u, v) - s0) ** 2
    n = max(len(members), 1)
    return {c: np.sqrt(a / n).astype(np.float32) for c, a in acc.items()}


# ---------------------------------------------------------------------- calibration
class RatioHistogram:
    """Histogramme de |erreur| / sigma par composante et par niveau (log10)."""

    def __init__(self, nz: int) -> None:
        self.nz = nz
        self.h = {c: np.zeros((nz, _H_N + 2), dtype=np.int64) for c in COMPOSANTES}

    def add(self, comp: str, k: int, err_abs: np.ndarray, sigma: np.ndarray,
            floor: float) -> None:
        r = np.log10(np.maximum(err_abs, 1e-12) / np.maximum(sigma, floor))
        idx = np.clip(((r - _H_LO) / (_H_HI - _H_LO) * _H_N).astype(np.int64) + 1, 0, _H_N + 1)
        self.h[comp][k] += np.bincount(idx, minlength=_H_N + 2)

    @staticmethod
    def _quantile(counts: np.ndarray, q: float) -> float:
        tot = counts.sum()
        if tot == 0:
            return float("nan")
        cum = np.cumsum(counts)
        b = int(np.searchsorted(cum, q * tot))
        b = min(max(b, 1), _H_N)
        return float(10 ** (_H_LO + b * (_H_HI - _H_LO) / _H_N))       # bord superieur

    def k(self, comp: str, level: int | None = None, q: float = 0.8) -> float:
        counts = self.h[comp].sum(0) if level is None else self.h[comp][level]
        return self._quantile(counts, q) / Z80

    def inside(self, comp: str, level: int, k: float) -> tuple[int, int]:
        """(couples avec |erreur| <= 1,2816 k sigma, couples) au niveau `level`."""
        counts = self.h[comp][level]
        t = (np.log10(Z80 * k) - _H_LO) / (_H_HI - _H_LO) * _H_N
        b = int(np.clip(np.floor(t + 1e-9), 0, _H_N + 1))
        return int(counts[:b + 1].sum()), int(counts.sum())


def fit_calibration(hist: RatioHistogram, geom: Geometry, jeu: str, rcfg, n_instants: int,
                    extra: dict | None = None) -> dict:
    """k par composante et par niveau, et k global par composante. Refuse le TEST."""
    if jeu != "validation":
        raise ValueError(f"calibration de la fiabilite sur « {jeu} » refusee : elle se fait sur "
                         "la VALIDATION uniquement (le test ne sert qu'au verdict DSO6)")
    levels = list(geom.eval_levels())
    q = float(getattr(rcfg, "calibration_q", 0.8))
    k_niv = {c: [hist.k(c, lv, q) if lv in levels else None for lv in range(geom.nz)]
             for c in COMPOSANTES}
    k_glob = {c: hist.k(c, None, q) for c in COMPOSANTES}
    data = {"method": METHOD, "jeu": "validation", "instants": int(n_instants),
            "members": rcfg.members, "keep": rcfg.keep, "seed": rcfg.seed,
            "sigma_floor_m_s": rcfg.sigma_floor_m_s, "z_levels_asl_m": geom.z_levels.tolist(),
            "par_niveau": rcfg.per_level, "quantile": q, "k": k_niv, "k_global": k_glob,
            # le contrat n'a qu'UN facteur : celui de la vitesse, toutes altitudes (les k par
            # composante et par niveau, ceux qui s'appliquent, sont dans `k`)
            "scale_factor": k_glob["speed"], **(extra or {})}
    # couverture OBTENUE sur la validation avec les k retenus (>= 0,8 par construction, a la
    # largeur d'une classe d'histogramme pres) — mesuree, pas recopiee de la cible
    cal = Calibration(data)
    cov = {}
    for c in COMPOSANTES:
        ins = tot = 0
        for lv in levels:
            a, b = hist.inside(c, lv, cal.factor(c, lv))
            ins, tot = ins + a, tot + b
        cov[c] = ins / tot if tot else float("nan")
    data["couverture80_validation"] = cov
    data["coverage_80_validation"] = cov["speed"]
    return data


class Calibration:
    """Facteurs k geles (JSON) ; « non_calibre » = k 1 partout (calibrated: false)."""

    def __init__(self, data: dict | None, path: Path | None = None,
                 sha256: str | None = None) -> None:
        self.data, self.path, self.sha256 = data, path, sha256
        if data is not None:
            if data.get("jeu") != "validation":
                raise ValueError("JSON de calibration de fiabilite : « jeu » doit valoir "
                                 f"« validation », recu {data.get('jeu')!r}")
            if data.get("method") != METHOD:
                raise ValueError(f"methode inconnue : {data.get('method')!r}")

    @classmethod
    def load(cls, source: str | Path) -> "Calibration":
        if str(source) == NON_CALIBRE:
            return cls(None)
        raw = Path(source).read_bytes()
        return cls(json.loads(raw.decode("utf-8")), Path(source), hashlib.sha256(raw).hexdigest())

    @property
    def calibrated(self) -> bool:
        return self.data is not None

    def factor(self, comp: str, level: int) -> float:
        if self.data is None:
            return 1.0
        if self.data.get("par_niveau", True):
            k = self.data["k"][comp][level]
            if k is not None and np.isfinite(k):
                return float(k)
        return float(self.data["k_global"][comp])

    def apply(self, sig: dict) -> dict:
        out = {}
        for c, s in sig.items():
            kk = np.array([self.factor(c, lv) for lv in range(s.shape[0])], np.float32)
            out[c] = (s * kk[:, None, None]).astype(np.float32)
        return out

    def uncertainty_method(self, members: int) -> dict:
        """Bloc `uncertainty_method` du contrat (common.schema.json), cles exactes."""
        d = self.data
        return {"method": METHOD, "members": int(members), "calibrated": self.calibrated,
                "scale_factor": float(d["scale_factor"]) if d else None,
                "coverage_80_validation": float(d["coverage_80_validation"]) if d else None}


# ---------------------------------------------------------------------- fiabilite
def reliability_index(sigma_speed: np.ndarray, coverage: np.ndarray,
                      sigma_ref: float) -> np.ndarray:
    """r = exp(-sigma_vitesse / sigma_ref) * (0,5 + 0,5 * couverture), dans [0, 1]."""
    return (np.exp(-np.asarray(sigma_speed, np.float64) / sigma_ref)
            * (0.5 + 0.5 * np.asarray(coverage, np.float64))).astype(np.float32)


def reliability_class(r: np.ndarray, high: float, mid: float) -> np.ndarray:
    """uint8 : 2 elevee (r >= high), 1 moyenne (r >= mid), 0 faible, 255 sous le relief."""
    out = np.where(r >= high, 2, np.where(r >= mid, 1, 0)).astype(np.uint8)
    out[~np.isfinite(r)] = NODATA_U8
    return out


def speed_interval(speed: np.ndarray, sigma_speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(p10, p90) autour de la vitesse servie (p50) : -/+ 1,2816 sigma, p10 >= 0."""
    half = Z80 * sigma_speed
    return np.maximum(speed - half, 0.0).astype(np.float32), (speed + half).astype(np.float32)


@dataclass
class Reliability:
    """Tout ce que la fiabilite ajoute a un Volume (NaN / 255 sous le relief)."""

    sigma_u: np.ndarray
    sigma_v: np.ndarray
    sigma_w: np.ndarray
    sigma_speed: np.ndarray
    status: np.ndarray
    coverage: np.ndarray
    reliability: np.ndarray
    reliability_class: np.ndarray
    uncertainty: dict
    info: dict = field(default_factory=dict)

    def speed_interval(self, speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return speed_interval(speed, self.sigma_speed)


def assess(served: tuple[np.ndarray, np.ndarray, np.ndarray],
           predict: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
           sensor: np.ndarray, geom: Geometry, rcfg, calibration: Calibration,
           real_sensors: bool = False) -> Reliability:
    """Fiabilite d'un volume servi. `predict(masque)` refait une passe (u, v, w) avec ce
    masque de capteurs (caracteristiques recalculees, correction M1 comprise si active)."""
    members = [predict(m) for m in member_masks(sensor, rcfg.members, rcfg.keep, rcfg.seed)]
    sig = calibration.apply(ensemble_sigma(served, members))
    cov = coverage_index(geom, sensor, rcfg.coverage_length_m, rcfg.coverage_height_m,
                         rcfg.coverage_density_ref)
    for c in sig:
        sig[c][~geom.valid] = np.nan
    r = reliability_index(sig["speed"], cov, rcfg.sigma_ref_m_s)
    r[~geom.valid] = np.nan
    info = {"methode": METHOD, "membres": rcfg.members, "keep": rcfg.keep, "graine": rcfg.seed,
            "calibration_sha256": calibration.sha256,
            "calibration_fichier": calibration.path.name if calibration.path else None,
            "statut_reel": bool(real_sensors)}
    return Reliability(sig["u"], sig["v"], sig["w"], sig["speed"],
                       node_status(geom, sensor, real_sensors, rcfg.dz_status_m), cov, r,
                       reliability_class(r, rcfg.class_high, rcfg.class_mid),
                       calibration.uncertainty_method(rcfg.members), info)


# ---------------------------------------------------------------------- DSO6
class DSO6Accumulator:
    """Couverture de l'intervalle a 80 % et Spearman(|erreur|, sigma), par niveau et global.

    La correlation se calcule sur un echantillon tire a graine fixe (au plus `cap` couples
    par niveau et composante, reservoir) : sur 849 184 noeuds x 120 instants, garder tous les
    couples couterait des Go. La couverture, elle, est exacte (comptage complet).
    """

    def __init__(self, nz: int, cap: int = 20000, seed: int = 6) -> None:
        self.nz, self.cap = nz, cap
        self.rng = np.random.default_rng(seed)
        self.n = {c: np.zeros(nz, np.int64) for c in COMPOSANTES}
        self.inside = {c: np.zeros(nz, np.int64) for c in COMPOSANTES}
        self.pairs = {c: [None] * nz for c in COMPOSANTES}
        self.seen = {c: np.zeros(nz, np.int64) for c in COMPOSANTES}

    def add(self, comp: str, k: int, err_abs: np.ndarray, sigma: np.ndarray) -> None:
        err_abs, sigma = np.asarray(err_abs, float).ravel(), np.asarray(sigma, float).ravel()
        ok = np.isfinite(err_abs) & np.isfinite(sigma)
        err_abs, sigma = err_abs[ok], sigma[ok]
        self.n[comp][k] += err_abs.size
        self.inside[comp][k] += int((err_abs <= Z80 * sigma).sum())
        # reservoir : chaque couple vu a la meme probabilite d'etre garde
        cur = self.pairs[comp][k]
        new = np.stack([err_abs, sigma], 1)
        allp = new if cur is None else np.concatenate([cur, new])
        self.seen[comp][k] += new.shape[0]
        if allp.shape[0] > self.cap:
            allp = allp[self.rng.choice(allp.shape[0], self.cap, replace=False)]
        self.pairs[comp][k] = allp

    @staticmethod
    def _spearman(p) -> float:
        from scipy.stats import spearmanr
        if p is None or p.shape[0] < 3 or np.ptp(p[:, 1]) == 0:
            return float("nan")
        return float(spearmanr(p[:, 0], p[:, 1]).statistic)

    def summary(self, geom: Geometry, coverage_target: float = 0.8) -> dict:
        levels = geom.eval_levels()
        par_niveau = []
        for k in levels:
            rec = {"z_m": int(geom.z_levels[k])}
            for c in COMPOSANTES:
                rec[f"couverture80_{c}"] = float(self.inside[c][k] / self.n[c][k]) \
                    if self.n[c][k] else float("nan")
                rec[f"spearman_{c}"] = self._spearman(self.pairs[c][k])
            par_niveau.append(rec)
        glob = {}
        for c in COMPOSANTES:
            n = sum(int(self.n[c][k]) for k in levels)
            ins = sum(int(self.inside[c][k]) for k in levels)
            pts = [self.pairs[c][k] for k in levels if self.pairs[c][k] is not None]
            glob[f"couverture80_{c}"] = ins / n if n else float("nan")
            glob[f"spearman_{c}"] = self._spearman(np.concatenate(pts)) if pts else float("nan")
        cov_ok = all(glob[f"couverture80_{c}"] >= coverage_target for c in ("u", "v", "w"))
        corr_ok = all(glob[f"spearman_{c}"] > 0 for c in ("u", "v", "w"))
        return {"par_niveau": par_niveau, "global": glob, "dso6_couverture_ok": bool(cov_ok),
                "dso6_correlation_ok": bool(corr_ok), "dso6_conforme": bool(cov_ok and corr_ok),
                "coverage_80": glob["couverture80_speed"],
                "corr_error_sigma": glob["spearman_speed"]}
