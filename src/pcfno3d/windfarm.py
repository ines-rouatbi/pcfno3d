"""Lot M5 — eolien : vent au moyeu, REWS, cisaillement, veer, puissance, parc (critere EOL1).

Post-traitement d'un Volume reconstruit (aucune passe du modele en plus). Formules, sources
et conventions dans docs/EOLIEN.md ; resume :

    hauteurs        : TOUJOURS au-dessus du SOL (hub_agl_m). Les niveaux du modele sont ASL :
                      h_k = z_k - relief(j, i), colonne la plus proche de l'eolienne. Jamais
                      de confusion sol / mer (D6 du plan).
    profil          : interpolation lineaire en hauteur entre noeuds valides ; sous le premier
                      noeud valide, loi de puissance ajustee sur les deux premiers noeuds
                      (alpha borne a [0, 1]) ; au-dessus du dernier niveau : indisponible.
    REWS            : (somme_i A_i v_i^3 / A)^(1/3), tranches horizontales du disque rotor
                      (IEC 61400-12-1:2017, pondere par l'aire, contrat rev. 2 D9).
    shear_alpha     : ln(v_haut / v_bas) / ln(h_haut / h_bas), bas et haut du rotor (D9).
    veer_deg        : direction(haut) - direction(bas), ramene a [-180, 180), positif dans
                      le sens horaire avec la hauteur (D9). Direction meteo = d'ou vient le vent.
    densite         : capteur visible le plus proche (rho de surface) ramene a la hauteur du
                      moyeu par le rapport ISA, sinon atmosphere standard ISA au moyeu.
    puissance       : v_n = REWS * (rho / rho_ref)^(1/3) (IEC 61400-12-1, normalisation en
                      densite), puis courbe de puissance ; etat sur le MEME v_n
                      (production <=> cut_in < v_n <= cut_out <=> P > 0).
    intervalles     : vitesses : p50 = passe servie, p10/p90 = -/+ 1,2816 sigma (D6), sigma =
                      dispersion des passes de l'ensemble M2 a la colonne x facteur k calibre ;
                      parc : quantiles sur les MEMBRES de l'ensemble (D9).

Strictement opt-in : `wind_farm.turbines` vaut None par defaut, et rien ne change alors.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .alerts import alert_id, iso_utc
from .geometry import Geometry
from .reliability import UNCERTAINTY_AUCUNE, Z80

RHO_REF = 1.225                 # kg/m3, densite de reference des courbes (ISA, niveau de la mer)
ALPHA_DEFAUT = 1.0 / 7.0        # exposant de repli quand le profil ne permet pas de l'estimer
HEURES_AN = 8760.0
ETATS = ("production", "arret_vent_faible", "arret_vent_fort", "indisponible")
SOURCES_DENSITE = ("atmosphere_standard", "extrapolation_surface", "mesure")
NIVEAUX_ALERTE = ("information", "avertissement", "important", "critique")
_ID = re.compile(r"^[A-Za-z0-9_-]+$")


# ---------------------------------------------------------------------- conventions
def meteo_direction(u, v):
    """Direction meteo (d'ou vient le vent), degres dans [0, 360) : (270 - atan2(v, u)) mod 360.
    u > 0 (vent soufflant vers l'est) -> 270 (vent d'ouest) ; v > 0 -> 180 (vent du sud)."""
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def wrap180(deg):
    """Angle ramene a [-180, 180) (sans arrondi : voir round_veer pour une valeur publiee)."""
    return (np.asarray(deg, float) + 180.0) % 360.0 - 180.0


def round_direction(deg, nd: int = 1):
    """Direction publiee dans [0, 360) : arrondi AVANT le modulo (contrat, regle m4).
    359,96 -> 360,0 -> 0,0 ; arrondir apres le modulo donnerait 360,0, hors du schema."""
    return np.round(np.round(np.asarray(deg, float), nd) % 360.0, nd)


def round_veer(deg, nd: int = 1):
    """Veer publie dans [-180, 180) : arrondi AVANT le modulo (regle m4).
    179,96 -> 180,0 -> -180,0 ; -179,96 -> -180,0."""
    return np.round((np.round(np.asarray(deg, float), nd) + 180.0) % 360.0 - 180.0, nd)


# --- conventions d'alerte : UNE definition, celle de M4 (pcfno3d.alerts) -----------------
# alert_id (sha8 de « kind|cible », D8) et iso_utc (iso_utc(None) leve une erreur : ce module
# ne lui passe jamais None, pas d'alerte ni de rose pour un instant sans horodatage).
# Re-exportes ici pour les appelants de windfarm ; toutes les alertes passent par _alert_meta.
def _alert_meta(kind: str, target: str, stamp) -> tuple[str, str]:
    """(id, raised_at) d'une alerte : unique point de contact avec les conventions de M4."""
    return alert_id(kind, target), iso_utc(stamp)


# ---------------------------------------------------------------------- courbe de puissance
@dataclass
class PowerCurve:
    """Courbe de puissance a la densite de reference rho_ref (table vitesse -> kW)."""

    model: str
    speed_ms: np.ndarray
    power_kw: np.ndarray
    rated_kw: float
    cut_in_ms: float
    cut_out_ms: float
    rho_ref: float = RHO_REF
    source: str = ""

    def __post_init__(self) -> None:
        self.speed_ms = np.asarray(self.speed_ms, float)
        self.power_kw = np.asarray(self.power_kw, float)
        if self.speed_ms.shape != self.power_kw.shape or self.speed_ms.size < 2:
            raise ValueError(f"courbe {self.model} : speed_ms et power_kw de meme longueur >= 2")
        if (np.diff(self.speed_ms) <= 0).any() or (self.speed_ms < 0).any():
            raise ValueError(f"courbe {self.model} : vitesses positives strictement croissantes")
        if (self.power_kw < 0).any() or self.power_kw.max() > self.rated_kw * (1 + 1e-9):
            raise ValueError(f"courbe {self.model} : puissances dans [0, rated_kw]")
        if not 0 <= self.cut_in_ms < self.cut_out_ms:
            raise ValueError(f"courbe {self.model} : 0 <= cut_in_ms < cut_out_ms")
        if self.speed_ms[-1] < self.cut_out_ms:
            raise ValueError(f"courbe {self.model} : la table doit aller jusqu'au cut-out")
        # regle d'etat unique : production <=> cut_in < v_n <= cut_out, et alors P > 0
        dedans = (self.speed_ms > self.cut_in_ms) & (self.speed_ms <= self.cut_out_ms)
        if (self.power_kw[dedans] <= 0).any():
            raise ValueError(f"courbe {self.model} : puissance nulle entre cut-in et cut-out "
                             "(l'etat « production » impose P > 0)")

    def running(self, v_n) -> np.ndarray:
        """En production : cut_in < v_n <= cut_out (v_n = vent normalise en densite)."""
        v = np.asarray(v_n, float)
        return np.isfinite(v) & (v > self.cut_in_ms) & (v <= self.cut_out_ms)

    def power(self, v_n) -> np.ndarray:
        """Puissance (kW) au vent NORMALISE v_n : la table hors production vaut 0 (meme
        regle que l'etat, `running`), et P > 0 en production."""
        v = np.asarray(v_n, float)
        p = np.interp(v, self.speed_ms, self.power_kw, left=0.0, right=0.0)
        return np.where(self.running(v), p, 0.0)

    def to_contract(self) -> dict:
        """Entree de power_curves (wind-farm.schema.json)."""
        return {"speed_ms": [float(x) for x in self.speed_ms],
                "power_kw": [float(x) for x in self.power_kw]}


def density_normalized_speed(v, rho, rho_ref: float = RHO_REF):
    """IEC 61400-12-1 (eoliennes a pas variable) : v_n = v * (rho / rho_ref)^(1/3)."""
    return np.asarray(v, float) * (np.asarray(rho, float) / rho_ref) ** (1.0 / 3.0)


# ---------------------------------------------------------------------- declaration du parc
@dataclass
class Turbine:
    id: str
    name: str
    x_m: float
    y_m: float
    hub_agl_m: float              # hauteur de moyeu AU-DESSUS DU SOL
    rotor_d_m: float
    model: str
    virtual: bool = True
    air_density: float | None = None   # densite mesuree au moyeu, si l'exploitant la fournit
    i: int = 0                    # colonne du modele utilisee (la plus proche)
    j: int = 0


@dataclass
class Farm:
    turbines: list[Turbine]
    curves: dict[str, PowerCurve]
    exemple: bool = False
    source: str | None = None
    sha256: str | None = None

    @property
    def columns(self) -> tuple[np.ndarray, np.ndarray]:
        """(jj, ii) des colonnes du modele, une par eolienne."""
        return (np.array([t.j for t in self.turbines], int),
                np.array([t.i for t in self.turbines], int))

    @property
    def rated_kw(self) -> float:
        return float(sum(self.curves[t.model].rated_kw for t in self.turbines))


def _nearest(geom: Geometry, x_m: float, y_m: float) -> tuple[int, int]:
    """Noeud le plus proche, meme regle que Volume.at_point."""
    return (int(np.clip(round(x_m / geom.dx), 0, geom.nx - 1)),
            int(np.clip(round(y_m / geom.dy), 0, geom.ny - 1)))


def farm_from_dict(data: dict, geom: Geometry, cfg=None, source: str | None = None,
                   sha256: str | None = None) -> Farm:
    """Parc depuis sa declaration (voir configs/eolien/parc_demo.yaml). Refuse ce qui est
    ambigu : identifiant invalide ou double, modele inconnu, position hors du domaine,
    rotor qui toucherait le sol, lat/lon sans domaine georeference."""
    curves = {}
    for name, c in (data.get("power_curves") or {}).items():
        curves[str(name)] = PowerCurve(
            model=str(name), speed_ms=c["speed_ms"], power_kw=c["power_kw"],
            rated_kw=float(c["rated_kw"]), cut_in_ms=float(c["cut_in_ms"]),
            cut_out_ms=float(c["cut_out_ms"]), rho_ref=float(c.get("rho_ref", RHO_REF)),
            source=str(c.get("source", "")))
    x_max, y_max = (geom.nx - 1) * geom.dx, (geom.ny - 1) * geom.dy
    turbines, seen = [], set()
    for t in data.get("turbines") or []:
        tid = str(t["id"])
        if not _ID.match(tid) or tid in seen:
            raise ValueError(f"eolienne {tid!r} : identifiant invalide ou en double "
                             "(lettres, chiffres, _ et -)")
        seen.add(tid)
        if str(t["model"]) not in curves:
            raise ValueError(f"eolienne {tid} : modele {t['model']!r} absent de power_curves")
        if "x_m" in t and "y_m" in t:
            x, y = float(t["x_m"]), float(t["y_m"])
        elif "lat" in t and "lon" in t:
            from .realdata import GeoRef
            if cfg is None:
                raise ValueError(f"eolienne {tid} : lat/lon demandent la configuration `real`")
            xa, ya = GeoRef.from_cfg(cfg).to_xy(float(t["lat"]), float(t["lon"]))
            x, y = float(xa), float(ya)
        else:
            raise ValueError(f"eolienne {tid} : position x_m/y_m ou lat/lon manquante")
        if not (-0.5 * geom.dx <= x <= x_max + 0.5 * geom.dx
                and -0.5 * geom.dy <= y <= y_max + 0.5 * geom.dy):
            raise ValueError(f"eolienne {tid} : ({x:.0f}, {y:.0f}) m hors du domaine "
                             f"[0, {x_max:.0f}] x [0, {y_max:.0f}] m")
        hub, d = float(t["hub_agl_m"]), float(t["rotor_d_m"])
        if d <= 0 or hub - d / 2 <= 0:
            raise ValueError(f"eolienne {tid} : le bas du rotor (hub_agl_m - rotor_d_m/2 = "
                             f"{hub - d / 2:.1f} m) doit etre au-dessus du sol")
        i, j = _nearest(geom, x, y)
        rho = t.get("air_density")
        turbines.append(Turbine(id=tid, name=str(t.get("name", tid)), x_m=x, y_m=y,
                                hub_agl_m=hub, rotor_d_m=d, model=str(t["model"]),
                                virtual=bool(t.get("virtual", True)),
                                air_density=float(rho) if rho is not None else None, i=i, j=j))
    if not turbines:
        raise ValueError("declaration de parc sans eolienne")
    return Farm(turbines, curves, exemple=bool(data.get("exemple", False)), source=source,
                sha256=sha256)


def load_farm(path: str | Path, geom: Geometry, cfg=None) -> Farm:
    """Declaration YAML ou JSON (le JSON est du YAML valide)."""
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw.decode("utf-8")) or {}
    return farm_from_dict(data, geom, cfg, source=Path(path).name,
                          sha256=hashlib.sha256(raw).hexdigest())


# ---------------------------------------------------------------------- profil vertical
def shear_alpha(v_low, v_high, h_low, h_high):
    """Exposant de loi de puissance entre deux hauteurs au-dessus du sol (NaN si indefini)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log(np.asarray(v_high, float) / np.asarray(v_low, float)) \
            / np.log(np.asarray(h_high, float) / np.asarray(h_low, float))
    ok = (np.asarray(v_low) > 0) & (np.asarray(v_high) > 0)
    return np.where(ok & np.isfinite(a), a, np.nan)


def profile_at(h_nodes: np.ndarray, u_col: np.ndarray, v_col: np.ndarray,
               heights: np.ndarray) -> dict:
    """Vent aux hauteurs `heights` (m AU-DESSUS DU SOL) dans une colonne.

    h_nodes (nz,) : hauteurs des niveaux au-dessus du sol de la colonne (z_k - relief) ;
    u_col, v_col : vent aux niveaux (NaN sous le relief). Retourne speed, u, v (NaN si
    indisponible), `extrapole` (au moins une hauteur sous le premier noeud valide) et
    `alpha_bas` (exposant utilise sous ce noeud, None sinon).

    La vitesse est interpolee pour elle-meme (contrat, D10), u et v servent a la direction.
    Sous le premier noeud valide : loi de puissance des deux premiers noeuds, alpha borne a
    [0, 1] (dans la couche de surface le vent croit avec la hauteur), 1/7 si incalculable.
    """
    heights = np.atleast_1d(np.asarray(heights, float))
    ok = np.isfinite(u_col) & np.isfinite(v_col) & (h_nodes > 0)
    hk, uk, vk = h_nodes[ok], np.asarray(u_col, float)[ok], np.asarray(v_col, float)[ok]
    sk = np.hypot(uk, vk)
    s = np.full(heights.shape, np.nan)
    u, v = s.copy(), s.copy()
    out = {"speed": s, "u": u, "v": v, "extrapole": False, "alpha_bas": None}
    if hk.size == 0:
        return out
    inside = (heights >= hk[0]) & (heights <= hk[-1])
    s[inside] = np.interp(heights[inside], hk, sk)
    u[inside] = np.interp(heights[inside], hk, uk)
    v[inside] = np.interp(heights[inside], hk, vk)
    below = (heights < hk[0]) & (heights > 0)
    if below.any():
        a = float(shear_alpha(sk[0], sk[1], hk[0], hk[1])) if hk.size >= 2 else float("nan")
        a = float(np.clip(a, 0.0, 1.0)) if np.isfinite(a) else ALPHA_DEFAUT
        f = (heights[below] / hk[0]) ** a
        s[below], u[below], v[below] = sk[0] * f, uk[0] * f, vk[0] * f
        out["extrapole"], out["alpha_bas"] = True, a
    return out


def rotor_strips(hub_agl_m: float, rotor_d_m: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n tranches horizontales de hauteur egale du disque rotor : (hauteurs au centre de
    chaque tranche, poids = aire de la tranche / aire du disque, somme 1).

    Aire d'une tranche entre y1 et y2 (y depuis le centre) : F(y2) - F(y1) avec
    F(y) = y sqrt(R^2 - y^2) + R^2 arcsin(y / R)  (aire du disque : F(R) - F(-R) = pi R^2).
    """
    r = rotor_d_m / 2.0
    y = np.linspace(-r, r, n + 1)
    f = y * np.sqrt(np.clip(r * r - y * y, 0.0, None)) + r * r * np.arcsin(np.clip(y / r, -1, 1))
    w = np.diff(f) / (math.pi * r * r)
    return hub_agl_m + 0.5 * (y[:-1] + y[1:]), w


def rews(speeds: np.ndarray, weights: np.ndarray) -> float:
    """Vent equivalent rotor pondere par l'aire : (somme w_i v_i^3)^(1/3), NaN si une
    tranche est indisponible."""
    s = np.asarray(speeds, float)
    if not np.isfinite(s).all():
        return float("nan")
    return float(np.sum(np.asarray(weights) * s ** 3) ** (1.0 / 3.0))


# ---------------------------------------------------------------------- densite de l'air
@dataclass
class FarmContext:
    """Ce que le calcul eolien garde d'une inference (lot M5), attache au Volume.

    columns     : (jj, ii) des colonnes des eoliennes ;
    members_uv  : (N, 2, nz, ncol) u et v des N passes de l'ensemble M2 a ces colonnes,
                  None si la fiabilite n'a pas tourne (pas d'intervalle alors) ;
    rho_surface : (ny, nx) densite mesuree en surface (canal air_density de S) ;
    sensor      : (ny, nx) capteurs visibles.
    """

    columns: tuple[np.ndarray, np.ndarray]
    members_uv: np.ndarray | None = None
    rho_surface: np.ndarray | None = None
    sensor: np.ndarray | None = None


def hub_density(turbine: Turbine, geom: Geometry, ctx: FarmContext | None,
                radius_m: float) -> tuple[float, str, dict]:
    """(rho au moyeu, source du contrat, details).

    mesure                : densite fournie dans la declaration de l'eolienne ;
    extrapolation_surface : rho du capteur VISIBLE le plus proche (<= radius_m), ramenee a
                            la hauteur du moyeu par rho_ISA(z_moyeu) / rho_ISA(z_capteur)
                            (altitudes ASL) ;
    atmosphere_standard   : rho_ISA(z_moyeu ASL) sinon.
    """
    from .realdata import isa

    z_hub = float(geom.zt[turbine.j, turbine.i]) + turbine.hub_agl_m       # ASL
    if turbine.air_density is not None:
        return float(turbine.air_density), "mesure", {"z_hub_asl_m": z_hub}
    if ctx is not None and ctx.rho_surface is not None and ctx.sensor is not None:
        r = int(math.ceil(radius_m / min(geom.dx, geom.dy)))
        j0, j1 = max(turbine.j - r, 0), min(turbine.j + r + 1, geom.ny)
        i0, i1 = max(turbine.i - r, 0), min(turbine.i + r + 1, geom.nx)
        rho = np.asarray(ctx.rho_surface, float)[j0:j1, i0:i1]
        ok = np.asarray(ctx.sensor, bool)[j0:j1, i0:i1] & np.isfinite(rho) \
            & (rho > 0.5) & (rho < 1.6)
        jj, ii = np.mgrid[j0:j1, i0:i1]
        d = np.hypot((ii - turbine.i) * geom.dx, (jj - turbine.j) * geom.dy)
        ok &= d <= radius_m
        if ok.any():
            k = np.argmin(np.where(ok, d, np.inf))
            js, is_ = int(jj.flat[k]), int(ii.flat[k])
            z_s = float(geom.z_sensor[js, is_])
            val = float(rho.flat[k]) * float(isa(z_hub)[2]) / float(isa(z_s)[2])
            return val, "extrapolation_surface", {"z_hub_asl_m": z_hub, "capteur_ij": [js, is_],
                                                   "distance_m": float(d.flat[k]),
                                                   "rho_capteur": float(rho.flat[k])}
    return float(isa(z_hub)[2]), "atmosphere_standard", {"z_hub_asl_m": z_hub}


# ---------------------------------------------------------------------- calcul du parc
@dataclass
class TurbineCalc:
    """Valeurs brutes (non arrondies) d'une eolienne a un instant."""

    turbine: Turbine
    curve: PowerCurve
    state: str
    hub_speed: float = float("nan")
    rews: float = float("nan")
    v_n: float = float("nan")          # REWS normalise en densite : etat ET puissance
    hub_direction: float | None = None
    shear_alpha: float | None = None
    veer_deg: float | None = None
    rho: float | None = None
    rho_source: str | None = None
    power_kw: float = 0.0
    hub_sigma: float | None = None
    rews_sigma: float | None = None
    power_p10: float | None = None
    power_p90: float | None = None
    k_calibration: float | None = None
    members_rews: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    def interval(self, x: float, sigma: float | None) -> dict:
        if not np.isfinite(x):
            return {"p10": None, "p50": None, "p90": None}
        if sigma is None:
            return {"p10": None, "p50": _r(x), "p90": None}
        return {"p10": _r(max(0.0, x - Z80 * sigma)), "p50": _r(x), "p90": _r(x + Z80 * sigma)}

    def to_contract(self) -> dict:
        t, c = self.turbine, self.curve
        dispo = self.state != "indisponible"
        return {"id": t.id, "name": t.name, "x_m": t.x_m, "y_m": t.y_m, "virtual": t.virtual,
                "model": t.model, "hub_agl_m": t.hub_agl_m, "rotor_d_m": t.rotor_d_m,
                "rated_kw": c.rated_kw, "cut_in_ms": c.cut_in_ms, "cut_out_ms": c.cut_out_ms,
                "hub_speed": self.interval(self.hub_speed, self.hub_sigma),
                "rews": self.interval(self.rews, self.rews_sigma),
                "shear_alpha": _r(self.shear_alpha, 3),
                "veer_deg": _rv(self.veer_deg),
                "power_kw": {"p10": _r(self.power_p10, 1), "p50": _r(self.power_kw, 1),
                             "p90": _r(self.power_p90, 1)} if dispo
                else {"p10": None, "p50": None, "p90": None},
                "state": self.state,
                "air_density": _r(self.rho, 4) if dispo else None,
                "air_density_source": self.rho_source if dispo else None}


def _r(x, nd: int = 2):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _rv(x):
    """Veer publie (arrondi avant le modulo), None si inconnu."""
    return float(round_veer(x)) if x is not None and np.isfinite(x) else None


def _rd(x):
    """Direction publiee dans [0, 360) (arrondi avant le modulo), None si inconnue."""
    return float(round_direction(x)) if x is not None and np.isfinite(x) else None


def _state(v_n: float, curve: PowerCurve) -> str:
    """REGLE UNIQUE (etat et puissance) sur le REWS normalise en densite v_n :
    v_n <= cut-in -> arret_vent_faible, v_n > cut-out -> arret_vent_fort, sinon production,
    et alors P = courbe(v_n) > 0 (garanti par PowerCurve)."""
    if not np.isfinite(v_n):
        return "indisponible"
    if v_n <= curve.cut_in_ms:
        return "arret_vent_faible"
    if v_n > curve.cut_out_ms:
        return "arret_vent_fort"
    return "production"


def _power(rews_, rho: float, curve: PowerCurve) -> np.ndarray:
    """Puissance (kW) = courbe(REWS normalise en densite), 0 hors production (meme regle
    que _state). Vectorise sur rews_."""
    return curve.power(density_normalized_speed(rews_, rho, curve.rho_ref))


def _k_at(calibration, geom: Geometry, z_asl: float) -> float:
    """Facteur k de la vitesse (M2) interpole lineairement en altitude ASL."""
    if calibration is None:
        return 1.0
    z = np.asarray(geom.z_levels, float)
    ks = np.array([calibration.factor("speed", k) for k in range(len(z))], float)
    return float(np.interp(z_asl, z, ks))


def turbine_calc(volume, turbine: Turbine, curve: PowerCurve, wcfg, ctx: FarmContext | None,
                 col: int, calibration=None) -> TurbineCalc:
    """Une eolienne a un instant (voir l'en-tete du module pour les formules)."""
    geom = volume.geom
    j, i = turbine.j, turbine.i
    h_nodes = np.asarray(geom.z_levels, float) - float(geom.zt[j, i])
    r = turbine.rotor_d_m / 2.0
    hs, w = rotor_strips(turbine.hub_agl_m, turbine.rotor_d_m, wcfg.rews_segments)
    heights = np.concatenate([[turbine.hub_agl_m, turbine.hub_agl_m - r,
                               turbine.hub_agl_m + r], hs])
    # seuls les noeuds VALIDES comptent (au-dessus du relief) : jamais une valeur sous le sol
    ok = geom.valid[:, j, i]
    prof = profile_at(h_nodes, np.where(ok, volume.u[:, j, i], np.nan),
                      np.where(ok, volume.v[:, j, i], np.nan), heights)
    s = prof["speed"]
    hub, rw = float(s[0]), rews(s[3:], w)
    info = {"noeud": {"i": i, "j": j, "x_m": i * geom.dx, "y_m": j * geom.dy},
            "z_terrain_m": round(float(geom.zt[j, i]), 2),
            "hub_asl_m": round(float(geom.zt[j, i]) + turbine.hub_agl_m, 2),
            "extrapolation_sous_premier_noeud": prof["extrapole"],
            "alpha_extrapolation": prof["alpha_bas"]}
    calc = TurbineCalc(turbine, curve, "indisponible", hub_speed=hub, rews=rw, info=info)
    if not (np.isfinite(hub) and np.isfinite(rw)):
        calc.hub_speed, calc.rews = float("nan"), float("nan")
        return calc
    calc.rho, calc.rho_source, info["densite"] = hub_density(turbine, geom, ctx,
                                                             wcfg.rho_radius_m)
    calc.v_n = float(density_normalized_speed(rw, calc.rho, curve.rho_ref))
    calc.state = _state(calc.v_n, curve)
    info["rews_normalise_ms"] = round(calc.v_n, 3)
    if hub >= wcfg.calm_ms:
        calc.hub_direction = float(meteo_direction(prof["u"][0], prof["v"][0]))
    a = float(shear_alpha(s[1], s[2], turbine.hub_agl_m - r, turbine.hub_agl_m + r))
    calc.shear_alpha = a if np.isfinite(a) else None
    if min(s[1], s[2]) >= wcfg.calm_ms:
        d_lo = meteo_direction(prof["u"][1], prof["v"][1])
        d_hi = meteo_direction(prof["u"][2], prof["v"][2])
        calc.veer_deg = float(wrap180(d_hi - d_lo))       # brut ; publie par round_veer
    calc.power_kw = float(_power(rw, calc.rho, curve))

    # ensemble M2 : sigma brut a la colonne x k calibre, puis intervalles (D6)
    if ctx is not None and ctx.members_uv is not None:
        mh, mr = [], []
        for m in range(ctx.members_uv.shape[0]):
            pm = profile_at(h_nodes, np.where(ok, ctx.members_uv[m, 0, :, col], np.nan),
                            np.where(ok, ctx.members_uv[m, 1, :, col], np.nan), heights)
            mh.append(pm["speed"][0])
            mr.append(rews(pm["speed"][3:], w))
        mh, mr = np.asarray(mh, float), np.asarray(mr, float)
        k = _k_at(calibration, geom, float(geom.zt[j, i]) + turbine.hub_agl_m)
        fin = np.isfinite(mh) & np.isfinite(mr)
        if fin.any():
            calc.k_calibration = k
            calc.hub_sigma = k * float(np.sqrt(np.mean((mh[fin] - hub) ** 2)))
            calc.rews_sigma = k * float(np.sqrt(np.mean((mr[fin] - rw) ** 2)))
            # membres ramenes a la dispersion calibree : x_0 + k (x_m - x_0)
            calc.members_rews = np.maximum(rw + k * (mr[fin] - rw), 0.0)
            # intervalle de puissance : image de l'intervalle du REWS par la courbe (meme
            # regle d'etat), p50 = passe servie
            t = np.linspace(-1.0, 1.0, 201)
            p = _power(np.maximum(rw + t * Z80 * calc.rews_sigma, 0.0), calc.rho, curve)
            calc.power_p10 = float(min(p.min(), calc.power_kw))
            calc.power_p90 = float(max(p.max(), calc.power_kw))
    return calc


@dataclass
class FarmCalc:
    """Le parc a un instant : eoliennes, puissance du parc, alertes."""

    farm: Farm
    turbines: list[TurbineCalc]
    timestamp: str | None
    power_kw: float
    power_p10: float | None
    power_p90: float | None
    n_members: int
    uncertainty: dict

    @property
    def n_production(self) -> int:
        return sum(t.state == "production" for t in self.turbines)

    @property
    def n_arret(self) -> int:
        return sum(t.state in ("arret_vent_faible", "arret_vent_fort") for t in self.turbines)

    def document(self) -> dict:
        """Objet de wind-farm.schema.json (contrat rev. 2), cles exactes."""
        used = sorted({t.turbine.model for t in self.turbines})
        return {"turbines": [t.to_contract() for t in self.turbines],
                "farm": {"power_kw": {"p10": _r(self.power_p10, 1), "p50": _r(self.power_kw, 1),
                                      "p90": _r(self.power_p90, 1)},
                         "rated_kw": self.farm.rated_kw, "n_production": self.n_production,
                         "n_arret": self.n_arret},
                "power_curves": {m: self.farm.curves[m].to_contract() for m in used}}

    def details(self) -> list[dict]:
        out = []
        for t in self.turbines:
            d = {"id": t.turbine.id, **t.info,
                 "hub_direction_deg": _rd(t.hub_direction),
                 "hub_sigma_ms": _r(t.hub_sigma, 3), "rews_sigma_ms": _r(t.rews_sigma, 3),
                 "k_calibration": _r(t.k_calibration, 4)}
            out.append(d)
        return out

    def alerts(self, wcfg) -> list[dict]:
        """Alertes M5 au format alerts.schema.json (vent_fort_moyeu, cisaillement, D8).
        `raised_at` = instant du volume : le journal de M4 le remplace par la premiere levee.
        Sans horodatage, aucune alerte n'est produite (on n'invente pas d'instant)."""
        if self.timestamp is None:
            raise ValueError("alertes impossibles : instant sans horodatage")
        items = []
        for t in self.turbines:
            if t.state == "indisponible":
                continue
            tb, c = t.turbine, t.curve
            loc = {"x_m": tb.x_m, "y_m": tb.y_m}
            # meme grandeur que l'etat : REWS normalise en densite (p90 si M2 a tourne)
            f = t.v_n / t.rews if t.rews > 0 else 1.0
            v90 = t.v_n + Z80 * t.rews_sigma * f if t.rews_sigma is not None else t.v_n
            lvl = thr = None
            if t.hub_speed >= wcfg.high_wind_critical_ms:
                lvl, val, thr, rec = "critique", t.hub_speed, wcfg.high_wind_critical_ms, (
                    "Vent extreme au moyeu : verifier la mise en securite du parc et suspendre "
                    "toute intervention sur site")
            elif t.state == "arret_vent_fort":
                lvl, val, thr, rec = "important", t.v_n, c.cut_out_ms, (
                    "Eolienne a l'arret (vent fort) : verifier la mise en drapeau et anticiper "
                    "la perte de production")
            elif v90 >= wcfg.high_wind_warn_frac * c.cut_out_ms:
                lvl, val, thr, rec = ("avertissement", t.v_n,
                                      wcfg.high_wind_warn_frac * c.cut_out_ms,
                                      f"Surveiller : arret de securite possible (cut-out "
                                      f"{c.cut_out_ms:g} m/s)")
            if lvl:
                items.append(_alert("vent_fort_moyeu", tb, lvl, loc, round(val, 2), thr, "m/s",
                                    f"Vent fort au moyeu de {tb.id}",
                                    f"Vent au moyeu {t.hub_speed:.1f} m/s a {tb.hub_agl_m:g} m "
                                    f"au-dessus du sol ; vent equivalent rotor normalise "
                                    f"{t.v_n:.1f} m/s (p90 {v90:.1f}) ; arret au-dela de "
                                    f"{c.cut_out_ms:g} m/s", rec, self.timestamp))
            a = t.shear_alpha
            if a is not None:
                lvl = None
                for niveau, seuil in (("important", wcfg.shear_important),
                                      ("avertissement", wcfg.shear_warn),
                                      ("information", wcfg.shear_info)):
                    if a > seuil:
                        lvl, thr = niveau, seuil
                        break
                if lvl:
                    rec = {"information": "Aucune action ; a suivre s'il persiste (charges de "
                                          "fatigue)",
                           "avertissement": "Verifier les charges de fatigue des pales : le "
                                            "cisaillement de conception IEC 61400-1 est 0,2",
                           "important": "Envisager un fonctionnement bride et une inspection "
                                        "des pales"}[lvl]
                    items.append(_alert("cisaillement", tb, lvl, loc, round(a, 3), thr, None,
                                        f"Cisaillement marque sur {tb.id}",
                                        f"Exposant de cisaillement {a:.2f} > {thr:g} entre "
                                        f"{tb.hub_agl_m - tb.rotor_d_m / 2:g} et "
                                        f"{tb.hub_agl_m + tb.rotor_d_m / 2:g} m au-dessus du sol",
                                        rec, self.timestamp))
        return items

    def to_json(self, wcfg) -> dict:
        # sans horodatage, pas d'alerte (raised_at inconnu) : liste vide et explication
        alertes = self.alerts(wcfg) if self.timestamp is not None else []
        out = {"timestamp_utc": self.timestamp, "wind_farm": self.document(),
               "alerts": {"items": alertes}, "details": self.details(),
               "uncertainty": self.uncertainty, "membres": self.n_members,
               "parc": {"source": self.farm.source, "sha256": self.farm.sha256,
                        "exemple": self.farm.exemple}}
        if self.timestamp is None:
            out["alertes_non_produites"] = "instant sans horodatage"
        return out


def _alert(kind, tb, level, loc, value, threshold, unit, title, cause, rec, stamp) -> dict:
    aid, raised = _alert_meta(kind, tb.id, stamp)
    return {"id": aid, "level": level, "kind": kind, "target": tb.id,
            "title": title, "cause": cause, "recommendation": rec, "raised_at": raised,
            "status": "ouverte", "station_id": None, "turbine_id": tb.id, "location": loc,
            "value": float(value) if value is not None else None,
            "threshold": float(threshold) if threshold is not None else None, "unit": unit}


def compute_farm(volume, farm: Farm, wcfg, calibration=None,
                 context: FarmContext | None = None) -> FarmCalc:
    """Le parc sur un Volume. `context` : par defaut celui attache au volume par
    WindFieldModel.predict (densite de surface, capteurs, membres de l'ensemble)."""
    ctx = context if context is not None else getattr(volume, "farm_context", None)
    calcs = [turbine_calc(volume, t, farm.curves[t.model], wcfg, ctx, col, calibration)
             for col, t in enumerate(farm.turbines)]
    total = float(sum(c.power_kw for c in calcs))
    members = ctx.members_uv.shape[0] if ctx is not None and ctx.members_uv is not None else 0
    p10 = p90 = None
    actives = [c for c in calcs if c.state != "indisponible"]
    if members and actives and all(c.members_rews is not None and c.members_rews.size == members
                                   for c in actives):
        # quantiles du parc sur les MEMBRES (D9) : puissance du parc par membre, pas la somme
        # des quantiles des eoliennes (les erreurs des eoliennes voisines sont correlees)
        per = np.zeros(members)
        for c in actives:
            per += _power(c.members_rews, c.rho, c.curve)
        p10 = float(min(np.quantile(per, 0.1), total))
        p90 = float(max(np.quantile(per, 0.9), total))
    if members and calibration is not None:
        unc = calibration.uncertainty_method(members)
    else:
        unc = dict(UNCERTAINTY_AUCUNE)
    return FarmCalc(farm, calcs, volume.timestamp, total, p10, p90, members, unc)


# ---------------------------------------------------------------------- KPI du domaine (D10)
def hub_wind_plane(volume, hub_agl_m: float) -> np.ndarray:
    """(ny, nx) vitesse a hub_agl_m AU-DESSUS DU SOL, interpolation lineaire de speed par
    colonne entre deux noeuds valides ; NaN si la hauteur n'est pas encadree."""
    geom = volume.geom
    h = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    s = np.where(geom.valid, volume.speed, np.nan)
    out = np.full((geom.ny, geom.nx), np.nan)
    for k in range(geom.nz - 1):
        a, b, sa, sb = h[k], h[k + 1], s[k], s[k + 1]
        m = np.isfinite(sa) & np.isfinite(sb) & (a > 0) & (a <= hub_agl_m) \
            & (hub_agl_m <= b) & np.isnan(out)
        f = (hub_agl_m - a[m]) / (b[m] - a[m])
        out[m] = sa[m] + f * (sb[m] - sa[m])
    return out


def hub_wind_kpi(volume, hub_agl_m: float) -> dict:
    """KPI hub_wind_* de snapshot.kpi : moyenne et max sur le DOMAINE (D10), null si aucune
    colonne n'encadre la hauteur."""
    p = hub_wind_plane(volume, hub_agl_m)
    f = p[np.isfinite(p)]
    return {"hub_wind_mean_ms": _r(f.mean()) if f.size else None,
            "hub_wind_max_ms": _r(f.max()) if f.size else None,
            "hub_height_agl_m": float(hub_agl_m)}


def farm_alerts(volume, wf) -> list[dict]:
    """Alertes M5 brutes d'un volume (vent_fort_moyeu, cisaillement), entree externe du
    moteur d'alertes M4 ; [] sans parc declare ou sans horodatage."""
    if getattr(wf, "farm", None) is None or volume.timestamp is None:
        return []
    wcfg = wf.cfg.wind_farm
    return compute_farm(volume, wf.farm, wcfg, calibration=wf.reliability).alerts(wcfg)


def farm_report(volume, wf) -> dict:
    """Reponse de /api/wind-farm et de l'outil MCP production_eolienne."""
    wcfg = wf.cfg.wind_farm
    out = compute_farm(volume, wf.farm, wcfg, calibration=wf.reliability).to_json(wcfg)
    out["kpi"] = hub_wind_kpi(volume, wcfg.hub_height_agl_m)
    return out


# ---------------------------------------------------------------------- Weibull, AEP, rose
def fit_weibull(speeds) -> tuple[float, float] | None:
    """(k, c) par maximum de vraisemblance sur les vitesses > 0 ; None si impossible.

    k est la racine de g(k) = sum(x^k ln x) / sum(x^k) - 1/k - mean(ln x) (croissante),
    c = (mean(x^k))^(1/k). Vitesses normalisees par leur maximum (pas de debordement).
    """
    from scipy.optimize import brentq

    x = np.asarray(speeds, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 2 or np.ptp(x) == 0:
        return None
    xm = x.max()
    y = x / xm
    ly = np.log(y)

    def g(k):
        yk = y ** k
        return np.sum(yk * ly) / np.sum(yk) - 1.0 / k - ly.mean()

    try:
        k = brentq(g, 0.05, 50.0, xtol=1e-10)
    except ValueError:
        return None
    c = xm * float(np.mean(y ** k)) ** (1.0 / k)
    return float(k), float(c)


def weibull_aep(curve: PowerCurve, k: float, c: float, rho: float = RHO_REF,
                hours: float = HEURES_AN) -> tuple[float, float]:
    """(AEP en MWh, facteur de capacite) : esperance de la courbe (normalisee en densite)
    sous la loi de Weibull (k, c), integree numeriquement (pas 0,01 m/s jusqu'a 60 m/s)."""
    v = np.linspace(0.0, 60.0, 6001)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = (k / c) * (v / c) ** (k - 1) * np.exp(-((v / c) ** k))
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    p = curve.power(density_normalized_speed(v, rho, curve.rho_ref))
    integrate = getattr(np, "trapezoid", None) or np.trapz
    mean_kw = float(integrate(p * pdf, v))
    return mean_kw * hours / 1000.0, mean_kw / curve.rated_kw


def wind_rose(speeds, directions, n_sectors: int, edges, calm_ms: float = 0.2
              ) -> tuple[int, int, list[dict]]:
    """(n_valid, n_calmes, secteurs de windrose.schema.json).

    Secteurs de largeur egale, le premier centre sur 0 ; dir_center_deg = PROVENANCE. La
    direction est arrondie a 0,1 degre AVANT le modulo et le classement (regle m4), comme
    elle est publiee. La derniere classe de vitesse est ouverte.

    n_valid = instants de vitesse connue, calmes compris (< calm_ms, direction indefinie).
    Le schema n'a pas de champ « calme » : chaque calme est reparti A PARTS EGALES entre les
    secteurs, dans sa classe de vitesse (la premiere). Somme des frequences = 1 des que
    n_valid > 0 ; n_valid = 0 ne donne pas de rose (voir windrose_document).
    """
    s, d = np.asarray(speeds, float), np.asarray(directions, float)
    calme = np.isfinite(s) & (s < calm_ms)
    dirige = np.isfinite(s) & np.isfinite(d) & ~calme
    width = 360.0 / n_sectors
    edges = np.asarray(edges, float)
    nb = edges.size - 1

    def classe(v):
        return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, nb - 1)

    dd = round_direction(d[dirige], 1)
    sec = np.floor(((dd + width / 2.0) % 360.0) / width).astype(int) % n_sectors
    counts = np.zeros((n_sectors, nb), float)
    np.add.at(counts, (sec, classe(s[dirige])), 1.0)
    nc = int(calme.sum())
    if nc:
        np.add.at(counts, (slice(None), classe(s[calme])), 1.0 / n_sectors)
    n = int(dirige.sum()) + nc
    out = []
    for k in range(n_sectors):
        bins = counts[k] / n if n else np.zeros(nb)
        out.append({"dir_center_deg": round(k * width, 6),
                    "frequency": round(float(counts[k].sum() / n), 4) if n else 0.0,
                    "speed_bins": [round(float(x), 4) for x in bins]})
    return n, nc, out


def windrose_document(site_id: str, x_m: float, y_m: float, hub_agl_m: float, times,
                      speeds, directions, wcfg, curve: PowerCurve | None = None,
                      rho: float | None = None) -> dict | None:
    """Objet de windrose.schema.json pour un site ou une eolienne, sur une serie d'instants
    (rose, Weibull, AEP et facteur de capacite si une courbe est fournie).

    None si aucun instant n'est valide : le schema impose des frequences numeriques de somme
    1 (regle m8) et n'admet ni null ni rose vide ; le fichier n'est alors pas produit."""
    n, _, sectors = wind_rose(speeds, directions, wcfg.sectors, wcfg.speed_bin_edges_ms,
                              wcfg.calm_ms)
    if n == 0:
        return None
    times = list(times)
    if any(x is None for x in times):
        raise ValueError("rose des vents : instant sans horodatage dans la serie")
    t = pd.to_datetime(pd.Series(times))
    fit = fit_weibull(speeds)
    weibull = None
    if fit is not None:
        k, c = fit
        aep = cf = None
        if curve is not None:
            aep, cf = weibull_aep(curve, k, c, rho if rho is not None else curve.rho_ref)
        weibull = {"k": round(k, 3), "c_ms": round(c, 3),
                   "aep_mwh": round(aep, 1) if aep is not None else None,
                   "capacity_factor": round(cf, 4) if cf is not None else None}
    return {"site_id": site_id, "x_m": float(x_m), "y_m": float(y_m),
            "hub_agl_m": float(hub_agl_m),
            "period": {"from": iso_utc(t.min()), "to": iso_utc(t.max())},
            "n_valid": n, "sector_width_deg": 360.0 / wcfg.sectors,
            "speed_bin_edges_ms": [float(e) for e in wcfg.speed_bin_edges_ms],
            "sectors": sectors, "weibull": weibull}


def farm_windroses(wf, farm: Farm, stems, keep: float = 1.0
                   ) -> tuple[dict[str, dict], dict[str, str]]:
    """Roses des vents et Weibull des eoliennes declarees sur une serie d'instants en
    cache (une passe servie par instant, sans ensemble).

    Retourne (documents par eolienne, eoliennes ecartees -> raison) : une eolienne sans
    aucun instant valide (indisponible sur toute la serie) n'a pas de fichier."""
    wcfg = wf.cfg.wind_farm
    rows: dict[str, list] = {t.id: [] for t in farm.turbines}
    times = []
    for stem in stems:
        S, hour, doy, stamp = wf.surface_from_cache(stem)
        vol = wf.predict(S, hour, doy, timestamp=stamp, keep=keep, reliability=False)
        ctx = FarmContext(farm.columns, None, S[4], S[7] > 0.5)
        calc = compute_farm(vol, farm, wcfg, context=ctx)
        times.append(stamp)
        for c in calc.turbines:
            rows[c.turbine.id].append((c.hub_speed, c.hub_direction if c.hub_direction
                                       is not None else np.nan,
                                       c.rho if c.rho is not None else np.nan))
    out, ecartees = {}, {}
    for t in farm.turbines:
        a = np.array(rows[t.id], float).reshape(-1, 3)
        rho = float(np.nanmean(a[:, 2])) if np.isfinite(a[:, 2]).any() else None
        doc = windrose_document(t.id, t.x_m, t.y_m, t.hub_agl_m, times, a[:, 0], a[:, 1],
                                wcfg, farm.curves[t.model], rho)
        if doc is None:
            ecartees[t.id] = (f"aucun instant valide sur {len(times)} (eolienne indisponible) "
                              ": le contrat impose une somme des frequences egale a 1, pas "
                              "de fichier")
        else:
            out[t.id] = doc
    return out, ecartees


# ---------------------------------------------------------------------- EOL1
def truth_volume(cfg, geom: Geometry, stem: str, timestamp: str | None = None,
                 y: np.ndarray | None = None):
    """Verite 3D du generateur (cache Y) sous forme de Volume, NaN sous le relief et la ou
    la cible est inconnue."""
    from .inference import Volume

    if y is None:
        y = np.load(cfg.paths.cache_dir / f"{stem}_Y.npy")
    ok = geom.valid & np.isfinite(y[0]) & np.isfinite(y[1])
    u, v, w = (np.where(ok, y[c], np.nan).astype(np.float32) for c in range(3))
    return Volume(u, v, w, geom, timestamp=timestamp)


def eol1_instants(cfg) -> list[str]:
    """Instants synoptiques de TEST (ceux de DSO1), tous les instants de test sinon."""
    from .cache import load_index
    from .dataset import is_synoptic

    idx = load_index(cfg)
    te = idx[idx["split"] == "test"]
    syn = te[[is_synoptic(t) for t in te["timestamp"]]]
    return [str(s) for s in (syn if len(syn) else te)["stem"]]


def eol1_dataset(cfg, geom: Geometry, keep: float, seed: int = 2000):
    """Le MEME jeu que DSO1 (Evaluator.run) : instants synoptiques de test, ANEO3D en
    evaluation, graine 2000. Le masque de l'instant en position p de cette selection est
    tire par default_rng(seed + p) : identique a DSO1 pour keep = mask.keep_eval."""
    from .cache import load_index
    from .dataset import ANEO3D, is_synoptic, make_splits

    splits = make_splits(cfg, load_index(cfg))
    te_s, te_h, te_d = splits.test
    syn = np.array([is_synoptic(splits.stamp_of[x]) for x in te_s], dtype=bool)
    idx = np.nonzero(syn)[0] if syn.any() else np.arange(len(te_s))
    return ANEO3D(te_s[idx], te_h[idx], te_d[idx], geom, train=False, seed=seed,
                  keep_eval=keep)


def eol1(wf, farm: Farm, keep: float = 1.0, seed: int = 2000,
         n_instants: int | None = None) -> dict:
    """EOL1 : production du parc depuis le champ RECONSTRUIT vs depuis la VERITE 3D du
    generateur, sur les instants de test.

        ecart = |E_rec - E_vrai| / E_vrai,   E = somme sur les instants de P_parc(t)

    Meme parc, meme densite (surface de l'instant, capteurs visibles), meme courbe : seul
    le champ de vent differe. Conforme si ecart < wind_farm.eol1_max_rel (5 %). Diagnostics :
    nMAE par instant (|P_rec - P_vrai| / puissance installee), RMSE du vent au moyeu, accord
    des etats.

    Instants et capteurs visibles : exactement ceux de l'evaluation DSO1 (`eol1_dataset` :
    ANEO3D, graine 2000, masque de la POSITION dans la selection synoptique, conservee meme
    si `n_instants` sous-echantillonne). La passe est celle de production (`predict`), avec
    ce masque comme seul jeu de capteurs.
    """
    wcfg = wf.cfg.wind_farm
    ds = eol1_dataset(wf.cfg, wf.geom, keep, seed)
    pos = list(range(len(ds)))
    if n_instants is not None and len(pos) > n_instants:
        pos = [int(x) for x in np.unique(np.linspace(0, len(pos) - 1, n_instants).astype(int))]
    if not pos:
        raise ValueError("aucun instant de test en cache pour EOL1")
    rows, hub_err, etats = [], {t.id: [] for t in farm.turbines}, []
    for p in pos:
        _s, sensor, _tf, y, j = ds[p]
        stem = str(ds.stems[j])
        S, hour, doy, stamp = wf.surface_from_cache(stem)
        S = np.array(S, copy=True)
        S[7] = sensor.numpy()                     # masque DSO1 ; keep=1.0 le garde tel quel
        vol = wf.predict(S, hour, doy, timestamp=stamp, keep=1.0, reliability=False)
        vis = S[7] > 0.5
        ctx = FarmContext(farm.columns, None, S[4], vis)
        rec = compute_farm(vol, farm, wcfg, context=ctx)
        vrai = compute_farm(truth_volume(wf.cfg, wf.geom, stem, stamp, y=y.numpy()), farm,
                            wcfg, context=ctx)
        rows.append({"stem": stem, "timestamp_utc": stamp, "position_dso1": p,
                     "capteurs": int(vis.sum()), "p_rec_kw": rec.power_kw,
                     "p_vrai_kw": vrai.power_kw, "n_production_rec": rec.n_production,
                     "n_production_vrai": vrai.n_production})
        for a, b in zip(rec.turbines, vrai.turbines):
            if np.isfinite(a.hub_speed) and np.isfinite(b.hub_speed):
                hub_err[a.turbine.id].append(a.hub_speed - b.hub_speed)
            etats.append(a.state == b.state)
    e_rec = float(sum(r["p_rec_kw"] for r in rows))
    e_vrai = float(sum(r["p_vrai_kw"] for r in rows))
    ecart = abs(e_rec - e_vrai) / e_vrai if e_vrai > 0 else None
    rated = farm.rated_kw
    nmae = float(np.mean([abs(r["p_rec_kw"] - r["p_vrai_kw"]) for r in rows])) / rated
    return {"critere": "EOL1", "definition": "|E_rec - E_vrai| / E_vrai, E = somme des "
            "puissances du parc sur les instants de test (champ reconstruit vs verite 3D)",
            "instants": len(rows), "capteurs_visibles": keep, "graine": seed,
            "masques": "ceux de DSO1 (ANEO3D, graine 2000 + position synoptique)",
            "parc": {"source": farm.source, "sha256": farm.sha256, "exemple": farm.exemple,
                     "eoliennes": len(farm.turbines), "rated_kw": rated},
            "somme_p_rec_kw": e_rec, "somme_p_vrai_kw": e_vrai,
            "ecart_relatif": ecart, "seuil": wcfg.eol1_max_rel,
            "eol1_conforme": (ecart < wcfg.eol1_max_rel) if ecart is not None else None,
            "nmae_puissance": nmae,
            "rmse_vent_moyeu_ms": {k: (float(np.sqrt(np.mean(np.square(v)))) if v else None)
                                   for k, v in hub_err.items()},
            "accord_etats": float(np.mean(etats)) if etats else None,
            "par_instant": rows}
