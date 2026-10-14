"""Lot M1 — correction POST-MODELE du vent moyen par niveau (DSO1 a 650-1000 m).

Portage fidele des cellules du notebook `fno3d_v7.ipynb` (Colab) ajoutees les 12-13/10/2026,
dont les copies exactes sont dans tests/reference/notebook_v7_correction/ :

    BLOC 10f  caracteristiques d'instant (13), regression ridge standardisee
    BLOC 10g  estimateur geostrophique v2 (Tv = p/(Rd rho), altitude dans le plan, rho_sl)
    BLOC 10h  caracteristiques de couche limite (30), choix d'alpha sur la VALIDATION, JSON
    BLOC 12   etape 2b : application de la correction depuis le JSON

Principe : au-dessus de 500 m, l'erreur de v7 est surtout un DECALAGE de tout le niveau. Une
ridge par niveau et par composante predit le vent moyen vrai du niveau a partir de donnees de
surface et de la moyenne predite par le modele ; on decale alors uniformement les noeuds
valides du niveau. La structure spatiale predite est conservee.

Strictement opt-in : `inference.correction` (chemin du JSON) vaut None par defaut, et v7
reste alors identique bit a bit. RESERVE (docs/M1_CORRECTION.md) : coefficients appris sur
donnees SYNTHETIQUES ; avec de vrais barometres (+-10 a 50 Pa), le signal geostrophique
(~26 Pa pour 10 m/s sur le domaine) est du meme ordre que l'erreur de mesure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .features import geostrophic_from_surface
from .geometry import Geometry

RD_GAS, G0, LAPSE_STD = 287.05, 9.80665, 0.0065

# BLOC 10f
FEATURES_10F = ["sin_h", "cos_h", "sin_doy", "cos_doy", "ug/10", "vg/10", "su*200", "sv*200",
                "u_capteurs", "v_capteurs", "T_capteurs", "u_niveau_modele", "v_niveau_modele"]
I_PM = 11                                   # indice de u_niveau_modele (v : I_PM + 1)
RIDGE_ALPHA_10F = 1.0                       # alpha du BLOC 10f, reutilise par le BLOC 10g
# BLOC 10h : les 30 caracteristiques, dans l'ordre du JSON `features`
FEATURES_10H = FEATURES_10F + ["ug_v2/10", "vg_v2/10", "stabilite_Pa_km", "su*stab", "sv*stab",
                               "rot25_u", "rot25_v", "rot50_u", "rot50_v", "rot75_u", "rot75_v",
                               "su*stab+", "su*stab-", "sv*stab+", "sv*stab-", "ug*stab",
                               "vg*stab"]
ALPHAS_10H = (0.01, 0.1, 1.0, 10.0, 100.0)
Z_HIGH_M = 650                              # niveaux du score de selection (>= 650 m)


# ---------------------------------------------------------------------- BLOC 10g
def geostrophic_v2(S: np.ndarray, sensor: np.ndarray, geom: Geometry) -> np.ndarray:
    """[ug, vg (rho_sl 1.2), ug, vg (rho_sl calculee), stabilite (Pa/m), residu tous points,
    residu retenus (Pa)] — BLOC 10g, a l'identique."""
    m = sensor
    if m.sum() < 100:
        return np.zeros(7)
    f = geom.cfg.f_coriolis
    p = S[6][m].astype(np.float64)
    tv = p / (RD_GAS * S[4][m].astype(np.float64))          # temperature virtuelle exacte
    z = geom.z_sensor[m]
    p_sl = p * np.exp(G0 * z / (RD_GAS * (tv + 0.5 * LAPSE_STD * z)))
    A = np.stack([np.ones(m.sum()), geom.xc[m], geom.yc[m], z], axis=1)
    coef, *_ = np.linalg.lstsq(A, p_sl, rcond=None)
    res = p_sl - A @ coef
    rms_all = float(np.sqrt(np.mean(res ** 2)))
    mad = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
    keep = np.abs(res - np.median(res)) < 5.0 * mad
    if keep.sum() > 100:
        coef, *_ = np.linalg.lstsq(A[keep], p_sl[keep], rcond=None)
        res = p_sl[keep] - A[keep] @ coef
    rho_calc = float(np.mean(p_sl / (RD_GAS * (tv + LAPSE_STD * z))))
    out = []
    for rho_sl in (1.2, rho_calc):
        out += [-coef[2] / (rho_sl * f), coef[1] / (rho_sl * f)]
    return np.array(out + [coef[3], rms_all, float(np.sqrt(np.mean(res ** 2)))])


def plane_residual_v1(S: np.ndarray, sensor: np.ndarray, geom: Geometry) -> float:
    """Residu RMS (Pa) du plan de l'estimateur actuel (BLOC 6) — BLOC 10g."""
    m = sensor
    if m.sum() < 100:
        return 0.0
    p = S[6][m].astype(np.float64)
    t_k = S[3][m].astype(np.float64) + 273.15
    z = geom.z_sensor[m]
    p_sl = p * np.exp(9.80665 * z / (287.05 * (t_k + 0.5 * 0.0065 * z)))
    A = np.stack([np.ones(m.sum()), geom.xc[m], geom.yc[m]], axis=1)
    coef, *_ = np.linalg.lstsq(A, p_sl, rcond=None)
    return float(np.sqrt(np.mean((p_sl - A @ coef) ** 2)))


def geo_row(S: np.ndarray, sensor: np.ndarray, geom: Geometry) -> np.ndarray:
    """10 valeurs : ug, vg actuels (BLOC 6), residu actuel, puis geostrophic_v2 (7)."""
    ug1, vg1 = geostrophic_from_surface(S, sensor, geom)
    return np.concatenate([[ug1, vg1, plane_residual_v1(S, sensor, geom)],
                           geostrophic_v2(S, sensor, geom)])


# ---------------------------------------------------------------------- BLOC 10f
def instant_features(S: np.ndarray, sensor: np.ndarray, tfeat: np.ndarray, u: np.ndarray,
                     v: np.ndarray, geom: Geometry) -> np.ndarray:
    """(nz, 13) : tfeat (8), moyennes u, v, T des capteurs visibles, moyennes u, v du niveau
    predites. Lignes nulles pour les niveaux hors evaluation (BLOC 10f).

    S : surface (>= 7 canaux, ny, nx) ; sensor : (ny, nx) booleen ; u, v : (nz, ny, nx).
    """
    m = sensor
    surf = [float(S[c][m].mean()) if m.any() else 0.0 for c in (0, 1, 3)]
    base = list(np.asarray(tfeat).astype(np.float64)) + surf
    out = np.zeros((geom.nz, len(FEATURES_10F)))
    for k in geom.eval_levels():
        mk = geom.valid[k]
        out[k] = base + [float(u[k][mk].mean()), float(v[k][mk].mean())]
    return out


def ridge_fit(X: np.ndarray, t: np.ndarray, alpha: float) -> dict:
    """Ridge standardisee, constante non penalisee (BLOC 10f / ridge_fit_a du 10h)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.c_[np.ones(len(X)), (X - mu) / sd]
    R = alpha * np.eye(A.shape[1])
    R[0, 0] = 0.0
    return {"mu": mu, "sd": sd, "coef": np.linalg.solve(A.T @ A + R, A.T @ t)}


def ridge_pred(p: dict, X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), (X - np.asarray(p["mu"])) / np.asarray(p["sd"])] \
        @ np.asarray(p["coef"])


# ---------------------------------------------------------------------- BLOC 10g / 10h
def x2(X: np.ndarray, G: np.ndarray, k: int, variant: str) -> np.ndarray:
    """Caracteristiques du BLOC 10f + vent geostrophique v2, stabilite, vent thermique x stab."""
    i = 3 if variant == "1.2" else 5
    d = G[:, 7] * 1e3                                        # Pa/km
    return np.c_[X[:, k], G[:, i] / 10.0, G[:, i + 1] / 10.0, d, X[:, k, 6] * d, X[:, k, 7] * d]


def rotate(u, v, a):
    return u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a)


def x3(X: np.ndarray, G: np.ndarray, k: int, variant: str) -> np.ndarray:
    """(n, 30) : caracteristiques du BLOC 10g + rotation et stabilite (couche limite, 10h)."""
    k_top = X.shape[1] - 1
    i_geo = 3 if variant == "1.2" else 5
    ug, vg = G[:, i_geo], G[:, i_geo + 1]                    # geostrophique v2
    us, vs = X[:, k_top, 8], X[:, k_top, 9]                  # vent moyen des capteurs visibles
    a = np.arctan2(vs, us) - np.arctan2(vg, ug)
    a = (a + np.pi) % (2 * np.pi) - np.pi                    # angle de rotation estime (rad)
    d = G[:, 7] * 1e3                                        # indicateur de stabilite (Pa/km)
    dp, dm = np.maximum(d, 0.0), np.minimum(d, 0.0)
    su, sv = X[:, k, 6], X[:, k, 7]                          # vent thermique (x 200)
    cols = [x2(X, G, k, variant)]
    for fr in (0.25, 0.5, 0.75):
        ru, rv = rotate(ug, vg, fr * a)
        cols.append(np.c_[ru / 10.0, rv / 10.0])
    cols.append(np.c_[su * dp, su * dm, sv * dp, sv * dm, ug / 10.0 * d, vg / 10.0 * d])
    return np.concatenate(cols, axis=1)


# ---------------------------------------------------------------------- ajustement (10g + 10h)
def fit_correction(X_tr, G_tr, T_tr, X_va, G_va, T_va, geom: Geometry,
                   alphas=ALPHAS_10H) -> tuple[dict, dict]:
    """Reproduit 10g (choix de rho_sl) puis 10h (choix d'alpha, niveaux corriges).

    X : (n, nz, 13) ; G : (n, 10) ; T : (n, nz, 2) moyennes vraies de u, v par niveau.
    Tous les choix se font sur la VALIDATION (jamais le test). Renvoie (JSON, rapport) ; le
    JSON a exactement le format de couche_limite_v7.json ecrit par le BLOC 10h.
    """
    levels = geom.eval_levels()
    if geom.nz - 1 not in levels:
        raise ValueError("le niveau du haut doit etre evalue (x3 y lit le vent des capteurs)")
    high = [k for k in levels if geom.z_levels[k] >= Z_HIGH_M] or levels

    def rmse(pred, t):
        return float(np.sqrt(np.mean((pred - t) ** 2)))

    # 10g : variante rho_sl, ridge a l'alpha du 10f
    score_v = {}
    for variant in ("1.2", "calc"):
        errs = []
        for k in high:
            for ci in range(2):
                p = ridge_fit(x2(X_tr, G_tr, k, variant), T_tr[:, k, ci], RIDGE_ALPHA_10F)
                errs.append(rmse(ridge_pred(p, x2(X_va, G_va, k, variant)), T_va[:, k, ci]))
        score_v[variant] = float(np.mean(errs))
    variant = min(score_v, key=score_v.get)

    # 10h : grille d'alpha
    fits = {}
    for alpha in alphas:
        for k in levels:
            for ci, c in enumerate("uv"):
                p = ridge_fit(x3(X_tr, G_tr, k, variant), T_tr[:, k, ci], alpha)
                e = rmse(ridge_pred(p, x3(X_va, G_va, k, variant)), T_va[:, k, ci])
                fits[(alpha, k, c)] = (p, e)
    score_a = {al: float(np.mean([fits[(al, k, c)][1] for k in high for c in "uv"]))
               for al in alphas}
    alpha = min(score_a, key=score_a.get)

    niveaux, par_niveau = {}, []
    for k in levels:
        for ci, c in enumerate("uv"):
            e_mod = rmse(X_va[:, k, I_PM + ci], T_va[:, k, ci])
            p, e3 = fits[(alpha, k, c)]
            niveaux[f"{int(geom.z_levels[k])}_{c}"] = {
                "utilise": bool(e3 < e_mod), "mu": p["mu"].tolist(), "sd": p["sd"].tolist(),
                "coef": p["coef"].tolist()}
            par_niveau.append({"z_m": int(geom.z_levels[k]), "comp": c, "erreur_modele": e_mod,
                               "erreur_corrigee": e3, "utilise": bool(e3 < e_mod)})
    data = {"variante_rho_sl": variant, "ridge_alpha": alpha, "features": list(FEATURES_10H),
            "niveaux": niveaux}
    rapport = {"score_variante": score_v, "variante": variant, "score_alpha": score_a,
               "alpha": alpha, "validation": par_niveau}
    return data, rapport


# ---------------------------------------------------------------------- application (12-2b)
class MeanWindCorrection:
    """Correction chargee depuis le JSON du BLOC 10h, appliquee comme l'etape 2b du BLOC 12."""

    def __init__(self, data: dict, path: str | Path | None = None,
                 sha256: str | None = None) -> None:
        if list(data.get("features", [])) != FEATURES_10H:
            raise ValueError("JSON de correction : les 30 caracteristiques attendues (BLOC 10h) ne "
                             "correspondent pas, dans cet ordre, a celles du fichier")
        if data.get("variante_rho_sl") not in ("1.2", "calc"):
            raise ValueError(f"variante_rho_sl inconnue : {data.get('variante_rho_sl')!r}")
        self.data = data
        self.variant = data["variante_rho_sl"]
        self.path = Path(path) if path else None
        self.sha256 = sha256
        self.levels = {key: {"utilise": bool(v["utilise"]), "mu": np.asarray(v["mu"], float),
                             "sd": np.asarray(v["sd"], float), "coef": np.asarray(v["coef"], float)}
                       for key, v in data["niveaux"].items()}

    @classmethod
    def load(cls, path: str | Path) -> "MeanWindCorrection":
        raw = Path(path).read_bytes()
        return cls(json.loads(raw.decode("utf-8")), path, hashlib.sha256(raw).hexdigest())

    def info(self) -> dict:
        """Ce qui part dans la meta et les exports : quelle correction, exactement."""
        return {"fichier": self.path.name if self.path else None, "sha256": self.sha256,
                "variante_rho_sl": self.variant, "ridge_alpha": self.data.get("ridge_alpha"),
                "corrige": sorted(k for k, v in self.levels.items() if v["utilise"])}

    def shifts(self, S: np.ndarray, sensor: np.ndarray, tfeat: np.ndarray, u: np.ndarray,
               v: np.ndarray, geom: Geometry) -> dict[tuple[int, int], float]:
        """{(niveau, composante 0/1) : decalage a ajouter aux noeuds valides du niveau}."""
        X = instant_features(S, sensor, tfeat, u, v, geom)[None]
        G = geo_row(S, sensor, geom)[None]
        out = {}
        for k in range(geom.nz):
            mk = geom.valid[k]
            if mk.sum() == 0:
                continue
            for ci, (c, fld) in enumerate((("u", u), ("v", v))):
                par = self.levels.get(f"{int(geom.z_levels[k])}_{c}")
                if par is None or not par["utilise"]:
                    continue
                target = float(ridge_pred(par, x3(X, G, k, self.variant))[0])
                out[(k, ci)] = target - float(fld[k][mk].mean())
        return out

    def apply(self, u: np.ndarray, v: np.ndarray, S: np.ndarray, sensor: np.ndarray,
              tfeat: np.ndarray, geom: Geometry) -> tuple[np.ndarray, np.ndarray, int]:
        """Nouveaux (u, v) : chaque niveau corrige est decale uniformement sur ses noeuds
        valides. Les noeuds sous le relief (NaN ou 0) ne sont jamais touches."""
        sh = self.shifts(S, sensor, tfeat, u, v, geom)
        out = [u.copy(), v.copy()]
        for (k, ci), delta in sh.items():
            out[ci][k][geom.valid[k]] += delta
        return out[0], out[1], len(sh)


def load_correction(path: str | Path | None) -> MeanWindCorrection | None:
    return MeanWindCorrection.load(path) if path else None

