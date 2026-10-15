"""Lot M8 — methodes de reference : profil logarithmique, loi de puissance, krigeage.

Cahier des charges v9 : section 2.2 (p. 8, « Extrapolation et interpolation classiques :
profil logarithmique, krigeage »), lot 5 (p. 27, « Comparaison avec les methodes de
reference : profil logarithmique, PyKrige »), tache 7 (p. 44, livrable L6).

Une methode de reference = interpolation HORIZONTALE des capteurs visibles (vent a hauteur
de capteur, 25-30 m du sol) puis loi VERTICALE qui l'extrapole aux noeuds 3D :

    log        U(h) = U_s . ln(h / z0) / ln(h_s / z0)   z0 : rugosite de la grille
                                                        (aucun parametre ajuste)
    puissance  U(h) = U_s . (h / h_s) ** alpha          alpha : ajuste sur la VALIDATION

h : hauteur du noeud au-dessus du sol ; h_s : hauteur du capteur au-dessus du sol (grille) ;
U_s : vent (u, v) interpole a la colonne. La direction ne tourne pas avec la hauteur et
w = 0 : ces lois n'en disent rien (c'est une de leurs limites, cahier p. 8).

Interpolation horizontale :

    idw        inverse des distances au carre, k plus proches capteurs (aucun parametre ajuste)
    krigeage   krigeage ORDINAIRE en voisinage local (k voisins), variogramme exponentiel
               ajuste sur la VALIDATION (valeurs standardisees par instant), jamais sur le test

Implementation numpy / scipy : PyKrige n'est pas une dependance. L'estimateur est le meme
(krigeage ordinaire, multiplicateur de Lagrange) ; il est ecrit ici en voisinage local pour
tenir sur ~125 000 colonnes et jusqu'a ~125 000 capteurs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

H_MIN_M = 1.0                       # hauteur minimale au-dessus du sol (m)
LOIS = ("log", "puissance")
INTERPOLATIONS = ("idw", "krigeage")
ALPHA_GRILLE = np.round(np.arange(0.0, 0.6001, 0.01), 2)
NOMS_REFERENCES = ("log_idw", "puissance_idw", "log_krigeage", "puissance_krigeage")


# ---------------------------------------------------------------------- geometrie
def xy_colonnes(geom) -> np.ndarray:
    """(ny*nx, 2) : x, y de chaque colonne en m depuis le coin sud-ouest (Volume.at_point)."""
    j, i = np.meshgrid(np.arange(geom.ny), np.arange(geom.nx), indexing="ij")
    return np.stack([i.ravel() * geom.dx, j.ravel() * geom.dy], axis=1).astype(np.float64)


def hauteur_noeuds(geom) -> np.ndarray:
    """(nz, ny, nx) : hauteur des noeuds au-dessus du sol (niveaux ASL - relief)."""
    return np.asarray(geom.z_levels, float)[:, None, None] - np.asarray(geom.zt, float)[None]


def hauteur_capteurs(geom) -> np.ndarray:
    """(ny, nx) : hauteur du capteur au-dessus du sol, au moins H_MIN_M (meme altitude de
    capteur que la detection de derive M4, `geom.z_sensor`)."""
    return np.maximum(np.asarray(geom.z_sensor, float) - np.asarray(geom.zt, float), H_MIN_M)


def facteur_vertical(loi: str, h: np.ndarray, h_s: np.ndarray, z0: np.ndarray | None = None,
                     alpha: float | None = None) -> np.ndarray:
    """Rapport U(h) / U(h_s) de la loi verticale (log ou puissance)."""
    h = np.maximum(np.asarray(h, float), H_MIN_M)
    h_s = np.asarray(h_s, float)
    if loi == "log":
        if z0 is None:
            raise ValueError("profil logarithmique : z0 requis")
        # z0 <= h_s / 2 : ln(h_s / z0) >= ln 2, jamais de division par ~0
        z0 = np.minimum(np.asarray(z0, float), 0.5 * h_s)
        return np.log(np.maximum(h, 1.5 * z0) / z0) / np.log(h_s / z0)
    if loi == "puissance":
        if alpha is None:
            raise ValueError("loi de puissance : alpha requis (ajuste sur la validation)")
        return (h / h_s) ** float(alpha)
    raise ValueError(f"loi verticale inconnue : {loi!r} ({', '.join(LOIS)})")


# ---------------------------------------------------------------------- horizontal
class IDW:
    """Inverse des distances a la puissance p, k plus proches capteurs."""

    nom = "idw"

    def __init__(self, voisins: int = 8, puissance: float = 2.0) -> None:
        self.voisins = int(voisins)
        self.puissance = float(puissance)

    def poids(self, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(indices (m, k) dans src, poids (m, k) de somme 1). Exact sur un capteur."""
        k = min(self.voisins, len(src))
        d, idx = cKDTree(src).query(dst, k=k)
        d, idx = d.reshape(len(dst), k), idx.reshape(len(dst), k)
        w = 1.0 / np.maximum(d, 1e-6) ** self.puissance
        exact = d[:, 0] < 1e-6
        w[exact] = 0.0
        w[exact, 0] = 1.0
        return idx, w / w.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict:
        return {"methode": self.nom, "voisins": self.voisins, "puissance": self.puissance}


@dataclass
class Variogramme:
    """Variogramme exponentiel : gamma(h) = pepite + palier (1 - exp(-3 h / portee)),
    gamma(0) = 0. `portee_m` est la portee pratique (95 % du palier partiel)."""

    pepite: float
    palier: float
    portee_m: float
    modele: str = "exponentiel"

    def gamma(self, h) -> np.ndarray:
        h = np.asarray(h, float)
        g = self.pepite + self.palier * (1.0 - np.exp(-3.0 * h / self.portee_m))
        return np.where(h > 0, g, 0.0)

    def to_dict(self) -> dict:
        return {"modele": self.modele, "pepite": self.pepite, "palier": self.palier,
                "portee_m": self.portee_m}

    @classmethod
    def from_dict(cls, d: dict) -> "Variogramme":
        if d.get("modele", "exponentiel") != "exponentiel":
            raise ValueError(f"modele de variogramme non gere : {d.get('modele')!r}")
        return cls(float(d["pepite"]), float(d["palier"]), float(d["portee_m"]))


class Krigeage:
    """Krigeage ordinaire en voisinage local : pour chaque cible, les k capteurs les plus
    proches, systeme (k+1) x (k+1) avec multiplicateur de Lagrange (poids de somme 1).
    Les memes poids servent a u et a v (un seul variogramme, sur valeurs standardisees :
    les poids du krigeage ne dependent pas de l'echelle du palier)."""

    nom = "krigeage"

    def __init__(self, variogramme: Variogramme, voisins: int = 16, bloc: int = 8192) -> None:
        self.variogramme = variogramme
        self.voisins = int(voisins)
        self.bloc = int(bloc)

    def poids(self, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = min(self.voisins, len(src))
        d, idx = cKDTree(src).query(dst, k=k)
        d, idx = d.reshape(len(dst), k), idx.reshape(len(dst), k)
        if k == 1:
            return idx, np.ones_like(d)
        vg = self.variogramme
        c0 = vg.pepite + vg.palier                      # C(0) : variance totale

        def cov(h):
            # forme covariance du variogramme borne : C(h) = C(0) - gamma(h), C(0) au
            # point lui-meme (pepite comprise) -> matrice symetrique definie positive
            return c0 - vg.gamma(h)

        w = np.empty_like(d)
        for a in range(0, len(dst), self.bloc):
            b = min(a + self.bloc, len(dst))
            p = src[idx[a:b]]                                        # (n, k, 2)
            dd = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=-1)
            C = cov(dd)
            C[:, np.arange(k), np.arange(k)] += 1e-10 * c0          # garde-fou numerique
            # deux seconds membres : c (covariances cible-voisins) et 1 (contrainte)
            rhs = np.stack([cov(d[a:b]), np.ones((b - a, k))], axis=-1)
            x = _resoudre_spd(C, rhs)                                # (n, k, 2)
            ci, un = x[..., 0], x[..., 1]                            # C^-1 c, C^-1 1
            # krigeage ordinaire : w = C^-1 (c - mu 1), mu = (1' C^-1 c - 1) / (1' C^-1 1)
            mu = (ci.sum(axis=1) - 1.0) / un.sum(axis=1)
            w[a:b] = ci - mu[:, None] * un
        return idx, w

    def to_dict(self) -> dict:
        return {"methode": self.nom, "voisins": self.voisins,
                "variogramme": self.variogramme.to_dict()}


def _resoudre_spd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Resout A X = B pour un lot de matrices symetriques definies positives (n, k, k),
    B (n, k, r), par elimination de Gauss vectorisee sur le lot (sans pivot : inutile pour
    une matrice SPD). Evite un appel LAPACK par matrice, ~100 fois plus lent ici."""
    A = A.astype(np.float64, copy=True)
    B = B.astype(np.float64, copy=True)
    k = A.shape[1]
    for p in range(k - 1):
        f = A[:, p + 1:, p] / A[:, p, p][:, None]                  # (n, k-p-1)
        A[:, p + 1:, p:] -= f[:, :, None] * A[:, p, None, p:]
        B[:, p + 1:] -= f[:, :, None] * B[:, p, None]
    X = np.empty_like(B)
    for p in range(k - 1, -1, -1):
        s = B[:, p] - np.einsum("nj,njr->nr", A[:, p, p + 1:], X[:, p + 1:])
        X[:, p] = s / A[:, p, p][:, None]
    return X


def interpoler(interp, src: np.ndarray, valeurs: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """valeurs (c, n) aux capteurs src (n, 2) -> (c, m) aux cibles dst (m, 2)."""
    valeurs = np.atleast_2d(np.asarray(valeurs, float))
    if len(src) == 0:
        return np.zeros((valeurs.shape[0], len(dst)))
    idx, w = interp.poids(src, dst)
    return np.einsum("mk,cmk->cm", w, valeurs[:, idx])


# ---------------------------------------------------------------------- variogramme
def paires_variogramme(xy: np.ndarray, rng: np.random.Generator, h_max: float,
                       n_centres: int = 2000, voisins: int = 8,
                       n_aleatoires: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Paires de capteurs (i, j) pour le variogramme empirique : les `voisins` plus proches
    de `n_centres` capteurs (petits ecarts) + des paires tirees au hasard (grands ecarts).
    Le choix des paires ne depend que des positions, pas des valeurs : l'estimateur par
    classe de distance reste sans biais."""
    n = len(xy)
    if n < 2:
        return np.zeros(0, int), np.zeros(0, int)
    c = rng.choice(n, size=min(n_centres, n), replace=False)
    k = min(voisins + 1, n)
    _, nb = cKDTree(xy).query(xy[c], k=k)
    nb = nb.reshape(len(c), k)
    i1 = np.repeat(c, k - 1)
    j1 = nb[:, 1:].ravel()
    i2 = rng.integers(0, n, n_aleatoires)
    j2 = rng.integers(0, n, n_aleatoires)
    i, j = np.concatenate([i1, i2]), np.concatenate([j1, j2])
    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    ok = (i != j) & (d <= h_max)
    return i[ok], j[ok]


class VariogrammeEmpirique:
    """Cumul de 0,5 (z_i - z_j)^2 par classe de distance (classes geometriques)."""

    def __init__(self, h_min: float, h_max: float, n_classes: int = 20) -> None:
        self.bords = np.geomspace(max(h_min, 1.0), h_max, n_classes + 1)
        self.bords[0] = 0.0
        self.somme = np.zeros(n_classes)
        self.somme_h = np.zeros(n_classes)
        self.n = np.zeros(n_classes)

    def ajouter(self, xy: np.ndarray, z: np.ndarray, i: np.ndarray, j: np.ndarray) -> None:
        d = np.linalg.norm(xy[i] - xy[j], axis=1)
        c = np.clip(np.searchsorted(self.bords, d, side="right") - 1, 0, len(self.n) - 1)
        np.add.at(self.somme, c, 0.5 * (z[i] - z[j]) ** 2)
        np.add.at(self.somme_h, c, d)
        np.add.at(self.n, c, 1)

    def table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ok = self.n > 0
        return (self.somme_h[ok] / self.n[ok], self.somme[ok] / self.n[ok], self.n[ok])


def ajuster_variogramme(h: np.ndarray, g: np.ndarray, n: np.ndarray, h_min: float,
                        h_max: float) -> Variogramme:
    """Moindres carres ponderes par le nombre de paires (modele exponentiel)."""
    from scipy.optimize import least_squares

    if len(h) < 3:
        raise ValueError("variogramme : moins de 3 classes de distance renseignees")
    sw = np.sqrt(n / n.sum())

    def res(p):
        return sw * (p[0] + p[1] * (1.0 - np.exp(-3.0 * h / p[2])) - g)

    p0 = [max(float(g[0]) * 0.5, 1e-4), max(float(g.max()), 1e-3), h_max / 3.0]
    lo, hi = [0.0, 1e-6, h_min], [np.inf, np.inf, 10.0 * h_max]
    p0 = list(np.clip(p0, lo, [x if np.isfinite(x) else 1e9 for x in hi]))
    fit = least_squares(res, p0, bounds=(lo, hi))
    return Variogramme(float(fit.x[0]), float(fit.x[1]), float(fit.x[2]))


def standardiser(x: np.ndarray) -> np.ndarray | None:
    """(x - moyenne) / ecart type de l'instant ; None si le champ est constant."""
    sd = float(np.std(x))
    return None if sd < 1e-6 else (x - float(np.mean(x))) / sd


# ---------------------------------------------------------------------- references
@dataclass
class Reference:
    """Une methode de reference : interpolation horizontale + loi verticale."""

    nom: str
    loi: str
    interp: object
    alpha: float | None = None

    def to_dict(self) -> dict:
        return {"nom": self.nom, "loi_verticale": self.loi,
                "alpha": self.alpha, "horizontal": self.interp.to_dict()}


def references(ajustement: dict | None = None, noms=None, voisins_idw: int = 8,
               voisins_krigeage: int = 16) -> dict[str, Reference]:
    """Les methodes de reference disponibles. Sans `ajustement` (validation), seule
    `log_idw` existe : c'est la seule sans parametre ajuste."""
    idw = IDW(voisins_idw)
    out: dict[str, Reference] = {"log_idw": Reference("log_idw", "log", idw)}
    if ajustement is not None:
        alpha = ajustement["alpha"]
        krig = Krigeage(Variogramme.from_dict(ajustement["variogramme"]), voisins_krigeage)
        out["puissance_idw"] = Reference("puissance_idw", "puissance", idw, alpha["idw"])
        out["log_krigeage"] = Reference("log_krigeage", "log", krig)
        out["puissance_krigeage"] = Reference("puissance_krigeage", "puissance", krig,
                                              alpha["krigeage"])
    if noms is None:
        return out
    manquants = [n for n in noms if n not in out]
    if manquants:
        raise ValueError(f"references indisponibles : {manquants} (connues : {list(out)} ; "
                         "puissance_* et *_krigeage demandent l'ajustement de validation)")
    return {n: out[n] for n in noms}


class CalculReferences:
    """Champs (u, v, w) de plusieurs references pour un instant, en ne calculant qu'une
    fois chaque interpolation horizontale."""

    def __init__(self, geom, refs: dict[str, Reference]) -> None:
        self.geom = geom
        self.refs = refs
        self.xy = xy_colonnes(geom)
        self.h = hauteur_noeuds(geom)
        self.h_s = hauteur_capteurs(geom)
        self.z0 = np.asarray(geom.z0, float)
        self.cols = np.flatnonzero(geom.valid.any(axis=0).ravel())   # colonnes utiles
        self._facteurs: dict[tuple, np.ndarray] = {}

    def facteur(self, ref: Reference) -> np.ndarray:
        key = (ref.loi, ref.alpha)
        if key not in self._facteurs:
            self._facteurs[key] = facteur_vertical(ref.loi, self.h, self.h_s[None],
                                                   self.z0[None], ref.alpha)
        return self._facteurs[key]

    def surface(self, interp, S: np.ndarray, sensor: np.ndarray) -> np.ndarray:
        """(2, ny, nx) : u, v interpoles aux colonnes au-dessus du relief (0 ailleurs)."""
        g = self.geom
        m = np.asarray(sensor, bool).ravel()
        vals = np.stack([np.asarray(S[0], float).ravel()[m], np.asarray(S[1], float).ravel()[m]])
        out = np.zeros((2, g.ny * g.nx))
        out[:, self.cols] = interpoler(interp, self.xy[m], vals, self.xy[self.cols])
        return out.reshape(2, g.ny, g.nx)

    def champs(self, S: np.ndarray, sensor: np.ndarray) -> dict[str, tuple]:
        """{nom: (u, v, w)} en float32 (nz, ny, nx), 0 sous le relief."""
        horiz: dict[int, np.ndarray] = {}
        out = {}
        valid = self.geom.valid
        for nom, ref in self.refs.items():
            key = id(ref.interp)
            if key not in horiz:
                horiz[key] = self.surface(ref.interp, S, sensor)
            us, vs = horiz[key]
            f = self.facteur(ref)
            u = np.where(valid, us[None] * f, 0.0).astype(np.float32)
            v = np.where(valid, vs[None] * f, 0.0).astype(np.float32)
            out[nom] = (u, v, np.zeros_like(u))
        return out


def predicteurs(calc: CalculReferences, device) -> dict:
    """Predicteurs au format de `evaluate.metriques_loader` (un par reference) ; les champs
    d'un instant sont calcules une fois et partages."""
    import torch

    # cle = l'objet `sensor` lui-meme (identite) : tous les predicteurs d'un instant recoivent
    # le meme tenseur ; un numero d'instant se repete d'un loader a l'autre (E1), pas l'objet
    # garde ici
    memo: dict = {"sensor": None, "champs": None}

    def champs(s_raw, sensor, j):
        if memo["sensor"] is not sensor:
            S = s_raw[0].detach().cpu().numpy()
            m = sensor[0].detach().cpu().numpy() > 0.5
            memo["sensor"], memo["champs"] = sensor, calc.champs(S, m)
        return memo["champs"]

    def fabrique(nom):
        def predire(s_raw, sensor, tfeat, y, j, stem):
            return [torch.from_numpy(a)[None].to(device) for a in champs(s_raw, sensor, j)[nom]]
        return predire

    return {nom: fabrique(nom) for nom in calc.refs}


# ---------------------------------------------------------------------- ajustement
def ajuster_sur_validation(cfg, geom, splits, n_instants: int | None = None,
                           keep: float | None = None, seed: int = 1000,
                           graine_paires: int = 8, verbose: bool = False) -> dict:
    """Variogramme puis alpha des lois de puissance, sur la VALIDATION seulement.

    Instants : selection synoptique fixe de la validation (`splits.val_selection`, graine
    1000, `mask.keep_eval` des capteurs), comme la calibration M2. Le test n'est jamais lu.
    alpha : minimise la somme des carres des erreurs de u et v sur les noeuds valides des
    niveaux evalues (couverture >= dso.min_level_cover), grille 0 a 0,6 par pas de 0,01,
    pour chaque interpolation horizontale (idw, krigeage).
    """
    from .dataset import ANEO3D

    s, h, d = splits.val
    idx = np.asarray(splits.val_selection)
    if n_instants is not None and len(idx) > n_instants:
        idx = idx[np.unique(np.linspace(0, len(idx) - 1, n_instants).astype(int))]
    if len(idx) == 0:
        raise ValueError("aucun instant de validation en cache : ajustement impossible")
    ds = ANEO3D(s[idx], h[idx], d[idx], geom, train=False, seed=seed, keep_eval=keep)
    xy = xy_colonnes(geom)
    rng = np.random.default_rng(graine_paires)
    larg, haut = geom.nx * geom.dx, geom.ny * geom.dy
    h_min, h_max = float(min(geom.dx, geom.dy)), 0.5 * float(min(larg, haut))

    # 1. variogramme empirique, valeurs standardisees par instant et par composante
    emp = VariogrammeEmpirique(h_min, h_max)
    for s_raw, sensor, _t, _y, _j in ds:
        m = sensor.numpy().ravel() > 0.5
        pts = xy[m]
        i, j = paires_variogramme(pts, rng, h_max)
        for c in (0, 1):
            z = standardiser(s_raw[c].numpy().astype(float).ravel()[m])
            if z is not None and len(i):
                emp.ajouter(pts, z, i, j)
    hh, gg, nn = emp.table()
    vario = ajuster_variogramme(hh, gg, nn, h_min, h_max)
    if verbose:
        print(f"  variogramme (validation) : pepite {vario.pepite:.3f} palier "
              f"{vario.palier:.3f} portee {vario.portee_m:.0f} m", flush=True)

    # 2. alpha par interpolation horizontale
    interps = {"idw": IDW(), "krigeage": Krigeage(vario)}
    calc = CalculReferences(geom, {})
    lv = np.zeros(geom.nz, bool)
    lv[geom.eval_levels()] = True
    sse = {k: np.zeros(len(ALPHA_GRILLE)) for k in interps}
    n_noeuds = 0
    for s_raw, sensor, _t, y, _j in ds:
        S, m, Y = s_raw.numpy(), sensor.numpy() > 0.5, y.numpy()
        sel = geom.valid & lv[:, None, None] & np.isfinite(Y[0]) & np.isfinite(Y[1])
        r = np.maximum(calc.h, H_MIN_M)[sel] / calc.h_s[None].repeat(geom.nz, 0)[sel]
        lr = np.log(r)
        yu, yv = Y[0][sel].astype(float), Y[1][sel].astype(float)
        n_noeuds += int(sel.sum())
        for nom, it in interps.items():
            us, vs = calc.surface(it, S, m)
            ui = np.broadcast_to(us[None], Y[0].shape)[sel]
            vi = np.broadcast_to(vs[None], Y[0].shape)[sel]
            for a, alpha in enumerate(ALPHA_GRILLE):
                f = np.exp(alpha * lr)
                sse[nom][a] += float(((ui * f - yu) ** 2).sum() + ((vi * f - yv) ** 2).sum())
    alpha = {k: float(ALPHA_GRILLE[int(np.argmin(v))]) for k, v in sse.items()}
    if verbose:
        print("  alpha (validation) : " + "  ".join(f"{k} {a:.2f}" for k, a in alpha.items()),
              flush=True)
    keep_eff = cfg.mask.keep_eval if keep is None else keep
    return {
        "produit_par": "pcfno3d.baselines", "jeu": "validation", "graine": seed,
        "instants": int(len(ds)), "capteurs_visibles": keep_eff,
        "variogramme": vario.to_dict(),
        "variogramme_empirique": {"h_m": hh.tolist(), "gamma": gg.tolist(),
                                  "paires": nn.astype(int).tolist(),
                                  "valeurs": "u et v standardises par instant"},
        # portee a la borne haute de l'ajustement : variogramme sans palier atteint sur le
        # domaine (champ tres lisse) ; le krigeage reste defini, mais le dire
        "portee_a_la_borne": bool(vario.portee_m >= 0.99 * 10.0 * h_max),
        "alpha": alpha,
        "rmse_alpha": {k: {f"{a:.2f}": float(np.sqrt(v[i] / max(2 * n_noeuds, 1)))
                           for i, a in enumerate(ALPHA_GRILLE)} for k, v in sse.items()},
        "niveaux_ajustes_m": [int(geom.z_levels[k]) for k in geom.eval_levels()],
    }


def ecrire_ajustement(data: dict, path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} existe deja : un ajustement n'est jamais ecrase")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return path


def charger_ajustement(path: str | Path) -> dict:
    """Relit un ajustement ; refuse tout fichier qui ne vient pas de la validation."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("jeu") != "validation":
        raise ValueError(f"{path} : ajustement fait sur {data.get('jeu')!r} ; seuls les "
                         "ajustements de VALIDATION sont acceptes (le test ne regle rien)")
    return data
