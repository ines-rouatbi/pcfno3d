"""Canal d'innovation du candidat D2 national (configs/national/regle_d2_national.json, PROPOSITION soumise a
l'accord ecrit d'Ines ; definition d'Ines du 11/11).

Constat du 11/11 (scripts/controle_capteurs.py) : +15 m/s aux capteurs ne deplace la sortie que de ~1e-4 m/s
pres d'eux. Le modele residuel recoit ERA5 partout (2 derniers canaux) et le treillis de surface repete ERA5
a 10 m : un point de mesure isole est noye. D2 donne la mesure au reseau sous une forme dense :

1. innovation a chaque point de mesure j (aeroport, capteur ANEO), par composante :
       d_j = vent mesure a 10 m - vent ERA5 a 10 m au-dessus du sol interpole au meme point
   (cellules ERA5 de l'instant, bilineaire comme forcage.InterpERA5 ; a l'entrainement, les cellules
   DEGRADEES de l'instant : la meme ERA5 que celle que voit le reseau). Aeroports : la valeur d'entree a
   leur cellule. Capteurs ANEO : `innovation.capteurs_10m` = « verite_10m » (le capteur simule rapporte la
   verite a 10 m, mask.aneo_hauteur_capteur_m = 10) ou « loi_log » (valeur a mask.aneo_hauteur_capteur_m
   ramenee a 10 m par le profil logarithmique de la maille : x ln(10/z0) / ln(h/z0)).
2. etalement sur toutes les colonnes g, `innovation.methode` :
     « oi » (interpolation optimale, krigeage simple) :
       P = rho(D_oo) + diag(r_j^2),  r_j = sigma_o / sigma_b,   increment(g) = rho(g, .) P^-1 d
       un point seul rend au point d / (1 + r^2) ; rho gaussienne exp(-d^2/2L^2) (ou exponentielle exp(-d/L)),
       tronquee a 4 L (7 L) ;
     « cressman » (version simple) : increment(g) = sum w_j d_j / max(sum w_j, 1),
       w = (R^2 - d^2) / (R^2 + d^2) pour d < R = 2 L (un point seul rend d au point).
   L et r : scripts/statistiques_innovation_d2.py (jours d'ENTRAINEMENT, paires d'aeroports, L borne a
   [20 ; 50] km).
3. decroissance verticale f(h) = exp(-max(h - 10 m, 0) / H) (correction maximale pres du sol, decroissante
   vers 1 km), h = hauteur du noeud au-dessus du sol (>= 5 m) ; H choisi sur la VALIDATION dans
   {200, 500, 1000} m (scripts/info_capteurs_d2.py reglage).
4. canaux d'entree (3 tenseurs, AVANT les 2 canaux ERA5) : champ d'innovation etale u, v (x f(h), normalise
   par l'ecart-type de la cible, sans moyenne, 0 sous le relief) et distance au point de mesure le plus proche
   (km / ECHELLE_DISTANCE_KM, recopiee sur les niveaux).
5. option `innovation.base_residuel` = « era5_oi » : la base de la sortie residuelle (2 derniers canaux)
   devient ERA5 + innovation etalee x f(h) au lieu d'ERA5 seule.

Cout (CPU, chargeur de donnees) : un pochoir de correlation precalcule ajoute une fois par point (fenetre de
la troncature), une distance euclidienne exacte (scipy.ndimage.distance_transform_edt). Les champs 2D
(3 x ny x nx) voyagent a la fin du vecteur tfeat, apres les cellules ERA5 : les loaders gardent leur forme ;
assemble_input les deploie en 3D sur le peripherique.

Reference « interpolation optimale seule » (sans reseau) : ERA5 aux noeuds + increment x f(h)
(predicteur_oi), w par continuite, mesuree comme le predicteur « era5 ».
"""

from __future__ import annotations

import math

import numpy as np
import torch

N_OI = 3                        # tenseurs d'entree : du, dv (champ d'innovation), distance
H_MESURE_M = 10.0               # hauteur des innovations (aeroports METAR, mats ramenes a 10 m)
ECHELLE_DISTANCE_KM = 100.0     # canal distance = km / 100
TRONCATURE = {"gaussienne": 4.0, "exponentielle": 7.0}     # en L : rho < 4e-4 / 1e-3 au-dela
RAYON_CRESSMAN_L = 2.0          # R = 2 L


class InnovationOI:
    """Operateur fixe (geometrie + parametres) : points de mesure -> champs 2D d'innovation."""

    def __init__(self, geom, icfg=None) -> None:
        icfg = geom.cfg.innovation if icfg is None else icfg
        if geom.forcage is None or geom.interp_era5 is None:
            raise ValueError("innovation.actif demande paths.forcage_era5 (ERA5 aux points de mesure)")
        self.cfg = icfg
        self.geom = geom
        self.ny, self.nx = int(geom.ny), int(geom.nx)
        self.dx, self.dy = float(geom.dx), float(geom.dy)
        self.L = float(icfg.L_km) * 1000.0
        self.methode, self.correlation = icfg.methode, icfg.correlation
        self.r_cut = (RAYON_CRESSMAN_L if self.methode == "cressman" else TRONCATURE[self.correlation]) * self.L
        # pochoir (une fois) : (2 ry + 1, 2 rx + 1), borne a la taille du domaine
        self.ry = min(int(math.ceil(self.r_cut / self.dy)), self.ny - 1)
        self.rx = min(int(math.ceil(self.r_cut / self.dx)), self.nx - 1)
        a = np.arange(-self.ry, self.ry + 1)[:, None] * self.dy
        b = np.arange(-self.rx, self.rx + 1)[None, :] * self.dx
        self.pochoir = self.poids(np.hypot(a, b)).astype(np.float32)
        # decroissance verticale aux noeuds (nz, ny, nx), 0 sous le relief
        from .forcage import MIN_ZREL_NODE
        z = np.asarray(geom.z_levels, np.float64)[:, None, None]
        h = np.maximum(z - np.asarray(geom.zt, np.float64)[None], MIN_ZREL_NODE)
        self.f3d = (np.exp(-np.maximum(h - H_MESURE_M, 0.0) / icfg.H_m) * np.asarray(geom.valid, bool)
                    ).astype(np.float32)
        self._f3d_t: dict = {}
        self.distance_max_km = math.hypot(self.ny * self.dy, self.nx * self.dx) / 1000.0
        self.n_queue = N_OI * self.ny * self.nx

    # ------------------------------------------------------------------ poids
    def rho(self, d_m: np.ndarray) -> np.ndarray:
        """Correlation horizontale des erreurs d'ERA5 (OI), tronquee."""
        d = np.asarray(d_m, np.float64)
        r = np.exp(-0.5 * (d / self.L) ** 2) if self.correlation == "gaussienne" else np.exp(-d / self.L)
        return np.where(d <= self.r_cut, r, 0.0)

    def poids(self, d_m: np.ndarray) -> np.ndarray:
        """Poids du pochoir : rho (OI) ou poids de Cressman (R^2 - d^2) / (R^2 + d^2), d < R."""
        if self.methode == "oi":
            return self.rho(d_m)
        d2, R2 = np.asarray(d_m, np.float64) ** 2, self.r_cut ** 2
        return np.where(d2 < R2, (R2 - d2) / (R2 + d2), 0.0)

    def _distances(self, iy_a, ix_a, iy_b, ix_b) -> np.ndarray:
        return np.hypot((np.asarray(iy_a, np.float64)[:, None] - np.asarray(iy_b, np.float64)[None]) * self.dy,
                        (np.asarray(ix_a, np.float64)[:, None] - np.asarray(ix_b, np.float64)[None]) * self.dx)

    # ------------------------------------------------------------------ points de mesure
    def era5_10m_aux_points(self, cellules: np.ndarray, iy, ix) -> tuple[np.ndarray, np.ndarray]:
        """u, v ERA5 a 10 m au-dessus du sol aux colonnes (iy, ix) : bilineaire dans le plan (poids
        d'InterpERA5), lineaire en hauteur entre les hauteurs du forcage qui encadrent 10 m."""
        it, fo = self.geom.interp_era5, self.geom.forcage
        c = np.asarray(cellules, np.float64).reshape(fo.forme_cellules)        # (2, nh, ny_e, nx_e)
        iy, ix = np.asarray(iy, np.int64), np.asarray(ix, np.int64)
        ey, ex, a, b = it.iy[iy, ix], it.ix[iy, ix], it.a[iy, ix], it.b[iy, ix]
        hs = fo.hauteurs
        hh = float(np.clip(H_MESURE_M, hs[0], hs[-1]))
        k = int(np.clip(np.searchsorted(hs, hh) - 1, 0, len(hs) - 2))
        t = (hh - hs[k]) / (hs[k + 1] - hs[k])
        ch = (1 - t) * c[:, k] + t * c[:, k + 1]                                # (2, ny_e, nx_e)
        val = ((1 - a) * (1 - b) * ch[:, ey, ex] + (1 - a) * b * ch[:, ey, ex + 1]
               + a * (1 - b) * ch[:, ey + 1, ex] + a * b * ch[:, ey + 1, ex + 1])
        return val[0], val[1]

    def points(self, S: np.ndarray, stations: np.ndarray | None, capteurs=None) -> dict:
        """Points de mesure de l'instant : stations (masque (ny, nx) des aeroports EN ENTREE, valeur de S a
        leur cellule) + capteurs ANEO (cellules (n, 2), valeur de S a leur cellule ramenee a 10 m selon
        innovation.capteurs_10m). Rapport d'erreurs r par type. Valeurs non finies ignorees."""
        g, c = self.geom, self.cfg
        if capteurs is not None and len(capteurs):
            cel = np.asarray(capteurs, np.int64).reshape(-1, 2)
            pris = np.zeros((self.ny, self.nx), bool)
            pris[cel[:, 0], cel[:, 1]] = True
        else:
            cel, pris = np.zeros((0, 2), np.int64), np.zeros((self.ny, self.nx), bool)
        sy, sx = (np.nonzero(np.asarray(stations, bool) & ~pris) if stations is not None
                  else (np.zeros(0, np.int64), np.zeros(0, np.int64)))   # un capteur prime sur la station
        iy, ix = np.concatenate([sy, cel[:, 0]]).astype(np.int64), np.concatenate([sx, cel[:, 1]]).astype(np.int64)
        u = np.asarray(S[0], np.float64)[iy, ix]
        v = np.asarray(S[1], np.float64)[iy, ix]
        r_c = c.rapport_o_b if c.rapport_o_b_capteur is None else c.rapport_o_b_capteur
        r = np.concatenate([np.full(len(sy), c.rapport_o_b), np.full(len(cel), r_c)])
        if len(cel):
            h_c = float(g.cfg.mask.aneo_hauteur_capteur_m)
            k = slice(len(sy), None)
            if c.capteurs_10m == "loi_log":
                z0 = np.maximum(np.asarray(g.z0, np.float64)[iy[k], ix[k]], 1e-4)
                fac = np.log(H_MESURE_M / z0) / np.log(h_c / z0)
                u[k], v[k] = u[k] * fac, v[k] * fac
            elif abs(h_c - H_MESURE_M) > 1e-6:
                raise ValueError(f"innovation.capteurs_10m = verite_10m : les capteurs simules doivent rapporter "
                                 f"le vent a 10 m (mask.aneo_hauteur_capteur_m = 10, recu {h_c:g})")
        ok = np.isfinite(u) & np.isfinite(v)
        return {"iy": iy[ok], "ix": ix[ok], "r": r[ok], "u": u[ok], "v": v[ok]}

    # ------------------------------------------------------------------ etalement
    def analyse(self, iy, ix, d_u, d_v, r) -> np.ndarray:
        """Innovations (d_u, d_v) aux colonnes (iy, ix), rapports d'erreurs r = sigma_o / sigma_b ->
        (3, ny, nx) float32 : increments u, v etales (m/s) et distance au point le plus proche (km)."""
        out = np.zeros((N_OI, self.ny, self.nx), np.float32)
        iy, ix = np.asarray(iy, np.int64), np.asarray(ix, np.int64)
        n = len(iy)
        if n == 0:
            out[2] = self.distance_max_km
            return out
        d = np.stack([np.asarray(d_u, np.float64), np.asarray(d_v, np.float64)], 1)       # (n, 2)
        if self.methode == "oi":
            P = self.rho(self._distances(iy, ix, iy, ix)) + np.diag(np.asarray(r, np.float64) ** 2)
            alpha = np.linalg.solve(P, d)
        else:
            alpha = d
            somme = np.zeros((self.ny, self.nx), np.float32)
        ry, rx, K = self.ry, self.rx, self.pochoir
        for j in range(n):
            y0, y1 = max(iy[j] - ry, 0), min(iy[j] + ry + 1, self.ny)
            x0, x1 = max(ix[j] - rx, 0), min(ix[j] + rx + 1, self.nx)
            k = K[y0 - iy[j] + ry:y1 - iy[j] + ry, x0 - ix[j] + rx:x1 - ix[j] + rx]
            out[0, y0:y1, x0:x1] += np.float32(alpha[j, 0]) * k
            out[1, y0:y1, x0:x1] += np.float32(alpha[j, 1]) * k
            if self.methode == "cressman":
                somme[y0:y1, x0:x1] += k
        if self.methode == "cressman":
            out[:2] /= np.maximum(somme, 1.0)[None]
        from scipy.ndimage import distance_transform_edt
        libre = np.ones((self.ny, self.nx), bool)
        libre[iy, ix] = False
        out[2] = distance_transform_edt(libre, sampling=(self.dy, self.dx)) / 1000.0
        return out

    def queue(self, S: np.ndarray, cellules: np.ndarray, stations: np.ndarray | None, capteurs=None) -> np.ndarray:
        """Champs 2D d'innovation de l'instant, aplatis (N_OI * ny * nx,) float32 : la fin du vecteur tfeat.
        cellules : cellules ERA5 de l'instant (celles que le reseau recoit, degradees a l'entrainement)."""
        p = self.points(S, stations, capteurs)
        eu, ev = self.era5_10m_aux_points(cellules, p["iy"], p["ix"])
        du, dv = p["u"] - eu, p["v"] - ev
        ok = np.ones(len(du), bool)
        if self.cfg.rejet_innovation_m_s is not None:
            ok = np.hypot(du, dv) <= self.cfg.rejet_innovation_m_s
        return self.analyse(p["iy"][ok], p["ix"][ok], du[ok], dv[ok], p["r"][ok]).ravel()

    # ------------------------------------------------------------------ cote reseau
    def f3d_t(self, device: torch.device) -> torch.Tensor:
        cle = str(device)
        if cle not in self._f3d_t:
            self._f3d_t[cle] = torch.from_numpy(self.f3d).to(device)
        return self._f3d_t[cle]

    def champs(self, q: torch.Tensor) -> torch.Tensor:
        """queue (B, N_OI * ny * nx) -> (B, N_OI, ny, nx) : du, dv (m/s), distance (km)."""
        return q.float().reshape(q.shape[0], N_OI, self.ny, self.nx)

    def increments_3d(self, q: torch.Tensor) -> torch.Tensor:
        """(B, 2, nz, ny, nx) en m/s : increment u, v etale x f(h) aux noeuds (0 sous le relief)."""
        c = self.champs(q)
        f = self.f3d_t(q.device)[None]
        return torch.stack([c[:, 0, None] * f, c[:, 1, None] * f], dim=1)

    def canaux(self, q: torch.Tensor, y_sd: torch.Tensor, nz: int) -> torch.Tensor:
        """(B, N_OI, nz, ny, nx) : du f / sd_u, dv f / sd_v, distance / ECHELLE_DISTANCE_KM."""
        inc = self.increments_3d(q)
        dist = (self.champs(q)[:, 2] / ECHELLE_DISTANCE_KM)[:, None, None].expand(-1, -1, nz, -1, -1)
        return torch.cat([inc[:, :1] / y_sd[0], inc[:, 1:] / y_sd[1], dist], dim=1)


def decouper_tfeat(tfeat: torch.Tensor, geom) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(cellules ERA5 (B, n_cellules), queue d'innovation (B, N_OI ny nx) ou None) depuis le vecteur tfeat."""
    from .forcage import N_SCAL
    from .geometry import N_TFEAT
    n = N_TFEAT + N_SCAL
    nc = geom.forcage.n_cellules
    cel = tfeat[:, n:n + nc]
    oi = getattr(geom, "oi", None)
    if oi is None:
        return cel, None
    q = tfeat[:, n + nc:]
    if q.shape[1] != oi.n_queue:
        raise ValueError(f"tfeat : {q.shape[1]} valeurs apres les cellules ERA5, {oi.n_queue} attendues "
                         "(canal d'innovation D2 : innovation.actif)")
    return cel, q


def predicteur_oi(geom):
    """Reference « interpolation optimale seule » : ERA5 aux noeuds + increment etale x f(h) (memes points de
    mesure que le modele), w par continuite ; format de metriques_loader / dso_hauteur_sol."""
    if getattr(geom, "oi", None) is None:
        raise ValueError("reference OI seule : innovation.actif requis")

    @torch.no_grad()
    def predire(s_raw, sensor, tfeat, y, j, stem):
        cel, q = decouper_tfeat(tfeat, geom)
        uv = geom.interp_era5.dense(cel) + geom.oi.increments_3d(q)
        valid = geom.valid_t[None]
        u, v = torch.where(valid, uv[:, 0], 0.0), torch.where(valid, uv[:, 1], 0.0)
        return [u, v, geom.w_continuite(u, v)]

    return predire


# ------------------------------------------------------------------ statistiques (jours d'entrainement)
def ajuster_statistiques(paires: list[dict], variances: dict, correlation: str = "gaussienne",
                         bornes_km: tuple[float, float] = (20.0, 50.0), pas_km: float = 1.0) -> dict:
    """Methode de Hollingsworth et Loennberg : covariance des innovations entre deux stations distinctes =
    sigma_b^2 rho(d ; L) (erreurs de mesure independantes), variance a une station = sigma_b^2 + sigma_o^2.
    paires : [{"distance_km", "covariance", "instants"}] ; variances : {station: variance} (moyenne u, v).
    L sur la grille [bornes] au pas `pas_km` et sigma_b^2 (fermeture lineaire) aux moindres carres ponderes par
    le nombre d'instants ; sigma_o^2 = variance moyenne - sigma_b^2. -> L_km, sigma_b, sigma_o, rapport_o_b,
    et si L est en butee."""
    d = np.array([p["distance_km"] for p in paires], float)
    c = np.array([p["covariance"] for p in paires], float)
    w = np.array([p.get("instants", 1) for p in paires], float)
    if len(d) == 0:
        raise ValueError("aucune paire de stations")
    var = float(np.mean(list(variances.values())))

    def rho(L):
        return np.exp(-0.5 * (d / L) ** 2) if correlation == "gaussienne" else np.exp(-d / L)

    grille = np.arange(bornes_km[0], bornes_km[1] + 1e-9, pas_km)
    best = None
    for L in grille:
        r = rho(L)
        sb2 = float(np.clip((w * c * r).sum() / max((w * r * r).sum(), 1e-12), 1e-6, var))
        cout = float((w * (c - sb2 * r) ** 2).sum())
        if best is None or cout < best[0]:
            best = (cout, float(L), sb2)
    _, L, sb2 = best
    so2 = max(var - sb2, 1e-6)
    return {"L_km": L, "sigma_b": math.sqrt(sb2), "sigma_o": math.sqrt(so2), "rapport_o_b": math.sqrt(so2 / sb2),
            "variance_innovation": var, "correlation": correlation, "bornes_km": list(bornes_km),
            "L_en_butee": bool(L <= bornes_km[0] + 1e-9 or L >= bornes_km[1] - 1e-9), "paires": int(len(d)),
            "distance_min_km": float(d.min())}
