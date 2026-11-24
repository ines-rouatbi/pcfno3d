"""OS4 national par OI 3D, regle configs/national/regle_os4_oi_v4_3.json (approuvee par Ines le 24/11/2026).

Experience OSSE distincte de C', CPU seulement. Capteurs ANEO simules a 25 m (verite + bruit 0,5 m/s), fond = ERA5
degradee, analyse = fond + OI 3D. Couche jugee : 0-100 m au-dessus du sol (RMSE vectorielle) ; 10 m descriptif ;
0-1 km et pire tranche « information ». Deux etapes, dans cet ordre :

    choisir : 72 reglages d'OI (forme, L, lambda, H) sur TOUTE la validation 2021, placement grille, N = 50 et 200 ;
              ecrit choix_validation/grille.csv et choix.json (jamais ecrases) ; ne lit pas le test.
    juger   : avec choix.json, les 5 placements, N = 10 ... 400 ; --lecture-test = les 180 instants du test 2022
              (deuxieme lecture, non independante, dite dans la regle) ; sans ce drapeau, essai sur 6 instants de
              VALIDATION (sortie suffixee _essai).

    python scripts/os4_oi_v4_3.py choisir -c national_1km_colab --degradation <runs>/degradation_era5_v4.json --sortie <dossier>
    python scripts/os4_oi_v4_3.py juger   -c national_1km_colab --degradation <runs>/degradation_era5_v4.json --sortie <dossier> [--lecture-test]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
REGLE = REPO / "configs" / "national" / "regle_os4_oi_v4_3.json"
BORNES = [0.0, 100.0, 300.0, 600.0, 1000.0]                 # = scripts/dso_hauteur_sol.BORNES
N_CHOIX = (50, 200)
EX_AEQUO = 0.002
_spec = importlib.util.spec_from_file_location("os4_oi_v4_base", REPO / "scripts" / "os4_oi_v4.py")
V2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V2)                                # CHOTTS, plus_eloigne, phase (inchanges)


# ---------------------------------------------------------------------------- OI (testable sans donnees)
def rho_h(d_km: np.ndarray, forme: str, L: float) -> np.ndarray:
    if forme == "gaussienne":
        return np.exp(-d_km ** 2 / (2 * L ** 2))
    if forme == "exponentielle":
        return np.exp(-d_km / L)
    raise ValueError(forme)


def alpha_oi(yx_km: np.ndarray, z: np.ndarray, d: np.ndarray, forme: str, L: float, lam: float, H: float) -> np.ndarray:
    """yx_km (n, 2), z (n,) hauteurs AGL, d (n, 2) innovations -> alpha (n, 2) = (R + lambda I)^-1 d."""
    dist = np.hypot(yx_km[:, None, 0] - yx_km[None, :, 0], yx_km[:, None, 1] - yx_km[None, :, 1])
    R = rho_h(dist, forme, L) * np.exp(-np.abs(z[:, None] - z[None, :]) / H)
    return np.linalg.solve(R + lam * np.eye(len(z)), d)


def glouton(variance: np.ndarray, cand_km: np.ndarray, n: int, forme: str, L: float, lam: float):
    """Meme regle que v4_2 avec la correlation retenue : candidat de variance maximale, puis
    var *= 1 - rho^2 / (1 + lambda) ; renvoie (ordre, gain marginal = baisse de la variance moyenne)."""
    v = variance.astype(np.float64).copy()
    ordre, g = [], []
    pris = np.zeros(len(v), bool)
    for _ in range(min(n, len(v))):
        j = int(np.argmax(np.where(pris, -np.inf, v)))
        avant = v.mean()
        r = rho_h(np.hypot(cand_km[:, 0] - cand_km[j, 0], cand_km[:, 1] - cand_km[j, 1]), forme, L)
        v = v * (1 - r ** 2 / (1 + lam))
        pris[j] = True
        ordre.append(j)
        g.append(float(avant - v.mean()))
    return np.asarray(ordre), np.asarray(g)


def choisir_reglage(grille: pd.DataFrame) -> dict:
    """grille : colonnes forme, L_km, lambda, H_m, rmse_0_100m_N50, rmse_0_100m_N200 -> ligne retenue (dict).
    Critere : moyenne sur N = 50 et 200 ; ex aequo (< 0,002 m/s) : gaussienne, L grand, lambda grand, H = 300."""
    g = grille.copy()
    g["critere"] = g[[f"rmse_0_100m_N{n}" for n in N_CHOIX]].mean(axis=1)
    m = g["critere"].min()
    ex = g[g["critere"] < m + EX_AEQUO].copy()
    ex["_f"] = (ex["forme"] != "gaussienne").astype(int)
    ex["_h"] = (ex["H_m"] != 300).astype(int)
    ex = ex.sort_values(["_f", "L_km", "lambda", "_h"], ascending=[True, False, False, True])
    r = ex.iloc[0]
    return {"forme": str(r["forme"]), "L_km": float(r["L_km"]), "lambda": float(r["lambda"]), "H_m": float(r["H_m"]),
            "critere_m_s": float(r["critere"]), "meilleur_m_s": float(m), "ex_aequo": int(len(ex))}


# ---------------------------------------------------------------------------- statistiques par instant
CHAMPS = ["b_sse", "b_su", "b_sv", "b_n", "b_tu", "b_tu2", "b_tv", "b_tv2", "b_absdvit", "b_dvit", "b_du", "b_dv",
          "b_dir", "b_ndir", "b_linf", "c_sse", "c_n", "m10_sse", "m10_n", "masse_sse", "masse_n"] + \
         [f"t{t}_s{c}" for t in range(4) for c in "uv"] + [f"t{t}_n" for t in range(4)]
IX = {k: i for i, k in enumerate(CHAMPS)}


def direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Direction meteorologique (d'ou vient le vent), degres."""
    return np.mod(270.0 - np.degrees(np.arctan2(v, u)), 360.0)


def stats_couche(au, av, tu, tv) -> dict:
    """Sommes d'une couche (tableaux 1D de meme taille, NaN ignores) : SSE vectorielle et par composante, sommes de la
    verite (R^2), vitesse, biais, direction (vitesse vraie >= 1 m/s), erreur vectorielle maximale."""
    ok = np.isfinite(au) & np.isfinite(av) & np.isfinite(tu) & np.isfinite(tv)
    au, av, tu, tv = (x[ok].astype(np.float64) for x in (au, av, tu, tv))
    du, dv = au - tu, av - tv
    va, vt = np.hypot(au, av), np.hypot(tu, tv)
    fort = vt >= 1.0
    dd = np.abs(direction(au[fort], av[fort]) - direction(tu[fort], tv[fort]))
    dd = np.minimum(dd, 360.0 - dd)
    e = np.hypot(du, dv)
    return {"b_sse": float((du ** 2 + dv ** 2).sum()), "b_su": float((du ** 2).sum()), "b_sv": float((dv ** 2).sum()),
            "b_n": float(du.size), "b_tu": float(tu.sum()), "b_tu2": float((tu ** 2).sum()), "b_tv": float(tv.sum()),
            "b_tv2": float((tv ** 2).sum()), "b_absdvit": float(np.abs(va - vt).sum()), "b_dvit": float((va - vt).sum()),
            "b_du": float(du.sum()), "b_dv": float(dv.sum()), "b_dir": float(dd.sum()), "b_ndir": float(dd.size),
            "b_linf": float(e.max()) if e.size else 0.0}


def metriques(S: np.ndarray, linf: np.ndarray) -> dict:
    """S (..., CHAMPS) sommes cumulees ; linf (...) maximum -> metriques (...)."""
    g = lambda k: S[..., IX[k]]
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.maximum(g("b_n"), 1)
        sst_u = g("b_tu2") - g("b_tu") ** 2 / n
        sst_v = g("b_tv2") - g("b_tv") ** 2 / n
        tr = np.stack([np.stack([np.sqrt(g(f"t{t}_s{c}") / np.maximum(g(f"t{t}_n"), 1)) for c in "uv"], -1)
                       for t in range(4)], -2)
        return {
            "rmse_vect_0_100m": np.sqrt(g("b_sse") / n),
            "rmse_u_0_100m": np.sqrt(g("b_su") / n), "rmse_v_0_100m": np.sqrt(g("b_sv") / n),
            "r2_u_0_100m": 1 - g("b_su") / sst_u, "r2_v_0_100m": 1 - g("b_sv") / sst_v,
            "mae_vitesse_0_100m": g("b_absdvit") / n, "biais_vitesse_0_100m": g("b_dvit") / n,
            "biais_u_0_100m": g("b_du") / n, "biais_v_0_100m": g("b_dv") / n,
            "mae_direction_0_100m_deg": g("b_dir") / np.maximum(g("b_ndir"), 1),
            "linf_0_100m": linf,
            "rmse_vect_10m": np.sqrt(g("m10_sse") / np.maximum(g("m10_n"), 1)),
            "rmse_vect_0_1km": np.sqrt(g("c_sse") / np.maximum(g("c_n"), 1)),
            "pire_tranche_0_1km": tr.max(axis=(-2, -1)),
            "residu_masse_rms": np.sqrt(g("masse_sse") / np.maximum(g("masse_n"), 1)),
        }


GAINS = ("rmse_vect_0_100m", "rmse_vect_10m", "rmse_vect_0_1km", "pire_tranche_0_1km")   # gain = 1 - x / sans reseau


def poids_bootstrap(jours: np.ndarray, n: int = 2000, graine: int = 2030) -> np.ndarray:
    """(n, instants) : nombre de fois que le jour de chaque instant est tire (bootstrap sur les jours)."""
    ju = np.unique(jours)
    rng = np.random.default_rng(graine)
    W = np.stack([np.bincount(rng.integers(0, len(ju), len(ju)), minlength=len(ju)) for _ in range(n)])
    return W[:, np.searchsorted(ju, jours)].astype(float)


def resumer(S: np.ndarray, L: np.ndarray, poids: np.ndarray) -> dict:
    """S (instants, cles, CHAMPS), L (instants, cles) linf, poids (B, instants) -> par metrique :
    valeur (cles,), ic (2, cles) de la valeur, et pour GAINS gain (cles,) et ic_gain (2, cles) vs la cle 0."""
    pt = metriques(S.sum(0), L.max(0))
    Sb = np.einsum("bi,ikc->bkc", poids, S)
    Lb = np.where(poids[:, :, None] > 0, L[None], -np.inf).max(1)
    bt = metriques(Sb, Lb)
    out = {}
    for k, v in pt.items():
        d = {"valeur": v, "ic": np.percentile(bt[k], [2.5, 97.5], axis=0)}
        if k in GAINS:
            d["gain"] = 1 - v / v[0]
            d["ic_gain"] = np.percentile(1 - bt[k] / bt[k][:, :1], [2.5, 97.5], axis=0)
        out[k] = d
    return out


def verdict(gain_ic_bas_N50: float) -> str:
    return "CONFORME" if gain_ic_bas_N50 >= 0.10 else "NON CONFORME"


# ---------------------------------------------------------------------------- donnees
class Contexte:
    """Geometrie, jeux, fond degrade, capteurs candidats (identiques a v4_2)."""

    def __init__(self, a, regle):
        from pcfno3d.aneo_entree import EntreeANEO
        from pcfno3d.cache import load_index
        from pcfno3d.config import Config
        from pcfno3d.dataset import is_synoptic, make_splits
        from pcfno3d.geometry import Geometry
        from pcfno3d.os6 import Contraintes, candidats
        from pcfno3d.realdata import georef_grille
        from pcfno3d.stations_reelles import cellules_codes
        p = regle["ce_qui_ne_change_pas"]
        deg_j = json.loads(Path(a.degradation).read_text(encoding="utf-8"))
        for k_j, val, tol in (("sigma_u", 1.91, 0.01), ("sigma_v", 2.02, 0.01), ("L_km", 55.0, 0.5)):
            if abs(float(deg_j[k_j]) - val) > tol:
                raise SystemExit(f"degradation : {k_j} {deg_j[k_j]} != regle {val}")
        assert "25 m" in p["capteurs"] and "0,5 m/s" in p["capteurs"]
        self.h_cap, self.bruit = 25.0, 0.5
        self.Ns = [int(n) for n in p["N"]]
        cfg = Config.load(a.config)
        geom = Geometry(cfg, device="cpu")
        self.cfg, self.geom, self.geo = cfg, geom, georef_grille(cfg, geom)
        self.dx, self.dy = float(geom.dx) / 1000, float(geom.dy) / 1000
        self.ny, self.nx, self.nz = geom.ny, geom.nx, geom.nz
        index = load_index(cfg)
        sp = make_splits(cfg, index)
        self.st = sp.stamp_of
        syn = lambda l: [s for s in l if is_synoptic(self.st[s])]
        train = syn(sp.train[0])
        self.train = list(np.random.default_rng(2030).choice(train, size=min(100, len(train)), replace=False))
        self.val, self.test = syn(sp.val[0]), syn(sp.test[0])
        assert not set(self.val) & set(sp.test[0])
        self.fo = geom.forcage
        self.deg = EntreeANEO(geom, a.degradation, 0.0, (50.0, 150.0))
        valid = np.asarray(geom.valid, bool)
        self.valid = valid
        hagl = np.asarray(geom.z_levels, float)[:, None, None] - np.asarray(geom.zt, float)[None]
        self.hagl = hagl
        self.ix_bas = np.flatnonzero(valid & (hagl >= 0) & (hagl < 100))
        self.ix_col = np.flatnonzero(valid & (hagl >= 0) & (hagl < 1000))
        self.ix_tr = [np.flatnonzero(valid & (hagl >= x) & (hagl < y)) for x, y in zip(BORNES[:-1], BORNES[1:])]
        self.terre10 = np.flatnonzero(((np.asarray(geom.z0) >= 0.001) & valid.any(axis=0)).ravel())
        polys = [[[a0, o0], [a0, o1], [a1, o1], [a1, o0]] for a0, a1, o0, o1 in V2.CHOTTS]
        cand = candidats(geom, Contraintes(exclusions=polys), self.geo)
        met = np.array(list(cellules_codes(geom).values()), float)
        if len(met):
            dmet = np.hypot((cand[:, None, 0] - met[None, :, 0]) * self.dy, (cand[:, None, 1] - met[None, :, 1]) * self.dx).min(1)
            cand = cand[dmet >= 8.0]
        self.cand = cand
        self.cand_km = np.stack([cand[:, 0] * self.dy, cand[:, 1] * self.dx], 1)
        terre = np.argwhere(np.asarray(geom.z0) >= 0.001)
        self.depart = int(np.argmin(np.hypot(cand[:, 0] - terre[:, 0].mean(), cand[:, 1] - terre[:, 1].mean())))
        self.gy, self.gx = (np.arange(self.ny) * self.dy)[:, None], (np.arange(self.nx) * self.dx)[None, :]

    def instant(self, stem, graine):
        """(Y (4, nz, ny, nx) verite, NaN hors valide pour u, v ; B (2, ...) fond degrade)."""
        Y = np.load(self.cfg.paths.cache_dir / f"{stem}_Y.npy").astype(np.float32)
        Y[:2] = np.where(self.valid[None], Y[:2], np.nan)
        v = self.fo.vecteur(self.st[stem])
        B = self.geom.interp_era5.dense_np(v[self.fo.n_extra - self.fo.n_cellules:self.fo.n_extra]).astype(np.float32)
        delta = self.deg.appliquer(np.zeros(self.fo.n_cellules, np.float32), np.random.default_rng(graine))[1]
        return Y, np.where(self.valid[None], B + delta[:, None], np.nan)

    def rho_sites(self, sites, forme, L):
        R = np.empty((len(sites), self.ny * self.nx), np.float32)
        for i, s in enumerate(sites):
            R[i] = rho_h(np.hypot(self.gy - self.cand_km[s, 0], self.gx - self.cand_km[s, 1]), forme, L).ravel()
        return R

    def vert(self, H):
        return np.exp(-np.abs(self.hagl - self.h_cap) / H).reshape(self.nz, -1).astype(np.float32)

    def stats(self, A, Y, t10, avec_masse: bool) -> tuple[np.ndarray, float]:
        """A (2, nz, ny, nx) analyse ; Y verite (4, ...) ; t10 verite a 10 m -> (vecteur CHAMPS, linf 0-100 m)."""
        from pcfno3d.atom_wind import champ_agl
        out = np.zeros(len(CHAMPS))
        au, av, tu, tv = A[0].reshape(-1), A[1].reshape(-1), Y[0].reshape(-1), Y[1].reshape(-1)
        b = stats_couche(au[self.ix_bas], av[self.ix_bas], tu[self.ix_bas], tv[self.ix_bas])
        for k, v in b.items():
            out[IX[k]] = v
        c = self.ix_col
        e = (au[c] - tu[c]) ** 2 + (av[c] - tv[c]) ** 2
        e = e[np.isfinite(e)]
        out[IX["c_sse"]], out[IX["c_n"]] = float(e.sum()), e.size
        for t, ix in enumerate(self.ix_tr):
            for cc, (a_, t_) in zip("uv", ((au, tu), (av, tv))):
                d = a_[ix] - t_[ix]
                d = d[np.isfinite(d)].astype(np.float64)
                out[IX[f"t{t}_s{cc}"]] = float((d ** 2).sum())
                out[IX[f"t{t}_n"]] = d.size
        a10u, a10v = champ_agl(A[0], A[1], self.geom, 10.0)
        e = ((a10u - t10[0]) ** 2 + (a10v - t10[1]) ** 2).reshape(-1)[self.terre10]
        e = e[np.isfinite(e)]
        out[IX["m10_sse"]], out[IX["m10_n"]] = float(e.sum()), e.size
        if avec_masse:
            import torch
            z = lambda x: torch.from_numpy(np.nan_to_num(x, nan=0.0).astype(np.float32))[None]
            r = self.geom.mass_residual(z(Y[3]), z(A[0]), z(A[1]), z(Y[2]))[0][self.geom.mass_mask_t].double()
            out[IX["masse_sse"]], out[IX["masse_n"]] = float((r ** 2).sum()), r.numel()
        return out, b["b_linf"]


def innovations(ctx, s, Y, B, b_sites_par_site):
    """Capteurs s (indices de candidats) : innovations (n, 2) a 25 m et masque des valeurs finies."""
    from pcfno3d.atom_wind import champ_agl
    tu, tv = champ_agl(Y[0], Y[1], ctx.geom, ctx.h_cap)
    bu, bv = champ_agl(B[0], B[1], ctx.geom, ctx.h_cap)
    iy, ix = ctx.cand[s, 0], ctx.cand[s, 1]
    d = np.stack([tu[iy, ix], tv[iy, ix]], 1) + b_sites_par_site - np.stack([bu[iy, ix], bv[iy, ix]], 1)
    return d, np.isfinite(d).all(axis=1)


def analyse(ctx, B, s, d, ok, RHO_s, vert, forme, L, lam, H):
    al = alpha_oi(ctx.cand_km[s][ok], np.full(int(ok.sum()), ctx.h_cap), d[ok], forme, L, lam, H)
    F = al.T @ RHO_s[ok]
    return B + (F[:, None, :] * vert[None]).reshape(2, ctx.nz, ctx.ny, ctx.nx)


# ---------------------------------------------------------------------------- etape 1 : choix sur la validation
def commande_choisir(a) -> int:
    from pcfno3d.atom_wind import champ_agl
    from pcfno3d.regles import sortie_neuve, verifier
    regle, h_regle = verifier(REGLE)
    e1 = regle["etape_1_choix_sur_validation_2021"]
    gr = e1["grille"]
    formes = ["gaussienne" if f.startswith("gaussienne") else "exponentielle" for f in gr["forme_horizontale"]]
    reglages = list(product(formes, [float(x) for x in gr["L_km"]], [float(x) for x in gr["lambda"]], [float(x) for x in gr["H_m"]]))
    assert len(reglages) == gr["combinaisons"] == 72
    nom = "choix_validation_essai" if a.essai else "choix_validation"
    sortie = sortie_neuve(Path(a.sortie) / nom, "choix.json")
    ctx = Contexte(a, regle)
    if a.essai:
        ctx.val = ctx.val[:2]                 # essai du code : 2 instants, sortie a part, jamais lue par juger
    print(f"regle {REGLE.name} {h_regle[:12]}... verifiee | ETAPE 1 : choix sur la validation 2021 "
          f"({len(ctx.val)} instants synoptiques), {len(reglages)} reglages", flush=True)
    plan = V2.plus_eloigne(ctx.cand_km, max(N_CHOIX), ctx.depart)
    sites = plan[:max(N_CHOIX)]
    RHO = {(f, L): ctx.rho_sites(sites, f, L) for f in formes for L in sorted({r[1] for r in reglages})}
    VERT = {H: ctx.vert(H) for H in sorted({r[3] for r in reglages})}
    cles = [("sans_reseau", 0)] + [(r, N) for r in reglages for N in N_CHOIX]
    S = np.zeros((len(ctx.val), len(cles), len(CHAMPS)))
    Lm = np.zeros((len(ctx.val), len(cles)))
    t0 = time.time()
    for n, stem in enumerate(ctx.val):
        Y, B = ctx.instant(stem, 1000 + n)
        t10 = champ_agl(Y[0], Y[1], ctx.geom, 10.0)
        S[n, 0], Lm[n, 0] = ctx.stats(B, Y, t10, avec_masse=False)
        bruit = np.random.default_rng(9000 + n).standard_normal((len(sites), 2)) * ctx.bruit
        d_all, ok_all = innovations(ctx, sites, Y, B, bruit)
        k = 1
        for r in reglages:
            for N in N_CHOIX:
                d, ok = d_all[:N], ok_all[:N]
                A = analyse(ctx, B, sites[:N], d, ok, RHO[(r[0], r[1])][:N], VERT[r[3]], *r)
                S[n, k], Lm[n, k] = ctx.stats(A, Y, t10, avec_masse=False)
                k += 1
        print(f"  {n + 1}/{len(ctx.val)} instants ({time.time() - t0:.0f} s)", flush=True)
    m = metriques(S.sum(0), Lm.max(0))
    lignes = []
    for r in reglages:
        l = {"forme": r[0], "L_km": r[1], "lambda": r[2], "H_m": r[3]}
        for N in N_CHOIX:
            k = cles.index((r, N))
            for nom, cle in (("rmse_0_100m", "rmse_vect_0_100m"), ("rmse_10m", "rmse_vect_10m"), ("pire_tranche", "pire_tranche_0_1km")):
                l[f"{nom}_N{N}"] = float(m[cle][k])
        lignes.append(l)
    grille = pd.DataFrame(lignes)
    grille.to_csv(sortie / "grille.csv", index=False)
    choix = choisir_reglage(grille)
    choix.update({"regle": REGLE.name, "regle_sha256": h_regle, "jeu": "validation 2021, tous les instants synoptiques",
                  "instants": len(ctx.val), "placement": "grille reguliere", "N": list(N_CHOIX),
                  "sans_reseau_rmse_0_100m": float(m["rmse_vect_0_100m"][0]),
                  "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    (sortie / "choix.json").write_text(json.dumps(choix, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CHOIX", json.dumps(choix, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------------------- etape 2 : jugement
def commande_juger(a) -> int:
    from pcfno3d.atom_wind import champ_agl
    from pcfno3d.regles import sortie_neuve, verifier
    regle, h_regle = verifier(REGLE)
    fchoix = Path(a.sortie) / "choix_validation" / "choix.json"
    if not fchoix.exists():
        raise SystemExit(f"{fchoix} absent : l'etape 1 (choisir) doit avoir tourne avant tout jugement")
    choix = json.loads(fchoix.read_text(encoding="utf-8"))
    if choix.get("regle_sha256") != h_regle:
        raise SystemExit("choix.json n'a pas ete produit par cette regle")
    forme, L, lam, H = choix["forme"], choix["L_km"], choix["lambda"], choix["H_m"]
    suffixe = "" if a.lecture_test else "_essai"
    sortie = sortie_neuve(Path(a.sortie + suffixe) if suffixe else Path(a.sortie), "bilan.json")
    ctx = Contexte(a, regle)
    if a.lecture_test:
        evalues = ctx.test
        assert len(evalues) == 180, len(evalues)
    else:
        evalues = ctx.val[:6]
    print(f"regle {REGLE.name} {h_regle[:12]}... verifiee | reglage {forme} L {L} km, lambda {lam}, H {H} m | "
          f"{'TEST 2022 (deuxieme lecture, non independante)' if a.lecture_test else 'essai du code (validation)'}", flush=True)
    nmax = max(ctx.Ns)
    # glouton : variance d'erreur du fond a 25 m sur 100 instants d'ENTRAINEMENT (comme v4_2), ordre avec le reglage retenu
    var = np.zeros(len(ctx.cand))
    for n, stem in enumerate(ctx.train):
        Y, B = ctx.instant(stem, 1000 + n)
        bu, bv = champ_agl(B[0], B[1], ctx.geom, ctx.h_cap)
        tu, tv = champ_agl(Y[0], Y[1], ctx.geom, ctx.h_cap)
        var += 0.5 * ((bu - tu) ** 2 + (bv - tv) ** 2)[ctx.cand[:, 0], ctx.cand[:, 1]]
    var = np.nan_to_num(var / len(ctx.train), nan=0.0)
    ordre_g, gains_g = glouton(var, ctx.cand_km, nmax, forme, L, lam)
    plans = {"glouton": ordre_g, "grille_reguliere": V2.plus_eloigne(ctx.cand_km, nmax, ctx.depart)}
    for g in (1, 2, 3):
        plans[f"aleatoire_{g}"] = np.random.default_rng(g).permutation(len(ctx.cand))[:nmax]
    sites = np.unique(np.concatenate(list(plans.values())))
    pos = {int(s): i for i, s in enumerate(sites)}
    RHO = ctx.rho_sites(sites, forme, L)
    vert = ctx.vert(H)
    print(f"plans prets : {len(sites)} sites, correlations {RHO.nbytes / 1e9:.1f} Go", flush=True)
    cles = [("sans_reseau", 0)] + [(nom, N) for nom in plans for N in ctx.Ns]
    S = np.zeros((len(evalues), len(cles), len(CHAMPS)))
    Lm = np.zeros((len(evalues), len(cles)))
    t0 = time.time()
    for n, stem in enumerate(evalues):
        Y, B = ctx.instant(stem, 1000 + 100000 + n)
        t10 = champ_agl(Y[0], Y[1], ctx.geom, 10.0)
        S[n, 0], Lm[n, 0] = ctx.stats(B, Y, t10, avec_masse=True)
        bruit = np.random.default_rng(9000 + n).standard_normal((len(sites), 2)) * ctx.bruit
        for nom, plan in plans.items():
            ps_all = np.asarray([pos[int(k)] for k in plan])
            d_all, ok_all = innovations(ctx, plan, Y, B, bruit[ps_all])
            for N in ctx.Ns:
                A = analyse(ctx, B, plan[:N], d_all[:N], ok_all[:N], RHO[ps_all[:N]], vert, forme, L, lam, H)
                k = cles.index((nom, N))
                S[n, k], Lm[n, k] = ctx.stats(A, Y, t10, avec_masse=True)
        print(f"  {n + 1}/{len(evalues)} instants ({time.time() - t0:.0f} s)", flush=True)
    jours = np.array([pd.Timestamp(ctx.st[s]).normalize().value for s in evalues])
    R = resumer(S, Lm, poids_bootstrap(jours))
    # sorties
    lignes = []
    for k, (nom, N) in enumerate(cles):
        l = {"strategie": nom, "N": N}
        for met, d in R.items():
            l[met] = float(d["valeur"][k])
            l[f"{met}_ic95_bas"], l[f"{met}_ic95_haut"] = float(d["ic"][0][k]), float(d["ic"][1][k])
            if "gain" in d:
                l[f"gain_{met}"] = float(d["gain"][k])
                l[f"gain_{met}_ic95_bas"], l[f"gain_{met}_ic95_haut"] = float(d["ic_gain"][0][k]), float(d["ic_gain"][1][k])
        lignes.append(l)
    tab = pd.DataFrame(lignes)
    tab.to_csv(sortie / "metriques.csv", index=False)
    tab[["strategie", "N"] + [c for c in tab.columns if c.startswith("gain_") or c in GAINS]].to_csv(sortie / "courbe.csv", index=False)
    verdicts, nmin = {}, {}
    for strat in plans:
        t = tab[tab.strategie == strat].sort_values("N")
        r50 = t[t.N == 50].iloc[0]
        verdicts[strat] = {"N": 50, "gain_0_100m": float(r50["gain_rmse_vect_0_100m"]),
                           "ic95": [float(r50["gain_rmse_vect_0_100m_ic95_bas"]), float(r50["gain_rmse_vect_0_100m_ic95_haut"])],
                           "verdict": verdict(float(r50["gain_rmse_vect_0_100m_ic95_bas"]))}
        ic = t[t["gain_rmse_vect_0_100m_ic95_bas"] >= 0.10]
        nmin[strat] = {"N_minimal_ic95_0_100m": int(ic.N.iloc[0]) if len(ic) else "non atteint jusqu'a N = 400"}
    (sortie / "n_minimal.json").write_text(json.dumps(nmin, indent=2, ensure_ascii=False), encoding="utf-8")

    def plan_geojson(ordre, fichier, gains=None):
        feats = []
        for rang, j in enumerate(ordre, start=1):
            lat, lon = ctx.geo.to_latlon(np.array([ctx.cand[j, 1] * ctx.geom.dx]), np.array([ctx.cand[j, 0] * ctx.geom.dy]))
            pr = {"id": f"ANEO-{rang:03d}", "priorite": rang, "phase": V2.phase(rang), "lat": round(float(lat[0]), 5),
                  "lon": round(float(lon[0]), 5), "hauteur_m": ctx.h_cap,
                  "source": "OS4 OI (simulation OSSE, regle_os4_oi_v4_3)"}
            if gains is not None:
                pr["gain_marginal"] = round(float(gains[rang - 1]), 6)
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [pr["lon"], pr["lat"]]},
                          "properties": pr})
        (sortie / fichier).write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False),
                                      encoding="utf-8")
    plan_geojson(plans["grille_reguliere"], "plan_grille.geojson")
    plan_geojson(ordre_g, "plan_glouton.geojson", gains_g)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for i, strat in enumerate(plans):
            t = tab[tab.strategie == strat].sort_values("N")
            col = f"C{i}"
            ax.plot(t.N, 100 * t.gain_rmse_vect_0_100m, marker="o", color=col, label=f"{strat} (0-100 m)")
            ax.fill_between(t.N, 100 * t.gain_rmse_vect_0_100m_ic95_bas, 100 * t.gain_rmse_vect_0_100m_ic95_haut,
                            color=col, alpha=0.12)
            ax.plot(t.N, 100 * t.gain_rmse_vect_0_1km, ls=":", color=col, alpha=0.7)
        ax.plot([], [], ls=":", color="grey", label="0-1 km (information, non visee par des capteurs a 25 m)")
        ax.axhline(10, color="r", ls="--", label="critere : +10 % (borne basse IC 95 %, N = 50)")
        ax.axvline(50, color="k", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("capteurs ANEO a 25 m (N)")
        ax.set_ylabel("gain de RMSE vectorielle (%)")
        ax.set_title("OS4 v4_3 : gain selon le nombre de capteurs (simulation, IC 95 %)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(sortie / "courbe.png", dpi=150)
    except Exception as exc:
        print("figure non produite :", exc)
    ref = {k: float(R[k]["valeur"][0]) for k in R}
    bilan = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "regle": REGLE.name,
             "regle_sha256": h_regle, "reglage_oi": {"forme": forme, "L_km": L, "lambda": lam, "H_m": H,
                                                     "source": "choix_validation/choix.json"},
             "lecture": ("test 2022, deuxieme lecture pour l'OS4, non independante (lu le 15/11/2026 sous regle_os4_oi_v4_2)"
                         if a.lecture_test else "essai du code (validation)"),
             "instants": len(evalues), "jours": int(len(np.unique(jours))), "candidats": int(len(ctx.cand)),
             "capteurs": "25 m au-dessus du sol, verite + bruit 0,5 m/s", "sans_reseau": ref,
             "verdict_N50_0_100m": verdicts, "N_minimal": nmin,
             "information": "0-1 km et pire tranche : non visees par des capteurs a 25 m, sans verdict",
             "v4_2": "regle_os4_oi_v4_2 reste publiee avec son verdict d'origine (critere 0-1 km non atteint jusqu'a N = 400)",
             "nature": "OSSE sur la verite generee, experience distincte de C'"}
    (sortie / "bilan.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    titre = "test 2022 (deuxieme lecture, non independante)" if a.lecture_test else "ESSAI du code (validation)"
    md = [f"# OS4 par OI 3D, regle v4_3 — {titre}", "",
          f"Regle `{REGLE.name}` (`{h_regle[:12]}…`). Reglage choisi sur la validation 2021 : {forme}, L {L:g} km, "
          f"lambda {lam:g}, H {H:g} m. Capteurs a 25 m.", "",
          "Verdict (borne basse de l'IC 95 % du gain 0-100 m >= 10 % a N = 50) :", ""]
    for s, v in verdicts.items():
        md.append(f"- {s} : gain {100 * v['gain_0_100m']:+.2f} % [{100 * v['ic95'][0]:+.2f} ; {100 * v['ic95'][1]:+.2f}] -> **{v['verdict']}**")
    md += ["", f"Sans reseau : 0-100 m {ref['rmse_vect_0_100m']:.3f} m/s, 10 m {ref['rmse_vect_10m']:.3f}, "
           f"0-1 km {ref['rmse_vect_0_1km']:.3f} (information), pire tranche {ref['pire_tranche_0_1km']:.3f} (information).", "",
           "```", tab[["strategie", "N", "gain_rmse_vect_0_100m", "gain_rmse_vect_0_100m_ic95_bas",
                       "gain_rmse_vect_0_100m_ic95_haut", "gain_rmse_vect_10m", "gain_rmse_vect_0_1km"]].round(4).to_string(index=False), "```"]
    (sortie / "bilan.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("etape", choices=["choisir", "juger"])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--degradation", required=True)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--lecture-test", action="store_true", help="juger : les 180 instants du test 2022")
    ap.add_argument("--essai", action="store_true", help="choisir : essai du code sur 2 instants (choix_validation_essai/)")
    a = ap.parse_args(argv)
    if a.etape == "juger" and a.essai:
        raise SystemExit("juger : l'essai est le mode sans --lecture-test")
    if a.etape == "choisir" and a.lecture_test:
        raise SystemExit("l'etape de choix ne lit jamais le test")
    return commande_choisir(a) if a.etape == "choisir" else commande_juger(a)


if __name__ == "__main__":
    raise SystemExit(main())
