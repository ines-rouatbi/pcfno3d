"""OS4 national par OI 3D (regle configs/national/regle_os4_oi_v4_2.json) : experience OSSE distincte de C', CPU.

Fond = ERA5 degradee (variante D) ; capteurs simules a 25 m au-dessus du sol (verite + bruit 0,5 m/s) ; analyse =
fond + OI 3D (L 150 km, lambda 4, H 300 m) ; N = 10 ... 400 ; placements glouton (variance du fond, ENTRAINEMENT),
regulier, aleatoire x 3 ; variante drone (4 profils). Metrique : pire tranche 0-1 km (RMSE u, v). Le test 2022 est lu
UNE fois (--lecture-test) ; sans ce drapeau, essai sur la VALIDATION (sortie suffixee _essai).

    python scripts/os4_oi_v4.py -c national_1km_colab --degradation <runs>/degradation_era5_v4.json --sortie <dossier> [--lecture-test]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
REGLE = REPO / "configs" / "national" / "regle_os4_oi_v4_2.json"
BORNES = [0.0, 100.0, 300.0, 600.0, 1000.0]                 # = scripts/dso_hauteur_sol.BORNES


# ---------------------------------------------------------------------------- OI 3D (testable sans donnees)
def alpha_oi(yx_km: np.ndarray, z: np.ndarray, d: np.ndarray, L: float, lam: float, H: float) -> np.ndarray:
    """yx_km (n, 2), z (n,) hauteurs AGL, d (n, 2) innovations -> alpha (n, 2) = (R + lambda I)^-1 d."""
    dist = np.hypot(yx_km[:, None, 0] - yx_km[None, :, 0], yx_km[:, None, 1] - yx_km[None, :, 1])
    R = np.exp(-dist ** 2 / (2 * L ** 2)) * np.exp(-np.abs(z[:, None] - z[None, :]) / H)
    return np.linalg.solve(R + lam * np.eye(len(z)), d)


def increment_3d(alpha: np.ndarray, rho_h: np.ndarray, z: np.ndarray, verticaux: dict) -> np.ndarray:
    """alpha (n, 2), rho_h (n, ny*nx) correlations horizontales des points avec les colonnes, z (n,) hauteurs,
    verticaux {z: (nz, ny*nx) exp(-|h - z| / H)} -> increment (2, nz, ny*nx)."""
    inc = None
    for zz in np.unique(z):
        k = z == zz
        F = alpha[k].T @ rho_h[k]                                  # (2, ny*nx)
        part = F[:, None, :] * verticaux[float(zz)][None]
        inc = part if inc is None else inc + part
    return inc


def glouton(variance: np.ndarray, cand_km: np.ndarray, n: int, L: float, lam: float, gains: bool = False):
    """Ordre glouton : candidat de variance courante maximale, puis var *= 1 - rho^2 / (1 + lambda).
    gains=True : aussi le gain marginal de chaque site (baisse de la variance moyenne sur les candidats)."""
    v = variance.astype(np.float64).copy()
    ordre, g = [], []
    pris = np.zeros(len(v), bool)
    for _ in range(min(n, len(v))):
        w = np.where(pris, -np.inf, v)
        j = int(np.argmax(w))
        avant = v.mean()
        r = np.exp(-((cand_km[:, 0] - cand_km[j, 0]) ** 2 + (cand_km[:, 1] - cand_km[j, 1]) ** 2) / (2 * L ** 2))
        v = v * (1 - r ** 2 / (1 + lam))
        pris[j] = True
        ordre.append(j)
        g.append(float(avant - v.mean()))
    return (np.asarray(ordre), np.asarray(g)) if gains else np.asarray(ordre)


def plus_eloigne(cand_km: np.ndarray, n: int, depart: int) -> np.ndarray:
    choisis = [int(depart)]
    dmin = np.hypot(cand_km[:, 0] - cand_km[depart, 0], cand_km[:, 1] - cand_km[depart, 1])
    for _ in range(min(n, len(cand_km)) - 1):
        k = int(np.argmax(dmin))
        choisis.append(k)
        dmin = np.minimum(dmin, np.hypot(cand_km[:, 0] - cand_km[k, 0], cand_km[:, 1] - cand_km[k, 1]))
    return np.asarray(choisis)


# ---------------------------------------------------------------------------- calcul
PHASES = ((1, 10), (11, 25), (26, 50), (51, 100), (101, 200), (201, 400))
CHOTTS = ((33.55, 34.15, 7.85, 8.95), (33.85, 34.20, 7.60, 8.10), (33.80, 34.00, 8.95, 9.60),
          (35.40, 35.65, 10.20, 10.50), (35.75, 35.95, 10.20, 10.35))   # lat_min, lat_max, lon_min, lon_max


def phase(rang: int) -> int:
    return next(k + 1 for k, (a, b) in enumerate(PHASES) if a <= rang <= b)


def metrique(sse: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """sse, cnt (..., tranches, 2) -> max sur tranches et u, v du RMSE (...)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(sse / np.maximum(cnt, 1)).max(axis=(-2, -1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--degradation", required=True)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--lecture-test", action="store_true", help="LA lecture unique du test 2022")
    ap.add_argument("--max-instants", type=int, default=None, help="essai (validation seulement)")
    a = ap.parse_args(argv)
    if a.lecture_test and a.max_instants:
        raise SystemExit("la lecture du test est complete ou n'a pas lieu")
    from pcfno3d.aneo_entree import EntreeANEO
    from pcfno3d.atom_wind import champ_agl
    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.dataset import is_synoptic, make_splits
    from pcfno3d.geometry import Geometry
    from pcfno3d.os6 import Contraintes, candidats
    from pcfno3d.realdata import georef_grille
    from pcfno3d.regles import sortie_neuve, verifier
    from pcfno3d.stations_reelles import cellules_codes
    regle, h_regle = verifier(REGLE)
    suffixe = "" if a.lecture_test else "_essai"
    sortie = sortie_neuve(Path(a.sortie + suffixe), "bilan.json")
    print(f"regle {REGLE.name} {h_regle[:12]}... verifiee | "
          f"{'LECTURE UNIQUE DU TEST' if a.lecture_test else 'essai du code (validation)'}", flush=True)
    p = regle["fond_sans_reseau"]["parametres"]
    deg_j = json.loads(Path(a.degradation).read_text(encoding="utf-8"))
    for k_r, k_j, tol in (("sigma_u_m_s", "sigma_u", 0.01), ("sigma_v_m_s", "sigma_v", 0.01), ("L_km", "L_km", 0.5)):
        if abs(float(p[k_r]) - float(deg_j[k_j])) > tol:
            raise SystemExit(f"degradation : {k_j} {deg_j[k_j]} != regle {p[k_r]}")
    L, lam, H = 150.0, 4.0, 300.0
    bruit_s = float(regle["capteurs"]["bruit_m_s"])
    h_cap = float(regle["capteurs"]["hauteur_m_agl"])
    Ns = [int(n) for n in regle["N"]]
    nmax = max(Ns)

    cfg = Config.load(a.config)
    geom = Geometry(cfg, device="cpu")
    geo = georef_grille(cfg, geom)
    dx, dy = float(geom.dx) / 1000, float(geom.dy) / 1000
    ny, nx, nz = geom.ny, geom.nx, geom.nz
    index = load_index(cfg)
    sp = make_splits(cfg, index)
    st = sp.stamp_of
    train = [s for s in sp.train[0] if is_synoptic(st[s])]
    train = list(np.random.default_rng(2030).choice(train, size=min(100, len(train)), replace=False))
    if a.lecture_test:
        evalues = [s for s in sp.test[0] if is_synoptic(st[s])]
        assert len(evalues) == 180, len(evalues)
    else:
        evalues = [s for s in sp.val[0] if is_synoptic(st[s])][:a.max_instants or 6]
        assert not set(evalues) & set(sp.test[0])
    fo = geom.forcage
    deg = EntreeANEO(geom, a.degradation, 0.0, (50.0, 150.0))
    valid = np.asarray(geom.valid, bool)
    hagl = np.asarray(geom.z_levels, float)[:, None, None] - np.asarray(geom.zt, float)[None]
    noms_tr = [f"{x:g}-{y:g} m" for x, y in zip(BORNES[:-1], BORNES[1:])]
    idx_tr = [np.flatnonzero(valid & (hagl >= x) & (hagl < y)) for x, y in zip(BORNES[:-1], BORNES[1:])]
    terre10 = np.flatnonzero(((np.asarray(geom.z0) >= 0.001) & valid.any(axis=0)).ravel())
    vert = np.exp(-np.abs(hagl - h_cap) / H).reshape(nz, -1).astype(np.float32)

    def instant(stem, graine):
        Y = np.load(cfg.paths.cache_dir / f"{stem}_Y.npy").astype(np.float32)[:2]
        Y = np.where(valid[None], Y, np.nan)
        v = fo.vecteur(st[stem])
        B = geom.interp_era5.dense_np(v[fo.n_extra - fo.n_cellules:fo.n_extra]).astype(np.float32)
        delta = deg.appliquer(np.zeros(fo.n_cellules, np.float32), np.random.default_rng(graine))[1]
        return Y, np.where(valid[None], B + delta[:, None], np.nan)

    # candidats : OS6 + hors chotts + 8 km des aeroports METAR
    polys = [[[a0, o0], [a0, o1], [a1, o1], [a1, o0]] for a0, a1, o0, o1 in CHOTTS]
    cand = candidats(geom, Contraintes(exclusions=polys), geo)
    met = np.array(list(cellules_codes(geom).values()), float)
    if len(met):
        dmet = np.hypot((cand[:, None, 0] - met[None, :, 0]) * dy, (cand[:, None, 1] - met[None, :, 1]) * dx).min(1)
        cand = cand[dmet >= 8.0]
    cand_km = np.stack([cand[:, 0] * dy, cand[:, 1] * dx], 1)
    print(f"candidats {len(cand)} | entrainement {len(train)} | evalues {len(evalues)}", flush=True)

    # 1. glouton sur la variance d'erreur du fond a 25 m (ENTRAINEMENT seulement)
    var = np.zeros(len(cand))
    for n, stem in enumerate(train):
        Y, B = instant(stem, 1000 + n)
        bu, bv = champ_agl(B[0], B[1], geom, h_cap)
        tu, tv = champ_agl(Y[0], Y[1], geom, h_cap)
        var += 0.5 * ((bu - tu) ** 2 + (bv - tv) ** 2)[cand[:, 0], cand[:, 1]]
    var = np.nan_to_num(var / len(train), nan=0.0)
    terre = np.argwhere(np.asarray(geom.z0) >= 0.001)
    depart = int(np.argmin(np.hypot(cand[:, 0] - terre[:, 0].mean(), cand[:, 1] - terre[:, 1].mean())))
    ordre_g, gains_g = glouton(var, cand_km, nmax, L, lam, gains=True)
    plans = {"glouton": ordre_g, "grille_reguliere": plus_eloigne(cand_km, nmax, depart)}
    for g in (1, 2, 3):
        plans[f"aleatoire_{g}"] = np.random.default_rng(g).permutation(len(cand))[:nmax]
    sites = np.unique(np.concatenate(list(plans.values())))
    pos = {int(s): i for i, s in enumerate(sites)}
    gy, gx = (np.arange(ny) * dy)[:, None], (np.arange(nx) * dx)[None, :]
    RHO = np.empty((len(sites), ny * nx), np.float32)
    for i, s in enumerate(sites):
        RHO[i] = np.exp(-((gy - cand_km[s, 0]) ** 2 + (gx - cand_km[s, 1]) ** 2) / (2 * L ** 2)).ravel()
    print(f"plans prets : {len(sites)} sites, correlations {RHO.nbytes / 1e9:.1f} Go", flush=True)

    # 2. evaluation : sommes d'erreurs PAR INSTANT (pour le bootstrap sur les jours)
    cles = [("sans_reseau", 0)] + [(nom, N) for nom in plans for N in Ns]
    ci = {c: i for i, c in enumerate(cles)}
    S3 = np.zeros((len(evalues), len(cles), len(idx_tr), 2))
    C3 = np.zeros_like(S3)
    S10 = np.zeros((len(evalues), len(cles)))
    C10 = np.zeros_like(S10)

    def ajouter(n, cle, A, Y, t10):
        k = ci[cle]
        for t, ix in enumerate(idx_tr):
            for c in (0, 1):
                e = A[c].reshape(-1)[ix] - Y[c].reshape(-1)[ix]
                e = e[np.isfinite(e)].astype(np.float64)
                S3[n, k, t, c], C3[n, k, t, c] = float((e ** 2).sum()), e.size
        a10u, a10v = champ_agl(A[0], A[1], geom, 10.0)
        e = ((a10u - t10[0]) ** 2 + (a10v - t10[1]) ** 2).reshape(-1)[terre10]
        e = e[np.isfinite(e)]
        S10[n, k], C10[n, k] = float(e.sum()), e.size

    t0 = time.time()
    for n, stem in enumerate(evalues):
        Y, B = instant(stem, 1000 + 100000 + n)
        t10 = champ_agl(Y[0], Y[1], geom, 10.0)
        ajouter(n, ("sans_reseau", 0), B, Y, t10)
        rng = np.random.default_rng(9000 + n)
        b_sites = rng.standard_normal((len(sites), 2)) * bruit_s
        tu, tv = champ_agl(Y[0], Y[1], geom, h_cap)
        bu, bv = champ_agl(B[0], B[1], geom, h_cap)
        for nom, plan in plans.items():
            for N in Ns:
                s = plan[:N]
                iy, ix = cand[s, 0], cand[s, 1]
                ps = np.asarray([pos[int(k)] for k in s])
                d = np.stack([tu[iy, ix], tv[iy, ix]], 1) + b_sites[ps] - np.stack([bu[iy, ix], bv[iy, ix]], 1)
                ok = np.isfinite(d).all(axis=1)
                al = alpha_oi(cand_km[s][ok], np.full(int(ok.sum()), h_cap), d[ok], L, lam, H)
                F = al.T @ RHO[ps[ok]]                                              # (2, ny*nx)
                inc = (F[:, None, :] * vert[None]).reshape(2, nz, ny, nx)
                ajouter(n, (nom, N), B + inc, Y, t10)
        print(f"  {n + 1}/{len(evalues)} instants ({time.time() - t0:.0f} s)", flush=True)

    # 3. resultats et IC 95 % par bootstrap sur les jours
    jours = np.array([pd.Timestamp(st[s]).normalize().value for s in evalues])
    ju = np.unique(jours)
    rng = np.random.default_rng(2030)
    W = np.stack([np.bincount(rng.integers(0, len(ju), len(ju)), minlength=len(ju)) for _ in range(2000)])
    poids = W[:, np.searchsorted(ju, jours)].astype(float)                      # (2000, n_instants)
    m_pt = metrique(S3.sum(0), C3.sum(0))                                        # (cles,)
    m_ref = float(m_pt[0])
    m_boot = metrique(np.einsum("bi,iktc->bktc", poids, S3), np.einsum("bi,iktc->bktc", poids, C3))  # (2000, cles)
    r10 = np.sqrt(S10.sum(0) / np.maximum(C10.sum(0), 1))
    bas = np.sqrt(S3.sum(0)[:, 0, :] / np.maximum(C3.sum(0)[:, 0, :], 1)).max(axis=1)
    lignes = []
    for cle, k in ci.items():
        g_boot = 1 - m_boot[:, k] / m_boot[:, 0]
        lignes.append({"strategie": cle[0], "N": cle[1], "pire_tranche_0_1km": float(m_pt[k]),
                       "gain": float(1 - m_pt[k] / m_ref), "gain_ic95_bas": float(np.percentile(g_boot, 2.5)),
                       "gain_ic95_haut": float(np.percentile(g_boot, 97.5)),
                       "tranche_0_100m": float(bas[k]), "gain_0_100m": float(1 - bas[k] / bas[0]),
                       "rmse_vect_10m": float(r10[k]), "gain_10m": float(1 - r10[k] / r10[0]),
                       **{f"rmse_{t}_{'uv'[c]}": float(np.sqrt(S3.sum(0)[k, ti, c] / max(C3.sum(0)[k, ti, c], 1)))
                          for ti, t in enumerate(noms_tr) for c in (0, 1)}})
    tab = pd.DataFrame(lignes)
    tab.to_csv(sortie / "courbe.csv", index=False)
    nmin = {}
    for strat in plans:
        t = tab[tab.strategie == strat].sort_values("N")
        ic = t[t.gain_ic95_bas >= 0.10]
        pt = t[t.gain >= 0.10]
        nmin[strat] = {"N_minimal_ic95": int(ic.N.iloc[0]) if len(ic) else "non atteint jusqu'a N = 400",
                       "N_minimal_ponctuel": int(pt.N.iloc[0]) if len(pt) else "non atteint jusqu'a N = 400"}
    (sortie / "n_minimal.json").write_text(json.dumps(nmin, indent=2, ensure_ascii=False), encoding="utf-8")
    feats = []
    for rang, (j, gm) in enumerate(zip(ordre_g, gains_g), start=1):
        lat, lon = geo.to_latlon(np.array([cand[j, 1] * geom.dx]), np.array([cand[j, 0] * geom.dy]))
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [round(float(lon[0]), 5), round(float(lat[0]), 5)]},
                      "properties": {"id": f"ANEO-{rang:03d}", "priorite": rang, "phase": phase(rang),
                                     "lat": round(float(lat[0]), 5), "lon": round(float(lon[0]), 5),
                                     "gain_marginal": round(float(gm), 6), "hauteur_m": h_cap,
                                     "source": "OS4 OI (simulation OSSE, regle_os4_oi_v4_2)"}})
    (sortie / "plan_glouton.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                                                            ensure_ascii=False), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for strat in plans:
            t = tab[tab.strategie == strat].sort_values("N")
            ax.plot(t.N, t.pire_tranche_0_1km, marker="o", label=strat)
            ax.fill_between(t.N, m_ref * (1 - t.gain_ic95_haut), m_ref * (1 - t.gain_ic95_bas), alpha=0.12)
        ax.axhline(m_ref, color="k", ls="--", label="sans reseau")
        ax.axhline(0.9 * m_ref, color="r", ls=":", label="-10 %")
        ax.set_xscale("log")
        ax.set_xlabel("capteurs ANEO (N)")
        ax.set_ylabel("RMSE pire tranche 0-1 km (m/s)")
        ax.set_title("OS4 : erreur selon le nombre de capteurs (simulation, IC 95 %)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(sortie / "courbe.png", dpi=150)
    except Exception as exc:
        print("figure non produite :", exc)
    bilan = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "regle": REGLE.name,
             "regle_sha256": h_regle,
             "lecture": "test 2022, lecture unique" if a.lecture_test else "essai du code (validation)",
             "instants": len(evalues), "jours": int(len(ju)), "candidats": int(len(cand)),
             "sans_reseau": {"pire_tranche_0_1km": m_ref, "tranche_0_100m": float(bas[0]), "rmse_vect_10m": float(r10[0])},
             "N_minimal": nmin, "nature": "OSSE sur la verite generee, experience distincte de C'"}
    (sortie / "bilan.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    titre = "test 2022 (lecture unique)" if a.lecture_test else "ESSAI du code (validation)"
    md = [f"# OS4 par OI 3D — {titre}", "",
          f"Regle `{REGLE.name}` (`{h_regle[:12]}…`). Sans reseau : pire tranche {m_ref:.3f} m/s, "
          f"0-100 m {bas[0]:.3f}, 10 m {r10[0]:.3f}.", "", f"N minimal : {json.dumps(nmin, ensure_ascii=False)}", "",
          "```", tab[["strategie", "N", "pire_tranche_0_1km", "gain", "gain_ic95_bas", "gain_ic95_haut",
                      "gain_0_100m", "gain_10m"]].round(4).to_string(index=False), "```"]
    (sortie / "bilan.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
