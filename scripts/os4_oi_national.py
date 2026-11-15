"""OS4 national par interpolation optimale (regle configs/national/regle_os4_oi_national.json) : simulation (OSSE).

Courbe « gain selon le nombre de capteurs ANEO » sans reentrainer de reseau : fond = ERA5 10 m degradee (statistiques
de l'ecart ERA5 - METAR, candidat D), capteurs simules = verite a 10 m + bruit, analyse = fond + OI. Parametres de
l'OI ajustes sur les jours d'ENTRAINEMENT ; courbe sur la VALIDATION ; 2022 non lu. CPU seulement.

    python scripts/os4_oi_national.py -c national_1km_colab --degradation <runs>/degradation_era5_v4.json --sortie <dossier>
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
REGLE = REPO / "configs" / "national" / "regle_os4_oi_national.json"


def point_le_plus_eloigne(cand: np.ndarray, n: int, depart: int, dx: float, dy: float) -> np.ndarray:
    """Indices (dans cand, (m, 2) iy, ix) des n premiers points de l'echantillonnage du point le plus eloigne."""
    y, x = cand[:, 0] * dy, cand[:, 1] * dx
    choisis = [int(depart)]
    dmin = np.hypot(y - y[depart], x - x[depart])
    for _ in range(min(n, len(cand)) - 1):
        k = int(np.argmax(dmin))
        choisis.append(k)
        dmin = np.minimum(dmin, np.hypot(y - y[k], x - x[k]))
    return np.asarray(choisis)


def statistiques(innov: np.ndarray, cellules: np.ndarray, dx: float, dy: float) -> dict:
    """innov (instants, n, 2) aux cellules (n, 2) -> ajuster_statistiques (Hollingsworth et Loennberg)."""
    from pcfno3d.innovation import ajuster_statistiques
    a = innov - np.nanmean(innov, axis=0, keepdims=True)
    n = cellules.shape[0]
    d = np.hypot((cellules[:, None, 0] - cellules[None, :, 0]) * dy, (cellules[:, None, 1] - cellules[None, :, 1]) * dx) / 1000
    cov = np.einsum("tic,tjc->ij", a, a) / (2 * a.shape[0])           # moyenne sur u, v et les instants
    var = {i: float(cov[i, i]) for i in range(n)}
    iu = np.triu_indices(n, 1)
    paires = [{"distance_km": float(d[i, j]), "covariance": float(cov[i, j]), "instants": int(a.shape[0])}
              for i, j in zip(*iu)]
    return ajuster_statistiques(paires, var, correlation="gaussienne", bornes_km=(10.0, 400.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--degradation", required=True)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--max-instants", type=int, default=None, help="essai rapide (pas un resultat)")
    a = ap.parse_args(argv)
    from pcfno3d.aneo_entree import EntreeANEO
    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.dataset import is_synoptic, make_splits
    from pcfno3d.forcage import charger_e
    from pcfno3d.geometry import Geometry
    from pcfno3d.os6 import Contraintes, candidats
    from pcfno3d.realdata import georef_grille
    from pcfno3d.regles import sortie_neuve, verifier
    from pcfno3d.vent10m_oi import couche_10m
    regle, h = verifier(REGLE)
    sortie = sortie_neuve(a.sortie, "bilan.json")
    print(f"regle {REGLE.name} {h[:12]}... verifiee", flush=True)

    cfg = Config.load(a.config)
    geom = Geometry(cfg, device="cpu")
    dx, dy = float(geom.dx), float(geom.dy)
    index = load_index(cfg)
    sp = make_splits(cfg, index)
    stamp = sp.stamp_of
    rng0 = np.random.default_rng(2030)
    train = [s for s in sp.train[0] if is_synoptic(stamp[s])]
    train = list(rng0.choice(train, size=min(200, len(train)), replace=False))
    val = [s for s in sp.val[0] if is_synoptic(stamp[s])]
    assert not set(val) & set(sp.test[0]) and not set(train) & set(sp.test[0])
    if a.max_instants:
        train, val = train[:a.max_instants], val[:a.max_instants]
    deg = EntreeANEO(geom, a.degradation, 0.0, (50.0, 150.0))
    nc = geom.forcage.n_cellules
    terre = (np.asarray(geom.z0) >= 0.001) & np.asarray(geom.valid).any(axis=0)
    cand = candidats(geom, Contraintes())
    print(f"instants : statistiques {len(train)} (entrainement), courbe {len(val)} (validation) | "
          f"candidats {len(cand)} | noeuds de terre {int(terre.sum())}", flush=True)

    sig_p = float(regle["capteurs"]["sigma_capteur_m_s"]["principal"])
    sigmas = [sig_p] + [float(s) for s in regle["capteurs"]["sigma_capteur_m_s"]["sensibilite"]]
    N = [int(n) for n in regle["N"]]
    nmax = max(N)

    def instant(stem, graine):
        S = np.load(cfg.paths.cache_dir / f"{stem}_S.npy")
        T = S[:2].astype(np.float64)
        E = charger_e(geom, cfg.paths.cache_dir / f"{stem}_E.npy").astype(np.float64)
        delta = deg.appliquer(np.zeros(nc, np.float32), np.random.default_rng(graine))[1].astype(np.float64)
        return T, E, E + delta

    # 1. statistiques de l'OI (jours d'entrainement), 200 capteurs aleatoires fixes
    fixes = cand[np.random.default_rng(2030).choice(len(cand), size=min(200, len(cand)), replace=False)]
    innov = {s: [] for s in sigmas}
    for n, stem in enumerate(train):
        T, _, B = instant(stem, 1000 + n)
        bruit = np.random.default_rng(5000 + n).standard_normal((len(fixes), 2))
        for s in sigmas:
            obs = T[:, fixes[:, 0], fixes[:, 1]].T + s * bruit
            innov[s].append(obs - B[:, fixes[:, 0], fixes[:, 1]].T)
    stats = {}
    for s in sigmas:
        st = statistiques(np.asarray(innov[s]), fixes, dx, dy)
        st["lambda"] = (st["sigma_o"] / st["sigma_b"]) ** 2
        stats[str(s)] = st
        print(f"sigma_capteur {s} : L {st['L_km']:.0f} km, sigma_b {st['sigma_b']:.3f}, sigma_o {st['sigma_o']:.3f}, "
              f"lambda {st['lambda']:.2f}{' (L en butee)' if st['L_en_butee'] else ''}", flush=True)

    # 2. placements
    yc, xc = np.argwhere(terre).mean(axis=0)
    depart = int(np.argmin(np.hypot(cand[:, 0] - yc, cand[:, 1] - xc)))
    places = {"regulier": cand[point_le_plus_eloigne(cand, nmax, depart, dx, dy)]}
    for g in (1, 2, 3):
        places[f"aleatoire_{g}"] = cand[np.random.default_rng(g).permutation(len(cand))[:nmax]]

    # 3. courbe (validation)
    sse = {}
    def ajouter(cle, A, T):
        e = ((A - T) ** 2).sum(axis=0)[terre]
        s, c = sse.get(cle, (0.0, 0))
        sse[cle] = (s + float(np.nansum(e)), c + int(np.isfinite(e).sum()))
    t0 = time.time()
    for n, stem in enumerate(val):
        T, E, B = instant(stem, 1000 + 100000 + n)
        ajouter(("era5_propre",), E, T)
        ajouter(("fond",), B, T)
        bruit = np.random.default_rng(9000 + n).standard_normal((nmax, 2))
        for s in sigmas:
            st = stats[str(s)]
            for nom, pl in places.items():
                if s != sig_p and nom != "regulier":
                    continue
                for k in N:
                    c = pl[:k]
                    cel = {str(i): (int(c[i, 0]), int(c[i, 1])) for i in range(k)}
                    mes = {str(i): tuple(T[:, c[i, 0], c[i, 1]] + s * bruit[i]) for i in range(k)}
                    A, _ = couche_10m(B, cel, mes, dx, dy, biais={}, L_km=st["L_km"], lam=st["lambda"])
                    ajouter((nom, s, k), A, T)
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(val)} instants ({time.time() - t0:.0f} s)", flush=True)

    rmse = {k: float(np.sqrt(v[0] / max(v[1], 1))) for k, v in sse.items()}
    r_fond = rmse[("fond",)]
    lignes = []
    for k, v in rmse.items():
        if len(k) == 3:
            lignes.append({"placement": k[0], "sigma_capteur": k[1], "N": k[2], "rmse_vect": v,
                           "rmse_fond": r_fond, "gain": 1 - v / r_fond})
    tab = pd.DataFrame(lignes).sort_values(["sigma_capteur", "placement", "N"])
    tab.to_csv(sortie / "courbe.csv", index=False)
    reg = tab[(tab.placement == "regulier") & (tab.sigma_capteur == sig_p)].sort_values("N")
    atteint = reg[reg.gain >= 0.10]
    n_min = int(atteint.N.iloc[0]) if len(atteint) else None
    if n_min:
        geo = georef_grille(cfg, geom)
        c = places["regulier"][:n_min]
        lat, lon = geo.to_latlon(c[:, 1] * dx, c[:, 0] * dy) if geo is not None else (np.nan, np.nan)
        pd.DataFrame({"iy": c[:, 0], "ix": c[:, 1], "lat": np.round(lat, 5), "lon": np.round(lon, 5)}).to_csv(
            sortie / "plan_regulier.csv", index=False)
    (sortie / "statistiques_oi.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    bilan = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "regle": REGLE.name,
             "regle_sha256": h, "nature": "simulation (OSSE) face a la verite generee",
             "instants_statistiques": len(train), "instants_courbe": len(val),
             "rmse_era5_propre": rmse[("era5_propre",)], "rmse_fond_degrade": r_fond,
             "N_minimal_gain_10pct": n_min, "gain_regulier": dict(zip(reg.N.astype(int).astype(str), reg.gain.round(4))),
             **({"essai": "essai rapide, pas un resultat"} if a.max_instants else {}),
             "note": "verite et ERA5 = jamais des mesures ; gain reel de l'OI des aeroports : -3,7 % (2023-2024)"}
    (sortie / "bilan.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# OS4 national par OI (simulation, validation)", "",
          f"Regle `{REGLE.name}` (`{h[:12]}…`). Fond = ERA5 degradee : RMSE {r_fond:.3f} m/s (ERA5 propre "
          f"{rmse[('era5_propre',)]:.3f}). N minimal pour 10 % de gain (placement regulier, capteur {sig_p} m/s) : "
          f"**{n_min if n_min else 'non atteint avec 200 capteurs'}**.", "", "```",
          tab.round(4).to_string(index=False), "```"]
    (sortie / "bilan.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
