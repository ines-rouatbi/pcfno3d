"""Diagnostic des leviers pour battre l'ERA5 aux aeroports (regle configs/national/regle_diag_leviers_aeroports.json).

VALIDATION 2021 seulement (ajustement 2015-2020) ; l'annee 2022 n'est jamais lue. Chaque aeroport est juge
RETIRE (ni dans l'ajustement de ses poids, ni en entree). Leviers : ERA5 seule, ERA5 + OI des innovations des
11 autres aeroports, MOS generique (predictions de validation de regle_mos_era5), MOS + OI des residus.

    python scripts/diag_leviers_aeroports.py --paires artifacts/national/mos/paires_29 \\
        --mos artifacts/national/mos/choix_29/predictions_validation.csv.gz --sortie artifacts/national/diag_leviers
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]
REGLE = REPO / "configs" / "national" / "regle_diag_leviers_aeroports.json"
L_KM = (25, 50, 75, 100, 150, 200)
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)


def distances_km(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    d = (np.sin((la[:, None] - la[None]) / 2) ** 2
         + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[:, None] - lo[None]) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(d))


def lire(dossier: Path, annees) -> pd.DataFrame:
    p = pd.concat([pd.read_csv(dossier / f"paires_{a}.csv.gz", parse_dates=["temps"]) for a in annees])
    p = p[p["aeroport"].astype(bool) & (p["temps"].dt.hour % 3 == 0) & (p["temps"].dt.minute == 0)]
    return p.dropna(subset=["u_obs", "v_obs", "u10_era5", "v10_era5"])


def tableaux(p: pd.DataFrame, codes, base=("u10_era5", "v10_era5")):
    """(temps, sites) : innovations du, dv = mesure - base, et la base elle-meme (NaN si absent)."""
    p = p.assign(_du=p["u_obs"] - p[base[0]], _dv=p["v_obs"] - p[base[1]])
    piv = {c: p.pivot_table(index="temps", columns="code", values=c, aggfunc="first").reindex(columns=codes)
           for c in ("_du", "_dv", "u_obs", "v_obs", base[0], base[1])}
    return piv


def oi_loo(du, dv, biais, dist, L, lam):
    """Estimation des innovations de chaque site a partir des AUTRES (memes instants) ; sites x temps."""
    n_t, n_s = du.shape
    rho = np.exp(-dist ** 2 / (2 * L ** 2))
    est_u = np.full((n_t, n_s), np.nan)
    est_v = np.full((n_t, n_s), np.nan)
    cu, cv = du - biais[0], dv - biais[1]
    ok = np.isfinite(cu) & np.isfinite(cv)
    motifs = {}
    for t in range(n_t):
        m = ok[t]
        cle = m.tobytes()
        if cle not in motifs:
            poids = np.zeros((n_s, n_s))
            for k in range(n_s):
                autres = np.where(m & (np.arange(n_s) != k))[0]
                if len(autres) == 0:
                    continue
                A = rho[np.ix_(autres, autres)] + lam * np.eye(len(autres))
                poids[k, autres] = np.linalg.solve(A, rho[autres, k])
            motifs[cle] = poids
        W = motifs[cle]
        x_u, x_v = np.where(m, cu[t], 0.0), np.where(m, cv[t], 0.0)
        est_u[t], est_v[t] = W @ x_u, W @ x_v
    return est_u, est_v


def rmse_sites(eu, ev):
    """erreurs (temps, sites) -> RMSE vectoriel par site."""
    return np.sqrt(np.nanmean(eu ** 2 + ev ** 2, axis=0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paires", type=Path, required=True)
    ap.add_argument("--mos", type=Path, required=True)
    ap.add_argument("--sortie", type=Path, required=True)
    a = ap.parse_args(argv)
    from pcfno3d.regles import verifier, sortie_neuve
    regle, h = verifier(REGLE)
    sortie = sortie_neuve(a.sortie, "bilan.json")
    print(f"regle {REGLE.name} {h[:12]}... verifiee")

    st = pd.read_csv(a.paires / "stations.csv")
    ad = lire(a.paires, range(2015, 2022))
    assert ad["temps"].dt.year.max() <= 2021, "2022 ne doit pas etre lue"
    # les 12 aeroports METAR de l'entree du modele (Monastir : le releve DTMB, pas DTTM Skanes)
    metar = set(pd.read_csv(REPO / "configs" / "national" / "stations_metar.csv")["code"])
    icao = st.set_index("code")["icao"]
    codes = sorted(c for c in ad["code"].unique() if icao.get(c) in metar)
    assert len(codes) == 12, codes
    ad = ad[ad["code"].isin(codes)]
    pos = st.set_index("code").loc[codes]
    dist = distances_km(pos["lat"].to_numpy(), pos["lon"].to_numpy())
    noms = dict(zip(codes, pos["nom"]))

    fit = ad[ad["temps"].dt.year <= 2020]
    val = ad[ad["temps"].dt.year == 2021]
    P = tableaux(fit, codes)
    du_f, dv_f = P["_du"].to_numpy(), P["_dv"].to_numpy()
    biais = (np.nanmean(du_f, axis=0), np.nanmean(dv_f, axis=0))

    # choix de L et lambda sur 2015-2020, chaque aeroport retire (son biais jamais utilise : l'estimation
    # ne lit que les autres sites ; on compare a l'innovation brute du site juge)
    grille = []
    for L in L_KM:
        for lam in LAMBDAS:
            eu, ev = oi_loo(du_f, dv_f, biais, dist, L, lam)
            r = rmse_sites(du_f - eu, dv_f - ev).mean()
            grille.append({"L_km": L, "lambda": lam, "rmse_moyen_2015_2020": float(r)})
            print(f"  L {L:>3} km  lambda {lam:<4}  {r:.4f}", flush=True)
    g = pd.DataFrame(grille)
    best = g.loc[g["rmse_moyen_2015_2020"].idxmin()]
    L, lam = float(best["L_km"]), float(best["lambda"])
    print(f"retenu sur 2015-2020 : L {L:g} km, lambda {lam:g} (ERA5 seule : "
          f"{rmse_sites(du_f, dv_f).mean():.4f})")

    # validation 2021
    V = tableaux(val, codes)
    du_v, dv_v = V["_du"].to_numpy(), V["_dv"].to_numpy()
    jours = V["_du"].index.normalize()
    eu, ev = oi_loo(du_v, dv_v, biais, dist, L, lam)
    err = {"era5": (du_v, dv_v), "oi_aeroports": (du_v - eu, dv_v - ev)}

    mos = pd.read_csv(a.mos, parse_dates=["temps"])
    mos = mos[mos["code"].isin(codes) & (mos["temps"].dt.hour % 3 == 0)]
    assert mos["temps"].dt.year.max() <= 2021
    M = tableaux(mos, codes, base=("u_mos", "v_mos"))
    rdu = M["_du"].reindex(V["_du"].index).to_numpy()
    rdv = M["_dv"].reindex(V["_du"].index).to_numpy()
    zero = (np.zeros(12), np.zeros(12))
    mu, mv = oi_loo(rdu, rdv, zero, dist, L, lam)
    err["mos_generique"] = (rdu, rdv)
    err["mos_plus_oi"] = (rdu - mu, rdv - mv)
    # memes instants pour tous les leviers
    commun = np.ones_like(du_v, dtype=bool)
    for eu_, ev_ in err.values():
        commun &= np.isfinite(eu_) & np.isfinite(ev_)
    err = {k: (np.where(commun, x, np.nan), np.where(commun, y, np.nan)) for k, (x, y) in err.items()}

    rng = np.random.default_rng(2030)
    jours_u = np.unique(jours)
    idx_j = [np.where(jours == j)[0] for j in jours_u]
    tirages = [np.concatenate([idx_j[i] for i in rng.integers(0, len(jours_u), len(jours_u))]) for _ in range(2000)]

    def moy(e, sel=slice(None)):
        return rmse_sites(e[0][sel], e[1][sel]).mean()

    lignes, resume = [], {}
    r_era5 = rmse_sites(*err["era5"])
    for nom, e in err.items():
        r = rmse_sites(*e)
        ecarts = np.array([moy(e, s) - moy(err["era5"], s) for s in tirages])
        ic = (float(np.percentile(ecarts, 2.5)), float(np.percentile(ecarts, 97.5)))
        mieux = int(np.sum(r < r_era5))
        resume[nom] = {"rmse_vect_moyen": float(r.mean()), "ecart_era5": float(r.mean() - r_era5.mean()),
                       "ic95_ecart": ic, "aeroports_ameliores": mieux,
                       "pire_degradation_relative": float(np.max(r / r_era5 - 1)),
                       "prometteur": bool(ic[1] < 0 and mieux >= 8) if nom != "era5" else None}
        for c, v, ve in zip(codes, r, r_era5):
            lignes.append({"levier": nom, "code": c, "aeroport": noms[c], "n": int(np.sum(commun[:, codes.index(c)])),
                           "rmse_vect": float(v), "rmse_vect_era5": float(ve), "ecart_relatif": float(v / ve - 1)})
    tab = pd.DataFrame(lignes)
    tab.to_csv(sortie / "par_aeroport.csv", index=False)
    g.to_csv(sortie / "grille_oi_2015_2020.csv", index=False)
    bilan = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "regle": REGLE.name,
             "regle_sha256": h, "jeu": "validation 2021 (ajustement 2015-2020) ; 2022 non lue",
             "oi_retenu": {"L_km": L, "lambda": lam}, "instants_communs": int(commun.any(axis=1).sum()),
             "resume": resume,
             "note": "ERA5 = reanalyse, jamais des mesures ; METAR/SYNOP = mesures ; diagnostic sans effet sur un modele"}
    (sortie / "bilan.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resume, indent=1))
    piv = tab.pivot_table(index="aeroport", columns="levier", values="rmse_vect").round(3)
    print(piv.to_string())
    (sortie / "par_aeroport_tableau.txt").write_text(piv.to_string(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
