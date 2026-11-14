"""Post-traitement « modele + OI des innovations des aeroports » a 10 m (regle configs/national/regle_oi_post_v4.json).

Lit <run>/validation_reelle/<jeu>/stations_paires.csv (valider_grand_tunis_reel.py : une ligne par aeroport juge et
instant, aeroport RETIRE de l'entree) ; pour chaque aeroport juge k, estime son innovation a partir des 11 AUTRES
(innovation de j prise dans la passe ou j est retire), L et lambda fixes par la regle. Sources : modele, era5,
modele_oi, era5_oi. Refuse le jeu « test » (2022 deja lue). Sortie : <run>/validation_reelle/<jeu>/oi_post/ (neuf).

    python scripts/oi_post_aeroports.py --run <runs>/national_1km_v4_Cp_s44 --jeu validation
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
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
REGLE = REPO / "configs" / "national" / "regle_oi_post_v4.json"
BIAIS = REPO / "configs" / "national" / "biais_innovations_aeroports_2015_2020.json"
L_KM, LAMBDA = 150.0, 4.0


def oi_autres(d_u, d_v, dist_km, L=L_KM, lam=LAMBDA):
    """d_u, d_v (temps, sites), NaN = absent -> estimation au site k a partir des autres sites seulement."""
    n_t, n_s = d_u.shape
    rho = np.exp(-dist_km ** 2 / (2 * L ** 2))
    eu, ev = np.full((n_t, n_s), np.nan), np.full((n_t, n_s), np.nan)
    ok = np.isfinite(d_u) & np.isfinite(d_v)
    cache = {}
    for t in range(n_t):
        m = ok[t]
        cle = m.tobytes()
        if cle not in cache:
            W = np.zeros((n_s, n_s))
            for k in range(n_s):
                a = np.where(m & (np.arange(n_s) != k))[0]
                if len(a):
                    W[k, a] = np.linalg.solve(rho[np.ix_(a, a)] + lam * np.eye(len(a)), rho[a, k])
            cache[cle] = W
        W = cache[cle]
        eu[t] = W @ np.where(m, d_u[t], 0.0)
        ev[t] = W @ np.where(m, d_v[t], 0.0)
    return eu, ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--jeu", default="val")
    a = ap.parse_args(argv)
    if a.jeu == "test":
        raise SystemExit("regle_oi_post_v4 : 2022 deja lue pour C' s44, le jeu test est refuse ici")
    from pcfno3d.regles import verifier, sortie_neuve
    regle, h = verifier(REGLE)
    from diag_leviers_aeroports import distances_km
    src = a.run / "validation_reelle" / a.jeu / "stations_paires.csv"
    sortie = sortie_neuve(src.parent / "oi_post", "bilan.json")
    print(f"regle {REGLE.name} {h[:12]}... verifiee | {src}")

    p = pd.read_csv(src, parse_dates=["temps"])
    meta = pd.read_csv(REPO / "configs" / "national" / "stations_metar.csv")
    code_de = dict(zip(meta["nom"], meta["code"]))
    p["code"] = p["station"].map(code_de)
    if p["code"].isna().any():
        raise SystemExit(f"stations inconnues : {sorted(p.loc[p['code'].isna(), 'station'].unique())}")
    if p["temps"].dt.year.max() >= 2022:
        raise SystemExit("une date de 2022 dans le jeu : refuse")
    codes = sorted(p["code"].unique())
    pos = meta.set_index("code").loc[codes]
    dist = distances_km(pos["lat"].to_numpy(), pos["lon"].to_numpy())
    biais = json.loads(BIAIS.read_text(encoding="utf-8"))["biais"]
    bu = np.array([biais[c]["du_moyen"] for c in codes])
    bv = np.array([biais[c]["dv_moyen"] for c in codes])

    piv = {c: p.pivot_table(index="temps", columns="code", values=c, aggfunc="first").reindex(columns=codes)
           for c in ("u_obs", "v_obs", "u_modele", "v_modele", "u_era5", "v_era5")}
    obs_u, obs_v = piv["u_obs"].to_numpy(), piv["v_obs"].to_numpy()
    err = {}
    for base in ("modele", "era5"):
        bu_, bv_ = piv[f"u_{base}"].to_numpy(), piv[f"v_{base}"].to_numpy()
        du, dv = obs_u - bu_, obs_v - bv_
        eu, ev = oi_autres(du - bu, dv - bv, dist)
        err[base] = (du, dv)
        err[f"{base}_oi"] = (du - eu, dv - ev)
    commun = np.ones(obs_u.shape, bool)
    for x, y in err.values():
        commun &= np.isfinite(x) & np.isfinite(y)
    err = {k: (np.where(commun, x, np.nan), np.where(commun, y, np.nan)) for k, (x, y) in err.items()}

    def rmse(e, sel=slice(None)):
        return np.sqrt(np.nanmean(e[0][sel] ** 2 + e[1][sel] ** 2, axis=0))

    jours = piv["u_obs"].index.normalize()
    ju = np.unique(jours)
    idx = [np.where(jours == j)[0] for j in ju]
    rng = np.random.default_rng(2030)
    tir = [np.concatenate([idx[i] for i in rng.integers(0, len(ju), len(ju))]) for _ in range(2000)]
    r_era5 = rmse(err["era5"])
    resume, lignes = {}, []
    for k, e in err.items():
        r = rmse(e)
        ec = np.array([rmse(e, s).mean() - rmse(err["era5"], s).mean() for s in tir])
        ic = (float(np.percentile(ec, 2.5)), float(np.percentile(ec, 97.5)))
        resume[k] = {"rmse_vect_moyen": float(r.mean()), "ecart_era5": float(r.mean() - r_era5.mean()),
                     "ic95_ecart_era5": ic, "aeroports_ameliores": int(np.sum(r < r_era5)),
                     "pire_degradation_relative": float(np.max(r / r_era5 - 1))}
        for c, v, ve in zip(codes, r, r_era5):
            lignes.append({"source": k, "code": c, "station": pos.loc[c, "nom"], "n": int(commun[:, codes.index(c)].sum()),
                           "rmse_vect": float(v), "rmse_vect_era5": float(ve), "ecart_relatif": float(v / ve - 1)})
    m = resume["modele_oi"]
    prometteur = bool(m["ic95_ecart_era5"][1] < 0 and m["aeroports_ameliores"] >= 8
                      and m["pire_degradation_relative"] <= 0.05)
    tab = pd.DataFrame(lignes)
    tab.to_csv(sortie / "par_aeroport.csv", index=False)
    bilan = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "regle": REGLE.name,
             "regle_sha256": h, "jeu": a.jeu, "run": a.run.name, "L_km": L_KM, "lambda": LAMBDA,
             "instants": int(commun.any(axis=1).sum()), "resume": resume, "prometteur": prometteur,
             "note": "aeroport juge retire de l'entree et des innovations ; METAR = mesures ; ERA5 et verite = jamais des mesures"}
    (sortie / "bilan.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    t = tab.pivot_table(index="station", columns="source", values="rmse_vect").round(3)
    t.loc["MOYENNE"] = t.mean().round(3)
    (sortie / "tableau.txt").write_text(t.to_string(), encoding="utf-8")
    print(t.to_string())
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                      for k, v in resume.items()}, indent=1))
    print("PROMETTEUR (regle_oi_post_v4) :", prometteur, "| resultats ->", sortie)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
