"""Complements de la solution de repli (demande d'Ines du 10/11, points 1 et 2), sans regeneration.

Regle : configs/national/regle_repli_complements.json (empreinte verifiee avant calcul).
1. Pour beta 0,25 / 0,5 / 0,75 : ecart RMSE verite - ERA5 aux 12 aeroports, IC 95 % par bootstrap sur les
   instants, verdict ; validation croisee par station (beta choisi sans Tabarka, Gafsa, Sfax, Djerba sur la
   calibration, juge sur ces 4 a l'acceptation).
2. Detail garde : ecart verite - ERA5 sur u, v a 300 / 800 m et rugosite verite / ERA5, v2 contre v3.
Un passage du generateur a beta = 1 par instant ; les beta s'en deduisent par linearite (controle contre les
RMSE publies de la phase 2).

    python scripts/repli_complements.py [--sortie docs/resultats/national/repli_complements]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
NAT = REPO / "artifacts" / "national"
REGLE = REPO / "configs" / "national" / "regle_repli_complements.json"
BETAS = (0.25, 0.5, 0.75)
RETENUES = ("DTKA", "DTTF", "DTTX", "DTTJ")
HAUTEURS = (300.0, 800.0)
CONFIGS = {"v3": {"passe_haut": True, "perturb": 0.0, "brise": 0.5, "relief": 1.0},
           "v2": {"passe_haut": False, "perturb": 0.4, "brise": 1.0, "relief": 1.0}}


def verifier_regle() -> str:
    h = hashlib.sha256(REGLE.read_bytes()).hexdigest()
    attendu = (REGLE.parent / (REGLE.name + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"regle modifiee depuis son hachage : {h} != {attendu}")
    return h


def rmse(df: pd.DataFrame, pu: str, pv: str) -> float:
    return float(np.sqrt(((df[pu] - df.uo) ** 2 + (df[pv] - df.vo) ** 2).mean()))


def ecart_ic(df: pd.DataFrame, pu: str, pv: str, n: int = 5000, graine: int = 2029) -> dict:
    """ecart RMSE(pu, pv) - RMSE(ERA5) et IC 95 % par bootstrap sur les instants."""
    d = df.assign(se=(df[pu] - df.uo) ** 2 + (df[pv] - df.vo) ** 2, ee=(df.ue - df.uo) ** 2 + (df.ve - df.vo) ** 2)
    g = d.groupby("instant")[["se", "ee"]].agg(["sum", "count"])
    se, ee, nn = g[("se", "sum")].to_numpy(), g[("ee", "sum")].to_numpy(), g[("se", "count")].to_numpy()
    point = np.sqrt(se.sum() / nn.sum()) - np.sqrt(ee.sum() / nn.sum())
    rng = np.random.default_rng(graine)
    i = rng.integers(0, len(se), size=(n, len(se)))
    b = np.sqrt(se[i].sum(1) / nn[i].sum(1)) - np.sqrt(ee[i].sum(1) / nn[i].sum(1))
    lo, hi = np.percentile(b, [2.5, 97.5])
    verdict = "meilleure qu'ERA5" if hi < 0 else ("moins bonne qu'ERA5" if lo > 0 else "equivalente a ERA5")
    return {"paires": int(nn.sum()), "instants": int(len(se)), "rmse_verite": float(np.sqrt(se.sum() / nn.sum())),
            "rmse_era5": float(np.sqrt(ee.sum() / nn.sum())), "ecart": float(point), "ic95_bas": float(lo),
            "ic95_haut": float(hi), "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sortie", default=str(REPO / "docs" / "resultats" / "national" / "repli_complements"))
    ap.add_argument("--instants", type=int, default=20, help="essai seulement : moins d'instants")
    a = ap.parse_args()
    h_regle = verifier_regle()
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)

    import calibrer_generateur_stations as C
    import calibrer_verite_v3 as V
    info = json.loads((NAT / "grille" / "grille_reelle.json").read_text(encoding="utf-8"))
    C._env(info, 1.0)
    import generate_aneo_3d as G
    from pcfno3d.stations_reelles import lire_observations, obs_a

    df = G.load_or_build_grid(NAT / "grille" / "aneo_grid_points_surface.csv", np.random.default_rng(0))
    S = G.build_static(df)
    P0 = G.precompute_static_derived(S)
    F = G.lire_forcage(str(NAT / "forcage_era5.csv"))
    CH = G.lire_champs(str(NAT / "forcage_era5.csv"))
    st_codes = {s["code"]: (int(s["iy"]), int(s["ix"])) for s in info["stations"].values() if s.get("dans_la_zone")}
    obs = lire_observations(NAT / "obs", verbose=False, codes=list(st_codes))
    k = G.STRIDE
    lat_v, lon_v = S["lat"][::k, ::k][:G.NY_V, :G.NX_V], S["lon"][::k, ::k][:G.NY_V, :G.NX_V]
    zt_v = S["zt"][::k, ::k][:G.NY_V, :G.NX_V]
    Z = np.asarray(G.Z_LEVELS, float)
    h_sol = np.maximum(Z[:, None, None] - zt_v[None], G.MIN_ZREL_NODE)
    a_rot1, a_div1 = G.A_ROT, G.A_DIV

    def a_hauteur(vol3, hh):
        z = zt_v + hh
        i = np.clip(np.searchsorted(Z, z) - 1, 0, len(Z) - 2)
        t = np.clip((z - Z[i]) / (Z[i + 1] - Z[i]), 0, 1)
        lo = np.take_along_axis(vol3, i[None], 0)[0]
        hi = np.take_along_axis(vol3, (i + 1)[None], 0)[0]
        ok = (z >= Z[0]) & (z <= Z[-1]) & (Z[i] - zt_v >= G.MIN_ZREL_NODE)
        return np.where(ok, (1 - t) * lo + t * hi, np.nan)

    def rug(f):
        m = np.isfinite(f)
        mx, my = m[:, 1:] & m[:, :-1], m[1:, :] & m[:-1, :]
        return np.abs(np.diff(f, axis=1))[mx].mean() + np.abs(np.diff(f, axis=0))[my].mean()

    i_cal = V.instants(20, 2026)
    i_acc = V.instants(20, 2027, exclure={j for j, _ in i_cal})
    i_cal, i_acc = i_cal[:a.instants], i_acc[:a.instants]
    paires, detail = [], []
    for nom_c, cfg in CONFIGS.items():
        G.A_ROT, G.A_DIV = a_rot1 * cfg["perturb"], a_div1 * cfg["perturb"]
        G.AMORT_PASSE_HAUT = cfg["passe_haut"]
        G.BRISE_FACTEUR, G.RELIEF_FACTEUR, G.DETAIL_BETA = cfg["brise"], cfg["relief"], 1.0
        for jeu, insts in (("calibration", i_cal), ("acceptation", i_acc)):
            for jour, hh in insts:
                seed = 2026 * 1000 + int((jour - pd.Timestamp("2021-01-01")).days) + 1
                st = G.DayStateERA5(0, jour, 0, S, seed, F, CH)
                surf, vol, _ = G.compute_fields(st, S, P0, hh * 3600.0, None)
                champ = st.champ_cible(hh * 3600.0)
                t = jour + pd.Timedelta(hours=hh)
                for code, (uo, vo) in obs_a(obs, t).items():
                    iy, ix = st_codes[code]
                    jy, jx = (iy // k) * k, (ix // k) * k
                    blk = (slice(jy, jy + k), slice(jx, jx + k))
                    ug, vg = (float(surf[x][blk].mean()) for x in ("u", "v"))
                    # ERA5 du melange beta, cellule par cellule du bloc (hauteur du capteur)
                    ec = [float(G._era5_au_sol(CH, champ[c], S["lat"][blk], S["lon"][blk], S["zrel"][blk]).mean())
                          for c in (0, 1)]
                    la, lo_ = np.array([[S["lat"][jy, jx]]]), np.array([[S["lon"][jy, jx]]])
                    ue, ve = (float(G._era5_au_sol(CH, champ[c], la, lo_, np.full((1, 1), 10.0))[0, 0]) for c in (0, 1))
                    paires.append({"config": nom_c, "jeu": jeu, "instant": str(t), "code": code, "uo": uo, "vo": vo,
                                   "ug1": ug, "vg1": vg, "ecu": ec[0], "ecv": ec[1], "ue": ue, "ve": ve})
                if jeu == "acceptation":
                    E = [np.stack([G._era5_au_sol(CH, champ[c], lat_v, lon_v, h_sol[kz]) for kz in range(len(Z))])
                         for c in (0, 1)]
                    for c, cn in ((0, "u"), (1, "v")):
                        V1 = vol["uv"[c]]
                        for z in HAUTEURS:
                            A1, AE = a_hauteur(V1, z), a_hauteur(E[c], z)
                            Eh = G._era5_au_sol(CH, champ[c], lat_v, lon_v, np.full(lat_v.shape, z))
                            Eh = np.where(np.isfinite(A1), Eh, np.nan)
                            for beta in ((1.0,) if nom_c == "v2" else BETAS + (1.0,)):
                                Tb = AE + beta * (A1 - AE)
                                d = Tb - Eh
                                m = np.isfinite(d)
                                detail.append({"config": nom_c, "beta": beta, "instant": str(t), "composante": cn,
                                               "hauteur_sol_m": int(z), "sse": float((d[m] ** 2).sum()), "n": int(m.sum()),
                                               "rug_verite": float(rug(Tb)), "rug_era5": float(rug(Eh)),
                                               "rug_detail": float(rug(Tb - Eh))})
                print(f"{nom_c} {jeu} {t:%Y-%m-%d %H} h", flush=True)
    G.A_ROT, G.A_DIV, G.AMORT_PASSE_HAUT, G.BRISE_FACTEUR, G.RELIEF_FACTEUR = a_rot1, a_div1, False, 1.0, 1.0

    P = pd.DataFrame(paires)
    for b in BETAS:
        P[f"ug_{b:g}"] = P.ecu + b * (P.ug1 - P.ecu)
        P[f"vg_{b:g}"] = P.ecv + b * (P.vg1 - P.ecv)
    P.to_csv(out / "paires_aeroports.csv", index=False)
    D = pd.DataFrame(detail)
    D.to_csv(out / "detail_par_instant.csv", index=False)

    # controle de linearite contre les RMSE publies (phase 2)
    pub = pd.concat([pd.read_csv(REPO / "docs" / "resultats" / "national" / f) for f in
                     ("verite_v3_repli_calibration.csv", "verite_v3_repli_acceptation.csv")])
    controle = []
    p3 = P[P.config == "v3"]
    for _, r in pub.iterrows():
        q = p3[p3.jeu == r.jeu]
        val = rmse(q, f"ug_{r.beta:g}", f"vg_{r.beta:g}")
        controle.append({"jeu": r.jeu, "beta": r.beta, "publie": r.rmse_aeroports_verite, "recalcule": val,
                         "ecart": abs(val - r.rmse_aeroports_verite)})
    controle = pd.DataFrame(controle)
    lineaire_ok = bool((controle.ecart <= 1e-6).all())

    # point 1 : ecart et IC par beta (acceptation, calibration, ensemble), v2 pour comparaison
    lignes = []
    for jeu in ("acceptation", "calibration", "ensemble"):
        q3 = p3 if jeu == "ensemble" else p3[p3.jeu == jeu]
        for b in BETAS:
            lignes.append({"verite": f"v3 beta {b:g}", "jeu": jeu, **ecart_ic(q3, f"ug_{b:g}", f"vg_{b:g}")})
        q2 = P[P.config == "v2"] if jeu == "ensemble" else P[(P.config == "v2") & (P.jeu == jeu)]
        lignes.append({"verite": "v2", "jeu": jeu, **ecart_ic(q2, "ug1", "vg1")})
    T1 = pd.DataFrame(lignes)
    T1.to_csv(out / "ecart_aeroports_ic.csv", index=False)

    # validation croisee par station
    cal8 = p3[(p3.jeu == "calibration") & ~p3.code.isin(RETENUES)]
    r8 = pd.DataFrame([{"beta": b, "rmse": rmse(cal8, f"ug_{b:g}", f"vg_{b:g}")} for b in BETAS])
    best = r8.rmse.min()
    beta_choisi = float(r8[r8.rmse <= best + 0.005].beta.max())
    acc4 = p3[(p3.jeu == "acceptation") & p3.code.isin(RETENUES)]
    vc = [{"beta": b, "choisi": b == beta_choisi, **ecart_ic(acc4, f"ug_{b:g}", f"vg_{b:g}")} for b in BETAS]
    q2 = P[(P.config == "v2") & (P.jeu == "acceptation") & P.code.isin(RETENUES)]
    vc.append({"beta": "v2", "choisi": False, **ecart_ic(q2, "ug1", "vg1")})
    VC = pd.DataFrame(vc)
    VC.to_csv(out / "validation_croisee_4_stations.csv", index=False)
    ch = VC[VC.choisi].iloc[0]

    # point 2 : detail garde
    D2 = D.groupby(["config", "beta", "composante", "hauteur_sol_m"]).apply(
        lambda g: pd.Series({"ecart_rms_verite_era5": np.sqrt(g.sse.sum() / g.n.sum()),
                             "rugosite_verite_sur_era5": (g.rug_verite / g.rug_era5).mean(),
                             # information, hors regle : gradients moyens en m/s par km (2 directions)
                             "info_grad_verite": g.rug_verite.mean(), "info_grad_era5": g.rug_era5.mean(),
                             "info_grad_detail": g.rug_detail.mean()}), include_groups=False).reset_index()
    D2.to_csv(out / "detail_garde.csv", index=False)

    verdict = {"regle": "configs/national/regle_repli_complements.json", "sha256_regle": h_regle,
               "controle_linearite_ok": lineaire_ok, "controle": controle.to_dict("records"),
               "par_beta_acceptation": T1[T1.jeu == "acceptation"].to_dict("records"),
               "validation_croisee": {"calibration_8_stations": r8.to_dict("records"), "beta_choisi": beta_choisi,
                                      "jugement_4_stations_acceptation": ch.to_dict(),
                                      "generalise": bool(ch.ecart <= 0)}}
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(controle.to_string(index=False))
    print(T1.round(4).to_string(index=False))
    print(r8.round(4).to_string(index=False), "\nbeta choisi sans les 4 :", beta_choisi)
    print(VC.round(4).to_string(index=False))
    print(D2.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
