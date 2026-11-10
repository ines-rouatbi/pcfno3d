"""Phase 2 nationale : recalibration du generateur de la verite d'entrainement v3 sur la VALIDATION 2021,
selon configs/national/regle_verite_v3.json (fixee et hachee avant le calcul ; l'empreinte est verifiee).

1. calibration : 20 instants (graine 2026), chaque variante de la regle (facteurs perturbations / brise /
   relief, amortissement passe-haut) ; mesures : RMSE vectoriel aux 12 aeroports (vent 10 m, bloc 2 x 2,
   METAR a 30 min au plus) contre ERA5 seule au meme point, et ecart rms verite - ERA5 a 300 / 800 m
   au-dessus du sol ;
2. choix de la variante selon la regle ;
3. acceptation : 20 autres instants de validation (graine 2027, jours disjoints), variante choisie contre
   ERA5 seule.
Aucune generation complete, aucun instant du test 2022. Les lignes sont ecrites au fil de l'eau
(reprise possible : une variante deja dans le CSV n'est pas recalculee).

    python scripts/calibrer_verite_v3.py [--sortie artifacts/national/verite_v3] [--instants 20]
    python scripts/calibrer_verite_v3.py --reference   (diagnostic 7 : generateur de la v2 sur les memes
        instants de calibration et d'acceptation, regle_phase1_complements.json ; ne rend aucun verdict)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
NAT = REPO / "artifacts" / "national"
REGLE = REPO / "configs" / "national" / "regle_verite_v3.json"
REGLE_REPLI = REPO / "configs" / "national" / "regle_verite_v3_repli.json"
REGLE_COMPLEMENTS = REPO / "configs" / "national" / "regle_phase1_complements.json"
VAL = (("2021-01-01", "2021-02-28"), ("2021-07-01", "2021-07-31"))
HEURES = (3, 9, 15, 21)


def charger_regle(f: Path = REGLE) -> tuple[dict, str]:
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    attendu = (f.parent / (f.name + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"regle modifiee depuis son hachage : {h} != {attendu}")
    return json.loads(f.read_text(encoding="utf-8")), h


def jours_validation() -> pd.DatetimeIndex:
    j = pd.DatetimeIndex([])
    for a, b in VAL:
        j = j.append(pd.date_range(a, b, freq="D"))
    return j


def instants(n: int, graine: int, exclure: set | None = None) -> list[tuple[pd.Timestamp, int]]:
    """n instants sur n // 2 jours de validation ; meme tirage que test_facteurs_perturbation pour la
    graine 2026 sans exclusion."""
    jours = jours_validation()
    idx = np.arange(len(jours))
    if exclure:
        idx = np.array([i for i in idx if jours[i] not in exclure])
    rng = np.random.default_rng(graine)
    choix = sorted(idx[rng.choice(len(idx), size=n // 2, replace=False)])
    return [(jours[i], h) for i in choix for h in (HEURES[i % 2], HEURES[2 + i % 2])]


def variantes(regle: dict) -> list[dict]:
    v = regle["variantes"]
    return [{"passe_haut": hp, "perturb": p, "brise": b, "relief": r}
            for hp, p, b, r in itertools.product(v["amortissement_passe_haut"], v["facteur_perturbations"],
                                                 v["facteur_brise"], v["facteur_relief"])]


def nom(v: dict) -> str:
    n = f"ph{int(v['passe_haut'])}_p{v['perturb']:g}_b{v['brise']:g}_r{v['relief']:g}"
    return n + (f"_beta{v['beta']:g}" if "beta" in v else "")


def choisir_beta(t: pd.DataFrame) -> pd.Series:
    """Critere de regle_verite_v3_repli.json : plus petit RMSE aux aeroports ; a 0.005 pres, le plus grand beta."""
    meilleur = t.rmse_aeroports_verite.min()
    return t[t.rmse_aeroports_verite <= meilleur + 0.005].sort_values("beta", ascending=False).iloc[0]


def choisir(t: pd.DataFrame, regle: dict) -> pd.Series:
    """Applique le critere de choix de la regle (texte recopie dans choix_sur_la_calibration)."""
    base = t[(t.perturb == 0.4) & (t.brise == 1.0) & (t.relief == 1.0) & t.passe_haut]
    if base.empty:
        raise SystemExit("variante de base (0.4 / 1 / 1, passe-haut) absente de la calibration")
    b = base.iloc[0]
    ok = t[(t.ecart_v300 <= 1.10 * b.ecart_v300) & (t.ecart_v800 <= 1.10 * b.ecart_v800)].copy()
    meilleur = ok.rmse_aeroports_verite.min()
    ok = ok[ok.rmse_aeroports_verite <= meilleur + 0.005].copy()
    ok["dist_v2"] = (ok.brise != 1.0).astype(int) + (ok.relief != 1.0).astype(int) + (ok.perturb != 0.4).astype(int)
    return ok.sort_values(["dist_v2", "rmse_aeroports_verite"]).iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sortie", default=str(NAT / "verite_v3"))
    ap.add_argument("--instants", type=int, default=20)
    ap.add_argument("--repli", action="store_true", help="option de repli (regle_verite_v3_repli.json)")
    ap.add_argument("--reference", action="store_true",
                    help="generateur de la v2 sur les instants de calibration et d'acceptation (diagnostic 7)")
    a = ap.parse_args()
    regle, h_regle = charger_regle()
    if a.reference:
        charger_regle(REGLE_COMPLEMENTS)
    if a.repli:
        v3 = json.loads((Path(a.sortie) / "verdict.json").read_text(encoding="utf-8"))
        if v3.get("acceptee"):
            raise SystemExit("la verite v3 est acceptee : le repli ne se declenche pas")
        regle, h_regle = charger_regle(REGLE_REPLI)
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)

    import calibrer_generateur_stations as C
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
    a_rot1, a_div1 = G.A_ROT, G.A_DIV

    def a_hauteur(vol3, hh):
        z = zt_v + hh
        i = np.clip(np.searchsorted(Z, z) - 1, 0, len(Z) - 2)
        t = np.clip((z - Z[i]) / (Z[i + 1] - Z[i]), 0, 1)
        lo = np.take_along_axis(vol3, i[None], 0)[0]
        hi = np.take_along_axis(vol3, (i + 1)[None], 0)[0]
        ok = (z >= Z[0]) & (z <= Z[-1]) & (Z[i] - zt_v >= G.MIN_ZREL_NODE)
        return np.where(ok, (1 - t) * lo + t * hi, np.nan)

    def mesurer(v: dict, insts, etiquette: str) -> dict:
        G.A_ROT, G.A_DIV = a_rot1 * v["perturb"], a_div1 * v["perturb"]
        G.AMORT_PASSE_HAUT = bool(v["passe_haut"])
        G.BRISE_FACTEUR, G.RELIEF_FACTEUR = v["brise"], v["relief"]
        G.DETAIL_BETA = float(v.get("beta", 1.0))
        err, dev = [], {key: np.zeros(2) for key in ("v300", "v800", "u300", "u800")}
        for jour, hh in insts:
            seed = 2026 * 1000 + int((jour - pd.Timestamp("2021-01-01")).days) + 1
            st = G.DayStateERA5(0, jour, 0, S, seed, F, CH)
            surf, vol, _ = G.compute_fields(st, S, P0, hh * 3600.0, None)
            champ = st.champ_cible(hh * 3600.0)
            for c, cn in ((0, "u"), (1, "v")):
                for z in (300.0, 800.0):
                    d = a_hauteur(vol["uv"[c]], z) - G._era5_au_sol(st.champs, champ[c], lat_v, lon_v,
                                                                     np.full(lat_v.shape, z))
                    m = np.isfinite(d)
                    dev[f"{cn}{int(z)}"] += [(d[m] ** 2).sum(), m.sum()]
            t = jour + pd.Timedelta(hours=hh)
            for code, (uo, vo) in obs_a(obs, t).items():
                iy, ix = st_codes[code]
                jy, jx = (iy // k) * k, (ix // k) * k
                ug, vg = (float(surf[x][jy:jy + k, jx:jx + k].mean()) for x in ("u", "v"))
                la, lo = np.array([[S["lat"][jy, jx]]]), np.array([[S["lon"][jy, jx]]])
                ue, ve = (float(G._era5_au_sol(st.champs, champ[c], la, lo, np.full((1, 1), 10.0))[0, 0])
                          for c in (0, 1))
                err.append((code, uo, vo, ug, vg, ue, ve))
            print(f"{etiquette} {nom(v)} {t:%Y-%m-%d %H} h", flush=True)
        e = pd.DataFrame(err, columns=["code", "uo", "vo", "ug", "vg", "ue", "ve"])
        par = e.assign(eg=(e.ug - e.uo) ** 2 + (e.vg - e.vo) ** 2, ee=(e.ue - e.uo) ** 2 + (e.ve - e.vo) ** 2) \
               .groupby("code")[["eg", "ee"]].mean() ** 0.5
        return {"jeu": etiquette, "variante": nom(v), **v, "paires_aeroports": len(e),
                "rmse_aeroports_verite": float(np.sqrt(((e.ug - e.uo) ** 2 + (e.vg - e.vo) ** 2).mean())),
                "rmse_aeroports_era5": float(np.sqrt(((e.ue - e.uo) ** 2 + (e.ve - e.vo) ** 2).mean())),
                "stations_verite_meilleure": int((par.eg <= par.ee).sum()), "stations": int(len(par)),
                **{f"ecart_{key}": float(np.sqrt(s[0] / s[1])) for key, s in dev.items()}}

    if a.reference:
        v_ref = {"passe_haut": False, "perturb": 0.4, "brise": 1.0, "relief": 1.0, "beta": 1.0}
        i_cal = instants(a.instants, 2026)
        i_acc = instants(a.instants, 2027, exclure={j for j, _ in i_cal})
        f_ref = out / "reference_v2.csv"
        lignes = [mesurer(v_ref, i_cal, "calibration"), mesurer(v_ref, i_acc, "acceptation")]
        pd.DataFrame(lignes).to_csv(f_ref, index=False)
        print(pd.DataFrame(lignes).round(3).to_string(index=False))
        return 0

    suffixe = "_repli" if a.repli else ""
    f_cal = out / f"calibration{suffixe}.csv"
    deja = pd.read_csv(f_cal) if f_cal.exists() else pd.DataFrame()
    i_cal = instants(a.instants, 2026)
    liste = ([{**v3["reglages"], "beta": float(b)} for b in regle["valeurs_beta"]] if a.repli
             else variantes(regle))
    for v in liste:
        if len(deja) and nom(v) in set(deja.variante):
            continue
        ligne = mesurer(v, i_cal, "calibration")
        deja = pd.concat([deja, pd.DataFrame([ligne])], ignore_index=True)
        deja.to_csv(f_cal, index=False)
    t = deja.copy()
    t["passe_haut"] = t["passe_haut"].astype(bool)
    choix = choisir_beta(t) if a.repli else choisir(t, regle)
    v_choix = {"passe_haut": bool(choix.passe_haut), "perturb": float(choix.perturb),
               "brise": float(choix.brise), "relief": float(choix.relief)}
    if a.repli:
        v_choix["beta"] = float(choix.beta)

    f_acc = out / f"acceptation{suffixe}.csv"
    if f_acc.exists() and nom(v_choix) in set(pd.read_csv(f_acc).variante):
        acc = pd.read_csv(f_acc).set_index("variante").loc[nom(v_choix)].to_dict()
    else:
        i_acc = instants(a.instants, 2027, exclure={j for j, _ in i_cal})
        acc = mesurer(v_choix, i_acc, "acceptation")
        pd.DataFrame([acc]).to_csv(f_acc, index=False)
    G.A_ROT, G.A_DIV, G.AMORT_PASSE_HAUT, G.BRISE_FACTEUR, G.RELIEF_FACTEUR = a_rot1, a_div1, False, 1.0, 1.0
    G.DETAIL_BETA = 1.0
    acceptee = acc["rmse_aeroports_verite"] <= acc["rmse_aeroports_era5"]
    verdict = {"regle": str((REGLE_REPLI if a.repli else REGLE).relative_to(REPO)), "sha256_regle": h_regle, "variante_choisie": nom(v_choix),
               "reglages": v_choix, "acceptation": {k_: acc[k_] for k_ in ("rmse_aeroports_verite", "rmse_aeroports_era5",
                                                                           "stations_verite_meilleure", "stations",
                                                                           "ecart_v300", "ecart_v800")},
               "acceptee": bool(acceptee),
               "suite": regle["si_acceptee"] if acceptee else regle["si_refusee"]}
    (out / f"verdict{suffixe}.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(t.drop(columns=["jeu"]).round(3).to_string(index=False))
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
