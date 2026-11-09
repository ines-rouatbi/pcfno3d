"""Banc de calibration du generateur NATIONAL aux 12 aeroports, sur des jours d'ENTRAINEMENT.

Le diagnostic du 09/11/2026 (docs/diagnostic_generateur_national.md) montre qu'aux aeroports le
detail local que le generateur ajoute a ERA5 n'est pas correle a l'erreur reelle d'ERA5
(alpha -0,08 sur le test 2022). Ce banc mesure, terme par terme, ce que chaque composante du
detail apporte, en coupant une composante a la fois (memes graines, memes jours) :

    base          generateur du jeu national v2 (perturbations 0,4)
    sans_relief   pas d'acceleration par le relief (pre["relief"] = 0)
    sans_pente    pas de vents de pente (A_SLOPE = 0)
    sans_brise    pas de brise de mer / de terre
    sans_perturb  pas de perturbations aleatoires (A_ROT = A_DIV = 0)
    z0_degage     z0 = 0,03 m a 1 km autour de chaque aeroport (terrain d'aerodrome)

Jours tires dans 2023-2025 (jamais le test 2022 ni la validation 2021), 4 instants par jour.
Le vent du generateur a une station = moyenne du bloc 2 x 2 de surface (comme le cache, S) ;
ERA5 a 10 m au noeud (comme _E) ; mesure METAR a 30 min au plus (stations_reelles.obs_a).

    python scripts/calibrer_generateur_stations.py --jours 30 --workers 2 \
        --sortie artifacts/national/calibration_generateur
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NAT = REPO / "artifacts" / "national"
NIVEAUX = "25,50,75,100,150,200,300,400,500,650,800,1000,1250,1500,2000,2500"
CONFIGS = ("base", "sans_relief", "sans_pente", "sans_brise", "sans_perturb", "z0_degage")
EXCLUS = (("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"))   # validation, test
HEURES = (3, 9, 15, 21)

_G = _STATIC = _PRE = _FORCAGE = _CHAMPS = _OBS = _ST = None


def _env(info: dict, perturb: float) -> None:
    z = info["zone"]
    os.environ.update({"ANEO_GEN_DX": str(float(z["dx_m"])), "ANEO_GEN_NIVEAUX": NIVEAUX,
                       "ANEO_GEN_NX": str(z["nx"]), "ANEO_GEN_NY": str(z["ny"]),
                       "ANEO_GEN_F": str(info["f_coriolis"]), "ANEO_GEN_PERTURB": str(perturb)})


def _init(grille: str, forcage: str, obs: str, info_path: str, perturb: float) -> None:
    global _G, _STATIC, _PRE, _FORCAGE, _CHAMPS, _OBS, _ST
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    _env(info, perturb)
    sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
    import generate_aneo_3d as G
    from pcfno3d.stations_reelles import lire_observations
    _G = G
    df = G.load_or_build_grid(Path(grille), np.random.default_rng(0))
    _STATIC = G.build_static(df)
    _PRE = G.precompute_static_derived(_STATIC)
    _FORCAGE = G.lire_forcage(forcage)
    _CHAMPS = G.lire_champs(forcage)
    _ST = {s["code"]: (int(s["iy"]), int(s["ix"])) for s in info["stations"].values()
           if s.get("dans_la_zone")}
    _OBS = lire_observations(Path(obs), verbose=False, codes=list(_ST))


def _variante(config: str):
    """(static, pre, A_ROT, A_DIV, A_SLOPE, brise_nulle) de la configuration."""
    G, static, pre = _G, _STATIC, _PRE
    a_rot, a_div, a_slope = G.A_ROT, G.A_DIV, G.A_SLOPE
    if config == "sans_relief":
        pre = {**pre, "relief": np.zeros_like(pre["relief"])}
    elif config == "sans_pente":
        a_slope = 0.0
    elif config == "sans_perturb":
        a_rot = a_div = 0.0
    elif config == "z0_degage":
        z0 = static["z0"].copy()
        for iy, ix in _ST.values():
            z0[max(iy - 2, 0):iy + 3, max(ix - 2, 0):ix + 3] = 0.03
        static = {**static, "z0": z0}
    return static, pre, a_rot, a_div, a_slope, config == "sans_brise"


def _jour(tache: tuple) -> list[dict]:
    from pcfno3d.stations_reelles import obs_a
    config, date, graine = tache
    G = _G
    static, pre, a_rot, a_div, a_slope, sans_brise = _variante(config)
    sauve = (G.A_ROT, G.A_DIV, G.A_SLOPE)
    G.A_ROT, G.A_DIV, G.A_SLOPE = a_rot, a_div, a_slope
    try:
        st = G.DayStateERA5(0, date, 0, static, graine, _FORCAGE, _CHAMPS)
        if sans_brise:
            st.brise = lambda t_sec, g: 0.0
        k = G.STRIDE
        lignes = []
        for h in HEURES:
            t = pd.Timestamp(date) + pd.Timedelta(hours=h)
            mesures = obs_a(_OBS, t)
            if not mesures:
                continue
            surf, _, _ = G.compute_fields(st, static, pre, h * 3600.0, None)
            champ = st.champ_cible(h * 3600.0)
            for code, (uo, vo) in mesures.items():
                iy, ix = _ST[code]
                jy, jx = (iy // k) * k, (ix // k) * k
                ug = float(surf["u"][jy:jy + k, jx:jx + k].mean())
                vg = float(surf["v"][jy:jy + k, jx:jx + k].mean())
                lat = np.array([[static["lat"][jy, jx]]])
                lon = np.array([[static["lon"][jy, jx]]])
                dix = np.full((1, 1), 10.0)
                ue = float(G._era5_au_sol(st.champs, champ[0], lat, lon, dix)[0, 0])
                ve = float(G._era5_au_sol(st.champs, champ[1], lat, lon, dix)[0, 0])
                lignes.append({"config": config, "code": code, "temps": t, "u_obs": uo, "v_obs": vo,
                               "u_gen": ug, "v_gen": vg, "u_era5": ue, "v_era5": ve})
        return lignes
    finally:
        G.A_ROT, G.A_DIV, G.A_SLOPE = sauve


def jours_entrainement(forcage: str, n: int, graine: int) -> list[pd.Timestamp]:
    t = pd.to_datetime(pd.read_csv(forcage, usecols=["timestamp_utc"])["timestamp_utc"])
    jours = pd.DatetimeIndex(t.dt.tz_localize(None).dt.normalize().unique())
    ok = np.ones(len(jours), bool)
    for a, b in EXCLUS:
        ok &= ~((jours >= pd.Timestamp(a)) & (jours <= pd.Timestamp(b)))
    # une journee complete de forcage (+-3 h autour)
    ok &= (jours > jours.min()) & (jours < jours.max())
    jours = jours[ok]
    rng = np.random.default_rng(graine)
    return sorted(pd.Timestamp(d) for d in rng.choice(jours, size=min(n, len(jours)), replace=False))


def bilan(p: pd.DataFrame) -> pd.DataFrame:
    """Par configuration (et station) : RMSE vectoriel generateur / ERA5, detail ajoute, alpha."""
    def un(g):
        du, dv = g.u_gen - g.u_era5, g.v_gen - g.v_era5
        eu, ev = g.u_obs - g.u_era5, g.v_obs - g.v_era5
        dd = float((du ** 2 + dv ** 2).sum())
        return pd.Series({
            "n": len(g),
            "rmse_gen": float(np.sqrt(np.mean((g.u_gen - g.u_obs) ** 2 + (g.v_gen - g.v_obs) ** 2))),
            "rmse_era5": float(np.sqrt(np.mean(eu ** 2 + ev ** 2))),
            "detail_rms": float(np.sqrt(np.mean(du ** 2 + dv ** 2))),
            "alpha": float((du * eu + dv * ev).sum() / dd) if dd > 0 else np.nan,
            "vit_obs": float(np.hypot(g.u_obs, g.v_obs).mean()),
            "vit_gen": float(np.hypot(g.u_gen, g.v_gen).mean()),
            "vit_era5": float(np.hypot(g.u_era5, g.v_era5).mean())})
    cols = ["u_obs", "v_obs", "u_gen", "v_gen", "u_era5", "v_era5"]
    return p.groupby("config", sort=False)[cols].apply(un).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grille", default=str(NAT / "grille" / "aneo_grid_points_surface.csv"))
    ap.add_argument("--info", default=str(NAT / "grille" / "grille_reelle.json"))
    ap.add_argument("--forcage", default=str(NAT / "forcage_era5.csv"))
    ap.add_argument("--obs", default=str(NAT / "obs"))
    ap.add_argument("--jours", type=int, default=30)
    ap.add_argument("--graine", type=int, default=2026)
    ap.add_argument("--perturbations", type=float, default=0.4, help="comme le jeu national v2")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sortie", default=str(NAT / "calibration_generateur"))
    a = ap.parse_args()

    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    configs = [c for c in a.configs.split(",") if c]
    inconnues = set(configs) - set(CONFIGS)
    if inconnues:
        raise SystemExit(f"configurations inconnues : {sorted(inconnues)}")
    jours = jours_entrainement(a.forcage, a.jours, a.graine)
    # meme graine par jour pour toutes les configurations : seule la composante coupee change
    taches = [(c, d, a.graine + int(d.strftime("%Y%m%d"))) for d in jours for c in configs]
    fichier = sortie / "paires.csv"
    faites = set()
    if fichier.exists():
        old = pd.read_csv(fichier, parse_dates=["temps"])
        faites = set(zip(old["config"], old["temps"].dt.normalize()))
    taches = [t for t in taches if (t[0], t[1]) not in faites]
    print(f"{len(jours)} jours d'entrainement, {len(configs)} configurations, "
          f"{len(taches)} journees a calculer", flush=True)

    initargs = (a.grille, a.forcage, a.obs, a.info, a.perturbations)
    t0 = time.time()
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(a.workers, initializer=_init, initargs=initargs)
        resultats = ex.map(_jour, taches)
    else:
        _init(*initargs)
        resultats = map(_jour, taches)
    for n, lignes in enumerate(resultats, 1):
        if lignes:
            pd.DataFrame(lignes).to_csv(fichier, mode="a", header=not fichier.exists(), index=False)
        print(f"{n}/{len(taches)} journees | {time.time() - t0:.0f} s", flush=True)

    p = pd.read_csv(fichier, parse_dates=["temps"]).dropna()
    b = bilan(p)
    b.to_csv(sortie / "bilan.csv", index=False)
    par_station = pd.concat([bilan(g).assign(code=code) for code, g in p.groupby("code")],
                            ignore_index=True)
    par_station.to_csv(sortie / "bilan_par_station.csv", index=False)
    pd.set_option("display.width", 200)
    print(b.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
