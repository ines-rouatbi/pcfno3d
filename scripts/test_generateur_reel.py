"""Point de decision J2 du generateur reel (Grand Tunis) : les champs generes sont-ils credibles ?

Genere des instants (en memoire, sans ecrire le jeu) et mesure, sans qu'aucune station n'entre
dans la generation :
  1. le residu de masse sur les noeuds 3D (seuil DSO2 1e-4) ;
  2. la moyenne du champ sur la zone, niveau par niveau, contre le vent ERA5 moyen au meme
     niveau (le generateur doit retrouver sa grande echelle) ;
  3. le vent a 10 m aux trois stations reelles (Tunis-Carthage, Zaghouan, Nabeul) contre la
     mesure, compare a l'ecart ERA5 - mesure au meme point : critere « pas pire qu'ERA5 ».

    python scripts/test_generateur_reel.py --dates 2021-01-05,2021-02-10,2022-03-15 \
        --grand-tunis C:/Users/inesr/AppData/Local/pcfno3d/grand_tunis

Sortie : <grand-tunis>/test_j2/ (bilan.json, niveaux.csv, stations.csv).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "pcfno3d"


def charger_generateur(nx: int, ny: int, f: float, perturb: float):
    os.environ.update({"ANEO_GEN_NX": str(nx), "ANEO_GEN_NY": str(ny), "ANEO_GEN_F": str(f),
                       "ANEO_GEN_PERTURB": str(perturb)})
    spec = importlib.util.spec_from_file_location("gen", REPO / "scripts" / "generate_aneo_3d.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


def observations(metar_dir: Path, isd_dir: Path) -> pd.DataFrame:
    fichiers = sorted(metar_dir.glob("metar_*.csv")) + sorted(isd_dir.glob("synop_*.csv"))
    obs = pd.concat([pd.read_csv(f) for f in fichiers], ignore_index=True)
    obs = obs[obs["station_id"].isin(["DTTA", "INM60729", "INM60728"])]
    obs["t"] = pd.to_datetime(obs["timestamp_utc"]).dt.tz_localize(None)
    # l'archive METAR garde vitesse et direction, souvent sans u, v : on les recalcule
    rad = np.radians(obs["wind_dir_deg"])
    calme = obs["wind_speed_m_s"] == 0
    obs["wind_u_m_s"] = obs["wind_u_m_s"].fillna((-obs["wind_speed_m_s"] * np.sin(rad)).where(~calme, 0.0))
    obs["wind_v_m_s"] = obs["wind_v_m_s"].fillna((-obs["wind_speed_m_s"] * np.cos(rad)).where(~calme, 0.0))
    return obs.dropna(subset=["wind_u_m_s", "wind_v_m_s"])


def era5_aux_stations(era5_dir: Path, stations: dict, temps: list) -> pd.DataFrame:
    import xarray as xr

    ds = []
    for f in sorted(era5_dir.glob("era5_sl_*.nc")):
        d = xr.open_dataset(f)
        tdim = "valid_time" if "valid_time" in d.dims else "time"
        d = d.rename({tdim: "temps"})[["u10", "v10"]]
        garde = np.intersect1d(d["temps"].values, np.array(temps, dtype="datetime64[ns]"))
        if len(garde):
            ds.append(d.sel(temps=garde).load())
    tout = xr.concat(ds, dim="temps")
    lignes = []
    for nom, (la, lo) in stations.items():
        p = tout.interp(latitude=la, longitude=lo)
        lignes.append(pd.DataFrame({"station": nom, "t": pd.to_datetime(p["temps"].values),
                                    "u_era5": p["u10"].values, "v_era5": p["v10"].values}))
    return pd.concat(lignes).drop_duplicates(["station", "t"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grand-tunis", default=str(BASE / "grand_tunis"))
    ap.add_argument("--era5", default=str(BASE / "era5" / "tunisie"))
    ap.add_argument("--metar", default=str(BASE / "metar" / "historique"))
    ap.add_argument("--isd", default=str(BASE / "isd"))
    ap.add_argument("--dates", required=True, help="AAAA-MM-JJ separes par des virgules")
    ap.add_argument("--heures", default="0,3,6,9,12,15,18,21")
    ap.add_argument("--perturbations", type=float, default=0.4)
    args = ap.parse_args()

    gt = Path(args.grand_tunis)
    info = json.loads((gt / "grille_reelle.json").read_text(encoding="utf-8"))
    nx, ny, dx = info["zone"]["nx"], info["zone"]["ny"], info["zone"]["dx_m"]
    gen = charger_generateur(nx, ny, info["f_coriolis"], args.perturbations)

    t0 = time.time()
    df = pd.read_csv(gt / "aneo_grid_points_surface.csv")
    df["z_rel_m"] = 10.0                          # « capteurs » a 10 m : hauteur des anemometres
    static = gen.build_static(df)
    pre = gen.precompute_static_derived(static)
    forcage = gen.lire_forcage(gt / "forcage_era5.csv")
    champs = gen.lire_champs(gt / "forcage_era5.csv")
    print("recalage", "SPATIAL (champs ERA5 par maille)" if champs is not None else "uniforme")
    print(f"grille {ny} x {nx} chargee en {time.time() - t0:.0f} s ; noeuds 3D {static['node_k'].size:,}")

    codes = {"DTTA": "Tunis-Carthage", "INM60729": "Zaghouan", "INM60728": "Nabeul"}
    pos = {}
    for nom, s in info["stations"].items():
        for code, court in codes.items():
            if court.split("-")[0] in nom and s["dans_la_zone"]:
                pos[court] = (s["iy"], s["ix"], s["lat"], s["lon"])
    obs = observations(Path(args.metar), Path(args.isd))

    nv = static["node_valid"]
    z = gen.Z_LEVELS
    dz = np.diff(z)
    paire = nv[1:] & nv[:-1]
    niveaux, stations, masses, temps = [], [], [], []
    for jour in args.dates.split(","):
        date = pd.Timestamp(jour).date()
        st = gen.DayStateERA5(1, date, 0, static, 2026, forcage, champs)
        for h in (int(x) for x in args.heures.split(",")):
            t1 = time.time()
            surf, vol, meta = gen.compute_fields(st, static, pre, h * 3600.0, None)
            ts = pd.Timestamp(date) + pd.Timedelta(hours=h)
            temps.append(ts)
            r = gen.mass_residual(vol["rho"], vol["u"], vol["v"], vol["w"], z,
                                  static["dx_s"] * gen.STRIDE, static["dy_s"] * gen.STRIDE)
            masses.append(float(np.sqrt(np.mean(r[paire] ** 2))))
            f = forcage.loc[ts]
            for k, zk in enumerate(z):
                ok = nv[k]
                if ok.mean() < 0.05:
                    continue
                niveaux.append({"t": ts, "z_m": zk, "couverture": float(ok.mean()),
                                "u_gen": float(vol["u"][k][ok].mean()), "v_gen": float(vol["v"][k][ok].mean()),
                                "u_era5": f[f"u_era5_{int(zk)}"], "v_era5": f[f"v_era5_{int(zk)}"]})
            for court, (iy, ix, la, lo) in pos.items():
                stations.append({"station": court, "t": ts, "u_gen": float(surf["u"][iy, ix]),
                                 "v_gen": float(surf["v"][iy, ix])})
            print(f"{ts} : {time.time() - t1:.0f} s | g {meta['g']:.1f} m/s, h_cl {meta['h_bl']:.0f} m, "
                  f"brise {meta['brise']:+.2f} m/s, residu de masse {masses[-1]:.2e}", flush=True)

    sortie = gt / "test_j2"
    sortie.mkdir(exist_ok=True)
    nivx = pd.DataFrame(niveaux)
    nivx.to_csv(sortie / "niveaux.csv", index=False)
    par_niveau = nivx.assign(du=nivx.u_gen - nivx.u_era5, dv=nivx.v_gen - nivx.v_era5).groupby("z_m").agg(
        couverture=("couverture", "mean"), biais_u=("du", "mean"), biais_v=("dv", "mean"),
        rmse_vect=("du", lambda s: np.nan)).reset_index()
    for i, zk in enumerate(par_niveau["z_m"]):
        g = nivx[nivx.z_m == zk]
        par_niveau.loc[i, "rmse_vect"] = float(np.sqrt(np.mean((g.u_gen - g.u_era5) ** 2 + (g.v_gen - g.v_era5) ** 2)))

    sta = pd.DataFrame(stations)
    era = era5_aux_stations(Path(args.era5), {k: v[2:] for k, v in pos.items()}, temps)
    sta = sta.merge(era, on=["station", "t"], how="left")
    o = obs.assign(station=obs["station_id"].map(codes))
    lignes = []
    for _, r in sta.iterrows():
        cand = o[(o.station == r.station) & ((o.t - r.t).abs() <= pd.Timedelta(minutes=30))]
        if cand.empty:
            continue
        c = cand.iloc[(cand.t - r.t).abs().argmin()]
        lignes.append({**r.to_dict(), "u_obs": c.wind_u_m_s, "v_obs": c.wind_v_m_s})
    sta = pd.DataFrame(lignes)
    sta.to_csv(sortie / "stations.csv", index=False)
    bil_sta = {}
    for court, g in sta.groupby("station"):
        e_gen = np.sqrt(np.mean((g.u_gen - g.u_obs) ** 2 + (g.v_gen - g.v_obs) ** 2))
        e_era = np.sqrt(np.mean((g.u_era5 - g.u_obs) ** 2 + (g.v_era5 - g.v_obs) ** 2))
        bil_sta[court] = {"n": len(g), "rmse_vect_genere": float(e_gen), "rmse_vect_era5": float(e_era),
                          "vitesse_moy_obs": float(np.hypot(g.u_obs, g.v_obs).mean()),
                          "vitesse_moy_genere": float(np.hypot(g.u_gen, g.v_gen).mean()),
                          "vitesse_moy_era5": float(np.hypot(g.u_era5, g.v_era5).mean())}
    bilan = {"instants": len(temps), "residu_masse_rms_max": max(masses),
             "residu_masse_rms_moyen": float(np.mean(masses)), "seuil_dso2": 1e-4,
             "niveaux": par_niveau.round(3).to_dict("records"), "stations": bil_sta}
    (sortie / "bilan.json").write_text(json.dumps(bilan, indent=2, default=str), encoding="utf-8")
    print(json.dumps(bilan, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
