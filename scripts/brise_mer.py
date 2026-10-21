"""Controle « brise de mer » du generateur reel (Grand Tunis), avant la generation complete.

Cycle journalier ETE de la direction (et de la vitesse) du vent a 10 m a Tunis-Carthage et a
Nabeul : mesure (METAR DTTA, SYNOP INM 60728), champ genere au point de la station, ERA5 au
meme point. Aucune station n'entre dans la generation : le champ vient d'ERA5, du relief et du
sol ; les stations ne servent qu'a mesurer l'ecart.

    python scripts/brise_mer.py --periode 2025-07-01:2025-08-23 --pas-jours 2

Sortie : <grand-tunis>/brise_mer/ (brise_mer_<periode>.png, instants.csv, bilan.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from test_generateur_reel import BASE, charger_generateur, era5_aux_stations, observations  # noqa: E402

CODES = {"DTTA": "Tunis-Carthage", "INM60728": "Nabeul"}
_G = {}


def _init(gt: str, perturb: float):
    """Processus de calcul : generateur, grille (capteurs a 10 m), forcage et champs ERA5."""
    gt = Path(gt)
    info = json.loads((gt / "grille_reelle.json").read_text(encoding="utf-8"))
    z = info["zone"]
    gen = charger_generateur(z["nx"], z["ny"], info["f_coriolis"], perturb)
    df = pd.read_csv(gt / "aneo_grid_points_surface.csv")
    df["z_rel_m"] = 10.0
    static = gen.build_static(df)
    _G.update(gen=gen, static=static, pre=gen.precompute_static_derived(static),
              forcage=gen.lire_forcage(gt / "forcage_era5.csv"),
              champs=gen.lire_champs(gt / "forcage_era5.csv"), pos=positions(info))


def positions(info) -> dict:
    pos = {}
    for nom, s in info["stations"].items():
        for court in CODES.values():
            if court.split("-")[0] in nom and s["dans_la_zone"]:
                pos[court] = (s["iy"], s["ix"], s["lat"], s["lon"])
    return pos


def un_jour(jour: str, heures: list[int]) -> list[dict]:
    g = _G
    date = pd.Timestamp(jour).date()
    st = g["gen"].DayStateERA5(1, date, 0, g["static"], 2026, g["forcage"], g["champs"])
    lignes = []
    for h in heures:
        surf, _, meta = g["gen"].compute_fields(st, g["static"], g["pre"], h * 3600.0, None)
        ts = pd.Timestamp(date) + pd.Timedelta(hours=h)
        for court, (iy, ix, _, _) in g["pos"].items():
            lignes.append({"station": court, "t": ts, "u_gen": float(surf["u"][iy, ix]),
                           "v_gen": float(surf["v"][iy, ix]), "brise": meta["brise"]})
    return lignes


def direction(u, v):
    """Direction meteorologique (d'ou vient le vent), degres."""
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def cycle(g: pd.DataFrame, cu: str, cv: str) -> pd.DataFrame:
    """Par heure : direction moyenne (moyenne des vecteurs unitaires), constance (longueur du
    vecteur moyen, 1 = toujours la meme direction) et vitesse moyenne."""
    sp = np.hypot(g[cu], g[cv])
    ok = sp > 0.5                                   # calmes : direction indefinie
    d = g[ok].assign(eu=g[cu] / sp, ev=g[cv] / sp)
    a = d.groupby("heure").agg(eu=("eu", "mean"), ev=("ev", "mean"))
    a["dir"] = direction(a.eu, a.ev)
    a["constance"] = np.hypot(a.eu, a.ev)
    a["vitesse"] = g.assign(sp=sp).groupby("heure")["sp"].mean()
    return a


def figure(df: pd.DataFrame, periode: str, sortie: Path, cotes: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = [("obs", "mesure (anemometre)", "k", "o"), ("gen", "champ genere", "tab:red", "s"),
               ("era5", "ERA5", "tab:blue", "^")]
    stations = [s for s in CODES.values() if s in set(df.station)]
    fig, ax = plt.subplots(2, len(stations), figsize=(6.2 * len(stations), 7.4), sharex=True,
                           squeeze=False)
    for j, s in enumerate(stations):
        g = df[df.station == s]
        for cle, lab, c, m in sources:
            a = cycle(g, f"u_{cle}", f"v_{cle}")
            ax[0, j].scatter(a.index, a["dir"], s=30 + 170 * a["constance"], c=c, marker=m,
                             alpha=0.85, label=lab, edgecolors="none")
            ax[0, j].plot(a.index, a["dir"], c=c, lw=0.8, alpha=0.5)
            ax[1, j].plot(a.index, a["vitesse"], c=c, marker=m, label=lab)
        mer = cotes.get(s)
        if mer is not None:
            lo, hi = mer
            ax[0, j].axhspan(lo, hi, color="tab:cyan", alpha=0.12, label="secteur mer")
        ax[0, j].set_title(f"{s} : direction, ete ({len(g.t.unique())} instants)")
        ax[0, j].set_ylim(0, 360)
        ax[0, j].set_yticks(range(0, 361, 45))
        ax[0, j].set_yticklabels(["N", "NE", "E", "SE", "S", "SO", "O", "NO", "N"])
        ax[1, j].set_title(f"{s} : vitesse moyenne a 10 m")
        ax[1, j].set_ylabel("m/s")
        ax[1, j].set_xlabel("heure UTC (heure locale = UTC + 1)")
        ax[1, j].set_xticks(range(0, 22, 3))
        for a_ in ax[:, j]:
            a_.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Brise de mer, cycle journalier d'ete {periode} — taille du point = constance "
                 "de la direction", fontsize=11)
    fig.tight_layout()
    chemin = sortie / f"brise_mer_{periode.replace(':', '_')}.png"
    fig.savefig(chemin, dpi=130)
    return chemin


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grand-tunis", default=str(BASE / "grand_tunis"))
    ap.add_argument("--era5", default=str(BASE / "era5" / "tunisie"))
    ap.add_argument("--metar", default=str(BASE / "metar" / "historique"))
    ap.add_argument("--isd", default=str(BASE / "isd"))
    ap.add_argument("--periode", required=True, help="AAAA-MM-JJ:AAAA-MM-JJ")
    ap.add_argument("--pas-jours", type=int, default=2)
    ap.add_argument("--heures", default="0,3,6,9,12,15,18,21")
    ap.add_argument("--perturbations", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    gt = Path(args.grand_tunis)
    a, b = args.periode.split(":")
    jours = [str(d.date()) for d in pd.date_range(a, b, freq=f"{args.pas_jours}D")]
    heures = [int(h) for h in args.heures.split(",")]
    sortie = gt / "brise_mer"
    sortie.mkdir(exist_ok=True)
    fichier = sortie / f"instants_{args.periode.replace(':', '_')}.csv"

    t0 = time.time()
    if fichier.exists():
        gen_df = pd.read_csv(fichier, parse_dates=["t"])
        print(f"champ genere relu : {fichier}")
    else:
        lignes = []
        with ProcessPoolExecutor(args.workers, initializer=_init,
                                 initargs=(str(gt), args.perturbations)) as ex:
            for n, l in enumerate(ex.map(un_jour, jours, [heures] * len(jours)), 1):
                lignes += l
                print(f"{n}/{len(jours)} jours ({time.time() - t0:.0f} s)", flush=True)
        gen_df = pd.DataFrame(lignes)
        gen_df.to_csv(fichier, index=False)

    info = json.loads((gt / "grille_reelle.json").read_text(encoding="utf-8"))
    pos = positions(info)
    temps = sorted(gen_df.t.unique())
    era = era5_aux_stations(Path(args.era5), {k: v[2:] for k, v in pos.items()}, list(temps))
    obs = observations(Path(args.metar), Path(args.isd))
    obs = obs[obs.station_id.isin(CODES)].assign(station=lambda d: d.station_id.map(CODES))
    df = gen_df.merge(era, on=["station", "t"], how="left")
    lignes = []
    for s, g in df.groupby("station"):
        o = obs[obs.station == s].sort_values("t")
        m = pd.merge_asof(g.sort_values("t"), o[["t", "wind_u_m_s", "wind_v_m_s"]], on="t",
                          direction="nearest", tolerance=pd.Timedelta(minutes=30))
        lignes.append(m)
    df = pd.concat(lignes).rename(columns={"wind_u_m_s": "u_obs", "wind_v_m_s": "v_obs"})
    df = df.dropna(subset=["u_obs", "v_obs", "u_era5", "v_era5"])
    df["heure"] = df.t.dt.hour

    # secteur « mer » vu de chaque station (cote : Tunis-Carthage golfe a l'E-NE, Nabeul au SE)
    cotes = {"Tunis-Carthage": (0.0, 135.0), "Nabeul": (45.0, 225.0)}
    bilan = {"periode": args.periode, "instants_apparies": int(len(df))}
    for s, g in df.groupby("station"):
        r = {}
        for cle in ("obs", "gen", "era5"):
            d = direction(g[f"u_{cle}"], g[f"v_{cle}"])
            lo, hi = cotes[s]
            jour = g.heure.isin([9, 12, 15])
            r[cle] = {"part_secteur_mer_9_15h_utc": round(float(((d >= lo) & (d <= hi))[jour].mean()), 3),
                      "part_secteur_mer_nuit": round(float(((d >= lo) & (d <= hi))[~jour].mean()), 3),
                      "vitesse_moy": round(float(np.hypot(g[f"u_{cle}"], g[f"v_{cle}"]).mean()), 2)}
        for cle in ("gen", "era5"):
            r[f"rmse_vect_{cle}"] = round(float(np.sqrt(np.mean(
                (g[f"u_{cle}"] - g.u_obs) ** 2 + (g[f"v_{cle}"] - g.v_obs) ** 2))), 2)
        r["n"] = int(len(g))
        bilan[s] = r
    df.to_csv(sortie / f"apparies_{args.periode.replace(':', '_')}.csv", index=False)
    chemin = figure(df, args.periode, sortie, cotes)
    (sortie / f"bilan_{args.periode.replace(':', '_')}.json").write_text(
        json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(bilan, indent=1, ensure_ascii=False))
    print(f"figure : {chemin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
