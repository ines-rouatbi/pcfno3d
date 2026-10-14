"""M1, etape 0 (CPU, sans GPU) : l'erreur du vent geostrophique estime, avant / apres correctif.

Le generateur (scripts/generate_aneo_3d.py) impose un plan EXACT de pression au niveau de
la mer, p_sl = p0 + 1,2 f (vg x - ug y), puis reduit chaque capteur a son altitude avec la
temperature VIRTUELLE moyenne de la colonne, 0,5 (T_sl + T) (1 + 0,608 q), ou T contient le
terme de stabilite de couche limite. L'estimation du pipeline (pcfno3d.features, BLOC 6)
ignore l'humidite et la stabilite, et divise par la densite moyenne des capteurs au lieu
de 1,2. Ce script mesure ce que coute chacun de ces ecarts, par rapport au geostrophique
VRAI de chaque instant.

Verite : le dataset ne contient pas (ug, vg). On les recalcule en rejouant les tirages du
generateur (graine 2026 * 1000 + jour, comme `main()`), secteur de direction compris (rejeu
du tirage du relief synthetique). Le rejeu est VERIFIE, pas suppose : la norme |G| doit
retrouver la colonne `g` de qc/qc_day_XXX.csv (ecart attendu ~1e-15), sinon arret.

Lecture seule : le cache (S, Y) et les CSV journaliers ne sont jamais modifies. Les jours
absents du cache sont lus en flux depuis le CSV (surface + 3 niveaux hauts), ~1 min/jour.
Uniquement des jours d'ENTRAINEMENT et de VALIDATION : le test reste reserve au verdict.

    python scripts/diag_geostrophique.py --data-dir <aneo_dataset_3d_v2> --cache-dir <cache>
        --days trainval --out diag.json          # 76 jours, ~25 min de CPU
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]

# constantes du generateur (scripts/generate_aneo_3d.py)
SEED = 2026
G = 9.80665
RD = 287.05
F_COR = 8.57e-5
LAPSE = 0.0065
RHO_GEN = 1.2                         # densite du plan de pression du generateur
HAUTS = (650.0, 800.0, 1000.0)        # niveaux ou le DSO1 echoue


# ------------------------------------------------------------------ verite (rejeu)
def secteurs(n_days: int = 100) -> np.ndarray:
    """Rejoue main() : relief synthetique (4 champs + z_rel) puis melange des secteurs."""
    rng = np.random.default_rng(SEED)
    for _ in range(4):
        rng.standard_normal((500, 1000))
    rng.uniform(20.0, 30.0, size=(500, 1000))
    sec = np.tile(np.arange(8), n_days // 8 + 1)[:n_days]
    rng.shuffle(sec)
    return sec


def verite(day: int, t_sec: float, sector: int, n_days: int = 100) -> dict:
    """Memes tirages, dans le meme ordre, que DayState.__init__ / geostrophic()."""
    rng = np.random.default_rng(SEED * 1000 + day)
    cat = rng.random()
    if cat < 0.12:
        g_mean = rng.uniform(1.0, 2.5)
    elif cat < 0.80:
        g_mean = rng.uniform(4.0, 12.0)
    else:
        g_mean = rng.uniform(15.0, 26.0)
    dir_mean = (sector * 45.0 + rng.uniform(-15.0, 15.0)) % 360.0
    g_phase, dir_phase = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    dir_swing = rng.uniform(10.0, 30.0)
    rng.uniform(0.0, 1.0)                                   # cloud
    rng.uniform(20.0, 40.0)                                 # veer
    rng.random()                                            # jet_day
    rng.uniform(0.2, 0.5)                                   # jet_amp
    rng.uniform(150.0, 400.0)                               # jet_z
    grad_mag, grad_dir = rng.uniform(0.0, 1.2e-5), rng.uniform(0.0, 2.0 * math.pi)
    date = dt.date(2025, 1, 1) + dt.timedelta(days=int(round((day - 1) * 365.0 / n_days)))
    doy = date.timetuple().tm_yday
    t_mean = 19.0 + 8.0 * math.sin(2 * math.pi * (doy - 110) / 365.0) + rng.normal(0, 1.5)

    h = t_sec / 3600.0
    g = g_mean * (1.0 + 0.15 * math.sin(2 * math.pi * (h - 3.0) / 24.0 + g_phase))
    d = math.radians((dir_mean + dir_swing * math.sin(2 * math.pi * h / 24.0 + dir_phase)) % 360)
    coef = G / (F_COR * (t_mean + 273.15))
    return {"g": g, "ug": -g * math.sin(d), "vg": -g * math.cos(d),
            "su": -coef * grad_mag * math.sin(grad_dir), "sv": coef * grad_mag * math.cos(grad_dir)}


# ------------------------------------------------------------------ estimateurs
def geo_estime(S: np.ndarray, m: np.ndarray, geom, tv: bool, zreg: bool,
               rho12: bool) -> tuple[float, float]:
    """tv/zreg/rho12 a False = exactement pcfno3d.features.geostrophic_from_surface."""
    p = S[6][m].astype(np.float64)
    z = geom.z_sensor[m]
    if tv:   # temperature virtuelle du capteur, exacte : rho = p / (Rd Tv)
        t_ref = p / (RD * S[4][m].astype(np.float64))
    else:
        t_ref = S[3][m].astype(np.float64) + 273.15
    p_sl = p * np.exp(G * z / (RD * (t_ref + 0.5 * LAPSE * z)))
    cols = [np.ones(m.sum()), geom.xc[m], geom.yc[m]] + ([z - z.mean()] if zreg else [])
    coef, *_ = np.linalg.lstsq(np.stack(cols, axis=1), p_sl, rcond=None)
    rho = RHO_GEN if rho12 else float(S[4][m].mean())
    return -coef[2] / (rho * F_COR), coef[1] / (rho * F_COR)


VARIANTES = {   # nom : (Tv, z regresseur, rho = 1,2)
    "actuel (v7)": (False, False, False),
    "rho=1,2 seul": (False, False, True),
    "Tv seul": (True, False, False),
    "z seul": (False, True, False),
    "Tv + z": (True, True, False),
    "Tv + rho=1,2": (True, False, True),
    "z + rho=1,2": (False, True, True),
    "correctif (Tv + z + rho=1,2)": (True, True, True),
}


# ------------------------------------------------------------------ donnees
def instants_du_cache(cache_dir: Path, day: int):
    idx = pd.read_csv(cache_dir / "index.csv")
    for r in idx[idx["day"] == day].itertuples():
        S = np.load(cache_dir / f"{r.stem}_S.npy")
        Y = np.load(cache_dir / f"{r.stem}_Y.npy", mmap_mode="r")
        yield r.stem, r.hour * 3600.0, S, Y[:2]


def instants_du_csv(path: Path, geom, kz: list[int]):
    """Lecture en flux (memoire bornee) : surface poolee + u, v aux niveaux kz."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    from pcfno3d.geometry import SURF_VARS

    cols = ["point_id", "timestamp_utc", *SURF_VARS, "mask_i_obs"]
    conv = pacsv.ConvertOptions(include_columns=cols, column_types={"timestamp_utc": "string"})
    reader = pacsv.open_csv(pa.input_stream(str(path), compression="gzip"),
                            read_options=pacsv.ReadOptions(block_size=64 << 20),
                            convert_options=conv)
    nz, ny, nx = geom.shape
    acc: dict[str, dict] = {}
    for batch in reader:
        ts = batch.column("timestamp_utc").to_pandas()
        pid = batch.column("point_id").to_numpy()
        data = {c: batch.column(c).to_numpy().astype(np.float32) for c in cols[2:]}
        for stamp in ts.unique():
            sel = (ts == stamp).to_numpy()
            a = acc.setdefault(stamp, {"S": np.zeros((8, geom.ny_s * geom.nx_s), np.float32),
                                       "Y": np.zeros((2, nz, ny * nx), np.float32)})
            p = pid[sel]
            s = p < geom.node_id_start
            flat = geom.pid_to_surf[p[s]]
            for ci, var in enumerate(SURF_VARS):
                a["S"][ci, flat] = data[var][sel][s]
            a["S"][7, flat] = data["mask_i_obs"][sel][s]
            nflat = geom.node_flat[p[~s] - geom.node_id_start]
            k = nflat // (ny * nx)
            keep = np.isin(k, kz)
            for ci, var in enumerate(("wind_u_m_s", "wind_v_m_s")):
                a["Y"][ci, k[keep], nflat[keep] % (ny * nx)] = data[var][sel][~s][keep]
    for stamp in sorted(acc):
        t = pd.Timestamp(stamp)
        a = acc.pop(stamp)
        yield (f"{t:%H%M%S}", t.hour * 3600.0 + t.minute * 60 + t.second,
               geom.pool_surface(a["S"]), a["Y"].reshape(2, nz, ny, nx))


# ------------------------------------------------------------------ previsibilite
def cv_niveaux(df: pd.DataFrame, keep: float) -> dict:
    """Le vent MOYEN d'un niveau haut est-il mieux previsible avec le geostrophique corrige ?

    Regression lineaire, validee en laissant un jour de cote (pas de fuite entre instants
    d'un meme jour), sur ce que le modele recoit comme nombres synoptiques : geostrophique,
    vent thermique, heure, et leurs produits (la couche limite depend de l'heure). Le
    jeu A est celui de la v7, B le correctif, C l'oracle (geostrophique et thermique vrais).
    Ce n'est pas le modele : c'est un plafond lineaire de l'information synoptique.
    """
    h = df["heure"].str[:2].astype(float).to_numpy() * 2 * np.pi / 24
    sh, ch = np.sin(h), np.cos(h)
    jeux = {"A actuel (v7)": (f"{keep}|actuel (v7)|ug", f"{keep}|actuel (v7)|vg",
                              f"{keep}|thermique|su", f"{keep}|thermique|sv"),
            "B correctif": (f"{keep}|correctif (Tv + z + rho=1,2)|ug",
                            f"{keep}|correctif (Tv + z + rho=1,2)|vg",
                            f"{keep}|thermique|su", f"{keep}|thermique|sv"),
            "C oracle": ("ug", "vg", "su", "sv")}
    out: dict = {}
    print("\n--- vent moyen de niveau, regression validee jour par jour : RMSE (m/s)")
    for z in HAUTS:
        ligne = {}
        for nom, (cu, cv, cs, ct) in jeux.items():
            ug, vg = df[cu].to_numpy(), df[cv].to_numpy()
            su, sv = df[cs].to_numpy() * 1e3, df[ct].to_numpy() * 1e3
            A = np.stack([np.ones(len(df)), ug, vg, su, sv, sh, ch,
                          ug * sh, ug * ch, vg * sh, vg * ch], axis=1)
            for comp in ("U", "V"):
                y = df[f"{comp}{int(z)}"].to_numpy()
                err = np.empty_like(y)
                for d in df["jour"].unique():
                    te = (df["jour"] == d).to_numpy()
                    c, *_ = np.linalg.lstsq(A[~te], y[~te], rcond=None)
                    err[te] = y[te] - A[te] @ c
                ligne[f"{comp}|{nom}"] = float(np.sqrt(np.mean(err ** 2)))
        out[int(z)] = ligne
        print(f"{int(z):5d} m  " + "  ".join(f"{k} {v:.3f}" for k, v in ligne.items()))
    return out


# ------------------------------------------------------------------ transferabilite au reel
def sensibilite_bruit(instants: list, geom, sigmas: list[float], seed: int) -> dict:
    """Le correctif tient-il avec de VRAIS barometres ? Erreur de G selon reseau et bruit.

    La pression synthetique est exacte (arrondie a 0,1 Pa). Un barometre reel a une erreur
    propre de 10 a 50 Pa (etalonnage), et une erreur d'altitude de 1 m en vaut ~12 Pa.
    On ajoute un bruit gaussien independant par capteur a p, puis on applique le correctif,
    pour trois reseaux : 50 % de la couche synthetique, un capteur tous les 500 m, tous les 1 km.
    """
    from pcfno3d.dataset import lattice_mask

    reseaux = {"50 % synthetique": None, "reseau 500 m": 500.0, "reseau 1 km": 1000.0}
    out: dict = {}
    print("\n--- correctif + bruit barometrique : RMSE vectorielle de G (m/s)")
    print(f"{'reseau':18s} {'capteurs':>8s} " + " ".join(f"{s:>7.0f}Pa" for s in sigmas))
    for nom, spacing in reseaux.items():
        ligne, n_moy = {}, []
        for s in sigmas:
            err = []
            for i, (day, t_sec, S, tr) in enumerate(instants):
                rng = np.random.default_rng(seed + i)
                if spacing is None:
                    m = (S[7] > 0.5) & (rng.random(S[7].shape) < 0.5)
                else:
                    m = (S[7] > 0.5) & lattice_mask(S[7].shape, spacing, geom.dx, rng)
                n_moy.append(int(m.sum()))
                Sb = S.copy()
                Sb[6] = S[6] + (rng.normal(0.0, s, S[6].shape) if s > 0 else 0.0)
                ug, vg = geo_estime(Sb, m, geom, True, True, True)
                err.append((ug - tr["ug"]) ** 2 + (vg - tr["vg"]) ** 2)
            ligne[f"{s:g}"] = float(np.sqrt(np.mean(err)))
        out[nom] = {"capteurs": int(np.mean(n_moy)), **ligne}
        print(f"{nom:18s} {out[nom]['capteurs']:8d} "
              + " ".join(f"{v:9.3f}" for v in ligne.values()))
    return out


# ------------------------------------------------------------------ programme
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="local_v7")
    ap.add_argument("--data-dir", required=True, help="aneo_dataset_3d_v2 (lecture seule)")
    ap.add_argument("--cache-dir", required=True, help="cache (S, Y) existant (lecture seule)")
    ap.add_argument("--days", default="2,3,14",
                    help="jours d'entrainement / validation, ou 'trainval' (tous, sans episodes)")
    ap.add_argument("--keeps", default="0.5,1.0,0.25", help="fractions de capteurs visibles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bruit-pa", default="",
                    help="ex. 0,10,30,50 : sensibilite du correctif au bruit des barometres")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from pcfno3d.config import Config
    from pcfno3d.features import geostrophic_from_surface, thermal_wind_from_surface
    from pcfno3d.geometry import Geometry

    data_dir, cache_dir = Path(args.data_dir), Path(args.cache_dir)
    tmp = Path(tempfile.mkdtemp(prefix="diag_geo_"))      # rien n'est ecrit dans le cache
    if (cache_dir / "geometry_v1.npz").exists():
        shutil.copy2(cache_dir / "geometry_v1.npz", tmp / "geometry_v1.npz")
    cfg = Config.load(args.config, paths__data_dir=data_dir, paths__cache_dir=tmp)
    geom = Geometry(cfg, device="cpu")
    split = json.loads((data_dir / "split_days.json").read_text(encoding="utf-8"))
    if args.days == "trainval":   # tous les jours train/val sans episode rapide (synoptiques)
        cal = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))["calendrier"]
        rapides = set(cal["jours_episodes_minute"]) | set(cal["jours_episodes_10s"])
        days = sorted(d for d in split["train_days"] + split["val_days"] if d not in rapides)
    else:
        days = [int(d) for d in args.days.split(",")]
    interdits = [d for d in days if d in split["test_days"]]
    if interdits:
        sys.exit(f"jours de TEST refuses (reserves au verdict) : {interdits}")
    kz = [int(np.argmin(np.abs(geom.z_levels - z))) for z in HAUTS]
    keeps = [float(k) for k in args.keeps.split(",")]
    sec = secteurs()
    cached = set(pd.read_csv(cache_dir / "index.csv")["day"]) if (
        cache_dir / "index.csv").exists() else set()

    sigmas = [float(s) for s in args.bruit_pa.split(",") if s.strip()]
    gardes = []                                         # instants gardes pour le bruit
    rows = []
    for day in days:
        t0 = time.time()
        qc = pd.read_csv(data_dir / "qc" / f"qc_day_{day:03d}.csv")
        qc_g = dict(zip(pd.to_datetime(qc["timestamp_utc"]).dt.strftime("%H%M%S"), qc["g"]))
        if day in cached:
            src = instants_du_cache(cache_dir, day)
        else:
            path = next(data_dir.glob(f"aneo_observations_day_{day:03d}_*_multitime.csv*"))
            src = instants_du_csv(path, geom, kz)
        for stem, t_sec, S, Y in src:
            hhmmss = stem.split("_")[-1]
            tr = verite(day, t_sec, int(sec[day - 1]))
            if abs(tr["g"] - qc_g[hhmmss]) > 1e-6:
                sys.exit(f"rejeu du generateur FAUX (jour {day} {hhmmss}) : "
                         f"|G| {tr['g']:.6f} contre qc {qc_g[hhmmss]:.6f}")
            row = {"jour": day, "split": "val" if day in split["val_days"] else "train",
                   "heure": hhmmss, **tr}
            for k, z in zip(kz, HAUTS):
                v = geom.valid[k]
                row[f"U{int(z)}"] = float(np.asarray(Y[0, k])[v].mean())
                row[f"V{int(z)}"] = float(np.asarray(Y[1, k])[v].mean())
            for keep in keeps:
                rng = np.random.default_rng(args.seed + len(rows))
                m = (S[7] > 0.5) & (rng.random(S[7].shape) < keep)
                ref = geostrophic_from_surface(S, m, geom)
                for name, opts in VARIANTES.items():
                    ug, vg = geo_estime(S, m, geom, *opts)
                    if name == "actuel (v7)":
                        assert abs(ug - ref[0]) < 1e-6 and abs(vg - ref[1]) < 1e-6, \
                            "la variante 'actuel' doit reproduire pcfno3d.features"
                    row[f"{keep}|{name}|ug"], row[f"{keep}|{name}|vg"] = ug, vg
                su, sv = thermal_wind_from_surface(S, m, geom)
                row[f"{keep}|thermique|su"], row[f"{keep}|thermique|sv"] = su, sv
            rows.append(row)
            if sigmas:   # ~4 Mo par instant : a reserver a quelques jours (ex. 2,3,14)
                gardes.append((day, t_sec, S, tr))
        print(f"jour {day:3d} ({row['split']}) : {time.time() - t0:5.0f} s", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)

    df = pd.DataFrame(rows)
    res = {"instants": len(df), "jours": days, "variantes": {}, "niveaux_hauts": {}}
    print(f"\n{len(df)} instants synoptiques (train/val), verite rejouee et verifiee sur qc.g\n")
    for keep in keeps:
        print(f"--- {keep:.0%} des capteurs visibles : erreur du geostrophique estime (m/s)")
        print(f"{'variante':32s} {'RMSE ug':>8s} {'RMSE vg':>8s} {'vect':>6s} {'p95':>6s} "
              f"{'pente':>6s} {'gain':>6s}")
        base = None
        for name in [*VARIANTES]:
            eu = df[f"{keep}|{name}|ug"] - df["ug"]
            ev = df[f"{keep}|{name}|vg"] - df["vg"]
            vec = np.sqrt(eu ** 2 + ev ** 2)
            rv = float(np.sqrt(np.mean(vec ** 2)))
            base = rv if base is None else base
            est = np.concatenate([df[f"{keep}|{name}|ug"], df[f"{keep}|{name}|vg"]])
            tru = np.concatenate([df["ug"], df["vg"]])
            pente = float(np.dot(est, tru) / np.dot(tru, tru))
            r = {"rmse_ug": float(np.sqrt(np.mean(eu ** 2))),
                 "rmse_vg": float(np.sqrt(np.mean(ev ** 2))), "rmse_vect": rv,
                 "p95_vect": float(np.percentile(vec, 95)), "pente": pente,
                 "reduction_vs_v7": 1.0 - rv / base}
            res["variantes"][f"{keep}|{name}"] = r
            print(f"{name:32s} {r['rmse_ug']:8.3f} {r['rmse_vg']:8.3f} {rv:6.3f} "
                  f"{r['p95_vect']:6.3f} {pente:6.3f} {r['reduction_vs_v7']:6.1%}")
        es = np.sqrt((df[f"{keep}|thermique|su"] - df["su"]) ** 2
                     + (df[f"{keep}|thermique|sv"] - df["sv"]) ** 2)
        vrai = float(np.sqrt(np.mean(df.su ** 2 + df.sv ** 2)))
        res[f"thermique_{keep}"] = {"rmse_vect_s-1": float(np.sqrt(np.mean(es ** 2))),
                                    "rms_vrai_s-1": vrai}
        print(f"vent thermique : RMSE {res[f'thermique_{keep}']['rmse_vect_s-1']:.2e} s-1 "
              f"(norme vraie RMS {res[f'thermique_{keep}']['rms_vrai_s-1']:.2e} s-1)\n")

    # niveaux hauts : part du vent moyen de niveau expliquee par le geostrophique (lineaire)
    print("--- vent moyen de niveau (verite) ~ a + b ug + c vg : RMS du residu (m/s)")
    keep = keeps[0]
    for z in HAUTS:
        out = {}
        for name, (cu, cv) in {"vrai": ("ug", "vg"),
                               "actuel (v7)": (f"{keep}|actuel (v7)|ug", f"{keep}|actuel (v7)|vg"),
                               "correctif": (f"{keep}|correctif (Tv + z + rho=1,2)|ug",
                                             f"{keep}|correctif (Tv + z + rho=1,2)|vg")}.items():
            A = np.stack([np.ones(len(df)), df[cu], df[cv]], axis=1)
            for comp in ("U", "V"):
                y = df[f"{comp}{int(z)}"].to_numpy()
                c, *_ = np.linalg.lstsq(A, y, rcond=None)
                out[f"{comp}|{name}"] = float(np.sqrt(np.mean((y - A @ c) ** 2)))
        res["niveaux_hauts"][int(z)] = out
        print(f"{int(z):5d} m  " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()))

    res["cv_niveaux_hauts"] = cv_niveaux(df, keeps[0])
    if sigmas:
        res["bruit_barometrique"] = sensibilite_bruit(gardes, geom, sigmas, args.seed)

    if args.out:
        args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        df.to_csv(args.out.with_suffix(".csv"), index=False)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
