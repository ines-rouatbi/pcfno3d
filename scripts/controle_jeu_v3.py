"""Controle qualite du jeu national v3 complet (demande d'Ines du 10/11), sans entrainement.
Protocole : docs/resultats/national/plan_v4_controle_et_place.md, point 1.

a) conservation de la masse : residu r(k+1/2) de meta.json (operateur du generateur, Geometry.mass_residual)
   sur les demi-niveaux dont les deux noeuds sont valides (et les 4 voisins horizontaux), sur les 1000 instants
   du cache v3 ; meme calcul sur les instants de validation et de test de la v2 (lus dans les zip du Drive) ;
b) manifeste : pcfno3d manifeste-cache (ecrit puis verifie) + controles : calendrier = celui de la v2,
   700 / 120 / 180 instants, _S _Y _E presents pour chaque instant, aucun NaN / infini ;
c) 10 instants de validation (graine 2031) : regeneres par le generateur lui-meme (--days, memes reglages,
   memes graine et secteur) -> ecart max au cache ; puis chiffres du rapport (ecart verite - ERA5 sur u, v a
   300 / 800 m au-dessus du sol, aeroports verite contre ERA5) sur le cache ET par le chemin de calcul du
   rapport (compute_fields, phase 2), memes instants.

    python scripts/controle_jeu_v3.py -c <config national pointant vers le jeu et le cache v3> \
        --nat /content/drive/MyDrive/pcfno3d_national --donnees /content/donnees/national --sortie DIR
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
HAUTEURS = (300.0, 800.0)


def residu_masse(geom, Y: np.ndarray) -> tuple[float, float, int]:
    """(somme des carres, max |r|, n) du residu sur les demi-niveaux interieurs valides."""
    import torch
    t = torch.from_numpy(Y.astype(np.float32))[:, None]
    r = geom.mass_residual(t[3], t[0], t[1], t[2])[0].numpy()
    m = np.asarray(geom.mass_mask, bool)          # demi-niveaux dont les deux noeuds sont interieurs valides
    rr = r[m].astype(np.float64)
    return float((rr ** 2).sum()), float(np.abs(rr).max()), int(rr.size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--nat", required=True, help="dossier pcfno3d_national du Drive")
    ap.add_argument("--donnees", required=True, help="dossier national (grille, forcage, obs, grille_reelle.json)")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--n-instants", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    NAT, G = Path(a.nat), Path(a.donnees)

    from pcfno3d.config import Config
    from pcfno3d.geometry import Geometry
    cfg = Config.load(a.config)
    geom = Geometry(cfg, device="cpu")
    cache = cfg.paths.cache_dir
    idx = pd.read_csv(cache / "index.csv").astype({"stem": str})
    bilan: dict = {}

    # ------------------------------------------------------------------ b) manifeste
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}"}
    w = subprocess.run([sys.executable, "-m", "pcfno3d", "-c", a.config, "manifeste-cache"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    v = subprocess.run([sys.executable, "-m", "pcfno3d", "-c", a.config, "manifeste-cache", "--verifier"],
                       cwd=REPO, env=env, capture_output=True, text=True)
    cal = json.loads((NAT / "calendrier.json").read_text(encoding="utf-8"))
    jours = {"train": set(cal["split"]["train_days"]), "val": set(cal["split"]["val_days"]),
             "test": set(cal["split"]["test_days"])}
    mal_range = [r.stem for r in idx.itertuples() if int(r.day) not in jours.get(r.split, set())]
    manquants, non_finis = [], []
    for s in idx.stem:
        for suf in ("_S", "_Y", "_E"):
            if not (cache / f"{s}{suf}.npy").exists():
                manquants.append(f"{s}{suf}")
    for s in idx.stem:
        S = np.load(cache / f"{s}_S.npy")
        E = np.load(cache / f"{s}_E.npy")
        Y = np.load(cache / f"{s}_Y.npy")
        val = np.asarray(geom.valid, bool)
        if not (np.isfinite(S).all() and np.isfinite(E).all() and all(np.isfinite(Y[c][val]).all() for c in range(4))):
            non_finis.append(s)
    bilan["manifeste"] = {
        "ecrit": w.stdout.strip()[-300:], "verification": v.stdout.strip()[-300:], "code_verification": v.returncode,
        "instants": int(len(idx)), "par_split": idx.split.value_counts().to_dict(),
        "attendu": {"train": 700, "val": 120, "test": 180},
        "calendrier_v2_relu": True, "instants_hors_de_leur_jeu": mal_range[:10], "fichiers_manquants": manquants[:10],
        "instants_non_finis": non_finis[:10]}
    ok_b = (v.returncode == 0 and bilan["manifeste"]["par_split"] == bilan["manifeste"]["attendu"]
            and not mal_range and not manquants and not non_finis)
    bilan["manifeste"]["verdict"] = "CONFORME" if ok_b else "ECART"
    print(json.dumps(bilan["manifeste"], indent=1, ensure_ascii=False, default=str), flush=True)

    # ------------------------------------------------------------------ a) masse (v3 : tout ; v2 : val + test)
    lignes = []
    for s, sp in zip(idx.stem, idx.split):
        ss, mx, n = residu_masse(geom, np.load(cache / f"{s}_Y.npy"))
        lignes.append({"verite": "v3", "stem": s, "split": sp, "rms": np.sqrt(ss / n), "max": mx})
    for k in (2, 3):
        f = NAT / f"cache_v2_partie_{k}.zip"
        if not f.exists():
            continue
        with zipfile.ZipFile(f) as zf:
            ix = next(n for n in zf.namelist() if n.endswith(f"index_partie_{k}.csv"))
            i2 = pd.read_csv(zf.open(ix)).astype({"stem": str})
            for s, sp in zip(i2.stem, i2.split):
                if sp not in ("val", "test"):
                    continue
                with zf.open(f"{s}_Y.npy") as fh:
                    Y = np.lib.format.read_array(fh)
                ss, mx, n = residu_masse(geom, Y)
                lignes.append({"verite": "v2", "stem": s, "split": sp, "rms": np.sqrt(ss / n), "max": mx})
    M = pd.DataFrame(lignes)
    M.to_csv(out / "residu_masse_par_instant.csv", index=False)
    bilan["masse"] = (M.groupby(["verite", "split"]).agg(instants=("rms", "size"), rms_moyen=("rms", "mean"),
                                                          rms_max=("rms", "max"), max_abs=("max", "max"))
                      .reset_index().to_dict("records"))
    print(pd.DataFrame(bilan["masse"]).to_string(index=False), flush=True)

    # ------------------------------------------------------------------ c) 10 instants de validation
    rng = np.random.default_rng(2031)
    vstems = sorted(idx[idx.split == "val"].stem)
    choix = sorted(rng.choice(vstems, size=min(a.n_instants, len(vstems)), replace=False))
    jours_c = sorted({int(s.split("_")[0]) for s in choix})
    regen = Path("/content/qc_regen") if Path("/content").exists() else out / "qc_regen"
    jeu_tmp = regen / "jeu"
    jeu_tmp.mkdir(parents=True, exist_ok=True)
    (jeu_tmp / "calendrier.json").write_text((NAT / "calendrier.json").read_text(encoding="utf-8"), encoding="utf-8")
    info = json.loads((G / "grille_reelle.json").read_text(encoding="utf-8"))
    z = info["zone"]
    cmd = [sys.executable, "-u", str(REPO / "scripts" / "generate_aneo_3d.py"),
           "--input-grid", str(G / "aneo_grid_points_surface.csv"), "--output-dir", str(jeu_tmp),
           "--cache-dir", str(regen / "cache"), "--forcage", str(G / "forcage_era5.csv"),
           "--nx", str(z["nx"]), "--ny", str(z["ny"]), "--dx", str(z["dx_m"]),
           "--niveaux", ",".join(f"{x:g}" for x in geom.z_levels), "--f-coriolis", str(info["f_coriolis"]),
           "--perturbations", "0", "--test", "2022-01-01:2022-03-31,2022-07-01:2022-08-31",
           "--val", "2021-01-01:2021-02-28,2021-07-01:2021-07-31", "--marge-jours", "7",
           "--instants", "1000", "--heures-par-jour", "4", "--workers", str(max(1, (os.cpu_count() or 2) - 1)),
           "--jeux", "val", "--days", ",".join(str(j) for j in jours_c)]
    env_v3 = {**os.environ, "ANEO_GEN_AMORT_PASSE_HAUT": "1", "ANEO_GEN_BRISE": "0.5", "ANEO_GEN_RELIEF": "1",
              "ANEO_GEN_DETAIL": "0.5"}
    r = subprocess.run(cmd, cwd=REPO, env=env_v3, capture_output=True, text=True)
    (out / "regeneration_10_instants.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    ecarts = []
    for s in choix:
        Yc = np.load(cache / f"{s}_Y.npy").astype(np.float32)
        f2 = regen / "cache" / f"{s}_Y.npy"
        Yr = np.load(f2).astype(np.float32) if f2.exists() else None
        ecarts.append({"stem": s, "regenere": Yr is not None,
                       "ecart_max_Y": float(np.abs(Yc - Yr).max()) if Yr is not None else None})
    bilan["regeneration_10"] = {"code_generateur": r.returncode, "instants": ecarts,
                                "identiques": bool(all(e["ecart_max_Y"] == 0.0 for e in ecarts))}
    print(json.dumps(bilan["regeneration_10"], indent=1), flush=True)

    # chiffres du rapport sur ces 10 instants : cache contre chemin de la phase 2 (compute_fields)
    import calibrer_generateur_stations as C
    C._env(info, 0.0)
    import generate_aneo_3d as GA
    from pcfno3d.stations_reelles import lire_observations, obs_a
    df = GA.load_or_build_grid(G / "aneo_grid_points_surface.csv", np.random.default_rng(0))
    S0 = GA.build_static(df)
    P0 = GA.precompute_static_derived(S0)
    F = GA.lire_forcage(str(G / "forcage_era5.csv"))
    CH = GA.lire_champs(str(G / "forcage_era5.csv"))
    st_codes = {s["code"]: (int(s["iy"]), int(s["ix"])) for s in info["stations"].values() if s.get("dans_la_zone")}
    obs = lire_observations(G / "obs", verbose=False, codes=list(st_codes))
    k = GA.STRIDE
    lat_v, lon_v = S0["lat"][::k, ::k][:GA.NY_V, :GA.NX_V], S0["lon"][::k, ::k][:GA.NY_V, :GA.NX_V]
    zt_v = S0["zt"][::k, ::k][:GA.NY_V, :GA.NX_V]
    Z = np.asarray(GA.Z_LEVELS, float)

    def a_hauteur(vol3, hh):
        zz = zt_v + hh
        i = np.clip(np.searchsorted(Z, zz) - 1, 0, len(Z) - 2)
        t = np.clip((zz - Z[i]) / (Z[i + 1] - Z[i]), 0, 1)
        lo = np.take_along_axis(vol3, i[None], 0)[0]
        hi = np.take_along_axis(vol3, (i + 1)[None], 0)[0]
        ok = (zz >= Z[0]) & (zz <= Z[-1]) & (Z[i] - zt_v >= GA.MIN_ZREL_NODE)
        return np.where(ok, (1 - t) * lo + t * hi, np.nan)

    GA.AMORT_PASSE_HAUT, GA.A_ROT, GA.A_DIV = True, 0.0, 0.0
    GA.BRISE_FACTEUR, GA.RELIEF_FACTEUR, GA.DETAIL_BETA = 0.5, 1.0, 0.5
    acc = {src: {f"{c}{int(h)}": np.zeros(2) for c in "uv" for h in HAUTEURS} for src in ("cache", "phase2")}
    paires = []
    stamp = dict(zip(idx.stem, idx.timestamp))
    for s in choix:
        ts = pd.Timestamp(stamp[s])
        jour, hh = ts.normalize(), ts.hour + ts.minute / 60
        seed = 2026 * 1000 + int((jour - pd.Timestamp("2021-01-01")).days) + 1
        st = GA.DayStateERA5(0, jour, 0, S0, seed, F, CH)
        surf, vol, _ = GA.compute_fields(st, S0, P0, hh * 3600.0, None)
        champ = st.champ_cible(hh * 3600.0)
        Yc = np.load(cache / f"{s}_Y.npy").astype(np.float32)
        Sc = np.load(cache / f"{s}_S.npy")
        val = np.asarray(geom.valid, bool)
        for c, cn in ((0, "u"), (1, "v")):
            for h in HAUTEURS:
                E = GA._era5_au_sol(CH, champ[c], lat_v, lon_v, np.full(lat_v.shape, h))
                for src, V in (("cache", np.where(val, Yc[c], np.nan)), ("phase2", vol["uv"[c]])):
                    d = a_hauteur(V, h) - E
                    m = np.isfinite(d)
                    acc[src][f"{cn}{int(h)}"] += [(d[m] ** 2).sum(), m.sum()]
        for code, (uo, vo) in obs_a(obs, ts).items():
            iy, ix = st_codes[code]
            jy, jx = (iy // k) * k, (ix // k) * k
            la, lo_ = np.array([[S0["lat"][jy, jx]]]), np.array([[S0["lon"][jy, jx]]])
            ue, ve = (float(GA._era5_au_sol(CH, champ[c], la, lo_, np.full((1, 1), 10.0))[0, 0]) for c in (0, 1))
            ug2, vg2 = (float(surf[x][jy:jy + k, jx:jx + k].mean()) for x in ("u", "v"))
            paires.append({"stem": s, "code": code, "uo": uo, "vo": vo, "ue": ue, "ve": ve,
                           "u_cache": float(Sc[0, iy // k, ix // k]), "v_cache": float(Sc[1, iy // k, ix // k]),
                           "u_phase2": ug2, "v_phase2": vg2})
        print("instant", s, flush=True)
    P = pd.DataFrame(paires)
    P.to_csv(out / "paires_aeroports_10.csv", index=False)

    def rm(pu, pv):
        return float(np.sqrt(((P[pu] - P.uo) ** 2 + (P[pv] - P.vo) ** 2).mean()))
    bilan["chiffres_10_instants"] = {
        "instants": list(map(str, choix)),
        "ecart_verite_era5": {src: {key: float(np.sqrt(s_[0] / s_[1])) for key, s_ in d.items()} for src, d in acc.items()},
        "aeroports": {"paires": int(len(P)), "rmse_verite_cache_S_bruite": rm("u_cache", "v_cache"),
                      "rmse_verite_phase2": rm("u_phase2", "v_phase2"), "rmse_era5": rm("ue", "ve")},
        "rapport_acceptation_20_instants": {"v300": 0.434, "v800": 0.320, "u300": 0.370, "u800": 0.274,
                                            "aeroports_verite": 2.925, "aeroports_era5": 2.940},
        "note": "le _S du cache contient le bruit de mesure simule du generateur (0,02-0,1 m/s), le chemin phase 2 "
                "non ; graine et secteur du chemin phase 2 (DayStateERA5(0, jour, 0, ..., 2026*1000+j)) different "
                "de ceux du generateur : les champs aleatoires du jour peuvent differer"}
    print(json.dumps(bilan["chiffres_10_instants"], indent=1, ensure_ascii=False), flush=True)
    (out / "bilan_controle_v3.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
