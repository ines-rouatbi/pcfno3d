"""Diagnostic du modele reel Grand Tunis sur les instants de VALIDATION (CPU), avant toute relance.

    python scripts/diag_grand_tunis.py -c grand_tunis_colab --donnees /content/donnees/grand_tunis \
        --ckpt <run_dir>/best.pt --sortie <dossier>

1. R2 et RMSE par niveau (u, v, w ; noeuds valides, sommes cumulees sur les instants comme
   train.validate) de trois predicteurs sur les memes instants :
     era5     ERA5 interpole sur la grille (bilineaire, meme hauteur au-dessus du sol que la
              grande echelle du generateur), w = 0 ;
     moyenne  vent moyen du domaine par niveau et par instant (ORACLE : lu dans la cible) ;
     modele   le checkpoint (EMA), entree de production `era5_stations`, graine de validation.
2. Alignement entree ERA5 / cible : heure UTC (stem, index, champs ERA5 a t et t +- 3 h),
   orientation nord/sud et est/ouest, ordre u/v, unites, normalisation du canal ERA5.
3. Part de la variance de la cible due aux perturbations aleatoires du generateur : chaque
   instant est regenere sans perturbation (A_ROT = A_DIV = 0, meme forcage, meme relief) ;
   quelques instants sont aussi regeneres avec la graine d'origine pour verifier que la
   regeneration reproduit le cache.

Rien n'est ecrit hors de --sortie ; le test n'est pas lu.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GEN = None          # module du generateur, charge apres avoir fixe ses variables d'environnement


def charger_generateur(info: dict, perturb: float):
    os.environ["ANEO_GEN_NX"] = str(info["zone"]["nx"])
    os.environ["ANEO_GEN_NY"] = str(info["zone"]["ny"])
    os.environ["ANEO_GEN_F"] = str(info["f_coriolis"])
    os.environ["ANEO_GEN_PERTURB"] = str(perturb)
    spec = importlib.util.spec_from_file_location("gen_aneo", REPO / "scripts" / "generate_aneo_3d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------- regeneration (processus)
_W = {}


def _init(static_path, forcage_path, info, perturb):
    global GEN
    GEN = charger_generateur(info, perturb)
    GEN._worker_init(static_path, forcage_path)
    _W["a_rot"], _W["a_div"] = GEN.A_ROT, GEN.A_DIV


def _regenerer(tache):
    """(stem, jour, date iso, t_sec, mode, sortie) -> chemin du Y (u, v, w) float32 regenere."""
    stem, jour, date, t_sec, mode, sortie = tache
    if mode == "zero":
        GEN.A_ROT, GEN.A_DIV = 0.0, 0.0
    else:
        GEN.A_ROT, GEN.A_DIV = _W["a_rot"], _W["a_div"]
    seed = 2026 * 1000 + jour                   # main() : args.seed * 1000 + day_index
    st = GEN.DayStateERA5(jour, dt.date.fromisoformat(date), 0, GEN._STATIC, seed,
                          GEN._FORCAGE, GEN._CHAMPS)
    _, vol, _ = GEN.compute_fields(st, GEN._STATIC, GEN._PRE, float(t_sec), None)
    nv = GEN._STATIC["node_valid"]
    Y = np.stack([np.where(nv, np.nan_to_num(vol[c]), 0.0) for c in ("u", "v", "w")]).astype(np.float32)
    out = Path(sortie) / f"{stem}_{mode}.npy"
    np.save(out, Y)
    return str(out)


# ---------------------------------------------------------------------- metriques
class Cumul:
    """Sommes par (composante, niveau) : sse, n, somme et somme des carres de la cible."""

    def __init__(self, nz):
        self.a = np.zeros((4, 3, nz))

    def ajouter(self, pred, y, valid):
        for c in range(3):
            for k in range(y.shape[1]):
                m = valid[k]
                if not m.any():
                    continue
                t = y[c, k][m].astype(np.float64)
                p = pred[c, k][m].astype(np.float64)
                self.a[:, c, k] += [((p - t) ** 2).sum(), t.size, t.sum(), (t ** 2).sum()]

    def table(self):
        sse, n, s, s2 = self.a
        with np.errstate(invalid="ignore", divide="ignore"):
            var = s2 - s ** 2 / np.maximum(n, 1)
            r2 = 1.0 - sse / var
            rmse = np.sqrt(sse / np.maximum(n, 1))
            g = self.a.sum(axis=2)
            r2_g = 1.0 - g[0] / (g[3] - g[2] ** 2 / np.maximum(g[1], 1))
            rmse_g = np.sqrt(g[0] / np.maximum(g[1], 1))
        return r2, rmse, r2_g, rmse_g


def corr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="grand_tunis_colab")
    ap.add_argument("--donnees", required=True, help="dossier grand_tunis (grille, forcage)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--perturbations", type=float, default=0.4, help="facteur de la generation")
    ap.add_argument("--n-replique", type=int, default=4, help="instants regeneres avec la graine d'origine")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--max-instants", type=int, default=None, help="tests rapides")
    args = ap.parse_args()

    sys.path[:0] = [str(REPO / "src")]
    import torch
    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.dataset import ANEO3D, assemble_input, cellules_stations, make_splits
    from pcfno3d.geometry import Geometry
    from pcfno3d.model import build_model, model_cfg_from_checkpoint
    from pcfno3d.normalize import Normalizer

    t_debut = time.time()
    sortie = Path(args.sortie)
    (sortie / "regen").mkdir(parents=True, exist_ok=True)
    G = Path(args.donnees)
    info = json.loads((G / "grille_reelle.json").read_text(encoding="utf-8"))
    cfg = Config.load(args.config)
    geom = Geometry(cfg, device="cpu")
    index = load_index(cfg)
    splits = make_splits(cfg, index)
    va_s, va_h, va_d = splits.val
    sel = splits.val_selection[: args.max_instants] if args.max_instants else splits.val_selection
    stems = list(va_s[sel])
    ts_of = dict(zip(index["stem"], pd.to_datetime(index["timestamp"])))
    jour_of = dict(zip(index["stem"], index["day"]))
    print(f"{len(stems)} instants de validation (selection de l'entrainement, graine 1000)",
          flush=True)
    cal = json.loads((cfg.paths.data_dir / "calendrier.json").read_text(encoding="utf-8"))
    date_of = {d["day_index"]: d["date"] for d in cal["days"]}
    valid = geom.valid
    nz = geom.nz
    rapport = {"instants": stems, "niveaux_m": geom.z_levels.tolist(),
               "couverture_niveau": geom.cover.tolist()}

    # ------------------------------------------------------------------ generateur (statique)
    global GEN
    GEN = charger_generateur(info, args.perturbations)
    static_path = cfg.paths.data_dir / "static_cache.npz"
    if not static_path.exists():
        df = GEN.load_or_build_grid(G / "aneo_grid_points_surface.csv", np.random.default_rng(0))
        np.savez(sortie / "static_cache.npz", **GEN.build_static(df))
        static_path = sortie / "static_cache.npz"
    st_np = np.load(static_path)
    sv = (slice(None, None, GEN.STRIDE), slice(None, None, GEN.STRIDE))
    lat_v, lon_v, zt_v = st_np["lat"][sv], st_np["lon"][sv], st_np["zt"][sv]
    nv_gen = st_np["node_valid"]
    champs = GEN.lire_champs(G / "forcage_era5.csv")
    print(f"ERA5 : lat {champs['lat']}, lon {champs['lon']}, hauteurs {champs['hauteurs_sol']}", flush=True)
    rapport["masque_valide_identique_generateur"] = bool((nv_gen == valid).all())

    def champ_a(t):
        """u, v ERA5 par maille a l'instant t (interpolation lineaire en temps)."""
        s = int(pd.Timestamp(t).value // 10 ** 9)
        tt = champs["temps"]
        i = int(np.clip(np.searchsorted(tt, s), 1, len(tt) - 1))
        a = float(np.clip((s - tt[i - 1]) / max(tt[i] - tt[i - 1], 1), 0, 1))
        return [(1 - a) * champs[c][i - 1] + a * champs[c][i] for c in ("u", "v")]

    def era5_niveaux(t, flip=None):
        cu, cv = champ_a(t)
        if flip == "ns":
            cu, cv = cu[:, ::-1, :], cv[:, ::-1, :]
        elif flip == "eo":
            cu, cv = cu[:, :, ::-1], cv[:, :, ::-1]
        out = np.zeros((3, nz) + lat_v.shape, np.float32)
        for k, z in enumerate(geom.z_levels):
            h = np.maximum(z - zt_v, GEN.MIN_ZREL_NODE)
            out[0, k] = GEN._era5_au_sol(champs, cu, lat_v, lon_v, h)
            out[1, k] = GEN._era5_au_sol(champs, cv, lat_v, lon_v, h)
        return out

    def era5_10m(t):
        cu, cv = champ_a(t)
        dix = np.full(lat_v.shape, 10.0)
        return np.stack([GEN._era5_au_sol(champs, c, lat_v, lon_v, dix) for c in (cu, cv)])

    # ------------------------------------------------------------------ 3. regeneration
    t0 = time.time()
    taches = []
    for i, stem in enumerate(stems):
        ts = ts_of[stem]
        jour = int(jour_of[stem])
        t_sec = (ts - pd.Timestamp(date_of[jour])).total_seconds()
        taches.append((stem, jour, date_of[jour], t_sec, "zero", str(sortie / "regen")))
        if i < args.n_replique:
            taches.append((stem, jour, date_of[jour], t_sec, "graine", str(sortie / "regen")))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(str(static_path), str(G / "forcage_era5.csv"), info,
                                       args.perturbations)) as ex:
        for k, _ in enumerate(ex.map(_regenerer, taches), 1):
            if k % 8 == 0 or k == len(taches):
                print(f"regeneration {k}/{len(taches)} ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------------ modele
    torch.set_num_threads(os.cpu_count() or 4)
    ck = torch.load(str(args.ckpt), map_location="cpu", weights_only=False, mmap=True)
    saved = model_cfg_from_checkpoint(ck)
    if saved is not None:
        cfg.model = saved
    with torch.device("meta"):
        model = build_model(cfg, geom.c_in)
    model.load_state_dict(ck["model"], strict=True, assign=True)
    model.eval()
    norm = Normalizer.from_file(cfg.paths.norm_stats, geom, cfg.loss.w_level_max)
    rapport["checkpoint"] = {"chemin": str(args.ckpt), "epoque": ck.get("epoch"),
                             "val_score": ck.get("val_score")}
    print(f"checkpoint : epoque {ck.get('epoch')}, val_score {ck.get('val_score')}", flush=True)
    ds = ANEO3D(va_s[sel], va_h[sel], va_d[sel], geom, train=False, seed=1000)

    cum = {k: Cumul(nz) for k in ("era5", "moyenne", "modele", "sans_perturbation")}
    cum_spatial = {k: Cumul(nz) for k in ("modele", "sans_perturbation", "era5")}   # anomalies
    pert = np.zeros((2, 3, nz))          # sse(Y - Y0), sse(Y - moyenne instant)
    al = {"heure": [], "flip": [], "uv": [], "unites": [], "norm": [], "stations": [], "geo": [],
          "replique": []}
    E_acc = []
    stations = cellules_stations(geom)
    t0 = time.time()
    for i, stem in enumerate(stems):
        ts = ts_of[stem]
        S_cache = np.load(cfg.paths.cache_dir / f"{stem}_S.npy")
        E = np.load(cfg.paths.cache_dir / f"{stem}_E.npy").astype(np.float32)
        Y = np.load(cfg.paths.cache_dir / f"{stem}_Y.npy").astype(np.float32)[:3]
        s_raw, sensor, tfeat, _, _ = ds[i]

        # --- heure : stem, index, et ERA5 a t / t +- 3 h
        hh = ts.strftime("%H%M%S")
        e_t = {h: float(np.sqrt(np.mean((era5_10m(ts + pd.Timedelta(hours=h)) - E) ** 2)))
               for h in (-3, 0, 3)}
        al["heure"].append({"stem": stem, "timestamp": str(ts), "stem_hhmmss_ok": stem.endswith(hh),
                            "hour_index": float(va_h[sel][i]), "rmse_E_vs_era5_t": e_t})

        # --- 1. predicteurs
        p_era5 = era5_niveaux(ts)
        p_moy = np.zeros_like(Y)
        for c in range(3):
            for k in range(nz):
                if valid[k].any():
                    p_moy[c, k] = Y[c, k][valid[k]].mean()
        with torch.inference_mode():
            x = assemble_input(s_raw[None], sensor[None], tfeat[None], geom, norm)
            p_mod = torch.stack(norm.denorm(model(x).float()), dim=1)[0].numpy()
        Y0 = np.load(sortie / "regen" / f"{stem}_zero.npy")
        for nom, p in (("era5", p_era5), ("moyenne", p_moy), ("modele", p_mod),
                       ("sans_perturbation", Y0)):
            cum[nom].ajouter(p, Y, valid)
        # anomalies spatiales (moyenne de l'instant retiree des deux cotes)
        for nom, p in (("modele", p_mod), ("sans_perturbation", Y0), ("era5", p_era5)):
            pa = p - np.where(valid, p, 0).sum(axis=(2, 3), keepdims=True) / np.maximum(valid.sum(axis=(1, 2))[None, :, None, None], 1)
            cum_spatial[nom].ajouter(pa, Y - p_moy, valid)
        for c in range(3):
            for k in range(nz):
                m = valid[k]
                if m.any():
                    pert[0, c, k] += float(((Y[c, k][m] - Y0[c, k][m]) ** 2).sum())
                    pert[1, c, k] += float(((Y[c, k][m] - p_moy[c, k][m]) ** 2).sum())
        rep = sortie / "regen" / f"{stem}_graine.npy"
        if rep.exists():
            Yr = np.load(rep)
            al["replique"].append({"stem": stem, "max_abs": float(np.abs((Yr - Y)[:, valid]).max()),
                                   "rmse": float(np.sqrt(np.mean((Yr - Y)[:, valid] ** 2)))})

        # --- orientation : grande echelle de la cible (lissee 12 km) contre ERA5, retourne ou non
        sig = 12000.0 / geom.dx
        flips = {}
        for nom, f in (("normal", None), ("nord_sud", "ns"), ("est_ouest", "eo")):
            pe = p_era5 if f is None else era5_niveaux(ts, f)
            err = []
            for k in range(nz):
                if geom.cover[k] < 0.5:
                    continue
                for c in range(2):
                    lisse = GEN._passe_bas(np.where(valid[k], Y[c, k], Y[c, k][valid[k]].mean()),
                                           sig, facteur=4)
                    err.append(np.mean((lisse - pe[c, k])[valid[k]] ** 2))
            flips[nom] = float(np.sqrt(np.mean(err)))
        al["flip"].append(flips)

        # --- u/v, unites : E contre la surface de la cible (S, 10 m) aux capteurs du cache
        m7 = S_cache[7] > 0.5
        al["uv"].append({"E0_S0": corr(E[0][m7], S_cache[0][m7]), "E0_S1": corr(E[0][m7], S_cache[1][m7]),
                         "E1_S1": corr(E[1][m7], S_cache[1][m7]), "E1_S0": corr(E[1][m7], S_cache[0][m7]),
                         "E0_S0_anom": corr(E[0][m7] - E[0][m7].mean(), S_cache[0][m7] - S_cache[0][m7].mean()),
                         "E0_S0_anom_flipNS": corr(E[0][::-1][m7] - E[0][m7].mean(),
                                                   S_cache[0][m7] - S_cache[0][m7].mean())})
        vit_E = np.hypot(E[0], E[1])[m7].mean()
        vit_S = np.hypot(S_cache[0], S_cache[1])[m7].mean()
        k1 = int(np.argmax(geom.cover > 0.5))
        vit_Y = np.hypot(Y[0, k1], Y[1, k1])[valid[k1]].mean()
        al["unites"].append({"vitesse_E10m": float(vit_E), "vitesse_S10m": float(vit_S),
                             f"vitesse_Y_{int(geom.z_levels[k1])}m": float(vit_Y)})
        sn = sensor.numpy() > 0.5
        mu = np.array(norm.stats["surf_mu"][:2])[:, None, None]
        sd = np.array(norm.stats["surf_sd"][:2])[:, None, None]
        z = (s_raw.numpy()[:2] - mu) / sd
        pseudo = sn.copy()
        for iy, ix in stations:
            pseudo[iy, ix] = False
        al["norm"].append({"n_capteurs": int(sn.sum()), "u_norm_moy": float(z[0][sn].mean()),
                           "u_norm_sd": float(z[0][sn].std()), "v_norm_moy": float(z[1][sn].mean()),
                           "v_norm_sd": float(z[1][sn].std()),
                           "entree_egale_E_aux_pseudo_capteurs": bool(np.allclose(
                               s_raw.numpy()[:2][:, pseudo], E[:, pseudo], atol=1e-3)),
                           "entree_nulle_hors_capteurs": bool(
                               (assemble_input(s_raw[None], sensor[None], tfeat[None], geom, norm)
                                [0, :7, 0].numpy()[:, ~sn] == 0).all())})
        E_acc.append(E[:, sn])
        al["stations"].append({f"{iy},{ix}": {"S": S_cache[:2, iy, ix].round(2).tolist(),
                                              "E": E[:, iy, ix].round(2).tolist(),
                                              "Y_premier": [round(float(Y[c, int(np.argmax(valid[:, iy, ix])), iy, ix]), 2)
                                                            for c in range(2)],
                                              "entree": s_raw.numpy()[:2, iy, ix].round(2).tolist(),
                                              "capteur": bool(sn[iy, ix])}
                               for iy, ix in stations})
        row = index.set_index("stem").loc[stem]
        al["geo"].append({"ug_entree": float(tfeat[4]) * 10, "vg_entree": float(tfeat[5]) * 10,
                          "ug_era5": float(row.get("era5_ug", np.nan)),
                          "vg_era5": float(row.get("era5_vg", np.nan))})
        if (i + 1) % 4 == 0 or i + 1 == len(stems):
            print(f"instant {i + 1}/{len(stems)} ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------------ synthese
    lignes = []
    for nom, cu in cum.items():
        r2, rmse, r2g, rmseg = cu.table()
        for c, comp in enumerate("uvw"):
            for k in range(nz):
                lignes.append({"predicteur": nom, "composante": comp, "niveau_m": int(geom.z_levels[k]),
                               "couverture": round(float(geom.cover[k]), 3),
                               "R2": float(r2[c, k]), "RMSE": float(rmse[c, k])})
            lignes.append({"predicteur": nom, "composante": comp, "niveau_m": "global",
                           "couverture": 1.0, "R2": float(r2g[c]), "RMSE": float(rmseg[c])})
    for nom, cu in cum_spatial.items():
        r2, rmse, r2g, rmseg = cu.table()
        for c, comp in enumerate("uvw"):
            for k in range(nz):
                lignes.append({"predicteur": f"{nom}_anomalie_spatiale", "composante": comp,
                               "niveau_m": int(geom.z_levels[k]), "couverture": round(float(geom.cover[k]), 3),
                               "R2": float(r2[c, k]), "RMSE": float(rmse[c, k])})
            lignes.append({"predicteur": f"{nom}_anomalie_spatiale", "composante": comp,
                           "niveau_m": "global", "couverture": 1.0, "R2": float(r2g[c]),
                           "RMSE": float(rmseg[c])})
    tab = pd.DataFrame(lignes)
    tab.to_csv(sortie / "r2_rmse_par_niveau.csv", index=False)

    var_tot = cum["moyenne"].a[3] - cum["moyenne"].a[2] ** 2 / np.maximum(cum["moyenne"].a[1], 1)
    part = {comp: {"part_variance_totale": (pert[0, c].sum() / var_tot[c].sum()),
                   "part_variance_spatiale": (pert[0, c].sum() / pert[1, c].sum()),
                   "par_niveau_totale": (pert[0, c] / var_tot[c]).tolist(),
                   "par_niveau_spatiale": (pert[0, c] / pert[1, c]).tolist()}
            for c, comp in enumerate("uvw")}
    E_all = np.concatenate(E_acc, axis=1)
    rapport.update({
        "perturbations": part,
        "alignement": {
            "heure_stem_ok": all(h["stem_hhmmss_ok"] for h in al["heure"]),
            "rmse_E_vs_era5": {str(h): float(np.mean([a["rmse_E_vs_era5_t"][h] for a in al["heure"]]))
                               for h in (-3, 0, 3)},
            "heures_index_vs_timestamp_ok": all(abs(a["hour_index"] - pd.Timestamp(a["timestamp"]).hour) < 1e-6
                                                for a in al["heure"]),
            "orientation_rmse_grande_echelle": {k: float(np.mean([f[k] for f in al["flip"]]))
                                                for k in al["flip"][0]},
            "uv_correlations": {k: float(np.nanmean([u[k] for u in al["uv"]])) for k in al["uv"][0]},
            "unites": {k: float(np.mean([u[k] for u in al["unites"]])) for k in al["unites"][0]},
            "normalisation": {
                "surf_mu_uv": norm.stats["surf_mu"][:2], "surf_sd_uv": norm.stats["surf_sd"][:2],
                "E_capteurs_moy": E_all.mean(axis=1).tolist(), "E_capteurs_sd": E_all.std(axis=1).tolist(),
                **{k: float(np.mean([u[k] for u in al["norm"]])) for k in al["norm"][0]}},
            "geostrophique_entree_vs_era5": {
                "corr_ug": corr([g["ug_entree"] for g in al["geo"]], [g["ug_era5"] for g in al["geo"]]),
                "corr_vg": corr([g["vg_entree"] for g in al["geo"]], [g["vg_era5"] for g in al["geo"]]),
                "rmse_ug": float(np.sqrt(np.nanmean([(g["ug_entree"] - g["ug_era5"]) ** 2 for g in al["geo"]]))),
                "rmse_vg": float(np.sqrt(np.nanmean([(g["vg_entree"] - g["vg_era5"]) ** 2 for g in al["geo"]])))},
            "replique_graine_origine": al["replique"],
            "stations_premier_instant": al["stations"][0],
        },
        "duree_s": time.time() - t_debut,
    })
    (sortie / "diagnostic.json").write_text(json.dumps(rapport, indent=1, default=float), encoding="utf-8")
    (sortie / "alignement_par_instant.json").write_text(json.dumps(al, indent=1, default=float),
                                                       encoding="utf-8")

    pd.set_option("display.width", 200)
    piv = tab[tab["predicteur"].isin(["era5", "moyenne", "modele", "sans_perturbation"])].pivot_table(
        index=["composante", "niveau_m"], columns="predicteur", values=["R2", "RMSE"], aggfunc="first")
    print(piv.round(3).to_string())
    print(json.dumps({k: v for k, v in rapport.items() if k in ("perturbations", "alignement", "checkpoint")},
                     indent=1, default=float)[:12000])
    print(f"ecrit : {sortie}  ({time.time() - t_debut:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
