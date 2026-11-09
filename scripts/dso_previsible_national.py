"""Modele national : DSO1 sur la verite PREVISIBLE (a titre d'information, aucun entrainement).

La verite generee contient une composante aleatoire (perturbations rotationnelle et divergente de
petite echelle du generateur, A_ROT / A_DIV) qu'aucune entree ne permet de prevoir. Quelques
instants de TEST sont regeneres SANS ces perturbations (meme forcage ERA5, meme relief, meme
graine 2026 x 1000 + jour, comme scripts/diag_grand_tunis.py en mode « zero ») ; deux sont aussi
regeneres AVEC les perturbations d'origine pour verifier que la regeneration reproduit le cache.
Sur ces memes instants et les memes entrees que DSO1 (graine 2000), on compare le modele a la
verite du cache et a la verite previsible : RMSE de u et v par niveau et par tranche de hauteur
au-dessus du sol (0-100, 100-300, 300-600, 600-1000 m ; > 1000 m information). Part de l'ecart sur
v due aux perturbations = 1 - (RMSE v contre previsible / RMSE v contre cache)^2, par niveau.

    python scripts/dso_previsible_national.py -c national_eval_v2 \
        --donnees /content/donnees/national [--instants 12] [--workers 4]
Sortie : <run_dir>/dso_previsible/ (resume.json, par_niveau.csv, par_tranche.csv, .md)
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
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
NIVEAUX = "25,50,75,100,150,200,300,400,500,650,800,1000,1250,1500,2000,2500"
BORNES = (0.0, 100.0, 300.0, 600.0, 1000.0)
STATUT = "information : verite sans perturbations aleatoires, pas un verdict DSO"

GEN = None
_W: dict = {}


def _env(info: dict, perturb: float) -> None:
    z = info["zone"]
    os.environ.update({"ANEO_GEN_DX": str(float(z["dx_m"])), "ANEO_GEN_NIVEAUX": NIVEAUX,
                       "ANEO_GEN_NX": str(z["nx"]), "ANEO_GEN_NY": str(z["ny"]),
                       "ANEO_GEN_F": str(info["f_coriolis"]), "ANEO_GEN_PERTURB": str(perturb)})


def _init(static_path: str, forcage_path: str, info: dict, perturb: float) -> None:
    global GEN
    _env(info, perturb)
    import generate_aneo_3d as G
    GEN = G
    G._worker_init(static_path, forcage_path)
    _W["a_rot"], _W["a_div"] = G.A_ROT, G.A_DIV


def _regenerer(tache) -> str:
    import datetime as dt
    stem, jour, date, t_sec, mode, sortie = tache
    GEN.A_ROT, GEN.A_DIV = (0.0, 0.0) if mode == "zero" else (_W["a_rot"], _W["a_div"])
    st = GEN.DayStateERA5(jour, dt.date.fromisoformat(date), 0, GEN._STATIC, 2026 * 1000 + jour,
                          GEN._FORCAGE, GEN._CHAMPS)
    _, vol, _ = GEN.compute_fields(st, GEN._STATIC, GEN._PRE, float(t_sec), None)
    nv = GEN._STATIC["node_valid"]
    Y = np.stack([np.where(nv, np.nan_to_num(vol[c]), 0.0) for c in ("u", "v")]).astype(np.float32)
    out = Path(sortie) / f"{stem}_{mode}.npy"
    np.save(out, Y)
    return str(out)


def main() -> int:
    import torch

    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.dataset import eval_loader, is_synoptic
    from pcfno3d.evaluate import Evaluator

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--donnees", required=True, help="dossier national (grille, forcage)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--instants", type=int, default=12)
    ap.add_argument("--n-replique", type=int, default=2)
    ap.add_argument("--perturbations", type=float, default=0.4, help="facteur de la generation v2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = Config.load(a.config)
    ev = Evaluator(cfg, checkpoint=a.checkpoint, device=a.device, verbose=False)
    geom, splits = ev.geom, ev.splits
    te_s = splits.test[0]
    syn = np.nonzero([is_synoptic(splits.stamp_of[s]) for s in te_s])[0]
    idx = syn[np.unique(np.linspace(0, len(syn) - 1, min(a.instants, len(syn))).astype(int))]
    loader, stems = eval_loader(cfg, geom, splits, idx, 2000)
    stems = [str(s) for s in stems]

    G = Path(a.donnees)
    info = json.loads((G / "grille_reelle.json").read_text(encoding="utf-8"))
    out = cfg.paths.run_dir / "dso_previsible"
    out.mkdir(parents=True, exist_ok=True)
    regen = Path("/content/regen_national") if Path("/content").exists() else out / "regen"
    regen.mkdir(parents=True, exist_ok=True)
    static_path = regen / "static_cache.npz"
    if not static_path.exists():
        _env(info, a.perturbations)
        import generate_aneo_3d as Gm
        df = Gm.load_or_build_grid(G / "aneo_grid_points_surface.csv", np.random.default_rng(0))
        np.savez(static_path, **Gm.build_static(df))
    index = load_index(cfg)
    ts_of = dict(zip(index["stem"].astype(str), pd.to_datetime(index["timestamp"])))
    jour_of = dict(zip(index["stem"].astype(str), index["day"]))
    cal = json.loads((cfg.paths.data_dir / "calendrier.json").read_text(encoding="utf-8"))
    date_of = {d["day_index"]: d["date"] for d in cal["days"]}
    taches = []
    for i, stem in enumerate(stems):
        jour = int(jour_of[stem])
        t_sec = (ts_of[stem] - pd.Timestamp(date_of[jour])).total_seconds()
        for mode in ("zero",) + (("graine",) if i < a.n_replique else ()):
            if not (regen / f"{stem}_{mode}.npy").exists():
                taches.append((stem, jour, date_of[jour], t_sec, mode, str(regen)))
    t0 = time.time()
    if taches:
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                                 initargs=(str(static_path), str(G / "forcage_era5.csv"), info,
                                           a.perturbations)) as ex:
            for k, _ in enumerate(ex.map(_regenerer, taches), 1):
                print(f"regeneration {k}/{len(taches)} ({time.time() - t0:.0f} s)", flush=True)

    nz = geom.nz
    valid = np.asarray(geom.valid, bool)
    agl = np.asarray(geom.z_levels, float)[:, None, None] - np.asarray(geom.zt, float)[None]
    tranches = [(f"{lo:g}-{hi:g} m", valid & (agl >= lo) & (agl < hi)) for lo, hi in zip(BORNES[:-1], BORNES[1:])]
    tranches.append((f"> {BORNES[-1]:g} m (info)", valid & (agl >= BORNES[-1])))
    acc_n = {r: np.zeros((nz, 2)) for r in ("cache", "previsible")}       # sse u, v par niveau
    acc_t = {r: np.zeros((len(tranches), 2)) for r in ("cache", "previsible")}
    cnt_n, cnt_t = np.zeros(nz), np.zeros(len(tranches))
    replique = []
    with torch.no_grad():
        for s_raw, sensor, tfeat, y, j in loader:
            s_raw, sensor, tfeat, y = (t.to(geom.device) for t in (s_raw, sensor, tfeat, y))
            stem = stems[int(j)]
            u, v, _ = ev.predire(s_raw, sensor, tfeat, y, int(j), stem)
            p = np.stack([u[0].float().cpu().numpy(), v[0].float().cpu().numpy()])
            yc = y[0, :2].float().cpu().numpy()
            y0 = np.load(regen / f"{stem}_zero.npy")
            for nom, ref in (("cache", yc), ("previsible", y0)):
                e2 = (p - ref) ** 2
                for k in range(nz):
                    m = valid[k]
                    acc_n[nom][k] += e2[:, k][:, m].sum(axis=1)
                for b, (_, mb) in enumerate(tranches):
                    acc_t[nom][b] += e2[:, mb].sum(axis=1)
            for k in range(nz):
                cnt_n[k] += valid[k].sum()
            for b, (_, mb) in enumerate(tranches):
                cnt_t[b] += mb.sum()
            rep = regen / f"{stem}_graine.npy"
            if rep.exists():
                replique.append({"stem": stem, "max_abs_m_s": float(np.abs(np.load(rep)[:, valid] - yc[:, valid]).max())})

    def table(acc, cnt, cles):
        rows = []
        for i, cle in enumerate(cles):
            n = max(cnt[i], 1)
            rc, rp = np.sqrt(acc["cache"][i] / n), np.sqrt(acc["previsible"][i] / n)
            rows.append({"cle": cle, "rmse_u_cache": rc[0], "rmse_v_cache": rc[1],
                         "rmse_u_previsible": rp[0], "rmse_v_previsible": rp[1],
                         "part_v_perturbations": 1 - (rp[1] / rc[1]) ** 2 if rc[1] > 0 else np.nan})
        return pd.DataFrame(rows)

    niv = table(acc_n, cnt_n, [int(z) for z in geom.z_levels])
    tr = table(acc_t, cnt_t, [t[0] for t in tranches])
    niv.to_csv(out / "par_niveau.csv", index=False)
    tr.to_csv(out / "par_tranche.csv", index=False)
    dom = tr.iloc[:-1]
    resume = {"statut": STATUT, "modele": ev.wf.checkpoint_path.name, "instants": len(stems),
              "perturbations_generateur": a.perturbations, "replique_graine_origine": replique,
              "dso1_previsible_0_1km_rmse_max": float(dom[["rmse_u_previsible", "rmse_v_previsible"]].max().max()),
              "dso1_cache_0_1km_rmse_max": float(dom[["rmse_u_cache", "rmse_v_cache"]].max().max())}
    (out / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
    L = [f"# DSO1 sur la verite previsible ({len(stems)} instants de test) - {STATUT}", "",
         "Replique avec la graine d'origine (ecart max au cache, m/s) : "
         + ", ".join(f"{r['max_abs_m_s']:.4f}" for r in replique), "",
         tr.round(3).to_string(index=False), "", niv.round(3).to_string(index=False)]
    (out / "dso_previsible.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"resultats -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
