"""Variantes PROPOSEES des DSO1 et DSO3 (Grand Tunis) — calculees, NON utilisees pour decider
tant que l'encadrant ne les a pas validees (consigne d'Ines, 2026-10-24).

DSO3 « previsible » : la verite generee contient une composante ALEATOIRE (perturbations
rotationnelle et divergente de petite echelle du generateur, A_ROT / A_DIV) qu'aucune entree ne
permet de prevoir. On regenere chaque instant de test SANS ces perturbations (meme forcage ERA5,
meme relief, meme graine : scripts/diag_grand_tunis.py, mode « zero ») et on calcule la rugosite
de la prediction rapportee a celle de cette verite previsible, globalement et par niveau. Quelques
instants sont aussi regeneres AVEC la graine d'origine : l'ecart au cache verifie que la
regeneration reproduit bien la verite du test.

DSO1 « u, v » : verdicts par niveau sur le RMSE et le R2 de u et v seulement, le R2 de w rapporte
a part — lu dans la table par niveau de `pcfno3d evaluate` (scripts/decision_v4_grand_tunis.py).

Memes instants, memes entrees (loader de DSO1, graine 2000) et meme passe que l'evaluation ; aucun
bruit n'est ajoute aux predictions.

    python scripts/dso_variantes_grand_tunis.py -c grand_tunis_v4_dso_colab \
        --donnees /content/donnees/grand_tunis [--w-continuite]

Sortie : <run_dir>/eval_pcfno3d/<modele>[_wcont]/dso3_previsible.json et .csv (par niveau).
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

STATUT = ("variante proposee, NON utilisee pour decider (en attente de validation par "
          "l'encadrant)")


def main() -> int:
    import torch

    import diag_grand_tunis as diag
    from pcfno3d.config import Config
    from pcfno3d.dataset import eval_loader, is_synoptic
    from pcfno3d.dso import dso3_conforme
    from pcfno3d.evaluate import Evaluator

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--donnees", required=True, help="dossier grand_tunis (grille, forcage)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--w-continuite", action="store_true", help="forme livree : w par continuite")
    ap.add_argument("--perturbations", type=float, default=0.4, help="facteur de la generation")
    ap.add_argument("--n-replique", type=int, default=3)
    ap.add_argument("--max-instants", type=int, default=None, help="tests rapides")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.w_continuite:
        cfg.inference.w_continuite = True
    ev = Evaluator(cfg, checkpoint=args.checkpoint, device=args.device, verbose=False)
    geom, splits = ev.geom, ev.splits
    te_s = splits.test[0]
    syn = np.array([is_synoptic(splits.stamp_of[s]) for s in te_s])
    syn_idx = np.nonzero(syn)[0] if syn.any() else np.arange(len(te_s))
    if args.max_instants:
        syn_idx = syn_idx[: args.max_instants]
    loader, stems = eval_loader(cfg, geom, splits, syn_idx, 2000)   # les entrees de DSO1
    stems = [str(s) for s in stems]

    # ------------------------------------------------------------ verite previsible (CPU)
    G = Path(args.donnees)
    info = json.loads((G / "grille_reelle.json").read_text(encoding="utf-8"))
    # ~55 Mo par instant : disque LOCAL de la session (le Drive est presque plein)
    # la verite previsible ne depend pas du modele : partagee par la v2 et la v4
    base = Path(cfg.paths.local_dir).parent if cfg.paths.local_dir else ev.out_dir
    regen = base / "regen_sans_perturbation"
    regen.mkdir(parents=True, exist_ok=True)
    static_path = cfg.paths.data_dir / "static_cache.npz"
    if not static_path.exists():
        gen = diag.charger_generateur(info, args.perturbations)
        df = gen.load_or_build_grid(G / "aneo_grid_points_surface.csv", np.random.default_rng(0))
        static_path = regen / "static_cache.npz"
        np.savez(static_path, **gen.build_static(df))
    from pcfno3d.cache import load_index
    index = load_index(cfg)
    ts_of = dict(zip(index["stem"], pd.to_datetime(index["timestamp"])))
    jour_of = dict(zip(index["stem"], index["day"]))
    cal = json.loads((cfg.paths.data_dir / "calendrier.json").read_text(encoding="utf-8"))
    date_of = {d["day_index"]: d["date"] for d in cal["days"]}
    taches = []
    for i, stem in enumerate(stems):
        jour = int(jour_of[stem])
        t_sec = (ts_of[stem] - pd.Timestamp(date_of[jour])).total_seconds()
        for mode in ("zero",) + (("graine",) if i < args.n_replique else ()):
            if not (regen / f"{stem}_{mode}.npy").exists():
                taches.append((stem, jour, date_of[jour], t_sec, mode, str(regen)))
    t0 = time.time()
    if taches:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=diag._init,
                                 initargs=(str(static_path), str(G / "forcage_era5.csv"), info,
                                           args.perturbations)) as ex:
            for k, _ in enumerate(ex.map(diag._regenerer, taches), 1):
                if k % 10 == 0 or k == len(taches):
                    print(f"regeneration {k}/{len(taches)} ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------ rugosites (meme passe que DSO1)
    rough = ev.loss_fn.roughness
    gx_m, gy_m = geom.grad_mask_x_t, geom.grad_mask_y_t

    def rug_k(a, k):
        return ((a[0, k, :, 1:] - a[0, k, :, :-1]).abs()[gx_m[k]].mean()
                + (a[0, k, 1:, :] - a[0, k, :-1, :]).abs()[gy_m[k]].mean())

    nz = geom.nz
    lignes, replique = [], []
    par_niveau = np.zeros((nz, 3))             # n, somme rapport (verite), somme (previsible)
    with torch.no_grad():
        for n, (s_raw, sensor, tfeat, y, j) in enumerate(loader):
            s_raw, sensor, tfeat, y = (t.to(geom.device) for t in (s_raw, sensor, tfeat, y))
            stem = stems[int(j)]
            u, v, _w = ev.predire(s_raw, sensor, tfeat, y, int(j), stem)
            y0 = torch.from_numpy(np.load(regen / f"{stem}_zero.npy"))[None].to(geom.device)
            if not bool(torch.isfinite(y[:, :2]).all()):
                continue                       # verite partielle : rugosite sans objet (comme DSO3)
            r_pred = rough(u) + rough(v)
            r_vrai = rough(y[:, 0]) + rough(y[:, 1])
            r_prev = rough(y0[:, 0]) + rough(y0[:, 1])
            lignes.append({"stem": stem, "rugosite_verite": float(r_pred / r_vrai),
                           "rugosite_previsible": float(r_pred / r_prev),
                           "verite_sur_previsible": float(r_vrai / r_prev)})
            for k in range(nz):
                if gx_m[k].any() and gy_m[k].any():
                    p = rug_k(u, k) + rug_k(v, k)
                    par_niveau[k] += (1.0, float(p / (rug_k(y[:, 0], k) + rug_k(y[:, 1], k))),
                                      float(p / (rug_k(y0[:, 0], k) + rug_k(y0[:, 1], k))))
            rep = regen / f"{stem}_graine.npy"
            if rep.exists():
                yr = np.load(rep)
                ok = geom.valid
                replique.append({"stem": stem, "max_abs_m_s": float(
                    np.abs(yr[:3][:, ok] - y[0, :3].cpu().numpy()[:, ok]).max())})
            if (n + 1) % 20 == 0:
                print(f"  {n + 1}/{len(stems)} instants", flush=True)

    t = pd.DataFrame(lignes)
    niv = pd.DataFrame({"z_m": geom.z_levels.astype(int), "couverture": geom.cover,
                        "rugosite_verite": par_niveau[:, 1] / np.maximum(par_niveau[:, 0], 1),
                        "rugosite_previsible": par_niveau[:, 2] / np.maximum(par_niveau[:, 0], 1)})
    niv = niv[niv["couverture"] >= cfg.dso.min_level_cover]
    plage = cfg.dso.roughness_range
    moy_prev = float(t["rugosite_previsible"].mean())
    resume = {"statut": STATUT, "modele": ev.wf.checkpoint_path.name,
              "w_continuite": bool(cfg.inference.w_continuite), "instants": int(len(t)),
              "perturbations_generateur": args.perturbations,
              "dso3_rugosite_verite": float(t["rugosite_verite"].mean()),
              "dso3_rugosite_previsible": moy_prev,
              "dso3_previsible_conforme": dso3_conforme(moy_prev, plage),
              "verite_sur_previsible": float(t["verite_sur_previsible"].mean()),
              "plage": list(plage),
              "replique_graine_origine": replique,
              "par_niveau": niv.to_dict(orient="records")}
    (ev.out_dir / "dso3_previsible.json").write_text(
        json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
    niv.to_csv(ev.out_dir / "dso3_previsible_par_niveau.csv", index=False)
    print(json.dumps({k: v for k, v in resume.items() if k != "par_niveau"}, indent=1,
                     ensure_ascii=False))
    print(niv.round(3).to_string(index=False))
    print(f"resultats -> {ev.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
