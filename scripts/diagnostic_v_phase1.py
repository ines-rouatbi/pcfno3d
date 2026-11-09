"""Phase 1 nationale, diagnostics 1 a 3 de l'erreur sur v (sans entrainement).

1. Modes de Fourier : model.modes / width / pad, grille (nz, ny, nx), plus petite longueur d'onde
   vue par la partie spectrale dans chaque direction (taille du domaine augmentee du pad / modes).
2. Spectre de l'erreur : a 300 et 800 m au-dessus du sol, instants synoptiques du TEST, spectre de
   puissance le long de x (est-ouest) et de y (nord-sud) de l'erreur modele - verite, de la verite
   et de (verite - ERA5), pour u et v ; part de l'energie par bande de longueur d'onde.
3. Entrainement contre test : RMSE u, v par tranche au-dessus du sol sur 60 instants
   d'ENTRAINEMENT et 60 du TEST (memes entrees de production, graine fixe).

    python scripts/diagnostic_v_phase1.py -c national_eval_v2 [--n-train 60]
Sortie : <run_dir>/diagnostic_v_phase1/ (resume.json, spectres.csv, bandes.csv, tranches.csv, .md)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
HAUTEURS = (300.0, 800.0)
BANDES_KM = ((200.0, np.inf), (100.0, 200.0), (50.0, 100.0), (26.0, 50.0), (13.0, 26.0), (0.0, 13.0))


def modes_info(cfg, geom) -> dict:
    m = cfg.model
    nz, ny, nx = geom.shape
    pz, py, px = m.pad
    dy_km, dx_km = geom.dy / 1000.0, geom.dx / 1000.0
    return {"modes_z_y_x": list(m.modes), "width": m.width, "pad_z_y_x": list(m.pad),
            "local_conv": bool(m.local_conv), "grille_nz_ny_nx": [nz, ny, nx], "pas_km": [dy_km, dx_km],
            # mode le plus eleve garde = modes - 1 (indices 0 .. modes-1 sur la grille augmentee du pad)
            "lambda_min_y_km": (ny + py) * dy_km / (m.modes[1] - 1),
            "lambda_min_x_km": (nx + px) * dx_km / (m.modes[2] - 1),
            "niveaux_z": nz + pz, "modes_z": m.modes[0]}


def spectre(champ: np.ndarray, axe: int) -> np.ndarray:
    """Puissance moyenne par nombre d'onde le long de `axe` (0 = y, 1 = x) d'un champ 2D a trous :
    moyenne retiree sur les points valides, trous a 0, normalisee par la part de points valides."""
    m = np.isfinite(champ)
    if not m.any():
        return None
    a = np.where(m, champ - np.nanmean(champ), 0.0)
    f = np.abs(np.fft.rfft(a, axis=axe)) ** 2
    p = f.mean(axis=1 - axe) / max(m.mean(), 1e-6)
    return p / a.shape[axe]


def main() -> int:
    import torch

    from pcfno3d.config import Config
    from pcfno3d.dataset import ANEO3D, eval_loader, is_synoptic
    from pcfno3d.dso_national import predicteur_era5
    from pcfno3d.dso_tranches import TRANCHES_DEFAUT, masques_tranches, metriques_tranches
    from pcfno3d.evaluate import Evaluator

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--max-instants", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--hauteurs", default="300,800", help="hauteurs au-dessus du sol (m)")
    a = ap.parse_args()
    hauteurs = tuple(float(x) for x in a.hauteurs.split(","))

    cfg = Config.load(a.config)
    ev = Evaluator(cfg, device=a.device, verbose=False)
    geom, splits = ev.geom, ev.splits
    out = cfg.paths.run_dir / "diagnostic_v_phase1"
    out.mkdir(parents=True, exist_ok=True)
    info = modes_info(cfg, geom)

    # ------------------------------------------------------------ 2. spectres (test)
    te_s = splits.test[0]
    idx = np.nonzero([is_synoptic(splits.stamp_of[s]) for s in te_s])[0]
    if a.max_instants:
        idx = idx[:a.max_instants]
    loader, stems = eval_loader(cfg, geom, splits, idx, 2000)
    stems = [str(s) for s in stems]
    pera5 = predicteur_era5(geom, ev.norm)
    z = np.asarray(geom.z_levels, float)
    zt = np.asarray(geom.zt, float)
    valid = np.asarray(geom.valid, bool)

    def a_hauteur(c3, h):
        zz = zt + h
        i = np.clip(np.searchsorted(z, zz) - 1, 0, len(z) - 2)
        t = np.clip((zz - z[i]) / (z[i + 1] - z[i]), 0, 1)
        lo = np.take_along_axis(c3, i[None], 0)[0]
        hi = np.take_along_axis(c3, (i + 1)[None], 0)[0]
        ok = (zz >= z[0]) & (zz <= z[-1]) & np.take_along_axis(valid, i[None], 0)[0] \
            & np.take_along_axis(valid, (i + 1)[None], 0)[0]
        return np.where(ok, (1 - t) * lo + t * hi, np.nan)

    acc: dict = {}
    with torch.no_grad():
        for s_raw, sensor, tfeat, y, j in loader:
            s_raw, sensor, tfeat, y = (t.to(geom.device) for t in (s_raw, sensor, tfeat, y))
            stem = stems[int(j)]
            u, v, _ = ev.predire(s_raw, sensor, tfeat, y, int(j), stem)
            ue, ve, _ = pera5(s_raw, sensor, tfeat, y, int(j), stem)
            P = {"u": u[0].float().cpu().numpy(), "v": v[0].float().cpu().numpy()}
            E = {"u": ue[0].float().cpu().numpy(), "v": ve[0].float().cpu().numpy()}
            Y = {"u": y[0, 0].float().cpu().numpy(), "v": y[0, 1].float().cpu().numpy()}
            for h in hauteurs:
                for c in "uv":
                    champs = {"erreur_modele": a_hauteur(P[c] - Y[c], h), "verite": a_hauteur(Y[c], h),
                              "verite_moins_era5": a_hauteur(Y[c] - E[c], h)}
                    for nom, ch in champs.items():
                        for axe, an in ((0, "y"), (1, "x")):
                            sp = spectre(ch, axe)
                            if sp is None:
                                continue
                            k = (int(h), c, nom, an)
                            acc[k] = acc.get(k, 0) + sp
    n = len(stems)
    lignes, bandes = [], []
    for (h, c, nom, an), sp in acc.items():
        sp = sp / n
        L = (geom.ny * geom.dy if an == "y" else geom.nx * geom.dx) / 1000.0
        kk = np.arange(len(sp))
        lam = np.where(kk > 0, L / np.maximum(kk, 1), np.inf)
        for k_, l_, p_ in zip(kk, lam, sp):
            lignes.append({"hauteur_sol_m": h, "composante": c, "champ": nom, "axe": an,
                           "k": int(k_), "longueur_onde_km": float(l_), "puissance": float(p_)})
        tot = sp[1:].sum()
        for lo, hi in BANDES_KM:
            sel = (lam >= lo) & (lam < hi) & (kk > 0)
            bandes.append({"hauteur_sol_m": h, "composante": c, "champ": nom, "axe": an,
                           "bande_km": f"{lo:g}-{hi:g}", "part_energie": float(sp[sel].sum() / tot) if tot > 0 else np.nan})
    pd.DataFrame(lignes).to_csv(out / "spectres.csv", index=False)
    b = pd.DataFrame(bandes, columns=["hauteur_sol_m", "composante", "champ", "axe", "bande_km", "part_energie"])
    b.to_csv(out / "bandes.csv", index=False)
    # longueur d'onde qui partage l'energie de l'erreur en deux (mediane spectrale)
    med = []
    for (h, c, nom, an), sp in acc.items():
        sp = (sp / n)[1:]
        L = (geom.ny * geom.dy if an == "y" else geom.nx * geom.dx) / 1000.0
        cum = np.cumsum(sp) / sp.sum()
        k50 = int(np.searchsorted(cum, 0.5)) + 1
        med.append({"hauteur_sol_m": h, "composante": c, "champ": nom, "axe": an,
                    "longueur_onde_mediane_km": L / k50})
    med = pd.DataFrame(med, columns=["hauteur_sol_m", "composante", "champ", "axe", "longueur_onde_mediane_km"])
    med.to_csv(out / "longueur_onde_mediane.csv", index=False)

    # ------------------------------------------------------------ 3. entrainement contre test
    tranches = masques_tranches(geom, TRANCHES_DEFAUT)
    tr_s, tr_h, tr_d = splits.train
    dispo = np.array([(cfg.paths.cache_dir / f"{s}_Y.npy").exists() for s in tr_s])
    ti = np.nonzero(dispo & np.array([is_synoptic(splits.stamp_of[s]) for s in tr_s]))[0]
    ti = ti[np.unique(np.linspace(0, len(ti) - 1, min(a.n_train, len(ti))).astype(int))] if len(ti) else ti
    res_tr = None
    if len(ti):
        ds = ANEO3D(tr_s[ti], tr_h[ti], tr_d[ti], geom, train=False, seed=2000)
        from torch.utils.data import DataLoader
        res_tr = metriques_tranches(DataLoader(ds, batch_size=1), [str(s) for s in tr_s[ti]], geom,
                                    {"modele": ev.predire}, tranches)["modele"]
    te_i = idx[np.unique(np.linspace(0, len(idx) - 1, min(a.n_train, len(idx))).astype(int))]
    res_te = metriques_tranches(*eval_loader(cfg, geom, splits, te_i, 2000), geom,
                                {"modele": ev.predire}, tranches)["modele"]
    t = res_te[["tranche", "rmse_u", "rmse_v"]].rename(columns={"rmse_u": "test_rmse_u", "rmse_v": "test_rmse_v"})
    if res_tr is not None:
        t = res_tr[["tranche", "rmse_u", "rmse_v"]].rename(
            columns={"rmse_u": "entrainement_rmse_u", "rmse_v": "entrainement_rmse_v"}).merge(t, on="tranche")
        t["entrainement_v_sur_u"] = t["entrainement_rmse_v"] / t["entrainement_rmse_u"]
    t["test_v_sur_u"] = t["test_rmse_v"] / t["test_rmse_u"]
    t.to_csv(out / "tranches.csv", index=False)

    resume = {"modes": info, "instants_spectre": n, "instants_entrainement": int(len(ti)),
              "instants_test_tranches": int(len(te_i))}
    (out / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 220)
    piv = (b.pivot_table(index=["hauteur_sol_m", "composante", "champ", "axe"], columns="bande_km",
                         values="part_energie").reindex(columns=[f"{lo:g}-{hi:g}" for lo, hi in BANDES_KM])
           if len(b) else b)
    L = ["# Phase 1, diagnostics 1 a 3", "", json.dumps(info, indent=1), "",
         "## Part de l'energie par bande de longueur d'onde (km)", piv.round(3).to_string(), "",
         "## Longueur d'onde mediane (km)", med.round(1).to_string(index=False), "",
         "## RMSE par tranche au-dessus du sol : entrainement contre test", t.round(3).to_string(index=False)]
    (out / "diagnostic_v_phase1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"resultats -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
