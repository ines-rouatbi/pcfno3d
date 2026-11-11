"""Evaluation d'INFORMATION de la regle v4 nationale (ajout d'Ines, 10/11) : sans effet sur le choix.

Sur la VALIDATION de la verite v3 (memes 120 instants que le score de selection, entree complete), pour
un entrainement (C' ou D) et pour l'ERA5 seule vue a travers la meme entree, trois conditions :
  propre                     : entree de production (ERA5 10 m + 12 aeroports), comme le score ;
  era5_degradee              : ERA5 degradee par le champ correle du candidat D (degradation_era5.json),
                               sans masquage par zones ;
  era5_degradee_50_capteurs  : la meme degradation (meme tirage par instant) + 50 capteurs ANEO
                               simules (mats a 25 m, bruit 0,1 m/s + 2 %, candidats de l'OS6).
Le gain des capteurs = RMSE (era5_degradee) - RMSE (era5_degradee_50_capteurs) montre lequel des
candidats exploite les capteurs. Metriques sur 0-1 km au-dessus du sol, tranches de
scripts/dso_hauteur_sol.py : RMSE u, v par tranche, rmse_max_uv (max sur les tranches), et sur le
domaine 0-1 km. Graine des tirages : 1000 + instant (celle du loader de validation). Le test n'est
jamais lu ici.

    python scripts/info_capteurs_v4.py -c national_1km_colab --run <run_dir> --degradation <json>
Sortie : <run_dir>/info_capteurs_validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--degradation", required=True)
    ap.add_argument("--set", action="append", default=[], help="section.cle=valeur")
    ap.add_argument("--n-capteurs", type=int, default=50)
    ap.add_argument("--max-instants", type=int, default=None, help="essai rapide (pas un resultat)")
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from dso_hauteur_sol import masques_tranches
    from pcfno3d.aneo_entree import EntreeANEO
    from pcfno3d.config import Config
    from pcfno3d.dataset import ANEO3D
    from pcfno3d.evaluate import Evaluator
    from pcfno3d.forcage import N_SCAL
    from pcfno3d.geometry import N_TFEAT

    cfg = Config.load(a.config)
    from pcfno3d.cli import _parse_set
    over = _parse_set(a.set)
    over["paths.run_dir"] = a.run
    cfg.apply_overrides(over)
    ev = Evaluator(cfg, device=None, verbose=False)
    geom, splits = ev.geom, ev.splits
    s, h, d = splits.val
    sel = splits.val_selection
    if a.max_instants:
        sel = sel[:a.max_instants]
    m = cfg.mask
    deg = EntreeANEO(geom, a.degradation, 0.0, m.aneo_rayon_km)
    deg50 = EntreeANEO(geom, a.degradation, 0.0, m.aneo_rayon_km, (a.n_capteurs, a.n_capteurs),
                       m.aneo_hauteur_capteur_m)
    conditions = {"propre": None, "era5_degradee": deg, f"era5_degradee_{a.n_capteurs}_capteurs": deg50}
    tranches, _ = masques_tranches(geom)

    def pred_era5(s_raw, sensor, tfeat, y, j, stem):
        nc = geom.forcage.n_cellules            # avant une eventuelle queue d'innovation (D2)
        uv = geom.interp_era5.dense(tfeat[:, N_TFEAT + N_SCAL:N_TFEAT + N_SCAL + nc])
        return [uv[:, 0] * geom.valid_t, uv[:, 1] * geom.valid_t]

    preds = {"modele": lambda *x: ev.predire(*x)[:2], "era5": pred_era5}
    res = {}
    for nom, aneo in conditions.items():
        ds = ANEO3D(s[sel], h[sel], d[sel], geom, train=False, seed=1000, keep_eval=1.0, aneo_eval=aneo)
        acc = {p: {t: np.zeros((2, 2)) for t in tranches} for p in preds}
        n_capt = []
        with torch.no_grad():
            for i, (s_raw, sensor, tfeat, y, j) in enumerate(DataLoader(ds, batch_size=1)):
                s_raw, sensor, tfeat, y = (t.to(geom.device) for t in (s_raw, sensor, tfeat, y))
                n_capt.append(int(sensor.sum().item()))
                for p, f in preds.items():
                    out = f(s_raw, sensor, tfeat, y, int(j), str(s[sel][i]))
                    for c in (0, 1):
                        for t, mk in tranches.items():
                            k = mk & torch.isfinite(y[0, c])
                            e = (out[c][0][k].double() - y[0, c][k].double())
                            acc[p][t][c] += [k.sum().item(), (e ** 2).sum().item()]
        res[nom] = {"points_d_entree_moyen": float(np.mean(n_capt))}
        for p in preds:
            r = {t: {"rmse_u": float(np.sqrt(acc[p][t][0, 1] / acc[p][t][0, 0])),
                     "rmse_v": float(np.sqrt(acc[p][t][1, 1] / acc[p][t][1, 0]))} for t in tranches}
            par = [k for k in r if "domaine" not in k and "info" not in k]
            res[nom][p] = {"tranches": r, "rmse_max_uv": max(max(r[k]["rmse_u"], r[k]["rmse_v"]) for k in par)}
        print(nom, {p: round(res[nom][p]["rmse_max_uv"], 4) for p in preds}, flush=True)
    k0, k1 = "era5_degradee", f"era5_degradee_{a.n_capteurs}_capteurs"
    out = {"run": Path(a.run).name, "jeu": "validation", "instants": int(len(sel)),
           "regle": "configs/national/regle_v4_national.json (information, sans effet sur le choix)",
           "degradation": json.loads(Path(a.degradation).read_text(encoding="utf-8")),
           "conditions": res,
           "gain_des_capteurs_rmse_max_uv": res[k0]["modele"]["rmse_max_uv"] - res[k1]["modele"]["rmse_max_uv"],
           "note": "verite v3 = verite d'entrainement ; capteurs simules = verite a 25 m + bruit, pas des mesures"}
    out["degradation"].pop("paires", None)
    p = Path(a.run) / ("info_capteurs_validation.json" if not a.max_instants else
                       f"info_capteurs_validation_essai{a.max_instants}.json")
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("run", "instants", "gain_des_capteurs_rmse_max_uv")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
