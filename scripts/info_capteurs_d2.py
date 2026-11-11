"""Candidat D2 national (configs/national/regle_d2_national.json, PROPOSITION) : choix de l'echelle verticale
H et evaluation d'information, sur la VALIDATION seulement (le test n'est jamais lu ici).

Deux etapes :
  reglage : SANS modele. Reference « interpolation optimale seule » (ERA5 + innovation etalee x f(h), w par
            continuite) pour chaque H de la grille fixee d'avance (--H 200,500,1000 m ; L et sigma_o / sigma_b
            viennent de scripts/statistiques_innovation_d2.py, jours d'entrainement), dans deux conditions :
            propre (ERA5 10 m + 12 aeroports, l'entree de production) et ERA5 degradee (champ correle du
            candidat D, sans zones) + n capteurs ANEO simules (n = 50). H retenu = celui dont le rmse_max_uv
            0-1 km de l'OI seule est le plus petit dans la condition --condition (defaut : propre) ; ecrit
            AVANT l'entrainement.
  info    : un entrainement D2 (run_dir) et ses deux references (ERA5 seule, OI seule) dans les trois
            conditions de scripts/info_capteurs_v4.py : propre, era5_degradee, era5_degradee_<n>_capteurs ;
            gain des capteurs = rmse_max_uv (degradee) - rmse_max_uv (degradee + capteurs), pour le modele
            et pour l'OI seule. Sans effet sur le choix de l'epoque (validation propre, comme C' et D).
Metriques : RMSE u, v par tranche de hauteur au-dessus du sol (scripts/dso_hauteur_sol.py), rmse_max_uv sur
les tranches 0-1 km. Graine des tirages : 1000 + instant (celle du loader de validation), memes tirages pour
tous les H et tous les entrainements.

    python scripts/info_capteurs_d2.py reglage -c national_1km_colab --degradation <json> --sortie <json> \
        --set innovation.actif=true --set innovation.L_km=<...> --set innovation.rapport_o_b=<...> \
        --set mask.aneo_hauteur_capteur_m=10 [--H 200,500,1000] [--condition propre]
    python scripts/info_capteurs_d2.py info -c national_1km_colab --run <run_dir D2> --degradation <json>
Sorties : <sortie> (reglage), <run_dir>/info_capteurs_d2_validation.json (info) ; jamais ecrasees.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]


def mesurer(geom, splits, sel, conditions: dict, preds: dict) -> dict:
    """{condition: {predicteur: {tranches, rmse_max_uv}, points_d_entree_moyen}} sur la validation."""
    import torch
    from torch.utils.data import DataLoader

    from dso_hauteur_sol import masques_tranches
    from pcfno3d.dataset import ANEO3D

    s, h, d = splits.val
    tranches, _ = masques_tranches(geom)
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
                            e = out[c][0][k].double() - y[0, c][k].double()
                            acc[p][t][c] += [k.sum().item(), (e ** 2).sum().item()]
        res[nom] = {"points_d_entree_moyen": float(np.mean(n_capt))}
        for p in preds:
            r = {t: {"rmse_u": float(np.sqrt(acc[p][t][0, 1] / max(acc[p][t][0, 0], 1))),
                     "rmse_v": float(np.sqrt(acc[p][t][1, 1] / max(acc[p][t][1, 0], 1))),
                     "noeuds": int(acc[p][t][0, 0])} for t in tranches}
            par = [k for k in r if "domaine" not in k and "info" not in k and r[k]["noeuds"] > 0]
            res[nom][p] = {"tranches": r,
                           "rmse_max_uv": max(max(r[k]["rmse_u"], r[k]["rmse_v"]) for k in par)}
        print(nom, {p: round(res[nom][p]["rmse_max_uv"], 4) for p in preds}, flush=True)
    return res


def pred_era5(geom):
    from pcfno3d.innovation import decouper_tfeat

    def f(s_raw, sensor, tfeat, y, j, stem):
        cel, _q = decouper_tfeat(tfeat, geom)
        uv = geom.interp_era5.dense(cel)
        return [uv[:, 0] * geom.valid_t, uv[:, 1] * geom.valid_t]
    return f


def conditions_info(geom, cfg, degradation: str, n: int) -> dict:
    from pcfno3d.aneo_entree import EntreeANEO
    m = cfg.mask
    return {"propre": None,
            "era5_degradee": EntreeANEO(geom, degradation, 0.0, m.aneo_rayon_km),
            f"era5_degradee_{n}_capteurs": EntreeANEO(geom, degradation, 0.0, m.aneo_rayon_km, (n, n),
                                                       m.aneo_hauteur_capteur_m)}


def _sortie(p: Path) -> Path:
    if p.exists():
        raise SystemExit(f"{p} existe deja : rien n'est recalcule ni ecrase")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def reglage(a, cfg) -> int:
    from pcfno3d.cache import load_index
    from pcfno3d.dataset import make_splits
    from pcfno3d.geometry import Geometry
    from pcfno3d.innovation import predicteur_oi

    sortie = _sortie(Path(a.sortie))
    if not cfg.innovation.actif:
        raise SystemExit("reglage : --set innovation.actif=true (parametres de l'OI de la regle D2)")
    geom = Geometry(cfg, device=a.device or "cpu")
    splits = make_splits(cfg, load_index(cfg))
    sel = splits.val_selection[:a.max_instants] if a.max_instants else splits.val_selection
    cond = conditions_info(geom, cfg, a.degradation, a.n_capteurs)
    cle_deg = f"era5_degradee_{a.n_capteurs}_capteurs"
    cond = {"propre": cond["propre"], cle_deg: cond[cle_deg]}
    cle = "propre" if a.condition == "propre" else cle_deg
    par_H = {}
    for H in [float(x) for x in a.H.split(",")]:
        geom.configurer_innovation(dataclasses.replace(cfg.innovation, H_m=H))
        print(f"H = {H:g} m", flush=True)
        par_H[f"{H:g}"] = mesurer(geom, splits, sel, cond, {"era5": pred_era5(geom), "era5_oi": predicteur_oi(geom)})
    # premier de la grille en cas d'egalite exacte (ordre fixe d'avance)
    retenu = min(par_H, key=lambda k: par_H[k][cle]["era5_oi"]["rmse_max_uv"])
    deg = json.loads(Path(a.degradation).read_text(encoding="utf-8"))
    deg.pop("paires", None)
    out = {"jeu": "validation", "instants": int(len(sel)),
           "regle": "configs/national/regle_d2_national.json (PROPOSITION soumise a l'accord ecrit d'Ines)",
           "critere": f"min rmse_max_uv 0-1 km de l'OI seule (sans reseau), condition {cle}",
           "H_candidats_m": [float(k) for k in par_H], "H_retenu_m": float(retenu),
           "innovation": dataclasses.asdict(dataclasses.replace(cfg.innovation, H_m=float(retenu))),
           "degradation": deg, "resultats_par_H": par_H,
           "note": "verite v3 = verite d'entrainement ; capteurs simules = verite + bruit, pas des mesures"}
    sortie.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("instants", "H_candidats_m", "H_retenu_m")}, indent=1))
    return 0


def info(a, cfg) -> int:
    from pcfno3d.evaluate import Evaluator
    from pcfno3d.innovation import predicteur_oi

    cfg.apply_overrides({"paths.run_dir": a.run})
    ev = Evaluator(cfg, device=a.device, verbose=False)
    geom, splits = ev.geom, ev.splits
    if geom.oi is None:
        raise SystemExit(f"{a.run} : le modele n'a pas le canal d'innovation (candidat D2)")
    nom = ("info_capteurs_d2_validation.json" if not a.max_instants
           else f"info_capteurs_d2_validation_essai{a.max_instants}.json")
    sortie = _sortie(Path(a.run) / nom)
    sel = splits.val_selection[:a.max_instants] if a.max_instants else splits.val_selection
    preds = {"modele": lambda *x: ev.predire(*x)[:2], "era5": pred_era5(geom), "era5_oi": predicteur_oi(geom)}
    res = mesurer(geom, splits, sel, conditions_info(geom, cfg, a.degradation, a.n_capteurs), preds)
    k0, k1 = "era5_degradee", f"era5_degradee_{a.n_capteurs}_capteurs"
    deg = json.loads(Path(a.degradation).read_text(encoding="utf-8"))
    deg.pop("paires", None)
    out = {"run": Path(a.run).name, "jeu": "validation", "instants": int(len(sel)),
           "regle": "configs/national/regle_d2_national.json (information, sans effet sur le choix)",
           "innovation": dataclasses.asdict(geom.cfg.innovation), "degradation": deg, "conditions": res,
           "gain_des_capteurs_rmse_max_uv": {p: res[k0][p]["rmse_max_uv"] - res[k1][p]["rmse_max_uv"]
                                             for p in preds},
           "note": "verite v3 = verite d'entrainement ; capteurs simules = verite a 25 m + bruit, pas des mesures"}
    sortie.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("run", "instants", "gain_des_capteurs_rmse_max_uv")}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("etape", choices=["reglage", "info"])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--degradation", required=True)
    ap.add_argument("--run", default=None, help="info : run_dir de l'entrainement D2")
    ap.add_argument("--sortie", default=None, help="reglage : JSON ecrit (jamais ecrase)")
    ap.add_argument("--H", default="200,500,1000", help="reglage : echelles verticales candidates (m)")
    ap.add_argument("--condition", default="propre", choices=["propre", "degradee_capteurs"],
                    help="reglage : condition du critere (propre = entree de production)")
    ap.add_argument("--set", action="append", default=[], help="section.cle=valeur")
    ap.add_argument("--n-capteurs", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-instants", type=int, default=None, help="essai rapide (pas un resultat)")
    a = ap.parse_args()

    from pcfno3d.cli import _parse_set
    from pcfno3d.config import Config
    cfg = Config.load(a.config)
    cfg.apply_overrides(_parse_set(a.set))
    if a.etape == "reglage":
        if not a.sortie:
            raise SystemExit("reglage : --sortie requis")
        return reglage(a, cfg)
    if not a.run:
        raise SystemExit("info : --run requis")
    return info(a, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
