"""PREV1 / PREV2 (cahier v12, OS7) : le vent PREVU a +24 h / +48 h aux stations de l'INM, sur le test 2022,
contre GFS brut et la persistance.

Pour chaque instant synoptique du test ou une station mesure :
  - prevision ATOM Wind : le modele, avec en entree la prevision GFS de l'echeance (forcage et reseau
    a 10 m : `paths.forcage_era5` = forcage_gfs_f24.csv / f48, `mask.e_source: forcage`) et AUCUNE
    station (leurs mesures de demain n'existent pas) ; vent a 10 m a la station comme
    scripts/valider_grand_tunis_reel.py (noeuds de la colonne, profil logarithmique sous le premier) ;
  - GFS brut : le vent GFS a 10 m de la meme echeance, interpole a la cellule de la station ;
  - persistance : la mesure de la station `echeance` heures plus tot (a 30 min pres).

    python scripts/evaluer_prevision.py -c grand_tunis_v5_gfs_colab --set paths.forcage_era5=<f48.csv> \
        --obs /content/obs --echeance 48 --sortie <dossier>

Sorties : prevision_paires.csv (u_obs, u_prevision, u_gfs, u_persistance...), prevision_bilan.csv
(par station et source : erreur vectorielle RMS, biais de vitesse, erreur de direction), et
prevision_bilan.json avec les verdicts PREV : erreur vectorielle de la prevision < GFS brut et <
persistance (toutes stations confondues et par station). Les decisions de traitement de la veille
(PREV3) : scripts/verifier_decisions.py --paires prevision_paires.csv --sources prevision,gfs,persistance.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]


def _valider():
    spec = importlib.util.spec_from_file_location("valider", REPO / "scripts" / "valider_grand_tunis_reel.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def evaluer(cfg, dossier_obs: Path, echeance_h: int, checkpoint=None, device=None,
            max_instants: int | None = None, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from pcfno3d.cache import load_index
    from pcfno3d.dataset import is_synoptic, make_splits
    from pcfno3d.forcage import charger_e
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.stations_reelles import (CODES, cellules_codes, entree_reelle, lire_observations,
                                          obs_a, vent_a_hauteur)

    V = _valider()
    if cfg.mask.e_source != "forcage" or cfg.mask.stations_entree:
        raise ValueError("evaluer_prevision : la configuration doit etre celle de la prevision "
                         "(mask.e_source: forcage, mask.stations_entree: false)")
    wf = WindFieldModel(cfg, checkpoint=checkpoint, device=device, verbose=False)
    geom = wf.geom
    index = load_index(cfg)
    splits = make_splits(cfg, index)
    stems = [s for s in splits.test[0] if is_synoptic(splits.stamp_of[s])]
    if max_instants:
        stems = stems[:max_instants]
    row = index.set_index("stem")
    cellules = cellules_codes(geom)
    obs = lire_observations(Path(dossier_obs))
    h_noeuds = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    retard = pd.Timedelta(hours=echeance_h)
    paires, t0 = [], time.time()
    for n, stem in enumerate(stems):
        r = row.loc[stem]
        t = pd.Timestamp(r["timestamp"])
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        mesures = obs_a(obs, t)
        if not mesures:
            continue
        try:
            E = charger_e(geom, cfg.paths.cache_dir / f"{stem}_E.npy")
        except ValueError:                       # pas d'echeance GFS pour cet instant
            continue
        S = np.load(cfg.paths.cache_dir / f"{stem}_S.npy")
        S_in, cap = entree_reelle(S, E, geom, cellules, {})              # aucune station
        S_in = S_in.copy()
        S_in[7] = cap
        vol = wf.predict(S_in, float(r["hour"]), int(r["doy"]), timestamp=t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         reliability=False)
        avant = obs_a(obs, t - retard)
        for code, (uo, vo) in mesures.items():
            iy, ix = cellules[code]
            um, vm = vent_a_hauteur(vol.u[:, iy, ix], vol.v[:, iy, ix], h_noeuds[:, iy, ix], geom.z0[iy, ix])
            up, vp = avant.get(code, (np.nan, np.nan))
            paires.append({"station": CODES[code], "stem": stem, "temps": t, "echeance_h": echeance_h,
                           "u_obs": uo, "v_obs": vo, "u_prevision": um, "v_prevision": vm,
                           "u_gfs": float(E[0, iy, ix]), "v_gfs": float(E[1, iy, ix]),
                           "u_persistance": up, "v_persistance": vp})
        if verbose and (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(stems)} instants ({time.time() - t0:.0f} s)", flush=True)
    p = pd.DataFrame(paires)
    sources = ("prevision", "gfs", "persistance")
    b = V.bilan_stations(p, sources) if len(p) else pd.DataFrame()
    verdicts = {}
    if len(b):
        for station in ["toutes"] + sorted(p["station"].unique()):
            q = p if station == "toutes" else p[p["station"] == station]
            e = {}
            for s in sources:
                ok = q[[f"u_{s}", f"v_{s}"]].notna().all(axis=1) & q[["u_obs", "v_obs"]].notna().all(axis=1)
                d = q[ok]
                e[s] = float(np.sqrt(np.mean((d[f"u_{s}"] - d["u_obs"]) ** 2 + (d[f"v_{s}"] - d["v_obs"]) ** 2))) \
                    if len(d) else float("nan")
            verdicts[station] = {"rmse_vect": e, "n": int(len(q)),
                                 "mieux_que_gfs": bool(e["prevision"] < e["gfs"]),
                                 "mieux_que_persistance": bool(e["prevision"] < e["persistance"])}
    code = "PREV1" if echeance_h == 24 else "PREV2"
    resume = {"critere": code, "echeance_h": echeance_h, "modele": wf.checkpoint_path.name,
              "instants": len(stems), "paires": int(len(p)), "verdicts": verdicts,
              "conforme": bool(verdicts.get("toutes", {}).get("mieux_que_gfs")
                               and verdicts.get("toutes", {}).get("mieux_que_persistance"))}
    return p, b, resume


def main() -> int:
    from pcfno3d.config import Config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--obs", required=True)
    ap.add_argument("--echeance", type=int, required=True, choices=(24, 48))
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--max-instants", type=int)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg.apply_overrides({k: v})
    p, b, resume = evaluer(cfg, Path(args.obs), args.echeance, args.checkpoint, args.device, args.max_instants)
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    p.to_csv(sortie / "prevision_paires.csv", index=False)
    b.to_csv(sortie / "prevision_bilan.csv", index=False)
    (sortie / "prevision_bilan.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
    print(b.round(3).to_string(index=False) if len(b) else "aucune paire")
    print(json.dumps(resume["verdicts"].get("toutes", {}), indent=1))
    print(f"-> {sortie}")
    return 0 if resume["conforme"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
