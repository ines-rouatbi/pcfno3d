"""Validation du modele reel (Grand Tunis) sur de VRAIES mesures : stations et radiosondages.

Jusqu'ici, le modele n'a ete juge que contre la verite GENEREE. Ici, pour chaque instant
synoptique du jeu choisi (test par defaut, jamais vu a l'entrainement) :

  - entree de PRODUCTION : vent ERA5 a 10 m tous les `mask.pas_era5_m` + les anemometres
    REELS (METAR Tunis-Carthage, SYNOP Zaghouan et Nabeul, a 30 min au plus de l'instant) a
    la place de la verite generee aux cellules des stations ;
  - chaque station est jugee SANS elle-meme en entree (une passe par station retiree :
    validation croisee « station exclue ») ; la passe avec toutes les stations sert aux
    radiosondages ;
  - vent a 10 m au-dessus du sol a la station : noeud valide le plus bas de la maille, ramene
    a 10 m par le profil logarithmique (z0 de la maille), ou interpole entre deux noeuds ;
    references : ERA5 a 10 m (le _E du cache, meme interpolation qu'en production) et la
    couche de surface du generateur (S du cache, 10 m) ;
  - radiosondages Tunis-Carthage (IGRA, apparies sur l'heure nominale « temps ») entre 0 et
    1 000 m au-dessus du sol : colonne du modele, ERA5 aux noeuds (pcfno3d.forcage) et
    verite generee, toutes contre le ballon.

Les mois ou une station ne mesure que des zeros (panne d'anemometre : Nabeul juin, juillet
et septembre 2021) sont ecartes.

    python scripts/valider_grand_tunis_reel.py -c grand_tunis_v2_long_colab --obs <dossier>

<dossier> : metar_*.csv, synop_*.csv (format pcfno3d.metar / scripts/isd_synop.py) et
radiosondages_TSM00060715.csv (scripts/igra_radiosondages.py).
Sortie : run_dir/validation_reelle/<jeu>/ (stations_paires.csv, stations_bilan.csv,
radiosondages_paires.csv, radiosondages_bilan.csv, bilan.json).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcfno3d.stations_reelles import (CODES, H_STATION, cellules_codes,  # noqa: E402,F401
                                     entree_reelle, lire_observations, obs_a, vent_a_hauteur)

SITE_SONDE = "TSM00060715"             # radiosondage de Tunis-Carthage


def direction(u, v):
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def bilan_stations(p: pd.DataFrame, sources: tuple[str, ...]) -> pd.DataFrame:
    """Par station et source : n, RMSE u, v, erreur vectorielle RMS, biais de vitesse, erreur
    absolue moyenne de direction (vents mesures >= 1 m/s)."""
    lignes = []
    for station, g in p.groupby("station"):
        for src in sources:
            q = g.dropna(subset=[f"u_{src}", f"v_{src}"])
            du, dv = q[f"u_{src}"] - q["u_obs"], q[f"v_{src}"] - q["v_obs"]
            fort = np.hypot(q["u_obs"], q["v_obs"]) >= 1.0
            dd = np.abs((direction(q[f"u_{src}"], q[f"v_{src}"]) - direction(q["u_obs"], q["v_obs"])
                         + 180.0) % 360.0 - 180.0)
            lignes.append({
                "station": station, "source": src, "n": len(q),
                "rmse_u": float(np.sqrt(np.mean(du ** 2))), "rmse_v": float(np.sqrt(np.mean(dv ** 2))),
                "rmse_vect": float(np.sqrt(np.mean(du ** 2 + dv ** 2))),
                "biais_vitesse": float(np.mean(np.hypot(q[f"u_{src}"], q[f"v_{src}"])
                                               - np.hypot(q["u_obs"], q["v_obs"]))),
                "mae_direction_deg": float(dd[fort].mean()) if fort.any() else math.nan})
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------- programme
def main() -> int:
    from pcfno3d.config import Config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--obs", required=True, help="dossier des metar_*.csv, synop_*.csv, radiosondages")
    ap.add_argument("--jeu", choices=("test", "val"), default="test")
    ap.add_argument("--max-instants", type=int, default=None)
    args = ap.parse_args()
    valider(Config.load(args.config), Path(args.obs), args.jeu, args.checkpoint, args.max_instants)
    return 0


def valider(cfg, dossier_obs: Path, jeu: str = "test", checkpoint=None,
            max_instants: int | None = None, device: str | None = None) -> dict:
    import torch

    from pcfno3d.cache import load_index
    from pcfno3d.dataset import assemble_input, is_synoptic, make_splits
    from pcfno3d.geometry import N_SURF
    from pcfno3d.inference import WindFieldModel, caracteristiques
    from pcfno3d.radiosondages import bilan as bilan_sondes, colonne_aux_hauteurs

    if cfg.mask.entree != "era5_stations":
        raise SystemExit("validation reelle : modele en mode era5_stations seulement")
    wf = WindFieldModel(cfg, checkpoint=checkpoint, device=device)
    geom, norm, dev = wf.geom, wf.norm, wf.device
    if geom.forcage is None:
        raise SystemExit("ce script suppose un modele a entree ERA5 (paths.forcage_era5)")
    index = load_index(cfg)
    splits = make_splits(cfg, index)
    stems = splits.test[0] if jeu == "test" else splits.val[0]
    stems = [s for s in stems if is_synoptic(splits.stamp_of[s])]
    if max_instants:
        stems = stems[:max_instants]
    row = index.set_index("stem")
    cellules = cellules_codes(geom)
    obs = lire_observations(Path(dossier_obs))
    f_sonde = Path(dossier_obs) / f"radiosondages_{SITE_SONDE}.csv"
    sondes = pd.read_csv(f_sonde) if f_sonde.exists() else pd.DataFrame()
    if len(sondes):
        sondes["t"] = pd.to_datetime(sondes["temps"], utc=True).dt.tz_localize(None)
        sondes = sondes[sondes["h_agl_m"].between(0, 1000) & sondes["u"].notna() & sondes["v"].notna()]
    print(f"{len(stems)} instants synoptiques ({jeu}) | stations {cellules} | "
          f"{len(obs)} mesures | {sondes['t'].nunique() if len(sondes) else 0} radiosondages", flush=True)

    h_noeuds = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    valid = geom.valid

    @torch.no_grad()
    def passe(S_in, capteur, hour, doy, extra):
        tf = caracteristiques(S_in, capteur, hour, doy, geom, extra)
        s_t = torch.from_numpy(np.ascontiguousarray(S_in[:N_SURF]))[None].to(dev)
        m_t = torch.from_numpy(capteur.astype(np.float32))[None].to(dev)
        t_t = torch.from_numpy(tf)[None].to(dev)
        with torch.autocast(device_type=dev.type, dtype=wf.amp_dtype, enabled=wf.use_amp):
            pred = wf.model(assemble_input(s_t, m_t, t_t, geom, norm))
        u, v, _ = (np.where(valid, a[0].float().cpu().numpy(), np.nan) for a in norm.denorm(pred.float()))
        return u, v

    paires, p_sondes = [], []
    t0 = time.time()
    for n, stem in enumerate(stems):
        r = row.loc[stem]
        t = pd.Timestamp(r["timestamp"])
        hour, doy = float(r["hour"]), int(r["doy"])
        mesures = obs_a(obs, t)
        a_sonder = len(sondes) and t.hour in (0, 12) and (sondes["t"] == t).any()
        if not mesures and not a_sonder:
            continue
        cache = cfg.paths.cache_dir
        S = np.load(cache / f"{stem}_S.npy")
        E = np.load(cache / f"{stem}_E.npy").astype(np.float32)
        extra = geom.forcage.vecteur(t)
        for code, (uo, vo) in mesures.items():
            iy, ix = cellules[code]
            S_in, cap = entree_reelle(S, E, geom, cellules, mesures, exclue=code)
            u, v = passe(S_in, cap, hour, doy, extra)
            um, vm = vent_a_hauteur(u[:, iy, ix], v[:, iy, ix], h_noeuds[:, iy, ix], geom.z0[iy, ix])
            paires.append({"station": CODES[code], "stem": stem, "temps": t, "u_obs": uo, "v_obs": vo,
                           "u_modele": um, "v_modele": vm,
                           "u_era5": float(E[0, iy, ix]), "v_era5": float(E[1, iy, ix]),
                           "u_generateur": float(S[0, iy, ix]), "v_generateur": float(S[1, iy, ix])})
        if a_sonder:
            iy, ix = cellules["DTTA"]
            S_in, cap = entree_reelle(S, E, geom, cellules, mesures)
            u, v = passe(S_in, cap, hour, doy, extra)
            Y = np.load(cache / f"{stem}_Y.npy").astype(np.float32)
            dense = geom.interp_era5.dense_np(extra[geom.forcage.n_extra - geom.forcage.n_cellules:])
            g = sondes[sondes["t"] == t]
            h_obs = g["h_agl_m"].to_numpy(float)
            hn = h_noeuds[:, iy, ix]
            col = {"modele": (u[:, iy, ix], v[:, iy, ix]),
                   "era5": (dense[0][:, iy, ix], dense[1][:, iy, ix]),
                   "generateur": tuple(np.where(valid[:, iy, ix], Y[c][:, iy, ix], np.nan) for c in (0, 1))}
            d = {"site": "Tunis-Carthage", "temps": t, "h_agl_m": h_obs,
                 "u_obs": g["u"].to_numpy(float), "v_obs": g["v"].to_numpy(float)}
            for src, (cu, cv) in col.items():
                d[f"u_{src}"], d[f"v_{src}"] = colonne_aux_hauteurs(hn, cu, cv, h_obs)
            p_sondes.append(pd.DataFrame(d))
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(stems)} instants | {time.time() - t0:.0f} s", flush=True)

    sortie = cfg.paths.run_dir / "validation_reelle" / jeu
    sortie.mkdir(parents=True, exist_ok=True)
    sources = ("modele", "era5", "generateur")
    resume = {"jeu": jeu, "instants": len(stems), "checkpoint": str(wf.checkpoint_path.name)}
    if paires:
        p = pd.DataFrame(paires)
        p.to_csv(sortie / "stations_paires.csv", index=False)
        b = bilan_stations(p, sources)
        b.to_csv(sortie / "stations_bilan.csv", index=False)
        print("\nSTATIONS (10 m, station exclue de l'entree) :")
        print(b.round(3).to_string(index=False))
        resume["stations"] = b.to_dict(orient="records")
    if p_sondes:
        ps = pd.concat(p_sondes, ignore_index=True).dropna(subset=["u_obs", "v_obs"])
        ps.to_csv(sortie / "radiosondages_paires.csv", index=False)
        bs = bilan_sondes(ps.dropna(subset=[f"u_{s}" for s in sources]), sources)
        bs.to_csv(sortie / "radiosondages_bilan.csv", index=False)
        print("\nRADIOSONDAGES Tunis-Carthage (0-1 000 m au-dessus du sol) :")
        print(bs.round(3).to_string(index=False))
        resume["radiosondages"] = bs.to_dict(orient="records")
    (sortie / "bilan.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False, default=str),
                                       encoding="utf-8")
    print(f"\nresultats -> {sortie}")
    return resume


if __name__ == "__main__":
    raise SystemExit(main())
