"""EOL2 national (phase 7, cahier v12) : le profil du modele national contre les VRAIS radiosondages de
0 a 250 m au-dessus du sol, compare a ERA5, selon configs/national/regle_eolien_national.json
(PROPOSITION, empreinte verifiee avant toute lecture).

Sites : les radiosondages IGRA du dossier --obs (radiosondages_<site>.csv, scripts/igra_radiosondages.py)
dont le site est dans le domaine : Tunis-Carthage (TSM00060715) et Tozeur (TSM00060760) au national.
Instants : test 2022, instants synoptiques de 00 et 12 UTC ayant un radiosondage a l'heure nominale.
Entree : production (ERA5 10 m tous les pas_era5_m + tous les aeroports avec leurs METAR reels,
aucun retire). Colonnes de la maille du site : modele, ERA5 (profil aux noeuds) et verite
d'entrainement, interpolees aux hauteurs du ballon (sans extrapolation).

    python scripts/eol2_national.py -c <config> --obs <dossier> --sortie <dossier>

Sorties : eol2_paires.csv, eol2_bilan.csv (site x tranche), eol2.json (verdict), eol2.md.
Le script refuse de recalculer si eol2.json existe deja.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

REGLE = REPO / "configs" / "national" / "regle_eolien_national.json"
SOURCES = ("modele", "era5", "generateur")


def _rmse(du, dv) -> float:
    return float(np.sqrt(np.mean(np.asarray(du) ** 2 + np.asarray(dv) ** 2))) if len(du) else float("nan")


def sites_du_domaine(geom, dossier_obs: Path) -> dict[str, dict]:
    """{site: {nom, iy, ix, sondes}} : les fichiers radiosondages_<site>.csv dont le site (coordonnees
    de pcfno3d.radiosondages.SITES) tombe dans la grille."""
    from pcfno3d.radiosondages import SITES
    from pcfno3d.realdata import GeoRef

    geo = GeoRef.from_cfg(geom.cfg)
    out = {}
    for f in sorted(Path(dossier_obs).glob("radiosondages_*.csv")):
        site = f.stem.split("_", 1)[1]
        if site not in SITES:
            print(f"{f.name} : site inconnu de pcfno3d.radiosondages.SITES, ignore")
            continue
        nom, lat, lon = SITES[site]
        x, y = geo.to_xy(lat, lon)
        iy, ix = int(round(float(y) / geom.dy)), int(round(float(x) / geom.dx))
        if not (0 <= iy < geom.ny and 0 <= ix < geom.nx):
            print(f"{nom} ({site}) hors du domaine, ignore")
            continue
        s = pd.read_csv(f)
        s["t"] = pd.to_datetime(s["temps"], utc=True).dt.tz_localize(None)
        s = s[s["h_agl_m"].between(0, 1000) & s["u"].notna() & s["v"].notna()]
        out[site] = {"nom": nom, "iy": iy, "ix": ix, "sondes": s}
    return out


def verdict(p: pd.DataFrame, lo: float, hi: float, minimum: int) -> dict:
    q = p[(p["h_agl_m"] >= lo) & (p["h_agl_m"] <= hi)].dropna(
        subset=["u_obs", "v_obs", "u_modele", "v_modele", "u_era5", "v_era5"])
    n_sondes = int(q.groupby("site")["temps"].nunique().sum()) if len(q) else 0
    m = _rmse(q.u_modele - q.u_obs, q.v_modele - q.v_obs)
    e = _rmse(q.u_era5 - q.u_obs, q.v_era5 - q.v_obs)
    statut = "à mesurer" if n_sondes < minimum else ("conforme" if m <= e else "non conforme")
    par_site = {}
    for s, g in q.groupby("site"):
        par_site[s] = {"radiosondages": int(g["temps"].nunique()), "points": int(len(g)),
                       **{f"rmse_vect_{src}": _rmse(g[f"u_{src}"] - g.u_obs, g[f"v_{src}"] - g.v_obs)
                          for src in SOURCES if f"u_{src}" in g}}
    return {"tranche_m": [lo, hi], "statut": statut, "rmse_vect_modele": m, "rmse_vect_era5": e,
            "points": int(len(q)), "radiosondages": n_sondes, "minimum_radiosondages": minimum,
            "par_site": par_site,
            "note": "ERA5 assimile les radiosondages : critere non totalement independant"}


def mesurer(cfg, dossier_obs: Path, checkpoint=None, device=None, max_instants=None) -> pd.DataFrame:
    import torch

    from pcfno3d.cache import load_index
    from pcfno3d.dataset import assemble_input, is_synoptic, make_splits
    from pcfno3d.forcage import N_SCAL, charger_e
    from pcfno3d.geometry import N_SURF
    from pcfno3d.inference import WindFieldModel, caracteristiques
    from pcfno3d.radiosondages import colonne_aux_hauteurs
    from pcfno3d.stations_reelles import cellules_codes, entree_reelle, lire_observations, obs_a

    if cfg.mask.entree != "era5_stations":
        raise SystemExit("EOL2 national : modele en mode era5_stations seulement")
    wf = WindFieldModel(cfg, checkpoint=checkpoint, device=device, verbose=False)
    geom, norm, dev = wf.geom, wf.norm, wf.device
    if geom.forcage is None:
        raise SystemExit("EOL2 : modele a entree ERA5 (paths.forcage_era5) requis")
    sites = sites_du_domaine(geom, dossier_obs)
    if not sites:
        raise SystemExit(f"aucun radiosondage d'un site du domaine dans {dossier_obs}")
    cellules = cellules_codes(geom)
    obs = lire_observations(Path(dossier_obs), codes=list(cellules))
    index = load_index(cfg)
    splits = make_splits(cfg, index)
    stems = [s for s in splits.test[0] if is_synoptic(splits.stamp_of[s])]
    row = index.set_index("stem")
    a_sonder = []
    for stem in stems:
        t = pd.Timestamp(row.loc[stem, "timestamp"])
        if t.tz is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        if t.hour in (0, 12) and any((s["sondes"]["t"] == t).any() for s in sites.values()):
            a_sonder.append((stem, t))
    if max_instants:
        a_sonder = a_sonder[:max_instants]
    print(f"EOL2 : {len(a_sonder)} instants de test avec radiosondage | sites "
          + ", ".join(f"{s['nom']} ({k})" for k, s in sites.items()), flush=True)
    h_noeuds = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    valid = geom.valid
    lignes, t0 = [], time.time()
    for n, (stem, t) in enumerate(a_sonder):
        r = row.loc[stem]
        cache = cfg.paths.cache_dir
        S = np.load(cache / f"{stem}_S.npy")
        E = charger_e(geom, cache / f"{stem}_E.npy")
        extra = geom.forcage.vecteur(t)
        S_in, cap = entree_reelle(S, E, geom, cellules, obs_a(obs, t))
        tf = caracteristiques(S_in, cap, float(r["hour"]), int(r["doy"]), geom, extra)
        with torch.no_grad():
            s_t = torch.from_numpy(np.ascontiguousarray(S_in[:N_SURF]))[None].to(dev)
            m_t = torch.from_numpy(cap.astype(np.float32))[None].to(dev)
            t_t = torch.from_numpy(tf)[None].to(dev)
            with torch.autocast(device_type=dev.type, dtype=wf.amp_dtype, enabled=wf.use_amp):
                pred = wf.model(assemble_input(s_t, m_t, t_t, geom, norm))
            u_t, v_t, _ = norm.denorm(pred.float())
        u = np.where(valid, u_t[0].float().cpu().numpy(), np.nan)
        v = np.where(valid, v_t[0].float().cpu().numpy(), np.nan)
        dense = geom.interp_era5.dense_np(extra[N_SCAL:])
        Y = np.load(cache / f"{stem}_Y.npy").astype(np.float32)
        for site, s in sites.items():
            g = s["sondes"][s["sondes"]["t"] == t]
            if g.empty:
                continue
            iy, ix = s["iy"], s["ix"]
            hn = h_noeuds[:, iy, ix]
            h_obs = g["h_agl_m"].to_numpy(float)
            d = {"site": s["nom"], "code_site": site, "stem": stem, "temps": t, "h_agl_m": h_obs,
                 "u_obs": g["u"].to_numpy(float), "v_obs": g["v"].to_numpy(float)}
            col = {"modele": (u[:, iy, ix], v[:, iy, ix]),
                   "era5": tuple(np.where(valid[:, iy, ix], dense[c][:, iy, ix], np.nan) for c in (0, 1)),
                   "generateur": tuple(np.where(valid[:, iy, ix], Y[c][:, iy, ix], np.nan) for c in (0, 1))}
            for src, (cu, cv) in col.items():
                d[f"u_{src}"], d[f"v_{src}"] = colonne_aux_hauteurs(hn, cu, cv, h_obs)
            lignes.append(pd.DataFrame(d))
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(a_sonder)} instants ({time.time() - t0:.0f} s)", flush=True)
    if not lignes:
        return pd.DataFrame(columns=["site", "temps", "h_agl_m", "u_obs", "v_obs"]
                            + [f"{c}_{s}" for s in SOURCES for c in "uv"])
    return pd.concat(lignes, ignore_index=True).dropna(subset=["u_obs", "v_obs"])


def main(argv=None) -> int:
    from pcfno3d.config import Config
    from pcfno3d.radiosondages import bilan
    from pcfno3d.regles import sortie_neuve, verifier

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--obs", required=True, help="dossier : metar_*.csv et radiosondages_<site>.csv")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--max-instants", type=int, default=None, help="essai (pas un verdict)")
    args = ap.parse_args(argv)
    regle, h = verifier(REGLE)
    r = regle["eol2"]
    sortie = sortie_neuve(args.sortie, "eol2.json")
    cfg = Config.load(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg.apply_overrides({k: v})
    p = mesurer(cfg, Path(args.obs), args.checkpoint, args.device, args.max_instants)
    p.to_csv(sortie / "eol2_paires.csv", index=False)
    lo, hi = (float(x) for x in r["tranche_m"])
    minimum = 10
    v = verdict(p, lo, hi, minimum)
    tab = bilan(p.dropna(subset=[f"u_{s}" for s in SOURCES]), SOURCES) if len(p) else pd.DataFrame()
    tab.to_csv(sortie / "eol2_bilan.csv", index=False)
    out = {"regle": str(REGLE.relative_to(REPO)), "sha256_regle": h, "statut_regle": "PROPOSITION",
           "run": str(cfg.paths.run_dir), "essai": bool(args.max_instants), "EOL2": v,
           "mesures": "radiosondages IGRA (mesures) ; ERA5 et la verite generee sont la verite d'entrainement"}
    (sortie / "eol2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    L = [f"# EOL2 national (PROPOSITION, regle sha256 {h[:12]}…){' — ESSAI' if args.max_instants else ''}", "",
         f"Radiosondages {lo:g}-{hi:g} m au-dessus du sol, test 2022 : {v['radiosondages']} radiosondages, "
         f"{v['points']} points.", "",
         f"RMSE vectoriel modele {v['rmse_vect_modele']:.3f} m/s, ERA5 {v['rmse_vect_era5']:.3f} m/s : **{v['statut']}** "
         "(modele <= ERA5).", "", "| Site | Radiosondages | Modele | ERA5 | Verite d'entrainement |", "|---|---|---|---|---|"]
    for s, d in v["par_site"].items():
        L.append(f"| {s} | {d['radiosondages']} | {d.get('rmse_vect_modele', float('nan')):.3f} | "
                 f"{d.get('rmse_vect_era5', float('nan')):.3f} | {d.get('rmse_vect_generateur', float('nan')):.3f} |")
    L += ["", "ERA5 assimile les radiosondages : critere non totalement independant. Tranches 250-1000 m : "
          "eol2_bilan.csv, sans verdict."]
    (sortie / "eol2.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"-> {sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
