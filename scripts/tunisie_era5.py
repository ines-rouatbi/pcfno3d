"""Modele national « Tunisie ERA5 » : METAR des aeroports -> vent 3D sur toute la Tunisie.

Etapes (pcfno3d.tunisie decrit le modele) :
    python scripts/tunisie_era5.py masque        # masque terre / mer ERA5 (une requete CDS)
    python scripts/tunisie_era5.py preparer      # ERA5 + METAR -> jeu.npz
    python scripts/tunisie_era5.py entrainer     # 2021-2023 ; validation 2024
    python scripts/tunisie_era5.py evaluer       # test 2025 contre climatologie et IDW
    python scripts/tunisie_era5.py predire       # dernier instant des METAR collectes

Fichiers dans --dossier (defaut : %LOCALAPPDATA%/pcfno3d/tunisie) ; ERA5 lu dans --era5
(download_reanalysis.py --bbox 37.75,7.25,30.0,11.75 --hours 0,3,...,21).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]
BASE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "pcfno3d"
BBOX = [37.75, 7.25, 30.0, 11.75]


def cmd_masque(args) -> int:
    import cdsapi

    cible = Path(args.era5) / "masque_terre_mer.nc"
    if cible.exists():
        print(f"deja la : {cible}")
        return 0
    req = {"product_type": ["reanalysis"], "variable": ["land_sea_mask"], "year": ["2025"],
           "month": ["01"], "day": ["01"], "time": ["00:00"], "area": BBOX,
           "data_format": "netcdf", "download_format": "unarchived"}
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", req, str(cible))
    print(f"-> {cible}")
    return 0


def _masque_terre(era5: Path, ny: int, nx: int) -> np.ndarray | None:
    f = Path(era5) / "masque_terre_mer.nc"
    if not f.exists():
        return None
    import xarray as xr

    with xr.open_dataset(f) as d:
        m = d["lsm"].values.reshape(-1, ny, nx)[0]
    return m >= 0.5


def cmd_preparer(args) -> int:
    import pandas as pd
    from pcfno3d import metar, tunisie

    t0 = time.time()
    era = tunisie.lire_era5(Path(args.era5))
    Y = np.concatenate([era["U"], era["V"]], 1)                     # (t, 12, y, x)
    print(f"ERA5 : {len(era['temps'])} instants {era['temps'][0]} -> {era['temps'][-1]}, "
          f"grille {Y.shape[-2]} x {Y.shape[-1]}, cibles connues {np.isfinite(Y).mean():.1%}")
    fichiers = sorted(Path(args.metar).glob("metar_*.csv"))
    obs = tunisie.lire_metar(fichiers)
    stations = list(metar.STATIONS_TUNISIE)
    X = tunisie.entree_stations(obs, era["temps"], stations)
    coord = pd.concat([pd.read_csv(f, usecols=["station_id", "lat", "lon"]) for f in fichiers]
                      ).groupby("station_id")[["lat", "lon"]].median()
    st_lat = [float(coord.loc[s, "lat"]) for s in stations]
    st_lon = [float(coord.loc[s, "lon"]) for s in stations]
    print(f"METAR : {len(obs)} mesures ; presence par station a +/- 30 min : "
          + ", ".join(f"{s} {X[:, k, -1].mean():.0%}" for k, s in enumerate(stations)))
    terre = _masque_terre(Path(args.era5), *Y.shape[-2:])
    sortie = Path(args.dossier) / "jeu.npz"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sortie, X=X, Y=Y.astype(np.float32), temps=era["temps"].astype("int64"),
                        lat=era["lat"], lon=era["lon"], orographie_m=era["orographie_m"],
                        stations=np.array(stations), st_lat=st_lat, st_lon=st_lon,
                        terre=terre if terre is not None else np.ones(Y.shape[-2:], bool))
    print(f"-> {sortie} ({sortie.stat().st_size / 1e6:.0f} Mo, {time.time() - t0:.0f} s)"
          + ("" if terre is not None else " ; masque terre absent : toutes les mailles"))
    return 0


def _charger(dossier: Path) -> dict:
    d = dict(np.load(Path(dossier) / "jeu.npz", allow_pickle=False))
    d["temps"] = d["temps"].astype("datetime64[s]")
    return d


def _decoupe(d: dict, partie: str) -> np.ndarray:
    from pcfno3d.tunisie import ANNEES, annee_de

    return np.isin(annee_de(d["temps"]), ANNEES[partie])


def _predire(modele, norm, X, tv, lot: int = 256) -> np.ndarray:
    import torch

    modele.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), lot):
            x = torch.from_numpy(norm.entree(X[i:i + lot]))
            p = modele(x, torch.from_numpy(tv[i:i + lot])).numpy()
            out.append(p * norm.y_ect[None, :, None, None] + norm.y_moy[None, :, None, None])
    return np.concatenate(out).astype(np.float32)


def cmd_entrainer(args) -> int:
    import torch
    from pcfno3d import tunisie

    torch.manual_seed(args.graine)
    np.random.seed(args.graine)
    d = _charger(Path(args.dossier))
    tr, va = _decoupe(d, "entrainement"), _decoupe(d, "validation")
    X, Y, tv = d["X"], d["Y"], tunisie.variables_temps(d["temps"])
    norm = tunisie.Normalisation.depuis(X[tr], Y[tr])
    Xn = torch.from_numpy(norm.entree(X))
    Yn = torch.from_numpy(((Y - norm.y_moy[None, :, None, None]) / norm.y_ect[None, :, None, None]).astype(np.float32))
    TV = torch.from_numpy(tv)
    itr, iva = np.where(tr)[0], np.where(va)[0]
    S, F = X.shape[1], X.shape[2]
    modele = tunisie.construire_modele(S, F, Y.shape[2], Y.shape[3], Y.shape[1])
    opt = torch.optim.AdamW(modele.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoques)
    gen = torch.Generator().manual_seed(args.graine)
    meilleur, attente, journal = np.inf, 0, []
    chemin = Path(args.dossier) / "modele.pt"
    print(f"entrainement {len(itr)} instants, validation {len(iva)} ; "
          f"{sum(p.numel() for p in modele.parameters()) / 1e6:.2f} M parametres")
    for ep in range(1, args.epoques + 1):
        t0 = time.time()
        modele.train()
        perm = itr[torch.randperm(len(itr), generator=gen).numpy()]
        pertes = []
        for i in range(0, len(perm), args.lot):
            b = perm[i:i + args.lot]
            x = tunisie.masquer_stations(Xn[b], args.panne_max, gen)
            loss = tunisie.perte_masquee(modele(x, TV[b]), Yn[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            pertes.append(float(loss))
        sched.step()
        modele.eval()
        with torch.no_grad():
            pv = torch.cat([modele(Xn[iva[i:i + 512]], TV[iva[i:i + 512]]) for i in range(0, len(iva), 512)])
            val = float(tunisie.perte_masquee(pv, Yn[iva]))
        journal.append({"epoque": ep, "perte": float(np.mean(pertes)), "validation": val})
        mieux = val < meilleur
        if mieux:
            meilleur, attente = val, 0
            torch.save({"etat": modele.state_dict(), "normalisation": norm.etat(),
                        "forme": [S, F, int(Y.shape[2]), int(Y.shape[3]), int(Y.shape[1])],
                        "stations": d["stations"].tolist(), "st_lat": d["st_lat"].tolist(),
                        "st_lon": d["st_lon"].tolist(), "lat": d["lat"].tolist(),
                        "lon": d["lon"].tolist(), "hauteurs_agl_m": list(tunisie.HAUTEURS_AGL),
                        "epoque": ep, "validation_mse_norm": val}, chemin)
        else:
            attente += 1
        print(f"epoque {ep:3d} perte {np.mean(pertes):.4f} validation {val:.4f}"
              f"{' *' if mieux else ''} ({time.time() - t0:.0f} s)", flush=True)
        if attente >= args.patience:
            print(f"arret : {args.patience} epoques sans progres")
            break
    (Path(args.dossier) / "journal_entrainement.json").write_text(json.dumps(journal, indent=1))
    print(f"meilleur modele (validation {meilleur:.4f}) -> {chemin}")
    return 0


def _charger_modele(dossier: Path):
    import torch
    from pcfno3d import tunisie

    ck = torch.load(Path(dossier) / "modele.pt", map_location="cpu", weights_only=False)
    m = tunisie.construire_modele(*ck["forme"])
    m.load_state_dict(ck["etat"])
    return m, tunisie.Normalisation.charger(ck["normalisation"]), ck


def cmd_evaluer(args) -> int:
    from pcfno3d import tunisie

    d = _charger(Path(args.dossier))
    tr, te = _decoupe(d, "entrainement"), _decoupe(d, "test")
    modele, norm, ck = _charger_modele(Path(args.dossier))
    X, Y = d["X"][te], d["Y"][te]
    preds = {
        "modele": _predire(modele, norm, X, tunisie.variables_temps(d["temps"][te])),
        "climatologie": tunisie.reference_climatologie(d["Y"][tr], d["temps"][tr], d["temps"][te]),
        "idw_loi_puissance": tunisie.reference_idw(X, d["st_lat"], d["st_lon"], d["lat"], d["lon"]),
    }
    res = {"instants_test": int(te.sum()), "epoque": ck["epoque"],
           "mailles": {"toutes": int(d["terre"].size), "terre": int(d["terre"].sum())}}
    for zone, masque in (("terre", d["terre"]), ("toutes", None)):
        res[zone] = {k: tunisie.rmse_par_niveau(p, Y, masque) for k, p in preds.items()}
    out = Path(args.dossier) / "evaluation_test_2025.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"test 2025 : {res['instants_test']} instants ; RMSE de la vitesse (m/s), mailles terrestres")
    print("| hauteur | modele | climatologie | IDW + loi puissance |")
    print("|---|---|---|---|")
    for h in res["terre"]["modele"]:
        r = [res["terre"][k][h]["vitesse"] for k in ("modele", "climatologie", "idw_loi_puissance")]
        print(f"| {h} | {r[0]:.2f} | {r[1]:.2f} | {r[2]:.2f} |")
    print(f"-> {out}")
    return 0


def cmd_predire(args) -> int:
    import pandas as pd
    from pcfno3d import tunisie

    modele, norm, ck = _charger_modele(Path(args.dossier))
    fichiers = sorted(Path(args.metar_temps_reel).glob("metar_*.csv"))[-2:]
    obs = tunisie.lire_metar(fichiers)
    instant = pd.Timestamp(args.instant) if args.instant else obs["t"].max().floor("h")
    temps = np.array([np.datetime64(instant.to_datetime64(), "s")])
    X = tunisie.entree_stations(obs, temps, ck["stations"])
    p = _predire(modele, norm, X, tunisie.variables_temps(temps))[0]
    H = len(ck["hauteurs_agl_m"])
    nom = f"tunisie_{instant:%Y%m%d_%H%M}"
    sortie = Path(args.dossier) / "predictions" / f"{nom}.npz"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sortie, u=p[:H], v=p[H:], lat=ck["lat"], lon=ck["lon"],
                        hauteurs_agl_m=ck["hauteurs_agl_m"], stations_presentes=X[0, :, -1])
    n = int(X[0, :, -1].sum())
    vit = np.hypot(p[:H], p[H:])
    print(f"{instant} UTC : {n}/{len(ck['stations'])} stations ; vitesse moyenne "
          + ", ".join(f"{int(h)} m {vit[k].mean():.1f}" for k, h in enumerate(ck["hauteurs_agl_m"]))
          + f" m/s -> {sortie}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lat, lon = np.asarray(ck["lat"]), np.asarray(ck["lon"])
        jeu = Path(args.dossier) / "jeu.npz"
        terre = np.load(jeu)["terre"] if jeu.exists() else None
        fig, axes = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)
        for ax, k in zip(axes, (0, 3)):
            im = ax.pcolormesh(lon, lat, vit[k], shading="nearest", cmap="viridis")
            if terre is not None:
                ax.contour(lon, lat, terre.astype(float), levels=[0.5], colors="k", linewidths=0.8)
            ax.quiver(lon, lat, p[k], p[H + k], color="w", scale=150, width=0.003)
            ax.scatter(ck["st_lon"], ck["st_lat"], c=["r" if x else "0.5" for x in X[0, :, -1]], s=18)
            ax.set_title(f"{int(ck['hauteurs_agl_m'][k])} m au-dessus du sol")
            ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
            fig.colorbar(im, ax=ax, label="vitesse (m/s)", shrink=0.6)
        fig.suptitle(f"Tunisie ERA5 — reconstruction depuis les METAR, {instant:%Y-%m-%d %H:%M} UTC")
        fig.savefig(sortie.with_suffix(".png"), dpi=110)
        print(f"carte -> {sortie.with_suffix('.png')}")
    except ImportError:
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("commande", choices=("masque", "preparer", "entrainer", "evaluer", "predire"))
    ap.add_argument("--dossier", default=str(BASE / "tunisie"))
    ap.add_argument("--era5", default=str(BASE / "era5" / "tunisie"))
    ap.add_argument("--metar", default=str(BASE / "metar" / "historique"))
    ap.add_argument("--metar-temps-reel", default=str(BASE / "metar"))
    ap.add_argument("--instant", help="predire : instant UTC (defaut : le plus recent)")
    ap.add_argument("--epoques", type=int, default=80)
    ap.add_argument("--lot", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--panne-max", type=float, default=0.5, help="part max de stations masquees")
    ap.add_argument("--graine", type=int, default=2026)
    args = ap.parse_args()
    return {"masque": cmd_masque, "preparer": cmd_preparer, "entrainer": cmd_entrainer,
            "evaluer": cmd_evaluer, "predire": cmd_predire}[args.commande](args)


if __name__ == "__main__":
    raise SystemExit(main())
