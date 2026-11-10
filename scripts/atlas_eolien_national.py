"""Phase 7 de la feuille de route nationale : carte du potentiel eolien de 50 a 150 m au-dessus du sol sur
toute la grille nationale (1 km), meilleures zones, et zoom Grand Tunis a 100 m, selon
configs/national/regle_eolien_national.json (PROPOSITION, empreinte verifiee avant toute lecture).

  carte   une passe du modele par instant du cache (entree de production), vitesse a 50, 100 et 150 m
          (pcfno3d.atom_wind.champ_agl), moments par groupe d'instants ; carte PRINCIPALE = validation
          + test (jamais vus a l'entrainement), carte d'INFORMATION = tous les instants ; reference
          ERA5 seule sur les memes instants ; Weibull (moments), densite de puissance, facteur de
          charge et production de l'eolienne generique avec la densite de chaque maille ; 10
          meilleures zones a 100 m.
  zoom    l'atlas du Grand Tunis a 100 m (modele livre v4, carnet OS6) contre la carte nationale a
          100 m decoupee sur la meme boite : aucune passe du modele.

    python scripts/atlas_eolien_national.py carte -c <config> --sortie <dossier> [--territoire poly.json]
    python scripts/atlas_eolien_national.py zoom --grand-tunis <runs GT>/atlas_eolien \
        --national <dossier de la carte> --sortie <dossier>

Sorties (carte) : atlas_national.json, atlas_national.npz, <couche>_<h>m.tif, atlas_national.png,
ecart_era5.png, zones.csv, resume.csv, blocs_5km.csv. (zoom) : zoom.json, zoom.png, zoom.csv.
Chaque etape refuse de recalculer une sortie existante.
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

REGLE = REPO / "configs" / "national" / "regle_eolien_national.json"
PRINCIPAL = ("val", "test")
BLOC_CSV_KM = 5.0


def _moy_terre(a, terre) -> float:
    v = np.where(terre, a, np.nan)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def etape_carte(args) -> int:
    from pcfno3d import atlas
    from pcfno3d.agriculture import latlon_noeuds
    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.regles import sortie_neuve, verifier
    from pcfno3d.sig import ecrire_geotiff, grille_geo

    regle, h_regle = verifier(REGLE)
    ra = regle["atlas_national"]
    zr = ra["meilleures_zones"]
    sortie = sortie_neuve(args.sortie, "atlas_national.json")
    cfg = Config.load(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg.apply_overrides({k: v})
    wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device, verbose=False)
    geom = wf.geom
    idx = load_index(cfg).sort_values("timestamp")
    noms = ("val", "test") if args.sans_information else ("val", "test", "train")
    groupes = {g: list(idx[idx["split"] == g]["stem"]) for g in noms}
    if args.max_instants:
        groupes = {g: s[:: max(1, len(s) // args.max_instants)][:args.max_instants] for g, s in groupes.items()}
    hauteurs = tuple(float(x) for x in ra["hauteurs_agl_m"])
    print({g: len(s) for g, s in groupes.items()}, "instants | hauteurs", hauteurs, flush=True)
    res = atlas.construire_hauteurs(wf, groupes, hauteurs, era5=True)
    courbe = atlas.charger_courbe(REPO / "configs" / "eolien" / "parc_demo.yaml", atlas.COURBE_DEFAUT)
    table = atlas.table_aep(courbe, courbe.rho_ref)
    variantes = {"principal": PRINCIPAL}
    if not args.sans_information:
        variantes["information"] = tuple(groupes)
    sources = [s for s in ("modele", "era5") if any(k[0] == s for k in res["cumuls"])]
    C = {}                                  # (source, variante, h) -> couches
    for src in sources:
        for var, gs in variantes.items():
            for h in hauteurs:
                cum = atlas.combiner(res["cumuls"], src, gs, h)
                if cum is not None and cum.a[0].max() > 0:
                    C[(src, var, h)] = atlas.couches(cum, geom, h, courbe, table)
    terre = atlas.masque_terre(geom)
    polygone = None
    if args.territoire:
        polygone = json.loads(Path(args.territoire).read_text(encoding="utf-8"))
        polygone = polygone.get("polygone", polygone) if isinstance(polygone, dict) else polygone
    h_z = float(zr["hauteur_m"])
    fc = C[("modele", "principal", h_z)]["facteur_charge"]
    zones = atlas.meilleures_zones(fc, geom, int(zr["nombre"]), float(zr["distance_min_m"]),
                                   float(zr["pente_max_deg"]), polygone=polygone)
    lat, lon = latlon_noeuds(geom)
    g = grille_geo(geom)

    # GeoTIFF : carte principale du modele, chaque couche a chaque hauteur
    tif = []
    for h in hauteurs:
        for k in atlas.COUCHES_NATIONAL:
            f = ecrire_geotiff(sortie / f"{k}_{int(h)}m.tif", C[("modele", "principal", h)][k],
                               g["lon0"], g["lat0"], g["dlon"], g["dlat"])
            tif.append(f.name)
    np.savez_compressed(sortie / "atlas_national.npz", lat=lat.astype(np.float32), lon=lon.astype(np.float32),
                        terre=terre, **{f"{s}_{v}_{int(h)}m_{k}": np.asarray(c[k], np.float32)
                                        for (s, v, h), c in C.items() for k in atlas.COUCHES_NATIONAL})
    # CSV : zones, resume par hauteur / source / variante, couches par blocs de 5 km
    lignes_z = []
    for r, (iy, ix) in enumerate(zones, 1):
        z = {"rang": r, "lat": round(float(lat[iy, ix]), 5), "lon": round(float(lon[iy, ix]), 5),
             "altitude_m": round(float(geom.zt[iy, ix]), 0), "pente_deg": round(float(geom.slope[iy, ix]), 1),
             "territoire": "dans le polygone fourni" if polygone is not None else "territoire non verifie"}
        for h in hauteurs:
            c = C[("modele", "principal", h)]
            z.update({f"vitesse_{int(h)}m": round(float(c["moyenne"][iy, ix]), 2),
                      f"densite_puissance_{int(h)}m": round(float(c["densite_puissance_w_m2"][iy, ix]), 0),
                      f"facteur_charge_{int(h)}m": round(float(c["facteur_charge"][iy, ix]), 3),
                      f"aep_mwh_{int(h)}m": round(float(c["aep_mwh"][iy, ix]), 0)})
            ce = C.get(("era5", "principal", h))
            if ce is not None:
                z[f"vitesse_era5_{int(h)}m"] = round(float(ce["moyenne"][iy, ix]), 2)
        lignes_z.append(z)
    pd.DataFrame(lignes_z).to_csv(sortie / "zones.csv", index=False)
    resume_l = [{"source": s, "carte": v, "hauteur_m": h, "instants": int(sum(res["groupes"][x] for x in variantes[v])),
                 "vitesse_moyenne_terre": round(_moy_terre(c["moyenne"], terre), 3),
                 "densite_puissance_terre_w_m2": round(_moy_terre(c["densite_puissance_w_m2"], terre), 1),
                 "facteur_charge_terre": round(_moy_terre(c["facteur_charge"], terre), 4),
                 "aep_mwh_terre": round(_moy_terre(c["aep_mwh"], terre), 0)}
                for (s, v, h), c in sorted(C.items())]
    pd.DataFrame(resume_l).to_csv(sortie / "resume.csv", index=False)
    b = max(int(round(BLOC_CSV_KM * 1000 / geom.dx)), 1)
    atlas.blocs_csv({h: C[("modele", "principal", h)] for h in hauteurs}, geom, b).to_csv(
        sortie / "blocs_5km.csv", index=False)
    _figures(C, hauteurs, zones, lat, lon, terre, sortie)

    ecarts = {}
    if not args.sans_information:
        for h in hauteurs:
            a, i = C[("modele", "principal", h)]["moyenne"], C[("modele", "information", h)]["moyenne"]
            ecarts[f"{int(h)}m"] = round(_moy_terre(i - a, terre), 3)
    out = {"regle": str(REGLE.relative_to(REPO)), "sha256_regle": h_regle, "statut_regle": "PROPOSITION",
           "run": str(cfg.paths.run_dir), "checkpoint": wf.checkpoint_path.name, "essai": bool(args.max_instants),
           "entree": ("METAR reels aux aeroports (paths.stations_obs)" if cfg.paths.stations_obs
                      else "valeur du cache a la cellule des aeroports (pas de dossier obs)"),
           "instants": res["groupes"], "carte_principale": list(PRINCIPAL),
           "hauteurs_agl_m": list(hauteurs), "eolienne": courbe.model, "rated_kw": courbe.rated_kw,
           "resume": resume_l, "zones": lignes_z, "geotiff": tif,
           "ecart_vitesse_information_moins_principale_terre_m_s": ecarts or None,
           "limites": ("carte principale = validation et test : hiver et ete seulement (janvier-mars, juillet-aout), "
                       "pas une moyenne annuelle ; aucune production mesuree ; eolienne generique (table de "
                       "demonstration) ; sillages et pertes non comptes ; densite ISA ; zones hors Tunisie possibles "
                       "sans polygone du territoire ; aide a la decision a confirmer par une campagne de mesure"),
           "verite": "aucune verite : le modele et ERA5 seule sont publies cote a cote"}
    (sortie / "atlas_national.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf-8")
    print(pd.DataFrame(resume_l).to_string(index=False))
    print(pd.DataFrame(lignes_z).to_string(index=False))
    print(f"-> {sortie}")
    return 0


def _figures(C, hauteurs, zones, lat, lon, terre, sortie: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    fig, axes = plt.subplots(2, len(hauteurs), figsize=(4.2 * len(hauteurs), 11), squeeze=False,
                             constrained_layout=True)
    for c, h in enumerate(hauteurs):
        cc = C[("modele", "principal", h)]
        for r, (cle, titre, cmap) in enumerate((("moyenne", f"vitesse moyenne a {h:g} m (m/s)", "viridis"),
                                                ("facteur_charge", f"facteur de charge a {h:g} m", "magma"))):
            ax = axes[r, c]
            im = ax.imshow(np.where(terre, cc[cle], np.nan), origin="lower", extent=ext, cmap=cmap)
            fig.colorbar(im, ax=ax, shrink=0.6)
            if h == 100.0:
                for k, (iy, ix) in enumerate(zones, 1):
                    ax.plot(lon[iy, ix], lat[iy, ix], "w*", ms=10, mec="k")
                    ax.annotate(str(k), (lon[iy, ix], lat[iy, ix]), xytext=(4, 4), textcoords="offset points",
                                color="w", fontsize=8, weight="bold")
            ax.set_title(titre, fontsize=9)
    fig.suptitle("Potentiel eolien national (modele, validation 2021 + test 2022 : hiver et ete)", fontsize=10)
    fig.savefig(sortie / "atlas_national.png", dpi=90)
    plt.close(fig)
    if all(("era5", "principal", h) in C for h in hauteurs):
        fig, axes = plt.subplots(1, len(hauteurs), figsize=(4.2 * len(hauteurs), 5.5), squeeze=False,
                                 constrained_layout=True)
        for c, h in enumerate(hauteurs):
            d = C[("modele", "principal", h)]["moyenne"] - C[("era5", "principal", h)]["moyenne"]
            lim = float(np.nanpercentile(np.abs(np.where(terre, d, np.nan)), 98)) if np.isfinite(d).any() else 1.0
            im = axes[0, c].imshow(np.where(terre, d, np.nan), origin="lower", extent=ext, cmap="RdBu_r",
                                   vmin=-lim, vmax=lim)
            fig.colorbar(im, ax=axes[0, c], shrink=0.6)
            axes[0, c].set_title(f"modele - ERA5 seule, vitesse moyenne a {h:g} m (m/s)", fontsize=9)
        fig.savefig(sortie / "ecart_era5.png", dpi=90)
        plt.close(fig)


# ---------------------------------------------------------------------- zoom Grand Tunis
def agreger(fin: np.ndarray, gf: dict, gg: dict, forme: tuple[int, int]) -> np.ndarray:
    """Moyenne de la carte fine (grille gf) dans chaque maille de la carte grossiere (grille gg,
    forme `forme`) : chaque maille fine va a la maille grossiere dont elle est la plus proche."""
    ny, nx = fin.shape
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    la, lo = gf["lat0"] + jj * gf["dlat"], gf["lon0"] + ii * gf["dlon"]
    gy = np.rint((la - gg["lat0"]) / gg["dlat"]).astype(int)
    gx = np.rint((lo - gg["lon0"]) / gg["dlon"]).astype(int)
    ok = np.isfinite(fin) & (gy >= 0) & (gy < forme[0]) & (gx >= 0) & (gx < forme[1])
    s = np.zeros(forme)
    n = np.zeros(forme)
    np.add.at(s, (gy[ok], gx[ok]), fin[ok])
    np.add.at(n, (gy[ok], gx[ok]), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan)


def etape_zoom(args) -> int:
    from pcfno3d.regles import sortie_neuve, verifier
    from pcfno3d.sig import lire_geotiff

    regle, h_regle = verifier(REGLE)
    sortie = sortie_neuve(args.sortie, "zoom.json")
    gt = Path(args.grand_tunis)
    info_gt = json.loads((gt / "atlas.json").read_text(encoding="utf-8")) if (gt / "atlas.json").exists() else None
    if info_gt is None or not (gt / "atlas_moyenne.tif").exists():
        out = {"statut": "absent", "raison": f"atlas du Grand Tunis absent dans {gt} (carnet OS6)"}
    elif abs(float(info_gt.get("hub_agl_m", -1)) - 100.0) > 1e-6:
        out = {"statut": "absent", "raison": f"atlas du Grand Tunis au moyeu {info_gt.get('hub_agl_m')} m, pas 100 m"}
    else:
        fin, gf = lire_geotiff(gt / "atlas_moyenne.tif")
        nat, gn = lire_geotiff(Path(args.national) / "moyenne_100m.tif")
        # boite du Grand Tunis dans la grille nationale
        lat_min, lon_min = gf["lat0"], gf["lon0"]
        lat_max = gf["lat0"] + (fin.shape[0] - 1) * gf["dlat"]
        lon_max = gf["lon0"] + (fin.shape[1] - 1) * gf["dlon"]
        j0 = max(int(np.floor((lat_min - gn["lat0"]) / gn["dlat"])), 0)
        j1 = min(int(np.ceil((lat_max - gn["lat0"]) / gn["dlat"])) + 1, nat.shape[0])
        i0 = max(int(np.floor((lon_min - gn["lon0"]) / gn["dlon"])), 0)
        i1 = min(int(np.ceil((lon_max - gn["lon0"]) / gn["dlon"])) + 1, nat.shape[1])
        if j1 <= j0 or i1 <= i0:
            raise SystemExit("la boite du Grand Tunis ne recoupe pas la grille nationale")
        coupe = nat[j0:j1, i0:i1]
        gc = {"lat0": gn["lat0"] + j0 * gn["dlat"], "lon0": gn["lon0"] + i0 * gn["dlon"],
              "dlat": gn["dlat"], "dlon": gn["dlon"]}
        fin_1km = agreger(fin, gf, gc, coupe.shape)
        ok = np.isfinite(fin_1km) & np.isfinite(coupe)
        d = (coupe - fin_1km)[ok]
        out = {"statut": "calcule", "mailles_communes_1km": int(ok.sum()),
               "vitesse_moyenne_national_m_s": float(coupe[ok].mean()) if ok.any() else None,
               "vitesse_moyenne_grand_tunis_1km_m_s": float(fin_1km[ok].mean()) if ok.any() else None,
               "ecart_moyen_national_moins_gt_m_s": float(d.mean()) if ok.any() else None,
               "ecart_rms_m_s": float(np.sqrt(np.mean(d ** 2))) if ok.any() else None,
               "correlation": float(np.corrcoef(coupe[ok], fin_1km[ok])[0, 1]) if ok.sum() > 2 else None,
               "grand_tunis": {"source": str(gt), "instants": info_gt.get("instants"), "courbe": info_gt.get("courbe"),
                               "sites": info_gt.get("sites")},
               "limite": "instants differents (Grand Tunis : son cache ; national : validation + test) ; sans verdict"}
        if args.decision_gt and Path(args.decision_gt).exists():
            out["grand_tunis"]["modele_livre"] = json.loads(Path(args.decision_gt).read_text()).get("modele_livre")
        pd.DataFrame({"lat": (gc["lat0"] + np.nonzero(ok)[0] * gc["dlat"]).round(5),
                      "lon": (gc["lon0"] + np.nonzero(ok)[1] * gc["dlon"]).round(5),
                      "vitesse_national_100m": coupe[ok].round(3),
                      "vitesse_grand_tunis_100m_agregee": fin_1km[ok].round(3)}).to_csv(sortie / "zoom.csv", index=False)
        _figure_zoom(fin, gf, coupe, gc, fin_1km, sortie / "zoom.png")
    out.update(regle=str(REGLE.relative_to(REPO)), sha256_regle=h_regle, statut_regle="PROPOSITION")
    (sortie / "zoom.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "grand_tunis"}, indent=1, ensure_ascii=False))
    print(f"-> {sortie}")
    return 0


def _figure_zoom(fin, gf, coupe, gc, fin_1km, chemin: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def ext(a, g):
        return (g["lon0"] - g["dlon"] / 2, g["lon0"] + (a.shape[1] - 0.5) * g["dlon"],
                g["lat0"] - g["dlat"] / 2, g["lat0"] + (a.shape[0] - 0.5) * g["dlat"])
    vals = np.concatenate([x[np.isfinite(x)] for x in (fin, coupe)]) if np.isfinite(coupe).any() else fin[np.isfinite(fin)]
    lo, hi = (float(np.percentile(vals, 2)), float(np.percentile(vals, 98))) if vals.size else (0.0, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (a, g, t) in zip(axes[:2], ((fin, gf, "Grand Tunis v4, 100 m (m/s, moyeu 100 m)"),
                                        (coupe, gc, "national, 1 km (m/s, 100 m)"))):
        im = ax.imshow(a, origin="lower", extent=ext(a, g), cmap="viridis", vmin=lo, vmax=hi)
        ax.set_title(t, fontsize=9)
    fig.colorbar(im, ax=axes[:2], shrink=0.8)
    d = coupe - fin_1km
    lim = float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 1.0
    im = axes[2].imshow(d, origin="lower", extent=ext(d, gc), cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title("national - Grand Tunis agrege a 1 km (m/s)", fontsize=9)
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="etape", required=True)
    a = sp.add_parser("carte")
    a.add_argument("-c", "--config", required=True)
    a.add_argument("--set", action="append", default=[])
    a.add_argument("--checkpoint")
    a.add_argument("--device")
    a.add_argument("--territoire", default=None, help="JSON [[lat, lon], ...] du territoire (facultatif)")
    a.add_argument("--sans-information", action="store_true", help="pas de carte sur tous les instants")
    a.add_argument("--max-instants", type=int, default=None, help="par groupe ; essai, pas la carte de la regle")
    a.add_argument("--sortie", required=True)
    b = sp.add_parser("zoom")
    b.add_argument("--grand-tunis", required=True, help="dossier atlas_eolien du Grand Tunis (carnet OS6)")
    b.add_argument("--national", required=True, help="dossier de l'etape carte")
    b.add_argument("--decision-gt", default=None, help="runs/decision_v4.json du Grand Tunis (provenance)")
    b.add_argument("--sortie", required=True)
    args = ap.parse_args(argv)
    return etape_carte(args) if args.etape == "carte" else etape_zoom(args)


if __name__ == "__main__":
    raise SystemExit(main())
