"""Phase 1 nationale, diagnostic 5 : coherence du vent ERA5 vu par le modele et par le generateur.

Regle : configs/national/regle_phase1_complements.json (empreinte verifiee avant tout calcul).
Cote modele : ForcageERA5.vecteur(horodatage) -> InterpERA5.dense_np, exactement ce que le chargeur et
assemble_input donnent au reseau (canaux ERA5 de l'entree, base du residuel u, v = ERA5 + correction).
Cote generateur : DayStateERA5.champ_cible(t) -> _era5_au_sol aux lat / lon des noeuds de la grille et a
h = max(z - relief, MIN_ZREL_NODE), le vent du recalage de grande echelle.
Controles : ordre des latitudes, instant UTC, signe de v (npz a 10 m contre fichier ERA5 brut u10 / v10),
hauteurs au-dessus du sol, position des noeuds.

    python scripts/diagnostic_coherence_era5.py --sortie docs/resultats/national/phase1_diag5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
NAT = REPO / "artifacts" / "national"
REGLE = REPO / "configs" / "national" / "regle_phase1_complements.json"
ERA5_BRUT = Path.home() / "AppData" / "Local" / "pcfno3d" / "era5" / "tunisie"
HAUTEURS = (300.0, 800.0)


def verifier_regle() -> str:
    h = hashlib.sha256(REGLE.read_bytes()).hexdigest()
    attendu = (REGLE.parent / (REGLE.name + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"regle modifiee depuis son hachage : {h} != {attendu}")
    return h


def config_locale(sortie: Path) -> Path:
    """La config nationale, chemins vers la grille et le forcage locaux (jeu d'essai = grille complete)."""
    import yaml
    c = yaml.safe_load((REPO / "configs" / "national_1km_colab.yaml").read_text(encoding="utf-8"))
    c["paths"].update({"data_dir": str(NAT / "essai_jeu"), "cache_dir": str(NAT / "essai_cache"),
                       "run_dir": str(sortie / "run"), "forcage_era5": str(NAT / "forcage_era5.csv"),
                       "local_dir": None, "web_dir": str(sortie / "web")})
    f = sortie / "national_diag5.yaml"
    f.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sortie", default=str(REPO / "docs" / "resultats" / "national" / "phase1_diag5"))
    a = ap.parse_args()
    h_regle = verifier_regle()
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)

    import calibrer_generateur_stations as C
    import calibrer_verite_v3 as V
    info = json.loads((NAT / "grille" / "grille_reelle.json").read_text(encoding="utf-8"))
    C._env(info, 0.4)
    import generate_aneo_3d as G
    from pcfno3d.config import Config
    from pcfno3d.forcage import metres_par_degre
    from pcfno3d.geometry import Geometry

    cfg = Config.load(str(config_locale(out)))
    geom = Geometry(cfg, device="cpu")
    itp, fo = geom.interp_era5, geom.forcage

    df = G.load_or_build_grid(NAT / "grille" / "aneo_grid_points_surface.csv", np.random.default_rng(0))
    S = G.build_static(df)
    F = G.lire_forcage(str(NAT / "forcage_era5.csv"))
    CH = G.lire_champs(str(NAT / "forcage_era5.csv"))
    k = G.STRIDE
    lat_v, lon_v = S["lat"][::k, ::k][:G.NY_V, :G.NX_V], S["lon"][::k, ::k][:G.NY_V, :G.NX_V]
    zt_v = S["zt"][::k, ::k][:G.NY_V, :G.NX_V]
    Z = np.asarray(G.Z_LEVELS, float)
    valid = np.asarray(geom.valid, bool)

    ctrl = {}
    # ---- position des noeuds, relief, hauteurs
    ctrl["noeuds_meme_forme"] = list(lat_v.shape) == [geom.ny, geom.nx]
    ctrl["ecart_max_lat_deg"] = float(np.abs(itp.lat - lat_v).max())
    ctrl["ecart_max_lon_deg"] = float(np.abs(itp.lon - lon_v).max())
    m_lon, m_lat = metres_par_degre(float(lat_v.mean()))
    ctrl["ecart_max_position_m"] = float(np.hypot((itp.lat - lat_v) * m_lat, (itp.lon - lon_v) * m_lon).max())
    ctrl["ecart_max_relief_m"] = float(np.abs(np.asarray(geom.zt) - zt_v).max())
    h_gen = np.clip(np.maximum(Z[:, None, None] - zt_v[None], G.MIN_ZREL_NODE), CH["hauteurs_sol"][0],
                    CH["hauteurs_sol"][-1])
    hs = fo.hauteurs
    h_mod = hs[itp.k] + itp.t * (hs[itp.k + 1] - hs[itp.k])
    ctrl["ecart_max_hauteur_sol_m"] = float(np.abs(h_mod - h_gen)[valid].max())
    from pcfno3d.forcage import MIN_ZREL_NODE as MZ
    ctrl["min_zrel_modele_generateur"] = [float(MZ), float(G.MIN_ZREL_NODE)]
    ctrl["lat_forcage_croissantes"] = bool(np.all(np.diff(fo.lat) > 0))
    ctrl["meme_npz_modele_generateur"] = bool(np.array_equal(fo.lat, CH["lat"]) and np.array_equal(fo.lon, CH["lon"])
                                              and np.array_equal(fo.temps, CH["temps"]))

    # ---- instants (validation 2021, graine 2028)
    insts = V.instants(20, 2028)
    lignes, niv = [], {}
    for jour, hh in insts:
        stamp = jour + pd.Timedelta(hours=hh)
        # cote generateur : le jour et l'heure, comme compute_fields (DayStateERA5.champ_cible)
        st = G.DayStateERA5(0, pd.Timestamp(jour), 0, S, 1, F, CH)
        cu, cv = st.champ_cible(hh * 3600.0)
        gen = np.stack([np.stack([G._era5_au_sol(CH, c, lat_v, lon_v, h_gen[kz]) for kz in range(len(Z))])
                        for c in (cu, cv)])
        vec = fo.vecteur(stamp.strftime("%Y-%m-%d %H:%M:%S"))
        mod = itp.dense_np(vec[len(vec) - fo.n_cellules:])
        d = mod.astype(np.float64) - gen
        for c, cn in ((0, "u"), (1, "v")):
            for kz in range(len(Z)):
                m = valid[kz]
                s = niv.setdefault((cn, f"{Z[kz]:g} m ASL"), np.zeros(3))
                s += [(d[c, kz][m] ** 2).sum(), d[c, kz][m].sum(), m.sum()]
            for hz in HAUTEURS:
                hh_ = np.full(lat_v.shape, hz)
                g2 = G._era5_au_sol(CH, (cu, cv)[c], lat_v, lon_v, hh_)
                kk = np.clip(np.searchsorted(hs, hz) - 1, 0, len(hs) - 2)
                tt = (hz - hs[kk]) / (hs[kk + 1] - hs[kk])
                cel = vec[len(vec) - fo.n_cellules:].reshape(fo.forme_cellules)[c]
                pl = (1 - tt) * cel[kk] + tt * cel[kk + 1]
                m2 = (1 - itp.a) * (1 - itp.b) * pl[itp.iy, itp.ix] + (1 - itp.a) * itp.b * pl[itp.iy, itp.ix + 1] \
                    + itp.a * (1 - itp.b) * pl[itp.iy + 1, itp.ix] + itp.a * itp.b * pl[itp.iy + 1, itp.ix + 1]
                dd = m2 - g2
                s = niv.setdefault((cn, f"{hz:g} m au-dessus du sol"), np.zeros(3))
                s += [(dd ** 2).sum(), dd.sum(), dd.size]
        lignes.append({"instant_utc": str(stamp), "rmse_u": float(np.sqrt((d[0][valid] ** 2).mean())),
                       "rmse_v": float(np.sqrt((d[1][valid] ** 2).mean())),
                       "biais_u": float(d[0][valid].mean()), "biais_v": float(d[1][valid].mean())})
        print(lignes[-1], flush=True)

    t = pd.DataFrame([{"composante": c, "niveau": n, "rmse": float(np.sqrt(s[0] / s[2])), "biais": float(s[1] / s[2]),
                       "points": int(s[2])} for (c, n), s in niv.items()])
    t.to_csv(out / "coherence_par_niveau.csv", index=False)
    pd.DataFrame(lignes).to_csv(out / "coherence_par_instant.csv", index=False)

    # ---- npz contre fichier ERA5 brut (u10, v10) : ordre des latitudes, instant, signe de v
    import xarray as xr
    brut = []
    fichiers = sorted(ERA5_BRUT.glob("era5_sl_2021_m01-06.nc")) or sorted(ERA5_BRUT.glob("era5_sl_2021*.nc"))
    ds = xr.open_dataset(fichiers[0])
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    k10 = int(np.argmin(np.abs(fo.hauteurs - 10.0)))
    for jour, hh in insts[:3]:
        stamp = jour + pd.Timedelta(hours=hh)
        it = int(np.searchsorted(fo.temps, stamp.value // 10 ** 9))
        for la, lo in ((30.5, 8.0), (34.0, 9.5), (37.25, 10.25)):
            iy, ix = int(np.argmin(np.abs(fo.lat - la))), int(np.argmin(np.abs(fo.lon - lo)))
            r = ds.sel({tdim: stamp, "latitude": fo.lat[iy], "longitude": fo.lon[ix]}, method="nearest")
            brut.append({"instant_utc": str(stamp), "temps_npz": str(pd.Timestamp(fo.temps[it], unit="s")),
                         "temps_brut": str(pd.Timestamp(r[tdim].values)), "lat": float(fo.lat[iy]),
                         "lon": float(fo.lon[ix]), "u10_npz": float(fo.u[it, k10, iy, ix]),
                         "u10_brut": float(r["u10"]), "v10_npz": float(fo.v[it, k10, iy, ix]),
                         "v10_brut": float(r["v10"])})
    b = pd.DataFrame(brut)
    b.to_csv(out / "npz_contre_brut.csv", index=False)
    ctrl["fichier_brut"] = fichiers[0].name
    ctrl["lat_brut_decroissantes"] = bool(ds["latitude"].values[0] > ds["latitude"].values[-1])
    ctrl["instants_npz_egaux_brut"] = bool((b.temps_npz == b.temps_brut).all() and (b.temps_npz == b.instant_utc).all())
    ctrl["ecart_max_u10_npz_brut"] = float((b.u10_npz - b.u10_brut).abs().max())
    ctrl["ecart_max_v10_npz_brut"] = float((b.v10_npz - b.v10_brut).abs().max())
    ctrl["signe_v_identique"] = bool((np.sign(b.v10_npz) == np.sign(b.v10_brut)).all())
    idx = pd.read_csv(NAT / "essai_cache" / "index.csv")
    ctrl["horodatage_cache_contre_generateur"] = [
        {"stem": s, "timestamp_index": ts,
         "instant_generateur": str(G.JOUR_0_ERA5 + pd.Timedelta(days=int(d) - 1) + pd.Timedelta(hours=float(h)))}
        for s, ts, d, h in idx[["stem", "timestamp", "day", "hour"]].itertuples(index=False)]
    ctrl["horodatage_cache_coherent"] = all(c["timestamp_index"] == c["instant_generateur"]
                                            for c in ctrl["horodatage_cache_contre_generateur"])

    ok_niv = bool((t.rmse <= 0.01).all() and (t.biais.abs() <= 0.005).all())
    ok_ctrl = (ctrl["lat_forcage_croissantes"] and ctrl["instants_npz_egaux_brut"] and ctrl["signe_v_identique"]
               and ctrl["ecart_max_u10_npz_brut"] <= 0.01 and ctrl["ecart_max_v10_npz_brut"] <= 0.01
               and ctrl["horodatage_cache_coherent"] and ctrl["noeuds_meme_forme"])
    verdict = {"regle": "configs/national/regle_phase1_complements.json", "sha256_regle": h_regle,
               "instants": [str(j + pd.Timedelta(hours=h)) for j, h in insts],
               "rmse_max": float(t.rmse.max()), "biais_max_abs": float(t.biais.abs().max()),
               "controles": ctrl, "verdict": "COHERENT" if ok_niv and ok_ctrl else "BUG"}
    (out / "verdict_diag5.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    print(b.round(3).to_string(index=False))
    print(json.dumps({k_: v_ for k_, v_ in verdict.items() if k_ != "instants"}, indent=2, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
