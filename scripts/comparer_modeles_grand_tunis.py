"""Ecart entre deux modeles Grand Tunis (ex. v3 affine sur les stations - v2) a 10 m et 100 m
au-dessus du sol, en carte et par distance aux stations.

Question : l'affinage sur les stations corrige-t-il le champ ENTIER, ou laisse-t-il des
« taches » autour des stations (correction de site qui ne se propage pas) ? Sur des instants
synoptiques de test, avec l'entree de production (ERA5 10 m + les VRAIS anemometres, toutes
stations en entree), on calcule pour chaque modele le vent a 10 m et a 100 m au-dessus du sol,
puis la moyenne de l'ecart vectoriel |B - A| et de l'ecart de vitesse (B - A), carte par carte.

    python scripts/comparer_modeles_grand_tunis.py --conf-a grand_tunis_v2_long_colab \
        --conf-b grand_tunis_v3_stations_colab --obs /content/obs --sortie <dossier>

Sortie : ecart_<b>_moins_<a>.png (4 cartes, stations marquees), ecart_<b>_moins_<a>.json
(moyennes par anneau de distance a chaque station : 0-1, 1-3, 3-10, > 10 km).
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

ANNEAUX_KM = ((0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 1e9))
HAUTEURS = (10.0, 100.0)


def champ_agl(u: np.ndarray, v: np.ndarray, geom, h: float) -> tuple[np.ndarray, np.ndarray]:
    """u, v (nz, ny, nx ; NaN sous le relief) a h m au-dessus du sol, colonne par colonne :
    interpolation entre deux noeuds valides, ou profil log depuis le noeud valide le plus bas
    (z0 de la maille) si h est sous ce noeud — comme pcfno3d.stations_reelles.poids_10m."""
    valid = geom.valid
    hk = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    uo = np.full(u.shape[1:], np.nan)
    vo = np.full(u.shape[1:], np.nan)
    for k in range(len(geom.z_levels) - 1):
        m = valid[k] & valid[k + 1] & (hk[k] <= h) & (h <= hk[k + 1]) & np.isnan(uo)
        a = np.where(m, (h - hk[k]) / np.maximum(hk[k + 1] - hk[k], 1e-6), 0.0)
        uo = np.where(m, (1 - a) * u[k] + a * u[k + 1], uo)
        vo = np.where(m, (1 - a) * v[k] + a * v[k + 1], vo)
    k0 = np.argmax(valid, axis=0)
    a_un = valid.any(axis=0)
    h0 = np.take_along_axis(hk, k0[None], 0)[0]
    bas = a_un & (h < h0) & np.isnan(uo)
    z0 = np.maximum(geom.z0, 1e-4)
    f = np.where(bas, np.log(h / z0) / np.log(np.maximum(h0, 1.01 * z0) / z0), 0.0)
    u0 = np.take_along_axis(u, k0[None], 0)[0]
    v0 = np.take_along_axis(v, k0[None], 0)[0]
    uo = np.where(bas, f * u0, uo)
    vo = np.where(bas, f * v0, vo)
    return uo, vo


def stats_anneaux(carte: np.ndarray, geom, cellules: dict) -> dict:
    """Moyenne de la carte par anneau de distance a chaque station."""
    jj, ii = np.meshgrid(np.arange(geom.ny), np.arange(geom.nx), indexing="ij")
    out = {}
    for code, (iy, ix) in cellules.items():
        d = np.hypot((jj - iy) * geom.dy, (ii - ix) * geom.dx) / 1000.0
        out[code] = {f"{a:g}-{b:g} km" if b < 1e8 else f"> {a:g} km":
                     float(np.nanmean(np.where((d >= a) & (d < b), carte, np.nan)))
                     for a, b in ANNEAUX_KM}
    out["domaine"] = float(np.nanmean(carte))
    return out


def main() -> int:
    import torch

    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.dataset import assemble_input, is_synoptic, make_splits
    from pcfno3d.geometry import N_SURF
    from pcfno3d.inference import WindFieldModel, caracteristiques
    from pcfno3d.stations_reelles import CODES, cellules_codes, entree_reelle, lire_observations, obs_a

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--conf-a", required=True, help="modele de reference (ex. v2)")
    ap.add_argument("--conf-b", required=True, help="modele compare (ex. v3)")
    ap.add_argument("--obs", required=True)
    ap.add_argument("--n-instants", type=int, default=30)
    ap.add_argument("--sortie", required=True)
    args = ap.parse_args()

    cfg_a, cfg_b = Config.load(args.conf_a), Config.load(args.conf_b)
    wa = WindFieldModel(cfg_a)
    wb = WindFieldModel(cfg_b)
    geom, dev = wa.geom, wa.device
    index = load_index(cfg_a)
    splits = make_splits(cfg_a, index)
    stems = [s for s in splits.test[0] if is_synoptic(splits.stamp_of[s])]
    stems = [stems[i] for i in np.unique(np.linspace(0, len(stems) - 1, args.n_instants).astype(int))]
    row = index.set_index("stem")
    cellules = cellules_codes(geom)
    obs = lire_observations(Path(args.obs))

    @torch.no_grad()
    def passe(wf, S_in, capteur, hour, doy, extra):
        tf = caracteristiques(S_in, capteur, hour, doy, wf.geom, extra)
        s_t = torch.from_numpy(np.ascontiguousarray(S_in[:N_SURF]))[None].to(dev)
        m_t = torch.from_numpy(capteur.astype(np.float32))[None].to(dev)
        t_t = torch.from_numpy(tf)[None].to(dev)
        with torch.autocast(device_type=dev.type, dtype=wf.amp_dtype, enabled=wf.use_amp):
            pred = wf.model(assemble_input(s_t, m_t, t_t, wf.geom, wf.norm))
        u, v, _ = (np.where(geom.valid, x[0].float().cpu().numpy(), np.nan)
                   for x in wf.norm.denorm(pred.float()))
        return u, v

    somme = {h: {"vect": 0.0, "vitesse": 0.0, "n": 0.0} for h in HAUTEURS}
    t0 = time.time()
    for n, stem in enumerate(stems):
        r = row.loc[stem]
        t = pd.Timestamp(r["timestamp"])
        S = np.load(cfg_a.paths.cache_dir / f"{stem}_S.npy")
        E = np.load(cfg_a.paths.cache_dir / f"{stem}_E.npy").astype(np.float32)
        S_in, cap = entree_reelle(S, E, geom, cellules, obs_a(obs, t))
        extra = geom.forcage.vecteur(t)
        ua, va_ = passe(wa, S_in, cap, float(r["hour"]), int(r["doy"]), extra)
        ub, vb = passe(wb, S_in, cap, float(r["hour"]), int(r["doy"]), extra)
        for h in HAUTEURS:
            a_u, a_v = champ_agl(ua, va_, geom, h)
            b_u, b_v = champ_agl(ub, vb, geom, h)
            ok = np.isfinite(a_u) & np.isfinite(b_u)
            somme[h]["vect"] = somme[h]["vect"] + np.where(ok, np.hypot(b_u - a_u, b_v - a_v), 0.0)
            somme[h]["vitesse"] = somme[h]["vitesse"] + np.where(ok, np.hypot(b_u, b_v) - np.hypot(a_u, a_v), 0.0)
            somme[h]["n"] = somme[h]["n"] + ok
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(stems)} instants | {time.time() - t0:.0f} s", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    na, nb = cfg_a.name, cfg_b.name
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    bilan = {"modele_a": na, "modele_b": nb, "instants": len(stems)}
    for i, h in enumerate(HAUTEURS):
        nn = np.where(somme[h]["n"] > 0, somme[h]["n"], np.nan)
        vect, vit = somme[h]["vect"] / nn, somme[h]["vitesse"] / nn
        bilan[f"{h:g} m"] = {"ecart_vectoriel_moyen": stats_anneaux(vect, geom, cellules),
                             "ecart_vitesse_moyen": stats_anneaux(vit, geom, cellules)}
        lim = float(np.nanpercentile(np.abs(vit), 99)) or 1.0
        for j, (carte, titre, cmap, vmin, vmax) in enumerate((
                (vect, f"|{nb} - {na}| moyen a {h:g} m (m/s)", "viridis", 0, float(np.nanpercentile(vect, 99))),
                (vit, f"vitesse {nb} - {na} a {h:g} m (m/s)", "RdBu_r", -lim, lim))):
            ax = axes[i, j]
            im = ax.imshow(carte, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=(0, geom.nx * geom.dx / 1000, 0, geom.ny * geom.dy / 1000))
            for code, (iy, ix) in cellules.items():
                ax.plot(ix * geom.dx / 1000, iy * geom.dy / 1000, "k^", ms=8)
                ax.annotate(CODES.get(code, code), (ix * geom.dx / 1000, iy * geom.dy / 1000),
                            xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.set_title(titre, fontsize=10)
            ax.set_xlabel("km")
            fig.colorbar(im, ax=ax, shrink=0.8)
    nom = f"ecart_{nb}_moins_{na}"
    fig.savefig(sortie / f"{nom}.png", dpi=110)
    (sortie / f"{nom}.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(bilan, indent=1, ensure_ascii=False))
    print(f"resultats -> {sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
