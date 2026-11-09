"""Test rapide (plan B1, approuve par Rayen) : facteur des perturbations aleatoires du generateur
national, sur 20 instants de VALIDATION (jamais le test), sans generation complete.

Pour chaque facteur (0 ; 0,1 ; 0,2 ; 0,4 = jeu national v2), memes instants, meme forcage, meme
graine : la verite est recalculee par compute_fields, puis on mesure
  - l'erreur aux 12 aeroports (vent a 10 m, bloc 2 x 2 comme le cache, METAR a 30 min au plus) :
    RMSE vectoriel du generateur et d'ERA5 seule contre la mesure, et alpha (part du detail ajoute
    qui va dans le sens de l'erreur reelle d'ERA5) ;
  - l'ecart verite - ERA5 sur v (et u) : a 300 et 800 m au-dessus du sol, et sur tous les noeuds.
Une ligne d'information « 0,4 + amortissement passe-haut » (option ANEO_GEN_AMORT_PASSE_HAUT) est
ajoutee, separee : c'est l'amortissement de w qui domine l'ecart sur v (docs/diagnostic_v_national.md).

    python scripts/test_facteurs_perturbation.py --sortie artifacts/national/test_facteurs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
NAT = REPO / "artifacts" / "national"
FACTEURS = (0.0, 0.1, 0.2, 0.4)
VAL = (("2021-01-01", "2021-02-28"), ("2021-07-01", "2021-07-31"))     # blocs de validation
HEURES = (3, 9, 15, 21)


def instants_validation(n: int, graine: int = 2026) -> list[tuple[pd.Timestamp, int]]:
    jours = pd.DatetimeIndex([])
    for a, b in VAL:
        jours = jours.append(pd.date_range(a, b, freq="D"))
    rng = np.random.default_rng(graine)
    choix = sorted(rng.choice(len(jours), size=n // 2, replace=False))
    return [(jours[i], h) for i in choix for h in (HEURES[i % 2], HEURES[2 + i % 2])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instants", type=int, default=20)
    ap.add_argument("--sortie", default=str(NAT / "test_facteurs"))
    a = ap.parse_args()
    import calibrer_generateur_stations as C
    info = json.loads((NAT / "grille" / "grille_reelle.json").read_text(encoding="utf-8"))
    C._env(info, 1.0)                                  # A_ROT, A_DIV de base ; facteur applique ensuite
    import generate_aneo_3d as G
    from pcfno3d.stations_reelles import lire_observations, obs_a
    from scipy.ndimage import gaussian_filter

    df = G.load_or_build_grid(NAT / "grille" / "aneo_grid_points_surface.csv", np.random.default_rng(0))
    S = G.build_static(df)
    P = G.precompute_static_derived(S)
    F = G.lire_forcage(str(NAT / "forcage_era5.csv"))
    CH = G.lire_champs(str(NAT / "forcage_era5.csv"))
    st_codes = {s["code"]: (int(s["iy"]), int(s["ix"])) for s in info["stations"].values() if s.get("dans_la_zone")}
    obs = lire_observations(NAT / "obs", verbose=False, codes=list(st_codes))
    k = G.STRIDE
    lat_v, lon_v = S["lat"][::k, ::k][:G.NY_V, :G.NX_V], S["lon"][::k, ::k][:G.NY_V, :G.NX_V]
    zt_v = S["zt"][::k, ::k][:G.NY_V, :G.NX_V]
    Z = np.asarray(G.Z_LEVELS, float)
    nv = S["node_valid"]
    a_rot1, a_div1 = G.A_ROT, G.A_DIV

    def a_hauteur(vol3, h):
        z = zt_v + h
        i = np.clip(np.searchsorted(Z, z) - 1, 0, len(Z) - 2)
        t = np.clip((z - Z[i]) / (Z[i + 1] - Z[i]), 0, 1)
        lo = np.take_along_axis(vol3, i[None], 0)[0]
        hi = np.take_along_axis(vol3, (i + 1)[None], 0)[0]
        ok = (z >= Z[0]) & (z <= Z[-1]) & (Z[i] - zt_v >= G.MIN_ZREL_NODE)
        return np.where(ok, (1 - t) * lo + t * hi, np.nan)

    insts = instants_validation(a.instants)
    variantes = [(f"facteur {f:g}", f, False) for f in FACTEURS] + [("0,4 + amortissement passe-haut (info)", 0.4, True)]
    lignes = []
    for nom, f, hp in variantes:
        G.A_ROT, G.A_DIV = a_rot1 * f, a_div1 * f
        G.AMORT_PASSE_HAUT = hp
        st_err = []
        dev = {key: np.zeros(2) for key in ("v300", "v800", "u300", "u800", "v_tous", "u_tous")}
        for jour, h in insts:
            seed = 2026 * 1000 + int((jour - pd.Timestamp("2021-01-01")).days) + 1
            st = G.DayStateERA5(0, jour, 0, S, seed, F, CH)
            surf, vol, _ = G.compute_fields(st, S, P, h * 3600.0, None)
            champ = st.champ_cible(h * 3600.0)
            for c, cn in ((0, "u"), (1, "v")):
                for hh in (300.0, 800.0):
                    d = a_hauteur(vol["uv"[c]], hh) - G._era5_au_sol(st.champs, champ[c], lat_v, lon_v,
                                                                      np.full(lat_v.shape, hh))
                    m = np.isfinite(d)
                    dev[f"{cn}{int(hh)}"] += [(d[m] ** 2).sum(), m.sum()]
                for kk in range(len(Z)):
                    hs = np.maximum(Z[kk] - zt_v, G.MIN_ZREL_NODE)
                    d = (vol["uv"[c]][kk] - G._era5_au_sol(st.champs, champ[c], lat_v, lon_v, hs))[nv[kk]]
                    dev[f"{cn}_tous"] += [(d ** 2).sum(), d.size]
            t = jour + pd.Timedelta(hours=h)
            for code, (uo, vo) in obs_a(obs, t).items():
                iy, ix = st_codes[code]
                jy, jx = (iy // k) * k, (ix // k) * k
                ug, vg = (float(surf[x][jy:jy + k, jx:jx + k].mean()) for x in ("u", "v"))
                la, lo = np.array([[S["lat"][jy, jx]]]), np.array([[S["lon"][jy, jx]]])
                ue, ve = (float(G._era5_au_sol(st.champs, champ[c], la, lo, np.full((1, 1), 10.0))[0, 0]) for c in (0, 1))
                st_err.append((code, uo, vo, ug, vg, ue, ve))
            print(f"{nom} {t:%Y-%m-%d %H} h", flush=True)
        e = pd.DataFrame(st_err, columns=["code", "uo", "vo", "ug", "vg", "ue", "ve"])
        du, dv, eu, ev = e.ug - e.ue, e.vg - e.ve, e.uo - e.ue, e.vo - e.ve
        lignes.append({
            "variante": nom, "paires_aeroports": len(e),
            "rmse_aeroports_generateur": float(np.sqrt(((e.ug - e.uo) ** 2 + (e.vg - e.vo) ** 2).mean())),
            "rmse_aeroports_era5": float(np.sqrt(((e.ue - e.uo) ** 2 + (e.ve - e.vo) ** 2).mean())),
            "rmse_v_aeroports_generateur": float(np.sqrt(((e.vg - e.vo) ** 2).mean())),
            "alpha": float((du * eu + dv * ev).sum() / max((du ** 2 + dv ** 2).sum(), 1e-12)),
            **{f"ecart_{key}": float(np.sqrt(s[0] / s[1])) for key, s in dev.items()}})
    G.A_ROT, G.A_DIV, G.AMORT_PASSE_HAUT = a_rot1 * 0.4, a_div1 * 0.4, False
    t = pd.DataFrame(lignes)
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "test_facteurs.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
