"""Ce que le generateur national ajoute sur v par rapport a ERA5, composante par composante.

Diagnostic (pas une regeneration du jeu) : quelques instants de jours d'ENTRAINEMENT sont
calcules par compute_fields en coupant une composante a la fois (memes graines), et l'ecart
verite - ERA5 est mesure a 300 et 800 m AU-DESSUS DU SOL (interpolation dans chaque colonne),
pour u et v, en total et dans sa partie de grande echelle (> 12 km).

    actuel        generateur du jeu national v2
    sans_amort    sans l'amortissement de w (ecoulement potentiel ajoute apres le recalage)
    passe_haut    amortissement filtre < 12 km (option ANEO_GEN_AMORT_PASSE_HAUT, correction)
    sans_brise    sans brise de mer / de terre
    sans_pente    sans vents de pente
    sans_relief   sans acceleration par le relief
    sans_perturb  sans perturbations aleatoires

    python scripts/attribution_v_generateur.py --sortie artifacts/national/attribution_v
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
CONFIGS = ("actuel", "sans_amort", "passe_haut", "sans_brise", "sans_pente", "sans_relief", "sans_perturb")
HAUTEURS = (300.0, 800.0)
INSTANTS = (("2023-02-14", 3), ("2023-02-14", 15), ("2024-08-06", 3), ("2024-08-06", 15))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sortie", default=str(NAT / "attribution_v"))
    a = ap.parse_args()
    import calibrer_generateur_stations as C
    info = json.loads((NAT / "grille" / "grille_reelle.json").read_text(encoding="utf-8"))
    C._env(info, 0.4)
    import generate_aneo_3d as G
    from scipy.ndimage import gaussian_filter

    df = G.load_or_build_grid(NAT / "grille" / "aneo_grid_points_surface.csv", np.random.default_rng(0))
    S = G.build_static(df)
    P0 = G.precompute_static_derived(S)
    F = G.lire_forcage(str(NAT / "forcage_era5.csv"))
    CH = G.lire_champs(str(NAT / "forcage_era5.csv"))
    k = G.STRIDE
    lat_v, lon_v = S["lat"][::k, ::k][:G.NY_V, :G.NX_V], S["lon"][::k, ::k][:G.NY_V, :G.NX_V]
    zt_v = S["zt"][::k, ::k][:G.NY_V, :G.NX_V]
    Z = np.asarray(G.Z_LEVELS, float)
    base = (G.A_ROT, G.A_DIV, G.A_SLOPE, G.W_DAMPING_M)

    def a_hauteur(vol3, h):
        """champ (nz, ny, nx) aux niveaux ASL -> a h m au-dessus du sol (NaN hors colonne)."""
        z = zt_v + h
        i = np.clip(np.searchsorted(Z, z) - 1, 0, len(Z) - 2)
        t = (z - Z[i]) / (Z[i + 1] - Z[i])
        lo = np.take_along_axis(vol3, i[None], 0)[0]
        hi = np.take_along_axis(vol3, (i + 1)[None], 0)[0]
        ok = (z >= Z[0]) & (z <= Z[-1]) & (Z[i] - zt_v >= G.MIN_ZREL_NODE)
        return np.where(ok, (1 - np.clip(t, 0, 1)) * lo + np.clip(t, 0, 1) * hi, np.nan)

    lignes = []
    for cfg in CONFIGS:
        G.A_ROT, G.A_DIV, G.A_SLOPE, G.W_DAMPING_M = base
        G.AMORT_PASSE_HAUT = cfg == "passe_haut"
        P = P0
        if cfg == "sans_amort":
            G.W_DAMPING_M = 1e30
        elif cfg == "sans_pente":
            G.A_SLOPE = 0.0
        elif cfg == "sans_perturb":
            G.A_ROT = G.A_DIV = 0.0
        elif cfg == "sans_relief":
            P = {**P0, "relief": np.zeros_like(P0["relief"])}
        acc = {}
        for jour, h_utc in INSTANTS:
            st = G.DayStateERA5(0, pd.Timestamp(jour), 0, S, 7 + int(jour[-2:]), F, CH)
            if cfg == "sans_brise":
                st.brise = lambda t_sec, g: 0.0
            _, vol, _ = G.compute_fields(st, S, P, h_utc * 3600.0, None)
            champ = st.champ_cible(h_utc * 3600.0)
            for h in HAUTEURS:
                for c, nom in ((0, "u"), (1, "v")):
                    vrai = a_hauteur(vol["uv"[c]], h)
                    e = G._era5_au_sol(st.champs, champ[c], lat_v, lon_v, np.full(lat_v.shape, h))
                    d = vrai - e
                    m = np.isfinite(d)
                    d0 = np.where(m, d, 0.0)
                    lp = gaussian_filter(d0, 12) / np.maximum(gaussian_filter(m * 1.0, 12), 1e-3)
                    s = acc.setdefault((h, nom), np.zeros(3))
                    s += [(d0[m] ** 2).sum(), (lp[m] ** 2).sum(), m.sum()]
            print(f"{cfg} {jour} {h_utc:02d} h", flush=True)
        for (h, nom), s in acc.items():
            lignes.append({"config": cfg, "hauteur_sol_m": int(h), "composante": nom,
                           "ecart_rms": float(np.sqrt(s[0] / s[2])),
                           "ecart_grande_echelle": float(np.sqrt(s[1] / s[2]))})
    G.A_ROT, G.A_DIV, G.A_SLOPE, G.W_DAMPING_M = base
    t = pd.DataFrame(lignes)
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "attribution_v.csv", index=False)
    piv = t.pivot_table(index="config", columns=["hauteur_sol_m", "composante"],
                        values="ecart_rms").reindex(list(CONFIGS)).round(3)
    pd.set_option("display.width", 200)
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
