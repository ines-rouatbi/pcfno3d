"""DSO previsibles de la verite v3 avant tout entrainement : ERA5 seule (la base du residuel du modele)
contre la verite d'entrainement v2 et v3, par tranche au-dessus du sol, sur la VALIDATION 2021.

Regle : configs/national/regle_dso_previsible_v3.json (empreinte verifiee avant calcul). Rapport seulement.

    python scripts/dso_previsible_v3.py [--sortie docs/resultats/national/dso_previsible_v3]
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
REGLE = REPO / "configs" / "national" / "regle_dso_previsible_v3.json"
BORNES = (0.0, 100.0, 300.0, 600.0, 1000.0)
VERITES = {"v2": {"passe_haut": False, "perturb": 0.4, "brise": 1.0, "relief": 1.0, "beta": 1.0},
           "v3": {"passe_haut": True, "perturb": 0.0, "brise": 0.5, "relief": 1.0, "beta": 0.5}}


def verifier_regle() -> str:
    h = hashlib.sha256(REGLE.read_bytes()).hexdigest()
    attendu = (REGLE.parent / (REGLE.name + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"regle modifiee depuis son hachage : {h} != {attendu}")
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sortie", default=str(REPO / "docs" / "resultats" / "national" / "dso_previsible_v3"))
    a = ap.parse_args()
    h_regle = verifier_regle()
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)

    import calibrer_generateur_stations as C
    import calibrer_verite_v3 as V
    info = json.loads((NAT / "grille" / "grille_reelle.json").read_text(encoding="utf-8"))
    C._env(info, 1.0)
    import generate_aneo_3d as G

    df = G.load_or_build_grid(NAT / "grille" / "aneo_grid_points_surface.csv", np.random.default_rng(0))
    S = G.build_static(df)
    P0 = G.precompute_static_derived(S)
    F = G.lire_forcage(str(NAT / "forcage_era5.csv"))
    CH = G.lire_champs(str(NAT / "forcage_era5.csv"))
    k = G.STRIDE
    lat_v, lon_v = S["lat"][::k, ::k][:G.NY_V, :G.NX_V], S["lon"][::k, ::k][:G.NY_V, :G.NX_V]
    zt_v = S["zt"][::k, ::k][:G.NY_V, :G.NX_V]
    Z = np.asarray(G.Z_LEVELS, float)
    nv = np.asarray(S["node_valid"], bool) if "node_valid" in S else None
    a_rot1, a_div1 = G.A_ROT, G.A_DIV
    agl = Z[:, None, None] - zt_v[None]
    h_sol = np.maximum(agl, G.MIN_ZREL_NODE)

    i_cal = V.instants(20, 2026)
    insts = V.instants(20, 2027, exclure={j for j, _ in i_cal})
    lignes = []
    for nom_v, v in VERITES.items():
        G.A_ROT, G.A_DIV = a_rot1 * v["perturb"], a_div1 * v["perturb"]
        G.AMORT_PASSE_HAUT = v["passe_haut"]
        G.BRISE_FACTEUR, G.RELIEF_FACTEUR, G.DETAIL_BETA = v["brise"], v["relief"], v["beta"]
        nb = len(BORNES)
        acc = np.zeros((nb, 2, 4))
        rug = [[] for _ in range(nb)]
        for jour, hh in insts:
            seed = 2026 * 1000 + int((jour - pd.Timestamp("2021-01-01")).days) + 1
            st = G.DayStateERA5(0, jour, 0, S, seed, F, CH)
            _, vol, _ = G.compute_fields(st, S, P0, hh * 3600.0, None)
            champ = st.champ_cible(hh * 3600.0)
            valid = nv if nv is not None else np.isfinite(vol["u"]) & (agl >= G.MIN_ZREL_NODE)
            E = [np.stack([G._era5_au_sol(CH, champ[c], lat_v, lon_v, h_sol[kz]) for kz in range(len(Z))])
                 for c in (0, 1)]
            Y = [vol["u"], vol["v"]]
            masques = [valid & (agl >= lo) & (agl < hi) for lo, hi in zip(BORNES[:-1], BORNES[1:])]
            masques.append(valid & (agl >= BORNES[-1]))
            for b, m in enumerate(masques):
                for c in (0, 1):
                    t, p = Y[c][m].astype(np.float64), E[c][m].astype(np.float64)
                    acc[b, c] += [t.size, ((p - t) ** 2).sum(), t.sum(), (t ** 2).sum()]
                mx = m[:, :, 1:] & m[:, :, :-1]
                my = m[:, 1:, :] & m[:, :-1, :]
                if mx.any() and my.any():
                    def gm(a):
                        return np.abs(np.diff(a, axis=2))[mx].mean() + np.abs(np.diff(a, axis=1))[my].mean()
                    den = gm(Y[0]) + gm(Y[1])
                    if den > 0:
                        rug[b].append(float((gm(E[0]) + gm(E[1])) / den))
            print(f"{nom_v} {jour:%Y-%m-%d} {hh:02d} h", flush=True)
        noms = [f"{lo:g}-{hi:g} m" for lo, hi in zip(BORNES[:-1], BORNES[1:])] + [f"> {BORNES[-1]:g} m (information)"]
        for b, tn in enumerate(noms):
            rec = {"verite": nom_v, "tranche": tn, "noeuds_par_instant": int(acc[b, 0, 0] / len(insts))}
            for c, cn in enumerate("uv"):
                n, sse, st_, st2 = acc[b, c]
                rec[f"rmse_{cn}"] = float(np.sqrt(sse / n)) if n else float("nan")
                sst = st2 - st_ ** 2 / n if n else 0
                rec[f"r2_{cn}"] = float(1 - sse / sst) if sst > 0 else float("nan")
            rec["rugosite_era5_sur_verite"] = float(np.mean(rug[b])) if rug[b] else float("nan")
            lignes.append(rec)
    G.A_ROT, G.A_DIV, G.AMORT_PASSE_HAUT, G.BRISE_FACTEUR, G.RELIEF_FACTEUR, G.DETAIL_BETA = \
        a_rot1, a_div1, False, 1.0, 1.0, 1.0
    t = pd.DataFrame(lignes)
    t.to_csv(out / "era5_contre_verite_par_tranche.csv", index=False)
    (out / "resume.json").write_text(json.dumps({"regle": "configs/national/regle_dso_previsible_v3.json",
                                                 "sha256_regle": h_regle,
                                                 "instants": [str(j + pd.Timedelta(hours=h)) for j, h in insts]},
                                                indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
