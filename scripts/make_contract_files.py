"""Fabrique les jeux de non-regression du contrat de donnees ANEO (tests/data/contract/).

Trois petits fichiers au format EXACT du generateur (memes colonnes, meme format
d'horodatage), sur une grille minuscule de 6 x 4 stations (50 m) et 6 noeuds 3D, 4 instants
synoptiques (pas de 3 h) :

    conforme.csv   physiquement coherent (densite par temperature virtuelle), 2 valeurs
                   interpolees declarees : verdict « conforme », sans avertissement
    corrompu.csv   types faux, hors plage, doublons, point_id inconnu, mask et source
                   invalides, heure et pas incoherents, densite fausse : verdict « rejet »
    panne.csv      stations muettes (absentes 3 instants), anemometres figes, 30 % de
                   valeurs interpolees : verdict « quarantaine »

plus meta.json, la grille declaree. Deterministe (graine fixe) :

    python scripts/make_contract_files.py               # -> tests/data/contract/
    python scripts/make_contract_files.py autre/dossier
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NX_S, NY_S, DX = 6, 4, 50.0
STRIDE = 2
Z_NODE = 150.0
DAY_INDEX = 4
DATE = pd.Timestamp("2025-01-07")
HOURS = [0, 3, 6, 9]
SRC = {"capteur": "ANEO_PHYSICAL_SYNTHETIC_MULTITIME",
       "interpole": "ANEO_PHYSICAL_INTERPOLATED_MULTITIME",
       "noeud_3d": "ANEO_VIRTUAL_NODE_3D"}
COLS = ["point_id", "timestamp_utc", "day_index", "hour_utc", "temperature_c",
        "potential_temperature_k", "air_density_kg_m3", "relative_humidity_pct",
        "water_vapor_pressure_pa", "pressure_pa", "wind_u_m_s", "wind_v_m_s",
        "wind_w_m_s", "aneo_quality_score", "mask_i_obs", "source_type"]
DECIMALS = {"temperature_c": 3, "potential_temperature_k": 3, "air_density_kg_m3": 5,
            "relative_humidity_pct": 2, "water_vapor_pressure_pa": 1, "pressure_pa": 1,
            "wind_u_m_s": 3, "wind_v_m_s": 3, "wind_w_m_s": 4, "aneo_quality_score": 4}
RD, KAPPA = 287.05, 0.2857
# conforme : (station, instant) interpoles et declares comme tels
INTERP_CONFORME = [(5, 0), (17, 2)]
# panne
INTERP_PANNE = [4, 7, 9, 13, 15, 20, 22]      # interpolees a tous les instants (29 %)
FIGEES_PANNE = [1, 2]                         # anemometres bloques : u, v constants
ABSENTES_PANNE = [17, 18, 23, 24]             # coin nord-est muet aux instants 1, 2, 3


def meta() -> dict:
    n_nodes = (NX_S // STRIDE) * (NY_S // STRIDE)
    return {
        "producteur": "Tukhnanutha (jeux de test du contrat ANEO)",
        "version_generateur": 2,
        "grille_surface": {"nx": NX_S, "ny": NY_S, "dx_m": DX, "dy_m": DX},
        "noeuds_3d": {"stride": STRIDE, "nx": NX_S // STRIDE, "ny": NY_S // STRIDE,
                      "dx_m": DX * STRIDE, "z_levels_asl_m": [Z_NODE], "min_z_rel_m": 5.0,
                      "point_id_debut": NX_S * NY_S + 1, "nombre": n_nodes},
        "source_type": SRC,
    }


def magnus(t_c):
    return 611.2 * np.exp(17.62 * t_c / (243.12 + t_c))


def instant(hour: int, rng: np.random.Generator) -> pd.DataFrame:
    """Une coupe : stations (point_id 1..24) puis noeuds 3D, comme le generateur."""
    ix_s, iy_s = np.meshgrid(np.arange(NX_S), np.arange(NY_S))
    ix_s, iy_s = ix_s.ravel(), iy_s.ravel()
    zt = 30.0 + 4.0 * ix_s + 2.0 * iy_s
    z_st = zt + 25.0
    jj, ii = np.meshgrid(np.arange(NY_S // STRIDE), np.arange(NX_S // STRIDE), indexing="ij")
    zt_n = 30.0 + 4.0 * (ii.ravel() * STRIDE) + 2.0 * (jj.ravel() * STRIDE)
    z = np.concatenate([z_st, np.full(zt_n.size, Z_NODE)])
    x = np.concatenate([ix_s * DX, ii.ravel() * STRIDE * DX]) / 1000.0
    n_s, n = ix_s.size, z.size

    t_c = 14.0 + 6.0 * np.sin(2 * np.pi * (hour - 3) / 24) - 0.0065 * z
    p = 101325.0 * np.exp(-z / 8400.0) - 20.0 * x
    rh = np.clip(65.0 + 12.0 * np.cos(2 * np.pi * hour / 24) - 0.01 * z, 5.0, 100.0)
    e = rh / 100.0 * magnus(t_c)
    q = 0.622 * e / (p - 0.378 * e)
    rho = p / (RD * (t_c + 273.15) * (1.0 + 0.608 * q))        # temperature virtuelle
    speed = (6.0 + 1.5 * np.sin(2 * np.pi * hour / 24)) * np.log(np.maximum(z - 20.0, 2.0)
                                                                    / 0.1) / np.log(300.0)
    ang = np.radians(220.0 + 20.0 * np.sin(2 * np.pi * hour / 24) + 3.0 * x)
    u, v = speed * np.cos(ang), speed * np.sin(ang)
    w = 0.02 * u + 0.01 * v
    noise = np.r_[np.ones(n_s), np.zeros(n - n_s)]               # bruit capteur, surface
    u = u + noise * rng.normal(0, 0.05, n)
    v = v + noise * rng.normal(0, 0.05, n)
    t_c = t_c + noise * rng.normal(0, 0.03, n)

    stamp = DATE + pd.Timedelta(hours=hour)
    df = pd.DataFrame({
        "point_id": np.arange(1, n + 1),
        "timestamp_utc": stamp.strftime("%Y-%m-%d %H:%M:%S"),
        "day_index": DAY_INDEX, "hour_utc": hour,
        "temperature_c": t_c, "potential_temperature_k": (t_c + 273.15) * (1e5 / p) ** KAPPA,
        "air_density_kg_m3": rho, "relative_humidity_pct": rh,
        "water_vapor_pressure_pa": e, "pressure_pa": p,
        "wind_u_m_s": u, "wind_v_m_s": v, "wind_w_m_s": w,
        "aneo_quality_score": np.r_[rng.uniform(0.85, 0.95, n_s), np.ones(n - n_s)],
        "mask_i_obs": np.r_[np.ones(n_s, int), np.zeros(n - n_s, int)],
        "source_type": [SRC["capteur"]] * n_s + [SRC["noeud_3d"]] * (n - n_s),
    })
    for c, d in DECIMALS.items():
        df[c] = df[c].round(d)
    return df[COLS]


def interpoler(df: pd.DataFrame, rows) -> None:
    df.loc[rows, "mask_i_obs"] = 0
    df.loc[rows, "source_type"] = SRC["interpole"]
    df.loc[rows, "aneo_quality_score"] = 0.72


def conforme(rng) -> pd.DataFrame:
    frames = [instant(h, rng) for h in HOURS]
    for pid, k in INTERP_CONFORME:
        interpoler(frames[k], frames[k]["point_id"] == pid)
    return pd.concat(frames, ignore_index=True)


def corrompu(base: pd.DataFrame) -> pd.DataFrame:
    df = base.astype(object).copy()
    k0 = base.index[base["hour_utc"] == 0]
    k1 = base.index[base["hour_utc"] == 3]
    k3 = base.index[base["hour_utc"] == 9]
    df.loc[k0[0], "temperature_c"] = "abc"                    # type faux
    df.loc[k0[1], "wind_u_m_s"] = "12,5"                      # virgule decimale
    df.loc[k0[2], "point_id"] = "x7"                          # identifiant illisible
    df.loc[k0[3], "temperature_c"] = 80.0                     # hors plage (et densite)
    df.loc[k0[4], "relative_humidity_pct"] = 130.0            # hors plage
    df.loc[k0[6], "pressure_pa"] = 50000.0                    # hors plage (et densite)
    df.loc[k0[7], "wind_w_m_s"] = 25.0                        # hors plage
    df.loc[k0[8], "point_id"] = 999                           # inconnu de la grille
    df.loc[k0[9], "mask_i_obs"] = 2                           # ni 0 ni 1
    df.loc[k0[10], "source_type"] = "ANEO_STATION_INCONNUE"   # source non declaree
    df.loc[k1[0], "hour_utc"] = 5                             # heure incoherente
    df.loc[k1[1], "air_density_kg_m3"] = 1.35                 # gaz parfaits violes
    df.loc[k1[2], "day_index"] = None                         # valeur obligatoire absente
    df.loc[k1[-1], "timestamp_utc"] = "2025-01-07 03:00:07"   # hors du pas de 10 s
    df.loc[k3[0], "timestamp_utc"] = "2025-01-07 25:00:00"    # horodatage illisible
    df.loc[k3[1], "point_id"] = 30                            # station declaree en noeud 3D
    dup = df.loc[k3[-3:-1]]                                   # 2 doublons en fin de fichier
    return pd.concat([df, dup], ignore_index=True)


def panne(rng) -> pd.DataFrame:
    frames = [instant(h, rng) for h in HOURS]
    ref = frames[0].set_index("point_id")
    for k, f in enumerate(frames):
        interpoler(f, f["point_id"].isin(INTERP_PANNE))
        frozen = f["point_id"].isin(FIGEES_PANNE)
        for c in ("wind_u_m_s", "wind_v_m_s"):
            f.loc[frozen, c] = ref.loc[f.loc[frozen, "point_id"], c].to_numpy()
        if k > 0:
            frames[k] = f[~f["point_id"].isin(ABSENTES_PANNE)]
    return pd.concat(frames, ignore_index=True)


def ecrire(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default=str(REPO / "tests" / "data" / "contract"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_bytes(
        (json.dumps(meta(), indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    base = conforme(np.random.default_rng(0))
    ecrire(base, out / "conforme.csv")
    ecrire(corrompu(base), out / "corrompu.csv")
    ecrire(panne(np.random.default_rng(1)), out / "panne.csv")
    for name in ("meta.json", "conforme.csv", "corrompu.csv", "panne.csv"):
        print(f"{out / name}  ({(out / name).stat().st_size} octets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
