"""Fabrique un mini-dataset ANEO au meme format que le generateur, pour les tests.

But : pouvoir executer tout le pipeline (cache -> statistiques -> entrainement ->
evaluation -> reconstruction -> export web) en quelques secondes sur un portable, sans
Colab et sans les 35 Go du dataset reel. Les champs sont physiquement coherents mais
grossiers : ils servent a verifier le code, pas la qualite du modele.

    python scripts/make_mini_dataset.py artifacts/mini
    python -m pcfno3d -c test build-cache

Format produit (identique a `aneo_dataset_3d_v2`) :
    meta.json, split_days.json, aneo_grid_points.csv,
    aneo_observations_day_XXX_YYYYMMDD_multitime.csv.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

# grille minuscule : 40 x 20 capteurs (50 m) -> 20 x 10 noeuds (100 m), 4 niveaux
NX_S, NY_S = 40, 20
DX_S = DY_S = 50.0
STRIDE = 2
Z_LEVELS = [25.0, 50.0, 100.0, 200.0]
HOURS = [0, 6, 12, 18]
PENTE = 12.0          # m de relief par km vers l'est
N_DAYS = 6
START = pd.Timestamp("2025-01-01")

SURF_COLS = ["point_id", "timestamp_utc", "day_index", "hour_utc", "temperature_c",
             "potential_temperature_k", "air_density_kg_m3", "relative_humidity_pct",
             "water_vapor_pressure_pa", "pressure_pa", "wind_u_m_s", "wind_v_m_s",
             "wind_w_m_s", "aneo_quality_score", "mask_i_obs", "source_type"]


def build_grid(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Capteurs de surface puis noeuds 3D, avec les memes conventions que le generateur."""
    nx, ny = NX_S // STRIDE, NY_S // STRIDE
    ix_s, iy_s = np.meshgrid(np.arange(NX_S), np.arange(NY_S))
    ix_s, iy_s = ix_s.ravel(), iy_s.ravel()

    # relief lisse : une colline et une pente generale. Volontairement bas (5 a 80 m) pour
    # que plusieurs niveaux soient au-dessus du sol, comme dans le dataset reel.
    xk = ix_s * DX_S / 1000.0
    yk = iy_s * DY_S / 1000.0
    zt = 30.0 + 18.0 * np.sin(xk / 0.8) * np.cos(yk / 0.6) + PENTE * xk
    slope = np.abs(np.gradient(zt)) * 2.0
    surf = pd.DataFrame({
        "point_id": np.arange(1, NX_S * NY_S + 1),
        "ix": ix_s, "iy": iy_s,
        "x_m": ix_s * DX_S, "y_m": iy_s * DY_S,
        "z_terrain_m": np.round(zt, 2),
        "z_rel_m": np.round(rng.uniform(21.0, 29.0, zt.size), 2),
        "z0_m": 0.1,
        "slope_deg": np.round(np.clip(slope, 0.0, 20.0), 3),
        "aspect_deg": np.round(rng.uniform(0.0, 360.0, zt.size), 3),
        "land_use_code": 3, "terrain_type": "agricole",
    })

    # noeuds 3D : un point sur STRIDE, aux niveaux au-dessus du relief (min_z_rel_m = 5 m)
    node_id_start = NX_S * NY_S + 1
    zt_model = surf["z_terrain_m"].values.reshape(NY_S, NX_S)[::STRIDE, ::STRIDE]
    rows = []
    pid = node_id_start
    for k, z in enumerate(Z_LEVELS):
        for j in range(ny):
            for i in range(nx):
                z_rel = z - zt_model[j, i]
                if z_rel < 5.0:
                    continue
                rows.append((pid, i * STRIDE, j * STRIDE, i * STRIDE * DX_S,
                             j * STRIDE * DY_S, zt_model[j, i], round(float(z_rel), 2)))
                pid += 1
    nodes = pd.DataFrame(rows, columns=["point_id", "ix", "iy", "x_m", "y_m",
                                        "z_terrain_m", "z_rel_m"])
    nodes["z0_m"] = 0.1
    nodes["slope_deg"] = 0.0
    nodes["aspect_deg"] = 0.0
    nodes["land_use_code"] = 3
    nodes["terrain_type"] = "agricole"

    grid = pd.concat([surf, nodes], ignore_index=True)
    grid["geom_wkt"] = ["POINT (" + f"{x} {y})" for x, y in zip(grid["x_m"], grid["y_m"])]

    meta = {
        "producteur": "Tukhnanutha (mini-dataset de test)",
        "version_generateur": 2,
        "grille_surface": {"nx": NX_S, "ny": NY_S, "dx_m": DX_S, "dy_m": DY_S},
        "noeuds_3d": {
            "stride": STRIDE, "nx": nx, "ny": ny, "dx_m": DX_S * STRIDE,
            "z_levels_asl_m": Z_LEVELS, "min_z_rel_m": 5.0,
            "point_id_debut": node_id_start, "nombre": int(len(nodes)),
        },
        "source_type": {"capteur": "ANEO_PHYSICAL_SYNTHETIC_MULTITIME",
                        "noeud_3d": "ANEO_VIRTUAL_NODE_3D"},
        "conventions": {"u": "positif vers l'Est", "v": "positif vers le Nord",
                        "w": "positif vers le haut"},
        "calendrier": {"jours": N_DAYS, "instants": N_DAYS * len(HOURS)},
    }
    return grid, meta


def wind_at(x_km, y_km, z_asl, zt, hour, day, rng):
    """Profil logarithmique + rotation diurne + w deduit de la pente : physiquement plausible."""
    z_agl = np.maximum(z_asl - zt, 1.0)
    base = 7.0 + 2.0 * np.sin(2 * np.pi * hour / 24) + 0.3 * day
    prof = np.log(z_agl / 0.1) / np.log(30.0 / 0.1)          # vent de reference a 30 m
    angle = np.radians(220.0 + 25.0 * np.sin(2 * np.pi * hour / 24) + 0.4 * x_km)
    speed = base * prof * (1.0 + 0.04 * np.sin(x_km / 0.7) * np.cos(y_km / 0.5))
    u = speed * np.cos(angle)
    v = speed * np.sin(angle)
    dzt_dx = np.gradient(zt.reshape(-1))[0] if np.ndim(zt) else 0.0
    w = 0.02 * u * (1.0 + 0.1 * np.sin(y_km)) + 0.01 * v - 0.0005 * dzt_dx
    return u, v, w


def build_day(day: int, grid: pd.DataFrame, meta: dict, rng: np.random.Generator,
              out_dir: Path) -> Path:
    node_start = meta["noeuds_3d"]["point_id_debut"]
    is_surf = grid["point_id"].values < node_start
    z_asl = np.where(is_surf, grid["z_terrain_m"] + grid["z_rel_m"],
                     grid["z_terrain_m"] + grid["z_rel_m"])
    x_km = grid["x_m"].values / 1000.0
    y_km = grid["y_m"].values / 1000.0
    zt = grid["z_terrain_m"].values

    frames = []
    date = START + pd.Timedelta(days=day - 1)
    for hour in HOURS:
        stamp = date + pd.Timedelta(hours=hour)
        u, v, w = wind_at(x_km, y_km, z_asl, zt, hour, day, rng)
        t_c = 14.0 + 8.0 * np.sin(2 * np.pi * (hour - 3) / 24) - 0.0065 * z_asl
        p = 101325.0 * np.exp(-z_asl / 8400.0) - 20.0 * x_km
        rho = p / (287.05 * (t_c + 273.15))
        rh = np.clip(60.0 + 15.0 * np.cos(2 * np.pi * hour / 24), 10.0, 100.0)
        # quelques capteurs manquants, comme dans le dataset reel
        mask = np.where(is_surf, (rng.random(len(grid)) < 0.9).astype(int), 0)
        frames.append(pd.DataFrame({
            "point_id": grid["point_id"].values,
            "timestamp_utc": stamp.strftime("%Y-%m-%d %H:%M:%S"),
            "day_index": day, "hour_utc": hour,
            "temperature_c": np.round(t_c, 3),
            "potential_temperature_k": np.round(t_c + 273.15 + 0.0098 * z_asl, 3),
            "air_density_kg_m3": np.round(rho, 5),
            "relative_humidity_pct": np.round(rh, 2),
            "water_vapor_pressure_pa": np.round(rh * 12.0, 1),
            "pressure_pa": np.round(p, 1),
            "wind_u_m_s": np.round(u, 3), "wind_v_m_s": np.round(v, 3),
            "wind_w_m_s": np.round(w, 4),
            "aneo_quality_score": 0.95, "mask_i_obs": mask,
            "source_type": np.where(is_surf, meta["source_type"]["capteur"],
                                    meta["source_type"]["noeud_3d"]),
        }))
    df = pd.concat(frames, ignore_index=True)[SURF_COLS]
    path = out_dir / f"aneo_observations_day_{day:03d}_{date.strftime('%Y%m%d')}_multitime.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
    return path


def main() -> int:
    global NX_S, NY_S, DX_S, DY_S, PENTE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default="artifacts/mini", help="dossier de sortie")
    ap.add_argument("--seed", type=int, default=0)
    # taille de la grille de surface (scripts/dev_service.py : la zone du Grand Tunis a 500 m)
    ap.add_argument("--nx", type=int, default=NX_S, help="capteurs en x (pair)")
    ap.add_argument("--ny", type=int, default=NY_S, help="capteurs en y (pair)")
    ap.add_argument("--dx", type=float, default=DX_S, help="pas de la grille de surface (m)")
    ap.add_argument("--pente", type=float, default=PENTE, help="relief : m par km vers l'est")
    args = ap.parse_args()
    NX_S, NY_S, DX_S, DY_S, PENTE = args.nx, args.ny, args.dx, args.dx, args.pente

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    grid, meta = build_grid(rng)
    grid.to_csv(out_dir / "aneo_grid_points.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    # 4 jours d'entrainement, 1 de validation, 1 de test
    split = {"train_days": [1, 2, 3, 4], "val_days": [5], "test_days": [6]}
    (out_dir / "split_days.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    for day in range(1, N_DAYS + 1):
        path = build_day(day, grid, meta, rng, out_dir)
        print(f"jour {day:03d} -> {path.name}")

    print(f"\nmini-dataset pret dans {out_dir} "
          f"({len(grid):,} points, {N_DAYS * len(HOURS)} instants)")
    print("etape suivante : python -m pcfno3d -c test build-cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
