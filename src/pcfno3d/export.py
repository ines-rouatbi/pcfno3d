"""Export du volume reconstruit vers l'application web 3D (taches T6 / T8 du Gantt).

Un volume complet fait 12 x 250 x 500 x 3 float32 ~ 18 Mo par instant : trop pour un
navigateur. L'export produit donc des binaires Float32 bruts, eventuellement decimes,
plus un `manifest.json` qui decrit la grille. Le visualiseur (`web/`) lit le manifeste,
puis charge les binaires avec un seul fetch chacun.

Format :
    manifest.json   grille, niveaux, bornes, liste des champs, valeur de remplacement
    terrain.bin     (ny, nx)        float32, altitude du relief en m
    u.bin v.bin w.bin speed.bin     (nz, ny, nx) float32, NaN remplace par `nodata`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .inference import Volume

NODATA = -9999.0


def export_split(wf, split: str = "test", stride: int = 2, synoptic_only: bool = False,
                 verbose: bool = True) -> dict:
    """Exporte tous les instants d'un jeu (`test` par defaut) avec UN SEUL modele charge.

    Un instant deja exporte (manifeste present : il est ecrit en dernier) est saute, donc
    la commande se relance sans tout refaire. Ecrit dans `cfg.paths.web_root/recon_<stem>`,
    la ou l'API (`/api/export/...`) les sert.
    """
    from .cache import load_index
    from .dataset import is_synoptic

    cfg = wf.cfg
    index = load_index(cfg)
    rows = index[index["split"] == split]
    if synoptic_only:
        rows = rows[[is_synoptic(str(t)) for t in rows["timestamp"]]]
    if rows.empty:
        raise ValueError(f"aucun instant du jeu '{split}' dans le cache {cfg.paths.cache_dir}")
    faits = sautes = 0
    for stem in rows["stem"].astype(str):
        out = cfg.paths.web_root / f"recon_{stem}"
        if (out / "manifest.json").exists():
            sautes += 1
            continue
        export_web(wf.predict_stem(stem), out, stride=stride)
        faits += 1
        if verbose and faits % 20 == 0:
            print(f"  {faits + sautes}/{len(rows)} instants exportes", flush=True)
    if verbose:
        print(f"export '{split}' : {faits} nouveau(x), {sautes} deja la -> {cfg.paths.web_root}")
    return {"exportes": faits, "deja_la": sautes, "total": int(len(rows)),
            "dossier": str(cfg.paths.web_root)}


def export_web(volume: Volume, out_dir: str | Path, stride: int = 1,
               fields: tuple[str, ...] = ("u", "v", "w", "speed")) -> Path:
    """Ecrit les binaires + le manifeste. `stride` decime horizontalement (2 = 4x plus leger)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geom = volume.geom
    s = max(int(stride), 1)

    data = {"u": volume.u, "v": volume.v, "w": volume.w,
            "speed": volume.speed, "direction": volume.direction}
    written = {}
    for name in fields:
        if name not in data:
            raise KeyError(f"champ inconnu : {name}")
        a = np.ascontiguousarray(data[name][:, ::s, ::s], dtype=np.float32)
        a = np.nan_to_num(a, nan=NODATA, posinf=NODATA, neginf=NODATA)
        (out_dir / f"{name}.bin").write_bytes(a.tobytes())
        finite = a[a != NODATA]
        written[name] = {
            "file": f"{name}.bin",
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
        }

    terrain = np.ascontiguousarray(geom.zt[::s, ::s], dtype=np.float32)
    (out_dir / "terrain.bin").write_bytes(terrain.tobytes())

    nz, ny, nx = volume.u.shape
    manifest = {
        "format": "pcfno3d-web/1",
        "timestamp_utc": volume.timestamp,
        "dtype": "float32",
        "nodata": NODATA,
        "ordre": "z, y, x (ligne par ligne)",
        "grille": {
            "nz": nz, "ny": len(range(0, ny, s)), "nx": len(range(0, nx, s)),
            "dx_m": geom.dx * s, "dy_m": geom.dy * s, "stride": s,
        },
        "z_levels_asl_m": geom.z_levels.tolist(),
        "terrain": {"file": "terrain.bin", "unite": "m ASL",
                    "min": float(terrain.min()), "max": float(terrain.max())},
        "champs": written,
        "inference_s": volume.inference_s,
        "capteurs_utilises": volume.n_sensors,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    return out_dir


def export_levels_csv(volume: Volume, out_dir: str | Path,
                      altitudes: tuple[float, ...] = (50.0, 100.0, 120.0, 150.0)) -> list[Path]:
    """Une carte CSV par altitude de moyeu — format attendu par les etudes eoliennes."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geom = volume.geom
    x = np.arange(geom.nx) * geom.dx
    y = np.arange(geom.ny) * geom.dy
    paths = []
    for z in altitudes:
        speed = volume.at_altitude(z)
        u = volume.at_altitude(z, volume.u)
        v = volume.at_altitude(z, volume.v)
        ok = np.isfinite(speed)
        jj, ii = np.nonzero(ok)
        df = pd.DataFrame({
            "x_m": x[ii], "y_m": y[jj], "z_asl_m": z,
            "wind_u_m_s": np.round(u[jj, ii], 3), "wind_v_m_s": np.round(v[jj, ii], 3),
            "wind_speed_m_s": np.round(speed[jj, ii], 3),
            "wind_direction_deg": np.round((270.0 - np.degrees(
                np.arctan2(v[jj, ii], u[jj, ii]))) % 360.0, 1),
        })
        path = out_dir / f"carte_{int(z):04d}m.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths
