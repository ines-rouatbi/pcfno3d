"""BLOC 3/4 — construction du cache de tenseurs, une fois par dataset.

Pour chaque instant du dataset, deux fichiers :
    <jour>_<HHMMSS>_S.npy  surface (8, ny, nx)      = u, v, w, T, rho, HR, p, capteur_dispo
    <jour>_<HHMMSS>_Y.npy  volume  (4, nz, ny, nx)  = u, v, w, rho  (0 sous le relief)
plus `index.csv` : stem, day, timestamp, split, hour (decimale), doy.

Reprise : les jours deja presents dans l'index sont ignores, donc l'operation peut etre
interrompue et relancee.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

from .config import Config
from .geometry import SURF_VARS, Geometry

# colonnes lues dans les CSV journaliers (les autres sont ignorees : lecture plus rapide)
READ_COLS = ["point_id", "timestamp_utc", "temperature_c", "air_density_kg_m3",
             "relative_humidity_pct", "pressure_pa", "wind_u_m_s", "wind_v_m_s",
             "wind_w_m_s", "mask_i_obs"]
TARGET_VARS = ["wind_u_m_s", "wind_v_m_s", "wind_w_m_s", "air_density_kg_m3"]
DAY_GLOB = "aneo_observations_day_*_multitime.csv*"


def day_of(path: Path) -> int:
    """aneo_observations_day_004_20250107_multitime.csv.gz -> 4."""
    return int(path.name.split("_")[3])


def day_files(data_dir: Path) -> list[Path]:
    """Fichiers journaliers, hors telechargements partiels de Drive."""
    return [p for p in sorted(Path(data_dir).glob(DAY_GLOB))
            if not p.name.endswith((".part", ".crdownload", ".tmp"))]


def build_day(path: Path, geom: Geometry, cache_dir: Path) -> list[dict]:
    """Un CSV journalier -> un couple (S, Y) par instant. Renvoie les lignes d'index."""
    opts = pacsv.ConvertOptions(include_columns=READ_COLS,
                                column_types={"timestamp_utc": "string"})
    table = pacsv.read_csv(path, convert_options=opts)
    pid = table.column("point_id").to_numpy()
    ts = table.column("timestamp_utc").to_pandas()
    codes, uniq = pd.factorize(ts, sort=True)
    cols = {c: table.column(c).to_numpy().astype(np.float32) for c in READ_COLS[2:]}
    del table

    day = day_of(path)
    nz, ny, nx = geom.shape
    rows: list[dict] = []
    for c, stamp in enumerate(uniq):
        sel = np.nonzero(codes == c)[0]
        p = pid[sel]
        is_surface = p < geom.node_id_start
        sel_s, sel_n = sel[is_surface], sel[~is_surface]
        ps, pn = p[is_surface], p[~is_surface]

        # surface : 8 canaux sur la grille fine, puis agregation stride x stride
        S = np.zeros((8, geom.ny_s * geom.nx_s), dtype=np.float32)
        flat = geom.pid_to_surf[ps]
        for ci, var in enumerate(SURF_VARS):
            S[ci, flat] = cols[var][sel_s]
        S[7, flat] = cols["mask_i_obs"][sel_s]       # 1 = mesure reelle, 0 = interpolee
        S = geom.pool_surface(S)

        # cible : u, v, w, rho aux noeuds 3D (0 sous le relief)
        Y = np.zeros((4, nz * ny * nx), dtype=np.float32)
        nflat = geom.node_flat[pn - geom.node_id_start]
        for ci, var in enumerate(TARGET_VARS):
            Y[ci, nflat] = cols[var][sel_n]
        Y = Y.reshape(4, nz, ny, nx)

        stamp_ts = pd.Timestamp(stamp)
        stem = f"{day:03d}_{stamp_ts.strftime('%H%M%S')}"
        np.save(cache_dir / f"{stem}_S.npy", S)
        np.save(cache_dir / f"{stem}_Y.npy", Y)
        rows.append({
            "stem": stem, "day": day, "timestamp": str(stamp_ts), "split": geom.split_of(day),
            "hour": stamp_ts.hour + stamp_ts.minute / 60 + stamp_ts.second / 3600,
            "doy": stamp_ts.dayofyear,
        })
    return rows


def build_cache(cfg: Config, geom: Geometry, days: Iterable[int] | None = None,
                verbose: bool = True) -> pd.DataFrame:
    """Construit (ou complete) le cache et renvoie l'index des instants."""
    cfg.paths.mkdirs()
    cache_dir, index_path = cfg.paths.cache_dir, cfg.paths.index_csv
    done = set(pd.read_csv(index_path)["day"]) if index_path.exists() else set()
    wanted = set(days) if days is not None else None

    files = day_files(cfg.paths.data_dir)
    if not files:
        raise FileNotFoundError(f"Aucun fichier {DAY_GLOB} dans {cfg.paths.data_dir}")

    for path in files:
        day = day_of(path)
        if day in done or (wanted is not None and day not in wanted):
            continue
        rows = build_day(path, geom, cache_dir)
        df = pd.DataFrame(rows)
        header = not index_path.exists()
        df.to_csv(index_path, mode="a", header=header, index=False)
        done.add(day)
        mirrored = mirror_day(cfg, [r["stem"] for r in rows])
        if verbose:
            print(f"jour {day:03d} : {len(rows):4d} instants -> {cache_dir.name}"
                  + (f" (+ copie : {mirrored} fichiers)" if mirrored else ""))

    index = load_index(cfg)
    ecrire_manifeste(cfg)
    if verbose:
        counts = index["split"].value_counts().to_dict()
        print(f"cache : {len(index)} instants ({counts}) ; manifeste {MANIFESTE}")
    return index


def mirror_day(cfg: Config, stems: list[str]) -> int:
    """Copie vers `paths.cache_mirror` les tenseurs d'UN jour qui vient d'etre construit.

    Jour par jour, et l'index en dernier : si la copie s'interrompt, l'index du miroir ne
    designe jamais un fichier absent. On ne recopie jamais tout le cache — il fait ~28 Go
    pour la v7 et ne change pas une fois construit.
    """
    import os
    import shutil

    mirror = cfg.paths.cache_mirror
    if mirror is None or mirror.resolve() == cfg.paths.cache_dir.resolve():
        return 0
    mirror.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem in stems:
        for suffix in ("_S.npy", "_Y.npy"):
            src = cfg.paths.cache_dir / f"{stem}{suffix}"
            if src.exists():
                shutil.copy2(src, mirror / src.name)
                n += 1
    index = cfg.paths.index_csv
    if index.exists():
        tmp = mirror / "index.csv.tmp"
        shutil.copy2(index, tmp)
        os.replace(tmp, mirror / "index.csv")
        n += 1
    return n


def load_index(cfg: Config) -> pd.DataFrame:
    """Index des instants en cache, trie, sans doublon de stem."""
    path = cfg.paths.index_csv
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent : lance d'abord `python -m pcfno3d build-cache`"
        )
    index = pd.read_csv(path)
    return (index.drop_duplicates("stem", keep="last")
            .sort_values(["day", "timestamp"]).reset_index(drop=True))


def check_ground_truth(cfg: Config, geom: Geometry, n: int = 3) -> list[dict]:
    """BLOC 4 — controle : la verite ne doit etre non nulle que sur les noeuds valides."""
    index = load_index(cfg)
    report: list[dict] = []
    for stem in index["stem"].head(n):
        Y = np.load(cfg.paths.cache_dir / f"{stem}_Y.npy")
        outside = np.abs(Y[:3][:, ~geom.valid]).max() if (~geom.valid).any() else 0.0
        inside = Y[:3][:, geom.valid]
        report.append({
            "stem": stem,
            "max_hors_domaine": float(outside),              # doit etre 0
            "u_moyen": float(inside[0].mean()),
            "vitesse_max": float(np.hypot(inside[0], inside[1]).max()),
            "rho_min": float(Y[3][geom.valid].min()),        # doit etre ~1.0-1.3
        })
    return report


# ---------------------------------------------------------------------- manifeste (BF15, L2)
MANIFESTE = "cache_manifest.json"


def _sha256(path: Path) -> str | None:
    import hashlib
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def _commit() -> str | None:
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:                                  # noqa: BLE001 — pas de git (Colab, archive)
        return None


def manifeste(cfg: Config) -> dict:
    """Empreinte du cache : ce qui a servi a le construire (geometrie, decoupage) et ce qu'il contient
    (index). Deux caches de meme manifeste donnent les memes tenseurs d'entree."""
    from .geometry import GEOMETRY_CACHE_VERSION
    idx = cfg.paths.index_csv
    index = pd.read_csv(idx) if idx.exists() else pd.DataFrame(columns=["split"])
    return {"meta_json_sha256": _sha256(cfg.paths.data_dir / "meta.json"),
            "split_days_sha256": _sha256(cfg.paths.data_dir / "split_days.json"),
            "index_csv_sha256": _sha256(idx),
            "instants": int(len(index)),
            "par_split": {str(k): int(v) for k, v in index["split"].value_counts().items()},
            "geometry_cache_version": GEOMETRY_CACHE_VERSION,
            "entree": cfg.mask.entree, "forcage": str(cfg.paths.forcage_era5) if cfg.paths.forcage_era5 else None,
            "commit": _commit()}


def ecrire_manifeste(cfg: Config) -> Path:
    import json
    import time
    m = {**manifeste(cfg), "ecrit_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    f = cfg.paths.cache_dir / MANIFESTE
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(f)
    return f


def verifier_manifeste(cfg: Config) -> list[str]:
    """Ecarts entre le manifeste ecrit et le cache actuel ([] = conforme ; absent = un ecart)."""
    import json
    f = cfg.paths.cache_dir / MANIFESTE
    if not f.exists():
        return [f"manifeste absent : {f} (pcfno3d manifeste-cache)"]
    ecrit = json.loads(f.read_text(encoding="utf-8"))
    actuel = manifeste(cfg)
    return [f"{k} : {ecrit.get(k)} -> {actuel[k]}" for k in
            ("meta_json_sha256", "split_days_sha256", "index_csv_sha256", "instants", "geometry_cache_version")
            if ecrit.get(k) != actuel[k]]
