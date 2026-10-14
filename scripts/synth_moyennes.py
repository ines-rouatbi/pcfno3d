"""Vent synthetique MOYEN du domaine, par instant et par niveau, sur les 100 jours.

Lit les CSV journaliers du generateur en flux (pyarrow, par blocs : peu de memoire) et ne
garde, pour chaque instant et chaque niveau, que les sommes de u, v et de la vitesse sur
les noeuds 3D. Resultat : un petit CSV (instants x niveaux), reutilise par
compare_synth_era5.py pour comparer les statistiques de l'annee a celles d'ERA5.

    python scripts/synth_moyennes.py -c local_v7          # reprend la ou il s'est arrete
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="local_v7")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "reanalysis" / "comparaison"
                                         / "synth_moyennes.csv"))
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.csv as pacsv

    from pcfno3d.cache import day_files, day_of
    from pcfno3d.config import Config
    from pcfno3d.geometry import Geometry

    cfg = Config.load(args.config)
    geom = Geometry(cfg, device="cpu")
    nyx = geom.ny * geom.nx
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out)["day"]) if out.exists() else set()

    cols = ["point_id", "timestamp_utc", "wind_u_m_s", "wind_v_m_s"]
    for path in day_files(cfg.paths.data_dir):
        day = day_of(path)
        if day in done:
            continue
        acc: dict[tuple[str, int], np.ndarray] = {}
        reader = pacsv.open_csv(
            path, read_options=pacsv.ReadOptions(block_size=64 << 20),
            convert_options=pacsv.ConvertOptions(include_columns=cols,
                                                 column_types={"timestamp_utc": pa.string()}))
        for batch in reader:
            pid = batch.column("point_id").to_numpy()
            node = pid >= geom.node_id_start
            if not node.any():
                continue
            ts = np.asarray(batch.column("timestamp_utc").to_pylist(), dtype=object)[node]
            u = batch.column("wind_u_m_s").to_numpy()[node].astype(np.float64)
            v = batch.column("wind_v_m_s").to_numpy()[node].astype(np.float64)
            k = geom.node_flat[pid[node] - geom.node_id_start] // nyx
            frame = pd.DataFrame({"ts": ts, "k": k, "u": u, "v": v, "s": np.hypot(u, v)})
            g = frame.groupby(["ts", "k"])[["u", "v", "s"]].agg(["sum", "count"])
            for (t, kk), row in g.iterrows():
                a = acc.setdefault((t, int(kk)), np.zeros(4))
                a += [row[("u", "sum")], row[("v", "sum")], row[("s", "sum")],
                      row[("u", "count")]]
        rows = [{"day": day, "timestamp": t, "z_m": int(geom.z_levels[kk]),
                 "u": a[0] / a[3], "v": a[1] / a[3], "vitesse": a[2] / a[3], "n": int(a[3])}
                for (t, kk), a in sorted(acc.items())]
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"jour {day:03d} : {len({r['timestamp'] for r in rows})} instants", flush=True)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
