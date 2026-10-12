"""BLOC 11/12 — reconstruction 3D d'un instant : fichiers de sortie et figures.

Deux sources d'entree :
    A) un instant deja en cache (ex. un jour de test)  -> `stem="004_120000"`
    B) un CSV de mesures de surface d'un seul instant  -> `csv=Path(...)`

Sorties dans `run_dir/reconstructions/` :
    <nom>_3d.npz          u, v, w, vitesse, direction (nz, ny, nx) + niveaux et coordonnees
    <nom>_3d.csv          un noeud 3D par ligne (format de la base ANEO)
    <nom>_carte_coupe.png carte a l'altitude demandee + coupe verticale
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .inference import Volume, WindFieldModel


def compare_to_truth(volume: Volume, y_true: np.ndarray) -> pd.DataFrame:
    """RMSE par niveau et R2 de la vitesse, quand la verite est connue.

    Seuls les niveaux couvrant au moins `dso.min_level_cover` du domaine sont gardes,
    comme dans l'evaluation. Le BLOC 12 du notebook se contentait de « au moins 10
    noeuds » et affichait un R2 de -11 899 a 25 m : sur une poignee de noeuds de fond de
    vallee, la variance de la verite est quasi nulle et le R2 n'a plus de sens.
    """
    geom = volume.geom
    min_cover = geom.cfg.dso.min_level_cover
    rows = []
    for k in range(geom.nz):
        m = geom.valid[k]
        if m.sum() < 10 or geom.cover[k] < min_cover:
            continue
        rmse = [float(np.sqrt(np.mean((p[k][m] - y_true[c, k][m]) ** 2)))
                for c, p in enumerate((volume.u, volume.v, volume.w))]
        true_speed = np.hypot(y_true[0, k][m], y_true[1, k][m])
        r2 = 1 - np.sum((volume.speed[k][m] - true_speed) ** 2) \
            / (np.sum((true_speed - true_speed.mean()) ** 2) + 1e-12)
        rows.append({"z_m": int(geom.z_levels[k]), "rmse_u": rmse[0], "rmse_v": rmse[1],
                     "rmse_w": rmse[2], "r2_vitesse": float(r2)})
    return pd.DataFrame(rows)


def plot_map_and_section(volume: Volume, altitude: float, out_path: Path,
                         section_row: int | None = None) -> Path:
    """Carte du vent a une altitude + coupe verticale (matplotlib, sans affichage)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geom = volume.geom
    row = geom.ny // 2 if section_row is None else section_row
    x_km = np.arange(geom.nx) * geom.dx / 1000
    y_km = np.arange(geom.ny) * geom.dy / 1000

    spd = volume.at_altitude(altitude)
    u_map = volume.at_altitude(altitude, volume.u)
    v_map = volume.at_altitude(altitude, volume.v)

    fig, ax = plt.subplots(1, 2, figsize=(17, 5))
    im = ax[0].imshow(spd, origin="lower", cmap="viridis",
                      extent=[0, x_km[-1], 0, y_km[-1]])
    step = max(geom.nx // 25, 1)
    Xq, Yq = np.meshgrid(x_km[::step], y_km[::step])
    ax[0].quiver(Xq, Yq, u_map[::step, ::step], v_map[::step, ::step], color="white",
                 width=0.002)
    ax[0].set_title(f"Vent a {altitude:.0f} m ASL — {volume.timestamp}")
    ax[0].set_xlabel("x (km)"); ax[0].set_ylabel("y (km)")
    plt.colorbar(im, ax=ax[0], label="Vitesse (m/s)")

    im = ax[1].pcolormesh(x_km, geom.z_levels, volume.speed[:, row, :], cmap="viridis",
                          shading="nearest")
    ax[1].fill_between(x_km, 0, geom.zt[row], color="saddlebrown", alpha=0.9, label="relief")
    ax[1].set_ylim(0, geom.z_levels[-1])
    ax[1].set_title(f"Coupe verticale (iy = {row})")
    ax[1].set_xlabel("x (km)"); ax[1].set_ylabel("Altitude ASL (m)")
    ax[1].legend(loc="upper right")
    plt.colorbar(im, ax=ax[1], label="Vitesse (m/s)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def reconstruct(cfg: Config, stem: str | None = None, csv: str | Path | None = None,
                checkpoint: str | Path | None = None, keep: float = 1.0,
                altitude: float = 1000.0, section_row: int | None = None,
                figures: bool = True, web: bool = False, device: str | None = None,
                verbose: bool = True) -> dict:
    """Reconstruit un instant et ecrit npz + csv (+ figures, + export web)."""
    wf = WindFieldModel(cfg, checkpoint=checkpoint, device=device, verbose=verbose)
    out_dir = cfg.paths.run_dir / "reconstructions"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = None
    if csv is not None:
        volume = wf.predict_csv(csv, keep=keep)
        name = f"recon_{pd.Timestamp(volume.timestamp).strftime('%Y%m%d_%H%M%S')}"
    else:
        from .cache import load_index
        if stem is None:
            index = load_index(cfg)
            test = index[index["split"] == "test"]
            if test.empty:
                raise ValueError("Aucun instant de test : precise --stem ou --csv")
            stem = str(test["stem"].iloc[0])
        volume = wf.predict_stem(stem, keep=keep)
        name = f"recon_{stem}"
        y_path = cfg.paths.cache_dir / f"{stem}_Y.npy"
        if y_path.exists():
            y_true = np.load(y_path)

    npz = volume.save_npz(out_dir / f"{name}_3d.npz")
    nodes = volume.to_nodes_frame()
    csv_path = out_dir / f"{name}_3d.csv"
    nodes.to_csv(csv_path, index=False)

    result = {
        "timestamp": volume.timestamp, "capteurs": volume.n_sensors,
        "inference_s": volume.inference_s, "noeuds": int(len(nodes)),
        "npz": str(npz), "csv": str(csv_path),
    }
    if verbose:
        print(f"instant {volume.timestamp} | capteurs utilises : {volume.n_sensors:,} | "
              f"reconstruction en {volume.inference_s * 1000:.0f} ms")

    if y_true is not None:
        table = compare_to_truth(volume, y_true)
        table.to_csv(out_dir / f"{name}_vs_verite.csv", index=False)
        result["vs_verite"] = table.to_dict(orient="records")
        if verbose:
            print(table.to_string(index=False))

    if figures:
        # 1000 m par defaut (le haut de la grille v7) : on le ramene dans la grille si
        # celle-ci est moins haute, plutot que d'echouer sur « altitude hors domaine »
        levels = wf.geom.z_levels
        altitude = float(min(max(altitude, levels[0]), levels[-1]))
        fig = plot_map_and_section(volume, altitude, out_dir / f"{name}_carte_coupe.png",
                                   section_row)
        result["figure"] = str(fig)

    if web:
        from .export import export_web
        result["web"] = str(export_web(volume, cfg.paths.web_root / name))

    if verbose:
        print(f"sauvegarde : {name}_3d.npz et {name}_3d.csv ({len(nodes):,} noeuds) -> {out_dir}")
    return result
