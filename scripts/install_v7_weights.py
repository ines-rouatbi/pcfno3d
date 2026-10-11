"""Verifie puis installe les poids v7 telecharges du Drive, pour les servir localement.

    python scripts/install_v7_weights.py C:/Users/inesr/Downloads
    python scripts/install_v7_weights.py <dossier> --config local_v7

Cherche dans le dossier : best_dso_v7.pt (sinon best.pt) et norm_stats.json, tels que
ranges dans aneo_fno3d_runs/v7/ sur le Drive.

Tout est VERIFIE AVANT la moindre copie : un telechargement partiel ou un mauvais fichier
ne doit jamais remplacer un modele qui fonctionne. Les controles :
  - le checkpoint se charge, et ses poids entrent SANS ecart dans l'architecture v7 ;
  - norm_stats.json a les cles et les longueurs attendues (7 variables, 3 composantes,
    un ecart-type de w par niveau) ;
  - une passe avant sur la vraie grille donne des valeurs finies et un vent plausible.
Puis copie atomique dans run_dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

CHECKPOINT_NAMES = ("best_dso_v7.pt", "best.pt")
MIN_CHECKPOINT_BYTES = 1_500_000_000      # la v7 fait 1,81 Go ; en dessous, c'est tronque


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def find_files(source: Path) -> tuple[Path, Path]:
    ckpt = next((source / n for n in CHECKPOINT_NAMES if (source / n).exists()), None)
    if ckpt is None:
        raise FileNotFoundError(
            f"aucun de {CHECKPOINT_NAMES} dans {source}. Sur le Drive : "
            "aneo_fno3d_runs/v7/best_dso_v7.pt")
    stats = source / "norm_stats.json"
    if not stats.exists():
        raise FileNotFoundError(
            f"{stats} absent. Il est indispensable : sans lui la sortie du modele ne peut "
            "pas etre remise en m/s. Sur le Drive : aneo_fno3d_runs/v7/norm_stats.json")
    return ckpt, stats


def check_stats(stats_path: Path, nz: int) -> dict:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    attendu = {"surf_mu": 7, "surf_sd": 7, "y_mu": 3, "y_sd": 3, "w_level_sd": nz}
    for key, n in attendu.items():
        if key not in stats:
            raise ValueError(f"norm_stats.json : cle '{key}' absente")
        if len(stats[key]) != n:
            raise ValueError(f"norm_stats.json : '{key}' a {len(stats[key])} valeurs, "
                             f"{n} attendues")
    if any(sd <= 0 for sd in stats["y_sd"] + stats["surf_sd"]):
        raise ValueError("norm_stats.json : ecart-type nul ou negatif")
    return stats


def check_checkpoint(ckpt_path: Path, cfg, geom) -> dict:
    import torch

    from pcfno3d.model import build_model

    size = ckpt_path.stat().st_size
    if size < MIN_CHECKPOINT_BYTES:
        raise ValueError(f"{ckpt_path.name} ne fait que {size / 1e9:.2f} Go (la v7 fait "
                         "1,81 Go) : telechargement incomplet ?")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise ValueError(f"{ckpt_path.name} : pas de cle 'model' — ce n'est pas un "
                         "checkpoint du pipeline")
    model = build_model(cfg, geom.c_in)
    # strict=True : la moindre cle manquante ou en trop est une erreur, pas un avertissement
    model.load_state_dict(ck["model"], strict=True)
    return {"epoch": ck.get("epoch"), "val_score": ck.get("val_score"), "model": model}


def smoke_inference(model, cfg, geom, stats: dict) -> dict:
    """Une passe avant sur un vrai instant : valeurs finies, vent dans un ordre plausible."""
    import numpy as np
    import torch

    from pcfno3d.cache import load_index
    from pcfno3d.dataset import assemble_input
    from pcfno3d.features import time_geo_features
    from pcfno3d.normalize import Normalizer

    norm = Normalizer(stats, geom, cfg.loss.w_level_max)
    index = load_index(cfg)
    row = index.iloc[0]
    S = np.load(cfg.paths.cache_dir / f"{row['stem']}_S.npy")
    sensor = S[7] > 0.5
    tfeat = time_geo_features(S, sensor, float(row["hour"]), int(row["doy"]), geom)
    x = assemble_input(torch.from_numpy(S[:7])[None],
                       torch.from_numpy(sensor.astype(np.float32))[None],
                       torch.from_numpy(tfeat)[None], geom, norm)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model.eval()(x)
    dt = time.perf_counter() - t0
    u, v, _ = (p[0].numpy() for p in norm.denorm(pred))
    speed = np.hypot(u, v)[geom.valid]
    if not np.isfinite(speed).all():
        raise ValueError("la passe avant produit des valeurs non finies")
    if not 0.1 < float(np.median(speed)) < 60.0:
        raise ValueError(f"vitesse mediane {np.median(speed):.2f} m/s : invraisemblable, "
                         "les statistiques ne correspondent sans doute pas aux poids")
    y_path = cfg.paths.cache_dir / f"{row['stem']}_Y.npy"
    out = {"stem": row["stem"], "inference_s": round(dt, 1),
           "vitesse_mediane_m_s": round(float(np.median(speed)), 2)}
    if y_path.exists():
        Y = np.load(y_path)
        err = np.sqrt(np.mean((u[geom.valid] - Y[0][geom.valid]) ** 2))
        out["rmse_u_global_m_s"] = round(float(err), 3)
    return out


def install(src: Path, dest: Path) -> None:
    """Copie atomique : on ecrit a cote puis on renomme."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="dossier contenant best_dso_v7.pt et norm_stats.json")
    ap.add_argument("--config", default="local_v7")
    ap.add_argument("--no-smoke", action="store_true", help="sauter la passe avant d'essai")
    args = ap.parse_args()

    from pcfno3d.config import Config
    from pcfno3d.geometry import Geometry

    cfg = Config.load(args.config)
    source = Path(args.source).expanduser()
    print(f"source : {source}")
    ckpt, stats_path = find_files(source)
    print(f"  checkpoint : {ckpt.name} ({ckpt.stat().st_size / 1e9:.2f} Go)")

    geom = Geometry(cfg, device="cpu")
    stats = check_stats(stats_path, geom.nz)
    print(f"  norm_stats.json : conforme (cible mu = {[round(x, 3) for x in stats['y_mu']]})")

    print("  chargement du checkpoint dans l'architecture v7 (strict)...")
    info = check_checkpoint(ckpt, cfg, geom)
    print(f"  poids : conformes | epoque {(info['epoch'] or 0) + 1} | "
          f"score de selection {info['val_score']}")

    if not args.no_smoke:
        print("  passe avant d'essai sur un vrai instant (CPU : ~15 s)...")
        smoke = smoke_inference(info["model"], cfg, geom, stats)
        print(f"  essai : {smoke}")
    del info

    run_dir = cfg.paths.run_dir
    print(f"\ninstallation dans {run_dir}")
    install(ckpt, run_dir / ckpt.name)
    install(stats_path, run_dir / "norm_stats.json")
    print(f"  {ckpt.name}  sha256 {sha256(run_dir / ckpt.name)[:16]}...")
    print("  norm_stats.json")
    print("\nv7 installee. Pour la servir :")
    print(f"  python -m pcfno3d -c {args.config} serve --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
