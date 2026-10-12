"""Fabrique l'archive du code a deposer sur le Google Drive, pour servir depuis Colab.

Le dataset et les poids sont deja sur le Drive ; il n'y manque que le code. Cette archive
ne contient QUE les sources (quelques centaines de kilo-octets) : ni donnees, ni
checkpoints, ni cles d'API.

    python scripts/package_for_colab.py
    # -> artifacts/pcfno3d_code.zip, a deposer a la racine de MyDrive

Ensuite, dans Colab, le notebook `notebooks/current/pcfno3d_api_colab.ipynb` la
decompresse et installe le paquet.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# dossiers et fichiers embarques : le code et ce qu'il faut pour l'installer
INCLUDE_DIRS = ["src", "service", "configs", "webapp", "scripts", "tests"]
INCLUDE_FILES = ["pyproject.toml", "requirements.txt", "README.md", "CLAUDE.md"]

# jamais d'archive : artefacts, caches, et tout ce qui ressemble a un secret
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints",
                 "artifacts", ".git"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".npy", ".npz", ".rar", ".zip"}
EXCLUDE_NAMES = {"api_keys.json", ".env"}


def keep(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES or path.name in EXCLUDE_NAMES:
        return False
    return True


VERSION_FILE = "PCFNO3D_VERSION.txt"


def version_stamp() -> str:
    """Commit et date du code embarque : le notebook l'affiche et refuse une archive trop
    ancienne (une copie oubliee sur le Drive, d'avant une correction)."""
    import subprocess
    from datetime import datetime

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "service",
                                "configs"], cwd=REPO_ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = "inconnu", ""
    return (f"commit {sha}{' + modifications non commitees' if dirty else ''}\n"
            f"archive {datetime.now().isoformat(timespec='seconds')}\n")


def build(out_path: Path) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(VERSION_FILE, version_stamp())
        count += 1
        for name in INCLUDE_FILES:
            src = REPO_ROOT / name
            if src.exists():
                zf.write(src, name)
                count += 1
        for folder in INCLUDE_DIRS:
            root = REPO_ROOT / folder
            if not root.exists():
                continue
            for src in sorted(root.rglob("*")):
                if src.is_file() and keep(src.relative_to(REPO_ROOT)):
                    zf.write(src, str(src.relative_to(REPO_ROOT)).replace("\\", "/"))
                    count += 1
    return count, out_path.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(REPO_ROOT / "artifacts" / "pcfno3d_code.zip"))
    args = ap.parse_args()

    out = Path(args.out)
    count, size = build(out)
    print(f"{count} fichiers, {size / 1024:.0f} Ko -> {out}")

    # garde-fou : une archive de code ne doit transporter aucun secret
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    suspects = [n for n in names if "api_keys" in n or n.endswith((".pt", ".env"))]
    if suspects:
        print("ATTENTION, fichiers sensibles dans l'archive :", suspects)
        return 1

    print("\nEtapes suivantes :")
    print(f"  1. deposer {out.name} a la racine de ton Google Drive (MyDrive)")
    print("  2. ouvrir notebooks/current/pcfno3d_api_colab.ipynb dans Colab")
    print("  3. executer les cellules : l'URL publique et la cle d'API s'affichent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
