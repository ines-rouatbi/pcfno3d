"""Carnet KAGGLE de secours (quand Colab ne fournit pas de machine) : diagnostics 2 et 3 de la phase 1
nationale sur le modele national v2, SANS entrainement.

    python scripts/make_kaggle_phase1_nb.py [--commit SHA]
    -> notebooks/current/pcfno3d_national_phase1_kaggle.ipynb

Le Drive ne se monte pas sur Kaggle : les fichiers sont COPIES par rclone, dont la configuration (jeton
OAuth de l'utilisateur) vient du secret Kaggle RCLONE_CONF, jamais du depot ni du carnet. Les archives
de cache (15-20 Go chacune) ne sont pas telechargees : seuls les instants utiles sont lus a l'interieur,
par lectures partielles (`rclone cat --offset --count`). Resultats : /kaggle/working et
MyDrive/pcfno3d_national/runs/national_1km_v2/diagnostic_v_phase1_kaggle (dossier neuf, rien d'ecrase).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def md(texte: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texte.strip("\n").splitlines(keepends=True)}


def code(texte: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": texte.strip("\n").splitlines(keepends=True)}


TITRE = """
# PC-FNO3D réel NATIONAL — phase 1, diagnostics 2 et 3 (Kaggle, secours de Colab, SANS entraînement)

Évalue le modèle national **v2** déjà entraîné (`runs/national_1km_v2/best.pt`) ; rien n'est entraîné,
rien n'est généré.
- diagnostic 2 : spectres de l'erreur à 300 et 800 m au-dessus du sol, instants synoptiques du test 2022 ;
- diagnostic 3 : RMSE u, v par tranche, 60 instants d'entraînement contre 60 du test.

Les verdicts suivent `configs/national/regle_phase1_diagnostics.json` (fixée et hachée avant le calcul,
empreinte vérifiée par `scripts/juger_phase1.py`).

**Avant de lancer** (une seule fois) :
1. Compte Kaggle avec téléphone vérifié ; **Settings → Accelerator : GPU T4** (ou P100) ; **Internet : on**.
2. Sur votre PC : `rclone config` → un remote de type *drive* (votre compte Google), puis copier le contenu
   de `rclone.conf` dans **Add-ons → Secrets** sous le nom `RCLONE_CONF`, et l'attacher à ce carnet.
   Le jeton reste dans le secret : ne le collez jamais dans une cellule.

Puis **Run all**. Sur T4 / P100 le modèle tourne en float32 (pas de bf16 natif) : écarts numériques
minimes avec Colab (A100, bf16).
"""

INSTALL = r'''
import io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
from kaggle_secrets import UserSecretsClient

COMMIT = "@COMMIT@"
TMP = Path("/tmp/national")              # disque de la session (pas /kaggle/working, limite a 20 Go)
TMP.mkdir(parents=True, exist_ok=True)
print("place libre sur /tmp :", round(shutil.disk_usage("/tmp").free / 1e9), "Go")

# rclone (version figee) et sa configuration depuis le secret ; rien n'est affiche
if not Path("/usr/local/bin/rclone").exists():
    subprocess.run("cd /tmp && curl -sSLO https://downloads.rclone.org/v1.68.2/rclone-v1.68.2-linux-amd64.zip"
                   " && unzip -oq rclone-v1.68.2-linux-amd64.zip"
                   " && cp rclone-v1.68.2-linux-amd64/rclone /usr/local/bin/", shell=True, check=True)
conf = Path.home() / ".config" / "rclone" / "rclone.conf"
conf.parent.mkdir(parents=True, exist_ok=True)
conf.write_text(UserSecretsClient().get_secret("RCLONE_CONF"))
conf.chmod(0o600)
import configparser
cp = configparser.ConfigParser(); cp.read(conf)
REMOTE = next(s for s in cp.sections() if cp[s].get("type") == "drive")
NAT = f"{REMOTE}:pcfno3d_national"
RUN = f"{NAT}/runs/national_1km_v2"
print("remote :", REMOTE)
print(subprocess.run(["rclone", "lsf", NAT, "--max-depth", "1"], capture_output=True, text=True, check=True).stdout)

# code : depot public au commit fige
CODE = Path("/kaggle/working/PFE")
if not CODE.exists():
    subprocess.run(f"git init -q {CODE} && cd {CODE} && git remote add origin https://github.com/ines-rouatbi/pcfno3d.git"
                   f" && git fetch -q --depth 1 origin {COMMIT} && git checkout -q FETCH_HEAD", shell=True, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(CODE)], check=True)
sys.path[:0] = [str(CODE), str(CODE / "src"), str(CODE / "scripts")]
import torch
print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "PAS DE GPU")
'''

DONNEES = r'''
class FichierDistant(io.RawIOBase):
    """Fichier du Drive lu par morceaux (rclone cat --offset --count), assez pour zipfile."""
    BLOC = 64 << 20

    def __init__(self, chemin: str):
        self.chemin, self.pos, self._bloc = chemin, 0, (-1, b"")
        info = json.loads(subprocess.run(["rclone", "lsjson", chemin], capture_output=True, text=True,
                                         check=True).stdout)
        self.taille = int(info[0]["Size"])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.taille + off}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.taille - self.pos if n is None or n < 0 else min(n, self.taille - self.pos)
        out = bytearray()
        while n > 0:
            b0 = self.pos // self.BLOC * self.BLOC
            if self._bloc[0] != b0:
                d = subprocess.run(["rclone", "cat", "--offset", str(b0), "--count",
                                    str(min(self.BLOC, self.taille - b0)), self.chemin],
                                   capture_output=True, check=True).stdout
                self._bloc = (b0, d)
            d = self._bloc[1][self.pos - b0:self.pos - b0 + n]
            if not d:
                break
            out += d
            self.pos += len(d)
            n -= len(d)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)

# poids et statistiques du modele v2 (copie locale : le run_dir du Drive n'est jamais ecrit)
RUN_LOC = TMP / "run_national_1km_v2"
RUN_LOC.mkdir(exist_ok=True)
for f in ("best.pt", "best_dso_v7.pt", "norm_stats.json"):
    subprocess.run(["rclone", "copy", f"{RUN}/{f}", str(RUN_LOC)], check=False)
print(sorted(p.name for p in RUN_LOC.iterdir()))
assert (RUN_LOC / "norm_stats.json").exists() and any((RUN_LOC / f).exists() for f in ("best.pt", "best_dso_v7.pt"))

# jeu (meta.json, split_days.json, grille) et donnees (forcage ERA5)
for zipnom, dest in (("jeu_v2.zip", TMP / "jeu"), ("national_donnees.zip", TMP / "donnees")):
    if not dest.exists():
        subprocess.run(["rclone", "copy", f"{NAT}/{zipnom}", "/tmp"], check=True)
        with zipfile.ZipFile(f"/tmp/{zipnom}") as zf:
            zf.extractall(dest)
        os.remove(f"/tmp/{zipnom}")
FORCAGE = next((TMP / "donnees").rglob("forcage_era5.csv"))
print("forcage :", FORCAGE)

# cache : index des 3 parties, puis seulement les instants utiles
CACHE = TMP / "cache"
CACHE.mkdir(exist_ok=True)
import numpy as np, pandas as pd
archives = {k: zipfile.ZipFile(FichierDistant(f"{NAT}/cache_v2_partie_{k}.zip")) for k in (1, 2, 3)}
idx = []
for k, zf in archives.items():
    with zf.open(f"index_partie_{k}.csv") as fh:
        idx.append(pd.read_csv(fh).assign(partie=k))
idx = pd.concat(idx).drop_duplicates("stem")
idx.drop(columns="partie").to_csv(CACHE / "index.csv", index=False)
print(idx["split"].value_counts().to_string())

test = set(idx.loc[idx["split"] == "test", "stem"].astype(str))
tr = idx[idx["split"] == "train"]["stem"].astype(str).tolist()
train = {tr[i] for i in np.unique(np.linspace(0, len(tr) - 1, 60).astype(int))}   # comme la cellule Colab
voulus = test | train
for k, zf in archives.items():
    membres = [n for n in zf.namelist() if n.endswith(".npy") and n.rsplit("_", 1)[0] in voulus
               and not (CACHE / n).exists()]
    print(f"partie {k} : {len(membres)} fichiers a lire")
    for i, n in enumerate(membres):
        zf.extract(n, CACHE)
        if i % 100 == 0:
            print(f"  {i}/{len(membres)}", flush=True)
print("instants test :", sum((CACHE / f"{s}_Y.npy").exists() for s in test),
      "| entrainement :", sum((CACHE / f"{s}_Y.npy").exists() for s in train))
print("place libre sur /tmp :", round(shutil.disk_usage("/tmp").free / 1e9), "Go")
'''

LANCER = r'''
import yaml
c = yaml.safe_load(open(CODE / "configs" / "national_1km_colab.yaml"))
c["name"] = "national_eval_v2_kaggle"
c["paths"].update({"data_dir": str(TMP / "jeu"), "cache_dir": str(CACHE), "run_dir": str(RUN_LOC),
                   "forcage_era5": str(FORCAGE), "local_dir": None, "web_dir": str(TMP / "web")})
c["train"]["init_from"] = None
with open(CODE / "configs" / "national_eval_v2_kaggle.yaml", "w") as fh:
    yaml.safe_dump(c, fh, sort_keys=False)

def lancer(cmd: str) -> None:
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for ligne in p.stdout:
        print(ligne, end="")
    if p.wait():
        raise RuntimeError(f"echec ({p.returncode}) : {cmd}")

lancer("python -u scripts/diagnostic_v_phase1.py -c national_eval_v2_kaggle --n-train 60")
OUT = RUN_LOC / "diagnostic_v_phase1"
lancer(f"python -u scripts/juger_phase1.py --phase1 {OUT} --sortie {OUT / 'verdict_phase1.md'}")

# resultats : /kaggle/working et un dossier NEUF sur le Drive (rien n'est ecrase)
shutil.copytree(OUT, "/kaggle/working/diagnostic_v_phase1", dirs_exist_ok=True)
subprocess.run(["rclone", "copy", str(OUT), f"{RUN}/diagnostic_v_phase1_kaggle"], check=True)
print("resultats sur le Drive :", f"{RUN}/diagnostic_v_phase1_kaggle")
'''


def carnet(commit: str) -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
            "cells": [md(TITRE),
                      md("## 1. rclone, code"), code(INSTALL.replace("@COMMIT@", commit)),
                      md("## 2. Poids v2, jeu, forçage, instants utiles du cache"), code(DONNEES),
                      md("## 3. Diagnostics 2 et 3, verdicts, résultats"), code(LANCER)]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", default=None, help="commit du depot public (defaut : HEAD)")
    a = ap.parse_args()
    commit = a.commit or subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                        text=True, check=True).stdout.strip()
    out = REPO / "notebooks" / "current" / "pcfno3d_national_phase1_kaggle.ipynb"
    out.write_text(json.dumps(carnet(commit), indent=1, ensure_ascii=False), encoding="utf-8")
    print(out, commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
