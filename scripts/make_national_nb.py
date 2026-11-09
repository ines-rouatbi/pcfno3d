"""Carnets Colab du modele NATIONAL (Tunisie entiere, noeuds 1 km) : la methode du Grand Tunis v4
etendue au pays (configs/national_1km_colab.yaml).

    python scripts/make_national_nb.py
    -> notebooks/current/pcfno3d_national_generation_colab.ipynb  (jeu, environnement CPU)
       notebooks/current/pcfno3d_national_colab.ipynb             (entrainement + test, A100)

Sur le Drive :
    MyDrive/pcfno3d_grand_tunis/pcfno3d_code.zip    le code (le meme que pour le Grand Tunis)
    MyDrive/pcfno3d_grand_tunis/runs/<livre>/       les poids de depart (modele Grand Tunis livre)
    MyDrive/pcfno3d_national/national_donnees.zip   grille, grille_reelle.json, forcage ERA5
                                                    (scripts/package_national_colab.py)
Le jeu (1 000 instants) est genere en 3 parties ; chaque partie terminee est un point de reprise
(cache_v2_partie_<k>.zip dans MyDrive/pcfno3d_national/). Environ 50 Go au total : verifier la place
sur le Drive avant de lancer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_grand_tunis_nb import CHOIX_POIDS, carnet, code, md  # noqa: E402

# niveaux des noeuds (m au-dessus de la mer) : ceux de la v7 puis jusqu'a 2500 m (relief 1544 m)
# jeu v2 (07/11/2026) : forcage aux profils combles au-dessus de 700 hPa ; le premier jeu
# (cache_partie_*.zip) avait un vent nul a 0,5 % des noeuds de 2500 m et n'est plus lu
NIVEAUX = "25,50,75,100,150,200,300,400,500,650,800,1000,1250,1500,2000,2500"

COMMUN = r'''
from google.colab import drive
import os as _os
if not _os.path.isdir("/content/drive/MyDrive"):   # deja monte (panneau Fichiers) : ne pas redemander
    drive.mount("/content/drive")

import json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

GT = Path("/content/drive/MyDrive/pcfno3d_grand_tunis")   # code et poids du Grand Tunis
NAT = Path("/content/drive/MyDrive/pcfno3d_national")
NAT.mkdir(parents=True, exist_ok=True)
JOURNAUX = NAT / "journaux"
JOURNAUX.mkdir(exist_ok=True)
libre = shutil.disk_usage("/content/drive/MyDrive").free / 1e9
print(f"place libre sur le Drive : {libre:.0f} Go (le jeu national en demande ~50)")

CODE = Path("/content/PFE")
if CODE.exists():
    shutil.rmtree(CODE)
CODE.mkdir(parents=True)
with zipfile.ZipFile(GT / "pcfno3d_code.zip") as zf:
    zf.extractall(CODE)
tampon = CODE / "PCFNO3D_VERSION.txt"
print(tampon.read_text().strip() if tampon.exists() else "tampon : absent")
%cd /content/PFE
!pip install -q --no-deps -e . 2>&1 | tail -2
!pip install -q pandera xarray netCDF4 h5netcdf 2>&1 | tail -2
sys.path[:0] = ["/content/PFE", "/content/PFE/src"]

def lancer(nom: str, commande: str) -> None:
    """Commande obligatoire ; sortie affichee ET journalisee sur le Drive ; arret si echec."""
    t0 = time.time()
    with open(JOURNAUX / f"{nom}.log", "a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} : {commande}\n")
        p = subprocess.Popen(commande, shell=True, cwd="/content/PFE", stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for ligne in p.stdout:
            print(ligne, end="")
            log.write(ligne)
            log.flush()
        p.wait()
    print(f"[{nom}] {time.time() - t0:.0f} s, code {p.returncode}")
    if p.returncode:
        raise RuntimeError(f"{nom} a echoue (code {p.returncode}) : voir {JOURNAUX / (nom + '.log')}")

# donnees : grille nationale, grille_reelle.json, forcage ERA5 (une archive sur le Drive)
DON = Path("/content/donnees")
G = DON / "national"
if not (G / "forcage_era5.csv").exists():
    with zipfile.ZipFile(NAT / "national_donnees.zip") as zf:
        zf.extractall(DON)
print(sorted(p.name for p in G.iterdir()))
'''

GEN = r'''
TEST = "2022-01-01:2022-03-31,2022-07-01:2022-08-31"     # memes blocs que le Grand Tunis
VAL = "2021-01-01:2021-02-28,2021-07-01:2021-07-31"
LOC = Path("/content/national")
JEU = LOC / "jeu"
JEU.mkdir(parents=True, exist_ok=True)
INFO = json.loads((G / "grille_reelle.json").read_text(encoding="utf-8"))
print("domaine :", INFO["zone"], "| echelle est-ouest :", INFO.get("facteur_echelle_est_ouest"))

if (NAT / "calendrier.json").exists():
    shutil.copy2(NAT / "calendrier.json", JEU / "calendrier.json")

PARTIES = {1: [("train", "1/2")], 2: [("train", "2/2"), ("val", None)], 3: [("test", None)]}
faites = [k for k in PARTIES if (NAT / f"cache_v2_partie_{k}.zip").exists()]
PARTIE = next((k for k in PARTIES if k not in faites), None)
print("parties deja sur le Drive :", faites, "| a faire maintenant :", PARTIE)
'''

GEN_RUN = r'''
if PARTIE is None:
    print("les 3 parties sont sur le Drive : passer au carnet d'entrainement")
else:
    cache = LOC / f"cache_p{PARTIE}"
    if cache.exists():
        shutil.rmtree(cache)                 # partie interrompue : on la refait entiere
    nw = max(1, (os.cpu_count() or 2) - 1)
    base = (f"python -u scripts/generate_aneo_3d.py --input-grid {G / 'aneo_grid_points_surface.csv'} "
            f"--output-dir {JEU} --cache-dir {cache} --forcage {G / 'forcage_era5.csv'} "
            f"--nx {INFO['zone']['nx']} --ny {INFO['zone']['ny']} --dx {INFO['zone']['dx_m']} "
            f"--niveaux NIVEAUX_NOEUDS --f-coriolis {INFO['f_coriolis']} "
            f"--perturbations 0.4 --test {TEST} --val {VAL} --marge-jours 7 "
            f"--instants 1000 --heures-par-jour 4 --workers {nw}")
    for jeux, part in PARTIES[PARTIE]:
        lancer(f"generation_p{PARTIE}", base + f" --jeux {jeux}" + (f" --partie {part}" if part else ""))
    if not (NAT / "calendrier.json").exists():
        shutil.copy2(JEU / "calendrier.json", NAT / "calendrier.json")

    # jours du calendrier sans forcage pour cette partie : refuse le point de reprise
    cal = json.loads((JEU / "calendrier.json").read_text())
    import pandas as pd
    t = pd.to_datetime(pd.read_csv(G / "forcage_era5.csv", usecols=["timestamp_utc"])["timestamp_utc"], utc=True)
    forces = set(t.dt.tz_localize(None).dt.normalize())
    idx = pd.read_csv(cache / "index.csv")
    attendus = {k for jeux, _ in PARTIES[PARTIE] for k in cal["split"][f"{jeux}_days"]}
    dates = {d["day_index"]: d["date"] for d in cal["days"]}
    manque = sorted(dates[k] for k in attendus if pd.Timestamp(dates[k]) not in forces)
    print(f"partie {PARTIE} : {len(idx)} instants generes ; jours sans forcage ERA5 : {len(manque)}")
    if manque and PARTIE in (2, 3):
        raise RuntimeError(f"forcage ERA5 absent pour {manque[:5]}... : deposer un forcage a jour")

    tmp = NAT / f"cache_v2_partie_{PARTIE}.zip.part"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in sorted(cache.iterdir()):
            nom = f"index_partie_{PARTIE}.csv" if f.name == "index.csv" else f.name
            zf.write(f, nom)
    tmp.replace(NAT / f"cache_v2_partie_{PARTIE}.zip")
    if not (NAT / "jeu_v2.zip").exists():
        with zipfile.ZipFile(NAT / "jeu_v2.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for f in sorted(JEU.iterdir()):
                if f.is_file():
                    zf.write(f, f.name)
    print(f"point de reprise {PARTIE}/3 sur le Drive : "
          f"{(NAT / f'cache_v2_partie_{PARTIE}.zip').stat().st_size / 1e9:.1f} Go")
'''.replace("NIVEAUX_NOEUDS", NIVEAUX)

ENT = r'''
LOC = Path("/content/national")
CACHE, JEU = LOC / "cache", LOC / "jeu"
for k in (1, 2, 3):
    if not (NAT / f"cache_v2_partie_{k}.zip").exists():
        raise FileNotFoundError(f"cache_v2_partie_{k}.zip absent : finir le carnet de generation")
if not (CACHE / "index.csv").exists():
    CACHE.mkdir(parents=True, exist_ok=True)
    for k in (1, 2, 3):
        with zipfile.ZipFile(NAT / f"cache_v2_partie_{k}.zip") as zf:
            zf.extractall(CACHE)
    with zipfile.ZipFile(NAT / "jeu_v2.zip") as zf:
        zf.extractall(JEU)
    import pandas as pd
    idx = pd.concat([pd.read_csv(CACHE / f"index_partie_{k}.csv") for k in (1, 2, 3)])
    idx.drop_duplicates("stem").to_csv(CACHE / "index.csv", index=False)
import pandas as pd
print(pd.read_csv(CACHE / "index.csv")["split"].value_counts().to_string())
lancer("controle", "python -u -m pcfno3d -c national_1km_colab check")
'''

MEM = r'''
import torch
from pcfno3d.config import Config
from pcfno3d.geometry import Geometry
from pcfno3d.model import build_model

def essai(recalcul: bool) -> float:
    cfg = Config.load("configs/national_1km_colab.yaml")
    cfg.model.recalcul_activations = recalcul
    geom = Geometry(cfg, device="cuda")
    m = build_model(cfg, geom.c_in).cuda().train()
    x = torch.randn(cfg.train.batch_size, geom.c_in, *geom.shape, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(x).float().pow(2).mean().backward()
    pic = torch.cuda.max_memory_allocated() / 1e9
    del m, x
    torch.cuda.empty_cache()
    return pic

total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(torch.cuda.get_device_name(0), f"{total:.0f} Go")
try:
    pic = essai(False)
    RECALCUL = pic > 0.85 * total
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    pic, RECALCUL = float("nan"), True
if RECALCUL:
    pic = essai(True)
print(f"pic {pic:.1f} Go sur {total:.0f} | recalcul des activations : {RECALCUL}")
OPTS = f"--set model.recalcul_activations={str(RECALCUL).lower()}"
'''

TRAIN = r'''
import shlex
from google.colab import runtime
import traceback
RUNS_GT = GT / "runs"
DECISION = RUNS_GT / "decision_v4.json"
livre = json.loads(DECISION.read_text())["modele_livre"] if DECISION.exists() else "grand_tunis_v2_long"
INIT = shlex.quote(str(choisir_poids(RUNS_GT / livre)))
RUN = NAT / "runs" / "national_1km_v2"
print("depart :", INIT)
try:
    # choix de l'epoque sur DSO1, DSO2, DSO3 de la validation, passe ici aussi : le carnet marche
    # alors avec une archive de code anterieure a ce reglage dans configs/national_1km_colab.yaml
    SEL = shlex.quote('train.select_criteres=["dso1_rmse","dso2_masse","dso3_rugosite"]')
    lancer("entrainer", f"python -u -m pcfno3d -c national_1km_colab {OPTS} --set {SEL} --set train.sync_last=true --set train.init_from={INIT} train")
    if not (RUN / "eval_pcfno3d" / "best_wcont" / "test_summary.json").exists():
        lancer("evaluer", f"python -u -m pcfno3d -c national_1km_colab {OPTS} evaluate")
    # DSO6 (fiabilite) sans reentrainement : ensemble par masquage, calibration sur la VALIDATION,
    # couverture 80 % et correlation incertitude / erreur sur le test (lu une fois)
    CAL = RUN / "fiabilite_pcfno3d" / "best_wcont" / "calibration_validation.json"
    D6 = RUN / "eval_pcfno3d" / "best_wcont" / "dso6_modele_livre.json"
    if not CAL.exists():
        lancer("calibrer_dso6", f"python -u -m pcfno3d -c national_1km_colab {OPTS} calibrate-reliability --out {CAL}")
    if not D6.exists():
        try:
            lancer("dso6", f"python -u -m pcfno3d -c national_1km_colab {OPTS} dso6 --calibration {CAL} --out {D6}")
        except RuntimeError as exc:                     # code 3 = NON CONFORME : verdict ecrit
            if "(code 3)" not in str(exc):
                raise
            print(exc)
    # validation sur les VRAIES mesures : 12 aeroports METAR (chacun retire de l'entree quand il
    # est juge) et radiosondages de Tunis-Carthage ; test 2022
    if (G / "obs").exists() and not (RUN / "validation_reelle" / "test" / "bilan.json").exists():
        lancer("validation_reelle", f"python -u scripts/valider_grand_tunis_reel.py -c national_1km_colab "
                                    f"--obs {G / 'obs'} --jeu test")
    # DSO nationaux (docs/dso_national.md) : gain sur ERA5 interpole par niveau, aeroports ;
    # code 3 = au moins un DSO national non conforme, verdict ecrit
    if not (RUN / "eval_pcfno3d" / "best_wcont" / "verdict_national.json").exists():
        try:
            lancer("verdict_national", f"python -u -m pcfno3d -c national_1km_colab {OPTS} verdict-national")
        except RuntimeError as exc:
            if "(code 3)" not in str(exc):
                raise
            print(exc)
    for f in sorted(RUN.rglob("test_summary.json")):
        print(f); print(f.read_text()[:4000])
    for f in sorted(RUN.rglob("test_par_niveau_synoptique.csv")):
        print(pd.read_csv(f).round(3).to_string())
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''


def generation() -> dict:
    return carnet([
        md(f"""
# PC-FNO3D réel NATIONAL (Tunisie, nœuds 1 km) — génération du jeu (Colab, CPU)

Vérité d'entraînement = le générateur forcé par **ERA5**, sur le relief (Copernicus DEM) et le sol
(ESA WorldCover) réels de toute la Tunisie ; **aucune station n'y entre**. Surface 500 m
(1656 × 776), nœuds 1 km, niveaux {NIVEAUX} m au-dessus de la mer. 1 000 instants, mêmes blocs de
test et de validation que le Grand Tunis.

**Environnement : CPU, mémoire RAM élevée.** Un seul **Tout exécuter** par partie : le carnet fait
la partie suivante qui manque sur le Drive (`MyDrive/pcfno3d_national/`). **Place nécessaire : ~50 Go
sur le Drive** (affichée par la première cellule).

| partie | contenu | point de reprise |
|---|---|---|
| 1 | entraînement, 1re moitié | `cache_v2_partie_1.zip` |
| 2 | entraînement, 2e moitié + validation | `cache_v2_partie_2.zip` |
| 3 | test (janv.–mars et juillet–août 2022) | `cache_v2_partie_3.zip` |
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN),
        md("## 2. Calendrier et partie à faire"),
        code(GEN),
        md("## 3. Génération de la partie, puis point de reprise sur le Drive"),
        code(GEN_RUN),
    ], gpu=False)


def entrainement() -> dict:
    return carnet([
        md("""
# PC-FNO3D réel NATIONAL (Tunisie, nœuds 1 km) — entraînement et test (Colab, A100)

Même méthode que le Grand Tunis v4 (34 canaux : ERA5 aux nœuds + forçage de l'instant, sortie u, v
= ERA5 + correction, w par continuité), sur le jeu national (carnet de génération, 3 parties sur le
Drive). Entrée de production : ERA5 10 m tous les 2 km + les 12 aéroports METAR. **Départ : les
poids du modèle Grand Tunis livré** (`decision_v4.json`), simple point de départ : l'échelle d'un
mode de Fourier est ici 10 fois plus grande.

**Environnement : GPU A100.** Reprise automatique : relancer **Tout exécuter** après une coupure
reprend à la dernière copie sur le Drive. La session s'arrête à la fin. Résultats :
`MyDrive/pcfno3d_national/runs/national_1km_v2/`.
"""),
        md("## 1. Drive, code et données"),
        code(COMMUN),
        md("## 2. Cache (3 parties du Drive) et contrôle de la vérité"),
        code(ENT),
        md("## 3. Mémoire du GPU"),
        code(MEM),
        md("## 4. Entraînement, test 2022, fin de la session"),
        code(CHOIX_POIDS + TRAIN),
    ], gpu=True)


def main() -> int:
    out = REPO / "notebooks" / "current"
    for nom, nb in (("pcfno3d_national_generation_colab.ipynb", generation()),
                    ("pcfno3d_national_colab.ipynb", entrainement())):
        (out / nom).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(out / nom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
