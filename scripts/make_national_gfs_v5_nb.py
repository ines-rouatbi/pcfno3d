"""Carnets Colab de la regle configs/national/regle_national_gfs_v5_1.json (approuvee par Ines le 22/11/2026 au soir,
sha256 570713841039... ; remplace regle_national_gfs_v5, jamais executee, gardee telle quelle) : rendre valides, par
des corrections etudiees et jugees honnetement, les trois cases du champ 3D national nourri par GFS refusees le 18/11
par regle_national_gfs_v4.

    python scripts/make_national_gfs_v5_nb.py [--commit <sha a 40 caracteres>]
    -> notebooks/current/pcfno3d_national_gfs_v5_A_cpu.ipynb    (Colab CPU : GFS f0 / f24 / f48, reprise possible)
       notebooks/current/pcfno3d_national_gfs_v5_B1_a100.ipynb  (Colab A100 : affinage C1, etapes 1 + 3)
       notebooks/current/pcfno3d_national_gfs_v5_B2_l4.ipynb    (Colab L4 ou A100 : controle puis jugement 2026)

A (CPU) : GFS (archive NOAA sur AWS, boite nationale, `pcfno3d.gfs --national`) aux instants d'entrainement 2023-2025
et de validation 2021 (index du cache v3), puis a l'ECHANTILLON 2026 de la regle (2026-01-01 -> 2026-10-31, jours de
l'annee impairs, 00 / 06 / 12 / 18 UTC : `--periode ... --jours-impairs --heures 0,6,12,18`), a chacune des trois
echeances ; une echeance deja telechargee n'est pas refaite.
B1 (A100) : affinage C1 (configs/national_v5_gfs_mixte_colab.yaml, run neuf runs/national_1km_v5_gfs), etapes 1 + 3 sur
la validation 2021 (scripts/choisir_national_gfs_v5.py) ; s'arrete la. Si la selection dit ARRET, 2026 n'est jamais lu.
B2 (L4 ou A100, >= 22 Go) : refuse sans selection, si elle dit ARRET ou si le jugement existe deja ; controle prealable
(parties vent, scripts/juger_national_gfs_v5.py controle), puis METAR 2026 (archive IEM) et jugement unique. C' s44
sans correction (rapporte sans verdict) : RAPPORT_CP_S44 (vrai par defaut, a mettre a False faute de budget).

Budget (regle v5.1) : plafond 90 unites Colab pour A + B1 + B2 ; chaque carnet affiche son cout ESTIME et refuse de
tourner tant que UNITES_DEJA_UTILISEES (unites deja consommees par la regle) n'est pas renseigne, ou si l'estimation
haute ferait depasser le plafond. Les carnets prennent le code du depot public au commit FIGE (COMMIT ci-dessous) et
verifient les empreintes des regles avant tout ; tant que COMMIT n'est pas un vrai commit (40 caracteres
hexadecimaux), ils refusent de tourner et liberent la session. Sorties dans des dossiers NEUFS :
MyDrive/pcfno3d_national/gfs_v5_regle/ et runs/national_1km_v5_gfs (rien n'est supprime ni ecrase ; un entrainement
coupe est mis de cote, renomme). Chaque carnet libere sa session a la fin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_grand_tunis_nb import CHOIX_POIDS, carnet, code, md  # noqa: E402
from make_national_nb import CODE_GIT, COMMUN, GARDE_GPU, MEM, _ent_v3  # noqa: E402

# TODO (Ines) : commit du depot public contenant ce code, a renseigner une fois le commit et le push autorises
# (ou passer --commit <sha>). Tant qu'il vaut ce texte, les carnets refusent de tourner.
COMMIT = "A_REMPLIR_COMMIT_FIGE"
NAT_REGLES = REPO / "configs" / "national"
REGLE = "regle_national_gfs_v5_1.json"
NOMS = {"A": "pcfno3d_national_gfs_v5_A_cpu.ipynb", "B1": "pcfno3d_national_gfs_v5_B1_a100.ipynb",
        "B2": "pcfno3d_national_gfs_v5_B2_l4.ipynb"}
ACCORD = "Ines, 22/11/2026 (soir) : regle_national_gfs_v5_1 approuvee et hachee (570713841039...)"
CONFIG_C1 = "national_v5_gfs_mixte_colab"
EPOQUES_C1 = 12
PLAFOND_UNITES = 90
# couts ESTIMES (unites Colab) : A100 ~11,8 unites/h, L4 ~4,8 unites/h, CPU ~0,1 unite/h (estimations, non mesures)
COUTS = {"A": ("Colab CPU", 0.1, 3.0, 8.0), "B1": ("A100", 11.8, 2.0, 3.0), "B2": ("L4", 4.8, 2.5, 5.0)}


def empreinte(nom: str) -> str:
    f = NAT_REGLES / nom
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    attendu = (NAT_REGLES / (nom + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"{nom} modifiee depuis son hachage ({h} != {attendu})")
    return h


GARDE_COMMIT = r'''
# Commit FIGE du code (depot public). Tant qu'il n'est pas renseigne, rien ne tourne : session liberee.
import re as _re
COMMIT_FIGE = "@COMMIT@"
if not _re.fullmatch(r"[0-9a-f]{40}", COMMIT_FIGE):
    from google.colab import runtime
    print("commit du code non renseigne (scripts/make_national_gfs_v5_nb.py, COMMIT) : rien n'est lance, "
          "session liberee")
    runtime.unassign()
    raise SystemExit("commit non fige")
print("commit fige :", COMMIT_FIGE)
'''

BUDGET = r'''
# BUDGET (regle v5.1) : plafond @PLAFOND@ unites Colab pour toute la regle (A + B1 + B2). Renseigner les unites DEJA
# consommees par la regle (0 avant la partie A) en lisant le solde Colab. Cout de cette partie : ESTIMATION.
UNITES_DEJA_UTILISEES = None
_mat, _taux, _hmin, _hmax = "@MATERIEL@", @TAUX@, @HMIN@, @HMAX@
_bas, _haut = _taux * _hmin, _taux * _hmax
print(f"cout ESTIME de la partie @PARTIE@ (@MATERIEL@, ~@TAUX@ unites/h, estimation) : {_hmin:g}-{_hmax:g} h, "
      f"soit ~{_bas:.0f}-{_haut:.0f} unites ; plafond de la regle : @PLAFOND@ unites (A + B1 + B2)")
if UNITES_DEJA_UTILISEES is None or float(UNITES_DEJA_UTILISEES) + _haut > @PLAFOND@:
    from google.colab import runtime
    print("budget : UNITES_DEJA_UTILISEES non renseigne, ou plafond risque d'etre depasse : rien n'est lance, "
          "demander a Ines (regle v5.1, « budget ») ; session liberee")
    runtime.unassign()
    raise SystemExit("budget non verifie")
print(f"budget : {float(UNITES_DEJA_UTILISEES):g} deja utilisees + {_haut:.0f} au plus <= @PLAFOND@")
'''

GARDE_GPU_B2 = r'''
# Partie B2 : inference seulement (le national tient sur L4 en service). GPU d'au moins 22 Go (L4 ou A100) exige ;
# sinon la session est liberee AVANT le montage du Drive.
import subprocess as _sp
_g = _sp.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
             capture_output=True, text=True)
_nom, _mo = ([s.strip() for s in _g.stdout.strip().splitlines()[0].split(",")] if _g.returncode == 0
             and _g.stdout.strip() else ("aucun GPU", "0"))
print("GPU attribue :", _nom, f"{float(_mo) / 1024:.1f} Go")
if float(_mo) < 22 * 1024 or not any(k in _nom for k in ("L4", "A100")):
    from google.colab import runtime
    print("partie B2 : L4 ou A100 (>= 22 Go) exige : session liberee, rien n'a ete lance.")
    runtime.unassign()
    raise SystemExit("GPU L4 / A100 de 22 Go au moins exige")
'''

DOSSIER = r'''
GV5 = NAT / "gfs_v5_regle"                 # dossier NEUF de la regle (rien d'autre n'y ecrit)
GV5.mkdir(exist_ok=True)
ECH = (0, 24, 48)
F = {e: GV5 / f"forcage_gfs_f{e}.csv" for e in ECH}                 # entrainement 2023-2025 + validation 2021
TN = {e: GV5 / f"gfs_f{e}_tn" for e in ECH}                         # echeances telechargees (boite nationale)
F26 = {e: GV5 / "jugement_2026" / f"forcage_gfs_2026_f{e}.csv" for e in ECH}
TN26 = {e: GV5 / f"gfs_2026_f{e}_tn" for e in ECH}
# archives deja telechargees, LUES SEULEMENT (rien n'y est ecrit) : OS7 du Grand Tunis, OS7 national s'il existe
LECTURE = {e: [d for d in (GT / "os7" / f"gfs_f{e}_instants", NAT / "os7" / f"gfs_f{e}_tn") if d.exists()] for e in ECH}
VAL = GV5 / "validation_2021"              # etapes 1 + 3
RES = GV5 / "resultat"                     # etape 4 (controle, puis jugement 2026)
METAR26 = GV5 / "metar_2026"               # METAR 2026 (archive IEM), telecharges seulement a l'etape 4
RCP = NAT / "runs" / "national_1km_v4_Cp_s44"          # C' s44 tel que livre (lu seulement)
R5 = NAT / "runs" / "national_1km_v5_gfs"              # C1 : run NEUF
SHA_V5_1 = "@SHA_V5_1@"
# echantillon 2026 de la regle v5.1 : jours de l'annee impairs, heures de cycle (analyse toujours f000)
PERIODE_2026 = "--periode 2026-01-01T00:00,2026-10-31T18:00 --jours-impairs --heures 0,6,12,18"
print("accord :", "@ACCORD@")
'''

PARTIE_A = r'''
# PARTIE A (Colab CPU) : GFS f0 / f24 / f48 aux instants d'entrainement et de validation, puis a l'echantillon 2026
!pip install -q herbie-data cfgrib eccodes 2>&1 | tail -1
from google.colab import runtime
import traceback
def tolerer_code4(nom, cmd):
    # code 4 = plus de 2 % d'echeances absentes de l'archive apres 3 passes : le controle de la regle est fait par les
    # scripts de jugement sur les instants juges (validation 2021, 2026) ; ici on garde ce qui a ete telecharge
    try:
        lancer(nom, cmd)
    except RuntimeError as exc:
        if "(code 4)" not in str(exc):
            raise
        print(exc)
try:
    CACHE = Path("/content/national/cache")          # l'index du cache v3 seulement (pas les ~50 Go)
    CACHE.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    parts = []
    for k in (1, 2, 3):
        with zipfile.ZipFile(NAT / f"cache_v3_partie_{k}.zip") as zf:
            with zf.open(f"index_partie_{k}.csv") as fi:
                parts.append(pd.read_csv(fi))
    pd.concat(parts).drop_duplicates("stem").to_csv(CACHE / "index.csv", index=False)
    for e in ECH:                                     # 1. instants d'entrainement et de validation (C1, etapes 1 + 3)
        if F[e].exists():
            print(f"f{e} entrainement + validation deja ecrit : {F[e]}")
            continue
        lecture = " ".join(f"--lecture {d}" for d in LECTURE[e])
        tolerer_code4(f"gfs_v5_f{e}", f"python -u -m pcfno3d.gfs --config national_1km_colab --national --echeance {e} "
                                     f"--splits train,val --sortie {F[e]} --reprise {TN[e]} {lecture} --travailleurs 8")
    for e in ECH:                                     # 2. echantillon 2026 (etape 4 ; GFS seulement, aucune mesure lue)
        if F26[e].exists():
            print(f"f{e} 2026 deja ecrit : {F26[e]}")
            continue
        tolerer_code4(f"gfs_v5_2026_f{e}", f"python -u -m pcfno3d.gfs --config national_1km_colab --national "
                                          f"--echeance {e} {PERIODE_2026} --sortie {F26[e]} --reprise {TN26[e]} "
                                          f"--travailleurs 8")
    for nom, d in (("entrainement + validation", F), ("2026", F26)):
        for e in ECH:
            err = d[e].with_suffix(".erreurs.json")
            print(f"{nom} f{e} :", d[e].exists(), "| echeances absentes :",
                  len(json.loads(err.read_text())) if err.exists() else 0)
    print("partie A terminee : lancer le carnet B1 (A100)")
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''

COMMUN_B = r'''
import yaml, shlex
import pandas as pd
def preparer_forcages_et_configs():
    """Forcages GFS (entrainement + validation) sur le disque local, configuration de C' s44 tel que livre."""
    for e in ECH:
        if not F[e].exists():
            raise FileNotFoundError(f"{F[e]} absent : lancer d'abord le carnet A (Colab CPU)")
    locf = G / "gfs_v5"                            # le chemin lu par configs/@CONFIG_C1@.yaml
    locf.mkdir(exist_ok=True)
    for e in ECH:
        for suf in (".csv", ".npz"):
            if not (locf / F[e].with_suffix(suf).name).exists():
                shutil.copy2(F[e].with_suffix(suf), locf / F[e].with_suffix(suf).name)
    c = yaml.safe_load(open(CODE / "configs" / "national_1km_colab.yaml"))     # C' s44 tel que livre
    c["name"] = "national_v4_eval"
    c["paths"]["run_dir"], c["paths"]["local_dir"] = str(RCP), "/content/run_local_v4_sCp_s44"
    yaml.safe_dump(c, open(CODE / "configs" / "national_v4_eval.yaml", "w"), sort_keys=False)
    return locf
'''

PARTIE_B1 = r'''
# PARTIE B1 (Colab A100) : C1, puis etapes 1 + 3 sur la validation 2021 ; rien de 2026 n'est lu ici
from google.colab import runtime
import traceback
try:
    LOCF = preparer_forcages_et_configs()
    INIT = choisir_poids(RCP)
    print("C' s44 :", INIT)
    # 1. C1 = national_1km_v5_gfs : run NEUF ; un dossier d'une autre origine n'est jamais touche
    marqueur = R5 / "regle_gfs_v5_1.json"
    if R5.exists() and any(R5.iterdir()) and not marqueur.exists():
        raise SystemExit(f"{R5} existe et ne vient pas de cette regle : rien n'est ecrase (verifier sur le Drive)")
    h = R5 / "history.json"
    fini = h.exists() and any(r.get("epoch") == @EPOQUES@ - 1 for r in json.loads(h.read_text()))
    commence = h.exists() or (R5 / "best.pt").exists()
    if commence and not fini and not Path("/content/run_local_national_1km_v5_gfs/last.pt").exists():
        cote = R5.with_name(R5.name + "_interrompu_" + time.strftime("%Y%m%d_%H%M%S"))   # mis de cote, JAMAIS supprime
        shutil.move(str(R5), str(cote))
        print("entrainement interrompu mis de cote :", cote)
    R5.mkdir(parents=True, exist_ok=True)
    if not marqueur.exists():
        marqueur.write_text(json.dumps({"regle": "regle_national_gfs_v5_1.json", "regle_sha256": SHA_V5_1,
                                        "depart": str(INIT), "commit": COMMIT}, indent=1))
    if not (R5 / "norm_stats.json").exists():      # normalisation de C' s44 (memes donnees d'entrainement)
        shutil.copy2(RCP / "norm_stats.json", R5 / "norm_stats.json")
    if not fini:
        lancer("entrainer_c1_gfs_v5", f"python -u -m pcfno3d -c @CONFIG_C1@ {OPTS} "
                                      f"--set train.init_from={shlex.quote(str(INIT))} train")
    INIT_C1 = choisir_poids(R5)
    print("C1 :", INIT_C1, "|", (R5 / "history.json").read_text()[-1500:])
    # 2. etapes 1 + 3 sur la validation 2021 (13 variantes x 3 cases, choix ; aucune lecture de 2022 ni de 2026)
    if not (VAL / "selection.json").exists():
        lancer("choisir_gfs_v5", f"python -u scripts/choisir_national_gfs_v5.py --cp national_v4_eval --c1 @CONFIG_C1@ "
                                 f"--gfs {LOCF} --obs {G / 'obs'} --sortie {VAL} "
                                 f"--checkpoint-cp {shlex.quote(str(INIT))} --checkpoint-c1 {shlex.quote(str(INIT_C1))}")
    sel = json.loads((VAL / "selection.json").read_text())
    print((VAL / "validation_gfs_v5.md").read_text())
    if sel["arret"]:
        print("ARRET (regle, etape 3) : aucune variante ne passe les trois cases sur 2021 ; 2026 n'est pas lu, "
              "le carnet B2 refusera de tourner")
    else:
        print("variante retenue :", sel["retenue"], "-> carnet B2 (L4), apres verification du budget")
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''

PREP_B2 = r'''
# Partie B2 : refus AVANT toute extraction si la selection manque, dit ARRET, ou si le jugement 2026 existe deja
def refuser(raison):
    from google.colab import runtime
    print("partie B2 refusee :", raison, "; session liberee")
    runtime.unassign()
    raise SystemExit(raison)
_sel_f = VAL / "selection.json"
if not _sel_f.exists():
    refuser(f"{_sel_f} absent : faire d'abord le carnet B1")
_sel = json.loads(_sel_f.read_text())
if _sel.get("regle_sha256") != SHA_V5_1:
    refuser("selection.json ne vient pas de la regle v5.1")
if _sel.get("arret"):
    refuser("selection : ARRET (regle, etape 3) ; 2026 n'est pas lu")
_deja = [n for n in ("verdicts_gfs_v5.json", "paires.csv", "arret.json") if (RES / n).exists()]
if _deja:
    refuser(f"{RES} contient deja {_deja} : une seule lecture de 2026, rien n'est ecrase")
# cache v3 : seulement _S et _E des instants de VALIDATION (controle prealable), jeu v3 (stations, grille)
CACHE = Path("/content/national/cache")
CACHE.mkdir(parents=True, exist_ok=True)
import pandas as pd
from pcfno3d.dataset import is_synoptic
parts = []
for k in (1, 2, 3):
    with zipfile.ZipFile(NAT / f"cache_v3_partie_{k}.zip") as zf:
        with zf.open(f"index_partie_{k}.csv") as fi:
            parts.append(pd.read_csv(fi))
idx = pd.concat(parts).drop_duplicates("stem")
idx.to_csv(CACHE / "index.csv", index=False)
val = idx[(idx["split"] == "val") & idx["timestamp"].map(is_synoptic)]
voulus = {f"{s}_{x}.npy" for s in val["stem"] for x in ("S", "E")}
for k in (1, 2, 3):
    with zipfile.ZipFile(NAT / f"cache_v3_partie_{k}.zip") as zf:
        for m in zf.namelist():
            if Path(m).name in voulus and not (CACHE / m).exists():
                zf.extract(m, CACHE)
print(len(val), "instants synoptiques de validation,", sum(1 for _ in CACHE.rglob("*_S.npy")), "fichiers _S extraits")
JEU = Path("/content/national/jeu")
if not (JEU / "meta.json").exists():
    JEU.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(NAT / "jeu_v3.zip") as zf:
        zf.extractall(JEU)
'''

PARTIE_B2 = r'''
# PARTIE B2 (Colab L4 ou A100) : controle prealable (parties vent, validation 2021), puis 2026 lu UNE fois
from google.colab import runtime
import traceback
# C' s44 sans correction (rapporte sans verdict, regle v5.1) : False faute de budget -> « non calcule faute de budget »
# (sans effet si la variante retenue est un C2 : C' s44 tourne alors de toute facon)
RAPPORT_CP_S44 = True
try:
    LOCF = preparer_forcages_et_configs()
    INIT, INIT_C1 = choisir_poids(RCP), choisir_poids(R5)
    print("C' s44 :", INIT, "| C1 :", INIT_C1)
    tn_val = " ".join(f"--tn {d}" for e in ECH for d in [TN[e]] + LECTURE[e])
    if not (RES / "controle_entree.json").exists():
        try:
            lancer("controle_gfs_v5", f"python -u scripts/juger_national_gfs_v5.py controle --cp national_v4_eval "
                                      f"--gfs {LOCF} {tn_val} --obs {G / 'obs'} --sortie {RES}")
        except RuntimeError as exc:            # code 5 = controle non passe : ecrit, arret
            if "(code 5)" not in str(exc):
                raise
            print(exc)
    ctrl = json.loads((RES / "controle_entree.json").read_text())
    print(json.dumps({k: ctrl[k] for k in ("parties_vent", "rapporte_sans_arret")}, indent=1))
    if not ctrl["passe"]:
        print("ARRET (regle v5.1, etape 4) : parties vent de l'entree de production differentes du cache ; 2026 non lu")
    else:
        for e in ECH:
            if not F26[e].exists():
                raise FileNotFoundError(f"{F26[e]} absent : finir le carnet A (GFS de l'echantillon 2026)")
        codes = ",".join(pd.read_csv(CODE / "configs" / "national" / "stations_metar.csv")["code"])
        if not (METAR26 / "metar_2026.csv").exists():
            lancer("metar_2026", f"python -u scripts/historique_metar.py --debut 2025-12-31 --fin 2026-10-31 "
                                 f"--stations {codes} --sortie {METAR26}")
        tn26 = " ".join(f"--tn {TN26[e]}" for e in ECH)
        lancer("juger_gfs_v5", f"python -u scripts/juger_national_gfs_v5.py juger --cp national_v4_eval "
                               f"--c1 @CONFIG_C1@ --selection {VAL / 'selection.json'} {tn26} --obs {METAR26} "
                               f"--sortie {RES} --checkpoint-cp {shlex.quote(str(INIT))} "
                               f"--checkpoint-c1 {shlex.quote(str(INIT_C1))}"
                               + ("" if RAPPORT_CP_S44 else " --sans-cp-s44"))
        print((RES / "gfs_v5.md").read_text())
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''


def budget(partie: str) -> str:
    mat, taux, hmin, hmax = COUTS[partie]
    return (BUDGET.replace("@PLAFOND@", str(PLAFOND_UNITES)).replace("@MATERIEL@", mat).replace("@TAUX@", str(taux))
            .replace("@HMIN@", str(hmin)).replace("@HMAX@", str(hmax)).replace("@PARTIE@", partie))


def carnets(commit: str) -> dict[str, dict]:
    sha = empreinte(REGLE)
    sha_v4 = empreinte("regle_v4_national.json")
    sha_g4 = empreinte("regle_national_gfs_v4.json")
    ancien = '(("regle_v4_national.json", "@SHA_V4@"),)'
    if ancien not in CODE_GIT:
        raise SystemExit("CODE_GIT de make_national_nb.py a change : adapter la verification des regles")
    git = (CODE_GIT.replace(ancien, f'(("regle_v4_national.json", "{sha_v4}"), ("regle_national_gfs_v4.json", '
                                    f'"{sha_g4}"), ("{REGLE}", "{sha}"))')
           .replace('"| regle v4 verifiee"', '"| regles v4, gfs_v4 et gfs_v5_1 verifiees"')
           .replace("@COMMIT@", commit))
    garde = GARDE_COMMIT.replace("@COMMIT@", commit)
    prep = COMMUN + git + DOSSIER.replace("@ACCORD@", ACCORD).replace("@SHA_V5_1@", sha)
    commun_b = COMMUN_B.replace("@CONFIG_C1@", CONFIG_C1)
    b1 = PARTIE_B1.replace("@CONFIG_C1@", CONFIG_C1).replace("@EPOQUES@", str(EPOQUES_C1))
    b2 = PARTIE_B2.replace("@CONFIG_C1@", CONFIG_C1)
    tete = (f"Règle `configs/national/{REGLE}` (sha256 `{sha[:12]}…`), approuvée par Ines le 22/11/2026 (soir) et "
            f"vérifiée par le carnet ; elle remplace `regle_national_gfs_v5` (jamais exécutée). Code : dépôt public au "
            f"commit `{commit[:12]}` (le carnet refuse de tourner tant qu'il n'est pas renseigné). Sorties : "
            "`MyDrive/pcfno3d_national/gfs_v5_regle/` et `runs/national_1km_v5_gfs` (dossiers neufs, rien n'est "
            "supprimé ni écrasé). Choix sur la validation 2021 seulement ; 2022 n'est pas relu ; l'échantillon 2026 "
            "(jours impairs, 00/06/12/18 UTC, ~608 instants) n'est lu qu'une fois, et seulement si une variante passe "
            "les trois cases sur 2021 puis si les parties vent de l'entrée de production sont identiques à celles du "
            "cache. GFS et ERA5 ne sont jamais appelés mesures (seuls les METAR le sont). Aucun bruit ajouté. "
            f"Plafond de la règle : {PLAFOND_UNITES} unités Colab (A + B1 + B2) ; le coût affiché est une estimation. "
            "La session est libérée à la fin.")
    a = carnet([
        md("# PC-FNO3D national — règle GFS v5.1, partie A (Colab CPU) : GFS f0 / f24 / f48\n\n" + tete),
        md("## 0. Commit figé, budget"), code(garde), code(budget("A")),
        md("## 1. Drive, code au commit figé, données"), code(prep),
        md("## 2. GFS (archive NOAA sur AWS), boîte nationale : entraînement 2023-2025 + validation 2021, puis "
           "l'échantillon 2026 ; reprise possible ; fin de la session"),
        code(PARTIE_A),
    ], gpu=False)
    b1 = carnet([
        md("# PC-FNO3D national — règle GFS v5.1, partie B1 (Colab A100) : affinage C1, étapes 1 + 3 (validation "
           "2021)\n\n" + tete),
        md("## 0. Commit figé, budget, GPU (A100 obligatoire)"), code(garde), code(budget("B1")), code(GARDE_GPU),
        md("## 1. Drive, code au commit figé, données"), code(prep),
        md("## 2. Mémoire du GPU (recalcul des activations si besoin)"), code(MEM),
        md("## 3. Cache v3 complet (entraînement de C1)"), code(_ent_v3()),
        md("## 4. C1, puis étapes 1 + 3 ; arrêt si la sélection dit ARRÊT ; fin de la session"),
        code(CHOIX_POIDS + commun_b + b1),
    ], gpu=True)
    b2 = carnet([
        md("# PC-FNO3D national — règle GFS v5.1, partie B2 (Colab L4) : contrôle préalable puis jugement 2026\n\n"
           + tete),
        md("## 0. Commit figé, budget, GPU (L4 ou A100, au moins 22 Go)"), code(garde), code(budget("B2")),
        code(GARDE_GPU_B2),
        md("## 1. Drive, code au commit figé, données ; refus sans sélection, si ARRÊT ou si 2026 est déjà jugé"),
        code(prep), code(PREP_B2),
        md("## 2. Contrôle (parties vent), METAR 2026, jugement unique ; fin de la session"),
        code(CHOIX_POIDS + commun_b + b2),
    ], gpu=True, gpu_type="L4")
    return {"A": a, "B1": b1, "B2": b2}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", help=f"commit fige du code (defaut : la constante COMMIT, {COMMIT!r})")
    args = ap.parse_args()
    commit = args.commit or COMMIT
    if args.commit and not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit(f"--commit : 40 caracteres hexadecimaux attendus, recu {commit!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        print(f"ATTENTION : commit non renseigne ({commit}) ; les carnets refuseront de tourner")
    for k, nb in carnets(commit).items():
        f = REPO / "notebooks" / "current" / NOMS[k]
        f.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
