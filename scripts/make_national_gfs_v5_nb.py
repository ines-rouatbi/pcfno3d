"""Carnets Colab de la regle configs/national/regle_national_gfs_v5_2.json (approuvee par Ines le 23/11/2026, sha256
542e676c... ; remplace la v5.1, arretee sans verdict le 23/11 par son controle d'echeances, et la v5 jamais executee ;
toutes deux gardees telles quelles) : rendre valides, par des corrections etudiees et jugees honnetement, les trois
cases du champ 3D national nourri par GFS refusees le 18/11 par regle_national_gfs_v4.

    python scripts/make_national_gfs_v5_nb.py [--commit <sha a 40 caracteres>]
    -> notebooks/current/pcfno3d_national_gfs_v5_2_C_l4.ipynb   (Colab L4 ou A100 : etapes 1 + 3, validation 2021)
       notebooks/current/pcfno3d_national_gfs_v5_2_B2_l4.ipynb  (Colab L4 ou A100 : controle puis jugement 2026)

Deja fait sous la v5.1 le 22-23/11 et REUTILISE tel quel, comme le dit la v5.2 : la partie A (GFS f0 / f24 / f48 des
instants d'entrainement, de validation et de l'echantillon 2026, MyDrive/pcfno3d_national/gfs_v5_regle/) et
l'affinage C1 (runs/national_1km_v5_gfs, epoque 2 retenue, marqueur regle_gfs_v5_1.json). Les carnets A, B1 et B2 de
la v5.1 restent dans notebooks/current/ (epingles sur b3ac84a) et ne sont plus regeneres ; le B2 de la v5.1 refuse de
tourner (aucune selection v5.1).
C (L4) : verifie C1 (marqueur de la v5.1, 12 epoques finies, best.pt = epoque 2), puis etapes 1 + 3 sur la validation
2021 (scripts/choisir_national_gfs_v5.py, inference seulement) dans le dossier NEUF validation_2021_v5_2/.
B2 (L4) : refuse sans selection v5.2, si elle dit ARRET ou si le jugement existe deja ; controle prealable (parties
vent, scripts/juger_national_gfs_v5.py controle), puis METAR 2026 (archive IEM) et jugement unique. C' s44 sans
correction (rapporte sans verdict) : RAPPORT_CP_S44 (vrai par defaut, a mettre a False faute de budget).

Budget : plafond 90 unites Colab pour toute la famille v5.1 + v5.2 (A + B1 de la v5.1, ~17 unites, puis C + B2) ;
chaque carnet affiche son cout ESTIME et refuse de tourner tant que UNITES_DEJA_UTILISEES n'est pas renseigne, ou si
l'estimation haute ferait depasser le plafond. Les carnets prennent le code du depot public au commit FIGE (COMMIT
ci-dessous ou --commit) et verifient les empreintes des regles avant tout ; tant que le commit n'est pas un vrai
commit (40 caracteres hexadecimaux), ils refusent de tourner et liberent la session. Sorties dans des dossiers NEUFS
(rien n'est supprime ni ecrase). Chaque carnet libere sa session a la fin.
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
from make_national_nb import CODE_GIT, COMMUN  # noqa: E402

# commit du depot public contenant ce code (ou --commit <sha>). Tant qu'il vaut ce texte, les carnets refusent de tourner.
COMMIT = "A_REMPLIR_COMMIT_FIGE"
NAT_REGLES = REPO / "configs" / "national"
REGLE = "regle_national_gfs_v5_2.json"
REGLE_V5_1 = "regle_national_gfs_v5_1.json"        # celle sous laquelle C1 a ete entraine (marqueur de R5)
NOMS = {"C": "pcfno3d_national_gfs_v5_2_C_l4.ipynb", "B2": "pcfno3d_national_gfs_v5_2_B2_l4.ipynb"}
ACCORD = "Ines, 23/11/2026 : regle_national_gfs_v5_2 approuvee et hachee (542e676c...)"
CONFIG_C1 = "national_v5_gfs_mixte_colab"
EPOQUES_C1 = 12
EPOQUE_C1_RETENUE = 2                              # retenue le 23/11 par le critere DSO sur 2021 (regle v5.2)
PLAFOND_UNITES = 90
# couts ESTIMES (unites Colab) : L4 ~4,8 unites/h (estimation, non mesuree)
COUTS = {"C": ("L4", 4.8, 1.0, 2.5), "B2": ("L4", 4.8, 2.5, 5.0)}


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
# BUDGET (regle v5.2) : plafond @PLAFOND@ unites Colab pour toute la famille v5.1 + v5.2 (A + B1 de la v5.1, ~17
# unites le 23/11, puis C + B2). Renseigner les unites DEJA consommees (lues sur le solde Colab). Cout : ESTIMATION.
UNITES_DEJA_UTILISEES = None
_mat, _taux, _hmin, _hmax = "@MATERIEL@", @TAUX@, @HMIN@, @HMAX@
_bas, _haut = _taux * _hmin, _taux * _hmax
print(f"cout ESTIME de la partie @PARTIE@ (@MATERIEL@, ~@TAUX@ unites/h, estimation) : {_hmin:g}-{_hmax:g} h, "
      f"soit ~{_bas:.0f}-{_haut:.0f} unites ; plafond : @PLAFOND@ unites (v5.1 + v5.2)")
if UNITES_DEJA_UTILISEES is None or float(UNITES_DEJA_UTILISEES) + _haut > @PLAFOND@:
    from google.colab import runtime
    print("budget : UNITES_DEJA_UTILISEES non renseigne, ou plafond risque d'etre depasse : rien n'est lance, "
          "demander a Ines (regle v5.2, « budget ») ; session liberee")
    runtime.unassign()
    raise SystemExit("budget non verifie")
print(f"budget : {float(UNITES_DEJA_UTILISEES):g} deja utilisees + {_haut:.0f} au plus <= @PLAFOND@")
'''

GARDE_GPU_INFERENCE = r'''
# Inference seulement (le national tient sur L4 en service). GPU d'au moins 22 Go (L4 ou A100) exige ;
# sinon la session est liberee AVANT le montage du Drive.
import subprocess as _sp
_g = _sp.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
             capture_output=True, text=True)
_nom, _mo = ([s.strip() for s in _g.stdout.strip().splitlines()[0].split(",")] if _g.returncode == 0
             and _g.stdout.strip() else ("aucun GPU", "0"))
print("GPU attribue :", _nom, f"{float(_mo) / 1024:.1f} Go")
if float(_mo) < 22 * 1024 or not any(k in _nom for k in ("L4", "A100")):
    from google.colab import runtime
    print("L4 ou A100 (>= 22 Go) exige : session liberee, rien n'a ete lance.")
    runtime.unassign()
    raise SystemExit("GPU L4 / A100 de 22 Go au moins exige")
'''

DOSSIER = r'''
GV5 = NAT / "gfs_v5_regle"                 # dossier de la famille v5 (rien d'autre n'y ecrit)
ECH = (0, 24, 48)
F = {e: GV5 / f"forcage_gfs_f{e}.csv" for e in ECH}                 # entrainement 2023-2025 + validation 2021
TN = {e: GV5 / f"gfs_f{e}_tn" for e in ECH}                         # echeances telechargees (boite nationale)
F26 = {e: GV5 / "jugement_2026" / f"forcage_gfs_2026_f{e}.csv" for e in ECH}
TN26 = {e: GV5 / f"gfs_2026_f{e}_tn" for e in ECH}
# archives deja telechargees, LUES SEULEMENT (rien n'y est ecrit) : OS7 du Grand Tunis, OS7 national s'il existe
LECTURE = {e: [d for d in (GT / "os7" / f"gfs_f{e}_instants", NAT / "os7" / f"gfs_f{e}_tn") if d.exists()] for e in ECH}
VAL = GV5 / "validation_2021_v5_2"         # etapes 1 + 3 de la v5.2 (dossier NEUF ; validation_2021/ = arret v5.1)
RES = GV5 / "resultat"                     # etape 4 (controle, puis jugement 2026)
METAR26 = GV5 / "metar_2026"               # METAR 2026 (archive IEM), telecharges seulement a l'etape 4
RCP = NAT / "runs" / "national_1km_v4_Cp_s44"          # C' s44 tel que livre (lu seulement)
R5 = NAT / "runs" / "national_1km_v5_gfs"              # C1 : entraine le 23/11 sous la v5.1, lu seulement
SHA_REGLE = "@SHA_REGLE@"                              # regle executee : v5.2
SHA_V5_1 = "@SHA_V5_1@"                                # marqueur de C1
print("accord :", "@ACCORD@")
'''

COMMUN_B = r'''
import yaml, shlex
import pandas as pd
def preparer_forcages_et_configs():
    """Forcages GFS (entrainement + validation) sur le disque local, configuration de C' s44 tel que livre."""
    for e in ECH:
        if not F[e].exists():
            raise FileNotFoundError(f"{F[e]} absent : la partie A (v5.1) doit etre faite")
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
def verifier_c1():
    """C1 REUTILISE (regle v5.2, etape 2, deja_fait) : marqueur de la v5.1, 12 epoques finies, best.pt = epoque 2."""
    import torch
    m = R5 / "regle_gfs_v5_1.json"
    if not m.exists() or json.loads(m.read_text()).get("regle_sha256") != SHA_V5_1:
        raise SystemExit(f"{m} absent ou d'une autre regle : ce n'est pas le C1 de la v5.1")
    if not any(r.get("epoch") == @EPOQUES@ - 1 for r in json.loads((R5 / "history.json").read_text())):
        raise SystemExit("affinage C1 non termine (@EPOQUES@ epoques) : rien n'est relance ici")
    ck = choisir_poids(R5)
    ep = torch.load(str(ck), map_location="cpu", mmap=True, weights_only=False).get("epoch", -2) + 1
    if ep != @EPOQUE_RETENUE@:
        raise SystemExit(f"{ck} est l'epoque {ep} ; la regle v5.2 dit @EPOQUE_RETENUE@ : verifier sur le Drive")
    print("C1 verifie :", ck, "| epoque", ep, "| marqueur de la v5.1")
    return ck
'''

REFUS = r'''
def refuser(raison):
    from google.colab import runtime
    print("partie @PARTIE@ refusee :", raison, "; session liberee")
    runtime.unassign()
    raise SystemExit(raison)
'''

PREP_C = r'''
# Partie C : refus AVANT toute extraction si la selection v5.2 existe deja (une seule selection, rien n'est ecrase)
if (VAL / "selection.json").exists():
    refuser(f"{VAL} contient deja une selection")
for _e in ECH:
    if not F[_e].exists():
        refuser(f"{F[_e]} absent : la partie A (v5.1) doit etre faite")
'''

PREP_B2 = r'''
# Partie B2 : refus AVANT toute extraction si la selection v5.2 manque, dit ARRET, ou si le jugement 2026 existe deja
_sel_f = VAL / "selection.json"
if not _sel_f.exists():
    refuser(f"{_sel_f} absent : faire d'abord le carnet C")
_sel = json.loads(_sel_f.read_text())
if _sel.get("regle_sha256") != SHA_REGLE:
    refuser("selection.json ne vient pas de la regle v5.2")
if _sel.get("arret"):
    refuser("selection : ARRET (regle, etape 3) ; 2026 n'est pas lu")
_deja = [n for n in ("verdicts_gfs_v5.json", "paires.csv", "arret.json") if (RES / n).exists()]
if _deja:
    refuser(f"{RES} contient deja {_deja} : une seule lecture de 2026, rien n'est ecrase")
'''

EXTRAIRE_VAL = r'''
# cache v3 : seulement _S et _E des instants de VALIDATION, jeu v3 (stations, grille)
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

PARTIE_C = r'''
# PARTIE C (Colab L4) : etapes 1 + 3 sur la validation 2021 (13 variantes x 3 cases) ; rien de 2022 ni de 2026
from google.colab import runtime
import traceback
try:
    LOCF = preparer_forcages_et_configs()
    INIT = choisir_poids(RCP)
    INIT_C1 = verifier_c1()
    print("C' s44 :", INIT, "| C1 :", INIT_C1)
    lancer("choisir_gfs_v5_2", f"python -u scripts/choisir_national_gfs_v5.py --cp national_v4_eval --c1 @CONFIG_C1@ "
                               f"--gfs {LOCF} --obs {G / 'obs'} --sortie {VAL} "
                               f"--checkpoint-cp {shlex.quote(str(INIT))} --checkpoint-c1 {shlex.quote(str(INIT_C1))}")
    sel = json.loads((VAL / "selection.json").read_text())
    print((VAL / "validation_gfs_v5.md").read_text())
    if sel["arret"]:
        print("ARRET (regle, etape 3) : aucune variante ne passe les trois cases sur 2021 ; 2026 n'est pas lu, "
              "le carnet B2 refusera de tourner")
    else:
        print("variante retenue :", sel["retenue"], "-> carnet B2 (L4), apres l'accord d'Ines et le budget")
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''

PARTIE_B2 = r'''
# PARTIE B2 (Colab L4 ou A100) : controle prealable (parties vent, validation 2021), puis 2026 lu UNE fois
from google.colab import runtime
import traceback
# C' s44 sans correction (rapporte sans verdict) : False faute de budget -> « non calcule faute de budget »
# (sans effet si la variante retenue est un C2 : C' s44 tourne alors de toute facon)
RAPPORT_CP_S44 = True
try:
    LOCF = preparer_forcages_et_configs()
    INIT, INIT_C1 = choisir_poids(RCP), verifier_c1()
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
        print("ARRET (regle v5.2, etape 4) : parties vent de l'entree de production differentes du cache ; 2026 non lu")
    else:
        for e in ECH:
            if not F26[e].exists():
                raise FileNotFoundError(f"{F26[e]} absent : la partie A (v5.1) n'a pas l'echantillon 2026")
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
    sha_v51 = empreinte(REGLE_V5_1)
    sha_v4 = empreinte("regle_v4_national.json")
    sha_g4 = empreinte("regle_national_gfs_v4.json")
    ancien = '(("regle_v4_national.json", "@SHA_V4@"),)'
    if ancien not in CODE_GIT:
        raise SystemExit("CODE_GIT de make_national_nb.py a change : adapter la verification des regles")
    git = (CODE_GIT.replace(ancien, f'(("regle_v4_national.json", "{sha_v4}"), ("regle_national_gfs_v4.json", '
                                    f'"{sha_g4}"), ("{REGLE_V5_1}", "{sha_v51}"), ("{REGLE}", "{sha}"))')
           .replace('"| regle v4 verifiee"', '"| regles v4, gfs_v4, gfs_v5_1 et gfs_v5_2 verifiees"')
           .replace("@COMMIT@", commit))
    garde = GARDE_COMMIT.replace("@COMMIT@", commit)
    prep = (COMMUN + git + DOSSIER.replace("@ACCORD@", ACCORD).replace("@SHA_REGLE@", sha)
            .replace("@SHA_V5_1@", sha_v51))
    commun_b = (COMMUN_B.replace("@CONFIG_C1@", CONFIG_C1).replace("@EPOQUES@", str(EPOQUES_C1))
                .replace("@EPOQUE_RETENUE@", str(EPOQUE_C1_RETENUE)))
    tete = (f"Règle `configs/national/{REGLE}` (sha256 `{sha[:12]}…`), approuvée par Ines le 23/11/2026 et vérifiée "
            "par le carnet ; elle remplace la v5.1 (arrêtée sans verdict le 23/11 par son contrôle d'échéances, avant "
            "tout calcul aux aéroports) sur ce seul point. La partie A (GFS) et l'affinage C1 (époque 2) de la v5.1 "
            f"sont réutilisés tels quels. Code : dépôt public au commit `{commit[:12]}` (le carnet refuse de tourner "
            "tant qu'il n'est pas renseigné). Sorties : `MyDrive/pcfno3d_national/gfs_v5_regle/validation_2021_v5_2/` "
            "puis `resultat/` (dossiers neufs, rien n'est supprimé ni écrasé). Choix sur la validation 2021 seulement ; "
            "2022 n'est pas relu ; l'échantillon 2026 (jours impairs, 00/06/12/18 UTC, ~608 instants) n'est lu qu'une "
            "fois, et seulement si une variante passe les trois cases sur 2021 puis si les parties vent de l'entrée de "
            "production sont identiques à celles du cache. Un instant sans l'une des trois échéances GFS est retiré "
            "pour toutes les sources et publié. GFS et ERA5 ne sont jamais appelés mesures (seuls les METAR le sont). "
            f"Aucun bruit ajouté. Plafond : {PLAFOND_UNITES} unités Colab pour v5.1 + v5.2 ; le coût affiché est une "
            "estimation. La session est libérée à la fin.")
    c = carnet([
        md("# PC-FNO3D national — règle GFS v5.2, partie C (Colab L4) : étapes 1 + 3 (validation 2021)\n\n" + tete),
        md("## 0. Commit figé, budget, GPU (L4 ou A100, au moins 22 Go)"), code(garde), code(budget("C")),
        code(GARDE_GPU_INFERENCE),
        md("## 1. Drive, code au commit figé, données ; refus si la sélection v5.2 existe déjà"),
        code(prep), code(REFUS.replace("@PARTIE@", "C") + PREP_C + EXTRAIRE_VAL),
        md("## 2. Vérification de C1, puis diagnostic et choix sur 2021 ; fin de la session"),
        code(CHOIX_POIDS + commun_b + PARTIE_C.replace("@CONFIG_C1@", CONFIG_C1)),
    ], gpu=True, gpu_type="L4")
    b2 = carnet([
        md("# PC-FNO3D national — règle GFS v5.2, partie B2 (Colab L4) : contrôle préalable puis jugement 2026\n\n"
           + tete),
        md("## 0. Commit figé, budget, GPU (L4 ou A100, au moins 22 Go)"), code(garde), code(budget("B2")),
        code(GARDE_GPU_INFERENCE),
        md("## 1. Drive, code au commit figé, données ; refus sans sélection v5.2, si ARRÊT ou si 2026 est déjà jugé"),
        code(prep), code(REFUS.replace("@PARTIE@", "B2") + PREP_B2 + EXTRAIRE_VAL),
        md("## 2. Contrôle (parties vent), METAR 2026, jugement unique ; fin de la session"),
        code(CHOIX_POIDS + commun_b + PARTIE_B2.replace("@CONFIG_C1@", CONFIG_C1)),
    ], gpu=True, gpu_type="L4")
    return {"C": c, "B2": b2}


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
