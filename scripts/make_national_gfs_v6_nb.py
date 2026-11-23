"""Carnet Colab de la regle configs/national/regle_national_gfs_v6.json (approuvee par Ines le 23/11/2026, sha256
9babee63...) : choix sur la validation 2021 avec la marge (+3 %), GFS et METAR du 1er au 23 novembre 2026, controle
prealable, puis jugement UNIQUE ; tout sur une L4 (ou A100), en une session.

    python scripts/make_national_gfs_v6_nb.py --commit <sha a 40 caracteres>
    -> notebooks/current/pcfno3d_national_gfs_v6_l4.ipynb

Le carnet refuse de tourner : sans commit fige, sans UNITES_DEJA_UTILISEES, sur un GPU de moins de 22 Go (ou sans
GPU : la session est liberee), si la selection v6 ou le jugement existent deja. Il reutilise les resultats 2021 de
regle_national_gfs_v5_2 (gfs_v5_regle/validation_2021_v5_2/variantes.json), les poids de C' s44 et de C1, et ecrit
dans un dossier NEUF : MyDrive/pcfno3d_national/gfs_v6_regle/. Il libere sa session a la fin.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_national_gfs_v5_nb as V5  # noqa: E402
from make_grand_tunis_nb import CHOIX_POIDS, carnet, code, md  # noqa: E402
from make_national_nb import CODE_GIT, COMMUN  # noqa: E402

REGLE = "regle_national_gfs_v6.json"
NOM = "pcfno3d_national_gfs_v6_l4.ipynb"
ACCORD = "Ines, 23/11/2026 : regle_national_gfs_v6 approuvee et hachee (9babee63...)"
# cout ESTIME : L4 ~4,8 unites/h (Colab affichait ~1,9 le 23/11), 1,5 a 3 h ; plafond de la regle : 15 unites
TAUX, HMIN, HMAX, PLAFOND_REGLE, PLAFOND_FAMILLE = 4.8, 1.5, 3.0, 15, 90
PERIODE = "2026-11-01T00:00,2026-11-23T21:00"

BUDGET = r'''
# BUDGET (regle v6) : au plus @PLAFOND_REGLE@ unites pour cette regle, et le plafond de @PLAFOND_FAMILLE@ unites de la famille
# v5 + v6 (22,8 deja utilisees le 23/11). Renseigner les unites DEJA consommees (solde Colab). Cout : ESTIMATION.
UNITES_DEJA_UTILISEES = None
_haut = @TAUX@ * @HMAX@
print(f"cout ESTIME (L4, ~@TAUX@ unites/h) : @HMIN@-@HMAX@ h, soit au plus ~{_haut:.0f} unites")
if UNITES_DEJA_UTILISEES is None or _haut > @PLAFOND_REGLE@ or float(UNITES_DEJA_UTILISEES) + _haut > @PLAFOND_FAMILLE@:
    from google.colab import runtime
    print("budget : UNITES_DEJA_UTILISEES non renseigne, ou plafond risque d'etre depasse : rien n'est lance ; session liberee")
    runtime.unassign()
    raise SystemExit("budget non verifie")
print(f"budget : {float(UNITES_DEJA_UTILISEES):g} deja utilisees + {_haut:.0f} au plus <= @PLAFOND_FAMILLE@")
'''

# garde GPU : une session sans GPU (nvidia-smi absent) est liberee, comme une trop petite (le 23/11, un environnement
# CPU s'est connecte et l'ancienne garde plantait sur FileNotFoundError sans liberer la session)
GARDE_GPU = r'''
import subprocess as _sp
try:
    _g = _sp.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                 capture_output=True, text=True)
    _nom, _mo = ([s.strip() for s in _g.stdout.strip().splitlines()[0].split(",")] if _g.returncode == 0
                 and _g.stdout.strip() else ("aucun GPU", "0"))
except FileNotFoundError:
    _nom, _mo = "aucun GPU (nvidia-smi absent)", "0"
print("GPU attribue :", _nom, f"{float(_mo) / 1024:.1f} Go")
if float(_mo) < 22 * 1024 or not any(k in _nom for k in ("L4", "A100")):
    from google.colab import runtime
    print("L4 ou A100 (>= 22 Go) exige : session liberee, rien n'a ete lance.")
    runtime.unassign()
    raise SystemExit("GPU L4 / A100 de 22 Go au moins exige")
'''

DOSSIER_V6 = r'''
GV6 = NAT / "gfs_v6_regle"                 # dossier NEUF de la regle v6 (rien d'autre n'y ecrit)
GV6.mkdir(exist_ok=True)
RES6 = GV6 / "resultat"
TN_NOV = {e: GV6 / f"gfs_novembre_f{e}_tn" for e in ECH}
F_NOV = {e: GV6 / f"forcage_gfs_novembre_f{e}.csv" for e in ECH}
METAR_NOV = GV6 / "metar_novembre"
VARIANTES_2021 = GV5 / "validation_2021_v5_2" / "variantes.json"
SHA_V6 = "@SHA_V6@"
PERIODE_NOV = "--periode @PERIODE@"
print("accord :", "@ACCORD@")
'''

REFUS_V6 = r'''
def refuser(raison):
    from google.colab import runtime
    print("regle v6 refusee :", raison, "; session liberee")
    runtime.unassign()
    raise SystemExit(raison)
if (GV6 / "selection.json").exists():
    refuser(f"{GV6} contient deja une selection (une seule)")
_deja = [n for n in ("verdicts_gfs_v6.json", "paires.csv", "arret.json") if (RES6 / n).exists()]
if _deja:
    refuser(f"{RES6} contient deja {_deja} : une seule lecture de novembre")
if not VARIANTES_2021.exists():
    refuser(f"{VARIANTES_2021} absent (resultats 2021 de la v5.2)")
'''

PARTIE_V6 = r'''
# REGLE v6 : choix 2021 (marge +3 %) -> GFS + METAR de novembre -> controle -> jugement unique ; fin de la session
from google.colab import runtime
import traceback
def tolerer_code4(nom, cmd):
    # code 4 = plus de 2 % d'echeances absentes apres 3 passes : le jugement l'arrete lui-meme (controle_d_echeances)
    try:
        lancer(nom, cmd)
    except RuntimeError as exc:
        if "(code 4)" not in str(exc):
            raise
        print(exc)
try:
    # 1. choix sur 2021 : aucun calcul de modele
    try:
        lancer("choisir_gfs_v6", f"python -u scripts/juger_national_gfs_v6.py choisir --variantes {VARIANTES_2021} "
                                 f"--sortie {GV6}")
    except RuntimeError as exc:                      # code 6 = aucune variante admissible : ARRET ecrit
        if "(code 6)" not in str(exc):
            raise
    sel = json.loads((GV6 / "selection.json").read_text())
    if sel["arret"]:
        print("ARRET (regle v6, etape 1) :", sel["raison"])
    else:
        print("variante retenue :", sel["retenue"], sel["variante"])
        # 2. GFS de novembre (archive NOAA AWS), trois echeances ; 3. METAR de novembre (IEM)
        !pip install -q herbie-data cfgrib eccodes 2>&1 | tail -1
        for e in ECH:
            if not F_NOV[e].exists():
                tolerer_code4(f"gfs_v6_f{e}", f"python -u -m pcfno3d.gfs --config national_1km_colab --national "
                                              f"--echeance {e} {PERIODE_NOV} --sortie {F_NOV[e]} --reprise {TN_NOV[e]} "
                                              f"--travailleurs 8")
        codes = ",".join(pd.read_csv(CODE / "configs" / "national" / "stations_metar.csv")["code"])
        if not (METAR_NOV / "metar_2026.csv").exists():
            lancer("metar_novembre", f"python -u scripts/historique_metar.py --debut 2026-10-31 --fin 2026-11-24 "
                                     f"--stations {codes} --sortie {METAR_NOV}")
        # 4. controle prealable (validation 2021), puis 5. jugement unique
        LOCF = preparer_forcages_et_configs()
        INIT = choisir_poids(RCP)
        INIT_C1 = verifier_c1() if sel["variante"]["famille"] in ("c1", "c3") else INIT
        tn_val = " ".join(f"--tn {d}" for e in ECH for d in [TN[e]] + LECTURE[e])
        if not (RES6 / "controle_entree.json").exists():
            try:
                lancer("controle_gfs_v6", f"python -u scripts/juger_national_gfs_v6.py controle --cp national_v4_eval "
                                          f"--gfs {LOCF} {tn_val} --obs {G / 'obs'} --sortie {RES6}")
            except RuntimeError as exc:
                if "(code 5)" not in str(exc):
                    raise
                print(exc)
        ctrl = json.loads((RES6 / "controle_entree.json").read_text())
        print(json.dumps({k: ctrl[k] for k in ("parties_vent", "rapporte_sans_arret")}, indent=1))
        if not ctrl["passe"]:
            print("ARRET (regle v6) : parties vent de l'entree de production differentes du cache ; novembre non lu")
        else:
            tn = " ".join(f"--tn {TN_NOV[e]}" for e in ECH)
            lancer("juger_gfs_v6", f"python -u scripts/juger_national_gfs_v6.py juger --cp national_v4_eval "
                                   f"--c1 @CONFIG_C1@ --selection {GV6 / 'selection.json'} {tn} --obs {METAR_NOV} "
                                   f"--sortie {RES6} --checkpoint-cp {shlex.quote(str(INIT))} "
                                   f"--checkpoint-c1 {shlex.quote(str(INIT_C1))}")
            print((RES6 / "gfs_v6.md").read_text())
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''


def carnet_v6(commit: str) -> dict:
    sha = V5.empreinte(REGLE)
    sha_v52, sha_v51 = V5.empreinte(V5.REGLE), V5.empreinte(V5.REGLE_V5_1)
    sha_v4, sha_g4 = V5.empreinte("regle_v4_national.json"), V5.empreinte("regle_national_gfs_v4.json")
    ancien = '(("regle_v4_national.json", "@SHA_V4@"),)'
    if ancien not in CODE_GIT:
        raise SystemExit("CODE_GIT de make_national_nb.py a change : adapter la verification des regles")
    git = (CODE_GIT.replace(ancien, f'(("regle_v4_national.json", "{sha_v4}"), ("regle_national_gfs_v4.json", '
                                    f'"{sha_g4}"), ("{V5.REGLE_V5_1}", "{sha_v51}"), ("{V5.REGLE}", "{sha_v52}"), '
                                    f'("{REGLE}", "{sha}"))')
           .replace('"| regle v4 verifiee"', '"| regles v4, gfs_v4, gfs_v5_1, gfs_v5_2 et gfs_v6 verifiees"')
           .replace("@COMMIT@", commit))
    garde = V5.GARDE_COMMIT.replace("@COMMIT@", commit)
    budget = (BUDGET.replace("@PLAFOND_REGLE@", str(PLAFOND_REGLE)).replace("@PLAFOND_FAMILLE@", str(PLAFOND_FAMILLE))
              .replace("@TAUX@", str(TAUX)).replace("@HMIN@", str(HMIN)).replace("@HMAX@", str(HMAX)))
    prep = (COMMUN + git + V5.DOSSIER.replace("@ACCORD@", ACCORD).replace("@SHA_REGLE@", sha_v52)
            .replace("@SHA_V5_1@", sha_v51)
            + DOSSIER_V6.replace("@SHA_V6@", sha).replace("@PERIODE@", PERIODE).replace("@ACCORD@", ACCORD))
    commun_b = (V5.COMMUN_B.replace("@CONFIG_C1@", V5.CONFIG_C1).replace("@EPOQUES@", str(V5.EPOQUES_C1))
                .replace("@EPOQUE_RETENUE@", str(V5.EPOQUE_C1_RETENUE)))
    tete = (f"Règle `configs/national/{REGLE}` (sha256 `{sha[:12]}…`), approuvée par Ines le 23/11/2026 et vérifiée par "
            "le carnet. Choix sur la validation 2021 seulement (résultats déjà calculés par la v5.2, marge : chaque "
            "aéroport ≤ +3 % et IC ≤ 0 dans les trois cases), puis jugement **unique** du 1er au 23 novembre 2026 "
            "(184 instants, toutes les 3 h, jamais utilisés), critère inchangé (IC ≤ 0, aucun aéroport > +5 %). "
            f"Code : dépôt public au commit `{commit[:12]}`. Sorties : `MyDrive/pcfno3d_national/gfs_v6_regle/` "
            "(dossier neuf). GFS et ERA5 ne sont jamais appelés mesures (seuls les METAR le sont). Aucun bruit ajouté. "
            "La session est libérée à la fin.")
    return carnet([
        md("# PC-FNO3D national — règle GFS v6 (Colab L4) : choix 2021 avec marge, jugement de novembre 2026\n\n" + tete),
        md("## 0. Commit figé, budget, GPU (L4 ou A100, au moins 22 Go)"), code(garde), code(budget), code(GARDE_GPU),
        md("## 1. Drive, code au commit figé, données ; refus si la règle v6 a déjà choisi ou jugé"),
        code(prep), code(REFUS_V6 + V5.EXTRAIRE_VAL),
        md("## 2. Choix 2021, GFS et METAR de novembre, contrôle, jugement unique ; fin de la session"),
        code(CHOIX_POIDS + commun_b + PARTIE_V6.replace("@CONFIG_C1@", V5.CONFIG_C1)),
    ], gpu=True, gpu_type="L4")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        raise SystemExit(f"--commit : 40 caracteres hexadecimaux attendus, recu {a.commit!r}")
    f = REPO / "notebooks" / "current" / NOM
    f.write_text(json.dumps(carnet_v6(a.commit), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
