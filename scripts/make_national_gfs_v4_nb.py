"""Carnets Colab de la regle configs/national/regle_national_gfs_v4.json (accord ecrit d'Ines du 16/11/2026,
« 1 start execution ») : le national C' s44 nourri par GFS et la couche 10 m GFS + aeroports, contre GFS brut,
a l'analyse, +24 h et +48 h, aux 12 aeroports, sur les instants synoptiques du test 2022 (seconde lecture).

    python scripts/make_national_gfs_v4_nb.py [--commit <sha>]
    -> notebooks/current/pcfno3d_national_gfs_v4_A_cpu.ipynb   (Colab CPU : GFS f0 / f24 / f48, boite nationale)
       notebooks/current/pcfno3d_national_gfs_v4_B_a100.ipynb  (Colab A100 : passes du modele, verdicts)

Les deux carnets prennent le code du depot public au commit fige et verifient l'empreinte de la regle avant
tout. Sorties dans un dossier NEUF : MyDrive/pcfno3d_national/gfs_v4_regle/ (rien d'ecrase ; les echeances
deja telechargees par l'OS7 du Grand Tunis sont relues sans y ecrire). Chaque carnet libere sa session a la fin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_grand_tunis_nb import carnet, code, md  # noqa: E402
from make_national_nb import CODE_GIT, COMMUN, GARDE_GPU  # noqa: E402

NAT_REGLES = REPO / "configs" / "national"
REGLE = "regle_national_gfs_v4.json"
NOMS = {"A": "pcfno3d_national_gfs_v4_A_cpu.ipynb", "B": "pcfno3d_national_gfs_v4_B_a100.ipynb"}
ACCORD = "Ines, 16/11/2026 : « 1 start execution » (regle approuvee telle quelle, puis calcul)"


def empreinte(nom: str) -> str:
    f = NAT_REGLES / nom
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    attendu = (NAT_REGLES / (nom + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"{nom} modifiee depuis son hachage ({h} != {attendu})")
    return h


DOSSIER = r'''
GV4 = NAT / "gfs_v4_regle"                 # dossier NEUF de la regle (rien d'autre n'y ecrit)
GV4.mkdir(exist_ok=True)
F = {e: GV4 / f"forcage_gfs_f{e}.csv" for e in (0, 24, 48)}
RES = GV4 / "resultat"
print("accord :", "@ACCORD@")
'''

PARTIE_A = r'''
# PARTIE A (Colab CPU) : GFS a l'analyse, +24 h et +48 h aux instants synoptiques du test 2022, boite nationale
!pip install -q herbie-data cfgrib eccodes 2>&1 | tail -1
from google.colab import runtime
import traceback
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
    for e in (0, 24, 48):
        if F[e].exists():
            print(f"f{e} deja ecrit : {F[e]}")
            continue
        lecture = f"--lecture {GT / 'os7' / f'gfs_f{e}_instants'}" if e else ""
        # code 4 = plus de 2 % d'echeances absentes de l'archive apres 3 passes (regle) : arret sans verdict
        lancer(f"gfs_v4_f{e}", f"python -u -m pcfno3d.gfs --config national_1km_colab --national --echeance {e} "
                               f"--splits test --synoptiques --sortie {F[e]} --reprise {GV4 / f'gfs_f{e}_tn'} "
                               f"{lecture} --travailleurs 8")
    for e in (0, 24, 48):
        err = F[e].with_suffix(".erreurs.json")
        print(f"f{e} :", F[e].exists(), "| echeances absentes :", len(json.loads(err.read_text())) if err.exists() else 0)
    print("partie A terminee : lancer le carnet B (A100)")
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''

PARTIE_B = r'''
# PARTIE B (Colab A100) : cache v3 des instants du test seulement, passes du modele, verdicts
from google.colab import runtime
import traceback, yaml, shlex
try:
    for e in (0, 24, 48):
        if not F[e].exists():
            raise FileNotFoundError(f"{F[e]} absent : lancer d'abord le carnet A (Colab CPU)")
    if (RES / "verdicts_gfs_v4.json").exists():
        raise SystemExit(f"{RES} a deja ses verdicts : une seule lecture")
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
    test = idx[(idx["split"] == "test") & idx["timestamp"].map(is_synoptic)]
    voulus = {f"{s}_{x}.npy" for s in test["stem"] for x in ("S", "E")}
    for k in (1, 2, 3):                           # seulement _S et _E des instants du test (pas les _Y)
        with zipfile.ZipFile(NAT / f"cache_v3_partie_{k}.zip") as zf:
            for m in zf.namelist():
                if Path(m).name in voulus and not (CACHE / m).exists():
                    zf.extract(m, CACHE)
    print(len(test), "instants synoptiques du test,", sum(1 for _ in CACHE.rglob("*_S.npy")), "fichiers _S extraits")
    RUN = NAT / "runs" / "national_1km_v4_Cp_s44"      # entrainement fixe par regle_choix_v4_Cp_s44
    c = yaml.safe_load(open(CODE / "configs" / "national_1km_colab.yaml"))
    c["name"] = "national_v4_eval"
    c["paths"]["run_dir"], c["paths"]["local_dir"] = str(RUN), "/content/run_local_v4_sCp_s44"
    yaml.safe_dump(c, open(CODE / "configs" / "national_v4_eval.yaml", "w"), sort_keys=False)
    lancer("juger_gfs_v4", f"python -u scripts/juger_national_gfs_v4.py -c national_v4_eval --gfs {GV4} "
                           f"--obs {G / 'obs'} --sortie {RES}")
    print((RES / "gfs_v4.md").read_text())
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''


def carnets(commit: str) -> dict[str, dict]:
    sha = empreinte(REGLE)
    sha_v4 = empreinte("regle_v4_national.json")
    ancien = '(("regle_v4_national.json", "@SHA_V4@"),)'
    if ancien not in CODE_GIT:
        raise SystemExit("CODE_GIT de make_national_nb.py a change : adapter la verification des regles")
    git = (CODE_GIT.replace(ancien, f'(("regle_v4_national.json", "{sha_v4}"), ("{REGLE}", "{sha}"))')
           .replace('"| regle v4 verifiee"', '"| regles v4 et gfs_v4 verifiees"').replace("@COMMIT@", commit))
    prep = COMMUN + git + DOSSIER.replace("@ACCORD@", ACCORD)
    tete = (f"Règle `configs/national/{REGLE}` (sha256 `{sha[:12]}…`), approuvée par Ines le 16/11/2026 et vérifiée "
            f"par le carnet. Code : dépôt public au commit `{commit[:12]}`. Sorties : "
            "`MyDrive/pcfno3d_national/gfs_v4_regle/` (dossier neuf). **Seconde lecture** de 2022 (mêmes METAR que le "
            "DSO7). GFS et ERA5 ne sont jamais appelés mesures. Aucun réentraînement, aucun bruit ajouté. La session "
            "est libérée à la fin.")
    a = carnet([
        md("# PC-FNO3D national — règle GFS v4, partie A (Colab CPU) : GFS f0 / f24 / f48\n\n" + tete),
        md("## 1. Drive, code au commit figé, données"), code(prep),
        md("## 2. GFS (archive NOAA sur AWS), boîte nationale, instants synoptiques du test ; fin de la session"),
        code(PARTIE_A),
    ], gpu=False)
    b = carnet([
        md("# PC-FNO3D national — règle GFS v4, partie B (Colab A100) : passes du modèle et verdicts\n\n" + tete),
        md("## 0. GPU (A100 obligatoire)"), code(GARDE_GPU),
        md("## 1. Drive, code au commit figé, données"), code(prep),
        md("## 2. Aéroport retiré, trois échéances, critère de la règle ; fin de la session"), code(PARTIE_B),
    ], gpu=True)
    return {"A": a, "B": b}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", help="commit du code (defaut : HEAD)")
    args = ap.parse_args()
    commit = args.commit or subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                           text=True, check=True).stdout.strip()
    for k, nb in carnets(commit).items():
        f = REPO / "notebooks" / "current" / NOMS[k]
        f.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
