"""Carnet Colab de l'OS6 NATIONAL (phase 5 de docs/feuille_de_route_nationale.md) : reseau ANEO sur toute
la Tunisie avec le candidat D, regle configs/national/regle_os6_national.json (PROPOSITION).

    python scripts/make_national_os6_nb.py [--commit <sha>]
    -> notebooks/current/pcfno3d_national_os6_colab.ipynb   (A100, sans entrainement)

Le code vient du depot public au commit fige (par defaut HEAD : le commit doit etre pousse avant de
lancer le carnet) ; l'empreinte de la regle est inscrite dans le carnet et verifiee avant tout calcul.
Reprend les conventions de scripts/make_national_nb.py (COMMUN, CODE_GIT, cache v3, lancer,
choisir_poids), sans le modifier.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_grand_tunis_nb import CHOIX_POIDS, carnet, code, md  # noqa: E402
from make_national_nb import CODE_GIT, COMMUN, _ent_v3  # noqa: E402

REGLE = REPO / "configs" / "national" / "regle_os6_national.json"
NOM = "pcfno3d_national_os6_colab.ipynb"

VERIF_OS6 = '''
# regle OS6 nationale (PROPOSITION validee par Ines) : empreinte inscrite dans ce carnet, verifiee ici
from pcfno3d import os6
REGLE = CODE / "configs" / "national" / "regle_os6_national.json"
regle, sha_os6 = os6.charger_regle(REGLE)
assert sha_os6 == "@SHA_OS6@", f"regle OS6 modifiee depuis la generation du carnet : {sha_os6}"
print("regle OS6 verifiee :", sha_os6[:12], "|", regle["statut"][:120])
'''

GPU = r'''
import torch
assert torch.cuda.is_available(), "pas de GPU : choisir A100 (ou L4) dans Execution > Modifier le type d'execution"
nom_gpu = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1e9
# estimation (non mesuree, regle_os6_national.json) : ~5-6 Go par disposition a 16 x 828 x 388 ;
# un lot trop grand est de toute facon recoupe en deux (WindFieldModel._members)
LOT = 4 if total >= 39 else 2
print(f"{nom_gpu}, {total:.0f} Go | {LOT} dispositions par passage du reseau")
'''

OS6 = r'''
import shlex
import traceback
import yaml
from google.colab import runtime
P = regle["parametres"]
try:
    # 1. graine de D : minimum de val_score de la VALIDATION (history.json) ; le test n'est pas lu
    choix = os6.meilleure_graine(NAT / "runs", "national_1km_v4_D", (42, 43, 44))
    choix["regle_os6_sha256"] = sha_os6
    print(json.dumps(choix, indent=1))
    RUN = Path(choix["run_dir"])
    SORTIE = NAT / "runs" / f"os6_national_D_s{choix['graine']}"      # dossier neuf, hors du run_dir de D
    SORTIE.mkdir(parents=True, exist_ok=True)
    F_CHOIX = SORTIE / "choix_modele.json"
    if F_CHOIX.exists() and json.loads(F_CHOIX.read_text())["graine"] != choix["graine"]:
        raise RuntimeError(f"{F_CHOIX} porte une autre graine : ne rien melanger, voir avec Ines")
    F_CHOIX.write_text(json.dumps(choix, indent=2))
    # 2. configuration : celle du national, run_dir = celui de D (lu seulement : poids et norm_stats.json)
    c = yaml.safe_load(open(CODE / "configs" / "national_1km_colab.yaml"))
    c["name"] = "national_os6_D"
    c["paths"]["run_dir"], c["paths"]["local_dir"] = str(RUN), "/content/run_local_os6_D"
    yaml.safe_dump(c, open(CODE / "configs" / "national_os6_D.yaml", "w"), sort_keys=False)
    CKPT = shlex.quote(str(choisir_poids(RUN)))
    # 3. courbe d'information (ERA5 degradee, parametres de la phase 3, jours d'entrainement)
    DEG = NAT / "runs" / "degradation_era5_v4.json"
    if P["courbe_information_era5_degradee"] and not DEG.exists():
        raise FileNotFoundError(f"{DEG} absent : la regle demande la courbe d'information")
    ZONES = NAT / "os6_zones_exclues.geojson"          # fourni par Tukhnanutha, facultatif ; aucune zone inventee
    print("zones exclues :", ZONES if ZONES.exists() else "aucune")
    # 4. placement (validation), courbe (test, une fois ; une reprise relit les fichiers), plan
    lancer("os6_national",
           f"python -u scripts/os6_reseau_aneo.py -c national_os6_D --checkpoint {CKPT} "
           f"--regle {shlex.quote(str(REGLE))} --modele-info {shlex.quote(str(F_CHOIX))} "
           f"--sortie {shlex.quote(str(SORTIE))} --lot {LOT}"
           + (f" --degradation {shlex.quote(str(DEG))}" if P["courbe_information_era5_degradee"] else "")
           + (f" --zones-exclues {shlex.quote(str(ZONES))}" if ZONES.exists() else ""))
    r = json.loads((SORTIE / "os6.json").read_text())
    print(json.dumps({k: r.get(k) for k in ("nombre_recommande", "metrique_seuil", "candidats",
                                             "capteurs_places", "reseau_actuel", "comparaison_net1_sol")},
                     indent=1, ensure_ascii=False))
    import pandas as pd
    print(pd.read_csv(SORTIE / "courbe_test.csv").round(3).to_string(index=False))
    if (SORTIE / "courbe_test_era5_degradee.csv").exists():
        print("information (ERA5 degradee, sans effet sur le nombre recommande) :")
        print(pd.read_csv(SORTIE / "courbe_test_era5_degradee.csv").round(3).to_string(index=False))
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''


def os6_national(commit: str) -> dict:
    """Le carnet : code au commit fige, regle verifiee, cache v3, GPU, puis OS6 national et fin de session."""
    sha_v4 = (REPO / "configs" / "national" / "regle_v4_national.json.sha256").read_text(encoding="ascii").split()[0]
    sha = (REPO / "configs" / "national" / "regle_os6_national.json.sha256").read_text(encoding="ascii").split()[0]
    regle = json.loads(REGLE.read_text(encoding="utf-8"))
    P = regle["parametres"]
    git = CODE_GIT.replace("@COMMIT@", commit).replace("@SHA_V4@", sha_v4)
    titre = f"""
# PC-FNO3D NATIONAL — OS6 : réseau ANEO sur toute la Tunisie, candidat D (Colab, A100)

**À lancer uniquement avec l'accord écrit d'Ines.** Règle `configs/national/regle_os6_national.json`
(sha256 `{sha[:12]}…`, statut : PROPOSITION tant qu'Ines ne l'a pas validée), vérifiée par le carnet
avant tout calcul ; code du dépôt au commit figé `{commit[:12]}`. **Aucun entraînement.**

Expériences de simulation de systèmes d'observation (OSSE) sur la vérité d'entraînement v3 :

1. **modèle** : le candidat D (`runs/national_1km_v4_D_s<graine>`), graine choisie sur la **validation**
   (minimum de `val_score` de `history.json`), écrite dans `choix_modele.json` ; le run_dir de D n'est que lu ;
2. **capteur ANEO simulé** comme à l'entraînement de D : vérité d'entraînement à
   {P["hauteur_capteur_m"]:g} m au-dessus du sol + bruit de mesure 0,1 m/s + 2 % de la vitesse, entré comme
   une station (ce ne sont pas des mesures ; aucun bruit ajouté aux prédictions) ;
3. **candidats** sur toute la Tunisie : terre, pente ≤ 20°, altitude ≤ 650 m, à 1 km du bord, des
   aéroports et entre capteurs ; zones exclues seulement si `os6_zones_exclues.geojson` est déposé ;
4. **placement glouton** sur la validation ({P["n_val"]} instants, {P["par_etape"]} capteurs par étape,
   jusqu'à {P["n_max"]}) ;
5. **courbe de dimensionnement** sur le **test 2022, lu une fois** : optimisé, grille régulière, hasard, pour
   {", ".join(str(n) for n in P["tailles"])} capteurs ; nombre recommandé = plus petit nombre dont le RMSE u, v
   maximal sur 0–1 km au-dessus du sol est ≤ {P["seuil_m_s"]:g} m/s ; courbe ERA5 dégradée en information ;
6. **plan** : `MyDrive/pcfno3d_national/runs/os6_national_D_s<graine>/` (`plan_aneo.csv`, `.geojson`,
   `courbe_test.csv`, `carte.png`, `os6.json`…), au format du Grand Tunis (onglet admin de la phase 8).

Ressources estimées (non mesurées) : A100 40 Go, ~20–24 Go de GPU par lot de 4 dispositions, ~3 h 30
(2 à 3 fois plus sur L4). Reprise : une étape déjà écrite est relue. La session s'arrête à la fin.
"""
    return carnet([
        md(titre),
        md("## 1. Drive, code au commit figé, règles vérifiées, données"),
        code(COMMUN + git + VERIF_OS6.replace("@SHA_OS6@", sha)),
        md("## 2. Cache de la vérité v3"),
        code(_ent_v3()),
        md("## 3. GPU"),
        code(GPU),
        md("## 4. Graine de D (validation), placement, courbe (test, une fois), plan — puis fin de la session"),
        code(CHOIX_POIDS + OS6),
    ], gpu=True, gpu_type="A100")


def main() -> int:
    if "--commit" in sys.argv:
        commit = sys.argv[sys.argv.index("--commit") + 1]
    else:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                                check=True).stdout.strip()
    f = REPO / "notebooks" / "current" / NOM
    f.write_text(json.dumps(os6_national(commit), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
