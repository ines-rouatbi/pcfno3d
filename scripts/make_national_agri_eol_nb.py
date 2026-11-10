"""Carnet Colab des phases 6 (agriculture nationale) et 7 (eolien national) de la feuille de route
(docs/feuille_de_route_nationale.md), sur le modele national LIVRE, sans entrainement.

    python scripts/make_national_agri_eol_nb.py [--commit <sha>]
    -> notebooks/current/pcfno3d_national_agri_eolien_colab.ipynb

Le carnet fige le code au commit donne (par defaut HEAD : il doit etre pousse sur le depot public
avant l'execution) et verifie l'empreinte des deux regles PROPOSITION avant tout calcul :
configs/national/regle_agri_national.json et configs/national/regle_eolien_national.json.
Il ne s'execute qu'avec l'accord ecrit d'Ines. scripts/make_national_nb.py n'est pas modifie : ses
cellules communes (Drive, donnees, cache v3) sont reprises telles quelles.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_grand_tunis_nb import CHOIX_POIDS, carnet, code, md  # noqa: E402
from make_national_nb import COMMUN, _ent_v3  # noqa: E402

REGLES = ("regle_agri_national.json", "regle_eolien_national.json")
NOM = "pcfno3d_national_agri_eolien_colab.ipynb"


def empreintes() -> dict[str, str]:
    return {f: (REPO / "configs" / "national" / (f + ".sha256")).read_text(encoding="ascii").split()[0]
            for f in REGLES}


CODE_GIT = '''
# code du depot public au commit FIGE (les scripts des phases 6 et 7 y sont) ; regles verifiees
COMMIT = "@COMMIT@"
os.chdir("/content")    # le noyau etait dans /content/PFE : le supprimer laisserait pip dans un dossier efface
subprocess.run(f"rm -rf /content/PFE && git init -q /content/PFE && cd /content/PFE && git remote add origin "
               f"https://github.com/ines-rouatbi/pcfno3d.git && git fetch -q --depth 1 origin {COMMIT} "
               f"&& git checkout -q FETCH_HEAD", shell=True, check=True)
os.chdir("/content/PFE")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "/content/PFE"],
                   capture_output=True, text=True)
if r.returncode:
    print(r.stdout[-3000:], r.stderr[-3000:])
    raise SystemExit("pip install -e /content/PFE a echoue (sortie ci-dessus)")
import importlib
importlib.invalidate_caches()
import hashlib
# regles PROPOSITION ecrites et hachees AVANT tout calcul : une regle modifiee arrete le carnet
for f, h in @EMPREINTES@.items():
    lu = hashlib.sha256((CODE / "configs" / "national" / f).read_bytes()).hexdigest()
    assert lu == h, f"{f} : empreinte {lu} != {h} (regle modifiee depuis son hachage)"
print("code", COMMIT, "| regles verifiees :", ", ".join(f"{f} {h[:12]}" for f, h in @EMPREINTES@.items()))
'''

CHOIX_MODELE = r'''
# ----- modele national : celui que designe la decision v4 (runs/decision_v4_national.json), sinon la v2 -----
MODELE = None          # None = la decision ; ou un nom de run impose, ex. "national_1km_v4_Cp_s43" (ecrit dans les sorties)
RUNS = NAT / "runs"
DEC = RUNS / "decision_v4_national.json"
if MODELE is None:
    MODELE = json.loads(DEC.read_text())["modele_livre_national"] if DEC.exists() else "national_1km_v2"
    print("decision v4 nationale :", DEC if DEC.exists() else "absente -> modele livre v2")
if (RUNS / "choix_v4.json").exists():
    print("choix_v4.json (entrainement choisi sur la validation) :",
          json.loads((RUNS / "choix_v4.json").read_text())["entrainement_choisi"])
RUN = RUNS / MODELE
# la v2 jugee sur la verite v3 (memes poids, dossier neuf du carnet de test v4) porte les paires du test v3
if MODELE == "national_1km_v2" and (RUNS / "national_1km_v2_sur_verite_v3" / "best.pt").exists():
    RUN = RUNS / "national_1km_v2_sur_verite_v3"
CKPT = choisir_poids(RUN)
SORTIE = NAT / "phase6_7" / RUN.name           # dossier neuf, a part du run_dir ; rien n'est ecrase
SORTIE.mkdir(parents=True, exist_ok=True)
OBS = G / "obs"                                 # METAR des 12 aeroports, radiosondages IGRA
import yaml
c = yaml.safe_load(open(CODE / "configs" / "national_1km_colab.yaml"))
c["name"] = "national_phase67"
c["paths"]["run_dir"], c["paths"]["local_dir"] = str(RUN), "/content/run_local_phase67"
c["paths"]["stations_obs"] = str(OBS) if OBS.exists() else None     # entree de production : vrais METAR
c.setdefault("inference", {})["w_continuite"] = True
yaml.safe_dump(c, open(CODE / "configs" / "national_phase67.yaml", "w"), sort_keys=False)
print("modele :", MODELE, "| run :", RUN, "| poids :", CKPT.name, "| sorties :", SORTIE)
print("mesures :", sorted(p.name for p in OBS.iterdir()) if OBS.exists() else "AUCUN dossier obs")
'''

PHASE6 = r'''
import shlex
from google.colab import runtime
import traceback
ARRETER = True          # liberer la machine a la fin de la phase 7 (cellule suivante)
try:
    # AGRI1 / AGRI2 : paires « station retiree » du test 2022 (ecrites UNE fois par le carnet de test v4) ;
    # si elles manquent pour ce modele, le meme script les ecrit une fois
    PAIRES = RUN / "validation_reelle" / "test" / "stations_paires.csv"
    if not PAIRES.exists():
        lancer("validation_reelle_phase6", f"python -u scripts/valider_grand_tunis_reel.py -c national_phase67 "
                                           f"--checkpoint {shlex.quote(str(CKPT))} --obs {OBS} --jeu test")
    D = SORTIE / "agriculture" / "agri1_agri2"
    if not (D / "verdict_agri_national.json").exists():
        lancer("agri1_agri2", f"python -u scripts/agri_national.py decisions --run {shlex.quote(str(RUN))} --sortie {D}")
    print((D / "verdict_agri_national.md").read_text())

    # fenetres J0-J+2 et FAO-56 sur la grille nationale (jours de base de la regle : validation 2021)
    GFS = next((f for f in (NAT / "gfs_tunisie_f24.npz", GT / "os7" / "gfs_tunisie_f24.npz") if f.exists()), None)
    print("cube GFS national :", GFS or "absent -> repli signale (T, HR : verite d'entrainement ; Rs = 0,72 Ra)")
    P = SORTIE / "agriculture" / "produit_j0_j2"
    if not (P / "produit_agricole.json").exists():
        lancer("produit_agricole", f"python -u scripts/agri_national.py produit -c national_phase67 "
                                   f"--checkpoint {shlex.quote(str(CKPT))} --sortie {P}"
                                   + (f" --gfs {GFS}" if GFS else ""))
    print((P / "produit_agricole.json").read_text()[:4000])
except Exception:
    traceback.print_exc()
'''

PHASE7 = r'''
try:
    # carte du potentiel 50-150 m (principale : validation + test ; information : tous les instants)
    A = SORTIE / "eolien" / "atlas_national"
    if not (A / "atlas_national.json").exists():
        TERR = NAT / "territoire_tunisie.json"     # polygone [[lat, lon], ...] facultatif, depose par Ines
        lancer("atlas_national", f"python -u scripts/atlas_eolien_national.py carte -c national_phase67 "
                                 f"--checkpoint {shlex.quote(str(CKPT))} --sortie {A}"
                                 + (f" --territoire {TERR}" if TERR.exists() else ""))
    print(pd.read_csv(A / "resume.csv").to_string(index=False))
    print(pd.read_csv(A / "zones.csv").to_string(index=False))

    # zoom Grand Tunis a 100 m : l'atlas du modele livre v4 (carnet OS6), aucune nouvelle passe
    Z = SORTIE / "eolien" / "zoom_grand_tunis_100m"
    if not (Z / "zoom.json").exists():
        lancer("zoom_grand_tunis", f"python -u scripts/atlas_eolien_national.py zoom "
                                   f"--grand-tunis {GT / 'runs' / 'atlas_eolien'} --national {A} "
                                   f"--decision-gt {GT / 'runs' / 'decision_v4.json'} --sortie {Z}")
    print((Z / "zoom.json").read_text()[:3000])

    # EOL2 : radiosondages 0-250 m (Tunis-Carthage, Tozeur s'ils sont dans obs) contre ERA5, test 2022
    E = SORTIE / "eolien" / "eol2"
    if not (E / "eol2.json").exists():
        lancer("eol2_national", f"python -u scripts/eol2_national.py -c national_phase67 "
                                f"--checkpoint {shlex.quote(str(CKPT))} --obs {OBS} --sortie {E}")
    print((E / "eol2.md").read_text())
except Exception:
    traceback.print_exc()
finally:
    if ARRETER:
        runtime.unassign()
'''


def construire(commit: str) -> dict:
    h = empreintes()
    git = CODE_GIT.replace("@COMMIT@", commit).replace("@EMPREINTES@", json.dumps(h))
    tete = f"""
# PC-FNO3D NATIONAL — phases 6 (agriculture) et 7 (éolien) : produits et critères, sans entraînement (Colab, A100)

**Exécution uniquement avec l'accord écrit d'Ines.** Aucun entraînement, aucun téléchargement, aucun
réglage : le carnet applique deux règles **PROPOSITION**, écrites et hachées avant tout calcul et vérifiées
par la première cellule :

| Règle | sha256 |
|---|---|
| `configs/national/regle_agri_national.json` (AGRI1, AGRI2, J0–J+2, FAO-56) | `{h[REGLES[0]][:12]}…` |
| `configs/national/regle_eolien_national.json` (carte 50–150 m, zoom, EOL2) | `{h[REGLES[1]][:12]}…` |

Code figé au commit `{commit[:12]}` du dépôt public (à pousser avant l'exécution).

**Modèle** : le modèle national livré, désigné par `runs/decision_v4_national.json` (entraînement choisi :
`runs/choix_v4.json`), sinon la v2 ; on peut l'imposer avec `MODELE` (cellule 3). Sorties dans un dossier
neuf, `MyDrive/pcfno3d_national/phase6_7/<run>/` ; chaque script refuse de recalculer une sortie existante.

- **Phase 6** : AGRI1 / AGRI2 aux 12 aéroports (test 2022, chaque station retirée de l'entrée quand elle est
  jugée : post-traitement des paires du carnet de test v4, aucune nouvelle passe) ; fenêtres de traitement et
  d'aspersion J0–J+2 et besoin en eau FAO-56 sur la grille de 1 km, vent à 2 et 10 m au-dessus du sol
  (rejeu sur analyse ERA5 tant que la phase 4 n'a pas fourni le forçage GFS national).
- **Phase 7** : carte du potentiel de 50 à 150 m (vitesse, densité de puissance, Weibull, production d'une
  éolienne générique de 2 MW ; GeoTIFF, PNG, CSV) sur la validation 2021 et le test 2022 (instants jamais vus
  à l'entraînement ; hiver et été seulement, c'est dit), carte d'information sur les 1 000 instants, ERA5
  seule à côté ; zoom Grand Tunis à 100 m depuis l'atlas du modèle livré v4 ; EOL2 contre les radiosondages
  0–250 m (Tunis-Carthage, Tozeur).

ERA5 et la vérité générée sont la **vérité d'entraînement**, jamais des mesures ; les METAR et les
radiosondages sont les mesures. La session s'arrête à la fin.
"""
    return carnet([
        md(tete),
        md("## 1. Drive, code figé, règles vérifiées, données"), code(COMMUN + git),
        md("## 2. Cache v3 (3 parties du Drive)"), code(_ent_v3()),
        md("## 3. Modèle national retenu et configuration de production"), code(CHOIX_POIDS + CHOIX_MODELE),
        md("## 4. Phase 6 : AGRI1 / AGRI2, fenêtres J0–J+2, FAO-56"), code(PHASE6),
        md("## 5. Phase 7 : carte 50–150 m, zoom Grand Tunis 100 m, EOL2 — puis fin de la session"), code(PHASE7),
    ], gpu=True)


def main() -> int:
    commit = None
    if "--commit" in sys.argv:
        commit = sys.argv[sys.argv.index("--commit") + 1]
    if commit is None:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                                check=True).stdout.strip()
    f = REPO / "notebooks" / "current" / NOM
    f.write_text(json.dumps(construire(commit), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
