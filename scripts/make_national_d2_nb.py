"""Carnets Colab du candidat D2 national (configs/national/regle_d2_national.json) : canal d'innovation.

    python scripts/make_national_d2_nb.py
ecrit, au commit courant (HEAD, a pousser avant de lancer) :
    notebooks/current/pcfno3d_national_v4_D2_s42_colab.ipynb   statistiques (jours d'entrainement), choix de H
                                                               (validation), entrainement graine 42, information
    notebooks/current/pcfno3d_national_v4_D2_s43_colab.ipynb   attend les deux JSON de la graine 42, entraine 43
    notebooks/current/pcfno3d_national_v4_D2_s44_colab.ipynb   idem, graine 44
    notebooks/current/pcfno3d_national_v4_D2_test_colab.ipynb  choix sur la validation, lecture UNIQUE du test,
                                                               references, mesures reelles, decision
Aucune lecture du test dans les carnets d'entrainement. Le carnet de test demande ACCORD_TEST_INES = True.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts")]
import make_national_nb as NN  # noqa: E402
from make_grand_tunis_nb import CHOIX_POIDS, carnet, code, md  # noqa: E402

GRAINES = (42, 43, 44)

VERIF_D2 = '''
assert hashlib.sha256((CODE / "configs" / "national" / "regle_d2_national.json").read_bytes()).hexdigest() \\
    == "@SHA_D2@", "regle D2 modifiee"
print("regle D2 verifiee")
'''

TRAIN_D2 = r'''
import shlex
from google.colab import runtime
import traceback
GRAINE = @GRAINE@
RUN = NAT / "runs" / f"national_1km_v4_D2_s{GRAINE}"
RUN.mkdir(parents=True, exist_ok=True)
DEGV4 = NAT / "runs" / "degradation_era5_v4.json"          # celle de D (regle v4), jours d'entrainement
STATS = NAT / "runs" / "statistiques_innovation_d2.json"   # L, sigma_o / sigma_b (jours d'entrainement)
REG = NAT / "runs" / "reglage_H_d2.json"                   # H (validation, OI seule, sans reseau)

def attendre(f, heures=4):
    """Graines 43 et 44 : les JSON communs sont ecrits par le carnet de la graine 42 (jamais deux fois)."""
    t0 = time.time()
    while not f.exists():
        if time.time() - t0 > heures * 3600:
            raise TimeoutError(f"{f} absent apres {heures} h : lancer d'abord le carnet de la graine 42")
        print(f"attente de {f.name} (carnet de la graine 42)...", flush=True)
        time.sleep(120)

try:
    if GRAINE == 42:
        if not DEGV4.exists():
            lancer("degradation_era5_v4", f"python -u scripts/degradation_era5.py --donnees {G} "
                                          f"--calendrier {NAT / 'calendrier.json'} --sortie {DEGV4}")
        if not STATS.exists():
            lancer("statistiques_innovation_d2", f"python -u scripts/statistiques_innovation_d2.py --donnees {G} "
                                                 f"--calendrier {NAT / 'calendrier.json'} --sortie {STATS}")
    attendre(DEGV4)
    attendre(STATS)
    st = json.loads(STATS.read_text(encoding="utf-8"))
    print({k: st[k] for k in ("L_km", "rapport_o_b", "sigma_b", "sigma_o", "L_en_butee", "distance_min_km", "paires")})
    INN = (f"--set innovation.actif=true --set innovation.L_km={st['L_km']:.3f} "
           f"--set innovation.rapport_o_b={st['rapport_o_b']:.6f} --set mask.aneo_hauteur_capteur_m=10")
    if GRAINE == 42 and not REG.exists():
        lancer("reglage_H_d2", f"python -u scripts/info_capteurs_d2.py reglage -c national_1km_colab {OPTS} {INN} "
                               f"--degradation {shlex.quote(str(DEGV4))} --sortie {shlex.quote(str(REG))} "
                               "--H 200,500,1000 --condition propre")
    attendre(REG)
    H = float(json.loads(REG.read_text(encoding="utf-8"))["H_retenu_m"])
    INN += f" --set innovation.H_m={H:g}"
    print("H retenu (validation) :", H, "m | innovation :", INN)
    DEG = RUN / "degradation_era5.json"
    if not DEG.exists():
        shutil.copy2(DEGV4, DEG)
    INNJ = RUN / "innovation_d2.json"
    if not INNJ.exists():
        INNJ.write_text(json.dumps({"regle_sha256": "@SHA_D2@", "L_km": st["L_km"], "rapport_o_b": st["rapport_o_b"],
                                    "H_m": H, "L_en_butee": st["L_en_butee"], "statistiques": str(STATS),
                                    "reglage_H": str(REG)}, indent=2), encoding="utf-8")
    EXTRA = ("--set loss.lambda_rough=1.0 "
             f"--set mask.aneo_degradation={shlex.quote(str(DEG))} --set mask.aneo_p_zone=0.5 "
             f"--set {shlex.quote('mask.aneo_capteurs=[20,200]')} {INN}")
    INIT = shlex.quote(str(choisir_poids(NAT / "runs" / "national_1km_v2")))
    print("depart :", INIT, "| run :", RUN)
    SEL = shlex.quote('train.select_criteres=["dso1_rmse","dso2_masse","dso3_rugosite"]')
    lancer(f"entrainer_v4_D2_s{GRAINE}",
           f"python -u -m pcfno3d -c national_1km_colab {OPTS} {EXTRA} --set {SEL} --set train.sync_last=false "
           f"--set train.init_from={INIT} --set train.lr=1e-4 --set train.seed={GRAINE} "
           f"--set paths.run_dir={shlex.quote(str(RUN))} --set paths.local_dir=/content/run_local_v4_D2_s{GRAINE} train")
    # AUCUNE lecture du test ici (regle_d2_national.json)
    print((RUN / "history.json").read_text()[-3000:])
    # information, sans effet sur le choix : D2, ERA5 seule et OI seule ; propre / degradee / + 50 capteurs
    if not (RUN / "info_capteurs_d2_validation.json").exists():
        lancer(f"info_capteurs_d2_s{GRAINE}",
               f"python -u scripts/info_capteurs_d2.py info -c national_1km_colab {OPTS} {INN} "
               f"--run {shlex.quote(str(RUN))} --degradation {shlex.quote(str(DEGV4))}")
    print((RUN / "info_capteurs_d2_validation.json").read_text()[:4000])
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''

TEST_D2 = r'''
ACCORD_TEST_INES = False     # a passer a True a la main, seulement avec l'accord ecrit d'Ines pour lire le test
if not ACCORD_TEST_INES:
    raise SystemExit("lecture du test D2 : accord ecrit d'Ines requis (ACCORD_TEST_INES = True)")
import shlex
import statistics
from google.colab import runtime
import traceback
import yaml
regle = json.loads((CODE / "configs" / "national" / "regle_d2_national.json").read_text(encoding="utf-8"))
scores = {}
for g in regle["candidat_D2"]["graines"]:
    h = NAT / "runs" / f"national_1km_v4_D2_s{g}" / "history.json"
    if not h.exists():
        raise FileNotFoundError(f"D2 graine {g} pas entraine ({h})")
    scores[f"D2_s{g}"] = min(float(e["val_score"]) for e in json.loads(h.read_text()) if "val_score" in e)
print("scores de selection (validation) :", scores)
CHOIX = min(scores, key=scores.get)
RUN = NAT / "runs" / f"national_1km_v4_{CHOIX}"
choix = {"regle_sha256": "@SHA_D2@", "scores_validation": scores, "entrainement_choisi": CHOIX,
         "moyenne": statistics.mean(scores.values()), "ecart_type": statistics.pstdev(scores.values())}
(NAT / "runs" / "choix_d2.json").write_text(json.dumps(choix, indent=2))
inn = json.loads((RUN / "innovation_d2.json").read_text(encoding="utf-8"))

def config_eval(nom, run, local, innovation=None):
    c = yaml.safe_load(open(CODE / "configs" / "national_1km_colab.yaml"))
    c["name"] = nom
    c["paths"]["run_dir"], c["paths"]["local_dir"] = str(run), local
    if innovation:
        c["innovation"] = innovation
        c["mask"]["aneo_hauteur_capteur_m"] = 10.0
    yaml.safe_dump(c, open(CODE / "configs" / f"{nom}.yaml", "w"), sort_keys=False)

config_eval("national_d2_eval", RUN, f"/content/run_local_{CHOIX}",
            {"actif": True, "L_km": inn["L_km"], "rapport_o_b": inn["rapport_o_b"], "H_m": inn["H_m"]})
O = OPTS

def etapes(conf, run, suffixe, dso6=True):
    """evaluate, DSO6 (candidat seulement, comme la regle v4), validation reelle, verdict-national, DSO 0-1 km
    (+ --keep 0.25) ; chaque etape une fois (fichier deja present : sautee)."""
    ev = run / "eval_pcfno3d" / "best_wcont"
    if not (ev / "test_summary.json").exists():                                 # test lu une fois
        lancer(f"evaluer_{suffixe}", f"python -u -m pcfno3d -c {conf} {O} evaluate")
    CAL = run / "fiabilite_pcfno3d" / "best_wcont" / "calibration_validation.json"
    D6 = ev / "dso6_modele_livre.json"
    if dso6 and not CAL.exists():
        lancer(f"calibrer_dso6_{suffixe}", f"python -u -m pcfno3d -c {conf} {O} calibrate-reliability --out {CAL}")
    if dso6 and not D6.exists():
        try:
            lancer(f"dso6_{suffixe}", f"python -u -m pcfno3d -c {conf} {O} dso6 --calibration {CAL} --out {D6}")
        except RuntimeError as exc:
            if "(code 3)" not in str(exc):
                raise
            print(exc)
    if not (run / "validation_reelle" / "test" / "bilan.json").exists():
        lancer(f"validation_reelle_{suffixe}", f"python -u scripts/valider_grand_tunis_reel.py -c {conf} "
                                               f"--obs {G / 'obs'} --jeu test")
    sol = run / "dso_hauteur_sol"
    for fichier, cmd, nom in (
            (ev / "verdict_national.json", f"python -u -m pcfno3d -c {conf} {O} verdict-national", f"verdict_national_{suffixe}"),
            (sol / "regle.json", f"python -u scripts/dso_hauteur_sol.py -c {conf} regle", f"dso_sol_regle_{suffixe}"),
            (sol / "bilan.json", f"python -u scripts/dso_hauteur_sol.py -c {conf} mesure", f"dso_sol_{suffixe}"),
            (sol / "bilan_keep25.json", f"python -u scripts/dso_hauteur_sol.py -c {conf} mesure --keep 0.25",
             f"dso_sol_dso5_{suffixe}")):
        if fichier.exists():
            print("deja fait :", fichier)
            continue
        try:
            lancer(nom, cmd)
        except RuntimeError as exc:
            if "(code 3)" not in str(exc):
                raise
            print(exc)

try:
    etapes("national_d2_eval", RUN, "d2")
    # modele livre (v2, poids inchanges) sur la MEME verite v3 : dossier de la regle v4, reutilise ou cree
    REF = NAT / "runs" / "national_1km_v2_sur_verite_v3"
    REF.mkdir(parents=True, exist_ok=True)
    V2 = NAT / "runs" / "national_1km_v2"
    for src, dst in ((choisir_poids(V2), REF / "best.pt"), (V2 / "norm_stats.json", REF / "norm_stats.json")):
        if not dst.exists():
            shutil.copy2(src, dst)
    config_eval("national_v2_sur_v3_eval", REF, "/content/run_local_v2_sur_v3")
    etapes("national_v2_sur_v3_eval", REF, "v2_sur_v3", dso6=False)
    try:
        lancer("decision_d2_national", f"python -u scripts/decision_d2_national.py --d2 {RUN} --livre {REF} "
                                       f"--sortie {NAT / 'runs'}")
    except RuntimeError as exc:
        if "(code 3)" not in str(exc):
            raise
        print(exc)
    for f in sorted(RUN.rglob("bilan*.json")) + sorted(RUN.rglob("verdict_national.json")) \
            + [NAT / "runs" / "decision_d2_national.json"]:
        print(f); print(f.read_text()[:3000])
except Exception:
    traceback.print_exc()
finally:
    runtime.unassign()
'''


def carnets(commit: str) -> dict[str, dict]:
    sha_v4 = (REPO / "configs" / "national" / "regle_v4_national.json.sha256").read_text(encoding="ascii").split()[0]
    sha = (REPO / "configs" / "national" / "regle_d2_national.json.sha256").read_text(encoding="ascii").split()[0]
    git = NN.CODE_GIT.replace("@COMMIT@", commit).replace("@SHA_V4@", sha_v4) + VERIF_D2.replace("@SHA_D2@", sha)
    tete = (f"Règle `configs/national/regle_d2_national.json` (sha256 `{sha[:12]}…`), vérifiée par le carnet ; "
            f"code au commit `{commit[:7]}`. **Uniquement avec l'accord écrit d'Ines** (accord du 11/11 au soir "
            "pour les 3 entraînements, transmis par le coordinateur). Vérité d'entraînement v3 (`cache_v3_partie_*`).")
    out = {}
    for g in GRAINES:
        quoi = ("statistiques d'innovation (jours d'entraînement) et choix de H (validation, OI seule), puis "
                if g == 42 else "attend les JSON communs écrits par le carnet de la graine 42, puis ")
        out[f"pcfno3d_national_v4_D2_s{g}_colab.ipynb"] = carnet([
            md(f"# PC-FNO3D NATIONAL v4, candidat D2 (D + canal d'innovation) — graine {g} (Colab, A100)\n\n"
               f"{tete}\n\nCe carnet : {quoi}entraînement (30 époques, recette de D, départ v2) et évaluation "
               "d'information (validation). **Aucune lecture du test.**"),
            md("## 1. Drive, code et données"), code(NN.COMMUN + git),
            md("## 2. Cache v3"), code(NN._ent_v3()),
            md("## 3. Mémoire du GPU"), code(NN.MEM),
            md("## 4. Statistiques, H, entraînement, information, fin de la session"),
            code(CHOIX_POIDS + TRAIN_D2.replace("@GRAINE@", str(g)).replace("@SHA_D2@", sha))], gpu=True)
    out["pcfno3d_national_v4_D2_test_colab.ipynb"] = carnet([
        md(f"# PC-FNO3D NATIONAL v4 D2 — lecture UNIQUE du test 2022 (Colab, A100)\n\n{tete}\n\n"
           "**Lecture du test : uniquement avec l'accord écrit d'Ines** (mettre `ACCORD_TEST_INES = True`). "
           "L'entraînement D2 est choisi sur le score de validation des trois graines ; le test est lu une fois "
           "pour lui et (s'il ne l'a pas déjà été) pour le modèle livré v2 sur la même vérité. Tables : D2, "
           "modèle livré, ERA5 seule, interpolation optimale seule."),
        md("## 1. Drive, code et données"), code(NN.COMMUN + git),
        md("## 2. Cache v3"), code(NN._ent_v3()),
        md("## 3. Mémoire du GPU"), code(NN.MEM),
        md("## 4. Choix, test, DSO, mesures réelles, décision"),
        code(CHOIX_POIDS + TEST_D2.replace("@SHA_D2@", sha))], gpu=True)
    return out


def main() -> int:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    for nom, nb in carnets(commit).items():
        f = REPO / "notebooks" / "current" / nom
        f.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f, "| code", commit[:7])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
