"""Construit le notebook Colab AUTONOME de la session M1 (mesure decisive BLOC 10i).

    python scripts/make_m1_colab_nb.py            # ecrit notebooks/current/m1_bloc10i_colab.ipynb
    python scripts/make_m1_colab_nb.py --check    # reconstruit en memoire et verifie le fichier

Sources (lues, jamais modifiees) :
  - notebooks/current/fno3d_v7.ipynb : BLOCS 0 a 8 et 10, copies TELS QUELS (titre markdown +
    code), sorties videes ;
  - notebooks/current/cellule_10i.py : cellule 10i figee, copiee entiere ; le sha256 de son code
    (apres la ligne « # ==== DEBUT ») doit egaler celui de son en-tete et SHA_10I ci-dessous ;
  - docs/M1_PROTOCOLE.md : texte de la regle pre-enregistree et de l'avenant n 1, imprime par la
    cellule M1-a avant tout calcul.

Cellules ajoutees (prefixe « M1- », aucune ne touche au code des cellules copiees) : controle des
fichiers figes et impression de la regle (M1-a), remplacement du BLOC 9 (M1-b), redirection des
ecritures du BLOC 10 hors du dossier fige eval_best_dso_v7/ (M1-c), dossier de sortie
eval_m1_bloc10i/ pour 10i (M1-d), archivage (M1-e), jeu vierge hors regle (M1-f). Voir l'avenant
du 2026-10-16 dans docs/M1_PROTOCOLE.md.

--check verifie aussi : syntaxe Python de chaque cellule (lignes ! et % neutralisees), egalite
octet a octet des cellules copiees avec la source, sha256 du code 10i, et noms globaux lus par
chaque cellule sans etre definis par une cellule precedente.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_NB = REPO / "notebooks" / "current" / "fno3d_v7.ipynb"
SRC_10I = REPO / "notebooks" / "current" / "cellule_10i.py"
PROTOCOLE = REPO / "docs" / "M1_PROTOCOLE.md"
OUT_NB = REPO / "notebooks" / "current" / "m1_bloc10i_colab.ipynb"

SHA_10I = "a1fcc878ed3c6c6047538e5b48df25c6633c972a413d6bd42f415926599fbd6f"   # en-tete de cellule_10i.py
SHA_POIDS = "b2529442ee6fcdb18d3f35fe176663022604395743c161a25541ccd4e365e7de"  # protocole, § 4 et § 8
SHA_NORM = "7bae6748f270fa5e3f08a69a7f8e919f01d641d29b125981168cc15a03ee25f7"   # protocole, § 4 et § 8
SHA_TEST_SUMMARY = "7944ba0aa0311a08cd1e1555c1a225239b3deabc5c9c2b55dcf4406b5da17425"  # protocole, § 8

DEBUT = "# ==== DEBUT\n"

# ------------------------------------------------------------------------------------------
# Cellules ajoutees. Les marqueurs @@...@@ sont remplaces a la generation.
# ------------------------------------------------------------------------------------------
INTRO_MD = """# Session M1 — mesure décisive BLOC 10i (v7, DSO1 à 650–1000 m)

Notebook **autonome**, généré par `scripts/make_m1_colab_nb.py` (ne pas l'éditer à la main :
relancer le script). Protocole : `docs/M1_PROTOCOLE.md` (règle pré-enregistrée, avenants 1 et 2,
§ 5 et § 9, avenant du 2026-10-16).

**Lancer** : *Exécution → Modifier le type d'exécution → GPU* (A100 demandé par le § 9 pour que
le DSO4 reste comparable aux 91–104 ms de la v7 ; sur L4 la mesure est valable mais le DSO4 doit
être signalé comme non comparable), puis *Exécution → Tout exécuter*, accepter la fenêtre
d'autorisation Google Drive (compte qui contient `aneo_fno3d_runs`), et garder l'onglet ouvert
jusqu'au message `M1-e OK`. **Aucun secret** n'est utilisé. Ne pas lancer le notebook atom sur le
même environnement d'exécution.

| Cellule | Source |
|---|---|
@@TABLE@@

- Code des cellules copiées : **identique** à `fno3d_v7.ipynb` (sha256 du fichier source
  `@@SHA_NB@@`) et à `cellule_10i.py` (sha256 du code 10i `@@SHA_10I@@`). Sorties vidées.
- **Non exécutés** : BLOC 9 (risque de réentraînement, remplacé par M1-b), BLOCS 10c, 10d,
  10e, 10f, 10g, 10h (ajustements et variantes déjà vues sur le test : la correction est figée
  dans `couche_limite_v7.json`, § 4), BLOCS 11 et 12.
- **Écritures sur le Drive** : `aneo_fno3d_runs/v7/eval_m1_bloc10i/` seulement (le BLOC 3 recopie
  en outre `aneo_cache_v7/index.csv`, à l'identique). Rien n'est écrit dans `eval_best_dso_v7/`
  ni sur les JSON figés (vérifié par sha256 avant et après, M1-a et M1-e).
- La cellule 10i ne s'exécute **qu'une fois** (§ 5.2) : M1-a refuse de démarrer si
  `decision_10i.json` existe déjà.
"""

M1A = '''# M1-a — Controle des fichiers figes et impression de la regle (AVANT TOUT CALCUL)
# Cellule ajoutee pour la session M1 (docs/M1_PROTOCOLE.md, § 5.1 point 1, avenant du 2026-10-16).
# Lit seulement le Drive, n'ecrit rien. S'arrete (exception) si un fichier fige manque ou a change,
# ou si une mesure 10i complete existe deja (regle d'execution unique, § 5.2).
import hashlib, datetime

M1_DRIVE_RUN = Path("/content/drive/MyDrive/aneo_fno3d_runs/v7")
M1_EVAL_DIR = M1_DRIVE_RUN / "eval_m1_bloc10i"
M1_SHA_PUBLIE = {                      # docs/M1_PROTOCOLE.md, § 4 et § 8
    "best_dso_v7.pt": "@@SHA_POIDS@@",
    "norm_stats.json": "@@SHA_NORM@@",
}
M1_SHA_INFO = {"eval_best_dso_v7/test_summary.json": "@@SHA_TEST_SUMMARY@@"}   # § 8 (controle, non bloquant)
M1_SHA_10I = "@@SHA_10I@@"             # code de la cellule 10i (en-tete de cellule_10i.py)
M1_JSON_CORRECTION = ("correction_moyenne_v7.json", "geostrophique_v2_v7.json", "couche_limite_v7.json")


def m1_sha256(p, bloc=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bloc), b""):
            h.update(b)
    return h.hexdigest()


_manque = [n for n in (*M1_SHA_PUBLIE, *M1_JSON_CORRECTION) if not (M1_DRIVE_RUN / n).exists()]
if _manque:
    raise FileNotFoundError(f"Fichiers figes absents de {M1_DRIVE_RUN} : {_manque} (§ 9 : a deposer avant la session)")
for _d in (M1_EVAL_DIR, M1_DRIVE_RUN / "eval_best_dso_v7"):
    if (_d / "decision_10i.json").exists():
        raise RuntimeError(f"{_d / 'decision_10i.json'} existe deja : une mesure 10i complete a eu lieu. "
                           "Regle d'execution unique (§ 5.2) : rapporter ce resultat, ne pas remesurer.")
t0 = time.time()
M1_SHA = {n: m1_sha256(M1_DRIVE_RUN / n) for n in (*M1_SHA_PUBLIE, *M1_JSON_CORRECTION, *M1_SHA_INFO)
          if (M1_DRIVE_RUN / n).exists()}
_ecart = [n for n, s in M1_SHA_PUBLIE.items() if M1_SHA[n] != s]
if _ecart:
    raise RuntimeError(f"sha256 different de la valeur publiee (§ 4) : {_ecart} -> ce ne sont pas les fichiers de la v7")
with open(M1_DRIVE_RUN / "norm_stats.json", encoding="utf-8") as f:
    assert "w_level_sd" in json.load(f), "norm_stats.json sans w_level_sd : le BLOC 5 le reecrirait"
with open(M1_DRIVE_RUN / "couche_limite_v7.json", encoding="utf-8") as f:
    assert abs(json.load(f)["ridge_alpha"] - 0.01) < 1e-12, "couche_limite_v7.json : alpha different de 0.01"
M1_GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "aucun"
M1_DEBUT = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

M1_REGLE = """@@REGLE@@"""

print("=" * 96)
print(f"SESSION M1 — mesure decisive BLOC 10i | {M1_DEBUT} | GPU : {M1_GPU}")
print("Regle pre-enregistree : commit 0b64978 (salon #325) ; avenant n 1 : commit 33da965 (salon #332)")
print("=" * 96)
print(M1_REGLE)
print("=" * 96)
print(f"sha256 (calcules sur le Drive en {time.time() - t0:.0f} s) :")
for n, s in M1_SHA.items():
    etat = ""
    if n in M1_SHA_PUBLIE:
        etat = "  (= valeur publiee)"
    elif n in M1_SHA_INFO:
        etat = "  (= valeur publiee)" if s == M1_SHA_INFO[n] else "  (DIFFERENT de la valeur publiee § 8 : a signaler)"
    print(f"  {s}  {n}{etat}")
print(f"  {M1_SHA_10I}  code de la cellule 10i (publie ; recontrole apres execution par M1-e)")
if "A100" not in M1_GPU:
    print(f"ATTENTION : GPU {M1_GPU}, pas A100 : le DSO4 n'est pas comparable aux 91-104 ms de la v7 "
          "(§ 5.1 point 5, § 9). A signaler avec le verdict.")
print("M1-a OK")
'''

M1B = '''# M1-b — A LA PLACE DU BLOC 9 (non execute) : aucun entrainement
# Le BLOC 9 reprend depuis last.pt (~7 Go, Drive). Si ce fichier manque, il REENTRAINE 140 epoques
# (~5 h) et reecrit best.pt et history.json sur le Drive : il n'est donc pas execute.
# Seules les deux lignes du BLOC 9 dont le BLOC 10 et la cellule 10i ont besoin sont reprises,
# a l'identique. Le checkpoint est charge par le BLOC 10 lui-meme, comme dans le notebook v7 :
# RUN_DIR/best_dso_v7.pt (sha256 verifie par M1-a et M1-c).
amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
use_amp = device.type == "cuda"
print(f"BLOC 9 non execute | amp_dtype = {amp_dtype} | use_amp = {use_amp}")
print("M1-b OK")
'''

M1C = '''# M1-c — Le BLOC 10 ecrit dans RUN_DIR / "eval_best_dso_v7" : les CSV et le test_summary.json FIGES de la v7
# (sha256 publie, § 8). Pour l'executer tel quel sans les reecrire, RUN_DIR pointe, le temps du BLOC 10,
# vers un dossier LOCAL qui contient une copie de best_dso_v7.pt (sha256 verifie). Le BLOC 10 y charge
# le meme modele et y ecrit ses resultats rejoues, archives ensuite par M1-e dans eval_m1_bloc10i/bloc10_rejoue/.
# RUN_DIR revient au Drive dans M1-d, avant la cellule 10i.
import shutil

assert RUN_DIR == M1_DRIVE_RUN, f"RUN_DIR inattendu : {RUN_DIR}"
M1_LOCAL_RUN = Path("/content/m1_run_v7")
M1_LOCAL_RUN.mkdir(parents=True, exist_ok=True)
_dst = M1_LOCAL_RUN / "best_dso_v7.pt"
t0 = time.time()
if not _dst.exists() or m1_sha256(_dst) != M1_SHA["best_dso_v7.pt"]:
    shutil.copy2(M1_DRIVE_RUN / "best_dso_v7.pt", _dst)
assert m1_sha256(_dst) == M1_SHA_PUBLIE["best_dso_v7.pt"], "copie locale de best_dso_v7.pt corrompue"
RUN_DIR = M1_LOCAL_RUN
print(f"best_dso_v7.pt copie en local ({time.time() - t0:.0f} s, sha256 conforme) | RUN_DIR (BLOC 10) = {RUN_DIR}")
print("M1-c OK")
'''

M1D = '''# M1-d — Avant la cellule 10i : RUN_DIR revient au Drive (10i lit RUN_DIR / "couche_limite_v7.json", fige,
# sans l'ecrire) et EVAL_DIR pointe vers eval_m1_bloc10i/ (§ 5.1 point 8) : 10i y ecrit
# decision_10i_par_niveau.csv et decision_10i.json, jamais dans eval_best_dso_v7/.
M1_BLOC10_DIR = Path(EVAL_DIR)                       # resultats rejoues du BLOC 10 (disque local)
assert str(M1_BLOC10_DIR).startswith(str(M1_LOCAL_RUN)), f"le BLOC 10 a ecrit hors du dossier local : {M1_BLOC10_DIR}"
RUN_DIR = M1_DRIVE_RUN
assert m1_sha256(RUN_DIR / "couche_limite_v7.json") == M1_SHA["couche_limite_v7.json"], "couche_limite_v7.json a change"
if (M1_EVAL_DIR / "decision_10i.json").exists():
    raise RuntimeError("decision_10i.json existe deja : regle d'execution unique (§ 5.2)")
M1_EVAL_DIR.mkdir(parents=True, exist_ok=True)
_restes = [p for p in M1_EVAL_DIR.iterdir() if not p.name.startswith("_tentative_interrompue_")]
if _restes:                                          # tentative interrompue (§ 5.2) : gardee a part, jamais effacee
    _dest = M1_EVAL_DIR / f"_tentative_interrompue_{time.strftime('%Y%m%d_%H%M%S')}"
    _dest.mkdir()
    for p in _restes:
        shutil.move(str(p), str(_dest / p.name))
    print(f"Fichiers d'une tentative interrompue deplaces dans {_dest.name} : {[p.name for p in _restes]}")
EVAL_DIR = M1_EVAL_DIR
print(f"RUN_DIR = {RUN_DIR} | EVAL_DIR (10i) = {EVAL_DIR}")
print(f"Cellule suivante : BLOC 10i, code fige, sha256 publie {M1_SHA_10I}")
print("M1-d OK")
'''

M1E = '''# M1-e — Archivage (§ 5.1 point 8) dans eval_m1_bloc10i/ : sha256.txt, regle.txt, resultats rejoues du BLOC 10.
# Controle le sha256 du code 10i REELLEMENT execute (historique IPython) et verifie que les fichiers figes
# n'ont pas change pendant la session. N'ecrit que dans eval_m1_bloc10i/.
assert EVAL_DIR == M1_EVAL_DIR and (EVAL_DIR / "decision_10i.json").exists(), "la cellule 10i n'a pas termine"
_marque_debut, _marque_fin = "# ==== " + "DEBUT\\n", "BLOC 10i" + " OK"
_hist = [c for c in globals().get("In", []) if _marque_debut in c and _marque_fin in c]
if _hist:
    _code = _hist[-1].split(_marque_debut, 1)[1]
    M1_SHA_10I_EXEC = hashlib.sha256((_code.rstrip("\\n") + "\\n").encode("utf-8")).hexdigest()
else:
    M1_SHA_10I_EXEC = "non disponible (historique IPython absent)"
_ok10i = M1_SHA_10I_EXEC == M1_SHA_10I
t0 = time.time()
_apres = {n: m1_sha256(M1_DRIVE_RUN / n) for n in M1_SHA}
_change = [n for n in M1_SHA if _apres[n] != M1_SHA[n]]
_bloc10 = EVAL_DIR / "bloc10_rejoue"
_bloc10.mkdir(exist_ok=True)
for p in sorted(M1_BLOC10_DIR.iterdir()):
    if p.is_file():
        shutil.copy2(p, _bloc10 / p.name)
with open(EVAL_DIR / "sha256.txt", "w", encoding="utf-8") as f:
    f.write(f"# Session M1, BLOC 10i | debut {M1_DEBUT} | GPU {M1_GPU}\\n")
    f.write("# fichiers figes (Drive aneo_fno3d_runs/v7/), sha256 en debut de session :\\n")
    for n, s in M1_SHA.items():
        f.write(f"{s}  {n}\\n")
    f.write(f"# code de la cellule 10i : publie {M1_SHA_10I} | execute {M1_SHA_10I_EXEC}\\n")
    f.write(f"# fichiers figes modifies pendant la session : {_change if _change else 'aucun'}\\n")
    for p in sorted(EVAL_DIR.glob("decision_10i*")):
        f.write(f"{m1_sha256(p)}  eval_m1_bloc10i/{p.name}\\n")
with open(EVAL_DIR / "regle.txt", "w", encoding="utf-8") as f:
    f.write(M1_REGLE + "\\n")
print(f"Code 10i execute : {M1_SHA_10I_EXEC} -> {'CONFORME au sha256 publie' if _ok10i else 'NON CONFORME / non verifie : a signaler'}")
print(f"Fichiers figes inchanges pendant la session : {'oui' if not _change else 'NON : ' + str(_change)} ({time.time() - t0:.0f} s)")
print("Archive :", EVAL_DIR)
for p in sorted(EVAL_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(EVAL_DIR)}  ({p.stat().st_size:,} o)")
print("M1-e OK")
'''

M1F = '''# M1-f — Jeu VIERGE graine 2027 (avenant n 2) : mesure HORS REGLE, pour information. N'entre pas dans C1/C2/C3.
# Ne s'execute que si le jeu complet est sur le Drive (M1_VIERGE_DIR, 15 jours, aucun fichier .part) ;
# sinon, message et rien d'autre. Une seule execution : refuse si VIERGE_OUT/summary.json existe.
# Protocole de l'avenant n 2 : instants synoptiques, 50 % des capteurs, graine 2000, niveaux evalues de 10i
# (min_level_cover 0.05), v7 seul et v7 + correction figee (fonctions de la cellule 10i : predict_corrected,
# sampler/boot non utilises pour les grappes, rmse_lv, ci, worst), DSO2, DSO3, DSO5 (25 %, 40 instants,
# graine 3000), IC 95 % par bootstrap sur les JOURS entiers (1000 tirages, graine fixe).
# Ecrit seulement : cache local /content/aneo_cache_seed2027 et EVAL_DIR / "jeu_vierge_seed2027" / *.
M1_VIERGE_DIR = Path("/content/drive/MyDrive/aneo_dataset_3d_seed2027")
M1_VIERGE_CACHE = Path("/content/aneo_cache_seed2027")
M1_VIERGE_OUT = M1_EVAL_DIR / "jeu_vierge_seed2027"
M1_VIERGE_JOURS = 15


def m1_vierge_absent():
    if not (M1_VIERGE_DIR / "meta.json").exists():
        return f"{M1_VIERGE_DIR}/meta.json introuvable : le jeu vierge n'est pas sur le Drive"
    if not (M1_VIERGE_DIR / "aneo_grid_points.csv").exists():
        return f"{M1_VIERGE_DIR}/aneo_grid_points.csv introuvable"
    fichiers = sorted(M1_VIERGE_DIR.glob("aneo_observations_day_*_multitime.csv*"))
    partiels = [p.name for p in fichiers if p.name.endswith(".part")]
    complets = [p for p in fichiers if not p.name.endswith(".part")]
    if partiels:
        return f"fichiers incomplets (.part) : {partiels}"
    if len(complets) != M1_VIERGE_JOURS:
        return f"{len(complets)} jours sur le Drive au lieu de {M1_VIERGE_JOURS}"
    return None


def m1_vierge_grille():
    """Meme geometrie que le jeu v2 (BLOCS 1-2) : meta.json et aneo_grid_points.csv compares."""
    with open(M1_VIERGE_DIR / "meta.json", encoding="utf-8") as f:
        meta2 = json.load(f)
    for key in ("grille_surface", "noeuds_3d"):
        assert meta2[key] == META[key], f"meta.json : '{key}' differe du jeu v2"
    assert meta2["calendrier"]["jours"] == M1_VIERGE_JOURS, meta2["calendrier"]
    g2 = pd.read_csv(M1_VIERGE_DIR / "aneo_grid_points.csv",
                     usecols=["point_id", "ix", "iy", "z_terrain_m", "z_rel_m", "z0_m", "slope_deg", "aspect_deg"])
    s2 = g2[g2["point_id"] < NODE_ID_START].sort_values("point_id").reset_index(drop=True)
    n2 = g2[g2["point_id"] >= NODE_ID_START].sort_values("point_id").reset_index(drop=True)
    s1 = surf.sort_values("point_id").reset_index(drop=True)
    n1 = nodes.sort_values("point_id").reset_index(drop=True)
    for a, b, nom in ((s1, s2, "surface"), (n1, n2, "noeuds 3D")):
        assert len(a) == len(b), f"grille {nom} : {len(b)} points au lieu de {len(a)}"
        for c in ("point_id", "ix", "iy"):
            assert np.array_equal(a[c].values, b[c].values), f"grille {nom} : colonne {c} differente"
        for c in ("z_terrain_m", "z_rel_m") + (("z0_m", "slope_deg", "aspect_deg") if nom == "surface" else ()):
            e = float(np.nanmax(np.abs(a[c].values - b[c].values)))
            assert e < 1e-3, f"grille {nom} : colonne {c}, ecart max {e}"
    return meta2


def m1_vierge_run(ds):
    """Comme run() de la cellule 10i, avec en plus la masse et la rugosite AVANT correction (v7 seul)."""
    N = len(ds)
    R = {"avant": np.zeros((N, NZ, 2)), "apres": np.zeros((N, NZ, 2)),
         "st": np.zeros((N, NZ, 2)), "st2": np.zeros((N, NZ, 2)),
         "mass_avant": [], "mass_apres": [], "rough_avant": [], "rough_apres": []}
    for i, (s_raw, sensor, tfeat, y, _) in enumerate(DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS)):
        u, v, w, uc, vc = predict_corrected(s_raw, sensor, tfeat)
        y = y.to(device)
        for k in EVAL_LEVELS:
            mk = VALID_T[k]
            for ci_ in range(2):
                tk = y[0, ci_, k][mk].double()
                R["avant"][i, k, ci_] = (((u, v)[ci_][0, k][mk].double() - tk) ** 2).sum().item()
                R["apres"][i, k, ci_] = (((uc, vc)[ci_][0, k][mk].double() - tk) ** 2).sum().item()
                R["st"][i, k, ci_], R["st2"][i, k, ci_] = tk.sum().item(), (tk ** 2).sum().item()
        for lab, (a, b) in (("avant", (u, v)), ("apres", (uc, vc))):
            r = mass_residual(y[:, 3], a, b, w)
            R["mass_" + lab].append(torch.sqrt((r[:, MASS_MASK_T] ** 2).mean()).item())
            R["rough_" + lab].append(((roughness(a) + roughness(b)) / (roughness(y[:, 0]) + roughness(y[:, 1]))).item())
    return R


def m1_vierge_mesure():
    global CACHE_DIR
    t0 = time.time()
    meta2 = m1_vierge_grille()
    sha_meta = m1_sha256(M1_VIERGE_DIR / "meta.json")
    sha_grille = m1_sha256(M1_VIERGE_DIR / "aneo_grid_points.csv")
    print(f"Geometrie identique au jeu v2 | sha256 meta.json {sha_meta} | grille {sha_grille} ({time.time() - t0:.0f} s)")
    cache_v2 = CACHE_DIR
    try:
        CACHE_DIR = M1_VIERGE_CACHE                    # lu par build_day (BLOC 3) et ANEO3D (BLOC 6)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        idx_path = CACHE_DIR / "index.csv"
        faits = set(pd.read_csv(idx_path)["day"]) if idx_path.exists() else set()
        for path in sorted(M1_VIERGE_DIR.glob("aneo_observations_day_*_multitime.csv*")):
            if day_of(path) in faits:
                continue
            rows = pd.DataFrame(build_day(path)).assign(split="vierge")
            rows.to_csv(idx_path, mode="a", header=not idx_path.exists(), index=False)
            print(f"  cache jeu vierge : jour {day_of(path):03d}, {len(rows)} instants ({time.time() - t0:.0f} s)")
        idx = pd.read_csv(idx_path).drop_duplicates("stem").sort_values("stem").reset_index(drop=True)
        idx = idx[[is_synoptic(t) for t in idx["timestamp"]]].reset_index(drop=True)
        st_, hr_, dy_ = idx["stem"].values, idx["hour"].values, idx["doy"].values
        jours = idx["day"].values
        print(f"Jeu vierge : {len(idx)} instants synoptiques, {len(set(jours))} jours")
        R = m1_vierge_run(ANEO3D(st_, hr_, dy_, train=False, seed=2000))
        sub_v = np.unique(np.linspace(0, len(st_) - 1, min(40, len(st_))).astype(int))
        R25 = run_keep(sub_v, 3000, 0.25, stems=st_, hours=hr_, doys=dy_)
    finally:
        CACHE_DIR = cache_v2
    groupes = [np.nonzero(jours == d)[0] for d in pd.unique(jours)]
    rng = np.random.default_rng(2027)
    full, dr = boot(R, lambda: np.concatenate([groupes[g] for g in rng.integers(0, len(groupes), len(groupes))]))
    N = len(st_)
    n_all = N * NK[:, None]
    sst = R["st2"].sum(0) - R["st"].sum(0) ** 2 / n_all
    r2 = {n: 1 - R[n].sum(0) / sst for n in ("avant", "apres")}
    f25 = {n: rmse_lv(R25[n].sum(0), R25["avant"].shape[0]) for n in ("avant", "apres")}
    rows = []
    print("=" * 96)
    print(f"JEU VIERGE graine 2027 (HORS REGLE) : {N} instants, 50 % des capteurs, IC 95 % bootstrap par jours ({len(groupes)} grappes)")
    print(" Niveau |  u avant [IC]          u apres [IC]          R2 apres |  v avant [IC]          v apres [IC]          R2 apres")
    for k in LV:
        cells = []
        for ci_, c in enumerate("uv"):
            ia, ip = ci(dr["avant"][:, k, ci_]), ci(dr["apres"][:, k, ci_])
            dd = ci(dr["apres"][:, k, ci_] - dr["avant"][:, k, ci_])
            cells.append(f"{full['avant'][k, ci_]:.3f} [{ia[0]:.3f}, {ia[1]:.3f}]  {full['apres'][k, ci_]:.3f} "
                         f"[{ip[0]:.3f}, {ip[1]:.3f}]  {r2['apres'][k, ci_]:.4f}")
            rows.append({"z_m": int(Z_LEVELS[k]), "comp": c,
                         "avant": full["avant"][k, ci_], "avant_ic_bas": ia[0], "avant_ic_haut": ia[1],
                         "apres": full["apres"][k, ci_], "apres_ic_bas": ip[0], "apres_ic_haut": ip[1],
                         "delta": full["apres"][k, ci_] - full["avant"][k, ci_], "delta_ic_bas": dd[0], "delta_ic_haut": dd[1],
                         "r2_avant": r2["avant"][k, ci_], "r2_apres": r2["apres"][k, ci_],
                         "cap25_avant": f25["avant"][k, ci_], "cap25_apres": f25["apres"][k, ci_]})
        print(f" {int(Z_LEVELS[k]):5d} m |  {cells[0]} |  {cells[1]}")
    W = {}
    for ci_, c in enumerate("uv"):
        for n in ("avant", "apres"):
            W[f"{c}_{n}"], W[f"{c}_{n}_ic95"] = worst(full, dr, n, ci_)
        print(f"Pire niveau {c} : v7 seul {W[c + '_avant']:.3f} {tuple(round(x, 3) for x in W[c + '_avant_ic95'])}"
              f" -> v7 + correction {W[c + '_apres']:.3f} {tuple(round(x, 3) for x in W[c + '_apres_ic95'])} m/s")
    print("Difference appariee apres - avant, 650-1000 m (IC 95 %) :")
    for r_ in rows:
        if r_["z_m"] >= 650:
            print(f"  {r_['z_m']:5d} m {r_['comp']} : {r_['delta']:+.3f} [{r_['delta_ic_bas']:+.3f}, {r_['delta_ic_haut']:+.3f}]")
    dso = {n: {"dso2": float(np.mean(R["mass_" + n])), "dso3": float(np.mean(R["rough_" + n])),
               "dso5_25pct": float(f25[n][LV].max())} for n in ("avant", "apres")}
    for n, lab in (("avant", "v7 seul"), ("apres", "v7 + correction")):
        print(f"{lab:16s} : DSO2 {dso[n]['dso2']:.2e} | DSO3 {dso[n]['dso3']:.3f} | DSO5 (25 %, {len(sub_v)} instants) {dso[n]['dso5_25pct']:.3f}")
    print("Aucun seuil n'est applique ; aucune decision n'en depend (avenant n 2).")
    M1_VIERGE_OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(M1_VIERGE_OUT / "par_niveau.csv", index=False)
    with open(M1_VIERGE_OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"hors_regle": True, "avenant": "n 2 (2026-10-14, salon #341)", "graine_jeu": 2027,
                   "sha256_meta_json": sha_meta, "sha256_grille": sha_grille, "calendrier": meta2["calendrier"],
                   "instants": int(N), "jours": sorted(int(d) for d in set(jours)), "capteurs_visibles": 0.5,
                   "graine_masques": 2000, "bootstrap": {"grappes": "jours", "tirages": B, "graine": 2027},
                   "pire_niveau": W, "dso": dso, "correction_sha256": CORR_SHA,
                   "instants_dso5": int(len(sub_v))}, f, indent=2)
    print("Resultats :", M1_VIERGE_OUT, f"({(time.time() - t0) / 60:.1f} min)")


_raison = m1_vierge_absent()
if _raison is not None:
    print("JEU VIERGE NON MESURE (hors regle, sans effet sur la decision 10i) : " + _raison)
    print(f"Pour le mesurer : deposer le jeu complet (meta.json, split_days.json, aneo_grid_points.csv, "
          f"{M1_VIERGE_JOURS} fichiers aneo_observations_day_*_multitime.csv.gz) dans {M1_VIERGE_DIR}, puis executer "
          "cette cellule seule dans la meme session (apres 10i).")
elif (M1_VIERGE_OUT / "summary.json").exists():
    print(f"Jeu vierge deja mesure ({M1_VIERGE_OUT / 'summary.json'}) : une seule execution (avenant n 2).")
else:
    m1_vierge_mesure()
print("M1-f OK")
'''


# ------------------------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_10i(text: str) -> str:
    """Code de la cellule (apres la ligne DEBUT), termine par un seul saut de ligne."""
    assert text.count(DEBUT) == 1, "ligne « # ==== DEBUT » absente ou repetee"
    return text.split(DEBUT, 1)[1].rstrip("\n") + "\n"


def extrait_regle(md: str) -> str:
    lignes = md.split("\n")

    def bloc(debut: str, fin: str) -> str:
        i = next(n for n, l in enumerate(lignes) if l.startswith(debut))
        j = next(n for n, l in enumerate(lignes) if n > i and l.startswith(fin))
        return "\n".join(lignes[i:j]).strip()

    regle = bloc("## Règle de décision pré-enregistrée", "> **Lecture de la règle**")
    av1 = bloc("## Avenant n° 1", "## Avenant n° 2")
    texte = regle + "\n\n" + av1
    assert '"""' not in texte and "\\" not in texte, "le texte de la regle ne peut pas etre mis tel quel dans une chaine"
    return texte


def blocs_source(nb: dict) -> dict[str, tuple[dict | None, dict]]:
    """BLOC -> (cellule markdown de titre, cellule de code) dans fno3d_v7.ipynb."""
    out: dict[str, tuple[dict | None, dict]] = {}
    cells = nb["cells"]
    for n, c in enumerate(cells):
        src = "".join(c["source"])
        m = re.match(r"# BLOC (\w+) — ", src)
        if c["cell_type"] == "code" and m:
            assert m.group(1) not in out, f"BLOC {m.group(1)} en double"
            prev = cells[n - 1] if n and cells[n - 1]["cell_type"] == "markdown" else None
            if prev is not None and not "".join(prev["source"]).startswith(f"## BLOC {m.group(1)} "):
                prev = None
            out[m.group(1)] = (prev, c)
    return out


def lignes(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def md_cell(cid: str, text: str, prov: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {"id": cid, "m1_source": prov}, "source": lignes(text)}


def code_cell(cid: str, src: list[str] | str, prov: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {"id": cid, "m1_source": prov}, "outputs": [],
            "source": src if isinstance(src, list) else lignes(src)}


def copie(c: dict, prov: str) -> dict:
    """Copie TELLE QUELLE du source, sorties videes, metadonnees Colab de l'execution retirees."""
    if c["cell_type"] == "code":
        return code_cell(c["id"], list(c["source"]), prov)
    return md_cell(c["id"], "".join(c["source"]), prov)


def build() -> tuple[dict, list[tuple[str, str]]]:
    nb_bytes = SRC_NB.read_bytes()
    nb = json.loads(nb_bytes.decode("utf-8"))
    t10i = SRC_10I.read_bytes().decode("utf-8")
    assert "\r" not in t10i, "cellule_10i.py doit etre en LF"
    sha10 = sha256_bytes(code_10i(t10i).encode("utf-8"))
    entete = re.search(r"^#\s+([0-9a-f]{64})\s*$", t10i, re.M)
    assert entete and entete.group(1) == sha10 == SHA_10I, \
        f"sha256 du code 10i : calcule {sha10}, en-tete {entete and entete.group(1)}, attendu {SHA_10I}"
    regle = extrait_regle(PROTOCOLE.read_text(encoding="utf-8"))
    src = blocs_source(nb)
    for b in ("9", "10"):
        assert b in src, f"BLOC {b} introuvable dans {SRC_NB.name}"

    cells: list[dict] = []
    ordre: list[tuple[str, str]] = []

    def ajoute_bloc(b: str) -> None:
        md, code = src[b]
        prov = f"{SRC_NB.name}, BLOC {b} (cellule {code['id']}), copie telle quelle"
        if md is not None:
            cells.append(copie(md, prov))
        cells.append(copie(code, prov))
        ordre.append((f"BLOC {b}", prov))

    def ajoute(cid: str, titre: str, code: str, prov: str) -> None:
        cells.append(md_cell(cid + "-titre", f"## {titre}", prov))
        cells.append(code_cell(cid, code, prov))
        ordre.append((titre, prov))

    remplace = {"@@SHA_POIDS@@": SHA_POIDS, "@@SHA_NORM@@": SHA_NORM, "@@SHA_TEST_SUMMARY@@": SHA_TEST_SUMMARY,
                "@@SHA_10I@@": SHA_10I, "@@REGLE@@": regle}

    def sub(t: str) -> str:
        for k, v in remplace.items():
            t = t.replace(k, v)
        assert "@@" not in t
        return t

    ajoute_bloc("0")
    ajoute("m1-a", "M1-a — Contrôle des fichiers figés, impression de la règle", sub(M1A),
           "ajoutee (§ 5.1 point 1)")
    for b in ("1", "2", "3", "4", "5", "6", "7", "8"):
        ajoute_bloc(b)
    ajoute("m1-b", "M1-b — À la place du BLOC 9 (aucun entraînement)", sub(M1B),
           "ajoutee : 2 lignes du BLOC 9, a l'identique")
    ajoute("m1-c", "M1-c — BLOC 10 sans écrire dans eval_best_dso_v7/", sub(M1C), "ajoutee")
    ajoute_bloc("10")
    ajoute("m1-d", "M1-d — Sorties de 10i dans eval_m1_bloc10i/", sub(M1D), "ajoutee (§ 5.1 point 8)")
    prov10i = f"{SRC_10I.name}, fichier entier, copie tel quel (sha256 du code {SHA_10I})"
    cells.append(md_cell("m1-10i-titre", "## BLOC 10i — Mesure décisive (code figé)", prov10i))
    cells.append(code_cell("m1-10i", t10i, prov10i))
    ordre.append(("BLOC 10i", prov10i))
    ajoute("m1-e", "M1-e — Archivage et contrôles de fin", sub(M1E), "ajoutee (§ 5.1 point 8)")
    ajoute("m1-f", "M1-f — Jeu vierge graine 2027 (avenant n° 2, hors règle)", sub(M1F),
           "ajoutee (avenant n 2), gardee par la presence du jeu sur le Drive")

    table = "\n".join(f"| {t} | {p} |" for t, p in ordre)
    intro = INTRO_MD.replace("@@TABLE@@", table).replace("@@SHA_NB@@", sha256_bytes(nb_bytes)) \
        .replace("@@SHA_10I@@", SHA_10I)
    cells.insert(0, md_cell("m1-intro", intro, "ajoutee"))
    out = {"cells": cells,
           "metadata": {"accelerator": "GPU",
                        "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
                        "kernelspec": {"display_name": "Python 3", "name": "python3"},
                        "language_info": {"name": "python"}},
           "nbformat": 4, "nbformat_minor": 5}
    return out, ordre


def serialise(nb: dict) -> str:
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


# ------------------------------------------------------------------------------------------
def neutralise(src: str) -> str:
    return "".join(("pass  # " + l.lstrip()) if l.lstrip().startswith(("!", "%")) else l for l in lignes(src))


def noms(tree: ast.AST) -> tuple[set[str], set[str]]:
    """(noms definis, noms lus) — approximation suffisante pour reperer une globale manquante."""
    defs, lus = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (defs if isinstance(n.ctx, (ast.Store, ast.Del)) else lus).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            defs.add(n.name)
        elif isinstance(n, ast.arg):
            defs.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                defs.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defs.add(n.name)
        elif isinstance(n, ast.Global):
            defs.update(n.names)
    return defs, lus


def check(nb: dict) -> int:
    err = 0
    src_nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    par_id = {c["id"]: c for c in src_nb["cells"]}
    connus = set(dir(builtins)) | {"In", "Out", "get_ipython"}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        titre = s.splitlines()[0][:70]
        try:
            tree = ast.parse(neutralise(s))
        except SyntaxError as e:
            print(f"ERREUR syntaxe : {titre} : {e}")
            err += 1
            continue
        if c["id"] in par_id and par_id[c["id"]]["source"] != c["source"]:
            print(f"ERREUR : {titre} differe de la cellule source {c['id']}")
            err += 1
        if c["id"] == "m1-10i":
            if sha256_bytes(code_10i(s).encode("utf-8")) != SHA_10I:
                print("ERREUR : sha256 du code 10i dans le notebook")
                err += 1
        defs, lus = noms(tree)
        manquants = sorted(lus - defs - connus)
        if manquants:
            print(f"ERREUR : {titre} lit des noms non definis avant : {manquants}")
            err += 1
        connus |= defs
    print(f"{sum(c['cell_type'] == 'code' for c in nb['cells'])} cellules de code : "
          f"{'syntaxe, copies, sha256 10i et globales OK' if not err else str(err) + ' erreur(s)'}")
    return err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="verifie le notebook ecrit au lieu de l'ecrire")
    args = ap.parse_args()
    nb, ordre = build()
    texte = serialise(nb)
    err = check(nb)
    if args.check:
        if not OUT_NB.exists() or OUT_NB.read_bytes().decode("utf-8") != texte:
            print(f"ERREUR : {OUT_NB.name} absent ou different de la generation (relancer sans --check)")
            err += 1
    elif not err:
        OUT_NB.write_bytes(texte.encode("utf-8"))
        print(f"Ecrit : {OUT_NB.relative_to(REPO)} ({len(texte.encode('utf-8')):,} o)")
    for n, (t, p) in enumerate(ordre, 1):
        print(f"  {n:2d}. {t:60s} <- {p}")
    print(f"sha256 du code 10i : {SHA_10I} (en-tete de {SRC_10I.name} : conforme)")
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
