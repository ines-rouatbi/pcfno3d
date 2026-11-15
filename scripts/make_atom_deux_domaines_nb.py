"""Carnet Colab OS5 : ATOM Wind sert DEUX domaines a la fois (Grand Tunis + Tunisie entiere), choix par la position.

Objectif d'Ines du 16/11/2026, chantier B : C' s44 (w servi 0,75) a cote du Grand Tunis livre, sans le remplacer ;
couche 10 m corrigee par les aeroports ; DSO4 mesure sur le GPU du service (mediane de 20 passes, p95, memoire
maximale) ; demonstration de bout en bout sur 4 villes hors du Grand Tunis (+ Tunis). Aucun secret dans le carnet :
les identifiants du tunnel viennent du secret Colab TUNNEL_CRED_CONTENTS, la cle de demonstration est creee et
revoquee en memoire, jamais affichee.

    python scripts/make_atom_deux_domaines_nb.py --commit <sha>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_grand_tunis_nb import CHOIX_POIDS  # noqa: E402  (meme choix de poids que le carnet de service)

REPO =Path(__file__).resolve().parents[1]
SOURCE = REPO / "notebooks" / "current" / "pcfno3d_national_v4_test_colab.ipynb"
SORTIE = REPO / "notebooks" / "current" / "pcfno3d_atom_deux_domaines_colab.ipynb"
SHA_PRODUIT = (REPO / "configs" / "national" / "regle_produit_vent10m_v4.json.sha256").read_text(encoding="ascii").split()[0]


def cellule(kind: str, texte: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": texte.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


CACHE_NAT = r'''# national : instants de TEST seulement (le service ne sert que le test 2022 et le temps reel), + le jeu
import pandas as pd
LOCN = Path("/content/national")
CACHEN, JEUN = LOCN / "cache", LOCN / "jeu"
CACHEN.mkdir(parents=True, exist_ok=True)
JEUN.mkdir(parents=True, exist_ok=True)
if not (CACHEN / "index.csv").exists():
    with zipfile.ZipFile(NAT / "jeu_v3.zip") as zf:
        zf.extractall(JEUN)
    index = []
    for k in (1, 2, 3):
        with zipfile.ZipFile(NAT / f"cache_v3_partie_{k}.zip") as zf:
            with zf.open(f"index_partie_{k}.csv") as f:
                idx = pd.read_csv(f)
            index.append(idx)
            test = set(idx.loc[idx["split"] == "test", "stem"].astype(str))
            zf.extractall(CACHEN, members=[x for x in zf.namelist() if x.rsplit("_", 1)[0] in test])
    pd.concat(index).drop_duplicates("stem").to_csv(CACHEN / "index.csv", index=False)
print("national :", len(list(CACHEN.glob("*_Y.npy"))), "instants de test ; METAR :", (G / "obs").exists())
'''

GRAND_TUNIS = CHOIX_POIDS + r'''# Grand Tunis : donnees, instants de test, vraies mesures ; modele livre (decision_v4.json). Si la copie de
# production manque (le carnet de service Grand Tunis n'a pas tourne depuis la decision v4), elle est faite ici
# exactement comme dans ce carnet : copie seulement, rien n'est supprime, une ancienne copie est renommee.
GT = Path("/content/drive/MyDrive/pcfno3d_grand_tunis")
RUNS = GT / "runs"
PROD = RUNS / "grand_tunis_v9_production"
DECISION = RUNS / "decision_v4.json"
livre = json.loads(DECISION.read_text())["modele_livre"] if DECISION.exists() else "grand_tunis_v2_long"
SRC = RUNS / livre
print("modele Grand Tunis livre :", livre)
if (PROD / "PROVENANCE.txt").exists() and f"runs/{livre} " not in (PROD / "PROVENANCE.txt").read_text():
    import re
    ancien = re.search(r"Copie de runs/(\S+)", (PROD / "PROVENANCE.txt").read_text())
    cote = RUNS / f"grand_tunis_v9_production_avant_{livre}_{ancien[1] if ancien else 'inconnu'}"
    if cote.exists():
        raise RuntimeError(f"{cote} existe deja : verifie avant de remplacer {PROD}")
    PROD.rename(cote)
    print("ancienne copie de production mise de cote :", cote.name)
if not (PROD / "best.pt").exists():
    PROD.mkdir(parents=True, exist_ok=True)
    shutil.copy2(choisir_poids(SRC), PROD / "best.pt")
    for nom in ("norm_stats.json", "history.json"):
        shutil.copy2(SRC / nom, PROD / nom)
    for d in ("eval_pcfno3d", "validation_reelle", "fiabilite_pcfno3d", "figures"):
        if (SRC / d).exists():
            shutil.copytree(SRC / d, PROD / d, dirs_exist_ok=True)
    for nom in ("decision_v4.json", "rapport_v4.md"):
        if (RUNS / nom).exists():
            shutil.copy2(RUNS / nom, PROD / nom)
    (PROD / "PROVENANCE.txt").write_text(
        f"Copie de runs/{livre} : le modele livre PC-FNO3D Grand Tunis, servi avec w recalcule par "
        "continuite (decision d'Ines, 2026-10-24 ; runs/decision_v4.json et rapport_v4.md).", encoding="utf-8")
    print("copie de production faite :", PROD)
assert (PROD / "best.pt").exists(), PROD
if not (DON / "grand_tunis" / "forcage_era5.csv").exists():
    with zipfile.ZipFile(GT / "grand_tunis_donnees.zip") as zf:
        zf.extractall(DON)
import collections
morceaux = collections.defaultdict(list)
for f in sorted(GT.glob("maj_*")):                    # mises a jour deposees apres coup sur le Drive
    nom = f.name[4:]
    tete, _, queue = nom.rpartition(".")
    morceaux[tete if queue.isdigit() and tete else nom].append(f)
for nom, fs in morceaux.items():
    with open(DON / "grand_tunis" / nom, "wb") as out:
        for f in sorted(fs):
            out.write(f.read_bytes())
LOCG = Path("/content/grand_tunis")
CACHEG, JEUG = LOCG / "cache", LOCG / "jeu"
CACHEG.mkdir(parents=True, exist_ok=True)
JEUG.mkdir(parents=True, exist_ok=True)
if not (CACHEG / "index.csv").exists():
    with zipfile.ZipFile(GT / "jeu.zip") as zf:
        zf.extractall(JEUG)
    index = []
    for k in (1, 2, 3):
        with zipfile.ZipFile(GT / f"cache_partie_{k}.zip") as zf:
            with zf.open(f"index_partie_{k}.csv") as f:
                idx = pd.read_csv(f)
            index.append(idx)
            test = set(idx.loc[idx["split"] == "test", "stem"].astype(str))
            zf.extractall(CACHEG, members=[x for x in zf.namelist() if x.rsplit("_", 1)[0] in test])
    pd.concat(index).drop_duplicates("stem").to_csv(CACHEG / "index.csv", index=False)
OBS = Path("/content/obs")
with zipfile.ZipFile(GT / "grand_tunis_obs.zip") as zf:
    zf.extractall(OBS)
print("Grand Tunis :", len(list(CACHEG.glob("*_Y.npy"))), "instants de test ; production :", PROD.name)
'''

CONFIG = r'''import yaml
RUN = NAT / "runs" / "national_1km_v4_Cp_s44"      # modele national livre (decision_v4_national.json)
assert (RUN / "best.pt").exists(), RUN
c = yaml.safe_load(open(CODE / "configs" / "national_1km_colab.yaml"))   # w_continuite + w_beta 0,75
c["name"] = "national"
c["paths"]["run_dir"], c["paths"]["local_dir"] = str(RUN), "/content/run_local_national_service"
c["paths"]["stations_obs"] = str(G / "obs")       # vraies mesures METAR des 12 aeroports
c.setdefault("inference", {})["couche_10m_oi"] = True
yaml.safe_dump(c, open(CODE / "configs" / "national_v4_service.yaml", "w"), sort_keys=False)
OS5 = NAT / "os5"
OS5.mkdir(exist_ok=True)
!pip install -q fastapi uvicorn python-multipart httpx mcp==2.3.0 anthropic 2>&1 | tail -2
# front-end React : webapp/assets/ est ignore par git, il faut le construire avant de servir /app/atom-wind/
!cd /content/PFE && npm ci --no-audit --no-fund 2>&1 | tail -1 && npm run build 2>&1 | grep -E "built in|rror"
assert list((CODE / "webapp" / "assets").glob("atom-*.js")), "front-end non construit"
print("config national_v4_service :", c["inference"])
'''

DSO4 = r'''# B5 : DSO4 sur le GPU du service (processus separe, avant le service) : mediane de 20 passes, p95, memoire max
MESURE = r"""
import json, sys, time, numpy as np, torch
from pcfno3d.config import Config
from pcfno3d.inference import WindFieldModel
from pcfno3d.cache import load_index
out = {}
for nom in ("national_v4_service", "grand_tunis_v2_service_colab"):
    wf = WindFieldModel(Config.load(nom))
    idx = load_index(wf.cfg)
    stem = str(idx.loc[idx["split"] == "test", "stem"].iloc[0])
    S, h, d, ts = wf.surface_from_cache(stem)
    for _ in range(3):
        wf.predict(S, h, d, timestamp=ts, mask_nan=False)
    torch.cuda.reset_peak_memory_stats()
    t = [wf.predict(S, h, d, timestamp=ts, mask_nan=False).inference_s for _ in range(20)]
    out[nom] = {"gpu": torch.cuda.get_device_name(0), "n": 20, "mediane_s": float(np.median(t)),
                "p95_s": float(np.quantile(t, 0.95)), "max_s": float(np.max(t)),
                "memoire_max_go": torch.cuda.max_memory_allocated() / 1e9, "seuil_s": wf.cfg.dso.time_s,
                "conforme": bool(np.median(t) < wf.cfg.dso.time_s)}
    del wf
    torch.cuda.empty_cache()
print("DSO4", json.dumps(out))
"""
Path("/content/dso4_service.py").write_text(MESURE)
r = subprocess.run([sys.executable, "/content/dso4_service.py"], cwd="/content/PFE", capture_output=True, text=True)
ligne = next((l for l in r.stdout.splitlines() if l.startswith("DSO4 ")), None)
if ligne is None:
    print(r.stdout[-3000:], r.stderr[-3000:])
    raise RuntimeError("mesure DSO4 echouee")
dso4 = json.loads(ligne[5:])
(OS5 / "dso4_service.json").write_text(json.dumps(dso4, indent=2))
print(json.dumps(dso4, indent=1))
'''

SERT = r'''from google.colab import userdata
from service.colab import serve_public

os.environ["PCFNO3D_CONFIG_NATIONAL"] = "national_v4_service"    # le national A COTE du Grand Tunis (B1)
TUNNEL_ID = "e552f718-2478-4fe3-82b4-8d33a4b9eb6e"
HOTE = "atom-origin.zelqane.com"
CLE_NOM = "atom-worker"                      # la cle de lecture du Worker de Rayen (empreinte seulement)
CLE_EMPREINTE = "ce7965c54284eb6d1a274119839a71d9785731609d04b17bd08fb6da7d2c30fd"
try:
    identifiants = userdata.get("TUNNEL_CRED_CONTENTS")
except Exception:
    raise RuntimeError("secret Colab TUNNEL_CRED_CONTENTS introuvable ou non autorise : panneau de gauche, icone cle, "
                       "ajoute-le (contenu remis par Rayen) et active « Acces au notebook ».") from None
try:
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
except Exception:
    print("assistant : pas de secret ANTHROPIC_API_KEY (assistant desactive)")
info = serve_public(config="grand_tunis_v2_service_colab", key_name=CLE_NOM, scope="read", key_hash=CLE_EMPREINTE,
                    quota_minute=1200, named_tunnel={"id": TUNNEL_ID, "hostname": HOTE, "credentials": identifiants})
del identifiants
'''

DEMO = r'''# B7 : demonstration de bout en bout (service local, cle de demonstration creee puis revoquee, jamais affichee)
import urllib.request, urllib.parse
from service.auth import ApiKeyStore
req = urllib.request.Request(f"https://{HOTE}/api/health", headers={"User-Agent": "pcfno3d-colab/1.0"})
with urllib.request.urlopen(req, timeout=120) as r:
    sante = json.loads(r.read().decode())
assert sante["status"] == "ok", sante
store = ApiKeyStore(PROD / "api_keys.json")
secret, _ = store.create("demo-os5", scope="read")
def api(chemin, **q):
    u = f"http://127.0.0.1:8000/api/{chemin}?" + urllib.parse.urlencode(q)
    rq = urllib.request.Request(u, headers={"X-API-Key": secret, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(rq, timeout=900) as r:
        return json.loads(r.read().decode())
try:
    doms = api("atom-wind/domaines")
    print(json.dumps(doms, ensure_ascii=False, indent=1)[:1500])
    villes = {"Sfax": (34.7406, 10.7603), "Gafsa": (34.4250, 8.7842), "Tozeur": (33.9197, 8.1335),
              "Gabes": (33.8815, 10.0982), "Tunis": (36.8065, 10.1815)}
    demo = {}
    for v, (la, lo) in villes.items():
        demo[v] = {h: api("atom-wind/point", lat=la, lon=lo, heure=12, hauteur_m=h, j=0) for h in (10, 100)}
        print(v, "|", demo[v][10]["domaine"], "|", demo[v][10]["vitesse_m_s"], "m/s a 10 m :",
              demo[v][10]["source_vent"], "|", demo[v][100]["vitesse_m_s"], "m/s a 100 m :", demo[v][100]["source_vent"])
    (OS5 / "demo_points.json").write_text(json.dumps({"domaines": doms, "points": demo}, indent=2, ensure_ascii=False))
finally:
    store.revoke("demo-os5")
print(f"\nEN LIGNE : https://{HOTE}/app/atom-wind/ — Grand Tunis ET Tunisie entiere (choix par la position).")
# captures de la page (Chromium headless, cle temporaire revoquee) -> NAT/os5/captures, a copier dans docs/demo/
!pip install -q playwright 2>&1 | tail -1; playwright install --with-deps chromium 2>&1 | tail -1
!cd /content/PFE && PYTHONPATH=/content/PFE:/content/PFE/src python scripts/captures_os5.py --cles {PROD}/api_keys.json --sortie {NAT}/os5/captures 2>&1 | tail -12
'''

VIE = r'''try:
    while True:
        tunnel_ok = info["tunnel"].alive
        service_ok = info["process"].poll() is None
        print(time.strftime("%H:%M"), "| tunnel", "ok" if tunnel_ok else "ARRETE",
              "| service", "ok" if service_ok else "ARRETE", flush=True)
        if not (tunnel_ok and service_ok):
            print("journaux :", info["log_path"], "et", info["tunnel"].log_path)
            break
        time.sleep(600)
except KeyboardInterrupt:
    print("surveillance arretee — le service tourne toujours")
'''

ARRET = r'''# Garde-fou : « Tout executer » atteint aussi cette cellule des que la surveillance s'arrete.
ARRETER = False
if ARRETER:
    from service.colab import stop
    stop(info)
    from google.colab import runtime
    runtime.unassign()
else:
    print("service laisse en marche (ARRETER = False)")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    prep = "".join(code[1]["source"])
    ancien = prep.split('COMMIT = "')[1].split('"')[0]
    prep = prep.replace(ancien, a.commit).replace(
        'for f, h in ((', f'for f, h in (("regle_produit_vent10m_v4.json", "{SHA_PRODUIT}"), (', 1)
    cells = [
        cellule("markdown", "# ATOM Wind — Grand Tunis ET Tunisie entière en ligne (OS5)\n\n"
                "Un seul service sert **deux domaines** : le modèle Grand Tunis livré (inchangé) et le **modèle "
                "national C' s44** (six DSO conformes sur le test 2022, w servi β = 0,75), avec la **couche 10 m "
                "corrigée par les aéroports** (`regle_produit_vent10m_v4`). Le domaine est choisi par la position "
                "(Tunis → Grand Tunis ; Sfax, Gafsa, Tozeur, Gabès → national). Le carnet mesure le **DSO4 sur le GPU "
                "du service** puis fait la **démonstration** sur 5 villes.\n\n"
                "**Avant** : secret Colab `TUNNEL_CRED_CONTENTS` (remis par Rayen), « Accès au notebook » activé. "
                "**GPU** : L4 (inférence, ~2 unités/h). Arrêt : `ARRETER = True` dans la dernière cellule.\n"),
        cellule("markdown", "## 1. Drive, code et données nationales"), cellule("code", prep),
        cellule("markdown", "## 2. National : instants de test"), cellule("code", CACHE_NAT),
        cellule("markdown", "## 3. Grand Tunis : données et modèle livré"), cellule("code", GRAND_TUNIS),
        cellule("markdown", "## 4. Configuration du national (couche 10 m)"), cellule("code", CONFIG),
        cellule("markdown", "## 5. DSO4 sur le GPU du service (B5)"), cellule("code", DSO4),
        cellule("markdown", "## 6. Service à deux domaines derrière le tunnel nommé"), cellule("code", SERT),
        cellule("markdown", "## 7. Démonstration de bout en bout (B7)"), cellule("code", DEMO),
        cellule("markdown", "## 8. Garder en vie"), cellule("code", VIE), cellule("code", ARRET),
    ]
    nb["cells"] = cells
    nb.setdefault("metadata", {})["accelerator"] = "GPU"
    nb["metadata"].setdefault("colab", {})["gpuType"] = "L4"
    SORTIE.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(SORTIE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
