"""Carnet Colab : le modele NATIONAL livre (C' s44, w servi 0,75) en ligne dans ATOM Wind (atom.zelqane.com), avec
la couche 10 m « fond + OI des aeroports » (regle_produit_vent10m_v4). Reprend les cellules Drive / code / cache du
carnet de test v4 et les cellules de service du carnet Grand Tunis (meme tunnel nomme `atom`).

    python scripts/make_national_atom_nb.py --commit <sha>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "notebooks" / "current" / "pcfno3d_national_v4_test_colab.ipynb"
SORTIE = REPO / "notebooks" / "current" / "pcfno3d_national_atom_colab.ipynb"
SHA = (REPO / "configs" / "national" / "regle_produit_vent10m_v4.json.sha256").read_text(encoding="ascii").split()[0]


def cellule(kind: str, texte: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": texte.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


CONFIG = r'''import yaml
RUN = NAT / "runs" / "national_1km_v4_Cp_s44"      # modele national livre (decision_v4_national.json)
assert (RUN / "best.pt").exists(), RUN
c = yaml.safe_load(open(CODE / "configs" / "national_1km_colab.yaml"))   # w_continuite + w_beta 0,75
c["name"] = "national_v4_service"
c["paths"]["run_dir"], c["paths"]["local_dir"] = str(RUN), "/content/run_local_national_service"
c["paths"]["stations_obs"] = str(G / "obs")       # vraies mesures METAR des 12 aeroports
c.setdefault("inference", {})["couche_10m_oi"] = True
yaml.safe_dump(c, open(CODE / "configs" / "national_v4_service.yaml", "w"), sort_keys=False)
print("config national_v4_service :", c["inference"])
!pip install -q fastapi uvicorn python-multipart httpx mcp==2.3.0 anthropic 2>&1 | tail -2
'''

SERT = r'''from google.colab import userdata
from service.colab import serve_public

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
    print("assistant : cle presente")
except Exception:
    print("assistant : pas de secret ANTHROPIC_API_KEY (assistant desactive)")
info = serve_public(config="national_v4_service", key_name=CLE_NOM, scope="read", key_hash=CLE_EMPREINTE,
                    quota_minute=1200, named_tunnel={"id": TUNNEL_ID, "hostname": HOTE, "credentials": identifiants})
del identifiants
'''

VERIFIE = r'''import urllib.request
req = urllib.request.Request(f"https://{HOTE}/api/health", headers={"User-Agent": "pcfno3d-colab/1.0"})
with urllib.request.urlopen(req, timeout=120) as r:
    sante = json.loads(r.read().decode())
print(json.dumps(sante, ensure_ascii=False, indent=1))
assert sante["status"] == "ok", "le service ne repond pas correctement par le tunnel"
print(f"\nEN LIGNE : https://{HOTE}/app/atom-wind/ — modele NATIONAL (C' s44), couche 10 m avec les aeroports.")
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
    _garde, prep, cache = ("".join(c["source"]) for c in code[:3])
    ancien = prep.split('COMMIT = "')[1].split('"')[0]
    prep = prep.replace(ancien, a.commit).replace(
        'for f, h in ((', f'for f, h in (("regle_produit_vent10m_v4.json", "{SHA}"), (', 1)
    cache = cache.replace('lancer("controle", "python -u -m pcfno3d -c national_1km_colab check")',
                          "# controle de la verite deja fait par les carnets d'entrainement")
    cells = [
        cellule("markdown", "# PC-FNO3D NATIONAL v4 — ATOM Wind en ligne (atom.zelqane.com)\n\n"
                "Sert le **modèle national livré** (C' graine 44, `decision_v4_national.json` : six DSO conformes "
                "sur le test 2022, −21 % d'erreur face à l'ERA5 de 0 à 1 km) à l'application ATOM Wind, avec la "
                "**couche de vent à 10 m** qui intègre les mesures des 12 aéroports (`pcfno3d.vent10m_oi`, "
                f"règle `regle_produit_vent10m_v4.json`, `{SHA[:12]}…` : −3,7 % face à l'ERA5 sur 2023–2024).\n\n"
                "Même adresse que le Grand Tunis (tunnel nommé `atom`) : pendant que ce carnet tourne, "
                "atom.zelqane.com sert la Tunisie entière. Instants servis : les 180 instants du test 2022 "
                "(cache) et le temps réel (GFS + METAR).\n\n"
                "**Avant** : secret Colab `TUNNEL_CRED_CONTENTS` (remis par Rayen), avec « Accès au notebook ». "
                "**GPU** : L4 demandé (inférence seulement, ~2 unités/h) ; en cas de manque de mémoire : A100.\n"),
        cellule("markdown", "## 1. Drive, code et données"), cellule("code", prep),
        cellule("markdown", "## 2. Cache v3 (instants du test 2022)"), cellule("code", cache),
        cellule("markdown", "## 3. Configuration de service (modèle national + couche 10 m)"), cellule("code", CONFIG),
        cellule("markdown", "## 4. Démarrer le service derrière le tunnel nommé"), cellule("code", SERT),
        cellule("markdown", "## 5. Vérifier de l'extérieur"), cellule("code", VERIFIE),
        cellule("markdown", "## 6. Garder en vie"), cellule("code", VIE), cellule("code", ARRET),
    ]
    nb["cells"] = cells
    nb.setdefault("metadata", {})["accelerator"] = "GPU"
    nb["metadata"].setdefault("colab", {})["gpuType"] = "L4"
    SORTIE.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(SORTIE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
