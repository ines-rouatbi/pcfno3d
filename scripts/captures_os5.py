"""B7 OS5 : captures d'ecran de la page ATOM Wind servie par le service a deux domaines (Colab).

Chromium headless (Playwright) charge la page du service LOCAL (http://127.0.0.1:8000), avec une cle de
lecture temporaire creee ici, posee dans le localStorage du navigateur headless, jamais affichee ni ecrite,
puis revoquee a la fin. Captures : vue nationale « Mon vent » pour Sfax, Gafsa, Tozeur, Gabes ; Tunis sur le
domaine Grand Tunis ; carte 3D nationale ; onglet « Reseau ANEO » (plan OS4) en session de demonstration admin
(comptes de demonstration locaux de la page). Les PNG vont dans --sortie (le Drive), a copier dans docs/demo/.

    python scripts/captures_os5.py --cles <run_dir>/api_keys.json --sortie <dossier>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

URL = "http://127.0.0.1:8000/app/atom-wind/"
VILLES_NATIONAL = ("Sfax", "Gafsa", "Tozeur", "Gab")       # « Gabes » / « Gabès » selon stations.json
ADMIN = {"id": "u1", "name": "Admin ATOM Wind", "email": "admin@demo.atomwind.tn", "role": "admin",
         "status": "actif", "demo": True, "created": "2026-10-01"}


def attendre_donnees(page, delai_s: float) -> str:
    """Attend que l'ecran de chargement (#aw-chargement, retire par lancerPage) ait disparu, au plus delai_s."""
    fin = time.time() + delai_s
    while time.time() < fin:
        if page.evaluate("() => !!document.body && !document.getElementById('aw-chargement')"):
            time.sleep(10)                         # rendu de la carte et du bandeau
            return page.evaluate("() => document.body.innerText")
        time.sleep(5)
    print("ATTENTION : page encore en chargement apres", delai_s, "s", flush=True)
    return page.evaluate("() => document.body ? document.body.innerText : ''")


def choisir_ville(page, morceau: str) -> str | None:
    trouve = page.evaluate(
        """(m) => { const s = document.getElementById('town'); if(!s) return null;
                    const o = [...s.options].find(o => o.text.includes(m)); return o ? [o.value, o.text] : null; }""",
        morceau)
    if not trouve:
        return None
    page.locator("#town").scroll_into_view_if_needed()
    page.select_option("#town", value=trouve[0])          # vrai choix utilisateur (input + change)
    return trouve[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cles", required=True, help="api_keys.json du service (run_dir du domaine principal)")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--delai", type=float, default=900, help="attente maximale des donnees par vue (s)")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    from service.auth import ApiKeyStore

    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    store = ApiKeyStore(Path(a.cles))
    secret, _ = store.create("captures-os5", scope="read")
    # onglet « Reseau ANEO » : /atom-wind/reseau-aneo?detail=true demande la portee admin (cle temporaire, revoquee)
    secret_admin, _ = store.create("captures-os5-admin", scope="admin")
    bilan = {}
    try:
        with sync_playwright() as p:
            nav = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--enable-webgl",
                                          "--ignore-gpu-blocklist"])
            ctx = nav.new_context(viewport={"width": 1440, "height": 900}, locale="fr-FR")
            ctx.add_init_script(
                "try { localStorage.setItem('pcfno3d.apikey', %s); localStorage.setItem('aw-langue', 'fr'); "
                "localStorage.setItem('aw-tour', '1'); } catch(e) {}"
                % json.dumps(secret))
            page = ctx.new_page()
            page.set_default_timeout(int(a.delai * 1000))

            def capture(nom: str) -> None:
                f = sortie / f"{nom}.png"
                page.screenshot(path=str(f))
                bilan[nom] = {"fichier": f.name, "url": page.url}
                print("capture :", f.name, flush=True)

            # 1. national : Mon vent pour chaque ville
            page.goto(URL + "?domaine=national#monvent")
            texte = attendre_donnees(page, a.delai)
            # liste des lieux du domaine national (aeroports) : prete quand elle contient Sfax
            page.wait_for_function("() => [...(document.getElementById('town')||{options:[]}).options].some(o => o.text.includes('Sfax'))", timeout=120000)
            time.sleep(10)
            bilan["bandeau_national"] = next((l for l in texte.splitlines() if "PC-FNO3D" in l), "")[:400]
            for v in VILLES_NATIONAL:
                choisie = choisir_ville(page, v)
                time.sleep(8)
                capture(f"national_monvent_{v.lower()}")
                bilan[f"national_monvent_{v.lower()}"]["lieu"] = choisie
            # 2. national : carte 3D
            page.goto(URL + "?domaine=national#carte3d")
            attendre_donnees(page, a.delai)
            # relief 3D (WebGL logiciel) : attendre la fin de « Chargement du terrain », au plus 5 min
            fin = time.time() + 300
            while time.time() < fin and "Chargement du terrain" in page.evaluate("() => document.body.innerText"):
                time.sleep(10)
            time.sleep(15)
            capture("national_carte3d")
            # 3. Grand Tunis : Mon vent a Tunis
            page.goto(URL + "#monvent")
            texte = attendre_donnees(page, a.delai)
            bilan["bandeau_grand_tunis"] = next((l for l in texte.splitlines() if "PC-FNO3D" in l), "")[:400]
            choisie = choisir_ville(page, "Tunis")
            time.sleep(8)
            capture("grand_tunis_monvent_tunis")
            bilan["grand_tunis_monvent_tunis"]["lieu"] = choisie
            # 4. national : onglet Reseau ANEO (plan OS4), session de demonstration admin de la page
            ctx2 = nav.new_context(viewport={"width": 1440, "height": 900}, locale="fr-FR")
            ctx2.add_init_script(
                "try { localStorage.setItem('pcfno3d.apikey', %s); localStorage.setItem('aw-langue', 'fr'); "
                "localStorage.setItem('aw-tour', '1'); localStorage.setItem('aw-users', %s); "
                "localStorage.setItem('aw-session', %s); } catch(e) {}"
                % (json.dumps(secret_admin), json.dumps(json.dumps([ADMIN])), json.dumps(json.dumps("u1"))))
            page = ctx2.new_page()
            page.set_default_timeout(int(a.delai * 1000))
            page.goto(URL + "?domaine=national#reseau")
            attendre_donnees(page, a.delai)
            time.sleep(20)
            capture("national_reseau_aneo")
            nav.close()
    finally:
        store.revoke("captures-os5")
        store.revoke("captures-os5-admin")
    (sortie / "captures.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(bilan, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
