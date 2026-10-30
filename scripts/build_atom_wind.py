"""Construit le front-end ATOM Wind branche sur l'API a partir de la maquette d'Ines.

    python scripts/build_atom_wind.py [--source docs/atom_wind/atom_wind_maquette.html]

La maquette (un seul fichier HTML, vent d'exemple calcule dans la page) reste la source : ce
script y pose quelques points d'accroche, sans toucher au reste du design, et ecrit
webapp/atom-wind/index.html, servi par le service a /app/atom-wind/. La couche de donnees est
webapp/atom-wind/atom-data.js (API /api/atom-wind/...). Une accroche introuvable (maquette
modifiee) arrete le script : rien n'est ecrit a moitie.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ACCROCHES = [
    # 1. le script de la page attend les donnees (atom-data.js le lance)
    ('<script>\n(() => {\n"use strict";\n// ---------------- géographie',
     '<script src="atom-data.js"></script>\n<script type="text/plain" id="atom-main">\n(() => {\n'
     '"use strict";\n// ---------------- géographie'),
    # 2. relief, eau et rugosite reels, avant le calcul de la distance a la cote
    ('\n// distance signée à la côte marine',
     '\nif(window.ATOM) ATOM.appliquerGrille({NX, NY, G, elev, wat, z0, LAKES}); // relief réel (API)'
     '\n// distance signée à la côte marine'),
    # 3. le vent : celui du modele quand il est charge, sinon la journee d'exemple
    ('function wind(x, y, z, h){\n',
     'function wind(x, y, z, h){\n  if(window.ATOM && ATOM.pret()) return ATOM.vent(x, y, z, h);\n'),
    # 4. le reseau ANEO prevu : le plan OS6 quand il existe
    ('const ANEO = (() => {',
     'const ANEO = (window.ATOM && ATOM.planXY(xy)) || (() => {'),
    # 5. fiabilite du reseau actuel : sigma calibre du modele (M2, DSO6) quand il est servi
    ("function sigma(x,y,mode=relMode){ let d=1e9, who=null; for(const s of STATIONS){ const dd="
     "Math.hypot(x-s.p[0],y-s.p[1]); if(dd<d){d=dd; who=s.n;} }",
     "function sigma(x,y,mode=relMode){ let d=1e9, who=null; for(const s of STATIONS){ const dd="
     "Math.hypot(x-s.p[0],y-s.p[1]); if(dd<d){d=dd; who=s.n;} }\n"
     "  if(mode===\"reel\" && window.ATOM && ATOM.aSigma()){ const sg=ATOM.sigma(x,y,ACTS[state.act].z,"
     "state.hour); if(sg!==null && isFinite(sg)) return {s:sg, d, who}; }"),
    # 6. bandeau : source des donnees et choix de la journee
    ("<span>Démonstration : journée d'exemple</span>", "<span id=\"aw-fresh\">Démonstration : journée d'exemple</span>"),
    ("<div class=\"proto\">Prototype : <b>journée d'exemple d'été</b>, calculée sur la zone réelle du "
     "Grand Tunis. Ce ne sont pas des mesures en direct. ",
     "<div class=\"proto\"><span id=\"aw-source\">Prototype : <b>journée d'exemple d'été</b>, calculée "
     "sur la zone réelle du Grand Tunis. Ce ne sont pas des mesures en direct.</span> "
     "<select id=\"aw-jour\" hidden aria-label=\"Journée\" style=\"font-size:13px\"></select> "),
    # 7. changement de journee : caches vides, cartes et vues redessinees
    ('\nrenderAlerts(); setHour(15); renderRel();\n',
     '\nif(window.ATOM) ATOM.rafraichir = () => { for(const k in DSTAT) delete DSTAT[k]; '
     'MAPS.forEach(m=>m.redraw()); setHour(state.hour); renderRel(); }; if(window.ATOM) ATOM.setHour = setHour;'
     '\nrenderAlerts(); setHour(15); renderRel();\n'),
    # 8. nombre de capteurs ANEO : celui du plan OS6 (ou de la disposition d'exemple)
    ('["Réseau ANEO prévu","36 capteurs",', '["Réseau ANEO prévu",`${ANEO.length} capteurs`,'),
    ("(36 capteurs, points orange)", "(${ANEO.length} capteurs, points orange)"),
    ("les 36 emplacements proposés", "les ${ANEO.length} emplacements proposés"),
    # 9. vue Fiabilite : d'ou viennent la fiabilite et le reseau ANEO affiches
    ('document.getElementById("relstats").innerHTML = relMode==="reel"',
     'document.getElementById("relstats").innerHTML = (window.ATOM ? ATOM.noteFiabilite() : "") + (relMode==="reel"'),
    ("Zones d'eau exclues.</p>`;", "Zones d'eau exclues.</p>`);"),
    # 10. espace agriculteur (cahier v12, OS8) : J0, J+1, J+2 du temps reel ; temperature, humidite et
    #     rayonnement de GFS (grandeurs que le modele ne calcule pas, signalees par la page)
    ("function windD(x,y,z,h,d){ const [u,v]=wind(x,y,z,h), m=DAYMOD[d]",
     "function windD(x,y,z,h,d){ if(window.ATOM && ATOM.pretJour(d)) return ATOM.ventJour(d,x,y,z,h); "
     "const [u,v]=wind(x,y,z,h), m=DAYMOD[d]"),
    ("const Tair = (h,x,y,d=0) => ",
     "const Tair = (h,x,y,d=0) => (window.ATOM && ATOM.aT(d)) ? ATOM.tJour(d,x,y,h) : "),
    ("const RHair = (h,x,y,d=0) => ",
     "const RHair = (h,x,y,d=0) => (window.ATOM && ATOM.aT(d)) ? ATOM.hrJour(d,x,y,h) : "),
    ("Rs=.72*Ra", "Rs=(window.ATOM && ATOM.rsJour(d)!==null) ? ATOM.rsJour(d) : .72*Ra"),
    ("const J=200+d", "const J=(window.ATOM && ATOM.doyJour(d)) || 200+d"),
    # 11. prevision probabiliste (E19) : probabilite de pouvoir traiter sous le titre de la carte
    ('<div class="card"><h3>Quand traiter ${jour} ?</h3>',
     '<div class="card"><h3>Quand traiter ${jour} ?</h3>${window.ATOM && ATOM.htmlProba ? ATOM.htmlProba(d, x, y) : ""}'),
    # 12. espace eolien : source des jours (temps reel / prevision) et atlas annuel (E8, E15)
    ("<p class=\"src\" style=\"margin:8px 0 0\">Prototype : aujourd'hui est une journée d'exemple, demain et "
     "après-demain en sont des variantes simulées. En service : prévision GFS affinée par ATOM Wind, à 24 et 48 h.</p>",
     "<p class=\"src\" style=\"margin:8px 0 0\" id=\"aw-eol-src\">Prototype : aujourd'hui est une journée d'exemple, "
     "demain et après-demain en sont des variantes simulées. En service : prévision GFS affinée par ATOM Wind, à 24 et 48 h.</p>"),
    ("</section>\n\n<!-- ADMINISTRATION -->",
     "  <div class=\"card\" id=\"aw-atlas\" hidden></div>\n</section>\n\n<!-- ADMINISTRATION -->"),
]


def construire(source: str) -> str:
    html = source.replace("\r\n", "\n")
    for ancien, nouveau in ACCROCHES:
        n = html.count(ancien)
        if n != 1:
            raise SystemExit(f"accroche trouvee {n} fois (1 attendue) : {ancien[:70]!r}")
        html = html.replace(ancien, nouveau)
    if not html.lstrip().lower().startswith("<!doctype"):
        # la maquette commence a <title> : squelette standard (mode standard, UTF-8, mobile)
        squelette = ['<!doctype html>', '<html lang="fr">', '<meta charset="utf-8">',
                     '<meta name="viewport" content="width=device-width, initial-scale=1">']
        html = "\n".join(squelette) + "\n" + html
    return html


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(REPO / "docs" / "atom_wind" / "atom_wind_maquette.html"))
    ap.add_argument("--sortie", default=str(REPO / "webapp" / "atom-wind" / "index.html"))
    args = ap.parse_args()
    html = construire(Path(args.source).read_text(encoding="utf-8"))
    Path(args.sortie).write_bytes(html.encode("utf-8"))
    print(f"{args.sortie} ({len(html) // 1024} Ko)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
