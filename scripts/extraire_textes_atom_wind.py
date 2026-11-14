"""Extrait les textes francais de la page ATOM Wind construite (webapp/atom-wind/index.html) pour la
traduction arabe / anglaise (cahier v12, E17 et TB7).

    python scripts/extraire_textes_atom_wind.py

Ecrit webapp/atom-wind/i18n/textes_fr.json : la liste triee des textes a traduire. Trois sources :
  - les noeuds de texte du HTML (hors script et style) ;
  - les attributs aria-label, title et placeholder ;
  - les litteraux du script de la page, lus par un petit lecteur JavaScript (chaines, gabarits et leurs
    expressions imbriquees ; commentaires et expressions regulieres sautes) : un gabarit `...${expr}...`
    est coupe aux balises HTML et chaque morceau garde ses expressions sous la forme {0}, {1}...
    (atom-i18n.js reconnait ces motifs une fois les valeurs inserees, et traduit aussi les valeurs).
Les dictionnaires i18n/ar.json et i18n/en.json associent chaque texte a sa traduction ; un texte absent
reste en francais. Relancer apres une modification de la maquette, puis completer les dictionnaires
(--manquants LANGUE liste ce qui manque).
"""
from __future__ import annotations

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOSSIER = REPO / "frontend" / "public" / "atom-wind" / "i18n"
MOT = re.compile(r"[A-Za-zÀ-ÿ]{2,}")
BALISE = re.compile(r"<[^>]*>")
GUILLEMETS = "\"'"
MOTS_FR = re.compile(r"[À-ÿ]|\s(le|la|les|de|du|des|et|un|une|en|au|aux|à|pour|sur|par)\s", re.I)


class _Html(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.dans_script = 0
        self.textes: set[str] = set()

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.dans_script += 1
        for k, v in attrs:
            if k in ("aria-label", "title", "placeholder") and v and MOT.search(v):
                self.textes.add(" ".join(v.split()))

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.dans_script = max(0, self.dans_script - 1)

    def handle_data(self, d):
        t = " ".join(d.split())
        if not self.dans_script and MOT.search(t):
            self.textes.add(t)


# ---------------------------------------------------------------------- lecteur de litteraux JS
def _fin_chaine(src: str, i: int, q: str) -> int:
    """Indice du guillemet fermant (i : premier caractere apres l'ouvrant)."""
    while i < len(src) and src[i] != q:
        if src[i] == "\\":
            i += 1
        elif src[i] == "\n":
            return i
        i += 1
    return i


def _fin_expression(src: str, i: int) -> int:
    """Accolade fermante d'un ${...} (i : apres « ${ »), chaines et gabarits imbriques compris."""
    prof = 1
    while i < len(src):
        c = src[i]
        if c in GUILLEMETS:
            i = _fin_chaine(src, i + 1, c)
        elif c == "`":
            i = _fin_gabarit(src, i + 1)
        elif c == "{":
            prof += 1
        elif c == "}":
            prof -= 1
            if prof == 0:
                return i
        i += 1
    return i


def _fin_gabarit(src: str, i: int) -> int:
    """Accent grave fermant d'un gabarit (i : apres l'ouvrant)."""
    while i < len(src):
        if src[i] == "\\":
            i += 2
            continue
        if src[i] == "`":
            return i
        if src.startswith("${", i):
            i = _fin_expression(src, i + 2)
        i += 1
    return i


def litteraux(src: str) -> list[tuple[str, str]]:
    """[(« q » chaine | « t » gabarit, contenu brut)] d'un script ; les expressions des gabarits sont lues
    a leur tour."""
    out: list[tuple[str, str]] = []
    i, n, prec = 0, len(src), ""
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in GUILLEMETS:
            j = _fin_chaine(src, i + 1, c)
            out.append(("q", src[i + 1:j]))
            i, prec = j + 1, "x"
            continue
        if c == "`":
            j = _fin_gabarit(src, i + 1)
            corps = src[i + 1:j]
            out.append(("t", corps))
            k = corps.find("${")
            while k >= 0:
                f = _fin_expression(corps, k + 2)
                out.extend(litteraux(corps[k + 2:f]))
                k = corps.find("${", f + 1)
            i, prec = j + 1, "x"
            continue
        if c == "/" and (prec == "" or prec in "(,=:[!&|?{};+-*%<>~^"):
            j, classe = i + 1, False                # expression reguliere : sautee
            while j < n and src[j] != "\n":
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "[":
                    classe = True
                elif src[j] == "]":
                    classe = False
                elif src[j] == "/" and not classe:
                    break
                j += 1
            i, prec = j + 1, "x"
            continue
        if not c.isspace():
            prec = "a" if (c.isalnum() or c in "_$") else c
        i += 1
    return out


# ---------------------------------------------------------------------- morceaux a traduire
def _expressions(s: str) -> str:
    """`${...}` -> {0}, {1}..."""
    out, n, i = [], 0, 0
    while i < len(s):
        if s.startswith("${", i):
            j = _fin_expression(s, i + 2)
            out.append(f"{{{n}}}")
            n += 1
            i = j + 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _morceaux(gabarit: str) -> list[str]:
    s = _expressions(gabarit)
    out = []
    for m in BALISE.split(s):
        t = " ".join(m.split())
        # au moins un vrai mot en dehors des {n}, et pas du code
        mots = re.sub(r"\{\d+\}", "", t)
        if (MOT.search(mots) or re.search(r"(^|\s)à\s", mots)) and not re.search(r"[;=]\s*$|=>|\bfunction\b|\(\)|\.\w+\(", t):
            noms: dict[str, str] = {}               # {n} renumerotes a partir de 0 dans chaque morceau
            out.append(re.sub(r"\{(\d+)\}", lambda m: "{%s}" % noms.setdefault(m[1], str(len(noms))), t))
    return out


EVENEMENTS = {"click", "input", "change", "resize", "moveend", "movestart", "zoomend", "zoomstart", "keydown",
              "pointerdown", "pointerup", "pointermove", "use", "strict", "now", "sel", "dark", "light"}


def _affichable(t: str, classes: set[str]) -> bool:
    """Chaine affichee : des mots, une espace ou un accent ; pas un selecteur, des classes CSS, des noms
    d'evenements ou du code (sauf si elle a l'air d'une phrase francaise)."""
    if not MOT.search(t) or not (" " in t or re.search(r"[À-ÿ]", t)):
        return False
    if re.fullmatch(r"[\w-]+( [\w-]+)*", t) and all(w in classes or w in EVENEMENTS for w in t.split()):
        return False
    code = re.search(r"^[#.\[(]|=>|[{}=]|;\S|\w\(|^\d+(px)? ", t)
    return not code or bool(MOTS_FR.search(f" {t} "))


# couche de donnees, vue 3D, onglet d'administration « Reseau ANEO » (maquette de planification et d'installation),
# fonctions du service branchees dans les vues : textes affiches eux aussi
MODULES = ("atom-data.js", "atom-3d.js", "reseau-aneo-plan.js", "services-atom.js")


def textes_frontend() -> list[str]:
    """Extract source JSX and controller strings; never inspect minified build output."""
    src = REPO / "frontend" / "src"
    markup = (REPO / "frontend/i18n-source.html").read_text(encoding="utf-8")
    modules = tuple((src / "engine" / name).read_text(encoding="utf-8")
                    for name in (*MODULES, "atom-runtime.js"))
    # The existing lexer also extracts page headings and navigation names from TSX.
    modules += tuple((src / name).read_text(encoding="utf-8")
                     for name in ("components.tsx", "main.tsx", "territory-map.ts", "terrain/terrain-viewer.ts"))
    return avec_supplements(extraire(markup, modules))


def extraire(html: str, modules: tuple[str, ...] = ()) -> list[str]:
    p = _Html()
    p.feed(html)
    textes = set(p.textes)
    debut = html.find('id="atom-main"')
    script = html[html.find(">", debut) + 1:html.find("</script>", debut)] if debut >= 0 else ""
    script = "\n".join([script, *modules])
    classes = {c for attr in re.findall(r'class="([^"$]*)"', html) for c in attr.split()}
    classes |= set(re.findall(r"\.([a-z][\w-]*)", html[html.find("<style"):html.find("</style>")]))
    for genre, t in litteraux(script):
        # attributs lus par atom-i18n.js, ecrits dans les gabarits et chaines du script
        for v in re.findall(r'(?:aria-label|title|placeholder)=\\?"([^"\\]*)', t):
            if MOT.search(re.sub(r"\$\{[^}]*\}", "", v)):
                textes.update(_morceaux(v))
        if genre == "t":
            textes.update(_morceaux(t))
        elif _affichable(BALISE.sub(" ", t), classes):
            textes.update(_morceaux(t.replace("\\'", "'").replace('\\"', '"')))
    # les morceaux sans {n} d'abord : les plus nombreux et les plus simples
    return sorted(textes, key=lambda t: ("{0}" in t, t.lower(), t))


def avec_supplements(textes: list[str]) -> list[str]:
    """Ajoute i18n/textes_supplementaires.json : textes releves a l'affichage (ATOM.nonTraduits dans la
    console de la page) que la lecture du code ne trouve pas (mots isoles des tableaux de donnees : lieux,
    cultures, niveaux)."""
    f = DOSSIER / "textes_supplementaires.json"
    if not f.exists():
        return textes
    vus = set(textes)
    plus = [t for t in json.loads(f.read_text(encoding="utf-8")) if t not in vus]
    return sorted(vus | set(plus), key=lambda t: ("{0}" in t, t.lower(), t))


def renumeroter(fr: str, tr: str) -> tuple[str, str]:
    """{n} de la cle renumerotes a partir de 0 dans l'ordre d'apparition, la traduction suivant."""
    noms: dict[str, str] = {}
    for m in re.finditer(r"\{(\d+)\}", fr):
        noms.setdefault(m[1], str(len(noms)))
    f = lambda m: "{%s}" % noms.get(m[1], m[1])          # noqa: E731
    return re.sub(r"\{(\d+)\}", f, fr), re.sub(r"\{(\d+)\}", f, tr)


def normaliser(langue: str) -> int:
    """Renumerote les cles d'un dictionnaire (ecrit en place) ; renvoie le nombre de cles changees."""
    d = DOSSIER / f"{langue}.json"
    dico = json.loads(d.read_text(encoding="utf-8"))
    neuf = dict(renumeroter(k, v) for k, v in dico.items())
    d.write_text(json.dumps(neuf, ensure_ascii=False, indent=1), encoding="utf-8")
    return sum(1 for k in dico if k not in neuf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manquants", metavar="LANGUE", help="ecrire i18n/manquants_LANGUE.json (textes sans traduction)")
    ap.add_argument("--normaliser", metavar="LANGUE", help="renumeroter les {n} des cles de i18n/LANGUE.json")
    a = ap.parse_args()
    if a.normaliser:
        print(f"{a.normaliser} : {normaliser(a.normaliser)} cles renumerotees")
    textes = textes_frontend()
    DOSSIER.mkdir(exist_ok=True)
    sortie = DOSSIER / "textes_fr.json"
    sortie.write_text(json.dumps(textes, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{sortie} : {len(textes)} textes ({sum(map(len, textes))} caracteres)")
    if a.manquants:
        d = DOSSIER / f"{a.manquants}.json"
        dico = json.loads(d.read_text(encoding="utf-8")) if d.exists() else {}
        manque = [t for t in textes if t not in dico]
        (DOSSIER / f"manquants_{a.manquants}.json").write_text(json.dumps(manque, ensure_ascii=False, indent=0),
                                                               encoding="utf-8")
        print(f"{a.manquants} : {len(manque)} textes sans traduction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
