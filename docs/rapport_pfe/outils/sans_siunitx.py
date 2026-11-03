"""Retire siunitx (version incompatible avec certains noyaux LaTeX) : unités écrites en texte."""
import pathlib
import re

RACINE = pathlib.Path(__file__).resolve().parents[1]
for p in [RACINE / "main.tex", *(RACINE / "chapitres").glob("*.tex"), *(RACINE / "annexes").glob("*.tex")]:
    s = p.read_text(encoding="utf-8")
    t = re.sub(r"^\\usepackage\{siunitx\}\r?\n", "", s, flags=re.M)
    t = re.sub(r"^\\sisetup\{.*\}\r?\n", "", t, flags=re.M)
    t = t.replace(r"\si{m/s}", "m/s").replace(r"\si{s}", "s")
    if t != s:
        p.write_text(t, encoding="utf-8")
        print("modifié :", p.name)
reste = [p.name for p in RACINE.rglob("*.tex") if re.search(r"\\si\{|siunitx|sisetup", p.read_text(encoding="utf-8"))]
print("restes :", reste)
