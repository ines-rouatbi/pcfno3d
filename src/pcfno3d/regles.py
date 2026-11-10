"""Regles ecrites, datees et hachees AVANT le calcul (discipline de la feuille de route nationale) :
configs/national/regle_*.json + .sha256. Un script verifie l'empreinte avant toute lecture des
donnees ; une regle modifiee depuis son hachage est refusee."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def empreinte(chemin: str | Path) -> str:
    return hashlib.sha256(Path(chemin).read_bytes()).hexdigest()


def verifier(chemin: str | Path) -> tuple[dict, str]:
    """(regle, sha256) ; SystemExit si le fichier .sha256 manque ou ne correspond plus."""
    chemin = Path(chemin)
    f = chemin.with_name(chemin.name + ".sha256")
    if not f.exists():
        raise SystemExit(f"{f} absent : la regle doit etre hachee avant le calcul")
    attendu = f.read_text(encoding="ascii").split()[0]
    h = empreinte(chemin)
    if h != attendu:
        raise SystemExit(f"regle {chemin.name} modifiee depuis son hachage : {h} != {attendu}")
    return json.loads(chemin.read_text(encoding="utf-8")), h


def hacher(chemin: str | Path) -> str:
    """Ecrit <regle>.sha256 (format sha256sum) ; refuse si une empreinte DIFFERENTE existe deja."""
    chemin = Path(chemin)
    f = chemin.with_name(chemin.name + ".sha256")
    h = empreinte(chemin)
    if f.exists() and f.read_text(encoding="ascii").split()[0] != h:
        raise SystemExit(f"{f} existe avec une autre empreinte : une regle hachee ne se reecrit pas")
    f.write_text(f"{h}  {chemin.name}\n", encoding="ascii")
    return h


def sortie_neuve(dossier: str | Path, temoin: str) -> Path:
    """Dossier de sortie ; SystemExit si `temoin` y existe deja (jamais d'ecrasement)."""
    d = Path(dossier)
    if (d / temoin).exists():
        raise SystemExit(f"{d / temoin} existe deja : rien n'est recalcule ni ecrase")
    d.mkdir(parents=True, exist_ok=True)
    return d
