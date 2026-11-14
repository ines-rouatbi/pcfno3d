"""Variables d'environnement locales du service, lues dans le fichier `.env` a la racine du depot.

Sert aux secrets du poste (OPENROUTER_API_KEY de l'assistant, cles SMTP...) sans les taper a chaque
lancement. Le fichier est ignore par git (.gitignore) et refuse par scripts/package_for_colab.py ; le
modele a copier est `.env.example`. Une variable deja posee dans l'environnement garde sa valeur : le
fichier ne remplace jamais une valeur explicite. Aucune valeur n'est jamais affichee.

Format : une ligne `NOM=valeur` par variable ; lignes vides et `# commentaires` ignorees ; guillemets
simples ou doubles autour de la valeur retires ; `export NOM=valeur` accepte.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

FICHIER = Path(__file__).resolve().parents[1] / ".env"
_LIGNE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def charger(chemin: Path | None = None) -> list[str]:
    """Pose les variables du fichier absentes de l'environnement ; rend leurs NOMS (jamais les valeurs)."""
    chemin = chemin or Path(os.environ.get("PCFNO3D_ENV_FILE") or FICHIER)
    if not chemin.is_file():
        return []
    poses = []
    for ligne in chemin.read_text(encoding="utf-8-sig").splitlines():
        if not ligne.strip() or ligne.lstrip().startswith("#"):
            continue
        m = _LIGNE.match(ligne)
        if not m:
            continue
        nom, valeur = m.group(1), m.group(2)
        if len(valeur) >= 2 and valeur[0] == valeur[-1] and valeur[0] in "\"'":
            valeur = valeur[1:-1]
        elif " #" in valeur:
            valeur = valeur.split(" #", 1)[0].rstrip()
        if valeur and nom not in os.environ:
            os.environ[nom] = valeur
            poses.append(nom)
    return poses
