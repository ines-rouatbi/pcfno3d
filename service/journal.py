"""Journal d'acces JSONL : une ligne par requete authentifiee ou refusee.

Chaque ligne :

    {"ts": "2026-10-14T12:00:00.123Z", "cle": "app-web", "empreinte": "1a2b3c4d",
     "portee": "read", "methode": "GET", "chemin": "/api/point", "params": "x=1&y=2&z=120",
     "statut": 200, "duree_ms": 12.3, "octets": 245, "ip": "127.0.0.1"}

Ce qui n'y entre JAMAIS : le secret d'une cle, ni son hash complet (seulement les 8
premiers caracteres, `empreinte`). Le parametre `?api_key=` est retire des `params`, et
toute valeur qui ressemble a une cle (`pcf_...`) est masquee. Une cle refusee n'est pas
identifiee du tout (`cle` = null) : rien de ce qu'un client a tape de faux n'est garde.

Emplacement : `run_dir/logs/acces.jsonl` par defaut ; `PCFNO3D_ACCESS_LOG` le change (un
chemin relatif est pris dans run_dir ; `off` coupe le journal). Rotation par taille :
au-dela de `PCFNO3D_ACCESS_LOG_MAX_MB` (defaut 10 Mo), le fichier devient `acces.jsonl.1`
et les anciens glissent jusqu'a `.5` ; le plus ancien est supprime.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from .auth import KEY_PREFIX

ENV_CHEMIN = "PCFNO3D_ACCESS_LOG"
ENV_MAX_MO = "PCFNO3D_ACCESS_LOG_MAX_MB"
CHEMIN_DEFAUT = Path("logs") / "acces.jsonl"
MAX_MO_DEFAUT = 10.0
GARDER = 5
PARAMS_SECRETS = {"api_key", "apikey", "key", "token", "access_token"}


def params_sans_secret(query) -> str:
    """Query string sans la cle : `api_key` retire, toute valeur `pcf_...` masquee."""
    garde = []
    for nom, valeur in query.multi_items() if hasattr(query, "multi_items") else query.items():
        if nom.lower() in PARAMS_SECRETS:
            continue
        if KEY_PREFIX in valeur:
            valeur = "***"
        garde.append((nom, valeur))
    return urlencode(garde)


class JournalAcces:
    """Ecrit les lignes, sous verrou ; la rotation se fait avant d'ecrire."""

    def __init__(self, path: str | Path, max_octets: int = int(MAX_MO_DEFAUT * 1024 * 1024),
                 garder: int = GARDER) -> None:
        self.path = Path(path)
        self.max_octets = int(max_octets)
        self.garder = int(garder)
        self._lock = threading.Lock()
        self._signale = False

    @classmethod
    def from_env(cls, run_dir: str | Path) -> "JournalAcces | None":
        brut = (os.environ.get(ENV_CHEMIN) or "").strip()
        if brut.lower() in {"off", "0", "non", "none"}:
            return None
        path = Path(brut) if brut else CHEMIN_DEFAUT
        if not path.is_absolute():
            path = Path(run_dir) / path
        max_mo = float(os.environ.get(ENV_MAX_MO) or MAX_MO_DEFAUT)
        return cls(path, max_octets=int(max_mo * 1024 * 1024))

    def _rotation(self, a_ecrire: int) -> None:
        try:
            taille = self.path.stat().st_size
        except FileNotFoundError:
            return
        if taille == 0 or taille + a_ecrire <= self.max_octets:
            return

        def numero(i: int) -> Path:
            return self.path.with_name(f"{self.path.name}.{i}")

        numero(self.garder).unlink(missing_ok=True)
        for i in range(self.garder - 1, 0, -1):          # .4 -> .5, ..., .1 -> .2
            if numero(i).exists():
                os.replace(numero(i), numero(i + 1))
        os.replace(self.path, numero(1))

    def ecrire(self, **champs) -> None:
        champs = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                  .replace("+00:00", "Z"), **champs}
        ligne = json.dumps(champs, ensure_ascii=False) + "\n"
        donnees = ligne.encode("utf-8")
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._rotation(len(donnees))
                with open(self.path, "ab") as fh:
                    fh.write(donnees)
            except OSError as exc:
                # un disque plein ne doit pas couper le service, mais ca doit se voir
                if not self._signale:
                    self._signale = True
                    print(f"journal d'acces {self.path} : ecriture impossible ({exc})",
                          file=sys.stderr)
