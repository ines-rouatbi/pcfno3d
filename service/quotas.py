"""Quotas par cle d'API : requetes par minute et par jour.

Limites par defaut du service (variables d'environnement, 0 = illimite) :

    PCFNO3D_QUOTA_MINUTE   requetes par minute glissante et par cle (defaut 300)
    PCFNO3D_QUOTA_JOUR     requetes par jour UTC et par cle (defaut 50000)

Une cle peut porter ses propres limites dans le trousseau (`quota_minute`, `quota_jour`,
voir `pcfno3d apikey quota`), qui remplacent celles-ci.

Depassement : `429 Too Many Requests` avec l'en-tete `Retry-After` (secondes). Une
requete refusee n'est pas comptee : un client qui insiste ne prolonge pas son attente.

LIMITE ASSUMEE : les compteurs vivent en memoire, dans le processus. Le service tourne
avec un seul processus uvicorn (cas actuel : poste local, Colab) ; avec N processus,
chaque processus compte de son cote et la limite effective devient N fois plus grande.
Un redemarrage remet aussi les compteurs a zero. Pour plusieurs processus, il faudrait un
compteur partage (Redis), hors du perimetre actuel.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

ENV_MINUTE = "PCFNO3D_QUOTA_MINUTE"
ENV_JOUR = "PCFNO3D_QUOTA_JOUR"
DEFAUT_MINUTE = 300
DEFAUT_JOUR = 50_000
FENETRE_S = 60.0
JOUR_S = 86_400


@dataclass
class Depassement:
    """Une requete refusee : quelle limite, et quand reessayer."""

    limite: str          # "minute" ou "jour"
    quota: int
    retry_after: int     # secondes, >= 1

    def message(self, nom: str) -> str:
        unite = "par minute" if self.limite == "minute" else "par jour (UTC)"
        return (f"quota depasse pour la cle '{nom}' : {self.quota} requetes {unite} ; "
                f"reessayer dans {self.retry_after} s")


def _env_int(nom: str, defaut: int) -> int:
    brut = (os.environ.get(nom) or "").strip()
    if not brut:
        return defaut
    valeur = int(brut)
    if valeur < 0:
        raise ValueError(f"{nom} doit etre >= 0 (0 = illimite)")
    return valeur


class Quotas:
    """Compteurs en memoire, par empreinte de cle. Surs entre fils d'execution."""

    def __init__(self, par_minute: int = DEFAUT_MINUTE, par_jour: int = DEFAUT_JOUR,
                 horloge=time.time) -> None:
        self.par_minute = int(par_minute)
        self.par_jour = int(par_jour)
        self._horloge = horloge
        self._lock = threading.Lock()
        self._minute: dict[str, deque] = {}
        self._jour: dict[str, list[int]] = {}       # empreinte -> [jour UTC, compte]

    @classmethod
    def from_env(cls) -> "Quotas":
        return cls(_env_int(ENV_MINUTE, DEFAUT_MINUTE), _env_int(ENV_JOUR, DEFAUT_JOUR))

    def limites(self, record) -> tuple[int, int]:
        """(par minute, par jour) effectifs pour cette cle ; 0 = illimite."""
        m = record.quota_minute if record.quota_minute is not None else self.par_minute
        j = record.quota_jour if record.quota_jour is not None else self.par_jour
        return int(m), int(j)

    def consommer(self, record) -> Depassement | None:
        """Compte une requete ; renvoie le depassement si elle doit etre refusee."""
        par_minute, par_jour = self.limites(record)
        cle = record.key_hash
        with self._lock:
            now = self._horloge()
            fenetre = self._minute.setdefault(cle, deque())
            while fenetre and fenetre[0] <= now - FENETRE_S:
                fenetre.popleft()
            jour = int(now // JOUR_S)
            compte = self._jour.get(cle)
            if compte is None or compte[0] != jour:
                compte = [jour, 0]
                self._jour[cle] = compte
            if par_jour and compte[1] >= par_jour:
                return Depassement("jour", par_jour,
                                   max(1, math.ceil((jour + 1) * JOUR_S - now)))
            if par_minute and len(fenetre) >= par_minute:
                return Depassement("minute", par_minute,
                                   max(1, math.ceil(fenetre[0] + FENETRE_S - now)))
            if par_minute:                  # illimite : rien a memoriser
                fenetre.append(now)
            compte[1] += 1
        return None

    def public(self) -> dict:
        return {"par_minute": self.par_minute or None, "par_jour": self.par_jour or None,
                "portee": "par cle, en memoire du processus"}
