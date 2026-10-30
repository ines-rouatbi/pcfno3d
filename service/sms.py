"""Alertes par SMS pour les agriculteurs (cahier v12, E12) : un message court, en francais ou en arabe,
pour la parcelle de chaque abonne et le lendemain : creneaux de traitement, vent fort, vent chaud.

Fournisseurs :
  - « journal » (defaut) : les messages sont ecrits dans run_dir/sms/journal.jsonl, rien n'est envoye ;
  - « twilio » : envoi reel par l'API REST de Twilio, avec TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN et
    TWILIO_FROM en variables d'environnement du service (jamais dans le depot). PCFNO3D_SMS=twilio.

Les abonnes (numero E.164, parcelle, langue) sont des donnees personnelles : fichier run_dir/sms/
abonnes.json, routes reservees a la portee administrateur, numeros masques dans les reponses.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

E164 = re.compile(r"^\+[1-9]\d{7,14}$")
_VERROU = threading.Lock()

TEXTES = {
    "fr": {"titre": "ATOM Wind {lieu}, demain :", "traiter": "traiter {c}.", "pas_traiter": "pas de creneau pour traiter.",
           "fort": "Vent fort vers {h} h ({v} km/h).", "chaud": "Vent chaud vers {h} h, irriguez tot.",
           "calme": "Pas d'alerte de vent."},
    "ar": {"titre": "ATOM Wind {lieu}، غدًا:", "traiter": "الرش ممكن {c}.", "pas_traiter": "لا وقت مناسب للرش.",
           "fort": "رياح قوية حوالي {h} س ({v} كم/س).", "chaud": "رياح حارة حوالي {h} س، اسقِ باكرًا.",
           "calme": "لا تنبيه رياح."},
}
SEUIL_FORT_KMH = 40.0


def masquer(numero: str) -> str:
    return numero[:4] + "*" * max(len(numero) - 7, 0) + numero[-3:]


class Abonnes:
    def __init__(self, dossier: Path) -> None:
        self.chemin = Path(dossier) / "abonnes.json"

    def lire(self) -> list[dict]:
        return json.loads(self.chemin.read_text(encoding="utf-8")) if self.chemin.exists() else []

    def _ecrire(self, liste: list[dict]) -> None:
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.chemin.with_suffix(".tmp")
        tmp.write_text(json.dumps(liste, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.chemin)

    def ajouter(self, numero: str, langue: str = "fr", lieu: str | None = None,
                lat: float | None = None, lon: float | None = None) -> dict:
        if not E164.match(numero):
            raise ValueError("numero au format international E.164 (ex. +21612345678)")
        if langue not in TEXTES:
            raise ValueError("langue : fr ou ar")
        if not lieu and (lat is None or lon is None):
            raise ValueError("donner un lieu ou lat / lon")
        with _VERROU:
            liste = [a for a in self.lire() if a["numero"] != numero]
            a = {"numero": numero, "langue": langue, "lieu": lieu, "lat": lat, "lon": lon,
                 "depuis": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            liste.append(a)
            self._ecrire(liste)
        return a

    def retirer(self, numero: str) -> bool:
        with _VERROU:
            liste = self.lire()
            reste = [a for a in liste if a["numero"] != numero]
            self._ecrire(reste)
        return len(reste) < len(liste)


def message(wf, abonne: dict, jour: int = 1) -> str:
    """Le SMS du jour `jour` pour un abonne (court : un segment quand c'est possible)."""
    from service import atom_outils as ao
    t = TEXTES[abonne.get("langue", "fr")]
    # les coordonnees priment ; le lieu sert alors d'etiquette dans le message
    if abonne.get("lat") is not None and abonne.get("lon") is not None:
        loc = {"lieu": None, "lat": abonne["lat"], "lon": abonne["lon"]}
    else:
        loc = {"lieu": abonne.get("lieu"), "lat": None, "lon": None}
    d = ao.decision_agricole(wf, jour, **loc)
    nom = (abonne.get("lieu") or "parcelle").split(" (")[0]
    morceaux = [t["titre"].format(lieu=nom)]
    morceaux.append(t["traiter"].format(c=", ".join(d["creneaux_traitement"]).replace(" h - ", "-").replace(" h", "h"))
                    if d["creneaux_traitement"] else t["pas_traiter"])
    fort = max(d["heures"], key=lambda r: r["vent_10m_km_h"])
    alerte = False
    if fort["vent_10m_km_h"] >= SEUIL_FORT_KMH:
        morceaux.append(t["fort"].format(h=fort["heure"], v=round(fort["vent_10m_km_h"])))
        alerte = True
    x, y, _ = ao.position(wf, **loc)
    p = ao.jour(wf, jour)
    for h in range(10, 19):
        T = ao._surface(wf, p, "t2m", x, y, float(h))
        HR = ao._surface(wf, p, "hr2m", x, y, float(h))
        k = d["heures"][h]["vent_10m_km_h"]
        if T is not None and HR is not None and T >= 35 and HR <= 30 and k >= 20:
            morceaux.append(t["chaud"].format(h=h))
            alerte = True
            break
    if not alerte:
        morceaux.append(t["calme"])
    return " ".join(morceaux)


class FournisseurJournal:
    nom = "journal"

    def __init__(self, dossier: Path) -> None:
        self.chemin = Path(dossier) / "journal.jsonl"

    def envoyer(self, numero: str, texte: str) -> dict:
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        with open(self.chemin, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "a": masquer(numero),
                                "texte": texte}, ensure_ascii=False) + "\n")
        return {"envoye": False, "mode": "journal"}


class FournisseurTwilio:
    nom = "twilio"

    def __init__(self, ouvrir=None, expediteur: str | None = None) -> None:
        self.sid = os.environ["TWILIO_ACCOUNT_SID"]
        self.jeton = os.environ["TWILIO_AUTH_TOKEN"]
        self.expediteur = expediteur or os.environ["TWILIO_FROM"]
        self.ouvrir = ouvrir or urllib.request.urlopen

    def envoyer(self, numero: str, texte: str) -> dict:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json"
        corps = urllib.parse.urlencode({"To": numero, "From": self.expediteur, "Body": texte}).encode()
        auth = base64.b64encode(f"{self.sid}:{self.jeton}".encode()).decode()
        req = urllib.request.Request(url, data=corps, method="POST",
                                     headers={"Authorization": f"Basic {auth}",
                                              "Content-Type": "application/x-www-form-urlencoded"})
        with self.ouvrir(req, timeout=20) as r:
            d = json.loads(r.read())
        return {"envoye": True, "mode": "twilio", "statut": d.get("status"), "sid": d.get("sid")}


def fournisseur(dossier: Path):
    if os.environ.get("PCFNO3D_SMS", "journal") == "twilio":
        return FournisseurTwilio()
    return FournisseurJournal(dossier)


def envoyer_tous(wf, dossier: Path, jour: int = 1, simulation: bool = False, fourn=None) -> dict:
    """Un SMS par abonne ; un echec n'arrete pas les autres."""
    ab = Abonnes(dossier).lire()
    fourn = fourn or (FournisseurJournal(dossier) if simulation else fournisseur(dossier))
    resultats = []
    for a in ab:
        try:
            texte = message(wf, a, jour)
            r = fourn.envoyer(a["numero"], texte)
            resultats.append({"a": masquer(a["numero"]), "texte": texte, **r})
        except Exception as exc:                                     # noqa: BLE001
            resultats.append({"a": masquer(a["numero"]), "erreur": str(exc)[:200]})
    return {"fournisseur": fourn.nom, "abonnes": len(ab), "resultats": resultats}
