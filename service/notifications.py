"""Notification des alertes importantes (cahier v12, BF27, BF83) : chaque alerte n'est envoyee qu'une fois
(empreinte gardee dans run_dir/notifications/envoyees.json), sur les canaux configures :

  - journal (toujours) : run_dir/notifications/journal.jsonl, destinataires masques ;
  - e-mail : PCFNO3D_SMTP_HOTE, PCFNO3D_SMTP_PORT (587), PCFNO3D_SMTP_UTILISATEUR, PCFNO3D_SMTP_MOT_DE_PASSE,
    PCFNO3D_SMTP_DE et PCFNO3D_NOTIF_EMAILS (liste separee par des virgules) ; STARTTLS ;
  - WhatsApp : par Twilio (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_DE) vers
    PCFNO3D_NOTIF_WHATSAPP (numeros E.164 separes par des virgules).

Les secrets ne viennent QUE de l'environnement du service, jamais du depot, d'une requete ou d'un journal.
Les alertes viennent de service.atom_outils.alertes_service (temps reel du jour et prevision du lendemain).
"""
from __future__ import annotations

import json
import os
import smtplib
import threading
import time
from email.message import EmailMessage
from pathlib import Path

from service.sms import E164, FournisseurTwilio, masquer

NIVEAUX_NOTIFIES = ("important", "critique")
_VERROU = threading.Lock()


def _liste(var: str) -> list[str]:
    return [x.strip() for x in os.environ.get(var, "").split(",") if x.strip()]


def _texte(a: dict) -> str:
    return f"ATOM Wind — {a['titre']}. {a['cause']} {a['recommandation']}".strip()


class CanalJournal:
    nom = "journal"

    def __init__(self, dossier: Path) -> None:
        self.chemin = Path(dossier) / "journal.jsonl"

    def envoyer(self, alerte: dict, destinataires: list[str]) -> dict:
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        with open(self.chemin, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "id": alerte["id"],
                                "niveau": alerte["niveau"], "texte": _texte(alerte),
                                "a": [masquer(d) if d.startswith("+") else d.split("@")[0][:2] + "…@" + d.split("@")[-1]
                                      for d in destinataires]}, ensure_ascii=False) + "\n")
        return {"canal": self.nom, "envoye": False}


class CanalEmail:
    nom = "email"

    def __init__(self, smtp=None) -> None:
        self.hote = os.environ["PCFNO3D_SMTP_HOTE"]
        self.port = int(os.environ.get("PCFNO3D_SMTP_PORT", "587"))
        self.utilisateur = os.environ.get("PCFNO3D_SMTP_UTILISATEUR")
        self.mot_de_passe = os.environ.get("PCFNO3D_SMTP_MOT_DE_PASSE")
        self.de = os.environ["PCFNO3D_SMTP_DE"]
        self.smtp = smtp or smtplib.SMTP

    def envoyer(self, alerte: dict, destinataires: list[str]) -> dict:
        m = EmailMessage()
        m["Subject"], m["From"], m["To"] = f"ATOM Wind : {alerte['titre']}", self.de, ", ".join(destinataires)
        m.set_content(_texte(alerte) + "\n\nMessage automatique du service ATOM Wind (PC-FNO3D).")
        with self.smtp(self.hote, self.port, timeout=20) as s:
            s.starttls()
            if self.utilisateur:
                s.login(self.utilisateur, self.mot_de_passe or "")
            s.send_message(m)
        return {"canal": self.nom, "envoye": True, "destinataires": len(destinataires)}


class CanalWhatsApp:
    nom = "whatsapp"

    def __init__(self, ouvrir=None) -> None:
        self.twilio = FournisseurTwilio(ouvrir=ouvrir, expediteur="whatsapp:" + os.environ["TWILIO_WHATSAPP_DE"])

    def envoyer(self, alerte: dict, destinataires: list[str]) -> dict:
        r = [self.twilio.envoyer("whatsapp:" + d, _texte(alerte)) for d in destinataires]
        return {"canal": self.nom, "envoye": True, "destinataires": len(r)}


def canaux(dossier: Path) -> list[tuple[object, list[str]]]:
    """(canal, destinataires) configures ; le journal est toujours la."""
    out: list[tuple[object, list[str]]] = []
    emails = _liste("PCFNO3D_NOTIF_EMAILS")
    whatsapp = [n for n in _liste("PCFNO3D_NOTIF_WHATSAPP") if E164.match(n)]
    out.append((CanalJournal(dossier), emails + whatsapp))
    if emails and os.environ.get("PCFNO3D_SMTP_HOTE") and os.environ.get("PCFNO3D_SMTP_DE"):
        out.append((CanalEmail(), emails))
    if whatsapp and all(os.environ.get(v) for v in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_DE")):
        out.append((CanalWhatsApp(), whatsapp))
    return out


def notifier(alertes: list[dict], dossier: Path, liste_canaux=None) -> dict:
    """Envoie les alertes importantes pas encore envoyees ; un canal en echec n'arrete pas les autres."""
    dossier = Path(dossier)
    deja_f = dossier / "envoyees.json"
    liste_canaux = liste_canaux if liste_canaux is not None else canaux(dossier)
    rapport = {"canaux": [c.nom for c, _ in liste_canaux], "envoyees": [], "deja": 0, "erreurs": []}
    with _VERROU:
        deja = set(json.loads(deja_f.read_text(encoding="utf-8"))) if deja_f.exists() else set()
        for a in alertes:
            if a["niveau"] not in NIVEAUX_NOTIFIES:
                continue
            if a["id"] in deja:
                rapport["deja"] += 1
                continue
            ok = False
            for canal, dest in liste_canaux:
                try:
                    canal.envoyer(a, dest)
                    ok = True
                except Exception as exc:                          # noqa: BLE001
                    rapport["erreurs"].append({"canal": canal.nom, "id": a["id"], "erreur": str(exc)[:200]})
            if ok:
                deja.add(a["id"])
                rapport["envoyees"].append({"id": a["id"], "titre": a["titre"], "niveau": a["niveau"]})
        dossier.mkdir(parents=True, exist_ok=True)
        tmp = deja_f.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(deja)), encoding="utf-8")
        tmp.replace(deja_f)
    return rapport
