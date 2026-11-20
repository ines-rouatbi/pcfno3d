"""E-mails des comptes (demande d'Ines, 20/11/2026) : lien de verification de l'adresse (24 h) et lien « mot de
passe oublie » (1 h).

  - envoi par le meme serveur SMTP que les alertes : PCFNO3D_SMTP_HOTE, PCFNO3D_SMTP_PORT (587),
    PCFNO3D_SMTP_UTILISATEUR, PCFNO3D_SMTP_MOT_DE_PASSE, PCFNO3D_SMTP_DE ; STARTTLS ;
  - adresse de l'application dans les liens : PCFNO3D_URL_APP (par defaut https://atom.zelqane.com/app/atom-wind/).
    Elle ne vient JAMAIS de la requete (en-tetes Host / Origin) : sinon un tiers pourrait faire envoyer a la
    victime un lien de reinitialisation pointant vers son propre site, et y lire le jeton ;
  - le jeton voyage apres « # » : le navigateur ne l'envoie a aucun serveur et il n'apparait dans aucun journal ;
  - developpement sans SMTP (PCFNO3D_COURRIELS_LOCAUX=1, pose par scripts/dev_service.py) : le message est ecrit
    dans run_dir/courriels_locaux/ au lieu d'etre envoye ; sans SMTP ni ce reglage, rien n'est envoye et
    l'administrateur peut confirmer l'adresse lui-meme depuis la page Administration.

Aucun secret (mot de passe SMTP, jeton) n'est ecrit dans un journal.
"""
from __future__ import annotations

import os
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path

URL_APP_DEFAUT = "https://atom.zelqane.com/app/atom-wind/"


def url_app() -> str:
    u = os.environ.get("PCFNO3D_URL_APP", "").strip() or URL_APP_DEFAUT
    return u if (u.endswith("/") or "?" in u) else u + "/"      # ?domaine=national garde tel quel


def lien(sorte: str, jeton: str) -> str:
    return f"{url_app()}#{sorte}={jeton}"


def smtp_configure() -> bool:
    """Serveur, expediteur et, si un identifiant est donne, son mot de passe (cle SMTP) : un reglage a moitie
    rempli (cle encore vide) compte comme absent."""
    e = os.environ.get
    return bool(e("PCFNO3D_SMTP_HOTE") and e("PCFNO3D_SMTP_DE")
                and (not e("PCFNO3D_SMTP_UTILISATEUR") or e("PCFNO3D_SMTP_MOT_DE_PASSE")))


def envoyer(destinataire: str, sujet: str, texte: str, dossier: Path, smtp=None) -> dict:
    """Envoie un e-mail ; renvoie {"envoye": bool, "canal": "smtp" | "local" | None}. Ne leve jamais."""
    m = EmailMessage()
    m["Subject"], m["To"] = sujet, destinataire
    m["From"] = os.environ.get("PCFNO3D_SMTP_DE", "ATOM Wind <ne-pas-repondre@atom-wind.local>")
    m.set_content(texte + "\n\nMessage automatique du service ATOM Wind. Si vous n'êtes pas à l'origine de cette "
                          "demande, ignorez ce message : rien ne sera changé.")
    if smtp_configure():
        try:
            with (smtp or smtplib.SMTP)(os.environ["PCFNO3D_SMTP_HOTE"],
                                        int(os.environ.get("PCFNO3D_SMTP_PORT", "587")), timeout=20) as s:
                s.starttls()
                if os.environ.get("PCFNO3D_SMTP_UTILISATEUR"):
                    s.login(os.environ["PCFNO3D_SMTP_UTILISATEUR"], os.environ.get("PCFNO3D_SMTP_MOT_DE_PASSE", ""))
                s.send_message(m)
            return {"envoye": True, "canal": "smtp"}
        except (OSError, smtplib.SMTPException) as exc:
            # le motif (jamais le mot de passe) dans le journal du service, pour comprendre un refus du serveur
            import logging
            logging.getLogger("pcfno3d.service").warning("e-mail de compte non envoye : %s", type(exc).__name__)
            if os.environ.get("PCFNO3D_COURRIELS_LOCAUX") != "1":
                return {"envoye": False, "canal": "smtp"}
    if os.environ.get("PCFNO3D_COURRIELS_LOCAUX") == "1":
        d = Path(dossier) / "courriels_locaux"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{time.time_ns()}_{destinataire.replace('@', '_at_')}.eml"
        f.write_bytes(bytes(m))
        return {"envoye": True, "canal": "local"}
    return {"envoye": False, "canal": None}


def verification(destinataire: str, nom: str, jeton: str, dossier: Path, heures: int = 24) -> dict:
    texte = (f"Bonjour{(' ' + nom) if nom else ''},\n\n"
             "Pour activer votre compte ATOM Wind, confirmez votre adresse e-mail en ouvrant ce lien :\n\n"
             f"{lien('verifier-email', jeton)}\n\n"
             f"Ce lien ne sert qu'une fois et expire dans {heures} heures. Vous pourrez ensuite vous connecter avec "
             "votre adresse et votre mot de passe.")
    return envoyer(destinataire, "ATOM Wind : confirmez votre adresse e-mail", texte, dossier)


def reinitialisation(destinataire: str, nom: str, jeton: str, dossier: Path, minutes: int = 60) -> dict:
    texte = (f"Bonjour{(' ' + nom) if nom else ''},\n\n"
             "Un nouveau mot de passe a été demandé pour votre compte ATOM Wind. Pour le choisir, ouvrez ce lien :\n\n"
             f"{lien('nouveau-mot-de-passe', jeton)}\n\n"
             f"Ce lien ne sert qu'une fois et expire dans {minutes} minutes. Votre mot de passe actuel reste valable "
             "tant que vous n'en choisissez pas un nouveau.")
    return envoyer(destinataire, "ATOM Wind : choisir un nouveau mot de passe", texte, dossier)
