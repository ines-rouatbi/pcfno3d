"""Notifications des alertes importantes (BF27, BF83) : une seule fois chacune, journal toujours, e-mail et
WhatsApp simules (aucun reseau, aucun secret reel)."""

from __future__ import annotations

import json

from test_temps_reel import _jeu_reel, cfg_era5, fausse_source, forcage, jeu  # noqa: F401
from test_assistant import wf  # noqa: F401

ALERTE = {"id": "abcd1234", "kind": "vent_fort", "niveau": "important", "titre": "vent fort a 10 m",
          "cible": "domaine", "cause": "maximum 60 km/h.", "recommandation": "Eviter les travaux en hauteur."}


def test_une_seule_fois_et_journal_masque(tmp_path, monkeypatch):
    from service import notifications as n
    monkeypatch.setenv("PCFNO3D_NOTIF_EMAILS", "agri@exemple.tn")
    monkeypatch.setenv("PCFNO3D_NOTIF_WHATSAPP", "+21612345678,pas-un-numero")
    for v in ("PCFNO3D_SMTP_HOTE", "TWILIO_ACCOUNT_SID"):
        monkeypatch.delenv(v, raising=False)
    r = n.notifier([ALERTE, {**ALERTE, "id": "x", "niveau": "information"}], tmp_path)
    assert r["canaux"] == ["journal"] and [a["id"] for a in r["envoyees"]] == ["abcd1234"]
    assert n.notifier([ALERTE], tmp_path)["deja"] == 1                 # pas deux fois
    ligne = json.loads((tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert ligne["a"] == ["ag…@exemple.tn", "+216*****678"] and "12345" not in json.dumps(ligne)


def test_email_et_whatsapp_simules(tmp_path, monkeypatch):
    from service import notifications as n
    vus = {}

    class FauxSMTP:
        def __init__(self, hote, port, timeout):
            vus["smtp"] = (hote, port)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def starttls(self):
            vus["tls"] = True
        def login(self, u, p):
            vus["login"] = u
        def send_message(self, m):
            vus["sujet"], vus["a"] = m["Subject"], m["To"]

    class Rep:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b'{"status": "queued", "sid": "SM1"}'

    def ouvrir(req, timeout):
        vus.setdefault("twilio", []).append(req.data.decode())
        return Rep()

    for k, v in (("PCFNO3D_SMTP_HOTE", "smtp.exemple.tn"), ("PCFNO3D_SMTP_DE", "atom@exemple.tn"),
                 ("PCFNO3D_SMTP_UTILISATEUR", "atom"), ("PCFNO3D_SMTP_MOT_DE_PASSE", "secret"),
                 ("TWILIO_ACCOUNT_SID", "AC1"), ("TWILIO_AUTH_TOKEN", "t"), ("TWILIO_WHATSAPP_DE", "+15550000000")):
        monkeypatch.setenv(k, v)
    canaux = [(n.CanalJournal(tmp_path), ["agri@exemple.tn", "+21612345678"]),
              (n.CanalEmail(smtp=FauxSMTP), ["agri@exemple.tn"]),
              (n.CanalWhatsApp(ouvrir=ouvrir), ["+21612345678"])]
    r = n.notifier([ALERTE], tmp_path, canaux)
    assert r["canaux"] == ["journal", "email", "whatsapp"] and not r["erreurs"]
    assert vus["smtp"] == ("smtp.exemple.tn", 587) and vus["tls"] and "vent fort" in vus["sujet"]
    assert "whatsapp%3A%2B21612345678" in vus["twilio"][0] and "whatsapp%3A%2B15550000000" in vus["twilio"][0]


def test_alertes_service(wf, monkeypatch):  # noqa: F811
    """Les alertes du service ont un identifiant stable et un niveau ; la journee de test produit au moins
    les stations muettes (la fausse source ne donne des mesures qu'a 07 h)."""
    from service import atom_outils as ao
    a = ao.alertes_service(wf)
    assert all({"id", "kind", "niveau", "titre", "cause", "recommandation"} <= set(x) for x in a)
    assert len({x["id"] for x in a}) == len(a)
    assert ao.alertes_service(wf) == a
