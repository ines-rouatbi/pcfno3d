"""Comptes et roles cote serveur (cahier v12, BF19, BF43) : grand public et agriculteur actifs tout de suite,
exploitant eolien valide par l'administrateur, administrateur cree par la ligne de commande.

  - mot de passe : scrypt (n = 2^14, r = 8, p = 1) avec un sel aleatoire par compte ; jamais garde en clair,
    jamais renvoye, jamais journalise ; 8 caracteres au moins, dont un chiffre ;
  - adresse e-mail VERIFIEE avant toute connexion (demande d'Ines, 20/11/2026) : l'inscription envoie un lien a
    usage unique qui expire au bout de 24 h ; tant qu'il n'est pas ouvert, la connexion est refusee meme avec le
    bon mot de passe ; un nouveau lien remplace l'ancien ;
  - mot de passe oublie : lien a usage unique qui expire au bout d'1 h ; le changement ferme la session ouverte ;
  - les liens ne contiennent qu'un jeton aleatoire ; le serveur n'en garde que l'empreinte SHA-256 ;
  - connexion : une CLE DE SESSION du meme trousseau que les cles d'API (portee read, ou admin pour un
    administrateur), nommee compte-<empreinte de l'e-mail>, qui EXPIRE au bout de 8 h (il faut alors redonner
    son mot de passe) ; une seule session par compte (une nouvelle connexion revoque la precedente) ; un compte
    desactive perd sa session aussitot ;
  - 5 echecs de connexion en 15 min bloquent le compte 15 min (contre la force brute) ; les reponses aux
    demandes de lien sont les memes que l'adresse existe ou non (pas d'enumeration des comptes) ;
  - le role est relu a chaque requete (`role_de_cle`) : chaque espace (agriculteur, exploitant eolien,
    administration) est reserve a SON role.

Donnees personnelles (e-mail, nom) : run_dir/comptes.json, jamais dans le depot ; la liste des comptes est
reservee a la portee admin. Les comptes crees avant la verification d'e-mail sont consideres verifies.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[A-Za-z]{2,24}$")
ROLES = ("public", "agri", "eolien", "admin")
ROLES_IMMEDIATS = ("public", "agri")
ESPACE_EOLIEN = ("eolien",)          # espace exploitant eolien : son role seulement
ESPACE_AGRI = ("agri",)              # espace agriculteur : son role seulement
ECHECS_MAX, BLOCAGE_S = 5, 15 * 60
VERIFICATION_S = 24 * 3600           # validite du lien de verification d'e-mail
REINITIALISATION_S = 3600            # validite du lien « mot de passe oublie »
SESSION_S = 8 * 3600                 # duree d'une session avant de redonner son mot de passe
RENVOI_MIN_S = 60                    # delai minimal entre deux liens envoyes a la meme adresse
_VERROU = threading.Lock()


class EmailNonVerifie(Exception):
    """Bon mot de passe, mais l'adresse n'a pas encore ete verifiee par le lien recu."""


class LienInvalide(ValueError):
    """Lien inconnu, deja utilise ou expire."""


def _maintenant() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def nom_session(email: str) -> str:
    return "compte-" + hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:12]


def _hacher(mot_de_passe: str, sel: bytes) -> str:
    return hashlib.scrypt(mot_de_passe.encode("utf-8"), salt=sel, n=2 ** 14, r=8, p=1, dklen=32).hex()


def _empreinte(jeton: str) -> str:
    return hashlib.sha256(jeton.strip().encode("utf-8")).hexdigest()


def _verifier_mdp(mot_de_passe: str) -> None:
    if len(mot_de_passe) < 8 or not any(c.isdigit() for c in mot_de_passe):
        raise ValueError("mot de passe : 8 caracteres au moins, dont un chiffre")


class Comptes:
    def __init__(self, dossier: Path, store) -> None:
        self.chemin = Path(dossier) / "comptes.json"
        self.store = store
        self._echecs: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ fichier
    def _lire(self) -> list[dict]:
        return json.loads(self.chemin.read_text(encoding="utf-8")) if self.chemin.exists() else []

    def _ecrire(self, liste: list[dict]) -> None:
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.chemin.with_suffix(".tmp")
        tmp.write_text(json.dumps(liste, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.chemin)

    @staticmethod
    def public(c: dict) -> dict:
        d = {k: c.get(k) for k in ("email", "nom", "role", "statut", "demande_role", "cree_le", "connecte_le")}
        d["email_verifie"] = c.get("email_verifie", True)          # compte d'avant la verification : verifie
        return d

    def _trouver(self, liste: list[dict], email: str) -> dict | None:
        e = email.strip().lower()
        return next((c for c in liste if c["email"] == e), None)

    def _modifier(self, email: str, f) -> dict:
        """Applique f(compte) sous verrou et ecrit ; KeyError si le compte n'existe pas."""
        with _VERROU:
            liste = self._lire()
            c = self._trouver(liste, email)
            if c is None:
                raise KeyError(email)
            f(c)
            self._ecrire(liste)
        return self.public(c)

    # ------------------------------------------------------------------ inscription
    def creer(self, email: str, mot_de_passe: str, nom: str = "", role: str = "public",
              verifie: bool = False) -> dict:
        """Cree le compte, adresse NON verifiee (sauf `verifie=True` : comptes locaux de developpement).
        Le lien de verification est fabrique a part (`jeton_verification`)."""
        email = email.strip().lower()
        if not EMAIL.match(email):
            raise ValueError("adresse e-mail invalide")
        if role not in ROLES or role == "admin":
            raise ValueError("role : public, agri ou eolien (un administrateur se cree par la ligne de commande)")
        _verifier_mdp(mot_de_passe)
        sel = os.urandom(16)
        c = {"email": email, "nom": nom.strip()[:120], "sel": sel.hex(), "hachage": _hacher(mot_de_passe, sel),
             "role": role if role in ROLES_IMMEDIATS else "public", "statut": "actif",
             "demande_role": None if role in ROLES_IMMEDIATS else role, "cree_le": _maintenant(), "connecte_le": None,
             "email_verifie": bool(verifie)}
        with _VERROU:
            liste = self._lire()
            if self._trouver(liste, email):
                raise ValueError("un compte existe deja pour cette adresse")
            liste.append(c)
            self._ecrire(liste)
        return self.public(c)

    def creer_admin(self, email: str, mot_de_passe: str, nom: str = "Administrateur") -> dict:
        email = email.strip().lower()
        if not EMAIL.match(email):
            raise ValueError("adresse e-mail invalide")
        _verifier_mdp(mot_de_passe)
        sel = os.urandom(16)
        with _VERROU:
            liste = self._lire()
            c = self._trouver(liste, email)
            if c is None:
                c = {"email": email, "nom": nom, "cree_le": _maintenant(), "connecte_le": None}
                liste.append(c)
            c.update({"sel": sel.hex(), "hachage": _hacher(mot_de_passe, sel), "role": "admin", "statut": "actif",
                      "demande_role": None, "email_verifie": True})
            self._ecrire(liste)
        return self.public(c)

    # ------------------------------------------------------------------ liens a usage unique
    def _nouveau_jeton(self, email: str, sorte: str, duree_s: int, condition) -> str | None:
        """Pose un jeton `sorte` (verif / reinit) sur le compte et renvoie le secret en clair, ou None si le
        compte n'existe pas, ne remplit pas `condition`, ou a deja recu un lien il y a moins d'une minute."""
        t = time.time()
        jeton = secrets.token_urlsafe(32)
        with _VERROU:
            liste = self._lire()
            c = self._trouver(liste, email)
            if c is None or not condition(c):
                return None
            if t - float(c.get(f"{sorte}_envoye", 0) or 0) < RENVOI_MIN_S:
                return None
            c[f"{sorte}_hachage"], c[f"{sorte}_expire"], c[f"{sorte}_envoye"] = _empreinte(jeton), t + duree_s, t
            self._ecrire(liste)
        return jeton

    def _consommer(self, jeton: str, sorte: str, appliquer) -> dict:
        """Trouve le compte du jeton, verifie l'expiration, applique `appliquer(c)` et efface le jeton."""
        if not jeton or len(jeton) > 200:
            raise LienInvalide("lien invalide ou deja utilise")
        h = _empreinte(jeton)
        with _VERROU:
            liste = self._lire()
            c = next((x for x in liste if x.get(f"{sorte}_hachage")
                      and hmac.compare_digest(x[f"{sorte}_hachage"], h)), None)
            if c is None:
                raise LienInvalide("lien invalide ou deja utilise")
            if time.time() > float(c.get(f"{sorte}_expire", 0) or 0):
                raise LienInvalide("lien expire : demandez un nouveau lien")
            appliquer(c)
            for k in ("hachage", "expire", "envoye"):
                c.pop(f"{sorte}_{k}", None)
            self._ecrire(liste)
        return self.public(c)

    def jeton_verification(self, email: str) -> str | None:
        """Nouveau lien de verification (l'ancien ne vaut plus rien) ; None si rien a envoyer."""
        return self._nouveau_jeton(email, "verif", VERIFICATION_S, lambda c: not c.get("email_verifie", True))

    def verifier_email(self, jeton: str) -> dict:
        return self._consommer(jeton, "verif", lambda c: c.update({"email_verifie": True}))

    def jeton_reinitialisation(self, email: str) -> str | None:
        """Lien « mot de passe oublie » (comptes actifs seulement) ; None si rien a envoyer."""
        return self._nouveau_jeton(email, "reinit", REINITIALISATION_S, lambda c: c.get("statut") == "actif")

    def reinitialiser(self, jeton: str, mot_de_passe: str) -> dict:
        _verifier_mdp(mot_de_passe)
        sel = os.urandom(16)

        def appliquer(c):
            c.update({"sel": sel.hex(), "hachage": _hacher(mot_de_passe, sel),
                      "email_verifie": True})               # le lien recu prouve aussi l'adresse
        compte = self._consommer(jeton, "reinit", appliquer)
        self.deconnecter(compte["email"])                    # une session ouverte avec l'ancien mot de passe tombe
        self._echecs.pop(compte["email"], None)
        return compte

    # ------------------------------------------------------------------ session
    def connecter(self, email: str, mot_de_passe: str) -> tuple[str, dict]:
        """(cle de session en clair, compte) ; ValueError si refus (message identique : pas d'enumeration),
        EmailNonVerifie si le mot de passe est bon mais l'adresse pas encore verifiee."""
        email = email.strip().lower()
        t = time.time()
        echecs = [x for x in self._echecs.get(email, []) if t - x < BLOCAGE_S]
        if len(echecs) >= ECHECS_MAX:
            raise PermissionError("trop d'essais : reessayer dans 15 minutes")
        with _VERROU:
            liste = self._lire()
            c = self._trouver(liste, email)
            ok = c is not None and hmac.compare_digest(_hacher(mot_de_passe, bytes.fromhex(c["sel"])), c["hachage"])
            if not ok or c.get("statut") != "actif":
                echecs.append(t)
                self._echecs[email] = echecs
                raise ValueError("adresse ou mot de passe incorrect, ou compte desactive")
            self._echecs.pop(email, None)
            if not c.get("email_verifie", True):
                raise EmailNonVerifie("adresse e-mail non verifiee : ouvrez le lien recu par e-mail")
            nom = nom_session(email)
            self.store.revoke(nom)                                   # une seule session par compte
            secret, _ = self.store.create(nom, scope="admin" if c["role"] == "admin" else "read",
                                          duree_s=SESSION_S)
            c["connecte_le"] = _maintenant()
            self._ecrire(liste)
        return secret, self.public(c)

    def deconnecter(self, email: str) -> bool:
        return self.store.revoke(nom_session(email))

    # ------------------------------------------------------------------ administration
    def lister(self) -> list[dict]:
        return [self.public(c) for c in self._lire()]

    def valider(self, email: str, accepter: bool = True) -> dict:
        def f(c):
            if c.get("demande_role") and accepter:
                c["role"] = c["demande_role"]
            c["demande_role"] = None
        return self._modifier(email, f)

    def changer_role(self, email: str, role: str) -> dict:
        """L'administrateur change le role d'un compte (public, agri ou eolien ; jamais admin, jamais le role d'un
        administrateur : un administrateur se gere par la ligne de commande)."""
        if role not in ROLES or role == "admin":
            raise ValueError("role : public, agri ou eolien")

        def f(c):
            if c.get("role") == "admin":
                raise ValueError("le role d'un administrateur ne se change pas ici")
            c["role"], c["demande_role"] = role, None
        return self._modifier(email, f)

    def marquer_verifie(self, email: str) -> dict:
        """L'administrateur confirme l'adresse lui-meme (par exemple si l'e-mail ne peut pas etre envoye)."""
        def f(c):
            c["email_verifie"] = True
            for k in ("hachage", "expire", "envoye"):
                c.pop(f"verif_{k}", None)
        return self._modifier(email, f)

    def changer_statut(self, email: str, actif: bool) -> dict:
        compte = self._modifier(email, lambda c: c.update({"statut": "actif" if actif else "desactive"}))
        if not actif:
            self.deconnecter(email)                                  # la session tombe aussitot
        return compte

    def supprimer(self, email: str) -> dict:
        """L'administrateur supprime un compte : l'entree (e-mail, nom, empreinte du mot de passe, liens en cours)
        est retiree de comptes.json et la session tombe aussitot. Un compte administrateur ne se supprime pas
        ici (ligne de commande). Irreversible : la personne devra se reinscrire."""
        e = email.strip().lower()
        with _VERROU:
            liste = self._lire()
            c = self._trouver(liste, e)
            if c is None:
                raise KeyError(email)
            if c.get("role") == "admin":
                raise ValueError("un compte administrateur ne se supprime pas ici")
            liste.remove(c)
            self._ecrire(liste)
        self.deconnecter(e)
        self._echecs.pop(e, None)
        return self.public(c)

    def compte_de_cle(self, nom_cle: str | None) -> dict | None:
        """Le compte d'une cle de session (None pour une cle de service)."""
        if not nom_cle or not nom_cle.startswith("compte-"):
            return None
        return next((self.public(c) for c in self._lire() if nom_session(c["email"]) == nom_cle), None)
