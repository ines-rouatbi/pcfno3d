"""Comptes et roles cote serveur (cahier v12, BF19, BF43) : grand public et agriculteur actifs tout de suite,
exploitant eolien valide par l'administrateur, administrateur cree par la ligne de commande.

  - mot de passe : scrypt (n = 2^14, r = 8, p = 1) avec un sel aleatoire par compte ; jamais garde en clair,
    jamais renvoye, jamais journalise ; 8 caracteres au moins, dont un chiffre ;
  - connexion : une CLE DE SESSION du meme trousseau que les cles d'API (portee read, ou admin pour un
    administrateur), nommee compte-<empreinte de l'e-mail> ; une seule session par compte (une nouvelle
    connexion revoque la precedente) ; un compte desactive perd sa session aussitot ;
  - 5 echecs de connexion en 15 min bloquent le compte 15 min (contre la force brute) ;
  - le role est relu a chaque requete (`role_de_cle`) : l'espace eolien est reserve aux roles eolien et admin
    pour une cle de session ; une cle de service (Worker, CLI) garde son acces de toujours.

Donnees personnelles (e-mail, nom) : run_dir/comptes.json, jamais dans le depot ; la liste des comptes est
reservee a la portee admin.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import time
from pathlib import Path

EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[A-Za-z]{2,24}$")
ROLES = ("public", "agri", "eolien", "admin")
ROLES_IMMEDIATS = ("public", "agri")
ESPACE_EOLIEN = ("eolien", "admin")
ECHECS_MAX, BLOCAGE_S = 5, 15 * 60
_VERROU = threading.Lock()


def _maintenant() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def nom_session(email: str) -> str:
    return "compte-" + hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:12]


def _hacher(mot_de_passe: str, sel: bytes) -> str:
    return hashlib.scrypt(mot_de_passe.encode("utf-8"), salt=sel, n=2 ** 14, r=8, p=1, dklen=32).hex()


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
        return {k: c.get(k) for k in ("email", "nom", "role", "statut", "demande_role", "cree_le", "connecte_le")}

    def _trouver(self, liste: list[dict], email: str) -> dict | None:
        e = email.strip().lower()
        return next((c for c in liste if c["email"] == e), None)

    # ------------------------------------------------------------------ inscription
    def creer(self, email: str, mot_de_passe: str, nom: str = "", role: str = "public") -> dict:
        email = email.strip().lower()
        if not EMAIL.match(email):
            raise ValueError("adresse e-mail invalide")
        if role not in ROLES or role == "admin":
            raise ValueError("role : public, agri ou eolien (un administrateur se cree par la ligne de commande)")
        _verifier_mdp(mot_de_passe)
        sel = os.urandom(16)
        c = {"email": email, "nom": nom.strip()[:120], "sel": sel.hex(), "hachage": _hacher(mot_de_passe, sel),
             "role": role if role in ROLES_IMMEDIATS else "public", "statut": "actif",
             "demande_role": None if role in ROLES_IMMEDIATS else role, "cree_le": _maintenant(), "connecte_le": None}
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
                      "demande_role": None})
            self._ecrire(liste)
        return self.public(c)

    # ------------------------------------------------------------------ session
    def connecter(self, email: str, mot_de_passe: str) -> tuple[str, dict]:
        """(cle de session en clair, compte) ; ValueError si refus (message identique : pas d'enumeration)."""
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
            nom = nom_session(email)
            self.store.revoke(nom)                                   # une seule session par compte
            secret, _ = self.store.create(nom, scope="admin" if c["role"] == "admin" else "read")
            c["connecte_le"] = _maintenant()
            self._ecrire(liste)
        return secret, self.public(c)

    def deconnecter(self, email: str) -> bool:
        return self.store.revoke(nom_session(email))

    # ------------------------------------------------------------------ administration
    def lister(self) -> list[dict]:
        return [self.public(c) for c in self._lire()]

    def valider(self, email: str, accepter: bool = True) -> dict:
        with _VERROU:
            liste = self._lire()
            c = self._trouver(liste, email)
            if c is None:
                raise KeyError(email)
            if c.get("demande_role") and accepter:
                c["role"] = c["demande_role"]
            c["demande_role"] = None
            self._ecrire(liste)
        return self.public(c)

    def changer_statut(self, email: str, actif: bool) -> dict:
        with _VERROU:
            liste = self._lire()
            c = self._trouver(liste, email)
            if c is None:
                raise KeyError(email)
            c["statut"] = "actif" if actif else "desactive"
            self._ecrire(liste)
        if not actif:
            self.deconnecter(email)                                  # la session tombe aussitot
        return self.public(c)

    def compte_de_cle(self, nom_cle: str | None) -> dict | None:
        """Le compte d'une cle de session (None pour une cle de service)."""
        if not nom_cle or not nom_cle.startswith("compte-"):
            return None
        return next((self.public(c) for c in self._lire() if nom_session(c["email"]) == nom_cle), None)
