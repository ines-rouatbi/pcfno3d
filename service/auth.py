"""Authentification par cle d'API, pour l'API HTTP et pour le serveur MCP.

Une cle a la forme `pcf_<32 caracteres>`. Elle n'est **stockee que sous forme de hash**
SHA-256 dans `run_dir/api_keys.json` : le secret n'apparait qu'une fois, a la creation.
Perdre une cle veut donc dire en creer une autre, pas la retrouver.

Deux portees :
    read   lire les instants, les champs, les points, lancer une reconstruction
    admin  en plus : exporter des fichiers sur le disque du serveur, gerer les cles

Trois facons de fournir une cle, dans cet ordre :
    en-tete `X-API-Key: pcf_...`
    en-tete `Authorization: Bearer pcf_...`
    parametre `?api_key=pcf_...`  (dernier recours, pour un <img>/<a> du navigateur)

Ferme par defaut : si aucune cle n'est definie (ni fichier ni `PCFNO3D_API_KEYS`), TOUT
est refuse (401) sauf `/api/health`, qui le dit. Un mode ouvert n'existe que sur demande
explicite (`pcfno3d serve --insecure-open`, ou `PCFNO3D_INSECURE_OPEN=1`), pour le
developpement local ; il est ignore sur le chemin Colab / tunnel et derriere un proxy.

Revocation immediate : le serveur relit `api_keys.json` des que le fichier change (date,
taille ou inode : un `stat` par requete), donc `pcfno3d apikey revoke` lance depuis un
autre processus prend effet a la requete suivante. Le serveur n'ecrit JAMAIS ce fichier
en verifiant une cle : la date de derniere utilisation va dans `api_keys_usage.json`.
Sinon une ecriture du serveur pourrait ecraser une revocation faite au meme instant.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path


KEY_PREFIX = "pcf_"
KEY_BYTES = 24                       # -> 32 caracteres en base64url
SCOPES = ("read", "admin")
ENV_KEYS = "PCFNO3D_API_KEYS"        # "cle1,cle2" ou "nom:cle1,nom:cle2" (portee read)
ENV_INSECURE_OPEN = "PCFNO3D_INSECURE_OPEN"   # "1" : mode ouvert, developpement local
ENV_FORBID_OPEN = "PCFNO3D_FORBID_OPEN"       # pose par service/colab.py : jamais ouvert
USAGE_FLUSH_S = 60.0                 # date de derniere utilisation : ecrite au plus 1/min


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(KEY_BYTES)[:32]


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vrai(valeur: str | None) -> bool:
    return (valeur or "").strip().lower() in {"1", "true", "oui", "yes", "on"}


OPEN_WARNING = (f"MODE OUVERT ({ENV_INSECURE_OPEN}) : sans cle active, l'API accepte tout. "
                "Developpement local uniquement, ne pas exposer au reseau.")


def insecure_open_policy() -> tuple[bool, str]:
    """(mode ouvert autorise, explication). Ouvert SEULEMENT sur opt-in explicite.

    L'opt-in est ignore si le processus a ete lance par le chemin Colab / tunnel
    (`PCFNO3D_FORBID_OPEN`) ou si des hotes externes sont declares
    (`PCFNO3D_ALLOWED_HOSTS` : proxy, domaine, tunnel) : dans ces cas le service est
    joignable de l'exterieur, et l'ouvrir serait une fuite.
    """
    if not _vrai(os.environ.get(ENV_INSECURE_OPEN)):
        return False, "ferme par defaut"
    if os.environ.get(ENV_FORBID_OPEN):
        return False, f"{ENV_INSECURE_OPEN} ignore : interdit sur le chemin Colab / tunnel"
    if (os.environ.get("PCFNO3D_ALLOWED_HOSTS") or "").strip():
        return False, (f"{ENV_INSECURE_OPEN} ignore : des hotes externes sont declares "
                       "(PCFNO3D_ALLOWED_HOSTS), le service est expose")
    return True, OPEN_WARNING


@dataclass
class ApiKeyRecord:
    """Ce qui est conserve d'une cle : son hash, jamais le secret."""

    name: str
    key_hash: str
    scope: str = "read"
    created_at: str = field(default_factory=_now)
    last_used_at: str | None = None
    revoked: bool = False
    # quotas propres a cette cle ; None = valeur par defaut du service, 0 = illimite
    quota_minute: int | None = None
    quota_jour: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ApiKeyRecord":
        # tolerant : un champ ajoute par une version future ne doit pas tout casser
        connus = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in connus})

    def to_dict(self) -> dict:
        out = {"name": self.name, "key_hash": self.key_hash, "scope": self.scope,
               "created_at": self.created_at, "last_used_at": self.last_used_at,
               "revoked": self.revoked}
        # ecrits seulement s'ils sont poses : un trousseau sans quota reste lisible par
        # une version anterieure du service
        if self.quota_minute is not None:
            out["quota_minute"] = self.quota_minute
        if self.quota_jour is not None:
            out["quota_jour"] = self.quota_jour
        return out

    @property
    def empreinte(self) -> str:
        """Prefixe court du hash, pour le journal : jamais le hash complet."""
        return self.key_hash[:8]

    def public(self) -> dict:
        """Vue sans secret, pour /keys et la ligne de commande."""
        return {"name": self.name, "scope": self.scope, "created_at": self.created_at,
                "last_used_at": self.last_used_at, "revoked": self.revoked,
                "empreinte": self.key_hash[:12],
                "quota_minute": self.quota_minute, "quota_jour": self.quota_jour}


def _quota(value: int | None, nom: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{nom} doit etre >= 0 (0 = illimite)")
    return value


class ApiKeyStore:
    """Le trousseau : un fichier JSON + les cles de l'environnement.

    `allow_open` n'a d'effet que s'il n'existe AUCUNE cle active : c'est le mode ouvert
    explicite du developpement local (voir `insecure_open_policy`).
    """

    def __init__(self, path: str | Path, allow_open: bool = False) -> None:
        self.path = Path(path)
        self.allow_open = allow_open
        self.records: list[ApiKeyRecord] = []
        self.erreur: str | None = None
        self._env: dict[str, ApiKeyRecord] = {}
        self._sig: tuple | None = None
        self._lock = threading.RLock()
        self._usage: dict[str, str] = {}
        self._usage_flushed: float | None = None
        self.load()

    @property
    def usage_path(self) -> Path:
        return self.path.with_name(self.path.stem + "_usage.json")

    # ------------------------------------------------------------------ fichier
    def _signature(self) -> tuple | None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        # os.replace change l'inode : une reecriture de meme taille dans la meme tranche
        # d'horloge est donc vue quand meme
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    def load(self) -> "ApiKeyStore":
        with self._lock:
            self._sig = self._signature()
            self.records = []
            self.erreur = None
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    self.records = [ApiKeyRecord.from_dict(r) for r in data.get("keys", [])]
                except (ValueError, TypeError, AttributeError, OSError) as exc:
                    # trousseau illisible : on ferme (aucune cle valide, pas de mode ouvert)
                    self.records = []
                    self.erreur = f"trousseau illisible ({type(exc).__name__})"
            self._env = {}
            for item in (os.environ.get(ENV_KEYS) or "").split(","):
                item = item.strip()
                if not item:
                    continue
                name, _, secret = item.rpartition(":")
                secret = secret.strip()
                if not secret:
                    continue
                self._env[hash_key(secret)] = ApiKeyRecord(
                    name=(name.strip() or "env"), key_hash=hash_key(secret), scope="read")
        return self

    def refresh(self) -> bool:
        """Relit le fichier s'il a change depuis la derniere lecture."""
        with self._lock:
            if self._signature() != self._sig:
                self.load()
                return True
        return False

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "keys": [r.to_dict() for r in self.records]}
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
            self._sig = self._signature()

    # ------------------------------------------------------------------ gestion
    def create(self, name: str, scope: str = "read", quota_minute: int | None = None,
               quota_jour: int | None = None) -> tuple[str, ApiKeyRecord]:
        """Renvoie (secret en clair, enregistrement). Le secret n'est plus jamais lisible."""
        if scope not in SCOPES:
            raise ValueError(f"portee inconnue : {scope} (attendu {SCOPES})")
        record = ApiKeyRecord(name=name, key_hash="", scope=scope,
                              quota_minute=_quota(quota_minute, "quota_minute"),
                              quota_jour=_quota(quota_jour, "quota_jour"))
        with self._lock:
            self.refresh()
            if any(r.name == name and not r.revoked for r in self.records):
                raise ValueError(f"une cle active s'appelle deja '{name}'")
            secret = generate_key()
            record.key_hash = hash_key(secret)
            self.records.append(record)
            self.save()
        return secret, record

    def register_hash(self, name: str, key_hash: str, scope: str = "read",
                      quota_minute: int | None = None,
                      quota_jour: int | None = None) -> ApiKeyRecord:
        """Enregistre une cle dont on ne connait que l'EMPREINTE (SHA-256).

        Pour un deploiement ou la cle a ete generee ailleurs et remise de main a main :
        le serveur n'a jamais besoin du secret, et l'empreinte peut figurer dans un
        notebook sans rien reveler. Idempotent : une empreinte deja active est renvoyee
        telle quelle (ses quotas sont mis a jour s'ils sont donnes).
        """
        key_hash = key_hash.strip().lower()
        if len(key_hash) != 64 or any(c not in "0123456789abcdef" for c in key_hash):
            raise ValueError("empreinte attendue : SHA-256 en hexadecimal (64 caracteres)")
        if scope not in SCOPES:
            raise ValueError(f"portee inconnue : {scope} (attendu {SCOPES})")
        quota_minute = _quota(quota_minute, "quota_minute")
        quota_jour = _quota(quota_jour, "quota_jour")
        with self._lock:
            self.refresh()
            for r in self.records:
                if r.key_hash == key_hash and not r.revoked:
                    if r.scope != scope:
                        raise ValueError(f"cette empreinte existe deja avec la portee {r.scope}")
                    if quota_minute is not None or quota_jour is not None:
                        self._apply_quota(r, quota_minute, quota_jour)
                    return r
            if any(r.name == name and not r.revoked for r in self.records):
                raise ValueError(f"une cle active s'appelle deja '{name}'")
            record = ApiKeyRecord(name=name, key_hash=key_hash, scope=scope,
                                  quota_minute=quota_minute, quota_jour=quota_jour)
            self.records.append(record)
            self.save()
        return record

    def _apply_quota(self, record: ApiKeyRecord, minute: int | None,
                     jour: int | None) -> None:
        if minute is not None:
            record.quota_minute = minute
        if jour is not None:
            record.quota_jour = jour
        self.save()

    def set_quota(self, name: str, quota_minute: int | None = None,
                  quota_jour: int | None = None, reset: bool = False) -> ApiKeyRecord | None:
        """Surcharge les quotas d'une cle active (`reset` : retour aux valeurs par defaut)."""
        quota_minute = _quota(quota_minute, "quota_minute")
        quota_jour = _quota(quota_jour, "quota_jour")
        with self._lock:
            self.refresh()
            for r in self.records:
                if r.name == name and not r.revoked:
                    if reset:
                        r.quota_minute = r.quota_jour = None
                        self.save()
                    else:
                        self._apply_quota(r, quota_minute, quota_jour)
                    return r
        return None

    def revoke(self, name: str) -> bool:
        with self._lock:
            self.refresh()
            found = False
            for r in self.records:
                if r.name == name and not r.revoked:
                    r.revoked = True
                    found = True
            if found:
                self.save()
        return found

    def list_public(self) -> list[dict]:
        self.refresh()
        usage = self._read_usage()
        out = []
        for r in self.records:
            row = r.public()
            vu = usage.get(r.key_hash)
            if vu and (row["last_used_at"] is None or vu > row["last_used_at"]):
                row["last_used_at"] = vu
            out.append(row)
        return out

    # ------------------------------------------------------------------ utilisation
    def _read_usage(self) -> dict[str, str]:
        try:
            disque = json.loads(self.usage_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            disque = {}
        if not isinstance(disque, dict):
            disque = {}
        for h, vu in self._usage.items():
            if vu > str(disque.get(h, "")):
                disque[h] = vu
        return disque

    def _mark_used(self, record: ApiKeyRecord) -> None:
        record.last_used_at = _now()
        with self._lock:
            self._usage[record.key_hash] = record.last_used_at
            now = time.monotonic()
            if self._usage_flushed is not None and now - self._usage_flushed < USAGE_FLUSH_S:
                return
            self._usage_flushed = now
            try:
                merged = self._read_usage()
                tmp = self.usage_path.with_name(f"{self.usage_path.name}.{os.getpid()}.tmp")
                tmp.write_text(json.dumps(merged, indent=1), encoding="utf-8")
                os.replace(tmp, self.usage_path)
            except OSError:
                pass                      # une date d'utilisation ne vaut pas un 500

    # ------------------------------------------------------------------ controle
    @property
    def empty(self) -> bool:
        """Aucune cle active (le fichier est relu s'il a change)."""
        self.refresh()
        return not self.active_records()

    @property
    def open_mode(self) -> bool:
        """Mode ouvert : opt-in explicite ET aucune cle active ET trousseau lisible."""
        return self.allow_open and self.empty and self.erreur is None

    def active_records(self) -> list[ApiKeyRecord]:
        return [r for r in self.records if not r.revoked] + list(self._env.values())

    def verify(self, secret: str | None) -> ApiKeyRecord | None:
        """Renvoie l'enregistrement si la cle est valide, sinon None."""
        if not secret:
            return None
        self.refresh()                    # une revocation faite ailleurs compte tout de suite
        h = hash_key(secret.strip())
        for r in self.active_records():
            # comparaison a temps constant sur le hash : pas de fuite par le temps
            if secrets.compare_digest(r.key_hash, h):
                self._mark_used(r)
                return r
        return None


def extract_key(headers, query) -> str | None:
    """Lit la cle dans X-API-Key, Authorization: Bearer, ou ?api_key=."""
    key = headers.get("x-api-key")
    if key:
        return key.strip()
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (query.get("api_key") or "").strip() or None


def scope_allows(record: ApiKeyRecord | None, required: str) -> bool:
    """`admin` couvre `read` ; une portee inconnue ne couvre rien."""
    if record is None:
        return False
    if record.scope == "admin":
        return True
    return record.scope == required


def default_store_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "api_keys.json"


NO_KEY_MESSAGE = ("aucune cle configuree : toutes les requetes sont refusees sauf "
                  "/api/health. Creer une cle : pcfno3d apikey create <nom>")


def summarize(store: ApiKeyStore) -> dict:
    """Etat du trousseau, pour /health (jamais de secret ni de hash complet)."""
    store.refresh()
    active = store.active_records()
    out: dict = {"cles_actives": len(active),
                 "portees": sorted({r.scope for r in active}) or []}
    if store.open_mode:
        out["auth"] = "ouvert"
        out["avertissement"] = OPEN_WARNING
    elif not active:
        out["auth"] = "aucune_cle"
        out["message"] = (f"{store.erreur} ; " if store.erreur else "") + NO_KEY_MESSAGE
    else:
        out["auth"] = "cle_api"
    return out
