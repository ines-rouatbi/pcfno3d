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

Mode ouvert : si aucune cle n'est definie (ni fichier ni `PCFNO3D_API_KEYS`), l'API
accepte tout et le signale dans /health. Pratique en local, a ne pas exposer au reseau.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


KEY_PREFIX = "pcf_"
KEY_BYTES = 24                       # -> 32 caracteres en base64url
SCOPES = ("read", "admin")
ENV_KEYS = "PCFNO3D_API_KEYS"        # "cle1,cle2" ou "nom:cle1,nom:cle2" (portee read)


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(KEY_BYTES)[:32]


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ApiKeyRecord:
    """Ce qui est conserve d'une cle : son hash, jamais le secret."""

    name: str
    key_hash: str
    scope: str = "read"
    created_at: str = field(default_factory=_now)
    last_used_at: str | None = None
    revoked: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "key_hash": self.key_hash, "scope": self.scope,
                "created_at": self.created_at, "last_used_at": self.last_used_at,
                "revoked": self.revoked}

    def public(self) -> dict:
        """Vue sans secret, pour /keys et la ligne de commande."""
        return {"name": self.name, "scope": self.scope, "created_at": self.created_at,
                "last_used_at": self.last_used_at, "revoked": self.revoked,
                "empreinte": self.key_hash[:12]}


class ApiKeyStore:
    """Le trousseau : un fichier JSON + les cles de l'environnement."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: list[ApiKeyRecord] = []
        self._env: dict[str, ApiKeyRecord] = {}
        self.load()

    # ------------------------------------------------------------------ fichier
    def load(self) -> "ApiKeyStore":
        self.records = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.records = [ApiKeyRecord(**r) for r in data.get("keys", [])]
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

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "keys": [r.to_dict() for r in self.records]}
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ gestion
    def create(self, name: str, scope: str = "read") -> tuple[str, ApiKeyRecord]:
        """Renvoie (secret en clair, enregistrement). Le secret n'est plus jamais lisible."""
        if scope not in SCOPES:
            raise ValueError(f"portee inconnue : {scope} (attendu {SCOPES})")
        if any(r.name == name and not r.revoked for r in self.records):
            raise ValueError(f"une cle active s'appelle deja '{name}'")
        secret = generate_key()
        record = ApiKeyRecord(name=name, key_hash=hash_key(secret), scope=scope)
        self.records.append(record)
        self.save()
        return secret, record

    def register_hash(self, name: str, key_hash: str, scope: str = "read") -> ApiKeyRecord:
        """Enregistre une cle dont on ne connait que l'EMPREINTE (SHA-256).

        Pour un deploiement ou la cle a ete generee ailleurs et remise de main a main :
        le serveur n'a jamais besoin du secret, et l'empreinte peut figurer dans un
        notebook sans rien reveler. Idempotent : une empreinte deja active est renvoyee
        telle quelle.
        """
        key_hash = key_hash.strip().lower()
        if len(key_hash) != 64 or any(c not in "0123456789abcdef" for c in key_hash):
            raise ValueError("empreinte attendue : SHA-256 en hexadecimal (64 caracteres)")
        if scope not in SCOPES:
            raise ValueError(f"portee inconnue : {scope} (attendu {SCOPES})")
        for r in self.records:
            if r.key_hash == key_hash and not r.revoked:
                if r.scope != scope:
                    raise ValueError(f"cette empreinte existe deja avec la portee {r.scope}")
                return r
        if any(r.name == name and not r.revoked for r in self.records):
            raise ValueError(f"une cle active s'appelle deja '{name}'")
        record = ApiKeyRecord(name=name, key_hash=key_hash, scope=scope)
        self.records.append(record)
        self.save()
        return record

    def revoke(self, name: str) -> bool:
        found = False
        for r in self.records:
            if r.name == name and not r.revoked:
                r.revoked = True
                found = True
        if found:
            self.save()
        return found

    def list_public(self) -> list[dict]:
        return [r.public() for r in self.records]

    # ------------------------------------------------------------------ controle
    @property
    def open_mode(self) -> bool:
        """Aucune cle active : l'API est ouverte (local uniquement)."""
        return not self.active_records()

    def active_records(self) -> list[ApiKeyRecord]:
        return [r for r in self.records if not r.revoked] + list(self._env.values())

    def verify(self, secret: str | None) -> ApiKeyRecord | None:
        """Renvoie l'enregistrement si la cle est valide, sinon None."""
        if not secret:
            return None
        h = hash_key(secret.strip())
        for r in self.active_records():
            # comparaison a temps constant sur le hash : pas de fuite par le temps
            if secrets.compare_digest(r.key_hash, h):
                r.last_used_at = _now()
                if r in self.records:
                    self.save()
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


def summarize(store: ApiKeyStore) -> dict:
    """Etat du trousseau, pour /health (jamais de secret ni de hash complet)."""
    active = store.active_records()
    return {"auth": "ouvert" if store.open_mode else "cle_api",
            "cles_actives": len(active),
            "portees": sorted({r.scope for r in active}) or []}
