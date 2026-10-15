"""Worker ATOM simule (tests du lot M6) : la partie publication de src/v1.js (branche
agent/rayen-audit-atom/f1-r2), en Python, sur http.server, avec injection de pannes.

Regles reprises du Worker : jeton Bearer (401 sinon) ; PUT idempotent si meme sha256,
refuse apres commit (409) ; X-Content-SHA256 obligatoire (422) ; 8 Mo maximum (413) ;
sha256 du corps verifie (422) ; commit du modele : meta.json et terrain.f32 listes, taille
et sha256 de chaque fichier, terrain nx.ny.4 octets ; commit de l'instantane : modele
commite, snapshot.json present, ni meta.json ni terrain.f32, taille de chaque champ
nz.ny.nx.octets, puis taille et sha256 de chaque fichier ; ordre d'ecriture _commit, index,
latest EN DERNIER, et latest ne recule jamais (valid_time). Lecture : /v1/latest,
/v1/times, /v1/meta/{sha8}.

Pannes : `couper(predicat, apres_traitement=False)` ferme la connexion SANS reponse pour la
prochaine requete qui satisfait le predicat (avant ou apres l'avoir traitee).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BYTES = 8 * 1024 * 1024
ID_RE = re.compile(r"^[0-9A-Za-z_]+_[0-9a-f]{8}$")
SHA8_RE = re.compile(r"^[0-9a-f]{8}$")
PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)*$")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Stockage:
    """R2 en memoire : cle -> (octets, sha256)."""

    def __init__(self) -> None:
        self.objets: dict[str, tuple[bytes, str | None]] = {}
        self.lock = threading.Lock()

    def put(self, key: str, data: bytes, sha: str | None = None) -> None:
        self.objets[key] = (data, sha)

    def put_json(self, key: str, doc) -> None:
        self.put(key, json.dumps(doc).encode("utf-8"))

    def json(self, key: str):
        o = self.objets.get(key)
        return json.loads(o[0].decode("utf-8")) if o else None


class Worker:
    def __init__(self, jeton: str) -> None:
        self.jeton = jeton
        self.r2 = Stockage()
        self.journal: list[tuple[str, str, int]] = []
        self.pannes: list[tuple] = []
        self.heartbeats = 0

    def couper(self, predicat, apres_traitement: bool = False) -> None:
        self.pannes.append((predicat, apres_traitement))

    def _panne(self, methode: str, chemin: str):
        for p in list(self.pannes):
            if p[0](methode, chemin):
                self.pannes.remove(p)
                return p
        return None

    # -------------------------------------------------------------- logique
    def traiter(self, methode: str, chemin: str, entetes, corps: bytes) -> tuple[int, dict]:
        parts = [urllib.parse.unquote(p) for p in chemin.split("?")[0].split("/") if p]
        if parts[:1] != ["v1"]:
            return 404, {"error": "route_inconnue"}
        parts = parts[1:]
        with self.r2.lock:
            if parts and (parts[0] == "publish" or (parts[0] == "heartbeat"
                                                    and methode == "POST")):
                return self._publication(methode, parts, entetes, corps)
            return self._lecture(parts)

    def _lecture(self, parts) -> tuple[int, dict]:
        if parts == ["latest"]:
            doc = self.r2.json("latest.json")
            return (200, doc) if doc else (404, {"error": "absent"})
        if parts == ["times"]:
            items = self.r2.json("index/times.json") or []
            return 200, {"total": len(items), "next": None, "items": items}
        if len(parts) == 2 and parts[0] == "meta":
            if not self.r2.json(f"models/{parts[1]}/_commit.json"):
                return 404, {"error": "modele_inconnu"}
            return 200, self.r2.json(f"models/{parts[1]}/meta.json")
        return 404, {"error": "route_inconnue"}

    def _publication(self, methode, parts, entetes, corps) -> tuple[int, dict]:
        h = entetes.get("authorization") or ""
        if not h.startswith("Bearer ") or h[7:] != self.jeton:
            return 401, {"error": "non_autorise"}
        if parts[0] == "heartbeat":
            self.heartbeats += 1
            self.r2.put_json("heartbeat.json", {"at": _now()})
            return 204, {}
        if len(parts) > 1 and parts[1] == "model":
            sha8 = parts[2] if len(parts) > 2 else ""
            if not SHA8_RE.match(sha8):
                return 422, {"error": "sha8_invalide"}
            commite = f"models/{sha8}/_commit.json" in self.r2.objets
            if len(parts) == 4 and parts[3] == "commit" and methode == "POST":
                return self._commit_modele(sha8, commite, corps)
            if len(parts) >= 5 and parts[3] == "files" and methode == "PUT":
                return self._put(f"models/{sha8}", commite, "/".join(parts[4:]), entetes, corps)
            return 405, {"error": "methode"}
        sid = parts[1] if len(parts) > 1 else ""
        if not ID_RE.match(sid):
            return 422, {"error": "id_invalide"}
        commite = f"snapshots/{sid}/_commit.json" in self.r2.objets
        if len(parts) == 3 and parts[2] == "commit" and methode == "POST":
            return self._commit_instantane(sid, commite, corps)
        if len(parts) >= 4 and parts[2] == "files" and methode == "PUT":
            return self._put(f"snapshots/{sid}", commite, "/".join(parts[3:]), entetes, corps)
        return 405, {"error": "methode"}

    def _put(self, prefixe, commite, rel, entetes, corps) -> tuple[int, dict]:
        if not PATH_RE.match(rel):
            return 422, {"error": "chemin_invalide"}
        declare = (entetes.get("x-content-sha256") or "").lower()
        if not re.match(r"^[0-9a-f]{64}$", declare):
            return 422, {"error": "sha256_absent"}
        if int(entetes.get("content-length") or 0) > MAX_BYTES:
            return 413, {"error": "trop_gros"}
        key = f"{prefixe}/{rel}"
        ex = self.r2.objets.get(key)
        if ex and ex[1] == declare:
            return 200, {"ok": True, "path": rel, "unchanged": True}
        if commite:
            return 409, {"error": "deja_commite"}
        if len(corps) > MAX_BYTES:
            return 413, {"error": "trop_gros"}
        if _sha(corps) != declare:
            return 422, {"error": "sha256_different"}
        self.r2.put(key, corps, declare)
        return 201, {"ok": True, "path": rel, "bytes": len(corps)}

    def _verifier(self, prefixe, files) -> list[str]:
        pb = []
        for f in files:
            o = self.r2.objets.get(f"{prefixe}/{f['path']}")
            if o is None:
                pb.append(f"{f['path']} : absent")
            elif len(o[0]) != f["bytes"]:
                pb.append(f"{f['path']} : taille")
            elif o[1] != f["sha256"]:
                pb.append(f"{f['path']} : sha256 different")
        return pb

    def _commit_modele(self, sha8, commite, corps) -> tuple[int, dict]:
        if commite:
            return 200, {"ok": True, "model_sha8": sha8, "unchanged": True}
        m = json.loads(corps.decode("utf-8"))
        if m.get("model_sha8") != sha8 or not m["versions"]["model_sha256"].startswith(sha8):
            return 422, {"error": "manifeste_invalide"}
        listes = {f["path"] for f in m["files"]}
        pb = [f"{p} : absent du manifeste" for p in ("meta.json", "terrain.f32")
              if p not in listes]
        pb += self._verifier(f"models/{sha8}", m["files"]) if not pb else []
        if not pb:
            meta = self.r2.json(f"models/{sha8}/meta.json")
            g = meta["grid"]
            terr = self.r2.objets[f"models/{sha8}/terrain.f32"]
            if len(terr[0]) != g["nx"] * g["ny"] * 4:
                pb.append("terrain.f32 : taille")
            if meta["terrain"]["sha256"] != terr[1]:
                pb.append("terrain.f32 : sha256 != meta")
        if pb:
            return 422, {"error": "publication_refusee", "problems": pb}
        self.r2.put_json(f"models/{sha8}/_commit.json", {**m, "committed_at": _now()})
        return 200, {"ok": True, "model_sha8": sha8, "files": len(m["files"])}

    def _commit_instantane(self, sid, commite, corps) -> tuple[int, dict]:
        if commite:
            return 200, {"ok": True, "snapshot_id": sid, "unchanged": True}
        m = json.loads(corps.decode("utf-8"))
        if m.get("snapshot_id") != sid or not m.get("files") or m.get("data_status") not in (
                "complete", "degrade"):
            return 422, {"error": "manifeste_invalide"}
        sha8 = sid[-8:]
        if m["versions"]["model_sha256"][:8] != sha8:
            return 422, {"error": "manifeste_invalide"}
        if not self.r2.json(f"models/{sha8}/_commit.json"):
            return 422, {"error": "modele_non_commite"}
        meta = self.r2.json(f"models/{sha8}/meta.json")
        par = {f["path"]: f for f in m["files"]}
        pb = [f"{p} : interdit" for p in ("meta.json", "terrain.f32") if p in par]
        if "snapshot.json" not in par:
            pb.append("snapshot.json : absent du manifeste")
        g = meta["grid"]
        for v in meta["fields"]:
            enc = meta["encoding"][v]
            n = g["nx"] * g["ny"] * g["nz"] * (1 if enc["dtype"] == "uint8" else 4)
            f = par.get(enc["file"])
            if f is None:
                pb.append(f"{enc['file']} : absent du manifeste")
            elif f["bytes"] != n:
                pb.append(f"{enc['file']} : taille declaree")
        if pb:
            return 422, {"error": "publication_refusee", "problems": pb}
        pb = self._verifier(f"snapshots/{sid}", m["files"])
        if pb:
            return 422, {"error": "publication_refusee", "problems": pb}
        publie = _now()
        latest = {"snapshot_id": sid, "valid_time": m["valid_time"],
                  "generated_at": m.get("generated_at") or publie, "published_at": publie,
                  "last_observation_at": m.get("last_observation_at"),
                  "data_status": m["data_status"], "versions": m["versions"],
                  "href": f"/v1/snapshots/{sid}"}
        self.r2.put_json(f"snapshots/{sid}/_commit.json", {**m, "published_at": publie})
        alertes = self.r2.json(f"snapshots/{sid}/alerts.json")
        ouvertes = sum(a["status"] == "ouverte" for a in alertes["items"]) \
            if alertes is not None else None
        times = [t for t in (self.r2.json("index/times.json") or []) if t["snapshot_id"] != sid]
        times.append({"snapshot_id": sid, "valid_time": m["valid_time"],
                      "open_alerts": ouvertes})
        times.sort(key=lambda t: t["valid_time"], reverse=True)
        self.r2.put_json("index/times.json", times)
        courant = self.r2.json("latest.json")
        if not courant or courant["valid_time"] <= m["valid_time"]:
            self.r2.put_json("latest.json", latest)
        return 200, {"ok": True, "snapshot_id": sid, "published_at": publie,
                     "files": len(m["files"])}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    worker: Worker = None

    def log_message(self, *a) -> None:              # silencieux
        pass

    def _repondre(self) -> None:
        n = int(self.headers.get("content-length") or 0)
        corps = self.rfile.read(n) if n else b""
        entetes = {k.lower(): v for k, v in self.headers.items()}
        panne = self.worker._panne(self.command, self.path)
        if panne is not None and not panne[1]:
            self._couper()
            return
        status, doc = self.worker.traiter(self.command, self.path, entetes, corps)
        self.worker.journal.append((self.command, self.path, status))
        if panne is not None:
            self._couper()                          # traite, mais la reponse est perdue
            return
        data = b"" if status == 204 else json.dumps(doc).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _couper(self) -> None:
        self.close_connection = True
        try:
            self.connection.shutdown(2)
        except OSError:
            pass

    do_GET = do_PUT = do_POST = _repondre


class ServeurSimule:
    """Worker simule sur 127.0.0.1:<port libre>, dans un thread."""

    def __init__(self, jeton: str) -> None:
        self.worker = Worker(jeton)
        handler = type("H", (_Handler,), {"worker": self.worker})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "ServeurSimule":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
