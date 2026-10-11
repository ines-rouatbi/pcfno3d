"""Lien vers une instance distante : l'application locale sert les donnees de Colab.

Le probleme a resoudre : le modele entraine tourne sur Colab, derriere un tunnel dont
l'URL change a chaque relance. Une application qui pointe directement dessus casse au
premier redemarrage.

En mode lien, le service local ne charge aucun modele : il transmet `/api/...` et `/mcp`
a l'instance distante. L'adresse que tout le monde utilise reste stable
(`http://127.0.0.1:8000`), et quand le tunnel Colab change, une seule commande est a
refaire :

    pcfno3d link set https://xxx.trycloudflare.com --key pcf_...
    pcfno3d serve

Deux cles, et c'est voulu :

    navigateur --(cle LOCALE)--> service local --(cle AMONT)--> Colab

La cle de l'instance Colab reste sur cette machine : le navigateur ne la voit jamais, et
revoquer une cle locale ne touche pas a l'amont.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# importes au niveau du module, et non dans install() : avec
# `from __future__ import annotations` les annotations sont des chaines, que FastAPI
# resout dans les globales du module. Importes localement, `Request` et `Response`
# seraient introuvables et FastAPI prendrait `request` pour un parametre de requete.
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

# en-tetes propres au saut local : ils ne doivent pas etre retransmis tels quels
HOP_BY_HOP = {"host", "connection", "keep-alive", "proxy-authenticate",
              "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade",
              "content-length", "authorization", "x-api-key", "cookie"}
TIMEOUT_S = 180.0      # une reconstruction sur un GPU partage peut etre lente


@dataclass
class Upstream:
    """L'instance distante a laquelle le service local est relie."""

    url: str
    key: str | None = None
    label: str = "colab"

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"URL amont invalide : {self.url!r} (attendu http(s)://...)")

    # ------------------------------------------------------------------ stockage
    @staticmethod
    def path_for(run_dir: str | Path) -> Path:
        return Path(run_dir) / "upstream.json"

    def save(self, run_dir: str | Path) -> Path:
        path = self.path_for(run_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"url": self.url, "key": self.key, "label": self.label},
                                  indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, run_dir: str | Path) -> "Upstream | None":
        path = cls.path_for(run_dir)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("url"):
            return None
        return cls(url=data["url"], key=data.get("key"),
                   label=data.get("label", "colab"))

    @classmethod
    def clear(cls, run_dir: str | Path) -> bool:
        path = cls.path_for(run_dir)
        if path.exists():
            path.unlink()
            return True
        return False

    @classmethod
    def resolve(cls, run_dir: str | Path) -> "Upstream | None":
        """L'environnement a la priorite sur le fichier (pratique en conteneur)."""
        url = (os.environ.get("PCFNO3D_UPSTREAM") or "").strip()
        if url:
            return cls(url=url, key=(os.environ.get("PCFNO3D_UPSTREAM_KEY") or "").strip()
                       or None, label=os.environ.get("PCFNO3D_UPSTREAM_LABEL") or "amont")
        return cls.load(run_dir)

    # ------------------------------------------------------------------ vue publique
    def public(self) -> dict:
        """Jamais la cle amont : elle ne doit pas sortir de cette machine."""
        return {"url": self.url, "label": self.label, "cle_amont": bool(self.key)}


def forward_headers(incoming, extra_key: str | None) -> dict[str, str]:
    """En-tetes a transmettre : on retire le saut local et on pose la cle amont."""
    out = {k: v for k, v in incoming.items() if k.lower() not in HOP_BY_HOP}
    if extra_key:
        out["X-API-Key"] = extra_key
    return out


async def check(upstream: Upstream, timeout: float = 20.0) -> dict:
    """Interroge /api/health de l'amont. Ne leve pas : renvoie l'erreur lisible."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{upstream.url}/api/health")
            r.raise_for_status()
            return {"joignable": True, **r.json()}
    except Exception as exc:                      # noqa: BLE001 — diagnostic, pas de controle
        return {"joignable": False, "erreur": f"{type(exc).__name__}: {exc}"}


def install(app, upstream: Upstream) -> None:
    """Monte les routes de transmission sur l'application.

    `/api/health` reste local (il decrit le lien lui-meme) ; tout le reste de `/api`, et
    `/mcp`, partent vers l'amont. L'application web n'a rien a savoir : elle appelle les
    memes chemins relatifs.
    """
    import httpx

    async def relay(request: Request, chemin: str) -> Response:
        cible = f"{upstream.url}/{chemin.lstrip('/')}"
        body = await request.body()
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
                amont = await client.request(
                    request.method, cible,
                    params=dict(request.query_params),
                    headers=forward_headers(request.headers, upstream.key),
                    content=body or None)
        except httpx.TimeoutException as exc:
            raise HTTPException(504, f"l'amont ({upstream.label}) n'a pas repondu a "
                                     f"temps : {exc}") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                502, f"l'amont ({upstream.label}) est injoignable : {exc}. "
                     "Si c'est une session Colab, son URL a peut-etre change : "
                     "`pcfno3d link set <nouvelle URL> --key <cle>`") from exc

        garde = {k: v for k, v in amont.headers.items()
                 if k.lower() in {"content-type", "cache-control", "mcp-session-id"}}
        return Response(content=amont.content, status_code=amont.status_code, headers=garde)

    @app.api_route("/api/{chemin:path}", methods=["GET", "POST", "PUT", "DELETE"],
                   include_in_schema=False)
    async def relais_api(request: Request, chemin: str) -> Response:
        if chemin == "health":                    # traite localement, voir create_app
            raise HTTPException(404)
        return await relay(request, f"api/{chemin}")

    @app.api_route("/mcp", methods=["GET", "POST", "DELETE"], include_in_schema=False)
    @app.api_route("/mcp/", methods=["GET", "POST", "DELETE"], include_in_schema=False)
    async def relais_mcp(request: Request) -> Response:
        return await relay(request, "mcp")

    @app.get("/api/lien", include_in_schema=False)
    async def etat_du_lien() -> JSONResponse:
        return JSONResponse({"mode": "lien", "amont": upstream.public(),
                             "sante_amont": await check(upstream)})
