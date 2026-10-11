"""Tunnel public pour exposer le service depuis une machine sans adresse publique.

Cas d'usage : le modele entraine et le GPU sont sur Colab, mais l'application qui
consomme les donnees est ailleurs. Un tunnel Cloudflare « quick tunnel » donne une URL
https publique en quelques secondes, sans compte ni configuration.

Le tunnel est demarre AVANT le service, parce que son nom d'hote est tire au hasard et
que le transport MCP refuse un en-tete Host qu'on ne lui a pas declare : il faut donc
connaitre le nom avant de lancer uvicorn.

    from service.tunnel import start_tunnel
    tunnel = start_tunnel(port=8000)
    print(tunnel.public_url)       # https://xxx-yyy-zzz.trycloudflare.com

Un quick tunnel est PUBLIC : l'URL suffit a atteindre le service. C'est la cle d'API qui
protege les donnees, pas l'obscurite de l'URL.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CLOUDFLARED_URLS = {
    ("Linux", "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    ("Linux", "aarch64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
    ("Darwin", "arm64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64",
    ("Darwin", "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64",
    ("Windows", "AMD64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
}
URL_PATTERN = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")


def find_url(text: str) -> str | None:
    """Extrait l'URL publique de la sortie de cloudflared."""
    match = URL_PATTERN.search(text or "")
    return match.group(0) if match else None


def ensure_cloudflared(dest_dir: str | Path = "/tmp") -> Path:
    """Renvoie le chemin du binaire cloudflared, en le telechargeant au besoin."""
    existing = shutil.which("cloudflared")
    if existing:
        return Path(existing)

    key = (platform.system(), platform.machine())
    url = CLOUDFLARED_URLS.get(key)
    if url is None:
        raise RuntimeError(
            f"cloudflared n'est pas installe et aucun binaire connu pour {key}. "
            "Installe-le a la main (https://github.com/cloudflare/cloudflared) ou utilise "
            "un autre tunnel."
        )
    dest = Path(dest_dir) / ("cloudflared.exe" if key[0] == "Windows" else "cloudflared")
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        urllib.request.urlretrieve(url, tmp)     # noqa: S310 — URL fixe, pas une entree utilisateur
        tmp.replace(dest)
        dest.chmod(0o755)
    return dest


@dataclass
class Tunnel:
    """Un tunnel en cours d'execution."""

    public_url: str
    host: str
    process: subprocess.Popen
    log_path: Path

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()

    @property
    def alive(self) -> bool:
        return self.process.poll() is None


def start_tunnel(port: int = 8000, timeout: float = 60.0,
                 log_path: str | Path = "/tmp/cloudflared.log",
                 binary_dir: str | Path = "/tmp") -> Tunnel:
    """Demarre un quick tunnel vers `port` et attend que l'URL publique apparaisse.

    Le service local n'a pas besoin d'etre deja demarre : cloudflared se contente de
    transmettre, et renverra une erreur de passerelle en attendant.
    """
    binary = ensure_cloudflared(binary_dir)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    with open(log_path, "ab") as log:
        process = subprocess.Popen(
            [str(binary), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "TUNNEL_METRICS": "127.0.0.1:0"},
        )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"cloudflared s'est arrete (code {process.returncode}). "
                f"Journal : {log_path}\n"
                + log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            )
        url = find_url(log_path.read_text(encoding="utf-8", errors="replace"))
        if url:
            return Tunnel(public_url=url, host=url.split("://", 1)[1], process=process,
                          log_path=log_path)
        time.sleep(0.5)

    process.terminate()
    raise TimeoutError(
        f"aucune URL publique apres {timeout:.0f} s. Journal : {log_path}\n"
        + log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
    )
