"""Servir le modele depuis Colab, ou vivent le GPU et les poids entraines.

Une seule fonction a appeler depuis un notebook :

    from service.colab import serve_public
    info = serve_public(config="colab")
    print(info["public_url"], info["api_key"])

Elle enchaine, dans cet ordre : verification des poids -> cle d'API -> tunnel public
(son nom d'hote doit etre connu AVANT le service, cf. service/tunnel.py) -> uvicorn en
sous-processus -> attente de /api/health.

Le service tourne en sous-processus et non dans le noyau du notebook : une cellule qui
plante ou un redemarrage de l'execution ne laisse pas un serveur a moitie vivant, et les
journaux restent lisibles dans un fichier.

Rappels qui comptent sur Colab :
  - la session est ephemere (inactivite ~90 min, 12 h au maximum) : l'URL publique meurt
    avec elle, il faut la repartager a chaque relance ;
  - un quick tunnel est PUBLIC : seule la cle d'API protege les donnees. Le mode ouvert
    sans cle (`PCFNO3D_INSECURE_OPEN`) n'est donc JAMAIS transmis au service lance ici :
    `service_env` le retire et pose `PCFNO3D_FORBID_OPEN`, que le service respecte ;
  - sans `best_dso_v7.pt` (ou `best.pt`) et `norm_stats.json` dans run_dir, il n'y a rien
    a servir : la fonction s'arrete avec un message clair plutot que de demarrer a vide.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _health(base_url: str, timeout: float = 5.0) -> dict | None:
    # un User-Agent explicite : Cloudflare refuse souvent celui de urllib par defaut
    req = urllib.request.Request(f"{base_url}/api/health",
                                 headers={"User-Agent": "pcfno3d-colab/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None


def wait_for_health(base_url: str, timeout: float = 300.0,
                    log_path: Path | None = None) -> dict:
    """Attend que le service reponde. Le premier demarrage lit la grille (127 Mo) et les
    poids : plusieurs minutes sur Colab, d'ou le delai genereux."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = _health(base_url)
        if body is not None:
            return body
        time.sleep(2.0)
    extra = ""
    if log_path and log_path.exists():
        extra = "\n--- journal du service ---\n" + log_path.read_text(
            encoding="utf-8", errors="replace")[-2500:]
    raise TimeoutError(f"{base_url} ne repond pas apres {timeout:.0f} s.{extra}")


def service_env(cfg, config: str, port: int, device: str | None = None) -> dict:
    """Environnement du sous-processus uvicorn.

    Le mode ouvert est retire et interdit : sur ce chemin le service finit derriere un
    tunnel public, une variable heritee du noyau ne doit pas pouvoir l'ouvrir.
    """
    from .auth import ENV_FORBID_OPEN, ENV_INSECURE_OPEN

    env = {**os.environ,
           "PCFNO3D_CONFIG": config,
           "PCFNO3D_PORT": str(port),
           "PCFNO3D_RUN_DIR": str(cfg.paths.run_dir),
           "PCFNO3D_DATA_DIR": str(cfg.paths.data_dir),
           "PCFNO3D_CACHE_DIR": str(cfg.paths.cache_dir),
           "PYTHONPATH": os.pathsep.join(
               [str(REPO_ROOT), str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    if env.pop(ENV_INSECURE_OPEN, None):
        print(f"{ENV_INSECURE_OPEN} ignore : jamais de mode ouvert sur le chemin Colab / "
              "tunnel")
    env[ENV_FORBID_OPEN] = "1"
    if device:
        env["PCFNO3D_DEVICE"] = device
    return env


def ensure_api_key(cfg, name: str = "colab", scope: str = "read",
                   quota_minute: int | None = None,
                   quota_jour: int | None = None) -> tuple[str, bool]:
    """Renvoie (secret, cree_maintenant).

    Un secret ne peut pas etre relu : si une cle de ce nom existe deja, on en cree une
    autre, suffixee, plutot que de pretendre retrouver l'ancienne.
    """
    from .auth import ApiKeyStore, default_store_path

    store = ApiKeyStore(default_store_path(cfg.paths.run_dir))
    candidate = name
    suffix = 2
    while any(r.name == candidate and not r.revoked for r in store.records):
        candidate = f"{name}-{suffix}"
        suffix += 1
    secret, _ = store.create(candidate, scope=scope, quota_minute=quota_minute,
                             quota_jour=quota_jour)
    return secret, candidate != name


def check_model_files(cfg) -> Path:
    """Verifie qu'il y a bien un modele a servir, et explique quoi faire sinon."""
    try:
        checkpoint = cfg.paths.checkpoint_for_inference()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\n\nIl faut un checkpoint entraine dans {cfg.paths.run_dir} :\n"
            "  - best_dso_v7.pt (raffinement DSO) ou best.pt (meilleur de l'entrainement)\n"
            "Sur Colab, c'est le dossier de run du Drive. Lance l'entrainement "
            "(`pcfno3d -c colab train`) ou corrige run_dir."
        ) from exc
    if not cfg.paths.norm_stats.exists():
        raise FileNotFoundError(
            f"{cfg.paths.norm_stats} absent : les statistiques de normalisation font "
            "partie du modele, sans elles la sortie ne peut pas etre denormalisee.\n"
            "Elles sont ecrites par `pcfno3d -c colab stats` (ou par l'entrainement)."
        )
    return checkpoint


def serve_public(config: str = "colab", port: int = 8000, key_name: str = "colab",
                 scope: str = "read", device: str | None = None, tunnel: bool = True,
                 build_days: list[int] | None = None, wait: float = 300.0,
                 log_path: str | Path = "/tmp/pcfno3d_service.log",
                 named_tunnel: dict | None = None,
                 key_hash: str | None = None,
                 quota_minute: int | None = None,
                 quota_jour: int | None = None) -> dict:
    """Demarre le service (API + MCP + interface) et l'expose par un tunnel public.

    Args:
        config: nom du YAML de configuration (`colab` sur Colab).
        port: port local du service.
        key_name: nom de la cle d'API creee pour ce demarrage.
        scope: `read` (lecture) ou `admin` (+ ecriture d'exports).
        device: `cuda` / `cpu` ; None = GPU si disponible.
        tunnel: False pour rester en local (utile pour tester sans exposer).
        build_days: jours a mettre en cache avant de servir, ex. [4, 11]. Sans cache
            d'instants, seules les routes qui partent d'un CSV de surface fonctionnent.
        wait: delai d'attente du premier /api/health.
        log_path: journal du service.
        named_tunnel: {"id", "hostname", "credentials"} pour un tunnel NOMME a URL fixe
            (au lieu d'un quick tunnel). `credentials` est secret : il n'est transmis
            qu'au processus cloudflared.
        key_hash: empreinte SHA-256 d'une cle remise de main a main. Elle est enregistree
            (portee `scope`, nom `key_name`) au lieu de creer une nouvelle cle : le
            secret ne passe jamais par Colab.
        quota_minute, quota_jour: quotas propres a cette cle (None = defaut du service,
            0 = illimite). Utile pour une cle partagee par beaucoup d'utilisateurs, comme
            celle du Worker atom.

    Returns:
        Un dictionnaire : public_url, api_key, mcp_url, health, checkpoint, process...
    """
    from pcfno3d.config import Config

    # memes chemins que le service lancera (PCFNO3D_RUN_DIR / _DATA_DIR / _CACHE_DIR) :
    # sans cela, on verifiait le checkpoint du YAML et on imposait ces chemins au service
    cfg = Config.load(config).apply_service_env()
    checkpoint = check_model_files(cfg)

    if build_days:
        from pcfno3d.cache import build_cache
        from pcfno3d.geometry import Geometry
        print(f"construction du cache pour les jours {build_days} "
              "(une fois ; sans cela, seules les requetes depuis un CSV de surface "
              "fonctionnent)")
        geom = Geometry(cfg, device="cpu")
        build_cache(cfg, geom, days=build_days)
        del geom

    if key_hash:
        from .auth import ApiKeyStore, default_store_path
        record = ApiKeyStore(default_store_path(cfg.paths.run_dir)).register_hash(
            key_name, key_hash, scope=scope, quota_minute=quota_minute, quota_jour=quota_jour)
        api_key, renamed = None, False
        print(f"cle '{record.name}' ({record.scope}) enregistree par son empreinte "
              f"{record.key_hash[:12]}... : le secret n'est pas sur cette machine")
    else:
        api_key, renamed = ensure_api_key(cfg, name=key_name, scope=scope,
                                          quota_minute=quota_minute, quota_jour=quota_jour)

    tun = None
    base_url = f"http://127.0.0.1:{port}"
    env = service_env(cfg, config, port, device)

    if tunnel and named_tunnel:
        from .tunnel import start_named_tunnel
        print(f"rattachement au tunnel nomme {named_tunnel['hostname']}...")
        tun = start_named_tunnel(named_tunnel["id"], named_tunnel["hostname"],
                                 named_tunnel["credentials"], port=port)
        base_url = tun.public_url
        env["PCFNO3D_ALLOWED_HOSTS"] = f"{tun.host},{tun.host}:443"
        print(f"tunnel : {tun.public_url} (URL fixe)")
    elif tunnel:
        from .tunnel import start_tunnel
        print("ouverture du tunnel public...")
        tun = start_tunnel(port=port)
        base_url = tun.public_url
        # le transport MCP refuse un Host non declare : il faut le nom du tunnel, d'ou
        # l'ordre tunnel puis service
        env["PCFNO3D_ALLOWED_HOSTS"] = f"{tun.host},{tun.host}:443"
        print(f"tunnel : {tun.public_url}")

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    with open(log_path, "ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "service.app:app",
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)

    try:
        # on attend le service EN LOCAL : verifier sa propre URL publique depuis Colab
        # echouait (2026-10-12) alors que le service repondait 200 de l'exterieur, et
        # l'echec faisait tout arreter. L'acces public se verifie ensuite, sans rien couper.
        health = wait_for_health(f"http://127.0.0.1:{port}", timeout=wait, log_path=log_path)
    except Exception:
        process.terminate()
        if tun:
            tun.stop()
        raise

    info = {
        "public_url": base_url,
        "api_key": api_key,
        "key_name": key_name if not renamed else f"{key_name} (renommee, nom deja pris)",
        "mcp_url": f"{base_url}/mcp",
        "web_app": f"{base_url}/app/",
        "api_docs": f"{base_url}/api/docs",
        "checkpoint": checkpoint.name,
        "health": health,
        "process": process,
        "tunnel": tun,
        "log_path": str(log_path),
    }
    print(summary(info))
    return info


def summary(info: dict) -> str:
    """Le texte a partager a l'application qui consomme les donnees."""
    if info["api_key"] is None:
        return "\n".join([
            "", "=" * 72, "SERVICE PC-FNO3D EN LIGNE (tunnel nomme)", "=" * 72,
            f"  URL fixe        : {info['public_url']}",
            f"  sante (public)  : {info['public_url']}/api/health",
            f"  serveur MCP     : {info['mcp_url']}",
            f"  modele          : {info['checkpoint']}  "
            f"({(info.get('health') or {}).get('device', '?')})",
            "  cle d'API       : celle remise de main a main (non affichee : seule son",
            "                    empreinte est sur cette machine)",
            "=" * 72])
    base = info["public_url"]
    health = info.get("health") or {}
    lines = [
        "", "=" * 72,
        "SERVICE PC-FNO3D EN LIGNE",
        "=" * 72,
        f"  application web : {info['web_app']}",
        f"  API             : {info['api_docs']}",
        f"  serveur MCP     : {info['mcp_url']}",
        f"  modele          : {info['checkpoint']}  ({health.get('device', '?')})",
        f"  cle d'API       : {info['api_key']}",
        "",
        "  Essai :",
        f"    curl -H \"X-API-Key: {info['api_key']}\" \\",
        f"      \"{base}/api/point?x=5000&y=5000&z=120\"",
        "",
        "  Declaration MCP cote client :",
        "    {\"mcpServers\": {\"pcfno3d\": {\"type\": \"http\",",
        f"      \"url\": \"{info['mcp_url']}\",",
        f"      \"headers\": {{\"X-API-Key\": \"{info['api_key']}\"}}}}}}",
        "",
        "  ATTENTION : l'URL est publique et la session Colab est ephemere.",
        "  C'est la cle d'API qui protege les donnees. Ne la mets pas dans un depot,",
        "  ni dans le salon des agents. A la prochaine execution, tout change.",
        "=" * 72,
    ]
    return "\n".join(lines)


def stop(info: dict) -> None:
    """Arrete le service et le tunnel."""
    process = info.get("process")
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
    tun = info.get("tunnel")
    if tun is not None:
        tun.stop()
    print("service et tunnel arretes")


def tail_log(info: dict, lines: int = 40) -> str:
    path = Path(info.get("log_path", ""))
    if not path.exists():
        return "(aucun journal)"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
