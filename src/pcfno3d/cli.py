"""Point d'entree en ligne de commande : `python -m pcfno3d <commande>`.

    build-cache   CSV journaliers -> tenseurs .npy + index.csv        (BLOC 3)
    stats         statistiques de normalisation                       (BLOC 5)
    check         controle de la verite terrain et de la geometrie    (BLOC 2/4)
    train         entrainement, reprise automatique                   (BLOC 9)
    evaluate      metriques de test et verdict DSO1-DSO5              (BLOC 10)
    reconstruct   reconstruction 3D d'un instant (npz, csv, figures)  (BLOC 12)
    export-web    volume -> binaires + manifeste pour le visualiseur
    serve         application web + API JSON + serveur MCP (fastapi + uvicorn)
    apikey        creer / lister / revoquer les cles d'API du service
    link          relier le service local a une instance distante (Colab)
    mcp           serveur MCP en transport stdio (client local)
    info          configuration effective et etat des artefacts

Toute commande accepte `-c/--config` (nom dans configs/ ou chemin) et `--set a.b=valeur`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config


def _parse_set(items: list[str] | None) -> dict:
    """--set train.epochs=2 --set model.width=8 -> {"train.epochs": 2, ...}"""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set attend cle=valeur, recu : {item}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out[key.strip()] = value
    return out


def _load_cfg(args) -> Config:
    cfg = Config.load(args.config)
    cfg.apply_overrides(_parse_set(args.set))
    for attr, key in (("data_dir", "paths.data_dir"), ("cache_dir", "paths.cache_dir"),
                      ("run_dir", "paths.run_dir")):
        value = getattr(args, attr, None)
        if value:
            cfg.apply_overrides({key: Path(value)})
    return cfg


def _geometry(cfg: Config, device: str | None = None):
    from .config import pick_device
    from .geometry import Geometry
    return Geometry(cfg, device=pick_device(device))


# ------------------------------------------------------------------ commandes
def cmd_build_cache(args) -> int:
    from .cache import build_cache

    cfg = _load_cfg(args)
    geom = _geometry(cfg, "cpu")
    print(geom.summary())
    days = [int(d) for d in args.days.split(",")] if args.days else None
    build_cache(cfg, geom, days=days)
    return 0


def cmd_stats(args) -> int:
    from .cache import load_index
    from .normalize import Normalizer

    cfg = _load_cfg(args)
    geom = _geometry(cfg, "cpu")
    Normalizer.load_or_compute(cfg, geom, load_index(cfg))
    print(f"-> {cfg.paths.norm_stats}")
    return 0


def cmd_check(args) -> int:
    from .cache import check_ground_truth

    cfg = _load_cfg(args)
    geom = _geometry(cfg, "cpu")
    print(geom.summary())
    print(f"niveaux ASL : {geom.z_levels.tolist()}")
    print(f"couverture par niveau : {[round(float(c), 3) for c in geom.cover]}")
    print(f"niveaux evalues (DSO) : {[int(geom.z_levels[k]) for k in geom.eval_levels()]}")
    if cfg.paths.index_csv.exists():
        for row in check_ground_truth(cfg, geom):
            print(row)
    else:
        print(f"(pas encore de cache : {cfg.paths.index_csv})")
    return 0


def cmd_train(args) -> int:
    from .train import train

    cfg = _load_cfg(args)
    train(cfg, device=args.device)
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate

    cfg = _load_cfg(args)
    evaluate(cfg, checkpoint=args.checkpoint, device=args.device)
    return 0


def cmd_reconstruct(args) -> int:
    from .reconstruct import reconstruct

    cfg = _load_cfg(args)
    reconstruct(cfg, stem=args.stem, csv=args.csv, checkpoint=args.checkpoint,
                keep=args.keep, altitude=args.altitude, section_row=args.section_row,
                figures=not args.no_figures, web=args.web, device=args.device)
    return 0


def cmd_export_web(args) -> int:
    from .export import export_web
    from .inference import WindFieldModel

    cfg = _load_cfg(args)
    wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device)
    volume = wf.predict_csv(args.csv) if args.csv else wf.predict_stem(args.stem)
    out = export_web(volume, args.out or (cfg.paths.run_dir / "web" / "latest"),
                     stride=args.stride)
    print(f"export web -> {out}")
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn absent : pip install -r service/requirements.txt", file=sys.stderr)
        return 1
    import os

    cfg = _load_cfg(args)
    os.environ.setdefault("PCFNO3D_CONFIG", str(args.config or "default"))
    os.environ.setdefault("PCFNO3D_RUN_DIR", str(cfg.paths.run_dir))
    os.environ.setdefault("PCFNO3D_DATA_DIR", str(cfg.paths.data_dir))
    os.environ.setdefault("PCFNO3D_CACHE_DIR", str(cfg.paths.cache_dir))
    if args.device:
        os.environ.setdefault("PCFNO3D_DEVICE", args.device)
    # le garde-fou anti-DNS-rebinding du MCP compare l'en-tete Host, port inclus
    os.environ["PCFNO3D_PORT"] = str(args.port)
    if args.host not in ("127.0.0.1", "localhost", "0.0.0.0"):
        hosts = os.environ.get("PCFNO3D_ALLOWED_HOSTS", "")
        extra = f"{args.host},{args.host}:{args.port}"
        os.environ["PCFNO3D_ALLOWED_HOSTS"] = f"{hosts},{extra}".strip(",")

    store = _key_store(cfg)
    if store.open_mode:
        print("ATTENTION : aucune cle d'API definie, le service repond sans "
              "authentification.\n            `pcfno3d apikey create <nom>` pour en creer une.")
    print(f"application web : http://{args.host}:{args.port}/app/")
    print(f"API             : http://{args.host}:{args.port}/api/docs")
    print(f"serveur MCP     : http://{args.host}:{args.port}/mcp")
    uvicorn.run("service.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _key_store(cfg: Config):
    """Le trousseau vit dans run_dir, a cote des checkpoints."""
    from service.auth import ApiKeyStore, default_store_path

    return ApiKeyStore(default_store_path(cfg.paths.run_dir))


def cmd_apikey(args) -> int:
    if args.action in ("create", "revoke") and not args.name:
        print(f"`apikey {args.action}` attend un nom de cle", file=sys.stderr)
        return 1
    cfg = _load_cfg(args)
    try:
        store = _key_store(cfg)
    except ImportError:
        print("le module `service` n'est pas importable : lance la commande depuis la "
              "racine du depot", file=sys.stderr)
        return 1

    if args.action == "create":
        try:
            secret, record = store.create(args.name, scope=args.scope)
        except ValueError as exc:
            print(f"erreur : {exc}", file=sys.stderr)
            return 1
        print(f"cle creee  : {record.name} (portee {record.scope})")
        print(f"secret     : {secret}")
        print("\nIl n'est affiche qu'une fois : seul son hash est conserve dans")
        print(f"{store.path}")
        print("\nUsage : curl -H \"X-API-Key: <secret>\" http://127.0.0.1:8000/api/meta")
        return 0

    if args.action == "revoke":
        if store.revoke(args.name):
            print(f"cle revoquee : {args.name}")
            return 0
        print(f"aucune cle active nommee '{args.name}'", file=sys.stderr)
        return 1

    rows = store.list_public()
    if not rows:
        print(f"aucune cle dans {store.path} (le service repond sans authentification)")
        return 0
    print(f"{'nom':20s} {'portee':8s} {'etat':9s} {'empreinte':14s} derniere utilisation")
    for r in rows:
        etat = "revoquee" if r["revoked"] else "active"
        print(f"{r['name']:20s} {r['scope']:8s} {etat:9s} {r['empreinte']:14s} "
              f"{r['last_used_at'] or '-'}")
    return 0


def cmd_link(args) -> int:
    """Relier le service local a une instance distante (Colab, serveur...)."""
    cfg = _load_cfg(args)
    try:
        from service.link import Upstream
    except ImportError:
        print("lance la commande depuis la racine du depot", file=sys.stderr)
        return 1

    if args.action == "set":
        if not args.url:
            print("`link set` attend une URL, ex. https://xxx.trycloudflare.com",
                  file=sys.stderr)
            return 1
        try:
            upstream = Upstream(url=args.url, key=args.key, label=args.label)
        except ValueError as exc:
            print(f"erreur : {exc}", file=sys.stderr)
            return 1
        path = upstream.save(cfg.paths.run_dir)
        print(f"lien enregistre : {upstream.url} ({upstream.label})")
        print(f"  cle amont : {'oui' if upstream.key else 'non (amont en acces ouvert ?)'}")
        print(f"  fichier   : {path}")
        if not args.no_check:
            import asyncio

            from service.link import check
            print("\nverification de l'amont...")
            state = asyncio.run(check(upstream))
            if state.get("joignable"):
                print(f"  joignable : modele {state.get('modele')} sur "
                      f"{state.get('device')} | auth {state.get('auth')}")
            else:
                print(f"  INJOIGNABLE : {state.get('erreur')}")
                print("  (le lien est quand meme enregistre : relance la verification "
                      "avec `pcfno3d link show`)")
        print("\n`pcfno3d serve` servira desormais les donnees de l'amont.")
        return 0

    if args.action == "clear":
        if Upstream.clear(cfg.paths.run_dir):
            print("lien supprime : le service rechargera le modele local")
            return 0
        print("aucun lien enregistre")
        return 0

    upstream = Upstream.resolve(cfg.paths.run_dir)
    if upstream is None:
        print(f"aucun lien ({Upstream.path_for(cfg.paths.run_dir)})")
        print("le service charge le modele local")
        return 0
    import asyncio

    from service.link import check
    print(f"lien      : {upstream.url} ({upstream.label})")
    print(f"cle amont : {'oui' if upstream.key else 'non'}")
    state = asyncio.run(check(upstream))
    if state.get("joignable"):
        print(f"etat      : joignable | modele {state.get('modele')} sur "
              f"{state.get('device')} | auth {state.get('auth')}")
    else:
        print(f"etat      : INJOIGNABLE — {state.get('erreur')}")
        print("            une URL de tunnel Colab change a chaque relance : "
              "`pcfno3d link set <nouvelle URL> --key <cle>`")
    return 0


def cmd_mcp(args) -> int:
    """Serveur MCP en transport stdio (client local, ex. Claude Desktop)."""
    import os

    cfg = _load_cfg(args)
    os.environ.setdefault("PCFNO3D_CONFIG", str(args.config or "default"))
    os.environ.setdefault("PCFNO3D_RUN_DIR", str(cfg.paths.run_dir))
    os.environ.setdefault("PCFNO3D_DATA_DIR", str(cfg.paths.data_dir))
    os.environ.setdefault("PCFNO3D_CACHE_DIR", str(cfg.paths.cache_dir))
    try:
        from service.mcp_server import main as mcp_main
    except ImportError as exc:
        print(f"MCP indisponible ({exc}) : pip install -r service/requirements.txt",
              file=sys.stderr)
        return 1
    return mcp_main()


def cmd_info(args) -> int:
    cfg = _load_cfg(args)
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    print("\netat des artefacts :")
    checks = [("dataset", cfg.paths.data_dir / "meta.json"),
              ("index du cache", cfg.paths.index_csv),
              ("statistiques", cfg.paths.norm_stats),
              ("checkpoint best", cfg.paths.ckpt_best),
              ("checkpoint DSO", cfg.paths.ckpt_best_dso),
              ("checkpoint last", cfg.paths.ckpt_last)]
    for label, path in checks:
        mark = "present" if path.exists() else "absent "
        print(f"  [{mark}] {label:18s} {path}")
    return 0


# ------------------------------------------------------------------ parseur
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pcfno3d", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", help="nom dans configs/ (ex. test) ou chemin d'un YAML")
    p.add_argument("--set", action="append", metavar="CLE=VALEUR",
                   help="surcharge, ex. --set train.epochs=2")
    p.add_argument("--data-dir"), p.add_argument("--cache-dir"), p.add_argument("--run-dir")
    sub = p.add_subparsers(dest="commande", required=True)

    s = sub.add_parser("build-cache", help="CSV journaliers -> tenseurs .npy")
    s.add_argument("--days", help="liste de jours, ex. 1,2,3 (defaut : tous)")
    s.set_defaults(func=cmd_build_cache)

    s = sub.add_parser("stats", help="statistiques de normalisation")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("check", help="controle geometrie + verite terrain")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("train", help="entrainement (reprise automatique)")
    s.add_argument("--device")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="metriques de test et verdict DSO")
    s.add_argument("--checkpoint"), s.add_argument("--device")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("reconstruct", help="reconstruction 3D d'un instant")
    s.add_argument("--stem", help='instant en cache, ex. "004_120000"')
    s.add_argument("--csv", help="CSV de mesures de surface d'un instant")
    s.add_argument("--checkpoint"), s.add_argument("--device")
    s.add_argument("--keep", type=float, default=1.0, help="fraction de capteurs utilises")
    s.add_argument("--altitude", type=float, default=120.0, help="altitude ASL de la carte")
    s.add_argument("--section-row", type=int, help="ligne iy de la coupe verticale")
    s.add_argument("--no-figures", action="store_true")
    s.add_argument("--web", action="store_true", help="exporte aussi pour le visualiseur")
    s.set_defaults(func=cmd_reconstruct)

    s = sub.add_parser("export-web", help="volume -> binaires + manifeste")
    s.add_argument("--stem"), s.add_argument("--csv"), s.add_argument("--out")
    s.add_argument("--checkpoint"), s.add_argument("--device")
    s.add_argument("--stride", type=int, default=1, help="decimation horizontale")
    s.set_defaults(func=cmd_export_web)

    s = sub.add_parser("serve", help="application web + API + serveur MCP")
    s.add_argument("--host", default="127.0.0.1"), s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true"), s.add_argument("--device")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("apikey", help="gerer les cles d'API du service")
    s.add_argument("action", choices=["create", "list", "revoke"])
    s.add_argument("name", nargs="?", help="nom de la cle (create / revoke)")
    s.add_argument("--scope", default="read", choices=["read", "admin"],
                   help="read : lecture ; admin : + export sur disque et gestion des cles")
    s.set_defaults(func=cmd_apikey)

    s = sub.add_parser("link", help="relier le service local a une instance distante")
    s.add_argument("action", nargs="?", default="show", choices=["set", "show", "clear"])
    s.add_argument("url", nargs="?", help="URL de l'amont, ex. https://xxx.trycloudflare.com")
    s.add_argument("--key", help="cle d'API de l'amont (reste sur cette machine)")
    s.add_argument("--label", default="colab", help="nom affiche de l'amont")
    s.add_argument("--no-check", action="store_true", help="ne pas verifier l'amont")
    s.set_defaults(func=cmd_link)

    s = sub.add_parser("mcp", help="serveur MCP en transport stdio")
    s.set_defaults(func=cmd_mcp)

    s = sub.add_parser("info", help="configuration effective et etat des artefacts")
    s.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
