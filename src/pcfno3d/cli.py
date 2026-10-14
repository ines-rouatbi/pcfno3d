"""Point d'entree en ligne de commande : `python -m pcfno3d <commande>`.

    build-cache   CSV journaliers -> tenseurs .npy + index.csv        (BLOC 3)
    stats         statistiques de normalisation                       (BLOC 5)
    check         controle de la verite terrain et de la geometrie    (BLOC 2/4)
    train         entrainement, reprise automatique                   (BLOC 9)
    evaluate      metriques de test et verdict DSO1-DSO5 (--tta N)    (BLOC 10)
    dso-weights   poids u, v par niveau, mesures sur la VALIDATION    (v8)
    reconstruct   reconstruction 3D d'un instant (npz, csv, figures)  (BLOC 12)
    export-web    volume -> binaires + manifeste pour le visualiseur
    import-real   donnees REELLES (stations + radiosondes / champ maille) -> cache
    validate      contrat de donnees ANEO : verdict, rapport qualite, quarantaine
    serve         application web + API JSON + serveur MCP (fastapi + uvicorn)
    apikey        creer / lister / revoquer les cles d'API du service, regler leurs quotas
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

import numpy as np

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
    evaluate(cfg, checkpoint=args.checkpoint, device=args.device,
             tta=args.tta, tta_keep=args.tta_keep)
    return 0


def cmd_dso_weights(args) -> int:
    """Poids de u, v par niveau, mesures sur la VALIDATION d'un modele existant."""
    from . import dso
    from .cache import load_index
    from .dataset import make_loaders, make_splits
    from .inference import WindFieldModel

    cfg = _load_cfg(args)
    cfg.train.num_workers = 0
    wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device)
    splits = make_splits(cfg, load_index(cfg))
    loader = make_loaders(cfg, wf.geom, splits)["val"]
    if len(loader) == 0:
        print("aucun instant de validation en cache : construis le cache des jours de "
              "validation (split_days.json)", file=sys.stderr)
        return 1
    print(f"mesure sur {len(loader)} instants de VALIDATION synoptiques "
          f"(jamais le test : il ne sert qu'au verdict)\n")
    metrics = dso.measure(wf.model, loader, wf.geom, wf.norm, wf.device,
                          wf.amp_dtype, wf.use_amp)
    print(dso.describe_levels(metrics, cfg, wf.geom))
    sc = dso.score(metrics, cfg, wf.geom)
    print(f"\nscore DSO (validation) = {sc['score']:.3f} | limitant : {sc['limitant']} | "
          f"RMSE max {sc['rmse_max']:.3f} m/s a {sc['niveau_le_pire_m']} m")

    weights = dso.level_weights(metrics, cfg, wf.geom, floor=args.floor, cap=args.cap)
    print("\npoids u, v par niveau :")
    for z, w in zip(wf.geom.z_levels, weights):
        bar = "#" * int(round(w * 10))
        print(f"  {int(z):5d} m  {w:6.3f}  {bar}")

    import json
    out = Path(args.out) if args.out else cfg.paths.run_dir / "uv_level_weight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "uv_level_weight": weights,
        "jeu": "validation",
        "instants": len(loader),
        "modele": wf.checkpoint_path.name,
        "z_levels_asl_m": wf.geom.z_levels.tolist(),
        "rmse_u": [None if not np.isfinite(x) else round(float(x), 4)
                   for x in metrics.rmse()[0]],
        "rmse_v": [None if not np.isfinite(x) else round(float(x), 4)
                   for x in metrics.rmse()[1]],
        "score_dso_validation": sc,
        "floor": args.floor, "cap": args.cap,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")
    print(f"   a utiliser dans la configuration : loss.uv_level_weight: {out}")
    return 0


def cmd_reconstruct(args) -> int:
    from .reconstruct import reconstruct

    cfg = _load_cfg(args)
    reconstruct(cfg, stem=args.stem, csv=args.csv, checkpoint=args.checkpoint,
                keep=args.keep, altitude=args.altitude, section_row=args.section_row,
                figures=not args.no_figures, web=args.web, device=args.device)
    return 0


def _read_table(path: str):
    import pandas as pd
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def cmd_import_real(args) -> int:
    """Stations (+ radiosondes ou champ maille) -> cache S/Y + index, format du pipeline."""
    import pandas as pd

    from .geometry import Geometry
    from .realdata import GeoRef, open_gridded, points_to_target, grid_to_target, import_real

    cfg = _load_cfg(args)
    stations = _read_table(args.stations)
    target = None
    if args.points and args.grid:
        print("--points et --grid sont exclusifs : choisis une seule cible", file=sys.stderr)
        return 2
    if args.points or args.grid:
        geom, geo = Geometry(cfg, device="cpu"), GeoRef.from_cfg(cfg)
    if args.points:
        pts = _read_table(args.points)
        pts["timestamp_utc"] = pd.to_datetime(pts["timestamp_utc"], utc=True).dt.tz_localize(None)
        pts["bin"] = pts["timestamp_utc"].dt.floor(cfg.real.time_bin)
        by_bin = dict(tuple(pts.groupby("bin")))

        def target(ts):
            g = by_bin.get(ts)
            return None if g is None else points_to_target(g, geom, geo)
    elif args.grid:
        if args.grid_format == "era5-pl":
            from .realdata import open_era5_pressure_levels
            at = open_era5_pressure_levels(args.grid)           # motif accepte : era5_pl_*.nc
        else:
            at = open_gridded(args.grid, json.loads(args.grid_vars))

        def target(ts):
            got = at(ts)
            return None if got is None else grid_to_target(*got, geom, geo)
    rapport = import_real(cfg, stations, target)
    print(json.dumps(rapport, indent=2, ensure_ascii=False))
    return 0


def cmd_validate(args) -> int:
    """Contrat de donnees ANEO (section 8) : un verdict et un rapport par fichier.

    Code de retour : 0 conforme (avertissements compris), 1 quarantaine, 2 rejet.
    """
    from .contract import Grille, Seuils, code_retour, controler, ecrire_rapport

    if args.meta:
        meta = Path(args.meta)
    else:
        meta = _load_cfg(args).paths.data_dir / "meta.json"
    if not meta.exists():
        print(f"grille declaree introuvable : {meta} (option --meta)", file=sys.stderr)
        return 2
    grille = Grille.depuis_meta(meta)
    seuils = Seuils()
    if args.pas is not None:
        seuils.pas_attendu_s = args.pas
    if args.tuile is not None:
        seuils.tuile_m = args.tuile
    rapports = []
    for f in args.fichiers:
        path = Path(f)
        if not path.exists():
            print(f"[REJET]       {path} : fichier introuvable", file=sys.stderr)
            rapports.append({"fichier": str(path), "verdict": "rejet",
                             "motifs": ["fichier introuvable"]})
            continue
        r = controler(path, grille, quarantaine=args.quarantine, copier=args.copier,
                      seuils=seuils, surface_seule=args.surface_seule)
        rapports.append(r)
        print(f"[{r['verdict'].upper()}]".ljust(14) + f"{path}")
        for m in r["motifs"]:
            print(f"    - {m}")
        for a in r["avertissements"]:
            print(f"    ! {a}")
        if "quarantaine" in r:
            print(f"    -> {r['quarantaine']['mode']} vers {r['quarantaine']['fichier']}")
    if args.report:
        out = rapports[0] if len(rapports) == 1 else rapports
        ecrire_rapport(out, args.report)
        print(f"rapport : {args.report}")
    return code_retour([r["verdict"] for r in rapports])


def cmd_export_web(args) -> int:
    from .export import export_split, export_web
    from .inference import WindFieldModel

    cfg = _load_cfg(args)
    wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device)
    if args.split:
        export_split(wf, split=args.split, stride=args.stride)
        return 0
    volume = wf.predict_csv(args.csv) if args.csv else wf.predict_stem(args.stem)
    out = export_web(volume, args.out or (cfg.paths.web_root / "latest"),
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

    from service.auth import ENV_INSECURE_OPEN, insecure_open_policy, insecure_open_requested

    # verifie AVANT de toucher a l'environnement : un refus ne laisse rien derriere lui.
    # Meme regle que l'opt-in vienne de l'option ou de la variable d'environnement.
    if (args.insecure_open or insecure_open_requested()) and args.host not in LOOPBACK_HOSTS:
        origine = "--insecure-open" if args.insecure_open else ENV_INSECURE_OPEN
        print(f"{origine} refuse avec --host {args.host} : le mode ouvert est reserve a "
              "la boucle locale (127.0.0.1, localhost).", file=sys.stderr)
        return 2

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

    if args.insecure_open:
        os.environ[ENV_INSECURE_OPEN] = "1"
    ouvert, raison = insecure_open_policy()
    store = _key_store(cfg)
    if insecure_open_requested() and not ouvert:
        print(f"ATTENTION : {raison}")
    if store.empty and ouvert:
        print("ATTENTION : MODE OUVERT (--insecure-open) : aucune cle, le service repond "
              "sans authentification.\n            Developpement local uniquement.")
    elif store.empty:
        print("ATTENTION : aucune cle d'API active : toutes les requetes seront refusees "
              "(401) sauf /api/health.\n            `pcfno3d apikey create <nom>` pour en "
              "creer une (ou --insecure-open en local).")
    print(f"application web : http://{args.host}:{args.port}/app/")
    print(f"API             : http://{args.host}:{args.port}/api/docs")
    print(f"serveur MCP     : http://{args.host}:{args.port}/mcp")
    uvicorn.run("service.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


LOOPBACK_HOSTS = ("127.0.0.1", "localhost")


def _key_store(cfg: Config):
    """Le trousseau vit dans run_dir, a cote des checkpoints."""
    from service.auth import ApiKeyStore, default_store_path

    return ApiKeyStore(default_store_path(cfg.paths.run_dir))


def cmd_apikey(args) -> int:
    if args.action in ("create", "revoke", "quota") and not args.name:
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
            secret, record = store.create(args.name, scope=args.scope,
                                          quota_minute=args.per_minute,
                                          quota_jour=args.per_day)
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
            # le service relit le trousseau quand il change : effet a la requete suivante
            print(f"cle revoquee : {args.name} (refusee des la prochaine requete, "
                  "sans redemarrer le service)")
            return 0
        print(f"aucune cle active nommee '{args.name}'", file=sys.stderr)
        return 1

    if args.action == "quota":
        if args.per_minute is None and args.per_day is None and not args.reset:
            print("`apikey quota` attend --per-minute, --per-day ou --reset", file=sys.stderr)
            return 1
        try:
            record = store.set_quota(args.name, quota_minute=args.per_minute,
                                     quota_jour=args.per_day, reset=args.reset)
        except ValueError as exc:
            print(f"erreur : {exc}", file=sys.stderr)
            return 1
        if record is None:
            print(f"aucune cle active nommee '{args.name}'", file=sys.stderr)
            return 1
        print(f"quotas de {record.name} : {_quota_txt(record.quota_minute)} / minute, "
              f"{_quota_txt(record.quota_jour)} / jour (pris en compte a la requete suivante)")
        return 0

    rows = store.list_public()
    if not rows:
        print(f"aucune cle dans {store.path} : le service refuse toutes les requetes")
        return 0
    print(f"{'nom':20s} {'portee':8s} {'etat':9s} {'empreinte':14s} {'quota min/jour':16s} "
          "derniere utilisation")
    for r in rows:
        etat = "revoquee" if r["revoked"] else "active"
        quota = f"{_quota_txt(r['quota_minute'])}/{_quota_txt(r['quota_jour'])}"
        print(f"{r['name']:20s} {r['scope']:8s} {etat:9s} {r['empreinte']:14s} {quota:16s} "
              f"{r['last_used_at'] or '-'}")
    return 0


def _quota_txt(value) -> str:
    if value is None:
        return "defaut"
    return "illimite" if value == 0 else str(value)


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
    s.add_argument("--tta", type=int, help="moyenne de N passes (defaut : inference.tta)")
    s.add_argument("--tta-keep", type=float, help="capteurs gardes par passe (defaut 0.7)")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("dso-weights",
                       help="poids u, v par niveau mesures sur la validation (v8)")
    s.add_argument("--checkpoint"), s.add_argument("--device")
    s.add_argument("--floor", type=float, default=0.5, help="poids minimal (defaut 0.5)")
    s.add_argument("--cap", type=float, default=4.0, help="poids maximal (defaut 4)")
    s.add_argument("--out", help="fichier JSON (defaut : run_dir/uv_level_weight.json)")
    s.set_defaults(func=cmd_dso_weights)

    s = sub.add_parser("reconstruct", help="reconstruction 3D d'un instant")
    s.add_argument("--stem", help='instant en cache, ex. "004_120000"')
    s.add_argument("--csv", help="CSV de mesures de surface d'un instant")
    s.add_argument("--checkpoint"), s.add_argument("--device")
    s.add_argument("--keep", type=float, default=1.0, help="fraction de capteurs utilises")
    s.add_argument("--altitude", type=float, default=1000.0,
                   help="altitude ASL de la carte (defaut 1000 m, comme MAP_ALTITUDE du notebook v7)")
    s.add_argument("--section-row", type=int, help="ligne iy de la coupe verticale")
    s.add_argument("--no-figures", action="store_true")
    s.add_argument("--web", action="store_true", help="exporte aussi pour le visualiseur")
    s.set_defaults(func=cmd_reconstruct)

    s = sub.add_parser("import-real", help="donnees reelles -> cache du pipeline")
    s.add_argument("--stations", required=True,
                   help="CSV/Parquet : station_id, timestamp_utc, lat, lon, height_agl_m, vent...")
    s.add_argument("--points", help="cible ponctuelle (radiosondes, mats) : CSV/Parquet")
    s.add_argument("--grid", help="cible maillee (reanalyse, WRF) : NetCDF")
    s.add_argument("--grid-vars", default='{"u": "u", "v": "v", "z": "z"}',
                   help="noms des variables du fichier maille (JSON)")
    s.add_argument("--grid-format", choices=["generic", "era5-pl"], default="generic",
                   help="era5-pl : fichiers ERA5 niveaux de pression (download_reanalysis.py)")
    s.set_defaults(func=cmd_import_real)

    s = sub.add_parser("validate", help="contrat de donnees ANEO (verdict + rapport)")
    s.add_argument("fichiers", nargs="+", help="CSV de mesures (.csv ou .csv.gz)")
    s.add_argument("--meta", help="meta.json de la grille declaree (defaut : "
                                  "paths.data_dir/meta.json)")
    s.add_argument("--quarantine", metavar="DIR",
                   help="dossier ou deplacer les fichiers non conformes (+ rapport)")
    s.add_argument("--copier", action="store_true",
                   help="copier au lieu de deplacer vers la quarantaine")
    s.add_argument("--report", metavar="OUT.json", help="ecrit le(s) rapport(s) JSON")
    s.add_argument("--pas", type=int, help="pas de temps attendu en s (defaut 10)")
    s.add_argument("--tuile", type=float, help="cote des tuiles de couverture en m "
                                               "(defaut 1000)")
    s.add_argument("--surface-seule", action="store_true",
                   help="ignorer les noeuds 3D (fichiers journaliers complets)")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("export-web", help="volume -> binaires + manifeste")
    s.add_argument("--stem"), s.add_argument("--csv"), s.add_argument("--out")
    s.add_argument("--checkpoint"), s.add_argument("--device")
    s.add_argument("--stride", type=int, default=1, help="decimation horizontale")
    s.add_argument("--split", choices=["train", "val", "test"],
                   help="exporte TOUS les instants de ce jeu (un modele charge une fois)")
    s.set_defaults(func=cmd_export_web)

    s = sub.add_parser("serve", help="application web + API + serveur MCP")
    s.add_argument("--host", default="127.0.0.1"), s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true"), s.add_argument("--device")
    s.add_argument("--insecure-open", action="store_true",
                   help="DEVELOPPEMENT LOCAL : sans aucune cle, accepter les requetes au lieu "
                        "de les refuser (boucle locale seulement, jamais sur Colab / tunnel)")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("apikey", help="gerer les cles d'API du service")
    s.add_argument("action", choices=["create", "list", "revoke", "quota"])
    s.add_argument("name", nargs="?", help="nom de la cle (create / revoke / quota)")
    s.add_argument("--scope", default="read", choices=["read", "admin"],
                   help="read : lecture ; admin : + export sur disque et gestion des cles")
    s.add_argument("--per-minute", type=int,
                   help="quota propre a la cle : requetes par minute (0 = illimite)")
    s.add_argument("--per-day", type=int,
                   help="quota propre a la cle : requetes par jour UTC (0 = illimite)")
    s.add_argument("--reset", action="store_true",
                   help="quota : revenir aux quotas par defaut du service")
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
