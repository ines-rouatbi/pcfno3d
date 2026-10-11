"""Point d'entree en ligne de commande : `python -m pcfno3d <commande>`.

    build-cache   CSV journaliers -> tenseurs .npy + index.csv        (BLOC 3)
    stats         statistiques de normalisation                       (BLOC 5)
    check         controle de la verite terrain et de la geometrie    (BLOC 2/4)
    train         entrainement, reprise automatique                   (BLOC 9)
    evaluate      metriques de test et verdict DSO1-DSO5              (BLOC 10)
    reconstruct   reconstruction 3D d'un instant (npz, csv, figures)  (BLOC 12)
    export-web    volume -> binaires + manifeste pour le visualiseur
    serve         API HTTP d'inference (necessite fastapi + uvicorn)
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
    os.environ.setdefault("PCFNO3D_CONFIG", str(args.config or ""))
    os.environ.setdefault("PCFNO3D_RUN_DIR", str(cfg.paths.run_dir))
    os.environ.setdefault("PCFNO3D_DATA_DIR", str(cfg.paths.data_dir))
    uvicorn.run("service.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


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

    s = sub.add_parser("serve", help="API HTTP d'inference")
    s.add_argument("--host", default="127.0.0.1"), s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("info", help="configuration effective et etat des artefacts")
    s.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
