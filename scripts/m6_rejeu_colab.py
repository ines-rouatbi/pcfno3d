"""Lot M6 — rejeu « comme en direct » des instants de test (criteres HIS1 et DE3), sur GPU.

A lancer dans une session Colab, depuis le depot installe, avec le cache et le checkpoint :

    !python scripts/m6_rejeu_colab.py -c colab                       # local (defaut)
    !python scripts/m6_rejeu_colab.py -c colab --remote https://atom.zelqane.com

Chaque instant synoptique de TEST (ceux de DSO1, 120 pour v7) est reconstruit DANS L'ORDRE
CHRONOLOGIQUE puis publie aussitot, comme le ferait le service en direct : instantane au
format du contrat ATOM rev. 2 (pcfno3d.publish), historique (index/times.json), latest qui
avance, alertes M4 si `alerts.enabled`, roses des vents cumulees sur les instants deja
publies (jamais sur le futur), fiabilite M2 et parc M5 s'ils sont configures. Stem du rejeu :
celui du cache (JJJ_HHMMSS, common.schema.json).

Mode LOCAL par defaut : tout s'ecrit dans --sortie (defaut run_dir/publication_rejeu), avec
la disposition des fixtures du contrat (validate.mjs s'y applique). Avec --remote URL, chaque
instantane est en plus publie vers le Worker (modele une fois, puis fichiers, puis commit :
latest ne bascule qu'apres un commit verifie).

JETON : jamais en argument. Lu dans la variable d'environnement ATOM_PUBLISH_TOKEN, sinon
dans les secrets Colab (userdata « ATOM_PUBLISH_TOKEN ») ; il n'est ni imprime, ni ecrit.

Reprise : un instant deja ecrit (et deja publie pour --remote) est saute, sans recalcul ;
relancer la meme commande apres une coupure termine le rejeu.

Codes de retour : 0 rejeu termine ; 1 erreur de calcul ou publication refusee ; 2 jeton
absent avec --remote.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]

ENV = "ATOM_PUBLISH_TOKEN"


def jeton() -> str | None:
    """Environnement, puis secrets Colab ; None si absent. Jamais imprime."""
    from pcfno3d.publish import jeton_publication

    tok = jeton_publication(ENV)
    if tok:
        return tok
    try:
        from google.colab import userdata          # type: ignore
        tok = (userdata.get(ENV) or "").strip()
    except Exception:                              # hors Colab, ou secret non autorise
        tok = ""
    return tok or None


def _publies(racine: Path, base: str) -> set[str]:
    p = racine / "index" / "publies_distant.json"
    return set(json.loads(p.read_bytes().decode("utf-8")).get(base, [])) if p.exists() else set()


def _marquer_publie(racine: Path, base: str, sid: str) -> None:
    from pcfno3d.publish import _ecrire_atomique, json_bytes

    p = racine / "index" / "publies_distant.json"
    doc = json.loads(p.read_bytes().decode("utf-8")) if p.exists() else {}
    doc[base] = sorted(set(doc.get(base, [])) | {sid})
    _ecrire_atomique(p, json_bytes(doc))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="colab")
    ap.add_argument("--set", action="append", metavar="CLE=VALEUR")
    ap.add_argument("--data-dir"), ap.add_argument("--cache-dir"), ap.add_argument("--run-dir")
    ap.add_argument("--checkpoint"), ap.add_argument("--device")
    ap.add_argument("--sortie", help="depot local (defaut : run_dir/publication_rejeu)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true", help="ecriture locale seule (defaut)")
    mode.add_argument("--remote", help="URL du Worker (jeton : ATOM_PUBLISH_TOKEN)")
    ap.add_argument("--limite", type=int, help="nombre d'instants (defaut : tout le test)")
    ap.add_argument("--cadence", type=float, default=0.0,
                    help="pause entre deux instants, en s (simulation du direct)")
    ap.add_argument("--data-kind", default="synthetique",
                    choices=("synthetique", "reel", "test"))
    ap.add_argument("--dso6", help="resume DSO6 (quality.calibration)")
    ap.add_argument("--exports", action="store_true", help="NetCDF + CSV par instant (DE3)")
    ap.add_argument("--purger", action="store_true",
                    help="avec --remote : supprimer le dossier local apres un commit reussi")
    args = ap.parse_args(argv)

    from pcfno3d.cli import _load_cfg
    from pcfno3d.publish import (ClientPublication, DepotLocal, PublicationRefusee,
                                 instantane_du_cache, sha256_fichier, snapshot_id)

    cfg = _load_cfg(args)
    client = None
    if args.remote:
        tok = jeton()
        if tok is None:
            print(f"jeton absent : {ENV} (environnement ou secret Colab)", file=sys.stderr)
            return 2
        client = ClientPublication(args.remote, token=tok)
        del tok
    racine = Path(args.sortie) if args.sortie else cfg.paths.run_dir / "publication_rejeu"
    depot = DepotLocal(racine)

    from pcfno3d.inference import WindFieldModel
    from pcfno3d.windfarm import eol1_instants
    try:
        wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1
    moteur = None
    if cfg.alerts.enabled:
        from pcfno3d.alerts import MoteurAlertes
        moteur = MoteurAlertes(cfg, journal=racine / "alertes.jsonl")
    dso6 = json.loads(Path(args.dso6).read_bytes().decode("utf-8")) if args.dso6 else None

    from pcfno3d.cache import load_index
    idx = load_index(cfg)
    ordre = dict(zip(idx["stem"].astype(str), idx["timestamp"].astype(str)))
    stems = sorted(eol1_instants(cfg), key=lambda s: ordre[s])
    if args.limite:
        stems = stems[:args.limite]
    if not stems:
        print("ERREUR : aucun instant de test en cache", file=sys.stderr)
        return 1
    sha8 = sha256_fichier(wf.checkpoint_path)[:8]
    deja = _publies(racine, client.base) if client else set()
    print(f"rejeu de {len(stems)} instants | modele {wf.checkpoint_path.name} ({sha8}) | "
          f"fiabilite {'oui' if wf.reliability else 'non'} | alertes "
          f"{'oui' if moteur else 'non'} | parc {'oui' if wf.farm else 'non'} | "
          f"{'publication ' + client.base if client else 'local'} -> {racine}")

    t_tot = time.perf_counter()
    for n, stem in enumerate(stems, 1):
        sid = snapshot_id(stem, sha8)
        t0 = time.perf_counter()
        # deja fait : ecrit en local (mode local), ou deja commite sur ce Worker (--remote,
        # meme si --purger a supprime le dossier local)
        if (sid in deja) if client is not None else depot.existe(sid):
            print(f"{n:3d}/{len(stems)} {sid} deja publie")
            continue
        if not depot.existe(sid):
            try:
                inst = instantane_du_cache(wf, stem, data_kind=args.data_kind, depot=depot,
                                           moteur=moteur, dso6=dso6,
                                           garder_champs=args.exports)
            except ValueError as exc:
                print(f"ERREUR {stem} : {exc}", file=sys.stderr)
                return 1
            depot.ecrire(inst)
            if args.exports:
                from pcfno3d.export_noeuds import exporter
                exporter(inst, wf.geom, racine / "exports" / sid)
            etat = f"{inst.data_status}, {inst.open_alerts} alerte(s) ouverte(s)"
        else:
            etat = "deja ecrit"
        if client is not None:
            try:
                client.publier(depot.dossier(sid))
            except PublicationRefusee as exc:
                print(f"ERREUR publication {sid} : {exc}", file=sys.stderr)
                return 1
            _marquer_publie(racine, client.base, sid)
            if args.purger:
                shutil.rmtree(depot.dossier(sid))
            etat += " | publie"
        print(f"{n:3d}/{len(stems)} {sid} {etat} | {time.perf_counter() - t0:.2f} s")
        if args.cadence and n < len(stems):
            time.sleep(args.cadence)
    latest = depot.latest()
    print(f"termine en {time.perf_counter() - t_tot:.1f} s | latest local : "
          f"{latest['snapshot_id'] if latest else None} | historique : {len(depot.times())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
