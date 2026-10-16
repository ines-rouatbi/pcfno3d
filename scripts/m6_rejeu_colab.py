"""Lot M6 — rejeu « comme en direct » des instants de test (criteres HIS1 et DE3), sur GPU.

A lancer dans une session Colab, depuis le depot installe, avec le cache et le checkpoint :

    !python scripts/m6_rejeu_colab.py -c colab                       # local (defaut)
    !python scripts/m6_rejeu_colab.py -c colab --remote https://atom.zelqane.com --purger

Chaque instant synoptique de TEST (ceux de DSO1, 120 pour v7) est reconstruit DANS L'ORDRE
CHRONOLOGIQUE puis publie aussitot, comme le ferait le service en direct : instantane au
format du contrat ATOM rev. 2 (pcfno3d.publish), historique (index/times.json), latest qui
avance, alertes M4 si `alerts.enabled`, roses des vents cumulees sur les instants deja
publies (jamais sur le futur), fiabilite M2 et parc M5 s'ils sont configures. Stem du rejeu :
celui du cache (JJJ_HHMMSS, common.schema.json). La configuration doit porter un ancrage
(`real.origin_lat/lon`, comme configs/colab.yaml) : sans lui, rien n'est publie.

Mode LOCAL par defaut. Racine par defaut : le disque LOCAL de Colab
(/content/pcfno3d_publication_rejeu), jamais le Drive (~52 Mo par instant v7 avec M2) ;
hors Colab, run_dir/publication_rejeu. La taille est estimee et l'espace libre verifie
AVANT de commencer. Exports DE3 (NetCDF + CSV, ~250 Mo par instant v7 au plus) avec
--exports. Avec --remote URL, chaque instantane est en plus publie vers le Worker (modele
une fois, fichiers, commit, puis relecture de /v1/times et /v1/latest).

NE JAMAIS lancer ce rejeu en meme temps qu'une autre publication vers le MEME Worker : le
Worker reecrit index/times.json par lecture-modification-ecriture, deux commits
simultanes peuvent perdre une entree.

JETON : jamais en argument. Lu dans la variable d'environnement ATOM_PUBLISH_TOKEN, sinon
dans les secrets Colab (userdata « ATOM_PUBLISH_TOKEN ») ; il n'est ni imprime, ni ecrit.

Reprise : un instant deja ecrit (local) ou deja publie et verifie sur ce Worker est saute
sans recalcul ; un export DE3 manquant est refait depuis le dossier publie ; relancer la
meme commande apres une coupure termine le rejeu.

Codes de retour : 0 rejeu termine ; 1 erreur de calcul, configuration differente d'un
instantane deja ecrit, ou publication refusee ; 2 jeton absent avec --remote ;
3 publication INCOMPLETE (commit present mais index / latest du Worker non a jour apres
rejeu du commit : reparation cote Worker, rien n'est marque publie ni purge) ;
4 espace disque insuffisant.
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
RACINE_COLAB = Path("/content/pcfno3d_publication_rejeu")
OCTETS_LIGNE_CSV = 190          # ligne de noeud du CSV 8.4, mesure sur le mini-jeu (+ marge)


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


def estimation_octets(geom, fiabilite: bool) -> dict:
    """Taille d'un instant v7 ou mini : instantane publie (champs + JSON + modele) et
    exports DE3 (NetCDF non compresse : borne haute ; CSV : ~190 octets par noeud)."""
    n = int(geom.nz * geom.ny * geom.nx)
    f32, u8 = (8, 3) if fiabilite else (4, 0)
    instantane = n * (4 * f32 + u8) + geom.ny * geom.nx * 4 + 400_000
    nc = n * 4 * (6 + (8 if fiabilite else 0)) + geom.ny * geom.nx * 8 * 3
    csv = int(geom.valid.sum()) * OCTETS_LIGNE_CSV
    return {"instantane": instantane, "exports": nc + csv}


def _libre(chemin: Path) -> int:
    p = chemin
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="colab")
    ap.add_argument("--set", action="append", metavar="CLE=VALEUR")
    ap.add_argument("--data-dir"), ap.add_argument("--cache-dir"), ap.add_argument("--run-dir")
    ap.add_argument("--checkpoint"), ap.add_argument("--device")
    ap.add_argument("--sortie", help="depot local (defaut : /content/... sur Colab, sinon "
                                     "run_dir/publication_rejeu)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true", help="ecriture locale seule (defaut)")
    mode.add_argument("--remote", help="URL du Worker (jeton : ATOM_PUBLISH_TOKEN)")
    ap.add_argument("--limite", type=int, help="nombre d'instants (defaut : tout le test)")
    ap.add_argument("--cadence", type=float, default=0.0,
                    help="pause entre deux instants, en s (simulation du direct)")
    ap.add_argument("--data-kind", default="synthetique",
                    choices=("synthetique", "reel", "test"))
    ap.add_argument("--dso6", help="resume DSO6 (quality.calibration)")
    ap.add_argument("--exports", action="store_true",
                    help="NetCDF + CSV par instant (DE3) ; desactive par defaut (volumineux)")
    ap.add_argument("--purger", action="store_true",
                    help="avec --remote : supprimer le dossier local apres une publication "
                         "VERIFIEE")
    args = ap.parse_args(argv)

    from pcfno3d.cli import _load_cfg
    from pcfno3d.export_noeuds import exporter_dossier
    from pcfno3d.publish import (ClientPublication, DepotLocal, PublicationIncomplete,
                                 PublicationRefusee, champs_attendus,
                                 differences_configuration, instantane_du_cache, snapshot_id,
                                 versions_modele)

    cfg = _load_cfg(args)
    if cfg.real.origin_lat is None or cfg.real.origin_lon is None:
        print("ERREUR : configuration sans ancrage (real.origin_lat/lon) : M6 ne publie pas "
              "un domaine sans position (voir configs/colab.yaml)", file=sys.stderr)
        return 1
    client = None
    if args.remote:
        tok = jeton()
        if tok is None:
            print(f"jeton absent : {ENV} (environnement ou secret Colab)", file=sys.stderr)
            return 2
        client = ClientPublication(args.remote, token=tok)
        del tok
    if args.sortie:
        racine = Path(args.sortie)
    elif RACINE_COLAB.parent.is_dir():
        racine = RACINE_COLAB                      # disque local de Colab, pas le Drive
    else:
        racine = cfg.paths.run_dir / "publication_rejeu"
    depot = DepotLocal(racine)
    if depot.repares:
        print(f"index local complete depuis les dossiers presents : {depot.repares}")

    from pcfno3d.inference import WindFieldModel
    from pcfno3d.windfarm import eol1_instants
    try:
        wf = WindFieldModel(cfg, checkpoint=args.checkpoint, device=args.device)
        versions = versions_modele(wf, args.data_kind)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1
    champs = champs_attendus(wf)
    moteur = None
    if cfg.alerts.enabled:
        from pcfno3d.alerts import MoteurAlertes
        moteur = MoteurAlertes(cfg, journal=racine / "alertes.jsonl")
    dso6 = json.loads(Path(args.dso6).read_bytes().decode("utf-8")) if args.dso6 else None
    rc = cfg.reliability

    from pcfno3d.cache import load_index
    idx = load_index(cfg)
    ordre = dict(zip(idx["stem"].astype(str), idx["timestamp"].astype(str)))
    stems = sorted(eol1_instants(cfg), key=lambda s: ordre[s])
    if args.limite:
        stems = stems[:args.limite]
    if not stems:
        print("ERREUR : aucun instant de test en cache", file=sys.stderr)
        return 1
    sha8 = versions["model_sha256"][:8]
    deja = _publies(racine, client.base) if client else set()

    # ---- disque : estimation AVANT de commencer
    est = estimation_octets(wf.geom, wf.reliability is not None)
    a_faire = [s for s in stems if not ((snapshot_id(s, sha8) in deja) if client
                                        else depot.existe(snapshot_id(s, sha8)))]
    par_instant = est["instantane"] + (est["exports"] if args.exports else 0)
    besoin = int(1.1 * (par_instant if (client and args.purger and not args.exports)
                        else par_instant * len(a_faire)))
    libre = _libre(racine)
    print(f"estimation : {est['instantane'] / 1e6:.1f} Mo par instantane"
          + (f" + {est['exports'] / 1e6:.1f} Mo d'exports" if args.exports else
             " (exports DE3 desactives : --exports)")
          + f" ; {len(a_faire)} instant(s) a faire -> {besoin / 1e9:.2f} Go necessaires, "
          f"{libre / 1e9:.2f} Go libres sur {racine}")
    if besoin > libre:
        print(f"ERREUR : espace disque insuffisant sur {racine} ({besoin / 1e9:.2f} Go "
              f"necessaires, {libre / 1e9:.2f} Go libres) : liberer de la place, reduire "
              "--limite, retirer --exports ou publier avec --remote --purger",
              file=sys.stderr)
        return 4
    print(f"rejeu de {len(stems)} instants | modele {wf.checkpoint_path.name} ({sha8}) | "
          f"fiabilite {'oui' if wf.reliability else 'non'} | alertes "
          f"{'oui' if moteur else 'non'} | parc {'oui' if wf.farm else 'non'} | "
          f"{'publication ' + client.base if client else 'local'} -> {racine}")

    t_tot = time.perf_counter()
    for n, stem in enumerate(stems, 1):
        sid = snapshot_id(stem, sha8)
        t0 = time.perf_counter()
        dossier = depot.dossier(sid)
        etat = []
        if depot.existe(sid):
            diff = differences_configuration(dossier, versions, champs)
            if diff:
                print(f"ERREUR {sid} existe avec une autre configuration : "
                      + " ; ".join(diff), file=sys.stderr)
                return 1
        elif not (client is not None and sid in deja):
            try:
                inst = instantane_du_cache(wf, stem, data_kind=args.data_kind, depot=depot,
                                           moteur=moteur, dso6=dso6)
            except ValueError as exc:
                print(f"ERREUR {stem} : {exc}", file=sys.stderr)
                return 1
            depot.ecrire(inst)
            etat.append(f"{inst.data_status}, alertes ouvertes {inst.open_alerts}")
            # temps : journal local (le delai de fiabilite n'a pas de place dans le contrat)
            etat.append(f"valeur {inst.info['inference_s'] * 1000:.0f} ms"
                        + (f", fiabilite {inst.info['fiabilite_s']:.2f} s"
                           if inst.info.get("fiabilite_s") is not None else ""))
        # exports DE3 manquants, toujours depuis le dossier publie
        if args.exports and depot.existe(sid):
            faits = exporter_dossier(dossier, wf.geom, racine / "exports", rc.class_high,
                                     rc.class_mid)
            if faits:
                etat.append("exports " + "+".join(faits))
        if client is not None and sid not in deja:
            try:
                client.publier(dossier)
            except PublicationIncomplete as exc:
                print(f"PUBLICATION INCOMPLETE {sid} : {exc}", file=sys.stderr)
                return 3
            except PublicationRefusee as exc:
                print(f"ERREUR publication {sid} : {exc}", file=sys.stderr)
                return 1
            # publication VERIFIEE (index + latest relus) : alors seulement on marque, on purge
            _marquer_publie(racine, client.base, sid)
            deja.add(sid)
            if args.purger:
                shutil.rmtree(dossier)
            etat.append("publie")
        if not etat:
            print(f"{n:3d}/{len(stems)} {sid} deja publie")
            continue
        print(f"{n:3d}/{len(stems)} {sid} {' | '.join(etat)} | "
              f"{time.perf_counter() - t0:.2f} s")
        if args.cadence and n < len(stems):
            time.sleep(args.cadence)
    latest = depot.latest()
    print(f"termine en {time.perf_counter() - t_tot:.1f} s | latest local : "
          f"{latest['snapshot_id'] if latest else None} | historique : {len(depot.times())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
