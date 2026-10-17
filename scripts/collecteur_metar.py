"""Collecteur METAR : mesures reelles des aeroports tunisiens -> archive + reconstruction.

A chaque passage :
  1. telecharge les METAR recents (aviationweather.gov, gratuit, sans cle) ;
  2. les convertit au format des stations reelles (pcfno3d.metar) et les ajoute, sans
     doublon, a l'archive CSV du jour (reutilisable par `pcfno3d import-real --stations`) ;
  3. indique pour chaque station sa distance au domaine (ancrage `real.origin_*` de la
     configuration ; position SUPPOSEE tant que Tukhnanutha ne l'a pas fournie) ;
  4. avec --envoyer, poste l'instant le plus recent a POST /api/reconstruct/stations
     (cle d'API dans la variable d'environnement PCFNO3D_API_KEY, jamais sur la ligne).

Les METAR servent a TESTER le circuit : les aeroports sont trop epars pour la v7 et, avec la
position supposee du domaine, aucun n'y tombe (le service repond alors 400, sans rien
inventer). Aucune station n'est deplacee.

    python scripts/collecteur_metar.py                         # un passage, archive seule
    python scripts/collecteur_metar.py --envoyer http://127.0.0.1:8001
    python scripts/collecteur_metar.py --envoyer http://127.0.0.1:8001 --toutes 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]

ENV_CLE = "PCFNO3D_API_KEY"


def passage(args, geo, etendue) -> int:
    from pcfno3d import metar

    ids = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    try:
        brut = metar.telecharger(ids, heures=args.heures)
    except OSError as exc:
        print(f"telechargement impossible : {exc}", file=sys.stderr)
        return 1
    lignes, ecartees = [], []
    for obs in brut:
        l, raison = metar.metar_vers_station(obs)
        (lignes.append(l) if l else ecartees.append(f"{obs.get('icaoId')} : {raison}"))
    jour = datetime.now(timezone.utc).strftime("%Y%m%d")
    archive = Path(args.sortie) / f"metar_{jour}.csv"
    n = metar.ajouter_archive(archive, lignes)
    print(f"{len(brut)} METAR, {len(lignes)} exploitables, {n} nouveaux -> {archive}")
    for e in ecartees:
        print(f"  ecarte {e}")
    instant, choix = metar.instant_commun(lignes)
    if instant is None:
        print("aucune mesure exploitable", file=sys.stderr)
        return 1
    print(f"instant {instant} : {len(choix)} stations")
    if geo is not None:
        pos = {p["station_id"]: p for p in metar.position_domaine(choix, geo, *etendue)}
        dedans = [s for s, p in pos.items() if p["distance_domaine_km"] == 0]
        for l in choix:
            p = pos[l["station_id"]]
            vent = ("calme" if l["wind_speed_m_s"] == 0 else
                    f"{l['wind_speed_m_s']:.1f} m/s du {l['wind_dir_deg']:.0f} deg")
            lieu = "DANS le domaine" if p["distance_domaine_km"] == 0 \
                else f"a {p['distance_domaine_km']} km du domaine"
            print(f"  {l['station_id']} {metar.STATIONS_TUNISIE.get(l['station_id'], '')}: "
                  f"{vent}, {lieu}")
        print(f"{len(dedans)} station(s) dans le domaine (position {args.position})")
    if args.envoyer:
        cle = os.environ.get(ENV_CLE)
        if not cle:
            print(f"cle d'API absente : definir {ENV_CLE} dans l'environnement", file=sys.stderr)
            return 2
        statut, rep = metar.envoyer(args.envoyer, cle, metar.corps_reconstruction(instant, choix))
        if statut == 200:
            print(f"reconstruction {rep.get('stem')} : {json.dumps(rep.get('stations'))}, "
                  f"vitesse max {rep.get('vitesse_max_m_s')} m/s, "
                  f"inference {rep.get('inference_s')} s, georeferencement "
                  f"{rep.get('georeferencement')}")
        else:
            print(f"service : HTTP {statut} : {rep.get('detail', rep)}", file=sys.stderr)
            return 3
    return 0


def main() -> int:
    from pcfno3d.metar import STATIONS_TUNISIE

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stations", default=",".join(STATIONS_TUNISIE),
                    help="codes OACI separes par des virgules (defaut : aeroports tunisiens)")
    ap.add_argument("--heures", type=int, default=2, help="profondeur du telechargement (h)")
    ap.add_argument("--sortie", default=str(Path(os.environ.get("LOCALAPPDATA", Path.home()))
                                            / "pcfno3d" / "metar"),
                    help="dossier de l'archive CSV (un fichier par jour UTC)")
    ap.add_argument("-c", "--config", default="local_v7",
                    help="configuration, pour l'ancrage du domaine (section real)")
    ap.add_argument("--data-dir", help="dossier des donnees (meta.json : etendue du domaine)")
    ap.add_argument("--envoyer", metavar="URL", help="service PC-FNO3D, ex. http://127.0.0.1:8001")
    ap.add_argument("--toutes", type=float, metavar="MIN",
                    help="repeter toutes les MIN minutes (sinon un seul passage)")
    args = ap.parse_args()

    geo, etendue, args.position = None, None, "inconnue"
    try:
        from pcfno3d.config import Config
        from pcfno3d.geometry import Meta
        from pcfno3d.realdata import GeoRef

        cfg = Config.load(args.config)
        if args.data_dir:
            cfg.apply_overrides({"paths.data_dir": Path(args.data_dir)})
        geo = GeoRef.from_cfg(cfg)
        m = Meta.load(cfg.paths.data_dir)
        etendue = ((m.nx_s - 1) * m.dx_s, (m.ny_s - 1) * m.dy_s)
        args.position = "supposee" if cfg.real.origin_assumed else "mesuree"
    except (OSError, ValueError, KeyError) as exc:
        print(f"position du domaine non verifiee ({exc})", file=sys.stderr)
        geo = None

    if not args.toutes:
        return passage(args, geo, etendue)
    while True:
        rc = passage(args, geo, etendue)
        print(f"-- code {rc} ; prochain passage dans {args.toutes:g} min", flush=True)
        time.sleep(args.toutes * 60)


if __name__ == "__main__":
    raise SystemExit(main())
