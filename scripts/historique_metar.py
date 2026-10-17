"""Historique METAR des aeroports tunisiens -> jeu de donnees reel (archive IEM).

Telecharge, mois par mois, l'archive METAR de l'Iowa Environmental Mesonet (gratuite, sans
cle ; reseau TN__ASOS), la convertit au format des stations reelles (pcfno3d.metar, memes
conversions que le collecteur temps reel) et l'ecrit sans doublon dans un CSV par annee :
<sortie>/metar_<annee>.csv, lisible par `pcfno3d import-real --stations`. Relancer la
commande complete les fichiers sans rien dupliquer.

    python scripts/historique_metar.py                          # 2021-01-01 -> aujourd'hui
    python scripts/historique_metar.py --debut 2025-01-01 --fin 2025-12-31
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]


def mois(debut: datetime, fin: datetime):
    """Tranches [d, f] d'un mois au plus, de debut a fin (inclus)."""
    d = debut
    while d <= fin:
        suivant = (d.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield d, min(fin, suivant - timedelta(days=1))
        d = suivant


def main() -> int:
    from pcfno3d import metar

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--debut", default="2021-01-01", help="AAAA-MM-JJ (UTC)")
    ap.add_argument("--fin", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--stations", default=",".join(metar.STATIONS_TUNISIE))
    ap.add_argument("--sortie", default=str(Path(os.environ.get("LOCALAPPDATA", Path.home()))
                                            / "pcfno3d" / "metar" / "historique"))
    ap.add_argument("--pause", type=float, default=2.0, help="secondes entre deux requetes")
    args = ap.parse_args()

    ids = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    debut = datetime.strptime(args.debut, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    fin = datetime.strptime(args.fin, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    total, ecarts, erreurs = 0, collections.Counter(), 0
    for d, f in mois(debut, fin):
        for essai in range(3):
            try:
                brut = metar.telecharger_iem(ids, d, f + timedelta(days=1))
                break
            except OSError as exc:
                print(f"{d:%Y-%m} : essai {essai + 1} echoue ({exc})", file=sys.stderr)
                time.sleep(10 * (essai + 1))
        else:
            erreurs += 1
            continue
        lignes = []
        for row in brut:
            l, raison = metar.metar_vers_station(metar.iem_vers_obs(row))
            if l is not None:
                if d <= datetime.fromisoformat(l["timestamp_utc"].replace("Z", "+00:00")) < f + timedelta(days=1):
                    lignes.append(l)
            else:
                ecarts[raison.split(" (")[0]] += 1
        n = metar.ajouter_archive(Path(args.sortie) / f"metar_{d.year}.csv", lignes)
        total += n
        print(f"{d:%Y-%m} : {len(brut)} messages, {len(lignes)} exploitables, {n} nouveaux", flush=True)
        time.sleep(args.pause)
    print(f"total : {total} mesures nouvelles -> {args.sortie}")
    for raison, n in ecarts.most_common():
        print(f"  ecartees ({raison}) : {n}")
    if erreurs:
        print(f"{erreurs} mois non telecharges : relancer la commande", file=sys.stderr)
    return 1 if erreurs else 0


if __name__ == "__main__":
    raise SystemExit(main())
