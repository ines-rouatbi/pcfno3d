"""Jeu VIERGE pour la confirmation hors regle de M1 (avenant n 2 de docs/M1_PROTOCOLE.md).

Lance le generateur Tukhnanutha v2 (scripts/generate_aneo_3d.py) SANS LE MODIFIER, en ne
gardant que les instants synoptiques (0, 3, ..., 21 h UTC) de chaque jour : la mesure prevue
ne porte que sur eux, et les episodes (minute / 10 s) feraient passer le jeu de ~4 Go a ~13 Go,
plus que la place libre sur ines-pc. Le calendrier, les graines par jour et la physique restent
ceux du generateur ; seule la liste d'instants de chaque jour est filtree.

    python scripts/generate_jeu_vierge.py --input-grid .../aneo_dataset_3d_v2/aneo_grid_points.csv \
        --output-dir C:/Users/inesr/aneo_dataset_3d_seed2027 --n-days 15 --seed 2027 --workers 2

Tous les arguments sont ceux de generate_aneo_3d.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

GEN = Path(__file__).resolve().parent / "generate_aneo_3d.py"
# niveau module : les workers (spawn, meme sous python -I) reexecutent ce fichier et doivent
# pouvoir importer generate_aneo_3d pour retrouver generate_day
if str(GEN.parent) not in sys.path:
    sys.path.insert(0, str(GEN.parent))


def main() -> None:
    spec = importlib.util.spec_from_file_location("generate_aneo_3d", GEN)
    gen = importlib.util.module_from_spec(spec)
    sys.modules["generate_aneo_3d"] = gen          # les workers (spawn) reimportent ce nom
    spec.loader.exec_module(gen)
    calendrier_complet = gen.build_calendar

    def calendrier_synoptique(n_days, split, rng):
        days, schedule, minute_days, sec_days = calendrier_complet(n_days, split, rng)
        schedule = {i: [t for t in inst if t[1] is None] for i, inst in schedule.items()}
        # plus aucun episode : les listes d'episodes du meta.json doivent le dire
        return days, schedule, [], []

    gen.build_calendar = calendrier_synoptique
    gen.main()


if __name__ == "__main__":
    main()
