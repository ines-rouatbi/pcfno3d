"""Bilan de la validation croisee spatiale de l'affinage sur les stations (v3) et regle de livraison.

Dans chaque pli, une station n'a JAMAIS servi de cible pendant l'affinage : son erreur sur le
test (station retiree de l'entree, scripts/valider_grand_tunis_reel.py) est le seul chiffre qui
juge si l'affinage ameliore le champ au-dela des emplacements appris. On la compare, a la meme
station, a la v2 (avant affinage) et a ERA5.

Regle de livraison (decision du 2026-10-23) : la v3 n'est livree que si les plis montrent un gain
aux stations jamais vues ; sinon la v2 reste le modele livre. Ici, un gain doit depasser le bruit
(~170 instants par station) : erreur vectorielle moyenne des 3 stations jamais vues au moins
GAIN_MOYEN plus basse que celle de la v2, et au moins 2 stations sur 3 en progres d'au moins
GAIN_STATION.

    python scripts/bilan_plis_grand_tunis.py --runs /content/drive/MyDrive/pcfno3d_grand_tunis/runs

Code de sortie : 0 = gain (v3 livrable), 3 = pas de gain (la v2 reste livree), 2 = plis incomplets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PLIS = {"grand_tunis_v3_cv_sans_dtta": "Tunis-Carthage",
        "grand_tunis_v3_cv_sans_zaghouan": "Zaghouan",
        "grand_tunis_v3_cv_sans_nabeul": "Nabeul"}
GAIN_MOYEN = 0.05                      # -5 % en moyenne sur les stations jamais vues
GAIN_STATION = 0.03                    # -3 % a une station pour la compter en progres
REFERENCE = "grand_tunis_v2_long"
FINAL = "grand_tunis_v3_stations"


def lire(runs: Path, run: str) -> pd.DataFrame | None:
    f = runs / run / "validation_reelle" / "test" / "stations_bilan.csv"
    return pd.read_csv(f) if f.exists() else None


def valeur(b: pd.DataFrame, station: str, source: str, col: str) -> float:
    q = b[(b["station"] == station) & (b["source"] == source)]
    return float(q[col].iloc[0]) if len(q) else float("nan")


def bilan(runs: Path) -> tuple[pd.DataFrame, dict]:
    ref = lire(runs, REFERENCE)
    fin = lire(runs, FINAL)
    if ref is None:
        raise FileNotFoundError(f"{REFERENCE} : validation reelle absente")
    lignes = []
    for pli, station in PLIS.items():
        b = lire(runs, pli)
        lignes.append({
            "station_jamais_cible": station, "pli": pli,
            "pli_rmse_vect": valeur(b, station, "modele", "rmse_vect") if b is not None else float("nan"),
            "v2_rmse_vect": valeur(ref, station, "modele", "rmse_vect"),
            "era5_rmse_vect": valeur(ref, station, "era5", "rmse_vect"),
            "v3_final_rmse_vect": valeur(fin, station, "modele", "rmse_vect") if fin is not None else float("nan"),
            "pli_biais_vitesse": valeur(b, station, "modele", "biais_vitesse") if b is not None else float("nan"),
            "v2_biais_vitesse": valeur(ref, station, "modele", "biais_vitesse"),
        })
    t = pd.DataFrame(lignes)
    complet = t["pli_rmse_vect"].notna().all()
    t["gain_relatif"] = 1.0 - t["pli_rmse_vect"] / t["v2_rmse_vect"]
    progres = int((t["gain_relatif"] >= GAIN_STATION).sum())
    gain_moyen = 1.0 - t["pli_rmse_vect"].mean() / t["v2_rmse_vect"].mean()
    gain = bool(complet and gain_moyen >= GAIN_MOYEN and progres >= 2)
    decision = {"plis_complets": bool(complet), "stations_en_progres": progres,
                "gain_moyen_relatif": float(gain_moyen),
                "moyenne_plis": float(t["pli_rmse_vect"].mean()),
                "moyenne_v2": float(t["v2_rmse_vect"].mean()),
                "moyenne_era5": float(t["era5_rmse_vect"].mean()),
                "gain": gain,
                "modele_livre": "v3 (affinage valide par les plis)" if gain else "v2 (pas de gain aux stations jamais vues)"}
    return t, decision


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    args = ap.parse_args()
    runs = Path(args.runs)
    t, d = bilan(runs)
    print(t.round(3).to_string(index=False))
    print(json.dumps(d, indent=1, ensure_ascii=False))
    (runs / "bilan_plis_v3.json").write_text(
        json.dumps({"plis": t.to_dict(orient="records"), "decision": d}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    t.to_csv(runs / "bilan_plis_v3.csv", index=False)
    if not d["plis_complets"]:
        return 2
    return 0 if d["gain"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
