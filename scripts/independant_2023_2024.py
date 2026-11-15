"""Jugement independant 2023-2024 aux 12 aeroports (regle configs/national/regle_independant_2023_2024.json).

ERA5 10 m seule contre ERA5 + OI des innovations des 11 AUTRES aeroports (L 150 km, lambda 4, biais 2015-2020 :
rien n'est reajuste sur 2023-2024). Instants synoptiques ; mesure METAR a 30 min au plus, la plus proche. Lecture
unique : la sortie n'est jamais ecrasee.

    python scripts/independant_2023_2024.py --donnees artifacts/national/independant_2023_2024
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
REGLE = REPO / "configs" / "national" / "regle_independant_2023_2024.json"
BIAIS = REPO / "configs" / "national" / "biais_innovations_aeroports_2015_2020.json"
ANNEES = (2023, 2024)


def metar_synoptique(dossier: Path, codes) -> pd.DataFrame:
    m = pd.concat([pd.read_csv(dossier / f"metar_{a}.csv") for a in ANNEES])
    m = m[m["station_id"].isin(codes)].dropna(subset=["wind_u_m_s", "wind_v_m_s"])
    t = pd.to_datetime(m["timestamp_utc"], utc=True).dt.tz_localize(None)
    m = m.assign(t=t, heure=t.dt.round("3h"))
    m = m[(m["t"] - m["heure"]).abs() <= pd.Timedelta(minutes=30)]
    m = m.assign(ecart=(m["t"] - m["heure"]).abs()).sort_values("ecart")
    m = m.drop_duplicates(["station_id", "heure"])
    m = m[m["heure"].dt.year.isin(ANNEES)]
    return m.rename(columns={"station_id": "code", "heure": "temps", "wind_u_m_s": "u_obs", "wind_v_m_s": "v_obs"})[
        ["code", "temps", "u_obs", "v_obs"]]


def era5_points(dossier: Path, codes) -> pd.DataFrame:
    out = []
    for c in codes:
        for a in ANNEES:
            e = pd.read_csv(dossier / f"{c}_{a}.csv", parse_dates=["temps"])
            vit, d = e["wind_speed_10m"].astype(float), np.radians(e["wind_direction_10m"].astype(float))
            out.append(pd.DataFrame({"code": c, "temps": e["temps"], "u_era5": -vit * np.sin(d), "v_era5": -vit * np.cos(d)}))
    return pd.concat(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--donnees", type=Path, required=True)
    a = ap.parse_args(argv)
    from pcfno3d.regles import verifier, sortie_neuve
    from diag_leviers_aeroports import distances_km
    from oi_post_aeroports import oi_autres
    regle, h = verifier(REGLE)
    sortie = sortie_neuve(a.donnees / "jugement", "bilan.json")
    print(f"regle {REGLE.name} {h[:12]}... verifiee")
    meta = pd.read_csv(REPO / "configs" / "national" / "stations_metar.csv")
    codes = sorted(meta["code"])
    pos = meta.set_index("code").loc[codes]
    dist = distances_km(pos["lat"].to_numpy(), pos["lon"].to_numpy())
    b = json.loads(BIAIS.read_text(encoding="utf-8"))["biais"]
    bu, bv = np.array([b[c]["du_moyen"] for c in codes]), np.array([b[c]["dv_moyen"] for c in codes])

    p = metar_synoptique(a.donnees / "metar", codes).merge(era5_points(a.donnees / "era5", codes), on=["code", "temps"])
    piv = {c: p.pivot_table(index="temps", columns="code", values=c, aggfunc="first").reindex(columns=codes)
           for c in ("u_obs", "v_obs", "u_era5", "v_era5")}
    du = piv["u_obs"].to_numpy() - piv["u_era5"].to_numpy()
    dv = piv["v_obs"].to_numpy() - piv["v_era5"].to_numpy()
    eu, ev = oi_autres(du - bu, dv - bv, dist)
    err = {"era5": (du, dv), "era5_oi": (du - eu, dv - ev)}
    commun = np.isfinite(du) & np.isfinite(dv) & np.isfinite(eu) & np.isfinite(ev)
    err = {k: (np.where(commun, x, np.nan), np.where(commun, y, np.nan)) for k, (x, y) in err.items()}

    def rmse(e, sel=slice(None)):
        return np.sqrt(np.nanmean(e[0][sel] ** 2 + e[1][sel] ** 2, axis=0))

    jours = piv["u_obs"].index.normalize()
    ju = np.unique(jours)
    idx = [np.where(jours == j)[0] for j in ju]
    rng = np.random.default_rng(2030)
    tir = [np.concatenate([idx[i] for i in rng.integers(0, len(ju), len(ju))]) for _ in range(2000)]
    r0, r1 = rmse(err["era5"]), rmse(err["era5_oi"])
    ec = np.array([rmse(err["era5_oi"], s).mean() - rmse(err["era5"], s).mean() for s in tir])
    ic = (float(np.percentile(ec, 2.5)), float(np.percentile(ec, 97.5)))
    mieux, pire = int(np.sum(r1 < r0)), float(np.max(r1 / r0 - 1))
    bat = bool(ic[1] < 0 and mieux >= 8 and pire <= 0.05)
    tab = pd.DataFrame({"code": codes, "aeroport": pos["nom"].to_numpy(), "n": commun.sum(0),
                        "rmse_vect_era5": r0, "rmse_vect_era5_oi": r1, "ecart_relatif": r1 / r0 - 1})
    tab.to_csv(sortie / "par_aeroport.csv", index=False)
    bilan = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "regle": REGLE.name,
             "regle_sha256": h, "periode": "2023-01-01 -> 2024-12-31 (jamais utilisee)",
             "instants": int(commun.any(axis=1).sum()), "paires": int(commun.sum()),
             "era5": float(r0.mean()), "era5_oi": float(r1.mean()), "ecart": float(r1.mean() - r0.mean()),
             "ic95_ecart": ic, "aeroports_ameliores": mieux, "pire_degradation_relative": pire,
             "bat_era5_selon_la_regle": bat,
             "note": "METAR = mesures ; ERA5 = reanalyse ; le modele C' s44 n'est pas juge ici (forcage 2023-2024 absent)"}
    (sortie / "bilan.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(tab.round(3).to_string(index=False))
    print(json.dumps(bilan, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
