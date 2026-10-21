"""Observations SYNOP des stations INM via NOAA ISD (Integrated Surface Database), au format METAR.

Zaghouan (60729) et Nabeul (60728) ne sont pas des aeroports : pas de METAR, mais leurs
messages SYNOP (toutes les 3 h) sont diffuses sur le SMT de l'OMM et archives par la NOAA.
Ce script les telecharge (gratuit, sans compte) et les ecrit avec les MEMES colonnes que
l'archive METAR (pcfno3d.metar), pour qu'ils entrent tels quels dans le pipeline.

    python scripts/isd_synop.py --stations 60729099999,60728099999 --annees 2021-2025 \
        --sortie C:/Users/inesr/AppData/Local/pcfno3d/isd

Source a citer : NOAA NCEI, Global Hourly - Integrated Surface Database (ISD),
doi:10.25921/xs0j-1g49 (a verifier) ; observations des stations de l'INM (Tunisie).
Champs ISD (format CSV « global-hourly ») : WND = direction, qualite, type, vitesse (m/s x 10),
qualite ; TMP / DEW = degC x 10, qualite ; MA1 = altimetre, qualite, pression station (hPa x 10),
qualite ; 9999 / 99999 = absent. Codes qualite retenus : 0, 1, 4, 5, 9 (non suspects).
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/{annee}/{station}.csv"
BONS = {"0", "1", "4", "5", "9"}
HAUTEUR_ANEMOMETRE = 10.0           # norme OMM pour les stations synoptiques


def _champ(s: pd.Series, i: int) -> pd.Series:
    return s.fillna("").str.split(",").str[i]


def convertir(brut: pd.DataFrame) -> pd.DataFrame:
    """CSV ISD -> colonnes METAR (station_id, timestamp_utc, ..., source)."""
    wnd = brut["WND"]
    direc = pd.to_numeric(_champ(wnd, 0), errors="coerce")
    vit = pd.to_numeric(_champ(wnd, 3), errors="coerce") / 10.0
    # vitesse valide ; la direction peut manquer (999) : Nabeul n'en transmet presque pas en
    # 2023-2024. Ces observations gardent la vitesse (validation) avec u, v absents.
    ok_vent = _champ(wnd, 4).isin(BONS) & (vit < 999)
    ok_dir = ok_vent & _champ(wnd, 1).isin(BONS) & (direc <= 360)
    calme = ok_vent & (vit == 0)
    ok_dir = ok_dir | calme
    t = pd.to_numeric(_champ(brut["TMP"], 0), errors="coerce") / 10.0
    t = t.where(_champ(brut["TMP"], 1).isin(BONS) & (t.abs() < 99))
    td = pd.to_numeric(_champ(brut["DEW"], 0), errors="coerce") / 10.0
    td = td.where(_champ(brut["DEW"], 1).isin(BONS) & (td.abs() < 99))
    if "MA1" in brut:
        p = pd.to_numeric(_champ(brut["MA1"], 2), errors="coerce") / 10.0
        p = p.where(_champ(brut["MA1"], 3).isin(BONS) & (p < 9999)) * 100.0
    else:
        p = pd.Series(np.nan, index=brut.index)
    hr = 100.0 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
    rad = np.radians(direc.where(~calme, 0.0))
    vv = vit.where(ok_vent)
    vd = vit.where(ok_dir)
    out = pd.DataFrame({
        "station_id": "INM" + brut["STATION"].astype(str).str[:5],
        "timestamp_utc": pd.to_datetime(brut["DATE"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lat": pd.to_numeric(brut["LATITUDE"]), "lon": pd.to_numeric(brut["LONGITUDE"]),
        "height_agl_m": HAUTEUR_ANEMOMETRE, "elevation_m": pd.to_numeric(brut["ELEVATION"]),
        "wind_speed_m_s": vv, "wind_dir_deg": direc.where(ok_dir & ~calme),
        "wind_u_m_s": -vd * np.sin(rad), "wind_v_m_s": -vd * np.cos(rad),
        "temperature_c": t, "pressure_pa": p, "relative_humidity_pct": hr.clip(0, 100),
        "quality": 1.0, "source": "synop_isd"})
    return out[ok_vent].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stations", default="60729099999,60728099999")
    ap.add_argument("--annees", default="2021-2025")
    ap.add_argument("--sortie", required=True)
    args = ap.parse_args()
    a, b = (int(x) for x in args.annees.split("-"))
    sortie = Path(args.sortie)
    (sortie / "brut").mkdir(parents=True, exist_ok=True)
    tout = []
    for st in args.stations.split(","):
        for annee in range(a, b + 1):
            f = sortie / "brut" / f"{st}_{annee}.csv"
            if not f.exists():
                try:
                    with urllib.request.urlopen(URL.format(annee=annee, station=st), timeout=120) as r:
                        f.write_bytes(r.read())
                except Exception as exc:                  # noqa: BLE001
                    print(f"{st} {annee} : absent ({type(exc).__name__})")
                    continue
            brut = pd.read_csv(f, dtype=str, low_memory=False)
            if "WND" not in brut:
                print(f"{st} {annee} : fichier inattendu, ignore")
                continue
            obs = convertir(brut)
            tout.append(obs)
            print(f"{st} {annee} : {len(obs)} observations avec vitesse, "
                  f"{int(obs['wind_u_m_s'].notna().sum())} avec direction")
    if not tout:
        return 1
    df = pd.concat(tout).drop_duplicates(["station_id", "timestamp_utc"])
    for annee, g in df.groupby(df["timestamp_utc"].str[:4]):
        g.to_csv(sortie / f"synop_{annee}.csv", index=False)
    print(f"-> {sortie} : {len(df)} observations, stations {sorted(df['station_id'].unique())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
