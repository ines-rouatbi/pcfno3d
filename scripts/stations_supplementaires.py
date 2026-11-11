"""Stations tunisiennes NOAA ISD + OGIMET 2015-2022 : inventaire, telechargement, controle, sites.

Demande d'Ines (2026-11-11) : toutes les stations tunisiennes disponibles (NOAA ISD et SYNOP sur
OGIMET), vent a 10 m, 2015 a 2022, avec la part de messages valides par annee, le controle qualite
(pannes, calmes suspects, anemometres abrites comme Nabeul) et les descripteurs du site (altitude,
pente et z0 de la maille de 500 m, distance a la cote, TPI 5 et 20 km). Deux usages :
  - stations de validation SUPPLEMENTAIRES du modele national : jugees seulement, JAMAIS en entree
    (C' et D et la regle v4 restent sur les 12 aeroports METAR) ; liste
    configs/national/stations_supplementaires.csv, lue par scripts/valider_grand_tunis_reel.py ;
  - stations de la regle MOS (configs/national/regle_mos_era5.json, scripts/mos_era5.py) :
    aeroports compris.
Aeroport : code OACI parmi configs/national/stations_metar.csv ou moins de 3 km de l'un d'eux.

    python scripts/stations_supplementaires.py --dossier artifacts/national/obs_supp --telecharger

Etapes (chacune reprend ce qui est deja sur le disque ; --telecharger autorise le reseau) :
  1. inventaire : isd-history.csv de la NOAA (stations 607xxx, pays TS ou TN, actives sur les annees)
     et liste OGIMET des indicatifs SYNOP du bloc 607 (une seule page, telechargee une fois) ;
  2. fichiers annuels ISD de CHAQUE station (aeroports compris) dans <dossier>/brut/ ; tailles lues
     d'abord (HEAD), arret au-dela de MAX_OCTETS ;
  3. conversion (scripts/isd_synop.py, colonnes METAR) et controle qualite
     (pcfno3d.stations_reelles.controle_qualite : exclusions manuelles, mois en panne, plages, valeurs
     figees) ; part des creneaux de 3 h (00, 03, ..., 21 UTC) avec un message de vent valide a 30 min
     au plus, par annee, avant et apres controle ; part de calmes par annee ;
  4. indicateurs d'abri (signales, jamais exclus d'office) : vitesse moyenne de la station rapportee a
     celle de ses voisines a moins de 100 km aux memes creneaux (2015-2021) et a celle d'ERA5 10 m
     interpolee a la station (2021, fichiers ERA5 locaux ; 2022 = test, jamais lu ici) ;
  5. descripteurs du site (pcfno3d.sites_stations, lecture par plages d'octets, --descripteurs) ;
  6. ecrit <dossier>/synop_<annee>.csv (stations RETENUES pour la validation), <dossier>/qc_mensuel.csv,
     configs/national/stations_supplementaires.csv, docs/resultats/national/stations_supplementaires.csv
     (inventaire complet), _qc.csv (controle par station et par mois) et .md.

Station RETENUE (validation supplementaire) : dans le domaine national, pas un aeroport, au moins
MIN_TAUX_TEST des creneaux de 3 h de 2022 (annee de test) valides apres controle. Les donnees brutes
restent dans <dossier> (artifacts/, ignore par git) ; seuls la liste et les bilans sont versionnes.
Sources a citer : NOAA NCEI Integrated Surface Database (Global Hourly) ; OGIMET (G. Ballester Valor) ;
observations de l'INM (Tunisie) ; ERA5 (Copernicus C3S) ; voir aussi pcfno3d.sites_stations.
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pcfno3d.stations_reelles import MOTIFS, controle_qualite, preparer_observations  # noqa: E402

URL_HISTORIQUE = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
URL_OGIMET = ("https://www.ogimet.com/display_stations.php?lang=en&tipo=AND&isyn=607&oaci=&nombre=&estado="
              "&Send=Send")
AGENT = "pcfno3d-pfe/1.0 (these d'ingenieur, faible volume)"
ANNEES = tuple(range(2015, 2023))
ANNEE_TEST = 2022
ANNEE_ERA5_ABRI = 2021                 # validation ; 2022 (test) n'est jamais lu ici
# domaine national (configs/national_1km_colab.yaml, grille_reelle.json)
DOMAINE = {"lat_sud": 30.15, "lat_nord": 37.615, "lon_ouest": 7.45, "lon_est": 11.644}
DIST_AEROPORT_KM = 3.0
RAYON_VOISINS_KM = 100.0
MIN_TAUX_TEST = 0.5
# indicateurs (signalement, pas exclusion) : abri probable si la station mesure moins de 70 % du vent
# de ses voisines ET moins de 70 % du vent d'ERA5 ; calmes suspects si la part annuelle de calmes
# depasse 2 fois la mediane du reseau cette annee-la et 25 %
SEUIL_ABRI = 0.7
SEUIL_CALMES = (2.0, 0.25)
MAX_OCTETS = 1e9
PAUSE_S = 1.0
FENETRE = pd.Timedelta(minutes=30)
METAR = REPO / "configs" / "national" / "stations_metar.csv"
# coordonnees verifiees contre le relief Copernicus et l'altitude officielle (scripts/grille_reelle.py,
# STATIONS) : priment sur OGIMET et ISD
COORDONNEES_VERIFIEES = {
    "INM60728": (36.45, 10.7333, "catalogue INM (grille_reelle.STATIONS) : 9 m pour 1 m officiel ; OGIMET et ISD "
                                 "(36,467 N 10,700 E) donnent 45 a 55 m, douteux"),
}
ERA5_LOCAL = Path.home() / "AppData" / "Local" / "pcfno3d" / "era5" / "tunisie"
SORTIE_DOC = REPO / "docs" / "resultats" / "national"
SORTIE_CFG = REPO / "configs" / "national" / "stations_supplementaires.csv"


def _isd_synop():
    spec = importlib.util.spec_from_file_location("isd_synop", REPO / "scripts" / "isd_synop.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def distance_km(la1, lo1, la2, lo2) -> float:
    p1, p2 = math.radians(la1), math.radians(la2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def dans_domaine(lat: float, lon: float) -> bool:
    return (DOMAINE["lat_sud"] <= lat <= DOMAINE["lat_nord"]) and (DOMAINE["lon_ouest"] <= lon <= DOMAINE["lon_est"])


# ---------------------------------------------------------------------- inventaire
def inventaire_isd(chemin: Path, annees=ANNEES) -> pd.DataFrame:
    """Stations tunisiennes de isd-history.csv (identifiant 607xxx, pays TS ou TN), actives sur
    `annees` (debut <= 31/12 de la derniere, fin >= 01/01 de la premiere), coordonnees connues."""
    h = pd.read_csv(chemin, dtype=str)
    for c in ("LAT", "LON", "ELEV(M)", "BEGIN", "END"):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    t = h[h["USAF"].str.startswith("607") & h["CTRY"].isin(["TS", "TN"]) & h["LAT"].notna()
          & (h["BEGIN"] <= annees[-1] * 10000 + 1231) & (h["END"] >= annees[0] * 10000 + 101)]
    return pd.DataFrame({
        "isd": t["USAF"] + t["WBAN"], "usaf": t["USAF"],
        "wmo": t["USAF"].where(t["USAF"].str[5] == "0").str[:5],    # 607403 : identifiant ISD seul
        "icao_isd": t["ICAO"].fillna(""), "nom_isd": t["STATION NAME"].str.title(),
        "lat_isd": t["LAT"], "lon_isd": t["LON"], "elev_isd": t["ELEV(M)"],
        "debut_isd": t["BEGIN"].astype(int).astype(str), "fin_isd": t["END"].astype(int).astype(str)}
    ).reset_index(drop=True)


def _dms(s: str) -> float:
    """« 36-58-33N » / « 008-52-33E » -> degres decimaux."""
    m = re.fullmatch(r"(\d+)-(\d+)(?:-(\d+))?([NSEW])", s.strip())
    if not m:
        return math.nan
    v = int(m[1]) + int(m[2]) / 60 + (int(m[3]) if m[3] else 0) / 3600
    return -v if m[4] in "SW" else v


COLONNES_OGIMET = ["wmo", "icao_ogimet", "nom_ogimet", "lat_ogimet", "lon_ogimet", "alt_ogimet", "ouverture_ogimet"]


def lire_ogimet(chemin: Path) -> pd.DataFrame:
    """Page OGIMET display_stations (bloc 607) -> stations de Tunisie OUVERTES (sans date de
    fermeture) : wmo, icao, nom, lat, lon, altitude, ouverture."""
    t = chemin.read_text(encoding="latin-1")
    lignes = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", t, flags=re.S | re.I):
        c = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", x))).strip()
             for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S | re.I)]
        if len(c) < 10 or c[4] != "Tunisia" or c[9] != "----":
            continue
        wmo = re.search(r"\d{5}", c[1])
        icao = re.search(r"\b[A-Z]{4}\b", c[2].split("'")[0])
        lignes.append({"wmo": wmo[0] if wmo else "", "icao_ogimet": icao[0] if icao else "",
                       "nom_ogimet": c[3], "lat_ogimet": _dms(c[5]), "lon_ogimet": _dms(c[6]),
                       "alt_ogimet": pd.to_numeric(c[7], errors="coerce"), "ouverture_ogimet": c[8]})
    return pd.DataFrame(lignes, columns=COLONNES_OGIMET).drop_duplicates("wmo")


def fusionner(isd: pd.DataFrame, ogi: pd.DataFrame, aeroports: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par station ISD (+ les SYNOP d'OGIMET absents d'ISD). Coordonnees retenues :
    celles d'OGIMET (WIGOS, a la seconde d'arc) si elles existent, sinon celles d'ISD. Aeroport :
    code OACI parmi les 12 METAR, ou moins de DIST_AEROPORT_KM d'un des 12."""
    t = isd.merge(ogi.reindex(columns=COLONNES_OGIMET), on="wmo", how="outer")
    t["source"] = np.where(t["isd"].notna() & t["nom_ogimet"].notna(), "ISD + OGIMET",
                           np.where(t["isd"].notna(), "ISD", "OGIMET"))
    t["lat"] = t["lat_ogimet"].fillna(t["lat_isd"]).astype(float)
    t["lon"] = t["lon_ogimet"].fillna(t["lon_isd"]).astype(float)
    t["elevation_m"] = t["alt_ogimet"].fillna(t["elev_isd"]).astype(float)
    t["nom"] = t["nom_ogimet"].fillna(t["nom_isd"])
    t["icao"] = t["icao_ogimet"].fillna("").where(t["icao_ogimet"].fillna("") != "", t["icao_isd"].fillna(""))
    t["ecart_isd_ogimet_km"] = [
        round(distance_km(a, b, c, d), 1) if np.isfinite(np.array([a, b, c, d], float)).all() else math.nan
        for a, b, c, d in zip(t["lat_isd"], t["lon_isd"], t["lat_ogimet"], t["lon_ogimet"])]
    t["code"] = np.where(t["wmo"].notna(), "INM" + t["wmo"].fillna(""), "ISD" + t["isd"].fillna(""))
    t["coordonnees"] = np.where(t["lat_ogimet"].notna(), "OGIMET (WIGOS)", "ISD")
    for code, (la, lo, src) in COORDONNEES_VERIFIEES.items():
        m = t["code"] == code
        t.loc[m, "lat"], t.loc[m, "lon"], t.loc[m, "coordonnees"] = la, lo, src
    codes = set(aeroports["code"])
    ae, dist = [], []
    for r in t.itertuples():
        d = {a.code: distance_km(r.lat, r.lon, a.lat, a.lon) for a in aeroports.itertuples()}
        proche = min(d, key=d.get)
        icaos = {r.icao, str(r.icao_isd)} & codes
        ae.append(sorted(icaos)[0] if icaos else (proche if d[proche] < DIST_AEROPORT_KM else ""))
        dist.append(round(d[proche], 1))
    t["aeroport_metar"] = ae
    t["dist_aeroport_km"] = dist
    t["dans_domaine"] = [dans_domaine(a, b) for a, b in zip(t["lat"], t["lon"])]
    return t.sort_values(["wmo", "isd"]).reset_index(drop=True)


# ---------------------------------------------------------------------- telechargement
def _get(url: str, timeout: int = 120, methode: str = "GET"):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}, method=methode),
                                  timeout=timeout)


def telecharger_fichier(url: str, f: Path) -> bool:
    if f.exists():
        return True
    try:
        with _get(url) as r:
            data = r.read()
    except Exception as exc:                              # noqa: BLE001
        print(f"absent : {url} ({type(exc).__name__})")
        return False
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(f)
    time.sleep(PAUSE_S)
    return True


def telecharger_isd(ids, brut: Path, url: str, annees=ANNEES) -> int:
    """Fichiers annuels ISD de `ids` dans `brut` ; taille verifiee AVANT (HEAD) : arret au-dela de
    MAX_OCTETS. Renvoie le nombre d'octets sur le disque."""
    a_prendre, total = [], 0
    for s in ids:
        for an in annees:
            f = brut / f"{s}_{an}.csv"
            if f.exists():
                total += f.stat().st_size
                continue
            try:
                with _get(url.format(annee=an, station=s), timeout=30, methode="HEAD") as r:
                    total += int(r.headers.get("Content-Length", 0))
                a_prendre.append((s, an, f))
            except Exception as exc:                      # noqa: BLE001
                print(f"{s} {an} : absent d'ISD ({type(exc).__name__})")
            time.sleep(0.2)
    if total > MAX_OCTETS:
        raise SystemExit(f"telechargement de {total / 1e6:.0f} Mo > {MAX_OCTETS / 1e6:.0f} Mo : arret")
    print(f"ISD : {len(a_prendre)} fichiers a telecharger, {total / 1e6:.1f} Mo au total", flush=True)
    for s, an, f in a_prendre:
        telecharger_fichier(url.format(annee=an, station=s), f)
    return sum(f.stat().st_size for f in brut.glob("*.csv"))


# ---------------------------------------------------------------------- taux et controle
def creneaux(annee: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{annee}-01-01", f"{annee}-12-31 21:00", freq="3h")


def au_creneau(obs: pd.DataFrame) -> pd.DataFrame:
    """Mesures ramenees au creneau de 3 h le plus proche (a FENETRE au plus), la plus proche
    gardee : colonnes creneau, wind_speed_m_s, wind_u_m_s, wind_v_m_s."""
    if obs.empty:
        return pd.DataFrame(columns=["creneau", "wind_speed_m_s", "wind_u_m_s", "wind_v_m_s"])
    c = obs["t"].dt.round("3h")
    d = (obs["t"] - c).abs()
    o = obs.assign(creneau=c, _d=d)[d <= FENETRE].sort_values("_d")
    return o.drop_duplicates("creneau")[["creneau", "wind_speed_m_s", "wind_u_m_s", "wind_v_m_s"]] \
        .sort_values("creneau").reset_index(drop=True)


def taux_valides(t: pd.Series, annee: int) -> float:
    """Part des creneaux de 3 h de l'annee avec au moins un message a FENETRE au plus."""
    if len(t) == 0:
        return 0.0
    c = t.dt.round("3h")
    ok = c[((t - c).abs() <= FENETRE) & (c.dt.year == annee)].unique()
    return float(len(ok) / len(creneaux(annee)))


def convertir_station(isd_mod, brut: Path, isd_id: str, code: str, annees=ANNEES) -> pd.DataFrame:
    """Fichiers ISD de la station -> colonnes METAR (isd_synop.convertir), station_id = code."""
    out = []
    for an in annees:
        f = brut / f"{isd_id}_{an}.csv"
        if not f.exists():
            continue
        b = pd.read_csv(f, dtype=str, low_memory=False)
        if "WND" not in b:
            continue
        for c in ("TMP", "DEW"):
            if c not in b:
                b[c] = "+9999,9"
        o = isd_mod.convertir(b)
        o["station_id"] = code
        out.append(o)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def lat_lon_fichiers(brut: Path, isd_id: str, annees=ANNEES) -> tuple[float, float]:
    """Coordonnees les plus frequentes dans les messages ISD (verification de l'inventaire)."""
    la, lo = [], []
    for an in annees:
        f = brut / f"{isd_id}_{an}.csv"
        if f.exists():
            b = pd.read_csv(f, usecols=["LATITUDE", "LONGITUDE"], dtype=float)
            la.append(b["LATITUDE"])
            lo.append(b["LONGITUDE"])
    if not la:
        return math.nan, math.nan
    return float(pd.concat(la).mode().iloc[0]), float(pd.concat(lo).mode().iloc[0])


def era5_10m_aux_stations(dossier_era5: Path, annee: int, lat, lon) -> pd.DataFrame | None:
    """Vitesse ERA5 10 m interpolee (bilineaire) aux stations, fichiers era5_sl_<annee>*.nc locaux :
    colonnes creneau, k (indice de station), v_era5. None sans fichier."""
    import xarray as xr

    fs = sorted(Path(dossier_era5).glob(f"era5_sl_{annee}*.nc"))
    if not fs:
        return None
    morceaux = []
    for f in fs:
        with xr.open_dataset(f) as d:
            tv = "valid_time" if "valid_time" in d.coords else "time"
            pts = d[["u10", "v10"]].interp(latitude=xr.DataArray(np.asarray(lat), dims="k"),
                                           longitude=xr.DataArray(np.asarray(lon), dims="k")).load()
            t = pd.to_datetime(pts[tv].values)
            vit = np.hypot(pts["u10"].values, pts["v10"].values)              # (t, k)
            morceaux.append(pd.DataFrame({"creneau": np.repeat(t, vit.shape[1]),
                                          "k": np.tile(np.arange(vit.shape[1]), len(t)),
                                          "v_era5": vit.ravel()}))
    e = pd.concat(morceaux).drop_duplicates(["creneau", "k"])
    return e[e["creneau"].dt.hour % 3 == 0]


def indicateurs_abri(inv: pd.DataFrame, series: dict[str, pd.DataFrame], dossier_era5: Path | None) -> pd.DataFrame:
    """Rapports de vitesse moyenne station / voisines (< RAYON_VOISINS_KM, memes creneaux, annees
    avant le test) et station / ERA5 10 m (ANNEE_ERA5_ABRI). series : {code: au_creneau(gardees)}."""
    inv = inv.copy()
    avant = {c: s[s["creneau"].dt.year < ANNEE_TEST].set_index("creneau")["wind_speed_m_s"]
             for c, s in series.items()}
    for k, r in inv.iterrows():
        c = r["code"]
        if c not in avant or avant[c].empty:
            continue
        vois = [o.code for o in inv.itertuples() if o.code != c and o.code in avant
                and distance_km(r["lat"], r["lon"], o.lat, o.lon) < RAYON_VOISINS_KM]
        inv.loc[k, "n_voisines"] = len(vois)
        if vois:
            m = pd.concat({v: avant[v] for v in vois}, axis=1).mean(axis=1)
            j = pd.concat({"s": avant[c], "v": m}, axis=1).dropna()
            if len(j) >= 100 and j["v"].mean() > 0:
                inv.loc[k, "rapport_vitesse_voisines"] = round(float(j["s"].mean() / j["v"].mean()), 3)
    if dossier_era5 is not None:
        e = era5_10m_aux_stations(dossier_era5, ANNEE_ERA5_ABRI, inv["lat"], inv["lon"])
        if e is not None:
            for k, r in inv.iterrows():
                s = series.get(r["code"])
                if s is None or s.empty:
                    continue
                ek = e[e["k"] == inv.index.get_loc(k)].set_index("creneau")["v_era5"]
                j = pd.concat({"s": s.set_index("creneau")["wind_speed_m_s"], "e": ek}, axis=1).dropna()
                if len(j) >= 100:
                    inv.loc[k, "rapport_vitesse_era5_2021"] = round(float(j["s"].mean() / j["e"].mean()), 3)
                    inv.loc[k, "n_creneaux_era5_2021"] = len(j)
    rv = inv.get("rapport_vitesse_voisines", pd.Series(np.nan, index=inv.index))
    re5 = inv.get("rapport_vitesse_era5_2021", pd.Series(np.nan, index=inv.index))
    inv["abri_probable"] = (rv < SEUIL_ABRI) & (re5 < SEUIL_ABRI)
    return inv


def controler(inv: pd.DataFrame, brut: Path, annees=ANNEES, dossier_era5: Path | None = None):
    """Taux de messages valides, calmes et controle qualite de chaque station de l'inventaire.
    Renvoie (inventaire complete, rapport mensuel de toutes les stations, mesures brutes converties
    des stations retenues pour la validation)."""
    isd_mod = _isd_synop()
    inv = inv.copy()
    rapports, brutes_tout, series = [], {}, {}
    for k, r in inv.iterrows():
        if pd.isna(r["isd"]):
            continue
        la, lo = lat_lon_fichiers(brut, r["isd"], annees)
        inv.loc[k, "lat_messages"], inv.loc[k, "lon_messages"] = la, lo
        brutes = convertir_station(isd_mod, brut, r["isd"], r["code"], annees)
        if brutes.empty:
            continue
        obs = preparer_observations(brutes)                   # u, v, calme : meme lecture qu'en validation
        gardees, rap = controle_qualite(obs)
        for an in annees:
            inv.loc[k, f"taux_brut_{an}"] = round(taux_valides(obs["t"], an), 4)
            inv.loc[k, f"taux_qc_{an}"] = round(taux_valides(gardees["t"], an), 4)
            g = gardees[gardees["t"].dt.year == an]
            inv.loc[k, f"calmes_{an}"] = round(float((g["wind_speed_m_s"] == 0).mean()), 4) if len(g) else np.nan
        for m in MOTIFS:
            inv.loc[k, f"n_{m}"] = int(rap[m].sum())
        inv.loc[k, "vitesse_moyenne_qc"] = round(float(gardees["wind_speed_m_s"].mean()), 2)
        inv.loc[k, "mois_ecartes"] = ", ".join(f"{x.mois} ({x.motif_mois})" for x in rap.itertuples() if x.mois_ecarte)
        rapports.append(rap)
        brutes_tout[r["code"]] = brutes
        series[r["code"]] = au_creneau(gardees)
        print(f"{r['code']} {r['nom']} : " + " ".join(f"{an} {inv.loc[k, f'taux_qc_{an}']:.0%}" for an in annees),
              flush=True)
    # calmes suspects : part annuelle > 2 x la mediane du reseau et > 25 %
    suspects = []
    for k, r in inv.iterrows():
        ans = [an for an in annees if f"calmes_{an}" in inv
               and r[f"calmes_{an}"] > max(SEUIL_CALMES[0] * inv[f"calmes_{an}"].median(), SEUIL_CALMES[1])]
        suspects.append(", ".join(map(str, ans)))
    inv["calmes_suspects"] = suspects
    inv = indicateurs_abri(inv, series, dossier_era5)
    taux_test = inv.get(f"taux_qc_{ANNEE_TEST}", pd.Series(0.0, index=inv.index)).fillna(0.0)
    inv["retenue"] = inv["dans_domaine"].astype(bool) & (inv["aeroport_metar"] == "") & (taux_test >= MIN_TAUX_TEST)
    motif = []
    for k, r in inv.iterrows():
        if r["aeroport_metar"]:
            motif.append(f"aeroport METAR {r['aeroport_metar']} (deja juge, en entree)")
        elif not r["dans_domaine"]:
            motif.append("hors du domaine national")
        elif taux_test[k] < MIN_TAUX_TEST:
            motif.append(f"moins de {MIN_TAUX_TEST:.0%} de creneaux valides en {ANNEE_TEST}")
        else:
            motif.append("")
    inv["motif_non_retenue"] = motif
    retenues = set(inv.loc[inv["retenue"], "code"])
    obs_ret = pd.concat([b for c, b in brutes_tout.items() if c in retenues], ignore_index=True) \
        if retenues else pd.DataFrame()
    rapport = pd.concat(rapports, ignore_index=True) if rapports else pd.DataFrame()
    return inv, rapport, obs_ret


def ajouter_descripteurs(inv: pd.DataFrame, cache: Path, fichier: Path) -> tuple[pd.DataFrame, int]:
    """Descripteurs du site (pcfno3d.sites_stations) de chaque station, gardes dans `fichier`
    (une ligne par code, recalcules seulement si les coordonnees changent)."""
    from pcfno3d.sites_stations import descripteurs, trait_de_cote

    deja = pd.read_csv(fichier, dtype={"code": str}) if fichier.exists() else pd.DataFrame(columns=["code"])
    compteur: dict = {}
    cote = trait_de_cote(cache, compteur)
    lignes = []
    for r in inv.itertuples():
        old = deja[(deja["code"] == r.code)]
        if len(old) and abs(float(old["lat"].iloc[0]) - r.lat) < 1e-6 and abs(float(old["lon"].iloc[0]) - r.lon) < 1e-6:
            lignes.append(old.iloc[0].to_dict())
            continue
        d = descripteurs(r.lat, r.lon, cache, cote=cote, compteur=compteur)
        lignes.append({"code": r.code, "lat": r.lat, "lon": r.lon, **d})
        print(f"site {r.code} {r.nom} : {d}", flush=True)
    t = pd.DataFrame(lignes)
    t.to_csv(fichier, index=False)
    return inv.merge(t.drop(columns=["lat", "lon"]), on="code", how="left"), int(compteur.get("octets", 0))


# ---------------------------------------------------------------------- ecriture
def _pct(x) -> str:
    return "—" if pd.isna(x) else f"{100 * x:.0f}"


def _num(x, f="{:.0f}") -> str:
    return "—" if pd.isna(x) else f.format(x)


def ecrire_doc(inv: pd.DataFrame, n_octets: dict, chemin_md: Path, annees=ANNEES) -> None:
    ret = inv[inv["retenue"]]
    L = ["# Stations tunisiennes NOAA ISD et OGIMET, 2015–2022 (modèle national)", "",
         f"Écrit par `scripts/stations_supplementaires.py` (demandes d'Ines du 11/11/2026). Sources : NOAA NCEI ISD "
         f"(`isd-history.csv` et fichiers annuels {annees[0]}–{annees[-1]}), liste OGIMET des indicatifs SYNOP du bloc "
         "607, ERA5 10 m locale (2021, indicateur d'abri), Copernicus DEM GLO-30/GLO-90, ESA WorldCover 2021, "
         "Natural Earth (trait de côte).", "",
         "**Deux usages.** (1) Stations de validation **supplémentaires** du modèle national : jugées seulement, "
         "**jamais en entrée** (C' et D gardent les 12 aéroports METAR de `stations.json`, la règle v4 `60671e84…` "
         "aussi) ; bilan à part du DSO7 (`validation_reelle/<jeu>/stations_supp_*.csv`, `bilan_supp.json`), hors "
         "décision v4. Les mettre en entrée serait une décision d'Ines. (2) Stations de la règle MOS proposée "
         "(`configs/national/regle_mos_era5.json`), aéroports compris.", "",
         f"- stations tunisiennes actives en {annees[0]}–{annees[-1]} : **{len(inv)}** (ISD : "
         f"{int(inv['isd'].notna().sum())} ; OGIMET : {int(inv['nom_ogimet'].notna().sum())} indicatifs SYNOP ouverts) ;",
         f"- enregistrements des 12 aéroports METAR (code OACI ou moins de {DIST_AEROPORT_KM:.0f} km) : "
         f"**{int((inv['aeroport_metar'] != '').sum())}** (aéroports couverts : "
         f"{inv.loc[inv['aeroport_metar'] != '', 'aeroport_metar'].nunique()} ; Monastir a deux enregistrements, "
         "le SYNOP INM 60740 et l'identifiant ISD 607403) ; hors du domaine national : "
         f"**{int((~inv['dans_domaine'].astype(bool)).sum())}** ;",
         f"- **retenues** comme stations de validation supplémentaires : **{len(ret)}** (dans le domaine, pas un "
         f"aéroport, au moins {MIN_TAUX_TEST:.0%} des créneaux de 3 h de {ANNEE_TEST} valides après contrôle) : "
         + ", ".join(f"{r.nom} ({r.code})" for r in ret.itertuples()) + " ;",
         f"- téléchargé : ISD {n_octets.get('isd', 0) / 1e6:.0f} Mo, sites (plages d'octets des COG + trait de côte) "
         f"{n_octets.get('sites', 0) / 1e6:.0f} Mo ; dossiers ignorés par git, jamais versionnés.", "",
         "## Liste et part de messages valides par année", "",
         "Part (%) des créneaux de 3 h (00, 03, …, 21 UTC ; 2 920 par an) avec un message de vent valide (qualité ISD "
         "non suspecte, vitesse, et direction ou calme) à 30 min au plus, **après contrôle qualité** (avant contrôle : "
         "colonnes `taux_brut_*` du CSV).", "",
         "| Code | Nom | OACI | lat | lon | alt. (m) | aéroport METAR | " + " | ".join(str(a) for a in annees)
         + " | retenue (validation) |",
         "|---" * (8 + len(annees)) + "|"]
    for r in inv.itertuples():
        d = r._asdict()
        L.append(f"| {r.code} | {r.nom} | {r.icao or '—'} | {r.lat:.4f} | {r.lon:.4f} | {_num(r.elevation_m)} | "
                 f"{r.aeroport_metar or '—'} | " + " | ".join(_pct(d.get(f"taux_qc_{a}")) for a in annees)
                 + f" | {'**oui**' if r.retenue else 'non'} |")
    L += ["", "## Site de chaque station", "",
          "Maille de 500 m de la grille nationale, recalculée autour de la station avec le code de "
          "`scripts/grille_reelle.py` (`pcfno3d.sites_stations`) : altitude et pente du relief GLO-30 lissé comme la "
          "grille, z0 et occupation majoritaire de 5 × 5 sous-points WorldCover (moyenne logarithmique) ; TPI = "
          "altitude GLO-90 de la station moins l'altitude moyenne dans un disque de 5 et 20 km (mer à 0 m) ; "
          "distance au trait de côte Natural Earth 1:10 M (lagunes et sebkhas non comprises).", "",
          "| Code | Nom | alt. officielle | alt. maille | pente (°) | z0 (m) | occupation | TPI 5 km | TPI 20 km "
          "| côte (km) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in inv.itertuples():
        d = r._asdict()
        L.append(f"| {r.code} | {r.nom} | {_num(r.elevation_m)} | {_num(d.get('z_maille_500m_m'))} | "
                 f"{_num(d.get('pente_maille_deg'), '{:.1f}')} | {_num(d.get('z0_maille_m'), '{:.3f}')} | "
                 f"{d.get('occupation_maille') if isinstance(d.get('occupation_maille'), str) else '—'} | "
                 f"{_num(d.get('tpi_5km_m'))} | {_num(d.get('tpi_20km_m'))} | {_num(d.get('distance_cote_km'), '{:.1f}')} |")
    L += ["", "## Contrôle qualité", "",
          "Règles existantes de `pcfno3d.stations_reelles` (mêmes que pour Nabeul 2021) et deux contrôles du "
          "contrat ANEO adaptés au SYNOP (`controle_qualite`) :",
          "- **exclusions manuelles** (`EXCLUSIONS` : Nabeul 2023–2024, hors des années étudiées ici) ;",
          "- **mois en panne** : vent non nul dans moins de 10 % des messages du mois (au moins 20 messages) ;",
          "- **calme** : 0 m/s gardé comme mesure (u = v = 0) ; une vitesse sans direction (et non calme) est "
          "écartée (u, v inconnus) ;",
          "- **plages** : vitesse hors de 0–60 m/s, direction hors de 0–360°, |u| ou |v| > 60 m/s ;",
          "- **valeur figée** : même vitesse (≥ 0,5 m/s) et même direction sur au moins 3 messages consécutifs "
          "couvrant au moins 24 h (le contrat dit 30 min pour des données à 10 s ; un SYNOP toutes les 3 h, "
          "quantifié à 1 m/s et 10°, répète banalement deux ou trois messages par vent établi) ;",
          "- doublons (station, instant) : le premier message est gardé.",
          "",
          "Indicateurs **signalés, jamais exclus d'office** (décision d'Ines, station par station) :",
          f"- **calmes suspects** : part annuelle de calmes > {SEUIL_CALMES[0]:.0f} × la médiane du réseau cette "
          f"année-là et > {SEUIL_CALMES[1]:.0%} ;",
          f"- **abri probable** : vitesse moyenne < {SEUIL_ABRI:.0%} de celle des voisines à moins de "
          f"{RAYON_VOISINS_KM:.0f} km (mêmes créneaux, {annees[0]}–{ANNEE_TEST - 1}) **et** < {SEUIL_ABRI:.0%} de "
          f"celle d'ERA5 10 m à la station ({ANNEE_ERA5_ABRI}, fichiers locaux ; {ANNEE_TEST} = test, jamais lu).", "",
          "| Code | Nom | écartés (exclusion / panne / plage / figée) | mois écartés | calmes 2021 | années à calmes "
          "suspects | v / voisines | v / ERA5 2021 | abri probable |", "|---|---|---|---|---|---|---|---|---|"]
    for r in inv.itertuples():
        d = r._asdict()
        n = ["—" if pd.isna(inv.loc[r.Index, f"n_{m}"]) else str(int(inv.loc[r.Index, f"n_{m}"]))
             for m in MOTIFS] if f"n_{MOTIFS[0]}" in inv else ["—"] * 4
        L.append(f"| {r.code} | {r.nom} | {' / '.join(n)} | {d.get('mois_ecartes') or 'aucun'} | "
                 f"{_pct(d.get('calmes_2021'))} % | {d.get('calmes_suspects') or '—'} | "
                 f"{_num(d.get('rapport_vitesse_voisines'), '{:.2f}')} | {_num(d.get('rapport_vitesse_era5_2021'), '{:.2f}')} "
                 f"| {'**oui**' if d.get('abri_probable') else 'non'} |")
    L += ["", "Détail par station et par mois : `stations_supplementaires_qc.csv`.", "", "## Remarques", ""]
    rq = []
    for r in inv.itertuples():
        if np.isfinite(r.ecart_isd_ogimet_km) and r.ecart_isd_ogimet_km > 2:
            rq.append(f"- {r.code} {r.nom} : coordonnées ISD et OGIMET distantes de {r.ecart_isd_ogimet_km:.0f} km "
                      f"(ISD {r.lat_isd:.3f}, {r.lon_isd:.3f} ; OGIMET {r.lat_ogimet:.3f}, {r.lon_ogimet:.3f}) ; "
                      "les coordonnées OGIMET (WIGOS) sont retenues.")
    for code, (la, lo, src) in COORDONNEES_VERIFIEES.items():
        rq.append(f"- {code} : coordonnées {la}, {lo} retenues ({src}).")
    for r in inv.itertuples():
        z = r._asdict().get("z_maille_500m_m")
        if z is not None and np.isfinite(z) and np.isfinite(r.elevation_m) and abs(z - r.elevation_m) > 30:
            rq.append(f"- {r.code} {r.nom} : altitude de la maille {z:.0f} m pour {r.elevation_m:.0f} m officiels "
                      "(position ou altitude officielle à vérifier).")
    L += rq or ["- aucune."]
    L += ["", "## Utilisation", "", "```bash",
          "python scripts/stations_supplementaires.py --dossier artifacts/national/obs_supp --telecharger --descripteurs",
          "python scripts/valider_grand_tunis_reel.py -c <config nationale> --obs <obs> --jeu test \\",
          "    --stations-supp configs/national/stations_supplementaires.csv --obs-supp <dossier des synop_*.csv>",
          "# run deja valide (bilan.json des aeroports present) : ajouter --supp-seul, rien d'autre n'est reecrit",
          "```", ""]
    chemin_md.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dossier", default=str(REPO / "artifacts" / "national" / "obs_supp"))
    ap.add_argument("--annees", default=f"{ANNEES[0]}-{ANNEES[-1]}")
    ap.add_argument("--telecharger", action="store_true", help="autorise le reseau (NOAA, une page OGIMET)")
    ap.add_argument("--descripteurs", action="store_true",
                    help="descripteurs du site (relief, z0, cote ; lecture par plages d'octets)")
    ap.add_argument("--era5", default=str(ERA5_LOCAL), help="dossier des era5_sl_*.nc (indicateur d'abri)")
    ap.add_argument("--sortie-doc", default=str(SORTIE_DOC))
    ap.add_argument("--sortie-config", default=str(SORTIE_CFG))
    args = ap.parse_args(argv)
    a, b = (int(x) for x in args.annees.split("-"))
    annees = tuple(range(a, b + 1))
    dossier = Path(args.dossier)
    brut = dossier / "brut"
    brut.mkdir(parents=True, exist_ok=True)
    hist, ogi_html = dossier / "isd-history.csv", dossier / "ogimet" / "stations_607.html"
    if args.telecharger:
        telecharger_fichier(URL_HISTORIQUE, hist)
        telecharger_fichier(URL_OGIMET, ogi_html)
    if not hist.exists():
        raise SystemExit(f"{hist} absent : relancer avec --telecharger")
    isd = inventaire_isd(hist, annees)
    ogi = lire_ogimet(ogi_html) if ogi_html.exists() else pd.DataFrame(columns=COLONNES_OGIMET)
    inv = fusionner(isd, ogi, pd.read_csv(METAR))
    print(f"inventaire : {len(inv)} stations ({int(inv['isd'].notna().sum())} ISD, "
          f"{int(inv['nom_ogimet'].notna().sum())} OGIMET)", flush=True)
    octets = {}
    if args.telecharger:
        absents = inv[inv["isd"].isna()]
        if len(absents):
            print("SYNOP d'OGIMET absents d'ISD (non telecharges) :", ", ".join(absents["code"]))
        octets["isd"] = telecharger_isd(inv["isd"].dropna(), brut, _isd_synop().URL, annees)
    else:
        octets["isd"] = sum(f.stat().st_size for f in brut.glob("*.csv"))
    era5 = Path(args.era5) if args.era5 and Path(args.era5).exists() else None
    inv, rapport, obs = controler(inv, brut, annees, era5)
    if args.descripteurs:
        inv, octets["sites"] = ajouter_descripteurs(inv, dossier.parent / "sites_cache", dossier / "descripteurs.csv")
    elif (dossier / "descripteurs.csv").exists():
        d = pd.read_csv(dossier / "descripteurs.csv", dtype={"code": str})
        inv = inv.merge(d.drop(columns=["lat", "lon"]), on="code", how="left")
    for an in annees:
        g = obs[obs["timestamp_utc"].str[:4] == str(an)] if len(obs) else obs
        if len(g):
            g.drop_duplicates(["station_id", "timestamp_utc"]).to_csv(dossier / f"synop_{an}.csv", index=False)
    if len(rapport):
        rapport.to_csv(dossier / "qc_mensuel.csv", index=False)
    sd = Path(args.sortie_doc)
    sd.mkdir(parents=True, exist_ok=True)
    tete = ["code", "wmo", "isd", "icao", "nom", "lat", "lon", "coordonnees", "elevation_m", "source", "dans_domaine",
            "aeroport_metar", "dist_aeroport_km", "retenue", "motif_non_retenue"]
    sites = ["z_dem_point_m", "z_maille_500m_m", "pente_maille_deg", "z0_maille_m", "occupation_maille",
             "classes_worldcover", "tpi_5km_m", "tpi_20km_m", "distance_cote_km"]
    qc = (["vitesse_moyenne_qc", "mois_ecartes", *[f"n_{m}" for m in MOTIFS], "calmes_suspects",
           "n_voisines", "rapport_vitesse_voisines", "rapport_vitesse_era5_2021", "n_creneaux_era5_2021",
           "abri_probable"]
          + [f"{p}_{an}" for p in ("taux_qc", "taux_brut", "calmes") for an in annees])
    fin = ["nom_isd", "lat_isd", "lon_isd", "elev_isd", "debut_isd", "fin_isd", "lat_messages", "lon_messages",
           "nom_ogimet", "lat_ogimet", "lon_ogimet", "alt_ogimet", "ecart_isd_ogimet_km"]
    inv.reindex(columns=tete + sites + qc + fin).to_csv(sd / "stations_supplementaires.csv", index=False)
    if len(rapport):
        rapport.merge(inv[["code", "nom"]], left_on="station_id", right_on="code").drop(columns="code") \
            .to_csv(sd / "stations_supplementaires_qc.csv", index=False)
    ecrire_doc(inv, octets, sd / "stations_supplementaires.md", annees)
    ret = inv[inv["retenue"]]
    cfg = pd.DataFrame({"code": ret["code"], "nom": ret["nom"], "lat": ret["lat"].round(5),
                        "lon": ret["lon"].round(5), "elevation_m": ret["elevation_m"], "wmo": ret["wmo"],
                        "isd": ret["isd"], "retenue": 1,
                        "abri_probable": ret["abri_probable"].astype(int),
                        "source": "SYNOP INM via NOAA ISD (scripts/stations_supplementaires.py) ; coordonnees "
                                  + ret["coordonnees"]})
    cfg.to_csv(args.sortie_config, index=False)
    print(f"{len(ret)} stations retenues -> {args.sortie_config} ; bilans -> {sd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
