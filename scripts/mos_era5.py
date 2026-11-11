"""MOS de l'ERA5 10 m aux stations tunisiennes, sous la regle configs/national/regle_mos_era5.json.

Un modele leger (ridge ou gradient boosting) predit l'erreur de l'ERA5 a 10 m (u, v) a une station a
partir des descripteurs du site, de l'heure, du mois, du vent ERA5 et de la stabilite ; il est juge
en validation croisee PAR SITE (chaque site retire de l'ajustement, puis juge) contre l'ERA5 seule.
La regle est verifiee (empreinte) avant toute lecture et toutes ses valeurs sont lues dans le fichier.

    python scripts/mos_era5.py era5 --stations <inventaire.csv> --sortie artifacts/national/mos/era5_openmeteo
    python scripts/mos_era5.py preparer --inventaire docs/resultats/national/stations_supplementaires.csv \
        --brut artifacts/national/obs_supp/brut --era5 artifacts/national/mos/era5_openmeteo \
        --relief-era5 <era5_sl_2021_m01-06.nc> --sortie artifacts/national/mos/paires
    python scripts/mos_era5.py choisir --paires artifacts/national/mos/paires --sortie artifacts/national/mos/<date>
    python scripts/mos_era5.py tester  --paires artifacts/national/mos/paires --sortie artifacts/national/mos/<date>

  era5     : series horaires ERA5 servies par Open-Meteo (models=era5, cell_selection=nearest), une
             requete par station et par annee, pause proportionnelle au poids de la requete ;
  preparer : paires (station, creneau de 3 h) mesure + ERA5 + variables, un fichier PAR ANNEE
             (paires_<annee>.csv.gz) : l'etape choisir ne lit jamais paires_2022 ;
  choisir  : grille de la regle, validation croisee par site sur 2021 (ajustement 2015-2020), choix,
             critere « bat ERA5 » sur la validation (IC bootstrap), modeles de la configuration choisie
             gardes par site (aucun reajustement au test) ; refuse un dossier ou choix.json existe ;
  tester   : lit 2022 UNE fois, seulement si la validation bat ERA5 selon le critere ; refuse si
             test.json existe.
"""
from __future__ import annotations

import argparse
import os
import importlib.util
import json
import math
import pickle
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))      # sans sous-processus wmic (Windows)
sys.path.insert(0, str(REPO / "src"))

from pcfno3d.regles import verifier  # noqa: E402

REGLE = REPO / "configs" / "national" / "regle_mos_era5.json"
URL_OM = "https://archive-api.open-meteo.com/v1/archive"
VARS_OM = ("wind_speed_10m", "wind_direction_10m", "wind_speed_100m", "wind_direction_100m", "temperature_2m",
           "boundary_layer_height", "shortwave_radiation", "cloud_cover")
AGENT = "pcfno3d-pfe/1.0 (these d'ingenieur, faible volume)"
APPELS_PAR_HEURE = 4500.0              # sous la limite Open-Meteo (5 000 appels ponderes par heure)
R_SITE_KM = 10.0
MIN_TAUX_JUGE = 0.5


def _script(nom):
    spec = importlib.util.spec_from_file_location(nom, REPO / "scripts" / f"{nom}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def regle() -> tuple[dict, str]:
    return verifier(REGLE)


# ---------------------------------------------------------------------- ERA5 (Open-Meteo)
def poids_requete(n_vars: int, n_jours: int) -> float:
    """Poids d'une requete Open-Meteo (appels fractionnaires : > 10 variables ou > 14 jours)."""
    return max(1.0, n_vars / 10.0) * max(1.0, n_jours / 14.0)


def url_open_meteo(lat: float, lon: float, debut: str, fin: str, variables, cellule: str = "nearest") -> str:
    q = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "start_date": debut, "end_date": fin,
         "hourly": ",".join(variables), "models": "era5", "wind_speed_unit": "ms", "timezone": "GMT",
         "cell_selection": cellule}
    return URL_OM + "?" + "&".join(f"{k}={v}" for k, v in q.items())


def lire_open_meteo(js: dict) -> pd.DataFrame:
    """Reponse JSON -> DataFrame (temps UTC naif + variables + lat/lon de la maille retournee)."""
    h = js["hourly"]
    d = pd.DataFrame({k: v for k, v in h.items() if k != "time"})
    d.insert(0, "temps", pd.to_datetime(h["time"]))
    d["lat_maille"], d["lon_maille"] = float(js["latitude"]), float(js["longitude"])
    return d


def telecharger_point(lat, lon, annee, variables, fichier: Path, cellule="nearest", lire=None) -> bool:
    """Une annee d'un point -> fichier CSV (reprise : un fichier present n'est pas redemande). Deux
    echecs -> False (l'appelant arrete l'etape). `lire(url) -> dict` remplace le reseau (tests)."""
    if fichier.exists():
        return True
    url = url_open_meteo(lat, lon, f"{annee}-01-01", f"{annee}-12-31", variables, cellule)
    for essai in range(2):
        try:
            if lire is not None:
                js = lire(url)
            else:
                from pcfno3d.temps_reel import _contexte_tls      # magasin du systeme (CA Python perime)
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}),
                                            timeout=180, context=_contexte_tls()) as r:
                    js = json.loads(r.read().decode("utf-8"))
            d = lire_open_meteo(js)
            fichier.parent.mkdir(parents=True, exist_ok=True)
            d.to_csv(fichier.with_suffix(".part"), index=False)
            fichier.with_suffix(".part").replace(fichier)
            return True
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as exc:
            print(f"echec {essai + 1}/2 {fichier.name} : {type(exc).__name__} {exc}", flush=True)
            if lire is None:
                time.sleep(90 if essai == 0 else 0)
    return False


def etape_era5(stations: pd.DataFrame, sortie: Path, annees, points_mer, lire=None, pause=True) -> dict:
    """ERA5 Open-Meteo de chaque station (VARS_OM) et de chaque point marin (temperature_2m)."""
    taches = [(f"{r.code}_{a}.csv", r.lat, r.lon, VARS_OM, "nearest", a) for r in stations.itertuples() for a in annees]
    taches += [(f"mer{k}_{a}.csv", la, lo, ("temperature_2m",), "sea", a)
               for k, (la, lo) in enumerate(points_mer) for a in annees]
    echecs = []
    for nom, la, lo, var, cel, a in taches:
        f = sortie / nom
        if f.exists():
            continue
        if not telecharger_point(la, lo, a, var, f, cel, lire=lire):
            echecs.append(nom)
            if len(echecs) >= 2:
                print("deux requetes en echec : etape era5 arretee", flush=True)
                break
            continue
        if pause and lire is None:
            jours = 366 if a % 4 == 0 else 365
            time.sleep(poids_requete(len(var), jours) * 3600.0 / APPELS_PAR_HEURE)
    return {"taches": len(taches), "echecs": echecs}


# ---------------------------------------------------------------------- paires et variables
def sites(stations: pd.DataFrame, r_km: float = R_SITE_KM) -> dict[str, str]:
    """{code: site} : composantes connexes des stations a moins de r_km l'une de l'autre ; le site
    porte le plus petit code de sa composante."""
    dist = _script("stations_supplementaires").distance_km
    codes = list(stations["code"])
    parent = {c: c for c in codes}

    def racine(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    pts = list(zip(codes, stations["lat"], stations["lon"]))
    for i, (a, la, lo) in enumerate(pts):
        for b, lb, lob in pts[i + 1:]:
            if dist(la, lo, lb, lob) < r_km:
                ra, rb = racine(a), racine(b)
                parent[max(ra, rb)] = min(ra, rb)
    return {c: racine(c) for c in codes}


def variables_era5(e: pd.DataFrame) -> pd.DataFrame:
    """Series Open-Meteo -> u, v a 10 et 100 m, vitesse, direction, rapport V100/V10."""
    out = pd.DataFrame({"temps": e["temps"]})
    for h in (10, 100):
        vit, d = e[f"wind_speed_{h}m"].astype(float), np.radians(e[f"wind_direction_{h}m"].astype(float))
        out[f"u{h}_era5"], out[f"v{h}_era5"] = -vit * np.sin(d), -vit * np.cos(d)
    out["vitesse10_era5"] = e["wind_speed_10m"].astype(float)
    d10 = np.radians(e["wind_direction_10m"].astype(float))
    out["sin_dir10_era5"], out["cos_dir10_era5"] = np.sin(d10), np.cos(d10)
    out["rapport_v100_v10_era5"] = e["wind_speed_100m"].astype(float) / np.maximum(out["vitesse10_era5"], 0.5)
    out["blh_era5_m"] = e["boundary_layer_height"].astype(float)
    out["rayonnement_era5_w_m2"] = e["shortwave_radiation"].astype(float)
    out["nebulosite_era5_pct"] = e["cloud_cover"].astype(float)
    out["t2m_era5"] = e["temperature_2m"].astype(float)
    out["lat_maille"], out["lon_maille"] = e["lat_maille"], e["lon_maille"]
    return out


def construire_paires(obs: pd.DataFrame, era5: pd.DataFrame, t_mer: pd.Series, site: dict, z_era5: float) -> pd.DataFrame:
    """Une station : obs (creneau, wind_u_m_s, wind_v_m_s, wind_speed_m_s) + era5 (variables_era5) +
    temperature du point marin le plus proche (Series indexee par le temps) + descripteurs `site`
    -> paires avec les variables de la regle et les cibles. Paire incomplete ecartee."""
    e = variables_era5(era5).set_index("temps")
    e["dt_terre_mer_k"] = e["t2m_era5"] - t_mer.reindex(e.index)
    p = obs.rename(columns={"creneau": "temps", "wind_u_m_s": "u_obs", "wind_v_m_s": "v_obs",
                            "wind_speed_m_s": "vitesse_obs"}).merge(e.reset_index(), on="temps", how="inner")
    p["z_maille_500m_m"] = site["z_maille_500m_m"]
    p["pente_maille_deg"] = site["pente_maille_deg"]
    p["log10_z0_maille"] = math.log10(max(float(site["z0_maille_m"]), 1e-5))
    p["tpi_5km_m"], p["tpi_20km_m"] = site["tpi_5km_m"], site["tpi_20km_m"]
    p["log1p_distance_cote_km"] = math.log1p(max(float(site["distance_cote_km"]), 0.0))
    p["dz_maille_moins_era5_m"] = float(site["z_maille_500m_m"]) - z_era5
    h = p["temps"].dt.hour + p["temps"].dt.minute / 60.0
    j = p["temps"].dt.dayofyear
    p["sin_heure"], p["cos_heure"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    p["sin_jour"], p["cos_jour"] = np.sin(2 * np.pi * (j - 1) / 365.25), np.cos(2 * np.pi * (j - 1) / 365.25)
    p["du"], p["dv"] = p["u_obs"] - p["u10_era5"], p["v_obs"] - p["v10_era5"]
    p["dvitesse"] = np.hypot(p["u_obs"], p["v_obs"]) - p["vitesse10_era5"]
    p["annee"] = p["temps"].dt.year
    return p


def z_relief_era5(fichier_sl: Path):
    """Fonction (lat, lon) -> altitude (m) du relief ERA5 de la maille la plus proche (z / g)."""
    import xarray as xr

    with xr.open_dataset(fichier_sl) as d:
        tv = "valid_time" if "valid_time" in d.dims else "time"
        z = d["z"].isel({tv: 0}).load() / 9.80665

    def f(lat, lon):
        return float(z.sel(latitude=lat, longitude=lon, method="nearest"))
    return f


def etape_preparer(inv: pd.DataFrame, brut: Path, dossier_era5: Path, z_f, sortie: Path, r, annees) -> dict:
    """Paires de toutes les stations admises -> sortie/paires_<annee>.csv.gz + stations.csv."""
    from pcfno3d.stations_reelles import controle_qualite, preparer_observations

    S = _script("stations_supplementaires")
    isd_mod = S._isd_synop()
    adm = inv[inv["dans_domaine"].astype(bool) & inv["isd"].notna() & inv["z0_maille_m"].notna()].copy()
    st_site = sites(adm)
    adm["site"] = adm["code"].map(st_site)
    mer = []
    for k, _ in enumerate(r["donnees"]["points_mer"]):
        fs = [dossier_era5 / f"mer{k}_{a}.csv" for a in annees]
        if all(f.exists() for f in fs):
            m = pd.concat([pd.read_csv(f, parse_dates=["temps"]) for f in fs])
            mer.append((float(m["lat_maille"].iloc[0]), float(m["lon_maille"].iloc[0]),
                        m.set_index("temps")["temperature_2m"]))
    if not mer:
        raise SystemExit("aucun point marin ERA5 : lancer l'etape era5")
    tout = []
    for s in adm.itertuples():
        fs = [dossier_era5 / f"{s.code}_{a}.csv" for a in annees]
        if not all(f.exists() for f in fs):
            print(f"{s.code} : ERA5 incomplete, station ecartee", flush=True)
            continue
        e = pd.concat([pd.read_csv(f, parse_dates=["temps"]) for f in fs], ignore_index=True)
        brutes = S.convertir_station(isd_mod, brut, s.isd, s.code, annees)
        if brutes.empty:
            continue
        gardees, _ = controle_qualite(preparer_observations(brutes))
        obs = S.au_creneau(gardees)
        dmer = [S.distance_km(s.lat, s.lon, a, b) for a, b, _ in mer]
        t_mer = mer[int(np.argmin(dmer))][2]
        p = construire_paires(obs, e, t_mer, s._asdict(), z_f(float(e["lat_maille"].iloc[0]),
                                                               float(e["lon_maille"].iloc[0])))
        p.insert(0, "code", s.code)
        p.insert(1, "site", s.site)
        p["aeroport"] = bool(s.aeroport_metar) if isinstance(s.aeroport_metar, str) else False
        p["abri_probable"] = bool(s.abri_probable) if not pd.isna(s.abri_probable) else False
        tout.append(p)
        print(f"{s.code} {s.nom} : {len(p)} paires", flush=True)
    paires = pd.concat(tout, ignore_index=True)
    sortie.mkdir(parents=True, exist_ok=True)
    garde = ["code", "site", "temps", "annee", "aeroport", "abri_probable", "u_obs", "v_obs", "vitesse_obs",
             "du", "dv", "dvitesse", *r["variables"]]
    for a, g in paires.groupby("annee"):
        g[garde].to_csv(sortie / f"paires_{a}.csv.gz", index=False)
    adm.to_csv(sortie / "stations.csv", index=False)
    return {"paires": len(paires), "stations": paires["code"].nunique(), "sites": paires["site"].nunique()}


def lire_paires(dossier: Path, annees) -> pd.DataFrame:
    """paires_<annee>.csv.gz des SEULES annees demandees (le test n'est jamais ouvert par choisir)."""
    fs = [Path(dossier) / f"paires_{a}.csv.gz" for a in annees]
    manque = [f.name for f in fs if not f.exists()]
    if manque:
        raise SystemExit(f"paires absentes : {manque}")
    return pd.concat([pd.read_csv(f, parse_dates=["temps"]) for f in fs], ignore_index=True)


# ---------------------------------------------------------------------- modeles
def grille(r: dict) -> list[dict]:
    m = r["modeles"]
    g = [{"famille": "ridge", "alpha": a} for a in m["ridge"]["alpha"]]
    gb = m["gradient_boosting"]
    g += [{"famille": "gradient_boosting", "learning_rate": lr, "max_depth": md, "max_iter": mi,
           "min_samples_leaf": gb["min_samples_leaf"], "l2_regularization": gb["l2_regularization"]}
          for lr in gb["learning_rate"] for md in gb["max_depth"] for mi in gb["max_iter"]]
    return g


def nom_config(c: dict) -> str:
    if c["famille"] == "ridge":
        return f"ridge_a{c['alpha']:g}"
    return f"gb_lr{c['learning_rate']:g}_d{c['max_depth']}_n{c['max_iter']}"


def cle_simplicite(c: dict) -> tuple:
    """Ordre de simplicite de la regle : ridge avant gradient boosting, alpha le plus grand, puis
    max_iter et max_depth les plus petits."""
    if c["famille"] == "ridge":
        return (0, -c["alpha"], 0, 0)
    return (1, 0, c["max_iter"], c["max_depth"])


def modele(c: dict, graine: int):
    """Ridge : variables normalisees (moyenne et ecart-type des paires d'ajustement, StandardScaler),
    ordonnee libre ; gradient boosting : HistGradientBoostingRegressor de la regle."""
    if c["famille"] == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=c["alpha"]))
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="squared_error", learning_rate=c["learning_rate"],
                                         max_depth=c["max_depth"], max_iter=c["max_iter"],
                                         min_samples_leaf=c["min_samples_leaf"],
                                         l2_regularization=c["l2_regularization"], early_stopping=False,
                                         random_state=graine)


def ajuster(c: dict, train: pd.DataFrame, variables, graine: int, cibles=("du", "dv")) -> dict:
    X = train[variables].to_numpy(float)
    return {k: modele(c, graine).fit(X, train[k].to_numpy(float)) for k in cibles}


def corriger(m: dict, p: pd.DataFrame, variables) -> pd.DataFrame:
    X = p[variables].to_numpy(float)
    out = p[["code", "site", "temps", "aeroport", "abri_probable", "u_obs", "v_obs", "u10_era5", "v10_era5",
             "vitesse10_era5"]].copy()
    out["u_mos"] = p["u10_era5"].to_numpy() + m["du"].predict(X)
    out["v_mos"] = p["v10_era5"].to_numpy() + m["dv"].predict(X)
    if "dvitesse" in m:
        out["vitesse_mos"] = p["vitesse10_era5"].to_numpy() + m["dvitesse"].predict(X)
    return out


# ---------------------------------------------------------------------- jugement
def juges(stations: pd.DataFrame, annee: int) -> dict[str, str]:
    """{site: station jugee} : station la plus complete du site sur l'annee (taux_qc_<annee>, puis
    code le plus petit), si au moins MIN_TAUX_JUGE."""
    col = f"taux_qc_{annee}"
    out = {}
    for s, g in stations.groupby("site"):
        g = g.assign(_t=g[col].fillna(0.0)).sort_values(["_t", "code"], ascending=[False, True])
        if g["_t"].iloc[0] >= MIN_TAUX_JUGE:
            out[s] = g["code"].iloc[0]
    return out


def par_site(pred: pd.DataFrame) -> pd.DataFrame:
    """RMSE vectoriel corrige et ERA5 par site (et biais de vitesse, MAE de direction)."""
    lignes = []
    for s, g in pred.groupby("site"):
        e_m = (g["u_mos"] - g["u_obs"]) ** 2 + (g["v_mos"] - g["v_obs"]) ** 2
        e_e = (g["u10_era5"] - g["u_obs"]) ** 2 + (g["v10_era5"] - g["v_obs"]) ** 2
        vo = np.hypot(g["u_obs"], g["v_obs"])
        fort = vo >= 1.0

        def mae_dir(u, v):
            d = (np.degrees(np.arctan2(-u, -v)) - np.degrees(np.arctan2(-g["u_obs"], -g["v_obs"])) + 180) % 360 - 180
            return float(np.abs(d[fort]).mean()) if fort.any() else math.nan
        r = {"site": s, "code": g["code"].iloc[0], "n": len(g), "aeroport": bool(g["aeroport"].iloc[0]),
             "abri_probable": bool(g["abri_probable"].iloc[0]),
             "rmse_vect_mos": float(np.sqrt(e_m.mean())), "rmse_vect_era5": float(np.sqrt(e_e.mean())),
             "biais_vitesse_mos": float((np.hypot(g["u_mos"], g["v_mos"]) - vo).mean()),
             "biais_vitesse_era5": float((g["vitesse10_era5"] - vo).mean()),
             "mae_dir_mos_deg": mae_dir(g["u_mos"], g["v_mos"]), "mae_dir_era5_deg": mae_dir(g["u10_era5"], g["v10_era5"])}
        if "vitesse_mos" in g:
            r["rmse_vitesse_mos_dvitesse"] = float(np.sqrt(((g["vitesse_mos"] - vo) ** 2).mean()))
            r["rmse_vitesse_era5"] = float(np.sqrt(((g["vitesse10_era5"] - vo) ** 2).mean()))
        lignes.append(r)
    t = pd.DataFrame(lignes)
    t["ecart_relatif"] = t["rmse_vect_mos"] / t["rmse_vect_era5"] - 1.0
    return t


def bootstrap_ecart(pred: pd.DataFrame, tirages: int, graine: int) -> tuple[float, float, float]:
    """Ecart (moyenne sur les sites du RMSE vectoriel corrige - celle de l'ERA5) et son IC 95 % par
    bootstrap des JOURS UTC (toutes les paires d'un jour ensemble)."""
    p = pred.assign(jour=pred["temps"].dt.floor("D"),
                    em=(pred["u_mos"] - pred["u_obs"]) ** 2 + (pred["v_mos"] - pred["v_obs"]) ** 2,
                    ee=(pred["u10_era5"] - pred["u_obs"]) ** 2 + (pred["v10_era5"] - pred["v_obs"]) ** 2)
    a = p.groupby(["jour", "site"]).agg(em=("em", "sum"), ee=("ee", "sum"), n=("em", "size")).reset_index()
    jours, sites_ = np.sort(a["jour"].unique()), np.sort(a["site"].unique())
    ij = np.searchsorted(jours, a["jour"].to_numpy())
    js = np.searchsorted(sites_, a["site"].to_numpy())
    M = {k: np.zeros((len(jours), len(sites_))) for k in ("em", "ee", "n")}
    for k in M:
        np.add.at(M[k], (ij, js), a[k].to_numpy(float))

    def ecart(w):
        n = w @ M["n"]
        ok = n > 0
        rm = np.sqrt(np.where(ok, (w @ M["em"]) / np.where(ok, n, 1), np.nan))
        re = np.sqrt(np.where(ok, (w @ M["ee"]) / np.where(ok, n, 1), np.nan))
        return np.nanmean(rm, axis=-1) - np.nanmean(re, axis=-1)
    e0 = float(ecart(np.ones(len(jours))))
    rng = np.random.default_rng(graine)
    W = rng.multinomial(len(jours), np.full(len(jours), 1.0 / len(jours)), size=tirages).astype(float)
    d = ecart(W)
    return e0, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def verdict(t: pd.DataFrame, ic: tuple[float, float, float], r: dict) -> dict:
    c = r["critere_bat_era5_sur_les_stations_jamais_vues"]
    m_mos, m_era5 = float(t["rmse_vect_mos"].mean()), float(t["rmse_vect_era5"].mean())
    part = float((t["rmse_vect_mos"] < t["rmse_vect_era5"]).mean())
    pire = float(t["ecart_relatif"].max())
    c1 = (m_mos < m_era5) and ic[2] < 0
    c2 = part >= c["part_minimale_sites_ameliores"]
    c3 = pire <= c["degradation_max_par_site"]
    return {"sites": len(t), "moyenne_rmse_vect_mos": m_mos, "moyenne_rmse_vect_era5": m_era5,
            "ecart": ic[0], "ic95_ecart": [ic[1], ic[2]], "part_sites_ameliores": part,
            "pire_degradation_relative": pire, "pire_site": str(t.loc[t["ecart_relatif"].idxmax(), "code"]),
            "condition_1_moyenne_et_ic": bool(c1), "condition_2_sites_ameliores": bool(c2),
            "condition_3_aucune_degradation": bool(c3), "bat_era5": bool(c1 and c2 and c3)}


def anomalies(t: pd.DataFrame, pred: pd.DataFrame) -> list[str]:
    out = []
    if pred[["u_mos", "v_mos"]].isna().any().any():
        out.append("NaN dans les predictions")
    rap = t["rmse_vect_mos"] / t["rmse_vect_era5"]
    if (rap < 0.3).any():
        out.append(f"trop beau : {list(t.loc[rap < 0.3, 'code'])}")
    if (rap > 3.0).any():
        out.append(f"divergence : {list(t.loc[rap > 3.0, 'code'])}")
    return out


def sous_ensembles(t: pd.DataFrame) -> dict:
    out = {}
    for nom, m in (("aeroports", t["aeroport"]), ("non_aeroports", ~t["aeroport"]),
                   ("sans_abri_probable", ~t["abri_probable"])):
        g = t[m]
        if len(g):
            out[nom] = {"sites": len(g), "moyenne_rmse_vect_mos": float(g["rmse_vect_mos"].mean()),
                        "moyenne_rmse_vect_era5": float(g["rmse_vect_era5"].mean()),
                        "sites_ameliores": int((g["rmse_vect_mos"] < g["rmse_vect_era5"]).sum())}
    return out


def etape_choisir(paires_dir: Path, sortie: Path, r: dict, h: str, log=print) -> dict:
    if (sortie / "choix.json").exists():
        raise SystemExit(f"{sortie / 'choix.json'} existe deja : rien n'est recalcule ni ecrase")
    sortie.mkdir(parents=True, exist_ok=True)
    tr, va = r["decoupage"]["entrainement"], r["decoupage"]["validation"]
    p = lire_paires(paires_dir, tr + va)                     # jamais l'annee de test
    st = pd.read_csv(Path(paires_dir) / "stations.csv", dtype={"code": str})
    variables, graine = r["variables"], r["modeles"]["graine"]
    j_val = {s: c for s, c in juges(st, va[0]).items() if c in set(p["code"])}
    sites_tous = sorted(p["site"].unique())
    train, val = p[p["annee"].isin(tr)], p[p["annee"].isin(va)]
    log(f"paires : entrainement {len(train)}, validation {len(val)} ; sites juges en validation {len(j_val)}")
    scores, preds, modeles_choisis = [], {}, {}
    for c in grille(r):
        t0 = time.time()
        morceaux = []
        for s, code in j_val.items():
            m = ajuster(c, train[train["site"] != s], variables, graine)
            morceaux.append(corriger(m, val[val["code"] == code], variables))
        pr = pd.concat(morceaux, ignore_index=True)
        t = par_site(pr)
        preds[nom_config(c)] = pr
        scores.append({**c, "config": nom_config(c), "score_validation": float(t["rmse_vect_mos"].mean()),
                       "era5_seule": float(t["rmse_vect_era5"].mean()), "duree_s": round(time.time() - t0, 1)})
        log(f"{nom_config(c)} : MOS {scores[-1]['score_validation']:.4f} | ERA5 seule {scores[-1]['era5_seule']:.4f} "
            f"m/s ({scores[-1]['duree_s']} s)")
    sc = pd.DataFrame(scores)
    sc.to_csv(sortie / "grille_validation.csv", index=False)
    vals = [preds[k][["u_mos", "v_mos"]].to_numpy() for k in preds]
    if len(vals) > 1 and all(np.allclose(vals[0], v) for v in vals[1:]):
        raise SystemExit("anomalie : predictions identiques pour toutes les configurations")
    best = sc["score_validation"].min()
    cands = [c for c in grille(r) if sc.loc[sc["config"] == nom_config(c), "score_validation"].iloc[0] <= best + 1e-4]
    choisie = sorted(cands, key=cle_simplicite)[0]
    pr = preds[nom_config(choisie)]
    t = par_site(pr)
    anom = anomalies(t, pr)
    b = r["jugement_test"]["bootstrap"]
    ic = bootstrap_ecart(pr, b["tirages"], b["graine"])
    v = verdict(t, ic, r)
    t.to_csv(sortie / "validation_sites.csv", index=False)
    pr.to_csv(sortie / "predictions_validation.csv.gz", index=False)
    # modeles de la configuration choisie pour CHAQUE site (aucun reajustement au test), dvitesse compris
    for s in sites_tous:
        modeles_choisis[s] = ajuster(choisie, train[train["site"] != s], variables, graine, ("du", "dv", "dvitesse"))
    with open(sortie / "modeles_choisis.pkl", "wb") as f:
        pickle.dump({"config": choisie, "variables": variables, "modeles": modeles_choisis}, f)
    choix = {"regle": REGLE.name, "sha256_regle": h, "config_choisie": nom_config(choisie), "parametres": choisie,
             "validation": {**v, "sous_ensembles": sous_ensembles(t)}, "anomalies": anom,
             "test_autorise": bool(v["bat_era5"] and not anom),
             "sites_juges_validation": j_val, "annees_ajustement": tr, "annee_validation": va}
    (sortie / "choix.json").write_text(json.dumps(choix, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"choisie : {nom_config(choisie)} | validation : MOS {v['moyenne_rmse_vect_mos']:.4f} contre ERA5 seule "
        f"{v['moyenne_rmse_vect_era5']:.4f} m/s, ecart {v['ecart']:+.4f} [{ic[1]:+.4f}, {ic[2]:+.4f}] | "
        f"bat ERA5 : {v['bat_era5']} | anomalies : {anom or 'aucune'}")
    return choix


def etape_tester(paires_dir: Path, sortie: Path, r: dict, h: str, log=print) -> dict:
    choix = json.loads((sortie / "choix.json").read_text(encoding="utf-8"))
    if choix["sha256_regle"] != h:
        raise SystemExit("regle changee depuis l'etape choisir : test refuse")
    if not choix["test_autorise"]:
        raise SystemExit("la validation ne bat pas ERA5 selon le critere (ou anomalie) : le test n'est pas lu")
    if (sortie / "test.json").exists():
        raise SystemExit(f"{sortie / 'test.json'} existe deja : le test a ete lu, il ne se relit pas")
    with open(sortie / "modeles_choisis.pkl", "rb") as f:
        M = pickle.load(f)
    te = r["decoupage"]["test"][0]
    st = pd.read_csv(Path(paires_dir) / "stations.csv", dtype={"code": str})
    p = lire_paires(paires_dir, [te])
    j_te = {s: c for s, c in juges(st, te).items() if c in set(p["code"]) and s in M["modeles"]}
    pr = pd.concat([corriger(M["modeles"][s], p[p["code"] == c], M["variables"]) for s, c in j_te.items()],
                   ignore_index=True)
    t = par_site(pr)
    anom = anomalies(t, pr)
    b = r["jugement_test"]["bootstrap"]
    ic = bootstrap_ecart(pr, b["tirages"], b["graine"])
    v = verdict(t, ic, r)
    t.to_csv(sortie / "test_sites.csv", index=False)
    pr.to_csv(sortie / "predictions_test.csv.gz", index=False)
    res = {"regle": REGLE.name, "sha256_regle": h, "config": choix["config_choisie"], "annee_test": te,
           "lu_le": pd.Timestamp.now().isoformat(timespec="seconds"), "test": {**v, "sous_ensembles": sous_ensembles(t)},
           "anomalies": anom, "sites_juges_test": j_te}
    (sortie / "test.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"TEST {te} : MOS {v['moyenne_rmse_vect_mos']:.4f} contre ERA5 seule {v['moyenne_rmse_vect_era5']:.4f} m/s, "
        f"ecart {v['ecart']:+.4f} [{ic[1]:+.4f}, {ic[2]:+.4f}] | bat ERA5 : {v['bat_era5']} | anomalies : {anom or 'aucune'}")
    return res


def etape_rapport(sortie: Path, fichier_md: Path) -> None:
    """Resume en markdown (chaque nombre a cote de l'ERA5 seule) depuis choix.json et test.json."""
    choix = json.loads((sortie / "choix.json").read_text(encoding="utf-8"))
    g = pd.read_csv(sortie / "grille_validation.csv")
    v = choix["validation"]

    def bloc(titre, v, t):
        L = [f"### {titre}", "",
             f"- moyenne sur {v['sites']} sites du RMSE vectoriel : **MOS {v['moyenne_rmse_vect_mos']:.3f}** contre "
             f"ERA5 seule {v['moyenne_rmse_vect_era5']:.3f} m/s ; écart {v['ecart']:+.3f} m/s, IC 95 % "
             f"[{v['ic95_ecart'][0]:+.3f}, {v['ic95_ecart'][1]:+.3f}] ;",
             f"- sites améliorés : {v['part_sites_ameliores']:.0%} ; pire dégradation relative "
             f"{v['pire_degradation_relative']:+.1%} ({v['pire_site']}) ;",
             f"- conditions (1) moyenne et IC : {v['condition_1_moyenne_et_ic']}, (2) 2 sites sur 3 : "
             f"{v['condition_2_sites_ameliores']}, (3) aucun site > 5 % : {v['condition_3_aucune_degradation']} "
             f"→ **{'bat ERA5' if v['bat_era5'] else 'ne bat pas ERA5'}**.", ""]
        for nom, s in v.get("sous_ensembles", {}).items():
            L.append(f"- {nom} ({s['sites']} sites) : MOS {s['moyenne_rmse_vect_mos']:.3f} contre ERA5 seule "
                     f"{s['moyenne_rmse_vect_era5']:.3f} m/s, {s['sites_ameliores']} sites améliorés")
        L += ["", "| Station | n | RMSE vect. MOS | RMSE vect. ERA5 seule | écart relatif | biais vitesse MOS | "
              "biais vitesse ERA5 | MAE dir. MOS | MAE dir. ERA5 |", "|---|---|---|---|---|---|---|---|---|"]
        for r in t.itertuples():
            L.append(f"| {r.code} | {r.n} | {r.rmse_vect_mos:.3f} | {r.rmse_vect_era5:.3f} | {r.ecart_relatif:+.1%} | "
                     f"{r.biais_vitesse_mos:+.2f} | {r.biais_vitesse_era5:+.2f} | {r.mae_dir_mos_deg:.0f} | "
                     f"{r.mae_dir_era5_deg:.0f} |")
        return L + [""]
    L = ["# MOS de l'ERA5 10 m aux stations tunisiennes : résultats", "",
         f"Règle `configs/national/{choix['regle']}` (sha256 `{choix['sha256_regle']}`), `scripts/mos_era5.py`. "
         "Validation croisée par site : chaque site retiré de l'ajustement (2015–2020), puis jugé. ERA5 servie par "
         "Open-Meteo (`models=era5`, maille la plus proche).", "",
         "## Grille (validation 2021)", "", "| Configuration | MOS | ERA5 seule | durée (s) |", "|---|---|---|---|"]
    L += [f"| {r.config} | {r.score_validation:.4f} | {r.era5_seule:.4f} | {r.duree_s:.0f} |" for r in g.itertuples()]
    L += ["", f"Configuration choisie : **{choix['config_choisie']}** ; anomalies : {choix['anomalies'] or 'aucune'} ; "
          f"test autorisé : **{choix['test_autorise']}**.", ""]
    L += bloc("Validation 2021", v, pd.read_csv(sortie / "validation_sites.csv"))
    if (sortie / "test.json").exists():
        te = json.loads((sortie / "test.json").read_text(encoding="utf-8"))
        L += bloc(f"Test {te['annee_test']} (lu une fois, le {te['lu_le']})", te["test"], pd.read_csv(sortie / "test_sites.csv"))
    else:
        L += ["### Test 2022", "", "Non lu (la validation ne bat pas ERA5 selon la règle, ou anomalie).", ""]
    fichier_md.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="etape", required=True)
    a = sub.add_parser("era5")
    a.add_argument("--stations", required=True, help="CSV code, lat, lon (inventaire)")
    a.add_argument("--sortie", required=True)
    b = sub.add_parser("preparer")
    b.add_argument("--inventaire", default=str(REPO / "docs" / "resultats" / "national" / "stations_supplementaires.csv"))
    b.add_argument("--brut", required=True)
    b.add_argument("--era5", required=True)
    b.add_argument("--relief-era5", required=True, help="era5_sl_*.nc local (geopotentiel z)")
    b.add_argument("--sortie", required=True)
    for nom in ("choisir", "tester"):
        c = sub.add_parser(nom)
        c.add_argument("--paires", required=True)
        c.add_argument("--sortie", required=True)
    e = sub.add_parser("rapport")
    e.add_argument("--sortie", required=True)
    e.add_argument("--md", default=str(REPO / "docs" / "resultats" / "national" / "mos_era5_resultats.md"))
    args = ap.parse_args(argv)
    if args.etape == "rapport":
        etape_rapport(Path(args.sortie), Path(args.md))
        return 0
    r, h = regle()
    d = r["decoupage"]
    annees = d["entrainement"] + d["validation"] + d["test"]
    print(f"regle {REGLE.name} sha256 {h}", flush=True)
    if args.etape == "era5":
        st = pd.read_csv(args.stations, dtype={"code": str})
        print(etape_era5(st, Path(args.sortie), annees, r["donnees"]["points_mer"]))
    elif args.etape == "preparer":
        inv = pd.read_csv(args.inventaire, dtype={"code": str, "isd": str, "wmo": str})
        print(etape_preparer(inv, Path(args.brut), Path(args.era5), z_relief_era5(Path(args.relief_era5)),
                             Path(args.sortie), r, annees))
    elif args.etape == "choisir":
        etape_choisir(Path(args.paires), Path(args.sortie), r, h)
    else:
        etape_tester(Path(args.paires), Path(args.sortie), r, h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
