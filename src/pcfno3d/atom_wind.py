"""Donnees du front-end ATOM Wind (webapp/atom-wind/) : le champ du modele, compact.

Le front-end raisonne en hauteur AU-DESSUS DU SOL (une grue a 40 m, un drone a 100 m...) sur une
grille de 500 m (68 x 56 km pour le Grand Tunis). Un volume complet (12 x 560 x 680, 100 m) pese
~55 Mo : on envoie a la place, pour chaque instant d'une journee, u et v a quelques hauteurs AGL,
moyennes par blocs de 500 m, en centimetres par seconde (int16) — ~0,6 Mo par instant avant gzip.

    grille(...)  relief, eau, z0, stations, georeferencement (une fois)
    jours(...)   les journees servies (heure locale de Tunis, UTC+1, sans heure d'ete)
    jour(...)    les champs d'une journee
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HAUTEURS_AGL = (10, 25, 50, 80, 100, 150, 200, 300, 500, 800, 1000)
PAS_M = 500.0
UTC_LOCAL_H = 1                      # Tunisie : UTC+1 toute l'annee
NODATA_I16 = -32768


def champ_agl(u: np.ndarray, v: np.ndarray, geom, h: float) -> tuple[np.ndarray, np.ndarray]:
    """u, v (nz, ny, nx ; NaN sous le relief) a h m au-dessus du sol, colonne par colonne :
    interpolation entre deux noeuds valides, ou profil log depuis le noeud valide le plus bas
    (z0 de la maille) si h est sous ce noeud — comme pcfno3d.stations_reelles.poids_10m.
    Au-dessus du dernier noeud valide : valeur de ce noeud."""
    valid = geom.valid
    hk = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    uo = np.full(u.shape[1:], np.nan)
    vo = np.full(u.shape[1:], np.nan)
    for k in range(len(geom.z_levels) - 1):
        m = valid[k] & valid[k + 1] & (hk[k] <= h) & (h <= hk[k + 1]) & np.isnan(uo)
        a = np.where(m, (h - hk[k]) / np.maximum(hk[k + 1] - hk[k], 1e-6), 0.0)
        uo = np.where(m, (1 - a) * u[k] + a * u[k + 1], uo)
        vo = np.where(m, (1 - a) * v[k] + a * v[k + 1], vo)
    k0 = np.argmax(valid, axis=0)
    a_un = valid.any(axis=0)
    h0 = np.take_along_axis(hk, k0[None], 0)[0]
    bas = a_un & (h < h0) & np.isnan(uo)
    z0 = np.maximum(geom.z0, 1e-4)
    f = np.where(bas, np.log(h / z0) / np.log(np.maximum(h0, 1.01 * z0) / z0), 0.0)
    u0 = np.take_along_axis(u, k0[None], 0)[0]
    v0 = np.take_along_axis(v, k0[None], 0)[0]
    uo = np.where(bas, f * u0, uo)
    vo = np.where(bas, f * v0, vo)
    # au-dessus du dernier noeud valide de la colonne
    k1 = valid.shape[0] - 1 - np.argmax(valid[::-1], axis=0)
    haut = a_un & np.isnan(uo) & (h > np.take_along_axis(hk, k1[None], 0)[0])
    uo = np.where(haut, np.take_along_axis(u, k1[None], 0)[0], uo)
    vo = np.where(haut, np.take_along_axis(v, k1[None], 0)[0], vo)
    return uo, vo


def blocs(a: np.ndarray, b: int, fn=np.nanmean) -> np.ndarray:
    """Moyenne (ou autre) par blocs b x b ; le bord incomplet est garde."""
    ny, nx = a.shape
    py, px = (-ny) % b, (-nx) % b
    p = np.pad(a.astype(float), ((0, py), (0, px)), constant_values=np.nan)
    p = p.reshape((ny + py) // b, b, (nx + px) // b, b)
    with np.errstate(invalid="ignore"), _sans_alerte():
        return fn(p, axis=(1, 3))


class _sans_alerte:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


def facteur(geom) -> int:
    return max(int(round(PAS_M / geom.dx)), 1)


def grille(geom, georef, cfg) -> dict:
    b = facteur(geom)
    terrain = blocs(geom.zt, b)
    eau = blocs((geom.z0 < 0.001).astype(float), b) > 0.5
    z0 = np.exp(blocs(np.log(np.maximum(geom.z0, 1e-4)), b))
    stations = []
    f = cfg.paths.data_dir / "stations.json"
    if f.exists():
        for s in json.loads(f.read_text(encoding="utf-8")):
            x, y = s["ix"] * geom.dx, s["iy"] * geom.dy
            lat, lon = georef.to_latlon(x, y) if georef is not None else (None, None)
            stations.append({"nom": s.get("nom", "station"), "x_km": x / 1000, "y_km": y / 1000,
                             "lat": None if lat is None else round(float(lat), 5),
                             "lon": None if lon is None else round(float(lon), 5)})
    return {
        "format": "atom-wind/1", "nx": int(terrain.shape[1]), "ny": int(terrain.shape[0]),
        "pas_km": geom.dx * b / 1000, "largeur_km": geom.nx * geom.dx / 1000,
        "hauteur_km": geom.ny * geom.dy / 1000,
        "lat0": None if georef is None else georef.lat0,
        "lon0": None if georef is None else georef.lon0,
        "rotation_deg": None if georef is None else georef.rotation_deg,
        "hauteurs_agl_m": list(HAUTEURS_AGL),
        "terrain_m": np.round(np.nan_to_num(terrain), 1).tolist(),
        "eau": eau.astype(int).tolist(),
        "z0_m": np.round(z0, 5).tolist(),
        "stations": stations,
    }


def heure_locale(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t + pd.Timedelta(hours=UTC_LOCAL_H)


def jours(index: pd.DataFrame, split: str | None = "test") -> list[dict]:
    idx = index if split is None else index[index["split"] == split]
    out: dict[str, list] = {}
    for stem, ts in zip(idx["stem"], idx["timestamp"]):
        loc = heure_locale(ts)
        out.setdefault(loc.strftime("%Y-%m-%d"), []).append(
            {"stem": str(stem), "timestamp_utc": pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "heure_locale": round(loc.hour + loc.minute / 60, 3)})
    return [{"date": d, "instants": sorted(v, key=lambda r: r["heure_locale"])}
            for d, v in sorted(out.items())]


def encoder(champs: np.ndarray) -> str:
    """float (m/s, NaN) -> int16 en cm/s, petit-boutiste, base64."""
    q = np.where(np.isfinite(champs), np.clip(np.round(champs * 100), -32767, 32767), NODATA_I16)
    return base64.b64encode(q.astype("<i2").tobytes()).decode("ascii")


def instant(volume, geom) -> np.ndarray:
    """(n_hauteurs, 2, ny, nx) : u, v aux hauteurs AGL, moyennes par blocs de 500 m."""
    b = facteur(geom)
    out = []
    for h in HAUTEURS_AGL:
        u, v = champ_agl(volume.u, volume.v, geom, float(h))
        out.append([blocs(u, b), blocs(v, b)])
    return np.asarray(out, dtype=np.float32)


def jour(predire, geom, j: dict, fiabilite=None) -> dict:
    """Les champs d'une journee. predire(stem) -> Volume ; fiabilite(volume) -> (n_h, ny, nx)
    sigma de la vitesse (m/s) ou None."""
    instants, blocs_vent, blocs_sigma = [], [], []
    for it in j["instants"]:
        vol = predire(it["stem"])
        blocs_vent.append(instant(vol, geom))
        if fiabilite is not None:
            s = fiabilite(vol)
            if s is not None:
                blocs_sigma.append(s)
        instants.append({**it, "inference_s": round(float(vol.inference_s), 3),
                         "capteurs_utilises": int(vol.n_sensors)})
    a = np.stack(blocs_vent)
    out = {"format": "atom-wind/1", "date": j["date"], "instants": instants,
           "hauteurs_agl_m": list(HAUTEURS_AGL), "forme": list(a.shape),
           "unite": "cm/s, int16 petit-boutiste, -32768 = sans valeur",
           "ordre": "instant, hauteur, (u, v), y, x", "vent": encoder(a)}
    if blocs_sigma and len(blocs_sigma) == len(instants):
        s = np.stack(blocs_sigma)
        out["sigma_vitesse"] = encoder(s)
        out["forme_sigma"] = list(s.shape)
    return out


def sigma_agl(volume, geom) -> np.ndarray | None:
    """sigma de la vitesse aux hauteurs AGL (meme interpolation que le vent), si la
    fiabilite (lot M2) du volume est prete."""
    rel = getattr(volume, "reliability", None)
    if rel is None:
        return None
    b = facteur(geom)
    s = rel.sigma_speed
    return np.asarray([blocs(champ_agl(s, s, geom, float(h))[0], b) for h in HAUTEURS_AGL],
                      dtype=np.float32)


def plan_aneo(run_dir: Path) -> Path | None:
    """Le plan OS6 (scripts/os6_reseau_aneo.py), s'il a ete calcule."""
    for d in (run_dir / "os6_reseau_aneo", run_dir.parent / "os6_reseau_aneo"):
        f = d / "plan_aneo.geojson"
        if f.exists():
            return f
    return None


# ---------------------------------------------------------------------- temps reel
def jour_direct(wf, source, date_locale: str, maintenant: pd.Timestamp | None = None) -> dict:
    """Les 24 heures locales d'une journee (pcfno3d.temps_reel) : heures passees = forcage GFS +
    vraies mesures des stations ; heures a venir = prevision GFS, sans station. Avec, pour
    l'espace agriculteur, la temperature et l'humidite a 2 m de GFS (ramenees au relief, grille de
    500 m) et le rayonnement solaire moyen du domaine : grandeurs que le modele ne calcule pas,
    signalees comme telles."""
    from . import temps_reel as tr

    geom = wf.geom
    e = source.journee(date_locale)
    f, c, obs = e["forcage"], e["champs"], e["obs"]
    maintenant = maintenant if maintenant is not None else \
        pd.Timestamp.now(tz="UTC").tz_localize(None)
    index_t = {int(s): i for i, s in enumerate(c["temps"])}
    instants, blocs_vent, blocs_t, blocs_hr = [], [], [], []
    b = facteur(geom)
    for h in range(24):
        t = pd.Timestamp(date_locale) + pd.Timedelta(hours=h - UTC_LOCAL_H)
        it = index_t.get(int(t.value // 10 ** 9))
        if it is None:
            continue
        passe = t <= maintenant
        mesures = tr.mesures_a(obs, t) if passe else {}
        S = tr.entree(geom, c, f, it, mesures)
        stamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")
        vol = wf.predict(S, t.hour + t.minute / 60, int(t.dayofyear), timestamp=stamp,
                         forcage=f.vecteur(t), reliability=False)
        blocs_vent.append(instant(vol, geom))
        blocs_t.append(blocs(S[3], b))           # temperature a 2 m (GFS), deg C
        blocs_hr.append(blocs(S[5], b))          # humidite relative a 2 m (GFS), %
        ray = c.get("shortwave_radiation")
        instants.append({"stem": f"direct_{t:%Y%m%dT%H}", "timestamp_utc": stamp,
                         "rayonnement_w_m2": None if ray is None else round(float(np.nanmean(ray[it])), 1),
                         "heure_locale": float(h), "source": "temps reel" if passe else "prevision",
                         "stations": sorted(mesures), "inference_s": round(float(vol.inference_s), 3),
                         "capteurs_utilises": int(vol.n_sensors)})
    if not instants:
        raise ValueError(f"aucune echeance GFS pour le {date_locale}")
    a = np.stack(blocs_vent)
    prevision = all(i["source"] == "prevision" for i in instants)
    return {"format": "atom-wind/1", "date": date_locale, "temps_reel": True,
            "prevision": prevision,
            "instants": instants, "hauteurs_agl_m": list(HAUTEURS_AGL), "forme": list(a.shape),
            "unite": "cm/s, int16 petit-boutiste, -32768 = sans valeur",
            "ordre": "instant, hauteur, (u, v), y, x", "vent": encoder(a),
            # espace agriculteur : grandeurs de GFS, PAS calculees par le modele (cahier v12, 16.4)
            "t2m": encoder(np.stack(blocs_t)), "hr2m": encoder(np.stack(blocs_hr)),
            "unite_t2m_hr2m": "t2m en deg C, hr2m en % ; meme codage int16 x100 que le vent ; source GFS (pas ATOM Wind)",
            "grande_echelle": "GFS 0,25 deg (NOAA, via Open-Meteo), formules du forcage ERA5",
            "stations_sources": "METAR DTTA (aviationweather.gov), SYNOP 60729 / 60728 (OGIMET)",
            "mis_a_jour_utc": maintenant.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "modele": wf.checkpoint_path.name}


def date_locale_aujourdhui(decalage_j: int = 0) -> str:
    return (pd.Timestamp.now(tz="UTC").tz_localize(None) + pd.Timedelta(hours=UTC_LOCAL_H)
            + pd.Timedelta(days=decalage_j)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------- valeur des stations (E10, E11)
HEURES_DENI = tuple(range(0, 24, 3))     # heures locales rejouees sans chaque station
SEUIL_INFLUENCE_M_S = 0.5


def valeur_stations(wf, source, date_locale: str, maintenant: pd.Timestamp | None = None,
                    heures=HEURES_DENI, h_carte: float = 10.0) -> dict:
    """Deni de donnees sur les heures PASSEES de la journee (cahier v12, E10 et E11) : le modele est
    relance sans chaque station. Pour chaque station :
      - valeur (E11) : erreur vectorielle du modele A SON EMPLACEMENT quand elle est retiree (ce que
        les autres stations et GFS ne savent pas deviner ; faible = station redondante), et erreur
        quand elle est presente ;
      - influence (E10) : ecart |V avec - V sans| a `h_carte` m du sol, moyenne sur les heures, par
        blocs de 500 m ; aire ou il depasse SEUIL_INFLUENCE_M_S.
    Un instant sans mesure ne compte pas ; une station sans mesure du jour n'apparait pas."""
    from . import temps_reel as tr
    from .stations_reelles import CODES, H_STATION, cellules_codes

    geom = wf.geom
    e = source.journee(date_locale)
    f, c, obs = e["forcage"], e["champs"], e["obs"]
    maintenant = maintenant if maintenant is not None else         pd.Timestamp.now(tz="UTC").tz_localize(None)
    index_t = {int(s): i for i, s in enumerate(c["temps"])}
    cell = cellules_codes(geom)
    b = facteur(geom)
    infl: dict[str, list] = {}
    err_avec: dict[str, list] = {}
    err_sans: dict[str, list] = {}
    vues = []

    def passe(S, t):
        vol = wf.predict(S, t.hour + t.minute / 60, int(t.dayofyear), timestamp=t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         forcage=f.vecteur(t), reliability=False)
        carte = champ_agl(vol.u, vol.v, geom, h_carte)
        sol = carte if h_carte == H_STATION else champ_agl(vol.u, vol.v, geom, H_STATION)
        return carte, sol

    for h in heures:
        t = pd.Timestamp(date_locale) + pd.Timedelta(hours=h - UTC_LOCAL_H)
        it = index_t.get(int(t.value // 10 ** 9))
        if it is None or t > maintenant:
            continue
        mes = {k: v for k, v in tr.mesures_a(obs, t).items() if k in cell}
        if not mes:
            continue
        vues.append(float(h))
        (ua, va), (u10, v10) = passe(tr.entree(geom, c, f, it, mes), t)
        for code, (um, vm) in mes.items():
            (ub, vb), (u10b, v10b) = passe(tr.entree(geom, c, f, it, mes, exclue=code), t)
            infl.setdefault(code, []).append(blocs(np.hypot(ua - ub, va - vb), b))
            iy, ix = cell[code]
            err_avec.setdefault(code, []).append(math.hypot(u10[iy, ix] - um, v10[iy, ix] - vm))
            err_sans.setdefault(code, []).append(math.hypot(u10b[iy, ix] - um, v10b[iy, ix] - vm))
    if not vues:
        raise ValueError(f"aucune heure mesuree le {date_locale} (heures passees : {list(heures)})")
    codes = sorted(infl)
    cartes = np.stack([np.nanmean(np.stack(infl[k]), axis=0) for k in codes])
    aire_bloc = (b * geom.dx / 1000.0) ** 2
    stations = []
    for k, carte in zip(codes, cartes):
        sans = float(np.sqrt(np.mean(np.square(err_sans[k]))))
        avec = float(np.sqrt(np.mean(np.square(err_avec[k]))))
        stations.append({"code": k, "nom": CODES.get(k, k), "instants": len(err_sans[k]),
                         "erreur_sans_elle_m_s": round(sans, 2), "erreur_avec_elle_m_s": round(avec, 2),
                         "apport_m_s": round(sans - avec, 2),
                         "influence_moyenne_m_s": round(float(np.nanmean(carte)), 3),
                         "influence_max_m_s": round(float(np.nanmax(carte)), 2),
                         "aire_influence_km2": round(float(np.sum(carte > SEUIL_INFLUENCE_M_S)) * aire_bloc, 1)})
    stations.sort(key=lambda r: -r["erreur_sans_elle_m_s"])
    return {"format": "atom-wind/valeur-stations/1", "date": date_locale, "heures_locales": vues,
            "hauteur_carte_m": h_carte, "seuil_influence_m_s": SEUIL_INFLUENCE_M_S, "stations": stations,
            "codes": codes, "forme": list(cartes.shape), "influence": encoder(cartes),
            "unite": "influence en cm/s, int16 petit-boutiste, ordre station, y, x (blocs de 500 m)",
            "methode": "deni de donnees : le modele relance sans chaque station aux heures mesurees",
            "mis_a_jour_utc": maintenant.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lecture": "erreur_sans_elle faible = station que GFS et les autres stations devinent deja "
                       "(redondante) ; forte = station precieuse, a entretenir en priorite",
            "modele": wf.checkpoint_path.name}


# ---------------------------------------------------------------------- ensemble GEFS (E19)
HEURES_ENSEMBLE = (6, 9, 12, 15, 18)    # heures locales des decisions agricoles
HAUTEURS_ENSEMBLE = (10, 100)           # traitement (10 m) ; moyeu eolien (100 m)
QUANTILES = (0.1, 0.5, 0.9)


def traitement_possible(vitesse_m_s: np.ndarray) -> np.ndarray:
    """« oui » de la regle de traitement de l'application (3 a 15 km/h a 10 m), comme
    scripts/verifier_decisions.py."""
    k = np.round(np.asarray(vitesse_m_s, float) * 3.6, 6)
    return (k >= 3) & (k <= 15)


def jour_ensemble(wf, source, date_locale: str, maintenant: pd.Timestamp | None = None) -> dict:
    """Prevision PROBABILISTE d'une journee (E19) : le modele passe sur les 31 membres GEFS (ensemble
    partiel : vent a 10 m des membres, profil vertical du GFS deterministe decale de l'ecart a
    10 m). Produits agreges sur la grille de 500 m aux heures de traitement : probabilite de pouvoir
    traiter (regle de l'application) et quantiles 10 / 50 / 90 % du vent a 10 m et 100 m."""
    from . import temps_reel as tr

    geom = wf.geom
    e = source.journee(date_locale)
    f, c, obs = e["forcage"], e["champs"], e["obs"]
    ens = source.ensemble(date_locale)
    maintenant = maintenant if maintenant is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    it_det = {int(s): i for i, s in enumerate(c["temps"])}
    it_ens = {int(s): i for i, s in enumerate(ens["temps"])}
    b = facteur(geom)
    instants, probas, quant = [], [], []
    for h in HEURES_ENSEMBLE:
        t = pd.Timestamp(date_locale) + pd.Timedelta(hours=h - UTC_LOCAL_H)
        s = int(t.value // 10 ** 9)
        if s not in it_det or s not in it_ens:
            continue
        it, ie = it_det[s], it_ens[s]
        mesures = tr.mesures_a(obs, t) if t <= maintenant else {}
        stamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")
        v10, v100 = [], []
        for m in range(ens["u10"].shape[0]):
            du = ens["u10"][m, ie] - f.u[it, 0]
            dv = ens["v10"][m, ie] - f.v[it, 0]
            fm = tr.forcage_membre(f, it, du, dv)
            S = tr.entree(geom, c, fm, it, mesures)
            vol = wf.predict(S, t.hour + t.minute / 60, int(t.dayofyear), timestamp=stamp,
                             forcage=fm.vecteur(t), reliability=False)
            for hh, liste in ((10.0, v10), (100.0, v100)):
                u, v = champ_agl(vol.u, vol.v, geom, hh)
                liste.append(blocs(np.hypot(u, v), b))
        v10, v100 = np.stack(v10), np.stack(v100)                  # (M, ny, nx)
        with np.errstate(invalid="ignore"):
            probas.append(np.where(np.isfinite(v10).all(0), traitement_possible(v10).mean(0), np.nan))
            quant.append(np.stack([np.nanquantile(a, QUANTILES, axis=0) for a in (v10, v100)]))
        instants.append({"heure_locale": float(h), "timestamp_utc": stamp, "membres": int(v10.shape[0]),
                         "source": "temps reel" if t <= maintenant else "prevision",
                         "stations": sorted(mesures)})
    if not instants:
        raise ValueError(f"aucune echeance GEFS pour le {date_locale}")
    P, Q = np.stack(probas), np.stack(quant)                         # (I, ny, nx), (I, 2, 3, ny, nx)
    return {"format": "atom-wind/1", "date": date_locale, "ensemble": True, "instants": instants,
            "hauteurs_agl_m": list(HAUTEURS_ENSEMBLE), "quantiles": list(QUANTILES),
            "proba_traitement": encoder(P), "forme_proba": list(P.shape),
            "vitesse_quantiles": encoder(Q), "forme_quantiles": list(Q.shape),
            "unite": "proba : fraction (codee x100) ; vitesses en m/s (codees x100) ; int16, -32768 = sans valeur",
            "methode": ("ensemble partiel GEFS (31 membres, Open-Meteo) : vent a 10 m de chaque membre, "
                        "profil vertical du GFS deterministe decale de l'ecart a 10 m (attenue avec la "
                        "hauteur) ; regle de traitement de l'application (3 a 15 km/h a 10 m)"),
            "mis_a_jour_utc": maintenant.strftime("%Y-%m-%dT%H:%M:%SZ"), "modele": wf.checkpoint_path.name}


# ---------------------------------------------------------------------- qualite du modele servi (TB5)
def qualite(run_dir: Path, dso, w_continuite: bool = False) -> dict:
    """Tableau de qualite du modele SERVI (cahier v12, TB5 : « mis a jour a chaque version »), lu dans son
    run_dir : evaluation sur le test (eval_pcfno3d/best_wcont si w est servi par continuite, sinon best),
    verdict DSO6, validation sur les vraies stations, decision de livraison. Rien n'est recalcule : un
    resultat absent est signale absent."""
    run_dir = Path(run_dir)
    ev_dir = run_dir / "eval_pcfno3d" / ("best_wcont" if w_continuite else "best")
    if not (ev_dir / "test_summary.json").exists():
        ev_dir = run_dir / "eval_pcfno3d" / "best"
    f = ev_dir / "test_summary.json"
    if not f.exists():
        raise FileNotFoundError(f"evaluation absente : {f}")
    s = json.loads(f.read_text(encoding="utf-8"))
    niv = [r for r in s.get("par_niveau_synoptique", []) if r.get("couverture", 1.0) >= dso.min_level_cover]
    pire = max((max(r["rmse_u"], r["rmse_v"]), r["z_m"]) for r in niv) if niv else (float("nan"), None)
    hors = [int(r["z_m"]) for r in niv if max(r["rmse_u"], r["rmse_v"]) > dso.rmse]
    lo, hi = dso.roughness_range
    lignes = [
        {"code": "DSO1", "libelle": f"précision : RMSE u, v ≤ {dso.rmse:g} m/s à chaque niveau",
         "valeur": round(pire[0], 3), "unite": "m/s", "conforme": bool(s.get("dso1_rmse")),
         "note": (f"au-dessus du seuil à {', '.join(map(str, hors))} m" if hors else "tous les niveaux")},
        {"code": "DSO2", "libelle": f"conservation de la masse ≤ {dso.mass_rms:g}",
         "valeur": s.get("dso2_rms_moyen"), "unite": "kg/m³/s", "conforme": bool(s.get("dso2_conforme")),
         "note": "w recalculé par continuité : garanti par construction" if w_continuite else ""},
        {"code": "DSO3", "libelle": f"finesse des détails (rugosité dans [{lo:g} ; {hi:g}])",
         "valeur": s.get("dso3_rugosite"), "unite": "", "conforme": bool(s.get("dso3_conforme")), "note": ""},
        {"code": "DSO4", "libelle": f"rapidité ≤ {dso.time_s:g} s par carte 3D",
         "valeur": s.get("dso4_inference_s"), "unite": "s", "conforme": bool(s.get("dso4_conforme")), "note": ""},
        {"code": "DSO5", "libelle": f"robustesse : RMSE ≤ {dso.rmse_25pct:g} m/s avec 25 % des capteurs",
         "valeur": s.get("dso5_rmse_25pct"), "unite": "m/s", "conforme": bool(s.get("dso5_conforme")), "note": ""},
    ]
    d6 = sorted(ev_dir.glob("dso6*.json"))
    if d6:
        v = json.loads(d6[0].read_text(encoding="utf-8"))
        g = v.get("global", {})
        cov = [g[k] for k in ("couverture80_u", "couverture80_v") if k in g]
        lignes.append({"code": "DSO6", "libelle": "fiabilité : couverture de l'intervalle à 80 %",
                       "valeur": round(min(cov), 3) if cov else None, "unite": "",
                       "conforme": bool(v.get("dso6_conforme")),
                       "note": f"{v.get('instants_synoptiques', '?')} instants du test"})
    else:
        lignes.append({"code": "DSO6", "libelle": "fiabilité : couverture de l'intervalle à 80 %",
                       "valeur": None, "unite": "", "conforme": None, "note": "pas encore mesuré"})
    for l in lignes:
        if isinstance(l["valeur"], float):
            l["valeur"] = float(f"{l['valeur']:.4g}")
    stations = []
    st = run_dir / "validation_reelle" / "test" / "stations_bilan.csv"
    if st.exists():
        t = pd.read_csv(st)
        for nom, g in t.groupby("station"):
            par = dict(zip(g["source"], g["rmse_vect"]))
            stations.append({"station": nom, "modele_m_s": round(float(par["modele"]), 2) if "modele" in par else None,
                             "era5_m_s": round(float(par["era5"]), 2) if "era5" in par else None})
    decision = None
    for dd in (run_dir / "decision_v4.json", run_dir.parent / "decision_v4.json"):
        if dd.exists():
            d = json.loads(dd.read_text(encoding="utf-8"))
            decision = {"modele_livre": d.get("modele_livre"), "texte": d.get("texte")}
            break
    prov = run_dir / "PROVENANCE.txt"
    return {"format": "atom-wind/qualite/1", "modele": s.get("modele"), "run": run_dir.name,
            "provenance": prov.read_text(encoding="utf-8").split(".")[0] if prov.exists() else None,
            "evaluation": ev_dir.name, "instants_synoptiques": s.get("instants_synoptiques"),
            "rmse_global_u_m_s": s.get("dso1_rmse_global_u"), "rmse_global_v_m_s": s.get("dso1_rmse_global_v"),
            "r2_min_uv": min((min(r["r2_u"], r["r2_v"]) for r in niv), default=None),
            "mae_direction_deg": s.get("mae_direction_deg"), "w_continuite": w_continuite,
            "dso": lignes, "stations": stations, "decision": decision,
            "jeu": "test (jours jamais vus a l'entrainement)"}
