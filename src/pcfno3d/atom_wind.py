"""Donnees du front-end ATOM Wind (webapp/atom-wind/) : le champ du modele, compact.

Le front-end raisonne en hauteur AU-DESSUS DU SOL (une grue a 40 m, un drone a 100 m...) sur une
grille de blocs. Un volume complet (12 x 560 x 680, 100 m, Grand Tunis) pese ~55 Mo : on envoie a la
place, pour chaque instant d'une journee, u et v a quelques hauteurs AGL, moyennes par blocs, en
centimetres par seconde (int16). La taille des blocs suit le domaine SERVI (`facteur`) :
    Grand Tunis  68 x 56 km, noeuds 100 m  -> blocs de 500 m, 136 x 112 = 15 232 blocs,
                 ~0,67 Mo par instant (11 hauteurs x u, v x int16) avant base64 et gzip ;
    national     388 x 828 km, noeuds 1 km  -> blocs de 4 km, 97 x 207 = 20 079 blocs,
                 ~0,88 Mo par instant, ~21 Mo pour les 24 heures du temps reel avant gzip.
La regle : le plus petit multiple du pas des noeuds, au moins 500 m, qui garde au plus MAX_BLOCS
blocs ; PCFNO3D_ATOM_PAS_M impose un pas (m) a la place.

    grille(...)  relief, eau, z0, stations, georeferencement et emprise du domaine (une fois)
    jours(...)   les journees servies (heure locale de Tunis, UTC+1, sans heure d'ete)
    jour(...)    les champs d'une journee
    reseau_aneo(...)  plan OS6 du reseau ANEO pour l'onglet d'administration « Reseau ANEO »
"""
from __future__ import annotations

import base64
import json
import math
import os
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


MAX_BLOCS = 25_000                   # ordre de la grille de 500 m du Grand Tunis (15 232 blocs)


def facteur(geom) -> int:
    """Noeuds par bloc (b x b) : 500 m au moins, au plus MAX_BLOCS blocs (domaine national : 4 km).
    PCFNO3D_ATOM_PAS_M (m) impose le pas, sans la limite du nombre de blocs."""
    impose = os.environ.get("PCFNO3D_ATOM_PAS_M")
    b = max(int(round((float(impose) if impose else PAS_M) / geom.dx)), 1)
    if not impose:
        while math.ceil(geom.nx / b) * math.ceil(geom.ny / b) > MAX_BLOCS:
            b += 1
    return b


def projection(georef) -> dict | None:
    """Ce que le navigateur doit savoir pour placer un lat/lon dans la grille (meme plan que le serveur)."""
    if georef is None:
        return None
    from .realdata import R_EARTH, PlanGrille
    if isinstance(georef, PlanGrille):
        return {"type": "plan local a pas constant (latitude centrale, scripts/grille_reelle.py)",
                "lat0": georef.lat0, "lon0": georef.lon0, "lat_centrale": round(georef.lat_centrale, 6),
                "km_par_deg_lon": georef.m_lon / 1000.0, "km_par_deg_lat": georef.m_lat / 1000.0}
    k = R_EARTH * math.pi / 180.0 / 1000.0
    return {"type": "plan tangent au coin sud-ouest (GeoRef)", "lat0": georef.lat0, "lon0": georef.lon0,
            "lat_centrale": georef.lat0, "rotation_deg": georef.rotation_deg,
            "km_par_deg_lon": k * math.cos(math.radians(georef.lat0)), "km_par_deg_lat": k}


def grille(geom, georef, cfg) -> dict:
    """Relief, eau, z0 et stations sur la grille de blocs du domaine servi. `georef` : la projection
    du domaine (realdata.georef_grille : celle du relief), ou None (domaine synthetique)."""
    b = facteur(geom)
    terrain = blocs(geom.zt, b)
    eau = blocs((geom.z0 < 0.001).astype(float), b) > 0.5
    z0 = np.exp(blocs(np.log(np.maximum(geom.z0, 1e-4)), b))
    stations = []
    f = cfg.paths.data_dir / "stations.json"
    if f.exists():
        from .stations_reelles import noms_stations
        noms = noms_stations(geom)
        for s in json.loads(f.read_text(encoding="utf-8")):
            x, y = s["ix"] * geom.dx, s["iy"] * geom.dy
            lat, lon = georef.to_latlon(x, y) if georef is not None else (None, None)
            code = s.get("code")
            stations.append({"nom": noms.get(code, s.get("nom", "station")) if code else s.get("nom", "station"),
                             "code": code, "x_km": x / 1000, "y_km": y / 1000,
                             "lat": None if lat is None else round(float(lat), 5),
                             "lon": None if lon is None else round(float(lon), 5)})
    largeur, hauteur = geom.nx * geom.dx / 1000, geom.ny * geom.dy / 1000
    emprise = None
    if georef is not None:
        la, lo = georef.to_latlon(np.array([0.0, largeur, 0.0, largeur]) * 1000,
                                  np.array([0.0, 0.0, hauteur, hauteur]) * 1000)
        emprise = {"lat_min": round(float(np.min(la)), 5), "lat_max": round(float(np.max(la)), 5),
                   "lon_min": round(float(np.min(lo)), 5), "lon_max": round(float(np.max(lo)), 5)}
    return {
        "format": "atom-wind/1", "nx": int(terrain.shape[1]), "ny": int(terrain.shape[0]),
        "pas_km": geom.dx * b / 1000, "largeur_km": largeur, "hauteur_km": hauteur,
        "pas_noeuds_m": float(geom.dx), "noeuds_par_bloc": b,
        "lat0": None if georef is None else georef.lat0,
        "lon0": None if georef is None else georef.lon0,
        "rotation_deg": None if georef is None else getattr(georef, "rotation_deg", 0.0),
        "projection": projection(georef), "emprise": emprise,
        "domaine": {"nom": cfg.name, "etendue": "national" if max(largeur, hauteur) > 300 else "local",
                    "niveaux_asl_m": [float(z) for z in geom.z_levels]},
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
    """(n_hauteurs, 2, ny, nx) : u, v aux hauteurs AGL, moyennes par blocs (`facteur`)."""
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
def jour_direct(wf, source, date_locale: str, maintenant: pd.Timestamp | None = None,
                fiabilite: bool = False) -> dict:
    """Les 24 heures locales d'une journee (pcfno3d.temps_reel) : heures passees = forcage GFS +
    vraies mesures des stations ; heures a venir = prevision GFS, sans station. Avec, pour
    l'espace agriculteur, la temperature et l'humidite a 2 m de GFS (ramenees au relief, grille de
    500 m) et le rayonnement solaire moyen du domaine : grandeurs que le modele ne calcule pas,
    signalees comme telles. fiabilite=True (modele calibre, `inference.reliability`) : sigma de la vitesse
    par l'ensemble par masquage (lot M2, BF8), environ 4 s de plus par instant sur GPU."""
    from . import temps_reel as tr
    from .alerts import physique, rho_instant

    geom = wf.geom
    e = source.journee(date_locale)
    f, c, obs = e["forcage"], e["champs"], e["obs"]
    maintenant = maintenant if maintenant is not None else \
        pd.Timestamp.now(tz="UTC").tz_localize(None)
    index_t = {int(s): i for i, s in enumerate(c["temps"])}
    instants, blocs_vent, blocs_t, blocs_hr, blocs_sigma, blocs_raf = [], [], [], [], [], []
    b = facteur(geom)
    fiabilite = bool(fiabilite and wf.cfg.inference.reliability is not None)
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
                         forcage=f.vecteur(t), reliability=fiabilite)
        blocs_vent.append(instant(vol, geom))
        if fiabilite:
            sg = sigma_agl(vol, geom)
            if sg is not None:
                blocs_sigma.append(sg)
        # physique de l'instant (BF11, BF31) : residu de continuite et condition au sol du champ servi,
        # densite ISA recalee sur la surface (pas de verite en temps reel)
        rho, src = rho_instant(geom, "extrapolation_surface", S, S[7] > 0.5)
        phys = physique(geom, vol.u, vol.v, vol.w, rho, src, wf.cfg.dso.mass_rms)
        blocs_t.append(blocs(S[3], b))           # temperature a 2 m (GFS), deg C
        # facteur de rafale a 10 m de GFS (rafale / vent moyen), ramene a la grille : remplace le facteur
        # heuristique de la page (BF54, BF80) ; borne a [1 ; 3]
        if "wind_gusts_10m" in c:
            fr = c["wind_gusts_10m"][it] / np.maximum(c["wind_speed_10m"][it], 0.5)
            blocs_raf.append(blocs(np.clip(tr._bil(fr, geom.interp_era5), 1.0, 3.0), b))
        blocs_hr.append(blocs(S[5], b))          # humidite relative a 2 m (GFS), %
        ray = c.get("shortwave_radiation")
        instants.append({"stem": f"direct_{t:%Y%m%dT%H}", "timestamp_utc": stamp,
                         "rayonnement_w_m2": None if ray is None else round(float(np.nanmean(ray[it])), 1),
                         "heure_locale": float(h), "source": "temps reel" if passe else "prevision",
                         "stations": sorted(mesures), "inference_s": round(float(vol.inference_s), 3),
                         "capteurs_utilises": int(vol.n_sensors),
                         "residu_masse_rms": phys["mass_residual_rms"], "residu_masse_ok": phys["mass_residual_ok"],
                         "condition_sol_rms_m_s": phys["ground_condition_rms_ms"]})
    if not instants:
        raise ValueError(f"aucune echeance GFS pour le {date_locale}")
    a = np.stack(blocs_vent)
    prevision = all(i["source"] == "prevision" for i in instants)
    extra = {}
    if blocs_raf and len(blocs_raf) == len(instants) and np.isfinite(np.stack(blocs_raf)).any():
        extra["facteur_rafale"] = encoder(np.stack(blocs_raf))
        extra["source_rafale"] = "GFS wind_gusts_10m / wind_speed_10m (0,25 deg), meme codage int16 x100"
    if blocs_sigma and len(blocs_sigma) == len(instants):
        sg = np.stack(blocs_sigma)
        extra.update({"sigma_vitesse": encoder(sg), "forme_sigma": list(sg.shape)})
    return {**extra, "format": "atom-wind/1", "date": date_locale, "temps_reel": True,
            "prevision": prevision,
            "instants": instants, "hauteurs_agl_m": list(HAUTEURS_AGL), "forme": list(a.shape),
            "unite": "cm/s, int16 petit-boutiste, -32768 = sans valeur",
            "ordre": "instant, hauteur, (u, v), y, x", "vent": encoder(a),
            # espace agriculteur : grandeurs de GFS, PAS calculees par le modele (cahier v12, 16.4)
            "t2m": encoder(np.stack(blocs_t)), "hr2m": encoder(np.stack(blocs_hr)),
            "unite_t2m_hr2m": "t2m en deg C, hr2m en % ; meme codage int16 x100 que le vent ; source GFS (pas ATOM Wind)",
            "grande_echelle": "GFS 0,25 deg (NOAA, via Open-Meteo), formules du forcage ERA5",
            "stations_sources": tr.description_stations(geom),
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
    from .stations_reelles import H_STATION, cellules_codes, noms_stations

    geom = wf.geom
    CODES = noms_stations(geom)                  # noqa: N806  (Grand Tunis : les 3 ; national : les aeroports)
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


# ---------------------------------------------------------------------- scenario de panne (BF88)
def scenario_panne(wf, source, date_locale: str, heure_locale: float, retirees: list[str],
                   maintenant: pd.Timestamp | None = None) -> dict:
    """Le champ d'une heure deja mesuree recalcule SANS les stations `retirees` (codes DTTA, INM60729,
    INM60728 ; aeroports METAR sur le domaine national) : ecart a 10 m et 100 m sur le domaine et a chaque
    station, contre le champ servi."""
    from . import temps_reel as tr
    from .stations_reelles import H_STATION, cellules_codes, noms_stations

    geom = wf.geom
    CODES = noms_stations(geom)                  # noqa: N806
    e = source.journee(date_locale)
    f, c, obs = e["forcage"], e["champs"], e["obs"]
    maintenant = maintenant if maintenant is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    t = pd.Timestamp(date_locale) + pd.Timedelta(hours=float(heure_locale) - UTC_LOCAL_H)
    it = {int(x): i for i, x in enumerate(c["temps"])}.get(int(t.value // 10 ** 9))
    if it is None or t > maintenant:
        raise ValueError(f"{date_locale} {heure_locale:g} h : heure pas encore mesuree ou sans GFS")
    cell = cellules_codes(geom)
    inconnues = [k for k in retirees if k not in CODES]
    if inconnues:
        raise ValueError(f"stations inconnues : {inconnues} (codes : {', '.join(CODES)})")
    mes = {k: v for k, v in tr.mesures_a(obs, t).items() if k in cell}
    reste = {k: v for k, v in mes.items() if k not in retirees}

    def passe(m):
        S = tr.entree(geom, c, f, it, m)
        vol = wf.predict(S, t.hour + t.minute / 60, int(t.dayofyear), timestamp=t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         forcage=f.vecteur(t), reliability=False)
        return {h: champ_agl(vol.u, vol.v, geom, float(h)) for h in (H_STATION, 100.0)}

    a, b = passe(mes), passe(reste)
    domaine = {}
    for h in (H_STATION, 100.0):
        d = np.hypot(a[h][0] - b[h][0], a[h][1] - b[h][1])
        domaine[f"{h:g}m"] = {"ecart_moyen_m_s": round(float(np.nanmean(d)), 3),
                             "ecart_max_m_s": round(float(np.nanmax(d)), 3),
                             "part_au_dessus_0_5_m_s": round(float(np.nanmean(d > 0.5)), 4)}
    stations = {}
    for k, (iy, ix) in cell.items():
        ua, va = a[H_STATION][0][iy, ix], a[H_STATION][1][iy, ix]
        ub, vb = b[H_STATION][0][iy, ix], b[H_STATION][1][iy, ix]
        r = {"retiree": k in retirees, "mesuree": k in mes, "ecart_m_s": round(float(math.hypot(ua - ub, va - vb)), 3)}
        if k in mes:
            um, vm = mes[k]
            r["erreur_avec_m_s"] = round(float(math.hypot(ua - um, va - vm)), 2)
            r["erreur_sans_m_s"] = round(float(math.hypot(ub - um, vb - vm)), 2)
        stations[CODES.get(k, k)] = r
    return {"date": date_locale, "heure_locale": float(heure_locale), "retirees": [CODES.get(k, k) for k in retirees],
            "mesurees": [CODES.get(k, k) for k in mes], "domaine": domaine, "stations": stations,
            "lecture": "ecart = ce que la panne change dans le champ ; erreur_sans a une station retiree = ce que "
                       "le reste du reseau et GFS devinent sans elle"}


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


# ---------------------------------------------------------------------- reseau ANEO (OS6) : onglet admin
# L'onglet d'administration « Reseau ANEO » (feuille de route nationale, phase 8) lit les sorties de
# scripts/os6_reseau_aneo.py (pcfno3d.os6) dans le run_dir SERVI ou son parent, Grand Tunis comme national :
#   plan_aneo.csv / plan_aneo.geojson   ordre, lat, lon, altitude, hauteur de mat, gain de chaque capteur
#   courbe_test.csv                     erreur selon le nombre de capteurs : optimise / grille / hasard
#   os6.json                            nombre recommande, seuil DSO1, reseau actuel, contraintes, limites
# et, a cote (run_dir, son parent ou le dossier du projet) :
#   os6_installes.csv                   capteurs deja poses (lat, lon [, id, installe_le]) : --installes (BF78)
#   os6_parametres.json                 parametres de Tukhnanutha : {"cout_unitaire": ..., "devise": "TND"}
# Sans sorties OS6 : un plan d'EXEMPLE (36 emplacements eloignes des stations, sans aucun chiffre d'erreur,
# de gain ni de cout), signale par le bandeau « Demonstration : plan d'exemple ».
#
# Phases de deploiement : les paliers de la courbe de dimensionnement (nombres de capteurs ou l'erreur a ete
# MESUREE sur le test, strategie optimisee). La phase k pose les capteurs d'ordre n_(k-1) + 1 a n_k ; son
# erreur attendue est celle de la courbe a n_k, son gain l'ecart a la phase precedente (au reseau actuel pour
# la premiere). Rien n'est interpole entre deux paliers.
NOM_OS6 = "os6_reseau_aneo"
FICHIERS_OS6 = ("plan_aneo.csv", "plan_aneo.geojson", "courbe_test.csv", "courbe_test_par_niveau.csv",
                "os6.json", "candidats.csv", "ordre_validation.csv", "courbe.png", "carte.png")
INSTALLES = "os6_installes.csv"
PARAMETRES = "os6_parametres.json"
N_EXEMPLE = 36
RAYON_INSTALLE_KM = 1.0                  # un capteur pose a moins de 1 km d'un emplacement prevu : le meme


def dossier_os6(run_dir: Path) -> Path | None:
    """Le dossier des sorties OS6 du modele servi (run_dir/os6_reseau_aneo, sinon a cote du run_dir)."""
    run_dir = Path(run_dir)
    for d in (run_dir / NOM_OS6, run_dir.parent / NOM_OS6):
        if (d / "plan_aneo.csv").exists() or (d / "plan_aneo.geojson").exists():
            return d
    return None


def _a_cote(run_dir: Path, nom: str) -> Path | None:
    """Fichier depose par l'exploitant : run_dir, puis son parent (runs/), puis le dossier du projet (Drive)."""
    run_dir = Path(run_dir)
    for d in (run_dir, run_dir.parent, run_dir.parent.parent):
        if (d / nom).exists():
            return d / nom
    return None


def lire_installes(run_dir: Path) -> tuple[Path | None, list[dict]]:
    f = _a_cote(run_dir, INSTALLES)
    if f is None:
        return None, []
    t = pd.read_csv(f, dtype={"id": str, "installe_le": str})
    if not {"lat", "lon"} <= set(t.columns):
        raise ValueError(f"{f} : colonnes lat et lon attendues")
    out = []
    for _, r in t.iterrows():
        out.append({"id": str(r["id"]) if "id" in t.columns and pd.notna(r["id"]) and str(r["id"]) else None,
                    "lat": float(r["lat"]), "lon": float(r["lon"]),
                    "installe_le": str(r["installe_le"]) if "installe_le" in t.columns
                    and pd.notna(r["installe_le"]) else None})
    return f, out


def lire_parametres(run_dir: Path) -> dict:
    """Cout unitaire d'un capteur (et sa devise) : parametre attendu de Tukhnanutha, jamais invente."""
    f = _a_cote(run_dir, PARAMETRES)
    out = {"unitaire": None, "devise": None, "source": None,
           "note": f"coût unitaire à renseigner ({PARAMETRES} : cout_unitaire, devise)"}
    if f is None:
        return out
    d = json.loads(f.read_text(encoding="utf-8"))
    c = d.get("cout_unitaire")
    if isinstance(c, (int, float)) and not isinstance(c, bool) and c > 0:
        out.update({"unitaire": float(c), "devise": d.get("devise") or "", "source": str(f),
                    "note": d.get("note") or "coût unitaire fourni par l'exploitant"})
    return out


def _courbe(d: Path, os6j: dict) -> dict | None:
    f = d / "courbe_test.csv"
    if f.exists():
        t = pd.read_csv(f)
    elif os6j.get("courbe"):
        t = pd.DataFrame(os6j["courbe"])
    else:
        return None
    strat = {}
    for s, g in t.groupby("strategie"):
        strat[str(s)] = [{"n": int(r["n_capteurs"]), "rmse_vect_m_s": round(float(r["rmse_vect"]), 4),
                          "pire_rmse_niveau_m_s": round(float(r["pire_rmse_niveau"]), 4),
                          "seuil_respecte": bool(r["seuil_respecte"]) if "seuil_respecte" in g.columns else None}
                         for _, r in g.sort_values("n_capteurs").iterrows()]
    actuel = strat.pop("actuel", [None])[0]
    return {"strategies": strat, "actuel": actuel,
            "tailles": sorted({p["n"] for v in strat.values() for p in v}),
            "jeu": "test (calculé une fois par scripts/os6_reseau_aneo.py)"}


def _nombre(v) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _plan_os6(d: Path, geom, georef) -> list[dict]:
    """Capteurs du plan, dans l'ordre d'installation (CSV, sinon GeoJSON)."""
    if (d / "plan_aneo.csv").exists():
        lignes = pd.read_csv(d / "plan_aneo.csv").to_dict(orient="records")
    else:
        g = json.loads((d / "plan_aneo.geojson").read_text(encoding="utf-8"))
        lignes = []
        for ft in g.get("features", []):
            p = dict(ft.get("properties") or {})
            if p.get("type") not in (None, "capteur ANEO propose") or (ft.get("geometry") or {}).get("type") != "Point":
                continue
            lo, la = ft["geometry"]["coordinates"][:2]
            lignes.append({**p, "lat": la, "lon": lo})
    out = []
    for k, r in enumerate(sorted(lignes, key=lambda r: _nombre(r.get("ordre")) or 0)):
        x_km = y_km = None
        if _nombre(r.get("ix")) is not None and _nombre(r.get("iy")) is not None:
            x_km, y_km = float(r["ix"]) * geom.dx / 1000, float(r["iy"]) * geom.dy / 1000
        elif georef is not None and _nombre(r.get("lat")) is not None:
            x, y = georef.to_xy(r["lat"], r["lon"])
            x_km, y_km = float(x) / 1000, float(y) / 1000
        etape = _nombre(r.get("etape"))
        out.append({"id": str(r.get("id") or f"ANEO-{k + 1:03d}"), "ordre": int(_nombre(r.get("ordre")) or k + 1),
                    "etape": None if etape is None else int(etape),
                    "lat": _nombre(r.get("lat")), "lon": _nombre(r.get("lon")), "x_km": x_km, "y_km": y_km,
                    "altitude_sol_m": _nombre(r.get("altitude_sol_m")), "hauteur_mat_m": _nombre(r.get("hauteur_mat_m")),
                    "gain_m_s": _nombre(r.get("gain_rmse_validation_m_s")),
                    "gain_cumule_m_s": _nombre(r.get("gain_cumule_m_s")),
                    "rmse_validation_apres_m_s": _nombre(r.get("rmse_validation_apres_m_s"))})
    return out


def plan_exemple(geom, georef, n: int = N_EXEMPLE) -> list[dict]:
    """Plan d'EXEMPLE (aucun calcul OS6) : n emplacements candidats (contraintes OS6 par defaut), chacun le
    plus loin possible des stations et des precedents. Aucune erreur, aucun gain : ce n'est pas un resultat."""
    from .dataset import cellules_stations
    from .os6 import Contraintes, candidats
    etendue = max(geom.nx * geom.dx, geom.ny * geom.dy)
    c = Contraintes(pas_candidats_m=max(500.0, etendue / 60.0))
    cand = candidats(geom, c, None)
    if not len(cand):
        return []
    xy = np.stack([cand[:, 1] * geom.dx, cand[:, 0] * geom.dy], 1)
    st = cellules_stations(geom)
    pris = [np.array([b * geom.dx, a * geom.dy], float) for a, b in st] or [xy.mean(0)]
    d = np.min([np.hypot(*(xy - p).T) for p in pris], axis=0)
    out = []
    for k in range(min(n, len(cand))):
        i = int(np.argmax(d))
        jy, ix = (int(v) for v in cand[i])
        lat = lon = None
        if georef is not None:
            la, lo = georef.to_latlon(ix * geom.dx, jy * geom.dy)
            lat, lon = round(float(la), 6), round(float(lo), 6)
        out.append({"id": f"EXEMPLE-{k + 1:03d}", "ordre": k + 1, "etape": None, "lat": lat, "lon": lon,
                    "x_km": ix * geom.dx / 1000, "y_km": jy * geom.dy / 1000,
                    "altitude_sol_m": round(float(geom.zt[jy, ix]), 1),
                    "hauteur_mat_m": round(float(geom.zrel_sensor[jy, ix]), 1),
                    "gain_m_s": None, "gain_cumule_m_s": None, "rmse_validation_apres_m_s": None})
        d = np.minimum(d, np.hypot(*(xy - xy[i]).T))
    return out


def phases(tailles, n_plan: int) -> list[tuple[int, int]]:
    """[(premier ordre, dernier ordre)] : paliers de la courbe (bornes au plan), puis le reste du plan."""
    bornes = sorted({int(t) for t in tailles if 0 < int(t) <= n_plan})
    if n_plan > 0 and (not bornes or bornes[-1] < n_plan):
        bornes.append(n_plan)
    out, prec = [], 0
    for b in bornes:
        out.append((prec + 1, b))
        prec = b
    return out


def _associer_installes(capteurs: list[dict], installes: list[dict], georef) -> list[dict]:
    """Statut « installe » des emplacements prevus (par identifiant, sinon le plus proche a moins de
    RAYON_INSTALLE_KM) ; renvoie les capteurs poses hors du plan."""
    par_id = {c["id"]: c for c in capteurs}
    hors = []
    for p in installes:
        c = par_id.get(p["id"]) if p["id"] else None
        if c is None and georef is not None:
            x, y = georef.to_xy(p["lat"], p["lon"])
            best, dmin = None, RAYON_INSTALLE_KM
            for q in capteurs:
                if q["x_km"] is None or q["statut"] == "installé":
                    continue
                dd = math.hypot(float(x) / 1000 - q["x_km"], float(y) / 1000 - q["y_km"])
                if dd <= dmin:
                    best, dmin = q, dd
            c = best
        if c is None:
            hors.append(p)
        else:
            c["statut"], c["installe_le"] = "installé", p["installe_le"]
    return hors


def commande_installes(cfg, checkpoint, d_os6: Path | None, run_dir: Path, installes: Path | None,
                       tailles=None, config: str | None = None) -> dict:
    """La commande qui relance le plan a partir des capteurs poses (BF78). Elle n'est JAMAIS lancee par le
    service (calcul GPU, carnet Colab) ; nouvelle sortie datee : le plan servi n'est pas ecrase."""
    import shlex
    base = d_os6.parent if d_os6 is not None else Path(run_dir).parent
    sortie = base / f"{NOM_OS6}_installes_{pd.Timestamp.now(tz='UTC'):%Y%m%d}"
    morceaux = ["python", "-u", "scripts/os6_reseau_aneo.py", "-c", config or cfg.name,
                "--checkpoint", str(checkpoint),
                "--set", f"inference.w_continuite={'true' if cfg.inference.w_continuite else 'false'}"]
    if installes is not None:
        morceaux += ["--installes", str(installes)]
    if tailles:
        morceaux += ["--tailles", ",".join(str(int(t)) for t in tailles)]
    morceaux += ["--sortie", str(sortie)]
    return {"commande": " ".join(shlex.quote(m) for m in morceaux), "sortie": str(sortie),
            "avec_installes": installes is not None, "lancee_par_le_service": False,
            "note": ("À lancer dans le carnet OS6 (GPU), jamais par le service. Le placement repart des capteurs "
                     "installés ; la courbe de dimensionnement est recalculée sur le test (deuxième lecture du "
                     "test : décision d'Ines). Une fois vérifié, renommer le dossier en "
                     f"{NOM_OS6} (l'ancien renommé, jamais effacé) pour que le service le serve.")}


def reseau_aneo(run_dir: Path, cfg, geom, georef, checkpoint="best.pt", config: str | None = None) -> dict:
    """Tout ce qu'affiche l'onglet « Reseau ANEO » : plan, phases, courbe, statut prevu / installe, cout,
    commande --installes. Les chiffres viennent des fichiers OS6 ; rien n'est recalcule ni invente."""
    from .os6 import TAILLES
    run_dir = Path(run_dir)
    d = dossier_os6(run_dir)
    os6j = json.loads((d / "os6.json").read_text(encoding="utf-8")) \
        if d is not None and (d / "os6.json").exists() else {}
    demo = d is None
    capteurs = plan_exemple(geom, georef) if demo else _plan_os6(d, geom, georef)
    courbe = None if demo else _courbe(d, os6j)
    for c in capteurs:
        c["statut"], c["installe_le"], c["phase"] = "prévu", None, None
    f_inst, inst = lire_installes(run_dir)
    hors = inst if demo else _associer_installes(capteurs, inst, georef)
    cout = lire_parametres(run_dir)
    seuil = float(os6j.get("seuil_rmse_m_s", cfg.dso.rmse))
    opt = {p["n"]: p for p in (courbe or {}).get("strategies", {}).get("optimise", [])}
    actuel = (courbe or {}).get("actuel")
    if actuel is None and os6j.get("reseau_actuel"):
        a = os6j["reseau_actuel"]
        actuel = {"n": 0, "rmse_vect_m_s": a.get("rmse_vect"), "pire_rmse_niveau_m_s": a.get("pire_rmse_niveau")}
    tailles = (courbe or {}).get("tailles") or list(TAILLES)
    lignes, prec = [], actuel
    for k, (a, b) in enumerate(phases(tailles, len(capteurs)), start=1):
        sel = [c for c in capteurs if a <= c["ordre"] <= b]
        for c in sel:
            c["phase"] = k
        mes = opt.get(b)
        gain = None
        if mes is not None and prec is not None and prec.get("pire_rmse_niveau_m_s") is not None:
            gain = round(prec["pire_rmse_niveau_m_s"] - mes["pire_rmse_niveau_m_s"], 4)
        lignes.append({"phase": k, "de": a, "a": b, "capteurs": len(sel), "cumul": b,
                       "pire_rmse_niveau_m_s": None if mes is None else mes["pire_rmse_niveau_m_s"],
                       "rmse_vect_m_s": None if mes is None else mes["rmse_vect_m_s"],
                       "gain_pire_niveau_m_s": gain,
                       "seuil_dso1_respecte": None if mes is None else bool(mes["pire_rmse_niveau_m_s"] <= seuil),
                       "installes": sum(c["statut"] == "installé" for c in sel),
                       "cout": None if cout["unitaire"] is None else round(cout["unitaire"] * len(sel), 2),
                       "cout_cumule": None if cout["unitaire"] is None else round(cout["unitaire"] * b, 2)})
        if mes is not None:
            prec = mes
    stations = []
    f = cfg.paths.data_dir / "stations.json"
    if f.exists():
        from .stations_reelles import noms_stations
        noms = noms_stations(geom)
        for s in json.loads(f.read_text(encoding="utf-8")):
            lat = lon = None
            if georef is not None:
                la, lo = georef.to_latlon(s["ix"] * geom.dx, s["iy"] * geom.dy)
                lat, lon = round(float(la), 5), round(float(lo), 5)
            code = s.get("code")
            stations.append({"nom": noms.get(code, s.get("nom")) if code else s.get("nom", "station"),
                             "code": code, "lat": lat, "lon": lon,
                             "x_km": s["ix"] * geom.dx / 1000, "y_km": s["iy"] * geom.dy / 1000})
    return {
        "format": "atom-wind/reseau-aneo/1", "demonstration": demo,
        "bandeau": ("Démonstration : plan d'exemple" if demo else
                    f"Plan OS6 calculé ({os6j.get('modele') or 'modèle servi'}, configuration "
                    f"{os6j.get('configuration') or cfg.name})"),
        "domaine": {"nom": cfg.name, "largeur_km": geom.nx * geom.dx / 1000, "hauteur_km": geom.ny * geom.dy / 1000},
        "dossier": None if d is None else str(d),
        "fichiers": [] if d is None else [n for n in FICHIERS_OS6 if (d / n).exists()],
        "seuil_dso1_m_s": seuil, "nombre_recommande": os6j.get("nombre_recommande"),
        "reseau_actuel": actuel, "courbe": courbe, "capteurs": capteurs, "phases": lignes,
        "regle_phases": ("paliers de la courbe de dimensionnement (erreur mesurée sur le test, stratégie "
                         "optimisée) ; sans courbe, paliers par défaut de pcfno3d.os6.TAILLES, sans chiffre"),
        "stations": stations, "cout": cout,
        "installes": {"fichier": None if f_inst is None else str(f_inst), "n": len(inst), "hors_plan": hors},
        "relance": commande_installes(cfg, checkpoint, d, run_dir, f_inst, (courbe or {}).get("tailles"), config),
        "contraintes": os6j.get("contraintes"), "candidats": os6j.get("candidats"),
        "limites": os6j.get("limites") or ("Plan d'exemple : emplacements éloignés des stations, sans calcul OS6 "
                                           "(ni erreur, ni gain, ni coût)." if demo else None),
    }


def ecrire_installes(run_dir: Path, plan: dict, ids: list[str]) -> Path:
    """Ecrit run_dir/os6_installes.csv (atomique) : les emplacements du plan marques installes (`ids`), plus les
    capteurs deja poses hors du plan (gardes tels quels). La date d'installation d'un capteur deja connu est
    conservee, celle d'un nouveau est la date du jour (UTC)."""
    if plan["demonstration"]:
        raise ValueError("plan d'exemple : aucun capteur à installer (calculer d'abord le plan OS6)")
    connus = {c["id"]: c for c in plan["capteurs"]}
    inconnus = [i for i in ids if i not in connus]
    if inconnus:
        raise KeyError(f"capteurs absents du plan : {', '.join(inconnus[:5])}")
    auj = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    lignes = []
    for i in dict.fromkeys(ids):
        c = connus[i]
        if c["lat"] is None or c["lon"] is None:
            raise ValueError(f"{i} : coordonnées absentes du plan")
        lignes.append({"id": i, "lat": c["lat"], "lon": c["lon"], "installe_le": c.get("installe_le") or auj})
    for p in plan["installes"]["hors_plan"]:
        lignes.append({"id": p["id"] or "", "lat": p["lat"], "lon": p["lon"], "installe_le": p["installe_le"] or ""})
    f = Path(run_dir) / INSTALLES
    tmp = f.with_name(f.name + ".tmp")
    pd.DataFrame(lignes, columns=["id", "lat", "lon", "installe_le"]).to_csv(tmp, index=False)
    tmp.replace(f)
    return f
