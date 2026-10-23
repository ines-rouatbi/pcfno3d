"""Anemometres REELS du domaine Grand Tunis : lecture, entree du modele, vent a 10 m, affinage.

Trois stations dans le domaine : Tunis-Carthage (METAR DTTA), Zaghouan et Nabeul (SYNOP INM
60729 / 60728, NOAA ISD). Elles n'entrent jamais dans la verite generee ; elles servent :
  - d'ENTREE (production : ERA5 10 m + les anemometres) ;
  - de VALIDATION (scripts/valider_grand_tunis_reel.py, station retiree de l'entree) ;
  - d'AFFINAGE hors blocs de test (`loss.lambda_station`, `paths.stations_obs`) : a chaque
    instant d'entrainement, une station mesuree est tiree comme CIBLE et retiree de l'entree,
    les autres stations mesurees sont en entree ; le vent a 10 m predit a la cible est
    rapproche de la mesure, en plus de la verite generee partout.

Le vent a 10 m au-dessus du sol est une combinaison LINEAIRE fixe de deux noeuds de la colonne
(interpolation entre noeuds, ou profil logarithmique depuis le noeud valide le plus bas) :
`poids_10m` donne cette combinaison, ce qui rend le terme d'affinage derivable.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CODES = {"DTTA": "Tunis-Carthage", "INM60729": "Zaghouan", "INM60728": "Nabeul"}
FENETRE = pd.Timedelta(minutes=30)
H_STATION = 10.0                       # hauteur des anemometres (m au-dessus du sol)
N_CIBLE = 10                           # k_a, k_b, w_a, w_b, iy, ix, u, v, presente, poids vitesse


# ---------------------------------------------------------------------- observations
def lire_observations(dossier: Path, verbose: bool = True) -> pd.DataFrame:
    """METAR + SYNOP des trois stations du domaine ; u, v recalcules s'ils manquent ; mois en
    panne (vent nul a plus de 90 %, au moins 20 mesures) ecartes."""
    dossier = Path(dossier)
    fichiers = sorted(dossier.glob("metar_*.csv")) + sorted(dossier.glob("synop_*.csv"))
    if not fichiers:
        raise FileNotFoundError(f"aucun metar_*.csv / synop_*.csv dans {dossier}")
    obs = pd.concat([pd.read_csv(f) for f in fichiers], ignore_index=True)
    obs = obs[obs["station_id"].isin(list(CODES))].copy()
    obs["t"] = pd.to_datetime(obs["timestamp_utc"], utc=True).dt.tz_localize(None)
    rad = np.radians(obs["wind_dir_deg"])
    calme = obs["wind_speed_m_s"] == 0
    obs["wind_u_m_s"] = obs["wind_u_m_s"].fillna((-obs["wind_speed_m_s"] * np.sin(rad)).where(~calme, 0.0))
    obs["wind_v_m_s"] = obs["wind_v_m_s"].fillna((-obs["wind_speed_m_s"] * np.cos(rad)).where(~calme, 0.0))
    obs = obs.dropna(subset=["wind_u_m_s", "wind_v_m_s"])
    mois = obs["t"].dt.to_period("M")
    part = (obs["wind_speed_m_s"] > 0).groupby([obs["station_id"], mois]).transform("mean")
    n = obs.groupby([obs["station_id"], mois])["t"].transform("size")
    panne = (part < 0.1) & (n >= 20)
    if panne.any() and verbose:
        ecart = obs[panne].groupby(["station_id", obs.loc[panne, "t"].dt.to_period("M")]).size()
        print("mois ecartes (anemometre en panne) :", ", ".join(f"{s} {m}" for s, m in ecart.index))
    return obs[~panne].sort_values("t").reset_index(drop=True)


def obs_a(obs: pd.DataFrame, t: pd.Timestamp) -> dict[str, tuple[float, float]]:
    """{code: (u, v)} : la mesure la plus proche de t a 30 min au plus, par station."""
    fen = obs[(obs["t"] >= t - FENETRE) & (obs["t"] <= t + FENETRE)]
    out = {}
    for code, g in fen.groupby("station_id"):
        r = g.iloc[int(np.argmin(np.abs((g["t"] - t).to_numpy())))]
        out[code] = (float(r["wind_u_m_s"]), float(r["wind_v_m_s"]))
    return out


# ---------------------------------------------------------------------- geometrie
def cellules_codes(geom) -> dict[str, tuple[int, int]]:
    """{code: (iy, ix)} des stations du domaine (data_dir/stations.json du generateur), lu une
    fois par geometrie."""
    cache = getattr(geom, "_cellules_codes", None)
    if cache is None:
        st = json.loads((geom.cfg.paths.data_dir / "stations.json").read_text(encoding="utf-8"))
        cache = {}
        for s in st:
            code = ("DTTA" if "DTTA" in s["nom"] else "INM60729" if "60729" in s["nom"]
                    else "INM60728" if "60728" in s["nom"] else None)
            if code:
                cache[code] = (int(s["iy"]), int(s["ix"]))
        geom._cellules_codes = cache
    return cache


def entree_reelle(S: np.ndarray, E: np.ndarray, geom, cellules: dict, mesures: dict,
                  exclue: str | None = None,
                  decalage: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Entree de production avec les VRAIES mesures (meme regle que
    pcfno3d.dataset.entree_era5_stations) : ERA5 10 m sur le reseau regulier (decale de
    `decalage`, par defaut au milieu de la maille comme en evaluation), les stations mesurees a
    leur cellule (une station prime sur le pseudo-capteur), w = 0.
    `exclue` : station retiree de l'entree (sa cellule redevient ERA5 si elle est sur le
    reseau, sinon sans capteur, comme une station absente a l'entrainement)."""
    k = max(int(round(geom.cfg.mask.pas_era5_m / geom.dx)), 1)
    oy, ox = decalage if decalage is not None else (k // 2, k // 2)
    reseau = np.zeros(S.shape[1:], dtype=bool)
    reseau[oy::k, ox::k] = True
    S = S.copy()
    S[0] = np.where(reseau, E[0], S[0])
    S[1] = np.where(reseau, E[1], S[1])
    S[2] = 0.0
    capteur = reseau.copy()
    for code, (iy, ix) in cellules.items():
        if code == exclue or code not in mesures:
            continue
        S[0, iy, ix], S[1, iy, ix] = mesures[code]
        capteur[iy, ix] = True
    return S, capteur & np.isfinite(S[:7]).all(axis=0)


def poids_10m(geom, iy: int, ix: int, h: float = H_STATION) -> tuple[int, int, float, float] | None:
    """(k_a, k_b, w_a, w_b) : vent a h m au-dessus du sol = w_a * noeud k_a + w_b * noeud k_b
    de la colonne (iy, ix). Interpolation entre deux noeuds valides si h est encadre, sinon
    profil logarithmique depuis le plus bas (z0 de la maille). None : colonne sans noeud."""
    ks = np.nonzero(geom.valid[:, iy, ix])[0]
    if len(ks) == 0:
        return None
    hk = np.asarray(geom.z_levels, float)[ks] - float(geom.zt[iy, ix])
    if hk[0] <= h <= hk[-1]:
        j = int(np.clip(np.searchsorted(hk, h) - 1, 0, len(ks) - 2)) if len(ks) > 1 else 0
        if len(ks) == 1:
            return int(ks[0]), int(ks[0]), 1.0, 0.0
        a = (h - hk[j]) / (hk[j + 1] - hk[j])
        return int(ks[j]), int(ks[j + 1]), float(1 - a), float(a)
    if h < hk[0]:
        z0 = max(float(geom.z0[iy, ix]), 1e-4)
        return int(ks[0]), int(ks[0]), float(math.log(h / z0) / math.log(hk[0] / z0)), 0.0
    return None


def vent_a_hauteur(u: np.ndarray, v: np.ndarray, h_noeuds: np.ndarray, z0: float,
                   h: float = H_STATION) -> tuple[float, float]:
    """u, v d'une colonne (NaN = noeud sous le relief) a h metres au-dessus du sol :
    interpolation entre noeuds si h est encadre, sinon profil logarithmique depuis le noeud
    valide le plus bas (z0 de la maille)."""
    ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(h_noeuds)
    if not ok.any():
        return math.nan, math.nan
    hh, uu, vv = h_noeuds[ok], u[ok], v[ok]
    o = np.argsort(hh)
    hh, uu, vv = hh[o], uu[o], vv[o]
    if hh[0] <= h <= hh[-1]:
        return float(np.interp(h, hh, uu)), float(np.interp(h, hh, vv))
    if h < hh[0]:
        z0 = max(float(z0), 1e-4)
        f = math.log(h / z0) / math.log(hh[0] / z0)
        return float(uu[0] * f), float(vv[0] * f)
    return math.nan, math.nan


# ---------------------------------------------------------------------- affinage
def table_stations(geom) -> dict[str, dict[str, tuple[float, float]]]:
    """{stem: {code: (u, v)}} des mesures a 30 min au plus de chaque instant du cache, lu une
    fois par geometrie (paths.stations_obs)."""
    cache = getattr(geom, "_table_stations", None)
    if cache is None:
        cfg = geom.cfg
        obs = lire_observations(cfg.paths.stations_obs, verbose=False)
        index = pd.read_csv(cfg.paths.index_csv, usecols=["stem", "timestamp"])
        cache = {str(s): obs_a(obs, pd.Timestamp(t)) for s, t in zip(index["stem"], index["timestamp"])}
        geom._table_stations = cache
    return cache


def entree_affinage(S: np.ndarray, chemin_e, geom, rng: np.random.Generator,
                    stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entree d'entrainement de l'affinage : une station mesuree (parmi `train.stations_cibles`)
    tiree comme cible et retiree de l'entree ; les autres stations mesurees en entree, chacune
    avec la probabilite `mask.p_station` ; ERA5 10 m sur un reseau au decalage aleatoire.
    Renvoie (S, capteur, cible) ; cible (N_CIBLE,) = k_a, k_b, w_a, w_b, iy, ix, u, v, presente,
    poids de la vitesse (`loss.poids_vitesse_station`, 1 par defaut ; 0 = direction seule, la
    cible est alors ecartee si le vent mesure est sous 1 m/s : direction non definie)."""
    cfg = geom.cfg
    k = max(int(round(cfg.mask.pas_era5_m / geom.dx)), 1)
    mesures = table_stations(geom).get(str(stem), {})
    autorisees = cfg.train.stations_cibles or list(CODES)
    candidates = sorted(c for c in mesures if c in autorisees)
    cible = candidates[int(rng.integers(len(candidates)))] if candidates else None
    entrees = {c: uv for c, uv in mesures.items()
               if c != cible and rng.random() < cfg.mask.p_station}
    E = np.load(chemin_e).astype(np.float32)
    dec = (int(rng.integers(0, k)), int(rng.integers(0, k)))
    cellules = cellules_codes(geom)
    S_in, capteur = entree_reelle(S, E, geom, cellules, entrees, decalage=dec)
    vec = np.zeros(N_CIBLE, np.float32)
    if cible is not None and cible in cellules:
        iy, ix = cellules[cible]
        w = poids_10m(geom, iy, ix)
        pv = float((cfg.loss.poids_vitesse_station or {}).get(cible, 1.0))
        vitesse = math.hypot(*mesures[cible])
        if w is not None and (pv >= 1.0 or vitesse >= 1.0):
            vec[:] = (*w, iy, ix, *mesures[cible], 1.0, pv)
    return S_in, capteur, vec


def perte_station(pred, cible, norm):
    """Erreur quadratique (normalisee par l'ecart-type de la cible) entre le vent a 10 m predit
    aux stations cibles et la mesure, moyenne sur les exemples du lot qui ont une cible.
    Poids de la vitesse pv (colonne 9) : la cible est pv * mesure + (1 - pv) * |prediction| *
    direction mesuree ; pv = 0 ne juge que la DIRECTION (station abritee dont la vitesse n'est
    pas representative du vent alentour, ex. Nabeul).
    pred (B, 3, nz, ny, nx) normalise ; cible (B, N_CIBLE)."""
    import torch

    ok = cible[:, 8]
    if float(ok.sum()) == 0.0:
        return pred.sum() * 0.0
    b = torch.arange(pred.shape[0], device=pred.device)
    ka, kb = cible[:, 0].long(), cible[:, 1].long()
    iy, ix = cible[:, 4].long(), cible[:, 5].long()
    wa, wb = cible[:, 2], cible[:, 3]
    pv = cible[:, 9]
    p10 = []
    for c in (0, 1):
        sd, mu = norm.y_sd[c], norm.y_mu[c]
        va = pred[b, c, ka, iy, ix].float() * sd + mu
        vb = pred[b, c, kb, iy, ix].float() * sd + mu
        p10.append(wa * va + wb * vb)
    vit_p = torch.sqrt(p10[0] ** 2 + p10[1] ** 2 + 1e-6)
    vit_o = torch.sqrt(cible[:, 6] ** 2 + cible[:, 7] ** 2).clamp_min(1e-6)
    err = 0.0
    for c in (0, 1):
        o = cible[:, 6 + c]
        visee = pv * o + (1.0 - pv) * vit_p * o / vit_o
        err = err + ((p10[c] - visee) / norm.y_sd[c]) ** 2
    return (ok * err).sum() / (2.0 * ok.sum())
