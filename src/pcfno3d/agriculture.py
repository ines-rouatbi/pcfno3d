"""Agriculture (cahier v12, OS8 ; feuille de route nationale, phase 6) : vent de 2 a 10 m au-dessus du
sol, fenetres de traitement et d'aspersion J0 a J+2, besoin en eau FAO-56, sur n'importe quelle grille
du modele (Grand Tunis 100 m, national 1 km).

Les niveaux du modele sont des altitudes au-dessus de la MER : le vent a 2 et 10 m au-dessus du sol
vient de `atom_wind.champ_agl` (interpolation entre noeuds, profil log depuis le noeud valide le plus
bas avec le z0 de la maille), comme le front-end et scripts/dso_hauteur_sol.py.

Regle de decision FIGEE (maquette ATOM Wind, ACTS.spray / ACTS.irr ; scripts/verifier_decisions.py,
service/atom_outils.decision_agricole) — vent a 10 m, en km/h, jamais ajustee au resultat :

    traitement   < 3 attention (air trop calme) ; <= 15 oui ; <= 19 attention (derive) ; > 19 non
    aspersion    <= 15 oui ; <= 25 attention ; > 25 non

tests/test_agri_eolien_national.py verifie qu'elle rend exactement les decisions de
scripts/verifier_decisions.py, de service/atom_outils.py et de atom_wind.traitement_possible.

FAO-56 (Allen et al. 1998, Penman-Monteith, pas journalier) : la fonction `et0_journalier` est le
portage de `et0Day` de la page (memes equations, memes constantes) ; u2 = 0,748 x vent a 10 m
(FAO-56 eq. 47 a z = 10 m), temperature, humidite et rayonnement de GFS, comme le Grand Tunis
(rayonnement absent : repli de la page, Rs = 0,72 Ra, signale).
"""
from __future__ import annotations

import math

import numpy as np

KMH = 3.6
CLASSES = ("oui", "attention", "non")
CODES = {"oui": 0, "attention": 1, "non": 2}           # codage int8 des cartes ; -1 = sans valeur
SANS_VALEUR = -1
U2_DEPUIS_U10 = 4.87 / math.log(67.8 * 10.0 - 5.42)    # FAO-56 eq. 47 a 10 m : 0,748
HAUTEURS_AGRI = (2.0, 10.0)                            # m au-dessus du sol : rampe, regle (10 m)
REPLI_RS_RA = 0.72                                     # la page, sans rayonnement GFS


# ---------------------------------------------------------------------- regle figee
def _kmh(vitesse_m_s) -> np.ndarray:
    return np.round(np.asarray(vitesse_m_s, float) * KMH, 6)      # 15 km/h reste 15


def decision_traitement(vitesse_m_s) -> np.ndarray:
    k = _kmh(vitesse_m_s)
    return np.select([k < 3, k <= 15, k <= 19], ["attention", "oui", "attention"], "non")


def decision_aspersion(vitesse_m_s) -> np.ndarray:
    k = _kmh(vitesse_m_s)
    return np.select([k <= 15, k <= 25], ["oui", "attention"], "non")


REGLES = {"traitement": decision_traitement, "aspersion": decision_aspersion}
TEXTE_REGLE = ("traitement : <3 attention, <=15 oui, <=19 attention, >19 non (km/h, 10 m) ; "
               "aspersion : <=15 oui, <=25 attention, >25 non")


def coder(decisions: np.ndarray, valide: np.ndarray | None = None) -> np.ndarray:
    """« oui » / « attention » / « non » -> 0 / 1 / 2 (int8) ; -1 la ou `valide` est faux."""
    d = np.asarray(decisions)
    out = np.select([d == "oui", d == "attention", d == "non"], [0, 1, 2], SANS_VALEUR).astype(np.int8)
    if valide is not None:
        out = np.where(valide, out, np.int8(SANS_VALEUR)).astype(np.int8)
    return out


def creneaux(oui_par_heure, heures=None) -> list[tuple[int, int]]:
    """Plages [debut, fin) d'heures consecutives a « oui » (fin exclue), comme
    service/atom_outils.decision_agricole et windowsOf de la page."""
    oui = [bool(x) for x in oui_par_heure]
    heures = list(range(len(oui))) if heures is None else [int(h) for h in heures]
    out, debut = [], None
    for k, (h, o) in enumerate(zip(heures, oui)):
        suivante = heures[k + 1] if k + 1 < len(heures) else None
        if o and debut is None:
            debut = h
        if debut is not None and (not o or suivante is None or suivante != h + 1):
            out.append((debut, h + 1 if o else h))
            debut = None
    return out


def texte_creneaux(plages: list[tuple[int, int]]) -> str:
    return ", ".join(f"{a} h - {b} h" for a, b in plages) if plages else "aucun creneau"


# ---------------------------------------------------------------------- vent pres du sol
def vent_pres_du_sol(volume, geom, hauteurs=HAUTEURS_AGRI) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """{h: (u, v)} (ny, nx) a h m au-dessus du sol (champ_agl ; NaN sans colonne)."""
    from .atom_wind import champ_agl
    return {float(h): champ_agl(volume.u, volume.v, geom, float(h)) for h in hauteurs}


def interpoler_heures(temps_h: np.ndarray, champs: np.ndarray, heures: np.ndarray) -> np.ndarray:
    """Interpolation LINEAIRE en temps (comme la page entre deux sorties du modele) des champs
    (n_instants, ...) aux `heures` ; NaN hors de [premier, dernier] instant (pas d'extrapolation)."""
    t = np.asarray(temps_h, float)
    o = np.argsort(t)
    t, c = t[o], np.asarray(champs, float)[o]
    out = np.full((len(heures),) + c.shape[1:], np.nan)
    for i, h in enumerate(np.asarray(heures, float)):
        if len(t) == 0 or h < t[0] - 1e-9 or h > t[-1] + 1e-9:
            continue
        j = int(np.clip(np.searchsorted(t, h) - 1, 0, max(len(t) - 2, 0)))
        if len(t) == 1 or abs(t[j] - h) < 1e-9:
            out[i] = c[j]
            continue
        if abs(t[j + 1] - h) < 1e-9:
            out[i] = c[j + 1]
            continue
        a = (h - t[j]) / (t[j + 1] - t[j])
        out[i] = (1 - a) * c[j] + a * c[j + 1]
    return out


# ---------------------------------------------------------------------- FAO-56
def esat(t_c):
    t_c = np.asarray(t_c, float)
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def rayonnement_extraterrestre(lat_deg, jour_annee) -> np.ndarray:
    """Ra (MJ/m2/j), FAO-56 eq. 21 a 25 (comme la page)."""
    phi = np.radians(np.asarray(lat_deg, float))
    j = np.asarray(jour_annee, float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    dl = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dl), -1.0, 1.0))
    return 24 * 60 / np.pi * 0.082 * dr * (ws * np.sin(phi) * np.sin(dl) + np.cos(phi) * np.cos(dl) * np.sin(ws))


def et0_journalier(tmax, tmin, hr_moy, u2, z_m, lat_deg, jour_annee, rs_mj=None) -> dict:
    """ET0 (mm/j) FAO-56 Penman-Monteith, pas journalier, VECTORISE (n'importe quelle forme) :
    portage exact de et0Day (docs/atom_wind/atom_wind_maquette.html) — ea = es x HR moyenne / 100,
    G = 0, albedo 0,23, Rso = (0,75 + 2e-5 z) Ra. rs_mj None : repli de la page Rs = 0,72 Ra.
    Seul ajout : Rs / Rso borne a 1 (FAO-56, eq. 39), sans effet sur le repli de la page
    (0,72 / 0,75 < 1) ; il compte quand le rayonnement GFS depasse le ciel clair."""
    tmax, tmin = np.asarray(tmax, float), np.asarray(tmin, float)
    t = (tmax + tmin) / 2
    z = np.asarray(z_m, float)
    d = 4098 * esat(t) / (t + 237.3) ** 2
    p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26
    g = 0.000665 * p
    es = (esat(tmax) + esat(tmin)) / 2
    ea = es * np.asarray(hr_moy, float) / 100
    ra = rayonnement_extraterrestre(lat_deg, jour_annee)
    repli = rs_mj is None
    rs = REPLI_RS_RA * ra if repli else np.asarray(rs_mj, float)
    rso = (0.75 + 2e-5 * z) * ra
    with np.errstate(invalid="ignore", divide="ignore"):
        rnl = (4.903e-9 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
               * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0))) * (1.35 * np.minimum(rs / rso, 1.0) - 0.35))
    rn = 0.77 * rs - rnl
    u2 = np.asarray(u2, float)
    et0 = (0.408 * d * rn + g * 900 / (t + 273) * u2 * (es - ea)) / (d + g * (1 + 0.34 * u2))
    return {"et0_mm": et0, "rs_mj": rs, "ra_mj": ra, "rayonnement_repli": repli}


def u2_fao(vitesse_10m) -> np.ndarray:
    """Vent a 2 m pour FAO-56 depuis le vent a 10 m (eq. 47, gazon de reference), comme la page."""
    return np.asarray(vitesse_10m, float) * U2_DEPUIS_U10


# ---------------------------------------------------------------------- GFS sur la grille du modele
def latlon_noeuds(geom) -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) (ny, nx) des noeuds : la projection de pcfno3d.forcage.InterpERA5 / sig.grille_geo."""
    from .sig import grille_geo
    g = grille_geo(geom)
    jj, ii = np.meshgrid(np.arange(geom.ny), np.arange(geom.nx), indexing="ij")
    return g["lat0"] + jj * g["dlat"], g["lon0"] + ii * g["dlon"]


def bilineaire(champ: np.ndarray, lat_g: np.ndarray, lon_g: np.ndarray, lat: np.ndarray,
               lon: np.ndarray) -> np.ndarray:
    """Champ (..., n_lat, n_lon) d'une grille reguliere croissante (GFS 0,25 deg) -> points (lat, lon),
    bilineaire, borne au bord de la grille."""
    lat_g, lon_g = np.asarray(lat_g, float), np.asarray(lon_g, float)
    fy = np.clip((lat - lat_g[0]) / (lat_g[1] - lat_g[0]), 0, len(lat_g) - 1.000001)
    fx = np.clip((lon - lon_g[0]) / (lon_g[1] - lon_g[0]), 0, len(lon_g) - 1.000001)
    iy, ix = np.floor(fy).astype(int), np.floor(fx).astype(int)
    a, b = fy - iy, fx - ix
    c = np.asarray(champ, float)
    return ((1 - a) * (1 - b) * c[..., iy, ix] + (1 - a) * b * c[..., iy, ix + 1]
            + a * (1 - b) * c[..., iy + 1, ix] + a * b * c[..., iy + 1, ix + 1])


class MeteoGFS:
    """Temperature et humidite a 2 m, rayonnement solaire descendant : cube GFS national
    (pcfno3d.gfs.assembler_national, `gfs_tunisie_f<L>.npz` : temps en s UTC, lat, lon, champs
    (temps, lat, lon)). La temperature est ramenee au relief de la grille (gradient standard
    0,0065 K/m depuis l'altitude GFS), comme l'entree temps reel du Grand Tunis."""

    def __init__(self, chemin, geom) -> None:
        with np.load(chemin, allow_pickle=False) as d:
            self.temps = np.asarray(d["temps"], np.int64)
            self.lat_g, self.lon_g = np.asarray(d["lat"], float), np.asarray(d["lon"], float)
            self.champs = {k: np.asarray(d[k], np.float32) for k in
                           ("temperature_2m", "relative_humidity_2m", "shortwave_radiation") if k in d.files}
            self.relief_g = np.asarray(d["elevation"], float) if "elevation" in d.files else None
            self.echeance = sorted(set(int(x) for x in np.atleast_1d(d["echeance_h"]))) if "echeance_h" in d.files else []
        self.lat, self.lon = latlon_noeuds(geom)
        self.zt = np.asarray(geom.zt, float)
        self.source = f"GFS 0,25 deg (cube national {getattr(chemin, 'name', chemin)})"

    def a(self, t_s: int) -> dict | None:
        """{t2m (deg C), hr2m (%), ray (W/m2)} sur la grille a l'instant t (s UTC) ; None si absent."""
        i = np.nonzero(self.temps == int(t_s))[0]
        if not len(i):
            return None
        i = int(i[0])
        out = {}
        if "temperature_2m" in self.champs:
            t = bilineaire(self.champs["temperature_2m"][i], self.lat_g, self.lon_g, self.lat, self.lon)
            if self.relief_g is not None:
                zg = bilineaire(self.relief_g, self.lat_g, self.lon_g, self.lat, self.lon)
                t = t - 0.0065 * (self.zt - zg)
            out["t2m"] = t
        if "relative_humidity_2m" in self.champs:
            out["hr2m"] = np.clip(bilineaire(self.champs["relative_humidity_2m"][i], self.lat_g, self.lon_g,
                                             self.lat, self.lon), 1.0, 100.0)
        if "shortwave_radiation" in self.champs:
            out["ray"] = np.maximum(bilineaire(self.champs["shortwave_radiation"][i], self.lat_g, self.lon_g,
                                               self.lat, self.lon), 0.0)
        return out


# ---------------------------------------------------------------------- journee agricole
def journee(instants: list[dict], geom, date_locale: str, utc_local_h: int = 1) -> dict:
    """Produit agricole d'une journee locale (Tunisie : UTC+1) sur la grille du modele.

    `instants` : [{"heure_locale": float, "u10", "v10", "u2", "v2" (ny, nx), "t2m", "hr2m", "ray"
    (ny, nx) ou absents}], les sorties du modele de la journee. Heures 0..23 : vent interpole
    lineairement entre instants (pas d'extrapolation : heure hors de la plage = sans valeur).
    Renvoie les decisions horaires codees (24, ny, nx), heures « oui » par maille, vent moyen a 2
    et 10 m, ET0 FAO-56 (mm/j) et la source de chaque grandeur."""
    heures = np.arange(24)
    th = np.array([float(i["heure_locale"]) for i in instants])
    u10 = interpoler_heures(th, np.stack([i["u10"] for i in instants]), heures)
    v10 = interpoler_heures(th, np.stack([i["v10"] for i in instants]), heures)
    u2 = interpoler_heures(th, np.stack([i["u2"] for i in instants]), heures)
    v2 = interpoler_heures(th, np.stack([i["v2"] for i in instants]), heures)
    s10, s2 = np.hypot(u10, v10), np.hypot(u2, v2)
    ok = np.isfinite(s10)
    out = {"date": date_locale, "heures_locales": heures.tolist(),
           "instants_modele": sorted(float(x) for x in th),
           "vitesse_10m": s10.astype(np.float32), "vitesse_2m": s2.astype(np.float32)}
    for nom, f in REGLES.items():
        d = coder(f(np.where(ok, s10, 0.0)), ok)
        out[nom] = d
        out[f"heures_oui_{nom}"] = np.where(ok.any(0), (d == CODES["oui"]).sum(0), np.nan).astype(np.float32)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["vent_moyen_10m"] = np.nanmean(s10, 0).astype(np.float32)
            out["vent_moyen_2m"] = np.nanmean(s2, 0).astype(np.float32)
    # FAO-56 : Tmax / Tmin / HR moyenne / Rs sur les instants de la journee qui les portent
    t_inst = [i["t2m"] for i in instants if i.get("t2m") is not None]
    h_inst = [i["hr2m"] for i in instants if i.get("hr2m") is not None]
    r_inst = [i["ray"] for i in instants if i.get("ray") is not None]
    if t_inst and h_inst:
        import pandas as pd
        lat, _ = latlon_noeuds(geom)
        T = np.stack(t_inst)
        rs = np.mean(np.stack(r_inst), 0) * 0.0864 if len(r_inst) == len(instants) else None   # W/m2 -> MJ/m2/j
        f = et0_journalier(T.max(0), T.min(0), np.mean(np.stack(h_inst), 0), u2_fao(out["vent_moyen_10m"]),
                           geom.zt, lat, pd.Timestamp(date_locale).dayofyear, rs)
        out["et0_mm"] = np.where(np.isfinite(out["vent_moyen_10m"]), f["et0_mm"], np.nan).astype(np.float32)
        out["rayonnement_repli"] = bool(f["rayonnement_repli"])
        out["instants_meteo"] = len(t_inst)
    else:
        out["et0_mm"] = None
        out["rayonnement_repli"] = None
        out["instants_meteo"] = 0
    return out


def creneaux_point(j: dict, iy: int, ix: int, regle: str = "traitement") -> dict:
    """Decisions horaires et creneaux « oui » d'une maille d'une journee de `journee`."""
    d = j[regle][:, iy, ix]
    noms = {v: k for k, v in CODES.items()}
    heures = [h for h in range(24) if d[h] != SANS_VALEUR]
    oui = [d[h] == CODES["oui"] for h in heures]
    return {"heures": {int(h): noms.get(int(d[h])) for h in heures},
            "creneaux": texte_creneaux(creneaux(oui, heures))}
