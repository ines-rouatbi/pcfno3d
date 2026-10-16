"""Lot M4 — alertes (critere AN1 ; besoins BF20 a BF23 ; contrat ATOM rev. 2, D8 et D11).

Un seul document par instant, conforme a `alerts.schema.json` (rev. 2) : toutes les alertes,
quel que soit le lot qui les detecte, avec un niveau, une cause et une recommandation.

    kind                    lot   detection
    hors_plage              M3    rapport du contrat ANEO (stations hors plage / densite)
    station_muette          M3    rapport du contrat ANEO (absences, valeurs figees)
    stations_insuffisantes  M3    rapport du contrat (ou part de capteurs, instants du cache)
    capteur_derive          M4    coherence spatiale (mesure vs mediane des voisins, v2) ou
                                  capteur masque compare a la reconstruction faite sans lui
    incoherence_physique    M4    residu de continuite de l'instant > seuil DSO2 (D11)
    derive_entrees          M4    statistiques des entrees vs entrainement (norm_stats.json)
    zone_mal_couverte       M2    indice de couverture < 0,2 sur une maille de 2 km
    vent_fort_moyeu         M5    entree externe (alertes_externes), calcul fait par M5
    cisaillement            M5    entree externe (alertes_externes), calcul fait par M5

Identifiant (D8) : id = sha8("kind|cible"), stable d'un instant a l'autre. Cible : capteur
(c<i>_<j>, indices du treillis a 50 m), eolienne, maille de 2 km (m<ix>_<iy>), ou
« domaine » pour une alerte qui porte sur tout l'instant (ecart documente). `raised_at` =
premiere levee de l'episode en cours, lue dans le journal JSONL (`JournalAlertes`) ;
`status` = « ouverte » ou « close ». Methode, seuils et ecarts au contrat : docs/ALERTES.md.

Strictement opt-in (`alerts.enabled`, defaut false) : les alertes ne modifient jamais la
valeur servie, et rien ne change quand elles sont desactivees.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import warnings
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .geometry import SURF_VARS, Geometry

METHOD = "alertes-m4-v2"
# seuils relisibles : v1 (innovation / influence seules) et v2 (detecteur spatial, entrees)
METHODES_SEUILS = ("alertes-m4-v1", METHOD)
KINDS = {"hors_plage": "M3", "station_muette": "M3", "stations_insuffisantes": "M3",
         "capteur_derive": "M4", "incoherence_physique": "M4", "derive_entrees": "M4",
         "zone_mal_couverte": "M2", "vent_fort_moyeu": "M5", "cisaillement": "M5"}
KINDS_EXTERNES = ("vent_fort_moyeu", "cisaillement")       # calcules par M5, pas ici
NIVEAUX = ("information", "avertissement", "important", "critique")
STATUTS = ("ouverte", "close")
# cles EXACTES d'une alerte (alerts.schema.json rev. 2, additionalProperties: false)
CHAMPS = ("id", "level", "kind", "target", "title", "cause", "recommendation", "raised_at",
          "status", "station_id", "turbine_id", "location", "value", "threshold", "unit")
RHO_SOURCES = ("atmosphere_standard", "extrapolation_surface", "verite_synthetique")
CIBLE_DOMAINE = "domaine"
# statistiques qui demandent des passes du modele sans groupe de capteurs (cote^2 passes)
STATISTIQUES_PASSES = ("influence", "innovation")
# spatiale : mesure vs mediane des voisins (aucune passe) ; combinee : spatiale + innovation
STATISTIQUES = ("influence", "innovation", "spatiale", "combinee")
MAD_NORMALE = 1.4826                 # MAD -> ecart-type pour une loi normale
RAYLEIGH_MEDIANE = 1.1774            # mediane de |d| pour d gaussien 2D d'ecart-type 1
# variables de la derive des entrees (v2) : le vent en est exclu (voir derive_entrees)
VARIABLES_ENTREES = ("temperature_c", "air_density_kg_m3", "relative_humidity_pct",
                     "pressure_pa")


# ---------------------------------------------------------------------- briques
def alert_id(kind: str, target: str) -> str:
    """sha8 de « kind|cible » (D8) : meme alerte, meme id, d'un instant a l'autre."""
    return hashlib.sha256(f"{kind}|{target}".encode("utf-8")).hexdigest()[:8]


def iso_utc(stamp) -> str:
    """Horodatage -> « AAAA-MM-JJTHH:MM:SSZ » (iso_utc du contrat). Sans fuseau = UTC.
    None leve une erreur : on n'invente pas d'instant."""
    if stamp is None:
        raise ValueError("horodatage absent : iso_utc n'invente pas d'instant")
    t = pd.Timestamp(stamp)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(x, chiffres: int = 4) -> float | None:
    """Nombre JSON a `chiffres` chiffres significatifs, None si inconnu (jamais NaN)."""
    if x is None:
        return None
    x = float(x)
    return float(f"{x:.{chiffres}g}") if np.isfinite(x) else None


def alerte(kind: str, target: str, level: str, title: str, cause: str, recommendation: str,
           *, station_id: str | None = None, turbine_id: str | None = None,
           location: tuple[float, float] | dict | None = None, value=None, threshold=None,
           unit: str | None = None) -> dict:
    """Une alerte « ouverte », aux cles exactes du contrat ; raised_at est pose par le
    journal (`JournalAlertes.mettre_a_jour`)."""
    if kind not in KINDS:
        raise ValueError(f"type d'alerte inconnu : {kind!r}")
    if level not in NIVEAUX:
        raise ValueError(f"niveau d'alerte inconnu : {level!r}")
    if isinstance(location, dict):
        location = (location["x_m"], location["y_m"])
    loc = None if location is None else {"x_m": round(float(location[0]), 1),
                                         "y_m": round(float(location[1]), 1)}
    return {"id": alert_id(kind, target), "level": level, "kind": kind, "target": str(target),
            "title": title, "cause": cause, "recommendation": recommendation,
            "raised_at": None, "status": "ouverte", "station_id": station_id,
            "turbine_id": turbine_id, "location": loc, "value": _num(value),
            "threshold": _num(threshold), "unit": unit}


def verifier_alerte(a: dict) -> None:
    """Controle les regles de alerts.schema.json (rev. 2) sans dependance ; ValueError."""
    if set(a) != set(CHAMPS):
        raise ValueError(f"cles d'alerte attendues {sorted(CHAMPS)}, recu {sorted(a)}")
    if a["kind"] not in KINDS or a["level"] not in NIVEAUX or a["status"] not in STATUTS:
        raise ValueError(f"kind / level / status invalide : {a['kind']}, {a['level']}, "
                         f"{a['status']}")
    for k in ("target", "title", "cause", "recommendation"):
        if not isinstance(a[k], str) or not a[k]:
            raise ValueError(f"alerte : « {k} » doit etre une chaine non vide")
    if a["id"] != alert_id(a["kind"], a["target"]):
        raise ValueError(f"id {a['id']} different de sha8(« {a['kind']}|{a['target']} »)")
    ra = a["raised_at"]
    if not isinstance(ra, str) or not ra.endswith("Z"):
        raise ValueError(f"raised_at doit etre un horodatage UTC en Z, recu {ra!r}")
    pd.Timestamp(ra)
    for k in ("station_id", "turbine_id", "unit"):
        if a[k] is not None and not isinstance(a[k], str):
            raise ValueError(f"alerte : « {k} » chaine ou null")
    for k in ("value", "threshold"):
        v = a[k]
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                              or not np.isfinite(v)):
            raise ValueError(f"alerte : « {k} » nombre fini ou null, recu {v!r}")
    loc = a["location"]
    if loc is not None and (not isinstance(loc, dict) or set(loc) != {"x_m", "y_m"}):
        raise ValueError("alerte : location = {x_m, y_m} ou null")


def id_capteur(ix: int, iy: int) -> str:
    """Capteur du treillis synthetique a 50 m (contrat rev. 2, B4)."""
    return f"c{int(ix)}_{int(iy)}"


def id_maille(x_m: float, y_m: float, maille_m: float = 2000.0) -> str:
    """Maille de 2 km contenant le point (x, y) : « m<ix>_<iy> »."""
    return f"m{int(np.floor(x_m / maille_m))}_{int(np.floor(y_m / maille_m))}"


def _plafonner(alertes: list[dict], n: int, cle=None) -> list[dict]:
    """Au plus n alertes d'un meme type (les plus fortes d'abord)."""
    if cle is not None:
        alertes = sorted(alertes, key=cle, reverse=True)
    return alertes[:max(int(n), 0)]


# ---------------------------------------------------------------------- M3 : contrat
def stations_hors_plage(path: str | Path, n_stations: int, node_id_start: int,
                        tol_densite_rel: float) -> list[int]:
    """point_id des stations dont au moins une ligne est hors plage (contrat 8.1) ou de
    densite incoherente (8.2), en relisant le fichier avec les REGLES du contrat
    (`PLAGES`, `_ecart_densite`) : le rapport M3 reste exactement celui du contrat."""
    from .contract import PLAGES, _ecart_densite, convertir, lire_csv

    brut, _ = lire_csv(path, True, node_id_start)
    if brut is None or brut.empty:
        return []
    dfc = convertir(brut)
    pid = dfc["point_id"]
    bad = pd.Series(False, index=dfc.index)
    for var, (lo, hi) in PLAGES.items():
        v = dfc[var]
        bad |= v.notna() & ~v.between(lo, hi)
    bad |= _ecart_densite(dfc) > tol_densite_rel
    ok = bad & pid.notna() & (pid >= 1) & (pid <= n_stations) & (pid % 1 == 0)
    return sorted(int(x) for x in pid[ok].unique())


def alertes_contrat(rapport: dict, nx_s: int, dx_s: float, dy_s: float,
                    max_par_type: int = 100, hors_plage: Iterable[int] = ()) -> list[dict]:
    """Alertes M3 depuis un rapport de `pcfno3d.contract` : station_muette,
    stations_insuffisantes, et hors_plage pour les `point_id` donnes
    (`stations_hors_plage`). Les seuils sont ceux du rapport (`rapport["seuils"]`)."""
    q = rapport.get("qualite")
    if not q:
        return []
    s = rapport.get("seuils") or {}

    def capteur(pid: int) -> tuple[str, tuple[float, float]]:
        ix, iy = (int(pid) - 1) % nx_s, (int(pid) - 1) // nx_s
        return id_capteur(ix, iy), (ix * dx_s, iy * dy_s)

    out: list[dict] = []
    for pid in list(hors_plage)[:max_par_type]:
        sid, loc = capteur(pid)
        out.append(alerte(
            "hors_plage", sid, "avertissement", f"Capteur {sid} hors plage",
            "Valeur hors de la plage physique ou densite incoherente (contrat ANEO 8.1-8.2)",
            "Verifier le capteur ; sa mesure est masquee avant la reconstruction",
            station_id=sid, location=loc))
    fr = q.get("fraicheur") or {}
    figees = set((fr.get("valeurs_figees") or {}).get("ids", []))
    for pid in (fr.get("stations_muettes") or {"ids": []})["ids"][:max_par_type]:
        sid, loc = capteur(pid)
        cause = ("Valeur figee : meme mesure sur au moins "
                 f"{s.get('n_instants_figee', 3)} instants et "
                 f"{s.get('duree_figee_s', 1800.0) / 60:.0f} min" if pid in figees
                 else "Absence de mesure sur au moins un instant du fichier")
        out.append(alerte("station_muette", sid, "avertissement", f"Capteur {sid} muet",
                          cause, "Planifier une intervention sur le capteur",
                          station_id=sid, location=loc))
    sv = q.get("stations_valides")
    if sv:
        part = sv["part_mesurees"] if s.get("absences_connues") else sv["part"]
        rejet = float(s.get("rejet_stations_valides", 0.30))
        quar = float(s.get("quarantaine_stations_valides", 0.70))
        if part is not None and part < rejet:
            out.append(_insuffisantes(part, rejet, "critique"))
        elif sv.get("part_pire_instant") is not None and sv["part_pire_instant"] < quar:
            out.append(_insuffisantes(sv["part_pire_instant"], quar, "important"))
    return out


def _insuffisantes(part: float, seuil: float, level: str) -> dict:
    return alerte("stations_insuffisantes", CIBLE_DOMAINE, level,
                  "Part de capteurs valides trop faible",
                  f"{100 * part:.1f} % de capteurs valides (seuil {100 * seuil:.0f} %)",
                  "Reconstruction degradee : verifier la transmission du reseau ; "
                  "lire l'incertitude avant toute decision" if level != "critique" else
                  "Reconstruction non fiable : ne pas l'utiliser avant le retour du reseau",
                  value=part, threshold=seuil)


def alertes_disponibilite(sensor: np.ndarray, part_min: float = 0.70,
                          part_critique: float = 0.30, reseau: str = "treillis_synthetique",
                          disponibles: int | None = None,
                          attendues: int | None = None) -> list[dict]:
    """Instants sans rapport du contrat : part de capteurs disponibles.

    treillis_synthetique : un capteur par maille du modele, part = mailles disponibles /
    mailles. Autre reseau (stations_aneo) : part = stations disponibles / stations
    ATTENDUES (declarees ou recues) ; `disponibles` absent = mailles portant une station.
    Sans nombre de stations attendues, pas d'alerte (jamais une alerte fausse : quelques
    dizaines de stations sur ~125 000 mailles ne sont pas une panne).
    """
    if reseau == "treillis_synthetique":
        part = float(np.asarray(sensor, bool).mean())
    else:
        if not attendues:
            return []
        n = int(np.asarray(sensor, bool).sum()) if disponibles is None else int(disponibles)
        part = min(n / float(attendues), 1.0)
    if part < part_critique:
        return [_insuffisantes(part, part_critique, "critique")]
    if part < part_min:
        return [_insuffisantes(part, part_min, "important")]
    return []


# ---------------------------------------------------------------------- M2 : couverture
def alertes_couverture(geom: Geometry, coverage: np.ndarray, maille_m: float = 2000.0,
                       seuil: float = 0.2, sigma_speed: np.ndarray | None = None,
                       max_par_type: int = 100) -> list[dict]:
    """zone_mal_couverte : couverture moyenne au premier noeud valide de chaque colonne,
    par maille de `maille_m`, sous `seuil`."""
    def premier_noeud(f: np.ndarray) -> np.ndarray:
        ok = geom.first_valid & np.isfinite(f)
        out = np.where(ok, f, 0.0).sum(axis=0)
        return np.where(ok.any(axis=0), out, np.nan)

    c2 = premier_noeud(np.asarray(coverage, float))
    sig2 = None if sigma_speed is None else premier_noeud(np.asarray(sigma_speed, float))
    jj, ii = np.meshgrid(np.arange(geom.ny), np.arange(geom.nx), indexing="ij")
    mx = np.floor(ii * geom.dx / maille_m).astype(int)
    my = np.floor(jj * geom.dy / maille_m).astype(int)
    ncx = int(mx.max()) + 1
    cell = (my * ncx + mx)
    ok = np.isfinite(c2)
    nb = ncx * (int(my.max()) + 1)

    def somme(w: np.ndarray) -> np.ndarray:
        return np.bincount(cell[ok], weights=w[ok], minlength=nb)

    n = np.bincount(cell[ok], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        moy = somme(c2) / n
        xc, yc = somme(ii * geom.dx) / n, somme(jj * geom.dy) / n
        sg = somme(np.nan_to_num(sig2)) / n if sig2 is not None else None
    out = []
    for c in np.nonzero((n > 0) & (moy < seuil))[0]:
        cx, cy, m = int(c % ncx), int(c // ncx), float(moy[c])
        cause = f"Indice de couverture moyen {m:.2f} < {seuil:.2f} sur la maille de " \
                f"{maille_m / 1000:.0f} km"
        if sg is not None:
            cause += f" ; sigma vitesse moyen {float(sg[c]):.2f} m/s"
        out.append(alerte(
            "zone_mal_couverte", f"m{cx}_{cy}", "information",
            f"Zone mal couverte (maille m{cx}_{cy})", cause,
            "Emplacement prioritaire pour un nouveau capteur ; lire l'intervalle a 80 %",
            location=(float(xc[c]), float(yc[c])), value=m, threshold=seuil))
    return _plafonner(out, max_par_type, cle=lambda a: -a["value"])


# ---------------------------------------------------------------------- M4 : physique (D11)
def rho_instant(geom: Geometry, source: str, S: np.ndarray | None = None,
                sensor: np.ndarray | None = None,
                Y: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Densite 3D (nz, ny, nx) pour le residu de masse, et sa source effective (D11).

    atmosphere_standard   : ISA, fonction de l'altitude seule.
    extrapolation_surface : ISA x un facteur unique = mediane(rho mesuree / rho ISA) aux
                            capteurs visibles (pas de bruit de mesure dans les gradients).
                            Sans densite mesuree : repli sur atmosphere_standard.
    verite_synthetique    : Y[3] du cache (rejeu du jeu synthetique, comme DSO2).
    """
    from .realdata import isa

    if source not in RHO_SOURCES:
        raise ValueError(f"rho_source inconnue : {source!r}")
    if source == "verite_synthetique":
        if Y is None:
            raise ValueError("rho_source = verite_synthetique : la verite (Y) est requise")
        return np.asarray(Y[3], np.float32), source
    z = np.asarray(geom.z_levels, float)
    rho = np.broadcast_to(isa(z)[2][:, None, None], geom.shape).astype(np.float32)
    if source == "extrapolation_surface" and S is not None and sensor is not None:
        m = np.asarray(sensor, bool) & np.isfinite(S[4]) & (S[4] > 0)
        if m.any():
            ratio = float(np.median(S[4][m] / isa(geom.z_sensor[m])[2]))
            if np.isfinite(ratio) and 0.5 < ratio < 1.5:
                return (rho * ratio).astype(np.float32), source
    return rho.copy(), "atmosphere_standard"


def physique(geom: Geometry, u: np.ndarray, v: np.ndarray, w: np.ndarray, rho: np.ndarray,
             rho_source: str, seuil_masse: float) -> dict:
    """Bloc `physics` du snapshot (contrat rev. 2, D11), cles exactes.

    mass_residual_rms / _p95 : RMS et 95e centile de |r| sur `mass_mask`, r = residu de
    continuite (meme operateur que DSO2, `Geometry.mass_residual`) ; mass_residual_ok :
    rms <= seuil DSO2 (cet instant ; DSO2 lui-meme est un agregat). ground_condition_rms_ms :
    RMS de w - (u dzs/dx + v dzs/dy) au premier noeud valide, relief lisse.
    """
    import torch

    dev = geom.device

    def t(a):
        return torch.from_numpy(np.nan_to_num(np.asarray(a, np.float32)))[None].to(dev)

    with torch.no_grad():
        r = geom.mass_residual(t(rho), t(u), t(v), t(w))[0].cpu().numpy()
    vals = np.abs(r[geom.mass_mask].astype(np.float64))
    rms = float(np.sqrt(np.mean(vals ** 2))) if vals.size else float("nan")
    p95 = float(np.quantile(vals, 0.95)) if vals.size else float("nan")
    fv = geom.first_valid
    w_bc = np.asarray(u, float) * geom.dzt_dx[None] + np.asarray(v, float) * geom.dzt_dy[None]
    diff = (np.asarray(w, float) - w_bc)[fv]
    diff = diff[np.isfinite(diff)]
    sol = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
    return {"mass_residual_rms": _num(rms, 6), "mass_residual_p95": _num(p95, 6),
            "mass_residual_ok": bool(np.isfinite(rms) and rms <= seuil_masse),
            "ground_condition_rms_ms": _num(sol, 6), "rho_source": rho_source}


def alertes_physique(phys: dict, seuil_masse: float, facteur_critique: float = 10.0,
                     seuil_sol: float | None = None) -> list[dict]:
    """incoherence_physique : residu de masse > seuil DSO2 ; condition au sol > seuil."""
    out = []
    rms = phys.get("mass_residual_rms")
    if rms is not None and not phys.get("mass_residual_ok", True):
        level = "critique" if rms >= facteur_critique * seuil_masse else "important"
        out.append(alerte(
            "incoherence_physique", f"{CIBLE_DOMAINE}:masse", level,
            "Conservation de la masse non respectee",
            f"Residu de continuite RMS {rms:.2e} > seuil DSO2 {seuil_masse:.0e} "
            f"(rho : {phys.get('rho_source')})",
            "Lire ce champ avec prudence (w et divergence) ; verifier les entrees et la "
            "version du modele", value=rms, threshold=seuil_masse, unit="kg/m3/s"))
    sol = phys.get("ground_condition_rms_ms")
    if seuil_sol is not None and sol is not None and sol > seuil_sol:
        out.append(alerte(
            "incoherence_physique", f"{CIBLE_DOMAINE}:sol", "avertissement",
            "Condition cinematique au sol non respectee",
            f"RMS de w - u.grad(zs) au premier noeud {sol:.3f} m/s > {seuil_sol:.3f} m/s",
            "Vent vertical pres du sol peu fiable sur cet instant",
            value=sol, threshold=seuil_sol, unit="m/s"))
    return out


# ---------------------------------------------------------------------- M4 : entrees
def derive_entrees(S: np.ndarray, sensor: np.ndarray, surf_mu, surf_sd,
                   z_max: float = 5.0, ecart_extreme: float = 5.0,
                   part_extreme: float = 0.25,
                   variables: Iterable[str] = VARIABLES_ENTREES) -> list[dict]:
    """Par variable de surface : z de la moyenne de l'instant et part de valeurs extremes
    (|z| > ecart_extreme), par rapport aux statistiques d'ENTRAINEMENT (norm_stats.json).

    Toutes les variables sont decrites ; seules celles de `variables` peuvent etre en
    derive (`evaluee`). v2 : le vent (u, v, w) n'en fait plus partie. Son z moyen et sa
    part de valeurs extremes suivent la meteo du jour (jour de vent fort ; w de surface a
    queue lourde en relief, 1,9 % de valeurs a plus de 5 sd un jour d'entrainement pour un
    seuil de 2 %) : c'est la source des fausses alertes de la v1 (docs/ALERTES.md). Une
    erreur d'unite du vent n'etait de toute facon pas detectable ainsi.
    """
    sensor = np.asarray(sensor, bool)
    variables = set(variables)
    out = []
    for c, var in enumerate(SURF_VARS):
        mu, sd = float(np.ravel(surf_mu)[c]), max(float(np.ravel(surf_sd)[c]), 1e-12)
        x = np.asarray(S[c], float)[sensor]
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        z = (x - mu) / sd
        zm = float(z.mean())
        pe = float((np.abs(z) > ecart_extreme).mean())
        ev = var in variables
        out.append({"variable": var, "z_moyen": zm, "part_extreme": pe, "n": int(x.size),
                    "evaluee": ev,
                    "derive": bool(ev and (abs(zm) >= z_max or pe >= part_extreme))})
    return out


def regle_entrees(stats: list[dict], z_max: float, part_extreme: float,
                  variables: Iterable[str] = VARIABLES_ENTREES) -> list[str]:
    """Variables en derive pour une regle donnee, depuis les statistiques brutes de
    `derive_entrees` (reglage sur la validation sans recalcul)."""
    variables = set(variables)
    return [s["variable"] for s in stats if s["variable"] in variables
            and (abs(s["z_moyen"]) >= z_max or s["part_extreme"] >= part_extreme)]


def alertes_entrees(stats: list[dict], z_max: float = 5.0, part_extreme: float = 0.25,
                    ecart_extreme: float = 5.0) -> list[dict]:
    out = []
    for s in stats:
        if not s["derive"]:
            continue
        fort = abs(s["z_moyen"]) >= 2 * z_max or s["part_extreme"] >= 0.9
        out.append(alerte(
            "derive_entrees", f"{CIBLE_DOMAINE}:{s['variable']}",
            "important" if fort else "avertissement",
            f"Entree {s['variable']} hors du domaine d'entrainement",
            f"Moyenne a {s['z_moyen']:+.1f} ecarts-types de l'entrainement ; "
            f"{100 * s['part_extreme']:.1f} % des capteurs a plus de {ecart_extreme:g} "
            "ecarts-types",
            "Verifier les unites et l'etalonnage de cette variable ; la reconstruction "
            "extrapole hors de ce que le modele a appris",
            value=s["z_moyen"], threshold=z_max, unit="ecart-type"))
    return out


# ---------------------------------------------------------------------- M4 : capteurs
def groupes(ny: int, nx: int, cote: int) -> np.ndarray:
    """(ny, nx) -> numero de groupe en treillis : (j mod c) c + (i mod c). Masquer un groupe
    retire 1/c^2 des capteurs, tous distants d'au moins c mailles : les voisins restent."""
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return (j % cote) * cote + (i % cote)


def interp_colonne(f: np.ndarray, z: np.ndarray, geom: Geometry) -> np.ndarray:
    """Champ (nz, ny, nx) a l'altitude z (ny, nx) ASL, lineaire entre niveaux VALIDES ; sous
    le premier noeud valide de la colonne, la valeur de ce noeud (pas d'extrapolation)."""
    zl, valid = np.asarray(geom.z_levels, float), geom.valid
    nz = len(zl)
    kf = np.argmax(valid, axis=0)
    ze = np.clip(np.maximum(np.asarray(z, float), zl[kf]), zl[0], zl[-1])
    k = np.clip(np.searchsorted(zl, ze, side="right") - 1, 0, nz - 2)
    a = np.clip((ze - zl[k]) / (zl[k + 1] - zl[k]), 0.0, 1.0)
    f0 = np.take_along_axis(f, k[None], 0)[0].astype(np.float64)
    f1 = np.take_along_axis(f, (k + 1)[None], 0)[0].astype(np.float64)
    with np.errstate(invalid="ignore"):
        out = np.where(a > 0, (1 - a) * f0 + a * np.nan_to_num(f1), f0)
    out[~valid.any(axis=0)] = np.nan
    return out


@dataclass
class Ecarts:
    """Ecarts d'un instant, par maille-capteur (NaN ou capteur non evalue)."""

    d: np.ndarray                 # (2, ny, nx) ecart brut (u, v), m/s
    a: np.ndarray                 # (ny, nx) |ecart centre|, m/s
    z: np.ndarray                 # (ny, nx) |ecart centre| / echelle
    echelle: float                # m/s : 1,4826 MAD des composantes (plancher)
    centre: tuple[float, float]
    echelle_locale: np.ndarray | None = None   # (ny, nx) m/s, detecteur spatial seulement


def normaliser(d: np.ndarray, plancher: float) -> Ecarts:
    """Centre par la mediane de l'instant (une derive commune n'est pas un capteur) et met a
    l'echelle par 1,4826 MAD : seuil en ecarts robustes, quel que soit le vent du jour."""
    ok = np.isfinite(d[0]) & np.isfinite(d[1])
    nan = np.full(d.shape[1:], np.nan)
    if not ok.any():
        return Ecarts(d, nan, nan.copy(), float("nan"), (float("nan"), float("nan")))
    med = np.median(d[:, ok], axis=1)
    dev = d - med[:, None, None]
    s = max(MAD_NORMALE * float(np.median(np.abs(dev[:, ok]))), plancher)
    a = np.hypot(dev[0], dev[1])
    return Ecarts(d, a, a / s, s, (float(med[0]), float(med[1])))


def ecarts_capteurs(S: np.ndarray, sensor: np.ndarray, geom: Geometry,
                    served_uv: tuple[np.ndarray, np.ndarray],
                    membre: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
                    cote: int = 3, statistiques: Iterable[str] = ("influence",),
                    plancher: float = 0.05) -> dict[str, Ecarts]:
    """Chaque maille-capteur comparee a la reconstruction faite SANS elle (groupes en
    treillis : cote^2 passes au lieu d'une par capteur), a l'altitude du capteur.

        innovation : d = mesure - reconstruction sans le capteur
        influence  : d = reconstruction servie (avec) - reconstruction sans le capteur

    L'influence annule l'ecart de representativite (capteur a ~25 m du sol, premier noeud
    valide parfois plus haut) qui rend l'innovation systematique en relief ; une derive du
    capteur passe dans la reconstruction servie, donc dans l'influence.
    """
    statistiques = tuple(statistiques)
    for st in statistiques:
        if st not in STATISTIQUES_PASSES:
            raise ValueError(f"statistique de derive (passes) inconnue : {st!r}")
    sensor = np.asarray(sensor, bool)
    zc = geom.z_sensor
    gid = groupes(geom.ny, geom.nx, cote)
    ref = {"innovation": (np.asarray(S[0], float), np.asarray(S[1], float))}
    if "influence" in statistiques:
        ref["influence"] = (interp_colonne(served_uv[0], zc, geom),
                            interp_colonne(served_uv[1], zc, geom))
    d = {st: np.full((2, geom.ny, geom.nx), np.nan) for st in statistiques}
    for g in range(cote * cote):
        grp = sensor & (gid == g)
        kept = sensor & ~grp
        if not grp.any() or not kept.any():
            continue
        u_g, v_g, _ = membre(kept)
        ru, rv = interp_colonne(u_g, zc, geom), interp_colonne(v_g, zc, geom)
        for st in statistiques:
            d[st][0][grp] = (ref[st][0] - ru)[grp]
            d[st][1][grp] = (ref[st][1] - rv)[grp]
    return {st: normaliser(d[st], plancher) for st in statistiques}


def voisins(f: np.ndarray, rayon: int) -> np.ndarray:
    """(ny, nx) -> (n, ny, nx) : les n = (2 rayon + 1)^2 - 1 voisines de chaque maille
    (centre exclu), NaN hors du domaine."""
    r = int(rayon)
    ny, nx = f.shape
    p = np.pad(np.asarray(f, float), r, constant_values=np.nan)
    return np.stack([p[r + dj:r + dj + ny, r + di:r + di + nx]
                     for dj in range(-r, r + 1) for di in range(-r, r + 1) if dj or di])


def mediane_pile(pile: np.ndarray) -> np.ndarray:
    """Mediane le long de l'axe 0 en ignorant les NaN (tri : les NaN vont a la fin) ; NaN
    si aucune valeur. Plus rapide que np.nanmedian sur ~125 000 mailles."""
    t = np.sort(pile, axis=0)
    n = np.isfinite(t).sum(axis=0)
    lo = np.take_along_axis(t, np.maximum((n - 1) // 2, 0)[None], 0)[0]
    hi = np.take_along_axis(t, np.maximum(n // 2, 0)[None], 0)[0]
    return np.where(n > 0, 0.5 * (lo + hi), np.nan)


SITES_METHODE = "sites-lineaires-v1"
N_CARAC_SITE = 5                     # 1, u voisins, v voisins, sin(heure), cos(heure)


def heure_utc(instant) -> float:
    """Heure UTC decimale d'un instant (« ...Z » ou horodatage sans fuseau = UTC)."""
    t = pd.Timestamp(iso_utc(instant))
    return t.hour + t.minute / 60.0 + t.second / 3600.0


def caracteristiques_site(med_u: np.ndarray, med_v: np.ndarray, heure: float) -> np.ndarray:
    """(5, ny, nx) : 1, vent des voisins (u, v), cycle diurne (sin, cos de l'heure UTC)."""
    un = np.ones_like(med_u, dtype=float)
    w = 2 * np.pi * float(heure) / 24.0
    return np.stack([un, med_u, med_v, un * np.sin(w), un * np.cos(w)])


class ModeleSites:
    """Correction de site, apprise sur les jours d'ENTRAINEMENT seulement (v2).

    L'ecart naturel d'un capteur a la mediane de ses voisins n'est pas un bruit : il vient
    de son site (relief, rugosite, vents de pente qui s'inversent entre le jour et la nuit)
    et se repete d'un jour a l'autre. Par maille, une regression lineaire (ridge) de cet
    ecart (u, v) sur [1, u voisins, v voisins, sin h, cos h] l'apprend ; le detecteur
    compare ensuite la mesure a « mediane des voisins + ecart de site attendu ». Le
    modele fait partie du reglage (comme norm_stats.json) : il est ajuste sans les jours
    de validation ni de test, et `charger` refuse tout autre jeu.
    """

    def __init__(self, coef: np.ndarray, rayon: int, info: dict | None = None) -> None:
        self.coef = np.asarray(coef, np.float32)          # (ny, nx, 5, 2)
        self.rayon = int(rayon)
        self.info = dict(info or {})

    def predire(self, med_u: np.ndarray, med_v: np.ndarray, heure: float) -> np.ndarray:
        X = np.nan_to_num(caracteristiques_site(med_u, med_v, heure))
        return (X[:, None] * np.moveaxis(self.coef, (2, 3), (0, 1))).sum(axis=0)

    def sauver(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({**self.info, "methode": SITES_METHODE, "rayon": self.rayon,
                           "jeu": "entrainement"}, ensure_ascii=False)
        with open(path, "wb") as f:
            np.savez_compressed(f, coef=self.coef, meta=np.array(meta))
        return path

    @classmethod
    def charger(cls, path: str | Path, forme: tuple[int, int] | None = None) -> "ModeleSites":
        with np.load(Path(path)) as z:
            coef, meta = z["coef"], json.loads(str(z["meta"]))
        if meta.get("jeu") != "entrainement" or meta.get("methode") != SITES_METHODE:
            raise ValueError("modele de sites : ajuste sur l'entrainement seulement "
                             f"({SITES_METHODE}), recu {meta.get('jeu')!r} / "
                             f"{meta.get('methode')!r}")
        if forme is not None and tuple(coef.shape[:2]) != tuple(forme):
            raise ValueError(f"modele de sites {coef.shape[:2]} != grille {tuple(forme)}")
        return cls(coef, meta["rayon"], meta)


def ajuster_sites(instants: Iterable[tuple[np.ndarray, float]], rayon: int,
                  ridge: float = 1.0, min_voisins: int = 3, info: dict | None = None
                  ) -> ModeleSites:
    """Moindres carres par maille (ridge `ridge`) sur des instants (S, heure UTC) des jours
    d'ENTRAINEMENT : ecart a la mediane des voisins ~ caracteristiques_site. Une passe,
    sommes accumulees (memoire independante du nombre d'instants)."""
    XtX = XtY = None
    n = 0
    for S, heure in instants:
        sensor = np.asarray(S[7], float) > 0.5
        r, med = ecart_voisins(S, sensor, rayon, min_voisins)
        X = caracteristiques_site(med[0], med[1], heure)
        ok = np.isfinite(r).all(axis=0) & np.isfinite(X).all(axis=0)
        X = np.where(ok, X, 0.0)
        r = np.where(ok, r, 0.0)
        a = X[:, None] * X[None]                            # (5, 5, ny, nx)
        b = X[:, None] * r[None]                            # (5, 2, ny, nx)
        XtX, XtY = (a, b) if XtX is None else (XtX + a, XtY + b)
        n += 1
    if XtX is None:
        raise ValueError("modele de sites : aucun instant d'entrainement")
    import torch

    # ~125 000 systemes 5 x 5 : torch les resout en lot ~50 fois plus vite que numpy
    A = np.ascontiguousarray(np.moveaxis(XtX, (0, 1), (2, 3)) + ridge * np.eye(N_CARAC_SITE))
    B = np.ascontiguousarray(np.moveaxis(XtY, (0, 1), (2, 3)))
    coef = torch.linalg.solve(torch.from_numpy(A), torch.from_numpy(B)).numpy()  # (ny,nx,5,2)
    return ModeleSites(coef, rayon, {**(info or {}), "instants": n, "ridge": ridge})


def ecart_voisins(S: np.ndarray, sensor: np.ndarray, rayon: int, min_voisins: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) mesure - mediane des voisins, et cette mediane ; NaN si non evaluable."""
    d = np.full((2,) + sensor.shape, np.nan)
    med = np.full((2,) + sensor.shape, np.nan)
    for c in (0, 1):
        f = np.where(sensor & np.isfinite(S[c]), np.asarray(S[c], float), np.nan)
        pile = voisins(f, rayon)
        n = np.isfinite(pile).sum(axis=0)
        m = mediane_pile(pile)
        ok = np.isfinite(f) & (n >= int(min_voisins))
        med[c] = np.where(n >= int(min_voisins), m, np.nan)
        d[c] = np.where(ok, f - m, np.nan)
    return d, med


def ecarts_spatiaux(S: np.ndarray, sensor: np.ndarray, rayon: int = 2,
                    rayon_echelle: int = 2, plancher: float = 0.05,
                    min_voisins: int = 3, sites: ModeleSites | None = None,
                    heure: float | None = None,
                    base: tuple[np.ndarray, np.ndarray] | None = None) -> Ecarts:
    """Coherence spatiale (v2) : chaque maille-capteur comparee a ses voisines, SANS passe
    du modele.

        d = (u, v) mesure - mediane des (u, v) des capteurs voisins dans un carre de
            (2 rayon + 1) mailles, centre exclu (au moins `min_voisins`)
            [- ecart de site attendu, si un `ModeleSites` appris sur l'entrainement]
        s = mediane des |d| des voisines (carre de rayon `rayon_echelle`) / 1,1774,
            plancher `plancher` : ecart-type local par composante (|d| d'un bruit
            gaussien 2D d'ecart-type 1 a pour mediane 1,1774)
        z = |d| / s

    La mediane resiste a un voisin en derive ; l'echelle locale suit le relief (les
    mailles de crete ou de vallee s'ecartent naturellement plus de leurs voisines).
    `base` : (ecart, mediane) deja calcules par `ecart_voisins` pour ce rayon (simulation
    AN1 : une seule mediane pour plusieurs echelles et corrections).
    """
    sensor = np.asarray(sensor, bool)
    d, med = base if base is not None else ecart_voisins(S, sensor, rayon, min_voisins)
    if sites is not None:
        if sites.rayon != int(rayon):
            raise ValueError(f"modele de sites de rayon {sites.rayon}, detecteur {rayon}")
        if heure is None:
            raise ValueError("correction de site : l'heure de l'instant est requise")
        d = d - sites.predire(med[0], med[1], heure)
    a = np.hypot(d[0], d[1])
    ok = np.isfinite(a)
    nan = np.full(sensor.shape, np.nan)
    if not ok.any():
        return Ecarts(d, nan, nan.copy(), float("nan"), (0.0, 0.0), nan.copy())
    s = mediane_pile(voisins(np.where(ok, a, np.nan), rayon_echelle)) / RAYLEIGH_MEDIANE
    globale = max(float(np.median(a[ok])) / RAYLEIGH_MEDIANE, plancher)
    s = np.maximum(np.where(np.isfinite(s), s, globale), plancher)
    z = np.where(ok, a / s, np.nan)
    return Ecarts(d, np.where(ok, a, np.nan), z, float(np.median(s[ok])), (0.0, 0.0),
                  np.where(ok, s, np.nan))


def drapeaux_derive(st: dict, kappa: float, min_m_s: float, min_instants: int
                    ) -> np.ndarray:
    """Maille signalee : au moins `min_instants` observations, z >= kappa, ecart >= min_m_s."""
    with np.errstate(invalid="ignore"):
        return (st["n"] >= min_instants) & (st["z"] >= kappa) & (st["a"] >= min_m_s)


class SuiviDerive:
    """Persistance : les `fenetre` derniers instants, par maille-capteur.

    mode « moyenne » (v1 : innovation, influence) : z = moyenne sur la fenetre de
    |ecart| / echelle, a = moyenne de |ecart|.
    mode « vecteur » (v2 : spatiale) : cumul vectoriel, a = |moyenne des ecarts (u, v)|,
    z = a / moyenne de l'echelle locale. Un biais, un gain ou une girouette decalent la
    mesure dans une direction qui persiste ; le bruit, lui, se compense en partie.

    Alerte si au moins `min_instants` observations, z >= kappa ET a >= min_m_s. En memoire
    seulement (un redemarrage repart d'une fenetre vide).
    """

    MODES = ("moyenne", "vecteur")

    def __init__(self, fenetre: int, mode: str = "moyenne") -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode de persistance inconnu : {mode!r}")
        self.fenetre = int(fenetre)
        self.mode = mode
        self.z: deque = deque(maxlen=self.fenetre)
        self.a: deque = deque(maxlen=self.fenetre)
        self.s: deque = deque(maxlen=self.fenetre)
        self.d: deque = deque(maxlen=self.fenetre)
        self.sl: deque = deque(maxlen=self.fenetre)
        self.instants: deque = deque(maxlen=self.fenetre)

    @property
    def dernier(self) -> str | None:
        return self.instants[-1] if self.instants else None

    def ajouter(self, instant: str, ec: Ecarts) -> None:
        self.instants.append(instant)
        self.z.append(np.asarray(ec.z, np.float32))
        self.s.append(float(ec.echelle))
        if self.mode == "vecteur":
            if ec.echelle_locale is None:
                raise ValueError("persistance vectorielle : echelle locale requise")
            self.d.append(np.asarray(ec.d, np.float32))
            self.sl.append(np.asarray(ec.echelle_locale, np.float32))
        else:
            self.a.append(np.asarray(ec.a, np.float32))

    def statistiques(self) -> dict:
        s = np.asarray(self.s, float)
        s = s[np.isfinite(s)]
        echelle = float(s.mean()) if s.size else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            if self.mode == "vecteur":
                D, L = np.stack(self.d), np.stack(self.sl)
                ok = np.isfinite(D[:, 0]) & np.isfinite(D[:, 1]) & np.isfinite(L)
                n = ok.sum(axis=0)
                m = np.maximum(n, 1)
                du = np.where(ok, D[:, 0], 0).sum(0) / m
                dv = np.where(ok, D[:, 1], 0).sum(0) / m
                sm = np.where(n > 0, np.where(ok, L, 0).sum(0) / m, np.nan)
                ab = np.where(n > 0, np.hypot(du, dv), np.nan)
                return {"z": ab / sm, "a": ab, "n": n, "echelle": echelle,
                        "echelle_locale": sm}
            Z, A = np.stack(self.z), np.stack(self.a)
            ok = np.isfinite(Z)
            n = ok.sum(axis=0)
            zb = np.where(n > 0, np.where(ok, Z, 0).sum(0) / np.maximum(n, 1), np.nan)
            ab = np.where(n > 0, np.where(ok, A, 0).sum(0) / np.maximum(n, 1), np.nan)
        return {"z": zb, "a": ab, "n": n, "echelle": echelle}

    def drapeaux(self, kappa: float, min_m_s: float, min_instants: int
                 ) -> tuple[np.ndarray, dict]:
        st = self.statistiques()
        return drapeaux_derive(st, kappa, min_m_s, min_instants), st


def combiner(st_spatiale: dict, st_passes: dict) -> dict:
    """Statistique combinee : moyenne des deux z (ecarts robustes), observations communes ;
    l'ecart en m/s et l'echelle locale sont ceux du detecteur spatial."""
    with np.errstate(invalid="ignore"):
        z = 0.5 * (st_spatiale["z"] + st_passes["z"])
    return {**st_spatiale, "z": z, "n": np.minimum(st_spatiale["n"], st_passes["n"])}


def composantes(statistique: str) -> tuple[str, ...]:
    """Ecarts a suivre pour une statistique de decision (combinee : spatiale + innovation)."""
    if statistique not in STATISTIQUES:
        raise ValueError(f"statistique de derive inconnue : {statistique!r}")
    return ("spatiale", "innovation") if statistique == "combinee" else (statistique,)


def mode_suivi(composante: str) -> str:
    """Persistance vectorielle pour le detecteur spatial, moyenne de z sinon (v1)."""
    return "vecteur" if composante == "spatiale" else "moyenne"


def alertes_derive(geom: Geometry, flag: np.ndarray, st: dict, kappa: float,
                   min_m_s: float, facteur_critique: float, statistique: str,
                   max_par_type: int = 100, rayon: int | None = None) -> list[dict]:
    """capteur_derive pour chaque maille-capteur signalee. Une maille du modele regroupe
    stride x stride capteurs a 50 m : elle est designee par son capteur sud-ouest.
    Seuil publie : kappa x echelle (locale pour le detecteur spatial), plancher min_m_s."""
    vois = f"ses voisins (rayon {rayon} mailles)"
    quoi = {"influence": "la reconstruction avec et sans le capteur",
            "innovation": "la mesure et la reconstruction sans le capteur",
            "spatiale": f"la mesure et la mediane de {vois}",
            "combinee": f"la mesure et {vois}, confirme par la reconstruction sans le "
                        "capteur"}[statistique]
    loc = st.get("echelle_locale")
    out = []
    for j, i in zip(*np.nonzero(flag)):
        sid = id_capteur(i * geom.stride, j * geom.stride)
        a, z, n = float(st["a"][j, i]), float(st["z"][j, i]), int(st["n"][j, i])
        e = float(loc[j, i]) if loc is not None else float(st["echelle"])
        seuil = max(kappa * e, min_m_s) if np.isfinite(e) else min_m_s
        level = "critique" if a >= facteur_critique * seuil else "important"
        out.append(alerte(
            "capteur_derive", sid, level, f"Capteur {sid} en derive",
            f"Ecart persistant de {a:.2f} m/s entre {quoi} ({n} instants, "
            f"{z:.1f} ecarts robustes ; maille de {geom.stride}x{geom.stride} capteurs)",
            "Planifier un etalonnage (anemometre, girouette) ; lire l'intervalle a 80 % "
            "autour de ce capteur", station_id=sid,
            location=(i * geom.dx, j * geom.dy), value=a, threshold=seuil, unit="m/s"))
    return _plafonner(out, max_par_type, cle=lambda x: x["value"])


# ---------------------------------------------------------------------- M5 : externes
def alertes_externes(items: Iterable[dict], kinds: Iterable[str] = KINDS_EXTERNES
                     ) -> list[dict]:
    """Entree des alertes calculees ailleurs (M5 : vent_fort_moyeu, cisaillement).

    Chaque alerte donne kind, target, level, title, cause, recommendation et, au besoin,
    station_id, turbine_id, location, value, threshold, unit. L'id est recalcule (sha8 de
    « kind|cible ») ; raised_at et status sont poses par le journal M4.
    """
    kinds = set(kinds)
    out = []
    for it in items:
        it = dict(it)
        if it.get("kind") not in kinds:
            raise ValueError(f"alerte externe : type {it.get('kind')!r} non accepte "
                             f"(attendu : {sorted(kinds)})")
        if "id" in it and it["id"] != alert_id(it["kind"], str(it.get("target"))):
            raise ValueError(f"alerte externe : id {it['id']} != sha8(kind|cible)")
        inconnues = set(it) - set(CHAMPS)
        if inconnues:
            raise ValueError(f"alerte externe : cles inconnues {sorted(inconnues)}")
        out.append(alerte(it["kind"], it["target"], it["level"], it["title"], it["cause"],
                          it["recommendation"], station_id=it.get("station_id"),
                          turbine_id=it.get("turbine_id"), location=it.get("location"),
                          value=it.get("value"), threshold=it.get("threshold"),
                          unit=it.get("unit")))
    return out


# ---------------------------------------------------------------------- agregation
_RANG = {n: k for k, n in enumerate(NIVEAUX)}


def agreger(*listes: Iterable[dict]) -> list[dict]:
    """Une alerte par id (la plus grave si deux sources donnent la meme cible)."""
    par_id: dict[str, dict] = {}
    for liste in listes:
        for a in liste:
            b = par_id.get(a["id"])
            if b is None or _RANG[a["level"]] > _RANG[b["level"]]:
                par_id[a["id"]] = a
    return list(par_id.values())


def trier(items: list[dict]) -> list[dict]:
    """Ouvertes d'abord, puis niveau decroissant, type, cible : ordre stable du document."""
    return sorted(items, key=lambda a: (a["status"] != "ouverte", -_RANG[a["level"]],
                                        a["kind"], a["target"]))


def items_externes(items: Iterable[dict], kinds: Iterable[str] = KINDS_EXTERNES
                   ) -> list[dict]:
    """Items publies (journal M4) des types externes : les alertes du parc avec leur
    raised_at de premiere levee et leur statut ouverte / close (/api/wind-farm)."""
    kinds = set(kinds)
    return [a for a in items if a["kind"] in kinds]


def compter_ouvertes(items: Iterable[dict]) -> dict:
    """`kpi.open_alerts` du snapshot (m9 : alertes OUVERTES par niveau)."""
    out = {n: 0 for n in NIVEAUX}
    for a in items:
        if a["status"] == "ouverte":
            out[a["level"]] += 1
    return out


def resume_stations(items: Iterable[dict], n_capteurs: int) -> dict:
    """`stations_summary` (station_state) : derive, muette, exclue (hors plage, masquee)
    d'apres les alertes OUVERTES ; le reste est actif."""
    etat: dict[str, str] = {}
    for a in items:
        if a["status"] != "ouverte" or a["station_id"] is None:
            continue
        e = {"capteur_derive": "derive", "station_muette": "muette",
             "hors_plage": "exclue"}.get(a["kind"])
        if e and etat.get(a["station_id"]) not in ("muette", "exclue"):
            etat[a["station_id"]] = e
    out = {"active": 0, "suspecte": 0, "muette": 0, "derive": 0, "exclue": 0}
    for e in etat.values():
        out[e] += 1
    out["active"] = max(int(n_capteurs) - sum(out.values()), 0)
    return out


# ---------------------------------------------------------------------- journal JSONL
class JournalAlertes:
    """Cycle de vie des alertes, un evenement par ligne (JSONL, LF) :

        {"evenement": "levee" | "niveau" | "close", "instant": "...Z", "id": ...,
         "kind": ..., "target": ..., "level": ..., "alerte": {...}}

    Relu au demarrage : les alertes ouvertes gardent leur `raised_at` (premiere levee de
    l'episode en cours), meme apres un redemarrage. Les instants doivent arriver dans
    l'ordre chronologique ; un instant anterieur au dernier journalise est evalue sans rien
    ecrire (consultation). path None : journal en memoire seulement.
    """

    def __init__(self, path: str | Path | None, retention_h: float = 24.0) -> None:
        self.path = Path(path) if path is not None else None
        self.retention = pd.Timedelta(hours=float(retention_h))
        self.ouvertes: dict[str, dict] = {}
        self.closes: dict[str, tuple[str, dict]] = {}
        self.dernier: str | None = None
        self.lignes_illisibles = 0
        self._lock = threading.Lock()
        if self.path is not None and self.path.exists():
            self._relire()

    def _relire(self) -> None:
        for ligne in self.path.read_bytes().decode("utf-8").splitlines():
            if not ligne.strip():
                continue
            try:
                ev = json.loads(ligne)
                a, t = ev["alerte"], ev["instant"]
                a["id"], str(t)
            except (ValueError, KeyError, TypeError):
                # ligne tronquee (arret brutal, synchro Drive) : on la saute, on la compte
                self.lignes_illisibles += 1
                warnings.warn(f"journal d'alertes {self.path} : ligne illisible ignoree "
                              f"({self.lignes_illisibles})", RuntimeWarning, stacklevel=2)
                continue
            if ev["evenement"] == "close":
                self.ouvertes.pop(a["id"], None)
                self.closes[a["id"]] = (t, a)
            else:
                self.ouvertes[a["id"]] = a
                self.closes.pop(a["id"], None)
            if self.dernier is None or t > self.dernier:
                self.dernier = t

    def _ecrire(self, evenements: list[dict]) -> None:
        if self.path is None or not evenements:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        txt = "".join(json.dumps(e, ensure_ascii=False, separators=(",", ":")) + "\n"
                      for e in evenements)
        with open(self.path, "ab") as f:
            f.write(txt.encode("utf-8"))

    def chronologique(self, instant: str) -> bool:
        return self.dernier is None or instant > self.dernier

    def mettre_a_jour(self, alertes: list[dict], instant: str) -> list[dict]:
        """Alertes de l'instant -> items publies (ouvertes + closes recentes), raised_at
        et status poses ; ecrit les transitions si l'instant est nouveau."""
        with self._lock:
            chrono = self.chronologique(instant)
            evs: list[dict] = []
            items = []
            courants = {a["id"]: a for a in alertes}
            for aid, a in courants.items():
                prev = self.ouvertes.get(aid)
                a = dict(a, status="ouverte",
                         raised_at=prev["raised_at"] if prev else instant)
                if chrono:
                    if prev is None:
                        evs.append(self._ev("levee", instant, a))
                    elif prev["level"] != a["level"]:
                        evs.append(self._ev("niveau", instant, a))
                    self.ouvertes[aid] = a
                    self.closes.pop(aid, None)
                items.append(a)
            if chrono:
                for aid in [k for k in self.ouvertes if k not in courants]:
                    a = dict(self.ouvertes.pop(aid), status="close")
                    self.closes[aid] = (instant, a)
                    evs.append(self._ev("close", instant, a))
                self.dernier = instant
                self._ecrire(evs)
            t = pd.Timestamp(instant)
            for aid, (tc, a) in list(self.closes.items()):
                age = t - pd.Timestamp(tc)
                if chrono and age > self.retention:
                    del self.closes[aid]
                elif aid not in courants and pd.Timedelta(0) <= age <= self.retention:
                    items.append(a)
            return trier(items)

    @staticmethod
    def _ev(nom: str, instant: str, a: dict) -> dict:
        return {"evenement": nom, "instant": instant, "id": a["id"], "kind": a["kind"],
                "target": a["target"], "level": a["level"], "alerte": a}


# ---------------------------------------------------------------------- seuils figes
def charger_seuils(path: str | Path) -> dict:
    """JSON des seuils regles sur la VALIDATION (scripts/m4_an1_colab.py) ; refuse le test."""
    data = json.loads(Path(path).read_bytes().decode("utf-8"))
    if data.get("jeu") != "validation":
        raise ValueError("seuils d'alerte : « jeu » doit valoir « validation » (les seuils "
                         f"ne se reglent jamais sur le test), recu {data.get('jeu')!r}")
    if data.get("methode") not in METHODES_SEUILS:
        raise ValueError(f"seuils d'alerte : methode inconnue {data.get('methode')!r}")
    return data


# cles des seuils (JSON de validation) recopiees dans la section alerts du moteur
CLES_SEUILS = ("derive_statistique", "derive_kappa", "derive_min_m_s", "derive_rayon",
               "derive_rayon_echelle", "derive_min_voisins", "derive_fenetre",
               "derive_min_instants", "derive_cote", "entrees_z_max",
               "entrees_part_extreme", "entrees_ecart_extreme", "entrees_variables")


# ---------------------------------------------------------------------- moteur
@dataclass
class ResultatAlertes:
    """Alertes d'un instant + ce que le snapshot en reprend (physics, kpi.open_alerts)."""

    instant: str
    items: list[dict]
    physics: dict | None = None
    entrees: list[dict] = field(default_factory=list)
    derive: dict = field(default_factory=dict)

    def document(self) -> dict:
        """GET /v1/alerts/{id} : exactement alerts.schema.json (rev. 2)."""
        return {"items": self.items}

    def ouvertes(self) -> dict:
        return compter_ouvertes(self.items)


class MoteurAlertes:
    """Etat du service : suivi de derive, journal, derniers resultats. Thread-safe."""

    MAX_RESULTATS = 16

    def __init__(self, cfg, journal: str | Path | None | bool = True) -> None:
        self.cfg = cfg
        self.acfg = copy.copy(cfg.alerts)          # les seuils figes ne touchent pas cfg
        self.seuils = charger_seuils(self.acfg.seuils) if self.acfg.seuils else None
        if self.seuils:
            for k in CLES_SEUILS:
                if k in self.seuils:
                    setattr(self.acfg, k, self.seuils[k])
            if self.seuils.get("derive_sites"):
                # modele de sites : chemin relatif au JSON des seuils (meme session)
                self.acfg.derive_sites = Path(self.acfg.seuils).parent / \
                    self.seuils["derive_sites"]
            elif "derive_sites" in self.seuils:
                self.acfg.derive_sites = None
            self.acfg.__post_init__()               # memes controles que la config
        self.sites = None
        if self.acfg.derive_sites is not None and "spatiale" in composantes(
                self.acfg.derive_statistique):
            self.sites = ModeleSites.charger(self.acfg.derive_sites)
        if journal is True:
            j = self.acfg.journal
            path = None if str(j).lower() == "off" else \
                (j if j is not None else cfg.paths.run_dir / "logs" / "alertes.jsonl")
        else:
            path = journal or None
        self.journal = JournalAlertes(path, self.acfg.retention_close_h)
        self.suivis = {c: SuiviDerive(self.acfg.derive_fenetre, mode_suivi(c))
                       for c in composantes(self.acfg.derive_statistique)}
        self.dernier: ResultatAlertes | None = None
        self._resultats: OrderedDict[str, ResultatAlertes] = OrderedDict()
        self._lock = threading.RLock()

    # -------------------------------------------------------------- evaluation
    def evaluer(self, geom: Geometry, S: np.ndarray, served: tuple, instant,
                membre: Callable | None, surf_mu, surf_sd, *, rapport: dict | None = None,
                hors_plage: Iterable[int] = (), reliability=None, Y: np.ndarray | None = None,
                externes: Iterable[dict] = (), cle: str | None = None,
                stations_disponibles: int | None = None) -> ResultatAlertes:
        """Toutes les alertes d'un instant. `served` = (u, v, w) de la passe servie (NaN
        sous le relief) ; `membre(masque)` refait une passe sans les capteurs retires
        (innovation, influence, combinee ; None : pas de detection pour celles-ci, le
        detecteur spatial n'en a pas besoin). Ne modifie ni S ni la valeur servie."""
        a = self.acfg
        if instant is None:
            raise ValueError("alertes : l'instant (horodatage du volume) est requis")
        t = iso_utc(instant)
        with self._lock:
            if cle is not None and cle in self._resultats:
                return self._resultats[cle]
            sensor = np.asarray(S[7], float) > 0.5
            listes = []
            # M3 : rapport du contrat, sinon part de capteurs disponibles
            if rapport is not None:
                listes.append(alertes_contrat(rapport, geom.nx_s, geom.dx_s, geom.dy_s,
                                              a.max_par_type, hors_plage))
            else:
                listes.append(alertes_disponibilite(
                    sensor, a.capteurs_part_min, a.capteurs_part_critique, a.reseau,
                    stations_disponibles, a.stations_attendues))
            # M2 : couverture (calculee si la fiabilite est desactivee)
            if reliability is not None:
                cov, sig = reliability.coverage, reliability.sigma_speed
            else:
                from .reliability import coverage_index
                rc = self.cfg.reliability
                cov = coverage_index(geom, sensor, rc.coverage_length_m, rc.coverage_height_m,
                                     rc.coverage_density_ref)
                sig = None
            listes.append(alertes_couverture(geom, cov, a.couverture_maille_m,
                                             a.couverture_min, sig, a.max_par_type))
            # M4 : physique (D11)
            # verite_synthetique sans verite (upload, direct) : extrapolation de surface
            source = a.rho_source
            if source == "verite_synthetique" and Y is None:
                source = "extrapolation_surface"
            rho, src = rho_instant(geom, source, S, sensor, Y)
            phys = physique(geom, *served, rho, src, self.cfg.dso.mass_rms)
            listes.append(alertes_physique(phys, self.cfg.dso.mass_rms,
                                           a.masse_facteur_critique, a.sol_rms_max_m_s))
            # M4 : entrees
            ent = derive_entrees(S, sensor, surf_mu, surf_sd, a.entrees_z_max,
                                 a.entrees_ecart_extreme, a.entrees_part_extreme,
                                 a.entrees_variables)
            listes.append(alertes_entrees(ent, a.entrees_z_max, a.entrees_part_extreme,
                                          a.entrees_ecart_extreme))
            # M4 : capteurs en derive (persistance : instants chronologiques seulement)
            derive = {"evaluee": False, "statistique": a.derive_statistique,
                      "kappa": a.derive_kappa}
            comp = composantes(a.derive_statistique)
            passes = [c for c in comp if c in STATISTIQUES_PASSES]
            dernier = self.suivis[comp[0]].dernier
            if passes and membre is None:
                derive["raison"] = "pas de passe sans capteur (membre absent)"
            elif not (dernier is None or t > dernier):
                derive["raison"] = "instant anterieur ou egal au dernier instant suivi"
            else:
                ecs = {}
                if "spatiale" in comp:
                    ecs["spatiale"] = ecarts_spatiaux(
                        S, sensor, a.derive_rayon, a.derive_rayon_echelle,
                        a.derive_echelle_plancher_m_s, a.derive_min_voisins, self.sites,
                        heure_utc(t))
                if passes:
                    ecs.update(ecarts_capteurs(S, sensor, geom, (served[0], served[1]),
                                               membre, a.derive_cote, passes,
                                               a.derive_echelle_plancher_m_s))
                for c in comp:
                    self.suivis[c].ajouter(t, ecs[c])
                sts = {c: self.suivis[c].statistiques() for c in comp}
                st = (combiner(sts["spatiale"], sts["innovation"])
                      if a.derive_statistique == "combinee" else sts[comp[0]])
                flag = drapeaux_derive(st, a.derive_kappa, a.derive_min_m_s,
                                       a.derive_min_instants)
                listes.append(alertes_derive(geom, flag, st, a.derive_kappa, a.derive_min_m_s,
                                             a.derive_facteur_critique, a.derive_statistique,
                                             a.max_par_type, a.derive_rayon))
                ec = ecs[comp[0]]
                derive.update({"evaluee": True, "passes": a.derive_cote ** 2 if passes else 0,
                               "echelle_m_s": _num(ec.echelle),
                               "capteurs_evalues": int(np.isfinite(ec.z).sum()),
                               "signales": int(flag.sum())})
                if "spatiale" in comp:
                    derive.update({"rayon_mailles": a.derive_rayon,
                                   "correction_site": self.sites is not None})
            listes.append(alertes_externes(externes))
            items = self.journal.mettre_a_jour(agreger(*listes), t)
            res = ResultatAlertes(t, items, phys, ent, derive)
            self.dernier = res
            if cle is not None:
                self._resultats[cle] = res
                while len(self._resultats) > self.MAX_RESULTATS:
                    self._resultats.popitem(last=False)
            return res

    def verite(self, wf, stem: str) -> np.ndarray | None:
        """Y du cache quand rho_source = verite_synthetique (meme regle pour l'API et le
        MCP), sinon None."""
        if self.acfg.rho_source != "verite_synthetique":
            return None
        path = wf.cfg.paths.cache_dir / f"{stem}_Y.npy"
        return np.load(path) if path.exists() else None

    def evaluer_modele(self, wf, S: np.ndarray, hour: float, doy: int, volume, **kw
                       ) -> ResultatAlertes:
        """Raccourci pour un `WindFieldModel` et le `Volume` servi de l'instant.

        Avec un parc declare (`wf.farm`, lot M5) et sans `externes` explicites, les alertes
        vent_fort_moyeu / cisaillement du volume y sont ajoutees (`windfarm.farm_alerts`) :
        elles passent par le journal (raised_at, ouverte / close) et par kpi.open_alerts.
        API, upload, MCP et CLI passent tous par ici."""
        # fiabilite differee (M2) : couverture, sigma et membres du parc doivent etre PRETS
        # (attendus, ou calcules ici) ; jamais un etat partiel dans les alertes
        if hasattr(volume, "wait_reliability"):
            volume.wait_reliability()
        if "externes" not in kw and getattr(wf, "farm", None) is not None:
            from .windfarm import farm_alerts
            kw["externes"] = farm_alerts(volume, wf)
        st = wf.norm.stats
        return self.evaluer(wf.geom, S, (volume.u, volume.v, volume.w), volume.timestamp,
                            lambda kept: wf._member(S, kept, hour, doy),
                            st["surf_mu"], st["surf_sd"],
                            reliability=kw.pop("reliability", volume.reliability), **kw)
