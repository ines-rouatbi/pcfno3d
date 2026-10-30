"""Donnees ATOM Wind partagees par l'API (/api/atom-wind/*), le serveur MCP et l'assistant (E6) :
journees temps reel / prevision et ensemble GEFS (memes caches), et lecture en un point (vent a une
hauteur et une heure, profil, decision agricole, production eolienne, alertes de vent).

Un point est lu comme dans le navigateur (webapp/atom-wind/atom-data.js) : maille de 500 m,
interpolation lineaire en heure entre les sorties du modele et en hauteur entre les niveaux, profil
logarithmique sous 10 m avec la rugosite de la maille.
"""
from __future__ import annotations

import base64
import math
import threading
import time
from pathlib import Path

import numpy as np

TTL_DIRECT_S = 1800.0
TTL_ENSEMBLE_S = 6 * 3600.0
_VERROU = threading.Lock()
_CACHES: dict = {"direct": {}, "ensemble": {}, "source": {}, "grille": {}, "valeur": {}}

# lieux de l'application (maquette ATOM Wind, PLACES) : le nom suffit a l'assistant
LIEUX = {
    "tunis-carthage": (36.8504, 10.2156), "aeroport": (36.8504, 10.2156), "tunis": (36.80, 10.18),
    "la marsa": (36.878, 10.325), "ben arous": (36.75, 10.23), "hammam-lif": (36.728, 10.34),
    "mornag": (36.68, 10.29), "djebel ressas": (36.61, 10.33), "soliman": (36.69, 10.49),
    "grombalia": (36.60, 10.50), "korbous": (36.82, 10.57), "nabeul": (36.455, 10.715),
    "zaghouan": (36.4333, 10.0833), "mohamedia": (36.68, 10.16),
}


def _cle(wf, nom: str) -> tuple:
    return (id(wf), nom)


def source(wf):
    from pcfno3d import temps_reel
    k = id(wf)
    s = _CACHES["source"].get(k)
    if s is None:
        s = _CACHES["source"][k] = temps_reel.SourceDirect(wf.geom, ttl_s=TTL_DIRECT_S)
    return s


def grille(wf) -> dict:
    from pcfno3d import atom_wind
    from pcfno3d.realdata import GeoRef
    k = id(wf)
    g = _CACHES["grille"].get(k)
    if g is None:
        try:
            geo = GeoRef.from_cfg(wf.cfg)
        except ValueError:
            geo = None
        g = atom_wind.grille(wf.geom, geo, wf.cfg)
        g["modele"] = wf.checkpoint_path.name
        g["configuration"] = wf.cfg.name
        _CACHES["grille"][k] = g
    return g


def _avec_cache(nom: str, wf, date: str, ttl: float, calcul) -> dict:
    caches = _CACHES[nom]
    e = caches.get(_cle(wf, date))
    if e is None or time.time() - e["t"] > ttl:
        with _VERROU:
            e = caches.get(_cle(wf, date))
            if e is None or time.time() - e["t"] > ttl:
                try:
                    out = calcul()
                except Exception as exc:
                    if e is not None:           # source muette : le dernier calcul, signale
                        return {**e["out"], "avertissement": f"donnees de {e['out']['mis_a_jour_utc']} : {exc}"}
                    raise
                e = caches[_cle(wf, date)] = {"t": time.time(), "out": out}
    return e["out"]


def jour(wf, j: int) -> dict:
    """La journee relative j (0 aujourd'hui, 1 demain, 2 apres-demain), cachee 30 min."""
    from pcfno3d import atom_wind
    date = atom_wind.date_locale_aujourdhui(j)
    return _avec_cache("direct", wf, date, TTL_DIRECT_S,
                       lambda: atom_wind.jour_direct(wf, source(wf), date))


def ensemble(wf, j: int) -> dict:
    from pcfno3d import atom_wind
    date = atom_wind.date_locale_aujourdhui(j)
    return _avec_cache("ensemble", wf, date, TTL_ENSEMBLE_S,
                       lambda: atom_wind.jour_ensemble(wf, source(wf), date))


def valeur_stations(wf) -> dict:
    """Valeur et influence des stations aujourd'hui (E10, E11), cachees 30 min (4 passes par heure rejouee)."""
    from pcfno3d import atom_wind
    date = atom_wind.date_locale_aujourdhui(0)
    return _avec_cache("valeur", wf, date, TTL_DIRECT_S,
                       lambda: atom_wind.valeur_stations(wf, source(wf), date))


# ---------------------------------------------------------------------- lecture en un point
def _decoder(b64: str) -> np.ndarray:
    a = np.frombuffer(base64.b64decode(b64), dtype="<i2").astype(float)
    return np.where(a == -32768, np.nan, a / 100.0)


def _tableau(payload: dict, cle: str, forme_cle: str = "forme") -> np.ndarray:
    k = "_" + cle
    if k not in payload:
        payload[k] = _decoder(payload[cle]).reshape(payload[forme_cle]) if cle in payload else None
    return payload[k]


def position(wf, lat: float | None = None, lon: float | None = None, lieu: str | None = None) -> tuple[float, float, str]:
    """(x_km, y_km, nom) depuis lat/lon ou un nom de lieu de l'application."""
    from pcfno3d.realdata import GeoRef
    if lieu:
        n = lieu.strip().lower()
        trouve = next((k for k in LIEUX if k in n or n in k), None)
        if trouve is None:
            raise ValueError(f"lieu inconnu : {lieu} (lieux connus : {', '.join(sorted(LIEUX))})")
        lat, lon = LIEUX[trouve]
        nom = trouve
    else:
        if lat is None or lon is None:
            raise ValueError("donner lat et lon, ou un lieu")
        nom = f"{lat:.4f} N, {lon:.4f} E"
    x, y = GeoRef.from_cfg(wf.cfg).to_xy(lat, lon)
    g = grille(wf)
    if not (0 <= x <= g["largeur_km"] * 1000 and 0 <= y <= g["hauteur_km"] * 1000):
        raise ValueError(f"{nom} est hors du domaine du Grand Tunis")
    return float(x) / 1000.0, float(y) / 1000.0, nom


def _crochets(payload: dict, h: float) -> tuple[int, int, float]:
    I = [i["heure_locale"] for i in payload["instants"]]
    if len(I) == 1 or h <= I[0]:
        return 0, 0, 0.0
    if h >= I[-1]:
        return len(I) - 1, len(I) - 1, 0.0
    a = max(k for k in range(len(I) - 1) if I[k] <= h)
    return a, a + 1, (h - I[a]) / max(I[a + 1] - I[a], 1e-6)


def _maille(wf, x_km: float, y_km: float) -> tuple[int, int, float]:
    g = grille(wf)
    i = min(g["nx"] - 1, max(0, int(x_km // g["pas_km"])))
    j = min(g["ny"] - 1, max(0, int(y_km // g["pas_km"])))
    return i, j, float(g["z0_m"][j][i])


def _au_niveau(payload, arr, it, z, i, j, comp, z0) -> float:
    H = payload["hauteurs_agl_m"]
    col = arr[it, :, comp, j, i]
    if z <= H[0]:
        k = next((k for k in range(len(H)) if np.isfinite(col[k])), None)
        if k is None:
            return 0.0
        zz = max(z, 1.5 * z0)
        return float(col[k] * max(0.0, math.log(zz / z0) / math.log(H[k] / z0)))
    k = max(kk for kk in range(len(H) - 1) if H[kk] < z) if z < H[-1] else len(H) - 2
    a = min(1.0, (z - H[k]) / (H[k + 1] - H[k]))
    v0, v1 = col[k], col[k + 1]
    if np.isfinite(v0) and np.isfinite(v1):
        return float((1 - a) * v0 + a * v1)
    return float(v1 if np.isfinite(v1) else v0 if np.isfinite(v0) else 0.0)


def vent(wf, payload: dict, x_km: float, y_km: float, z_agl: float, h: float) -> tuple[float, float]:
    arr = _tableau(payload, "vent")
    i, j, z0 = _maille(wf, x_km, y_km)
    a, b, w = _crochets(payload, h)
    ua, va = (_au_niveau(payload, arr, a, z_agl, i, j, c, z0) for c in (0, 1))
    if a == b or w == 0:
        return ua, va
    ub, vb = (_au_niveau(payload, arr, b, z_agl, i, j, c, z0) for c in (0, 1))
    return (1 - w) * ua + w * ub, (1 - w) * va + w * vb


def _surface(wf, payload, cle, x_km, y_km, h) -> float | None:
    if cle not in payload:
        return None
    if "_" + cle not in payload:
        g = grille(wf)
        payload["_" + cle] = _decoder(payload[cle]).reshape(len(payload["instants"]), g["ny"], g["nx"])
    arr = payload["_" + cle]
    i, j, _ = _maille(wf, x_km, y_km)
    a, b, w = _crochets(payload, h)
    va, vb = arr[a, j, i], arr[b, j, i]
    return float((1 - w) * va + w * vb) if np.isfinite(va) and np.isfinite(vb) else None


def _direction(u: float, v: float) -> float:
    return round((270.0 - math.degrees(math.atan2(v, u))) % 360.0, 0)


def lire_point(wf, j: int, z_agl: float, h: float, lat=None, lon=None, lieu=None) -> dict:
    x, y, nom = position(wf, lat, lon, lieu)
    p = jour(wf, j)
    u, v = vent(wf, p, x, y, z_agl, h)
    s = math.hypot(u, v)
    return {"lieu": nom, "date": p["date"], "heure_locale": h, "hauteur_sol_m": z_agl,
            "vitesse_m_s": round(s, 2), "vitesse_km_h": round(s * 3.6, 1), "direction_provenance_deg": _direction(u, v),
            "source": "prevision" if p.get("prevision") else "temps reel",
            "temperature_2m_c": _arrondi(_surface(wf, p, "t2m", x, y, h)),
            "humidite_2m_pct": _arrondi(_surface(wf, p, "hr2m", x, y, h)),
            "note": "vent : ATOM Wind (PC-FNO3D) ; temperature et humidite : GFS"}


def _arrondi(x, nd=1):
    return None if x is None else round(x, nd)


def profil(wf, j: int, h: float, lat=None, lon=None, lieu=None) -> dict:
    x, y, nom = position(wf, lat, lon, lieu)
    p = jour(wf, j)
    niveaux = []
    for z in p["hauteurs_agl_m"]:
        u, v = vent(wf, p, x, y, float(z), h)
        niveaux.append({"hauteur_sol_m": z, "vitesse_m_s": round(math.hypot(u, v), 2),
                        "direction_provenance_deg": _direction(u, v)})
    return {"lieu": nom, "date": p["date"], "heure_locale": h, "niveaux": niveaux}


def decision_agricole(wf, j: int, lat=None, lon=None, lieu=None) -> dict:
    """Creneaux de traitement et d'aspersion de la journee (regles de l'application, vent a 10 m),
    avec la probabilite GEFS de pouvoir traiter si l'ensemble est deja calcule."""
    x, y, nom = position(wf, lat, lon, lieu)
    p = jour(wf, j)
    heures = []
    for h in range(24):
        u, v = vent(wf, p, x, y, 10.0, float(h))
        k = math.hypot(u, v) * 3.6
        trait = "attention" if k < 3 else "oui" if k <= 15 else "attention" if k <= 19 else "non"
        asp = "oui" if k <= 15 else "attention" if k <= 25 else "non"
        heures.append({"heure": h, "vent_10m_km_h": round(k, 1), "traitement": trait, "aspersion": asp})
    creneaux = []
    debut = None
    for r in heures + [{"heure": 24, "traitement": "fin"}]:
        if r["traitement"] == "oui" and debut is None:
            debut = r["heure"]
        elif r["traitement"] != "oui" and debut is not None:
            creneaux.append(f"{debut} h - {r['heure']} h")
            debut = None
    proba = None
    e = _CACHES["ensemble"].get(_cle(wf, p["date"]))
    if e is not None:
        P = _tableau(e["out"], "proba_traitement", "forme_proba")
        i, jj, _ = _maille(wf, x, y)
        proba = {f"{int(it['heure_locale'])} h": round(float(P[k, jj, i]) * 100) for k, it in enumerate(e["out"]["instants"])
                 if np.isfinite(P[k, jj, i])}
    return {"lieu": nom, "date": p["date"], "source": "prevision" if p.get("prevision") else "temps reel",
            "regle": "traitement : oui de 3 a 15 km/h a 10 m, attention sous 3 ou de 15 a 19, non au-dela",
            "creneaux_traitement": creneaux, "heures": heures, "probabilite_traiter_pct_gefs": proba}


def production_eolienne(wf, j: int, hub_m: float = 80.0, lat=None, lon=None, lieu=None) -> dict:
    """Production d'une eolienne de reference de 2 MW au point (vent au moyeu heure par heure)."""
    from pcfno3d import atlas
    from pcfno3d.windfarm import RHO_REF
    courbe = atlas.charger_courbe(Path(__file__).resolve().parents[1] / "configs" / "eolien" / "parc_demo.yaml")
    x, y, nom = position(wf, lat, lon, lieu)
    p = jour(wf, j)
    i, jj, _ = _maille(wf, x, y)
    rho = RHO_REF * math.exp(-(float(grille(wf)["terrain_m"][jj][i]) + hub_m) / 8500.0)
    heures, energie = [], 0.0
    for h in range(24):
        u, v = vent(wf, p, x, y, hub_m, float(h))
        s = math.hypot(u, v)
        kw = float(courbe.power((rho / courbe.rho_ref) ** (1 / 3) * s))
        energie += kw
        heures.append({"heure": h, "vent_moyeu_m_s": round(s, 2), "puissance_kw": round(kw)})
    return {"lieu": nom, "date": p["date"], "moyeu_m": hub_m, "eolienne": f"{courbe.model} ({courbe.rated_kw / 1000:g} MW)",
            "energie_jour_mwh": round(energie / 1000.0, 2), "facteur_charge": round(energie / (24 * courbe.rated_kw), 3),
            "heures": heures}


def alertes_vent(wf, j: int, seuil_10m_km_h: float = 50.0, seuil_100m_m_s: float = 20.0) -> dict:
    """Vent fort sur le domaine : maximum a 10 m et a 100 m, ou et quand."""
    p = jour(wf, j)
    arr = _tableau(p, "vent")
    H = p["hauteurs_agl_m"]
    g = grille(wf)
    sortie = {"date": p["date"], "alertes": []}
    for z, seuil, unite in ((10, seuil_10m_km_h / 3.6, "km/h"), (100, seuil_100m_m_s, "m/s")):
        k = H.index(z)
        s = np.hypot(arr[:, k, 0], arr[:, k, 1])
        if not np.isfinite(s).any():
            continue
        it, jj, i = np.unravel_index(np.nanargmax(s), s.shape)
        vmax = float(s[it, jj, i])
        info = {"hauteur_sol_m": z, "max": round(vmax * (3.6 if unite == "km/h" else 1), 1), "unite": unite,
                "heure_locale": p["instants"][it]["heure_locale"],
                "x_km": round((i + 0.5) * g["pas_km"], 1), "y_km": round((jj + 0.5) * g["pas_km"], 1)}
        sortie[f"max_{z}m"] = info
        if vmax >= seuil:
            sortie["alertes"].append({**info, "niveau": "important", "cause": f"vent fort a {z} m"})
    return sortie


# parc eolien virtuel de l'application (maquette ATOM Wind, TURB : trois eoliennes pres de Djebel Ressas)
PARC_VIRTUEL = (("E1", 36.618, 10.315), ("E2", 36.612, 10.33), ("E3", 36.605, 10.345))


def _plages(heures: list[dict], ok: str, duree_min_h: int) -> list[str]:
    plages, debut = [], None
    for r in heures + [{"heure": 24, ok: False}]:
        if r[ok] and debut is None:
            debut = r["heure"]
        elif not r[ok] and debut is not None:
            if r["heure"] - debut >= duree_min_h:
                plages.append(f"{debut} h - {r['heure']} h")
            debut = None
    return plages


def fenetres_maintenance(wf, j: int, hauteur_m: float = 80.0, seuil_m_s: float = 12.0, duree_min_h: int = 2,
                         lat=None, lon=None, lieu=None) -> dict:
    """Creneaux surs pour le travail en hauteur (cahier v12, E7) : heures ou le vent moyen a la hauteur
    de la nacelle ou du mat reste sous le seuil, en plages d'au moins `duree_min_h` heures. Le seuil est
    celui du constructeur ou de la procedure du site (12 m/s par defaut, a confirmer) ; le modele donne un
    vent moyen horaire, pas les rafales : une marge reste necessaire."""
    if lieu is None and (lat is None or lon is None):
        lieu = "djebel ressas"                                   # le parc virtuel de l'application
    x, y, nom = position(wf, lat, lon, lieu)
    p = jour(wf, j)
    heures = []
    for h in range(24):
        u, v = vent(wf, p, x, y, hauteur_m, float(h))
        s = math.hypot(u, v)
        heures.append({"heure": h, "vent_m_s": round(s, 1), "sur": bool(s < seuil_m_s),
                       "marge_m_s": round(seuil_m_s - s, 1)})
    return {"lieu": nom, "date": p["date"], "source": "prevision" if p.get("prevision") else "temps reel",
            "hauteur_sol_m": hauteur_m, "seuil_m_s": seuil_m_s, "duree_min_h": duree_min_h,
            "creneaux_surs": _plages(heures, "sur", duree_min_h), "heures": heures,
            "note": "vent moyen horaire ATOM Wind, sans rafales : garder une marge et suivre la procedure du site"}


def variations_production(wf, j: int, moyeu_m: float = 80.0, seuil_pct: float = 20.0) -> dict:
    """Hausses et chutes rapides de production du parc virtuel (cahier v12, E13), utiles au gestionnaire
    du reseau : puissance du parc heure par heure et rampes d'au moins `seuil_pct` % de la puissance
    installee d'une heure a l'autre."""
    eol = [production_eolienne(wf, j, moyeu_m, lat=la, lon=lo) for _, la, lo in PARC_VIRTUEL]
    from pcfno3d import atlas
    courbe = atlas.charger_courbe(Path(__file__).resolve().parents[1] / "configs" / "eolien" / "parc_demo.yaml")
    installee = courbe.rated_kw * len(eol)
    puissance = [sum(e["heures"][h]["puissance_kw"] for e in eol) for h in range(24)]
    rampes = []
    for h in range(1, 24):
        d = puissance[h] - puissance[h - 1]
        pct = 100.0 * d / installee
        if abs(pct) >= seuil_pct:
            rampes.append({"de": h - 1, "a": h, "variation_kw": round(d), "variation_pct": round(pct, 1),
                           "sens": "hausse" if d > 0 else "chute",
                           "niveau": "important" if abs(pct) >= 2 * seuil_pct else "attention"})
    return {"date": eol[0]["date"], "parc": "parc virtuel de Djebel Ressas (3 eoliennes de reference)",
            "eoliennes": [n for n, _, _ in PARC_VIRTUEL], "puissance_installee_kw": installee, "moyeu_m": moyeu_m,
            "seuil_pct_par_heure": seuil_pct, "puissance_kw": puissance, "rampes": rampes,
            "energie_jour_mwh": round(sum(puissance) / 1000.0, 2),
            "note": "estimation sans mesure de production ; pas horaire (les rampes plus courtes ne sont pas vues)"}


def explication_point(wf, lat=None, lon=None, lieu=None) -> dict:
    """Pourquoi ce vent ici (cahier v12, E10) : les stations qui ont le plus influence la valeur du point
    aujourd'hui (ecart a 10 m quand chacune est retiree, moyenne sur les heures mesurees), le reste venant
    de la grande echelle GFS et du relief."""
    x, y, nom = position(wf, lat, lon, lieu)
    d = valeur_stations(wf)
    cartes = _tableau(d, "influence")
    i, jj, _ = _maille(wf, x, y)
    infl = [(st, float(cartes[k, jj, i])) for k, st in enumerate(d["codes"])]
    total = sum(v for _, v in infl if np.isfinite(v))
    from pcfno3d.stations_reelles import CODES
    sortie = [{"station": CODES.get(k, k), "code": k, "influence_m_s": round(v, 2),
               "part_pct": round(100.0 * v / total) if total > 0 else 0} for k, v in infl if np.isfinite(v)]
    sortie.sort(key=lambda r: -r["influence_m_s"])
    principale = sortie[0]["station"] if sortie and sortie[0]["influence_m_s"] >= 0.1 else None
    return {"lieu": nom, "date": d["date"], "hauteur_sol_m": d["hauteur_carte_m"], "heures_locales": d["heures_locales"],
            "stations": sortie,
            "texte": (f"Ici, la station qui compte le plus est {principale}." if principale else
                      "Ici, aucune station ne change la valeur de plus de 0,1 m/s : le vent vient surtout de la grande "
                      "echelle (GFS) et du relief."),
            "methode": d["methode"]}


# micro-eolienne de mat (E18) : courbe GENERIQUE de 400 W, exemple a remplacer par la fiche du fabricant
MICRO_EOLIENNE = {"nom": "micro-eolienne generique 400 W (exemple)", "puissance_w": 400.0,
                  "vitesse_m_s": [0, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12.5, 20, 20.01],
                  "puissance": [0, 0, 10, 30, 60, 100, 160, 230, 300, 350, 390, 400, 400, 0]}


def dimensionnement_mat(wf, lat=None, lon=None, lieu=None, hauteur_m: float = 30.0, conso_w: float = 5.0,
                        autonomie_j: float = 3.0, profondeur: float = 0.8, rendement: float = 0.9) -> dict:
    """Dimensionnement energetique d'un mat de mesure (cahier v12, E18) : production heure par heure d'une
    micro-eolienne a la hauteur du mat sur les jours disponibles (aujourd'hui a apres-demain), bilan avec la
    consommation de la station, et taille de batterie : le plus grand deficit cumule de la periode, au
    moins `autonomie_j` jours de consommation, ramene a la profondeur de decharge et au rendement."""
    x, y, nom = position(wf, lat, lon, lieu)
    vit, prod, dates = [], [], []
    for j in (0, 1, 2):
        try:
            p = jour(wf, j)
        except Exception:                                       # noqa: BLE001 — jour absent : on s'arrete
            break
        dates.append(p["date"])
        for h in range(24):
            u, v = vent(wf, p, x, y, hauteur_m, float(h))
            s = math.hypot(u, v)
            vit.append(s)
            prod.append(float(np.interp(s, MICRO_EOLIENNE["vitesse_m_s"], MICRO_EOLIENNE["puissance"])))
    if not prod:
        raise ValueError("aucune journee disponible")
    bilan = np.cumsum(np.asarray(prod) * rendement - conso_w)            # Wh, pas horaire
    # plus grand deficit cumule : chute maximale de la courbe de bilan depuis un sommet
    deficit = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], bilan]))[1:] - bilan, initial=0.0))
    besoin_wh = max(deficit, conso_w * 24 * autonomie_j)
    batterie_wh = besoin_wh / (profondeur * rendement)
    e_prod, e_conso = sum(prod) * rendement, conso_w * len(prod)
    return {"lieu": nom, "dates": dates, "heures": len(prod), "hauteur_mat_m": hauteur_m,
            "eolienne": MICRO_EOLIENNE["nom"], "vent_moyen_m_s": round(float(np.mean(vit)), 2),
            "production_wh_par_jour": round(e_prod * 24 / len(prod)), "consommation_wh_par_jour": round(conso_w * 24),
            "couverture_pct": round(100.0 * e_prod / e_conso) if e_conso > 0 else None,
            "plus_grand_deficit_wh": round(deficit), "batterie_wh": round(batterie_wh),
            "batterie_ah_12v": round(batterie_wh / 12.0),
            "hypotheses": {"conso_w": conso_w, "autonomie_j": autonomie_j, "profondeur_decharge": profondeur,
                           "rendement": rendement},
            "production_w": [round(w) for w in prod],
            "limite": (f"{len(prod) // 24} jour(s) de vent ATOM Wind seulement : un dimensionnement definitif se fait "
                       "sur l'atlas annuel et la courbe du fabricant ; la micro-eolienne est un exemple generique")}
