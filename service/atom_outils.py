"""Donnees ATOM Wind partagees par l'API (/api/atom-wind/*), le serveur MCP et l'assistant (E6) :
journees temps reel / prevision et ensemble GEFS (memes caches), et lecture en un point (vent a une
hauteur et une heure, profil, decision agricole, production eolienne, alertes de vent).

Un point est lu comme dans le navigateur (webapp/atom-wind/atom-data.js) : maille de 500 m,
interpolation lineaire en heure entre les sorties du modele et en hauteur entre les niveaux, profil
logarithmique sous 10 m avec la rugosite de la maille.
"""
from __future__ import annotations

import base64
import gzip
import json
import math
import os
import re
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


# ---------------------------------------------------------------------- historique (BF40, HIS1)
# Chaque journee calculee (temps reel, previsions J+1 et J+2 du jour d'emission) est gardee sur le disque
# du service, run_dir/historique/ : une reconstruction passee se retrouve avec ses indicateurs, et une
# prevision se compare plus tard au temps reel du meme jour. PCFNO3D_HISTORIQUE=0 : rien n'est ecrit.
# Environ 8 Mo par journee (gzip) : 3 par jour d'emission.
NOM_HISTORIQUE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(temps_reel|prevision_J[12]_emise_\d{4}-\d{2}-\d{2})\.json\.gz$")


def dossier_historique(wf) -> Path:
    return Path(wf.cfg.paths.run_dir) / "historique"


def archiver(wf, j: int, out: dict) -> dict:
    """Ecrit la journee dans l'historique (atomique) et la renvoie telle quelle."""
    if os.environ.get("PCFNO3D_HISTORIQUE", "1") == "0":
        return out
    from pcfno3d import atom_wind
    d = dossier_historique(wf)
    d.mkdir(parents=True, exist_ok=True)
    nom = (f"{out['date']}_temps_reel.json.gz" if j == 0 else
           f"{out['date']}_prevision_J{j}_emise_{atom_wind.date_locale_aujourdhui(0)}.json.gz")
    tmp = d / (nom + ".tmp")
    tmp.write_bytes(gzip.compress(json.dumps({**out, "archive": nom}, ensure_ascii=False).encode("utf-8"), 6))
    tmp.replace(d / nom)
    return out


def historique(wf) -> list[dict]:
    """Index de l'historique : une ligne par journee archivee, la plus recente d'abord."""
    d = dossier_historique(wf)
    lignes = []
    for f in (d.glob("*.json.gz") if d.exists() else []):
        m = NOM_HISTORIQUE.match(f.name)
        if not m:
            continue
        genre = m[2]
        lignes.append({"nom": f.name, "date": m[1], "type": "temps reel" if genre == "temps_reel" else "prevision",
                       "echeance_j": None if genre == "temps_reel" else int(genre[11]),
                       "emise_le": None if genre == "temps_reel" else genre[-10:],
                       "taille_ko": round(f.stat().st_size / 1024),
                       "ecrit_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_mtime))})
    return sorted(lignes, key=lambda r: (r["date"], r["type"] == "temps reel", r["nom"]), reverse=True)


def lire_historique(wf, nom: str) -> bytes:
    """Le fichier gzip d'une journee archivee (nom verifie : aucun chemin arbitraire)."""
    if not NOM_HISTORIQUE.match(nom):
        raise ValueError("nom d'archive invalide")
    f = dossier_historique(wf) / nom
    if not f.exists():
        raise FileNotFoundError(nom)
    return f.read_bytes()


def entree_production(wf, timestamp, points):
    """pcfno3d.temps_reel.entree_production avec la source GFS partagee du service."""
    from pcfno3d import temps_reel as tr
    return tr.entree_production(wf, source(wf), timestamp, points)


def entree_production_active(wf) -> bool:
    from pcfno3d import temps_reel as tr
    return tr.entree_production_active(wf)


def jour(wf, j: int) -> dict:
    """La journee relative j (0 aujourd'hui, 1 demain, 2 apres-demain), cachee 30 min."""
    from pcfno3d import atom_wind
    date = atom_wind.date_locale_aujourdhui(j)
    # incertitude en temps reel (BF8) : PCFNO3D_SIGMA_DIRECT=1 et un modele calibre (inference.reliability)
    fiab = os.environ.get("PCFNO3D_SIGMA_DIRECT", "0") == "1"
    return _avec_cache("direct", wf, date, TTL_DIRECT_S,
                       lambda: archiver(wf, j, atom_wind.jour_direct(wf, source(wf), date, fiabilite=fiab)))


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
    from pcfno3d.agriculture import decision_aspersion, decision_traitement
    x, y, nom = position(wf, lat, lon, lieu)
    p = jour(wf, j)
    heures = []
    for h in range(24):
        u, v = vent(wf, p, x, y, 10.0, float(h))
        s = math.hypot(u, v)
        # la regle figee (pcfno3d.agriculture, = scripts/verifier_decisions.py) : km/h arrondis a 1e-6,
        # 15 km/h reste « oui » (avant le 11/11/2026, 15/3,6 x 3,6 = 15,000000000000002 donnait « attention »)
        trait, asp = str(decision_traitement(s)), str(decision_aspersion(s))
        heures.append({"heure": h, "vent_10m_km_h": round(s * 3.6, 1), "traitement": trait, "aspersion": asp})
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
        heures.append({"heure": h, "vent_moyeu_m_s": round(s, 2), "direction_deg": _direction(u, v),
                       "puissance_kw": round(kw)})
    return {"lieu": nom, "date": p["date"], "moyeu_m": hub_m, "eolienne": f"{courbe.model} ({courbe.rated_kw / 1000:g} MW)",
            "x_km": x, "y_km": y, "facteur_densite": (rho / courbe.rho_ref) ** (1 / 3),
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
ROTOR_PARC_M = 90.0                      # generique-2MW (configs/eolien/parc_demo.yaml)


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
    from pcfno3d.sillage import jensen
    courbe = atlas.charger_courbe(Path(__file__).resolve().parents[1] / "configs" / "eolien" / "parc_demo.yaml")
    installee = courbe.rated_kw * len(eol)
    sans_sillage = [sum(e["heures"][h]["puissance_kw"] for e in eol) for h in range(24)]
    # sillages de Jensen (BF72) : vitesses au moyeu reduites selon la direction de l'heure
    xm, ym = [e["x_km"] * 1000 for e in eol], [e["y_km"] * 1000 for e in eol]
    puissance = []
    for h in range(24):
        libres = [e["heures"][h]["vent_moyeu_m_s"] for e in eol]
        dirs = np.radians([e["heures"][h]["direction_deg"] for e in eol])
        d_moy = float(np.degrees(np.arctan2(np.mean(np.sin(dirs)), np.mean(np.cos(dirs)))) % 360.0)
        v = jensen(xm, ym, [ROTOR_PARC_M] * len(eol), libres, d_moy)
        puissance.append(round(sum(float(courbe.power(e["facteur_densite"] * vi)) for e, vi in zip(eol, v))))
    rampes = []
    for h in range(1, 24):
        d = puissance[h] - puissance[h - 1]
        pct = 100.0 * d / installee
        if abs(pct) >= seuil_pct:
            rampes.append({"de": h - 1, "a": h, "variation_kw": round(d), "variation_pct": round(pct, 1),
                           "sens": "hausse" if d > 0 else "chute",
                           "niveau": "important" if abs(pct) >= 2 * seuil_pct else "attention"})
    perte = 1.0 - sum(puissance) / sum(sans_sillage) if sum(sans_sillage) > 0 else 0.0
    return {"date": eol[0]["date"], "parc": "parc virtuel de Djebel Ressas (3 eoliennes de reference)",
            "puissance_sans_sillage_kw": sans_sillage, "perte_sillage_pct": round(100.0 * perte, 2),
            "sillage": f"Jensen, k = 0,075 (terre), rotor {ROTOR_PARC_M:g} m",
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


# ---------------------------------------------------------------------- alertes du service (BF27, BF83)
STATIONS_CADENCE = {"DTTA": ("Tunis-Carthage", 1), "INM60729": ("Zaghouan", 3), "INM60728": ("Nabeul", 3)}
STATIONS_POSITION = {"DTTA": (36.8504, 10.2156), "INM60729": (36.4333, 10.0833), "INM60728": (36.45, 10.7333)}


def station_virtuelle(wf, code: str) -> dict | None:
    """Vent du modele a la place d'une station (BF37), a 10 m et a l'heure locale courante, avec l'erreur
    type de la validation croisee (station exclue, REEL1) si le run servi l'a ecrite."""
    from pcfno3d import atom_wind
    nom = STATIONS_CADENCE[code][0]
    la, lo = STATIONS_POSITION[code]
    h = float((time.gmtime(time.time() + 3600)).tm_hour)
    try:
        v = lire_point(wf, 0, 10.0, h, lat=la, lon=lo)
    except ValueError:
        return None
    err = None
    try:
        q = atom_wind.qualite(wf.cfg.paths.run_dir, wf.cfg.dso, bool(wf.cfg.inference.w_continuite))
        err = next((r["modele_m_s"] for r in q["stations"] if r["station"] == nom), None)
    except Exception:                                              # noqa: BLE001 — pas d'evaluation
        pass
    return {"station": nom, "heure_locale": h, "vitesse_m_s": v["vitesse_m_s"],
            "direction_deg": v["direction_provenance_deg"], "erreur_type_m_s": err}


def _id_alerte(*parts) -> str:
    import hashlib
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8]


def alertes_service(wf) -> list[dict]:
    """Alertes du service temps reel (meme regle que la page) : station muette selon sa cadence, heures au
    residu de masse au-dessus du seuil, vent fort aujourd'hui et demain. Une alerte a un identifiant stable
    par jour (kind | cible | date) : elle n'est notifiee qu'une fois."""
    out = []
    p = jour(wf, 0)
    I = [x for x in p["instants"] if x.get("source") == "temps reel" and isinstance(x.get("stations"), list)]
    for code, (nom, cad) in STATIONS_CADENCE.items():
        if not I:
            break
        n = 0
        for x in reversed(I):
            if code in x["stations"]:
                break
            n += 1
        if n >= 3 * cad:
            out.append({"id": _id_alerte("station_muette", code, p["date"]), "kind": "station_muette",
                        "niveau": "important" if n >= 6 * cad else "information",
                        "titre": f"station {nom} muette", "cible": nom,
                        "cause": f"aucune mesure depuis {n} h ; le vent est calcule sans elle.",
                        "recommandation": "Verifier la transmission (METAR ou SYNOP).",
                        "station_virtuelle": station_virtuelle(wf, code)})
    mauvais = [x for x in p["instants"] if x.get("residu_masse_ok") is False]
    if mauvais:
        out.append({"id": _id_alerte("physique", p["date"]), "kind": "physique", "niveau": "important",
                    "titre": "residu de masse au-dessus du seuil", "cible": "domaine",
                    "cause": f"{len(mauvais)} heure(s) ce jour, des {mauvais[0]['heure_locale']:.0f} h.",
                    "recommandation": "Consulter la vue Fiabilite ; signaler a l'equipe si cela se repete."})
    for j in (0, 1):
        try:
            a = alertes_vent(wf, j)
        except Exception:                                          # noqa: BLE001 — jour absent
            continue
        for x in a["alertes"]:
            out.append({"id": _id_alerte("vent_fort", x["hauteur_sol_m"], a["date"]), "kind": "vent_fort",
                        "niveau": "important", "titre": f"vent fort a {x['hauteur_sol_m']} m le {a['date']}",
                        "cible": "domaine",
                        "cause": f"maximum {x['max']} {x['unite']} vers {x['heure_locale']:.0f} h.",
                        "recommandation": "Eviter les travaux en hauteur et les traitements ; securiser serres et films."})
    return out


def scenario_panne(wf, heure: float, stations: list[str]) -> dict:
    """Scenario de panne d'aujourd'hui (BF88) : codes ou noms de stations a retirer."""
    from pcfno3d import atom_wind
    from pcfno3d.stations_reelles import CODES
    noms = {v.lower(): k for k, v in CODES.items()}
    codes = [noms.get(x.strip().lower(), x.strip().upper()) for x in stations if x.strip()]
    return atom_wind.scenario_panne(wf, source(wf), atom_wind.date_locale_aujourdhui(0), float(heure), codes)


# ---------------------------------------------------------------------- espaces en perspective (cahier v12, 6.11)
# Holding (BF85), chantier (BF86), environnement (BF87) : memes journees de vent que le reste de l'application,
# sans nouveau calcul du modele. Statistiques sur les journees disponibles seulement (aujourd'hui a
# apres-demain et l'historique temps reel du service) : une decision d'implantation se prend sur une annee.
HAUTEURS_PORTEUR = (15, 20, 25, 30)          # mats Porteur (cahier v12, 1.1 : 15, 20, 25 et 30 m)
RHO_SEC = 287.05                             # J/(kg K)


def _jours_disponibles(wf, historique_max: int = 60) -> list[dict]:
    """Aujourd'hui a apres-demain, puis les journees temps reel archivees (les plus recentes, au plus
    `historique_max`), une seule par date et de la meme grille que le modele servi."""
    g = grille(wf)
    out, vues = [], set()
    for j in (0, 1, 2):
        try:
            p = jour(wf, j)
        except Exception:                                       # noqa: BLE001 — jour absent
            continue
        out.append(p)
        vues.add(p["date"])
    for r in [r for r in historique(wf) if r["type"] == "temps reel"][:historique_max]:
        if r["date"] in vues:
            continue
        try:
            p = json.loads(gzip.decompress(lire_historique(wf, r["nom"])))
        except Exception:                                       # noqa: BLE001 — archive illisible : ignoree
            continue
        if list(p.get("forme", []))[-2:] == [g["ny"], g["nx"]]:
            out.append(p)
            vues.add(p["date"])
    if not out:
        raise ValueError("aucune journee de vent disponible")
    return out


def _vitesse_hauteur(wf, payload: dict, z: float) -> np.ndarray:
    """Vitesse (instants, ny, nx) a z m au-dessus du sol sur toute la grille, comme _au_niveau : lineaire
    entre niveaux, profil logarithmique sous le premier avec la rugosite de la maille."""
    arr = _tableau(payload, "vent")                              # (instants, hauteurs, 2, ny, nx)
    H = list(payload["hauteurs_agl_m"])
    if z <= H[0]:
        z0 = np.asarray(grille(wf)["z0_m"], float)
        f = np.maximum(0.0, np.log(np.maximum(z, 1.5 * z0) / z0) / np.log(H[0] / z0))
        uv = arr[:, 0] * f
    else:
        k = max(kk for kk in range(len(H) - 1) if H[kk] < z) if z < H[-1] else len(H) - 2
        a = min(1.0, (z - H[k]) / (H[k + 1] - H[k]))
        uv = np.where(np.isfinite(arr[:, k]) & np.isfinite(arr[:, k + 1]),
                      (1 - a) * arr[:, k] + a * arr[:, k + 1], arr[:, k + 1])
    return np.hypot(uv[:, 0], uv[:, 1])


def _latlon(wf, x_km, y_km):
    from pcfno3d.realdata import GeoRef
    la, lo = GeoRef.from_cfg(wf.cfg).to_latlon(np.asarray(x_km) * 1000.0, np.asarray(y_km) * 1000.0)
    return np.round(la, 5), np.round(lo, 5)


def _plan_aneo_km(wf) -> list[dict]:
    """Capteurs du plan OS6 (s'il est calcule), en km dans la grille."""
    from pcfno3d import atom_wind
    from pcfno3d.realdata import GeoRef
    f = atom_wind.plan_aneo(Path(wf.cfg.paths.run_dir))
    if f is None:
        return []
    geo = GeoRef.from_cfg(wf.cfg)
    out = []
    for ft in json.loads(f.read_text(encoding="utf-8")).get("features", []):
        if (ft.get("geometry") or {}).get("type") != "Point":
            continue
        lo, la = ft["geometry"]["coordinates"][:2]
        x, y = geo.to_xy(la, lo)
        out.append({"ordre": (ft.get("properties") or {}).get("ordre"), "lat": la, "lon": lo,
                    "x_km": float(x) / 1000.0, "y_km": float(y) / 1000.0})
    return out


def sites_porteurs(wf, hauteur_m: int = 20, n: int = 10, eolienne_w: float = 400.0, conso_w: float = 5.0,
                   ecart_km: float = 2.0, altitude_max_m: float = 650.0, marge_km: float = 1.0) -> dict:
    """Espace holding (cahier v12, BF85, /v1/holding/porteur-sites) : emplacements recommandes pour les mats
    Porteur, classes par la production de leur petite eolienne a la hauteur du mat. Terre ferme, altitude
    au plus `altitude_max_m` (regle d'OS6), `marge_km` du bord, sites espaces d'au moins `ecart_km`. Un Porteur
    porte aussi des capteurs ANEO : les capteurs du plan OS6 (s'il est calcule) sont classes de la meme facon,
    et chaque site dit a quelle distance est le capteur prevu le plus proche. Courbe de l'eolienne : la
    micro-eolienne generique d'E18 mise a l'echelle de `eolienne_w` (exemple, a remplacer par la fiche du
    fabricant ; la part solaire du Porteur n'est pas comptee)."""
    if int(hauteur_m) not in HAUTEURS_PORTEUR:
        raise ValueError(f"hauteur de mat Porteur : {', '.join(map(str, HAUTEURS_PORTEUR))} m")
    g = grille(wf)
    jours = _jours_disponibles(wf)
    V, P = MICRO_EOLIENNE["vitesse_m_s"], MICRO_EOLIENNE["puissance"]
    echelle = eolienne_w / MICRO_EOLIENNE["puissance_w"]
    vit, pw = [], []
    for p in jours:
        s = _vitesse_hauteur(wf, p, float(hauteur_m))
        vit.append(s)
        pw.append(np.where(np.isfinite(s), np.interp(np.nan_to_num(s), V, P) * echelle, np.nan))
    vit, pw = np.concatenate(vit), np.concatenate(pw)
    with np.errstate(invalid="ignore"):
        moy_w, moy_v = np.nanmean(pw, axis=0), np.nanmean(vit, axis=0)
    terrain = np.asarray(g["terrain_m"], float)
    ok = (np.asarray(g["eau"]) == 0) & (terrain <= altitude_max_m) & np.isfinite(moy_w)
    m = int(math.ceil(marge_km / g["pas_km"]))
    if m > 0:
        ok[:m, :] = ok[-m:, :] = ok[:, :m] = ok[:, -m:] = False
    jj, ii = np.nonzero(ok)
    ordre = np.argsort(-moy_w[jj, ii], kind="stable")
    plan = _plan_aneo_km(wf)
    pas = g["pas_km"]
    choisis = []
    for k in ordre:
        x, y = (ii[k] + 0.5) * pas, (jj[k] + 0.5) * pas
        if all(math.hypot(x - c[0], y - c[1]) >= ecart_km for c in choisis):
            choisis.append((x, y, int(jj[k]), int(ii[k])))
            if len(choisis) >= n:
                break

    def ligne(x, y, j, i):
        w = float(moy_w[j, i])
        la, lo = _latlon(wf, x, y)
        d = {"lat": float(la), "lon": float(lo), "x_km": round(x, 2), "y_km": round(y, 2),
             "altitude_m": round(float(terrain[j, i])), "vent_moyen_m_s": round(float(moy_v[j, i]), 2),
             "production_wh_par_jour": round(w * 24), "couverture_conso_pct": round(100.0 * w / conso_w) if conso_w > 0 else None}
        if plan:
            d["capteur_aneo_prevu_km"] = round(min(math.hypot(x - q["x_km"], y - q["y_km"]) for q in plan), 2)
        return d

    sites = [{"rang": r + 1, **ligne(*c)} for r, c in enumerate(choisis)]
    sur_plan = []
    for q in plan:
        i = min(g["nx"] - 1, max(0, int(q["x_km"] // pas)))
        j = min(g["ny"] - 1, max(0, int(q["y_km"] // pas)))
        if np.isfinite(moy_w[j, i]):
            sur_plan.append({"ordre_os6": q["ordre"], **ligne(q["x_km"], q["y_km"], j, i)})
    sur_plan.sort(key=lambda r: -r["production_wh_par_jour"])
    return {"hauteur_mat_m": int(hauteur_m), "eolienne": f"{MICRO_EOLIENNE['nom']}, mise a l'echelle de {eolienne_w:g} W",
            "dates": sorted(p["date"] for p in jours), "heures": int(vit.shape[0]),
            "regles": {"terre_ferme": True, "altitude_max_m": altitude_max_m, "marge_bord_km": marge_km,
                       "ecart_min_km": ecart_km, "conso_w": conso_w},
            "sites": sites, "plan_aneo": sur_plan or None,
            "limite": (f"{len(jours)} journee(s) de vent seulement (aujourd'hui a apres-demain et l'historique du "
                       "service) : le classement definitif se fait sur l'atlas annuel ; pente, acces, reseau et "
                       "foncier ne sont pas pris en compte (zones exclues : os6_contraintes.json)")}


def _rho(wf, payload, x_km, y_km, h, z_agl) -> float:
    """Densite de l'air a z_agl au-dessus de la maille : temperature GFS a 2 m, pression standard a l'altitude."""
    t = _surface(wf, payload, "t2m", x_km, y_km, h)
    i, j, _ = _maille(wf, x_km, y_km)
    alt = float(grille(wf)["terrain_m"][j][i]) + z_agl
    tk = (15.0 if t is None else t) + 273.15 - 0.0065 * z_agl
    return 101325.0 * math.exp(-alt / 8434.0) / (RHO_SEC * tk)


def facteur_rafale_hauteur(fr10: float | None, z0: float, z: float) -> float:
    """Facteur de rafale a z m : celui de GFS a 10 m (sinon la regle de la page selon la rugosite),
    decroissant avec la hauteur comme gustF de la page (1 + (g0 - 1) exp(-z / 300))."""
    g0 = fr10 if fr10 is not None and np.isfinite(fr10) else (1.65 if z0 > 0.5 else 1.45 if z0 > 0.05 else 1.22)
    return 1.0 + (g0 - 1.0) * math.exp(-max(z - 10.0, 0.0) / 300.0)


def vent_grue(wf, j: int, hauteur_m: float = 40.0, limite_rafale_m_s: float = 20.0, surface_charge_m2: float = 0.0,
              cf: float = 1.2, lat=None, lon=None, lieu=None) -> dict:
    """Espace chantier (cahier v12, BF86) : vent moyen et rafale a la hauteur de la fleche, heure par heure,
    verdict contre la limite de service de la grue (20 m/s en rafale par defaut, soit 72 km/h : la valeur
    usuelle des grues a tour ; celle du constructeur prime) et effort du vent sur la charge suspendue
    (0,5 rho Cf A v^2, avec la rafale). Le contrepoids et le lest se reglent avec les abaques du
    constructeur : ATOM Wind donne le vent qui y entre, pas le reglage."""
    x, y, nom = position(wf, lat, lon, lieu)
    p = jour(wf, j)
    _, _, z0 = _maille(wf, x, y)
    heures = []
    for h in range(24):
        u, v = vent(wf, p, x, y, hauteur_m, float(h))
        s = math.hypot(u, v)
        fr = facteur_rafale_hauteur(_surface(wf, p, "facteur_rafale", x, y, float(h)), z0, hauteur_m)
        raf = s * fr
        rho = _rho(wf, p, x, y, float(h), hauteur_m)
        force = 0.5 * rho * cf * surface_charge_m2 * raf ** 2
        statut = "arret" if raf >= limite_rafale_m_s else "attention" if raf >= 0.8 * limite_rafale_m_s else "ok"
        heures.append({"heure": h, "vent_m_s": round(s, 1), "rafale_m_s": round(raf, 1),
                       "rafale_km_h": round(raf * 3.6), "direction_deg": _direction(u, v), "statut": statut,
                       "ok": statut == "ok", "effort_charge_n": round(force), "effort_charge_kgf": round(force / 9.81)})
    return {"lieu": nom, "date": p["date"], "source": "prevision" if p.get("prevision") else "temps reel",
            "hauteur_fleche_m": hauteur_m, "limite_rafale_m_s": limite_rafale_m_s,
            "limite_rafale_km_h": round(limite_rafale_m_s * 3.6), "surface_charge_m2": surface_charge_m2, "cf": cf,
            "rafale_source": "GFS (facteur de rafale a 10 m)" if "facteur_rafale" in p else "regle de rugosite de la page",
            "creneaux_levage": _plages(heures, "ok", 1),
            "rafale_max_m_s": max(r["rafale_m_s"] for r in heures), "heures": heures,
            "note": ("vent moyen ATOM Wind, rafale = facteur GFS a 10 m reduit avec la hauteur ; limite et contrepoids : "
                     "notice du constructeur et procedure du chantier")}


_SECTEURS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO")
# zone abritee derriere une haie de porosite moyenne (40-50 %) : ordres de grandeur de la litterature
# agroforestiere (reduction du vent d'environ 50 % jusqu'a 5-8 H, 20 % vers 15 H, effet sensible jusqu'a
# 20-25 H sous le vent, 2-5 H au vent)
ABRI = {"forte_h": 8.0, "abri_h": 15.0, "au_vent_h": 3.0}


def _secteur(d: float) -> str:
    return _SECTEURS[int(((d % 360.0) + 11.25) // 22.5) % 16]


def plan_haies(wf, hauteur_haie_m: float = 3.0, longueur_m: float = 200.0, seuil_m_s: float = 5.0,
               hauteur_vent_m: float = 10.0, lat=None, lon=None, lieu=None) -> dict:
    """Espace environnement (cahier v12, BF87, /v1/windbreak/plan) : vent dominant au point, orientation
    des haies brise-vent et zone protegee. Les vents qui comptent sont ceux au-dessus de `seuil_m_s` a
    `hauteur_vent_m` (dessechement, verse, erosion), ponderes par v^2 (v - seuil) ; la haie se plante
    perpendiculaire a leur direction resultante. La zone protegee est un rectangle de la longueur de la haie,
    `ABRI` hauteurs de haie sous le vent (abri fort, abri) et au vent ; GeoJSON en lat/lon."""
    x, y, nom = position(wf, lat, lon, lieu)
    if lat is None or lon is None:
        la, lo = _latlon(wf, x, y)
        lat, lon = float(la), float(lo)
    jours = _jours_disponibles(wf)
    S, D = [], []
    for p in jours:
        for inst in p["instants"]:
            u, v = vent(wf, p, x, y, hauteur_vent_m, float(inst["heure_locale"]))
            S.append(math.hypot(u, v))
            D.append(_direction(u, v))
    S, D = np.asarray(S), np.asarray(D)
    poids = np.where(S >= seuil_m_s, S ** 2 * (S - seuil_m_s), 0.0)
    critere = f"vents de plus de {seuil_m_s:g} m/s a {hauteur_vent_m:g} m, ponderes par v^2 (v - seuil)"
    if poids.sum() <= 0:
        poids, critere = S ** 3, f"aucun vent au-dessus de {seuil_m_s:g} m/s : direction du vent dominant en energie (v^3)"
    r = np.radians(D)
    cx, cy = float((poids * np.sin(r)).sum()), float((poids * np.cos(r)).sum())
    dom = float(np.degrees(np.arctan2(cx, cy)) % 360.0)
    constance = math.hypot(cx, cy) / max(float(poids.sum()), 1e-9)          # 1 : une seule direction
    k = ((D % 360.0 + 11.25) // 22.5).astype(int) % 16
    rose = [{"secteur": _SECTEURS[s], "frequence_pct": round(100.0 * float((k == s).mean()), 1),
             "part_vents_forts_pct": round(100.0 * float(poids[k == s].sum() / poids.sum()), 1)} for s in range(16)]
    axe = (dom + 90.0) % 180.0                                   # azimut de la ligne de haie
    # geometrie locale en metres (est, nord) autour du point, puis lat/lon
    t = np.radians(axe)
    ax = np.array([math.sin(t), math.cos(t)])                    # le long de la haie
    sous = np.radians((dom + 180.0) % 360.0)
    dv = np.array([math.sin(sous), math.cos(sous)])              # vers ou va le vent : sous le vent
    R = 6371000.0

    def ll(pt):
        return [round(lon + math.degrees(pt[0] / (R * math.cos(math.radians(lat)))), 6),
                round(lat + math.degrees(pt[1] / R), 6)]

    def rect(de_h, a_h):
        a, b = de_h * hauteur_haie_m, a_h * hauteur_haie_m
        c = [-ax * longueur_m / 2 + dv * a, ax * longueur_m / 2 + dv * a, ax * longueur_m / 2 + dv * b,
             -ax * longueur_m / 2 + dv * b]
        return [[ll(q) for q in c + [c[0]]]]

    zones = [("abri_fort", 0.0, ABRI["forte_h"], "reduction du vent d'environ 50 %"),
             ("abri", ABRI["forte_h"], ABRI["abri_h"], "reduction du vent d'environ 20 a 50 %"),
             ("au_vent", -ABRI["au_vent_h"], 0.0, "faible reduction devant la haie")]
    geojson = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"type": "haie", "hauteur_m": hauteur_haie_m, "longueur_m": longueur_m,
                                           "azimut_deg": round(axe)},
         "geometry": {"type": "LineString", "coordinates": [ll(-ax * longueur_m / 2), ll(ax * longueur_m / 2)]}}]
        + [{"type": "Feature", "properties": {"type": z, "de_h": a, "a_h": b, "effet": e},
            "geometry": {"type": "Polygon", "coordinates": rect(a, b)}} for z, a, b, e in zones]}
    oppose = (axe + 180.0) % 360.0
    texte = (f"Vents a craindre surtout du {_secteur(dom)} ({dom:.0f} deg). Haie orientee {_secteur(axe)}-{_secteur(oppose)} "
             f"(azimut {axe:.0f} deg) ; une haie de {hauteur_haie_m:g} m abrite fortement environ "
             f"{ABRI['forte_h'] * hauteur_haie_m:.0f} m et sensiblement {ABRI['abri_h'] * hauteur_haie_m:.0f} m sous le vent, "
             f"cote {_secteur((dom + 180.0) % 360.0)}.")
    if constance < 0.4:
        texte += " Les vents forts viennent de directions variees : un maillage de haies (en L ou en carre) protege mieux."
    return {"lieu": nom, "lat": lat, "lon": lon, "dates": sorted(p["date"] for p in jours), "heures": int(S.size),
            "critere": critere, "direction_dominante_deg": round(dom), "secteur_dominant": _secteur(dom),
            "constance": round(constance, 2), "azimut_haie_deg": round(axe),
            "zone_protegee_m": {"abri_fort": ABRI["forte_h"] * hauteur_haie_m, "abri": ABRI["abri_h"] * hauteur_haie_m,
                                "au_vent": ABRI["au_vent_h"] * hauteur_haie_m},
            "rose": rose, "geojson": geojson, "texte": texte,
            "limite": (f"{len(jours)} journee(s) de vent seulement : le choix d'une plantation se fait sur au moins une "
                       "saison (historique du service ou atlas) ; distances d'abri = ordres de grandeur pour une haie de "
                       "porosite moyenne (40-50 %)")}
