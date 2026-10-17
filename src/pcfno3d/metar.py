"""Collecteur METAR (aviationweather.gov) -> tableau de stations du pipeline reel.

Source publique et gratuite, sans cle : un message METAR par aeroport, toutes les 30 ou 60
min. Il sert a TESTER le circuit des donnees reelles de bout en bout (collecte, archive au
format `import-real --stations`, envoi a `POST /api/reconstruct/stations`), pas a produire
le champ de vent : les aeroports tunisiens sont espaces de dizaines de kilometres, alors
que la v7 attend un reseau dense (docs/REAL_DATA.md).

Conversions (aucune valeur inventee ; une mesure douteuse est ecartee, jamais corrigee) :
    vent        : noeuds -> m/s (x 0,514444), direction meteo (d'ou vient le vent) ;
                  calme (0 kt) -> u = v = 0 ; « VRB » avec du vent -> direction inconnue,
                  station ecartee pour cet instant ;
    hauteur     : 10 m au-dessus du sol (anemometre standard OACI) ;
    pression    : QNH (ramenee au niveau de la mer) -> pression a l'altitude de la station
                  en atmosphere standard : p = QNH (1 - 0,0065 z / 288,15) ^ 5,25588 ;
    humidite    : depuis la temperature et le point de rosee (Magnus, realdata).
Le domaine n'est pas modifie : une station hors domaine reste hors domaine.
"""

from __future__ import annotations

import csv
import json
import math
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

URL_METAR = "https://aviationweather.gov/api/data/metar"
NOEUD_M_S = 0.514444
HAUTEUR_ANEMOMETRE_M = 10.0

# Aeroports tunisiens qui publient un METAR (verifie le 2026-10-17).
STATIONS_TUNISIE = {
    "DTTA": "Tunis-Carthage", "DTNH": "Enfidha-Hammamet", "DTMB": "Monastir",
    "DTTB": "Bizerte", "DTKA": "Tabarka", "DTTX": "Sfax", "DTTJ": "Djerba-Zarzis",
    "DTTG": "Gabes", "DTTF": "Gafsa", "DTTZ": "Tozeur", "DTTD": "Remada",
}

# Colonnes du tableau de stations de `pcfno3d import-real --stations` (realdata).
COLONNES = ["station_id", "timestamp_utc", "lat", "lon", "height_agl_m", "elevation_m",
            "wind_speed_m_s", "wind_dir_deg", "wind_u_m_s", "wind_v_m_s", "temperature_c",
            "pressure_pa", "relative_humidity_pct", "quality", "source"]


def pression_station(qnh_hpa: float, elevation_m: float) -> float:
    """Pression (Pa) a l'altitude de la station, depuis le QNH, en atmosphere standard."""
    return qnh_hpa * 100.0 * (1.0 - 0.0065 * elevation_m / 288.15) ** 5.25588


def humidite_relative(t_c: float, td_c: float) -> float:
    """Humidite relative (%) depuis la temperature et le point de rosee (Magnus)."""
    e = lambda t: 611.2 * math.exp(17.62 * t / (243.12 + t))  # noqa: E731
    return max(0.0, min(100.0, 100.0 * e(td_c) / e(t_c)))


def _instant(obs: dict) -> datetime | None:
    t = obs.get("reportTime")
    if t:
        return datetime.fromisoformat(str(t).replace("Z", "+00:00")).astimezone(timezone.utc)
    if obs.get("obsTime") is not None:
        return datetime.fromtimestamp(int(obs["obsTime"]), tz=timezone.utc)
    return None


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def metar_vers_station(obs: dict) -> tuple[dict | None, str | None]:
    """Un METAR (JSON d'aviationweather.gov) -> une ligne du tableau de stations, ou
    (None, raison) si la mesure n'est pas exploitable."""
    sid = obs.get("icaoId")
    t = _instant(obs)
    if not sid or t is None or obs.get("lat") is None or obs.get("lon") is None:
        return None, "identifiant, instant ou position absent"
    wspd, wdir = obs.get("wspd"), obs.get("wdir")
    if wspd is None:
        return None, "vent absent"
    vitesse = float(wspd) * NOEUD_M_S
    ligne = {c: None for c in COLONNES}
    ligne.update(station_id=sid, timestamp_utc=_iso(t), lat=float(obs["lat"]),
                 lon=float(obs["lon"]), height_agl_m=HAUTEUR_ANEMOMETRE_M,
                 elevation_m=float(obs.get("elev") or 0.0), quality=1.0, source="metar")
    if vitesse == 0.0:
        ligne.update(wind_u_m_s=0.0, wind_v_m_s=0.0, wind_speed_m_s=0.0)
    elif isinstance(wdir, (int, float)) and not isinstance(wdir, bool):
        ligne.update(wind_speed_m_s=round(vitesse, 3), wind_dir_deg=float(wdir) % 360.0)
    else:
        return None, f"direction variable ({wdir}) avec {wspd} kt : direction inconnue"
    if obs.get("temp") is not None:
        ligne["temperature_c"] = float(obs["temp"])
        if obs.get("dewp") is not None:
            ligne["relative_humidity_pct"] = round(
                humidite_relative(float(obs["temp"]), float(obs["dewp"])), 1)
    if obs.get("altim") is not None:
        ligne["pressure_pa"] = round(pression_station(float(obs["altim"]), ligne["elevation_m"]), 1)
    return ligne, None


def telecharger(ids: list[str], heures: int = 2, timeout: float = 20.0) -> list[dict]:
    """METAR des `heures` dernieres heures pour les aeroports `ids` (JSON)."""
    url = f"{URL_METAR}?ids={','.join(ids)}&format=json&hours={int(heures)}"
    req = urllib.request.Request(url, headers={"User-Agent": "pcfno3d-collecteur-metar/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        corps = r.read()
    return json.loads(corps) if corps.strip() else []


def instant_commun(lignes: list[dict], tolerance_min: int = 30) -> tuple[str | None, list[dict]]:
    """Instant le plus recent (arrondi a l'heure) et, par station, la mesure la plus proche
    a +/- `tolerance_min` minutes : un seul instant pour une reconstruction."""
    if not lignes:
        return None, []
    temps = [datetime.fromisoformat(l["timestamp_utc"].replace("Z", "+00:00")) for l in lignes]
    cible = max(temps).replace(minute=0, second=0, microsecond=0)
    tol = timedelta(minutes=tolerance_min)
    meilleur: dict[str, tuple[timedelta, dict]] = {}
    for l, t in zip(lignes, temps):
        ecart = abs(t - cible)
        if ecart <= tol and (l["station_id"] not in meilleur or ecart < meilleur[l["station_id"]][0]):
            meilleur[l["station_id"]] = (ecart, l)
    return _iso(cible), [v[1] for _, v in sorted(meilleur.items())]


def ajouter_archive(chemin: Path, lignes: list[dict]) -> int:
    """Ajoute les lignes au CSV d'archive (format import-real), sans doublon
    (station_id, timestamp_utc). Renvoie le nombre de lignes nouvelles."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    vues: set[tuple[str, str]] = set()
    if chemin.exists():
        with open(chemin, newline="", encoding="utf-8") as f:
            vues = {(r["station_id"], r["timestamp_utc"]) for r in csv.DictReader(f)}
    nouvelles = [l for l in lignes if (l["station_id"], l["timestamp_utc"]) not in vues]
    if nouvelles:
        neuf = not chemin.exists()
        with open(chemin, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLONNES)
            if neuf:
                w.writeheader()
            for l in sorted(nouvelles, key=lambda r: (r["timestamp_utc"], r["station_id"])):
                w.writerow({c: ("" if l.get(c) is None else l[c]) for c in COLONNES})
    return len(nouvelles)


def position_domaine(lignes: list[dict], geo, etendue_x_m: float, etendue_y_m: float) -> list[dict]:
    """Par station : coordonnees dans le repere du domaine (m) et distance au domaine
    (0 = dedans). `geo` : realdata.GeoRef. Estimation : le service tranche (maille)."""
    out = []
    for l in lignes:
        x, y = (float(v) for v in geo.to_xy(l["lat"], l["lon"]))
        dx = max(0.0, -x, x - etendue_x_m)
        dy = max(0.0, -y, y - etendue_y_m)
        out.append({"station_id": l["station_id"], "x_m": round(x), "y_m": round(y),
                    "distance_domaine_km": round(math.hypot(dx, dy) / 1000.0, 1)})
    return out


def corps_reconstruction(instant: str, lignes: list[dict]) -> dict:
    """Corps JSON de POST /api/reconstruct/stations (StationsRequest)."""
    champs = ["station_id", "lat", "lon", "height_agl_m", "wind_u_m_s", "wind_v_m_s",
              "wind_speed_m_s", "wind_dir_deg", "temperature_c", "pressure_pa",
              "relative_humidity_pct", "quality"]
    return {"timestamp_utc": instant,
            "stations": [{c: l[c] for c in champs if l.get(c) is not None} for l in lignes]}


def envoyer(base_url: str, cle: str, corps: dict, timeout: float = 300.0) -> tuple[int, dict]:
    """POST /api/reconstruct/stations ; cle d'API dans l'en-tete X-API-Key. Renvoie
    (statut HTTP, corps JSON) ; une erreur HTTP n'est pas une exception."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/reconstruct/stations",
        data=json.dumps(corps).encode("utf-8"), method="POST",
        headers={"content-type": "application/json", "X-API-Key": cle})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {"detail": str(e)}
