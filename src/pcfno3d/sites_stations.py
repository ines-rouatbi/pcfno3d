"""Descripteurs du site d'une station (relief, rugosite, cote) : memes sources et memes formules que
la grille nationale (scripts/grille_reelle.py), calcules autour de la station seulement.

  - altitude et pente de la maille de 500 m : Copernicus GLO-30 lisse comme `grille_reelle.construire`
    (gaussienne de sigma = 0,5 dx), echantillonne sur 5 x 5 points a 500 m autour de la station,
    pente = gradient centre (np.gradient) au point central ;
  - z0 de la maille : ESA WorldCover 10 m, 5 x 5 sous-points de la maille de 500 m, table WORLDCOVER
    de grille_reelle, moyenne logarithmique (meme regle que la grille) ; classe majoritaire ;
  - TPI (position dans le relief) : altitude GLO-90 a la station moins l'altitude moyenne GLO-90 dans
    un disque de 5 km et de 20 km (mer = 0 m, tuiles absentes de Copernicus = pleine mer) ;
  - distance a la cote : plus proche sommet du trait de cote Natural Earth 1:10 M (ne_10m_coastline,
    precision de l'ordre du kilometre ; lagunes et sebkhas non comprises).

Les tuiles COG sont lues PAR PLAGES D'OCTETS (HTTP Range) : seules les tuiles internes utiles sont
telechargees, puis gardees dans un cache disque (`cache`). Les fichiers lus sont des donnees, jamais
executes. Sources : Copernicus DEM GLO-30 / GLO-90 (doi:10.5270/ESA-c5d3d65), ESA WorldCover 2021 v200
(doi:10.5281/zenodo.7254221, CC BY 4.0), Natural Earth (domaine public).
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from pathlib import Path

import numpy as np

URL_GLO30 = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
URL_GLO90 = "https://copernicus-dem-90m.s3.amazonaws.com/{n}/{n}.tif"
URL_WORLDCOVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{n}"
URL_COTE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/"
            "ne_10m_coastline.geojson")
AGENT = "pcfno3d-pfe/1.0 (these d'ingenieur, faible volume)"
DX_MAILLE = 500.0                      # maille de surface de la grille nationale (m)
RAYONS_TPI_KM = (5.0, 20.0)


class LecteurHTTP:
    """Fichier distant en lecture seule (seek / tell / read) par requetes HTTP Range, blocs de
    `bloc` octets gardes dans `cache` (un fichier par bloc). Compte les octets telecharges."""

    def __init__(self, url: str, cache: Path, bloc: int = 1 << 18, taille: int | None = None):
        self.url, self.bloc, self.pos = url, bloc, 0
        self.cache = Path(cache) / hashlib.sha1(url.encode()).hexdigest()[:16]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.octets_telecharges = 0
        self.name = url.rsplit("/", 1)[-1]
        if taille is None:
            f = self.cache / "taille"
            if f.exists():
                taille = int(f.read_text())
            else:
                with urllib.request.urlopen(urllib.request.Request(url, method="HEAD",
                                                                   headers={"User-Agent": AGENT}), timeout=60) as r:
                    taille = int(r.headers["Content-Length"])
                f.write_text(str(taille))
        self.taille = taille

    # interface de fichier binaire utilisee par tifffile
    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, pos: int, whence: int = 0) -> int:
        self.pos = pos if whence == 0 else self.pos + pos if whence == 1 else self.taille + pos
        return self.pos

    def _bloc(self, k: int) -> bytes:
        f = self.cache / f"{k}.bin"
        if f.exists():
            return f.read_bytes()
        a, b = k * self.bloc, min((k + 1) * self.bloc, self.taille) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}", "User-Agent": AGENT})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.octets_telecharges += len(data)
        tmp = f.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(f)
        return data

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.taille - self.pos
        n = max(0, min(n, self.taille - self.pos))
        if n == 0:
            return b""
        k0, k1 = self.pos // self.bloc, (self.pos + n - 1) // self.bloc
        data = b"".join(self._bloc(k) for k in range(k0, k1 + 1))
        d = self.pos - k0 * self.bloc
        self.pos += n
        return data[d:d + n]

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def close(self) -> None:
        pass


def _existe(url: str) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": AGENT}),
                                    timeout=60):
            return True
    except Exception:                                      # noqa: BLE001 (404 : tuile de pleine mer)
        return False


def fenetre_mosaique(source: str, lat_s, lat_n, lon_o, lon_e, cache: Path, compteur: dict):
    """Fenetre (lat_s..lat_n, lon_o..lon_e) d'une mosaique de tuiles Copernicus 1 x 1 degre
    (source 'glo30' ou 'glo90'), tuiles absentes (pleine mer) = 0 m. -> (tableau, lat nord, lon
    ouest, pas en degres)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from grille_reelle import lire_fenetre                     # noqa: PLC0415

    res, url = (10, URL_GLO30) if source == "glo30" else (30, URL_GLO90)
    pas = 1.0 / 3600 if source == "glo30" else 3.0 / 3600
    ny, nx = int(math.ceil((lat_n - lat_s) / pas)), int(math.ceil((lon_e - lon_o) / pas))
    out = np.zeros((ny, nx), np.float32)
    for la in range(int(math.floor(lat_s)), int(math.floor(lat_n)) + 1):
        for lo in range(int(math.floor(lon_o)), int(math.floor(lon_e)) + 1):
            n = f"Copernicus_DSM_COG_{res}_N{la:02d}_00_E{lo:03d}_00_DEM"
            u = url.format(n=n)
            cle = Path(cache) / "absentes.json"
            absentes = set(json.loads(cle.read_text())) if cle.exists() else set()
            if n in absentes:
                continue
            if not _existe(u):
                absentes.add(n)
                cle.parent.mkdir(parents=True, exist_ok=True)
                cle.write_text(json.dumps(sorted(absentes)))
                continue
            fh = LecteurHTTP(u, Path(cache) / source)
            a, b = max(lat_s, la), min(lat_n, la + 1)
            c, d = max(lon_o, lo), min(lon_e, lo + 1)
            tab, n_, o_, p_ = lire_fenetre(fh, a, b, c, d)
            compteur["octets"] = compteur.get("octets", 0) + fh.octets_telecharges
            # pas en longitude des tuiles Copernicus : 1x au sud de 50 N (ce domaine)
            r0, c0 = int(round((lat_n - n_) / pas)), int(round((o_ - lon_o) / pas))
            r1, c1 = min(r0 + tab.shape[0], ny), min(c0 + tab.shape[1], nx)
            out[max(r0, 0):r1, max(c0, 0):c1] = tab[max(-r0, 0):r1 - r0, max(-c0, 0):c1 - c0]
    return out, lat_n, lon_o, pas


def classes_worldcover(lat: np.ndarray, lon: np.ndarray, cache: Path, compteur: dict) -> np.ndarray:
    """Classe ESA WorldCover aux points (plus proche pixel, grille_reelle.echantillonner_cog) ;
    hors de toute tuile publiee : 80 (eau), comme la grille."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from grille_reelle import echantillonner_cog               # noqa: PLC0415

    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    tuiles = sorted({(3 * math.floor(a / 3), 3 * math.floor(b / 3)) for a, b in zip(lat.ravel(), lon.ravel())})
    fichiers = []
    for a, b in tuiles:
        n = f"ESA_WorldCover_10m_2021_v200_N{a:02d}E{b:03d}_Map.tif"
        u = URL_WORLDCOVER.format(n=n)
        if _existe(u):
            fichiers.append(LecteurHTTP(u, Path(cache) / "worldcover"))
    out = echantillonner_cog(fichiers, lat, lon, defaut=80) if fichiers else np.full(lat.shape, 80)
    compteur["octets"] = compteur.get("octets", 0) + sum(f.octets_telecharges for f in fichiers)
    return out


def trait_de_cote(cache: Path, compteur: dict, boite=(25.0, 42.0, 0.0, 20.0)) -> np.ndarray:
    """Sommets (lat, lon) du trait de cote Natural Earth 1:10 M dans `boite` (lat_s, lat_n, lon_o,
    lon_e) ; telecharge une fois dans le cache."""
    f = Path(cache) / "ne_10m_coastline.geojson"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(URL_COTE, headers={"User-Agent": AGENT}),
                                    timeout=120) as r:
            data = r.read()
        compteur["octets"] = compteur.get("octets", 0) + len(data)
        f.write_bytes(data)
    pts = []
    for ft in json.loads(f.read_text(encoding="utf-8"))["features"]:
        g = ft["geometry"]
        lignes = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for ln in lignes:
            for lo, la in ln:
                if boite[0] <= la <= boite[1] and boite[2] <= lo <= boite[3]:
                    pts.append((la, lo))
    return np.asarray(pts, float)


def distance_cote_km(lat: float, lon: float, cote: np.ndarray) -> float:
    if not len(cote):
        return math.nan
    p1, p2 = np.radians(lat), np.radians(cote[:, 0])
    a = (np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(cote[:, 1] - lon) / 2) ** 2)
    return float(2 * 6371.0 * np.arcsin(np.sqrt(a)).min())


def descripteurs(lat: float, lon: float, cache: Path, cote: np.ndarray | None = None,
                 compteur: dict | None = None) -> dict:
    """Descripteurs du site (voir l'en-tete du module) pour une station."""
    import sys

    from scipy.ndimage import gaussian_filter
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from grille_reelle import NOMS, WORLDCOVER, echantillonner, metres_par_degre  # noqa: PLC0415

    compteur = compteur if compteur is not None else {}
    m_lon, m_lat = metres_par_degre(lat)
    dx = DX_MAILLE
    # relief GLO-30 : fenetre de +-2 km, lissage comme la grille, 5 x 5 points a 500 m
    dla, dlo = 2000.0 / m_lat, 2000.0 / m_lon
    tab, n_, o_, pas = fenetre_mosaique("glo30", lat - dla, lat + dla, lon - dlo, lon + dlo, cache, compteur)
    lisse = gaussian_filter(tab.astype(np.float64), sigma=0.5 * dx / (pas * m_lat))
    k = np.arange(-2, 3)
    gy, gx = np.meshgrid(lat + k * dx / m_lat, lon + k * dx / m_lon, indexing="ij")
    z5 = echantillonner(lisse, n_, o_, pas, gy, gx)
    ddy, ddx = np.gradient(z5, dx, dx)
    pente = float(np.degrees(np.arctan(np.hypot(ddx[2, 2], ddy[2, 2]))))
    z_maille = float(z5[2, 2])
    z_point = float(echantillonner(tab.astype(np.float64), n_, o_, pas, np.array([lat]), np.array([lon]))[0])
    # WorldCover : 5 x 5 sous-points de la maille, z0 en moyenne logarithmique
    sous = (np.arange(5) + 0.5) / 5 - 0.5
    sy, sx = np.meshgrid(lat + sous * dx / m_lat, lon + sous * dx / m_lon, indexing="ij")
    cl = classes_worldcover(sy, sx, cache, compteur).ravel()
    z0 = float(np.exp(np.mean([math.log(WORLDCOVER.get(int(c), (2, 0.03))[1]) for c in cl])))
    codes = [WORLDCOVER.get(int(c), (2, 0.03))[0] for c in cl]
    occ = NOMS[int(np.bincount(codes, minlength=6).argmax())]
    # TPI : GLO-90 dans un disque de 5 et 20 km
    r = max(RAYONS_TPI_KM) * 1000.0
    t90, n9, o9, p9 = fenetre_mosaique("glo90", lat - r / m_lat, lat + r / m_lat, lon - r / m_lon, lon + r / m_lon,
                                       cache, compteur)
    yy = (n9 - (np.arange(t90.shape[0]) + 0.5) * p9 - lat) * m_lat
    xx = (o9 + (np.arange(t90.shape[1]) + 0.5) * p9 - lon) * m_lon
    d = np.hypot(yy[:, None], xx[None, :])
    z90 = float(echantillonner(t90.astype(np.float64), n9, o9, p9, np.array([lat]), np.array([lon]))[0])
    out = {"z_dem_point_m": round(z_point, 1), "z_maille_500m_m": round(z_maille, 1),
           "pente_maille_deg": round(pente, 2), "z0_maille_m": round(z0, 4), "occupation_maille": occ,
           "classes_worldcover": "/".join(str(int(c)) for c in sorted(set(cl)))}
    for rk in RAYONS_TPI_KM:
        out[f"tpi_{int(rk)}km_m"] = round(z90 - float(t90[d <= rk * 1000.0].mean()), 1)
    out["distance_cote_km"] = round(distance_cote_km(lat, lon, cote if cote is not None
                                                     else trait_de_cote(cache, compteur)), 1)
    return out
