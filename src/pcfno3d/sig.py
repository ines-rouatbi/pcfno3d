"""Export vers les SIG (cahier v12, E15) : GeoTIFF mono-bande float32 en WGS 84 (EPSG:4326), sans
dependance (ni GDAL ni rasterio), lisible par QGIS, GDAL et les bibliotheques SIG courantes.

La grille du modele est equirectangulaire (pcfno3d.forcage.InterpERA5 : pas metrique converti en
degres a la latitude centrale) : un pas constant en longitude et en latitude, donc un GeoTIFF
« nord en haut » avec ModelPixelScale + ModelTiepoint suffit.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

NODATA = -9999.0


def grille_geo(geom) -> dict:
    """lon0, lat0 (centre de la maille (0, 0), coin sud-ouest), pas en degres, comme InterpERA5."""
    from .forcage import metres_par_degre

    real = geom.cfg.real
    if real.origin_lat is None or real.origin_lon is None:
        raise ValueError("export SIG : domaine non georeference (real.origin_lat / origin_lon)")
    if abs(float(real.rotation_deg)) > 1e-9:
        raise ValueError("export SIG : grille tournee non prise en charge")
    lat0, lon0 = float(real.origin_lat), float(real.origin_lon)
    lat_c = lat0 + geom.ny_s * geom.dy_s / 2 / metres_par_degre(lat0)[1]
    m_lon, m_lat = metres_par_degre(lat_c)
    return {"lat0": lat0, "lon0": lon0, "dlon": geom.dx / m_lon, "dlat": geom.dy / m_lat}


def ecrire_geotiff(chemin: str | Path, sud_nord: np.ndarray, lon0: float, lat0: float, dlon: float,
                   dlat: float, nodata: float = NODATA) -> Path:
    """`sud_nord` (ny, nx) : ligne 0 = la plus au SUD (convention du modele), centres de mailles a
    (lon0 + i dlon, lat0 + j dlat). Ecrit un GeoTIFF nord en haut, NaN -> nodata."""
    a = np.asarray(sud_nord, np.float32)[::-1]                      # nord en haut
    a = np.where(np.isfinite(a), a, np.float32(nodata)).astype("<f4")
    ny, nx = a.shape
    donnees = a.tobytes()
    lat_haut = lat0 + (ny - 1) * dlat
    echelle = struct.pack("<3d", dlon, dlat, 0.0)
    ancrage = struct.pack("<6d", 0.0, 0.0, 0.0, lon0 - dlon / 2, lat_haut + dlat / 2, 0.0)
    cles = struct.pack("<16H", 1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326)
    nd = (repr(float(nodata)) + "\0").encode("ascii")
    # (tag, type, nombre, valeur ou octets) ; types : 3 SHORT, 4 LONG, 12 DOUBLE, 2 ASCII
    entrees = [(256, 4, 1, nx), (257, 4, 1, ny), (258, 3, 1, 32), (259, 3, 1, 1), (262, 3, 1, 1),
               (273, 4, 1, None), (277, 3, 1, 1), (278, 4, 1, ny), (279, 4, 1, len(donnees)),
               (284, 3, 1, 1), (339, 3, 1, 3), (33550, 12, 3, echelle), (33922, 12, 6, ancrage),
               (34735, 3, 16, cles), (42113, 2, len(nd), nd)]
    n = len(entrees)
    debut_ext = 8 + 2 + n * 12 + 4
    ext = b""
    places = {}
    for tag, typ, cnt, val in entrees:
        if isinstance(val, bytes) and len(val) > 4:
            places[tag] = debut_ext + len(ext)
            ext += val + (b"\0" if len(val) % 2 else b"")
    debut_donnees = debut_ext + len(ext)
    ifd = struct.pack("<H", n)
    for tag, typ, cnt, val in entrees:
        if tag == 273:
            val = debut_donnees
        if isinstance(val, bytes):
            champ = struct.pack("<I", places[tag]) if tag in places else val.ljust(4, b"\0")
        elif typ == 3:
            champ = struct.pack("<HH", val, 0)
        else:
            champ = struct.pack("<I", val)
        ifd += struct.pack("<HHI", tag, typ, cnt) + champ
    ifd += struct.pack("<I", 0)
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_bytes(b"II*\0" + struct.pack("<I", 8) + ifd + ext + donnees)
    return chemin
