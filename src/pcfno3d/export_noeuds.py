"""Lot M6 — exports des noeuds virtuels (critere DE3 ; contrat de sortie, spec 8.4).

    NetCDF  champs 3D de dimensions (altitude, y, x) : vent, vitesse, direction, hauteur
            au-dessus du sol et, avec la fiabilite M2, sigma, couverture, fiabilite, statut ;
            latitude / longitude (projection D4 du contrat ATOM), unites, projection et
            versions en attributs.
    CSV     un noeud virtuel par ligne, colonnes de la spec 8.4 dans leur ordre :
            node_id, x_m, y_m, altitude_m, hauteur_sol_m, timestamp_utc, u, v, w, vitesse,
            direction, statut, source, sigma_u, sigma_v, sigma_w, couverture, fiabilite,
            classe_fiabilite, version_modele, version_donnees, version_normalisation.

Ecarts assumes a la spec 8.4 : `node_id` est le point_id du noeud 3D du GENERATEUR (unique),
pas un identifiant ANEO (le capteur d'un noeud mesure est dans `source`) ; u, v, w sont la
PASSE SERVIE (contrat rev. 2, D6), pas la moyenne de l'ensemble (qui degrade DSO1).

Les valeurs sont EXACTEMENT celles de l'instantane publie (`publish.ChampsPublies` : meme
masque B1, memes float32). « Chaque export est verifie par un schema de sortie avant d'etre
livre » (spec 8.4) : le CSV passe un schema Pandera, le NetCDF est relu et controle, et le
fichier n'apparait sous son nom final qu'une fois verifie.

NetCDF : netCDF4 (format NETCDF4, compression zlib) s'il est installe, sinon scipy
(NETCDF3 64 bits, sans compression) — scipy est une dependance du paquet, l'export marche
donc partout. Inconnu : NaN en float, -1 pour statut / classe (int8), cellule vide en CSV.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .publish import CALME_MS, ChampsPublies, Instantane, latlon
from .reliability import NODATA_U8, STATUT_MESURE, classe_octet
from .windfarm import round_direction

COLONNES_CSV = ("node_id", "x_m", "y_m", "altitude_m", "hauteur_sol_m", "timestamp_utc",
                "u", "v", "w", "vitesse", "direction", "statut", "source", "sigma_u",
                "sigma_v", "sigma_w", "couverture", "fiabilite", "classe_fiabilite",
                "version_modele", "version_donnees", "version_normalisation")
MESURE, ESTIME = "mesur\u00e9", "estim\u00e9"
SOURCE_MODELE = "mod\u00e8le PC-FNO3D"
CLASSES_FR = ("faible", "moyenne", "\u00e9lev\u00e9e")      # codes 0, 1, 2
DIMS = ("altitude", "y", "x")


def _contexte(inst: Instantane, geom) -> tuple[dict, dict]:
    if inst.champs is None:
        raise ValueError("instantane construit sans ses champs : construire_instantane("
                         "..., garder_champs=True)")
    meta = json.loads(inst.modele["meta.json"].decode("utf-8"))
    return meta, meta["georef"]


def version_modele(versions: dict) -> str:
    """« v7@e44d68da » : nom lisible + sha8 des poids (tracabilite DE3)."""
    return f"{versions['model']}@{versions['model_sha256'][:8]}"


# ---------------------------------------------------------------------- CSV
def noeuds_dataframe(inst: Instantane, geom) -> pd.DataFrame:
    """Les noeuds virtuels publies (z - relief >= min_agl_m), une ligne par noeud."""
    _contexte(inst, geom)
    ch = inst.champs
    k, j, i = np.nonzero(ch.valide)
    # point_id du noeud 3D du generateur (identifiant stable du dataset)
    pid_of = np.full(int(np.prod(geom.shape)), -1, np.int64)
    ok = geom.node_flat >= 0
    pid_of[geom.node_flat[ok]] = np.nonzero(ok)[0] + geom.node_id_start
    flat = (k * geom.ny + j) * geom.nx + i
    node_id = pid_of[flat]
    if (node_id < 0).any():
        raise ValueError("noeud publie sans point_id du generateur")
    z = np.asarray(geom.z_levels, float)[k]
    u, v, w = (ch.f32[c][k, j, i].astype(float) for c in ("u", "v", "w"))
    s = ch.f32["speed"][k, j, i].astype(float)
    d = round_direction(270.0 - np.degrees(np.arctan2(v, u)), 1)
    vers = inst.versions
    df = pd.DataFrame({
        "node_id": node_id, "x_m": i * float(geom.dx), "y_m": j * float(geom.dy),
        "altitude_m": z, "hauteur_sol_m": np.round(z - ch.terrain[j, i].astype(float), 2),
        "timestamp_utc": inst.valid_time,
        "u": np.round(u, 3), "v": np.round(v, 3), "w": np.round(w, 4),
        "vitesse": np.round(s, 3),
        # direction indefinie sous 0,2 m/s (contrat) : cellule vide
        "direction": np.where(s < CALME_MS, np.nan, d),
    })
    if "status" in ch.u8:
        st = ch.u8["status"][k, j, i]
        mes = st == STATUT_MESURE
        df["statut"] = np.where(mes, MESURE, ESTIME)
        src = np.array([f"c{a * geom.stride}_{b * geom.stride}" for a, b in zip(i, j)],
                       dtype=object) if mes.any() else np.array([], dtype=object)
        df["source"] = SOURCE_MODELE
        if mes.any():
            df.loc[mes, "source"] = src[mes]
        df["sigma_u"] = np.round(ch.f32["sigma_u"][k, j, i].astype(float), 3)
        df["sigma_v"] = np.round(ch.f32["sigma_v"][k, j, i].astype(float), 3)
        df["sigma_w"] = np.round(ch.f32["sigma_w"][k, j, i].astype(float), 4)
        df["couverture"] = np.round(ch.coverage[k, j, i].astype(float), 3)
        df["fiabilite"] = np.round(ch.reliability[k, j, i].astype(float), 3)
        cl = ch.reliability_class[k, j, i]
        df["classe_fiabilite"] = np.array(CLASSES_FR, dtype=object)[np.minimum(cl, 2)]
    else:
        # fiabilite M2 inactive : statut, sigma, couverture et fiabilite inconnus (B5)
        df["statut"] = None
        df["source"] = SOURCE_MODELE
        for c in ("sigma_u", "sigma_v", "sigma_w", "couverture", "fiabilite"):
            df[c] = np.nan
        df["classe_fiabilite"] = None
    df["version_modele"] = version_modele(vers)
    df["version_donnees"] = f"{vers['data']} ({vers['data_kind']})"
    df["version_normalisation"] = vers["norm"]
    return df[list(COLONNES_CSV)]


def schema_csv():
    """Schema de sortie du CSV (spec 8.4) : colonnes exactes et ordonnees, types, plages,
    valeurs permises ; inconnu = vide, jamais 0 (B5)."""
    import pandera.pandas as pa

    def reel(lo=None, hi=None, nullable=False):
        checks = []
        if lo is not None:
            checks.append(pa.Check.ge(lo))
        if hi is not None:
            checks.append(pa.Check.le(hi))
        return pa.Column(float, checks=checks, nullable=nullable, coerce=True)

    texte = pa.Column(str, nullable=False)
    return pa.DataFrameSchema({
        "node_id": pa.Column(np.int64, pa.Check.ge(0), unique=True, coerce=True),
        "x_m": reel(0.0), "y_m": reel(0.0), "altitude_m": reel(),
        "hauteur_sol_m": reel(0.0),
        "timestamp_utc": pa.Column(str, pa.Check.str_matches(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")),
        "u": reel(-80.0, 80.0), "v": reel(-80.0, 80.0), "w": reel(-80.0, 80.0),
        "vitesse": reel(0.0, 80.0),
        "direction": pa.Column(float, [pa.Check.ge(0.0), pa.Check.lt(360.0)], nullable=True,
                               coerce=True),
        "statut": pa.Column(object, pa.Check.isin([MESURE, ESTIME]), nullable=True),
        "source": texte,
        "sigma_u": reel(0.0, 40.0, True), "sigma_v": reel(0.0, 40.0, True),
        "sigma_w": reel(0.0, 40.0, True),
        "couverture": reel(0.0, 1.0, True), "fiabilite": reel(0.0, 1.0, True),
        "classe_fiabilite": pa.Column(object, pa.Check.isin(list(CLASSES_FR)), nullable=True),
        "version_modele": pa.Column(str, pa.Check.str_matches(r"^.+@[0-9a-f]{8}$")),
        "version_donnees": texte,
        "version_normalisation": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{8}$")),
    }, strict=True, ordered=True, name="noeuds_virtuels_8_4")


def exporter_csv(inst: Instantane, geom, path: str | Path) -> Path:
    """CSV des noeuds virtuels, verifie par `schema_csv` AVANT d'etre livre (fichier
    temporaire, puis renomme) ; UTF-8, fins de ligne LF, separateur virgule."""
    path = Path(path)
    df = noeuds_dataframe(inst, geom)
    schema_csv().validate(df, lazy=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            df.to_csv(f, index=False, lineterminator="\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)             # jamais de .tmp abandonne
    return path


# ---------------------------------------------------------------------- NetCDF
def _variables(inst: Instantane, geom, georef: dict) -> list[tuple]:
    """(nom, dims, tableau, attributs) de chaque variable du NetCDF."""
    ch = inst.champs
    nz, ny, nx = geom.shape
    x = np.arange(nx) * float(geom.dx)
    y = np.arange(ny) * float(geom.dy)
    xx, yy = np.meshgrid(x, y)
    lat, lon = latlon(georef, xx, yy)
    suppose = " - position SUPPOSEE (ancrage choisi, non mesure)" if georef["suppose"] else ""
    temps = (pd.Timestamp(inst.valid_time).tz_convert(None)
             - pd.Timestamp("1970-01-01")).total_seconds()
    u, v = ch.f32["u"], ch.f32["v"]
    with np.errstate(invalid="ignore"):
        d = round_direction(270.0 - np.degrees(np.arctan2(v, u)), 2).astype(np.float32)
    d[~(ch.f32["speed"] >= CALME_MS)] = np.nan
    z = np.asarray(geom.z_levels, np.float64)
    hsol = np.where(ch.valide, z[:, None, None] - ch.terrain[None].astype(np.float64),
                    np.nan).astype(np.float32)
    out = [
        ("altitude", ("altitude",), z, {"units": "m", "long_name": "altitude au-dessus du "
                                        "niveau de la mer", "positive": "up", "axis": "Z"}),
        ("y", ("y",), y, {"units": "m", "long_name": "y vers le nord depuis le noeud (0, 0)",
                          "standard_name": "projection_y_coordinate", "axis": "Y"}),
        ("x", ("x",), x, {"units": "m", "long_name": "x vers l'est depuis le noeud (0, 0)",
                          "standard_name": "projection_x_coordinate", "axis": "X"}),
        ("time", (), np.float64(temps), {"units": "seconds since 1970-01-01 00:00:00",
                                         "standard_name": "time", "calendar": "standard",
                                         "long_name": "instant des mesures sources (UTC)"}),
        ("crs", (), np.int8(0), {
            "long_name": "plan tangent equirectangulaire au noeud (0, 0) (contrat ATOM D4) ; "
                         "aucun grid_mapping_name CF ne le decrit : lat/lon 2D font foi",
            "projection": georef["projection"], "latitude_of_origin": float(georef["lat0"]),
            "longitude_of_origin": float(georef["lon0"]),
            "rotation_deg": float(georef["rotation_deg"]),
            "m_per_deg_lat": float(georef["m_per_deg_lat"]),
            "m_per_deg_lon": float(georef["m_per_deg_lon"]),
            "georef_suppose": int(bool(georef["suppose"]))}),
        ("lat", ("y", "x"), lat, {"units": "degrees_north", "standard_name": "latitude",
                                  "long_name": "latitude (projection D4 du contrat ATOM)"
                                               + suppose}),
        ("lon", ("y", "x"), lon, {"units": "degrees_east", "standard_name": "longitude",
                                  "long_name": "longitude (projection D4 du contrat ATOM)"
                                               + suppose}),
        ("z_terrain", ("y", "x"), ch.terrain, {"units": "m", "long_name": "altitude du relief",
                                               "standard_name": "surface_altitude"}),
        ("u", DIMS, u, {"units": "m s-1", "standard_name": "eastward_wind",
                        "long_name": "composante est du vent"}),
        ("v", DIMS, v, {"units": "m s-1", "standard_name": "northward_wind",
                        "long_name": "composante nord du vent"}),
        ("w", DIMS, ch.f32["w"], {"units": "m s-1", "standard_name": "upward_air_velocity",
                                  "long_name": "composante verticale du vent"}),
        ("vitesse", DIMS, ch.f32["speed"], {"units": "m s-1", "standard_name": "wind_speed",
                                            "long_name": "vitesse horizontale"}),
        ("direction", DIMS, d, {"units": "degree", "standard_name": "wind_from_direction",
                                "long_name": "direction meteorologique (d'ou vient le vent), "
                                             "indefinie sous 0,2 m/s"}),
        ("hauteur_sol", DIMS, hsol, {"units": "m", "long_name": "hauteur au-dessus du sol"}),
    ]
    if "status" in ch.u8:
        st = ch.u8["status"].astype(np.int16)
        st[st == NODATA_U8] = -1
        cl = ch.reliability_class.astype(np.int16)
        cl[cl == NODATA_U8] = -1
        for c, lib in (("sigma_u", "incertitude calibree de u"),
                       ("sigma_v", "incertitude calibree de v"),
                       ("sigma_w", "incertitude calibree de w"),
                       ("sigma_speed", "incertitude calibree de la vitesse")):
            out.append((c, DIMS, ch.f32[c], {"units": "m s-1", "long_name": lib}))
        out += [
            ("couverture", DIMS, ch.coverage, {"units": "1", "long_name": "indice de "
                                               "couverture par les capteurs visibles (0-1)"}),
            ("fiabilite", DIMS, ch.reliability, {"units": "1", "long_name": "indice de "
                                                 "fiabilite (0-1, 1 = tres fiable)"}),
            ("statut", DIMS, st.astype(np.int8), {"long_name": "1 mesure, 0 estime",
                                                  "flag_values": np.array([0, 1], np.int8),
                                                  "flag_meanings": "estime mesure"}),
            ("classe_fiabilite", DIMS, cl.astype(np.int8),
             {"long_name": "classe de fiabilite", "flag_values": np.array([0, 1, 2], np.int8),
              "flag_meanings": "faible moyenne elevee"}),
        ]
    for _, dims, _, at in out:
        if dims == DIMS:
            at["coordinates"] = "time lat lon"     # CF : temps et coordonnees geographiques
            at["grid_mapping"] = "crs"
    return out


def _attributs_globaux(inst: Instantane, meta: dict, georef: dict) -> dict:
    vers = inst.versions
    return {
        "title": "PC-FNO3D : champ de vent 3D reconstruit (noeuds virtuels, spec 8.4)",
        "Conventions": "CF-1.8",
        "snapshot_id": inst.snapshot_id, "timestamp_utc": inst.valid_time,
        "generated_at": inst.generated_at,
        "version_modele": version_modele(vers), "model": vers["model"],
        "model_sha256": vers["model_sha256"], "version_donnees": vers["data"],
        "data_kind": vers["data_kind"], "version_normalisation": vers["norm"],
        "contrat_atom": vers["contract"], "versions_json": json.dumps(vers),
        "projection": georef["projection"], "crs": "plan-tangent-local",
        "lat0": float(georef["lat0"]), "lon0": float(georef["lon0"]),
        "rotation_deg": float(georef["rotation_deg"]),
        "m_per_deg_lat": float(georef["m_per_deg_lat"]),
        "m_per_deg_lon": float(georef["m_per_deg_lon"]),
        "georef_suppose": int(bool(georef["suppose"])),
        "min_agl_m": float(meta["grid"]["min_agl_m"]),
        "nodata": "NaN (float), -1 (statut, classe_fiabilite)",
        "direction_convention": "meteo : (270 - atan2(v, u)) mod 360, d'ou vient le vent",
    }


def _ecrire_netcdf4(path: Path, variables: list[tuple], attrs: dict, shape) -> None:
    import netCDF4

    nz, ny, nx = shape
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        for nom, n in zip(DIMS, (nz, ny, nx)):
            ds.createDimension(nom, n)
        for nom, dims, a, at in variables:
            a = np.asarray(a)
            fill = np.int8(-1) if a.dtype == np.int8 else (
                np.float32(np.nan) if a.dtype == np.float32 and len(dims) == 3 else None)
            var = ds.createVariable(nom, a.dtype, dims, zlib=len(dims) == 3, complevel=4,
                                    fill_value=fill)
            var.setncatts(at)
            if dims:
                var[:] = a
            else:
                var.assignValue(a)
        ds.setncatts(attrs)


def _ecrire_scipy(path: Path, variables: list[tuple], attrs: dict, shape) -> None:
    from scipy.io import netcdf_file

    nz, ny, nx = shape
    with netcdf_file(path, "w", version=2) as ds:
        for nom, n in zip(DIMS, (nz, ny, nx)):
            ds.createDimension(nom, n)
        for nom, dims, a, at in variables:
            a = np.asarray(a)
            var = ds.createVariable(nom, a.dtype, dims)
            if dims:
                var[:] = a
            else:
                var.data[()] = a                    # scalaire (assignValue casse sous numpy 1.26)
            if a.dtype == np.int8:
                var._FillValue = np.int8(-1)
            elif a.dtype == np.float32 and len(dims) == 3:
                var._FillValue = np.float32(np.nan)
            for k, val in at.items():
                setattr(var, k, val)
        for k, val in attrs.items():
            setattr(ds, k, val)


def _lire_netcdf(path: Path) -> tuple[dict, dict, dict]:
    """(dimensions, attributs globaux, {variable: (dims, attributs)})."""
    try:
        import netCDF4
    except ImportError:
        netCDF4 = None
    if netCDF4 is not None:
        with netCDF4.Dataset(path) as ds:
            return ({k: len(d) for k, d in ds.dimensions.items()},
                    {k: ds.getncattr(k) for k in ds.ncattrs()},
                    {n: (tuple(v.dimensions), {k: v.getncattr(k) for k in v.ncattrs()})
                     for n, v in ds.variables.items()})
    from scipy.io import netcdf_file
    with netcdf_file(path, "r", mmap=False) as ds:
        dec = (lambda x: x.decode() if isinstance(x, bytes) else x)
        return (dict(ds.dimensions), {k: dec(v) for k, v in ds._attributes.items()},
                {n: (tuple(v.dimensions), {k: dec(a) for k, a in v._attributes.items()})
                 for n, v in ds.variables.items()})


def verifier_netcdf(path: str | Path, inst: Instantane, geom) -> dict:
    """Schema de sortie du NetCDF : dimensions (altitude, y, x) et tailles, variables du
    vent et coordonnees avec leurs unites, versions et projection en attributs."""
    dims, attrs, variables = _lire_netcdf(Path(path))
    nz, ny, nx = geom.shape
    erreurs = []
    if {k: dims.get(k) for k in DIMS} != {"altitude": nz, "y": ny, "x": nx}:
        erreurs.append(f"dimensions {dims} != altitude {nz}, y {ny}, x {nx}")
    for nom in ("u", "v", "w", "vitesse", "direction", "hauteur_sol", "lat", "lon",
                "altitude", "x", "y", "z_terrain", "time"):
        if nom not in variables:
            erreurs.append(f"variable {nom} absente")
            continue
        if "units" not in variables[nom][1]:
            erreurs.append(f"{nom} sans unite")
        if nom in ("u", "v", "w", "vitesse", "direction") and variables[nom][0] != DIMS:
            erreurs.append(f"{nom} : dimensions {variables[nom][0]} != {DIMS}")
        if nom in ("u", "v", "w") and variables[nom][1].get("grid_mapping") != "crs":
            erreurs.append(f"{nom} sans grid_mapping")
    for nom, sn in (("x", "projection_x_coordinate"), ("y", "projection_y_coordinate"),
                    ("time", "time")):
        if nom in variables and variables[nom][1].get("standard_name") != sn:
            erreurs.append(f"{nom} : standard_name != {sn}")
    if "crs" not in variables:
        erreurs.append("variable crs absente")
    for k in ("version_modele", "model_sha256", "version_donnees", "version_normalisation",
              "data_kind", "projection", "lat0", "lon0", "snapshot_id", "timestamp_utc"):
        if k not in attrs:
            erreurs.append(f"attribut global {k} absent")
    if attrs.get("model_sha256") not in (None, inst.versions["model_sha256"]):
        erreurs.append("model_sha256 different de l'instantane")
    if erreurs:
        raise ValueError("NetCDF non conforme : " + " ; ".join(erreurs))
    return {"dimensions": {k: dims[k] for k in DIMS}, "variables": sorted(variables)}


def exporter_netcdf(inst: Instantane, geom, path: str | Path, moteur: str | None = None
                    ) -> Path:
    """NetCDF (altitude, y, x) de l'instantane, verifie avant d'etre livre. moteur :
    "netcdf4", "scipy" ou None (netcdf4 s'il est installe)."""
    meta, georef = _contexte(inst, geom)
    if moteur is None:
        import importlib.util
        moteur = "netcdf4" if importlib.util.find_spec("netCDF4") else "scipy"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    variables = _variables(inst, geom, georef)
    attrs = _attributs_globaux(inst, meta, georef)
    ecrire = {"netcdf4": _ecrire_netcdf4, "scipy": _ecrire_scipy}[moteur]
    try:
        ecrire(tmp, variables, attrs, geom.shape)
        verifier_netcdf(tmp, inst, geom)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return path


def charger_instantane(dossier: str | Path, geom, class_high: float,
                       class_mid: float) -> Instantane:
    """Relit un instantane PUBLIE (dossier de DepotLocal) : les exports se font depuis ce
    qui est publie, jamais depuis un nouveau calcul. coverage / reliability sont relus
    quantifies (octet / 254) ; la classe de fiabilite, non publiee dans les champs, est
    recalculee depuis l'octet relu avec les seuils de la configuration
    (reliability.class_high/mid) : meme regle que reliability_class, donc meme classe.
    Refuse un dossier dont la grille ou le relief different de `geom`."""
    d = Path(dossier)
    man = json.loads((d / "publish-manifest.json").read_bytes().decode("utf-8"))
    sid = man["snapshot_id"]
    sha8 = sid[-8:]
    meta_b = (d / "models" / sha8 / "meta.json").read_bytes()
    meta = json.loads(meta_b.decode("utf-8"))
    g = meta["grid"]
    nz, ny, nx = geom.shape
    if (g["nz"], g["ny"], g["nx"]) != (nz, ny, nx) or \
            [float(z) for z in g["levels_asl_m"]] != [float(z) for z in geom.z_levels]:
        raise ValueError(f"{sid} : grille publiee differente de la geometrie du modele")
    terrain = np.fromfile(d / "models" / sha8 / "terrain.f32", "<f4").reshape(ny, nx)
    if not np.array_equal(terrain, np.asarray(geom.zt, np.float32)):
        raise ValueError(f"{sid} : relief publie different de celui du modele")
    z = np.asarray(g["levels_asl_m"], np.float64)[:, None, None]
    valide = (z - terrain.astype(np.float64)[None]) >= float(g["min_agl_m"])
    f32, u8 = {}, {}
    for c in meta["fields"]:
        enc = meta["encoding"][c]
        dt = "<f4" if enc["dtype"] == "float32" else "u1"
        a = np.fromfile(d / enc["file"], dt).reshape(nz, ny, nx)
        (u8 if dt == "u1" else f32)[c] = a.astype(np.float32) if dt == "<f4" else a
    ch = ChampsPublies(terrain, valide, float(g["min_agl_m"]), f32, u8)
    if "reliability" in u8:
        def relu(c):
            a = u8[c].astype(np.float32) * np.float32(meta["encoding"][c]["scale"])
            a[u8[c] == NODATA_U8] = np.nan
            return a
        ch.coverage, ch.reliability = relu("coverage"), relu("reliability")
        # meme regle que reliability_class : classe prise sur l'octet publie
        cl = classe_octet(u8["reliability"], class_high, class_mid)
        cl[~valide] = NODATA_U8
        ch.reliability_class = cl
    return Instantane(sid, sha8, man["valid_time"], man["generated_at"],
                      man["last_observation_at"], man["data_status"], man["versions"], {},
                      {"meta.json": meta_b}, man, {}, None, ch)


def chemins_exports(racine_exports: str | Path, sid: str) -> dict[str, Path]:
    d = Path(racine_exports) / sid
    return {"netcdf": d / f"{sid}.nc", "csv": d / f"{sid}_noeuds.csv"}


def exporter_dossier(dossier: str | Path, geom, racine_exports: str | Path,
                     class_high: float, class_mid: float, netcdf: bool = True,
                     csv: bool = True) -> dict[str, Path]:
    """Exports DE3 d'un instantane PUBLIE, seulement ceux qui MANQUENT (une relance refait
    un NetCDF ou un CSV absent, sans toucher a ceux qui existent). {nom: chemin} des
    fichiers ecrits."""
    sid = Path(dossier).name
    cibles = {k: p for k, p in chemins_exports(racine_exports, sid).items()
              if {"netcdf": netcdf, "csv": csv}[k] and not p.exists()}
    if not cibles:
        return {}
    inst = charger_instantane(dossier, geom, class_high, class_mid)
    out = {}
    if "netcdf" in cibles:
        out["netcdf"] = exporter_netcdf(inst, geom, cibles["netcdf"])
    if "csv" in cibles:
        out["csv"] = exporter_csv(inst, geom, cibles["csv"])
    return out


def exporter(inst: Instantane, geom, dossier: str | Path, netcdf: bool = True,
             csv: bool = True) -> dict[str, Path]:
    """Les deux exports DE3 d'un instantane dans `dossier` (<id>.nc, <id>_noeuds.csv)."""
    d = Path(dossier)
    out = {}
    if netcdf:
        out["netcdf"] = exporter_netcdf(inst, geom, d / f"{inst.snapshot_id}.nc")
    if csv:
        out["csv"] = exporter_csv(inst, geom, d / f"{inst.snapshot_id}_noeuds.csv")
    return out
