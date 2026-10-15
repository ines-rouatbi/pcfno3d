"""Contrat de donnees ANEO (cahier des charges, section 8) : schema, coherence, qualite.

Le controle est branche a deux endroits : la commande `pcfno3d validate` (avec mise en
quarantaine optionnelle) et l'upload de l'API (`POST /api/reconstruct/upload`, seul
chemin ou le rapport est joint a la reconstruction). `build-cache` ne l'appelle PAS :
valider les fichiers journaliers avant de construire le cache reste une etape manuelle.
Il en sort un rapport JSON lisible et un verdict :

    conforme      le fichier peut entrer dans la chaine (avertissements eventuels)
    quarantaine   structure correcte mais qualite degradee : un humain decide
                  (a l'upload : reconstruction servie, statut_qualite = quarantaine)
    rejet         le fichier viole le contrat (critere DE1) : il n'entre pas

Trois etages, tous en validation PARESSEUSE (lazy) : on collecte TOUTES les erreurs au
lieu de s'arreter a la premiere.

    1. schema brut (Pandera)   colonnes presentes, valeurs obligatoires, types lisibles
    2. schema type (Pandera)   plages 8.1, mask_i_obs, source_type, regles 8.2 ligne a ligne
    3. indicateurs 8.3          stations valides, interpolees, hors plage, muettes, couverture

Definitions retenues (voir docs/CONTRAT_ANEO.md) :

- station : point de surface, point_id < point_id_debut (meta.json). Les noeuds 3D
  (source ANEO_VIRTUAL_NODE_3D) sont controles par le schema mais exclus des indicateurs.
- valeur interpolee ou signalee : mask_i_obs = 0, OU source interpolee, OU
  aneo_quality_score < QUALITE_SIGNALEE.
- absence : couple (station, instant) sans ligne, ou avec u, v ou T manquant.
- valeur figee : meme valeur exacte de u, v ou T sur une serie d'au moins
  N_INSTANTS_FIGEE instants consecutifs couvrant au moins DUREE_FIGEE_S ; pour le vent,
  le calme (vitesse < VITESSE_CALME) ne compte pas (u = v = 0 est une mesure plausible).
- station muette : au moins une absence, OU une valeur figee. Profil upload
  (`Seuils.upload()`) : une station sans AUCUNE ligne est une panne connue (« absente »),
  ni muette ni comptee dans le seuil de rejet des stations valides.
- station valide (a un instant) : presente, mesuree (ni interpolee ni signalee), toutes
  variables dans leur plage, densite coherente, non figee.
- couverture : part de stations valides par tuile carree de TUILE_M metres.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CONTRAT_VERSION = "aneo-8/1"

# =============================================================================== SEUILS
# Tolerance de la loi des gaz parfaits : |rho - p / (Rd Tv)| / (p / (Rd Tv)).
# 2 % ~ 6 K d'erreur sur T, ou 2 kPa sur p. Avec Tv (formule du generateur), le residu
# de conforme.csv est < 0,04 % (arrondis) ; le mini-dataset (densite d'air SEC) < 0,35 %.
# L'ecart air sec / air humide atteint ~2 % a 35 degC sature : d'ou Tv.
TOL_DENSITE_REL = 0.02
# Pas de temps attendu : tout instant est un multiple de 10 s depuis minuit UTC (pas le
# plus fin du generateur : episodes 10 s ; les episodes minute et les heures synoptiques
# (3 h) en sont des multiples).
PAS_ATTENDU_S = 10
# Valeur figee : meme valeur exacte sur au moins 3 instants consecutifs ET 30 min. En
# duree, pas en nombre d'instants : a 10 s, trois valeurs egales (quantification a 0,1)
# sont banales. Le calme (vitesse < 0,5 m/s) n'est pas un anemometre bloque.
N_INSTANTS_FIGEE = 3
DUREE_FIGEE_S = 1800.0
VITESSE_CALME = 0.5
VARS_FIGEES = ("wind_u_m_s", "wind_v_m_s", "temperature_c")
# Score de qualite en dessous duquel une valeur est « signalee » (= real.min_quality).
QUALITE_SIGNALEE = 0.5
# Cote des tuiles de couverture spatiale (densite locale de stations valides).
TUILE_M = 1000.0

# --- rejet : le fichier n'entre pas dans la chaine
REJET_PART_ANOMALIES = 0.01          # > 1 % des lignes STATIONS hors plage ou densite
                                     # incoherente (les noeuds 3D ne diluent pas le seuil)
REJET_STATIONS_VALIDES = 0.30        # < 30 % de couples (station, instant) valides
REJET_ANOMALIES_MIN_LIGNES_UPLOAD = 5  # upload : rejet si anomalies > 1 % ET > 5 lignes
# --- quarantaine : structure correcte, qualite a revoir par un humain
QUARANTAINE_STATIONS_VALIDES = 0.70  # < 70 % de stations valides au pire instant
QUARANTAINE_INTERPOLEES = 0.25       # > 25 % de valeurs interpolees ou signalees
QUARANTAINE_MUETTES = 0.05           # > 5 % de stations muettes
QUARANTAINE_TUILES_VIDES = 0.20      # > 20 % de tuiles sans station valide (pire instant)
# Toute anomalie de mesure (hors plage, densite), meme isolee, met en quarantaine.
# --- avertissement : le fichier passe, le rapport le signale
AVERT_INTERPOLEES = 0.05             # > 5 % de valeurs interpolees ou signalees


@dataclass
class Seuils:
    """Seuils du verdict ; les valeurs par defaut sont les constantes ci-dessus.

    absences_connues : profil upload. Une station sans ligne, ou dont la ligne se declare
    non mesuree (mask_i_obs = 0, source interpolee) ou n'a pas u/v/T, est une PANNE CONNUE
    (c'est l'encodage du dataset pour un capteur manquant) : elle n'est pas muette, et le
    seuil de REJET des stations valides ne compte que les lignes qui se declarent mesurees.
    rejet_anomalies_min_lignes : plancher absolu ; le rejet pour anomalies exige aussi
    STRICTEMENT plus de lignes que ce plancher (un instant de 50 stations ne doit pas etre
    rejete pour une seule station fausse : elle est masquee, le fichier en quarantaine).
    """

    tol_densite_rel: float = TOL_DENSITE_REL
    pas_attendu_s: int = PAS_ATTENDU_S
    n_instants_figee: int = N_INSTANTS_FIGEE
    duree_figee_s: float = DUREE_FIGEE_S
    vitesse_calme: float = VITESSE_CALME
    absences_connues: bool = False
    qualite_signalee: float = QUALITE_SIGNALEE
    tuile_m: float = TUILE_M
    rejet_part_anomalies: float = REJET_PART_ANOMALIES
    rejet_anomalies_min_lignes: int = 0
    rejet_stations_valides: float = REJET_STATIONS_VALIDES
    quarantaine_stations_valides: float = QUARANTAINE_STATIONS_VALIDES
    quarantaine_interpolees: float = QUARANTAINE_INTERPOLEES
    quarantaine_muettes: float = QUARANTAINE_MUETTES
    quarantaine_tuiles_vides: float = QUARANTAINE_TUILES_VIDES
    avert_interpolees: float = AVERT_INTERPOLEES

    @classmethod
    def upload(cls) -> Seuils:
        """Profil de `POST /api/reconstruct/upload` : un instant, souvent un sous-ensemble
        de stations (panne partielle, DSO5). Le modele reconstruit avec les capteurs
        presents (masque a 0 ailleurs) : seul un fichier faux doit etre refuse."""
        return cls(absences_connues=True,
                   rejet_anomalies_min_lignes=REJET_ANOMALIES_MIN_LIGNES_UPLOAD)


# =============================================================================== SCHEMA 8.1
# Plages physiques du cahier (8.1), proposees pour le domaine pilote.
PLAGES: dict[str, tuple[float, float]] = {
    "temperature_c": (-20.0, 55.0),
    "potential_temperature_k": (250.0, 340.0),
    "air_density_kg_m3": (0.9, 1.4),
    "relative_humidity_pct": (0.0, 100.0),
    "water_vapor_pressure_pa": (0.0, 7000.0),
    "pressure_pa": (80000.0, 106000.0),
    "wind_u_m_s": (-60.0, 60.0),
    "wind_v_m_s": (-60.0, 60.0),
    "wind_w_m_s": (-10.0, 10.0),
    "aneo_quality_score": (0.0, 1.0),
}
ENTIERS = ("point_id", "day_index", "hour_utc", "mask_i_obs")
# ordre des colonnes des fichiers du generateur (generate_aneo_3d.OBS_COLUMNS)
COLONNES = ["point_id", "timestamp_utc", "day_index", "hour_utc", "temperature_c",
            "potential_temperature_k", "air_density_kg_m3", "relative_humidity_pct",
            "water_vapor_pressure_pa", "pressure_pa", "wind_u_m_s", "wind_v_m_s",
            "wind_w_m_s", "aneo_quality_score", "mask_i_obs", "source_type"]
# colonnes dont l'absence de valeur rend la ligne inexploitable (erreur bloquante)
OBLIGATOIRES = ("point_id", "timestamp_utc", "day_index", "hour_utc", "mask_i_obs",
                "source_type")
# role de chaque type de source : station mesuree, valeur interpolee, noeud virtuel 3D
SOURCES_DEFAUT = {"ANEO_PHYSICAL_SYNTHETIC_MULTITIME": "station",
                  "ANEO_PHYSICAL_INTERPOLATED_MULTITIME": "interpolee",
                  "ANEO_VIRTUAL_NODE_3D": "noeud_3d"}
ROLES_META = {"capteur": "station", "interpole": "interpolee", "noeud_3d": "noeud_3d"}
# aneo_observations_day_004_20250107_multitime.csv.gz -> jour 4, 2025-01-07
NOM_JOURNALIER = re.compile(r"aneo_observations_day_(\d+)_(\d{8})_")

RD, RV = 287.05, 461.5            # J/(kg.K) ; memes constantes que realdata.py
EPS = RD / RV

BLOQUANT, MESURE = "bloquant", "mesure"
# identifiant de regle -> (section du cahier, gravite, message)
REGLES: dict[str, tuple[str, str, str]] = {
    "fichier_illisible": ("8.1", BLOQUANT, "fichier CSV illisible"),
    "fichier_vide": ("8.1", BLOQUANT, "fichier sans aucune ligne de donnees"),
    "separateur_invalide": ("8.1", BLOQUANT,
                            "separateur de champs invalide : le contrat attend la virgule"),
    "jour_multiple": ("8.1", BLOQUANT,
                      "le fichier couvre plusieurs dates UTC (un fichier = un jour)"),
    "ligne_malformee": ("8.1", BLOQUANT, "ligne au nombre de champs incorrect"),
    "colonne_absente": ("8.1", BLOQUANT, "colonne obligatoire absente"),
    "valeur_manquante": ("8.1", BLOQUANT, "valeur obligatoire manquante"),
    "type_entier": ("8.1", BLOQUANT, "entier attendu"),
    "type_reel": ("8.1", BLOQUANT, "reel attendu"),
    "type_horodatage": ("8.1", BLOQUANT, "horodatage ISO 8601 attendu"),
    "hors_plage": ("8.1", MESURE, "valeur hors de la plage physique"),
    "mask_i_obs_invalide": ("8.1", BLOQUANT, "mask_i_obs doit valoir 0 ou 1"),
    "source_non_declaree": ("8.1", BLOQUANT, "source_type absent de la liste declaree"),
    "point_id_inconnu": ("8.2", BLOQUANT, "point_id hors de la grille declaree"),
    "source_incoherente": ("8.2", BLOQUANT,
                           "source_type incompatible avec le point (station / noeud 3D)"),
    "doublon": ("8.2", BLOQUANT, "doublon du couple (point_id, timestamp_utc)"),
    "horodatage_decroissant": ("8.2", BLOQUANT, "horodatage anterieur a une ligne precedente"),
    "pas_incompatible": ("8.2", BLOQUANT, "instant non aligne sur le pas de temps attendu"),
    "hour_utc_incoherent": ("8.1", BLOQUANT, "hour_utc different de l'heure de timestamp_utc"),
    "day_index_incoherent": ("8.1", BLOQUANT,
                             "day_index et date de timestamp_utc ne se correspondent pas"),
    "jour_du_fichier": ("8.1", BLOQUANT,
                        "timestamp_utc ou day_index different du jour du nom de fichier"),
    "densite_incoherente": ("8.2", MESURE,
                            "densite incompatible avec p et T (gaz parfaits, Tv)"),
}
# colonnes montrees dans les exemples des regles qui portent sur plusieurs colonnes
COLONNES_EXEMPLE = {
    "doublon": ["point_id", "timestamp_utc"],
    "horodatage_decroissant": ["point_id", "timestamp_utc"],
    "pas_incompatible": ["timestamp_utc"],
    "hour_utc_incoherent": ["timestamp_utc", "hour_utc"],
    "day_index_incoherent": ["timestamp_utc", "day_index"],
    "jour_multiple": ["point_id", "timestamp_utc"],
    "jour_du_fichier": ["timestamp_utc", "day_index"],
    "source_incoherente": ["point_id", "source_type"],
    "densite_incoherente": ["pressure_pa", "temperature_c", "air_density_kg_m3"],
}
N_EXEMPLES = 5
N_IDS = 100                       # listes d'identifiants tronquees dans le rapport


# =============================================================================== GRILLE
@dataclass
class Grille:
    """Grille declaree (meta.json) : stations 1..nx*ny, puis noeuds 3D consecutifs.

    Les stations sont numerotees ligne par ligne : point_id = iy * nx + ix + 1, comme dans
    le generateur et le mini-dataset.
    """

    nx_s: int
    ny_s: int
    dx_s: float
    dy_s: float
    node_id_start: int
    n_noeuds: int
    sources: dict[str, str] = field(default_factory=lambda: dict(SOURCES_DEFAUT))

    @property
    def n_stations(self) -> int:
        return self.nx_s * self.ny_s

    @classmethod
    def depuis_meta(cls, meta: dict | str | Path) -> Grille:
        """meta.json (dict, fichier, ou dossier du dataset) -> Grille."""
        if not isinstance(meta, dict):
            path = Path(meta)
            if path.is_dir():
                path = path / "meta.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
        surface, nodes = meta["grille_surface"], meta["noeuds_3d"]
        declared = meta.get("source_type")
        sources = ({code: ROLES_META.get(key, key) for key, code in declared.items()}
                   if declared else dict(SOURCES_DEFAUT))
        return cls(nx_s=int(surface["nx"]), ny_s=int(surface["ny"]),
                   dx_s=float(surface["dx_m"]), dy_s=float(surface["dy_m"]),
                   node_id_start=int(nodes["point_id_debut"]),
                   n_noeuds=int(nodes.get("nombre", 0)), sources=sources)

    def connu(self, pid: pd.Series) -> pd.Series:
        """point_id appartient a la grille declaree (stations ou noeuds 3D)."""
        p = pd.to_numeric(pid, errors="coerce")
        surface = (p >= 1) & (p <= self.n_stations)
        nodes = (p >= self.node_id_start) & (p < self.node_id_start + self.n_noeuds)
        return (surface | nodes) & (p % 1 == 0)




# =============================================================================== LECTURE
TAILLE_LOT = 16 << 20                 # octets lus par lot (lecture en flux)
RELECTURE_MAX = 64 << 20              # au-dela (8 Mo si .gz), pas de relecture pandas/python


def _entete(path: Path) -> str:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        return f.readline().strip("\r\n")


def _lire_par_lots(path: Path, types: dict, node_id_start: int | None) -> pd.DataFrame:
    """Lecture EN FLUX : chaque lot est converti puis filtre (surface_seule) avant le
    suivant, pour ne jamais tenir en memoire les ~1,35 M lignes de noeuds 3D par instant
    d'un fichier journalier complet."""
    import pyarrow.csv as pacsv

    reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=TAILLE_LOT),
                            convert_options=pacsv.ConvertOptions(column_types=types))
    frames, offset = [], 0
    for batch in reader:
        df = batch.to_pandas()
        df.index = np.arange(offset, offset + len(df))
        offset += len(df)
        if node_id_start is not None and "point_id" in df.columns:
            df = df[~(pd.to_numeric(df["point_id"], errors="coerce") >= node_id_start)]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=reader.schema.names)
    return pd.concat(frames) if len(frames) > 1 else frames[0]


def lire_csv(path: str | Path, surface_seule: bool = False,
             node_id_start: int | None = None) -> tuple[pd.DataFrame | None, list[dict]]:
    """CSV (eventuellement .gz) -> DataFrame BRUT, sans rien convertir de force.

    1. lecture en flux typee (pyarrow infere les types sur le premier lot) ;
    2. si un lot contredit ces types (« abc » dans une colonne reelle), relecture en flux
       avec toutes les colonnes en texte : c'est le schema qui signale la valeur ;
    3. si des lignes n'ont pas le bon nombre de champs, relecture tolerante par pandas qui
       les collecte — seulement pour un petit fichier (RELECTURE_MAX) ; un gros fichier
       est declare illisible avec le message de pyarrow.

    L'index du DataFrame est le numero de ligne de donnees (0 = ligne 2 du fichier) :
    les erreurs du rapport citent les lignes du fichier, meme apres `surface_seule`.
    """
    import pyarrow as pa_

    path = Path(path)
    filtre = node_id_start if surface_seule else None
    try:
        entete = _entete(path)
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        return None, [_erreur("fichier_illisible", None, 1, [], [str(exc)[:300]])]
    if not entete:
        return None, [_erreur("fichier_vide", None, 1, [], ["fichier sans en-tete"])]
    for sep in (";", "\t", "|"):
        if entete.count(sep) > entete.count(","):
            return None, [_erreur("separateur_invalide", None, 1, [1],
                                  [f"separateur {sep!r} detecte dans l'en-tete : {entete[:120]}"])]
    noms = next(csv.reader([entete]))

    df, err = None, None
    for types in ({"timestamp_utc": pa_.string(), "source_type": pa_.string()},
                  {n: pa_.string() for n in noms}):
        try:
            df = _lire_par_lots(path, types, filtre)
            break
        except (pa_.ArrowException, OSError, ValueError) as exc:
            err = exc
    erreurs: list[dict] = []
    if df is None:
        limite = RELECTURE_MAX // 8 if path.name.endswith(".gz") else RELECTURE_MAX
        if path.stat().st_size > limite:
            return None, [_erreur("fichier_illisible", None, 1, [], [str(err)[:300]])]
        mauvaises: list[list[str]] = []
        try:
            df = pd.read_csv(path, dtype=str, engine="python",
                             on_bad_lines=lambda bad: mauvaises.append(bad) or None)
        except Exception as exc:                         # noqa: BLE001 - tout est rapporte
            return None, [_erreur("fichier_illisible", None, 1, [], [str(exc)[:300]])]
        if mauvaises:
            erreurs.append(_erreur("ligne_malformee", None, len(mauvaises), [],
                                   [",".join(b) for b in mauvaises[:N_EXEMPLES]]))
        if filtre is not None and "point_id" in df.columns:
            df = df[~(pd.to_numeric(df["point_id"], errors="coerce") >= filtre)]
    if len(df) == 0:
        erreurs.append(_erreur("fichier_vide", None, 1, [], [entete[:120]]))
        return None, erreurs
    return df, erreurs


# =============================================================================== CONTROLES
def _num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return pd.Series(np.nan, index=s.index)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    return pd.to_numeric(s, errors="coerce").astype("float64")


def parse_horodatage(s: pd.Series) -> pd.Series:
    """ISO 8601 (avec ou sans « T », avec ou sans fuseau) -> datetime UTC naif ; NaT sinon.

    Sans fuseau, l'heure est lue comme UTC (format des fichiers du generateur).
    """
    out = pd.to_datetime(s.astype("string"), format="ISO8601", utc=True, errors="coerce")
    return out.dt.tz_localize(None)


def _lisible_reel(s: pd.Series) -> pd.Series:
    return _num(s).notna() | s.isna()


def _lisible_entier(s: pd.Series) -> pd.Series:
    v = _num(s)
    return (v.notna() & (v % 1 == 0)) | s.isna()


def _lisible_horodatage(s: pd.Series) -> pd.Series:
    return parse_horodatage(s).notna() | s.isna()


def pression_vapeur_saturante(t_c):
    """Magnus (Pa), T en degres C — meme formule que realdata.py."""
    t_c = np.asarray(t_c, float)
    return 611.2 * np.exp(17.62 * t_c / (243.12 + t_c))


def densite_gaz_parfaits(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """rho = p / (Rd Tv), Tv = T / (1 - (e/p)(1 - Rd/Rv)).

    e : water_vapor_pressure_pa s'il est present et plausible (0 <= e < 0,2 p), sinon
    deduit de relative_humidity_pct (Magnus), sinon 0 (air sec). Renvoie aussi le nombre de
    lignes traitees de chaque facon.
    """
    p = df["pressure_pa"]
    t_c = df["temperature_c"]
    nan = pd.Series(np.nan, index=df.index)
    e = df["water_vapor_pressure_pa"] if "water_vapor_pressure_pa" in df else nan
    e = e.where((e >= 0) & (e < 0.2 * p))
    rh = df["relative_humidity_pct"] if "relative_humidity_pct" in df else nan
    e_rh = (rh.clip(0, 100) / 100.0 * pression_vapeur_saturante(t_c)).where(rh.notna())
    e_used = e.fillna(e_rh)
    modes = {"pression_de_vapeur": int(e.notna().sum()),
             "humidite_relative": int((e.isna() & e_rh.notna()).sum()),
             "air_sec": int(e_used.isna().sum())}
    tv = (t_c + 273.15) / (1.0 - e_used.fillna(0.0) / p * (1.0 - EPS))
    return p / (RD * tv), modes


def _ecart_densite(df: pd.DataFrame) -> pd.Series:
    if not {"pressure_pa", "temperature_c", "air_density_kg_m3"} <= set(df.columns):
        return pd.Series(np.nan, index=df.index)
    rho_gp, _ = densite_gaz_parfaits(df)
    return ((df["air_density_kg_m3"] - rho_gp) / rho_gp).abs()


def _sans_doublon(df: pd.DataFrame) -> pd.Series:
    ok_in = df["point_id"].notna() & df["_ts"].notna()
    dup = df.duplicated(["point_id", "_ts"], keep="first")
    return ~(dup & ok_in)


def _croissant(df: pd.DataFrame) -> pd.Series:
    ts = df["_ts"]
    prev = ts.cummax().ffill().shift(1)
    return ts.isna() | prev.isna() | (ts >= prev)


def _aligne(df: pd.DataFrame, pas_s: int) -> pd.Series:
    ts = df["_ts"]
    secs = (ts - ts.dt.normalize()).dt.total_seconds()
    return ts.isna() | ((secs % pas_s) == 0)


def _heure_coherente(df: pd.DataFrame) -> pd.Series:
    ts, h = df["_ts"], df["hour_utc"]
    return ts.isna() | h.isna() | (h == ts.dt.hour)


def _jour_coherent(df: pd.DataFrame) -> pd.Series:
    """day_index <-> date : une seule date par day_index et un seul day_index par date.

    Le generateur repartit ses jours sur l'annee (date = 1er janvier + i * 365 / n) :
    day_index n'est ni le jour de l'annee ni un decalage de dates, seule la bijection est
    exigible. En cas de conflit, le couple majoritaire est tenu pour juste.
    """
    d, di = df["_ts"].dt.normalize(), df["day_index"]
    known = d.notna() & di.notna()
    if not known.any():
        return pd.Series(True, index=df.index)
    pairs = pd.DataFrame({"d": d[known], "i": di[known]}).value_counts()
    good = (set(pairs.groupby(level=0).idxmax().values)
            & set(pairs.groupby(level=1).idxmax().values))
    rows = pd.MultiIndex.from_arrays([d, di])
    return pd.Series(rows.isin(list(good)), index=df.index) | ~known


def _jour_unique(df: pd.DataFrame) -> pd.Series:
    """Un fichier = un jour UTC : les lignes d'une autre date que la date majoritaire
    sont signalees, que le nom du fichier suive ou non le motif journalier."""
    d = df["_ts"].dt.normalize()
    if d.notna().sum() == 0:
        return pd.Series(True, index=df.index)
    return d.isna() | (d == d.mode().iloc[0])


def _jour_du_fichier(df: pd.DataFrame, jour: tuple[int, pd.Timestamp]) -> pd.Series:
    day, date = jour
    ts, di = df["_ts"], df["day_index"]
    ok_date = ts.isna() | (ts.dt.normalize() == date)
    ok_day = di.isna() | (di == day)
    return ok_date & ok_day


def _source_coherente(df: pd.DataFrame, grille: Grille) -> pd.Series:
    role = df["source_type"].map(grille.sources)
    pid = df["point_id"]
    is_node = pid >= grille.node_id_start
    hors = role.isna() | pid.isna() | ~grille.connu(pid)
    return hors | ((role == "noeud_3d") == is_node)


def jour_depuis_nom(nom: str | None) -> tuple[int, pd.Timestamp] | None:
    """aneo_observations_day_004_20250107_multitime.csv.gz -> (4, 2025-01-07)."""
    m = NOM_JOURNALIER.search(Path(nom).name) if nom else None
    if not m:
        return None
    return int(m.group(1)), pd.Timestamp(dt.datetime.strptime(m.group(2), "%Y%m%d"))


def schema_brut():
    """Etage 1 (Pandera) : colonnes presentes, valeurs obligatoires, types lisibles."""
    import pandera.pandas as pa

    cols = {}
    for c in COLONNES:
        if c in ENTIERS:
            checks = [pa.Check(_lisible_entier, error="type_entier")]
        elif c in PLAGES:
            checks = [pa.Check(_lisible_reel, error="type_reel")]
        elif c == "timestamp_utc":
            checks = [pa.Check(_lisible_horodatage, error="type_horodatage")]
        else:
            checks = []
        cols[c] = pa.Column(checks=checks, nullable=c not in OBLIGATOIRES, required=True)
    return pa.DataFrameSchema(cols, strict=False, name="aneo_brut")


def schema_type(grille: Grille, presentes: set[str], seuils: Seuils | None = None,
                jour: tuple[int, pd.Timestamp] | None = None):
    """Etage 2 (Pandera) sur les colonnes converties : plages 8.1 et regles 8.2.

    Construit d'apres les colonnes presentes : une colonne absente est deja une erreur de
    l'etage 1, et une regle qui en depend serait sans objet.
    """
    import pandera.pandas as pa

    seuils = seuils or Seuils()
    cols: dict = {}
    for c, (lo, hi) in PLAGES.items():
        if c in presentes:
            cols[c] = pa.Column(checks=pa.Check(lambda s, lo=lo, hi=hi: s.between(lo, hi),
                                                error="hors_plage"), nullable=True)
    if "point_id" in presentes:
        cols["point_id"] = pa.Column(checks=pa.Check(grille.connu, error="point_id_inconnu"),
                                     nullable=True)
    if "mask_i_obs" in presentes:
        cols["mask_i_obs"] = pa.Column(checks=pa.Check(lambda s: s.isin([0, 1]),
                                                       error="mask_i_obs_invalide"),
                                       nullable=True)
    if "source_type" in presentes:
        declared = list(grille.sources)
        cols["source_type"] = pa.Column(checks=pa.Check(lambda s: s.isin(declared),
                                                        error="source_non_declaree"),
                                        nullable=True)
    has = presentes | ({"_ts"} if "timestamp_utc" in presentes else set())
    checks = []
    if {"point_id", "_ts"} <= has:
        checks.append(pa.Check(_sans_doublon, error="doublon"))
    if "_ts" in has:
        checks.append(pa.Check(_croissant, error="horodatage_decroissant"))
        checks.append(pa.Check(_jour_unique, error="jour_multiple"))
        checks.append(pa.Check(lambda df: _aligne(df, seuils.pas_attendu_s),
                               error="pas_incompatible"))
    if {"_ts", "hour_utc"} <= has:
        checks.append(pa.Check(_heure_coherente, error="hour_utc_incoherent"))
    if {"_ts", "day_index"} <= has:
        checks.append(pa.Check(_jour_coherent, error="day_index_incoherent"))
        if jour is not None:
            checks.append(pa.Check(lambda df: _jour_du_fichier(df, jour),
                                   error="jour_du_fichier"))
    if {"point_id", "source_type"} <= has:
        checks.append(pa.Check(lambda df: _source_coherente(df, grille),
                               error="source_incoherente"))
    if {"pressure_pa", "temperature_c", "air_density_kg_m3"} <= has:
        tol = seuils.tol_densite_rel
        checks.append(pa.Check(lambda df: ~(_ecart_densite(df) > tol),
                               error="densite_incoherente"))
    return pa.DataFrameSchema(cols, checks=checks, strict=False, name="aneo_type")


# =============================================================================== ERREURS
def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        return v.isoformat()
    if v is None or isinstance(v, (int, str, bool)):
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _erreur(regle: str, colonne: str | None, nombre: int, lignes: list[int],
            exemples: list) -> dict:
    section, gravite, message = REGLES[regle]
    return {"regle": regle, "section": section, "gravite": gravite, "message": message,
            "colonne": colonne, "nombre": int(nombre),
            "lignes": [int(x) for x in lignes[:2 * N_EXEMPLES]],
            "exemples": [_jsonable(x) for x in exemples[:N_EXEMPLES]]}


def _erreurs_pandera(exc, brut: pd.DataFrame) -> list[dict]:
    """failure_cases de Pandera -> erreurs du rapport, regroupees par regle et colonne.

    Les numeros de ligne sont ceux du fichier (en-tete = ligne 1).
    """
    fc = exc.failure_cases.copy()
    builtin = {"column_in_dataframe": "colonne_absente", "not_nullable": "valeur_manquante"}
    fc["regle"] = fc["check"].map(lambda c: builtin.get(str(c), str(c)))
    fc = fc[fc["regle"].isin(list(REGLES))]
    out: list[dict] = []
    absent = fc[fc["regle"] == "colonne_absente"]
    for col in absent["failure_case"].astype(str).unique():
        out.append(_erreur("colonne_absente", col, 1, [], [col]))
    fc = fc[fc["regle"] != "colonne_absente"]
    for (ctx, regle), g in fc.groupby(["schema_context", "regle"], sort=False):
        if ctx == "DataFrameSchema":
            # regle multi-colonnes : une ligne fautive apparait une fois par colonne
            idx = sorted(int(i) for i in pd.unique(g["index"].dropna()))
            show = [c for c in COLONNES_EXEMPLE.get(regle, []) if c in brut.columns]
            ex = [{c: brut.at[i, c] for c in show} for i in idx[:N_EXEMPLES]]
            out.append(_erreur(regle, None, len(idx), [i + 2 for i in idx], ex))
            continue
        for col, gc in g.groupby("column", sort=False):
            gc = gc.dropna(subset=["index"]).sort_values("index")
            idx = gc["index"].astype(int).tolist()
            vals = gc["failure_case"].tolist() if regle != "valeur_manquante" else []
            out.append(_erreur(regle, str(col), len(idx), [i + 2 for i in idx], vals))
    return out


def _valider_pandera(schema, df: pd.DataFrame, brut: pd.DataFrame) -> list[dict]:
    from pandera.errors import SchemaErrors

    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        return _erreurs_pandera(exc, brut)
    return []


def convertir(brut: pd.DataFrame) -> pd.DataFrame:
    """Colonnes brutes -> types du contrat ; illisible -> NaN / NaT (deja signale).

    Les colonnes absentes sont creees vides, pour que les indicateurs restent calculables.
    """
    out = pd.DataFrame(index=brut.index)
    for c in COLONNES:
        if c == "timestamp_utc":
            continue
        if c == "source_type":
            out[c] = (brut[c].astype("string").astype(object) if c in brut
                      else pd.Series(None, index=brut.index, dtype=object))
        else:
            out[c] = _num(brut[c]) if c in brut else np.nan
    out["timestamp_utc"] = brut["timestamp_utc"] if "timestamp_utc" in brut else None
    out["_ts"] = (parse_horodatage(brut["timestamp_utc"]) if "timestamp_utc" in brut
                  else pd.Series(pd.NaT, index=brut.index, dtype="datetime64[ns]"))
    return out


# =============================================================================== QUALITE 8.3
def _r(x, n: int = 4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def _stat(fn, x, n: int = 3):
    """min / mediane / max sur les valeurs finies, None s'il n'y en a pas (sans warning)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return _r(fn(x), n) if x.size else None


def _ids(mask: np.ndarray, offset: int = 1) -> dict:
    ids = np.nonzero(mask)[0] + offset
    return {"nombre": int(ids.size), "ids": ids[:N_IDS].tolist(),
            "tronque": bool(ids.size > N_IDS)}


def valeurs_figees(vals: np.ndarray, n: int, temps_s: np.ndarray | None = None,
                   duree_s: float = 0.0, actif: np.ndarray | None = None) -> np.ndarray:
    """(stations, instants) -> cellules d'une serie de valeurs egales assez longue.

    Une serie compte si elle a >= n valeurs ET couvre >= duree_s secondes (temps_s : date
    de chaque instant, en s). actif (stations, instants) : cellules ou une egalite a un
    sens (ex. vent hors calme) ; une cellule inactive coupe la serie.
    """
    ns, nk = vals.shape
    if nk < n or n < 2:
        return np.zeros((ns, nk), dtype=bool)
    eq = (vals[:, 1:] == vals[:, :-1]) & np.isfinite(vals[:, 1:])     # (ns, nk-1)
    if actif is not None:
        eq &= actif[:, 1:] & actif[:, :-1]
    run = np.zeros((ns, nk), dtype=int)           # egalites consecutives se terminant en k
    for k in range(1, nk):
        run[:, k] = np.where(eq[:, k - 1], run[:, k - 1] + 1, 0)
    out = run >= n - 1
    if temps_s is not None and duree_s > 0:
        t = np.asarray(temps_s, float)
        debut = np.arange(nk)[None, :] - run                          # premier instant
        out &= (t[None, :] - t[debut]) >= duree_s
    for k in range(nk - 2, -1, -1):
        out[:, k] |= out[:, k + 1] & eq[:, k]
    return out


def indicateurs(dfc: pd.DataFrame, grille: Grille, seuils: Seuils | None = None) -> dict | None:
    """Indicateurs 8.3 sur un DataFrame converti (voir `convertir`).

    Calcules sur les lignes exploitables (point_id connu, horodatage lisible, sans
    doublon) ; les valeurs hors plage sont comptees sur toutes les lignes.
    """
    seuils = seuils or Seuils()
    n_rows = len(dfc)
    pid, ts = dfc["point_id"], dfc["_ts"]
    # un instant hors du pas attendu (ex. 03:00:07) n'en est pas un : il ne doit pas
    # creer une colonne d'instant ou toutes les stations paraitraient absentes
    usable = (grille.connu(pid) & ts.notna() & _aligne(dfc, seuils.pas_attendu_s)
              & ~dfc.duplicated(["point_id", "_ts"]))
    base = dfc[usable]
    if base.empty:
        return None
    instants = np.sort(base["_ts"].unique())
    n_inst, n_st = len(instants), grille.n_stations

    # --- valeurs hors plage et densite, par variable, sur toutes les lignes
    hors_plage, anomalie = {}, pd.Series(False, index=dfc.index)
    is_station = (pid >= 1) & (pid <= n_st)
    for var, (lo, hi) in PLAGES.items():
        v = dfc[var]
        bad = v.notna() & ~v.between(lo, hi)
        anomalie |= bad
        hors_plage[var] = {"plage": [lo, hi], "nombre": int(bad.sum()),
                           "part": _r(bad.sum() / max(n_rows, 1), 6),
                           "manquantes": int(v.isna().sum()),
                           "stations": int(pid[bad & is_station].nunique()),
                           "min": _r(v.min()), "max": _r(v.max())}
    ecart = _ecart_densite(dfc)
    dens_bad = ecart > seuils.tol_densite_rel
    anomalie |= dens_bad
    _, modes = densite_gaz_parfaits(dfc)
    densite = {"tolerance_rel": seuils.tol_densite_rel, "incoherentes": int(dens_bad.sum()),
               "ecart_rel_max": _r(ecart.max(), 6), "ecart_rel_median": _r(ecart.median(), 6),
               "temperature_virtuelle": modes}

    # --- matrices (station, instant)
    st = base[base["point_id"] <= n_st]
    s_idx = st["point_id"].to_numpy().astype(int) - 1
    k_idx = np.searchsorted(instants, st["_ts"].to_numpy())

    def mat(values: np.ndarray, fill) -> np.ndarray:
        m = np.full((n_st, n_inst), fill, dtype=np.asarray(values).dtype)
        m[s_idx, k_idx] = values
        return m

    role = st["source_type"].map(grille.sources)
    present = mat(np.ones(len(st), bool), False)
    manque = mat(st[["wind_u_m_s", "wind_v_m_s", "temperature_c"]].isna().any(axis=1)
                 .to_numpy(), False)
    mask0 = (st["mask_i_obs"] == 0).to_numpy()
    src_interp = (role == "interpolee").to_numpy()
    low_q = (st["aneo_quality_score"] < seuils.qualite_signalee).to_numpy()
    interp = mat(mask0 | src_interp | low_q, False)
    anom = mat(anomalie[st.index].to_numpy(), False)
    figee = np.zeros((n_st, n_inst), dtype=bool)
    figees_par_var = {}
    temps_s = (instants - instants[0]) / np.timedelta64(1, "s")
    series = {v: mat(st[v].to_numpy(dtype=float), np.nan) for v in VARS_FIGEES}
    with np.errstate(invalid="ignore"):
        hors_calme = (np.hypot(series["wind_u_m_s"], series["wind_v_m_s"])
                      >= seuils.vitesse_calme)
    for var in VARS_FIGEES:
        f = valeurs_figees(series[var], seuils.n_instants_figee, temps_s,
                           seuils.duree_figee_s,
                           actif=hors_calme if var.startswith("wind_") else None)
        figees_par_var[var] = int(f.any(axis=1).sum())
        figee |= f
    mesure = present & ~manque
    valid = mesure & ~interp & ~anom & ~figee
    # lignes qui se DECLARENT mesurees : ni mask_i_obs = 0, ni source interpolee, avec u/v/T
    declaree = mesure & ~mat(mask0 | src_interp, False)
    # a masquer avant l'inference : presente, mais valeurs fausses ou inutilisables
    a_masquer = present & (manque | anom | figee)

    # --- 1. stations valides
    par_instant = valid.sum(axis=0) / n_st
    k_worst = int(np.argmin(par_instant))
    stations_valides = {
        "part": _r(valid.sum() / (n_st * n_inst)),
        # sur les seuls couples presents : seuil de rejet du profil upload (panne connue)
        "part_presentes": _r(valid.sum() / max(int(present.sum()), 1)),
        # sur les lignes qui se declarent mesurees : seuil de rejet du profil upload
        "part_mesurees": _r(valid.sum() / max(int(declaree.sum()), 1)),
        "lignes_mesurees": int(declaree.sum()),
        "part_pire_instant": _r(par_instant[k_worst]),
        "pire_instant": pd.Timestamp(instants[k_worst]).isoformat(),
        "par_instant": {pd.Timestamp(t).isoformat(): _r(x)
                        for t, x in zip(instants, par_instant)},
    }

    # --- 2. valeurs interpolees ou signalees (couples station x instant presents)
    n_present = int(present.sum())
    n_interp = int(interp.sum())
    interpolees = {
        "part": _r(n_interp / max(n_present, 1)), "nombre": n_interp,
        "mask_i_obs_0": int(mask0.sum()), "source_interpolee": int(src_interp.sum()),
        "qualite_faible": int(low_q.sum()),
        # declarees mesurees (mask 1) mais de source interpolee : interpolation detectee
        "interpolees_non_declarees": int(((st["mask_i_obs"] == 1).to_numpy()
                                          & src_interp).sum()),
        "stations_toujours_interpolees": int((present.any(axis=1)
                                              & ~(present & ~interp).any(axis=1)).sum()),
    }

    # --- 3. fraicheur : absences, stations muettes, valeurs figees
    absence = ~mesure
    figee_station = figee.any(axis=1)
    jamais_la = ~present.any(axis=1)
    # profil upload : absence = panne connue (sans ligne, mask 0, interpolee, sans u/v/T) ;
    # seule une valeur figee rend la station muette
    muette = figee_station if seuils.absences_connues else absence.any(axis=1) | figee_station
    has_any = mesure.any(axis=1)
    last = np.where(has_any, n_inst - 1 - np.argmax(mesure[:, ::-1], axis=1), 0)
    retard = (instants[-1] - instants[last]) / np.timedelta64(1, "s")
    fraicheur = {
        "absences": int(absence.sum()),
        "stations_absentes": _ids(jamais_la),
        "stations_muettes": {**_ids(muette), "part": _r(muette.sum() / n_st)},
        "stations_sans_mesure_dernier_instant": int(absence[:, -1].sum()),
        "retard_max_s": _r(retard[has_any].max(), 1) if has_any.any() else None,
        "valeurs_figees": {**_ids(figee_station), "n_instants": seuils.n_instants_figee,
                           "par_variable": figees_par_var},
    }

    # --- 4. couverture spatiale : stations valides par tuile
    ix = np.arange(n_st) % grille.nx_s
    iy = np.arange(n_st) // grille.nx_s
    tx = np.floor(ix * grille.dx_s / seuils.tuile_m).astype(int)
    ty = np.floor(iy * grille.dy_s / seuils.tuile_m).astype(int)
    ntx, nty = int(tx.max()) + 1, int(ty.max()) + 1
    tile = ty * ntx + tx
    n_tile = np.bincount(tile, minlength=ntx * nty)
    v_tile = np.stack([np.bincount(tile, weights=valid[:, k], minlength=ntx * nty)
                       for k in range(n_inst)], axis=1)                 # (tuiles, instants)
    vides = ((v_tile == 0) & (n_tile[:, None] > 0)).sum(axis=0) / max((n_tile > 0).sum(), 1)
    area_km2 = n_tile * grille.dx_s * grille.dy_s / 1e6
    with np.errstate(invalid="ignore", divide="ignore"):
        dens = np.where(n_tile > 0, v_tile.mean(axis=1) / area_km2, np.nan)
    frac = v_tile.mean(axis=1) / np.maximum(n_tile, 1)
    couverture = {
        "tuile_m": seuils.tuile_m, "tuiles_y_x": [nty, ntx],
        "part_tuiles_vides_pire_instant": _r(vides.max()),
        "densite_stations_valides_km2": {
            "moyenne": _r(valid.sum() / n_inst / area_km2.sum(), 3),
            "min": _stat(np.min, dens), "mediane": _stat(np.median, dens),
            "max": _stat(np.max, dens)},
        # part de stations valides par tuile (moyenne sur les instants), lignes = y
        "part_valide_par_tuile": [[_r(x, 3) for x in row]
                                  for row in frac.reshape(nty, ntx)],
    }

    n_anom = int(anomalie.sum())
    n_anom_st = int((anomalie & is_station).sum())
    n_rows_st = int(is_station.sum())
    return {
        "instants": n_inst, "stations_declarees": n_st,
        "stations_valides": stations_valides,
        "interpolees_ou_signalees": interpolees,
        "hors_plage": hors_plage,
        "densite": densite,
        # part calculee sur les lignes STATIONS : les noeuds 3D (~73 % des lignes d'un
        # fichier journalier) dilueraient le seuil de rejet
        "anomalies": {"lignes": n_anom, "lignes_stations": n_anom_st,
                      "lignes_noeuds_3d": n_anom - n_anom_st,
                      "part_lignes_stations": _r(n_anom_st / max(n_rows_st, 1), 6)},
        "fraicheur": fraicheur,
        "couverture": couverture,
        "stations_a_masquer": {**_ids(a_masquer.any(axis=1)),
                               "couples": int(a_masquer.sum())},
        # stations dont au moins une ligne est hors plage ou de densite incoherente : cibles
        # des alertes `hors_plage` du lot M4 (pcfno3d.alerts)
        "stations_hors_plage": _ids(anom.any(axis=1)),
        # retire par valider_dataframe (non JSON) : point_id a masquer, par instant
        "_masque": {pd.Timestamp(t): (np.nonzero(a_masquer[:, k])[0] + 1)
                    for k, t in enumerate(instants) if a_masquer[:, k].any()},
    }


# =============================================================================== VERDICT
def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def verdict(erreurs: list[dict], q: dict | None,
            seuils: Seuils | None = None) -> tuple[str, list[str], list[str]]:
    """(verdict, motifs, avertissements) a partir des erreurs et des indicateurs."""
    s = seuils or Seuils()
    rejet, quar, avert = [], [], []
    bloq = sorted({e["regle"] for e in erreurs if e["gravite"] == BLOQUANT})
    if bloq:
        rejet.append("erreurs bloquantes : " + ", ".join(bloq))
    if q is None:
        rejet.append("aucune ligne exploitable (point_id ou timestamp_utc illisibles)")
        return "rejet", rejet, avert

    an = q["anomalies"]
    if (an["part_lignes_stations"] > s.rejet_part_anomalies
            and an["lignes_stations"] > s.rejet_anomalies_min_lignes):
        rejet.append(f"{_pct(an['part_lignes_stations'])} des lignes stations hors plage ou "
                     f"de densite incoherente (> {_pct(s.rejet_part_anomalies)})")
    elif an["lignes"]:
        quar.append(f"{an['lignes']} ligne(s) hors plage ou de densite incoherente")

    sv = q["stations_valides"]
    part = sv["part_mesurees"] if s.absences_connues else sv["part"]
    if part < s.rejet_stations_valides:
        base = " des lignes declarees mesurees" if s.absences_connues else ""
        rejet.append(f"stations valides {_pct(part)}{base} "
                     f"(< {_pct(s.rejet_stations_valides)}) : reconstruction impossible")
    elif sv["part_pire_instant"] < s.quarantaine_stations_valides:
        quar.append(f"stations valides {_pct(sv['part_pire_instant'])} a "
                    f"{sv['pire_instant']} (< {_pct(s.quarantaine_stations_valides)})")

    ip = q["interpolees_ou_signalees"]["part"]
    if ip > s.quarantaine_interpolees:
        quar.append(f"valeurs interpolees ou signalees {_pct(ip)} "
                    f"(> {_pct(s.quarantaine_interpolees)})")
    elif ip > s.avert_interpolees:
        avert.append(f"valeurs interpolees ou signalees {_pct(ip)} "
                     f"(> {_pct(s.avert_interpolees)})")

    fr = q["fraicheur"]
    mu = fr["stations_muettes"]
    if mu["part"] > s.quarantaine_muettes:
        quar.append(f"stations muettes {_pct(mu['part'])} (> {_pct(s.quarantaine_muettes)})")
    elif mu["nombre"]:
        avert.append(f"{mu['nombre']} station(s) muette(s)")
    if fr["valeurs_figees"]["nombre"]:
        avert.append(f"{fr['valeurs_figees']['nombre']} station(s) a valeur figee")
    if fr["stations_absentes"]["nombre"]:
        avert.append(f"{fr['stations_absentes']['nombre']} station(s) absente(s) du fichier")

    nm = q["stations_a_masquer"]["nombre"]
    if nm:
        avert.append(f"{nm} station(s) a masquer avant reconstruction (hors plage, densite, "
                     "figee ou sans u/v/T)")
    tv = q["couverture"]["part_tuiles_vides_pire_instant"]
    if tv > s.quarantaine_tuiles_vides:
        quar.append(f"tuiles sans station valide {_pct(tv)} "
                    f"(> {_pct(s.quarantaine_tuiles_vides)})")
    elif tv > 0:
        avert.append(f"tuiles sans station valide {_pct(tv)} au pire instant")

    if rejet:
        return "rejet", rejet + quar, avert
    if quar:
        return "quarantaine", quar, avert
    return "conforme", [], avert


# =============================================================================== API
def valider_dataframe(brut: pd.DataFrame, grille: Grille, seuils: Seuils | None = None,
                      nom: str | None = None,
                      erreurs_lecture: list[dict] | None = None,
                      masque: dict | None = None) -> dict:
    """Valide un DataFrame BRUT (tel que lu, sans conversion) et renvoie le rapport.

    L'index doit etre le numero de ligne de donnees (0 = ligne 2 du fichier).
    masque : dict rempli (s'il est fourni) par {instant: point_id des stations a masquer
    avant l'inference} — hors plage, densite incoherente, figees, sans u/v/T.
    """
    seuils = seuils or Seuils()
    erreurs = list(erreurs_lecture or [])
    presentes = set(brut.columns) & set(COLONNES)
    erreurs += _valider_pandera(schema_brut(), brut, brut)
    dfc = convertir(brut)
    jour = jour_depuis_nom(nom)
    erreurs += _valider_pandera(schema_type(grille, presentes, seuils, jour), dfc, brut)
    q = (indicateurs(dfc, grille, seuils) if {"point_id", "timestamp_utc"} <= presentes
         else None)
    a_masquer = q.pop("_masque") if q is not None else {}
    if masque is not None:
        masque.update(a_masquer)
    v, motifs, avert = verdict(erreurs, q, seuils)
    extra = sorted(set(brut.columns) - set(COLONNES))
    if extra:
        avert.append("colonnes non prevues par le contrat (ignorees) : " + ", ".join(extra))
    ts = dfc["_ts"].dropna()
    pid = dfc["point_id"]
    return {
        "contrat": CONTRAT_VERSION,
        "fichier": nom,
        "controle_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdict": v,
        "motifs": motifs,
        "avertissements": avert,
        "resume": {
            "lignes": int(len(brut)),
            "lignes_stations": int(((pid >= 1) & (pid <= grille.n_stations)).sum()),
            "lignes_noeuds_3d": int((pid >= grille.node_id_start).sum()),
            "instants": int(ts.nunique()),
            "premier_instant": ts.min().isoformat() if len(ts) else None,
            "dernier_instant": ts.max().isoformat() if len(ts) else None,
            "jour_du_fichier": ({"day_index": jour[0], "date": jour[1].date().isoformat()}
                                if jour else None),
            "lignes_en_erreur_bloquante": int(sum(e["nombre"] for e in erreurs
                                                  if e["gravite"] == BLOQUANT)),
            "anomalies_de_mesure": int(sum(e["nombre"] for e in erreurs
                                           if e["gravite"] == MESURE)),
        },
        "erreurs": erreurs,
        "qualite": q,
        "seuils": asdict(seuils),
    }


def valider_fichier(path: str | Path, grille: Grille, seuils: Seuils | None = None,
                    nom: str | None = None, surface_seule: bool = False,
                    masque: dict | None = None) -> dict:
    """Lit et valide un fichier de mesures ; ne leve pas d'exception sur un fichier faux.

    surface_seule : ignore les noeuds 3D a la lecture (fichiers journaliers complets du
    dataset reel, ~1,5 million de noeuds par instant). La lecture en flux ne borne la
    memoire que pour la partie 3D : les lignes STATIONS gardees restent en memoire
    (~1 Ko par ligne avec les colonnes converties, soit ~0,5 Go par instant pour les
    500 000 stations de la grille reelle, et N x 0,5 Go pour un jour a N instants).
    """
    path = Path(path)
    nom = nom or path.name
    brut, erreurs = lire_csv(path, surface_seule, grille.node_id_start)
    if brut is None:
        v, motifs, avert = verdict(erreurs, None, seuils)
        rapport = {"contrat": CONTRAT_VERSION, "fichier": nom, "verdict": v,
                   "motifs": motifs, "avertissements": avert, "resume": {"lignes": 0},
                   "erreurs": erreurs, "qualite": None, "seuils": asdict(seuils or Seuils())}
    else:
        rapport = valider_dataframe(brut, grille, seuils, nom, erreurs, masque)
    rapport["taille_octets"] = path.stat().st_size
    return rapport


def ecrire_rapport(rapport: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(rapport, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return path


def chemin_rapport(path: str | Path) -> Path:
    """x.csv.gz -> x.csv.gz.rapport.json, dans le meme dossier."""
    path = Path(path)
    return path.with_name(path.name + ".rapport.json")


def _destination_libre(dossier: Path, name: str) -> Path:
    """Jamais d'ecrasement : x.csv.gz, puis x_1.csv.gz, x_2.csv.gz..."""
    base, dot, ext = name.partition(".")
    dest, n = dossier / name, 1
    while dest.exists() or chemin_rapport(dest).exists():
        dest = dossier / f"{base}_{n}{dot}{ext}"
        n += 1
    return dest


def mettre_en_quarantaine(path: str | Path, rapport: dict, dossier: str | Path,
                          copier: bool = False) -> Path:
    """Deplace (ou copie) le fichier dans `dossier` et ecrit son rapport a cote.

    Rien n'est jamais supprime ni ecrase : un nom deja pris est suffixe (_1, _2...).
    """
    src = Path(path)
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    dest = _destination_libre(dossier, src.name)
    if copier:
        shutil.copy2(src, dest)
    else:
        shutil.move(str(src), str(dest))
    rapport["quarantaine"] = {"fichier": str(dest), "origine": str(src),
                              "mode": "copie" if copier else "deplacement",
                              "rapport": str(chemin_rapport(dest))}
    ecrire_rapport(rapport, chemin_rapport(dest))
    return dest


def controler(path: str | Path, grille: Grille, quarantaine: str | Path | None = None,
              copier: bool = False, seuils: Seuils | None = None,
              surface_seule: bool = False) -> dict:
    """Valide un fichier ; s'il n'est pas conforme et qu'un dossier de quarantaine est
    donne, l'y place (deplace ou copie) avec son rapport. Renvoie le rapport."""
    rapport = valider_fichier(path, grille, seuils, surface_seule=surface_seule)
    if quarantaine is not None and rapport["verdict"] != "conforme":
        mettre_en_quarantaine(path, rapport, quarantaine, copier=copier)
    return rapport


def code_retour(verdicts: list[str]) -> int:
    """Code de sortie de `pcfno3d validate` : 0 conforme (avertissements compris),
    1 quarantaine, 2 rejet."""
    if "rejet" in verdicts:
        return 2
    return 1 if "quarantaine" in verdicts else 0
