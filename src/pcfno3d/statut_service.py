"""Mode, date et statut de chaque donnee servie (demande d'Ines du 16/11/2026).

Chaque reponse de « Mon vent » et de l'API dit :
- le MODE : « analyse » (journee calculee sur l'ERA5), « temps reel » (heure passee d'aujourd'hui : GFS +
  mesures des stations) ou « prevision » (heure a venir : GFS seul) ;
- la DATE de la donnee ;
- un STATUT : « valide » seulement quand une regle d'acceptation fixee avant le calcul a ete passee pour CE
  domaine, CE mode et CETTE couche ; sinon « experimental ». La preuve (ou l'absence de preuve) accompagne le
  statut.

Ce qui est valide aujourd'hui (preuves dans docs/resultats/) :
- national, analyse : C' s44 nourri par l'ERA5, DSO1-DSO6 conformes sur le test 2022 ; couche 10 m ERA5 +
  aeroports, -3,7 % contre l'ERA5 sur 2023-2024 (regle_produit_vent10m_v4) ;
- Grand Tunis, analyse : v4 livre par la regle decision_v4 (test 2022) ;
- Grand Tunis, prevision J+1 / J+2 : seulement si le modele servi est v5 (OS7 : PREV1 et PREV2 conformes) ;
- national, temps reel et prevision : seulement les cases ACCEPTEES de regle_national_gfs_v4, lues dans un
  fichier de verdicts qui porte l'empreinte hachee de cette regle ; tant qu'elle n'est pas passee : experimental ;
- champ 3D national, temps reel / J+1 / J+2 : aussi les cases ACCEPTEES de regle_national_gfs_v5_2 (variable
  PCFNO3D_VERDICTS_GFS_V5, fichier portant l'empreinte de CETTE regle). Ces cases valident le CANDIDAT RETENU par
  la regle (nomme dans le fichier) : la variable ne se renseigne que lorsque le service sert ce candidat (decision
  d'Ines apres lecture des resultats, regle_national_gfs_v5_2 « consequences »). Les verdicts gfs_v4 sont lus a part,
  inchanges.
Le temps reel du Grand Tunis (v4 nourri par GFS + stations) n'a pas de regle a courte echeance : experimental.

Statut de SERVICE (regle_service_zones_v5, hachee b239c5c6..., decision d'Ines du 23/11/2026, ecrite APRES le
jugement 2026 de regle_national_gfs_v5_2, dont le verdict « refuse » reste publie) : quand le service sert le candidat
c3_a0.75_H100 (C1 + ancrage GFS a0 0,75, H 100 m), le champ 3D national en temps reel et en prevision est
« valide » a plus de 50 km de l'aeroport de Tabarka et « experimental (zone Tabarka) » a 50 km ou moins. Ce n'est
pas une acceptation par test ; la regle le dit. Un cas refuse n'est jamais dit « pas encore passe ».
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

VALIDE, EXPERIMENTAL = "valide", "experimental"
ANALYSE, TEMPS_REEL, PREVISION = "analyse", "temps reel", "prevision"
REGLE_GFS = Path(__file__).resolve().parents[2] / "configs" / "national" / "regle_national_gfs_v4.json"
# regle executee : la v5.2 (la v5, jamais executee, et la v5.1, arretee sans verdict le 23/11, sont archivees et
# ne sont jamais lues ici)
REGLE_GFS_V5 = Path(__file__).resolve().parents[2] / "configs" / "national" / "regle_national_gfs_v5_2.json"
# cases jugees par regle_national_gfs_v5_2 (critere.cases_jugees) -> echeance servie
CASES_V5 = {"national 3D / analyse": 0, "national 3D / +24 h": 1, "national 3D / +48 h": 2}
# regle de service (zones) : recopie des valeurs de son texte, verifiee par tests/test_statut_service.py
REGLE_ZONES = Path(__file__).resolve().parents[2] / "configs" / "national" / "regle_service_zones_v5.json"
ZONE_TABARKA = {"code": "DTKA", "nom": "Tabarka", "lat": 36.95, "lon": 8.75, "rayon_km": 50.0}
LIBELLE_ZONE = "experimental (zone Tabarka)"
TOLERANCE_CENTRE_DEG = 0.01
RAYON_TERRE_KM = 6371.0


def regle_zones() -> dict | None:
    """La regle de service par zones si son empreinte correspond a son .sha256 : {"sha256", "zone"} ; sinon None."""
    sha_f = REGLE_ZONES.with_name(REGLE_ZONES.name + ".sha256")
    try:
        attendu = sha_f.read_text(encoding="ascii").split()[0]
        h = hashlib.sha256(REGLE_ZONES.read_bytes()).hexdigest()
    except (OSError, IndexError):
        return None
    return {"sha256": h, "zone": dict(ZONE_TABARKA)} if h == attendu else None


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance du grand cercle (km), rayon terrestre 6371 km (regle_service_zones_v5, zone.definition)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * RAYON_TERRE_KM * math.asin(min(1.0, math.sqrt(a)))


def dans_zone_tabarka(lat: float, lon: float) -> bool:
    z = ZONE_TABARKA
    return distance_km(lat, lon, z["lat"], z["lon"]) <= z["rayon_km"]


def _statut_zones(zones: dict, lat: float | None, lon: float | None) -> dict:
    sha = zones["sha256"][:8]
    z = zones["zone"]
    zone = {"centre": {"lat": z["lat"], "lon": z["lon"]}, "rayon_km": z["rayon_km"], "aeroport": z["nom"],
            "statut": LIBELLE_ZONE}
    if lat is not None and lon is not None and dans_zone_tabarka(lat, lon):
        return {"statut": EXPERIMENTAL, "libelle": LIBELLE_ZONE, "zone_experimentale": zone,
                "preuve": f"regle_service_zones_v5 ({sha}...) : a {z['rayon_km']:.0f} km ou moins de {z['nom']}, ou "
                          "regle_national_gfs_v5_2 a refuse le candidat (Tabarka +6 % face a GFS en 2026)"}
    return {"statut": VALIDE, "libelle": "valide" if lat is not None else "valide (hors zone Tabarka)",
            "zone_experimentale": zone,
            "preuve": f"regle_service_zones_v5 ({sha}...) du 23/11/2026 : valide a plus de {z['rayon_km']:.0f} km de "
                      f"{z['nom']} ; fondee sur le jugement 2026 de regle_national_gfs_v5_2 (refuse pour Tabarka seul ; "
                      "-10 % face a GFS, 11 aeroports sur 12)"}


def verdicts_gfs_v4(chemin: str | Path | None = None) -> dict[str, bool]:
    """Cases de regle_national_gfs_v4 ACCEPTEES. Vide tant que la regle n'est pas hachee (accord d'Ines), que le
    fichier de verdicts (PCFNO3D_VERDICTS_GFS_V4) manque ou qu'il ne porte pas l'empreinte de la regle."""
    sha_f = REGLE_GFS.with_name(REGLE_GFS.name + ".sha256")
    chemin = chemin or os.environ.get("PCFNO3D_VERDICTS_GFS_V4")
    if not sha_f.exists() or not chemin or not Path(chemin).exists():
        return {}
    try:
        v = json.loads(Path(chemin).read_text(encoding="utf-8"))
        sha = sha_f.read_text(encoding="utf-8").split()[0]
    except (OSError, ValueError, IndexError):
        return {}
    if v.get("regle_sha256") != sha:
        return {}
    return {str(k): x is True or x == "accepte" for k, x in (v.get("cases") or {}).items()}


def verdicts_gfs_v5(chemin: str | Path | None = None) -> dict[str, bool]:
    """Cases de regle_national_gfs_v5_2 ACCEPTEES (cles de CASES_V5). Vide si le fichier (PCFNO3D_VERDICTS_GFS_V5)
    manque, si la regle n'a pas d'empreinte ou si le fichier ne porte pas l'empreinte de CETTE regle."""
    sha_f = REGLE_GFS_V5.with_name(REGLE_GFS_V5.name + ".sha256")
    chemin = chemin or os.environ.get("PCFNO3D_VERDICTS_GFS_V5")
    if not sha_f.exists() or not chemin or not Path(chemin).exists():
        return {}
    try:
        v = json.loads(Path(chemin).read_text(encoding="utf-8"))
        sha = sha_f.read_text(encoding="utf-8").split()[0]
    except (OSError, ValueError, IndexError):
        return {}
    if v.get("regle_sha256") != sha:
        return {}
    return {str(k): x is True or x == "accepte" for k, x in (v.get("cases") or {}).items() if k in CASES_V5}


def mode_instant(source: str | None, archive_era5: bool = False) -> str:
    """Mode d'un instant servi : `source` des instants de /atom-wind/direct (« temps reel » / « prevision »)."""
    if archive_era5:
        return ANALYSE
    return PREVISION if source == PREVISION else TEMPS_REEL


def statut(national: bool, mode: str, couche_10m: bool = False, j: int = 0,
           prevision_os7: bool = False, verdicts: dict[str, bool] | None = None,
           verdicts_v5: dict[str, bool] | None = None, zones: dict | None = None,
           lat: float | None = None, lon: float | None = None) -> dict:
    """{"statut", "preuve"} d'une donnee. `couche_10m` : la valeur lue a 10 m vient de la couche fond + aeroports ;
    `j` : 0 aujourd'hui, 1 demain, 2 apres-demain ; `prevision_os7` : le modele Grand Tunis servi est v5 (OS7) ;
    `verdicts` : cases acceptees de regle_national_gfs_v4 ; `verdicts_v5` : celles de regle_national_gfs_v5_2 (champ
    3D national seulement) ; `zones` : regle_zones() quand le service sert le candidat c3_a0.75_H100 (sinon None),
    avec `lat`, `lon` du point (absents : statut d'une journee entiere, « valide (hors zone Tabarka) »)."""
    verdicts = verdicts or {}
    verdicts_v5 = verdicts_v5 or {}
    if mode == ANALYSE:
        if national and couche_10m:
            return {"statut": VALIDE, "preuve": "couche 10 m ERA5 + aeroports : -3,7 % contre l'ERA5 aux aeroports "
                                                "sur 2023-2024, periode jamais utilisee (regle_produit_vent10m_v4)"}
        if national:
            return {"statut": VALIDE, "preuve": "modele national C' s44 nourri par l'ERA5 : DSO1-DSO6 conformes sur "
                                                "le test 2022 (decision_v4_national)"}
        return {"statut": VALIDE, "preuve": "modele Grand Tunis v4 nourri par l'ERA5 : livre par la regle decision_v4 "
                                            "(test 2022)"}
    if national:
        if mode == TEMPS_REEL:
            cle = "b / analyse" if couche_10m else "a / analyse"
        elif couche_10m:
            return {"statut": EXPERIMENTAL, "preuve": "prevision : aucune mesure de l'heure visee, la couche 10 m vaut "
                                                      "GFS brut, sans correction"}
        else:
            cle = "a / +48 h" if j >= 2 else "a / +24 h"
        if verdicts.get(cle):
            return {"statut": VALIDE, "preuve": f"regle_national_gfs_v4, case « {cle} » acceptee"}
        if not couche_10m:
            cle5 = "national 3D / analyse" if mode == TEMPS_REEL else (
                "national 3D / +48 h" if j >= 2 else "national 3D / +24 h")
            if verdicts_v5.get(cle5):
                return {"statut": VALIDE, "preuve": f"regle_national_gfs_v5_2, case « {cle5} » acceptee (candidat "
                                                    "retenu sur la validation 2021, juge sur 2026)"}
            if zones:
                return _statut_zones(zones, lat, lon)
            # champ 3D nourri par GFS : refuse le 18/11 (gfs_v4, Tabarka) et le 23/11 (gfs_v5_2, Tabarka) ; un cas
            # refuse n'est jamais dit « pas encore passe »
            return {"statut": EXPERIMENTAL, "preuve": f"nourri par GFS : refuse par regle_national_gfs_v4 (case "
                                                      f"« {cle} », 18/11/2026) et par regle_national_gfs_v5_2 "
                                                      "(23/11/2026), a cause de Tabarka"}
        return {"statut": EXPERIMENTAL, "preuve": "couche 10 m nourrie par GFS : verdict de regle_national_gfs_v4 "
                                                  f"(case « {cle} ») non lu par ce service (PCFNO3D_VERDICTS_GFS_V4)"}
    if mode == PREVISION and j >= 1 and prevision_os7:
        return {"statut": VALIDE, "preuve": "modele Grand Tunis v5 nourri par GFS : PREV1 (+24 h) et PREV2 (+48 h) "
                                            "conformes (OS7, 2022)"}
    if mode == PREVISION:
        return {"statut": EXPERIMENTAL,
                "preuve": "prevision par v5 a moins de 24 h non jugee (PREV1/PREV2 a +24 h et +48 h)" if prevision_os7
                else "modele Grand Tunis v4 nourri par GFS sans affinage : non conforme a l'OS7 (moins bon que GFS brut "
                     "a Zaghouan) ; le modele valide en prevision est v5"}
    return {"statut": EXPERIMENTAL, "preuve": "temps reel (GFS + stations) : aucune regle de validation a courte "
                                              "echeance"}


def resume(statuts: list[dict]) -> str:
    """Statut d'une journee entiere : « valide » seulement si toutes ses heures le sont."""
    return VALIDE if statuts and all(s.get("statut") == VALIDE for s in statuts) else EXPERIMENTAL
