"""BLOC 0/1 — configuration : chemins, hyperparametres, seuils DSO.

Toutes les constantes qui etaient des variables globales du notebook sont ici, dans des
dataclasses chargeables depuis un YAML (`configs/*.yaml`). Rien n'est lu sur le disque a
l'import : `Config.load()` ne touche qu'au fichier de configuration.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _resolve(path: str | Path) -> Path:
    """Chemin absolu tel quel ; chemin relatif resolu depuis la racine du depot."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


# --------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------
@dataclass
class Paths:
    """Les trois dossiers du pipeline.

    data_dir  : dataset brut du generateur (aneo_grid_points.csv, meta.json, CSV journaliers)
    cache_dir : tenseurs .npy par instant (BLOC 3) — disque local rapide, regenerable
    run_dir   : checkpoints, statistiques, figures, evaluations — a conserver
    """

    data_dir: Path = REPO_ROOT / "aneo_dataset_3d_v2"
    cache_dir: Path = REPO_ROOT / "artifacts" / "cache"
    run_dir: Path = REPO_ROOT / "artifacts" / "runs" / "v7"
    # copie de securite du cache (Google Drive sur Colab) ; None = pas de copie.
    # Alimentee par build-cache, pour les SEULS jours nouvellement construits : le cache ne
    # change jamais pendant l'entrainement, il n'y a donc rien a recopier a ce moment-la.
    cache_mirror: Path | None = None
    # Disque local rapide pour les checkpoints PENDANT l'entrainement (Colab : /content/...).
    # last.pt pese ~7 Go (modele, EMA, etat d'Adam) : l'ecrire sur le Drive a chaque epoque
    # coute ~75 s par epoque. Avec local_dir, on ecrit en local a chaque epoque et on
    # recopie dans run_dir toutes les `train.save_every` epoques, a la fin, et sur
    # interruption — comme le faisait le BLOC 9 du notebook v7. None = ecriture directe dans
    # run_dir (poste local, ou run_dir est deja un disque rapide).
    local_dir: Path | None = None
    # Ou vivent les exports du visualiseur (binaires + manifeste, un dossier par instant).
    # None = run_dir/web. Sur Colab : le disque local, car 166 instants pesent ~1 Go et se
    # regenerent en ~1 min sur GPU ; inutile de les stocker sur le Drive.
    web_dir: Path | None = None
    # Modele reel (Grand Tunis) : forcage_era5.csv (scripts/forcage_era5.py, le .npz a cote).
    # Renseigne = le modele recoit aussi le vent ERA5 a chaque noeud et les 7 scalaires de
    # forcage de l'instant (pcfno3d.forcage) : 34 canaux au lieu de 25. None = v7.
    forcage_era5: Path | None = None
    # Affinage sur les stations (pcfno3d.stations_reelles) : dossier des metar_*.csv /
    # synop_*.csv des anemometres du domaine. Utilise seulement si loss.lambda_station > 0.
    stations_obs: Path | None = None

    def __post_init__(self) -> None:
        # un chemin relatif dans un YAML est relatif a la racine du depot, pas au cwd :
        # les commandes marchent donc depuis n'importe quel dossier
        self.data_dir = _resolve(self.data_dir)
        self.cache_dir = _resolve(self.cache_dir)
        self.run_dir = _resolve(self.run_dir)
        if self.cache_mirror is not None:
            self.cache_mirror = _resolve(self.cache_mirror)
        if self.local_dir is not None:
            self.local_dir = _resolve(self.local_dir)
        if self.web_dir is not None:
            self.web_dir = _resolve(self.web_dir)
        if self.forcage_era5 is not None:
            self.forcage_era5 = _resolve(self.forcage_era5)
        if self.stations_obs is not None:
            self.stations_obs = _resolve(self.stations_obs)

    @property
    def web_root(self) -> Path:
        """Racine des exports du visualiseur : un sous-dossier recon_<stem> par instant."""
        return self.web_dir if self.web_dir is not None else self.run_dir / "web"

    def mkdirs(self) -> None:
        dirs = [self.cache_dir, self.run_dir, self.run_dir / "figures"]
        if self.local_dir is not None:
            dirs.append(self.local_dir)
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # fichiers nommes une seule fois, pour que tous les modules pointent au meme endroit
    @property
    def index_csv(self) -> Path:
        return self.cache_dir / "index.csv"

    @property
    def norm_stats(self) -> Path:
        return self.run_dir / "norm_stats.json"

    @property
    def geometry_cache(self) -> Path:
        return self.cache_dir / "geometry_v1.npz"

    @property
    def ckpt_best(self) -> Path:
        return self.run_dir / "best.pt"

    @property
    def ckpt_best_dso(self) -> Path:
        return self.run_dir / "best_dso_v7.pt"

    @property
    def ckpt_last(self) -> Path:
        return self.run_dir / "last.pt"

    @property
    def history(self) -> Path:
        return self.run_dir / "history.json"

    def checkpoint_for_inference(self) -> Path:
        """Le modele retenu : raffinement DSO s'il existe, sinon le meilleur de l'entrainement."""
        for p in (self.ckpt_best_dso, self.ckpt_best):
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Aucun checkpoint dans {self.run_dir} (attendu best_dso_v7.pt ou best.pt)"
        )


@dataclass
class ModelCfg:
    """BLOC 7 — architecture."""

    width: int = 48                             # canaux internes
    modes: tuple[int, int, int] = (6, 32, 32)   # modes de Fourier conserves (z, y, x)
    n_layers: int = 4
    pad: tuple[int, int, int] = (4, 8, 8)       # rembourrage avant FFT (z, y, x)
    proj_hidden: int = 128
    local_conv: bool = True                     # branche convolutive locale 3x3x3
    pool_surface: bool = True                   # moyenne des capteurs par bloc stride x stride
    recalcul_activations: bool = False          # memoire GPU des grandes grilles (meme resultat)
    # sortie u, v = vent ERA5 au noeud + correction apprise (exige paths.forcage_era5) : le
    # modele part du niveau d'ERA5 (R2 0,95) au lieu de reconstruire la grande echelle
    residuel_era5: bool = False

    def __post_init__(self) -> None:
        self.modes = tuple(int(m) for m in self.modes)      # type: ignore[assignment]
        self.pad = tuple(int(p) for p in self.pad)          # type: ignore[assignment]


@dataclass
class TrainCfg:
    """BLOC 9 — entrainement."""

    epochs: int = 140
    steps_per_epoch: int = 400
    val_samples: int = 64
    batch_size: int = 1
    accum_steps: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_phys_epochs: int = 10
    num_workers: int = 4
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    save_every: int = 5
    seed: int = 42
    # --- raffinement a partir d'un modele deja entraine (v8) -------------------------
    # chemin d'un checkpoint dont on reprend les POIDS (ni l'optimiseur ni l'historique) :
    # on affine au lieu de repartir de zero. None = entrainement complet, comme la v7.
    init_from: str | None = None
    # critere de selection du meilleur modele :
    #   "val_score" : loss de validation (v7) ;
    #   "dso"       : pire rapport mesure / seuil des DSO sur la validation — c'est le
    #                 critere qui correspond a l'objectif, la loss moyenne ne l'est pas.
    select_by: str = "val_score"
    # Affinage sur les stations : codes des stations pouvant servir de CIBLE (None = toutes).
    # Validation croisee spatiale : on en retire une, jugee ensuite sans avoir jamais ete cible.
    stations_cibles: list[str] | None = None
    # False : last.pt (~7 Go) reste sur le disque de travail et n'est pas recopie dans
    # run_dir (Drive presque plein ; une coupure de session fait alors repartir de zero).
    sync_last: bool = True

    def __post_init__(self) -> None:
        if self.select_by not in ("val_score", "dso"):
            raise ValueError(f"train.select_by : 'val_score' ou 'dso', recu {self.select_by!r}")


@dataclass
class LossCfg:
    """BLOC 8 — ponderation des termes de la loss."""

    lambda_mass: float = 1.0
    mass_ref: float = 5e-7        # echelle du terme de masse (L_mass / mass_ref)
    lambda_bc: float = 0.1
    lambda_smooth: float = 0.0    # donnees deja lisses : desactive
    lambda_grad: float = 0.15
    w_weight: float = 1.0
    w_level_max: float = 25.0     # plafond du poids de w par niveau
    # Affinage sur les stations reelles (pcfno3d.stations_reelles.perte_station) : poids de
    # l'erreur du vent a 10 m a la station cible contre la mesure. 0 = desactive.
    lambda_station: float = 0.0
    # Poids de la VITESSE par station dans ce terme (1 par defaut) : 0 = direction seule, pour
    # une station abritee dont la vitesse n'est pas representative (Nabeul, INM60728).
    poids_vitesse_station: dict[str, float] | None = None
    # Poids de l'erreur de u et v PAR NIVEAU (une valeur par niveau, ou None).
    # La loss moyenne sur tous les noeuds ne correspond pas au DSO1, qui est un MAXIMUM
    # par niveau : la v7 echoue precisement aux niveaux hauts (650-1000 m) alors que la
    # moyenne est bonne. Ces poids, derives de la RMSE de VALIDATION par niveau
    # (`pcfno3d dso-weights`), donnent plus de poids aux niveaux hors seuil. None = v7.
    # Accepte aussi le chemin du JSON ecrit par `pcfno3d dso-weights`, qui garde la trace
    # du modele et du jeu (validation) dont les poids sont issus.
    uv_level_weight: list[float] | str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.uv_level_weight, (str, Path)):
            import json

            path = _resolve(self.uv_level_weight)
            if not path.exists():
                raise FileNotFoundError(f"loss.uv_level_weight : {path} introuvable "
                                        "(produit par `pcfno3d dso-weights`)")
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("jeu") == "test":
                raise ValueError(f"{path} a ete calcule sur le TEST : l'utiliser pour "
                                 "entrainer reviendrait a regler le modele sur son examen")
            self.uv_level_weight = data["uv_level_weight"]
        if self.uv_level_weight is not None:
            self.uv_level_weight = [float(w) for w in self.uv_level_weight]
            if any(w <= 0 for w in self.uv_level_weight):
                raise ValueError("loss.uv_level_weight : tous les poids doivent etre > 0")


@dataclass
class MaskCfg:
    """Masquage des capteurs d'entree (robustesse a un reseau incomplet)."""

    keep_train: tuple[float, float] = (0.2, 1.0)
    keep_eval: float = 0.5
    # Probabilite qu'un exemple d'entrainement garde TOUS ses capteurs.
    # En v7 (tirage uniforme dans keep_train), le cas « tous les capteurs » est un bord
    # de la distribution, et la v7 y est mesurablement PIRE qu'a 50 % (RMSE max u 0,705
    # contre 0,593). C'est pourtant le cas de production. 0 = comportement v7.
    p_full: float = 0.0
    # Entree du modele reel (Grand Tunis) : « capteurs » (v7 : couche de surface du cache,
    # masquee au hasard) ou « era5_stations » : vent ERA5 a 10 m (_E.npy) sur un reseau
    # regulier de pseudo-capteurs tous les `pas_era5_m`, plus les cellules des stations
    # reelles (data_dir/stations.json) avec la valeur du cache, chacune gardee avec la
    # probabilite `p_station` a l'entrainement (toutes en evaluation). C'est l'entree de
    # production : ERA5 (ou la prevision) + les anemometres.
    entree: str = "capteurs"
    pas_era5_m: float = 2500.0
    p_station: float = 0.8
    # D'ou vient le vent a 10 m du reseau regulier : « cache » (_E.npy du generateur, ERA5) ou
    # « forcage » : les cellules du forcage de l'instant (paths.forcage_era5) a 10 m au-dessus du
    # sol, interpolees comme le profil — c'est ainsi qu'une PREVISION GFS entre (OS7).
    e_source: str = "cache"
    # Stations en entree. False : prevision (OS7) — leurs mesures de demain n'existent pas.
    stations_entree: bool = True

    def __post_init__(self) -> None:
        if self.entree not in ("capteurs", "era5_stations"):
            raise ValueError(f"mask.entree : 'capteurs' ou 'era5_stations', recu {self.entree!r}")
        if self.e_source not in ("cache", "forcage"):
            raise ValueError(f"mask.e_source : 'cache' ou 'forcage', recu {self.e_source!r}")
        self.stations_entree = bool(self.stations_entree)
        self.keep_train = tuple(float(k) for k in self.keep_train)   # type: ignore[assignment]
        if not 0.0 <= float(self.p_full) <= 1.0:
            raise ValueError(f"mask.p_full doit etre dans [0, 1], recu {self.p_full}")
        self.p_full = float(self.p_full)


@dataclass
class InferenceCfg:
    """Reconstruction en production (API, MCP, reconstruct).

    tta : nombre de passes moyennees (test-time augmentation). Chaque passe masque une
          fraction differente des capteurs, avec les caracteristiques geostrophiques et
          thermiques RECALCULEES sur les capteurs gardes — sans quoi la passe sort de la
          distribution d'entrainement. 1 = une seule passe, comme la v7.
    tta_keep : fraction des capteurs gardee a chaque passe supplementaire.
    reliability : fiabilite par noeud (lot M2, pcfno3d.reliability, section `reliability`).
          None = desactivee : sortie et temps d'inference inchanges. Sinon, le chemin du JSON
          de calibration ecrit par `pcfno3d calibrate-reliability` (sur la VALIDATION), ou
          "non_calibre" pour des sigma bruts (calibrated: false dans la sortie).
    """

    tta: int = 1
    tta_keep: float = 0.7
    reliability: Path | str | None = None
    # w recalcule par CONTINUITE a partir de u, v predits (Geometry.w_continuite) : meme
    # schema discret que le residu de masse (DSO2), condition cinematique au pied de chaque
    # colonne, densite de l'atmosphere standard. False = w du reseau, comme la v7.
    w_continuite: bool = False

    def __post_init__(self) -> None:
        self.tta = int(self.tta)
        if self.tta < 1:
            raise ValueError("inference.tta doit etre >= 1")
        if not 0.0 < float(self.tta_keep) <= 1.0:
            raise ValueError("inference.tta_keep doit etre dans ]0, 1]")
        self.tta_keep = float(self.tta_keep)
        if self.reliability is not None and str(self.reliability) != "non_calibre":
            self.reliability = _resolve(self.reliability)


@dataclass
class ReliabilityCfg:
    """Fiabilite par noeud (lot M2, DSO6). Justification des valeurs : docs/FIABILITE.md."""

    members: int = 8                  # passes masquees de l'ensemble (sigma)
    keep: float = 0.7                 # fraction des capteurs gardee par passe (comme le TTA)
    seed: int = 2026                  # graine des masques : sortie deterministe
    sigma_floor_m_s: float = 0.01     # sigma plancher dans la calibration (division)
    per_level: bool = True            # k par niveau (sinon un k global par composante)
    dz_status_m: float = 15.0         # « mesure » : capteur visible a moins de dz (rev. 2, D7)
    coverage_length_m: float = 1000.0  # L : portee horizontale d'un capteur
    coverage_height_m: float = 500.0   # H : decroissance de la couverture avec la hauteur
    coverage_density_ref: float = 2.0  # D0 : densite de capteurs pour ~63 % de couverture
    sigma_ref_m_s: float = 1.0        # echelle de sigma_vitesse dans l'indice de fiabilite
    # seuils de classe = ceux du tableau de bord et des fixtures du contrat (0,75 / 0,5) :
    # le front reclasse depuis le champ `reliability` publie (docs/FIABILITE.md)
    class_high: float = 0.75          # fiabilite >= : « elevee »
    class_mid: float = 0.5            # fiabilite >= : « moyenne » (sinon « faible »)
    coverage_target: float = 0.8      # DSO6 : part des valeurs vraies dans l'intervalle 80 %
    # quantile de |erreur| / sigma vise par la calibration (validation). "auto" (defaut) : la
    # marge est choisie par une REGLE fixee d'avance, sur la validation seule (validation
    # croisee en 2 plis de jours, plus petit q de la grille 0,80..0,90 dont la couverture
    # hors-pli minimale sur u, v, w atteint coverage_target ; reliability.choose_q). Un
    # nombre impose q tel quel (ancien comportement ; 0,8 = intervalle a 80 % exactement).
    calibration_q: float | str = "auto"
    # passes masquees de l'ensemble faites en UN passage avant du reseau par lot (DSO4) ;
    # 1 = une passe a la fois (comportement d'avant). Memoire du passage x batch.
    batch: int = 4

    def __post_init__(self) -> None:
        self.members = int(self.members)
        if self.members < 2:
            raise ValueError("reliability.members doit etre >= 2")
        if not 0.0 < float(self.keep) < 1.0:
            raise ValueError("reliability.keep doit etre dans ]0, 1[")
        self.keep = float(self.keep)
        if not 0.0 <= self.class_mid < self.class_high <= 1.0:
            raise ValueError("reliability : 0 <= class_mid < class_high <= 1")
        q = self.calibration_q
        if isinstance(q, str) and q.strip().lower() == "auto":
            self.calibration_q = "auto"
        else:
            self.calibration_q = float(q)
            if not 0.5 <= self.calibration_q < 1.0:
                raise ValueError("reliability.calibration_q : \"auto\" ou un nombre de "
                                 "[0,5 ; 1[")
        self.batch = int(self.batch)
        if self.batch < 1:
            raise ValueError("reliability.batch doit etre >= 1")


@dataclass
class AlertsCfg:
    """Alertes (lot M4, critere AN1). Justification des valeurs : docs/ALERTES.md.

    Strictement opt-in : `enabled: false` par defaut, et rien ne change alors (ni la sortie
    du modele, ni les reponses de l'API ; /api/alerts repond 409).
    """

    enabled: bool = False
    journal: Path | str | None = None      # JSONL ; None = run_dir/logs/alertes.jsonl
    seuils: Path | str | None = None       # JSON regle sur la VALIDATION (m4_an1_colab.py)
    retention_close_h: float = 24.0        # une alerte close reste publiee 24 h
    max_par_type: int = 100                # au plus 100 alertes d'un meme type par instant
    # capteur en derive. v2 (defaut) : « spatiale », mesure comparee a la mediane de ses
    # voisins (aucune passe du modele) ; v1 : « innovation » / « influence », capteur masque
    # compare a la reconstruction faite sans lui ; « combinee » : spatiale + innovation.
    # Le choix et les seuils se reglent sur la validation (scripts/m4_an1_colab.py).
    derive_statistique: str = "spatiale"
    derive_rayon: int = 2                  # voisinage : carre de 2 r + 1 mailles (spatiale)
    derive_rayon_echelle: int = 2          # echelle locale : carre de 2 r + 1 mailles
    derive_min_voisins: int = 3            # voisins disponibles exiges (sinon non evalue)
    derive_sites: Path | str | None = None  # correction de site apprise sur l'ENTRAINEMENT
    derive_cote: int = 3                   # groupes en treillis cote x cote : cote^2 passes
    derive_fenetre: int = 4                # instants retenus pour la persistance
    derive_min_instants: int = 3           # instants observes exiges dans la fenetre
    derive_kappa: float = 6.0              # seuil en ecarts robustes (regle sur validation)
    derive_min_m_s: float = 0.5            # plancher de l'ecart moyen (tolerance OMM)
    derive_echelle_plancher_m_s: float = 0.05
    derive_facteur_critique: float = 2.0   # critique si ecart >= 2 x seuil
    # derive des entrees (statistiques de l'instant vs entrainement, norm_stats.json).
    # v2 : vent exclu (sa meteo faisait les fausses alertes de la v1), seuils regles sur
    # la validation par une grille fixee d'avance (an1.GRILLE_ENTREES).
    entrees_z_max: float = 5.0             # |moyenne - mu| / sd au-dela : derive
    entrees_ecart_extreme: float = 5.0     # valeur a plus de 5 sd : extreme
    entrees_part_extreme: float = 0.25     # derive si plus de 25 % des capteurs extremes
    entrees_variables: tuple[str, ...] = ("temperature_c", "air_density_kg_m3",
                                          "relative_humidity_pct", "pressure_pa")
    # coherence physique (contrat rev. 2, D11)
    rho_source: str = "extrapolation_surface"
    sol_rms_max_m_s: float | None = None   # condition au sol : pas d'alerte sans seuil regle
    masse_facteur_critique: float = 10.0
    # couverture (M2) : maille de 2 km
    couverture_maille_m: float = 2000.0
    couverture_min: float = 0.2
    # part de capteurs disponibles, sans rapport du contrat (instants du cache)
    capteurs_part_min: float = 0.70
    capteurs_part_critique: float = 0.30
    # reseau d'entree (snapshot.network.kind) : "treillis_synthetique" = un capteur par
    # maille ; "stations_aneo" = stations reelles, part rapportee aux stations ATTENDUES
    # (`stations_attendues` ; None = inconnu, pas d'alerte stations_insuffisantes)
    reseau: str = "treillis_synthetique"
    stations_attendues: int | None = None

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        if self.journal is not None and str(self.journal).lower() != "off":
            self.journal = _resolve(self.journal)
        if self.seuils is not None:
            self.seuils = _resolve(self.seuils)
        if self.derive_statistique not in ("influence", "innovation", "spatiale", "combinee"):
            raise ValueError("alerts.derive_statistique : 'spatiale', 'innovation', "
                             "'influence' ou 'combinee'")
        if self.derive_sites is not None:
            self.derive_sites = _resolve(self.derive_sites)
        self.entrees_variables = tuple(str(v) for v in self.entrees_variables)
        self.derive_rayon = int(self.derive_rayon)
        self.derive_rayon_echelle = int(self.derive_rayon_echelle)
        self.derive_min_voisins = int(self.derive_min_voisins)
        if self.derive_rayon < 1 or self.derive_rayon_echelle < 1:
            raise ValueError("alerts.derive_rayon et derive_rayon_echelle doivent etre >= 1")
        if self.reseau not in ("treillis_synthetique", "stations_aneo"):
            raise ValueError("alerts.reseau : treillis_synthetique ou stations_aneo")
        if self.rho_source not in ("atmosphere_standard", "extrapolation_surface",
                                   "verite_synthetique"):
            raise ValueError("alerts.rho_source : atmosphere_standard, "
                             "extrapolation_surface ou verite_synthetique")
        self.derive_cote = int(self.derive_cote)
        self.derive_fenetre = int(self.derive_fenetre)
        self.derive_min_instants = int(self.derive_min_instants)
        if self.derive_cote < 2:
            raise ValueError("alerts.derive_cote doit etre >= 2")
        if not 1 <= self.derive_min_instants <= self.derive_fenetre:
            raise ValueError("alerts : 1 <= derive_min_instants <= derive_fenetre")


@dataclass
class WindFarmCfg:
    """Eolien (lot M5, pcfno3d.windfarm, critere EOL1). Formules et sources : docs/EOLIEN.md.

    turbines : fichier YAML/JSON de declaration du parc (eoliennes + courbes de puissance).
          None = desactive : rien ne change (ni la sortie, ni le temps d'inference).
    Les seuils d'alerte sont des choix d'ingenierie documentes, a regler avec l'exploitant.
    """

    turbines: Path | str | None = None
    hub_height_agl_m: float = 100.0      # D10 : hauteur des KPI hub_wind_* (au-dessus du SOL)
    rews_segments: int = 20              # tranches horizontales du disque rotor (REWS)
    rho_radius_m: float = 2000.0         # capteur visible le plus proche pour la densite
    calm_ms: float = 0.2                 # direction indefinie en dessous (contrat dir_from_deg)
    shear_info: float = 0.3              # alerte cisaillement : alpha > seuil -> information
    shear_warn: float = 0.45             # -> avertissement
    shear_important: float = 0.6         # -> important
    high_wind_warn_frac: float = 0.9     # vent_fort_moyeu : p90 >= frac x cut-out -> avert.
    high_wind_critical_ms: float = 35.0  # vent_fort_moyeu : p50 >= -> critique
    sectors: int = 12                    # rose des vents : secteurs de largeur egale
    speed_bin_edges_ms: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 25.0)
    eol1_max_rel: float = 0.05           # EOL1 : ecart relatif d'energie du parc < 5 %

    def __post_init__(self) -> None:
        if self.turbines is not None:
            self.turbines = _resolve(self.turbines)
        self.rews_segments = int(self.rews_segments)
        if self.rews_segments < 3:
            raise ValueError("wind_farm.rews_segments doit etre >= 3 (IEC 61400-12-1)")
        self.sectors = int(self.sectors)
        if not 1 <= self.sectors <= 72:
            raise ValueError("wind_farm.sectors doit etre dans [1, 72]")
        edges = tuple(float(e) for e in self.speed_bin_edges_ms)
        if len(edges) < 2 or edges[0] != 0.0 or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("wind_farm.speed_bin_edges_ms : croissantes, la premiere a 0")
        self.speed_bin_edges_ms = edges
        if self.hub_height_agl_m <= 0:
            raise ValueError("wind_farm.hub_height_agl_m doit etre > 0 (au-dessus du sol)")


@dataclass
class DSOCfg:
    """Seuils des objectifs de l'entreprise (DSO1 a DSO5)."""

    rmse: float = 0.5                 # DSO1 : RMSE max de u, v par niveau (m/s)
    r2: float = 0.95                  # DSO1 : R2 min de u, v par niveau
    r2_all: float = 0.95              # objectif etendu : R2 min de u, v ET w
    mass_rms: float = 1e-4            # DSO2 : RMS du residu de continuite (kg.m-3.s-1)
    # DSO3 : rugosite prediction / verite dans [0,9 ; 1,1] (cahier des charges). Trop lisse
    # (< 0,9) est une faute autant que trop bruite (> 1,1). La v7 etait jugee sur (0,0 ; 1,1),
    # ce qui ne bornait que le haut ; sa valeur (0,951) est conforme dans les deux lectures.
    roughness_range: tuple[float, float] = (0.9, 1.1)
    time_s: float = 1.0               # DSO4 : temps d'inference par instant (s)
    rmse_25pct: float = 0.7           # DSO5 : RMSE max avec 25 % des capteurs
    min_level_cover: float = 0.05     # niveau ignore s'il couvre moins de 5 % du domaine
    # DSO1 REVISE, PROPOSE (pas encore valide par Tukhnanutha) : RMSE u, v par niveau
    # <= max(omm_floor ; omm_rel x vitesse moyenne du niveau), la tolerance de l'OMM sur la
    # mesure du vent (0,5 m/s jusqu'a 5 m/s, 10 % au-dela). Il COMPLETE le DSO1 officiel
    # (`rmse`), il ne le remplace pas : les deux verdicts sont toujours affiches.
    omm_floor: float = 0.5
    omm_rel: float = 0.10

    def __post_init__(self) -> None:
        self.roughness_range = tuple(float(r) for r in self.roughness_range)  # type: ignore[assignment]
        lo, hi = self.roughness_range
        if not 0.0 <= lo < 1.0 < hi:
            # le score DSO normalise par (1 - lo) et (hi - 1) : 1 doit etre strictement dedans
            raise ValueError(f"dso.roughness_range doit verifier 0 <= bas < 1 < haut, "
                             f"recu {self.roughness_range}")


@dataclass
class RealCfg:
    """Donnees reelles (pcfno3d.realdata) : ancrage geographique et controle qualite."""

    # coin SUD-OUEST du domaine, en degres. None = domaine non georeference (cas du jeu
    # synthetique) : l'import de donnees reelles et /api/reconstruct/stations refusent.
    origin_lat: float | None = None
    origin_lon: float | None = None
    rotation_deg: float = 0.0           # axe x du domaine depuis l'est, sens antihoraire
    # True = ancrage CHOISI, pas mesure : le domaine du generateur Tukhnanutha est
    # synthetique et n'a pas de position reelle. Affiche dans les reponses de l'API.
    origin_assumed: bool = False
    time_bin: str = "1h"                # regroupement temporel des mesures de stations
    ref_height_m: float | None = None   # hauteur cible du vent ; None = celle des capteurs
    min_quality: float = 0.5            # mesures de qualite inferieure ecartees
    max_speed_m_s: float = 75.0         # vent plus fort = mesure aberrante, ecartee
    split_period_days: int = 7          # 1 jour sur 7 en validation, 1 sur 7 en test


SECTIONS: dict[str, type] = {
    "paths": Paths, "model": ModelCfg, "train": TrainCfg,
    "loss": LossCfg, "mask": MaskCfg, "dso": DSOCfg, "inference": InferenceCfg,
    "real": RealCfg, "reliability": ReliabilityCfg, "alerts": AlertsCfg,
    "wind_farm": WindFarmCfg,
}


# --------------------------------------------------------------------------------------
# configuration complete
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    name: str = "v7"
    paths: Paths = field(default_factory=Paths)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    dso: DSOCfg = field(default_factory=DSOCfg)
    inference: InferenceCfg = field(default_factory=InferenceCfg)
    real: RealCfg = field(default_factory=RealCfg)
    reliability: ReliabilityCfg = field(default_factory=ReliabilityCfg)
    alerts: AlertsCfg = field(default_factory=AlertsCfg)
    wind_farm: WindFarmCfg = field(default_factory=WindFarmCfg)
    terrain_smooth_m: float = 300.0   # meme lissage du relief que le generateur
    f_coriolis: float = 8.57e-5       # s-1, latitude ~36 N (vent geostrophique)
    # capteurs visibles exiges pour ajuster le plan de pression / temperature (vent
    # geostrophique et thermique) ; en dessous ces 4 canaux valent 0. 100 = v7 (couche de
    # capteurs quasi continue) ; quelques-uns suffisent pour un plan a 3 parametres quand
    # les capteurs couvrent tout le domaine (Tunisie : 12 aeroports)
    capteurs_min_plan: int = 100

    # ---------------------------------------------------------------- chargement
    @classmethod
    def load(cls, source: str | Path | None = None, **overrides: Any) -> "Config":
        """Charge un YAML (nom court, chemin, ou None = defauts) puis applique les surcharges.

        `Config.load("test")` lit `configs/test.yaml`. Les surcharges sont pointees :
        `Config.load(train__epochs=2)`.
        """
        data: dict[str, Any] = {}
        if source is not None:
            path = Path(source)
            if not path.exists() and path.suffix == "":
                path = CONFIG_DIR / f"{source}.yaml"
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in SECTIONS:
                kwargs[key] = SECTIONS[key](**(value or {}))
            else:
                kwargs[key] = value
        cfg = cls(**kwargs)
        cfg.apply_overrides(overrides)
        cfg.apply_env()
        return cfg

    def apply_overrides(self, overrides: Mapping[str, Any]) -> None:
        """`{"train__epochs": 2}` ou `{"train.epochs": 2}` -> self.train.epochs = 2."""
        for key, value in overrides.items():
            if value is None:
                continue
            parts = key.replace("__", ".").split(".")
            target: Any = self
            for part in parts[:-1]:
                target = getattr(target, part)
            name = parts[-1]
            if not hasattr(target, name):
                raise KeyError(f"Option de configuration inconnue : {key}")
            setattr(target, name, value)
            post_init = getattr(target, "__post_init__", None)
            if post_init is not None:
                post_init()

    def apply_env(self) -> None:
        """Variables d'environnement du notebook (ANEO_DATA / ANEO_CACHE / ANEO_RUN)."""
        for attr, var in (("data_dir", "ANEO_DATA"), ("cache_dir", "ANEO_CACHE"),
                          ("run_dir", "ANEO_RUN")):
            if os.environ.get(var):
                setattr(self.paths, attr, Path(os.environ[var]))
        self.paths.__post_init__()

    def apply_service_env(self) -> "Config":
        """Chemins imposes par le service : PCFNO3D_RUN_DIR / _DATA_DIR / _CACHE_DIR.

        Un seul endroit pour ces variables : le service web (service/app.py) et le
        lanceur Colab (service/colab.serve_public) doivent resoudre les MEMES chemins,
        sinon le lanceur verifie un checkpoint ailleurs que la ou le service le lira.
        """
        for var, key in (("PCFNO3D_RUN_DIR", "paths.run_dir"),
                         ("PCFNO3D_DATA_DIR", "paths.data_dir"),
                         ("PCFNO3D_CACHE_DIR", "paths.cache_dir")):
            if os.environ.get(var):
                self.apply_overrides({key: Path(os.environ[var])})
        return self

    # ---------------------------------------------------------------- utilitaires
    def to_dict(self) -> dict[str, Any]:
        return _unbuild(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
        return path

    def seed_everything(self) -> None:
        import torch

        random.seed(self.train.seed)
        np.random.seed(self.train.seed)
        torch.manual_seed(self.train.seed)


def _unbuild(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _unbuild(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (tuple, list)):
        return [_unbuild(v) for v in obj]
    return obj


def pick_device(prefer: str | None = None):
    """GPU si disponible, avec TF32 active comme dans le notebook."""
    import torch

    name = prefer or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device
