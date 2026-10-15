"""Serveur MCP exposant la sortie du modele PC-FNO3D.

Il permet a une autre application (ou a un agent) d'interroger le champ de vent reconstruit
sans rien savoir du modele : chaque outil renvoie du JSON deja agrege. C'est le point
d'entree prevu pour l'app qui consommera ces donnees.

Deux transports, le meme jeu d'outils :

    stdio            `python -m service.mcp_server`            (client local, ex. Claude Desktop)
    streamable HTTP  monte sur /mcp par `service/app.py`       (client distant, avec cle d'API)

Outils :
    grid_meta                 geometrie de sortie, niveaux, modele charge
    list_instants             instants disponibles (filtre par ensemble)
    wind_at_point             vent en un point (x, y, altitude)
    wind_profile              profil vertical complet en un point
    wind_field_at_altitude    carte decimee a une altitude
    instant_summary           statistiques d'un instant (vitesse, direction, temps d'inference)
    dso_report                dernier verdict DSO1-DSO5 archive
    alertes_actives           alertes ouvertes d'un instant (lot M4), si alerts.enabled

Les outils de lecture n'ecrivent rien. `export_instant` ecrit sur le disque du serveur et
exige donc la portee `admin` lorsque l'authentification est active.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

import numpy as np
from mcp.server.mcpserver import MCPServer

from pcfno3d.config import Config
from pcfno3d.inference import WindFieldModel

INSTRUCTIONS = """\
Champ de vent 3D reconstruit sur la Tunisie (reseau de capteurs ANEO, modele PC-FNO3D).

Un « instant » est un pas de temps, identifie par un stem de la forme JJJ_HHMMSS
(ex. 004_120000). Commence par grid_meta pour connaitre la grille et les altitudes
disponibles, puis list_instants pour choisir un instant.

Conventions : u est positif vers l'est, v vers le nord, w vers le haut. La direction est
donnee en convention meteo (0 = vent venant du nord, 90 = de l'est). Les altitudes sont
en metres au-dessus du niveau de la mer (ASL) ; sous le relief, les valeurs sont nulles
(null en JSON). Les coordonnees x et y sont en metres depuis le coin sud-ouest du domaine.
"""

# ---------------------------------------------------------------------------- modele
_MODEL: WindFieldModel | None = None
_VOLUMES: dict[str, Any] = {}
_VOLUME_LOCK = threading.Lock()
MAX_CACHED_VOLUMES = 4


def load_config() -> Config:
    return Config.load(os.environ.get("PCFNO3D_CONFIG") or "default").apply_service_env()


def get_model() -> WindFieldModel:
    """Charge le modele au premier appel (geometrie + statistiques + poids)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = WindFieldModel(load_config(), device=os.environ.get("PCFNO3D_DEVICE") or None,
                               verbose=False)
    return _MODEL


def set_model(model: WindFieldModel) -> None:
    """Partage le modele deja charge par l'API HTTP (une seule copie en memoire)."""
    global _MODEL
    _MODEL = model
    _VOLUMES.clear()


_ALERTES: Any = None
_ALERTES_FIXE = False


def set_alertes(moteur) -> None:
    """Partage le moteur d'alertes de l'API HTTP (un seul journal), ou None (desactive)."""
    global _ALERTES, _ALERTES_FIXE
    _ALERTES, _ALERTES_FIXE = moteur, True


def get_alertes():
    """Moteur d'alertes (lot M4) ; en stdio, construit au premier appel si alerts.enabled."""
    global _ALERTES
    if _ALERTES is None and not _ALERTES_FIXE:
        cfg = get_model().cfg
        if cfg.alerts.enabled:
            from pcfno3d.alerts import MoteurAlertes
            _ALERTES = MoteurAlertes(cfg)
    return _ALERTES


def _default_stem(wf: WindFieldModel) -> str:
    from pcfno3d.cache import load_index

    index = load_index(wf.cfg)
    test = index[index["split"] == "test"]
    rows = test if len(test) else index
    if rows.empty:
        raise ValueError("aucun instant en cache")
    return str(rows["stem"].iloc[0])


def _volume(stem: str | None):
    """Volume d'un instant, memorise : une reconstruction coute une passe avant."""
    wf = get_model()
    stem = stem or _default_stem(wf)
    volume = _VOLUMES.get(stem)
    if volume is not None:
        return stem, volume
    # meme double verification que l'API : deux outils appeles ensemble sur un nouvel
    # instant ne doivent pas lancer deux passes avant
    with _VOLUME_LOCK:
        volume = _VOLUMES.get(stem)
        if volume is None:
            volume = wf.predict_stem(stem)
            _VOLUMES[stem] = volume
            if len(_VOLUMES) > MAX_CACHED_VOLUMES:
                _VOLUMES.pop(next(iter(_VOLUMES)))
    return stem, volume


def _clean(x: float) -> float | None:
    """JSON n'admet pas NaN : sous le relief, la valeur devient null."""
    return None if not np.isfinite(x) else round(float(x), 3)


# ---------------------------------------------------------------------------- serveur
mcp = MCPServer(
    name="pcfno3d-vent-3d",
    title="PC-FNO3D — champ de vent 3D (Tunisie)",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)


@mcp.tool()
def grid_meta() -> dict:
    """Geometrie de sortie du modele : grille, altitudes disponibles, modele charge.

    A appeler en premier : les autres outils attendent des altitudes et des coordonnees
    coherentes avec cette grille.
    """
    wf = get_model()
    meta = wf.meta()
    g = wf.geom
    meta["domaine_m"] = {"x_max": float((g.nx - 1) * g.dx), "y_max": float((g.ny - 1) * g.dy)}
    meta["altitude_min_asl_m"] = float(g.z_levels[0])
    meta["altitude_max_asl_m"] = float(g.z_levels[-1])
    return meta


@mcp.tool()
def list_instants(split: str | None = None, limit: int = 50) -> dict:
    """Instants reconstructibles, avec leur date et leur ensemble.

    Args:
        split: "train", "val" ou "test" pour filtrer ; None = tous.
        limit: nombre maximal d'instants renvoyes (1-500).
    """
    from pcfno3d.cache import load_index

    wf = get_model()
    index = load_index(wf.cfg)
    if split:
        if split not in {"train", "val", "test"}:
            raise ValueError('split attendu : "train", "val" ou "test"')
        index = index[index["split"] == split]
    limit = max(1, min(int(limit), 500))
    rows = index[["stem", "timestamp", "split"]].head(limit)
    return {"total": int(len(index)), "renvoyes": int(len(rows)),
            "instants": rows.to_dict(orient="records")}


@mcp.tool()
def wind_at_point(x_m: float, y_m: float, z_asl_m: float, stem: str | None = None) -> dict:
    """Vent reconstruit en un point : u, v, w, vitesse et direction meteo.

    Args:
        x_m: distance en metres vers l'est depuis le coin sud-ouest.
        y_m: distance en metres vers le nord.
        z_asl_m: altitude au-dessus du niveau de la mer, interpolee entre deux niveaux.
        stem: instant (JJJ_HHMMSS) ; None = premier instant de test.
    """
    stem, volume = _volume(stem)
    point = volume.at_point(x_m, y_m, z_asl_m)
    point["stem"] = stem
    return point


@mcp.tool()
def wind_profile(x_m: float, y_m: float, stem: str | None = None) -> dict:
    """Profil vertical complet en un point : une ligne par niveau du modele.

    Utile pour un calcul de production eolienne : le vent a la hauteur de moyeu se lit
    directement, et le cisaillement se voit d'un niveau a l'autre.

    Args:
        x_m: distance en metres vers l'est depuis le coin sud-ouest.
        y_m: distance en metres vers le nord.
        stem: instant (JJJ_HHMMSS) ; None = premier instant de test.
    """
    stem, volume = _volume(stem)
    g = volume.geom
    i = int(np.clip(round(x_m / g.dx), 0, g.nx - 1))
    j = int(np.clip(round(y_m / g.dy), 0, g.ny - 1))
    niveaux = []
    for k, z in enumerate(g.z_levels):
        u, v, w = volume.u[k, j, i], volume.v[k, j, i], volume.w[k, j, i]
        niveaux.append({
            "z_asl_m": float(z),
            "z_agl_m": round(float(z - g.zt[j, i]), 2),
            "sous_le_relief": not bool(np.isfinite(u)),
            "wind_u_m_s": _clean(u), "wind_v_m_s": _clean(v), "wind_w_m_s": _clean(w),
            "wind_speed_m_s": _clean(float(np.hypot(u, v))),
            "wind_direction_deg": _clean(float((270.0 - np.degrees(np.arctan2(v, u))) % 360.0)),
        })
    return {"stem": stem, "timestamp_utc": volume.timestamp,
            "x_m": i * g.dx, "y_m": j * g.dy, "z_terrain_m": float(g.zt[j, i]),
            "niveaux": niveaux}


@mcp.tool()
def wind_field_at_altitude(z_asl_m: float, stem: str | None = None, variable: str = "speed",
                           stride: int = 8) -> dict:
    """Carte d'une variable a une altitude, decimee pour rester lisible.

    Args:
        z_asl_m: altitude ASL en metres (interpolee entre niveaux).
        stem: instant (JJJ_HHMMSS) ; None = premier instant de test.
        variable: "speed", "u", "v", "w" ou "direction".
        stride: decimation horizontale (1-32) ; 8 donne une grille d'environ 60x30.
    """
    if variable not in {"speed", "u", "v", "w", "direction"}:
        raise ValueError('variable attendue : "speed", "u", "v", "w" ou "direction"')
    stem, volume = _volume(stem)
    source = {"speed": volume.speed, "u": volume.u, "v": volume.v, "w": volume.w,
              "direction": volume.direction}[variable]
    plane = volume.at_altitude(z_asl_m, source)
    stride = max(1, min(int(stride), 32))
    plane = plane[::stride, ::stride]
    g = volume.geom
    finite = plane[np.isfinite(plane)]
    return {
        "stem": stem, "timestamp_utc": volume.timestamp, "variable": variable,
        "z_asl_m": z_asl_m, "stride": stride,
        "dx_m": g.dx * stride, "dy_m": g.dy * stride,
        "ny": int(plane.shape[0]), "nx": int(plane.shape[1]),
        "min": _clean(float(finite.min())) if finite.size else None,
        "max": _clean(float(finite.max())) if finite.size else None,
        "ordre": "lignes du sud vers le nord, colonnes de l'ouest vers l'est",
        "valeurs": [[_clean(v) for v in row] for row in plane],
    }


@mcp.tool()
def instant_summary(stem: str | None = None) -> dict:
    """Resume chiffre d'un instant : vitesses par niveau, secteur dominant, temps d'inference.

    Args:
        stem: instant (JJJ_HHMMSS) ; None = premier instant de test.
    """
    stem, volume = _volume(stem)
    g = volume.geom
    niveaux = []
    for k, z in enumerate(g.z_levels):
        m = g.valid[k]
        if not m.any():
            continue
        s = volume.speed[k][m]
        niveaux.append({"z_asl_m": float(z), "couverture": round(float(g.cover[k]), 3),
                        "vitesse_moyenne_m_s": _clean(float(s.mean())),
                        "vitesse_max_m_s": _clean(float(s.max())),
                        "w_moyen_m_s": _clean(float(volume.w[k][m].mean()))})
    valid = g.valid
    dirs = volume.direction[valid]
    secteurs = np.histogram(dirs, bins=8, range=(0, 360))[0]
    labels = ["0", "45", "90", "135", "180", "225", "270", "315"]
    return {
        "stem": stem, "timestamp_utc": volume.timestamp,
        "capteurs_utilises": volume.n_sensors,
        "inference_s": round(volume.inference_s, 4),
        "noeuds_3d": int(valid.sum()),
        "vitesse_moyenne_m_s": _clean(float(volume.speed[valid].mean())),
        "vitesse_max_m_s": _clean(float(volume.speed[valid].max())),
        "secteur_dominant_deg": labels[int(secteurs.argmax())],
        "par_niveau": niveaux,
    }


@mcp.tool()
def dso_report() -> dict:
    """Dernier verdict DSO1-DSO5 archive pour le modele charge.

    Renvoie le contenu de `test_summary.json` produit par l'evaluation. Si aucune
    evaluation n'a encore ete archivee, le dit explicitement plutot que d'inventer
    des chiffres.
    """
    from pcfno3d.evaluate import EVAL_NAMESPACE

    wf = get_model()
    run_dir, stem = wf.cfg.paths.run_dir, wf.checkpoint_path.stem
    tta = wf.cfg.inference.tta
    # ordre de preference : l'evaluation qui correspond EXACTEMENT a ce qui est servi
    # (meme modele, meme nombre de passes), puis celle du modele sans moyenne, puis
    # celle ecrite par le notebook v7 (eval_<modele>/), puis n'importe laquelle
    preferes = [run_dir / EVAL_NAMESPACE / (f"{stem}_tta{tta}" if tta > 1 else stem),
                run_dir / EVAL_NAMESPACE / stem,
                run_dir / f"eval_{stem}"]
    out = next((p / "test_summary.json" for p in preferes
                if (p / "test_summary.json").exists()), None)
    if out is None:
        candidates = sorted(run_dir.glob(f"{EVAL_NAMESPACE}/*/test_summary.json")) \
            + sorted(run_dir.glob("eval_*/test_summary.json"))
        if not candidates:
            return {"disponible": False,
                    "message": "aucune evaluation archivee ; lancer "
                               "`python -m pcfno3d evaluate`",
                    "modele": wf.checkpoint_path.name}
        out = candidates[-1]
    summary = json.loads(out.read_text(encoding="utf-8"))
    # la table de robustesse (4 lignes) est gardee : c'est elle qui montre qu'un modele
    # fait mieux avec moins de capteurs, le defaut que corrige mask.p_full
    return {"disponible": True, "fichier": str(out), "resume": summary}


@mcp.tool()
def alertes_actives(stem: str | None = None) -> dict:
    """Alertes OUVERTES d'un instant (lot M4) : capteurs en derive ou muets, stations
    insuffisantes, incoherence physique, derive des entrees, zones mal couvertes.

    Chaque alerte porte un niveau (information, avertissement, important, critique), une
    cause et une recommandation ; format du contrat ATOM rev. 2 (alerts.schema.json).
    Renvoie aussi le nombre d'alertes ouvertes par niveau et le bloc `physique` (residu
    de masse de l'instant). Si les alertes sont desactivees (alerts.enabled = false), le
    dit au lieu d'inventer une liste vide.

    Args:
        stem: instant (JJJ_HHMMSS) ; None = dernieres alertes calculees, sinon le premier
              instant de test.
    """
    moteur = get_alertes()
    if moteur is None:
        return {"disponible": False,
                "message": "alertes desactivees (alerts.enabled = false dans la configuration)"}
    if stem is None and moteur.dernier is not None:
        res = moteur.dernier
    else:
        stem, volume = _volume(stem)
        wf = get_model()
        S, hour, doy, _ = wf.surface_from_cache(stem)
        res = moteur.evaluer_modele(wf, S, hour, doy, volume, cle=f"stem:{stem}")
    ouvertes = [a for a in res.items if a["status"] == "ouverte"]
    return {"disponible": True, "stem": stem, "instant": res.instant,
            "ouvertes_par_niveau": res.ouvertes(), "items": ouvertes,
            "closes_recentes": len(res.items) - len(ouvertes),
            "physique": res.physics, "derive_capteurs": res.derive}


@mcp.tool()
def export_instant(stem: str | None = None, stride: int = 2) -> dict:
    """Ecrit un instant sur le disque du serveur (binaires Float32 + manifeste).

    Destine a l'application qui consomme les donnees : elle lit ensuite le manifeste et
    les binaires. Ecrit des fichiers, donc reserve a la portee `admin`.

    Args:
        stem: instant (JJJ_HHMMSS) ; None = premier instant de test.
        stride: decimation horizontale (1 = pleine resolution).
    """
    from pcfno3d.export import export_web

    stem, volume = _volume(stem)
    wf = get_model()
    out = export_web(volume, wf.cfg.paths.web_root / f"recon_{stem}",
                     stride=max(1, min(int(stride), 8)))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return {"stem": stem, "dossier": str(out), "manifeste": manifest}


# outils qui modifient l'etat du serveur : portee admin quand l'auth est active
ADMIN_TOOLS = {"export_instant"}


def main() -> int:
    """Transport stdio, pour un client MCP local."""
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
