"""Supervision de l'activite du service (cahier v12, BF41) : un seul document qui dit si ATOM Wind travaille
normalement. Lecture seule de ce que le service a deja : journee temps reel en cache, historique, journal
d'acces (comptes seulement, aucune adresse ni cle), notifications, et l'ecart du modele aux stations sur
les heures deja mesurees du jour.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# anemometres du Grand Tunis (lat, lon) : memes points que la validation reelle
STATIONS = {"DTTA": ("Tunis-Carthage", 36.8504, 10.2156), "INM60729": ("Zaghouan", 36.4333, 10.0833),
            "INM60728": ("Nabeul", 36.45, 10.7333)}


def acces(chemin: Path | None, heures: float = 24.0) -> dict:
    """Comptes du journal d'acces sur les dernieres `heures` (rotation .1 comprise)."""
    if chemin is None:
        return {"journal": "absent"}
    limite = datetime.now(timezone.utc) - timedelta(hours=heures)
    n, statuts, routes, durees, cles = 0, Counter(), Counter(), [], set()
    for f in (Path(str(chemin) + ".1"), Path(chemin)):
        if not f.exists():
            continue
        for ligne in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                d = json.loads(ligne)
                t = datetime.fromisoformat(d["ts"].replace("Z", "+00:00"))
            except (ValueError, KeyError):
                continue
            if t < limite:
                continue
            n += 1
            statuts[f"{int(d.get('statut', 0)) // 100}xx"] += 1
            routes[d.get("chemin", "?")] += 1
            if isinstance(d.get("duree_ms"), (int, float)):
                durees.append(d["duree_ms"])
            if d.get("cle"):
                cles.add(d["cle"])
    durees.sort()
    return {"heures": heures, "requetes": n, "par_statut": dict(statuts),
            "refus": sum(v for k, v in statuts.items() if k == "4xx"), "erreurs_serveur": statuts.get("5xx", 0),
            "cles_actives": len(cles), "routes_les_plus_demandees": routes.most_common(5),
            "duree_mediane_ms": durees[len(durees) // 2] if durees else None,
            "duree_p95_ms": durees[int(len(durees) * 0.95)] if durees else None}


def ecart_stations(wf) -> dict:
    """Erreur vectorielle a 10 m du modele servi aux stations, sur les heures deja mesurees du jour (la
    station est dans l'entree : c'est un controle de coherence, pas une validation independante)."""
    from service import atom_outils as ao
    from pcfno3d import temps_reel as tr
    import pandas as pd

    p = ao.jour(wf, 0)
    obs = ao.source(wf).journee(p["date"])["obs"]
    sortie = {}
    for code, (nom, la, lo) in STATIONS.items():
        try:
            x, y, _ = ao.position(wf, lat=la, lon=lo)
        except ValueError:
            continue
        e2 = []
        for it in p["instants"]:
            if it.get("source") != "temps reel" or code not in it.get("stations", []):
                continue
            t = pd.Timestamp(it["timestamp_utc"]).tz_localize(None)
            m = tr.mesures_a(obs[obs["station_id"] == code], t).get(code)
            if m is None:
                continue
            u, v = ao.vent(wf, p, x, y, 10.0, float(it["heure_locale"]))
            e2.append((u - m[0]) ** 2 + (v - m[1]) ** 2)
        sortie[nom] = {"heures": len(e2), "rmse_vect_m_s": round(math.sqrt(sum(e2) / len(e2)), 2) if e2 else None}
    return {"date": p["date"], "stations": sortie,
            "note": "station presente dans l'entree : controle de coherence (la validation independante est REEL1)"}


def etat(wf, journal_acces: Path | None) -> dict:
    from service import atom_outils as ao
    t0 = time.time()
    run = Path(wf.cfg.paths.run_dir)
    direct = {}
    for (_, date), e in ao._CACHES["direct"].items():
        out = e["out"]
        mes = [i for i in out["instants"] if i.get("source") == "temps reel"]
        direct[date] = {"calcule_il_y_a_min": round((time.time() - e["t"]) / 60, 1),
                        "mis_a_jour_utc": out.get("mis_a_jour_utc"), "heures_mesurees": len(mes),
                        "stations_du_jour": sorted({s for i in mes for s in i.get("stations", [])}),
                        "heures_masse_hors_seuil": sum(1 for i in out["instants"] if i.get("residu_masse_ok") is False),
                        "inference_s_mediane": sorted(i["inference_s"] for i in out["instants"])[len(out["instants"]) // 2]
                        if out["instants"] else None}
    hist = ao.historique(wf)
    nd = run / "notifications"
    envoyees = json.loads((nd / "envoyees.json").read_text(encoding="utf-8")) if (nd / "envoyees.json").exists() else []
    try:
        ecart = ecart_stations(wf)
    except Exception as exc:                                        # noqa: BLE001 — source muette
        ecart = {"erreur": str(exc)[:200]}
    return {"genere_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "modele": wf.checkpoint_path.name, "configuration": wf.cfg.name,
            "temps_reel": direct or {"etat": "aucune journee calculee depuis le demarrage"},
            "historique": {"journees": len(hist), "derniere": hist[0]["nom"] if hist else None,
                           "taille_mo": round(sum(h["taille_ko"] for h in hist) / 1024, 1)},
            "acces_24h": acces(journal_acces),
            "notifications": {"alertes_deja_envoyees": len(envoyees)},
            "ecart_modele_stations": ecart, "duree_s": round(time.time() - t0, 2)}
