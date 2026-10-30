"""Rapport PDF automatique (cahier v12, E9) : une parcelle (agriculture), un site (eolienne de reference) ou
le parc virtuel, sur aujourd'hui, demain et apres-demain : vent heure par heure, rose des vents,
production, alertes, recommandations et fiabilite. Matplotlib seul (PdfPages), pas de dependance de plus.

Les valeurs viennent des memes fonctions que l'API et l'assistant (service.atom_outils) : le PDF dit ce
que dit l'application, avec les memes regles et les memes limites.
"""
from __future__ import annotations

import io
import re
import time

import numpy as np

TYPES = ("parcelle", "site", "parc")
ENCRE, ACCENT, SABLE = "#12263A", "#1E6FB8", "#B8925A"
COUL_DECISION = {"oui": "#1f9d55", "attention": "#d98a00", "non": "#d23a3a"}
NOMS_JOURS = ("Aujourd'hui", "Demain", "Après-demain")


def _nb(x: float, nd: int = 1) -> str:
    """Nombre a la francaise (virgule decimale)."""
    return f"{x:.{nd}f}".replace(".", ",")


def _jours(fn):
    """Les jours disponibles (0, 1, 2) : un jour absent arrete la serie."""
    out = []
    for j in (0, 1, 2):
        try:
            out.append((j, fn(j)))
        except Exception:                                             # noqa: BLE001
            break
    return out


def _rose(ax, directions, vitesses):
    secteurs = np.zeros(16)
    for d, s in zip(directions, vitesses):
        secteurs[int(((d + 11.25) % 360) // 22.5)] += s
    ang = np.radians(np.arange(16) * 22.5)
    ax.bar(ang, secteurs, width=np.radians(20), color=ACCENT, alpha=.8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "O"])
    ax.set_yticklabels([])
    ax.set_title("Rose des vents (provenance)", fontsize=9)


def _entete(fig, titre: str, sous_titre: str):
    fig.text(.06, .955, "ATOM Wind", fontsize=10, color=ACCENT, weight="bold")
    fig.text(.06, .925, titre, fontsize=16, color=ENCRE, weight="bold")
    fig.text(.06, .902, sous_titre, fontsize=9, color="#555")


def _pied(fig, texte: str):
    fig.text(.06, .03, texte, fontsize=7, color="#666", wrap=True)


def _texte(fig, y: float, lignes: list[str], taille: float = 9.5) -> float:
    for ligne in lignes:
        fig.text(.06, y, ligne, fontsize=taille, color=ENCRE, wrap=True)
        y -= .022
    return y


def construire(wf, type_: str, lat=None, lon=None, lieu=None, moyeu_m: float = 80.0) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from service import atom_outils as ao

    if type_ not in TYPES:
        raise ValueError(f"type de rapport : {', '.join(TYPES)}")
    genere = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    pied = (f"Généré le {genere} par ATOM Wind (modèle PC-FNO3D, {wf.checkpoint_path.name}). Vent : ATOM Wind ; "
            "température et humidité : GFS.\nAujourd'hui = temps réel, demain et après-demain = prévision. "
            "Estimation à vérifier avant une décision importante.")
    buf = io.BytesIO()
    with PdfPages(buf, metadata={"Title": f"ATOM Wind - rapport {type_}", "Creator": "ATOM Wind"}) as pdf:
        if type_ == "parcelle":
            jours = _jours(lambda j: ao.decision_agricole(wf, j, lat=lat, lon=lon, lieu=lieu))
            if not jours:
                raise ValueError("aucune journee disponible")
            nom = jours[0][1]["lieu"]
            fig = plt.figure(figsize=(8.27, 11.69))
            _entete(fig, f"Parcelle : {nom}", f"Traitements et irrigation, à partir du {jours[0][1]['date']}")
            y = .87
            for j, d in jours:
                cr = ", ".join(d["creneaux_traitement"]) or "aucun créneau"
                src = "temps réel" if d["source"] == "temps reel" else "prévision"
                y = _texte(fig, y, [f"{NOMS_JOURS[j]} ({d['date']}, {src}) : traiter {cr}."])
            y = _texte(fig, y - .005, ["Règle : traiter de 3 à 15 km/h de vent à 10 m ; attention sous 3 km/h et de 15 à "
                                        "19 km/h ; non au-delà.",
                                        "Respecter l'étiquette du produit et les zones non traitées."], 8.5)
            for k, (j, d) in enumerate(jours):
                ax = fig.add_axes([.08, y - .2 - k * .2, .84, .14])
                h = [r["heure"] for r in d["heures"]]
                v = [r["vent_10m_km_h"] for r in d["heures"]]
                ax.bar(h, v, color=[COUL_DECISION[r["traitement"]] for r in d["heures"]])
                ax.axhline(15, color=ENCRE, lw=.8, ls="--")
                ax.axhline(19, color=ENCRE, lw=.8, ls=":")
                ax.set_xlim(-.5, 23.5)
                ax.set_ylabel("km/h à 10 m", fontsize=8)
                ax.set_title(f"{NOMS_JOURS[j]} : vent à 10 m (vert : oui, orange : attention, rouge : non)", fontsize=9)
                ax.tick_params(labelsize=7)
            _pied(fig, pied)
            pdf.savefig(fig)
            plt.close(fig)
        elif type_ == "site":
            jours = _jours(lambda j: ao.production_eolienne(wf, j, moyeu_m, lat=lat, lon=lon, lieu=lieu))
            if not jours:
                raise ValueError("aucune journee disponible")
            nom = jours[0][1]["lieu"]
            prof = [ao.profil(wf, j, float(h), lat=lat, lon=lon, lieu=lieu) for j, _ in jours for h in range(0, 24, 3)]
            dirs = [n["direction_provenance_deg"] for p in prof for n in p["niveaux"] if n["hauteur_sol_m"] == 100]
            vits = [n["vitesse_m_s"] for p in prof for n in p["niveaux"] if n["hauteur_sol_m"] == 100]
            fig = plt.figure(figsize=(8.27, 11.69))
            _entete(fig, f"Site : {nom}", f"{jours[0][1]['eolienne']}, moyeu {moyeu_m:g} m")
            y = _texte(fig, .87, [f"{NOMS_JOURS[j]} ({e['date']}) : {_nb(e['energie_jour_mwh'])} MWh, facteur de charge "
                                  f"{100 * e['facteur_charge']:.0f} %" for j, e in jours])
            ax = fig.add_axes([.08, y - .3, .55, .25])
            for j, e in jours:
                ax.plot([r["heure"] for r in e["heures"]], [r["puissance_kw"] for r in e["heures"]], label=NOMS_JOURS[j])
            ax.set_xlabel("heure locale", fontsize=8)
            ax.set_ylabel("kW", fontsize=8)
            ax.set_title("Puissance prévue", fontsize=9)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
            _rose(fig.add_axes([.68, y - .3, .26, .25], projection="polar"), dirs, vits)
            m = ao.fenetres_maintenance(wf, 0, moyeu_m, 12.0, 2, lat=lat, lon=lon, lieu=lieu)
            _texte(fig, y - .36, ["Créneaux sûrs pour la maintenance aujourd'hui (vent moyen au moyeu sous 12 m/s) : "
                                  + (", ".join(m["creneaux_surs"]) or "aucun") + ".",
                                  "Production estimée pour une éolienne de référence, sans mesure de production."], 9)
            _pied(fig, pied)
            pdf.savefig(fig)
            plt.close(fig)
        else:
            jours = _jours(lambda j: ao.variations_production(wf, j, moyeu_m))
            if not jours:
                raise ValueError("aucune journee disponible")
            fig = plt.figure(figsize=(8.27, 11.69))
            _entete(fig, "Parc virtuel de Djebel Ressas", f"{jours[0][1]['puissance_installee_kw'] / 1000:g} MW installés, "
                    f"moyeu {moyeu_m:g} m")
            lignes = []
            for j, v in jours:
                r = v["rampes"]
                lignes.append(f"{NOMS_JOURS[j]} ({v['date']}) : {_nb(v['energie_jour_mwh'])} MWh ; "
                              + (f"{len(r)} rampe(s) d'au moins {v['seuil_pct_par_heure']:g} %/h : "
                                 + ", ".join(f"{x['sens']} {x['de']}-{x['a']} h ({x['variation_pct']:+.0f} %)" for x in r[:4])
                                 if r else "pas de variation brusque"))
            y = _texte(fig, .87, lignes)
            ax = fig.add_axes([.08, y - .3, .84, .25])
            for j, v in jours:
                ax.step(range(24), v["puissance_kw"], where="mid", label=NOMS_JOURS[j])
            ax.set_xlabel("heure locale", fontsize=8)
            ax.set_ylabel("kW", fontsize=8)
            ax.set_title("Puissance du parc", fontsize=9)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
            _pied(fig, pied)
            pdf.savefig(fig)
            plt.close(fig)

        # page 2 : alertes et fiabilite
        fig = plt.figure(figsize=(8.27, 11.69))
        _entete(fig, "Alertes et fiabilité", "Vent fort sur le Grand Tunis et origine des valeurs")
        lignes = []
        for j, a in _jours(lambda j: ao.alertes_vent(wf, j)):
            m10, m100 = a.get("max_10m"), a.get("max_100m")
            txt = f"{NOMS_JOURS[j]} ({a['date']}) : "
            if m10:
                txt += f"maximum à 10 m {m10['max']:.0f} {m10['unite']} vers {m10['heure_locale']:.0f} h"
            if m100:
                txt += f" ; à 100 m {_nb(m100['max'])} {m100['unite']}"
            n = len(a["alertes"])
            txt += " ; aucune alerte." if n == 0 else f" ; {n} alerte{'s' if n > 1 else ''}."
            lignes.append(txt)
        y = _texte(fig, .87, lignes or ["Alertes indisponibles."])
        fiab = ["Fiabilité : le vent est calculé à partir de GFS (grande échelle) et des anémomètres",
                "de Tunis-Carthage, Zaghouan et Nabeul ; loin des stations, l'erreur probable grandit",
                "(vue Fiabilité de l'application).",
                "Validation sur mesures réelles (test 2022) : erreur vectorielle à 10 m de 2,3 à 2,7 m/s",
                "aux stations, au niveau de la réanalyse ERA5 ; de 75 à 1 000 m, écart au champ de",
                "référence sous 0,42 m/s."]
        # influence des stations : seulement si le deni de donnees du jour est deja calcule (calcul lourd)
        from pcfno3d import atom_wind
        e = None
        if type_ != "parc" and ao._cle(wf, atom_wind.date_locale_aujourdhui(0)) in ao._CACHES["valeur"]:
            try:
                e = ao.explication_point(wf, lat=lat, lon=lon, lieu=lieu)
            except Exception:                                         # noqa: BLE001
                e = None
        if e and e["stations"]:
            fiab.append(e["texte"].replace("Ici", f"À {e['lieu']}").replace("echelle", "échelle"))
        _texte(fig, y - .02, fiab, 9)
        _pied(fig, pied)
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()


def nom_fichier(type_: str, lieu: str | None, lat, lon) -> str:
    base = (lieu or (f"{lat:.3f}N_{lon:.3f}E" if lat is not None and lon is not None else "parc")).lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return f"atom_wind_{type_}_{base}_{time.strftime('%Y%m%d')}.pdf"

