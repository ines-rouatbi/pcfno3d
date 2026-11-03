"""Figures du rapport, tracées à partir des valeurs mesurées et archivées (sources en commentaire).

Lancer depuis ce dossier : python figures.py  -> ../figures/*.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SORTIE = Path(__file__).resolve().parents[1] / "figures"
SORTIE.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "figure.dpi": 150})
BLEU, ORANGE, GRIS, VERT, ROUGE = "#2b6cb0", "#dd6b20", "#718096", "#2f855a", "#c53030"


def lignee():
    # CLAUDE.md (« Grand Tunis v2 », nuit du 22-23/10) ; decision_v4.json (24/10)
    noms = ["v2\n(20 ép.)", "v2 ANEO\nsimulé", "v2 prolongée\n(+40 ép.)", "v4 DSO\n(livrée)"]
    u = [0.478, 0.548, 0.377, 0.380]
    v = [0.453, 0.514, 0.371, 0.373]
    x = np.arange(len(noms))
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(x - 0.18, u, 0.36, label="u", color=BLEU)
    ax.bar(x + 0.18, v, 0.36, label="v", color=ORANGE)
    ax.axhline(0.5, color=ROUGE, ls="--", lw=1)
    ax.text(3.45, 0.505, "seuil DSO1", color=ROUGE, ha="right", va="bottom", fontsize=8)
    for i in range(len(noms)):
        ax.text(x[i] - 0.18, u[i] + 0.01, f"{u[i]:.3f}", ha="center", fontsize=7)
        ax.text(x[i] + 0.18, v[i] + 0.01, f"{v[i]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, noms)
    ax.set_ylabel("RMSE global (m/s)")
    ax.set_ylim(0, 0.62)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(SORTIE / "lignee_grand_tunis.pdf")


def references():
    # docs/BILAN_FINAL.md § 6 (M8, test synoptique, 120 instants, 50 % des capteurs)
    noms = ["PC-FNO3D v7", "loi puiss.\n+ IDW", "loi puiss.\n+ krigeage", "profil log\n+ IDW",
            "profil log\n+ krigeage"]
    val = [0.665, 5.027, 5.031, 6.481, 6.487]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    barres = ax.bar(noms, val, color=[BLEU] + [GRIS] * 4)
    ax.axhline(0.5, color=ROUGE, ls="--", lw=1)
    for b, v in zip(barres, val):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("pire RMSE de niveau, u ou v (m/s)")
    ax.set_ylim(0, 7.3)
    fig.savefig(SORTIE / "references_m8.pdf")


def stations():
    # CLAUDE.md, validation sur mesures réelles (23/10, v2 prolongée, station exclue de l'entrée)
    st = ["Nabeul", "Tunis-Carthage", "Zaghouan"]
    modele, era5, gen = [2.42, 2.70, 2.28], [2.57, 2.81, 2.16], [2.52, 2.63, 2.14]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.bar(x - 0.27, modele, 0.27, label="PC-FNO3D", color=BLEU)
    ax.bar(x, era5, 0.27, label="ERA5 à 10 m", color=GRIS)
    ax.bar(x + 0.27, gen, 0.27, label="vérité générée", color=VERT)
    ax.set_xticks(x, st)
    ax.set_ylabel("RMSE vectoriel à 10 m (m/s)")
    ax.set_ylim(0, 3.3)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=8)
    fig.savefig(SORTIE / "stations_reelles.pdf")


def v3():
    # CLAUDE.md (v3, 23-24/10) : modèle final (3 stations cibles) et validation croisée spatiale
    st = ["Tunis-Carthage", "Zaghouan", "Nabeul"]
    v2 = [2.70, 2.28, 2.42]
    v3_final = [1.99, 1.74, 2.70]
    v3_cv = [2.66, 2.27, 2.41]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.bar(x - 0.27, v2, 0.27, label="v2 prolongée", color=GRIS)
    ax.bar(x, v3_final, 0.27, label="v3, station cible", color=ORANGE)
    ax.bar(x + 0.27, v3_cv, 0.27, label="v3, station jamais vue", color=BLEU)
    ax.set_xticks(x, st)
    ax.set_ylabel("RMSE vectoriel à 10 m (m/s)")
    ax.set_ylim(0, 3.3)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=8)
    fig.savefig(SORTIE / "v3_validation_croisee.pdf")


def os6():
    # runs/os6_reseau_aneo/os6.json (31/10/2026, test, 180 instants synoptiques)
    n = [0, 10, 20, 30, 50, 75, 100, 150, 200]
    actuel = 0.5277757787653758
    grille = [actuel, 0.5277281109938833, 0.5277310966019558, 0.5276875034382226, 0.5276902261369908,
              0.5277566177104455, 0.527740051641517, 0.5277515751822975, 0.5276469818773859]
    hasard = [actuel, 0.5277477924513566, 0.5277221143361733, 0.5277100250331336, 0.5276920088408928,
              0.5276683766129002, 0.5276658873429142, 0.5276726059996942, 0.5276715698755698]
    optimise = [actuel, 0.528215258154499, 0.5285356756860381, 0.5282894508070405, 0.5281527486731926,
                0.5282438571916046, 0.5282753540543315, 0.5287133548134314, 0.5288165111950934]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(n, grille, "o-", color=GRIS, label="grille régulière", ms=4)
    ax.plot(n, hasard, "s-", color=VERT, label="hasard", ms=4)
    ax.plot(n, optimise, "^-", color=ORANGE, label="placement optimisé", ms=4)
    ax.axhline(0.5, color=ROUGE, ls="--", lw=1)
    ax.text(200, 0.5015, "seuil 0,5 m/s", color=ROUGE, ha="right", fontsize=8)
    ax.set_xlabel("capteurs ANEO ajoutés")
    ax.set_ylabel("pire RMSE de niveau (m/s)")
    ax.set_ylim(0.495, 0.535)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(SORTIE / "os6_courbe.pdf")


def dso6():
    # decision_v4.json, bloc dso6 (v4, 180 instants) ; docs/BILAN_FINAL.md § 2 (v7)
    noms = ["u", "v", "w", "vitesse"]
    v7_1 = [0.751, 0.811, 0.794, 0.796]
    v7_2 = [0.871, 0.912, 0.896, 0.904]
    v4 = [0.878, 0.883, 0.841, 0.879]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.bar(x - 0.27, v7_1, 0.27, label="v7, 1re lecture", color=GRIS)
    ax.bar(x, v7_2, 0.27, label="v7, 2e lecture", color=VERT)
    ax.bar(x + 0.27, v4, 0.27, label="Grand Tunis v4", color=BLEU)
    ax.axhline(0.8, color=ROUGE, ls="--", lw=1)
    ax.set_xticks(x, noms)
    ax.set_ylabel("couverture de l'intervalle à 80 %")
    ax.set_ylim(0.6, 1.0)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    fig.savefig(SORTIE / "dso6_couverture.pdf")


if __name__ == "__main__":
    for f in (lignee, references, stations, v3, os6, dso6):
        f()
    print("figures ->", SORTIE)
