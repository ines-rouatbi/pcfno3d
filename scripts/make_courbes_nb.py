"""Carnet « courbes de résultats » (Colab CPU) : trace les courbes du Grand Tunis v4 et de la prévision v5 à partir
des fichiers DÉJÀ écrits sur le Drive (history.json, test_par_niveau_synoptique.csv, prevision_bilan.json).
Aucun modèle n'est chargé, aucun entraînement, aucune nouvelle lecture du test : on trace des résultats existants.
Sortie : notebooks/current/pcfno3d_courbes_resultats_colab.ipynb ; figures copiées dans MyDrive/pcfno3d_courbes/."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "current" / "pcfno3d_courbes_resultats_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


C0 = r'''from google.colab import drive
import os
if not os.path.isdir("/content/drive/MyDrive"):
    drive.mount("/content/drive")
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RUNS = Path("/content/drive/MyDrive/pcfno3d_grand_tunis/runs")
OS7 = Path("/content/drive/MyDrive/pcfno3d_grand_tunis/os7")
FIG = Path("/content/drive/MyDrive/pcfno3d_courbes")          # dossier neuf : rien n'est ecrase ailleurs
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

def historique(run):
    h = json.loads((RUNS / run / "history.json").read_text())
    h = [r for r in h if r.get("epoch", -1) >= 0 and isinstance(r.get("train"), dict)]
    df = pd.DataFrame({"epoque": [r["epoch"] + 1 for r in h],
                       "train": [r["train"].get("data") for r in h],
                       "validation": [r["val"].get("data") for r in h],
                       "r2_u": [r.get("val_r2_u") for r in h], "r2_v": [r.get("val_r2_v") for r in h],
                       "r2_w": [r.get("val_r2_w") for r in h]})
    print(run, ":", len(df), "epoques lues dans", RUNS / run / "history.json")
    return df

def courbes_entrainement(df, titre, nom):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(df["epoque"], df["train"], "o-", label="entrainement")
    ax[0].plot(df["epoque"], df["validation"], "s-", label="validation")
    ax[0].set_yscale("log"); ax[0].set_xlabel("Epoque"); ax[0].set_title("Erreur sur les donnees (normalisee)"); ax[0].legend()
    for c, m in (("r2_u", "o"), ("r2_v", "s"), ("r2_w", "^")):
        ax[1].plot(df["epoque"], df[c], m + "-", label="R2 " + c[-1])
    ax[1].axhline(0.95, color="red", ls="--", label="seuil 0,95")
    ax[1].set_xlabel("Epoque"); ax[1].set_title("R2 sur la validation"); ax[1].legend()
    fig.suptitle(titre); fig.tight_layout(); fig.savefig(FIG / nom, dpi=150); plt.show()
'''

C1 = r'''# 1. Grand Tunis v4 : entrainement et validation (affinage de 15 epoques)
v4 = historique("grand_tunis_v4_dso")
courbes_entrainement(v4, "Grand Tunis v4 — entrainement et validation (history.json)", "01_gt_v4_entrainement.png")
print(v4.round(4).to_string(index=False))
'''

C2 = r'''# 2. Grand Tunis : toute la chaine v2 (20 epoques) -> v2_long (+40) -> v4 (+15), validation
chaine, debut = [], 0
for run in ("grand_tunis_v2", "grand_tunis_v2_long", "grand_tunis_v4_dso"):
    try:
        d = historique(run); d["epoque_totale"] = d["epoque"] + debut; d["run"] = run
        chaine.append(d); debut = int(d["epoque_totale"].max())
    except FileNotFoundError:
        print(run, ": history.json absent, saute")
if chaine:
    c = pd.concat(chaine)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(c["epoque_totale"], c["train"], label="entrainement")
    ax.plot(c["epoque_totale"], c["validation"], label="validation")
    for d in chaine[1:]:
        ax.axvline(d["epoque_totale"].min() - 0.5, color="grey", ls=":")
    ax.set_yscale("log"); ax.set_xlabel("Epoque cumulee (v2 | v2_long | v4)"); ax.legend()
    ax.set_title("Grand Tunis — erreur d'entrainement et de validation sur toute la chaine")
    fig.tight_layout(); fig.savefig(FIG / "02_gt_chaine_entrainement.png", dpi=150); plt.show()
'''

C3 = r'''# 3. Grand Tunis v4 : resultats du TEST 2022 par altitude (fichier deja ecrit le 24/10, le test n'est pas relu)
f = RUNS / "grand_tunis_v4_dso" / "eval_pcfno3d" / "best_wcont" / "test_par_niveau_synoptique.csv"
t = pd.read_csv(f); print("lu :", f); print(t.round(3).to_string(index=False))
t = t[t["couverture"] >= 0.05]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for c, m in (("u", "o"), ("v", "s"), ("w", "^")):
    ax[0].plot(t[f"r2_{c}"], t["z_m"], m + "-", label=c)
    ax[1].plot(t[f"rmse_{c}"], t["z_m"], m + "-", label=c)
ax[0].axvline(0.95, color="red", ls="--", label="seuil 0,95"); ax[1].axvline(0.5, color="red", ls="--", label="seuil 0,5 m/s")
ax[0].set_xlabel("R2"); ax[1].set_xlabel("RMSE (m/s)")
for a in ax: a.set_ylabel("Altitude ASL (m)"); a.legend()
ax[0].set_title("R2 par niveau (test 2022)"); ax[1].set_title("RMSE par niveau (test 2022)")
fig.suptitle("Grand Tunis v4 — test 2022, 180 instants (w par continuite, modele servi)")
fig.tight_layout(); fig.savefig(FIG / "03_gt_v4_test_niveaux.png", dpi=150); plt.show()
'''

C4 = r'''# 4. Prevision v5 : affinage de 12 epoques sur la prevision GFS a +24 h
v5 = historique("grand_tunis_v5_gfs")
courbes_entrainement(v5, "Prevision v5 — entrainement et validation (history.json)", "04_v5_entrainement.png")
print(v5.round(4).to_string(index=False))
'''

C5 = r'''# 5. Prevision v5 : erreur aux stations reelles en 2022, ATOM Wind contre GFS brut et persistance
import numpy as np
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
for a, e in zip(ax, (24, 48)):
    f = OS7 / f"v5_f{e}" / "prevision_bilan.json"
    v = json.loads(f.read_text())["verdicts"]; print("lu :", f)
    noms = [k for k in v if k != "toutes"] + ["toutes"]
    x = np.arange(len(noms)); l = 0.27
    for k, (s, lib) in enumerate((("prevision", "ATOM Wind v5"), ("gfs", "GFS brut"), ("persistance", "persistance"))):
        val = [v[n]["rmse_vect"][s] for n in noms]
        b = a.bar(x + (k - 1) * l, val, l, label=lib)
        a.bar_label(b, fmt="%.2f", fontsize=7)
    a.set_xticks(x); a.set_xticklabels(noms, rotation=15); a.set_title(f"Echeance +{e} h")
    a.set_ylabel("Erreur vectorielle a 10 m (m/s)")
ax[0].legend(); fig.suptitle("Prevision v5 — test 2022 aux stations reelles (prevision_bilan.json)")
fig.tight_layout(); fig.savefig(FIG / "05_v5_prevision_stations.png", dpi=150); plt.show()
print("figures ->", FIG, sorted(p.name for p in FIG.iterdir()))
'''

# ---------------------------------------------------------------- modele national C' s44 (modele servi)
N0 = r'''# Modele national C' s44 : fichiers deja ecrits sur le Drive (entrainement du 14/11, test 2022 lu une seule fois)
import numpy as np
from matplotlib.ticker import FuncFormatter
NAT = Path("/content/drive/MyDrive/pcfno3d_national/runs/national_1km_v4_Cp_s44")
EVAL = NAT / "eval_pcfno3d" / "best_wcont"
virgule = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))
print("dossier :", NAT, "| present :", NAT.exists())
'''

N1 = r'''# 6. National : entrainement et validation, epoque par epoque (history.json)
h = json.loads((NAT / "history.json").read_text())
h = [r for r in h if r.get("epoch", -1) >= 0 and isinstance(r.get("train"), dict)]
ep = [r["epoch"] + 1 for r in h]
print("history.json :", len(h), "epoques")
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
a = ax[0, 0]
a.semilogy(ep, [r["train"].get("data") for r in h], "o-", label="entraînement")
a.semilogy(ep, [r["val"].get("data") for r in h], "s-", label="validation")
a.set_title("Perte sur les données (normalisée)"); a.legend()
a = ax[0, 1]
if all(r["train"].get("mass") is not None for r in h):
    a.semilogy(ep, [r["train"]["mass"] for r in h], "o-", label="entraînement")
    a.semilogy(ep, [r["val"]["mass"] for r in h], "s-", label="validation")
    a.legend()
a.set_title("Perte de conservation de la masse (résidu au carré)")
a = ax[1, 0]
for c, m in zip("uvw", "os^"):
    a.plot(ep, [r.get(f"val_r2_{c}") for r in h], m + "-", label=f"$R^2$ {c}")
a.axhline(0.95, color="red", ls="--", label="seuil DSO1 (0,95)")
a.set_title("$R^2$ en validation"); a.legend(loc="lower right"); a.yaxis.set_major_formatter(virgule)
a = ax[1, 1]
a.plot(ep, [r.get("val_score") for r in h], "o-", color="tab:purple", label="score de sélection")
a.axhline(1.0, color="red", ls="--", label="1 = limite des critères DSO")
a.set_title("Score de sélection en validation (pire rapport mesure / seuil)"); a.legend()
a.yaxis.set_major_formatter(virgule)
for a in ax.flat: a.set_xlabel("Époque")
fig.suptitle("Modèle national C' s44 : entraînement et validation")
fig.tight_layout(); fig.savefig(FIG / "06_national_entrainement.png", dpi=150); plt.show()
print(pd.DataFrame({"epoque": ep, "train": [r["train"].get("data") for r in h], "validation": [r["val"].get("data") for r in h],
                    "score": [r.get("val_score") for r in h]}).round(4).to_string(index=False))
'''

N2 = r'''# 7. National : TEST 2022 par niveau (fichier deja ecrit le 14/11, le test n'est pas relu)
f = EVAL / "test_par_niveau_synoptique.csv"
t = pd.read_csv(f); print("lu :", f); print(t.round(3).to_string(index=False))
t = t[t["couverture"] >= 0.05]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for c, m in zip("uvw", "os^"):
    ax[0].plot(t[f"r2_{c}"], t["z_m"], m + "-", label=c)
    ax[1].plot(t[f"rmse_{c}"], t["z_m"], m + "-", label=c)
ax[0].axvline(0.95, color="red", ls="--", label="seuil DSO1 (0,95)")
ax[1].axvline(0.5, color="red", ls="--", label="seuil DSO1 (0,5 m/s)")
ax[0].set_xlabel("$R^2$"); ax[1].set_xlabel("RMSE (m/s)")
ax[0].set_title("$R^2$ par niveau (test 2022)"); ax[1].set_title("RMSE par niveau (test 2022)")
for a in ax:
    a.set_ylabel("Altitude au-dessus du niveau de la mer (m)"); a.legend(); a.xaxis.set_major_formatter(virgule)
fig.suptitle("Modèle national C' s44 : test 2022, instants synoptiques")
fig.tight_layout(); fig.savefig(FIG / "07_national_test_niveaux.png", dpi=150); plt.show()
'''

N3 = r'''# 8. National contre ERA5, niveau par niveau (verdict_national.json, test 2022)
f = EVAL / "verdict_national.json"
v = pd.DataFrame(json.loads(f.read_text())["par_niveau"]); print("lu :", f)
v = v[v["couverture"] >= 0.05]
print(v[["z_m", "rmse_vect", "rmse_vect_era5", "gain_era5"]].round(3).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(v["rmse_vect"], v["z_m"], "o-", label="modèle national")
ax[0].plot(v["rmse_vect_era5"], v["z_m"], "s--", label="ERA5 interpolée")
ax[0].set_xlabel("Erreur vectorielle (m/s)"); ax[0].set_ylabel("Altitude au-dessus du niveau de la mer (m)")
ax[0].set_title("Erreur par niveau"); ax[0].legend(); ax[0].xaxis.set_major_formatter(virgule)
ax[1].barh(v["z_m"].astype(int).astype(str), 100 * v["gain_era5"],
           color=["tab:green" if g >= 0 else "tab:red" for g in v["gain_era5"]])
ax[1].axvline(0, color="black", lw=0.8); ax[1].set_xlabel("Gain sur l'ERA5 (%) : positif = modèle meilleur")
ax[1].set_ylabel("Altitude au-dessus du niveau de la mer (m)"); ax[1].set_title("Gain du modèle sur l'ERA5")
fig.suptitle("Modèle national C' s44 contre ERA5, par niveau (test 2022)")
fig.tight_layout(); fig.savefig(FIG / "08_national_contre_era5_niveaux.png", dpi=150); plt.show()
'''

N4 = r'''# 9. National contre ERA5, par tranche au-dessus du sol (dso_hauteur_sol/tranches.csv, test 2022)
f = NAT / "dso_hauteur_sol" / "tranches.csv"
d = pd.read_csv(f); print("lu :", f); print(d.round(3).to_string(index=False))
col_p = next(c for c in d.columns if set(d[c].astype(str)) >= {"modele", "era5"})
col_t = next(c for c in d.columns if c != col_p and d[c].astype(str).str.contains("m").all())
d = d[~d[col_t].astype(str).str.contains("info")]
tr = list(dict.fromkeys(d[col_t])); x = np.arange(len(tr)); l = 0.38
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for k, (p, lib) in enumerate((("modele", "modèle national"), ("era5", "ERA5"))):
    q = d[d[col_p] == p].set_index(col_t).reindex(tr)
    b = ax[0].bar(x + (k - 0.5) * l, np.maximum(q["rmse_u"], q["rmse_v"]), l, label=lib)
    ax[0].bar_label(b, labels=[f"{y:.3f}".replace(".", ",") for y in b.datavalues], fontsize=7)
    b = ax[1].bar(x + (k - 0.5) * l, q["r2_w"], l, label=lib)
    ax[1].bar_label(b, labels=[f"{y:.2f}".replace(".", ",") for y in b.datavalues], fontsize=7)
for a in ax:
    a.set_xticks(x); a.set_xticklabels(tr, rotation=15); a.legend(); a.yaxis.set_major_formatter(virgule)
ax[0].set_ylabel("RMSE (m/s)"); ax[0].set_title("Vent horizontal : pire erreur sur u et v")
ax[1].axhline(0, color="black", lw=0.8); ax[1].set_ylabel("$R^2$ de w"); ax[1].set_title("Vent vertical : $R^2$")
fig.suptitle("Modèle national C' s44 contre ERA5, par tranche au-dessus du sol (test 2022)")
fig.tight_layout(); fig.savefig(FIG / "09_national_contre_era5_tranches.png", dpi=150); plt.show()
'''

N5 = r'''# 10. National contre ERA5 aux aeroports reels (validation_reelle/test/stations_bilan.csv ; aeroport juge retire de l'entree)
f = NAT / "validation_reelle" / "test" / "stations_bilan.csv"
s = pd.read_csv(f); print("lu :", f)
m = s.pivot(index="station", columns="source", values="rmse_vect")      # une ligne par aeroport, une colonne par source
print(m.round(3).to_string()); print("moyenne :", m.mean().round(3).to_dict())
src = [(c, lib) for c, lib in (("modele", "modèle national"), ("era5", "ERA5"), ("generateur", "vérité générée")) if c in m]
x = np.arange(len(m)); l = 0.8 / len(src)
fig, ax = plt.subplots(figsize=(13, 4.6))
for k, (c, lib) in enumerate(src):
    b = ax.bar(x + (k - (len(src) - 1) / 2) * l, m[c], l, label=lib)
    ax.bar_label(b, labels=[f"{y:.2f}".replace(".", ",") for y in b.datavalues], fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(m.index, rotation=30, ha="right"); ax.legend()
ax.set_ylabel("Erreur vectorielle à 10 m (m/s)"); ax.yaxis.set_major_formatter(virgule)
ax.set_title("Modèle national C' s44 contre ERA5 aux aéroports réels (test 2022)")
fig.tight_layout(); fig.savefig(FIG / "10_national_aeroports.png", dpi=150); plt.show()
print("figures ->", FIG, sorted(p.name for p in FIG.iterdir()))
'''

C6 = r'''# fin : la session CPU est liberee (les courbes restent affichees dans le carnet et copiees sur le Drive)
from google.colab import runtime
runtime.unassign()
'''

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": [md("# Courbes de résultats — Grand Tunis v4, prévision v5 et modèle national\n\nColab **CPU**. Le carnet "
                   "ne charge aucun modèle et ne relit pas le test : il trace les résultats déjà écrits sur le Drive "
                   "(`history.json`, `test_par_niveau_synoptique.csv`, `prevision_bilan.json`, `verdict_national.json`, "
                   "`tranches.csv`, `stations_bilan.csv`). Les figures sont aussi copiées dans `MyDrive/pcfno3d_courbes/`. "
                   "La session est libérée à la fin."),
                code(C0), md("## 1. Grand Tunis v4 : entraînement et validation"), code(C1),
                md("## 2. Grand Tunis : toute la chaîne d'entraînement"), code(C2),
                md("## 3. Grand Tunis v4 : test 2022 par altitude"), code(C3),
                md("## 4. Prévision v5 : entraînement et validation"), code(C4),
                md("## 5. Prévision v5 : stations réelles, +24 h et +48 h"), code(C5),
                md("## Modèle national C' s44 (modèle servi)"), code(N0),
                md("## 6. National : entraînement et validation"), code(N1),
                md("## 7. National : test 2022 par niveau"), code(N2),
                md("## 8. National contre ERA5 : test 2022 par niveau"), code(N3),
                md("## 9. National contre ERA5 : test 2022 par tranche au-dessus du sol"), code(N4),
                md("## 10. National contre ERA5 : aéroports réels (test 2022)"), code(N5),
                code(C6)]}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(OUT)
