# COURBES DE RESULTATS — modele national v4 contre ERA5 (traces depuis les fichiers deja ecrits ; aucun modele charge, test non relu)
from google.colab import drive
import os
if not os.path.isdir("/content/drive/MyDrive"):
    drive.mount("/content/drive")
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN = Path("/content/drive/MyDrive/pcfno3d_national/runs/national_1km_v4_Cp_s44")
FIG = Path("/content/drive/MyDrive/pcfno3d_courbes")          # dossier neuf : rien n'est ecrase ailleurs
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

# 6. Erreur par altitude : modele national contre ERA5 (verdict_national.json, test 2022, 180 instants)
f = RUN / "eval_pcfno3d" / "best_wcont" / "verdict_national.json"
v = json.loads(f.read_text()); print("lu :", f)
t = pd.DataFrame(v["par_niveau"]); t = t[t["couverture"] >= 0.05]
print(t[["z_m", "rmse_vect", "rmse_vect_era5", "gain_era5", "r2_u", "r2_v", "r2_w"]].round(3).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(t["rmse_vect"], t["z_m"], "o-", label="modele national v4")
ax[0].plot(t["rmse_vect_era5"], t["z_m"], "s--", label="ERA5 interpole")
ax[0].set_xlabel("Erreur vectorielle (m/s)"); ax[0].set_ylabel("Altitude ASL (m)"); ax[0].legend()
ax[0].set_title("Erreur par altitude (test 2022)")
ax[1].barh(t["z_m"].astype(str), 100 * t["gain_era5"], color=["tab:green" if g >= 0 else "tab:red" for g in t["gain_era5"]])
ax[1].axvline(0, color="black", lw=0.8); ax[1].set_xlabel("Gain face a ERA5 (%) : positif = modele meilleur")
ax[1].set_ylabel("Altitude ASL (m)"); ax[1].set_title("Gain du modele face a ERA5, niveau par niveau")
fig.suptitle("Modele national v4 contre ERA5 — test 2022 (verdict_national.json)")
fig.tight_layout(); fig.savefig(FIG / "06_national_vs_era5_altitude.png", dpi=150); plt.show()

# 7. Par tranche au-dessus du sol, 0 a 1 km (dso_hauteur_sol/tranches.csv)
f = RUN / "dso_hauteur_sol" / "tranches.csv"
try:
    d = pd.read_csv(f); print("lu :", f); print(d.round(3).to_string(index=False))
    col_p = next(c for c in d.columns if set(d[c].astype(str)) >= {"modele", "era5"})
    col_t = next(c for c in d.columns if c != col_p and d[c].astype(str).str.contains("m").all())
    d = d[~d[col_t].astype(str).str.contains("info")]
    tr = list(dict.fromkeys(d[col_t]))
    x = np.arange(len(tr)); l = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for k, (p, lib) in enumerate((("modele", "modele national v4"), ("era5", "ERA5"))):
        q = d[d[col_p] == p].set_index(col_t).reindex(tr)
        err = np.maximum(q["rmse_u"], q["rmse_v"])
        b = ax[0].bar(x + (k - 0.5) * l, err, l, label=lib); ax[0].bar_label(b, fmt="%.3f", fontsize=7)
        b = ax[1].bar(x + (k - 0.5) * l, q["r2_w"], l, label=lib); ax[1].bar_label(b, fmt="%.2f", fontsize=7)
    for a in ax: a.set_xticks(x); a.set_xticklabels(tr, rotation=15); a.legend()
    ax[0].set_ylabel("Pire erreur u, v (m/s)"); ax[0].set_title("Erreur du vent horizontal par tranche")
    ax[1].axhline(0, color="black", lw=0.8); ax[1].set_ylabel("R2 du vent vertical w"); ax[1].set_title("Vent vertical : R2 par tranche")
    fig.suptitle("Modele national v4 contre ERA5 — hauteur au-dessus du sol, test 2022 (tranches.csv)")
    fig.tight_layout(); fig.savefig(FIG / "07_national_vs_era5_tranches.png", dpi=150); plt.show()
except Exception as e:
    print("tranches : figure non tracee :", repr(e))

# 8. Aux 12 aeroports reels (validation_reelle/test/stations_bilan.csv)
f = RUN / "validation_reelle" / "test" / "stations_bilan.csv"
try:
    s = pd.read_csv(f); print("lu :", f); print(s.round(3).to_string(index=False))
    col_s = next(c for c in s.columns if s[c].dtype == object)
    cm = next(c for c in s.columns if "modele" in c.lower() and "rmse" in c.lower())
    ce = next(c for c in s.columns if "era5" in c.lower() and "rmse" in c.lower())
    x = np.arange(len(s)); l = 0.38
    fig, ax = plt.subplots(figsize=(13, 4.4))
    b = ax.bar(x - l / 2, s[cm], l, label="modele national v4"); ax.bar_label(b, fmt="%.2f", fontsize=7)
    b = ax.bar(x + l / 2, s[ce], l, label="ERA5"); ax.bar_label(b, fmt="%.2f", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(s[col_s], rotation=30, ha="right"); ax.legend()
    ax.set_ylabel("Erreur vectorielle a 10 m (m/s)")
    ax.set_title("Modele national v4 contre ERA5 aux aeroports reels (aeroport juge retire de l'entree) — stations_bilan.csv")
    fig.tight_layout(); fig.savefig(FIG / "08_national_vs_era5_aeroports.png", dpi=150); plt.show()
except Exception as e:
    print("aeroports : figure non tracee :", repr(e))
print("figures ->", FIG, sorted(p.name for p in FIG.iterdir()))

# fin : la session est liberee (les courbes restent affichees ici et copiees dans MyDrive/pcfno3d_courbes)
from google.colab import runtime
runtime.unassign()
