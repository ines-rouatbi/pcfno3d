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
# 9. ANIMATIONS (pyplot anime) : affichees dans le carnet et enregistrees en GIF sur le Drive
from matplotlib import animation
from IPython.display import HTML, display
plt.rcParams["animation.embed_limit"] = 60

# 9a. entrainement du modele national v4, epoque par epoque (history.json)
h = json.loads((RUN / "history.json").read_text())
h = [r for r in h if r.get("epoch", -1) >= 0 and isinstance(r.get("train"), dict)]
ep = [r["epoch"] + 1 for r in h]
tr = [r["train"].get("data") for r in h]; va = [r["val"].get("data") for r in h]
r2 = {c: [r.get(f"val_r2_{c}") for r in h] for c in "uvw"}
print("history.json :", len(h), "epoques")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
l_tr, = ax[0].plot([], [], "o-", label="entrainement"); l_va, = ax[0].plot([], [], "s-", label="validation")
ax[0].set_xlim(0.5, max(ep) + 0.5); ax[0].set_yscale("log")
ax[0].set_ylim(min(tr + va) * 0.8, max(tr + va) * 1.2); ax[0].set_xlabel("Epoque"); ax[0].legend()
ax[0].set_title("Erreur sur les donnees (normalisee)")
l_r2 = {c: ax[1].plot([], [], m + "-", label="R2 " + c)[0] for c, m in zip("uvw", "os^")}
ax[1].axhline(0.95, color="red", ls="--", label="seuil 0,95"); ax[1].set_xlim(0.5, max(ep) + 0.5)
ax[1].set_ylim(min(min(x for x in r2["w"] if x is not None), 0.9) - 0.02, 1.005); ax[1].set_xlabel("Epoque")
ax[1].set_title("R2 sur la validation"); ax[1].legend(loc="lower right")
titre = fig.suptitle("")
def image_ep(i):
    l_tr.set_data(ep[:i + 1], tr[:i + 1]); l_va.set_data(ep[:i + 1], va[:i + 1])
    for c in "uvw": l_r2[c].set_data(ep[:i + 1], r2[c][:i + 1])
    titre.set_text(f"Modele national v4 — entrainement, epoque {ep[i]}/{max(ep)}")
    return [l_tr, l_va, *l_r2.values(), titre]
anim1 = animation.FuncAnimation(fig, image_ep, frames=len(ep), interval=350, blit=False)
anim1.save(FIG / "09_national_entrainement_anime.gif", writer=animation.PillowWriter(fps=3))
plt.close(fig); display(HTML(anim1.to_jshtml()))

# 9b. modele national contre ERA5, altitude par altitude (verdict_national.json)
z = t["z_m"].to_numpy(); em = t["rmse_vect"].to_numpy(); ee = t["rmse_vect_era5"].to_numpy()
fig, ax = plt.subplots(figsize=(7, 5.2))
lm, = ax.plot([], [], "o-", lw=2, label="modele national v4"); le, = ax.plot([], [], "s--", lw=2, label="ERA5 interpole")
ax.set_xlim(0, max(em.max(), ee.max()) * 1.15); ax.set_ylim(0, z.max() * 1.05)
ax.set_xlabel("Erreur vectorielle (m/s)"); ax.set_ylabel("Altitude ASL (m)"); ax.legend(loc="lower right")
txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=10)
ax.set_title("Modele national v4 contre ERA5 — test 2022")
def image_z(i):
    lm.set_data(em[:i + 1], z[:i + 1]); le.set_data(ee[:i + 1], z[:i + 1])
    g = 100 * (1 - em[i] / ee[i])
    txt.set_text(f"{z[i]:.0f} m : modele {em[i]:.3f} m/s, ERA5 {ee[i]:.3f} m/s ({g:+.0f} %)")
    return [lm, le, txt]
anim2 = animation.FuncAnimation(fig, image_z, frames=len(z), interval=600, blit=False)
anim2.save(FIG / "10_national_vs_era5_anime.gif", writer=animation.PillowWriter(fps=2))
plt.close(fig); display(HTML(anim2.to_jshtml()))
print("figures ->", FIG, sorted(p.name for p in FIG.iterdir()))

# fin : la session est liberee (les courbes restent affichees ici et copiees dans MyDrive/pcfno3d_courbes)
from google.colab import runtime
runtime.unassign()
