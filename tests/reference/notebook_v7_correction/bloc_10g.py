# BLOC 10g — Vent geostrophique estime corrige (humidite, stabilite, densite) : test sans reentrainement
# Le generateur (generate_aneo_3d.py) fixe la pression au niveau de la mer par un plan exact
#   p_sl = p0 + 1.2 f (vg x - ug y)        (pression des capteurs sans bruit)
# et calcule la pression des capteurs avec une temperature VIRTUELLE moyenne de colonne (humidite)
# qui inclut le terme de stabilite. L'estimateur du BLOC 6 ignore l'humidite et la stabilite (erreur
# de reduction liee au relief : dizaines de Pa, pour un signal de 25 a 50 Pa) et divise par la
# densite des capteurs au lieu de celle du niveau de la mer.
# Estimateur v2 (physique standard, valable aussi sur donnees reelles) :
#   - temperature virtuelle exacte au capteur : Tv = p / (Rd rho)
#   - reduction au niveau de la mer avec Tv + 0.5 * gradient adiabatique * z
#   - plan ajuste AVEC l'altitude comme regresseur (absorbe le terme de stabilite ; son coefficient
#     sert d'indicateur de stabilite), puis second ajustement sans les points aberrants
#   - densite au niveau de la mer : 1.2 (convention du generateur) ou calculee, choisie sur la validation
# Protocole identique au BLOC 10f (train -> validation -> test). Necessite le BLOC 10f (meme session).
# N'ecrit que de nouveaux fichiers (EVAL_DIR/geostrophique_v2_test.csv, RUN_DIR/geostrophique_v2_v7.json).
for _v in ("X_tr", "X_va", "T_tr", "T_va", "REG", "USE", "I_PM", "predict_uv", "instant_features",
           "ridge_fit", "ridge_pred"):
    if _v not in globals():
        raise RuntimeError("Executer d'abord le BLOC 10f (dans la meme session).")
assert KEEP_EVAL == 0.5, "KEEP_EVAL doit valoir 0.5 (protocole DSO1)"
model.eval()
RD_GAS, G0, LAPSE_STD = 287.05, 9.80665, 0.0065
K_TOP = NZ - 1

def geostrophic_v2(S, sensor):
    """[ug, vg (rho_sl 1.2), ug, vg (rho_sl calculee), stabilite (Pa/m), residu tous points, residu retenus (Pa)]."""
    m = sensor
    if m.sum() < 100:
        return np.zeros(7)
    p = S[6][m].astype(np.float64)
    tv = p / (RD_GAS * S[4][m].astype(np.float64))          # temperature virtuelle exacte (gaz parfait)
    z = Z_SENSOR[m]
    p_sl = p * np.exp(G0 * z / (RD_GAS * (tv + 0.5 * LAPSE_STD * z)))
    A = np.stack([np.ones(m.sum()), XC[m], YC[m], z], axis=1)
    coef, *_ = np.linalg.lstsq(A, p_sl, rcond=None)
    res = p_sl - A @ coef
    rms_all = float(np.sqrt(np.mean(res ** 2)))
    mad = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
    keep = np.abs(res - np.median(res)) < 5.0 * mad
    if keep.sum() > 100:
        coef, *_ = np.linalg.lstsq(A[keep], p_sl[keep], rcond=None)
        res = p_sl[keep] - A[keep] @ coef
    rho_calc = float(np.mean(p_sl / (RD_GAS * (tv + LAPSE_STD * z))))
    out = []
    for rho_sl in (1.2, rho_calc):
        out += [-coef[2] / (rho_sl * F_CORIOLIS), coef[1] / (rho_sl * F_CORIOLIS)]
    return np.array(out + [coef[3], rms_all, float(np.sqrt(np.mean(res ** 2)))])

def plane_residual_v1(S, sensor):
    """Residu RMS (Pa) du plan de l'estimateur actuel (BLOC 6)."""
    m = sensor
    if m.sum() < 100:
        return 0.0
    p = S[6][m].astype(np.float64)
    t_k = S[3][m].astype(np.float64) + 273.15
    z = Z_SENSOR[m]
    p_sl = p * np.exp(9.80665 * z / (287.05 * (t_k + 0.5 * 0.0065 * z)))
    A = np.stack([np.ones(m.sum()), XC[m], YC[m]], axis=1)
    coef, *_ = np.linalg.lstsq(A, p_sl, rcond=None)
    return float(np.sqrt(np.mean((p_sl - A @ coef) ** 2)))

def geo_row(S, sensor):
    """10 valeurs : ug, vg actuels, residu actuel, puis geostrophic_v2 (7)."""
    ug1, vg1 = geostrophic_from_surface(S, sensor)
    return np.concatenate([[ug1, vg1, plane_residual_v1(S, sensor)], geostrophic_v2(S, sensor)])

def surf_iter(stems, seed):
    """Memes capteurs visibles que ANEO3D(train=False, seed) : surface seule, sans charger la cible."""
    for i, stem in enumerate(stems):
        rng = np.random.default_rng(seed + i)
        S = np.load(CACHE_DIR / f"{stem}_S.npy")
        yield S, (S[7] > 0.5) & (rng.random(S[7].shape) < KEEP_EVAL)

# ---------------------------------------------------------------- 1. Estimateur : qualite
t0 = time.time()
GEO = {"train": np.array([geo_row(S, m) for S, m in surf_iter(tr_s, 4000)]),
       "val": np.array([geo_row(S, m) for S, m in surf_iter(va_s[vsel], 1000)])}
assert len(GEO["train"]) == len(X_tr) and len(GEO["val"]) == len(X_va)
print(f"Vent geostrophique : train {len(GEO['train'])} | validation {len(GEO['val'])} instants ({time.time() - t0:.0f} s)")
Gt = GEO["train"]
print("=" * 90)
print("1) Plan de pression au niveau de la mer : residu RMS (mediane sur les instants d'entrainement)")
print(f"   estimateur actuel (BLOC 6)            : {np.median(Gt[:, 2]):8.2f} Pa")
print(f"   estimateur v2 (tous les points)        : {np.median(Gt[:, 8]):8.2f} Pa")
print(f"   estimateur v2 (sans points aberrants)  : {np.median(Gt[:, 9]):8.2f} Pa")
print("   (signal geostrophique : environ 26 Pa pour 10 m/s sur les 25 km du domaine en y)")
print(f"2) Ecart au vent moyen VRAI a {int(Z_LEVELS[K_TOP])} m (train ; indicatif : en altitude le vent vaut")
print("   le geostrophique + le vent thermique au-dessus de la couche limite)")
ut, vt = T_tr[:, K_TOP, 0], T_tr[:, K_TOP, 1]
for name, iu in (("actuel (BLOC 6)", 0), ("v2, rho_sl = 1.2", 3), ("v2, rho_sl calculee", 5)):
    eu = float(np.sqrt(np.mean((Gt[:, iu] - ut) ** 2)))
    ev = float(np.sqrt(np.mean((Gt[:, iu + 1] - vt) ** 2)))
    print(f"   {name:22s} : RMSE u {eu:5.2f}  v {ev:5.2f} m/s | correlation u "
          f"{np.corrcoef(Gt[:, iu], ut)[0, 1]:.3f}  v {np.corrcoef(Gt[:, iu + 1], vt)[0, 1]:.3f}")

# ---------------------------------------------------------------- 2. Regression : train -> validation
def x2(X, Gm, k, variant):
    """Caracteristiques du BLOC 10f + vent geostrophique v2, stabilite, vent thermique x stabilite."""
    i = 3 if variant == "1.2" else 5
    d = Gm[:, 7] * 1e3                                       # Pa/km
    return np.c_[X[:, k], Gm[:, i] / 10.0, Gm[:, i + 1] / 10.0, d, X[:, k, 6] * d, X[:, k, 7] * d]

FITS = {}
for variant in ("1.2", "calc"):
    for k in EVAL_LEVELS:
        for ci, c in enumerate("uv"):
            p = ridge_fit(x2(X_tr, GEO["train"], k, variant), T_tr[:, k, ci])
            e = float(np.sqrt(np.mean((ridge_pred(p, x2(X_va, GEO["val"], k, variant)) - T_va[:, k, ci]) ** 2)))
            FITS[(variant, k, c)] = (p, e)
HIGH = [k for k in EVAL_LEVELS if Z_LEVELS[k] >= 650]
score = {vr: float(np.mean([FITS[(vr, k, c)][1] for k in HIGH for c in "uv"])) for vr in ("1.2", "calc")}
VARIANT = min(score, key=score.get)
print("=" * 90)
print(f"3) Validation, erreur moyenne 650-1000 m : rho_sl 1.2 -> {score['1.2']:.3f} | calculee -> "
      f"{score['calc']:.3f} m/s  => variante retenue : {VARIANT}")
REG2, USE2 = {}, {}
print(" Niveau | comp | erreur du vent moyen (VALIDATION) : modele | BLOC 10f |  v2   | corrige ?")
for k in EVAL_LEVELS:
    for ci, c in enumerate("uv"):
        e_mod = float(np.sqrt(np.mean((X_va[:, k, I_PM + ci] - T_va[:, k, ci]) ** 2)))
        e_10f = float(np.sqrt(np.mean((ridge_pred(REG[(k, c)], X_va[:, k]) - T_va[:, k, ci]) ** 2)))
        p, e_v2 = FITS[(VARIANT, k, c)]
        REG2[(k, c)], USE2[(k, c)] = p, e_v2 < e_mod
        print(f" {int(Z_LEVELS[k]):5d} m |  {c}   |                                    {e_mod:.3f} |  {e_10f:.3f}"
              f"   | {e_v2:.3f} | {'oui' if USE2[(k, c)] else 'non'}")

# ---------------------------------------------------------------- 3. Test
NAMES_G = ("modele", "BLOC 10f", "v2")

def evaluate_v2(ds):
    acc_g = {n: {c: np.zeros((NZ, 4)) for c in "uv"} for n in NAMES_G}
    mass_g, rough_g = {n: [] for n in NAMES_G}, {n: [] for n in NAMES_G}
    for s_raw, sensor, tfeat, y, _ in DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS):
        u, v, w = predict_uv(s_raw, sensor, tfeat)
        X = instant_features(s_raw, sensor, tfeat, u, v)[None]
        Gm = geo_row(s_raw[0].numpy(), sensor[0].numpy() > 0.5)[None]
        y = y.to(device)
        preds = {"modele": (u, v), "BLOC 10f": (u.clone(), v.clone()), "v2": (u.clone(), v.clone())}
        for k in EVAL_LEVELS:
            mk = VALID_T[k]
            for ci, c in enumerate("uv"):
                if USE[(k, c)]:
                    f = preds["BLOC 10f"][ci]
                    f[0, k] += (float(ridge_pred(REG[(k, c)], X[:, k])[0]) - f[0, k][mk].mean()) * mk
                if USE2[(k, c)]:
                    f = preds["v2"][ci]
                    f[0, k] += (float(ridge_pred(REG2[(k, c)], x2(X, Gm, k, VARIANT))[0])
                                - f[0, k][mk].mean()) * mk
        for name in NAMES_G:
            pu, pv = preds[name]
            for ci, (c, p) in enumerate(zip("uv", (pu, pv))):
                for k in EVAL_LEVELS:
                    mk = VALID_T[k]
                    pk, tk = p[0, k][mk].double(), y[0, ci, k][mk].double()
                    acc_g[name][c][k] += np.array([tk.numel(), ((pk - tk) ** 2).sum().item(),
                                                   tk.sum().item(), (tk ** 2).sum().item()])
            r = mass_residual(y[:, 3], pu, pv, w)
            mass_g[name].append(torch.sqrt((r[:, MASS_MASK_T] ** 2).mean()).item())
            rough_g[name].append(((roughness(pu) + roughness(pv))
                                  / (roughness(y[:, 0]) + roughness(y[:, 1]))).item())
    return acc_g, mass_g, rough_g

def rmse_g(acc_g, name, c, k):
    return float(np.sqrt(acc_g[name][c][k][1] / acc_g[name][c][k][0]))

acc_g, mass_g, rough_g = evaluate_v2(ANEO3D(te_s[syn_idx], te_h[syn_idx], te_d[syn_idx], train=False, seed=2000))
rows = []
for name in NAMES_G:
    for k in EVAL_LEVELS:
        rec = {"variante": name, "z_m": int(Z_LEVELS[k])}
        for c in "uv":
            n, sse, st, st2 = acc_g[name][c][k]
            rec[f"rmse_{c}"] = float(np.sqrt(sse / n))
            rec[f"r2_{c}"] = float(1 - sse / (st2 - st ** 2 / n))
        rows.append(rec)
TABG = pd.DataFrame(rows)
TABG.to_csv(EVAL_DIR / "geostrophique_v2_test.csv", index=False)
print("=" * 90)
print(f"4) TEST SYNOPTIQUE : {len(syn_idx)} instants | capteurs visibles : {KEEP_EVAL:.0%}")
print(" Niveau | modele : RMSE u  RMSE v | BLOC 10f : RMSE u  RMSE v |   v2 : RMSE u  RMSE v")
for k in EVAL_LEVELS:
    r_ = [rmse_g(acc_g, n, c, k) for n in NAMES_G for c in "uv"]
    flag = "  <- > 0.5" if max(r_[4], r_[5]) > DSO1_RMSE else ""
    print(f" {int(Z_LEVELS[k]):5d} m |       {r_[0]:.3f}   {r_[1]:.3f}  |         {r_[2]:.3f}   {r_[3]:.3f}  |"
          f"       {r_[4]:.3f}   {r_[5]:.3f}{flag}")
print("(controle : les colonnes 'modele' doivent egaler le tableau du BLOC 10)")

# ---------------------------------------------------------------- 4. DSO5 et bilan
sub = syn_idx[np.unique(np.linspace(0, len(syn_idx) - 1, min(40, len(syn_idx))).astype(int))]
KEEP_EVAL_SAVE = KEEP_EVAL
try:
    KEEP_EVAL = 0.25
    acc5_g, _, _ = evaluate_v2(ANEO3D(te_s[sub], te_h[sub], te_d[sub], train=False, seed=3000))
finally:
    KEEP_EVAL = KEEP_EVAL_SAVE
r5_g = {n: max(rmse_g(acc5_g, n, c, k) for c in "uv" for k in EVAL_LEVELS) for n in NAMES_G}

def ok(b):
    return "CONFORME" if b else "NON CONFORME"

cor = TABG[TABG["variante"] == "v2"]
cov = np.array([COVER[k] for k in EVAL_LEVELS])
d1 = bool((cor[["rmse_u", "rmse_v"]].max(axis=1) <= DSO1_RMSE).all()) and bool((cor[["r2_u", "r2_v"]].min(axis=1) >= DSO1_R2).all())
g_ = {c: float(np.sqrt((cor[f"rmse_{c}"].values ** 2 * cov).sum() / cov.sum())) for c in "uv"}
m_c, r_c = float(np.mean(mass_g["v2"])), float(np.mean(rough_g["v2"]))
print("-" * 90)
print(f"Variante v2 (rho_sl {VARIANT}) :")
print(f"DSO1  RMSE u, v <= {DSO1_RMSE} et R2 >= {DSO1_R2} a chaque niveau -> {ok(d1)}")
print(f"      RMSE GLOBAL u = {g_['u']:.3f}  v = {g_['v']:.3f} m/s (seuil {DSO1_RMSE}) -> {ok(max(g_.values()) <= DSO1_RMSE)}")
print(f"DSO2  RMS residu moyen = {m_c:.2e} (modele seul {np.mean(mass_g['modele']):.2e}) -> {ok(m_c <= DSO2_RMS)}")
print(f"DSO3  rugosite = {r_c:.3f} (modele seul {np.mean(rough_g['modele']):.3f}) -> {ok(0.9 <= r_c <= 1.1)}")
print(f"DSO5  RMSE max a 25 % de capteurs = {r5_g['v2']:.3f} (modele seul {r5_g['modele']:.3f}) -> {ok(r5_g['v2'] <= DSO5_RMSE)}")
with open(RUN_DIR / "geostrophique_v2_v7.json", "w", encoding="utf-8") as f:
    json.dump({"variante_rho_sl": VARIANT, "ridge_alpha": RIDGE_ALPHA,
               "features": FEATURES + ["ug_v2/10", "vg_v2/10", "stabilite_Pa_km", "su*stab", "sv*stab"],
               "niveaux": {f"{int(Z_LEVELS[k])}_{c}": {"utilise": bool(USE2[(k, c)]),
                                                        "mu": REG2[(k, c)]["mu"].tolist(),
                                                        "sd": REG2[(k, c)]["sd"].tolist(),
                                                        "coef": REG2[(k, c)]["coef"].tolist()}
                           for (k, c) in REG2}}, f, indent=2)
print("Parametres enregistres :", RUN_DIR / "geostrophique_v2_v7.json")
print("BLOC 10g OK")
