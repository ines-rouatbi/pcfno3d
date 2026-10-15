# BLOC 10h — Physique de couche limite (rotation, stabilite) : test sans reentrainement
# Le BLOC 10g corrige l'estimation du vent geostrophique ; l'erreur restante a 800-1000 m vient de la
# couche limite : rotation du vent (Ekman) jusqu'au sommet de la couche limite, et vent thermique
# au-dessus, qui depend de la hauteur de couche limite (donc de la stabilite).
# Caracteristiques generiques ajoutees (aucune formule du generateur n'est recopiee) :
#   - angle de rotation estime : direction du vent moyen des capteurs - direction du geostrophique
#   - vent geostrophique tourne de 1/4, 1/2, 3/4 de cet angle (base pour une rotation selon l'altitude)
#   - vent thermique x stabilite, separe selon le signe de l'indicateur de stabilite
#   - vent geostrophique x stabilite (profil de vitesse dans la couche limite)
# Regularisation choisie sur la validation. Protocole : train -> validation -> test, une seule fois.
# Necessite les BLOCS 10f et 10g (meme session). N'ecrit que de nouveaux fichiers
# (EVAL_DIR/couche_limite_test.csv, RUN_DIR/couche_limite_v7.json).
for _v in ("GEO", "FITS", "REG2", "USE2", "x2", "geo_row", "VARIANT", "X_tr", "X_va", "T_tr", "T_va",
           "predict_uv", "instant_features", "ridge_pred"):
    if _v not in globals():
        raise RuntimeError("Executer d'abord les BLOCS 10f puis 10g (dans la meme session).")
assert KEEP_EVAL == 0.5, "KEEP_EVAL doit valoir 0.5 (protocole DSO1)"
model.eval()
K_TOP = NZ - 1
I_GEO = 3 if VARIANT == "1.2" else 5                         # colonnes ug, vg v2 dans geo_row

def rotate(u, v, a):
    return u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a)

def x3(X, Gm, k):
    """Caracteristiques du BLOC 10g + rotation et stabilite (couche limite)."""
    ug, vg = Gm[:, I_GEO], Gm[:, I_GEO + 1]                  # geostrophique v2
    us, vs = X[:, K_TOP, 8], X[:, K_TOP, 9]                  # vent moyen des capteurs visibles
    a = np.arctan2(vs, us) - np.arctan2(vg, ug)
    a = (a + np.pi) % (2 * np.pi) - np.pi                    # angle de rotation estime (rad)
    d = Gm[:, 7] * 1e3                                       # indicateur de stabilite (Pa/km)
    dp, dm = np.maximum(d, 0.0), np.minimum(d, 0.0)
    su, sv = X[:, k, 6], X[:, k, 7]                          # vent thermique (x 200)
    cols = [x2(X, Gm, k, VARIANT)]
    for f in (0.25, 0.5, 0.75):
        ru, rv = rotate(ug, vg, f * a)
        cols.append(np.c_[ru / 10.0, rv / 10.0])
    cols.append(np.c_[su * dp, su * dm, sv * dp, sv * dm, ug / 10.0 * d, vg / 10.0 * d])
    return np.concatenate(cols, axis=1)

def ridge_fit_a(X, t, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.c_[np.ones(len(X)), (X - mu) / sd]
    R = alpha * np.eye(A.shape[1])
    R[0, 0] = 0.0
    return {"mu": mu, "sd": sd, "coef": np.linalg.solve(A.T @ A + R, A.T @ t)}

# ---------------------------------------------------------------- 1. Train -> validation
FITS3 = {}
for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
    for k in EVAL_LEVELS:
        for ci, c in enumerate("uv"):
            p = ridge_fit_a(x3(X_tr, GEO["train"], k), T_tr[:, k, ci], alpha)
            e = float(np.sqrt(np.mean((ridge_pred(p, x3(X_va, GEO["val"], k)) - T_va[:, k, ci]) ** 2)))
            FITS3[(alpha, k, c)] = (p, e)
HIGH = [k for k in EVAL_LEVELS if Z_LEVELS[k] >= 650]
score3 = {al: float(np.mean([FITS3[(al, k, c)][1] for k in HIGH for c in "uv"])) for al in (0.01, 0.1, 1.0, 10.0, 100.0)}
ALPHA3 = min(score3, key=score3.get)
print("Validation, erreur moyenne 650-1000 m : " + " | ".join(f"alpha {al:g} -> {s:.3f}" for al, s in score3.items())
      + f"  => alpha retenu : {ALPHA3:g}")
REG3, USE3 = {}, {}
print("=" * 90)
print(" Niveau | comp | erreur du vent moyen (VALIDATION) : modele |  v2   | couche limite | corrige ?")
for k in EVAL_LEVELS:
    for ci, c in enumerate("uv"):
        e_mod = float(np.sqrt(np.mean((X_va[:, k, I_PM + ci] - T_va[:, k, ci]) ** 2)))
        e_v2 = FITS[(VARIANT, k, c)][1]
        p, e_v3 = FITS3[(ALPHA3, k, c)]
        REG3[(k, c)], USE3[(k, c)] = p, e_v3 < e_mod
        print(f" {int(Z_LEVELS[k]):5d} m |  {c}   |                                    {e_mod:.3f} | {e_v2:.3f} |"
              f"     {e_v3:.3f}     | {'oui' if USE3[(k, c)] else 'non'}")

# ---------------------------------------------------------------- 2. Test
NAMES_H = ("modele", "v2", "couche limite")

def evaluate_v3(ds):
    acc_h = {n: {c: np.zeros((NZ, 4)) for c in "uv"} for n in NAMES_H}
    mass_h, rough_h = {n: [] for n in NAMES_H}, {n: [] for n in NAMES_H}
    for s_raw, sensor, tfeat, y, _ in DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS):
        u, v, w = predict_uv(s_raw, sensor, tfeat)
        X = instant_features(s_raw, sensor, tfeat, u, v)[None]
        Gm = geo_row(s_raw[0].numpy(), sensor[0].numpy() > 0.5)[None]
        y = y.to(device)
        preds = {"modele": (u, v), "v2": (u.clone(), v.clone()), "couche limite": (u.clone(), v.clone())}
        for k in EVAL_LEVELS:
            mk = VALID_T[k]
            for ci, c in enumerate("uv"):
                if USE2[(k, c)]:
                    f = preds["v2"][ci]
                    f[0, k] += (float(ridge_pred(REG2[(k, c)], x2(X, Gm, k, VARIANT))[0]) - f[0, k][mk].mean()) * mk
                if USE3[(k, c)]:
                    f = preds["couche limite"][ci]
                    f[0, k] += (float(ridge_pred(REG3[(k, c)], x3(X, Gm, k))[0]) - f[0, k][mk].mean()) * mk
        for name in NAMES_H:
            pu, pv = preds[name]
            for ci, (c, p) in enumerate(zip("uv", (pu, pv))):
                for k in EVAL_LEVELS:
                    mk = VALID_T[k]
                    pk, tk = p[0, k][mk].double(), y[0, ci, k][mk].double()
                    acc_h[name][c][k] += np.array([tk.numel(), ((pk - tk) ** 2).sum().item(),
                                                   tk.sum().item(), (tk ** 2).sum().item()])
            r = mass_residual(y[:, 3], pu, pv, w)
            mass_h[name].append(torch.sqrt((r[:, MASS_MASK_T] ** 2).mean()).item())
            rough_h[name].append(((roughness(pu) + roughness(pv))
                                  / (roughness(y[:, 0]) + roughness(y[:, 1]))).item())
    return acc_h, mass_h, rough_h

def rmse_h(acc_h, name, c, k):
    return float(np.sqrt(acc_h[name][c][k][1] / acc_h[name][c][k][0]))

acc_h, mass_h, rough_h = evaluate_v3(ANEO3D(te_s[syn_idx], te_h[syn_idx], te_d[syn_idx], train=False, seed=2000))
rows = []
for name in NAMES_H:
    for k in EVAL_LEVELS:
        rec = {"variante": name, "z_m": int(Z_LEVELS[k])}
        for c in "uv":
            n, sse, st, st2 = acc_h[name][c][k]
            rec[f"rmse_{c}"] = float(np.sqrt(sse / n))
            rec[f"r2_{c}"] = float(1 - sse / (st2 - st ** 2 / n))
        rows.append(rec)
TABH = pd.DataFrame(rows)
TABH.to_csv(EVAL_DIR / "couche_limite_test.csv", index=False)
print("=" * 90)
print(f"TEST SYNOPTIQUE : {len(syn_idx)} instants | capteurs visibles : {KEEP_EVAL:.0%}")
print(" Niveau | modele : RMSE u  RMSE v |   v2 : RMSE u  RMSE v | couche limite : RMSE u  RMSE v")
for k in EVAL_LEVELS:
    r_ = [rmse_h(acc_h, n, c, k) for n in NAMES_H for c in "uv"]
    flag = "  <- > 0.5" if max(r_[4], r_[5]) > DSO1_RMSE else ""
    print(f" {int(Z_LEVELS[k]):5d} m |       {r_[0]:.3f}   {r_[1]:.3f}  |     {r_[2]:.3f}   {r_[3]:.3f}  |"
          f"              {r_[4]:.3f}   {r_[5]:.3f}{flag}")
print("(controle : les colonnes 'modele' doivent egaler le tableau du BLOC 10)")

# ---------------------------------------------------------------- 3. DSO5 et bilan
sub = syn_idx[np.unique(np.linspace(0, len(syn_idx) - 1, min(40, len(syn_idx))).astype(int))]
KEEP_EVAL_SAVE = KEEP_EVAL
try:
    KEEP_EVAL = 0.25
    acc5_h, _, _ = evaluate_v3(ANEO3D(te_s[sub], te_h[sub], te_d[sub], train=False, seed=3000))
finally:
    KEEP_EVAL = KEEP_EVAL_SAVE
r5_h = {n: max(rmse_h(acc5_h, n, c, k) for c in "uv" for k in EVAL_LEVELS) for n in NAMES_H}

def ok(b):
    return "CONFORME" if b else "NON CONFORME"

cor = TABH[TABH["variante"] == "couche limite"]
cov = np.array([COVER[k] for k in EVAL_LEVELS])
d1 = bool((cor[["rmse_u", "rmse_v"]].max(axis=1) <= DSO1_RMSE).all()) and bool((cor[["r2_u", "r2_v"]].min(axis=1) >= DSO1_R2).all())
g_ = {c: float(np.sqrt((cor[f"rmse_{c}"].values ** 2 * cov).sum() / cov.sum())) for c in "uv"}
m_c, r_c = float(np.mean(mass_h["couche limite"])), float(np.mean(rough_h["couche limite"]))
print("-" * 90)
print(f"Variante couche limite (alpha {ALPHA3:g}) :")
print(f"DSO1  RMSE u, v <= {DSO1_RMSE} et R2 >= {DSO1_R2} a chaque niveau -> {ok(d1)}")
print(f"      RMSE GLOBAL u = {g_['u']:.3f}  v = {g_['v']:.3f} m/s (seuil {DSO1_RMSE}) -> {ok(max(g_.values()) <= DSO1_RMSE)}")
print(f"DSO2  RMS residu moyen = {m_c:.2e} (modele seul {np.mean(mass_h['modele']):.2e}) -> {ok(m_c <= DSO2_RMS)}")
print(f"DSO3  rugosite = {r_c:.3f} (modele seul {np.mean(rough_h['modele']):.3f}) -> {ok(0.9 <= r_c <= 1.1)}")
print(f"DSO5  RMSE max a 25 % de capteurs = {r5_h['couche limite']:.3f} (v2 {r5_h['v2']:.3f}, modele seul "
      f"{r5_h['modele']:.3f}) -> {ok(r5_h['couche limite'] <= DSO5_RMSE)}")
with open(RUN_DIR / "couche_limite_v7.json", "w", encoding="utf-8") as f:
    json.dump({"variante_rho_sl": VARIANT, "ridge_alpha": ALPHA3,
               "features": FEATURES + ["ug_v2/10", "vg_v2/10", "stabilite_Pa_km", "su*stab", "sv*stab",
                                       "rot25_u", "rot25_v", "rot50_u", "rot50_v", "rot75_u", "rot75_v",
                                       "su*stab+", "su*stab-", "sv*stab+", "sv*stab-", "ug*stab", "vg*stab"],
               "niveaux": {f"{int(Z_LEVELS[k])}_{c}": {"utilise": bool(USE3[(k, c)]),
                                                        "mu": REG3[(k, c)]["mu"].tolist(),
                                                        "sd": REG3[(k, c)]["sd"].tolist(),
                                                        "coef": REG3[(k, c)]["coef"].tolist()}
                           for (k, c) in REG3}}, f, indent=2)
print("Parametres enregistres :", RUN_DIR / "couche_limite_v7.json")
print("BLOC 10h OK")
