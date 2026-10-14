# BLOC 10f — Correction du vent moyen par niveau (sans reentrainement)
# Le BLOC 10d montre qu'au-dessus de 500 m l'erreur est surtout un decalage de tout le niveau.
# Une regression lineaire (ridge) predit le vent moyen vrai (u, v) de chaque niveau a partir de
# donnees de surface uniquement (vents geostrophique et thermique estimes, moyennes des capteurs
# visibles, moyenne du niveau predite par le modele), puis remplace la moyenne du niveau predite :
# la structure spatiale de la prediction est conservee.
# Protocole : ajustement sur l'ENTRAINEMENT, choix des niveaux corriges sur la VALIDATION,
# resultat sur le TEST (instants synoptiques, 50 % des capteurs), une seule fois.
# N'ecrit que de nouveaux fichiers (EVAL_DIR/correction_moyenne_test.csv, RUN_DIR/correction_moyenne_v7.json).
if "syn_idx" not in globals() or "amp_dtype" not in globals():
    raise RuntimeError("Executer d'abord le BLOC 9 puis le BLOC 10.")
assert KEEP_EVAL == 0.5, "KEEP_EVAL doit valoir 0.5 (protocole DSO1) : relancer le BLOC 1"
model.eval()
RIDGE_ALPHA = 1.0
FEATURES = ["sin_h", "cos_h", "sin_doy", "cos_doy", "ug/10", "vg/10", "su*200", "sv*200",
            "u_capteurs", "v_capteurs", "T_capteurs", "u_niveau_modele", "v_niveau_modele"]
I_PM = 11                                        # indice de u_niveau_modele (v : I_PM + 1)

def predict_uv(s_raw, sensor, tfeat):
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        pred = model(assemble_input(s_raw.to(device), sensor.to(device), tfeat.to(device)))
    return denorm(pred.float())

def instant_features(s_raw, sensor, tfeat, u, v):
    """(NZ, 13) : tfeat (8), moyennes u, v, T des capteurs visibles, moyennes u, v du niveau predites."""
    m = sensor[0] > 0.5
    surf = [float(s_raw[0, c][m].mean()) if m.any() else 0.0 for c in (0, 1, 3)]
    base = list(tfeat[0].numpy().astype(np.float64)) + surf
    out = np.zeros((NZ, len(FEATURES)))
    for k in EVAL_LEVELS:
        mk = VALID_T[k]
        out[k] = base + [u[0, k][mk].mean().item(), v[0, k][mk].mean().item()]
    return out

def collect(ds):
    """Caracteristiques (n, NZ, 13) et moyennes vraies (n, NZ, 2) de u, v par niveau."""
    X, T = [], []
    for s_raw, sensor, tfeat, y, _ in DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS):
        u, v, _w = predict_uv(s_raw, sensor, tfeat)
        X.append(instant_features(s_raw, sensor, tfeat, u, v))
        T.append(np.array([[float(y[0, c, k][VALID[k]].mean()) if VALID[k].any() else 0.0
                            for c in (0, 1)] for k in range(NZ)]))
    return np.stack(X), np.stack(T)

def ridge_fit(X, t):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.c_[np.ones(len(X)), (X - mu) / sd]
    R = RIDGE_ALPHA * np.eye(A.shape[1])
    R[0, 0] = 0.0                                # pas de penalite sur la constante
    return {"mu": mu, "sd": sd, "coef": np.linalg.solve(A.T @ A + R, A.T @ t)}

def ridge_pred(p, X):
    return np.c_[np.ones(len(X)), (X - p["mu"]) / p["sd"]] @ p["coef"]

# ---------------------------------------------------------------- 1. Ajustement (train) et choix (validation)
t0 = time.time()
X_tr, T_tr = collect(ANEO3D(tr_s, tr_h, tr_d, train=False, seed=4000))
X_va, T_va = collect(ANEO3D(va_s[vsel], va_h[vsel], va_d[vsel], train=False, seed=1000))
print(f"Caracteristiques : train {len(X_tr)} | validation {len(X_va)} instants ({time.time() - t0:.0f} s)")
REG, USE = {}, {}
print("=" * 90)
print(" Niveau | comp | erreur du vent moyen (VALIDATION) : modele -> regression | corrige ?")
for k in EVAL_LEVELS:
    for ci, c in enumerate("uv"):
        p = ridge_fit(X_tr[:, k], T_tr[:, k, ci])
        e_mod = float(np.sqrt(np.mean((X_va[:, k, I_PM + ci] - T_va[:, k, ci]) ** 2)))
        e_reg = float(np.sqrt(np.mean((ridge_pred(p, X_va[:, k]) - T_va[:, k, ci]) ** 2)))
        REG[(k, c)], USE[(k, c)] = p, e_reg < e_mod
        print(f" {int(Z_LEVELS[k]):5d} m |  {c}   |              {e_mod:.3f} -> {e_reg:.3f}"
              f"                | {'oui' if USE[(k, c)] else 'non'}")

# ---------------------------------------------------------------- 2. Evaluation sur le test
def evaluate_corrected(ds):
    """Metriques globales par niveau pour le modele seul et le modele corrige."""
    acc = {n: {c: np.zeros((NZ, 4)) for c in "uv"} for n in ("modele", "corrige")}
    mass, rough = {"modele": [], "corrige": []}, {"modele": [], "corrige": []}
    for s_raw, sensor, tfeat, y, _ in DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS):
        u, v, w = predict_uv(s_raw, sensor, tfeat)
        X = instant_features(s_raw, sensor, tfeat, u, v)
        y = y.to(device)
        uc, vc = u.clone(), v.clone()
        for k in EVAL_LEVELS:
            mk = VALID_T[k]
            for c, f in (("u", uc), ("v", vc)):
                if USE[(k, c)]:
                    target = float(ridge_pred(REG[(k, c)], X[k][None])[0])
                    f[0, k] += (target - f[0, k][mk].mean()) * mk     # decalage des noeuds valides
        for name, (pu, pv) in (("modele", (u, v)), ("corrige", (uc, vc))):
            for ci, (c, p) in enumerate(zip("uv", (pu, pv))):
                for k in EVAL_LEVELS:
                    mk = VALID_T[k]
                    pk, tk = p[0, k][mk].double(), y[0, ci, k][mk].double()
                    acc[name][c][k] += np.array([tk.numel(), ((pk - tk) ** 2).sum().item(),
                                                 tk.sum().item(), (tk ** 2).sum().item()])
            r = mass_residual(y[:, 3], pu, pv, w)
            mass[name].append(torch.sqrt((r[:, MASS_MASK_T] ** 2).mean()).item())
            rough[name].append(((roughness(pu) + roughness(pv))
                                / (roughness(y[:, 0]) + roughness(y[:, 1]))).item())
    return acc, mass, rough

def rmse_of(acc, name, c, k):
    return float(np.sqrt(acc[name][c][k][1] / acc[name][c][k][0]))

te_syn = ANEO3D(te_s[syn_idx], te_h[syn_idx], te_d[syn_idx], train=False, seed=2000)
acc, mass, rough = evaluate_corrected(te_syn)
rows = []
for name in ("modele", "corrige"):
    for k in EVAL_LEVELS:
        rec = {"variante": name, "z_m": int(Z_LEVELS[k])}
        for c in "uv":
            n, sse, st, st2 = acc[name][c][k]
            rec[f"rmse_{c}"] = float(np.sqrt(sse / n))
            rec[f"r2_{c}"] = float(1 - sse / (st2 - st ** 2 / n))
        rows.append(rec)
TABF = pd.DataFrame(rows)
TABF.to_csv(EVAL_DIR / "correction_moyenne_test.csv", index=False)
print("=" * 90)
print(f"TEST SYNOPTIQUE : {len(te_syn)} instants | capteurs visibles : {KEEP_EVAL:.0%}")
print(" Niveau | modele : RMSE u  RMSE v | corrige : RMSE u  RMSE v |  R2 u    R2 v  (corrige)")
for k in EVAL_LEVELS:
    a = TABF[(TABF["variante"] == "modele") & (TABF["z_m"] == int(Z_LEVELS[k]))].iloc[0]
    b = TABF[(TABF["variante"] == "corrige") & (TABF["z_m"] == int(Z_LEVELS[k]))].iloc[0]
    flag = "  <- > 0.5" if max(b["rmse_u"], b["rmse_v"]) > DSO1_RMSE else ""
    print(f" {int(Z_LEVELS[k]):5d} m |       {a['rmse_u']:.3f}   {a['rmse_v']:.3f}  |        {b['rmse_u']:.3f}"
          f"   {b['rmse_v']:.3f}  | {b['r2_u']:.4f}  {b['r2_v']:.4f}{flag}")
print("(controle : les colonnes 'modele' doivent egaler le tableau du BLOC 10)")

# ---------------------------------------------------------------- 3. DSO5 : 25 % des capteurs
sub = syn_idx[np.unique(np.linspace(0, len(syn_idx) - 1, min(40, len(syn_idx))).astype(int))]
KEEP_EVAL_SAVE = KEEP_EVAL
try:
    KEEP_EVAL = 0.25
    acc5, _, _ = evaluate_corrected(ANEO3D(te_s[sub], te_h[sub], te_d[sub], train=False, seed=3000))
finally:
    KEEP_EVAL = KEEP_EVAL_SAVE
r5 = {n: max(rmse_of(acc5, n, c, k) for c in "uv" for k in EVAL_LEVELS) for n in ("modele", "corrige")}

# ---------------------------------------------------------------- 4. Bilan
def ok(b):
    return "CONFORME" if b else "NON CONFORME"

cor = TABF[TABF["variante"] == "corrige"]
cov = np.array([COVER[k] for k in EVAL_LEVELS])
d1 = bool((cor[["rmse_u", "rmse_v"]].max(axis=1) <= DSO1_RMSE).all()) and \
     bool((cor[["r2_u", "r2_v"]].min(axis=1) >= DSO1_R2).all())
g = {c: float(np.sqrt((cor[f"rmse_{c}"].values ** 2 * cov).sum() / cov.sum())) for c in "uv"}
m_c, r_c = float(np.mean(mass["corrige"])), float(np.mean(rough["corrige"]))
print("-" * 90)
print(f"DSO1  RMSE u, v <= {DSO1_RMSE} et R2 >= {DSO1_R2} a chaque niveau -> {ok(d1)}")
print(f"      RMSE GLOBAL u = {g['u']:.3f}  v = {g['v']:.3f} m/s (seuil {DSO1_RMSE}) -> {ok(max(g.values()) <= DSO1_RMSE)}")
print(f"DSO2  RMS residu moyen = {m_c:.2e} (modele seul {np.mean(mass['modele']):.2e}) -> {ok(m_c <= DSO2_RMS)}")
print(f"DSO3  rugosite = {r_c:.3f} (modele seul {np.mean(rough['modele']):.3f}) -> {ok(0.9 <= r_c <= 1.1)}")
print(f"DSO5  RMSE max a 25 % de capteurs = {r5['corrige']:.3f} (modele seul {r5['modele']:.3f}) "
      f"-> {ok(r5['corrige'] <= DSO5_RMSE)}")
with open(RUN_DIR / "correction_moyenne_v7.json", "w", encoding="utf-8") as f:
    json.dump({"ridge_alpha": RIDGE_ALPHA, "features": FEATURES,
               "niveaux": {f"{int(Z_LEVELS[k])}_{c}": {"utilise": bool(USE[(k, c)]),
                                                        "mu": REG[(k, c)]["mu"].tolist(),
                                                        "sd": REG[(k, c)]["sd"].tolist(),
                                                        "coef": REG[(k, c)]["coef"].tolist()}
                           for (k, c) in REG}}, f, indent=2)
print("Parametres enregistres :", RUN_DIR / "correction_moyenne_v7.json")
print("BLOC 10f OK")
