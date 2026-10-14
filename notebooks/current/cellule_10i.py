# ==== CELLULE 10i du notebook fno3d_v7.ipynb (lot M1) — a coller juste APRES le BLOC 10h ====
# SHA-256 du CODE DE LA CELLULE (tout ce qui suit la ligne '# ==== DEBUT', en UTF-8, fins de ligne LF,
# terminant par un seul saut de ligne) :
#   a1fcc878ed3c6c6047538e5b48df25c6633c972a413d6bd42f415926599fbd6f
# Lit seulement : globales des BLOCs 0 a 10 + RUN_DIR/couche_limite_v7.json
#   (Drive : /content/drive/MyDrive/aneo_fno3d_runs/v7/couche_limite_v7.json, fige, jamais reecrit).
# Ecrit seulement : EVAL_DIR/decision_10i_par_niveau.csv et EVAL_DIR/decision_10i.json.
# ==== DEBUT
# BLOC 10i — Evaluation FIGEE de la correction : mesure decisive (lot M1)
# Methode figee : v7 + correction du vent moyen par niveau du BLOC 10h (alpha 0.01, choisi sur la
# VALIDATION), rechargee depuis RUN_DIR/couche_limite_v7.json (meme calcul que le BLOC 12, etape 2b).
# Aucun reglage dans cette cellule : on mesure seulement.
#
# Variantes deja vues sur le TEST pendant la conception (l'estimation du test est donc optimiste) :
#   BLOC 10e (4 variantes, choix fait sur le test), 10f, 10g, 10h alpha 1, 10h alpha 0.01 (grille
#   etendue apres lecture du test de 10h alpha 1). Les 46 episodes n'ont servi a AUCUN choix.
#
# REGLE DE DECISION (fixee avant execution) : correction RETENUE si C1 ET C2 ET C3.
#   C1  test synoptique (DSO1, 50 % des capteurs) : pire niveau u et v apres correction <= 0.5 m/s et
#       R2 >= 0.95 a chaque niveau ; l'IC 95 % bootstrap est rapporte (ce n'est pas un seuil).
#   C2  episodes (46 instants jamais utilises) : la correction ne degrade pas.
#       Delta = RMSE apres - RMSE avant, bootstrap apparie (1000 tirages), u et v :
#       650-1000 m : borne haute de l'IC 95 % de Delta <= 0       (strict)
#       100-500 m  : borne haute de l'IC 95 % de Delta <= +0.02   (tolerant)
#   C3  DSO2, DSO3, DSO5 conformes et DSO4 (modele + correction) < 1 s.
#   Information, hors regle : validation avant/apres par niveau, test synoptique a 100 % des capteurs.
# BOOTSTRAP (choisi par Rayen avant execution) fixe l'IC utilise par C2 ; l'autre est rapporte pour information.
#   "blocs"    : episodes -> blocs de BLOC_LEN instants consecutifs, reechantillonnes A L'INTERIEUR de chaque
#                episode (les 46 instants viennent de 3 situations : les instants voisins sont correles) ;
#                test synoptique -> jours entiers reechantillonnes (grappes de 8 instants).
#                NB : avec 3 episodes seulement, l'IC est conditionnel a ces 3 situations.
#   "instants" : tirage simple des instants avec remise (IC trop etroit si les instants sont correles).
# Ecrit : EVAL_DIR/decision_10i_par_niveau.csv et EVAL_DIR/decision_10i.json (nouveaux fichiers).
import hashlib

BOOTSTRAP = "blocs"      # "blocs" | "instants" : IC utilise par C2 -- decide par Rayen (salon #332, avenant 33da965)
BLOC_LEN = 6             # longueur des blocs d'instants consecutifs (mode "blocs", episodes)
assert BOOTSTRAP in ("blocs", "instants")
for _v in ("syn_idx", "ep_idx", "EVAL_LEVELS", "COVER", "EVAL_DIR", "amp_dtype", "vsel"):
    if _v not in globals():
        raise RuntimeError("Executer d'abord le BLOC 9 puis le BLOC 10.")
assert KEEP_EVAL == 0.5, "KEEP_EVAL doit valoir 0.5 (protocole DSO1) : relancer le BLOC 1"
assert len(ep_idx) > 0, "aucun instant d'episode dans le test"
CORR_PATH = RUN_DIR / "couche_limite_v7.json"
with open(CORR_PATH, encoding="utf-8") as f:
    CORR = json.load(f)
assert abs(CORR["ridge_alpha"] - 0.01) < 1e-12, "couche_limite_v7.json doit venir du BLOC 10h avec alpha 0.01"
CORR_SHA = hashlib.sha256(CORR_PATH.read_bytes()).hexdigest()
C_PAR = {key: (np.array(p["mu"]), np.array(p["sd"]), np.array(p["coef"]))
         for key, p in CORR["niveaux"].items() if p["utilise"]}
model.eval()

print("=" * 96)
print("REGLE DE DECISION (fixee avant execution) : correction retenue si C1 et C2 et C3")
print("  C1  test synoptique : pire niveau u et v <= 0.5 m/s et R2 >= 0.95 a chaque niveau (IC 95 % rapporte)")
print("  C2  episodes : borne haute IC 95 % de (apres - avant) <= 0 a 650-1000 m, <= +0.02 a 100-500 m (u et v)")
print("  C3  DSO2, DSO3, DSO5 conformes et DSO4 (modele + correction) < 1 s")
print(f"IC pour C2 : BOOTSTRAP = '{BOOTSTRAP}' (blocs de {BLOC_LEN} instants a l'interieur de chaque episode ; "
      f"test : jours entiers) | l'autre mode est rapporte pour information")
print(f"Correction : {CORR_PATH.name} | alpha {CORR['ridge_alpha']:g} | rho_sl {CORR['variante_rho_sl']} "
      f"| {len(C_PAR)} niveaux x composantes | sha256 {CORR_SHA[:16]}")
print("=" * 96)


def correction_targets(S, sensor, tfeat, um, vm):
    """Moyenne cible de u, v par niveau : dict (k, c) -> m/s. Meme calcul que le BLOC 12, etape 2b.
    S (>= 7, NY, NX), sensor booleen, tfeat (8,), um / vm : moyennes par niveau predites par le modele."""
    m_s = sensor
    p_s = S[6][m_s].astype(np.float64)
    tv_s = p_s / (287.05 * S[4][m_s].astype(np.float64))      # temperature virtuelle (gaz parfait)
    z_s = Z_SENSOR[m_s]
    p_sl = p_s * np.exp(9.80665 * z_s / (287.05 * (tv_s + 0.5 * 0.0065 * z_s)))
    A_s = np.stack([np.ones(m_s.sum()), XC[m_s], YC[m_s], z_s], axis=1)
    cf, *_ = np.linalg.lstsq(A_s, p_sl, rcond=None)
    res_s = p_sl - A_s @ cf
    mad = 1.4826 * np.median(np.abs(res_s - np.median(res_s))) + 1e-9
    keep_s = np.abs(res_s - np.median(res_s)) < 5.0 * mad
    if keep_s.sum() > 100:
        cf, *_ = np.linalg.lstsq(A_s[keep_s], p_sl[keep_s], rcond=None)
    rho_sl = 1.2 if CORR["variante_rho_sl"] == "1.2" else float(np.mean(p_sl / (287.05 * (tv_s + 0.0065 * z_s))))
    ug2, vg2 = -cf[2] / (rho_sl * F_CORIOLIS), cf[1] / (rho_sl * F_CORIOLIS)
    d = cf[3] * 1e3                                          # indicateur de stabilite (Pa/km)
    surf_m = [float(S[c][m_s].mean()) for c in (0, 1, 3)]    # u, v, T moyens des capteurs visibles
    a = math.atan2(surf_m[1], surf_m[0]) - math.atan2(vg2, ug2)
    a = (a + math.pi) % (2 * math.pi) - math.pi              # angle de rotation estime
    rot = []
    for fr in (0.25, 0.5, 0.75):
        rot += [(ug2 * math.cos(fr * a) - vg2 * math.sin(fr * a)) / 10.0,
                (ug2 * math.sin(fr * a) + vg2 * math.cos(fr * a)) / 10.0]
    su, sv = float(tfeat[6]), float(tfeat[7])                 # vent thermique (x 200)
    tf = list(np.asarray(tfeat, dtype=np.float64))
    out = {}
    for k in range(NZ):
        for c in "uv":
            par = C_PAR.get(f"{int(Z_LEVELS[k])}_{c}")
            if par is None:
                continue
            x = np.array(tf + surf_m + [um[k], vm[k]] + [ug2 / 10.0, vg2 / 10.0, d, su * d, sv * d] + rot
                         + [su * max(d, 0.0), su * min(d, 0.0), sv * max(d, 0.0), sv * min(d, 0.0),
                            ug2 / 10.0 * d, vg2 / 10.0 * d])
            out[(k, c)] = float(np.r_[1.0, (x - par[0]) / par[1]] @ par[2])
    return out


def predict_corrected(s_raw, sensor, tfeat):
    """(u, v, w) du modele seul et (u, v) corriges ; tenseurs (1, NZ, NY, NX) sur le GPU."""
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        pred = model(assemble_input(s_raw.to(device), sensor.to(device), tfeat.to(device)))
    u, v, w = denorm(pred.float())
    um = np.array([u[0, k][VALID_T[k]].mean().item() if VALID[k].any() else 0.0 for k in range(NZ)])
    vm = np.array([v[0, k][VALID_T[k]].mean().item() if VALID[k].any() else 0.0 for k in range(NZ)])
    tg = correction_targets(s_raw[0].numpy(), sensor[0].numpy() > 0.5, tfeat[0].numpy(), um, vm)
    uc, vc = u.clone(), v.clone()
    for (k, c), t_ in tg.items():
        f_ = uc if c == "u" else vc
        mk = VALID_T[k]
        f_[0, k] += (t_ - f_[0, k][mk].mean()) * mk          # decalage des noeuds valides du niveau
    return u, v, w, uc, vc


def run(ds):
    """Par instant : SSE avant / apres (N, NZ, 2), sommes de la verite (R2), masse et rugosite apres."""
    N = len(ds)
    R = {"avant": np.zeros((N, NZ, 2)), "apres": np.zeros((N, NZ, 2)),
         "st": np.zeros((N, NZ, 2)), "st2": np.zeros((N, NZ, 2)), "mass": [], "rough": []}
    for i, (s_raw, sensor, tfeat, y, _) in enumerate(DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS)):
        u, v, w, uc, vc = predict_corrected(s_raw, sensor, tfeat)
        y = y.to(device)
        for k in EVAL_LEVELS:
            mk = VALID_T[k]
            for ci in range(2):
                tk = y[0, ci, k][mk].double()
                R["avant"][i, k, ci] = (((u, v)[ci][0, k][mk].double() - tk) ** 2).sum().item()
                R["apres"][i, k, ci] = (((uc, vc)[ci][0, k][mk].double() - tk) ** 2).sum().item()
                R["st"][i, k, ci], R["st2"][i, k, ci] = tk.sum().item(), (tk ** 2).sum().item()
        r = mass_residual(y[:, 3], uc, vc, w)
        R["mass"].append(torch.sqrt((r[:, MASS_MASK_T] ** 2).mean()).item())
        R["rough"].append(((roughness(uc) + roughness(vc)) / (roughness(y[:, 0]) + roughness(y[:, 1]))).item())
    return R


def run_keep(stems_idx, seed, keep, stems=None, hours=None, doys=None):
    global KEEP_EVAL
    saved = KEEP_EVAL
    try:
        KEEP_EVAL = keep
        st_, hr_, dy_ = (te_s, te_h, te_d) if stems is None else (stems, hours, doys)
        return run(ANEO3D(st_[stems_idx], hr_[stems_idx], dy_[stems_idx], train=False, seed=seed))
    finally:
        KEEP_EVAL = saved


NK = np.maximum(VALID.reshape(NZ, -1).sum(axis=1), 1).astype(np.float64)
LV = np.array(EVAL_LEVELS)
B = 1000


def rmse_lv(sse_sum, n_inst):
    return np.sqrt(sse_sum / (n_inst * NK[:, None]))              # (NZ, 2)


DAY_OF = dict(zip(INDEX["stem"], INDEX["day"]))


def sampler(stems, mode, kind, seed):
    """Fonction qui tire un vecteur d'indices d'instants (avec remise).
    mode "instants" : tirage simple. mode "blocs" : kind "episodes" -> blocs de BLOC_LEN instants consecutifs
    reechantillonnes a l'interieur de chaque episode (jour) ; kind "jours" -> jours entiers reechantillonnes."""
    rng = np.random.default_rng(seed)
    n_all = len(stems)
    days = np.array([DAY_OF[s] for s in stems])
    groups = [np.nonzero(days == d)[0] for d in pd.unique(days)]   # instants dans l'ordre chronologique
    if mode == "instants":
        return lambda: rng.integers(0, n_all, n_all)
    if kind == "jours":
        return lambda: np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])

    def draw():
        out = []
        for g in groups:
            n, L = len(g), min(BLOC_LEN, len(g))
            starts = rng.integers(0, n - L + 1, math.ceil(n / L))
            out.append(np.concatenate([g[s:s + L] for s in starts])[:n])
        return np.concatenate(out)
    return draw


def boot(R, draw):
    """RMSE complets et tirages bootstrap apparies (memes instants avant / apres)."""
    N = R["avant"].shape[0]
    full = {n: rmse_lv(R[n].sum(0), N) for n in ("avant", "apres")}
    dr = {n: np.empty((B, NZ, 2)) for n in ("avant", "apres")}
    for b in range(B):
        idx = draw()
        for n in ("avant", "apres"):
            dr[n][b] = rmse_lv(R[n][idx].sum(0), len(idx))
    return full, dr


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def worst(full, dr, n, ci_):
    return float(full[n][LV, ci_].max()), ci(dr[n][:, LV, ci_].max(axis=1))


# ---------------------------------------------------------------- 1. Mesures
t0 = time.time()
R_val = run(ANEO3D(va_s[vsel], va_h[vsel], va_d[vsel], train=False, seed=1000))
R_syn = run(ANEO3D(te_s[syn_idx], te_h[syn_idx], te_d[syn_idx], train=False, seed=2000))
R_ep = run(ANEO3D(te_s[ep_idx], te_h[ep_idx], te_d[ep_idx], train=False, seed=2500))
R_100 = run_keep(syn_idx, 2000, 1.0)
sub = syn_idx[np.unique(np.linspace(0, len(syn_idx) - 1, min(40, len(syn_idx))).astype(int))]
R_25 = run_keep(sub, 3000, 0.25)
print(f"Instants : validation {len(vsel)} | test synoptique {len(syn_idx)} | episodes {len(ep_idx)} "
      f"| 100 % capteurs {len(syn_idx)} | 25 % capteurs {len(sub)} ({time.time() - t0:.0f} s)")
BOOTS = {}
for mode in ("instants", "blocs"):
    BOOTS[("syn", mode)] = boot(R_syn, sampler(te_s[syn_idx], mode, "jours", 101))
    BOOTS[("ep", mode)] = boot(R_ep, sampler(te_s[ep_idx], mode, "episodes", 202))
F_syn, D_syn = BOOTS[("syn", BOOTSTRAP)]
F_ep, D_ep = BOOTS[("ep", BOOTSTRAP)]
OTHER = "instants" if BOOTSTRAP == "blocs" else "blocs"
D_syn_o, D_ep_o = BOOTS[("syn", OTHER)][1], BOOTS[("ep", OTHER)][1]
_ep_days = pd.Series([DAY_OF[s] for s in te_s[ep_idx]]).value_counts(sort=False)
print("Episodes par jour : " + ", ".join(f"jour {d} : {n} instants" for d, n in _ep_days.items()))
F_val = {n: rmse_lv(R_val[n].sum(0), R_val["avant"].shape[0]) for n in ("avant", "apres")}
F_100 = {n: rmse_lv(R_100[n].sum(0), R_100["avant"].shape[0]) for n in ("avant", "apres")}
F_25 = {n: rmse_lv(R_25[n].sum(0), R_25["avant"].shape[0]) for n in ("avant", "apres")}

# DSO4 : modele seul / modele + correction (assemblage de l'entree inclus, comme au BLOC 10)
s1, m1, t1_, _, _ = ANEO3D(te_s[syn_idx], te_h[syn_idx], te_d[syn_idx], train=False, seed=2000)[0]
s1, m1, t1_ = s1[None], m1[None], t1_[None]


def timed(correct):
    t_ = time.perf_counter()
    if correct:
        predict_corrected(s1, m1, t1_)
    else:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            denorm(model(assemble_input(s1.to(device), m1.to(device), t1_.to(device))).float())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t_


for _ in range(3):
    timed(True)
T_INF = {c: float(np.mean([timed(c) for _ in range(10 if TEST_MODE else 20)])) for c in (False, True)}

# ---------------------------------------------------------------- 2. Tableaux
rows = []
print("=" * 96)
print("VALIDATION (information) : RMSE par niveau, avant -> apres")
print(" Niveau |  u avant  u apres |  v avant  v apres")
for k in LV:
    print(f" {int(Z_LEVELS[k]):5d} m |   {F_val['avant'][k, 0]:.3f}   {F_val['apres'][k, 0]:.3f}  |"
          f"   {F_val['avant'][k, 1]:.3f}   {F_val['apres'][k, 1]:.3f}")
print("=" * 96)
print(f"TEST SYNOPTIQUE (C1) : {len(syn_idx)} instants, 50 % des capteurs")
print(" Niveau |  u avant  u apres  [IC 95 % apres] |  v avant  v apres  [IC 95 % apres]")
for k in LV:
    cu, cv = ci(D_syn["apres"][:, k, 0]), ci(D_syn["apres"][:, k, 1])
    print(f" {int(Z_LEVELS[k]):5d} m |   {F_syn['avant'][k, 0]:.3f}   {F_syn['apres'][k, 0]:.3f}  [{cu[0]:.3f}, {cu[1]:.3f}]"
          f"  |   {F_syn['avant'][k, 1]:.3f}   {F_syn['apres'][k, 1]:.3f}  [{cv[0]:.3f}, {cv[1]:.3f}]")
print("(controle : 'avant' = tableau du BLOC 10 ; 'apres' = colonne 'couche limite' du BLOC 10h, alpha 0.01)")
print("=" * 96)
print(f"EPISODES (C2) : {len(ep_idx)} instants jamais utilises, 50 % des capteurs | IC : bootstrap '{BOOTSTRAP}'"
      f" (borne haute en mode '{OTHER}' entre parentheses, information)")
print(" Niveau |  u avant  u apres  IC 95 % (apres-avant)          |  v avant  v apres  IC 95 % (apres-avant)          | seuil | C2")
c2 = True
for k in LV:
    tol = 0.0 if Z_LEVELS[k] >= 650 else 0.02
    d_u = ci(D_ep["apres"][:, k, 0] - D_ep["avant"][:, k, 0])
    d_v = ci(D_ep["apres"][:, k, 1] - D_ep["avant"][:, k, 1])
    o_u = ci(D_ep_o["apres"][:, k, 0] - D_ep_o["avant"][:, k, 0])[1]
    o_v = ci(D_ep_o["apres"][:, k, 1] - D_ep_o["avant"][:, k, 1])[1]
    ok_k = d_u[1] <= tol + 1e-12 and d_v[1] <= tol + 1e-12
    c2 = c2 and ok_k
    print(f" {int(Z_LEVELS[k]):5d} m |   {F_ep['avant'][k, 0]:.3f}   {F_ep['apres'][k, 0]:.3f}  [{d_u[0]:+.3f}, {d_u[1]:+.3f}] ({o_u:+.3f})"
          f"  |   {F_ep['avant'][k, 1]:.3f}   {F_ep['apres'][k, 1]:.3f}  [{d_v[0]:+.3f}, {d_v[1]:+.3f}] ({o_v:+.3f})"
          f"  | {tol:+.2f} | {'OK' if ok_k else 'NON'}")
print("(controle : 'avant' = tableau EPISODES RAPIDES du BLOC 10)")
for k in LV:
    for ci_, c in enumerate("uv"):
        rows.append({"z_m": int(Z_LEVELS[k]), "comp": c,
                     "val_avant": F_val["avant"][k, ci_], "val_apres": F_val["apres"][k, ci_],
                     "syn_avant": F_syn["avant"][k, ci_], "syn_apres": F_syn["apres"][k, ci_],
                     "syn_apres_ic_bas": ci(D_syn["apres"][:, k, ci_])[0],
                     "syn_apres_ic_haut": ci(D_syn["apres"][:, k, ci_])[1],
                     "ep_avant": F_ep["avant"][k, ci_], "ep_apres": F_ep["apres"][k, ci_],
                     "ep_delta_ic_bas": ci(D_ep["apres"][:, k, ci_] - D_ep["avant"][:, k, ci_])[0],
                     "ep_delta_ic_haut": ci(D_ep["apres"][:, k, ci_] - D_ep["avant"][:, k, ci_])[1],
                     "ep_delta_ic_haut_autre_mode": ci(D_ep_o["apres"][:, k, ci_] - D_ep_o["avant"][:, k, ci_])[1],
                     "bootstrap": BOOTSTRAP,
                     "cap100_avant": F_100["avant"][k, ci_], "cap100_apres": F_100["apres"][k, ci_],
                     "cap25_avant": F_25["avant"][k, ci_], "cap25_apres": F_25["apres"][k, ci_]})
pd.DataFrame(rows).to_csv(EVAL_DIR / "decision_10i_par_niveau.csv", index=False)

# ---------------------------------------------------------------- 3. Pires niveaux, criteres, verdict
def ok(b):
    return "CONFORME" if b else "NON CONFORME"


N_syn = R_syn["avant"].shape[0]
n_all = N_syn * NK[:, None]
r2_apres = 1 - R_syn["apres"].sum(0) / (R_syn["st2"].sum(0) - R_syn["st"].sum(0) ** 2 / n_all)
W = {}
print("=" * 96)
print(f"PIRE NIVEAU (IC 95 %, 1000 tirages ; mode '{BOOTSTRAP}', puis mode '{OTHER}' pour information)")
for lab, F_, D_, Do_ in (("test synoptique", F_syn, D_syn, D_syn_o), ("episodes", F_ep, D_ep, D_ep_o)):
    for ci_, c in enumerate("uv"):
        wa, ca = worst(F_, D_, "avant", ci_)
        wp, cp = worst(F_, D_, "apres", ci_)
        _, cao = worst(F_, Do_, "avant", ci_)
        _, cpo = worst(F_, Do_, "apres", ci_)
        W[f"{lab}_{c}"] = {"avant": wa, "avant_ic95": ca, "apres": wp, "apres_ic95": cp,
                           "avant_ic95_autre_mode": cao, "apres_ic95_autre_mode": cpo}
        print(f"  {lab:15s} {c} : avant {wa:.3f} [{ca[0]:.3f}, {ca[1]:.3f}] -> apres {wp:.3f} [{cp[0]:.3f}, {cp[1]:.3f}]"
              f" m/s   ({OTHER} : [{cao[0]:.3f}, {cao[1]:.3f}] -> [{cpo[0]:.3f}, {cpo[1]:.3f}])")
w100 = {c: (float(F_100["avant"][LV, ci_].max()), float(F_100["apres"][LV, ci_].max())) for ci_, c in enumerate("uv")}
print(f"  100 % capteurs (information) : u {w100['u'][0]:.3f} -> {w100['u'][1]:.3f} | v {w100['v'][0]:.3f} -> {w100['v'][1]:.3f} m/s")
c1 = (W["test synoptique_u"]["apres"] <= DSO1_RMSE and W["test synoptique_v"]["apres"] <= DSO1_RMSE
      and bool((r2_apres[LV] >= DSO1_R2).all()))
m_c, r_c = float(np.mean(R_syn["mass"])), float(np.mean(R_syn["rough"]))
r5 = float(F_25["apres"][LV].max())
c3 = (m_c <= DSO2_RMS) and (0.9 <= r_c <= 1.1) and (r5 <= DSO5_RMSE) and (T_INF[True] < DSO4_TIME)
print("=" * 96)
print(f"C1  DSO1 sur le test synoptique (pire niveau <= {DSO1_RMSE}, R2 min {r2_apres[LV].min():.4f})  -> {ok(c1)}")
print(f"C2  episodes : la correction ne degrade pas (650-1000 m strict, 100-500 m +0.02)        -> {ok(c2)}")
print(f"C3  DSO2 {m_c:.2e} | DSO3 {r_c:.3f} | DSO5 {r5:.3f} | DSO4 {T_INF[True] * 1000:.0f} ms "
      f"(modele seul {T_INF[False] * 1000:.0f} ms)  -> {ok(c3)}")
retained = bool(c1 and c2 and c3)
print("=" * 96)
print("DECISION : correction " + ("RETENUE (v7 + correction)" if retained else "NON RETENUE (v7 seul)"))
print("Reserve : coefficients appris sur donnees synthetiques ; sur donnees reelles, l'erreur des barometres")
print("(+-10 a 50 Pa) est du meme ordre que le signal geostrophique (~26 Pa pour 10 m/s) : a recalibrer.")
with open(EVAL_DIR / "decision_10i.json", "w", encoding="utf-8") as f:
    json.dump({"correction": CORR_PATH.name, "correction_sha256": CORR_SHA, "alpha": CORR["ridge_alpha"],
               "regle": {"C1": "pire niveau u, v <= 0.5 et R2 >= 0.95 (test synoptique, 50 %)",
                         "C2": "IC95 haut de (apres - avant) <= 0 a 650-1000 m, <= +0.02 a 100-500 m (episodes)",
                         "C3": "DSO2, DSO3, DSO5 conformes et DSO4 modele + correction < 1 s"},
               "bootstrap": {"mode_C2": BOOTSTRAP, "bloc_instants": BLOC_LEN, "tirages": B,
                             "episodes_par_jour": {str(d): int(n) for d, n in _ep_days.items()}},
               "pire_niveau": W, "cap100_pire": w100, "r2_min_test_apres": float(r2_apres[LV].min()),
               "dso2": m_c, "dso3": r_c, "dso5_25pct": r5,
               "dso4_s": {"modele": T_INF[False], "modele_plus_correction": T_INF[True]},
               "C1": bool(c1), "C2": bool(c2), "C3": bool(c3), "retenue": retained,
               "instants": {"validation": int(len(vsel)), "test_synoptique": int(len(syn_idx)),
                            "episodes": int(len(ep_idx)), "cap25": int(len(sub))}}, f, indent=2)
print("Resultats :", EVAL_DIR / "decision_10i_par_niveau.csv", "et", EVAL_DIR / "decision_10i.json")
print("BLOC 10i OK")
