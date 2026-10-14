# 2b. Correction du vent moyen par niveau (BLOC 10h) : v7 + correction = configuration qui atteint DSO1
# Coefficients appris sur l'entrainement, choisis sur la validation (RUN_DIR/couche_limite_v7.json).
# Seule la moyenne de chaque niveau est recalee : la structure spatiale predite est conservee.
APPLY_CORRECTION = True
CORR_PATH = RUN_DIR / "couche_limite_v7.json"
u_raw, v_raw = u.copy(), v.copy()                          # prediction du modele seul
corr_applied, t_corr = False, 0.0
if APPLY_CORRECTION and CORR_PATH.exists():
    t1 = time.perf_counter()
    with open(CORR_PATH, encoding="utf-8") as f:
        CORR = json.load(f)
    # vent geostrophique v2 (BLOC 10g) : temperature virtuelle exacte, altitude dans le plan
    m_s = sensor
    p_s = S[6][m_s].astype(np.float64)
    tv_s = p_s / (287.05 * S[4][m_s].astype(np.float64))
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
    d = cf[3] * 1e3                                        # indicateur de stabilite (Pa/km)
    surf_m = [float(S[c][m_s].mean()) for c in (0, 1, 3)]  # u, v, T moyens des capteurs visibles
    a = math.atan2(surf_m[1], surf_m[0]) - math.atan2(vg2, ug2)
    a = (a + math.pi) % (2 * math.pi) - math.pi            # angle de rotation estime
    rot = []
    for fr in (0.25, 0.5, 0.75):
        rot += [(ug2 * math.cos(fr * a) - vg2 * math.sin(fr * a)) / 10.0,
                (ug2 * math.sin(fr * a) + vg2 * math.cos(fr * a)) / 10.0]
    su, sv = float(tfeat[6]), float(tfeat[7])               # vent thermique (x 200)
    n_corr = 0
    for k in range(NZ):
        m = VALID[k]
        for c, fld in (("u", u), ("v", v)):
            par = CORR["niveaux"].get(f"{int(Z_LEVELS[k])}_{c}")
            if par is None or not par["utilise"] or m.sum() == 0:
                continue
            # memes 30 caracteristiques, dans le meme ordre, que le BLOC 10h
            x = np.array(list(tfeat.astype(np.float64)) + surf_m
                         + [float(u_raw[k][m].mean()), float(v_raw[k][m].mean())]
                         + [ug2 / 10.0, vg2 / 10.0, d, su * d, sv * d] + rot
                         + [su * max(d, 0.0), su * min(d, 0.0), sv * max(d, 0.0), sv * min(d, 0.0),
                            ug2 / 10.0 * d, vg2 / 10.0 * d])
            target = float(np.r_[1.0, (x - np.array(par["mu"])) / np.array(par["sd"])] @ np.array(par["coef"]))
            fld[k][m] += target - fld[k][m].mean()
            n_corr += 1
    corr_applied, t_corr = True, time.perf_counter() - t1
    print(f"Correction du vent moyen : {n_corr} niveaux x composantes recales en {t_corr * 1000:.0f} ms "
          f"(total avec le modele : {(t_rec + t_corr) * 1000:.0f} ms)")
elif APPLY_CORRECTION:
    print("Correction absente (executer le BLOC 10h) : reconstruction par le modele seul")
