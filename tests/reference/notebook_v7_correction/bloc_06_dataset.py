# BLOC 6 — Dataset, DataLoaders, assemblage de l'entree
def geostrophic_from_surface(S, sensor):
    """
    Vent geostrophique estime a partir des capteurs visibles :
    pression reduite au niveau de la mer, plan ajuste par moindres carres, equilibre geostrophique
    ug = -(1 / rho f) dp/dy ; vg = (1 / rho f) dp/dx.
    """
    m = sensor
    if m.sum() < 100:
        return 0.0, 0.0
    p = S[6][m].astype(np.float64)
    t_k = S[3][m].astype(np.float64) + 273.15
    z = Z_SENSOR[m]
    p_sl = p * np.exp(9.80665 * z / (287.05 * (t_k + 0.5 * 0.0065 * z)))
    A = np.stack([np.ones(m.sum()), XC[m], YC[m]], axis=1)
    coef, *_ = np.linalg.lstsq(A, p_sl, rcond=None)
    rho = float(S[4][m].mean())
    return -coef[2] / (rho * F_CORIOLIS), coef[1] / (rho * F_CORIOLIS)

def thermal_wind_from_surface(S, sensor):
    """
    Vent thermique (variation du vent geostrophique avec l'altitude, en s-1) estime a partir
    du gradient horizontal de temperature des capteurs visibles, ramenee au niveau de la mer :
    d(ug, vg)/dz = (g / (f T)) * (-dT/dy, dT/dx).
    """
    m = sensor
    if m.sum() < 100:
        return 0.0, 0.0
    t_red = S[3][m].astype(np.float64) + 0.0065 * Z_SENSOR[m]
    A = np.stack([np.ones(m.sum()), XC[m], YC[m]], axis=1)
    coef, *_ = np.linalg.lstsq(A, t_red, rcond=None)
    k = 9.80665 / (F_CORIOLIS * (float(t_red.mean()) + 273.15))
    return -k * coef[2], k * coef[1]

def time_geo_features(S, sensor, hour, doy):
    ug, vg = geostrophic_from_surface(S, sensor)
    su, sv = thermal_wind_from_surface(S, sensor)
    return np.array([math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
                     math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365),
                     ug / 10.0, vg / 10.0, su * 200.0, sv * 200.0], dtype=np.float32)

class ANEO3D(Dataset):
    """Renvoie la surface brute, les caracteristiques temporelles, la cible et le masque capteurs."""

    def __init__(self, stems, hours, doys, train, n_samples=None, seed=0):
        self.stems, self.hours, self.doys = list(stems), list(hours), list(doys)
        self.train = train
        self.n_samples = n_samples
        self.seed = seed

    def __len__(self):
        return self.n_samples if self.n_samples else len(self.stems)

    def __getitem__(self, i):
        if self.train:
            j = np.random.randint(len(self.stems))
            rng = np.random.default_rng()
            keep = rng.uniform(*KEEP_TRAIN)
        else:
            j = i
            rng = np.random.default_rng(self.seed + i)
            keep = KEEP_EVAL
        stem = self.stems[j]
        S = np.load(CACHE_DIR / f"{stem}_S.npy")
        Y = np.load(CACHE_DIR / f"{stem}_Y.npy")
        sensor = (S[7] > 0.5) & (rng.random(S[7].shape) < keep)
        tfeat = time_geo_features(S, sensor, self.hours[j], self.doys[j])
        return (torch.from_numpy(S[:N_SURF]), torch.from_numpy(sensor.astype(np.float32)),
                torch.from_numpy(tfeat), torch.from_numpy(Y), j)

def assemble_input(s_raw, sensor, tfeat):
    """Construit l'entree (B, C_IN, NZ, NY, NX) sur le GPU."""
    B = s_raw.shape[0]
    s = (s_raw - SURF_MU) / SURF_SD * sensor[:, None]
    s2d = torch.cat([s, sensor[:, None], STATIC_2D_T.expand(B, -1, -1, -1)], dim=1)
    x2d = s2d[:, :, None].expand(-1, -1, NZ, -1, -1)
    x3d = STATIC_3D_T.expand(B, -1, -1, -1, -1)
    xt = tfeat[:, :, None, None, None].expand(-1, -1, NZ, NY, NX)
    return torch.cat([x2d, x3d, xt], dim=1)

def subset(split):
    d = INDEX[INDEX["split"] == split]
    return d["stem"].values, d["hour"].values, d["doy"].values

tr_s, tr_h, tr_d = subset("train")
va_s, va_h, va_d = subset("val")
te_s, te_h, te_d = subset("test")
# Validation : instants synoptiques (toutes les 3 h) repartis regulierement sur les jours de validation
_ts_val = INDEX.set_index("stem").loc[va_s, "timestamp"]
_is_syn = np.array([(lambda t: t.minute == 0 and t.second == 0 and t.hour % 3 == 0)(pd.Timestamp(x))
                    for x in _ts_val])
_cand = np.nonzero(_is_syn)[0] if _is_syn.any() else np.arange(len(va_s))
vsel = _cand[np.unique(np.linspace(0, len(_cand) - 1, min(VAL_SAMPLES, len(_cand))).astype(int))]

train_ds = ANEO3D(tr_s, tr_h, tr_d, train=True, n_samples=STEPS_PER_EPOCH * BATCH_SIZE)
val_ds = ANEO3D(va_s[vsel], va_h[vsel], va_d[vsel], train=False, seed=1000)
test_ds = ANEO3D(te_s, te_h, te_d, train=False, seed=2000)

loader_kw = dict(num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
                 persistent_workers=NUM_WORKERS > 0)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kw)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kw)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **loader_kw)
print(f"Instants : train={len(tr_s)} | val={len(vsel)} (sur {len(va_s)}) | test={len(te_s)}")
print("BLOC 6 OK")