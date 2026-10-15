"""Lot M4 — simulation de pannes et de derives : mesure du critere AN1.

AN1 (cahier des charges) : au moins 90 % de detection et moins de 5 % de fausses alertes.

Protocole (detail et justification : docs/ALERTES.md) :

    scenario   un jour du jeu (instants synoptiques, dans l'ordre chronologique)
    injection  sur des mailles-capteurs disponibles a tous les instants du jour, tirees a
               graine fixe et espacees (pas deux defauts voisins) :
                 derive  biais (vecteur constant), gain (vitesse x (1 + g)),
                         girouette (rotation de la direction)
                 panne   valeur figee (u, v, T constants des le premier instant)
    chaine     la meme que le service : passe servie + cote^2 passes sans groupe de
               capteurs (`alerts.ecarts_capteurs`), persistance (`SuiviDerive`), regle M3
               des valeurs figees (`contract.valeurs_figees`)
    comptage   unite = (scenario, maille-capteur) ; signale = au moins une alerte
               capteur_derive a un instant du jour
                 detection        signales parmi les DERIVES d'amplitude effective
                                  >= e_ref (verdict)
                 fausses alertes  part des signales qui sont sains (verdict ; depend de la
                                  prevalence, d'ou le protocole fixe `PROTOCOLE`)
                 publies a cote   sains signales / sains ; fausses alertes par instant ;
                                  pannes (valeurs figees, regle M3), HORS verdict : leur
                                  detection par egalite exacte gonflerait le taux
    entrees    derive_entrees sur chaque instant propre (fausses alertes) et sur le meme instant
               avec une erreur d'entree connue (detection) : temperature en K, pression
               en hPa, humidite en fraction, densite en g/m3.

Les seuils (kappa, statistique) se reglent sur la VALIDATION (`choisir_kappa`) ; le test ne
sert qu'au verdict (`evaluer_an1`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .alerts import (METHOD, STATISTIQUES, SuiviDerive, derive_entrees, ecarts_capteurs,
                     iso_utc, physique, rho_instant)

DETECTION_MIN = 0.90
FAUSSES_MAX = 0.05
# Protocole du verdict officiel : la part de fausses alertes depend de la prevalence des
# defauts, donc leur nombre par jour est fixe. Tout autre reglage est « exploratoire ».
PROTOCOLE = {"derives_par_jour": 40, "pannes_par_jour": 10, "e_ref_m_s": 1.0}
Z_MIN = 2.0          # bas de la grille de kappa : statistiques gardees par instant
# (type, parametre) : biais en m/s, gain sans unite, girouette en degres
PLAN_DERIVES = (("biais", 0.5), ("biais", 1.0), ("biais", 2.0), ("biais", 3.0),
                ("gain", 0.15), ("gain", 0.30), ("girouette", 15.0), ("girouette", 30.0))
# erreurs d'entree connues : (variable, operation, valeur)
ERREURS_ENTREES = (("temperature_c", "plus", 273.15),        # kelvins au lieu de degC
                   ("pressure_pa", "fois", 0.01),            # hPa au lieu de Pa
                   ("relative_humidity_pct", "fois", 0.01),  # fraction au lieu de %
                   ("air_density_kg_m3", "fois", 1000.0))    # g/m3 au lieu de kg/m3
CANAUX = {"wind_u_m_s": 0, "wind_v_m_s": 1, "temperature_c": 3, "air_density_kg_m3": 4,
          "relative_humidity_pct": 5, "pressure_pa": 6}
CLASSES_AMPLITUDE = (0.0, 0.5, 1.0, 2.0, 3.0, np.inf)


def choisir_capteurs(present: np.ndarray, n: int, espacement: int,
                     rng: np.random.Generator) -> list[tuple[int, int]]:
    """n mailles (j, i) parmi `present`, deux a deux a au moins `espacement` mailles
    (distance de Chebyshev) : un defaut ne contamine pas la detection d'un autre."""
    cand = np.flatnonzero(present)
    rng.shuffle(cand)
    nx = present.shape[1]
    pris: list[tuple[int, int]] = []
    for c in cand:
        j, i = divmod(int(c), nx)
        if all(max(abs(j - a), abs(i - b)) >= espacement for a, b in pris):
            pris.append((j, i))
            if len(pris) >= n:
                break
    return pris


def appliquer_derive(u: float, v: float, type_: str, p: float, angle: float
                     ) -> tuple[float, float]:
    if type_ == "biais":
        return u + p * np.cos(angle), v + p * np.sin(angle)
    if type_ == "gain":
        return u * (1.0 + p), v * (1.0 + p)
    if type_ == "girouette":
        a = np.radians(p)
        return u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a)
    raise ValueError(f"type de derive inconnu : {type_!r}")


def erreur_entree(S: np.ndarray, sensor: np.ndarray, var: str, op: str, val: float
                  ) -> np.ndarray:
    out = S.copy()
    c = CANAUX[var]
    out[c] = np.where(sensor, out[c] + val if op == "plus" else out[c] * val, out[c])
    return out


def simuler_jour(wf, stems: list[str], acfg, rng: np.random.Generator, n_derives: int,
                 n_figees: int, statistiques=STATISTIQUES, plan=PLAN_DERIVES,
                 espacement: int | None = None) -> dict:
    """Un scenario : injection, chaine d'alertes instant par instant, resultats bruts.

    `acfg` : section `alerts` (cote, fenetre, min_instants, min_m_s, plancher, rho_source,
    seuils des entrees). Le seuil kappa n'intervient pas ici : on garde, par maille et par
    statistique, le maximum dans le temps de la statistique de decision (zmax), et kappa
    se choisit ensuite (`choisir_kappa`) ou s'applique (`evaluer_an1`).
    """
    from .contract import DUREE_FIGEE_S, N_INSTANTS_FIGEE, VITESSE_CALME, valeurs_figees

    geom = wf.geom
    stems = list(stems)
    data = [wf.surface_from_cache(s) for s in stems]
    present = np.logical_and.reduce([d[0][7] > 0.5 for d in data])
    esp = espacement or acfg.derive_cote + 1
    cells = choisir_capteurs(present, n_derives + n_figees, esp, rng)
    derives = [{"j": j, "i": i, "type": plan[k % len(plan)][0], "p": plan[k % len(plan)][1],
                "angle": float(rng.uniform(0, 2 * np.pi)), "ecart": 0.0, "n": 0}
               for k, (j, i) in enumerate(cells[:n_derives])]
    figees = [{"j": j, "i": i} for j, i in cells[n_derives:]]
    ref = data[0][0]
    suivis = {st: SuiviDerive(acfg.derive_fenetre) for st in statistiques}
    zmax = {st: np.full(geom.ny * geom.nx, -np.inf) for st in statistiques}
    vus = np.zeros(geom.ny * geom.nx, bool)
    series = {v: np.full((geom.ny * geom.nx, len(data)), np.nan)
              for v in ("wind_u_m_s", "wind_v_m_s", "temperature_c")}
    entrees = {"instants": 0, "fausses": 0,
               "detection": {f"{v}:{op}:{x:g}": 0 for v, op, x in ERREURS_ENTREES}}
    phys = []
    temps = []
    par_instant = {st: [] for st in statistiques}     # (cellules, z) eligibles, z >= Z_MIN
    for t, (S, hour, doy, stamp) in enumerate(data):
        S2 = S.copy()
        for d in derives:
            j, i = d["j"], d["i"]
            u, v = float(S2[0, j, i]), float(S2[1, j, i])
            u2, v2 = appliquer_derive(u, v, d["type"], d["p"], d["angle"])
            S2[0, j, i], S2[1, j, i] = u2, v2
            d["ecart"] += float(np.hypot(u2 - u, v2 - v))
            d["n"] += 1
        for f in figees:
            for c in (0, 1, 3):
                S2[c, f["j"], f["i"]] = ref[c, f["j"], f["i"]]
        sensor = S2[7] > 0.5
        vol = wf.predict(S2, hour, doy, timestamp=stamp, reliability=False)
        ec = ecarts_capteurs(S2, sensor, geom, (vol.u, vol.v),
                             lambda kept: wf._member(S2, kept, hour, doy),
                             acfg.derive_cote, statistiques, acfg.derive_echelle_plancher_m_s)
        inst = iso_utc(stamp)
        for st in statistiques:
            suivis[st].ajouter(inst, ec[st])
            s = suivis[st].statistiques()
            with np.errstate(invalid="ignore"):
                elig = (s["n"] >= acfg.derive_min_instants) & (s["a"] >= acfg.derive_min_m_s)
            zt = np.where(elig, s["z"], -np.inf).ravel()
            zmax[st] = np.fmax(zmax[st], zt)
            garde = np.flatnonzero(zt >= Z_MIN)
            par_instant[st].append((garde, zt[garde].astype(np.float32)))
            vus |= np.isfinite(ec[st].z).ravel()
        for v, c in (("wind_u_m_s", 0), ("wind_v_m_s", 1), ("temperature_c", 3)):
            series[v][:, t] = np.where(sensor, S2[c], np.nan).ravel()
        temps.append(pd.Timestamp(stamp))
        # entrees : instant tel quel (fausses alertes), puis erreurs connues (detection)
        st_mu, st_sd = wf.norm.stats["surf_mu"], wf.norm.stats["surf_sd"]
        kw = dict(z_max=acfg.entrees_z_max, ecart_extreme=acfg.entrees_ecart_extreme,
                  part_extreme=acfg.entrees_part_extreme)
        entrees["instants"] += 1
        # sur l'instant PROPRE : les derives de capteurs sont une autre famille
        propre = S[7] > 0.5
        entrees["fausses"] += int(any(x["derive"] for x in
                                      derive_entrees(S, propre, st_mu, st_sd, **kw)))
        for var, op, val in ERREURS_ENTREES:
            res = derive_entrees(erreur_entree(S, propre, var, op, val), propre, st_mu,
                                 st_sd, **kw)
            entrees["detection"][f"{var}:{op}:{val:g}"] += int(
                any(x["derive"] and x["variable"] == var for x in res))
        Y = (np.load(wf.cfg.paths.cache_dir / f"{stems[t]}_Y.npy")
             if acfg.rho_source == "verite_synthetique" else None)
        rho, src = rho_instant(geom, acfg.rho_source, S2, sensor, Y)
        phys.append(physique(geom, vol.u, vol.v, vol.w, rho, src, wf.cfg.dso.mass_rms))

    # panne : regle M3 des valeurs figees, sur la serie du jour
    temps_s = np.array([(x - temps[0]).total_seconds() for x in temps])
    with np.errstate(invalid="ignore"):
        actif = np.hypot(series["wind_u_m_s"], series["wind_v_m_s"]) >= VITESSE_CALME
    fig = np.zeros(geom.ny * geom.nx, bool)
    for v, s in series.items():
        fig |= valeurs_figees(s, N_INSTANTS_FIGEE, temps_s, DUREE_FIGEE_S,
                              actif=actif if v.startswith("wind_") else None).any(axis=1)
    nx = geom.nx
    return {"stems": stems, "zmax": zmax, "par_instant": par_instant, "vus": vus,
            "figee_signalee": fig,
            "derives": [{**d, "cellule": d["j"] * nx + d["i"],
                         "ecart_moyen": d["ecart"] / max(d["n"], 1)} for d in derives],
            "figees": [{**f, "cellule": f["j"] * nx + f["i"]} for f in figees],
            "entrees": entrees, "physique": phys}


def _taux(a: int, b: int) -> float | None:
    return a / b if b else None


def evaluer_an1(jours: list[dict], statistique: str, kappa: float, e_ref: float = 1.0
                ) -> dict:
    """Detection des derives, fausses alertes et verdict AN1 (capteurs) pour un kappa.

    Signale = alerte capteur_derive (zmax >= kappa). Les pannes (valeurs figees, regle M3)
    sont rapportees a part, hors verdict ; elles ne comptent ni comme saines ni comme
    fausses alertes."""
    det_ref = n_ref = fp = n_sains = signales_def = 0
    det_fig = det_fig_m3 = n_fig = fp_m3 = n_instants = fa_instants = 0
    par_type: dict[str, list[int]] = {}
    par_classe: dict[str, list[int]] = {}
    for jour in jours:
        flag = jour["zmax"][statistique] >= kappa
        defaut = np.zeros_like(flag)
        for d in jour["derives"]:
            c = d["cellule"]
            defaut[c] = True
            ok = bool(flag[c])
            k = f"{d['type']}:{d['p']:g}"
            par_type.setdefault(k, [0, 0])
            par_type[k][0] += ok
            par_type[k][1] += 1
            e = d["ecart_moyen"]
            b = int(np.searchsorted(CLASSES_AMPLITUDE, e, side="right") - 1)
            lab = f"[{CLASSES_AMPLITUDE[b]:g}, {CLASSES_AMPLITUDE[b + 1]:g})"
            par_classe.setdefault(lab, [0, 0])
            par_classe[lab][0] += ok
            par_classe[lab][1] += 1
            if e >= e_ref:
                det_ref += ok
                n_ref += 1
        for f in jour["figees"]:
            defaut[f["cellule"]] = True
            det_fig += bool(flag[f["cellule"]] or jour["figee_signalee"][f["cellule"]])
            det_fig_m3 += bool(jour["figee_signalee"][f["cellule"]])
            n_fig += 1
        sains = jour["vus"] & ~defaut
        fp += int((flag & sains).sum())
        fp_m3 += int((jour["figee_signalee"] & sains).sum())
        n_sains += int(sains.sum())
        signales_def += int((flag & defaut).sum())
        for cel, z in jour.get("par_instant", {}).get(statistique, []):
            n_instants += 1
            fa_instants += int((sains[cel] & (z >= kappa)).sum())
    detection = _taux(det_ref, n_ref)
    fausses = _taux(fp, fp + signales_def)
    if fausses is None:
        fausses = 0.0
    ok = detection is not None and detection >= DETECTION_MIN and fausses < FAUSSES_MAX
    return {"statistique": statistique, "kappa": float(kappa), "e_ref_m_s": float(e_ref),
            "detection": detection, "derives_ref": n_ref,
            "fausses_alertes": fausses, "fausses_alertes_n": fp,
            "taux_faux_positifs": _taux(fp, n_sains), "capteurs_sains": n_sains,
            "fausses_alertes_par_instant": _taux(fa_instants, n_instants),
            "detection_par_type": {k: _taux(*v) for k, v in par_type.items()},
            "detection_par_amplitude_m_s": {k: _taux(*v) for k, v in par_classe.items()},
            "pannes_hors_verdict": {"pannes": n_fig, "detection": _taux(det_fig, n_fig),
                                    "detection_regle_m3": _taux(det_fig_m3, n_fig),
                                    "fausses_regle_m3": fp_m3},
            "an1_capteurs_conforme": bool(ok)}


def evaluer_entrees(jours: list[dict]) -> dict:
    n = sum(j["entrees"]["instants"] for j in jours)
    fa = sum(j["entrees"]["fausses"] for j in jours)
    det: dict[str, int] = {}
    for j in jours:
        for k, x in j["entrees"]["detection"].items():
            det[k] = det.get(k, 0) + x
    total = sum(det.values())
    detection = _taux(total, n * len(det)) if det else None
    fausses = _taux(fa, n)
    ok = (detection is not None and detection >= DETECTION_MIN and fausses is not None
          and fausses < FAUSSES_MAX)
    return {"instants": n, "detection": detection,
            "detection_par_erreur": {k: _taux(x, n) for k, x in det.items()},
            "fausses_alertes": fausses, "an1_entrees_conforme": bool(ok)}


def choisir_kappa(jours: list[dict], statistique: str, grille=None, fdr_cible: float = 0.025,
                  e_ref: float = 1.0) -> dict:
    """Plus petit kappa (meilleure detection) dont la part de fausses alertes reste sous
    `fdr_cible` SUR LA VALIDATION (marge sous les 5 % de AN1). A n'appeler que sur la
    validation : le resultat porte « jeu: validation »."""
    grille = np.round(np.arange(2.0, 30.01, 0.25), 2) if grille is None else grille
    essais = [evaluer_an1(jours, statistique, k, e_ref) for k in grille]
    bons = [e for e in essais if e["fausses_alertes"] <= fdr_cible]
    choix = bons[0] if bons else essais[-1]
    return {"jeu": "validation", "methode": METHOD, "derive_statistique": statistique,
            "derive_kappa": choix["kappa"], "fdr_cible": fdr_cible, "e_ref_m_s": e_ref,
            "cible_atteinte": bool(bons), "validation": choix,
            "courbe": [{"kappa": e["kappa"], "detection": e["detection"],
                        "fausses_alertes": e["fausses_alertes"]} for e in essais]}
