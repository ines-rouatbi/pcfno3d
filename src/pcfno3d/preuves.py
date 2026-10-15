"""Lot M8 — preuves : comparaison aux methodes de reference (L6) et etudes d'aide au
dimensionnement du reseau E1, E2, E3 (L9).

Definitions (cahier des charges v9, section 19.1, p. 42 ; taches 30 a 32, p. 45) :

    E1  etude de densite du reseau : erreur par niveau selon la part de stations (5 a
        100 %) ; densite recommandee (besoin BF35, sortie S8). On y ajoute les reseaux
        reguliers de la section 18.2 (p. 41 : un capteur tous les 300 m, 500 m, 1 km),
        « a confirmer sur les 120 instants de test » (section 18.4, p. 42).
    E2  placement optimal des stations : ajout iteratif d'une station la ou l'incertitude
        est la plus forte ; carte des emplacements prioritaires ; gain par rapport a un
        placement au hasard (BF36, S9).
    E3  capteur virtuel de secours : valeur d'une station en panne donnee par son noeud
        virtuel ; erreur de remplacement chiffree (BF37, S10).

Protocole commun : instants synoptiques du TEST (graine 2000, comme DSO1), memes capteurs
visibles et memes noeuds valides pour toutes les methodes (`evaluate.metriques_loader`),
metriques cumulees par niveau. Tout ce qui se regle (variogramme, alpha, emplacements E2)
se regle sur la VALIDATION.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import (IDW, CalculReferences, Krigeage, Variogramme, interpoler,
                        predicteurs, references, xy_colonnes)
from .config import Config
from .dataset import ANEO3D, eval_loader, is_synoptic, lattice_mask, make_splits

FRACTIONS_E1 = (0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
ESPACEMENTS_E1 = (300.0, 500.0, 1000.0)
GRAINE_TEST = 2000            # meme graine que DSO1 (Evaluator.run)
GRAINE_VALIDATION = 1000      # meme graine que la validation de l'entrainement et M2


def _f(x) -> float | None:
    """float JSON strict : None pour NaN / inf."""
    x = float(x)
    return x if np.isfinite(x) else None


class Banc:
    """Ce que partagent les preuves : configuration, geometrie, decoupage, modele
    (facultatif, via un `Evaluator`) et methodes de reference.

    Sans modele, seules les references sont evaluees (CPU, sans poids) ; avec un
    `Evaluator`, PC-FNO3D passe par `Evaluator.predire`, le chemin exact de DSO1.
    """

    def __init__(self, cfg: Config, evaluator=None, ajustement: dict | None = None,
                 noms_references=None, verbose: bool = False) -> None:
        from .cache import load_index
        from .geometry import Geometry
        from .losses import PhysicsLoss

        self.cfg = cfg
        self.ev = evaluator
        self.verbose = verbose
        if evaluator is not None:
            self.geom, self.splits = evaluator.geom, evaluator.splits
            self.roughness_ratio = evaluator.loss_fn.roughness_ratio
        else:
            self.geom = Geometry(cfg, device="cpu")
            self.splits = make_splits(cfg, load_index(cfg))
            # la rugosite (DSO3) n'utilise que la geometrie : pas de statistiques requises
            self.roughness_ratio = PhysicsLoss(cfg, self.geom, None).roughness_ratio
        self.ajustement = ajustement
        self.refs = references(ajustement, noms_references)
        self.calc = CalculReferences(self.geom, self.refs)
        s = self.splits.test[0]
        syn = np.array([is_synoptic(self.splits.stamp_of[x]) for x in s], dtype=bool)
        # mini-jeu sans instant synoptique : tout le test (meme repli que Evaluator.run)
        self.syn_idx = np.nonzero(syn)[0] if syn.any() else np.arange(len(s))

    @property
    def wf(self):
        return None if self.ev is None else self.ev.wf

    def instants(self, n: int | None) -> np.ndarray:
        idx = self.syn_idx
        if n is not None and len(idx) > n:
            idx = idx[np.unique(np.linspace(0, len(idx) - 1, n).astype(int))]
        if len(idx) == 0:
            raise ValueError("aucun instant de test en cache")
        return idx

    def predicteurs(self, modele: bool = True, refs: bool = True) -> dict:
        p = {}
        if modele and self.ev is not None:
            p["pcfno3d"] = self.ev.predire
        if refs:
            p.update(predicteurs(self.calc, self.geom.device))
        if not p:
            raise ValueError("aucune methode a evaluer")
        return p

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


# ---------------------------------------------------------------------- comparaison (L6)
def _resume_methode(levels: pd.DataFrame, per: pd.DataFrame, cfg: Config) -> dict:
    from .dso import dso3_conforme
    from .evaluate import verdicts_dso1

    ev, d1 = verdicts_dso1(levels, cfg)
    rmax = ev[["rmse_u", "rmse_v"]].max(axis=1)
    rough = float(per["roughness_ratio"].mean())
    out = {"par_niveau": [{k: (_f(v) if isinstance(v, (float, np.floating)) else v)
                           for k, v in r.items()} for r in levels.to_dict(orient="records")],
           **d1,
           "rmse_max_uv": _f(rmax.max()),
           "niveau_rmse_max_m": int(ev.loc[rmax.idxmax(), "z_m"]) if len(ev) else None,
           "r2_min_uv": _f(ev[["r2_u", "r2_v"]].min(axis=1).min()),
           "dso2_rms_moyen": _f(per["mass_rms"].mean()),
           "dso2_conforme": bool(per["mass_rms"].mean() <= cfg.dso.mass_rms),
           "dso3_rugosite": _f(rough),
           "dso3_conforme": bool(dso3_conforme(rough, cfg.dso.roughness_range)),
           "mae_direction_deg": _f(per["mae_dir_deg"].mean())}
    return {k: (_f(v) if isinstance(v, float) else v) for k, v in out.items()}


class _Compteur:
    """Enveloppe d'un loader : compte les capteurs visibles par instant (E1)."""

    def __init__(self, loader) -> None:
        self.loader, self.n = loader, []

    def __iter__(self):
        for item in self.loader:
            self.n.append(int(item[1].sum()))
            yield item


def comparer(banc: Banc, n_instants: int | None = None, keep: float | None = None,
             spacing_m: float | None = None, reseau: np.ndarray | None = None,
             seed: int = GRAINE_TEST, modele: bool = True, refs: bool = True) -> dict:
    """PC-FNO3D et references sur les memes instants de test et les memes capteurs."""
    from .evaluate import metriques_loader

    cfg, geom = banc.cfg, banc.geom
    idx = banc.instants(n_instants)
    loader, stems = eval_loader(cfg, geom, banc.splits, idx, seed, keep=keep,
                                spacing_m=spacing_m, reseau=reseau)
    compteur = _Compteur(loader)
    res = metriques_loader(compteur, stems, geom, banc.splits.stamp_of,
                           banc.predicteurs(modele, refs), banc.roughness_ratio)
    if reseau is not None:
        desc = {"type": "fixe", "stations": int(np.asarray(reseau, bool).sum())}
    elif spacing_m:
        desc = {"type": "regulier", "espacement_m": float(spacing_m)}
    else:
        desc = {"type": "fraction", "capteurs_visibles": cfg.mask.keep_eval if keep is None
                else float(keep)}
    return {"jeu": "test", "instants": int(len(idx)), "graine": seed, "reseau": desc,
            "capteurs_visibles_moyen": _f(np.mean(compteur.n)) if compteur.n else None,
            "methodes": {nom: _resume_methode(lv, per, cfg) for nom, (lv, per) in res.items()}}


def tableau_niveaux(comp: dict) -> pd.DataFrame:
    """Une ligne par (methode, niveau) : la table du rapport L6."""
    rows = []
    for nom, r in comp["methodes"].items():
        for lv in r["par_niveau"]:
            rows.append({"methode": nom, **lv})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- E1
def _recommandation(lignes: list[dict], cle: str, seuil: float, plus_petit: bool) -> dict:
    """Plus petite fraction (ou plus grand espacement) dont RMSE max u, v <= seuil."""
    ok = [x for x in lignes if x["rmse_max_uv"] is not None and x["rmse_max_uv"] <= seuil]
    if not ok:
        return {"valeur": None, "seuil_m_s": seuil}
    best = min(ok, key=lambda x: x[cle]) if plus_petit else max(ok, key=lambda x: x[cle])
    return {"valeur": best[cle], "seuil_m_s": seuil}


def etude_densite(banc: Banc, fractions=FRACTIONS_E1, espacements=ESPACEMENTS_E1,
                  n_instants: int | None = None, seed: int = GRAINE_TEST) -> dict:
    """E1. Les masques d'une meme graine sont EMBOITES d'une fraction a l'autre (meme tirage
    uniforme compare a keep) : les capteurs a 25 % font partie de ceux a 50 %."""
    cfg = banc.cfg
    lignes, details = [], []
    for f in fractions:
        banc.log(f"  E1 : {f:.0%} des capteurs")
        comp = comparer(banc, n_instants, keep=float(f), seed=seed)
        details.append(comp)
        for nom, r in comp["methodes"].items():
            lignes.append({"methode": nom, "reseau": "fraction", "capteurs_visibles": float(f),
                           "espacement_m": None, "capteurs_moyen": comp["capteurs_visibles_moyen"],
                           "rmse_max_uv": r["rmse_max_uv"],
                           "rmse_global_u": r["dso1_rmse_global_u"],
                           "rmse_global_v": r["dso1_rmse_global_v"],
                           "niveau_rmse_max_m": r["niveau_rmse_max_m"]})
    for e in espacements:
        banc.log(f"  E1 : un capteur tous les {e:.0f} m")
        comp = comparer(banc, n_instants, spacing_m=float(e), seed=seed)
        details.append(comp)
        for nom, r in comp["methodes"].items():
            lignes.append({"methode": nom, "reseau": "regulier", "capteurs_visibles": None,
                           "espacement_m": float(e),
                           "capteurs_moyen": comp["capteurs_visibles_moyen"],
                           "rmse_max_uv": r["rmse_max_uv"],
                           "rmse_global_u": r["dso1_rmse_global_u"],
                           "rmse_global_v": r["dso1_rmse_global_v"],
                           "niveau_rmse_max_m": r["niveau_rmse_max_m"]})
    reco = {}
    for nom in sorted({x["methode"] for x in lignes}):
        fr = [x for x in lignes if x["methode"] == nom and x["reseau"] == "fraction"]
        es = [x for x in lignes if x["methode"] == nom and x["reseau"] == "regulier"]
        reco[nom] = {
            "fraction_min_dso1": _recommandation(fr, "capteurs_visibles", cfg.dso.rmse, True),
            "fraction_min_dso5": _recommandation(fr, "capteurs_visibles", cfg.dso.rmse_25pct,
                                                 True),
            "espacement_max_dso1": _recommandation(es, "espacement_m", cfg.dso.rmse, False),
            "espacement_max_dso5": _recommandation(es, "espacement_m", cfg.dso.rmse_25pct,
                                                   False)}
    return {"definition": "E1, cahier v9 p. 42 (section 19.1) et p. 41 (section 18.2)",
            "regle_recommandation": "plus petite densite (fraction) ou plus grand espacement "
            "dont la RMSE max de u, v sur les niveaux evalues est <= au seuil DSO1 "
            f"({cfg.dso.rmse} m/s) ou DSO5 ({cfg.dso.rmse_25pct} m/s) ; regle fixee avant "
            "la mesure", "lignes": lignes, "recommandation": reco, "details": details}


# ---------------------------------------------------------------------- E2
def _score_incertitude(banc: Banc, reseau: np.ndarray, idx_val: np.ndarray) -> np.ndarray:
    """(ny, nx) : sigma BRUT de la vitesse (ensemble M2, non calibre), moyen sur les
    niveaux evalues et sur les instants de validation ; NaN hors colonnes utiles."""
    from .reliability import ensemble_sigma, member_masks

    geom, wf, rc = banc.geom, banc.wf, banc.cfg.reliability
    s, h, d = banc.splits.val
    ds = ANEO3D(s[idx_val], h[idx_val], d[idx_val], geom, train=False,
                seed=GRAINE_VALIDATION, reseau=reseau)
    lv = geom.eval_levels()
    tot = np.zeros((geom.ny, geom.nx))
    n = np.zeros((geom.ny, geom.nx))
    for s_raw, sensor, _t, _y, j in ds:
        s_np, m_np = s_raw.numpy(), sensor.numpy() > 0.5
        hour, doy = float(ds.hours[j]), int(ds.doys[j])
        served = wf._member(s_np, m_np, hour, doy)
        members = [wf._member(s_np, mm, hour, doy)
                   for mm in member_masks(m_np, rc.members, rc.keep, rc.seed)]
        sig = ensemble_sigma(served, members)["speed"][lv].astype(float)
        ok = geom.valid[lv]
        tot += np.where(ok, sig, 0.0).sum(axis=0)
        n += ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, tot / np.maximum(n, 1), np.nan)


def _choisir(score: np.ndarray, candidats: np.ndarray, n: int,
             espacement: int) -> list[tuple[int, int]]:
    """n colonnes candidates de score maximal, deux a deux a au moins `espacement` mailles
    (Chebyshev) : un ajout ne se concentre pas sur une seule zone."""
    sc = np.where(candidats & np.isfinite(score), score, -np.inf)
    ordre = np.argsort(sc.ravel())[::-1]
    nx = score.shape[1]
    pris: list[tuple[int, int]] = []
    for c in ordre:
        if not np.isfinite(sc.ravel()[c]) or len(pris) >= n:
            break
        j, i = divmod(int(c), nx)
        if all(max(abs(j - a), abs(i - b)) >= espacement for a, b in pris):
            pris.append((j, i))
    return pris


def _capteurs_possibles(banc: Banc, stems) -> np.ndarray:
    """Mailles portant un capteur disponible (S[7] > 0,5) a TOUS les instants donnes."""
    cache = banc.cfg.paths.cache_dir
    ok = np.ones((banc.geom.ny, banc.geom.nx), bool)
    for st in stems:
        ok &= np.load(cache / f"{st}_S.npy", mmap_mode="r")[7] > 0.5
    return ok


def placement_optimal(banc: Banc, base_espacement_m: float | None = 1000.0,
                      base_fraction: float | None = None, n_etapes: int = 4,
                      par_etape: int | None = None, espacement_mailles: int = 2,
                      n_validation: int | None = 8, n_test: int | None = None,
                      n_hasard: int = 3, graine: int = 3232) -> dict:
    """E2 : reseau de base fixe, puis `n_etapes` ajouts de `par_etape` stations la ou le
    sigma de l'ensemble M2 (sur la VALIDATION) est le plus fort. Gain mesure sur le TEST
    contre `n_hasard` placements au hasard du meme nombre de stations, parmi les memes
    candidats. Demande le modele (l'incertitude vient de l'ensemble de passes masquees)."""
    if banc.wf is None:
        raise ValueError("E2 demande le modele (incertitude par ensemble)")
    geom = banc.geom
    rng = np.random.default_rng(graine)
    shape = (geom.ny, geom.nx)
    if base_fraction is not None:
        base = rng.random(shape) < float(base_fraction)
        base_desc = {"type": "fraction", "capteurs": float(base_fraction)}
    else:
        base = lattice_mask(shape, float(base_espacement_m), geom.dx, rng)
        base_desc = {"type": "regulier", "espacement_m": float(base_espacement_m)}
    vs = banc.splits.val[0]
    idx_val = np.asarray(banc.splits.val_selection)
    if n_validation is not None and len(idx_val) > n_validation:
        idx_val = idx_val[np.unique(np.linspace(0, len(idx_val) - 1, n_validation).astype(int))]
    if len(idx_val) == 0:
        raise ValueError("E2 : aucun instant de validation en cache")
    possibles = _capteurs_possibles(banc, vs[idx_val]) & geom.valid.any(axis=0)
    base &= possibles
    candidats = possibles & ~base
    n_base = int(base.sum())
    par_etape = max(1, n_base // 20) if par_etape is None else int(par_etape)

    reseau = base.copy()
    choisis, scores = [], []
    for e in range(n_etapes):
        banc.log(f"  E2 : etape {e + 1}/{n_etapes} (incertitude sur la validation)")
        score = _score_incertitude(banc, reseau, idx_val)
        scores.append(score)
        pris = _choisir(score, candidats & ~reseau, par_etape, espacement_mailles)
        for j, i in pris:
            choisis.append({"etape": e + 1, "iy": j * geom.stride, "ix": i * geom.stride,
                            "x_m": float(i * geom.dx), "y_m": float(j * geom.dy),
                            "sigma_vitesse_m_s": _f(score[j, i])})
            reseau[j, i] = True
    n_ajout = len(choisis)

    banc.log("  E2 : evaluation sur le test (base, optimal, hasard)")
    res = {"base": comparer(banc, n_test, reseau=base, refs=False),
           "optimal": comparer(banc, n_test, reseau=reseau, refs=False)}
    hasard = []
    cand = np.flatnonzero(candidats)
    for h in range(n_hasard):
        r_h = np.random.default_rng(graine + 1 + h)
        m = base.copy()
        m.ravel()[r_h.choice(cand, size=min(n_ajout, len(cand)), replace=False)] = True
        hasard.append(comparer(banc, n_test, reseau=m, refs=False))

    def cle(c, k):
        return c["methodes"]["pcfno3d"][k]

    resume = {}
    for k in ("rmse_max_uv", "dso1_rmse_global_u", "dso1_rmse_global_v"):
        vh = [cle(c, k) for c in hasard if cle(c, k) is not None]
        mh = float(np.mean(vh)) if vh else float("nan")
        resume[k] = {"base": cle(res["base"], k), "optimal": cle(res["optimal"], k),
                     "hasard_moyen": _f(mh), "hasard_ecart_type": _f(np.std(vh)) if vh else None,
                     "gain_vs_hasard_m_s": _f(mh - cle(res["optimal"], k))
                     if cle(res["optimal"], k) is not None else None}
    return {"definition": "E2, cahier v9 p. 42 (section 19.1), tache 32 p. 45",
            "base": {**base_desc, "stations": n_base}, "etapes": n_etapes,
            "stations_par_etape": par_etape, "stations_ajoutees": n_ajout,
            "espacement_min_mailles": espacement_mailles,
            "incertitude": "sigma brut de la vitesse (ensemble M2 de reliability.members "
            "passes masquees), moyen sur les niveaux evalues et les instants de validation",
            "instants_validation": int(len(idx_val)), "graine": graine,
            "emplacements": choisis, "resume": resume,
            "details": {"base": res["base"], "optimal": res["optimal"], "hasard": hasard},
            "_cartes": {"scores": np.stack(scores) if scores else None, "base": base,
                        "reseau_final": reseau}}


# ---------------------------------------------------------------------- E3
def capteur_secours(banc: Banc, n_instants: int | None = None, n_pannes: int = 50,
                    espacement: int = 3, keep: float | None = None, seed: int = GRAINE_TEST,
                    graine_pannes: int = 3030) -> dict:
    """E3 : a chaque instant de test, `n_pannes` stations visibles tombent en panne (deux a
    deux a au moins `espacement` mailles) ; leur valeur est remplacee par :

        pcfno3d     le noeud virtuel : reconstruction SANS ces stations (caracteristiques
                    geostrophiques recalculees, comme en production), interpolee a
                    l'altitude du capteur (`alerts.interp_colonne`, meme regle que M4)
        plus_proche la station visible la plus proche
        idw         inverse des distances (8 voisins)
        krigeage    krigeage ordinaire (variogramme de validation), si l'ajustement existe

    Erreur de remplacement = valeur de remplacement - mesure (u, v) a la station."""
    from .alerts import interp_colonne
    from .an1 import choisir_capteurs

    geom, cfg = banc.geom, banc.cfg
    idx = banc.instants(n_instants)
    te_s, te_h, te_d = banc.splits.test
    ds = ANEO3D(te_s[idx], te_h[idx], te_d[idx], geom, train=False, seed=seed, keep_eval=keep)
    xy = xy_colonnes(geom)
    methodes = {"plus_proche": IDW(voisins=1), "idw": IDW()}
    if banc.ajustement is not None:
        methodes["krigeage"] = Krigeage(Variogramme.from_dict(banc.ajustement["variogramme"]))
    lignes = []
    for s_raw, sensor, _t, _y, j in ds:
        s_np, m = s_raw.numpy(), sensor.numpy() > 0.5
        rng = np.random.default_rng(graine_pannes + int(j))
        pannes = choisir_capteurs(m, n_pannes, espacement, rng)
        if not pannes:
            continue
        jj = np.array([p[0] for p in pannes])
        ii = np.array([p[1] for p in pannes])
        kept = m.copy()
        kept[jj, ii] = False
        if not kept.any():
            continue
        vrai_u, vrai_v = s_np[0][jj, ii].astype(float), s_np[1][jj, ii].astype(float)
        cible = jj * geom.nx + ii
        src = xy[kept.ravel()]
        vals = np.stack([s_np[0].ravel()[kept.ravel()], s_np[1].ravel()[kept.ravel()]])
        est = {nom: interpoler(it, src, vals, xy[cible]) for nom, it in methodes.items()}
        if banc.wf is not None:
            u, v, _w = banc.wf._member(s_np, kept, float(ds.hours[j]), int(ds.doys[j]))
            est["pcfno3d"] = np.stack([interp_colonne(u, geom.z_sensor, geom)[jj, ii],
                                       interp_colonne(v, geom.z_sensor, geom)[jj, ii]])
        stem = str(ds.stems[j])
        for q in range(len(pannes)):
            row = {"stem": stem, "ix": int(ii[q] * geom.stride), "iy": int(jj[q] * geom.stride),
                   "u_mesure": float(vrai_u[q]), "v_mesure": float(vrai_v[q]),
                   "h_capteur_m": float(geom.z_sensor[jj[q], ii[q]] - geom.zt[jj[q], ii[q]])}
            for nom, e in est.items():
                row[f"u_{nom}"], row[f"v_{nom}"] = float(e[0][q]), float(e[1][q])
            lignes.append(row)
    df = pd.DataFrame(lignes)
    noms = list(methodes) + (["pcfno3d"] if banc.wf is not None else [])
    resume = {}
    for nom in noms:
        if df.empty:
            resume[nom] = None
            continue
        eu, ev = df[f"u_{nom}"] - df["u_mesure"], df[f"v_{nom}"] - df["v_mesure"]
        sv = np.hypot(df[f"u_{nom}"], df[f"v_{nom}"]) - np.hypot(df["u_mesure"], df["v_mesure"])
        resume[nom] = {"n": int(len(df)), "rmse_u": _f(np.sqrt((eu ** 2).mean())),
                       "rmse_v": _f(np.sqrt((ev ** 2).mean())),
                       "rmse_vecteur": _f(np.sqrt((eu ** 2 + ev ** 2).mean())),
                       "biais_u": _f(eu.mean()), "biais_v": _f(ev.mean()),
                       "mae_vitesse": _f(np.abs(sv).mean()),
                       "p95_erreur_vecteur": _f(np.quantile(np.hypot(eu, ev), 0.95))}
    return {"definition": "E3, cahier v9 p. 42 (section 19.1), tache 30 p. 45",
            "jeu": "test", "instants": int(len(idx)), "graine": seed,
            "graine_pannes": graine_pannes, "pannes_par_instant": n_pannes,
            "espacement_min_mailles": espacement,
            "capteurs_visibles": cfg.mask.keep_eval if keep is None else float(keep),
            "altitude": "pcfno3d : altitude du capteur (geom.z_sensor), sous le premier "
            "noeud valide la valeur de ce noeud ; references : interpolation horizontale "
            "entre capteurs, donc deja a hauteur de capteur",
            "resume": resume, "_lignes": df}
