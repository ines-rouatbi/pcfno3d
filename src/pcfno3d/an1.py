"""Lot M4 — simulation de pannes et de derives : mesure du critere AN1 (v2).

AN1 (cahier des charges) : au moins 90 % de detection et moins de 5 % de fausses alertes.

Protocole (detail et justification : docs/ALERTES.md) :

    scenario   un jour du jeu (instants synoptiques, dans l'ordre chronologique)
    injection  sur des mailles-capteurs disponibles a tous les instants du jour, tirees a
               graine fixe et espacees (pas deux defauts voisins) :
                 derive  biais (vecteur constant), gain (vitesse x (1 + g)),
                         girouette (rotation de la direction)
                 panne   valeur figee (u, v, T constants des le premier instant)
    chaine     la meme que le service : detecteurs de derive (`Detecteur` : spatiale,
               innovation, influence, combinee), persistance (`alerts.SuiviDerive`), regle
               M3 des valeurs figees (`contract.valeurs_figees`)
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
               en hPa, humidite en fraction, densite en g/m3. Les statistiques brutes sont
               gardees : la regle se regle ensuite sur la validation (`choisir_regle_entrees`).

v2 (apres la premiere lecture du test du 16/10/2026, non conforme) : detecteur spatial
(mediane des voisins, correction de site apprise sur l'entrainement, cumul vectoriel),
grille de detecteurs fixee d'avance (`GRILLE_DETECTEURS`) et regle des entrees sans le vent
(`GRILLE_ENTREES`). Tout se regle sur la VALIDATION (`choisir_detecteur`,
`choisir_regle_entrees`) ; le test ne sert qu'au verdict (`evaluer_an1`, `evaluer_entrees`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alerts import (METHOD, STATISTIQUES, STATISTIQUES_PASSES, VARIABLES_ENTREES,
                     ModeleSites, SuiviDerive, ajuster_sites, combiner, composantes,
                     derive_entrees, ecart_voisins, ecarts_capteurs, ecarts_spatiaux,
                     heure_utc, iso_utc, mode_suivi, physique, regle_entrees, rho_instant)
from .geometry import SURF_VARS

DETECTION_MIN = 0.90
FAUSSES_MAX = 0.05
# Protocole du verdict officiel : la part de fausses alertes depend de la prevalence des
# defauts, donc leur nombre par jour est fixe. Tout autre reglage est « exploratoire ».
PROTOCOLE = {"derives_par_jour": 40, "pannes_par_jour": 10, "e_ref_m_s": 1.0}
Z_MIN = 2.0          # bas de la grille de kappa : statistiques gardees (sains)
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

# Grilles fixees d'avance (v2, 16/10/2026, avant toute mesure GPU de la v2). Le choix se
# fait sur la validation seulement ; l'ordre sert a departager (le plus simple d'abord).
GRILLE_DETECTEURS = {"rayon": (1, 2, 3), "rayon_echelle": (2, 4), "fenetre": (3, 4, 8),
                     "sites": (True, False)}
GRILLE_KAPPA = tuple(float(k) for k in np.round(np.arange(2.0, 30.01, 0.25), 2))
GRILLE_ENTREES = {"z_max": (3.0, 4.0, 5.0, 6.0, 8.0),
                  "part_extreme": (0.02, 0.05, 0.10, 0.25, 0.50)}
# regle de la v1 (toutes les variables, vent compris), rejouee pour le diagnostic
REGLE_ENTREES_V1 = {"z_max": 3.0, "part_extreme": 0.02, "variables": tuple(SURF_VARS)}


@dataclass(frozen=True)
class Detecteur:
    """Un detecteur de derive candidat : statistique, persistance et voisinage."""

    statistique: str
    fenetre: int
    rayon: int | None = None
    rayon_echelle: int | None = None
    sites: bool = False

    def __post_init__(self) -> None:
        composantes(self.statistique)                     # statistique connue
        if "spatiale" in composantes(self.statistique) and not (self.rayon and
                                                                 self.rayon_echelle):
            raise ValueError("detecteur spatial : rayon et rayon_echelle requis")

    @property
    def spatial(self) -> bool:
        return "spatiale" in composantes(self.statistique)

    @property
    def min_instants(self) -> int:
        """Fenetre complete : le detecteur juge sur W instants consecutifs."""
        return int(self.fenetre)

    @property
    def cle(self) -> str:
        if not self.spatial:
            return f"{self.statistique}:W{self.fenetre}"
        return (f"{self.statistique}:R{self.rayon}:L{self.rayon_echelle}:W{self.fenetre}"
                f"{':site' if self.sites else ''}")

    def seuils(self) -> dict:
        """Cles `derive_*` du JSON des seuils (lues par `alerts.MoteurAlertes`)."""
        out = {"derive_statistique": self.statistique, "derive_fenetre": int(self.fenetre),
               "derive_min_instants": self.min_instants}
        if self.spatial:
            out.update({"derive_rayon": int(self.rayon),
                        "derive_rayon_echelle": int(self.rayon_echelle),
                        "derive_sites": fichier_sites(self.rayon) if self.sites else None})
        return out

    @classmethod
    def depuis_config(cls, acfg) -> "Detecteur":
        spatial = "spatiale" in composantes(acfg.derive_statistique)
        return cls(acfg.derive_statistique, int(acfg.derive_fenetre),
                   acfg.derive_rayon if spatial else None,
                   acfg.derive_rayon_echelle if spatial else None,
                   bool(spatial and acfg.derive_sites is not None))


def fichier_sites(rayon: int) -> str:
    return f"sites_R{int(rayon)}.npz"


def detecteurs(statistiques=STATISTIQUES, grille: dict | None = None) -> list[Detecteur]:
    """Candidats de la grille, dans l'ordre de preference a egalite : spatiale (aucune
    passe ; correction de site d'abord, fenetre courte d'abord), puis innovation /
    influence (cote^2 passes), puis combinee."""
    g = GRILLE_DETECTEURS if grille is None else grille
    out: list[Detecteur] = []
    ordre = [s for s in ("spatiale", "innovation", "influence", "combinee") if s in statistiques]
    for st in ordre:
        if st in ("spatiale", "combinee"):
            for site, w, r, ll in itertools.product(g["sites"], g["fenetre"], g["rayon"],
                                                    g["rayon_echelle"]):
                out.append(Detecteur(st, w, r, ll, site))
        else:
            out.extend(Detecteur(st, w) for w in g["fenetre"])
    return out


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


def cle_erreur(var: str, op: str, val: float) -> str:
    return f"{var}:{op}:{val:g}"


# ---------------------------------------------------------------------- sites
def ajuster_sites_entrainement(wf, stems: list[str], rayons, min_voisins: int = 3
                               ) -> dict[int, ModeleSites]:
    """Modeles de sites (un par rayon) sur des instants d'ENTRAINEMENT du cache. Refuse
    un instant d'un autre jeu : la validation et le test n'y entrent jamais."""
    from .cache import load_index

    idx = load_index(wf.cfg).set_index("stem")
    autres = [s for s in stems if idx.loc[s, "split"] != "train"]
    if autres:
        raise ValueError(f"modele de sites : instants hors entrainement {autres[:3]}")

    def instants():
        for s in stems:
            S, _, _, stamp = wf.surface_from_cache(s)
            yield S, heure_utc(stamp)

    info = {"jours": sorted({s.split("_")[0] for s in stems})}
    return {int(r): ajuster_sites(instants(), int(r), min_voisins=min_voisins, info=info)
            for r in rayons}


# ---------------------------------------------------------------------- simulation
def _serie(ecs: list, fenetre: int, mode: str) -> list[dict]:
    """Statistiques de persistance a chaque instant, avec la MEME classe que le service."""
    suivi = SuiviDerive(fenetre, mode)
    out = []
    for k, ec in enumerate(ecs):
        suivi.ajouter(f"{k:06d}", ec)
        out.append(suivi.statistiques())
    return out


def _compacte(ec):
    """Ecarts en float32, sans ce que la persistance ne relit pas (memoire)."""
    from .alerts import Ecarts
    f = (lambda x: None if x is None else np.asarray(x, np.float32))
    return Ecarts(f(ec.d), f(ec.a), f(ec.z), ec.echelle, ec.centre, f(ec.echelle_locale))


def simuler_jour(wf, stems: list[str], acfg, rng: np.random.Generator, n_derives: int,
                 n_figees: int, detecteurs_=None, plan=PLAN_DERIVES,
                 espacement: int | None = None, sites: dict | None = None,
                 par_instant: bool = False) -> dict:
    """Un scenario : injection, detecteurs instant par instant, resultats bruts.

    `acfg` : section `alerts` (cote, min_m_s, plancher, min_voisins, rho_source,
    ecart_extreme des entrees). `detecteurs_` : liste de `Detecteur` (defaut : celui de
    la config). `sites` : {rayon: ModeleSites} pour les detecteurs avec correction de site.
    Le seuil kappa n'intervient pas ici : on garde, par maille et par detecteur, le maximum
    dans le temps de la statistique de decision (zmax) ; kappa se choisit ensuite
    (`choisir_detecteur`) ou s'applique (`evaluer_an1`).
    """
    from .contract import DUREE_FIGEE_S, N_INSTANTS_FIGEE, VITESSE_CALME, valeurs_figees

    dets = [Detecteur.depuis_config(acfg)] if detecteurs_ is None else list(detecteurs_)
    sites = sites or {}
    for d in dets:
        if d.sites and d.rayon not in sites:
            raise ValueError(f"detecteur {d.cle} : modele de sites de rayon {d.rayon} absent")
    passes = sorted({c for d in dets for c in composantes(d.statistique)
                     if c in STATISTIQUES_PASSES})
    geom = wf.geom
    stems = list(stems)
    data = [wf.surface_from_cache(s) for s in stems]
    present = np.logical_and.reduce([d[0][7] > 0.5 for d in data])
    esp = espacement or acfg.derive_cote + 1           # protocole v1, inchange
    cells = choisir_capteurs(present, n_derives + n_figees, esp, rng)
    derives = [{"j": j, "i": i, "type": plan[k % len(plan)][0], "p": plan[k % len(plan)][1],
                "angle": float(rng.uniform(0, 2 * np.pi)), "ecart": 0.0, "n": 0}
               for k, (j, i) in enumerate(cells[:n_derives])]
    figees = [{"j": j, "i": i} for j, i in cells[n_derives:]]
    ref = data[0][0]
    S_inj, heures = [], []
    ec_passes = {st: [] for st in passes}
    series = {v: np.full((geom.ny * geom.nx, len(data)), np.nan)
              for v in ("wind_u_m_s", "wind_v_m_s", "temperature_c")}
    entrees = []
    phys = []
    temps = []
    st_mu, st_sd = wf.norm.stats["surf_mu"], wf.norm.stats["surf_sd"]
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
        S_inj.append(S2)
        heures.append(heure_utc(stamp))
        vol = wf.predict(S2, hour, doy, timestamp=stamp, reliability=False)
        if passes:
            ec = ecarts_capteurs(S2, sensor, geom, (vol.u, vol.v),
                                 lambda kept: wf._member(S2, kept, hour, doy),
                                 acfg.derive_cote, passes, acfg.derive_echelle_plancher_m_s)
            for st in passes:
                ec_passes[st].append(_compacte(ec[st]))
        for v, c in (("wind_u_m_s", 0), ("wind_v_m_s", 1), ("temperature_c", 3)):
            series[v][:, t] = np.where(sensor, S2[c], np.nan).ravel()
        temps.append(pd.Timestamp(stamp))
        # entrees : statistiques BRUTES de l'instant propre (fausses alertes) et des erreurs
        # connues (detection), toutes variables ; la regle s'applique ensuite
        propre = S[7] > 0.5
        kw = dict(ecart_extreme=acfg.entrees_ecart_extreme, variables=SURF_VARS)
        entrees.append({
            "propre": derive_entrees(S, propre, st_mu, st_sd, **kw),
            "erreurs": {cle_erreur(var, op, val): derive_entrees(
                erreur_entree(S, propre, var, op, val), propre, st_mu, st_sd, **kw)
                for var, op, val in ERREURS_ENTREES}})
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
    der = [{**d, "cellule": d["j"] * nx + d["i"], "ecart_moyen": d["ecart"] / max(d["n"], 1)}
           for d in derives]
    fgs = [{**f, "cellule": f["j"] * nx + f["i"]} for f in figees]
    defauts = np.array([d["cellule"] for d in der] + [f["cellule"] for f in fgs], int)
    est_defaut = np.zeros(geom.ny * geom.nx, bool)
    est_defaut[defauts] = True

    # detecteurs : series d'ecarts spatiaux calculees une fois par (rayon, echelle, site)
    spatiaux: dict[tuple, list] = {}
    bases: dict[int, list] = {}
    res_det = {}
    for det in dets:
        series_ec = {}
        for c in composantes(det.statistique):
            if c == "spatiale":
                if det.rayon not in bases:              # une mediane par rayon et instant
                    bases[det.rayon] = [ecart_voisins(S2, S2[7] > 0.5, det.rayon,
                                                      acfg.derive_min_voisins)
                                        for S2 in S_inj]
                k = (det.rayon, det.rayon_echelle, det.sites)
                if k not in spatiaux:
                    spatiaux[k] = [_compacte(ecarts_spatiaux(
                        S2, S2[7] > 0.5, det.rayon, det.rayon_echelle,
                        acfg.derive_echelle_plancher_m_s, acfg.derive_min_voisins,
                        sites.get(det.rayon) if det.sites else None, h, base))
                        for S2, h, base in zip(S_inj, heures, bases[det.rayon])]
                series_ec[c] = spatiaux[k]
            else:
                series_ec[c] = ec_passes[c]
        stats = {c: _serie(e, det.fenetre, mode_suivi(c)) for c, e in series_ec.items()}
        c0 = composantes(det.statistique)[0]
        vus = np.zeros(geom.ny * geom.nx, bool)
        for ec in series_ec[c0]:
            vus |= np.isfinite(ec.z).ravel()
        sains = vus & ~est_defaut
        zmax = np.full(geom.ny * geom.nx, -np.inf)
        inst = []
        for t in range(len(S_inj)):
            st = (combiner(stats["spatiale"][t], stats["innovation"][t])
                  if det.statistique == "combinee" else stats[c0][t])
            with np.errstate(invalid="ignore"):
                elig = (st["n"] >= det.min_instants) & (st["a"] >= acfg.derive_min_m_s)
            zt = np.where(elig, st["z"], -np.inf).ravel()
            zmax = np.fmax(zmax, zt)
            if par_instant:
                inst.append(zt[sains & (zt >= Z_MIN)].astype(np.float32))
        res_det[det.cle] = {"defauts": zmax[defauts].astype(float),
                            "sains_hauts": zmax[sains & (zmax >= Z_MIN)].astype(np.float32),
                            "n_sains": int(sains.sum()), "par_instant": inst,
                            "detecteur": det}
    return {"stems": stems, "detecteurs": res_det, "figee_signalee": fig[defauts],
            "figee_signalee_sains": int(fig[~est_defaut].sum()),
            "derives": der, "figees": fgs, "entrees": entrees, "physique": phys}


def _taux(a: int, b: int) -> float | None:
    return a / b if b else None


def evaluer_an1(jours: list[dict], cle: str, kappa: float, e_ref: float = 1.0) -> dict:
    """Detection des derives, fausses alertes et verdict AN1 (capteurs) pour un detecteur
    (`Detecteur.cle`) et un kappa.

    Signale = alerte capteur_derive (zmax >= kappa). Les pannes (valeurs figees, regle M3)
    sont rapportees a part, hors verdict ; elles ne comptent ni comme saines ni comme
    fausses alertes."""
    det_ref = n_ref = fp = n_sains = signales_def = 0
    det_fig = det_fig_m3 = n_fig = fp_m3 = n_instants = fa_instants = 0
    par_type: dict[str, list[int]] = {}
    par_classe: dict[str, list[int]] = {}
    for jour in jours:
        r = jour["detecteurs"][cle]
        flag = r["defauts"] >= kappa                       # derives puis pannes
        nd = len(jour["derives"])
        for d, ok in zip(jour["derives"], flag[:nd]):
            ok = bool(ok)
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
        for k, f in enumerate(jour["figees"]):
            m3 = bool(jour["figee_signalee"][nd + k])
            det_fig += bool(flag[nd + k] or m3)
            det_fig_m3 += m3
            n_fig += 1
        fp += int((r["sains_hauts"] >= kappa).sum())
        fp_m3 += int(jour["figee_signalee_sains"])
        n_sains += r["n_sains"]
        signales_def += int(flag.sum())
        for z in r["par_instant"]:
            n_instants += 1
            fa_instants += int((z >= kappa).sum())
    detection = _taux(det_ref, n_ref)
    fausses = _taux(fp, fp + signales_def)
    if fausses is None:
        fausses = 0.0
    ok = detection is not None and detection >= DETECTION_MIN and fausses < FAUSSES_MAX
    return {"detecteur": cle, "kappa": float(kappa), "e_ref_m_s": float(e_ref),
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


def choisir_kappa(jours: list[dict], cle: str, grille=GRILLE_KAPPA, fdr_cible: float = 0.025,
                  e_ref: float = 1.0) -> dict:
    """Plus petit kappa (meilleure detection) dont la part de fausses alertes reste sous
    `fdr_cible` SUR LA VALIDATION (marge sous les 5 % de AN1). A n'appeler que sur la
    validation."""
    essais = [evaluer_an1(jours, cle, k, e_ref) for k in grille]
    bons = [e for e in essais if e["fausses_alertes"] <= fdr_cible]
    choix = bons[0] if bons else essais[-1]
    return {"detecteur": cle, "derive_kappa": choix["kappa"], "cible_atteinte": bool(bons),
            "validation": choix,
            "courbe": [{"kappa": e["kappa"], "detection": e["detection"],
                        "fausses_alertes": e["fausses_alertes"]} for e in essais]}


def choisir_detecteur(jours: list[dict], fdr_cible: float = 0.025, e_ref: float = 1.0,
                      grille_kappa=GRILLE_KAPPA) -> dict:
    """Detecteur et kappa retenus SUR LA VALIDATION : pour chaque detecteur simule, le plus
    petit kappa sous `fdr_cible` ; on garde la meilleure detection (a egalite, la plus
    faible part de fausses alertes, puis le detecteur le plus simple : ordre de la grille).
    Le resultat porte « jeu: validation » (`alerts.charger_seuils` refuse tout autre jeu)."""
    cles = list(jours[0]["detecteurs"])
    choix = {c: choisir_kappa(jours, c, grille_kappa, fdr_cible, e_ref) for c in cles}

    def rang(c):
        v = choix[c]["validation"]
        return (choix[c]["cible_atteinte"], v["detection"] or 0.0, -v["fausses_alertes"],
                -cles.index(c))

    best = max(cles, key=rang)
    det = jours[0]["detecteurs"][best]["detecteur"]
    b = choix[best]
    return {"jeu": "validation", "methode": METHOD, **det.seuils(),
            "derive_kappa": b["derive_kappa"], "detecteur": best, "fdr_cible": fdr_cible,
            "e_ref_m_s": e_ref, "cible_atteinte": b["cible_atteinte"],
            "validation": b["validation"], "courbe": b["courbe"],
            "alternatives": {c: {"kappa": x["derive_kappa"],
                                 "cible_atteinte": x["cible_atteinte"],
                                 "detection": x["validation"]["detection"],
                                 "fausses_alertes": x["validation"]["fausses_alertes"]}
                             for c, x in choix.items()}}


# ---------------------------------------------------------------------- entrees
def evaluer_entrees(jours: list[dict], regle: dict | None = None) -> dict:
    """Detection des erreurs d'unite connues et fausses alertes (instants propres) pour une
    regle {z_max, part_extreme, variables} ; defaut : la regle par defaut de la config."""
    if regle is None:
        from .config import AlertsCfg
        a = AlertsCfg()
        regle = {"z_max": a.entrees_z_max, "part_extreme": a.entrees_part_extreme,
                 "variables": a.entrees_variables}
    var = tuple(regle.get("variables", VARIABLES_ENTREES))
    zx, pe = float(regle["z_max"]), float(regle["part_extreme"])
    n = fa = 0
    det: dict[str, int] = {}
    fa_var: dict[str, int] = {}
    for j in jours:
        for inst in j["entrees"]:
            n += 1
            flags = regle_entrees(inst["propre"], zx, pe, var)
            fa += bool(flags)
            for v in flags:
                fa_var[v] = fa_var.get(v, 0) + 1
            for k, stats in inst["erreurs"].items():
                det[k] = det.get(k, 0) + int(k.split(":")[0] in regle_entrees(stats, zx, pe,
                                                                               var))
    total = sum(det.values())
    detection = _taux(total, n * len(det)) if det else None
    fausses = _taux(fa, n)
    ok = (detection is not None and detection >= DETECTION_MIN and fausses is not None
          and fausses < FAUSSES_MAX)
    return {"regle": {"z_max": zx, "part_extreme": pe, "variables": list(var)},
            "instants": n, "detection": detection,
            "detection_par_erreur": {k: _taux(x, n) for k, x in det.items()},
            "fausses_alertes": fausses, "fausses_par_variable": fa_var,
            "an1_entrees_conforme": bool(ok)}


def choisir_regle_entrees(jours: list[dict], fdr_cible: float = 0.025,
                          grille: dict | None = None) -> dict:
    """Regle des entrees retenue SUR LA VALIDATION (grille fixee d'avance, vent exclu) :
    fausses alertes <= `fdr_cible`, puis la meilleure detection ; a egalite, les seuils les
    plus hauts (la plus grande marge contre la meteo). Rejoue aussi la regle v1."""
    g = GRILLE_ENTREES if grille is None else grille
    essais = [evaluer_entrees(jours, {"z_max": z, "part_extreme": p,
                                      "variables": VARIABLES_ENTREES})
              for z, p in itertools.product(g["z_max"], g["part_extreme"])]
    bons = [e for e in essais if e["fausses_alertes"] is not None
            and e["fausses_alertes"] <= fdr_cible]
    pool = bons or essais
    best = max(pool, key=lambda e: (e["detection"] or 0.0, -(e["fausses_alertes"] or 0.0),
                                    e["regle"]["z_max"], e["regle"]["part_extreme"]))
    return {"jeu": "validation", "entrees_z_max": best["regle"]["z_max"],
            "entrees_part_extreme": best["regle"]["part_extreme"],
            "entrees_variables": list(VARIABLES_ENTREES), "cible_atteinte": bool(bons),
            "validation": best, "regle_v1_validation": evaluer_entrees(jours, REGLE_ENTREES_V1),
            "courbe": [{**e["regle"], "detection": e["detection"],
                        "fausses_alertes": e["fausses_alertes"]} for e in essais]}


__all__ = ["DETECTION_MIN", "FAUSSES_MAX", "PROTOCOLE", "Detecteur", "detecteurs",
           "simuler_jour", "evaluer_an1", "choisir_kappa", "choisir_detecteur",
           "evaluer_entrees", "choisir_regle_entrees", "ajuster_sites_entrainement",
           "iso_utc"]
