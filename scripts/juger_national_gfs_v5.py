"""Regle configs/national/regle_national_gfs_v5_2.json (approuvee par Ines le 23/11/2026, hachee 542e676c... ; remplace
la v5.1, arretee sans verdict le 23/11 par son controle d'echeances, et la v5 jamais executee) : etape 4, le jugement UNIQUE du candidat retenu sur la validation 2021
(scripts/choisir_national_gfs_v5.py), sur l'ECHANTILLON 2026 fixe par la regle : du 2026-01-01 au 2026-10-31, jours
de l'annee impairs, 00, 06, 12 et 18 UTC (heures de cycle : l'analyse est toujours f000), ~608 instants.

Entree construite EXACTEMENT comme le service temps reel (`pcfno3d.temps_reel`) : champs GFS de l'echeance lus dans
l'archive NOAA telechargee (`pcfno3d.gfs`, boite nationale `*_tn.npz`, `gfs.lire_national`), forcage par
`temps_reel.forcage_gfs` / `ForcageDirect`, surface et reseau a 10 m par `temps_reel.entree`, METAR des 11 AUTRES
aeroports a l'analyse (`mesures` a 30 min au plus), aucun aux echeances +24 h / +48 h ; heure et jour de l'annee de
l'instant UTC comme `atom_wind.jour_direct`. Il n'y a pas de cache pour 2026.

Deux commandes, dans cet ordre :

    controle : sur les instants de VALIDATION 2021, AVANT toute lecture de 2026 : les parties VENT de l'entree de
               production (vecteur de forcage : scalaires et cellules ; reseau GFS 10 m des pseudo-capteurs ; carte et
               valeurs des stations) doivent etre identiques a 1e-5 pres a celles du chemin par le cache de gfs_v4 ;
               sinon ARRET. Les canaux de surface T, rho, HR, p et les 4 traits geostrophique / vent thermique sont
               ATTENDUS differents (verite generee contre GFS) : ecarts moyens et max rapportes, sans arret.
    juger    : seulement si la selection a retenu une variante ET si le controle est passe ; lit 2026 une fois.

    python scripts/juger_national_gfs_v5.py controle --cp national_v4_eval --gfs <forcages f0/f24/f48 val> \
        --tn <dossier *_tn.npz> [--tn ...] --obs <METAR 2021> --sortie <resultat>
    python scripts/juger_national_gfs_v5.py juger --cp national_v4_eval --c1 national_v5_gfs_mixte_colab \
        --selection <validation>/selection.json --tn <dossiers *_tn.npz 2026> --obs <METAR 2026 IEM> \
        --sortie <resultat> [--sans-cp-s44]

Sorties (dossier neuf, jamais ecrase) : controle_entree.json, paires.csv, bilan.csv, verdicts_gfs_v5.json (empreinte de
la v5.1 ; lu par le service : pcfno3d.statut_service, variable PCFNO3D_VERDICTS_GFS_V5) et gfs_v5.md. Sources : le
candidat retenu (juge), GFS brut (reference), C' s44 + GFS sans correction (rapporte sans verdict ; pour C1 / C3 il
demande un second modele : `--sans-cp-s44` le saute faute de budget, note « non calcule faute de budget »). Seuls les
METAR sont des mesures.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]
_spec = importlib.util.spec_from_file_location("choisir_gfs_v5", REPO / "scripts" / "choisir_national_gfs_v5.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

REGLE = C.REGLE
ECHEANCES, NOM_ECHEANCE, CASES, ECHEANCE_CASE = C.ECHEANCES, C.NOM_ECHEANCE, C.CASES, C.ECHEANCE_CASE
TIRAGES, GRAINE, ABSENTS_MAX = C.TIRAGES, C.GRAINE, C.ABSENTS_MAX
# echantillon de jugement (regle v5.1, « periodes.jugement ») : jours de l'annee impairs, heures de cycle GFS
DEBUT_2026, FIN_2026, HEURES_2026 = "2026-01-01T00:00", "2026-10-31T18:00", (0, 6, 12, 18)
TOLERANCE = 1e-5                               # « ecart max <= 1e-5 » (parties vent)
NON_CALCULE = "non calcule faute de budget"
# parties VENT de l'entree, qui doivent etre identiques ; le reste est rapporte (attendu different)
VENT = ("capteurs_differents", "forcage_scalaires", "forcage_cellules", "reseau_gfs_10m", "stations_valeurs")


def echantillon_2026() -> list[pd.Timestamp]:
    """Les instants juges : 2026-01-01 -> 2026-10-31, jours de l'annee impairs, 00 / 06 / 12 / 18 UTC (608)."""
    from pcfno3d.gfs import instants_echantillon
    return instants_echantillon(DEBUT_2026, FIN_2026, HEURES_2026, jours_impairs=True)
SORTIES = ("verdicts_gfs_v5.json", "paires.csv", "arret.json")


# ---------------------------------------------------------------------- entree de production (temps_reel)
def forcage_production(geom, instants, echeance_h: int, dossiers) -> tuple[dict, object, dict, list]:
    """Champs GFS de l'archive et forcage `ForcageDirect`, comme le service : (champs, forcage, {instant s : indice},
    instants absents de l'archive)."""
    from pcfno3d.gfs import lire_national
    from pcfno3d.temps_reel import ForcageDirect, forcage_gfs

    c, _, absents = lire_national(instants, echeance_h, geom.forcage.lat, geom.forcage.lon, dossiers)
    temps, scal, u, v = forcage_gfs(c, geom.forcage.hauteurs)
    f = ForcageDirect(temps, scal, u, v, geom.forcage)
    return c, f, {int(t): i for i, t in enumerate(c["temps"])}, absents


def secondes(t: pd.Timestamp) -> int:
    return int(pd.Timestamp(t).value // 10 ** 9)


def entree_production(geom, c, f, it: int, t: pd.Timestamp, mesures: dict, exclue=None):
    """(S, vecteur, heure, jour de l'annee) de l'instant t : `temps_reel.entree` et `ForcageDirect.vecteur`, l'heure
    UTC decimale et le jour de l'annee comme `atom_wind.jour_direct`."""
    from pcfno3d.temps_reel import entree
    S = entree(geom, c, f, it, mesures, exclue=exclue)
    return S, f.vecteur(t), t.hour + t.minute / 60, int(t.dayofyear)


def gfs_10m_production(geom, f, it: int) -> np.ndarray:
    """GFS 10 m aux colonnes, comme le fond du service (`temps_reel._bil` de la premiere hauteur, 10 m)."""
    from pcfno3d.temps_reel import _bil
    return np.stack([_bil(f.u[it, 0], geom.interp_era5), _bil(f.v[it, 0], geom.interp_era5)]).astype(np.float32)


# ---------------------------------------------------------------------- controle prealable (validation 2021)
def comparer(geom, norm, a: tuple, b: tuple, cellules_st: dict) -> dict:
    """Ecarts entre deux entrees (S, vecteur, heure, jour), partie par partie : {nom : (moyenne, max)} des ecarts
    absolus. VENT : carte des capteurs (nombre de cellules differentes), scalaires et cellules du forcage, vent 10 m
    (u, v, w) aux pseudo-capteurs du reseau GFS et aux stations. Rapporte : surface T, rho, HR, p (normalisee comme
    le modele la recoit, et brute), 4 traits geostrophique / vent thermique, 4 traits temps, heure, jour."""
    from pcfno3d.forcage import N_SCAL
    from pcfno3d.geometry import N_SURF, N_TFEAT
    from pcfno3d.inference import caracteristiques

    (Sa, va, ha, da), (Sb, vb, hb, db) = a, b
    ma, mb = Sa[7] > 0.5, Sb[7] > 0.5
    st = np.zeros(ma.shape, bool)
    for iy, ix in cellules_st.values():
        st[iy, ix] = True
    m = ma & mb
    mu, sd = (np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, float).reshape(-1)
              for x in (norm.surf_mu, norm.surf_sd))
    ta = caracteristiques(Sa, ma, ha, da, geom, va)
    tb = caracteristiques(Sb, mb, hb, db, geom, vb)

    def ecart(x, y, masque=None):
        d = np.abs(np.asarray(x, float) - np.asarray(y, float))
        if masque is not None:
            d = d[..., masque]
        d = d.ravel()
        return (float(d.mean()), float(d.max())) if d.size else (0.0, 0.0)

    n_diff = float((ma != mb).sum())
    out = {"capteurs_differents": (n_diff, n_diff),
           "forcage_scalaires": ecart(ta[N_TFEAT:N_TFEAT + N_SCAL], tb[N_TFEAT:N_TFEAT + N_SCAL]),
           "forcage_cellules": ecart(ta[N_TFEAT + N_SCAL:], tb[N_TFEAT + N_SCAL:]),
           "reseau_gfs_10m": ecart(Sa[:3], Sb[:3], m & ~st),
           "stations_valeurs": ecart(Sa[:3], Sb[:3], m & st)}
    for k, nom in zip(range(3, N_SURF), ("temperature", "densite", "humidite", "pression")):
        out[f"surface_{nom}_normalisee"] = ecart((Sa[k] - mu[k]) / sd[k], (Sb[k] - mu[k]) / sd[k], m)
        out[f"surface_{nom}_brute"] = ecart(Sa[k], Sb[k], m)
    out["traits_geostrophique_vent_thermique_4"] = ecart(ta[4:8], tb[4:8])
    out["traits_temps_4"] = ecart(ta[:4], tb[:4])
    out["heure"] = ecart(ha, hb)
    out["jour_annee"] = ecart(da, db)
    return out


def controler(cp_cfg, dossier_gfs: Path, dossiers_tn, dossier_obs: Path, sortie: Path, verbose: bool = True) -> dict:
    """Controle prealable de la regle (etape 4) sur les instants de VALIDATION 2021 : entree de production contre
    entree par le cache de gfs_v4, a chaque echeance, sans aeroport retire et avec chacun retire a l'analyse."""
    from types import SimpleNamespace

    from pcfno3d.forcage import ForcageERA5
    from pcfno3d.geometry import Geometry
    from pcfno3d.normalize import Normalizer
    from pcfno3d.stations_reelles import cellules_codes, entree_reelle, lire_observations, obs_a

    sha = C.verifier_regle()
    sortie = Path(sortie)
    if (sortie / "controle_entree.json").exists():
        raise SystemExit(f"{sortie / 'controle_entree.json'} existe deja : rien n'est recalcule ni ecrase")
    forcages = {e: Path(dossier_gfs) / f"forcage_gfs_f{e}.csv" for e in ECHEANCES}
    stems, row = C.stems_validation(cp_cfg)
    presents = {e: C.temps_du_forcage(f) for e, f in forcages.items()}
    stems = [s for s in stems if all(C.instant_utc(row, s) in presents[e] for e in ECHEANCES)]
    cfg = C.config_eval(cp_cfg, forcages[0])
    geom = Geometry(cfg)
    norm = Normalizer.from_file(cfg.paths.norm_stats, geom, cfg.loss.w_level_max)
    cellules = cellules_codes(SimpleNamespace(cfg=cfg))
    obs = lire_observations(Path(dossier_obs), verbose=verbose, codes=list(cellules))
    somme: dict[str, float] = {}
    pire: dict[str, float] = {}
    n, absents, t0 = 0, {}, time.time()
    for e in ECHEANCES:
        F = ForcageERA5(forcages[e])
        instants = [C.instant_utc(row, s) for s in stems]
        c, f, index_t, manque = forcage_production(geom, instants, e, dossiers_tn)
        absents[NOM_ECHEANCE[e]] = [str(t) for t in manque]
        for stem, t in zip(stems, instants):
            it = index_t.get(secondes(t))
            if it is None:
                continue
            stamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")
            mesures = {k: m for k, m in obs_a(obs, t).items() if k in cellules} if e == 0 else {}
            S = np.load(cfg.paths.cache_dir / f"{stem}_S.npy")
            E = F.e_10m(stamp, geom.interp_era5)
            vec = F.vecteur(stamp)
            r = row.loc[stem]
            for exclue in [None] + sorted(mesures):
                S_in, cap = entree_reelle(S, E, geom, cellules, mesures, exclue=exclue)
                S_in[7] = cap
                cache = (S_in, vec, float(r["hour"]), int(r["doy"]))
                prod = entree_production(geom, c, f, it, t, mesures, exclue)
                for k, (moy, mx) in comparer(geom, norm, cache, prod, cellules).items():
                    somme[k] = somme.get(k, 0.0) + moy
                    pire[k] = max(pire.get(k, 0.0), mx)
                n += 1
        if verbose:
            print(f"  controle {NOM_ECHEANCE[e]} : {n} entrees comparees ({time.time() - t0:.0f} s)", flush=True)
    vent = {k: {"moyenne": somme[k] / n, "max": pire[k]} for k in VENT} if n else {}
    autres = {k: {"moyenne": somme[k] / n, "max": pire[k]} for k in pire if k not in VENT} if n else {}
    passe = bool(n > 0 and vent["capteurs_differents"]["max"] == 0 and all(
        v["max"] <= TOLERANCE for k, v in vent.items() if k != "capteurs_differents"))
    out = {"regle": REGLE.name, "regle_sha256": sha, "tolerance": TOLERANCE, "entrees_comparees": n,
           "instants_validation": len(stems), "parties_vent": vent, "rapporte_sans_arret": autres,
           "note": "surface T, rho, HR, p et traits geostrophique / vent thermique : ATTENDUS differents (cache = "
                   "verite generee, production = GFS ramene au relief) ; rapportes sans arret (regle v5.1)",
           "absents_archive": absents, "passe": passe,
           "consequence": ("parties vent identiques a celles du cache : le jugement 2026 peut etre lu" if passe
                           else "ARRET (regle v5.1, etape 4) : les parties vent de l'entree de production different "
                                "de celles du chemin par le cache ; la periode 2026 n'est pas lue"),
           "date_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "controle_entree.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("passe", "entrees_comparees", "parties_vent", "rapporte_sans_arret")},
                     indent=1), flush=True)
    return out


# ---------------------------------------------------------------------- jugement 2026
def modeles_requis(famille: str, avec_cp_s44: bool = True) -> tuple[str, ...]:
    """Modeles a faire tourner : le candidat (C2 : C' s44 ancre ; C1 ; C3 : C1 ancre) et, si le budget le permet,
    C' s44 sans correction (rapporte sans verdict ; gratuit pour C2, qui le fait tourner de toute facon)."""
    if famille == "c2":
        return ("cp",)
    return ("c1", "cp") if avec_cp_s44 else ("c1",)


def passes_production(base, instants, dossiers_tn, obs, checkpoint=None, device=None, ancrage=None,
                      verbose: bool = True, nom: str = "") -> dict:
    """Passes du modele de `base` sur l'entree de PRODUCTION, mois par mois : {(echeance, instant, code) : (u10, v10,
    u10 ancre, v10 ancre, GFS brut u, v, mesure u, v)}. `ancrage` : (a0, H) ou None."""
    from types import SimpleNamespace

    from pcfno3d.ancrage_gfs import ancrer, colonne_gfs, hauteurs_noeuds
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.stations_reelles import cellules_codes, obs_a

    cfg = C.config_eval(base, base.paths.forcage_era5)
    wf = WindFieldModel(cfg, checkpoint=checkpoint, device=device, verbose=False)
    geom = wf.geom
    cellules = cellules_codes(SimpleNamespace(cfg=cfg))
    h_noeuds = hauteurs_noeuds(geom)
    out, t0, n = {}, time.time(), 0
    mois = pd.Series(instants).dt.to_period("M")
    for e in ECHEANCES:
        for _, groupe in pd.Series(instants).groupby(mois.values):
            ts = list(groupe)
            c, f, index_t, _ = forcage_production(geom, ts, e, dossiers_tn)
            for t in ts:
                it = index_t.get(secondes(t))
                mesures = {k: m for k, m in obs_a(obs, t).items() if k in cellules}
                if it is None or not mesures:
                    continue
                stamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")
                E = gfs_10m_production(geom, f, it)

                def passe(m, exclue=None):
                    S, vec, h, d = entree_production(geom, c, f, it, t, m, exclue)
                    return wf.predict(S, h, d, timestamp=stamp, reliability=False, tta=1, forcage=vec), vec

                seul = passe({}) if e else None              # heures a venir : aucun aeroport en entree
                for code, (uo, vo) in mesures.items():
                    iy, ix = cellules[code]
                    vol, vec = passe(mesures, exclue=code) if e == 0 else seul
                    cu, cv = vol.u[:, iy, ix].astype(float), vol.v[:, iy, ix].astype(float)
                    hh, z0 = h_noeuds[:, iy, ix], float(geom.z0[iy, ix])
                    u10, v10 = C.vent_colonne(cu, cv, hh, z0)
                    ua = va = math.nan
                    if ancrage is not None:
                        g = colonne_gfs(geom, vec, iy, ix)
                        au, av = ancrer(cu, cv, g[0], g[1], hh, *ancrage)
                        ua, va = C.vent_colonne(au, av, hh, z0)
                    out[(e, t, code)] = (u10, v10, ua, va, float(E[0, iy, ix]), float(E[1, iy, ix]), uo, vo)
                n += 1
                if verbose and n % 200 == 0:
                    print(f"  {nom} : {n} instants x echeances ({time.time() - t0:.0f} s)", flush=True)
    wf = None                                           # libere la memoire du GPU
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:                                   # noqa: BLE001
        pass
    return out


def paires_2026(passes: dict, famille: str, noms: dict) -> pd.DataFrame:
    """Paires du jugement : candidat (juge), gfs_brut (reference), cp_s44_gfs (rapporte sans verdict)."""
    cand_modele = "cp" if famille == "c2" else "c1"
    ancre = famille in ("c2", "c3")
    lignes = []
    avec_cp = "cp" in passes
    for (e, t, code), x in passes[cand_modele].items():
        u10, v10, ua, va, ug, vg, uo, vo = x
        cp = passes["cp"].get((e, t, code)) if avec_cp else (math.nan,) * 8
        if cp is None:
            raise ValueError(f"C' s44 sans la paire {(e, t, code)}")
        if avec_cp and (abs(cp[4] - ug) > 1e-6 or cp[6] != uo):
            raise ValueError("GFS brut ou mesure differents entre les modeles a une paire")
        lignes.append({"station": noms.get(code, code), "code": code, "temps": t, "jour": t.strftime("%Y-%m-%d"),
                       "echeance_h": e, "u_obs": uo, "v_obs": vo, "u_gfs_brut": ug, "v_gfs_brut": vg,
                       "u_candidat": ua if ancre else u10, "v_candidat": va if ancre else v10,
                       "u_cp_s44_gfs": cp[0], "v_cp_s44_gfs": cp[1]})
    return pd.DataFrame(lignes)


def juger_paires(p: pd.DataFrame, stations: list, sha: str, selection: dict) -> dict:
    """Les 3 cases par le critere de gfs_v4 (fonctions appelees telles quelles), memes tirages des jours partout ;
    C' s44 + GFS sans correction rapporte par la meme metrique, sans verdict."""
    J = C.critere_v4()
    jours = sorted(p["jour"].unique())
    w = J.tirages_jours(jours, TIRAGES, GRAINE)
    details, rapportes = {}, {}
    for case in CASES:
        q = p[p["echeance_h"] == ECHEANCE_CASE[case]]
        details[case] = J.juger_case(q, "candidat", w, jours, stations)
        if not q["u_cp_s44_gfs"].notna().any():
            rapportes[case] = NON_CALCULE
            continue
        r = J.juger_case(q, "cp_s44_gfs", w, jours, stations)
        r.pop("verdict", None)
        r.pop("raison", None)
        rapportes[case] = r
    return {"regle": REGLE.name, "regle_sha256": sha, "cases": {k: v["verdict"] for k, v in details.items()},
            "candidat": selection.get("variante"), "details": details,
            "rapporte_sans_verdict_cp_s44_gfs": rapportes, "jours": len(jours), "aeroports": stations}


def rapport(v: dict, b: pd.DataFrame, info: dict) -> str:
    L = ["# Règle `regle_national_gfs_v5_2` : jugement 2026 aux 12 aéroports", "",
         f"Règle `configs/national/regle_national_gfs_v5_2.json` (sha256 `{v['regle_sha256'][:12]}…`), approuvée par "
         "Ines le 23/11/2026, vérifiée avant le calcul. Candidat retenu sur la validation 2021 seulement : "
         f"`{(v.get('candidat') or {}).get('nom')}`. Échantillon fixé par la règle : 2026-01-01 → 2026-10-31, jours "
         "impairs, 00 / 06 / 12 / 18 UTC (analyse toujours f000 ; les heures 03 / 09 / 15 / 21 du service ne sont pas "
         "jugées), "
         "jamais utilisée, **lue une seule fois**. Entrée construite comme le service temps réel (archive GFS NOAA, "
         "METAR IEM des 11 autres aéroports à l'analyse, aucun aux échéances). Les seules mesures sont les METAR.", "",
         f"Instants : {info['instants']} sur {info['instants_attendus']} ({info['absents']} écartés pour toutes les "
         f"sources faute d'échéance GFS) ; {v['jours']} jours ; IC 95 % par {TIRAGES} tirages des jours (graine "
         f"{GRAINE}).", "",
         "| Case | Moyenne RMSE candidat | Moyenne RMSE GFS brut | Écart | IC 95 % | Aéroports > +5 % | Verdict |",
         "|---|---:|---:|---:|---|---|---|"]
    for case, d in v["details"].items():
        L.append(f"| {case} | {d['moyenne_rmse_source']:.3f} | {d['moyenne_rmse_gfs_brut']:.3f} | {d['ecart']:+.3f} | "
                 f"[{d['ic95'][0]:+.3f} ; {d['ic95'][1]:+.3f}] | "
                 f"{', '.join(d['aeroports_degrades_plus_5_pct']) or 'aucun'} | **{d['verdict']}** |")
    L += ["", "Rapporté sans verdict : C' s44 + GFS sans correction.", "",
          "| Case | Moyenne RMSE C' s44 + GFS | Écart à GFS brut | IC 95 % |", "|---|---:|---:|---|"]
    for case, d in v["rapporte_sans_verdict_cp_s44_gfs"].items():
        if isinstance(d, str):
            L.append(f"| {case} | {d} | | |")
            continue
        L.append(f"| {case} | {d['moyenne_rmse_source']:.3f} | {d['ecart']:+.3f} | "
                 f"[{d['ic95'][0]:+.3f} ; {d['ic95'][1]:+.3f}] |")
    L += ["", "## Par aéroport (m/s)", ""]
    t = b.pivot_table(index=["echeance", "station"], columns="source", values="rmse_vect").round(3)
    L.append(C.md_table(t.reset_index()))
    return "\n".join(L) + "\n"


def juger(cp_cfg, c1_cfg, selection_f: Path, dossiers_tn, dossier_obs: Path, sortie: Path, checkpoint_cp=None,
          checkpoint_c1=None, device=None, verbose: bool = True, avec_cp_s44: bool = True) -> dict:
    from types import SimpleNamespace

    from pcfno3d.gfs import fichier_national, trouver_national
    from pcfno3d.stations_reelles import cellules_codes, lire_observations, noms_stations

    sha = C.verifier_regle()
    sortie = Path(sortie)
    deja = [n for n in SORTIES if (sortie / n).exists()]
    if deja:
        raise SystemExit(f"{sortie} contient deja {deja} : une seule lecture de 2026, rien n'est ecrase")
    sel = json.loads(Path(selection_f).read_text(encoding="utf-8"))
    if sel.get("regle_sha256") != sha:
        raise SystemExit("selection.json ne vient pas de cette regle (empreinte differente)")
    if sel.get("arret") or not sel.get("variante"):
        raise SystemExit("selection : ARRET (aucune variante ne passe sur 2021) ; 2026 n'est pas lu")
    ctrl_f = sortie / "controle_entree.json"
    ctrl = json.loads(ctrl_f.read_text(encoding="utf-8")) if ctrl_f.exists() else {}
    if ctrl.get("regle_sha256") != sha or ctrl.get("passe") is not True:
        raise SystemExit(f"{ctrl_f} absent ou non passe : controle prealable exige avant 2026 (regle, etape 4)")
    variante = sel["variante"]
    famille = variante["famille"]
    ancrage = (float(variante["a0"]), float(variante["H"])) if famille in ("c2", "c3") else None
    # instants de la periode presents aux TROIS echeances (absent a l'une = retire pour toutes les sources)
    instants = echantillon_2026()
    dossiers = [Path(d) for d in dossiers_tn]
    communs = [t for t in instants if all(trouver_national(t, e, dossiers) is not None for e in ECHEANCES)]
    par_ech = {NOM_ECHEANCE[e]: sum(trouver_national(t, e, dossiers) is None for t in instants) for e in ECHEANCES}
    info = {"instants_attendus": len(instants), "instants": len(communs), "absents": len(instants) - len(communs),
            "absents_liste": [str(t) for t in sorted(set(instants) - set(communs))][:200],
            "absents_par_echeance": par_ech,
            "fichier_exemple": str(fichier_national(dossiers[0], instants[0], 0)) if dossiers else None}
    sortie.mkdir(parents=True, exist_ok=True)
    print(f"2026 : {len(communs)} instants sur {len(instants)} ; echeances absentes {par_ech}", flush=True)
    # regle v5.2, controle_d_echeances (a) : au plus 2 % d'echeances absentes PAR ECHEANCE sur l'echantillon
    # telecharge, sinon arret sans verdict ; (b) les instants incomplets sont retires partout, sans arret
    trop = {k: n for k, n in par_ech.items() if n > ABSENTS_MAX * len(instants)}
    if trop or not communs:
        (sortie / "arret.json").write_text(json.dumps({**info, "regle_sha256": sha, "verdict": "arret sans verdict"},
                                                      indent=2, ensure_ascii=False), encoding="utf-8")
        raise SystemExit(f"plus de {ABSENTS_MAX:.0%} d'echeances GFS absentes ({trop}) : arret sans verdict (regle)")
    geo = SimpleNamespace(cfg=cp_cfg)
    cellules, noms = cellules_codes(geo), noms_stations(geo)
    obs = lire_observations(Path(dossier_obs), verbose=verbose, codes=list(cellules))      # SEULE lecture de 2026
    passes = {}
    for m in modeles_requis(famille, avec_cp_s44):
        base, ck = (cp_cfg, checkpoint_cp) if m == "cp" else (c1_cfg, checkpoint_c1)
        anc = ancrage if (m == "cp" and famille == "c2") or (m == "c1" and famille == "c3") else None
        passes[m] = passes_production(base, communs, dossiers, obs, ck, device, anc, verbose, m)
    p = paires_2026(passes, famille, noms)
    p.to_csv(sortie / "paires.csv", index=False)
    v = juger_paires(p, sorted(p["station"].unique()), sha, sel)
    v["instants"] = info
    v["controle_entree"] = {k: ctrl.get(k) for k in ("passe", "tolerance", "entrees_comparees")}
    v["cp_s44_gfs_calcule"] = "cp" in passes
    b = C.bilan(p, ["gfs_brut", "candidat"] + (["cp_s44_gfs"] if "cp" in passes else []))
    b.to_csv(sortie / "bilan.csv", index=False)
    (sortie / "verdicts_gfs_v5.json").write_text(json.dumps(v, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
    (sortie / "gfs_v5.md").write_text(rapport(v, b, info), encoding="utf-8")
    return v


def main() -> int:
    from pcfno3d.config import Config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="commande", required=True)
    a = sp.add_parser("controle", help="controle prealable sur la validation 2021 (avant toute lecture de 2026)")
    a.add_argument("--cp", required=True)
    a.add_argument("--gfs", required=True, help="dossier des forcage_gfs_f{0,24,48}.csv (validation)")
    a.add_argument("--tn", action="append", required=True, help="dossier des *_tn.npz (repetable)")
    a.add_argument("--obs", required=True, help="METAR 2021")
    a.add_argument("--sortie", required=True)
    b = sp.add_parser("juger", help="jugement unique de 2026")
    b.add_argument("--cp", required=True)
    b.add_argument("--c1", required=True)
    b.add_argument("--selection", required=True)
    b.add_argument("--tn", action="append", required=True, help="dossier des *_tn.npz 2026 (repetable)")
    b.add_argument("--obs", required=True, help="METAR 2026 (archive IEM, scripts/historique_metar.py)")
    b.add_argument("--sortie", required=True)
    b.add_argument("--checkpoint-cp")
    b.add_argument("--checkpoint-c1")
    b.add_argument("--device")
    b.add_argument("--sans-cp-s44", action="store_true",
                   help="ne pas faire tourner C' s44 sans correction (rapporte sans verdict) : note « non calcule "
                        "faute de budget » (sans effet pour C2)")
    x = ap.parse_args()
    if x.commande == "controle":
        r = controler(Config.load(x.cp), Path(x.gfs), x.tn, Path(x.obs), Path(x.sortie))
        return 0 if r["passe"] else 5
    v = juger(Config.load(x.cp), Config.load(x.c1), Path(x.selection), x.tn, Path(x.obs), Path(x.sortie),
              x.checkpoint_cp, x.checkpoint_c1, x.device, avec_cp_s44=not x.sans_cp_s44)
    print(json.dumps(v["cases"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
