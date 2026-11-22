"""Regle configs/national/regle_national_gfs_v5_1.json (approuvee par Ines le 22/11/2026 au soir, remplace la v5 jamais executee, hachee) : etapes 1 et 3, sur
la VALIDATION 2021 seulement (aucune lecture de 2022 ni de 2026).

Etape 1, diagnostic : C' s44 tel que livre, nourri par GFS (f0-f5, f24, f48), aux instants synoptiques de la
validation 2021, aux 12 aeroports retires a tour de role (entree de gfs_v4 : 11 autres aeroports a l'analyse, aucun
aux echeances) ; GFS brut a la cellule de l'aeroport. Rapporte : RMSE vectoriel par aeroport et par echeance ; a
Tabarka : biais de vitesse, erreur de direction, ecart par secteur de vent (8 secteurs), jour / nuit, saison, profil
du modele de 10 a 300 m. Rien n'est choisi a cette etape.

Etape 3, choix : les 13 variantes (C1 ; C2 = ancrage GFS de C' s44, 6 reglages ; C3 = C1 + ancrage, 6 reglages) x
3 cases (analyse, +24 h, +48 h), critere de regle_national_gfs_v4 recopie a l'identique (les fonctions de
scripts/juger_national_gfs_v4.py sont appelees telles quelles) : ACCEPTE si la borne haute de l'IC 95 % (bootstrap
des JOURS, 2000 tirages, graine 2030, memes tirages partout) de l'ecart moyenne(variante) - moyenne(gfs_brut) est
<= 0 ET aucun aeroport degrade de plus de 5 %. Retenue : parmi les variantes qui passent les TROIS cases, celle de
plus bas RMSE vectoriel moyen (moyenne des trois cases) ; ex aequo a 0,005 m/s pres : la plus simple (C2, puis C1,
puis C3). Si aucune ne passe : ARRET (selection.json « arret »), 2026 n'est pas lu.

    python scripts/choisir_national_gfs_v5.py --cp national_v4_eval --c1 national_v5_gfs_eval \
        --gfs <dossier des forcage_gfs_f{0,24,48}.csv (instants d'entrainement et de validation)> \
        --obs <METAR des 12 aeroports> --sortie <dossier NEUF>

Sorties : paires_validation.csv, bilan_validation.csv, diagnostic_tabarka.csv, profil_tabarka.csv,
variantes.json, selection.json, validation_gfs_v5.md. GFS et ERA5 ne sont jamais appeles mesures : seuls les METAR
le sont.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
REGLE = REPO / "configs" / "national" / "regle_national_gfs_v5_1.json"
# recopies de la regle (verifies par tests/test_gfs_v5_national.py contre son texte)
ECHEANCES = (0, 24, 48)
NOM_ECHEANCE = {0: "analyse", 24: "+24 h", 48: "+48 h"}
CASES = ("national 3D / analyse", "national 3D / +24 h", "national 3D / +48 h")
ECHEANCE_CASE = {CASES[0]: 0, CASES[1]: 24, CASES[2]: 48}
TIRAGES, GRAINE, DEGRADATION_MAX, ABSENTS_MAX = 2000, 2030, 0.05, 0.02
EX_AEQUO = 0.005
ORDRE_SIMPLICITE = ("c2", "c1", "c3")          # « la plus simple (C2, puis C1, puis C3) »
TABARKA = "DTKA"
HAUTEURS_PROFIL = (10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0)
SECTEURS = ("N", "NE", "E", "SE", "S", "SO", "O", "NO")


def verifier_regle(f: Path = REGLE) -> str:
    """Empreinte de la regle, comparee a son .sha256 AVANT tout calcul (comme juger_national_gfs_v4)."""
    h = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    attendu = (Path(f).parent / (Path(f).name + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"regle modifiee depuis son hachage : {h} != {attendu}")
    return h


def _module(nom: str, fichier: str):
    spec = importlib.util.spec_from_file_location(nom, REPO / "scripts" / fichier)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def critere_v4():
    """Le module de regle_national_gfs_v4 : metrique, tirages des jours et critere par case, appeles tels quels."""
    return _module("juger_gfs_v4", "juger_national_gfs_v4.py")


def variantes() -> list[dict]:
    """Les 13 variantes de l'etape 3 : C1, puis C2 (6 reglages), puis C3 (6 reglages)."""
    from pcfno3d.ancrage_gfs import nom_reglage, reglages
    out = [{"nom": "c1", "famille": "c1", "a0": None, "H": None}]
    for fam in ("c2", "c3"):
        out += [{"nom": nom_reglage(fam, a0, H), "famille": fam, "a0": a0, "H": H} for a0, H in reglages()]
    return out


# ---------------------------------------------------------------------- vent a 10 m et ancrage aux colonnes
def vent_colonne(col_u, col_v, h, z0, hauteur: float = 10.0) -> tuple[float, float]:
    from pcfno3d.stations_reelles import vent_a_hauteur
    return vent_a_hauteur(np.asarray(col_u, float), np.asarray(col_v, float), np.asarray(h, float), float(z0),
                          hauteur)


def vent_ancre(r: dict, a0: float, H: float, hauteur: float = 10.0) -> tuple[float, float]:
    """Vent a `hauteur` du champ ancre (C2) de la colonne de la ligne r (col_u, col_v modele ; col_gu, col_gv GFS
    aux noeuds ; h ; z0) : ancrage noeud par noeud, puis lecture a 10 m comme le DSO7."""
    from pcfno3d.ancrage_gfs import ancrer
    u, v = ancrer(r["col_u"], r["col_v"], r["col_gu"], r["col_gv"], r["h"], a0, H)
    return vent_colonne(u, v, r["h"], r["z0"], hauteur)


# ---------------------------------------------------------------------- passes du modele (chemin du cache)
def stems_validation(base) -> tuple[list[str], dict]:
    """Stems synoptiques de la VALIDATION (2021) et leurs lignes d'index (instant UTC, heure, jour de l'annee)."""
    from pcfno3d.cache import load_index
    from pcfno3d.dataset import is_synoptic, make_splits

    idx = load_index(base)
    sp = make_splits(base, idx)
    row = idx.set_index("stem")
    stems = [s for s in sp.val[0] if is_synoptic(sp.stamp_of[s])]
    return stems, row


def instant_utc(row, stem) -> pd.Timestamp:
    t = pd.Timestamp(row.loc[stem, "timestamp"])
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def temps_du_forcage(chemin: Path) -> set:
    return set(pd.to_datetime(pd.read_csv(chemin, usecols=["timestamp_utc"])["timestamp_utc"], utc=True)
               .dt.tz_localize(None))


def config_eval(base, forcage_geometrie: Path):
    """La configuration d'un modele pour les passes : forcage de la geometrie, reseau a 10 m tire du forcage, aucun
    tirage d'echeance (paths.forcages_echeances vide : l'echeance est imposee)."""
    cfg = copy.deepcopy(base)
    cfg.apply_overrides({"paths.forcage_era5": str(forcage_geometrie), "mask.e_source": "forcage"})
    cfg.paths.forcages_echeances = None
    return cfg


def colonnes_cache(base, forcages: dict, stems: list, row, obs: pd.DataFrame, checkpoint=None, device=None,
                   verbose: bool = True, nom: str = "") -> list[dict]:
    """Passes du modele de `base` par le chemin du CACHE (celui de gfs_v4 : _S du cache, reseau a 10 m et vecteur
    tires du forcage GFS de l'echeance, aeroport juge retire ; aux echeances futures aucun aeroport). Une ligne par
    (echeance, instant, aeroport mesure) : mesure METAR, GFS brut a 10 m, colonne du modele et colonne GFS aux
    noeuds (pour l'ancrage), hauteurs des noeuds, z0."""
    from pcfno3d.ancrage_gfs import colonne_gfs, hauteurs_noeuds
    from pcfno3d.forcage import ForcageERA5
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.stations_reelles import cellules_codes, entree_reelle, noms_stations, obs_a

    cfg = config_eval(base, forcages[0])
    wf = WindFieldModel(cfg, checkpoint=checkpoint, device=device, verbose=False)
    geom = wf.geom
    cellules, noms = cellules_codes(geom), noms_stations(geom)
    h_noeuds = hauteurs_noeuds(geom)
    cache = cfg.paths.cache_dir
    lignes, t0 = [], time.time()
    for e in ECHEANCES:
        F = ForcageERA5(forcages[e])
        for n, stem in enumerate(stems):
            r, t = row.loc[stem], instant_utc(row, stem)
            stamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")
            mesures = {c: m for c, m in obs_a(obs, t).items() if c in cellules}
            if not mesures:
                continue
            S = np.load(cache / f"{stem}_S.npy")
            E = F.e_10m(stamp, geom.interp_era5)          # GFS 10 m de l'echeance aux colonnes (= charger_e)
            vec = F.vecteur(stamp)

            def passe(m, exclue=None):
                S_in, cap = entree_reelle(S, E, geom, cellules, m, exclue=exclue)
                S_in[7] = cap
                return wf.predict(S_in, float(r["hour"]), int(r["doy"]), timestamp=stamp, reliability=False,
                                  tta=1, forcage=vec)

            seul = passe({}) if e else None
            for code, (uo, vo) in mesures.items():
                iy, ix = cellules[code]
                vol = passe(mesures, exclue=code) if e == 0 else seul
                gu = colonne_gfs(geom, vec, iy, ix)
                lignes.append({"station": noms.get(code, code), "code": code, "stem": stem, "temps": t,
                               "jour": t.strftime("%Y-%m-%d"), "echeance_h": e, "u_obs": uo, "v_obs": vo,
                               "u_gfs_brut": float(E[0, iy, ix]), "v_gfs_brut": float(E[1, iy, ix]),
                               "col_u": vol.u[:, iy, ix].astype(float), "col_v": vol.v[:, iy, ix].astype(float),
                               "col_gu": gu[0].astype(float), "col_gv": gu[1].astype(float),
                               "h": h_noeuds[:, iy, ix], "z0": float(geom.z0[iy, ix])})
            if verbose and (n + 1) % 20 == 0:
                print(f"  {nom} {NOM_ECHEANCE[e]} : {n + 1}/{len(stems)} instants ({time.time() - t0:.0f} s)",
                      flush=True)
    wf = None                                           # libere la memoire du GPU
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:                                   # noqa: BLE001
        pass
    return lignes


def paires_variantes(cp: list[dict], c1: list[dict] | None) -> pd.DataFrame:
    """Lignes de colonnes de C' s44 (et de C1) -> paires a 10 m : gfs_brut, cp_s44 (rapporte), c1, c2_*, c3_*."""
    from pcfno3d.ancrage_gfs import nom_reglage, reglages
    cle = ("echeance_h", "stem", "code")
    par_c1 = {tuple(r[k] for k in cle): r for r in (c1 or [])}
    out = []
    for r in cp:
        p = {k: r[k] for k in ("station", "code", "stem", "temps", "jour", "echeance_h", "u_obs", "v_obs",
                               "u_gfs_brut", "v_gfs_brut")}
        p["u_cp_s44"], p["v_cp_s44"] = vent_colonne(r["col_u"], r["col_v"], r["h"], r["z0"])
        for a0, H in reglages():
            p[f"u_{nom_reglage('c2', a0, H)}"], p[f"v_{nom_reglage('c2', a0, H)}"] = vent_ancre(r, a0, H)
        q = par_c1.get(tuple(r[k] for k in cle))
        if c1 is not None:
            if q is None:
                raise ValueError(f"C1 sans la paire {tuple(r[k] for k in cle)} de C' s44 : entrees differentes")
            if abs(q["u_obs"] - r["u_obs"]) > 0 or abs(q["u_gfs_brut"] - r["u_gfs_brut"]) > 1e-6:
                raise ValueError("C1 et C' s44 n'ont pas la meme mesure ou le meme GFS brut a une paire")
            p["u_c1"], p["v_c1"] = vent_colonne(q["col_u"], q["col_v"], q["h"], q["z0"])
            for a0, H in reglages():
                p[f"u_{nom_reglage('c3', a0, H)}"], p[f"v_{nom_reglage('c3', a0, H)}"] = vent_ancre(q, a0, H)
        out.append(p)
    if c1 is not None and len(par_c1) != len(cp):
        raise ValueError(f"C1 a {len(par_c1)} paires, C' s44 {len(cp)} : entrees differentes")
    return pd.DataFrame(out)


# ---------------------------------------------------------------------- etape 1 : diagnostic
def direction(u, v):
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def secteur(u, v) -> np.ndarray:
    """Secteur de provenance (8 secteurs de 45 deg centres sur N, NE, ...) du vent MESURE ; « calme ou faible » sous
    1 m/s (direction peu definie, comme la MAE de direction de valider_grand_tunis_reel)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    k = (np.floor((direction(u, v) + 22.5) / 45.0).astype(int)) % 8
    s = np.array(SECTEURS, dtype=object)[k]
    return np.where(np.hypot(u, v) < 1.0, "calme ou faible", s)


def saison(mois) -> np.ndarray:
    m = np.asarray(mois, int)
    return np.select([np.isin(m, (12, 1, 2)), np.isin(m, (3, 4, 5)), np.isin(m, (6, 7, 8))],
                     ["hiver (DJF)", "printemps (MAM)", "ete (JJA)"], "automne (SON)")


def periode_jour(heure_utc) -> np.ndarray:
    """Jour : 06 a 17 h UTC (07 a 18 h a Tunis, UTC+1) ; nuit sinon. Aux instants du cache (00, 06, 12, 18 UTC) :
    06 et 12 = jour, 18 et 00 = nuit."""
    h = np.asarray(heure_utc, int)
    return np.where((h >= 6) & (h < 18), "jour", "nuit")


def _stats(q: pd.DataFrame, src: str) -> dict:
    du, dv = q[f"u_{src}"] - q["u_obs"], q[f"v_{src}"] - q["v_obs"]
    fort = np.hypot(q["u_obs"], q["v_obs"]) >= 1.0
    dd = np.abs((direction(q[f"u_{src}"], q[f"v_{src}"]) - direction(q["u_obs"], q["v_obs"]) + 180.0) % 360.0 - 180.0)
    return {"rmse_vect": float(np.sqrt(np.mean(du ** 2 + dv ** 2))) if len(q) else math.nan,
            "biais_vitesse": float(np.mean(np.hypot(q[f"u_{src}"], q[f"v_{src}"]) - np.hypot(q["u_obs"], q["v_obs"])))
            if len(q) else math.nan,
            "mae_direction_deg": float(dd[fort].mean()) if fort.any() else math.nan}


def diagnostic_tabarka(p: pd.DataFrame, modele: str = "cp_s44") -> pd.DataFrame:
    """A Tabarka (retire de l'entree), par echeance : tout, par secteur du vent mesure, jour / nuit, saison ; n,
    RMSE vectoriel, biais de vitesse et MAE de direction du modele et de GFS brut, ecart modele - GFS brut."""
    q = p[p["code"] == TABARKA].dropna(subset=[f"u_{modele}", f"v_{modele}", "u_gfs_brut", "v_gfs_brut",
                                              "u_obs", "v_obs"])
    t = pd.to_datetime(q["temps"])
    q = q.assign(secteur=secteur(q["u_obs"], q["v_obs"]), periode=periode_jour(t.dt.hour),
                 saison=saison(t.dt.month), tout="tout")
    lignes = []
    for e, g in q.groupby("echeance_h"):
        for decoupage in ("tout", "secteur", "periode", "saison"):
            for valeur, gg in g.groupby(decoupage):
                m, r = _stats(gg, modele), _stats(gg, "gfs_brut")
                lignes.append({"echeance": NOM_ECHEANCE[int(e)], "decoupage": decoupage, "groupe": valeur,
                               "n": len(gg), **{f"{k}_modele": x for k, x in m.items()},
                               **{f"{k}_gfs_brut": x for k, x in r.items()},
                               "ecart_rmse_modele_moins_gfs": m["rmse_vect"] - r["rmse_vect"]})
    return pd.DataFrame(lignes)


def profil_tabarka(lignes: list[dict], hauteurs=HAUTEURS_PROFIL) -> pd.DataFrame:
    """Profil moyen a Tabarka de 10 a 300 m au-dessus du sol, par echeance : vitesse moyenne du modele et du GFS
    aux memes hauteurs (colonnes aux noeuds, lues comme le DSO7), et vent moyen u, v."""
    out = []
    tab = [r for r in lignes if r["code"] == TABARKA]
    for e in ECHEANCES:
        rs = [r for r in tab if r["echeance_h"] == e]
        if not rs:
            continue
        for hh in hauteurs:
            m = np.array([vent_colonne(r["col_u"], r["col_v"], r["h"], r["z0"], hh) for r in rs], float)
            g = np.array([vent_colonne(r["col_gu"], r["col_gv"], r["h"], r["z0"], hh) for r in rs], float)
            ok = np.isfinite(m).all(1) & np.isfinite(g).all(1)
            out.append({"echeance": NOM_ECHEANCE[e], "hauteur_sol_m": hh, "n": int(ok.sum()),
                        "vitesse_modele": float(np.hypot(*m[ok].T).mean()) if ok.any() else math.nan,
                        "vitesse_gfs": float(np.hypot(*g[ok].T).mean()) if ok.any() else math.nan,
                        "u_modele": float(m[ok, 0].mean()) if ok.any() else math.nan,
                        "v_modele": float(m[ok, 1].mean()) if ok.any() else math.nan,
                        "u_gfs": float(g[ok, 0].mean()) if ok.any() else math.nan,
                        "v_gfs": float(g[ok, 1].mean()) if ok.any() else math.nan})
    return pd.DataFrame(out)


def bilan(p: pd.DataFrame, sources) -> pd.DataFrame:
    """Par echeance, source et aeroport : n, RMSE vectoriel, biais de vitesse, MAE direction (valider_grand_tunis_reel)."""
    v = _module("valider", "valider_grand_tunis_reel.py")
    out = []
    for e, g in p.groupby("echeance_h"):
        src = tuple(s for s in sources if f"u_{s}" in g and g[f"u_{s}"].notna().any())
        out.append(v.bilan_stations(g, src).assign(echeance=NOM_ECHEANCE[int(e)]))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------- etape 3 : critere et choix
def juger_variantes(p: pd.DataFrame, noms: list[str], stations: list) -> dict:
    """Pour chaque variante : verdict des 3 cases par le critere de gfs_v4 (memes tirages des jours pour toutes les
    variantes et echeances), RMSE vectoriel moyen par case et moyenne des trois cases."""
    J = critere_v4()
    jours = sorted(p["jour"].unique())
    w = J.tirages_jours(jours, TIRAGES, GRAINE)
    out = {}
    for nom in noms:
        cases = {}
        for case in CASES:
            q = p[p["echeance_h"] == ECHEANCE_CASE[case]]
            cases[case] = J.juger_case(q, nom, w, jours, stations)
        moyennes = [cases[c]["moyenne_rmse_source"] for c in CASES]
        out[nom] = {"cases": {c: d["verdict"] for c, d in cases.items()},
                    "passe_les_trois": all(d["verdict"] == "accepte" for d in cases.values()),
                    "rmse_moyen_par_case": dict(zip(CASES, moyennes)),
                    "rmse_moyen_trois_cases": float(np.mean(moyennes)),
                    "details": cases}
    return out


def choisir(resultats: dict, familles: dict) -> dict:
    """La regle de choix : parmi les variantes qui passent les TROIS cases, le plus bas RMSE moyen des trois cases ;
    ex aequo a EX_AEQUO pres du meilleur : la famille la plus simple (C2, puis C1, puis C3), puis le RMSE, puis le
    nom (ordre fixe). Aucune ne passe : arret (2026 n'est pas lu)."""
    passent = [n for n, r in resultats.items() if r["passe_les_trois"] and np.isfinite(r["rmse_moyen_trois_cases"])]
    if not passent:
        return {"arret": True, "retenue": None, "passent": [],
                "raison": "aucune variante ne passe le critere dans les trois cases sur la validation 2021 : "
                          "ARRET, la periode 2026 n'est pas lue (regle, etape 3)"}
    meilleur = min(resultats[n]["rmse_moyen_trois_cases"] for n in passent)
    proches = [n for n in passent if resultats[n]["rmse_moyen_trois_cases"] - meilleur <= EX_AEQUO + 1e-12]
    rang = {f: i for i, f in enumerate(ORDRE_SIMPLICITE)}
    retenue = min(proches, key=lambda n: (rang[familles[n]], resultats[n]["rmse_moyen_trois_cases"], n))
    return {"arret": False, "retenue": retenue, "famille": familles[retenue], "passent": sorted(passent),
            "ex_aequo_a_0_005": sorted(proches), "rmse_moyen_retenue": resultats[retenue]["rmse_moyen_trois_cases"],
            "rmse_moyen_meilleur": meilleur}


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:                                 # tabulate absent
        return df.to_string(index=False)


def rapport(sel: dict, res: dict, diag: pd.DataFrame, prof: pd.DataFrame, b: pd.DataFrame, info: dict,
            sha: str) -> str:
    L = ["# Règle `regle_national_gfs_v5_1` : diagnostic et choix sur la validation 2021", "",
         f"Règle `configs/national/regle_national_gfs_v5_1.json` (sha256 `{sha[:12]}…`), approuvée par Ines le "
         "22/11/2026 et vérifiée avant le calcul. Validation 2021 seulement (2022 et 2026 ne sont pas lus). Les "
         "seules mesures sont les METAR ; GFS est une prévision. Chaque aéroport jugé est retiré de l'entrée (11 "
         "autres à l'analyse, aucun aux échéances).", "",
         f"Instants : {info['instants']} sur {info['instants_attendus']} ({info['absents']} écartés pour toutes "
         f"les sources faute d'échéance GFS) ; {info['jours']} jours ; IC 95 % par {TIRAGES} tirages des jours "
         f"(graine {GRAINE}).", "", "## Étape 3 : les 13 variantes", "",
         "| Variante | analyse | +24 h | +48 h | RMSE moyen (3 cases) | passe les trois |", "|---|---|---|---|---:|---|"]
    for n, r in res.items():
        c = r["cases"]
        L.append(f"| {n} | {c[CASES[0]]} | {c[CASES[1]]} | {c[CASES[2]]} | {r['rmse_moyen_trois_cases']:.3f} | "
                 f"{'oui' if r['passe_les_trois'] else 'non'} |")
    L += ["", "**Retenue : " + (f"`{sel['retenue']}`**" if not sel["arret"] else "aucune — ARRÊT, 2026 non lu**"), ""]
    L += ["## Étape 1 : C' s44 + GFS, Tabarka (retiré de l'entrée)", "",
          md_table(diag.round(3)), "",
          "Profil moyen à Tabarka (m/s) :", "",
          md_table(prof.round(3)), "",
          "## RMSE vectoriel par aéroport et par échéance (m/s)", ""]
    t = b.pivot_table(index=["echeance", "station"], columns="source", values="rmse_vect").round(3)
    keep = [c for c in ("gfs_brut", "cp_s44", "c1") if c in t.columns]
    t = t[keep]
    L.append(md_table(t.reset_index()))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- tout
def calculer(cp_cfg, c1_cfg, dossier_gfs: Path, dossier_obs: Path, sortie: Path, checkpoint_cp=None,
             checkpoint_c1=None, device=None, verbose: bool = True) -> dict:
    from pcfno3d.stations_reelles import cellules_codes, lire_observations

    sha = verifier_regle()
    sortie = Path(sortie)
    if (sortie / "selection.json").exists():
        raise SystemExit(f"{sortie} contient deja une selection : rien n'est recalcule ni ecrase")
    forcages = {e: Path(dossier_gfs) / f"forcage_gfs_f{e}.csv" for e in ECHEANCES}
    for e, f in forcages.items():
        if not f.exists():
            raise SystemExit(f"{f} absent : telecharger d'abord l'echeance {NOM_ECHEANCE[e]}")
    stems, row = stems_validation(cp_cfg)
    presents = {e: temps_du_forcage(f) for e, f in forcages.items()}
    communs = [s for s in stems if all(instant_utc(row, s) in presents[e] for e in ECHEANCES)]
    info = {"instants_attendus": len(stems), "instants": len(communs), "absents": len(stems) - len(communs),
            "absents_liste": sorted(set(stems) - set(communs))}
    sortie.mkdir(parents=True, exist_ok=True)
    print(f"validation 2021 : {len(communs)} instants sur {len(stems)}", flush=True)
    if info["absents"] > ABSENTS_MAX * len(stems):
        (sortie / "arret.json").write_text(json.dumps({**info, "regle_sha256": sha, "verdict": "arret sans verdict"},
                                                      indent=2, ensure_ascii=False), encoding="utf-8")
        raise SystemExit(f"plus de {ABSENTS_MAX:.0%} d'instants sans echeance GFS : arret sans verdict (regle)")
    from types import SimpleNamespace
    cellules = cellules_codes(SimpleNamespace(cfg=cp_cfg))          # data_dir/stations.json : les 12 aeroports
    obs = lire_observations(Path(dossier_obs), verbose=verbose, codes=list(cellules))
    lc = colonnes_cache(cp_cfg, forcages, communs, row, obs, checkpoint_cp, device, verbose, "C' s44")
    l1 = colonnes_cache(c1_cfg, forcages, communs, row, obs, checkpoint_c1, device, verbose, "C1")
    p = paires_variantes(lc, l1)
    p.to_csv(sortie / "paires_validation.csv", index=False)
    stations = sorted(p["station"].unique())
    info["jours"] = int(p["jour"].nunique())
    vs = variantes()
    noms = [v["nom"] for v in vs]
    res = juger_variantes(p, noms, stations)
    # C' s44 + GFS sans correction : rapporte (meme critere), sans entrer dans le choix
    res_cp = juger_variantes(p, ["cp_s44"], stations)["cp_s44"]
    sel = choisir(res, {v["nom"]: v["famille"] for v in vs})
    retenue = next((v for v in vs if v["nom"] == sel["retenue"]), None)
    sel.update({"regle": REGLE.name, "regle_sha256": sha, "variante": retenue, "instants": info,
                "aeroports": stations, "date_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")})
    diag = diagnostic_tabarka(p)
    prof = profil_tabarka(lc)
    b = bilan(p, ["gfs_brut", "cp_s44"] + noms)
    diag.to_csv(sortie / "diagnostic_tabarka.csv", index=False)
    prof.to_csv(sortie / "profil_tabarka.csv", index=False)
    b.to_csv(sortie / "bilan_validation.csv", index=False)
    (sortie / "variantes.json").write_text(json.dumps({"regle_sha256": sha, "variantes": res,
                                                       "cp_s44_sans_correction_rapporte": res_cp},
                                                      indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (sortie / "validation_gfs_v5.md").write_text(rapport(sel, res, diag, prof, b, info, sha), encoding="utf-8")
    # ecrit en DERNIER : sa presence signifie que l'etape 3 est complete
    (sortie / "selection.json").write_text(json.dumps(sel, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")
    print(("ARRET : aucune variante ne passe les trois cases, 2026 n'est pas lu" if sel["arret"]
           else f"variante retenue : {sel['retenue']}"), flush=True)
    return sel


def main() -> int:
    from pcfno3d.config import Config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cp", required=True, help="configuration de C' s44 tel que livre (national_v4_eval)")
    ap.add_argument("--c1", required=True, help="configuration de C1 (run_dir runs/national_1km_v5_gfs)")
    ap.add_argument("--gfs", required=True, help="dossier des forcage_gfs_f0.csv, _f24.csv, _f48.csv (+ .npz)")
    ap.add_argument("--obs", required=True, help="METAR des 12 aeroports (2021)")
    ap.add_argument("--sortie", required=True, help="dossier NEUF")
    ap.add_argument("--checkpoint-cp")
    ap.add_argument("--checkpoint-c1")
    ap.add_argument("--device")
    a = ap.parse_args()
    sel = calculer(Config.load(a.cp), Config.load(a.c1), Path(a.gfs), Path(a.obs), Path(a.sortie),
                   a.checkpoint_cp, a.checkpoint_c1, a.device)
    print(json.dumps({k: sel[k] for k in ("arret", "retenue")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
