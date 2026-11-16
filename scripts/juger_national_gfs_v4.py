"""Regle configs/national/regle_national_gfs_v4.json (fixee et hachee avant tout calcul, accord d'Ines du 16/11/2026) :
ce que le service sert hors du Grand Tunis en temps reel et en prevision, juge aux 12 aeroports METAR sur les instants
synoptiques du test national 2022, sans aucun reentrainement.

Sources, a 10 m au-dessus du sol a la cellule de l'aeroport, l'aeroport juge RETIRE :
  - gfs_brut (reference) : le vent GFS 10 m de l'echeance interpole aux colonnes du modele (bilineaire,
    `forcage.charger_e` avec `mask.e_source: forcage`), exactement le fond de (b) ;
  - a_national_gfs : C' s44 nourri par le forcage GFS de l'echeance ; entree = pseudo-capteurs GFS 10 m + a l'analyse
    les 11 AUTRES aeroports mesures, aux echeances +24 h / +48 h aucun aeroport ; vent a 10 m comme le DSO7
    (`stations_reelles.vent_a_hauteur`) ;
  - b_gfs_oi_10m (analyse seulement) : fond GFS 10 m + OI des innovations des 11 AUTRES aeroports
    (`vent10m_oi.couche_10m`, L 150 km, lambda 4, biais 2015-2020 : les parametres de regle_produit_vent10m_v4).
Rapporte sans verdict : ERA5 10 m aux memes instants (le _E du cache, deja lu au DSO7).

Metrique et critere recopies de la regle (dont l'empreinte est verifiee avant toute lecture) : RMSE vectoriel par
aeroport, moyenne des 12, ecart = moyenne(source) - moyenne(gfs_brut), IC 95 % par bootstrap des JOURS (2000 tirages,
graine 2030, memes tirages partout) ; ACCEPTE si la borne haute <= 0 ET aucun aeroport degrade de plus de 5 %.

    python scripts/juger_national_gfs_v4.py -c national_v4_eval --gfs <dossier des forcage_gfs_f{0,24,48}.csv> \
        --obs <dossier des METAR> --sortie <dossier neuf>

Sorties : paires.csv, bilan.csv (par echeance, source et aeroport), verdicts_gfs_v4.json (lu par le service :
pcfno3d.statut_service, variable PCFNO3D_VERDICTS_GFS_V4) et gfs_v4.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]
REGLE = REPO / "configs" / "national" / "regle_national_gfs_v4.json"
# recopies de la regle (verifies par tests/test_gfs_v4_national.py contre son texte)
ECHEANCES = (0, 24, 48)
NOM_ECHEANCE = {0: "analyse", 24: "+24 h", 48: "+48 h"}
CASES = ("a / analyse", "a / +24 h", "a / +48 h", "b / analyse")
TIRAGES, GRAINE, DEGRADATION_MAX, ABSENTS_MAX = 2000, 2030, 0.05, 0.02
SOURCE_CASE = {"a": "a_national_gfs", "b": "b_gfs_oi_10m"}


def verifier_regle(f: Path = REGLE) -> str:
    h = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    attendu = (Path(f).parent / (Path(f).name + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"regle modifiee depuis son hachage : {h} != {attendu}")
    return h


# ---------------------------------------------------------------------- metrique (sans modele : testee seule)
def _erreurs(p: pd.DataFrame, src: str) -> pd.DataFrame:
    q = p.dropna(subset=[f"u_{src}", f"v_{src}", "u_obs", "v_obs"])
    return q.assign(e2=(q[f"u_{src}"] - q["u_obs"]) ** 2 + (q[f"v_{src}"] - q["v_obs"]) ** 2)


def rmse_par_aeroport(p: pd.DataFrame, src: str) -> pd.Series:
    q = _erreurs(p, src)
    return np.sqrt(q.groupby("station")["e2"].mean())


def tirages_jours(jours: list, n: int = TIRAGES, graine: int = GRAINE) -> np.ndarray:
    """(n, J) multiplicites des jours tires avec remise : les MEMES pour toutes les sources et echeances."""
    rng = np.random.default_rng(graine)
    idx = rng.integers(0, len(jours), size=(int(n), len(jours)))
    w = np.zeros((int(n), len(jours)))
    np.add.at(w, (np.arange(int(n))[:, None], idx), 1.0)
    return w


def _sommes(p: pd.DataFrame, src: str, jours: list, stations: list) -> tuple[np.ndarray, np.ndarray]:
    q = _erreurs(p, src)
    g = q.groupby(["jour", "station"])["e2"].agg(["sum", "size"])
    s = np.zeros((len(jours), len(stations)))
    k = np.zeros_like(s)
    ij = {j: i for i, j in enumerate(jours)}
    ia = {a: i for i, a in enumerate(stations)}
    for (j, a), r in g.iterrows():
        if j in ij and a in ia:
            s[ij[j], ia[a]], k[ij[j], ia[a]] = r["sum"], r["size"]
    return s, k


def ecart_ic(p: pd.DataFrame, src: str, ref: str, w: np.ndarray, jours: list, stations: list) -> dict:
    """Ecart moyenne des aeroports (RMSE vectoriel) src - ref et IC 95 % par tirage des jours (w). Sur les paires ou
    les deux sources existent ; dans un tirage, la moyenne porte sur les aeroports qui y ont au moins une paire."""
    q = p.dropna(subset=[f"u_{src}", f"v_{src}", f"u_{ref}", f"v_{ref}", "u_obs", "v_obs"])
    sa, k = _sommes(q, src, jours, stations)
    sb, _ = _sommes(q, ref, jours, stations)

    def moyenne(ss, kk):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nanmean(np.sqrt(ss / np.where(kk > 0, kk, np.nan)), axis=-1)

    plein = np.ones((1, len(jours)))
    e = float(moyenne(plein @ sa, plein @ k)[0] - moyenne(plein @ sb, plein @ k)[0])
    d = moyenne(w @ sa, w @ k) - moyenne(w @ sb, w @ k)
    return {"ecart": e, "ic95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
            "n_paires": int(len(q)), "n_jours": len(jours), "tirages": int(len(w)), "graine": GRAINE}


def juger_case(p: pd.DataFrame, src: str, w: np.ndarray, jours: list, stations: list) -> dict:
    """Le critere de la regle pour une source a une echeance (p : paires de cette echeance)."""
    ic = ecart_ic(p, src, "gfs_brut", w, jours, stations)
    r_src, r_ref = rmse_par_aeroport(p, src), rmse_par_aeroport(p, "gfs_brut")
    par = {a: {"rmse_source": float(r_src.get(a, np.nan)), "rmse_gfs_brut": float(r_ref.get(a, np.nan)),
               "degradation": float(r_src.get(a, np.nan) / r_ref.get(a, np.nan) - 1.0)} for a in stations}
    manque = [a for a, x in par.items() if not np.isfinite(x["degradation"])]
    pires = [a for a, x in par.items() if np.isfinite(x["degradation"]) and x["degradation"] > DEGRADATION_MAX]
    out = {"source": src, **ic, "par_aeroport": par, "aeroports_degrades_plus_5_pct": pires,
           "moyenne_rmse_source": float(np.mean([x["rmse_source"] for x in par.values()])),
           "moyenne_rmse_gfs_brut": float(np.mean([x["rmse_gfs_brut"] for x in par.values()]))}
    if manque:
        out.update(verdict="non juge", raison=f"aeroports sans paire : {manque}")
    else:
        ok = ic["ic95"][1] <= 0.0 and not pires
        out["verdict"] = "accepte" if ok else "refuse"
    return out


def juger(p: pd.DataFrame, stations: list, sha: str) -> dict:
    """Verdicts des 4 cases ; `p` : paires des trois echeances (colonne echeance_h, jour, station)."""
    jours = sorted(p["jour"].unique())
    w = tirages_jours(jours)
    cases = {}
    for case in CASES:
        lettre, nom = case.split(" / ")
        e = {v: k for k, v in NOM_ECHEANCE.items()}[nom]
        cases[case] = juger_case(p[p["echeance_h"] == e], SOURCE_CASE[lettre], w, jours, stations)
    return {"regle": REGLE.name, "regle_sha256": sha, "cases": {k: v["verdict"] for k, v in cases.items()},
            "details": cases, "jours": len(jours), "aeroports": stations}


def bilan(p: pd.DataFrame) -> pd.DataFrame:
    """Par echeance, source et aeroport : n, RMSE vectoriel, biais de vitesse, MAE direction (rapporte sans verdict)."""
    spec = importlib.util.spec_from_file_location("valider", REPO / "scripts" / "valider_grand_tunis_reel.py")
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    out = []
    for e, g in p.groupby("echeance_h"):
        sources = [s for s in ("gfs_brut", "a_national_gfs", "b_gfs_oi_10m", "era5") if g[f"u_{s}"].notna().any()]
        out.append(v.bilan_stations(g, tuple(sources)).assign(echeance=NOM_ECHEANCE[int(e)]))
    return pd.concat(out, ignore_index=True)


def rapport(verdicts: dict, b: pd.DataFrame, info: dict) -> str:
    L = ["# Règle `regle_national_gfs_v4` : national nourri par GFS, aux 12 aéroports (test 2022)", "",
         f"Règle `configs/national/regle_national_gfs_v4.json` (sha256 `{verdicts['regle_sha256'][:12]}…`), fixée et "
         "hachée avant le calcul (accord d'Ines du 16/11/2026). **Seconde lecture** de la période 2022 (les mêmes METAR "
         "ont servi au DSO7) : ce n'est pas un jugement indépendant. GFS et ERA5 sont des prévisions et une réanalyse ; "
         "les seules mesures sont les METAR. Aéroport jugé retiré de l'entrée et de l'OI.", "",
         f"Instants : {info['instants']} sur {info['instants_attendus']} ({info['absents']} écartés pour toutes les "
         f"sources faute d'échéance GFS) ; {verdicts['jours']} jours ; IC 95 % par {TIRAGES} tirages des jours "
         f"(graine {GRAINE}).", "",
         "| Case | Source | Moyenne RMSE source | Moyenne RMSE GFS brut | Écart | IC 95 % | Aéroports > +5 % | Verdict |",
         "|---|---|---:|---:|---:|---|---|---|"]
    for case, d in verdicts["details"].items():
        L.append(f"| {case} | {d['source']} | {d['moyenne_rmse_source']:.3f} | {d['moyenne_rmse_gfs_brut']:.3f} | "
                 f"{d['ecart']:+.3f} | [{d['ic95'][0]:+.3f} ; {d['ic95'][1]:+.3f}] | "
                 f"{', '.join(d['aeroports_degrades_plus_5_pct']) or 'aucun'} | **{d['verdict']}** |")
    L += ["", "(b) à +24 h et +48 h vaut GFS brut à l'identique (aucune mesure de l'heure visée) : non jugé, la page "
          "dit « GFS brut, sans correction ».", "", "## Par aéroport (m/s ; rapporté, l'ERA5 sans verdict)", ""]
    t = b.pivot_table(index=["echeance", "station"], columns="source", values="rmse_vect").round(3)
    L.append(t.reset_index().to_markdown(index=False) if hasattr(t, "to_markdown") else t.to_string())
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- passes du modele
def paires_echeance(base, forcage: Path, echeance_h: int, dossier_obs: Path, garder: set | None,
                    checkpoint=None, device=None, verbose: bool = True) -> tuple[pd.DataFrame, list, list]:
    """Paires d'une echeance : (paires, stems juges, stems synoptiques du test). `garder` : stems retenus (instants
    presents aux trois echeances) ; None = tous ceux dont l'instant est EXACTEMENT dans le forcage."""
    import copy

    from pcfno3d.cache import load_index
    from pcfno3d.dataset import is_synoptic, make_splits
    from pcfno3d.forcage import charger_e
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.stations_reelles import (cellules_codes, entree_reelle, lire_observations, noms_stations, obs_a,
                                          vent_a_hauteur)
    from pcfno3d.vent10m_oi import couche_10m

    test = make_splits(base, load_index(base))                 # les instants du test (forcage ERA5 : rien d'ecarte)
    attendus = [s for s in test.test[0] if is_synoptic(test.stamp_of[s])]
    cfg = copy.deepcopy(base)
    cfg.apply_overrides({"paths.forcage_era5": str(forcage), "mask.e_source": "forcage"})
    temps_f = set(pd.to_datetime(pd.read_csv(forcage, usecols=["timestamp_utc"])["timestamp_utc"], utc=True)
                  .dt.tz_localize(None))
    row = load_index(base).set_index("stem")

    def instant(stem):
        t = pd.Timestamp(row.loc[stem, "timestamp"])
        return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t

    stems = [s for s in attendus if instant(s) in temps_f and (garder is None or s in garder)]
    if garder is None:
        return pd.DataFrame(), stems, attendus
    wf = WindFieldModel(cfg, checkpoint=checkpoint, device=device, verbose=False)
    geom = wf.geom
    cellules = cellules_codes(geom)
    noms = noms_stations(geom)
    obs = lire_observations(Path(dossier_obs), verbose=verbose, codes=list(cellules))
    h_noeuds = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    cache = cfg.paths.cache_dir
    paires, t0 = [], time.time()

    def passe(S_in, cap, r, t):
        S_in = S_in.copy()
        S_in[7] = cap
        return wf.predict(S_in, float(r["hour"]), int(r["doy"]), timestamp=t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          reliability=False, tta=1)

    for n, stem in enumerate(stems):
        r, t = row.loc[stem], instant(stem)
        mesures = {c: m for c, m in obs_a(obs, t).items() if c in cellules}
        if not mesures:
            continue
        S = np.load(cache / f"{stem}_S.npy")
        E = charger_e(geom, cache / f"{stem}_E.npy")         # GFS 10 m de l'echeance aux colonnes (bilineaire)
        E5 = np.load(cache / f"{stem}_E.npy").astype(np.float32) if echeance_h == 0 else None   # ERA5 (information)
        vol_seul = None
        if echeance_h:                                       # heures a venir : aucun aeroport en entree
            S_in, cap = entree_reelle(S, E, geom, cellules, {})
            vol_seul = passe(S_in, cap, r, t)
        for code, (uo, vo) in mesures.items():
            iy, ix = cellules[code]
            if echeance_h == 0:
                S_in, cap = entree_reelle(S, E, geom, cellules, mesures, exclue=code)
                vol = passe(S_in, cap, r, t)
            else:
                vol = vol_seul
            um, vm = vent_a_hauteur(vol.u[:, iy, ix], vol.v[:, iy, ix], h_noeuds[:, iy, ix], geom.z0[iy, ix])
            p = {"station": noms.get(code, code), "code": code, "stem": stem, "temps": t, "jour": t.strftime("%Y-%m-%d"),
                 "echeance_h": echeance_h, "u_obs": uo, "v_obs": vo, "u_gfs_brut": float(E[0, iy, ix]),
                 "v_gfs_brut": float(E[1, iy, ix]), "u_a_national_gfs": um, "v_a_national_gfs": vm,
                 "u_b_gfs_oi_10m": np.nan, "v_b_gfs_oi_10m": np.nan, "u_era5": np.nan, "v_era5": np.nan}
            if echeance_h == 0:
                autres = {c: m for c, m in mesures.items() if c != code}
                c10, _ = couche_10m(E, cellules, autres, float(geom.dx), float(geom.dy))
                p.update(u_b_gfs_oi_10m=float(c10[0, iy, ix]), v_b_gfs_oi_10m=float(c10[1, iy, ix]),
                         u_era5=float(E5[0, iy, ix]), v_era5=float(E5[1, iy, ix]))
            paires.append(p)
        if verbose and (n + 1) % 20 == 0:
            print(f"  {NOM_ECHEANCE[echeance_h]} : {n + 1}/{len(stems)} instants ({time.time() - t0:.0f} s)", flush=True)
    del wf
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return pd.DataFrame(paires), stems, attendus


def calculer(base, dossier_gfs: Path, dossier_obs: Path, sortie: Path, checkpoint=None, device=None,
             verbose: bool = True) -> dict:
    """Toute la regle : instants communs aux trois echeances, paires, verdicts, sorties dans `sortie` (neuf)."""
    sha = verifier_regle()
    sortie = Path(sortie)
    if (sortie / "verdicts_gfs_v4.json").exists():
        raise SystemExit(f"{sortie} contient deja des verdicts : une seule lecture, rien n'est ecrase")
    forcages = {e: Path(dossier_gfs) / f"forcage_gfs_f{e}.csv" for e in ECHEANCES}
    for e, f in forcages.items():
        if not f.exists():
            raise SystemExit(f"{f} absent : telecharger d'abord l'echeance {NOM_ECHEANCE[e]}")
    # instants presents aux TROIS echeances (un instant sans echeance GFS est retire pour toutes les sources)
    communs, attendus = None, []
    for e, f in forcages.items():
        _, stems, attendus = paires_echeance(base, f, e, Path(dossier_obs), None)
        communs = set(stems) if communs is None else communs & set(stems)
    absents = len(attendus) - len(communs)
    info = {"instants_attendus": len(attendus), "instants": len(communs), "absents": absents,
            "absents_liste": sorted(set(attendus) - communs)}
    print(f"instants : {len(communs)} sur {len(attendus)} ({absents} sans echeance GFS a l'une des echeances)", flush=True)
    sortie.mkdir(parents=True, exist_ok=True)
    if absents > ABSENTS_MAX * len(attendus):
        (sortie / "arret.json").write_text(json.dumps({**info, "regle_sha256": sha, "verdict": "arret sans verdict"},
                                                      indent=2, ensure_ascii=False), encoding="utf-8")
        raise SystemExit(f"plus de {ABSENTS_MAX:.0%} d'instants sans echeance GFS : arret sans verdict (regle)")
    tout = [paires_echeance(base, f, e, Path(dossier_obs), communs, checkpoint, device, verbose)[0]
            for e, f in forcages.items()]
    p = pd.concat(tout, ignore_index=True)
    p.to_csv(sortie / "paires.csv", index=False)
    v = juger(p, sorted(p["station"].unique()), sha)
    v["instants"] = info
    b = bilan(p)
    b.to_csv(sortie / "bilan.csv", index=False)
    (sortie / "verdicts_gfs_v4.json").write_text(json.dumps(v, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
    (sortie / "gfs_v4.md").write_text(rapport(v, b, info), encoding="utf-8")
    return v


def main() -> int:
    from pcfno3d.config import Config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True, help="configuration nationale de C' s44 (run_dir du modele livre)")
    ap.add_argument("--gfs", required=True, help="dossier des forcage_gfs_f0.csv, _f24.csv, _f48.csv (+ .npz)")
    ap.add_argument("--obs", required=True, help="METAR des 12 aeroports")
    ap.add_argument("--sortie", required=True, help="dossier NEUF des resultats")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    args = ap.parse_args()
    v = calculer(Config.load(args.config), Path(args.gfs), Path(args.obs), Path(args.sortie), args.checkpoint,
                 args.device)
    print(json.dumps(v["cases"], indent=1, ensure_ascii=False))
    print(f"-> {args.sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
