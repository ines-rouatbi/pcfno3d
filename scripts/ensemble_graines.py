"""Moyenne des graines de C' (v4 nationale) sur la VALIDATION : metriques, DSO4, dispersion (DSO6).

Regle configs/national/regle_ensemble_cp.json (PROPOSITION soumise a l'accord ecrit d'Ines ; empreinte
.sha256 verifiee avant tout calcul ; tous les reglages viennent de ses `parametres`). Aucun entrainement,
le test n'est jamais lu.

  1. meilleure graine fixee AVANT le calcul : minimum de val_score (history.json, os6.meilleure_graine) ;
  2. metriques 0-1 km au-dessus du sol (scripts/dso_hauteur_sol.mesurer, memes formules) pour la moyenne,
     chaque graine et l'ERA5 seule, sur les 64 instants de la selection de validation, entree complete
     puis 25 % des points d'entree (DSO5) ;
  3. DSO4 : passe servie de la moyenne (3 passes, moyenne, w par continuite), mesuree comme le DSO4
     existant, sur un instant de validation ; temps de chaque graine seule a cote ;
  4. DSO6 (estimation de validation) : dispersion entre graines autour de la moyenne, k calibres sur
     la validation (regle auto des 2 plis de jours), couverture 80 % hors pli et Spearman ;
  5. les six criteres de la regle, bilan JSON + Markdown.

    python scripts/ensemble_graines.py -c national_1km_colab --runs <MyDrive>/pcfno3d_national/runs \\
        --regle configs/national/regle_ensemble_cp.json [--checkpoints c42.pt c43.pt c44.pt]

Sortie : <runs>/national_1km_v4_Cp_ens/ (dossier neuf, voir la regle). Code de retour : 0 = les six
criteres remplis (moyenne proposee pour le test, accord separe d'Ines), 3 = non remplis (bilan ecrit).
--max-instants N : essai rapide (fichiers suffixes _essaiN, pas un resultat).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

CRITERES = ("dso1_gain_sur_meilleure_graine", "dso3_rugosite", "dso5_25pct", "dso2_masse",
            "dso4_temps", "mieux_que_era5")


def maintenant() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dispersion_validation(mm, cfg, splits, index, P: dict, max_instants: int | None,
                          verbose: bool = True) -> tuple[dict, dict]:
    """DSO6 sur la validation : (calibration JSON, resume). Deux boucles sur les memes instants :
    histogrammes |erreur| / sigma par pli de jours -> k (regle auto) ; puis couverture et Spearman
    avec le sigma calibre (DSO6Accumulator, comme Evaluator.dso6)."""
    import torch

    from pcfno3d.dataset import ANEO3D
    from pcfno3d.evaluate import couples_fiabilite, plis_jours
    from pcfno3d.reliability import (COMPOSANTES, METHOD_GRAINES, Calibration, DSO6Accumulator,
                                     RatioHistogram, fit_calibration)

    geom = mm.geom
    rc = copy.deepcopy(cfg.reliability)
    rc.calibration_q = P["calibration_q"]
    rc.coverage_target = float(P["couverture_cible"])
    s, h, d = splits.val
    idx = np.asarray(splits.val_selection)
    if max_instants:
        idx = idx[:max_instants]
    ds = ANEO3D(s[idx], h[idx], d[idx], geom, train=False, seed=1000)
    levels = geom.eval_levels()
    pli, plis_info = plis_jours(ds.stems, dict(zip(index["stem"], index["day"])))

    def boucle(on_instant, etape: str) -> int:
        with torch.no_grad():
            for n, (s_raw, sensor, tfeat, y, j) in enumerate(ds):
                dev = mm.device
                sorties = mm.passes(s_raw[None].to(dev), sensor[None].to(dev), tfeat[None].to(dev))
                moy = mm.moyenne(sorties)
                sig = mm.dispersion(moy, sorties)
                servie = tuple(a[0].float().cpu().numpy() for a in moy)
                couples_fiabilite(geom, levels, servie, sig, y.numpy(), str(ds.stems[j]), on_instant)
                del sorties, moy
                if verbose and (n + 1) % 10 == 0:
                    print(f"  dispersion ({etape}) : {n + 1}/{len(ds)} instants", flush=True)
        return len(ds)

    hists = (RatioHistogram(geom.nz), RatioHistogram(geom.nz))
    brut = {c: [0.0, 0, np.inf] for c in COMPOSANTES}          # somme, n, min du sigma brut

    def calib(k, c, e, sg, stem):
        hists[pli[stem]].add(c, k, e, sg, rc.sigma_floor_m_s)
        brut[c][0] += float(np.sum(sg, dtype=np.float64))
        brut[c][1] += int(sg.size)
        brut[c][2] = min(brut[c][2], float(np.min(sg)))

    n = boucle(calib, "calibration")
    data = fit_calibration(RatioHistogram.total(list(hists)), geom, "validation", rc, n,
                           extra={"membres_noms": mm.noms,
                                  "modeles": [m.checkpoint_path.name for m in mm.membres],
                                  "capteurs_visibles": cfg.mask.keep_eval},
                           plis=hists, plis_info=plis_info)
    # meme format que la fiabilite, autre ensemble : la dispersion des N graines autour de leur moyenne
    data.update({"method": METHOD_GRAINES, "members": mm.n, "keep": None, "seed": None,
                 "sigma": "dispersion entre graines autour de la moyenne servie "
                          "(reliability.ensemble_sigma, 1/N)"})
    cal = Calibration(data)
    acc = DSO6Accumulator(geom.nz)
    floor = np.float32(rc.sigma_floor_m_s)
    boucle(lambda k, c, e, sg, _s: acc.add(c, k, e, np.maximum(sg, floor) * np.float32(cal.factor(c, k))),
           "couverture")
    resume = acc.summary(geom, rc.coverage_target)
    for cle in ("dso6_conforme", "dso6_couverture_ok", "dso6_correlation_ok"):
        resume.pop(cle)                         # pas un verdict DSO6 (celui-ci se lit sur le test)
    regle_q = data.get("regle_quantile") or {}
    hors_pli = regle_q.get("couverture_hors_pli", {}).get(f"{data['quantile']:.2f}")
    g = resume["global"]
    min_hp = hors_pli["min_uvw"] if hors_pli else float("nan")
    utile = bool(min_hp >= rc.coverage_target and all(g[f"spearman_{c}"] > 0 for c in "uvw"))
    resume = {"jeu": "validation", "instants": n,
              "nature": "estimation de validation, pas un verdict DSO6 (le verdict se lit sur le test)",
              "sigma": data["sigma"], "membres": mm.noms, "quantile": data["quantile"],
              "couverture80_hors_pli": hors_pli, "couverture80_hors_pli_min_uvw": min_hp,
              "couverture80_validation_k_finaux": data["couverture80_validation"],
              "sigma_brut_moyen_m_s": {c: (b[0] / b[1] if b[1] else float("nan")) for c, b in brut.items()},
              "sigma_brut_min_m_s": {c: (b[2] if b[1] else float("nan")) for c, b in brut.items()},
              "dispersion_utile": utile,
              "critere_d_information": "couverture hors pli minimale u, v, w >= "
                                       f"{rc.coverage_target:g} et Spearman u, v, w > 0",
              **resume}
    return data, resume


def criteres(vd: dict, vd25: dict, dso4: dict, meilleure: str, P: dict) -> dict:
    """Les six criteres de la regle (validation), chacun {valeur, reference, rempli}."""
    m, b = vd["moyenne"], vd[meilleure]
    def rug(x):
        return abs(x["dso3_rugosite"] - 1.0)
    out = {
        "dso1_gain_sur_meilleure_graine": {
            "moyenne": m["rmse_max_uv"], "meilleure_graine": b["rmse_max_uv"],
            "seuil": (1 - P["gain_min_dso1"]) * b["rmse_max_uv"],
            "rempli": bool(m["rmse_max_uv"] <= (1 - P["gain_min_dso1"]) * b["rmse_max_uv"])},
        "dso3_rugosite": {
            "moyenne": m["dso3_rugosite"], "meilleure_graine": b["dso3_rugosite"],
            "rempli": bool(rug(m) <= rug(b) + P["tol_rugosite"])},
        "dso5_25pct": {
            "moyenne": vd25["moyenne"]["rmse_max_uv"], "meilleure_graine": vd25[meilleure]["rmse_max_uv"],
            "rempli": bool(vd25["moyenne"]["rmse_max_uv"]
                           <= (1 + P["tol_dso5"]) * vd25[meilleure]["rmse_max_uv"])},
        "dso2_masse": {"moyenne": m["dso2_rms_moyen"], "rempli": bool(m["dso2_conforme"])},
        "dso4_temps": {"moyenne_s": dso4["mean_s"], "seuil_s": dso4["seuil_s"], "device": dso4["device"],
                       "rempli": bool(dso4["conforme"])},
    }
    if "era5" in vd:
        e = vd["era5"]
        out["mieux_que_era5"] = {"moyenne": m["rmse_max_uv"], "era5": e["rmse_max_uv"],
                                 "rugosite_moyenne": m["dso3_rugosite"], "rugosite_era5": e["dso3_rugosite"],
                                 "rempli": bool(m["rmse_max_uv"] < e["rmse_max_uv"] and rug(m) < rug(e))}
    else:
        out["mieux_que_era5"] = {"rempli": False, "note": "modele sans forcage ERA5 : pas de predicteur era5"}
    return out


def rapport_md(bilan: dict, tab, tab25) -> str:
    v, c = bilan["verdicts_validation"], bilan["criteres"]
    L = [f"# Moyenne des graines de C' — validation ({bilan['instants']} instants, verite d'entrainement v3)", "",
         f"Regle `{bilan['regle']}` (sha256 `{bilan['regle_sha256'][:12]}…`, PROPOSITION). "
         f"Meilleure graine (val_score) : **{bilan['meilleure_graine']}**. Le test n'est pas lu.", "",
         "| Predicteur | rmse_max_uv 0-1 km | pire tranche | rugosite | DSO2 | direction (deg) | rmse_max_uv 25 % |",
         "|---|---|---|---|---|---|---|"]
    for p in v:
        L.append(f"| {p} | {v[p]['rmse_max_uv']:.4f} | {v[p]['tranche_la_pire']} | {v[p]['dso3_rugosite']:.3f} "
                 f"| {v[p]['dso2_rms_moyen']:.2e} | {v[p]['mae_direction_deg']:.2f} "
                 f"| {bilan['verdicts_validation_keep25'][p]['rmse_max_uv']:.4f} |")
    L += ["", "| Critere | rempli |", "|---|---|"]
    L += [f"| {k} | {'oui' if c[k]['rempli'] else 'non'} |" for k in CRITERES]
    d4, d6 = bilan["dso4"], bilan["dso6_validation"]
    L += ["", f"DSO4 : {d4['mean_s'] * 1000:.0f} ms (p95 {d4['p95_s'] * 1000:.0f} ms) pour la passe servie de la "
              f"moyenne de {d4['membres']} graines sur {d4.get('gpu', d4['device'])}"
              + (" ; seules : " + ", ".join(f"{k} {x['mean_s'] * 1000:.0f} ms" for k, x in d4["par_membre"].items())
                 if "par_membre" in d4 else "")
              + (f" ; memoire GPU : poids {d4['memoire_gpu_go']['poids_et_geometrie']:.1f} Go, pic "
                 f"{d4['memoire_gpu_go']['pic_passe_servie']:.1f} Go sur {d4['memoire_gpu_go']['total']:.0f}"
                 if "memoire_gpu_go" in d4 else ""),
          f"DSO6 (estimation de validation, pas un verdict) : couverture 80 % hors pli minimale u, v, w "
          f"{d6['couverture80_hors_pli_min_uvw']:.3f} ; Spearman "
          + ", ".join(f"{x} {d6['global'][f'spearman_{x}']:.2f}" for x in ("u", "v", "w", "speed"))
          + f" ; dispersion utile : {'oui' if d6['dispersion_utile'] else 'non'}", "",
          f"**Moyenne proposee pour le test : {'OUI (accord ecrit separe d Ines)' if bilan['moyenne_proposee_pour_le_test'] else 'NON'}**",
          "", "Tranches (entree complete) :", "", "```", tab.round(4).to_string(index=False), "```",
          "", "Tranches (25 % des points d'entree) :", "", "```", tab25.round(4).to_string(index=False), "```"]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--runs", required=True, help="dossier des run_dir (<prefixe>_s<graine>)")
    ap.add_argument("--regle", required=True)
    ap.add_argument("--checkpoints", nargs="+", default=None, help="un par graine, dans l'ordre de la regle")
    ap.add_argument("--set", action="append", default=[], help="section.cle=valeur")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-instants", type=int, default=None, help="essai rapide (pas un resultat)")
    a = ap.parse_args()

    from pcfno3d import os6
    from pcfno3d.regles import verifier

    regle, sha_regle = verifier(a.regle)
    P = regle["parametres"]
    runs = Path(a.runs)
    sortie = runs / P["sortie"]
    suffixe = f"_essai{a.max_instants}" if a.max_instants else ""
    temoin = sortie / f"bilan_validation{suffixe}.json"
    if temoin.exists():
        raise SystemExit(f"{temoin} existe deja : rien n'est recalcule ni ecrase")
    copie = sortie / Path(a.regle).name
    if copie.exists() and copie.read_bytes() != Path(a.regle).read_bytes():
        raise SystemExit(f"{sortie} a ete commence avec une autre regle : choisir un dossier neuf")
    graines = [int(g) for g in P["graines"]]
    noms = [f"s{g}" for g in graines]
    run_dirs = [runs / f"{P['prefixe']}_s{g}" for g in graines]
    if a.checkpoints is not None and len(a.checkpoints) != len(graines):
        raise SystemExit("--checkpoints : un fichier par graine de la regle")
    # 1. meilleure graine : fixee par la validation de l'entrainement, AVANT tout calcul
    choix = os6.meilleure_graine(runs, P["prefixe"], graines)
    meilleure = f"s{choix['graine']}"
    print(f"regle {Path(a.regle).name} (sha256 {sha_regle[:12]}..., verifiee) | meilleure graine "
          f"(val_score) : {meilleure}", flush=True)

    import dso_hauteur_sol as H
    from pcfno3d.cache import load_index
    from pcfno3d.cli import _parse_set
    from pcfno3d.config import Config
    from pcfno3d.dataset import make_splits
    from pcfno3d.moyenne import ModeleMoyen

    cfg = Config.load(a.config)
    cfg.apply_overrides(_parse_set(a.set))
    if not cfg.inference.w_continuite:
        raise SystemExit("la regle demande w par continuite (inference.w_continuite: true)")
    mm = ModeleMoyen(cfg, run_dirs, checkpoints=a.checkpoints, noms=noms, device=a.device)
    index = load_index(cfg)
    splits = make_splits(cfg, index)
    if not a.max_instants and len(splits.val_selection) != int(P["n_val"]):
        raise SystemExit(f"selection de validation : {len(splits.val_selection)} instants, la regle en fixe "
                         f"{P['n_val']} (train.val_samples)")
    ev = SimpleNamespace(geom=mm.geom, splits=splits)

    # 2. metriques 0-1 km au-dessus du sol (validation), entree complete puis 25 % des points
    tab, info = H.mesurer(ev, cfg, None, a.max_instants, jeu="validation", predicteurs=mm.predicteurs())
    vd = H.verdicts(tab, info["global"], cfg, None)
    print({p: round(x["rmse_max_uv"], 4) for p, x in vd.items()}, flush=True)
    k25 = float(P["keep_dso5"])
    tab25, info25 = H.mesurer(ev, cfg, k25, a.max_instants, jeu="validation", predicteurs=mm.predicteurs())
    vd25 = H.verdicts(tab25, info25["global"], cfg, k25)

    # 3. DSO4 : passe servie de la moyenne, sur un instant de VALIDATION
    stem = str(splits.val[0][splits.val_selection[0]])
    S, hour, doy, stamp = mm.surface_from_cache(stem)
    dso4 = {"instant": stem, "jeu": "validation",
            **mm.benchmark(S, hour, doy, n=int(P["dso4_n"]), warmup=int(P["dso4_chauffe"]), timestamp=stamp)}
    print(f"DSO4 moyenne : {dso4['mean_s'] * 1000:.0f} ms", flush=True)

    # 4. DSO6 : dispersion entre graines, calibree sur la validation
    cal, dso6 = dispersion_validation(mm, cfg, splits, index, P, a.max_instants)

    # 5. criteres et sorties (ecrites a la fin : un calcul interrompu ne laisse rien de partiel)
    crit = criteres(vd, vd25, dso4, meilleure, P)
    proposee = all(crit[k]["rempli"] for k in CRITERES)
    membres = {"regle_sha256": sha_regle, "meilleure_graine": choix,
               "membres": [{"nom": n, "run_dir": str(r), "checkpoint": str(m.checkpoint_path),
                            "epoque": (m.epoch + 1) if m.epoch is not None else None,
                            "val_score": float(m.val_score) if m.val_score is not None else None,
                            "norm_stats_sha256": sha(m.cfg.paths.norm_stats)}
                           for n, r, m in zip(noms, run_dirs, mm.membres)]}
    sortie.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.regle, copie)
    (sortie / f"membres{suffixe}.json").write_text(json.dumps(membres, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    tab.to_csv(sortie / f"tranches_validation{suffixe}.csv", index=False)
    tab25.to_csv(sortie / f"tranches_validation_keep{int(round(k25 * 100))}{suffixe}.csv", index=False)
    (sortie / f"dso4_moyenne{suffixe}.json").write_text(json.dumps(dso4, indent=2), encoding="utf-8")
    pc = sortie / f"calibration_graines_validation{suffixe}.json"
    pc.write_text(json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8")
    dso6["calibration_sha256"] = sha(pc)
    (sortie / f"dso6_graines_validation{suffixe}.json").write_text(
        json.dumps(dso6, indent=2, ensure_ascii=False), encoding="utf-8")
    bilan = {"date_utc": maintenant(), "regle": Path(a.regle).name, "regle_sha256": sha_regle,
             "statut_regle": regle.get("statut"), "jeu": "validation", "instants": info["instants"],
             **({"essai": "essai rapide (--max-instants), pas un resultat"} if a.max_instants else {}),
             "membres": noms, "meilleure_graine": meilleure,
             "verdicts_validation": vd, "verdicts_validation_keep25": vd25,
             "dso4": dso4, "dso6_validation": {k: dso6[k] for k in (
                 "nature", "quantile", "couverture80_hors_pli_min_uvw", "dispersion_utile", "global",
                 "sigma_brut_moyen_m_s", "calibration_sha256")},
             "criteres": crit, "moyenne_proposee_pour_le_test": bool(proposee),
             "test": "non lu ; lecture du test de la moyenne seulement avec un accord ecrit separe d'Ines",
             "note": "verite v3 = verite d'entrainement (generee), pas des mesures"}
    temoin.write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    md = rapport_md(bilan, tab, tab25)
    (sortie / f"bilan_validation{suffixe}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"resultats -> {sortie}")
    return 0 if proposee else 3


if __name__ == "__main__":
    raise SystemExit(main())
