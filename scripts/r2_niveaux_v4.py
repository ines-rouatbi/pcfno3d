"""R2 u, v, w par niveau (validation) d'un entrainement v4 national, avant et apres post-traitement, et DSO4.

Regle configs/national/regle_postraitement_r2_v4.json (empreinte .sha256 verifiee avant toute lecture des
donnees). Aucun entrainement, le test n'est jamais lu, aucun bruit ajoute.

  passe 1 : statistiques des moindres carres par pli de jours, niveau et composante (u, v) -> a, b ;
  passe 2 : R2 / RMSE par niveau pour brut, w_continuite, calibre_w_continuite (hors pli), era5 ;
  DSO4    : 3 passes de chauffe puis 20 passes servies, mediane.

    python scripts/r2_niveaux_v4.py -c national_1km_colab --run <runs>/national_1km_v4_Cp_s43 \\
        --regle configs/national/regle_postraitement_r2_v4.json

Sortie : <run>/r2_niveaux_validation/ (dossier neuf). --max-instants N : essai (suffixe _essaiN).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

PREDS = ("brut", "w_continuite", "calibre_w_continuite", "era5")


def ajuster(st: np.ndarray) -> tuple[float, float]:
    """st = [n, sx, sy, sxx, sxy] -> (a, b) des moindres carres ; identite si degenere."""
    n, sx, sy, sxx, sxy = st
    var = sxx - sx * sx / n if n > 0 else 0.0
    if n < 2 or var <= 0:
        return 0.0, 1.0
    b = (sxy - sx * sy / n) / var
    return float((sy - b * sx) / n), float(b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--run", required=True, help="run_dir de l'entrainement (best.pt + norm_stats.json)")
    ap.add_argument("--regle", default=str(REPO / "configs" / "national" / "regle_postraitement_r2_v4.json"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-instants", type=int, default=None)
    ap.add_argument("--n-dso4", type=int, default=20)
    ap.add_argument("--chauffe-dso4", type=int, default=3)
    a = ap.parse_args(argv)

    from pcfno3d.regles import verifier
    regle, h = verifier(a.regle)
    run = Path(a.run)
    suffixe = f"_essai{a.max_instants}" if a.max_instants else ""
    sortie = run / "r2_niveaux_validation"
    temoin = sortie / f"bilan{suffixe}.json"
    if temoin.exists():
        raise SystemExit(f"{temoin} existe deja : rien n'est recalcule ni ecrase")
    print(f"regle {Path(a.regle).name} sha256 {h[:12]}... verifiee | run {run.name}", flush=True)

    import torch
    from torch.utils.data import DataLoader

    from pcfno3d.cache import load_index
    from pcfno3d.cli import _parse_set
    from pcfno3d.config import Config
    from pcfno3d.dataset import ANEO3D, assemble_input, make_splits
    from pcfno3d.evaluate import plis_jours
    from pcfno3d.forcage import N_SCAL
    from pcfno3d.geometry import N_TFEAT
    from pcfno3d.inference import WindFieldModel

    cfg = Config.load(a.config)
    cfg.apply_overrides(_parse_set(a.set))
    cfg.apply_overrides({"paths.run_dir": run, "inference.w_continuite": True})
    m = WindFieldModel(cfg, device=a.device)
    geom, dev = m.geom, m.device
    index = load_index(cfg)
    splits = make_splits(cfg, index)
    va_s, va_h, va_d = splits.val
    idx = np.asarray(splits.val_selection)
    if a.max_instants:
        idx = idx[:a.max_instants]
    ds = ANEO3D(va_s[idx], va_h[idx], va_d[idx], geom, train=False, seed=1000)
    stems = [str(s) for s in va_s[idx]]
    pli, plis_info = plis_jours(stems, dict(zip(index["stem"].astype(str), index["day"])))
    nz, levels = geom.nz, list(geom.eval_levels())

    @torch.no_grad()
    def reseau(s_raw, sensor, tfeat):
        with torch.autocast(device_type=dev.type, dtype=m.amp_dtype, enabled=m.use_amp):
            pred = m.model(assemble_input(s_raw, sensor, tfeat, geom, m.norm))
        return m.norm.denorm(pred.float())

    def iterer():
        for s_raw, sensor, tfeat, y, j in DataLoader(ds, batch_size=1):
            yield (s_raw.to(dev), sensor.to(dev), tfeat.to(dev), y.to(dev), stems[int(j)])

    # ---- passe 1 : moindres carres par pli, niveau, composante
    stats = np.zeros((2, 2, nz, 5))                 # pli, (u, v), niveau, [n, sx, sy, sxx, sxy]
    t0 = time.time()
    for n, (s_raw, sensor, tfeat, y, stem) in enumerate(iterer()):
        u, v, _ = reseau(s_raw, sensor, tfeat)
        for ci, p in enumerate((u, v)):
            t = y[0, ci]
            known = geom.valid_t & torch.isfinite(t)
            for k in range(nz):
                mk = known[k]
                if mk.sum() == 0:
                    continue
                x, z = p[0, k][mk].double(), t[k][mk].double()
                stats[pli[stem], ci, k] += [x.numel(), x.sum().item(), z.sum().item(),
                                            (x * x).sum().item(), (x * z).sum().item()]
        if (n + 1) % 16 == 0:
            print(f"  passe 1 : {n + 1}/{len(ds)} ({time.time() - t0:.0f} s)", flush=True)
    coef = {"pli_a": [[ajuster(stats[0, c, k]) for k in range(nz)] for c in range(2)],
            "pli_b": [[ajuster(stats[1, c, k]) for k in range(nz)] for c in range(2)],
            "servis": [[ajuster(stats[0, c, k] + stats[1, c, k]) for k in range(nz)] for c in range(2)]}
    # instants du pli a -> coefficients ajustes sur b, et inversement
    hors_pli = {0: coef["pli_b"], 1: coef["pli_a"]}

    def calibrer(p, cc):
        a_ = torch.tensor([x[0] for x in cc], device=dev, dtype=p.dtype).view(1, nz, 1, 1)
        b_ = torch.tensor([x[1] for x in cc], device=dev, dtype=p.dtype).view(1, nz, 1, 1)
        return (a_ + b_ * p) * geom.valid_t

    # ---- passe 2 : R2 / RMSE par niveau
    acc = {p: np.zeros((3, nz, 4)) for p in PREDS}     # composante, niveau, [n, sse, st, st2]
    t0 = time.time()
    for n, (s_raw, sensor, tfeat, y, stem) in enumerate(iterer()):
        u, v, w = reseau(s_raw, sensor, tfeat)
        cc = hors_pli[pli[stem]]
        uc, vc = calibrer(u, cc[0]), calibrer(v, cc[1])
        sorties = {"brut": (u, v, w), "w_continuite": (u, v, geom.w_continuite(u, v)),
                   "calibre_w_continuite": (uc, vc, geom.w_continuite(uc, vc))}
        if geom.interp_era5 is not None:
            nc = geom.forcage.n_cellules
            uv = geom.interp_era5.dense(tfeat[:, N_TFEAT + N_SCAL:N_TFEAT + N_SCAL + nc])
            ue, ve = uv[:, 0] * geom.valid_t, uv[:, 1] * geom.valid_t
            sorties["era5"] = (ue, ve, geom.w_continuite(ue, ve))
        for nom, champs in sorties.items():
            for ci, p in enumerate(champs):
                t = y[0, ci]
                known = geom.valid_t & torch.isfinite(t)
                for k in range(nz):
                    mk = known[k]
                    if mk.sum() == 0:
                        continue
                    pk, tk = p[0, k][mk].double(), t[k][mk].double()
                    acc[nom][ci, k] += [tk.numel(), ((pk - tk) ** 2).sum().item(), tk.sum().item(),
                                        (tk ** 2).sum().item()]
        if (n + 1) % 16 == 0:
            print(f"  passe 2 : {n + 1}/{len(ds)} ({time.time() - t0:.0f} s)", flush=True)

    import pandas as pd
    lignes, resume = [], {}
    seuil = float(regle["seuil"]["r2_uvw_par_niveau"])
    for nom in PREDS:
        if not acc[nom][0, :, 0].any():
            continue
        n_ok, rmse_uv, r2_min = 0, 0.0, (np.inf, None, None)
        for k in range(nz):
            rec = {"predicteur": nom, "z_m": int(geom.z_levels[k]), "couverture": float(geom.cover[k]),
                   "evalue_dso": k in levels}
            for ci, c in enumerate("uvw"):
                nn, sse, st, st2 = acc[nom][ci, k]
                if nn == 0:
                    continue
                sst = st2 - st * st / nn
                r2 = float(1 - sse / sst) if sst > 0 else float("nan")
                rec[f"r2_{c}"], rec[f"rmse_{c}"] = r2, float(np.sqrt(sse / nn))
                if k in levels and np.isfinite(r2):
                    n_ok += r2 >= seuil
                    if r2 < r2_min[0]:
                        r2_min = (r2, c, int(geom.z_levels[k]))
                    if c in "uv":
                        rmse_uv = max(rmse_uv, rec[f"rmse_{c}"])
            lignes.append(rec)
        resume[nom] = {"couples_r2_ok": int(n_ok), "couples_evalues": 3 * len(levels),
                       "r2_min": r2_min[0], "r2_min_composante": r2_min[1], "r2_min_niveau_m": r2_min[2],
                       "rmse_max_uv_niveaux_evalues": rmse_uv,
                       "dso1_r2_uvw_conforme_validation": bool(n_ok == 3 * len(levels))}
    tab = pd.DataFrame(lignes)
    wc, cal = resume["w_continuite"], resume["calibre_w_continuite"]
    retenu = ("calibre_w_continuite" if cal["couples_r2_ok"] > wc["couples_r2_ok"]
              and cal["rmse_max_uv_niveaux_evalues"] <= 1.01 * wc["rmse_max_uv_niveaux_evalues"]
              else "w_continuite")
    print({k: (v["couples_r2_ok"], round(v["r2_min"], 4), v["r2_min_composante"], v["r2_min_niveau_m"])
           for k, v in resume.items()}, "| retenu :", retenu, flush=True)

    # ---- DSO4 : mediane de 20 passes servies apres chauffe, grille complete
    stem0 = stems[0]
    S, hour, doy, stamp = m.surface_from_cache(stem0)
    for _ in range(a.chauffe_dso4):
        m.predict(S, hour, doy, timestamp=stamp, mask_nan=False)
    temps = [m.predict(S, hour, doy, timestamp=stamp, mask_nan=False).inference_s for _ in range(a.n_dso4)]
    dso4 = {"instant": stem0, "jeu": "validation", "grille": list(map(int, (nz, *geom.shape[1:]))),
            "chauffe": a.chauffe_dso4, "n": a.n_dso4, "mediane_s": float(np.median(temps)),
            "moyenne_s": float(np.mean(temps)), "p95_s": float(np.quantile(temps, 0.95)),
            "max_s": float(np.max(temps)), "seuil_s": float(cfg.dso.time_s),
            "conforme_mediane": bool(np.median(temps) < cfg.dso.time_s), "device": str(dev)}
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        dso4.update({"gpu": p.name, "memoire_gpu_go": p.total_memory / 1e9,
                     "pic_memoire_go": torch.cuda.max_memory_allocated(dev) / 1e9})
    print(f"DSO4 mediane {dso4['mediane_s'] * 1000:.0f} ms sur {dso4.get('gpu', dso4['device'])}", flush=True)

    sortie.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.regle, sortie / Path(a.regle).name)
    tab.to_csv(sortie / f"r2_par_niveau{suffixe}.csv", index=False)
    (sortie / f"coefficients_calibration{suffixe}.json").write_text(json.dumps(
        {"forme": "c = a + b * c_reseau par niveau ASL, [composante u, v][niveau] -> (a, b)",
         "niveaux_m": [int(z) for z in geom.z_levels], "plis": plis_info, **coef}, indent=1), encoding="utf-8")
    (sortie / f"dso4{suffixe}.json").write_text(json.dumps(dso4, indent=2), encoding="utf-8")
    bilan = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "regle": Path(a.regle).name, "regle_sha256": h, "run": run.name, "jeu": "validation",
             "instants": len(ds), "seuil_r2": seuil, "resume": resume,
             "post_traitement_retenu_pour_le_test": retenu, "dso4": dso4,
             **({"essai": "essai rapide, pas un resultat"} if a.max_instants else {}),
             "note": "verite v3 = verite d'entrainement (generee), jamais des mesures ; test non lu"}
    temoin.write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    L = [f"# R2 par niveau — {run.name} (validation, {len(ds)} instants)", "",
         f"Regle `{Path(a.regle).name}` (`{h[:12]}…`). Seuil R2 u, v, w >= {seuil} a chaque niveau evalue.", "",
         "| Predicteur | couples R2 >= seuil | R2 min (comp., niveau) | RMSE max u, v |", "|---|---|---|---|"]
    for k, v in resume.items():
        L.append(f"| {k} | {v['couples_r2_ok']}/{v['couples_evalues']} | {v['r2_min']:.4f} "
                 f"({v['r2_min_composante']}, {v['r2_min_niveau_m']} m) | {v['rmse_max_uv_niveaux_evalues']:.3f} |")
    L += ["", f"Post-traitement retenu pour le test : **{retenu}**", "",
          f"DSO4 : mediane {dso4['mediane_s'] * 1000:.0f} ms (moyenne {dso4['moyenne_s'] * 1000:.0f}, p95 "
          f"{dso4['p95_s'] * 1000:.0f}) sur {dso4.get('gpu', dso4['device'])}, seuil {dso4['seuil_s']} s", "",
          "```", tab.round(4).to_string(index=False), "```"]
    (sortie / f"bilan{suffixe}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))
    print(f"resultats -> {sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
