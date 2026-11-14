"""w servi du modele national v4 : melange w_continuite + beta (w_reseau - w_continuite), choisi sur la VALIDATION.

Regle configs/national/regle_w_servi_v4.json (empreinte .sha256 verifiee avant toute lecture des donnees).
Une passe sur les 64 instants de validation : R2 u, v, w par niveau evalue et residu de masse (DSO2, comme
dso.measure) pour chaque beta, sur la validation entiere et sur chacun des deux plis de jours. Aucun
entrainement, le test n'est jamais lu, aucun bruit ajoute.

    python scripts/w_servi_v4.py -c national_1km_colab --run <runs>/national_1km_v4_Cp_s44 \\
        --regle configs/national/regle_w_servi_v4.json

Sortie : <run>/w_servi_validation/ (dossier neuf). --max-instants N : essai (suffixe _essaiN).
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


def r2_de(acc: np.ndarray) -> np.ndarray:
    """acc (..., 4) = [n, sse, st, st2] -> R2 (NaN si variance nulle ou aucun noeud)."""
    n, sse, st, st2 = (acc[..., i] for i in range(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        sst = st2 - st * st / n
        return np.where((n > 0) & (sst > 0), 1 - sse / sst, np.nan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--run", required=True, help="run_dir de l'entrainement (best.pt + norm_stats.json)")
    ap.add_argument("--regle", default=str(REPO / "configs" / "national" / "regle_w_servi_v4.json"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-instants", type=int, default=None)
    a = ap.parse_args(argv)

    from pcfno3d.regles import verifier
    regle, h = verifier(a.regle)
    run = Path(a.run)
    suffixe = f"_essai{a.max_instants}" if a.max_instants else ""
    sortie = run / "w_servi_validation"
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
    betas = [float(b) for b in regle["candidats"]["beta"]]
    seuil_r2, seuil_masse = float(cfg.dso.r2_all), float(cfg.dso.mass_rms)

    # pli (2), beta, composante (u, v, w), niveau, [n, sse, st, st2] ; u, v identiques pour tous les beta
    acc = np.zeros((2, len(betas), 3, nz, 4))
    masse = {b: ([], []) for b in betas}            # par pli : RMS du residu par instant
    t0 = time.time()
    with torch.no_grad():
        for n, (s_raw, sensor, tfeat, y, j) in enumerate(DataLoader(ds, batch_size=1)):
            s_raw, sensor, tfeat, y = (t.to(dev) for t in (s_raw, sensor, tfeat, y))
            p_ = pli[stems[int(j)]]
            with torch.autocast(device_type=dev.type, dtype=m.amp_dtype, enabled=m.use_amp):
                pred = m.model(assemble_input(s_raw, sensor, tfeat, geom, m.norm))
            u, v, w_res = m.norm.denorm(pred.float())
            w_c = geom.w_continuite(u, v)
            for ib, b in enumerate(betas):
                w = w_c + b * (w_res - w_c)
                for ci, p in enumerate((u, v, w)):
                    t = y[0, ci]
                    known = geom.valid_t & torch.isfinite(t)
                    for k in range(nz):
                        mk = known[k]
                        if mk.sum() == 0:
                            continue
                        pk, tk = p[0, k][mk].double(), t[k][mk].double()
                        acc[p_, ib, ci, k] += [tk.numel(), ((pk - tk) ** 2).sum().item(), tk.sum().item(),
                                               (tk ** 2).sum().item()]
                r = geom.mass_residual(y[:, 3], u, v, w)
                masse[b][p_].append(torch.sqrt((r[:, geom.mass_mask_t] ** 2).mean()).item())
            if (n + 1) % 16 == 0:
                print(f"  {n + 1}/{len(ds)} ({time.time() - t0:.0f} s)", flush=True)

    import pandas as pd
    lignes, resume = [], {}
    jeux = {"validation": acc.sum(0), "pli_a": acc[0], "pli_b": acc[1]}
    for ib, b in enumerate(betas):
        res_b = {}
        for nom, A in jeux.items():
            r2 = r2_de(A[ib])                                       # (3, nz)
            ev = r2[:, levels]
            ok = int(np.sum(np.isfinite(ev) & (ev >= seuil_r2)))
            imin = np.unravel_index(np.nanargmin(ev), ev.shape)
            mm = (masse[b][0] + masse[b][1]) if nom == "validation" else masse[b][0 if nom == "pli_a" else 1]
            dso2 = float(np.mean(mm)) if mm else float("nan")
            res_b[nom] = {"couples_r2_ok": ok, "couples_evalues": 3 * len(levels), "r2_min": float(ev[imin]),
                          "r2_min_composante": "uvw"[imin[0]], "r2_min_niveau_m": int(geom.z_levels[levels[imin[1]]]),
                          "dso2_masse_rms": dso2, "dso1_r2_conforme": ok == 3 * len(levels),
                          "dso2_conforme": bool(dso2 <= seuil_masse)}
            if nom == "validation":
                for k in range(nz):
                    rec = {"beta": b, "z_m": int(geom.z_levels[k]), "evalue_dso": k in levels}
                    for ci, c in enumerate("uvw"):
                        nn, sse = A[ib, ci, k, 0], A[ib, ci, k, 1]
                        rec[f"r2_{c}"] = float(r2[ci, k])
                        rec[f"rmse_{c}"] = float(np.sqrt(sse / nn)) if nn else float("nan")
                    lignes.append(rec)
        res_b["admissible"] = all(res_b[j]["dso1_r2_conforme"] and res_b[j]["dso2_conforme"] for j in jeux)
        resume[str(b)] = res_b
    admissibles = [b for b in betas if resume[str(b)]["admissible"]]
    retenu = min(admissibles) if admissibles else 0.0
    print({b: (resume[str(b)]["validation"]["couples_r2_ok"], round(resume[str(b)]["validation"]["r2_min"], 4),
               f"{resume[str(b)]['validation']['dso2_masse_rms']:.2e}", resume[str(b)]["admissible"]) for b in betas},
          "| beta retenu :", retenu, flush=True)

    sortie.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.regle, sortie / Path(a.regle).name)
    tab = pd.DataFrame(lignes)
    tab.to_csv(sortie / f"r2_par_niveau_beta{suffixe}.csv", index=False)
    bilan = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "regle": Path(a.regle).name, "regle_sha256": h, "run": run.name, "jeu": "validation",
             "instants": len(ds), "plis": plis_info, "seuil_r2": seuil_r2, "seuil_dso2": seuil_masse,
             "resume": resume, "beta_admissibles": admissibles, "beta_retenu": retenu,
             "post_traitement_retenu_pour_le_test": ("w_continuite" if retenu == 0.0 else
                                                     f"w_continuite + {retenu} x (w_reseau - w_continuite)"),
             **({"essai": "essai rapide, pas un resultat"} if a.max_instants else {}),
             "note": "verite v3 = verite d'entrainement (generee), jamais des mesures ; test non lu"}
    temoin.write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    L = [f"# w servi — {run.name} (validation, {len(ds)} instants)", "",
         f"Regle `{Path(a.regle).name}` (`{h[:12]}…`). w = w_continuite + beta (w_reseau - w_continuite). "
         f"Seuils inchanges : R2 >= {seuil_r2} a chaque niveau evalue, DSO2 <= {seuil_masse:g}.", "",
         "| beta | couples R2 >= seuil | R2 min (comp., niveau) | DSO2 | pli a | pli b | admissible |",
         "|---|---|---|---|---|---|---|"]
    for b in betas:
        r = resume[str(b)]
        v = r["validation"]
        pa, pb = r["pli_a"], r["pli_b"]
        L.append(f"| {b} | {v['couples_r2_ok']}/{v['couples_evalues']} | {v['r2_min']:.4f} "
                 f"({v['r2_min_composante']}, {v['r2_min_niveau_m']} m) | {v['dso2_masse_rms']:.2e} | "
                 f"{pa['couples_r2_ok']}/{pa['couples_evalues']}, {pa['dso2_masse_rms']:.1e} | "
                 f"{pb['couples_r2_ok']}/{pb['couples_evalues']}, {pb['dso2_masse_rms']:.1e} | "
                 f"{'oui' if r['admissible'] else 'non'} |")
    L += ["", f"beta retenu : **{retenu}** ({bilan['post_traitement_retenu_pour_le_test']})", "",
          "```", tab[tab.evalue_dso].round(4).to_string(index=False), "```"]
    (sortie / f"bilan{suffixe}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))
    print(f"resultats -> {sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
