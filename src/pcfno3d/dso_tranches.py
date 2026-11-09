"""DSO1, R2, DSO3 et DSO5 par TRANCHES DE HAUTEUR AU-DESSUS DU SOL (modele national).

Regle (docs/regle_dso_0_1km.md, ecrite le 09/11/2026) : les DSO sont evalues de 0 a 1 km
au-dessus du sol, comme le demande le document du 25/01/2026 (ch. 1) ; les niveaux plus hauts
sont publies a titre d'information. Les niveaux du modele sont en altitude (au-dessus de la mer,
16 niveaux de 25 a 2500 m) : chaque noeud est range selon sa hauteur au-dessus du relief
(z_niveau - z_terrain du noeud), puis les metriques sont cumulees par tranche.

Tranches par defaut (m au-dessus du sol) : 0-100, 100-300, 300-600, 600-1000 ; plus > 1000
(information). Memes instants, capteurs et noeuds pour le modele et pour ERA5 interpole sur la
meme verite (une seule lecture du test, comme `evaluate.metriques_loader`).

    DSO1  RMSE u, v <= dso.rmse (0,5 m/s) dans chaque tranche ; critere OMM rapporte
    R2    R2 u, v >= dso.r2 (0,95) ; R2 de w rapporte
    DSO3  rugosite prediction / verite dans dso.roughness_range, paires de noeuds voisins
          toutes deux dans la tranche (moyenne sur les instants)
    DSO5  RMSE max u, v a 25 % des capteurs <= dso.rmse_25pct (0,7, cahier) ; degradation
          100 % -> 25 % rapportee (DSO5 national : <= 10 %)

Sortie : run_dir/eval_pcfno3d/<modele>/dso_tranches.json et .md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config

TRANCHES_DEFAUT = (0.0, 100.0, 300.0, 600.0, 1000.0)
FIXE_LE = "2026-11-09 (apres les tests de la national v2 et v3 : publie avec le verdict par niveau)"
ROBUSTESSE_INSTANTS = 40


def masques_tranches(geom, bornes) -> list[tuple[str, np.ndarray]]:
    """[(nom, masque (nz, ny, nx))] : noeuds valides dont la hauteur au-dessus du sol est dans
    [bas, haut[ ; la derniere entree (> derniere borne) est l'information hors domaine."""
    agl = np.asarray(geom.z_levels, float)[:, None, None] - np.asarray(geom.zt, float)[None]
    valid = np.asarray(geom.valid, bool)
    out = []
    for lo, hi in zip(bornes[:-1], bornes[1:]):
        out.append((f"{lo:g}-{hi:g} m", valid & (agl >= lo) & (agl < hi)))
    out.append((f"> {bornes[-1]:g} m (information)", valid & (agl >= bornes[-1])))
    return out


@torch.no_grad()
def metriques_tranches(loader, stems, geom, predicteurs: dict, tranches) -> dict[str, pd.DataFrame]:
    """{predicteur: table par tranche} : n, rmse u/v/w, R2 u/v/w, vitesse moyenne, rugosite."""
    dev = geom.device
    masques = [(nom, torch.from_numpy(m).to(dev)) for nom, m in tranches]
    gx_ok = [(m[:, :, 1:] & m[:, :, :-1] & geom.grad_mask_x_t) for _, m in masques]
    gy_ok = [(m[:, 1:, :] & m[:, :-1, :] & geom.grad_mask_y_t) for _, m in masques]
    nb = len(masques)
    acc = {p: np.zeros((nb, 3, 4)) for p in predicteurs}      # n, sse, somme, somme2
    vit = np.zeros(nb)
    rug = {p: [[] for _ in range(nb)] for p in predicteurs}

    def grad(a, b):
        gx = (a[0, :, :, 1:] - a[0, :, :, :-1]).abs()
        gy = (a[0, :, 1:, :] - a[0, :, :-1, :]).abs()
        return gx, gy

    for s_raw, sensor, tfeat, y, j in loader:
        s_raw, sensor, tfeat, y = (t.to(dev) for t in (s_raw, sensor, tfeat, y))
        stem = stems[int(j)]
        connu = torch.isfinite(y[0, 0]) & torch.isfinite(y[0, 1])
        for b, (_, m) in enumerate(masques):
            mb = m & connu
            vit[b] += torch.hypot(y[0, 0][mb], y[0, 1][mb]).double().sum().item()
        gxt_u, gyt_u = grad(y[:, 0], None)
        gxt_v, gyt_v = grad(y[:, 1], None)
        complet = bool(torch.isfinite(y[:, :2]).all())
        for nom, predire in predicteurs.items():
            u, v, w = predire(s_raw, sensor, tfeat, y, int(j), stem)
            for b, (_, m) in enumerate(masques):
                for c, p in enumerate((u, v, w)):
                    t = y[0, c]
                    mb = m & torch.isfinite(t)
                    if mb.sum() == 0:
                        continue
                    pk, tk = p[0][mb].double(), t[mb].double()
                    acc[nom][b, c] += np.array([tk.numel(), ((pk - tk) ** 2).sum().item(),
                                                tk.sum().item(), (tk ** 2).sum().item()])
            if complet:
                gxu, gyu = grad(u, None)
                gxv, gyv = grad(v, None)
                for b in range(nb):
                    mx, my = gx_ok[b], gy_ok[b]
                    if mx.sum() == 0 or my.sum() == 0:
                        continue
                    num = (gxu[mx].mean() + gyu[my].mean() + gxv[mx].mean() + gyv[my].mean())
                    den = (gxt_u[mx].mean() + gyt_u[my].mean() + gxt_v[mx].mean() + gyt_v[my].mean())
                    if den > 0:
                        rug[nom][b].append(float(num / den))
    out = {}
    for nom in predicteurs:
        lignes = []
        for b, (tn, _) in enumerate(tranches):
            rec = {"tranche": tn}
            n_u = acc[nom][b, 0, 0]
            rec["noeuds_par_instant"] = int(round(n_u / max(len(stems), 1)))
            rec["vitesse_moyenne"] = float(vit[b] / n_u) if n_u else float("nan")
            for c, cn in enumerate("uvw"):
                n, sse, st, st2 = acc[nom][b, c]
                if n == 0:
                    rec[f"rmse_{cn}"] = rec[f"r2_{cn}"] = float("nan")
                    continue
                sst = st2 - st ** 2 / n
                rec[f"rmse_{cn}"] = float(np.sqrt(sse / n))
                rec[f"r2_{cn}"] = float(1 - sse / sst) if sst > 0 else float("nan")
            rec["rugosite"] = float(np.mean(rug[nom][b])) if rug[nom][b] else float("nan")
            lignes.append(rec)
        out[nom] = pd.DataFrame(lignes)
    return out


def juger(t: pd.DataFrame, cfg: Config, t25: pd.DataFrame | None = None,
          t100: pd.DataFrame | None = None) -> dict:
    """Verdicts par tranche (sauf la derniere, information) et sur 0-1 km."""
    d = cfg.dso
    lo, hi = d.roughness_range
    t = t.copy()
    t["rmse_uv_max"] = t[["rmse_u", "rmse_v"]].max(axis=1)
    t["seuil_omm"] = np.maximum(d.omm_floor, d.omm_rel * t["vitesse_moyenne"])
    t["dso1"] = t["rmse_uv_max"] <= d.rmse
    t["omm"] = t["rmse_uv_max"] <= t["seuil_omm"]
    t["r2_uv"] = t[["r2_u", "r2_v"]].min(axis=1) >= d.r2
    t["dso3"] = t["rugosite"].between(lo, hi)
    if t25 is not None and t100 is not None:
        m25 = t25[["rmse_u", "rmse_v"]].max(axis=1).to_numpy()
        m100 = t100[["rmse_u", "rmse_v"]].max(axis=1).to_numpy()
        t["rmse_max_25pct"] = m25
        t["degradation_25pct"] = m25 / m100 - 1.0
        t["dso5_cahier"] = t["rmse_max_25pct"] <= d.rmse_25pct
        t["dso5_national"] = t["degradation_25pct"] <= cfg.dso_national.dso5_degradation_max
    dom = t.iloc[:-1]
    return {"par_tranche": t.round(4).to_dict(orient="records"),
            "domaine_0_1km": {
                "dso1_conforme": bool(dom["dso1"].all()),
                "critere_omm": bool(dom["omm"].all()),
                "r2_uv_conforme": bool(dom["r2_uv"].all()),
                "dso3_conforme": bool(dom["dso3"].all()),
                "dso5_cahier_conforme": bool(dom["dso5_cahier"].all()) if "dso5_cahier" in dom else None,
                "dso5_national_conforme": bool(dom["dso5_national"].all()) if "dso5_national" in dom else None,
                "rmse_uv_max": float(dom["rmse_uv_max"].max())}}


def rapport_md(v: dict) -> str:
    L = [f"# DSO par tranches au-dessus du sol — {v['poids']} (epoque {v['epoque']})", "",
         f"Regle : DSO evalues de 0 a 1 km au-dessus du sol (document du 25/01/2026, ch. 1), "
         f"ecrite le {v['fixe_le']}. Le verdict par niveau (au-dessus de la mer) reste publie.", ""]
    for src in ("modele", "era5"):
        if src not in v:
            continue
        L += [f"## {'Modele' if src == 'modele' else 'ERA5 interpole (meme verite)'}", "",
              "| Tranche | noeuds/instant | vitesse moy. | RMSE u | RMSE v | RMSE w | R2 u | R2 v | R2 w "
              "| rugosite | RMSE max 25 % | degradation |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in v[src]["par_tranche"]:
            L.append(f"| {r['tranche']} | {r['noeuds_par_instant']} | {r['vitesse_moyenne']:.2f} | "
                     f"{r['rmse_u']:.3f} | {r['rmse_v']:.3f} | {r['rmse_w']:.3f} | {r['r2_u']:.3f} | "
                     f"{r['r2_v']:.3f} | {r['r2_w']:.3f} | {r['rugosite']:.3f} | "
                     f"{r.get('rmse_max_25pct', float('nan')):.3f} | {r.get('degradation_25pct', float('nan')):+.1%} |")
        d = v[src]["domaine_0_1km"]
        L += ["", f"0-1 km : DSO1 {'CONFORME' if d['dso1_conforme'] else 'NON CONFORME'} "
                  f"(RMSE max {d['rmse_uv_max']:.3f} m/s) ; R2 u, v "
                  f"{'CONFORME' if d['r2_uv_conforme'] else 'NON CONFORME'} ; DSO3 "
                  f"{'CONFORME' if d['dso3_conforme'] else 'NON CONFORME'} ; DSO5 cahier "
                  f"{d['dso5_cahier_conforme']} ; DSO5 national {d['dso5_national_conforme']}", ""]
    return "\n".join(L) + "\n"


def dso_tranches(cfg: Config, checkpoint: str | Path | None = None, device: str | None = None,
                 bornes=TRANCHES_DEFAUT, verbose: bool = True) -> dict:
    from .dataset import eval_loader, is_synoptic
    from .dso_national import predicteur_era5
    from .evaluate import Evaluator

    ev = Evaluator(cfg, checkpoint=checkpoint, device=device, verbose=False)
    geom = ev.geom
    tranches = masques_tranches(geom, [float(b) for b in bornes])
    preds = {"modele": ev.predire}
    if geom.interp_era5 is not None:
        preds["era5"] = predicteur_era5(geom, ev.norm)
    te_s = ev.splits.test[0]
    syn = np.array([is_synoptic(ev.splits.stamp_of[s]) for s in te_s])
    idx = np.nonzero(syn)[0] if syn.any() else np.arange(len(te_s))
    res = metriques_tranches(*eval_loader(cfg, geom, ev.splits, idx, 2000), geom, preds, tranches)
    # DSO5 : memes 40 instants et graine que `evaluate` (robustesse), 25 % puis 100 % des capteurs
    sub = idx[np.unique(np.linspace(0, len(idx) - 1, min(ROBUSTESSE_INSTANTS, len(idx))).astype(int))]
    rob = {k: metriques_tranches(*eval_loader(cfg, geom, ev.splits, sub, 3000, keep=k), geom,
                                 preds, tranches) for k in (0.25, 1.0)}
    v = {"poids": ev.wf.checkpoint_path.name, "fixe_le": FIXE_LE, "bornes_m": [float(b) for b in bornes]}
    for src in preds:
        v[src] = juger(res[src], cfg, rob[0.25][src], rob[1.0][src])
    v["epoque"] = (ev.wf.epoch + 1) if ev.wf.epoch is not None else None
    (ev.out_dir / "dso_tranches.json").write_text(json.dumps(v, indent=2, ensure_ascii=False, default=float),
                                                  encoding="utf-8")
    (ev.out_dir / "dso_tranches.md").write_text(rapport_md(v), encoding="utf-8")
    if verbose:
        print(rapport_md(v))
        print(f"resultats -> {ev.out_dir / 'dso_tranches.json'}")
    return v
