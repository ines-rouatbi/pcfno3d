"""DSO evalues en HAUTEUR AU-DESSUS DU SOL, de 0 a 1 km, comme le demande le document fondateur
(R. Ben Amor, « Approche de Reconstruction Atmospherique 3D », 25/01/2026, ch. 1 : « depuis le sol
jusqu'a une altitude de 1 km »). Aucun entrainement, aucun bruit ajoute.

Pourquoi : les niveaux du modele sont des altitudes au-dessus de la MER. Au national, le pire
niveau du DSO1 est 2500 m ASL, soit 1 a 2,5 km au-dessus du sol : hors du domaine demande. Et le
niveau 25 m ASL ne couvre que 24 % du territoire (il est sous le relief ailleurs). Ici chaque
noeud valide est classe par sa hauteur au-dessus du sol h = z_niveau - z_terrain(x, y), dans des
tranches FIXEES AVANT le calcul (regle ecrite et hachee a la premiere execution).

Pour chaque tranche et pour deux predicteurs vus sur EXACTEMENT les memes instants, entrees et
noeuds (une seule lecture du loader) :
  - « modele » : la passe servie (w par continuite si la config le dit) ;
  - « era5 »   : le vent ERA5 aux noeuds, l'entree du modele residuel (w par continuite) —
                 la reference a battre (seulement pour un modele avec paths.forcage_era5).
on publie RMSE et R2 de u, v, w, la vitesse moyenne, le seuil OMM max(0,5 ; 10 % de la vitesse)
(OMM CIMO n° 8 §5.1.3), la rugosite (DSO3) et le residu de masse (DSO2) restreints au domaine
0-1 km, la MAE de direction. La tranche « au-dessus de 1 km » est publiee A TITRE D'INFORMATION.
Option --keep 0.25 : meme calcul avec un quart des points d'entree (DSO5).

    python scripts/dso_hauteur_sol.py -c national_1km_colab regle      # 1) regle datee
    python scripts/dso_hauteur_sol.py -c national_1km_colab mesure     # 2) test, un passage
    python scripts/dso_hauteur_sol.py -c national_1km_colab mesure --keep 0.25   # DSO5

Sorties : <run_dir>/dso_hauteur_sol/ (regle.json + .sha256, tranches[_keepXX].csv,
bilan[_keepXX].json, bilan[_keepXX].md).

Correction du 09/11/2026 a la reprise du script : la reference ERA5 lisait tfeat[:, N_TFEAT:],
qui commence par les N_SCAL scalaires du forcage ; les cellules ERA5 commencent a
N_TFEAT + N_SCAL (forcage.ForcageERA5.vecteur, tests/test_forcage_era5.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

DOSSIER = "dso_hauteur_sol"
BORNES = [0.0, 100.0, 300.0, 600.0, 1000.0]        # tranches PRE-ENREGISTREES (m au-dessus du sol)
DOMAINE_MAX = 1000.0
PART_MIN = 0.01        # tranche de moins de 1 % des noeuds du domaine : publiee sans verdict

REGLE = {
    "objet": "DSO evalues en hauteur au-dessus du sol, domaine 0-1 km",
    "source": "R. Ben Amor, Approche de Reconstruction Atmospherique 3D, 25/01/2026, chapitre 1 : "
              "« depuis le sol jusqu'a une altitude de 1 km » ; z compte depuis le sol (§4.4 : "
              "w(x, y, 0) = 0)",
    "hauteur": "h = z_niveau (ASL) - z_terrain(x, y) du modele, noeuds valides seulement",
    "tranches_m": BORNES,
    "domaine_dso_m": [0.0, DOMAINE_MAX],
    "information": f"tranche > {DOMAINE_MAX:g} m publiee sans verdict",
    "verdicts": {
        "DSO1": "RMSE u et v <= dso.rmse dans CHAQUE tranche du domaine ; R2 u, v >= dso.r2",
        "DSO1_omm": "RMSE u, v <= max(dso.omm_floor ; dso.omm_rel x vitesse moyenne de la tranche) "
                    "(OMM CIMO n° 8 §5.1.3), publie a cote",
        "R2_uvw": "R2 u, v, w >= dso.r2_all dans chaque tranche",
        "DSO2": "RMS du residu de masse sur mass_mask restreint au domaine <= dso.mass_rms",
        "DSO3": "rugosite prediction / verite (gradients horizontaux entre deux noeuds du domaine) "
                "dans dso.roughness_range",
        "DSO5": "meme calcul avec --keep 0.25, RMSE max <= dso.rmse_25pct",
        "reference": "ERA5 aux noeuds (entree du modele residuel), memes instants et noeuds",
    },
    "seuils": "ceux de cfg.dso, inchanges",
    "jeu": "test, instants synoptiques, graine 2000 (memes entrees que DSO1)",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def maintenant() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def masques_tranches(geom):
    """{nom: masque (nz, ny, nx) bool sur le device} + hauteur au-dessus du sol (numpy)."""
    import torch

    h = np.asarray(geom.z_levels, float)[:, None, None] - np.asarray(geom.zt, float)[None]
    valid = np.asarray(geom.valid, bool)
    tranches = {}
    for a, b in zip(BORNES[:-1], BORNES[1:]):
        tranches[f"{a:g}-{b:g} m"] = valid & (h >= a) & (h < b)
    tranches[f"0-{DOMAINE_MAX:g} m (domaine)"] = valid & (h >= 0) & (h < DOMAINE_MAX)
    tranches[f"> {DOMAINE_MAX:g} m (info)"] = valid & (h >= DOMAINE_MAX)
    return {k: torch.from_numpy(m).to(geom.device) for k, m in tranches.items()}, h


def mesurer(ev, cfg, keep: float | None, max_instants: int | None) -> tuple[pd.DataFrame, dict]:
    import torch

    from pcfno3d.dataset import eval_loader, is_synoptic
    from pcfno3d.forcage import N_SCAL
    from pcfno3d.geometry import N_TFEAT

    geom, splits = ev.geom, ev.splits
    te_s = splits.test[0]
    syn = np.array([is_synoptic(splits.stamp_of[s]) for s in te_s])
    idx = np.nonzero(syn)[0] if syn.any() else np.arange(len(te_s))
    if max_instants:
        idx = idx[:max_instants]
    loader, stems = eval_loader(cfg, geom, splits, idx, 2000, keep=keep)
    stems = [str(s) for s in stems]

    tranches, _ = masques_tranches(geom)
    dom = tranches[f"0-{DOMAINE_MAX:g} m (domaine)"]
    # paires horizontales entierement dans le domaine (DSO3) ; residu de masse dans le domaine
    gx = dom[:, :, 1:] & dom[:, :, :-1]
    gy = dom[:, 1:, :] & dom[:, :-1, :]
    mm = geom.mass_mask_t & dom[1:] & dom[:-1]

    def rug(a):
        return ((a[0, :, :, 1:] - a[0, :, :, :-1]).abs()[gx].mean()
                + (a[0, :, 1:, :] - a[0, :, :-1, :]).abs()[gy].mean())

    def pred_era5(s_raw, sensor, tfeat, y, j, stem):
        # cellules ERA5 apres les N_SCAL scalaires du forcage (correction du 09/11/2026)
        uv = geom.interp_era5.dense(tfeat[:, N_TFEAT + N_SCAL:])
        u, v = uv[:, 0] * geom.valid_t, uv[:, 1] * geom.valid_t
        return [u, v, geom.w_continuite(u, v)]

    preds = {"modele": ev.predire}
    if geom.interp_era5 is not None:
        preds["era5"] = pred_era5

    acc = {p: {t: np.zeros((3, 4)) for t in tranches} for p in preds}   # c x (n, sse, s1, s2)
    spd = {t: np.zeros(2) for t in tranches}
    rug_l = {p: [] for p in preds}
    masse = {p: [] for p in preds}
    dirs = {p: [] for p in preds}
    with torch.no_grad():
        for s_raw, sensor, tfeat, y, j in loader:
            s_raw, sensor, tfeat, y = (t.to(geom.device) for t in (s_raw, sensor, tfeat, y))
            stem = stems[int(j)]
            known_uv = torch.isfinite(y[0, 0]) & torch.isfinite(y[0, 1])
            for t, m in tranches.items():
                mk = m & known_uv
                spd[t] += [mk.sum().item(), torch.hypot(y[0, 0][mk], y[0, 1][mk]).double().sum().item()]
            for p, f in preds.items():
                u, v, w = f(s_raw, sensor, tfeat, y, int(j), stem)
                for c, a in enumerate((u, v, w)):
                    tc = y[0, c]
                    for t, m in tranches.items():
                        mk = m & torch.isfinite(tc)
                        if not mk.any():
                            continue
                        pk, tk = a[0][mk].double(), tc[mk].double()
                        acc[p][t][c] += [tk.numel(), ((pk - tk) ** 2).sum().item(),
                                         tk.sum().item(), (tk ** 2).sum().item()]
                r = geom.mass_residual(y[:, 3], u, v, w)
                masse[p].append(torch.sqrt((r[:, mm] ** 2).mean()).item())
                if bool(torch.isfinite(y[:, :2]).all()):
                    rug_l[p].append(((rug(u) + rug(v)) / (rug(y[:, 0]) + rug(y[:, 1]))).item())
                md = dom & known_uv & (torch.hypot(y[0, 0], y[0, 1]) >= 1.0)
                dp = torch.rad2deg(torch.atan2(v[0], u[0]))[md]
                dt = torch.rad2deg(torch.atan2(y[0, 1], y[0, 0]))[md]
                dd = torch.abs(dp - dt) % 360
                dirs[p].append(torch.minimum(dd, 360 - dd).mean().item())

    d = cfg.dso
    rows = []
    for p in preds:
        for t in tranches:
            rec = {"predicteur": p, "tranche": t, "noeuds_par_instant": int(acc[p][t][0, 0] / len(stems)),
                   "vitesse_moyenne": spd[t][1] / max(spd[t][0], 1)}
            for c, nom in enumerate("uvw"):
                n, sse, s1, s2 = acc[p][t][c]
                rec[f"rmse_{nom}"] = float(np.sqrt(sse / n)) if n else np.nan
                sst = s2 - s1 ** 2 / n if n else 0
                rec[f"r2_{nom}"] = float(1 - sse / sst) if sst > 0 else np.nan
            rec["seuil_omm"] = max(d.omm_floor, d.omm_rel * rec["vitesse_moyenne"])
            rows.append(rec)
    tab = pd.DataFrame(rows)
    glob = {p: {"dso2_rms_moyen": float(np.mean(masse[p])),
                "dso3_rugosite": float(np.mean(rug_l[p])) if rug_l[p] else float("nan"),
                "mae_direction_deg": float(np.mean(dirs[p]))} for p in preds}
    return tab, {"instants": len(stems), "keep": keep, "global": glob}


def verdicts(tab: pd.DataFrame, glob: dict, cfg, keep: float | None) -> dict:
    from pcfno3d.dso import dso3_conforme

    d = cfg.dso
    out = {}
    for p, g in glob.items():
        tp = tab[tab.predicteur == p]
        n_dom = float(tp[tp.tranche.str.contains("domaine")].noeuds_par_instant.iloc[0])
        t = tp[tp.tranche.str.match(r"^\d") & ~tp.tranche.str.contains("domaine")
               & (tp.noeuds_par_instant >= PART_MIN * n_dom)]
        rmse_max = float(t[["rmse_u", "rmse_v"]].max(axis=1).max())
        seuil = d.rmse_25pct if keep is not None and keep < 1 else d.rmse
        out[p] = {
            "rmse_max_uv": rmse_max,
            "tranche_la_pire": t.loc[t[["rmse_u", "rmse_v"]].max(axis=1).idxmax(), "tranche"],
            "dso1_conforme" if keep is None else "dso5_conforme": bool(rmse_max <= seuil),
            "dso1_omm_conforme": bool((t[["rmse_u", "rmse_v"]].max(axis=1) <= t["seuil_omm"]).all()),
            "r2_uv_min": float(t[["r2_u", "r2_v"]].min().min()),
            "dso1_r2_conforme": bool(t[["r2_u", "r2_v"]].min().min() >= d.r2),
            "r2_uvw_min": float(t[["r2_u", "r2_v", "r2_w"]].min().min()),
            "r2_uvw_conforme": bool(t[["r2_u", "r2_v", "r2_w"]].min().min() >= d.r2_all),
            "dso2_conforme": bool(g["dso2_rms_moyen"] <= d.mass_rms),
            "dso3_conforme": dso3_conforme(g["dso3_rugosite"], d.roughness_range),
            **g,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("etape", choices=["regle", "mesure"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep", type=float, default=None, help="part des points d'entree (DSO5 : 0.25)")
    ap.add_argument("--max-instants", type=int, default=None, help="essai rapide (pas un verdict)")
    args = ap.parse_args()

    from pcfno3d.config import Config
    from pcfno3d.evaluate import Evaluator

    cfg = Config.load(args.config)
    ev = Evaluator(cfg, checkpoint=args.checkpoint, device=args.device, verbose=False)
    out = cfg.paths.run_dir / DOSSIER
    out.mkdir(parents=True, exist_ok=True)
    pr = out / "regle.json"

    if args.etape == "regle":
        if pr.exists():
            raise SystemExit(f"{pr} existe deja : la regle ne se reecrit pas apres coup.")
        pr.write_text(json.dumps(dict(REGLE, date_utc=maintenant(), config=args.config,
                                      checkpoint=str(ev.wf.checkpoint_path)),
                                 indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "regle.sha256").write_text(sha(pr), encoding="utf-8")
        _, h = masques_tranches(ev.geom)
        v = np.asarray(ev.geom.valid, bool)
        print(f"regle ecrite -> {pr}\nhauteur au-dessus du sol des noeuds valides : "
              f"{h[v].min():.0f} a {h[v].max():.0f} m ; part dans 0-1 km : "
              f"{((h[v] < DOMAINE_MAX)).mean():.1%}")
        return 0

    if not pr.exists() or sha(pr) != (out / "regle.sha256").read_text(encoding="utf-8").strip():
        raise SystemExit("regle.json absente ou modifiee : lancer d'abord l'etape 'regle'.")
    suffixe = "" if args.keep is None else f"_keep{int(round(args.keep * 100))}"
    if args.max_instants:
        suffixe += f"_essai{args.max_instants}"
    pb = out / f"bilan{suffixe}.json"
    if pb.exists():
        raise SystemExit(f"{pb} existe deja : un seul passage sur le test.")

    tab, info = mesurer(ev, cfg, args.keep, args.max_instants)
    tab.to_csv(out / f"tranches{suffixe}.csv", index=False)
    v = verdicts(tab, info["global"], cfg, args.keep)
    pb.write_text(json.dumps({"date_utc": maintenant(), "regle_sha256": sha(pr), **info,
                              "verdicts": v}, indent=2, ensure_ascii=False), encoding="utf-8")

    L = [f"# DSO en hauteur au-dessus du sol ({info['instants']} instants de test"
         + (f", {args.keep:.0%} des points d'entree" if args.keep else "") + ")", "",
         "| Predicteur | Tranche | RMSE u | RMSE v | R2 u | R2 v | R2 w | vitesse | seuil OMM |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        L.append(f"| {r.predicteur} | {r.tranche} | {r.rmse_u:.3f} | {r.rmse_v:.3f} | {r.r2_u:.3f} "
                 f"| {r.r2_v:.3f} | {r.r2_w:.3f} | {r.vitesse_moyenne:.2f} | {r.seuil_omm:.2f} |")
    L += ["", "| Verdict (domaine 0-1 km) | " + " | ".join(v) + " |", "|---|" + "---|" * len(v)]
    for k in next(iter(v.values())):
        L.append(f"| {k} | " + " | ".join(
            (f"{x:.4g}" if isinstance(x, float) else str(x)) for x in (v[p][k] for p in v)) + " |")
    L += ["", "Verite d'entrainement : champ genere (ERA5 + relief), pas des mesures.",
          f"La tranche > {DOMAINE_MAX:g} m est publiee a titre d'information, sans verdict."]
    (out / f"bilan{suffixe}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
