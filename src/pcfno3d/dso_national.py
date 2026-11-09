"""DSO du modele NATIONAL (seuils : `config.DSONationalCfg`).

Le cahier des charges definit les DSO pour le Grand Tunis (reseau ANEO dense). Le domaine
national (12 aeroports sur 828 x 388 km, 16 niveaux jusqu'a 2500 m, couverture de 24 % a 25 m)
demande d'autres criteres, fixes le 09/11/2026 apres le test de la national v2 (ecrit dans le
verdict). Ils completent les DSO du cahier, qui restent calcules par `evaluate`.

    DSO1a  RMSE u, v <= max(0,5 ; 10 % de la vitesse moyenne) a chaque niveau <= 1000 m
    DSO1b  gain sur ERA5 interpole >= 0 a chaque niveau <= 1000 m (la valeur ajoutee du modele)
    DSO1c  R2 u, v >= 0,95 a chaque niveau <= 1000 m (R2 de w rapporte : w vient de la
           continuite, DSO2)
    DSO2   residu de masse (cahier, inchange)
    DSO3   rugosite dans [0,9 ; 1,1], moyenne des niveaux couvrant >= 50 % du domaine
    DSO4   temps d'inference (cahier, inchange)
    DSO5   degradation <= 10 % du RMSE max entre 100 % et 25 % des capteurs
    DSO6   fiabilite (cahier, inchange)
    DSO7   RMSE vectoriel moyen aux aeroports (station retiree de l'entree) <= ERA5 seul

ERA5 interpole = les deux derniers canaux d'entree du modele (vent ERA5 aux noeuds, normalise
comme la cible), w par continuite : memes instants, capteurs et noeuds que le modele (une
seule lecture du loader, `evaluate.metriques_loader`).

Sortie : run_dir/eval_pcfno3d/<modele>/verdict_national.json et .md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config

FIXE_LE = "2026-11-09 (apres le test de la national v2, avant celui de la v3)"


def predicteur_era5(geom, norm):
    """ERA5 interpole aux noeuds, au format de `metriques_loader` (u, v ; w par continuite)."""
    from .dataset import assemble_input

    if geom.interp_era5 is None:
        raise ValueError("ERA5 interpole : paths.forcage_era5 requis")

    @torch.no_grad()
    def predire(s_raw, sensor, tfeat, y, j, stem):
        x = assemble_input(s_raw, sensor, tfeat, geom, norm)
        u = x[:, -2] * norm.y_sd[0] + norm.y_mu[0]
        v = x[:, -1] * norm.y_sd[1] + norm.y_mu[1]
        valid = geom.valid_t[None]
        u, v = torch.where(valid, u, 0.0), torch.where(valid, v, 0.0)
        return [u, v, geom.w_continuite(u, v)]

    return predire


def rmse_vect(levels: pd.DataFrame) -> pd.Series:
    return np.sqrt(levels["rmse_u"] ** 2 + levels["rmse_v"] ** 2)


def juger(niveaux_modele: pd.DataFrame, niveaux_era5: pd.DataFrame, summary: dict,
          dso6: dict | None, stations: pd.DataFrame | None, cfg: Config) -> dict:
    """Verdicts nationaux. `summary` : test_summary.json du modele (DSO2, DSO4, robustesse) ;
    `stations` : stations_bilan.csv de la validation reelle (sources modele et era5)."""
    n, cover = cfg.dso_national, cfg.dso.min_level_cover
    m = niveaux_modele.set_index("z_m")
    e = niveaux_era5.set_index("z_m")
    t = pd.DataFrame({
        "couverture": m["couverture"], "vitesse_moyenne": m["vitesse_moyenne"],
        "rmse_u": m["rmse_u"], "rmse_v": m["rmse_v"], "rmse_vect": rmse_vect(m),
        "rmse_vect_era5": rmse_vect(e), "r2_u": m["r2_u"], "r2_v": m["r2_v"], "r2_w": m["r2_w"],
        "rugosite": m["rugosite"]})
    t["erreur_relative"] = t["rmse_vect"] / t["vitesse_moyenne"]
    t["seuil_dso1a"] = np.maximum(n.plancher, n.rel * t["vitesse_moyenne"])
    t["gain_era5"] = 1.0 - t["rmse_vect"] / t["rmse_vect_era5"]
    t = t.reset_index()
    juges = t[(t["couverture"] >= cover) & (t["z_m"] <= n.z_max_m)]
    hauts = t[(t["couverture"] >= cover) & (t["z_m"] > n.z_max_m)]

    d1a = bool((juges[["rmse_u", "rmse_v"]].max(axis=1) <= juges["seuil_dso1a"]).all())
    d1b = bool((juges["gain_era5"] >= n.gain_era5_min).all())
    d1c = bool((juges[["r2_u", "r2_v"]].min(axis=1) >= n.r2).all())

    rug = t[t["couverture"] >= n.dso3_couverture_min]["rugosite"].dropna()
    rug_moy = float(rug.mean()) if len(rug) else float("nan")
    lo, hi = cfg.dso.roughness_range
    d3 = bool(lo <= rug_moy <= hi)

    rob = pd.DataFrame(summary.get("robustesse", []))
    def rmse_max(keep):
        r = rob[np.isclose(rob["capteurs_visibles"], keep)] if len(rob) else rob
        return float(r[["rmse_u_max", "rmse_v_max"]].max(axis=1).iloc[0]) if len(r) else float("nan")
    r25, r100 = rmse_max(0.25), rmse_max(1.0)
    degr = r25 / r100 - 1.0 if r100 > 0 else float("nan")
    d5 = bool(degr <= n.dso5_degradation_max)

    out = {
        "fixe_le": FIXE_LE,
        "seuils": {k: getattr(n, k) for k in n.__dataclass_fields__},
        "par_niveau": t.round(4).to_dict(orient="records"),
        "dso1a_rmse_relatif": d1a,
        "dso1b_gain_era5": d1b,
        "dso1b_gain_min": float(juges["gain_era5"].min()) if len(juges) else float("nan"),
        "dso1c_r2_uv": d1c,
        "dso1_conforme": d1a and d1b and d1c,
        "au_dessus_de_z_max": {"erreur_relative_max": float(hauts["erreur_relative"].max())
                               if len(hauts) else None,
                               "gain_era5_min": float(hauts["gain_era5"].min())
                               if len(hauts) else None},
        "dso2_conforme": bool(summary["dso2_conforme"]),
        "dso2_rms_moyen": summary["dso2_rms_moyen"],
        "dso3_rugosite": rug_moy,
        "dso3_conforme": d3,
        "dso4_inference_s": summary["dso4_inference_s"],
        "dso4_conforme": bool(summary["dso4_conforme"]),
        "dso5_rmse_max_100pct": r100, "dso5_rmse_max_25pct": r25, "dso5_degradation": degr,
        "dso5_conforme": d5,
        "dso6_conforme": bool(dso6["dso6_conforme"]) if dso6 else None,
    }
    if stations is not None and len(stations):
        s = stations.pivot_table(index="station", columns="source", values="rmse_vect")
        moy_m, moy_e = float(s["modele"].mean()), float(s["era5"].mean())
        out.update({"dso7_rmse_vect_modele": moy_m, "dso7_rmse_vect_era5": moy_e,
                    "dso7_stations_meilleures_que_era5": int((s["modele"] < s["era5"]).sum()),
                    "dso7_stations": int(len(s)),
                    "dso7_conforme": bool(moy_m <= n.stations_rapport_max * moy_e)})
    else:
        out["dso7_conforme"] = None
    cles = [k for k in ("dso1_conforme", "dso2_conforme", "dso3_conforme", "dso4_conforme",
                        "dso5_conforme", "dso6_conforme", "dso7_conforme")]
    juges_ok = [out[k] for k in cles if out[k] is not None]
    out["conformes"] = f"{sum(juges_ok)}/{len(juges_ok)}"
    return out


def rapport_md(v: dict) -> str:
    def ok(b):
        return "—" if b is None else ("CONFORME" if b else "NON CONFORME")
    s = v["seuils"]
    L = [f"# Verdict DSO nationaux ({v['conformes']} conformes)", "",
         f"DSO nationaux fixes le {v['fixe_le']}.", "",
         "| Niveau (m) | couverture | vitesse moy. | RMSE u | RMSE v | seuil DSO1a | err. relative "
         "| RMSE vect. ERA5 | gain / ERA5 | R2 u | R2 v | R2 w | rugosite |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in v["par_niveau"]:
        L.append(f"| {int(r['z_m'])} | {r['couverture']:.0%} | {r['vitesse_moyenne']:.2f} | "
                 f"{r['rmse_u']:.3f} | {r['rmse_v']:.3f} | {r['seuil_dso1a']:.3f} | "
                 f"{r['erreur_relative']:.1%} | {r['rmse_vect_era5']:.3f} | {r['gain_era5']:+.1%} | "
                 f"{r['r2_u']:.3f} | {r['r2_v']:.3f} | {r['r2_w']:.3f} | {r['rugosite']:.3f} |")
    L += ["",
          f"- DSO1a RMSE u, v <= max({s['plancher']} ; {s['rel']:.0%} vitesse), <= {s['z_max_m']:.0f} m : "
          f"{ok(v['dso1a_rmse_relatif'])}",
          f"- DSO1b gain sur ERA5 interpole >= {s['gain_era5_min']:.0%}, <= {s['z_max_m']:.0f} m "
          f"(min {v['dso1b_gain_min']:+.1%}) : {ok(v['dso1b_gain_era5'])}",
          f"- DSO1c R2 u, v >= {s['r2']} : {ok(v['dso1c_r2_uv'])}",
          f"- DSO2 residu de masse {v['dso2_rms_moyen']:.2e} : {ok(v['dso2_conforme'])}",
          f"- DSO3 rugosite (niveaux couvrant >= {s['dso3_couverture_min']:.0%}) "
          f"{v['dso3_rugosite']:.3f} : {ok(v['dso3_conforme'])}",
          f"- DSO4 inference {v['dso4_inference_s'] * 1000:.0f} ms : {ok(v['dso4_conforme'])}",
          f"- DSO5 degradation 100 % -> 25 % des capteurs {v['dso5_degradation']:+.1%} "
          f"(max {s['dso5_degradation_max']:.0%}) : {ok(v['dso5_conforme'])}",
          f"- DSO6 fiabilite : {ok(v['dso6_conforme'])}"]
    if v.get("dso7_conforme") is not None:
        L.append(f"- DSO7 aeroports : modele {v['dso7_rmse_vect_modele']:.3f} contre ERA5 "
                 f"{v['dso7_rmse_vect_era5']:.3f} m/s ({v['dso7_stations_meilleures_que_era5']}/"
                 f"{v['dso7_stations']} stations meilleures) : {ok(v['dso7_conforme'])}")
    else:
        L.append("- DSO7 aeroports : validation reelle absente")
    h = v["au_dessus_de_z_max"]
    if h["erreur_relative_max"] is not None:
        L.append(f"- (rapporte) au-dessus de {s['z_max_m']:.0f} m : erreur relative max "
                 f"{h['erreur_relative_max']:.1%}, gain ERA5 min {h['gain_era5_min']:+.1%}")
    return "\n".join(L) + "\n"


def verdict_national(cfg: Config, checkpoint: str | Path | None = None,
                     device: str | None = None, verbose: bool = True) -> dict:
    """Relit le test synoptique (modele ET ERA5 interpole, une seule lecture), puis juge avec
    test_summary.json, le DSO6 et la validation reelle deja produits."""
    from .dataset import eval_loader, is_synoptic
    from .evaluate import Evaluator, metriques_loader

    ev = Evaluator(cfg, checkpoint=checkpoint, device=device, verbose=False)
    f = ev.out_dir / "test_summary.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} absent : lancer `evaluate` d'abord")
    summary = json.loads(f.read_text(encoding="utf-8"))
    d6 = next(iter(sorted(ev.out_dir.glob("dso6_*.json"))), None)
    dso6 = json.loads(d6.read_text(encoding="utf-8")) if d6 else None
    sb = cfg.paths.run_dir / "validation_reelle" / "test" / "stations_bilan.csv"
    stations = pd.read_csv(sb) if sb.exists() else None

    te_s = ev.splits.test[0]
    syn = np.array([is_synoptic(ev.splits.stamp_of[s]) for s in te_s])
    idx = np.nonzero(syn)[0] if syn.any() else np.arange(len(te_s))
    res = metriques_loader(*eval_loader(cfg, ev.geom, ev.splits, idx, 2000), ev.geom,
                           ev.splits.stamp_of,
                           {"modele": ev.predire, "era5": predicteur_era5(ev.geom, ev.norm)},
                           ev.loss_fn.roughness_ratio)
    v = juger(res["modele"][0], res["era5"][0], summary, dso6, stations, cfg)
    v["modele"] = ev.wf.checkpoint_path.name
    v["epoque"] = (ev.wf.epoch + 1) if ev.wf.epoch is not None else None
    (ev.out_dir / "verdict_national.json").write_text(
        json.dumps(v, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (ev.out_dir / "verdict_national.md").write_text(rapport_md(v), encoding="utf-8")
    if verbose:
        print(rapport_md(v))
        print(f"resultats -> {ev.out_dir / 'verdict_national.json'}")
    return v
