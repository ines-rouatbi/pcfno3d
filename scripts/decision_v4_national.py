"""Decision v4 nationale selon configs/national/regle_v4_national.json (empreinte verifiee avant toute lecture).

La v4 (graine choisie) et le modele livre (v2, evalue sur la verite v3 dans runs/national_1km_v2_sur_verite_v3)
sont compares sur la MEME verite v3 (dso_hauteur_sol : DSO1, DSO3, DSO5 ; DSO2 ; verdict-national : DSO4),
puis sur les mesures reelles (aeroports station exclue, radiosondages 0-1 km). Aucun seuil choisi ici.

    python scripts/decision_v4_national.py --v4 <run v4> --livre <run v2 sur v3> [--sortie <dossier>]
Code de sortie : 0 la v4 remplace le modele livre, 3 sinon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REGLE = REPO / "configs" / "national" / "regle_v4_national.json"


def verifier_regle() -> str:
    h = hashlib.sha256(REGLE.read_bytes()).hexdigest()
    attendu = (REGLE.parent / (REGLE.name + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"regle modifiee depuis son hachage : {h} != {attendu}")
    return h


def lire(run: Path) -> dict:
    d = run / "dso_hauteur_sol"
    vb = json.loads((d / "bilan.json").read_text(encoding="utf-8"))["verdicts"]
    b, be = vb["modele"], vb.get("era5", {})
    b25 = json.loads((d / "bilan_keep25.json").read_text(encoding="utf-8"))["verdicts"]["modele"]
    vn = sorted(run.glob("eval_pcfno3d/*/verdict_national.json"))
    v = json.loads(vn[-1].read_text(encoding="utf-8")) if vn else {}
    vr = run / "validation_reelle" / "test"
    return {"dso1_rmse_max_uv": b["rmse_max_uv"], "dso3_rugosite": b["dso3_rugosite"],
            "dso5_rmse_max_uv": b25["rmse_max_uv"], "dso2_conforme": b["dso2_conforme"],
            "era5_dso1_rmse_max_uv": be.get("rmse_max_uv"), "era5_dso3_rugosite": be.get("dso3_rugosite"),
            "dso4_conforme": v.get("dso4_conforme"), "verdict_national": str(vn[-1]) if vn else None,
            "stations": pd.read_csv(vr / "stations_paires.csv"), "sondes": pd.read_csv(vr / "radiosondages_paires.csv")}


def dso7(p: pd.DataFrame, src: str = "modele") -> float:
    """moyenne sur les stations du RMSE vectoriel par station (definition du DSO7 de verdict-national)"""
    e = (p[f"u_{src}"] - p.u_obs) ** 2 + (p[f"v_{src}"] - p.v_obs) ** 2
    return float(e.groupby(p.station).mean().pow(0.5).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v4", required=True)
    ap.add_argument("--livre", required=True)
    ap.add_argument("--sortie", default=None)
    a = ap.parse_args()
    h = verifier_regle()
    V, L = lire(Path(a.v4)), lire(Path(a.livre))

    c = {
        "dso1": V["dso1_rmse_max_uv"] <= 1.01 * L["dso1_rmse_max_uv"],
        "dso3": abs(V["dso3_rugosite"] - 1) <= abs(L["dso3_rugosite"] - 1) + 0.02,
        "dso5": V["dso5_rmse_max_uv"] <= 1.01 * L["dso5_rmse_max_uv"],
        "dso2_dso4": bool(V["dso2_conforme"]) and bool(V["dso4_conforme"]),
        # le candidat doit faire mieux que l'ERA5 seule sur la meme verite (predicteur era5 de son bilan)
        "mieux_que_era5_dso1": V["era5_dso1_rmse_max_uv"] is not None
        and V["dso1_rmse_max_uv"] < V["era5_dso1_rmse_max_uv"],
        "mieux_que_era5_dso3": V["era5_dso3_rugosite"] is not None
        and abs(V["dso3_rugosite"] - 1) < abs(V["era5_dso3_rugosite"] - 1),
    }
    # aeroports : memes paires (station, instant) pour les deux modeles
    cle = ["station", "temps"]
    sv, sl = V["stations"].dropna(subset=["u_modele", "v_modele"]), L["stations"].dropna(subset=["u_modele", "v_modele"])
    m = sv.merge(sl[cle + ["u_modele", "v_modele"]], on=cle, suffixes=("", "_livre"))
    pl = m.assign(u_modele=m.u_modele_livre, v_modele=m.v_modele_livre)
    a_v4, a_livre = dso7(m), dso7(pl)
    inst = m.temps.unique()
    rng = np.random.default_rng(2030)
    groupes = {t: g for t, g in m.groupby("temps")}
    boot = []
    for _ in range(5000):
        tir = rng.choice(inst, size=len(inst), replace=True)
        q = pd.concat([groupes[t] for t in tir], ignore_index=True)
        boot.append(dso7(q) - dso7(q.assign(u_modele=q.u_modele_livre, v_modele=q.v_modele_livre)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    c["aeroports"] = a_v4 <= a_livre
    # radiosondages : memes sondages et hauteurs, 0-1 km
    k = ["temps", "h_agl_m"]
    rv, rl = V["sondes"], L["sondes"]
    rm = rv.merge(rl[k + ["u_modele", "v_modele"]], on=k, suffixes=("", "_livre")).dropna(
        subset=["u_obs", "v_obs", "u_modele", "v_modele", "u_modele_livre", "v_modele_livre"])
    rm = rm[(rm.h_agl_m >= 0) & (rm.h_agl_m <= 1000)]
    r_v4 = float(np.sqrt(((rm.u_modele - rm.u_obs) ** 2 + (rm.v_modele - rm.v_obs) ** 2).mean()))
    r_livre = float(np.sqrt(((rm.u_modele_livre - rm.u_obs) ** 2 + (rm.v_modele_livre - rm.v_obs) ** 2).mean()))
    r_era5 = float(np.sqrt(((rm.u_era5 - rm.u_obs) ** 2 + (rm.v_era5 - rm.v_obs) ** 2).mean())) if "u_era5" in rm else None
    a_era5 = dso7(m, "era5") if "u_era5" in m else None
    c["radiosondages"] = r_v4 <= 1.03 * r_livre
    remplace = all(bool(x) for x in c.values())

    res = {"regle": "configs/national/regle_v4_national.json", "sha256_regle": h,
           "meme_verite_v3": {
               "dso1_rmse_max_uv": {"candidat": V["dso1_rmse_max_uv"], "livre": L["dso1_rmse_max_uv"],
                                    "era5_seule": V["era5_dso1_rmse_max_uv"]},
               "dso3_rugosite": {"candidat": V["dso3_rugosite"], "livre": L["dso3_rugosite"],
                                 "era5_seule": V["era5_dso3_rugosite"]},
               "dso5_rmse_max_uv": {"candidat": V["dso5_rmse_max_uv"], "livre": L["dso5_rmse_max_uv"]},
               "dso2_conforme": {"candidat": V["dso2_conforme"], "livre": L["dso2_conforme"]},
               "dso4_conforme": {"candidat": V["dso4_conforme"], "livre": L["dso4_conforme"]}},
           "aeroports": {"paires": int(len(m)), "instants": int(len(inst)), "candidat": a_v4, "livre": a_livre,
                         "era5_seule": a_era5, "ecart_candidat_livre": a_v4 - a_livre, "ic95": [float(lo), float(hi)]},
           "radiosondages_0_1km": {"paires": int(len(rm)), "candidat": r_v4, "livre": r_livre, "era5_seule": r_era5},
           "conditions": {k_: bool(v_) for k_, v_ in c.items()},
           "le_candidat_remplace_le_modele_livre": remplace, "candidat": Path(a.v4).name,
           "modele_livre_national": Path(a.v4).name if remplace else "national_1km_v2"}
    out = Path(a.sortie) if a.sortie else Path(a.v4).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision_v4_national.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str),
                                                   encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    return 0 if remplace else 3


if __name__ == "__main__":
    raise SystemExit(main())
