"""Decision D2 nationale selon configs/national/regle_d2_national.json (empreinte verifiee avant toute lecture).

Le D2 choisi (graine choisie sur la validation) et le modele livre (v2 sur la verite v3,
runs/national_1km_v2_sur_verite_v3) sont compares comme dans la regle v4 (memes huit conditions, memes
scripts : scripts/decision_v4_national.py pour la lecture et le DSO7), plus une condition propre a D2 : faire
mieux que la reference « interpolation optimale seule » (predicteur era5_oi de scripts/dso_hauteur_sol.py) sur
le DSO1 0-1 km. ERA5 seule et OI seule sont publiees partout (verite v3, aeroports, radiosondages).

    python scripts/decision_d2_national.py --d2 <run D2> --livre <run v2 sur v3> [--sortie <dossier>]
Code de sortie : 0 le D2 remplace le modele livre, 3 sinon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts")]
REGLE = REPO / "configs" / "national" / "regle_d2_national.json"


def verifier_regle() -> str:
    h = hashlib.sha256(REGLE.read_bytes()).hexdigest()
    attendu = (REGLE.parent / (REGLE.name + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"regle modifiee depuis son hachage : {h} != {attendu}")
    return h


def rmse_sondes(rm: pd.DataFrame, src: str) -> float | None:
    if f"u_{src}" not in rm:
        return None
    q = rm.dropna(subset=[f"u_{src}", f"v_{src}"])
    return float(np.sqrt(((q[f"u_{src}"] - q.u_obs) ** 2 + (q[f"v_{src}"] - q.v_obs) ** 2).mean()))


def main() -> int:
    import decision_v4_national as D4

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--d2", required=True)
    ap.add_argument("--livre", required=True)
    ap.add_argument("--sortie", default=None)
    a = ap.parse_args()
    h = verifier_regle()
    V, L = D4.lire(Path(a.d2)), D4.lire(Path(a.livre))
    vb = json.loads((Path(a.d2) / "dso_hauteur_sol" / "bilan.json").read_text(encoding="utf-8"))["verdicts"]
    oi = vb.get("era5_oi")
    if oi is None:
        raise SystemExit("bilan D2 sans reference era5_oi : le modele n'a pas le canal d'innovation")
    b25 = json.loads((Path(a.d2) / "dso_hauteur_sol" / "bilan_keep25.json").read_text(encoding="utf-8"))["verdicts"]

    c = {
        "dso1": V["dso1_rmse_max_uv"] <= 1.01 * L["dso1_rmse_max_uv"],
        "dso3": abs(V["dso3_rugosite"] - 1) <= abs(L["dso3_rugosite"] - 1) + 0.02,
        "dso5": V["dso5_rmse_max_uv"] <= 1.01 * L["dso5_rmse_max_uv"],
        "dso2_dso4": bool(V["dso2_conforme"]) and bool(V["dso4_conforme"]),
        "mieux_que_era5_dso1": V["era5_dso1_rmse_max_uv"] is not None
        and V["dso1_rmse_max_uv"] < V["era5_dso1_rmse_max_uv"],
        "mieux_que_era5_dso3": V["era5_dso3_rugosite"] is not None
        and abs(V["dso3_rugosite"] - 1) < abs(V["era5_dso3_rugosite"] - 1),
        # propre a D2 : le reseau doit apporter plus que l'etalement seul des memes innovations
        "mieux_que_oi_seule_dso1": V["dso1_rmse_max_uv"] < oi["rmse_max_uv"],
    }
    cle = ["station", "temps"]
    sv, sl = V["stations"].dropna(subset=["u_modele", "v_modele"]), L["stations"].dropna(subset=["u_modele", "v_modele"])
    m = sv.merge(sl[cle + ["u_modele", "v_modele"]], on=cle, suffixes=("", "_livre"))
    a_d2, a_livre = D4.dso7(m), D4.dso7(m.assign(u_modele=m.u_modele_livre, v_modele=m.v_modele_livre))
    inst = m.temps.unique()
    rng = np.random.default_rng(2030)
    groupes = {t: g for t, g in m.groupby("temps")}
    boot = []
    for _ in range(5000):
        q = pd.concat([groupes[t] for t in rng.choice(inst, size=len(inst), replace=True)], ignore_index=True)
        boot.append(D4.dso7(q) - D4.dso7(q.assign(u_modele=q.u_modele_livre, v_modele=q.v_modele_livre)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    c["aeroports"] = a_d2 <= a_livre
    k = ["temps", "h_agl_m"]
    rm = V["sondes"].merge(L["sondes"][k + ["u_modele", "v_modele"]], on=k, suffixes=("", "_livre")).dropna(
        subset=["u_obs", "v_obs", "u_modele", "v_modele", "u_modele_livre", "v_modele_livre"])
    rm = rm[(rm.h_agl_m >= 0) & (rm.h_agl_m <= 1000)]
    r_d2 = rmse_sondes(rm, "modele")
    r_livre = rmse_sondes(rm.assign(u_modele=rm.u_modele_livre, v_modele=rm.v_modele_livre), "modele")
    c["radiosondages"] = r_d2 <= 1.03 * r_livre
    remplace = all(bool(x) for x in c.values())
    res = {"regle": "configs/national/regle_d2_national.json", "sha256_regle": h,
           "meme_verite_v3": {
               "dso1_rmse_max_uv": {"candidat": V["dso1_rmse_max_uv"], "livre": L["dso1_rmse_max_uv"],
                                    "era5_seule": V["era5_dso1_rmse_max_uv"], "oi_seule": oi["rmse_max_uv"]},
               "dso3_rugosite": {"candidat": V["dso3_rugosite"], "livre": L["dso3_rugosite"],
                                 "era5_seule": V["era5_dso3_rugosite"], "oi_seule": oi["dso3_rugosite"]},
               "dso5_rmse_max_uv": {"candidat": V["dso5_rmse_max_uv"], "livre": L["dso5_rmse_max_uv"],
                                    "era5_seule": b25.get("era5", {}).get("rmse_max_uv"),
                                    "oi_seule": b25.get("era5_oi", {}).get("rmse_max_uv")},
               "dso2_conforme": {"candidat": V["dso2_conforme"], "livre": L["dso2_conforme"]},
               "dso4_conforme": {"candidat": V["dso4_conforme"], "livre": L["dso4_conforme"]}},
           "aeroports": {"paires": int(len(m)), "instants": int(len(inst)), "candidat": a_d2, "livre": a_livre,
                         "era5_seule": D4.dso7(m, "era5") if "u_era5" in m else None,
                         "oi_seule": D4.dso7(m.dropna(subset=["u_era5_oi"]), "era5_oi") if "u_era5_oi" in m else None,
                         "ecart_candidat_livre": a_d2 - a_livre, "ic95": [float(lo), float(hi)]},
           "radiosondages_0_1km": {"paires": int(len(rm)), "candidat": r_d2, "livre": r_livre,
                                   "era5_seule": rmse_sondes(rm, "era5"), "oi_seule": rmse_sondes(rm, "era5_oi")},
           "conditions": {k_: bool(v_) for k_, v_ in c.items()},
           "le_candidat_remplace_le_modele_livre": remplace, "candidat": Path(a.d2).name,
           "modele_livre_national": Path(a.d2).name if remplace else "national_1km_v2"}
    out = Path(a.sortie) if a.sortie else Path(a.d2).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision_d2_national.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str),
                                                   encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    return 0 if remplace else 3


if __name__ == "__main__":
    raise SystemExit(main())
