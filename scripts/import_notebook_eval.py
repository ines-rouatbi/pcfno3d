"""Importe l'evaluation DSO imprimee par le BLOC 10 d'un notebook, au format du paquet.

Pourquoi : la v7 a ete evaluee sur Colab (GPU, 120 instants synoptiques de test). Le
notebook a ecrit test_summary.json sur le Drive, pas sur ce poste ; l'application ne
trouve donc aucune evaluation. Refaire l'evaluation ici n'aurait pas de sens : seuls
quelques instants de test sont en cache, et le DSO4 serait mesure sur CPU (~20 s au lieu
de 91 ms), ce qui ferait passer pour non conforme un modele qui l'est.

On reprend donc les resultats DU NOTEBOOK, tels qu'il les a imprimes, et on le dit :
le resume porte sa source, le materiel de mesure et la precision (valeurs arrondies a
l'impression). Rien n'est recalcule, rien n'est estime.

    python scripts/import_notebook_eval.py notebooks/current/fno3d_v7.ipynb --config local_v7

Ecrit dans run_dir/eval_<modele>/ (l'emplacement du notebook) :
    test_summary.json, test_par_niveau_synoptique.csv, test_par_niveau_episodes.csv,
    robustesse_capteurs.csv

Le parseur echoue bruyamment si une valeur attendue manque : un resume partiel affiche
comme complet serait pire que pas de resume.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

NUM = r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?"


def bloc10_text(notebook: Path) -> str:
    nb = json.load(io.open(notebook, encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        if re.search(r"#\s*BLOC\s+10\b", "".join(cell["source"])):
            text = "".join("".join(o.get("text", "")) for o in cell.get("outputs", [])
                           if o.get("output_type") == "stream")
            if "BLOC 10 OK" not in text:
                raise ValueError(f"{notebook.name} : le BLOC 10 n'a pas termine "
                                 "(« BLOC 10 OK » absent de sa sortie)")
            return text
    raise ValueError(f"{notebook.name} : aucun BLOC 10 avec une sortie")


def _need(pattern: str, text: str, what: str, flags: int = 0) -> re.Match:
    m = re.search(pattern, text, flags)
    if m is None:
        raise ValueError(f"valeur introuvable dans la sortie du BLOC 10 : {what}")
    return m


def _verdict(line_pattern: str, text: str, what: str) -> bool:
    m = _need(line_pattern + r".*?->\s*(NON CONFORME|CONFORME)", text, what)
    return m.group(m.lastindex) == "CONFORME"


def parse_levels(section: str, with_w: bool) -> list[dict]:
    rows = []
    if with_w:
        pat = re.compile(rf"^\s*(\d+) m \|\s*({NUM})%\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s*\|"
                         rf"\s*({NUM})\s+({NUM})\s+({NUM})", re.M)
        for m in pat.finditer(section):
            z, cov, ru, rv, rw, r2u, r2v, r2w = m.groups()
            rows.append({"z_m": int(z), "couverture": float(cov) / 100.0,
                         "rmse_u": float(ru), "rmse_v": float(rv), "rmse_w": float(rw),
                         "r2_u": float(r2u), "r2_v": float(r2v), "r2_w": float(r2w)})
    else:
        pat = re.compile(rf"^\s*(\d+) m \|\s*({NUM})\s+({NUM})\s*\|\s*({NUM})\s+({NUM})", re.M)
        for m in pat.finditer(section):
            z, ru, rv, r2u, r2v = m.groups()
            rows.append({"z_m": int(z), "rmse_u": float(ru), "rmse_v": float(rv),
                         "r2_u": float(r2u), "r2_v": float(r2v)})
    return rows


def parse(text: str) -> dict:
    model = _need(r"Modele evalue : (\S+)", text, "modele").group(1)
    epoch_m = _need(rf"Meilleur modele : epoque (\d+) \(score val = ({NUM})\)", text, "epoque")
    syn_m = _need(r"TEST SYNOPTIQUE \(DSO1\) : (\d+) instants \| capteurs visibles : (\d+)%",
                  text, "nombre d'instants synoptiques")

    # sections : table synoptique (jusqu'a la ligne de tirets), episodes, robustesse
    syn_table = text[syn_m.end():text.index("-" * 20, syn_m.end())]
    levels = parse_levels(syn_table, with_w=True)
    if not levels:
        raise ValueError("table par niveau (synoptique) vide")

    ep_levels, n_ep = [], 0
    ep_m = re.search(r"EPISODES RAPIDES.*?: (\d+) instants", text)
    if ep_m:
        n_ep = int(ep_m.group(1))
        end = text.find("=" * 20, ep_m.end())
        ep_levels = parse_levels(text[ep_m.end():end], with_w=False)

    glob = _need(rf"RMSE GLOBAL.*?u = ({NUM})\s+v = ({NUM})", text, "RMSE global")
    dso2 = _need(rf"DSO2\s+RMS residu moyen = ({NUM}).*?p95 = ({NUM}) \| max = ({NUM})",
                 text, "DSO2")
    dso2_all = re.search(rf"episodes compris : RMS residu moyen = ({NUM})", text)
    dso3 = _need(rf"DSO3\s+rugosite prediction / verite = ({NUM})", text, "DSO3")
    dso3_all = re.search(rf"episodes compris : rugosite = ({NUM})", text)
    dso4 = _need(rf"DSO4\s+inference = ({NUM}) ms", text, "DSO4")
    dso5 = _need(rf"DSO5\s+RMSE max a 25 % de capteurs = ({NUM})", text, "DSO5")
    mae = _need(rf"Direction : MAE = ({NUM}) deg", text, "MAE de direction")

    robust = []
    rob_m = re.search(r"Robustesse.*?\n.*?capteurs_visibles.*?\n((?:\s+[\d.]+.*\n)+)", text)
    if rob_m:
        for line in rob_m.group(1).strip().splitlines():
            vals = [float(x) for x in re.findall(NUM, line)]
            if len(vals) == 5:
                robust.append(dict(zip(("capteurs_visibles", "rmse_u_max", "rmse_v_max",
                                        "r2_u_min", "r2_v_min"), vals)))

    return {
        "modele": model,
        "epoque": int(epoch_m.group(1)),
        "score_validation": float(epoch_m.group(2)),
        "instants_synoptiques": int(syn_m.group(1)),
        "instants_episodes": n_ep,
        "capteurs_visibles": int(syn_m.group(2)) / 100.0,
        "tta": 1, "tta_keep": None,
        "par_niveau_synoptique": levels,
        "par_niveau_episodes": ep_levels,
        "dso1_rmse": _verdict(r"DSO1\s+RMSE u, v <=", text, "DSO1 RMSE"),
        "dso1_r2": _verdict(r"R2 u, v >= [\d.]+ a chaque niveau\s", text, "DSO1 R2"),
        "dso1_r2_uvw": _verdict(r"R2 u, v, w >=", text, "R2 u, v, w"),
        "dso1_rmse_global_u": float(glob.group(1)),
        "dso1_rmse_global_v": float(glob.group(2)),
        "dso1_rmse_global_conforme": _verdict(r"RMSE GLOBAL", text, "RMSE global"),
        "critere_omm": _verdict(r"critere OMM", text, "critere OMM"),
        "dso2_rms_moyen": float(dso2.group(1)),
        "dso2_rms_p95": float(dso2.group(2)),
        "dso2_rms_max": float(dso2.group(3)),
        "dso2_rms_tous_instants": float(dso2_all.group(1)) if dso2_all else None,
        "dso2_conforme": _verdict(r"DSO2\s+RMS residu moyen", text, "DSO2"),
        "dso3_rugosite": float(dso3.group(1)),
        "dso3_rugosite_tous_instants": float(dso3_all.group(1)) if dso3_all else None,
        "dso3_conforme": _verdict(r"DSO3\s+rugosite", text, "DSO3"),
        "dso4_inference_s": float(dso4.group(1)) / 1000.0,
        "dso4_conforme": _verdict(r"DSO4\s+inference", text, "DSO4"),
        "dso5_rmse_25pct": float(dso5.group(1)),
        "dso5_conforme": _verdict(r"DSO5\s+RMSE max", text, "DSO5"),
        "mae_direction_deg": float(mae.group(1)),
        "robustesse": robust,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", help="notebook dont le BLOC 10 a tourne, ex. fno3d_v7.ipynb")
    ap.add_argument("--config", default="local_v7")
    ap.add_argument("--materiel", default="GPU Colab",
                    help="ou les mesures ont ete faites (affiche dans l'application)")
    args = ap.parse_args()

    import pandas as pd

    from pcfno3d.config import Config

    cfg = Config.load(args.config)
    notebook = Path(args.notebook)
    summary = parse(bloc10_text(notebook))
    summary.update({
        "produit_par": "notebook",
        "source": {
            "notebook": notebook.name,
            "bloc": 10,
            "materiel": args.materiel,
            "importe_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "precision": "valeurs telles qu'imprimees par le notebook (arrondies)",
            "note": "evaluation faite par le notebook, non recalculee sur ce poste",
        },
    })

    stem = Path(summary["modele"]).stem
    out = cfg.paths.run_dir / f"eval_{stem}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(summary["par_niveau_synoptique"]).to_csv(
        out / "test_par_niveau_synoptique.csv", index=False)
    if summary["par_niveau_episodes"]:
        pd.DataFrame(summary["par_niveau_episodes"]).to_csv(
            out / "test_par_niveau_episodes.csv", index=False)
    if summary["robustesse"]:
        pd.DataFrame(summary["robustesse"]).to_csv(out / "robustesse_capteurs.csv", index=False)

    ok = [k for k in ("dso1_rmse", "dso2_conforme", "dso3_conforme", "dso4_conforme",
                      "dso5_conforme") if summary[k]]
    print(f"evaluation de {summary['modele']} (epoque {summary['epoque']}) importee depuis "
          f"{notebook.name}, BLOC 10")
    print(f"  {summary['instants_synoptiques']} instants synoptiques, "
          f"{len(summary['par_niveau_synoptique'])} niveaux, "
          f"{summary['instants_episodes']} episodes, {len(summary['robustesse'])} lignes de "
          f"robustesse")
    print(f"  DSO conformes : {len(ok)} / 5 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
