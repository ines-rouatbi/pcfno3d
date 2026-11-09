"""Applique la regle d'interpretation de la phase 1 nationale (configs/national/regle_phase1_diagnostics.json,
fixee et hachee avant les calculs) aux sorties des diagnostics 2, 3 et 4. Aucun seuil n'est choisi ici :
ils sont recopies de la regle, dont l'empreinte sha256 est verifiee avant toute lecture.

    python scripts/juger_phase1.py [--attribution DIR] [--phase1 DIR] [--sortie FICHIER.md]

Chaque diagnostic absent est signale « non execute » ; rien n'est invente.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REGLE = REPO / "configs" / "national" / "regle_phase1_diagnostics.json"


def verifier_regle(f: Path = REGLE) -> str:
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    attendu = (f.parent / (f.name + ".sha256")).read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise SystemExit(f"regle modifiee depuis son hachage : {h} != {attendu}")
    return h


def juger_attribution(t: pd.DataFrame) -> list[str]:
    """Diagnostic 4 : reduction de l'ecart rms verite d'entrainement - ERA5 quand on coupe un mecanisme."""
    L = []
    p = t.pivot_table(index="config", columns=["hauteur_sol_m", "composante"], values="ecart_rms")
    ref = p.loc["actuel"]
    red = 1.0 - p / ref
    L.append("| Configuration | " + " | ".join(f"{c} {h} m" for h, c in p.columns) + " |")
    L.append("|---|" + "---|" * len(p.columns))
    for cfg in p.index:
        L.append(f"| {cfg} | " + " | ".join(f"{p.loc[cfg, k]:.3f} ({100 * red.loc[cfg, k]:+.0f} %)"
                                             for k in p.columns) + " |")
    L.append("")
    L.append("Pourcentages : reduction de l'ecart par rapport a la configuration actuelle (positif = ecart plus petit).")
    L.append("")
    v_cols = [k for k in p.columns if k[1] == "v"]
    responsables = [cfg for cfg in p.index if cfg not in ("actuel", "passe_haut")
                    and any(red.loc[cfg, k] >= 0.30 for k in v_cols)]
    L.append("**Mecanismes retenus pour correction en phase 2** (reduction >= 30 % sur v a 300 OU 800 m) : "
             + (", ".join(responsables) if responsables else "aucun"))
    if "passe_haut" in p.index:
        ok_v = all(red.loc["passe_haut", k] >= 0.30 for k in v_cols)
        u_cols = [k for k in p.columns if k[1] == "u"]
        ok_u = all(p.loc["passe_haut", k] <= 1.10 * ref[k] for k in u_cols)
        L.append(f"**Correction passe_haut** : v reduit d'au moins 30 % a 300 et 800 m : {'oui' if ok_v else 'non'} ; "
                 f"u augmente de 10 % au plus : {'oui' if ok_u else 'non'} -> "
                 f"{'RETENUE pour la phase 2' if ok_v and ok_u else 'non retenue'}")
    return L


def juger_spectres(b: pd.DataFrame) -> list[str]:
    """Diagnostic 2 : part de l'energie de l'erreur du modele par bande, axe y."""
    L = []
    e = b[(b["champ"] == "erreur_modele") & (b["axe"] == "y")]
    piste_y = piste_ge = False
    for h in sorted(e["hauteur_sol_m"].unique()):
        part = {c: e[(e["hauteur_sol_m"] == h) & (e["composante"] == c)].set_index("bande_km")["part_energie"]
                for c in "uv"}
        b1326 = {c: float(part[c].get("13-26", float("nan"))) for c in "uv"}
        ge = float(sum(part["v"].get(k, 0.0) for k in ("200-inf", "100-200", "50-100")))
        L.append(f"- {int(h)} m : part 13-26 km le long de y, u {b1326['u']:.3f} / v {b1326['v']:.3f} ; "
                 f"part > 50 km de l'erreur sur v {ge:.3f}")
        piste_y |= b1326["v"] >= 0.25 and b1326["v"] - b1326["u"] >= 0.05
        piste_ge |= ge >= 0.5
    L.append(f"**Ablation modes [6, 64, 32] en phase 3** : {'proposee' if piste_y else 'non proposee'}")
    L.append(f"**Erreur sur v dominee par la grande echelle (> 50 km)** : {'oui, priorite a la phase 2' if piste_ge else 'non'}")
    return L


def juger_tranches(t: pd.DataFrame) -> list[str]:
    """Diagnostic 3 : entrainement contre test par tranche au-dessus du sol."""
    L = [t.round(3).to_string(index=False), ""]
    if "entrainement_rmse_u" not in t:
        return L + ["Pas d'instants d'entrainement disponibles : sur-apprentissage non jugeable."]
    r_u = t["test_rmse_u"] / t["entrainement_rmse_u"]
    r_v = t["test_rmse_v"] / t["entrainement_rmse_v"]
    sur = [str(x) for x, a, b in zip(t["tranche"], r_u, r_v) if max(a, b) > 1.15]
    L.append(f"**Sur-apprentissage** (test / entrainement > 1,15) : {'tranches ' + ', '.join(sur) if sur else 'aucun'}")
    vu = t["entrainement_v_sur_u"]
    lourd = [str(x) for x, q in zip(t["tranche"], vu) if q >= 1.3]
    L.append("**Ecart v/u deja present a l'entrainement** (>= 1,3) : "
             + (f"oui ({', '.join(lourd)}) : il vient de la verite d'entrainement ou de l'architecture" if lourd else "non"))
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--attribution", default=str(REPO / "artifacts" / "national" / "attribution_v_validation"))
    ap.add_argument("--phase1", default=None, help="dossier diagnostic_v_phase1 (bandes.csv, tranches.csv)")
    ap.add_argument("--sortie", default=None)
    a = ap.parse_args()
    h = verifier_regle()
    L = ["# Phase 1 nationale : verdicts selon la regle fixee avant les calculs", "",
         f"Regle : `configs/national/regle_phase1_diagnostics.json`, sha256 `{h}` (verifie).", ""]
    ph = Path(a.phase1) if a.phase1 else None
    L.append("## Diagnostic 2 : spectres de l'erreur (test 2022)")
    L += juger_spectres(pd.read_csv(ph / "bandes.csv")) if ph and (ph / "bandes.csv").exists() else ["Non execute."]
    L += ["", "## Diagnostic 3 : entrainement contre test"]
    L += juger_tranches(pd.read_csv(ph / "tranches.csv")) if ph and (ph / "tranches.csv").exists() else ["Non execute."]
    L += ["", "## Diagnostic 4 : mecanismes du generateur (12 instants de validation 2021)"]
    f4 = Path(a.attribution) / "attribution_v.csv"
    L += juger_attribution(pd.read_csv(f4)) if f4.exists() else ["Non execute."]
    texte = "\n".join(L) + "\n"
    if a.sortie:
        Path(a.sortie).write_text(texte, encoding="utf-8")
    sys.stdout.write(texte)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
