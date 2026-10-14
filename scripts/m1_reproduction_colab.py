"""Lot M1 — le paquet reproduit-il la correction du notebook ? (session GPU, APRES le BLOC 10i)

A lancer dans la session Colab de confirmation, apres 10i, depuis le depot installe :

    !python scripts/m1_reproduction_colab.py -c colab

1. Rejoue 10f + 10g + 10h AVEC LE PAQUET (pcfno3d.correction) : caracteristiques sur
   l'entrainement (graine 4000) et la validation (selection fixe, graine 1000), 50 % des
   capteurs, choix de rho_sl puis d'alpha sur la VALIDATION.
2. Ecrit son JSON UNIQUEMENT dans run_dir/reproduction_paquet/ : les fichiers figes
   (couche_limite_v7.json, geostrophique_v2_v7.json, correction_moyenne_v7.json et leurs CSV)
   ne sont jamais ouverts en ecriture.
3. Critere de validite du portage, contre le couche_limite_v7.json FIGE :
   a) meme variante rho_sl, meme alpha, memes caracteristiques, memes niveaux « utilise » ;
   b) memes PREDICTIONS de la ridge (vent moyen cible de chaque niveau / composante utilise,
      instant par instant, sur la validation ET sur le test synoptique) a 1e-4 m/s pres ;
   c) RMSE par niveau du test synoptique (graine 2000, 50 %) avec la correction figee et la
      reproduite egales aux chiffres de desktop-d3 au millieme.
   Les ecarts de mu / sd / coef sont imprimes POUR INFORMATION seulement : les 30
   caracteristiques sont exactement colineaires (su*stab = su*stab+ + su*stab-, idem sv) et
   alpha vaut 0,01, donc un ecart relatif de 1e-7 sur les moyennes (numpy float32 contre
   torch bf16 -> fp32 du notebook) deplace les coefficients d'environ 3e-4 sans changer les
   predictions. Les sha256 des deux fichiers sont affiches.
4. INFORMATIF (sans effet sur le code de retour) : test synoptique a keep = 1,0 (tous les
   capteurs, comme predict() en production), modele seul contre corrige, RMSE par niveau.
   L'ajustement et la validation sont a 50 % des capteurs : on verifie l'absence de
   degradation hors de la distribution d'ajustement.

Codes de retour : 0 portage valide (a, b et c) ; 1 portage non valide ; 2 JSON fige absent ;
3 JSON fige modifie pendant le script. Lecture seule de tout le reste.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]

# desktop-d3, BLOC 10h, test synoptique : RMSE u / v par niveau apres correction
ATTENDU = {100: (0.416, 0.382), 150: (0.418, 0.381), 200: (0.393, 0.368), 300: (0.365, 0.344),
           400: (0.368, 0.347), 500: (0.380, 0.351), 650: (0.413, 0.339), 800: (0.404, 0.362),
           1000: (0.419, 0.355)}
TOL_PRED, TOL_RMSE = 1e-4, 1e-3                   # m/s : predictions de la ridge, RMSE
SEED_TR, SEED_VA, SEED_TE = 4000, 1000, 2000          # graines des BLOCS 10f / 10 du notebook


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def collecte(wf, cfg, splits, stems, hours, doys, seed):
    """X (n, nz, 13), G (n, 10), T (n, nz, 2) : comme collect() du 10f et GEO du 10g."""
    from torch.utils.data import DataLoader

    from pcfno3d.correction import geo_row, instant_features
    from pcfno3d.dataset import ANEO3D, assemble_input

    geom, norm = wf.geom, wf.norm
    ds = ANEO3D(stems, hours, doys, geom, train=False, seed=seed, keep_eval=0.5)
    X, G, T = [], [], []
    for s_raw, sensor, tfeat, y, _ in DataLoader(ds, batch_size=1):
        with torch.autocast(device_type=wf.device.type, dtype=wf.amp_dtype, enabled=wf.use_amp):
            pred = wf.model(assemble_input(s_raw.to(wf.device), sensor.to(wf.device),
                                           tfeat.to(wf.device), geom, norm))
        u, v, _ = norm.denorm(pred.float())
        S, m = s_raw[0].numpy(), sensor[0].numpy() > 0.5
        X.append(instant_features(S, m, tfeat[0].numpy(), u[0].cpu().numpy(),
                                  v[0].cpu().numpy(), geom))
        G.append(geo_row(S, m, geom))
        yy = y[0].numpy()
        T.append(np.array([[float(yy[c, k][geom.valid[k]].mean()) if geom.valid[k].any()
                            else 0.0 for c in (0, 1)] for k in range(geom.nz)]))
    return np.stack(X), np.stack(G), np.stack(T)


@torch.no_grad()
def test_synoptique(splits) -> np.ndarray:
    """Indices des instants synoptiques du test (ceux du DSO1)."""
    from pcfno3d.dataset import is_synoptic
    return np.array([i for i, s in enumerate(splits.test[0]) if is_synoptic(splits.stamp_of[s])])


@torch.no_grad()
def rmse_test(wf, cfg, splits, correction, keep: float = 0.5):
    """RMSE u, v par niveau evalue, test synoptique, avec la correction donnee (ou None)."""
    from pcfno3d.dataset import assemble_input, eval_loader

    geom, norm = wf.geom, wf.norm
    loader, _ = eval_loader(cfg, geom, splits, test_synoptique(splits), SEED_TE, keep=keep)
    acc = np.zeros((2, geom.nz, 2))                          # (composante, niveau, [n, sse])
    for s_raw, sensor, tfeat, y, _ in loader:
        with torch.autocast(device_type=wf.device.type, dtype=wf.amp_dtype, enabled=wf.use_amp):
            pred = wf.model(assemble_input(s_raw.to(wf.device), sensor.to(wf.device),
                                           tfeat.to(wf.device), geom, norm))
        u, v, _ = (a[0].float().cpu().numpy() for a in norm.denorm(pred.float()))
        if correction is not None:
            u, v, _ = correction.apply(u, v, s_raw[0].numpy(), sensor[0].numpy() > 0.5,
                                       tfeat[0].numpy(), geom)
        yy = y[0].numpy()
        for ci, p in enumerate((u, v)):
            for k in geom.eval_levels():
                m = geom.valid[k]
                acc[ci, k] += [m.sum(), float(((p[k][m] - yy[ci, k][m]) ** 2).sum())]
    return {int(geom.z_levels[k]): tuple(float(np.sqrt(acc[ci, k, 1] / acc[ci, k, 0]))
                                         for ci in range(2)) for k in geom.eval_levels()}


def compare_json(repro: dict, fige: dict) -> tuple[bool, list[str]]:
    """Critere (a) : variante, alpha, caracteristiques, niveaux « utilise ». Les ecarts de
    mu / sd / coef sont imprimes pour information (colinearite exacte, voir l'en-tete)."""
    msgs, ok = [], True
    for key in ("variante_rho_sl", "ridge_alpha", "features"):
        same = repro[key] == fige[key]
        ok &= same
        msgs.append(f"  {key:16s} : {'identique' if same else f'DIFFERENT {repro[key]!r} / {fige[key]!r}'}")
    if set(repro["niveaux"]) != set(fige["niveaux"]):
        return False, msgs + ["  niveaux : cles differentes"]
    ecarts = {"mu": 0.0, "sd": 0.0, "coef": 0.0}
    utilise_ok = True
    for nom, pa in repro["niveaux"].items():
        pb = fige["niveaux"][nom]
        if pa["utilise"] != pb["utilise"]:
            utilise_ok = False
            msgs.append(f"  {nom} : utilise {pa['utilise']} / fige {pb['utilise']}")
        for champ in ecarts:
            ecarts[champ] = max(ecarts[champ], float(np.max(np.abs(
                np.asarray(pa[champ]) - np.asarray(pb[champ])))))
    ok &= utilise_ok
    msgs.append(f"  niveaux utilises : {'identiques' if utilise_ok else 'DIFFERENTS'}")
    msgs.append("  (information) ecart max " + " | ".join(f"{c} {e:.2e}" for c, e in ecarts.items())
                + " -- non bloquant : caracteristiques colineaires, alpha 0,01")
    return ok, msgs


def cibles(data: dict, X: np.ndarray, G: np.ndarray, geom) -> dict[str, np.ndarray]:
    """Vent moyen cible predit par la ridge, instant par instant, pour chaque niveau /
    composante du JSON present sur la grille : {"800_u": (n,), ...}."""
    from pcfno3d.correction import ridge_pred, x3

    out = {}
    for k in geom.eval_levels():
        for c in "uv":
            nom = f"{int(geom.z_levels[k])}_{c}"
            par = data["niveaux"].get(nom)
            if par is not None:
                out[nom] = ridge_pred(par, x3(X, G, k, data["variante_rho_sl"]))
    return out


def compare_predictions(repro: dict, fige: dict, jeux: dict, geom) -> tuple[bool, list[str]]:
    """Critere (b) : predictions de la ridge egales a TOL_PRED pres, pour chaque niveau /
    composante utilise (dans l'un ou l'autre JSON), sur chaque jeu {nom: (X, G)}."""
    utilises = {n for d in (repro, fige) for n, p in d["niveaux"].items() if p["utilise"]}
    msgs, ok = [], True
    for nom_jeu, (X, G) in jeux.items():
        pa, pb = cibles(repro, X, G, geom), cibles(fige, X, G, geom)
        noms = sorted(utilises & set(pa) & set(pb))
        manque = sorted(utilises - (set(pa) & set(pb)))
        pire, ou = 0.0, "-"
        for n in noms:
            e = float(np.max(np.abs(pa[n] - pb[n]))) if len(pa[n]) else 0.0
            if not e <= pire:                                # un NaN compte comme un ecart
                pire, ou = e, n
        bon = bool(pire <= TOL_PRED) and not manque
        ok &= bon
        if manque:
            msgs.append(f"  {nom_jeu:10s} : niveaux utilises absents de la grille {manque}")
        msgs.append(f"  {nom_jeu:10s} : {len(X)} instants x {len(noms)} niveaux/composantes, "
                    f"ecart max {pire:.2e} m/s ({ou}, tolerance {TOL_PRED:g}) -> "
                    f"{'OK' if bon else 'ECART'}")
    return ok, msgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", default="colab")
    ap.add_argument("--checkpoint", help="defaut : best_dso_v7.pt du run_dir")
    ap.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR",
                    help="surcharge de configuration (essais sur le mini-jeu)")
    ap.add_argument("--sans-attendu", action="store_true",
                    help="ne pas comparer aux chiffres de desktop-d3 (mini-jeu : autres niveaux)")
    args = ap.parse_args(argv)

    from pcfno3d.cache import load_index
    from pcfno3d.config import Config
    from pcfno3d.correction import MeanWindCorrection, fit_correction
    from pcfno3d.dataset import make_splits
    from pcfno3d.inference import WindFieldModel

    cfg = Config.load(args.config)
    for item in args.set:
        cle, _, valeur = item.partition("=")
        cfg.apply_overrides({cle: valeur})
    cfg.inference.correction = None                         # le modele SEUL, puis on applique
    fige_path = cfg.paths.run_dir / "couche_limite_v7.json"
    if not fige_path.exists():
        print(f"{fige_path} absent : lancer d'abord le BLOC 10h (puis 10i)", file=sys.stderr)
        return 2
    out = cfg.paths.run_dir / "reproduction_paquet"
    out.mkdir(parents=True, exist_ok=True)
    sha_fige_avant = sha256(fige_path)

    wf = WindFieldModel(cfg, checkpoint=args.checkpoint, verbose=True)
    splits = make_splits(cfg, load_index(cfg))
    t0 = time.time()
    tr = collecte(wf, cfg, splits, *splits.train, SEED_TR)
    vs = splits.val_selection
    va = collecte(wf, cfg, splits, splits.val[0][vs], splits.val[1][vs], splits.val[2][vs],
                  SEED_VA)
    print(f"caracteristiques : train {len(tr[0])} | validation {len(va[0])} "
          f"({time.time() - t0:.0f} s)")
    data, rapport = fit_correction(*tr, *va, wf.geom)
    repro_path = out / "couche_limite_v7_paquet.json"
    repro_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    (out / "rapport_ajustement.json").write_text(json.dumps(rapport, indent=2), encoding="utf-8")
    print(f"variante rho_sl {rapport['variante']} | alpha {rapport['alpha']:g} -> {repro_path}")

    fige = json.loads(fige_path.read_text(encoding="utf-8"))
    print("\n1a) JSON reproduit / JSON fige du notebook : structure")
    print(f"  sha256 fige      : {sha_fige_avant}")
    print(f"  sha256 reproduit : {sha256(repro_path)}")
    ok_json, msgs = compare_json(data, fige)
    print("\n".join(msgs))

    te_s, te_h, te_d = splits.test
    syn = test_synoptique(splits)
    te = collecte(wf, cfg, splits, te_s[syn], te_h[syn], te_d[syn], SEED_TE)
    print("\n1b) Predictions de la ridge (vent moyen cible par instant) : reproduit / fige")
    ok_pred, msgs = compare_predictions(data, fige, {"validation": va[:2], "test": te[:2]},
                                        wf.geom)
    print("\n".join(msgs))

    print("\n2) Test synoptique (graine 2000, 50 % des capteurs) : RMSE u / v")
    r_seul = rmse_test(wf, cfg, splits, None)
    r_fige = rmse_test(wf, cfg, splits, MeanWindCorrection.load(fige_path))
    r_repro = rmse_test(wf, cfg, splits, MeanWindCorrection.load(repro_path))
    ok_rmse = True
    lignes = []
    attendu = {} if args.sans_attendu else ATTENDU
    if args.sans_attendu:
        for z in r_seul:
            print(f"  {z:5d} | seul {r_seul[z][0]:.3f} {r_seul[z][1]:.3f} | figee "
                  f"{r_fige[z][0]:.3f} {r_fige[z][1]:.3f} | paquet {r_repro[z][0]:.3f} "
                  f"{r_repro[z][1]:.3f}")
    print(" niveau | modele seul   | corr. figee   | corr. paquet  | attendu (d3)  | ecart max")
    for z, (eu, ev) in attendu.items():
        fu, fv = r_fige[z]
        pu, pv = r_repro[z]
        ecart = max(abs(fu - eu), abs(fv - ev), abs(pu - eu), abs(pv - ev))
        ok_rmse &= ecart <= TOL_RMSE
        lignes.append({"z_m": z, "modele_seul": r_seul[z], "figee": r_fige[z],
                       "paquet": r_repro[z], "attendu": (eu, ev), "ecart_max": ecart})
        print(f"  {z:5d} | {r_seul[z][0]:.3f} {r_seul[z][1]:.3f} | {fu:.3f} {fv:.3f} | "
              f"{pu:.3f} {pv:.3f} | {eu:.3f} {ev:.3f} | {ecart:.4f}"
              f"{'' if ecart <= TOL_RMSE else '  <- ECART'}")
    (out / "comparaison_test.json").write_text(json.dumps(lignes, indent=2), encoding="utf-8")

    print("\n3) INFORMATIF, sans effet sur le verdict : test synoptique a keep = 1,0 (tous les "
          "capteurs,\n   comme predict() en production) ; la correction a ete ajustee a 50 %")
    k_seul = rmse_test(wf, cfg, splits, None, keep=1.0)
    k_corr = rmse_test(wf, cfg, splits, MeanWindCorrection.load(fige_path), keep=1.0)
    print(" niveau | modele seul   | corr. figee   | delta u  delta v")
    keep1, degrades = [], []
    for z in k_seul:
        du, dv = k_corr[z][0] - k_seul[z][0], k_corr[z][1] - k_seul[z][1]
        degrade = bool(max(du, dv) > TOL_RMSE)
        if degrade:
            degrades.append(z)
        keep1.append({"z_m": z, "modele_seul": k_seul[z], "corrige": k_corr[z],
                      "delta": (du, dv), "degrade": degrade})
        print(f"  {z:5d} | {k_seul[z][0]:.3f} {k_seul[z][1]:.3f} | {k_corr[z][0]:.3f} "
              f"{k_corr[z][1]:.3f} | {du:+.3f}  {dv:+.3f}{'  <- DEGRADATION' if degrade else ''}")
    print("  keep = 1,0 : " + (f"DEGRADATION a {degrades} m" if degrades
                               else "aucune degradation") + " (informatif)")
    (out / "keep1_informatif.json").write_text(json.dumps(keep1, indent=2), encoding="utf-8")

    if sha256(fige_path) != sha_fige_avant:                 # pas d'assert : python -O l'ote
        print(f"ERREUR : le JSON fige {fige_path} a change pendant le script", file=sys.stderr)
        return 3
    print(f"\nJSON fige intact (sha256 inchange). Sorties : {out}")
    verdict = ok_json and ok_pred and ok_rmse
    print(f"VERDICT : structure {'OK' if ok_json else 'ECART'} | predictions "
          f"{'OK' if ok_pred else 'ECART'} | RMSE {'OK' if ok_rmse else 'ECART'} "
          f"-> {'PORTAGE VALIDE' if verdict else 'PORTAGE NON VALIDE'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
