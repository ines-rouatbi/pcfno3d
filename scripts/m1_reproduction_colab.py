"""Lot M1 — le paquet reproduit-il la correction du notebook ? (session GPU, APRES le BLOC 10i)

A lancer dans la session Colab de confirmation, apres 10i, depuis le depot installe :

    !python scripts/m1_reproduction_colab.py -c colab

1. Rejoue 10f + 10g + 10h AVEC LE PAQUET (pcfno3d.correction) : caracteristiques sur
   l'entrainement (graine 4000) et la validation (selection fixe, graine 1000), 50 % des
   capteurs, choix de rho_sl puis d'alpha sur la VALIDATION.
2. Ecrit son JSON UNIQUEMENT dans run_dir/reproduction_paquet/ : les fichiers figes
   (couche_limite_v7.json, geostrophique_v2_v7.json, correction_moyenne_v7.json et leurs CSV)
   ne sont jamais ouverts en ecriture.
3. Compare ce JSON au couche_limite_v7.json FIGE : variante, alpha, niveaux corriges, puis
   mu / sd / coef a 1e-6 ; affiche les sha256 des deux fichiers.
4. Applique la correction FIGEE et la reproduite au test synoptique (graine 2000, 50 %) et
   compare les RMSE par niveau aux chiffres de desktop-d3, au millieme.

Code de retour 0 seulement si 3 et 4 passent. Lecture seule de tout le reste.
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
TOL_COEF, TOL_RMSE = 1e-6, 1e-3
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
def rmse_test(wf, cfg, splits, correction):
    """RMSE u, v par niveau evalue, test synoptique, avec la correction donnee (ou None)."""
    from pcfno3d.dataset import assemble_input, eval_loader, is_synoptic

    geom, norm = wf.geom, wf.norm
    te_s = splits.test[0]
    syn = np.array([i for i, s in enumerate(te_s) if is_synoptic(splits.stamp_of[s])])
    loader, _ = eval_loader(cfg, geom, splits, syn, SEED_TE, keep=0.5)
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
    msgs, ok = [], True
    for key in ("variante_rho_sl", "ridge_alpha", "features"):
        same = repro[key] == fige[key]
        ok &= same
        msgs.append(f"  {key:16s} : {'identique' if same else f'DIFFERENT {repro[key]!r} / {fige[key]!r}'}")
    if set(repro["niveaux"]) != set(fige["niveaux"]):
        return False, msgs + ["  niveaux : cles differentes"]
    pire = 0.0
    for nom, a in repro["niveaux"].items():
        b = fige["niveaux"][nom]
        if a["utilise"] != b["utilise"]:
            ok = False
            msgs.append(f"  {nom} : utilise {a['utilise']} / fige {b['utilise']}")
        for champ in ("mu", "sd", "coef"):
            pire = max(pire, float(np.max(np.abs(np.asarray(a[champ]) - np.asarray(b[champ])))))
    ok &= pire <= TOL_COEF
    msgs.append(f"  ecart max mu/sd/coef : {pire:.2e} (tolerance {TOL_COEF:g}) -> "
                f"{'OK' if pire <= TOL_COEF else 'ECART'}")
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
    print("\n1) JSON reproduit / JSON fige du notebook")
    print(f"  sha256 fige      : {sha_fige_avant}")
    print(f"  sha256 reproduit : {sha256(repro_path)}")
    ok_json, msgs = compare_json(data, fige)
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

    assert sha256(fige_path) == sha_fige_avant, "le JSON fige a change pendant le script !"
    print(f"\nJSON fige intact (sha256 inchange). Sorties : {out}")
    verdict = ok_json and ok_rmse
    print(f"VERDICT : JSON {'OK' if ok_json else 'ECART'} | RMSE {'OK' if ok_rmse else 'ECART'} "
          f"-> {'PORTAGE VALIDE' if verdict else 'PORTAGE NON VALIDE'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
