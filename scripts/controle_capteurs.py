"""Controle du branchement des capteurs (demande d'Ines, 11/11) — AUCUN entrainement, aucune lecture du test.

Les gains des capteurs simules mesures par scripts/info_capteurs_v4.py (7,9e-8 m/s pour C' s42) et l'OS6 du
Grand Tunis (0,5326 m/s pour 10 comme pour 200 capteurs) font soupconner que les capteurs n'arrivent pas au
reseau. Quatre controles, sur des instants de VALIDATION :

  1. test absurde : 50 capteurs simules (meme tirage), puis +15 m/s sur u a ces 50 cellules ; max de
     |sortie modifiee - sortie normale| (u, v, noeuds valides, tous niveaux) a moins de `--rayon-km` des
     capteurs et ailleurs. Meme chose en +15 m/s sur u aux AEROPORTS (temoin : le modele les a vus).
     Deux chemins : celui de l'evaluation (ANEO3D + assemble_input, info_capteurs_v4) et celui de l'OS6
     (WindFieldModel._members avec une surface par masque).
  2. chemin : canaux de S et masque ecrits, cellules non nulles du masque avant / apres l'ajout des 50
     capteurs, valeurs du tenseur d'entree du reseau a ces cellules (canal u normalise, canal du masque).
  3. entrainement (avec --entrainement et une config D) : sur `--lots` exemples d'entrainement, nombre de
     capteurs simules tires et nombre de leurs cellules presentes dans le tenseur d'entree du reseau.

    python scripts/controle_capteurs.py -c national_1km_colab --run <run_dir> [--set ...] [--entrainement]
    python scripts/controle_capteurs.py -c grand_tunis_v4_dso_colab --run <run Grand Tunis> --set ...
Sortie : <run>/controle_capteurs.json (jamais ecrase : suffixe horodate s'il existe).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]


class _Trace:
    """Enveloppe d'une EntreeANEO qui garde les cellules des derniers capteurs tires."""

    def __init__(self, e):
        self.e, self.cellules, self.valeurs = e, [], []

    def appliquer(self, cel, rng):
        return self.e.appliquer(cel, rng)

    def capteurs(self, S, Y, rng, n=None):
        r = self.e.capteurs(S, Y, rng, n)
        self.cellules.append(None if r is None else r[0].copy())
        self.valeurs.append(None if r is None else (r[1].copy(), r[2].copy()))
        return r


def _proches(geom, cel: np.ndarray, rayon_km: float) -> np.ndarray:
    """(ny, nx) : a moins de rayon_km d'une des cellules."""
    yy, xx = np.meshgrid(np.arange(geom.ny) * geom.dy, np.arange(geom.nx) * geom.dx, indexing="ij")
    m = np.zeros((geom.ny, geom.nx), bool)
    for iy, ix in cel:
        m |= np.hypot(yy - iy * geom.dy, xx - ix * geom.dx) <= rayon_km * 1000.0
    return m


def _ecarts(geom, a, b, proche: np.ndarray) -> dict:
    """max |a - b| sur u, v (noeuds valides) pres des cellules et ailleurs."""
    val = np.asarray(geom.valid, bool)
    out = {}
    for c, nom in ((0, "u"), (1, "v")):
        d = np.abs(np.asarray(a[c], np.float64) - np.asarray(b[c], np.float64))
        d = np.where(val, d, 0.0)
        out[f"max_{nom}_proche"] = float(d[:, proche].max()) if proche.any() else None
        out[f"max_{nom}_ailleurs"] = float(d[:, ~proche].max()) if (~proche).any() else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--instants", type=int, default=3)
    ap.add_argument("--n-capteurs", type=int, default=50)
    ap.add_argument("--delta", type=float, default=15.0)
    ap.add_argument("--rayon-km", type=float, default=5.0)
    ap.add_argument("--entrainement", action="store_true", help="controle 3 (config du candidat D)")
    ap.add_argument("--lots", type=int, default=5)
    a = ap.parse_args()

    import torch

    from pcfno3d.aneo_entree import EntreeANEO
    from pcfno3d.cli import _parse_set
    from pcfno3d.config import Config
    from pcfno3d.dataset import ANEO3D, assemble_input, cellules_stations
    from pcfno3d.evaluate import Evaluator

    cfg = Config.load(a.config)
    over = _parse_set(a.set)
    over["paths.run_dir"] = a.run
    cfg.apply_overrides(over)
    ev = Evaluator(cfg, verbose=False)
    geom, norm, wf = ev.geom, ev.norm, ev.wf
    s, h, d = ev.splits.val
    sel = ev.splits.val_selection[:a.instants]
    res = {"run": Path(a.run).name, "config": a.config, "instants": [str(x) for x in s[sel]],
           "n_capteurs": a.n_capteurs, "delta_m_s": a.delta, "rayon_km": a.rayon_km,
           "mode_entree": cfg.mask.entree, "controles": []}

    def passe(s_raw, sensor, tfeat):
        with torch.no_grad(), torch.autocast(device_type=wf.device.type, dtype=wf.amp_dtype,
                                             enabled=wf.use_amp):
            x = assemble_input(s_raw, sensor, tfeat, geom, norm)
            pred = wf.model(x).float()
        u, v, _w = norm.denorm(pred)
        return x, [u[0].cpu().numpy(), v[0].cpu().numpy()]

    st = cellules_stations(geom)
    e50 = _Trace(EntreeANEO(geom, None, 0.0, cfg.mask.aneo_rayon_km, (a.n_capteurs, a.n_capteurs),
                            cfg.mask.aneo_hauteur_capteur_m))
    for k, i in enumerate(sel):
        args = (s[[i]], h[[i]], d[[i]], geom)
        base = ANEO3D(*args, train=False, seed=1000 + int(i), keep_eval=1.0)[0]
        avec = ANEO3D(*args, train=False, seed=1000 + int(i), keep_eval=1.0, aneo_eval=e50)[0]
        cel = e50.cellules[-1]
        cu, cv = e50.valeurs[-1]
        dev = wf.device
        sb, mb, tb = (t[None].to(dev) for t in base[:3])
        sa, ma, ta = (t[None].to(dev) for t in avec[:3])
        # controle 2 : chemin des capteurs
        nb, na = int(mb.sum().item()), int(ma.sum().item())
        dans_masque = int(ma[0, cel[:, 0], cel[:, 1]].sum().item())
        ecrit_u = float(np.abs(sa[0, 0, cel[:, 0], cel[:, 1]].cpu().numpy() - cu).max())
        xb, ob = passe(sb, mb, tb)
        xa, oa = passe(sa, ma, ta)
        canal_masque = 7                                   # 7 mesures normalisees, puis le masque
        entree_u = (sa[0, 0, cel[:, 0], cel[:, 1]] - norm.surf_mu.reshape(-1)[0]) / norm.surf_sd.reshape(-1)[0]
        xin = xa[0, 0, 0, cel[:, 0], cel[:, 1]].float()
        rec = {"instant": str(s[i]),
               "masque_avant": nb, "masque_apres": na, "capteurs_tires": int(len(cel)),
               "capteurs_dans_le_masque": dans_masque,
               "ecart_max_valeur_ecrite_S_u": ecrit_u,
               "entree_reseau_masque_aux_capteurs_min": float(xa[0, canal_masque, 0, cel[:, 0], cel[:, 1]].min()),
               "entree_reseau_u_aux_capteurs_ecart_max": float((xin - entree_u.float()).abs().max()),
               "sortie_avec_50_contre_sans": _ecarts(geom, oa, ob, _proches(geom, cel, a.rayon_km))}
        # controle 1 : test absurde (+delta sur u aux capteurs simules), chemin de l'evaluation
        sm = sa.clone()
        sm[0, 0, cel[:, 0], cel[:, 1]] += a.delta
        _xm, om = passe(sm, ma, ta)
        rec["absurde_capteurs_evaluation"] = _ecarts(geom, om, oa, _proches(geom, cel, a.rayon_km))
        # temoin : +delta sur u aux aeroports
        if len(st):
            sm2 = sb.clone()
            sm2[0, 0, st[:, 0], st[:, 1]] += a.delta
            _x2, o2 = passe(sm2, mb, tb)
            rec["absurde_aeroports_evaluation"] = _ecarts(geom, o2, ob, _proches(geom, st, a.rayon_km))
        # controle 1 bis : meme test par le chemin de l'OS6 (WindFieldModel._members, une surface par masque)
        S0 = base[0].numpy()
        m0 = base[1].numpy() > 0.5
        m1 = m0.copy()
        m1[cel[:, 0], cel[:, 1]] = True
        S1 = S0.copy()
        S1[0][cel[:, 0], cel[:, 1]], S1[1][cel[:, 0], cel[:, 1]] = cu, cv
        S2 = S1.copy()
        S2[0][cel[:, 0], cel[:, 1]] += a.delta
        from pcfno3d.geometry import N_TFEAT
        extra = base[2].numpy()[N_TFEAT:] if geom.forcage is not None else None
        sorties = wf._members(S0, [m1, m1], float(h[i]), int(d[i]), batch=1, extra=extra, surfaces=[S1, S2])
        rec["absurde_capteurs_chemin_os6"] = _ecarts(geom, [sorties[1][0], sorties[1][1]],
                                                     [sorties[0][0], sorties[0][1]],
                                                     _proches(geom, cel, a.rayon_km))
        # et sans surfaces (chemin du Grand Tunis : la valeur aux nouvelles cellules vient de S)
        sorties2 = wf._members(S0, [m0, m1], float(h[i]), int(d[i]), batch=1, extra=extra)
        rec["os6_sans_surfaces_masque_ajoute"] = _ecarts(geom, [sorties2[1][0], sorties2[1][1]],
                                                         [sorties2[0][0], sorties2[0][1]],
                                                         _proches(geom, cel, a.rayon_km))
        res["controles"].append(rec)
        print(json.dumps(rec, indent=1, ensure_ascii=False), flush=True)

    if a.entrainement:
        from pcfno3d import aneo_entree as AE
        ent = AE.entree_aneo(geom)
        if ent is None:
            raise SystemExit("--entrainement : la config n'active pas l'entree ANEO (mask.aneo_*)")
        tr = _Trace(ent)
        geom._entree_aneo = tr                              # le Dataset d'entrainement passe par entree_aneo
        ts, th, td = ev.splits.train
        ds = ANEO3D(ts, th, td, geom, train=True, n_samples=a.lots)
        lots = []
        for k in range(a.lots):
            s_raw, sensor, tfeat, _y, _j = ds[k][:5]
            cel = tr.cellules[-1]
            x = assemble_input(s_raw[None].to(wf.device), sensor[None].to(wf.device),
                               tfeat[None].to(wf.device), geom, norm)
            n_ok = 0 if cel is None else int((x[0, 7, 0, cel[:, 0], cel[:, 1]] > 0.5).sum().item())
            lots.append({"capteurs_tires": 0 if cel is None else int(len(cel)),
                         "cellules_capteurs_dans_l_entree_du_reseau": n_ok,
                         "masque_total": int(sensor.sum().item())})
            print(lots[-1], flush=True)
        res["entrainement"] = lots

    out = Path(a.run) / "controle_capteurs.json"
    if out.exists():
        out = out.with_name(f"controle_capteurs_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ecrit :", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
