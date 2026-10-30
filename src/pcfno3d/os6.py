"""OS6 — dimensionner et placer le reseau ANEO AVANT son deploiement (cahier v10, sections 6.9
et 18.5 ; besoins BF35-BF37, BF74-BF78 ; indicateurs NET1, NET2 ; livrable S18).

Methode : experiences de simulation de systemes d'observation (OSSE). Le jeu genere contient la
verite 3D complete : un capteur ANEO simule est une maille ou le modele voit la verite de la couche
de surface du cache (mat du generateur, 20 a 30 m, bruit de mesure compris), exactement comme une
station reelle dans l'entree de production (`dataset.entree_era5_stations`). On mesure l'erreur du
champ 3D obtenue avec n'importe quelle disposition, avant tout achat.

    1. emplacements candidats  : terre ferme, altitude, pente, marge au bord, zones exclues,
                                 distance minimale aux stations (BF74-BF77, `Contraintes`) ;
    2. placement optimise      : selection gloutonne sur la VALIDATION — a chaque etape, ajout des
                                 `par_etape` candidats ou l'erreur residuelle (lissee sur le rayon
                                 d'influence) est la plus forte, distance minimale respectee, puis
                                 nouvelle passe du modele ; gain mesure de chaque etape (BF37, BF78) ;
    3. courbe de dimensionnement : erreur par niveau sur le TEST (calcule une fois) selon le nombre
                                 de capteurs, placement optimise / grille reguliere / hasard a
                                 nombre egal (BF35, NET1) ; nombre minimal qui respecte le seuil
                                 cible (BF36, NET2) ;
    4. plan de deploiement     : ordre, coordonnees, hauteur de mat, gain ; CSV, GeoJSON, carte (S18).

Limites, dites dans le rapport : le modele livre a appris avec ERA5 + 3 stations ; un capteur
ANEO est pour lui une station de plus, a un endroit qu'il n'a jamais vu en entree. Le gain
mesure est donc celui du modele ACTUEL ; un modele reentraine avec des reseaux denses en
tirerait davantage. La verite est generee (relief reel, forcage ERA5) : le plan reste
provisoire jusqu'a la validation sur les premiers capteurs installes (BF78, etape 6).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .dataset import ANEO3D, cellules_stations, is_synoptic
from .geometry import N_TFEAT

TAILLES = (10, 20, 30, 50, 75, 100, 150, 200)


# ---------------------------------------------------------------------- contraintes
@dataclass
class Contraintes:
    """Contraintes d'implantation (BF74-BF77). Les zones exclues sont des polygones
    [[lat, lon], ...] (zones militaires, aeroport, aires protegees... fournies par l'exploitant) ;
    aucune n'est inventee ici."""

    altitude_max_m: float = 650.0          # acces et energie : pas de crete isolee
    pente_max_deg: float = 20.0
    marge_bord_m: float = 1000.0           # loin du bord du domaine (champ mal contraint)
    distance_min_m: float = 1000.0         # entre deux capteurs (et avec les stations)
    pas_candidats_m: float = 500.0         # un candidat tous les 500 m
    z0_min_m: float = 0.005                # sous ce z0 : mer, lac, sebkha
    exclusions: list = field(default_factory=list)
    n_max: int = 200                       # budget (nombre de capteurs)

    @classmethod
    def charger(cls, chemin: str | Path | None) -> "Contraintes":
        if chemin is None:
            return cls()
        d = json.loads(Path(chemin).read_text(encoding="utf-8"))
        inconnues = set(d) - set(cls.__dataclass_fields__)
        if inconnues:
            raise ValueError(f"contraintes inconnues : {sorted(inconnues)}")
        return cls(**d)


def _dans_polygone(lat: np.ndarray, lon: np.ndarray, poly) -> np.ndarray:
    """Lancer de rayon, vectorise sur les points."""
    p = np.asarray(poly, float)
    dedans = np.zeros(lat.shape, bool)
    for (a_lat, a_lon), (b_lat, b_lon) in zip(p, np.roll(p, -1, axis=0)):
        coupe = (a_lat > lat) != (b_lat > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (b_lon - a_lon) * (lat - a_lat) / (b_lat - a_lat) + a_lon
        dedans ^= coupe & (lon < x)
    return dedans


def candidats(geom, c: Contraintes, georef=None) -> np.ndarray:
    """(n, 2) mailles (iy, ix) ou un capteur ANEO peut etre installe."""
    ny, nx = geom.ny, geom.nx
    pas = max(int(round(c.pas_candidats_m / geom.dx)), 1)
    marge = int(math.ceil(c.marge_bord_m / geom.dx))
    ok = np.zeros((ny, nx), bool)
    ok[marge:ny - marge:pas, marge:nx - marge:pas] = True
    ok &= geom.valid.any(axis=0)                       # une colonne au-dessus du relief
    ok &= geom.z0 >= c.z0_min_m                        # terre ferme
    ok &= geom.zt <= c.altitude_max_m
    ok &= geom.slope <= c.pente_max_deg
    jj, ii = np.nonzero(ok)
    if c.exclusions:
        if georef is None:
            raise ValueError("zones exclues en lat/lon : il faut le georeferencement du domaine")
        lat, lon = georef.to_latlon(ii * geom.dx, jj * geom.dy)
        dehors = np.ones(len(jj), bool)
        for poly in c.exclusions:
            dehors &= ~_dans_polygone(lat, lon, poly)
        jj, ii = jj[dehors], ii[dehors]
    cand = np.stack([jj, ii], axis=1)
    st = cellules_stations(geom)
    if len(st) and len(cand):
        d = np.hypot((cand[:, None, 0] - st[None, :, 0]) * geom.dy,
                     (cand[:, None, 1] - st[None, :, 1]) * geom.dx).min(axis=1)
        cand = cand[d >= c.distance_min_m]
    return cand


def masque(geom, cellules) -> np.ndarray:
    m = np.zeros((geom.ny, geom.nx), bool)
    if len(cellules):
        c = np.asarray(cellules, int)
        m[c[:, 0], c[:, 1]] = True
    return m


# ---------------------------------------------------------------------- passes du modele
class Banc:
    """Les instants d'un jeu (validation ou test) et les passes du modele avec des reseaux
    ANEO ajoutes a l'entree de production. Toutes les dispositions d'un instant passent en
    LOTS dans le reseau (`WindFieldModel._members`) : memes instants, meme entree de base."""

    def __init__(self, wf, jeu: str, n_instants: int | None = None, verbose: bool = True):
        from .dataset import make_splits
        from .cache import load_index

        self.wf, self.geom, self.verbose = wf, wf.geom, verbose
        splits = make_splits(wf.cfg, load_index(wf.cfg))
        if jeu == "validation":
            s, h, d = splits.val
            idx, seed = splits.val_selection, 1000
        elif jeu == "test":
            s, h, d = splits.test
            syn = np.array([is_synoptic(splits.stamp_of[x]) for x in s], bool)
            idx, seed = (np.nonzero(syn)[0] if syn.any() else np.arange(len(s))), 2000
        else:
            raise ValueError(f"jeu inconnu : {jeu!r}")
        if n_instants is not None and len(idx) > n_instants:
            idx = idx[np.unique(np.linspace(0, len(idx) - 1, n_instants).astype(int))]
        self.jeu = jeu
        self.ds = ANEO3D(s[idx], h[idx], d[idx], self.geom, train=False, seed=seed,
                         keep_eval=1.0)
        g = self.geom
        self.niveaux = [k for k in range(g.nz) if g.cover[k] >= wf.cfg.dso.min_level_cover]

    def passe(self, reseaux: dict[str, np.ndarray], carte_erreur: bool = False) -> dict:
        """Pour chaque reseau (masque (ny, nx) des capteurs ANEO ajoutes) : sommes par niveau
        (n, sse u, sse v, somme et somme des carres de la verite u, v) et, sur demande, la
        carte de l'erreur vectorielle quadratique moyenne par colonne."""
        g, wf = self.geom, self.wf
        noms = list(reseaux)
        nz = g.nz
        acc = {n: np.zeros((nz, 7)) for n in noms}
        cartes = {n: np.zeros((2, g.ny, g.nx)) for n in noms} if carte_erreur else None
        t0 = time.time()
        for it, (s_raw, sensor, tfeat, y, j) in enumerate(self.ds):
            S = s_raw.numpy()
            base = sensor.numpy() > 0.5
            ok_s = np.isfinite(S[:7]).all(axis=0)
            extra = tfeat.numpy()[N_TFEAT:] if g.forcage is not None else None
            masks = [base | (reseaux[n] & ok_s) for n in noms]
            sorties = wf._members(S, masks, float(self.ds.hours[j]), int(self.ds.doys[j]),
                                  extra=extra)
            yu, yv = y[0].numpy(), y[1].numpy()
            for n, (u, v, _w) in zip(noms, sorties):
                for k in self.niveaux:
                    m = g.valid[k] & np.isfinite(yu[k]) & np.isfinite(yv[k])
                    if not m.any():
                        continue
                    du, dv = u[k][m] - yu[k][m], v[k][m] - yv[k][m]
                    acc[n][k] += [m.sum(), (du ** 2).sum(), (dv ** 2).sum(), yu[k][m].sum(),
                                  (yu[k][m] ** 2).sum(), yv[k][m].sum(), (yv[k][m] ** 2).sum()]
                    if carte_erreur:
                        e2 = np.zeros((g.ny, g.nx))
                        e2[m] = du ** 2 + dv ** 2
                        cartes[n][0] += e2
                        cartes[n][1] += m
            if self.verbose and (it + 1) % 10 == 0:
                print(f"  {self.jeu} : {it + 1}/{len(self.ds)} instants, {len(noms)} reseaux "
                      f"({time.time() - t0:.0f} s)", flush=True)
        out = {n: _tableau(acc[n], g) for n in noms}
        if carte_erreur:
            for n in noms:
                c = cartes[n]
                out[n]["carte"] = np.where(c[1] > 0, c[0] / np.maximum(c[1], 1), np.nan)
        return out


def _tableau(a: np.ndarray, g) -> dict:
    lignes = []
    for k in range(g.nz):
        n, su, sv, mu, mu2, mv, mv2 = a[k]
        if n == 0:
            continue
        var_u, var_v = mu2 - mu ** 2 / n, mv2 - mv ** 2 / n
        lignes.append({"z_m": int(g.z_levels[k]), "n": int(n),
                       "rmse_u": math.sqrt(su / n), "rmse_v": math.sqrt(sv / n),
                       "r2_u": 1 - su / var_u if var_u > 0 else float("nan"),
                       "r2_v": 1 - sv / var_v if var_v > 0 else float("nan")})
    t = pd.DataFrame(lignes)
    n = a[:, 0].sum()
    return {"niveaux": t,
            "rmse_vect": math.sqrt((a[:, 1].sum() + a[:, 2].sum()) / max(n, 1)),
            "pire_rmse_niveau": float(t[["rmse_u", "rmse_v"]].max().max()) if len(t) else float("nan")}


# ---------------------------------------------------------------------- placement
def _choisir(score: np.ndarray, cand: np.ndarray, pris: np.ndarray, n: int, geom,
             d_min_m: float) -> list[tuple[int, int]]:
    """Les n candidats au plus fort score, a au moins d_min de tout capteur deja pris."""
    s = score[cand[:, 0], cand[:, 1]]
    ordre = np.argsort(-np.nan_to_num(s, nan=-np.inf))
    deja = [tuple(p) for p in pris]
    choisis: list[tuple[int, int]] = []
    for i in ordre:
        if not np.isfinite(s[i]):
            break
        jy, ix = int(cand[i, 0]), int(cand[i, 1])
        if all(math.hypot((jy - a) * geom.dy, (ix - b) * geom.dx) >= d_min_m
               for a, b in deja + choisis):
            choisis.append((jy, ix))
            if len(choisis) == n:
                break
    return choisis


def placement_glouton(banc_val: Banc, cand: np.ndarray, c: Contraintes, par_etape: int = 5,
                      rayon_influence_m: float = 1500.0,
                      installes: list[tuple[int, int]] | None = None) -> pd.DataFrame:
    """Selection gloutonne sur la validation : ordre d'installation et gain de chaque etape.

    installes : capteurs ANEO DEJA poses [(iy, ix)] (BF78, mise a jour du plan apres les premiers
    capteurs) : ils font partie du reseau des le depart, ne sont pas recomptes dans le budget `n_max` et
    le plan propose la suite a partir d'eux."""
    g = banc_val.geom
    pris = cellules_stations(g).reshape(-1, 2)
    aneo: list[tuple[int, int]] = [(int(a), int(b)) for a, b in (installes or [])]
    n_installes = len(aneo)
    r = banc_val.passe({"reseau": masque(g, aneo)}, carte_erreur=True)["reseau"]
    lignes = []
    etape = 0
    while len(aneo) - n_installes < c.n_max:
        carte = r["carte"]
        lisse_n = gaussian_filter(np.nan_to_num(carte), rayon_influence_m / g.dx)
        lisse_d = gaussian_filter(np.isfinite(carte).astype(float), rayon_influence_m / g.dx)
        score = np.where(lisse_d > 1e-6, lisse_n / np.maximum(lisse_d, 1e-6), np.nan)
        nouveaux = _choisir(score, cand, np.array(pris.tolist() + aneo).reshape(-1, 2),
                            min(par_etape, c.n_max - (len(aneo) - n_installes)), g, c.distance_min_m)
        if not nouveaux:
            break
        avant = r["rmse_vect"]
        aneo += nouveaux
        etape += 1
        r = banc_val.passe({"reseau": masque(g, aneo)}, carte_erreur=True)["reseau"]
        gain = (avant - r["rmse_vect"]) / len(nouveaux)
        for (jy, ix) in nouveaux:
            lignes.append({"ordre": len(lignes) + 1, "etape": etape, "iy": jy, "ix": ix,
                           "score_erreur_m2_s2": float(score[jy, ix]),
                           "gain_rmse_validation_m_s": float(gain),
                           "rmse_validation_apres_m_s": float(r["rmse_vect"])})
        if banc_val.verbose:
            print(f"OS6 placement : {len(aneo)} capteurs, RMSE vectoriel validation "
                  f"{avant:.3f} -> {r['rmse_vect']:.3f} m/s", flush=True)
    return pd.DataFrame(lignes)


def grille_reguliere(cand: np.ndarray, n: int, geom) -> np.ndarray:
    """n candidats au plus pres d'une grille reguliere (espacement ajuste au nombre)."""
    if n >= len(cand):
        return cand.copy()
    aire = len(np.unique(cand[:, 0])) * len(np.unique(cand[:, 1]))
    meilleur = None
    for f in np.linspace(0.5, 3.0, 26):
        pas = max(int(round(math.sqrt(aire / n) * f)), 1)
        jy0, ix0 = cand[:, 0].min(), cand[:, 1].min()
        noeuds = {}
        for jy, ix in cand:
            cle = (int(round((jy - jy0) / pas)), int(round((ix - ix0) / pas)))
            d = abs(jy - jy0 - cle[0] * pas) + abs(ix - ix0 - cle[1] * pas)
            if cle not in noeuds or d < noeuds[cle][0]:
                noeuds[cle] = (d, (jy, ix))
        pts = np.array([v[1] for v in noeuds.values()])
        if meilleur is None or abs(len(pts) - n) < abs(len(meilleur) - n) \
                or (len(pts) >= n and len(meilleur) < n):
            meilleur = pts
        if len(pts) <= n:
            break
    if len(meilleur) > n:
        meilleur = meilleur[np.unique(np.linspace(0, len(meilleur) - 1, n).astype(int))]
    return meilleur


def au_hasard(cand: np.ndarray, n: int, geom, d_min_m: float, graine: int) -> np.ndarray:
    rng = np.random.default_rng(graine)
    pris = cellules_stations(geom).reshape(-1, 2).tolist()
    out = []
    for i in rng.permutation(len(cand)):
        jy, ix = (int(v) for v in cand[i])
        if all(math.hypot((jy - a) * geom.dy, (ix - b) * geom.dx) >= d_min_m for a, b in pris):
            out.append((jy, ix))
            pris.append((jy, ix))
            if len(out) == n:
                break
    return np.array(out).reshape(-1, 2)


def courbe(banc_test: Banc, ordre: pd.DataFrame, cand: np.ndarray, c: Contraintes,
           tailles=TAILLES, graines=(0, 1, 2)) -> pd.DataFrame:
    """Erreur sur le TEST selon le nombre de capteurs : optimise, grille reguliere, hasard
    (moyenne de `graines` tirages), a nombre egal ; plus le reseau actuel (0 ANEO)."""
    g = banc_test.geom
    opt = ordre[["iy", "ix"]].to_numpy()
    tailles = [n for n in tailles if n <= len(opt)]
    reseaux = {"actuel|0|0": masque(g, [])}
    for n in tailles:
        reseaux[f"optimise|{n}|0"] = masque(g, opt[:n])
        reseaux[f"grille|{n}|0"] = masque(g, grille_reguliere(cand, n, g))
        for s in graines:
            reseaux[f"hasard|{n}|{s}"] = masque(g, au_hasard(cand, n, g, c.distance_min_m, s))
    res = banc_test.passe(reseaux)
    lignes = []
    for cle, r in res.items():
        strat, n, s = cle.split("|")
        for _, lv in r["niveaux"].iterrows():
            lignes.append({"strategie": strat, "n_capteurs": int(n), "graine": int(s),
                           "n_effectif": int(reseaux[cle].sum()), **lv.to_dict(),
                           "rmse_vect": r["rmse_vect"], "pire_rmse_niveau": r["pire_rmse_niveau"]})
    return pd.DataFrame(lignes)


def resume_courbe(t: pd.DataFrame, seuil: float) -> pd.DataFrame:
    """Une ligne par (strategie, n) : RMSE vectoriel et pire RMSE de niveau (hasard : moyenne
    des tirages), et le respect du seuil cible."""
    a = (t.groupby(["strategie", "n_capteurs", "graine"])[["rmse_vect", "pire_rmse_niveau"]]
         .first().groupby(["strategie", "n_capteurs"]).mean().reset_index())
    a["seuil_respecte"] = a["pire_rmse_niveau"] <= seuil
    return a.sort_values(["strategie", "n_capteurs"]).reset_index(drop=True)


def nombre_recommande(r: pd.DataFrame, strategie: str = "optimise") -> int | None:
    ok = r[(r["strategie"] == strategie) & r["seuil_respecte"]]
    return int(ok["n_capteurs"].min()) if len(ok) else None


# ---------------------------------------------------------------------- plan (S18)
def plan(ordre: pd.DataFrame, geom, georef) -> pd.DataFrame:
    p = ordre.copy()
    p.insert(0, "id", [f"ANEO-{i:03d}" for i in p["ordre"]])
    p["x_m"] = p["ix"] * geom.dx
    p["y_m"] = p["iy"] * geom.dy
    if georef is not None:
        lat, lon = georef.to_latlon(p["x_m"].to_numpy(), p["y_m"].to_numpy())
        p["lat"], p["lon"] = np.round(lat, 6), np.round(lon, 6)
    p["altitude_sol_m"] = geom.zt[p["iy"], p["ix"]].round(1)
    p["hauteur_mat_m"] = geom.zrel_sensor[p["iy"], p["ix"]].round(1)
    p["gain_cumule_m_s"] = p["gain_rmse_validation_m_s"].cumsum()
    return p


def geojson(p: pd.DataFrame, geom, georef, meta: dict) -> dict:
    feats = []
    for _, r in p.iterrows():
        props = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()
                 if k not in ("lat", "lon")}
        props["type"] = "capteur ANEO propose"
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]}})
    st_path = geom.cfg.paths.data_dir / "stations.json"
    if st_path.exists():
        for s in json.loads(st_path.read_text(encoding="utf-8")):
            lat, lon = georef.to_latlon(s["ix"] * geom.dx, s["iy"] * geom.dy)
            feats.append({"type": "Feature",
                          "properties": {"id": s.get("nom", "station"), "type": "station existante",
                                         "iy": int(s["iy"]), "ix": int(s["ix"])},
                          "geometry": {"type": "Point", "coordinates": [round(float(lon), 6),
                                                                        round(float(lat), 6)]}})
    return {"type": "FeatureCollection", "metadata": meta, "features": feats}


def contraintes_dict(c: Contraintes) -> dict:
    return asdict(c)


def cellules_installes(geom, georef, chemin: str | Path) -> list[tuple[int, int]]:
    """CSV des capteurs ANEO deja poses (colonnes lat, lon) -> cellules (iy, ix) dans le domaine (BF78)."""
    t = pd.read_csv(chemin)
    if not {"lat", "lon"} <= set(t.columns):
        raise ValueError(f"{chemin} : colonnes lat et lon attendues")
    x, y = georef.to_xy(t["lat"].to_numpy(float), t["lon"].to_numpy(float))
    ix, iy = np.rint(np.asarray(x) / geom.dx).astype(int), np.rint(np.asarray(y) / geom.dy).astype(int)
    ok = (ix >= 0) & (ix < geom.nx) & (iy >= 0) & (iy < geom.ny)
    if not ok.all():
        print(f"OS6 : {int((~ok).sum())} capteur(s) installe(s) hors du domaine, ignore(s)", flush=True)
    return sorted({(int(a), int(b)) for a, b in zip(iy[ok], ix[ok])})
