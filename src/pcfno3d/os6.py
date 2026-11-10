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

Mode NATIONAL (phase 5 de docs/feuille_de_route_nationale.md, regle
configs/national/regle_os6_national.json) : le meme OSSE sur toute la Tunisie, avec le candidat D
(entraine avec des capteurs ANEO simules). Trois differences, toutes en parametre ; sans elles, le
Grand Tunis garde exactement son comportement et ses fichiers :
    - capteur simule (`CapteursSimules`) comme a l'entrainement de D : verite d'entrainement a la
      hauteur du mat (25 m) au-dessus du sol (`aneo_entree.EntreeANEO.valeur_a_hauteur`) + bruit de
      mesure 0,1 m/s + 2 % de la vitesse (`EntreeANEO.SIGMA_CAPTEUR`), entre comme une station
      (prioritaire sur le pseudo-capteur ERA5) ; un seul tirage par (instant, maille), commun a toutes
      les dispositions (comparaison a bruit egal) ;
    - metrique « sol » : RMSE par tranche de hauteur au-dessus du sol (0-100, 100-300, 300-600,
      600-1000 m, comme scripts/dso_hauteur_sol.py) ; critere = RMSE u, v maximal sur les tranches ;
    - georeferencement de la grille (`GeoRefGrille`, projection de scripts/grille_reelle.py et
      d'InterpERA5) : sur 388 km d'est en ouest, l'equirectangulaire prise au coin sud-ouest
      (`realdata.GeoRef`) decalerait l'est du domaine d'environ 15 km.
"""
from __future__ import annotations

import hashlib
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
TAILLES_NATIONAL = (10, 20, 30, 50, 75, 100, 150, 200, 300, 400)
# tranches de hauteur au-dessus du sol : celles de scripts/dso_hauteur_sol.py (BORNES, PART_MIN)
BORNES_SOL = (0.0, 100.0, 300.0, 600.0, 1000.0)
PART_MIN_TRANCHE = 0.01


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


def zones_geojson(chemin: str | Path) -> list:
    """Zones exclues d'un GeoJSON (Polygon / MultiPolygon, coordonnees [lon, lat]) -> polygones
    [[lat, lon], ...] de `Contraintes.exclusions`. Seul l'anneau exterieur compte (un trou du
    polygone reste exclu : prudence)."""
    d = json.loads(Path(chemin).read_text(encoding="utf-8"))
    if d.get("type") == "FeatureCollection":
        geoms = [f.get("geometry") or {} for f in d.get("features", [])]
    elif d.get("type") == "Feature":
        geoms = [d.get("geometry") or {}]
    else:
        geoms = [d]
    polys = []
    for g in geoms:
        if g.get("type") == "Polygon":
            anneaux = [g["coordinates"][0]]
        elif g.get("type") == "MultiPolygon":
            anneaux = [p[0] for p in g["coordinates"]]
        else:
            raise ValueError(f"{chemin} : geometrie {g.get('type')!r} non prise en charge "
                             "(Polygon, MultiPolygon)")
        polys += [[[float(pt[1]), float(pt[0])] for pt in a] for a in anneaux]
    return polys


@dataclass
class GeoRefGrille:
    """Georeferencement EXACT d'une grille de scripts/grille_reelle.py : plan local a pas metrique
    constant, pris a la latitude centrale de la grille fine (meme calcul que forcage.InterpERA5,
    donc les memes lat, lon de noeud que l'entree ERA5 du modele). Noeud (iy, ix) <-> x = ix dx,
    y = iy dy. Meme interface que realdata.GeoRef (to_latlon, to_xy)."""

    lat0: float
    lon0: float
    m_lon: float
    m_lat: float

    @classmethod
    def depuis_geometrie(cls, geom) -> "GeoRefGrille":
        from .forcage import metres_par_degre
        r = geom.cfg.real
        if r.origin_lat is None or r.origin_lon is None:
            raise ValueError("georeferencement de la grille : real.origin_lat / origin_lon requis")
        if abs(float(r.rotation_deg)) > 1e-9:
            raise ValueError("georeferencement de la grille : grille tournee non prise en charge")
        lat0, lon0 = float(r.origin_lat), float(r.origin_lon)
        lat_c = lat0 + geom.ny_s * geom.dy_s / 2 / metres_par_degre(lat0)[1]
        m_lon, m_lat = metres_par_degre(lat_c)
        return cls(lat0, lon0, m_lon, m_lat)

    def to_latlon(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        return (self.lat0 + np.asarray(y, float) / self.m_lat,
                self.lon0 + np.asarray(x, float) / self.m_lon)

    def to_xy(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        return ((np.asarray(lon, float) - self.lon0) * self.m_lon,
                (np.asarray(lat, float) - self.lat0) * self.m_lat)


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


# ---------------------------------------------------------------------- capteurs simules (national)
class CapteursSimules:
    """Capteurs ANEO simules EXACTEMENT comme a l'entrainement du candidat D national
    (`aneo_entree.EntreeANEO.capteurs`, dont le code est reutilise tel quel) : verite d'entrainement
    interpolee a `hauteur_m` au-dessus du sol dans la colonne (`EntreeANEO.valeur_a_hauteur` : couche
    de surface du cache + noeuds valides, lineaire en ln h) + bruit gaussien par composante
    d'ecart-type `EntreeANEO.SIGMA_CAPTEUR` (0,1 m/s + 2 % de la vitesse). Ce sont des valeurs
    SIMULEES a partir de la verite d'entrainement, pas des mesures.

    Le bruit d'une maille a un instant est tire une fois (graine, jeu, instant) : toutes les
    dispositions qui contiennent cette maille y voient la meme valeur (comparaison a bruit egal)."""

    def __init__(self, geom, hauteur_m: float = 25.0, graine: int = 0) -> None:
        from .aneo_entree import EntreeANEO
        self._e = EntreeANEO(geom, None, 0.0, (0.0, 0.0), (0, 0), hauteur_m)
        self.sigma = EntreeANEO.SIGMA_CAPTEUR
        self.hauteur_m = float(hauteur_m)
        self.graine = int(graine)

    def valeurs(self, S_cache: np.ndarray, Y: np.ndarray, cellules: np.ndarray,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """u, v simules (n,) aux cellules (n, 2). S_cache = surface BRUTE du cache (avant l'entree
        de production), comme dans ANEO3D a l'entrainement de D ; Y = verite d'entrainement."""
        ny, nx = S_cache.shape[1:]
        z = rng.standard_normal((2, ny, nx))            # un tirage par maille, independant du reseau
        a, b = self.sigma
        cel = np.asarray(cellules, int).reshape(-1, 2)
        u, v = np.empty(len(cel)), np.empty(len(cel))
        for i, (iy, ix) in enumerate(cel):
            uu, vv = self._e.valeur_a_hauteur(S_cache, Y, int(iy), int(ix))
            sd = a + b * float(np.hypot(uu, vv))
            u[i], v[i] = uu + sd * z[0, iy, ix], vv + sd * z[1, iy, ix]
        return u.astype(np.float32), v.astype(np.float32)


def etiquettes_sol(geom, bornes=BORNES_SOL) -> np.ndarray:
    """(nz, ny, nx) int8 : indice de la tranche de hauteur au-dessus du sol de chaque noeud valide
    (h = z_niveau - z_terrain, intervalle [a, b)), -1 ailleurs. Memes noeuds que
    scripts/dso_hauteur_sol.masques_tranches ; leur reunion est le domaine 0-1 km."""
    h = np.asarray(geom.z_levels, float)[:, None, None] - np.asarray(geom.zt, float)[None]
    lab = np.full(h.shape, -1, np.int8)
    valid = np.asarray(geom.valid, bool)
    for t, (a, b) in enumerate(zip(bornes[:-1], bornes[1:])):
        lab[valid & (h >= a) & (h < b)] = t
    return lab


# ---------------------------------------------------------------------- passes du modele
class Banc:
    """Les instants d'un jeu (validation ou test) et les passes du modele avec des reseaux
    ANEO ajoutes a l'entree de production. Toutes les dispositions d'un instant passent en
    LOTS dans le reseau (`WindFieldModel._members`) : memes instants, meme entree de base.

    capteurs    : None (Grand Tunis) = le capteur voit la surface du cache a sa maille ;
                  `CapteursSimules` (national) = valeur simulee comme a l'entrainement de D.
    degradation : JSON de la degradation d'ERA5 du candidat D (courbe d'information) : ERA5 degradee
                  par le champ correle (`EntreeANEO.appliquer`, sans zones), memes tirages que
                  scripts/info_capteurs_v4.py (graine du jeu + instant).
    metrique    : "niveaux" (Grand Tunis : critere = RMSE vectoriel des niveaux couverts) ou "sol"
                  (national : critere = RMSE vectoriel sur 0-1 km au-dessus du sol ; seuil sur le
                  RMSE u, v maximal des tranches)."""

    def __init__(self, wf, jeu: str, n_instants: int | None = None, verbose: bool = True,
                 capteurs: CapteursSimules | None = None, degradation: str | Path | None = None,
                 metrique: str = "niveaux"):
        from .dataset import make_splits
        from .cache import load_index

        if metrique not in ("niveaux", "sol"):
            raise ValueError(f"metrique inconnue : {metrique!r}")
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
        aneo_eval = None
        if degradation is not None:
            from .aneo_entree import EntreeANEO
            aneo_eval = EntreeANEO(self.geom, degradation, 0.0, wf.cfg.mask.aneo_rayon_km)
        self.ds = ANEO3D(s[idx], h[idx], d[idx], self.geom, train=False, seed=seed,
                         keep_eval=1.0, aneo_eval=aneo_eval)
        g = self.geom
        self.niveaux = [k for k in range(g.nz) if g.cover[k] >= wf.cfg.dso.min_level_cover]
        self.capteurs = capteurs
        self.metrique = metrique
        self.condition = "era5_degradee" if degradation is not None else "propre"
        self._sol = etiquettes_sol(g) if metrique == "sol" else None

    def passe(self, reseaux: dict[str, np.ndarray], carte_erreur: bool = False) -> dict:
        """Pour chaque reseau (masque (ny, nx) des capteurs ANEO ajoutes) : sommes par niveau
        (n, sse u, sse v, somme et somme des carres de la verite u, v), par tranche au-dessus du
        sol en metrique « sol », et, sur demande, la carte de l'erreur vectorielle quadratique
        moyenne par colonne (niveaux couverts ; domaine 0-1 km au-dessus du sol en metrique « sol »).

        Les dispositions passent par lots de `reliability.batch` : la memoire ne tient qu'un lot de
        sorties (u, v, w) a la fois, quel que soit le nombre de dispositions."""
        g, wf = self.geom, self.wf
        noms = list(reseaux)
        nz = g.nz
        lot = max(1, int(wf.cfg.reliability.batch))
        nt = len(BORNES_SOL) - 1
        acc = {n: np.zeros((nz, 7)) for n in noms}
        acc_sol = {n: np.zeros((nt, 3)) for n in noms} if self._sol is not None else None
        cartes = {n: np.zeros((2, g.ny, g.nx)) for n in noms} if carte_erreur else None
        sel_niv = np.zeros(nz, bool)
        sel_niv[self.niveaux] = True
        valid = np.asarray(g.valid, bool)
        t0 = time.time()
        for it, (s_raw, sensor, tfeat, y, j) in enumerate(self.ds):
            S = s_raw.numpy()
            base = sensor.numpy() > 0.5
            ok_s = np.isfinite(S[:7]).all(axis=0)
            extra = tfeat.numpy()[N_TFEAT:] if g.forcage is not None else None
            Y = y.numpy()
            yu, yv = Y[0], Y[1]
            fin = valid & np.isfinite(yu) & np.isfinite(yv)
            lv = fin & sel_niv[:, None, None]
            yu0 = np.where(lv, yu.astype(np.float64), 0.0)     # sommes en float64 (R2 : mu2 - mu^2/n)
            yv0 = np.where(lv, yv.astype(np.float64), 0.0)
            verite = np.stack([lv.sum(axis=(1, 2)), yu0.sum(axis=(1, 2)), (yu0 ** 2).sum(axis=(1, 2)),
                               yv0.sum(axis=(1, 2)), (yv0 ** 2).sum(axis=(1, 2))], axis=1)
            if self._sol is not None:
                lab = np.where(fin, self._sol, -1).ravel()
                dom = lab >= 0
                lab_d = lab[dom].astype(np.int64)
                n_sol = np.bincount(lab_d, minlength=nt)
                m_carte = dom.reshape(fin.shape)
            else:
                m_carte = lv
            # capteurs simules (national) : valeurs aux mailles de toutes les dispositions de l'instant
            U = V = None
            if self.capteurs is not None:
                union = np.zeros_like(base)
                for n in noms:
                    union |= reseaux[n] & ok_s
                cel = np.argwhere(union)
                U, V = np.zeros(union.shape, np.float32), np.zeros(union.shape, np.float32)
                if len(cel):
                    S_cache = np.load(self.ds.cache_dir / f"{self.ds.stems[j]}_S.npy")
                    rng = np.random.default_rng([self.capteurs.graine, self.ds.seed, int(j)])
                    cu, cv = self.capteurs.valeurs(S_cache, Y, cel, rng)
                    U[cel[:, 0], cel[:, 1]], V[cel[:, 0], cel[:, 1]] = cu, cv
            for i0 in range(0, len(noms), lot):
                part = noms[i0:i0 + lot]
                masks = [base | (reseaux[n] & ok_s) for n in part]
                surfaces = None
                if U is not None:
                    surfaces = []
                    for n in part:
                        m = reseaux[n] & ok_s
                        Sn = S.copy()
                        Sn[0][m], Sn[1][m] = U[m], V[m]     # le capteur prime sur le pseudo-capteur ERA5
                        surfaces.append(Sn)
                sorties = wf._members(S, masks, float(self.ds.hours[j]), int(self.ds.doys[j]),
                                      extra=extra, surfaces=surfaces)
                for n, (u, v, _w) in zip(part, sorties):
                    du2 = np.where(fin, (u.astype(np.float64) - yu) ** 2, 0.0)
                    dv2 = np.where(fin, (v.astype(np.float64) - yv) ** 2, 0.0)
                    a = acc[n]
                    a[:, 0] += verite[:, 0]
                    a[:, 1] += np.where(lv, du2, 0.0).sum(axis=(1, 2))
                    a[:, 2] += np.where(lv, dv2, 0.0).sum(axis=(1, 2))
                    a[:, 3:] += verite[:, 1:]
                    if acc_sol is not None:
                        acc_sol[n][:, 0] += n_sol
                        acc_sol[n][:, 1] += np.bincount(lab_d, weights=du2.ravel()[dom], minlength=nt)
                        acc_sol[n][:, 2] += np.bincount(lab_d, weights=dv2.ravel()[dom], minlength=nt)
                    if carte_erreur:
                        cartes[n][0] += np.where(m_carte, du2 + dv2, 0.0).sum(axis=0)
                        cartes[n][1] += m_carte.sum(axis=0)
                del sorties
            if self.verbose and it == 0 and wf.device.type == "cuda":
                import torch
                print(f"  memoire GPU de pointe : {torch.cuda.max_memory_allocated() / 1e9:.1f} Go "
                      f"(lots de {lot} dispositions)", flush=True)
            if self.verbose and (it + 1) % 10 == 0:
                print(f"  {self.jeu} : {it + 1}/{len(self.ds)} instants, {len(noms)} reseaux "
                      f"({time.time() - t0:.0f} s)", flush=True)
        out = {n: _tableau(acc[n], g) for n in noms}
        for n in noms:
            out[n]["rmse_critere"] = out[n]["rmse_vect"]
            if acc_sol is not None:
                out[n].update(_tableau_sol(acc_sol[n]))
                out[n]["rmse_critere"] = out[n]["rmse_vect_sol"]
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


def _tableau_sol(a: np.ndarray, bornes=BORNES_SOL) -> dict:
    """Sommes (n, sse u, sse v) par tranche au-dessus du sol -> tableau et criteres nationaux :
    rmse_vect_sol (domaine 0-1 km) et rmse_max_uv_sol (max de RMSE u, v sur les tranches qui
    portent au moins PART_MIN_TRANCHE des noeuds du domaine, comme dso_hauteur_sol.verdicts)."""
    n_dom = float(a[:, 0].sum())
    lignes = []
    for t, (lo, hi) in enumerate(zip(bornes[:-1], bornes[1:])):
        n, su, sv = a[t]
        lignes.append({"tranche": f"{lo:g}-{hi:g} m", "n": int(n),
                       "rmse_u": math.sqrt(su / n) if n else float("nan"),
                       "rmse_v": math.sqrt(sv / n) if n else float("nan"),
                       "retenue": bool(n_dom > 0 and n >= PART_MIN_TRANCHE * n_dom)})
    lignes.append({"tranche": f"{bornes[0]:g}-{bornes[-1]:g} m (domaine)", "n": int(n_dom),
                   "rmse_u": math.sqrt(a[:, 1].sum() / n_dom) if n_dom else float("nan"),
                   "rmse_v": math.sqrt(a[:, 2].sum() / n_dom) if n_dom else float("nan"),
                   "retenue": False})
    t = pd.DataFrame(lignes)
    r = t[t["retenue"]]
    return {"tranches": t,
            "rmse_vect_sol": math.sqrt((a[:, 1].sum() + a[:, 2].sum()) / max(n_dom, 1)),
            "rmse_max_uv_sol": float(r[["rmse_u", "rmse_v"]].max().max()) if len(r) else float("nan")}


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
    Score et gain portent sur le critere du banc (RMSE vectoriel des niveaux couverts, ou du
    domaine 0-1 km au-dessus du sol en metrique « sol ») ; memes colonnes dans les deux cas.

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
        avant = r["rmse_critere"]
        aneo += nouveaux
        etape += 1
        r = banc_val.passe({"reseau": masque(g, aneo)}, carte_erreur=True)["reseau"]
        gain = (avant - r["rmse_critere"]) / len(nouveaux)
        for (jy, ix) in nouveaux:
            lignes.append({"ordre": len(lignes) + 1, "etape": etape, "iy": jy, "ix": ix,
                           "score_erreur_m2_s2": float(score[jy, ix]),
                           "gain_rmse_validation_m_s": float(gain),
                           "rmse_validation_apres_m_s": float(r["rmse_critere"])})
        if banc_val.verbose:
            quoi = " (0-1 km au-dessus du sol)" if banc_val.metrique == "sol" else ""
            print(f"OS6 placement : {len(aneo)} capteurs, RMSE vectoriel validation{quoi} "
                  f"{avant:.3f} -> {r['rmse_critere']:.3f} m/s", flush=True)
    return pd.DataFrame(lignes)


def _noeuds_grille(cand: np.ndarray, pas: int) -> np.ndarray:
    """Pour chaque noeud d'une grille de pas `pas` (en mailles, origine au coin des candidats), le
    candidat le plus proche (distance de Manhattan ; a egalite, le premier dans l'ordre des
    candidats), les noeuds ranges dans l'ordre de leur premier candidat. Vectorise : meme resultat
    que la boucle d'origine (dictionnaire rempli dans l'ordre des candidats), sans boucle Python
    sur les ~150 000 candidats nationaux."""
    jy0, ix0 = cand[:, 0].min(), cand[:, 1].min()
    dy, dx = (cand[:, 0] - jy0).astype(np.int64), (cand[:, 1] - ix0).astype(np.int64)
    ky, kx = np.rint(dy / pas).astype(np.int64), np.rint(dx / pas).astype(np.int64)
    d = np.abs(dy - ky * pas) + np.abs(dx - kx * pas)
    cle = ky * (int(kx.max()) + 1) + kx
    rang = np.arange(len(cand))
    tri = np.lexsort((rang, d, cle))                   # par noeud, puis distance, puis ordre
    _, debut = np.unique(cle[tri], return_index=True)
    choisi = tri[debut]                                # un candidat par noeud (cles croissantes)
    _, premier = np.unique(cle, return_index=True)     # premiere apparition de chaque noeud
    return cand[choisi[np.argsort(premier, kind="stable")]]


def grille_reguliere(cand: np.ndarray, n: int, geom) -> np.ndarray:
    """n candidats au plus pres d'une grille reguliere (espacement ajuste au nombre)."""
    if n >= len(cand):
        return cand.copy()
    aire = len(np.unique(cand[:, 0])) * len(np.unique(cand[:, 1]))
    meilleur = None
    for f in np.linspace(0.5, 3.0, 26):
        pas = max(int(round(math.sqrt(aire / n) * f)), 1)
        pts = _noeuds_grille(cand, pas)
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


def reseaux_courbe(geom, ordre: pd.DataFrame, cand: np.ndarray, c: Contraintes,
                   tailles=TAILLES, graines=(0, 1, 2)) -> dict[str, np.ndarray]:
    """Les dispositions de la courbe : reseau actuel, puis optimise / grille / hasard par taille."""
    opt = ordre[["iy", "ix"]].to_numpy()
    tailles = [n for n in tailles if n <= len(opt)]
    reseaux = {"actuel|0|0": masque(geom, [])}
    for n in tailles:
        reseaux[f"optimise|{n}|0"] = masque(geom, opt[:n])
        reseaux[f"grille|{n}|0"] = masque(geom, grille_reguliere(cand, n, geom))
        for s in graines:
            reseaux[f"hasard|{n}|{s}"] = masque(geom, au_hasard(cand, n, geom, c.distance_min_m, s))
    return reseaux


def courbe(banc_test: Banc, ordre: pd.DataFrame, cand: np.ndarray, c: Contraintes,
           tailles=TAILLES, graines=(0, 1, 2), par_tranche: bool = False):
    """Erreur sur le TEST selon le nombre de capteurs : optimise, grille reguliere, hasard
    (moyenne de `graines` tirages), a nombre egal ; plus le reseau actuel (0 ANEO).

    Metrique « sol » : chaque ligne porte aussi rmse_vect_sol et rmse_max_uv_sol ; avec
    par_tranche=True, renvoie (par niveau, par tranche au-dessus du sol)."""
    g = banc_test.geom
    reseaux = reseaux_courbe(g, ordre, cand, c, tailles, graines)
    res = banc_test.passe(reseaux)
    lignes, lignes_t = [], []
    for cle, r in res.items():
        strat, n, s = cle.split("|")
        tete = {"strategie": strat, "n_capteurs": int(n), "graine": int(s),
                "n_effectif": int(reseaux[cle].sum())}
        sol = ({"rmse_vect_sol": r["rmse_vect_sol"], "rmse_max_uv_sol": r["rmse_max_uv_sol"]}
               if "rmse_vect_sol" in r else {})
        for _, lv in r["niveaux"].iterrows():
            lignes.append({**tete, **lv.to_dict(), "rmse_vect": r["rmse_vect"],
                           "pire_rmse_niveau": r["pire_rmse_niveau"], **sol})
        if "tranches" in r:
            for _, lt in r["tranches"].iterrows():
                lignes_t.append({**tete, **lt.to_dict(), **sol})
    t = pd.DataFrame(lignes)
    return (t, pd.DataFrame(lignes_t)) if par_tranche else t


def resume_courbe(t: pd.DataFrame, seuil: float, critere: str = "pire_rmse_niveau") -> pd.DataFrame:
    """Une ligne par (strategie, n) : RMSE vectoriel et pire RMSE de niveau (hasard : moyenne
    des tirages), et le respect du seuil cible par `critere` (national : rmse_max_uv_sol, avec les
    colonnes rmse_vect_sol et rmse_max_uv_sol en plus)."""
    cols = ["rmse_vect", "pire_rmse_niveau"] + [k for k in ("rmse_vect_sol", "rmse_max_uv_sol")
                                                if k in t.columns]
    a = (t.groupby(["strategie", "n_capteurs", "graine"])[cols]
         .first().groupby(["strategie", "n_capteurs"]).mean().reset_index())
    a["seuil_respecte"] = a[critere] <= seuil
    return a.sort_values(["strategie", "n_capteurs"]).reset_index(drop=True)


def nombre_recommande(r: pd.DataFrame, strategie: str = "optimise") -> int | None:
    ok = r[(r["strategie"] == strategie) & r["seuil_respecte"]]
    return int(ok["n_capteurs"].min()) if len(ok) else None


# ---------------------------------------------------------------------- plan (S18)
def plan(ordre: pd.DataFrame, geom, georef, hauteur_mat_m: float | None = None) -> pd.DataFrame:
    """Plan de deploiement. hauteur_mat_m : hauteur du mat simule (national : 25 m) ; sans elle,
    celle de la couche de surface du cache a la maille (Grand Tunis)."""
    p = ordre.copy()
    p.insert(0, "id", [f"ANEO-{i:03d}" for i in p["ordre"]])
    p["x_m"] = p["ix"] * geom.dx
    p["y_m"] = p["iy"] * geom.dy
    if georef is not None:
        lat, lon = georef.to_latlon(p["x_m"].to_numpy(), p["y_m"].to_numpy())
        p["lat"], p["lon"] = np.round(lat, 6), np.round(lon, 6)
    p["altitude_sol_m"] = geom.zt[p["iy"], p["ix"]].round(1)
    if hauteur_mat_m is None:
        p["hauteur_mat_m"] = geom.zrel_sensor[p["iy"], p["ix"]].round(1)
    else:
        p["hauteur_mat_m"] = round(float(hauteur_mat_m), 1)
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


# ---------------------------------------------------------------------- national : regle et modele
def charger_regle(chemin: str | Path) -> tuple[dict, str]:
    """La regle OS6 (JSON) et son empreinte, VERIFIEE contre le fichier `<regle>.sha256` voisin
    (fixe avant tout calcul) : une regle modifiee apres coup est refusee."""
    chemin = Path(chemin)
    h = hashlib.sha256(chemin.read_bytes()).hexdigest()
    f = chemin.parent / (chemin.name + ".sha256")
    if not f.exists():
        raise ValueError(f"{chemin} : empreinte {f.name} absente (regle non fixee)")
    attendu = f.read_text(encoding="ascii").split()[0]
    if h != attendu:
        raise ValueError(f"{chemin} : sha256 {h} different de l'empreinte fixee {attendu}")
    return json.loads(chemin.read_text(encoding="utf-8")), h


def meilleure_graine(runs: str | Path, prefixe: str = "national_1km_v4_D",
                     graines=(42, 43, 44)) -> dict:
    """L'entrainement retenu parmi les graines d'un candidat : minimum de val_score de la
    VALIDATION (history.json, meme score que le choix de la regle v4). Le test n'est jamais lu."""
    runs = Path(runs)
    scores = {}
    for g in graines:
        h = runs / f"{prefixe}_s{g}" / "history.json"
        if not h.exists():
            raise FileNotFoundError(f"{prefixe} graine {g} : {h} absent (entrainement non fait)")
        vals = [float(e["val_score"]) for e in json.loads(h.read_text(encoding="utf-8"))
                if isinstance(e, dict) and e.get("val_score") is not None]
        if not vals:
            raise ValueError(f"{h} : aucun val_score")
        scores[int(g)] = min(vals)
    g = min(scores, key=scores.get)
    return {"candidat": prefixe, "scores_validation": {str(k): v for k, v in scores.items()},
            "graine": int(g), "run_dir": str(runs / f"{prefixe}_s{g}"),
            "critere": "minimum de val_score (history.json, validation) ; le test n'est pas lu"}
