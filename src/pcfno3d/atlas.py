"""Atlas eolien local (cahier v12, E8) et carte du gisement (16.5) : vitesse moyenne, loi de Weibull,
production annuelle et facteur de charge a hauteur de moyeu, a la resolution du modele (100 m),
construits a partir de l'historique des reconstructions ; trois meilleurs emplacements espaces d'au
moins 6 km.

Par maille, on cumule les moments de la vitesse au moyeu (n, somme v, somme v^2, somme v^3) sur tous
les instants : la loi de Weibull vient de la moyenne et de l'ecart type (methode des moments,
Justus 1978 : k = (sigma / mu)^-1,086, c = mu / Gamma(1 + 1/k)), la production annuelle de la courbe
de puissance sous cette loi (`windfarm.weibull_aep`, tabulee en (k, c) pour les 380 000 mailles), la
densite de l'air de l'atmosphere standard a l'altitude du moyeu.

Limites, dites dans les sorties : l'historique est celui des instants du cache (saisons et heures de
la generation, pas une annee continue) ; aucune production mesuree ; une seule courbe de reference ;
sillages et pertes non comptes.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .atom_wind import champ_agl
from .windfarm import RHO_REF, PowerCurve, weibull_aep

COURBE_DEFAUT = "generique-2MW"
DISTANCE_SITES_M = 6000.0
PENTE_MAX_SITE_DEG = 15.0


def charger_courbe(chemin: str | Path, nom: str = COURBE_DEFAUT) -> PowerCurve:
    import yaml
    d = yaml.safe_load(Path(chemin).read_text(encoding="utf-8"))["power_curves"][nom]
    return PowerCurve(model=nom, speed_ms=d["speed_ms"], power_kw=d["power_kw"], rated_kw=d["rated_kw"],
                      cut_in_ms=d["cut_in_ms"], cut_out_ms=d["cut_out_ms"], rho_ref=d.get("rho_ref", RHO_REF),
                      source=d.get("source", ""))


class Cumul:
    """Moments de la vitesse au moyeu par maille."""

    def __init__(self, forme) -> None:
        self.a = np.zeros((4,) + tuple(forme))

    def ajouter(self, vitesse: np.ndarray) -> None:
        ok = np.isfinite(vitesse)
        v = np.where(ok, vitesse, 0.0)
        self.a += np.stack([ok, v, v ** 2, v ** 3])

    def moments(self) -> dict:
        n, s1, s2, s3 = self.a
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(n > 0, s1 / n, np.nan)
            sd = np.sqrt(np.maximum(np.where(n > 1, s2 / n - mu ** 2, np.nan), 0.0))
            cube = np.where(n > 0, s3 / n, np.nan)
        return {"n": n, "moyenne": mu, "ecart_type": sd, "cube_moyen": cube}


def weibull_moments(mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.clip((sd / mu) ** -1.086, 0.8, 10.0)
        c = mu / np.vectorize(lambda x: math.gamma(1 + 1 / x) if np.isfinite(x) else np.nan)(k)
    return k, c


def table_aep(courbe: PowerCurve, rho: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Facteur de charge tabule sur une grille (k, c) pour l'interpolation."""
    ks = np.linspace(0.8, 6.0, 53)
    cs = np.linspace(0.5, 20.0, 196)
    fc = np.array([[weibull_aep(courbe, k, c, rho)[1] for c in cs] for k in ks])
    return ks, cs, fc


def facteur_charge(k: np.ndarray, c: np.ndarray, table) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator
    ks, cs, fc = table
    f = RegularGridInterpolator((ks, cs), fc, bounds_error=False, fill_value=np.nan)
    pts = np.stack([np.clip(k, ks[0], ks[-1]), np.clip(c, cs[0], cs[-1])], -1)
    out = f(pts.reshape(-1, 2)).reshape(k.shape)
    return np.where(np.isfinite(k) & np.isfinite(c), out, np.nan)


def meilleurs_sites(fc: np.ndarray, geom, n: int = 3, d_min_m: float = DISTANCE_SITES_M) -> list[tuple[int, int]]:
    """Les n mailles au plus fort facteur de charge, sur terre ferme, pente <= 15 deg, a au moins
    d_min les unes des autres."""
    ok = np.isfinite(fc) & (geom.z0 >= 0.005) & (geom.slope <= PENTE_MAX_SITE_DEG)
    ordre = np.argsort(-np.where(ok, fc, -np.inf), axis=None)
    pris: list[tuple[int, int]] = []
    for flat in ordre:
        iy, ix = np.unravel_index(flat, fc.shape)
        if not ok[iy, ix]:
            break
        if all(math.hypot((iy - a) * geom.dy, (ix - b) * geom.dx) >= d_min_m for a, b in pris):
            pris.append((int(iy), int(ix)))
            if len(pris) == n:
                break
    return pris


def construire(wf, stems, hub_agl_m: float = 100.0, courbe: PowerCurve | None = None, verbose: bool = True) -> dict:
    """Atlas a la resolution du modele. Les reconstructions sont celles du service (predict_stem)."""
    import time

    geom = wf.geom
    cum = Cumul(geom.zt.shape)
    t0 = time.time()
    for i, stem in enumerate(stems):
        S, hour, doy, stamp = wf.surface_from_cache(stem)
        vol = wf.predict(S, hour, doy, timestamp=stamp, reliability=False)
        u, v = champ_agl(vol.u, vol.v, geom, hub_agl_m)
        cum.ajouter(np.hypot(u, v))
        if verbose and (i + 1) % 25 == 0:
            print(f"  atlas : {i + 1}/{len(stems)} instants ({time.time() - t0:.0f} s)", flush=True)
    m = cum.moments()
    k, c = weibull_moments(m["moyenne"], m["ecart_type"])
    # densite de l'atmosphere standard au moyeu (moyenne du domaine : la table reste unique)
    z_moyeu = float(np.nanmean(geom.zt)) + hub_agl_m
    rho = RHO_REF * math.exp(-z_moyeu / 8500.0)
    courbe = courbe or charger_courbe(Path(__file__).resolve().parents[2] / "configs" / "eolien" / "parc_demo.yaml")
    fc = facteur_charge(k, c, table_aep(courbe, rho))
    aep = fc * courbe.rated_kw * 8760.0 / 1000.0
    sites = meilleurs_sites(fc, geom)
    return {"hub_agl_m": hub_agl_m, "instants": len(stems), "courbe": courbe.model, "rated_kw": courbe.rated_kw,
            "rho_kg_m3": rho, "moyenne": m["moyenne"], "ecart_type": m["ecart_type"], "k": k, "c": c,
            "facteur_charge": fc, "aep_mwh": aep, "densite_puissance_w_m2": 0.5 * rho * m["cube_moyen"],
            "sites": sites}


def ecrire(atlas: dict, geom, sortie: str | Path) -> dict:
    """atlas.npz, GeoTIFF (E15) de chaque couche, atlas.png et atlas.json (sites en lat/lon)."""
    from .realdata import GeoRef
    from .sig import ecrire_geotiff, grille_geo

    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    couches = ("moyenne", "facteur_charge", "aep_mwh", "k", "c", "densite_puissance_w_m2")
    np.savez_compressed(sortie / "atlas.npz", **{k: np.asarray(atlas[k], np.float32) for k in couches})
    tif = {}
    try:
        g = grille_geo(geom)
        for k in couches:
            tif[k] = str(ecrire_geotiff(sortie / f"atlas_{k}.tif", atlas[k], g["lon0"], g["lat0"], g["dlon"], g["dlat"]).name)
    except ValueError as exc:
        tif["erreur"] = str(exc)
    try:
        geo = GeoRef.from_cfg(geom.cfg)
    except ValueError:
        geo = None
    sites = []
    for r, (iy, ix) in enumerate(atlas["sites"], 1):
        s = {"rang": r, "iy": iy, "ix": ix, "x_m": ix * geom.dx, "y_m": iy * geom.dy,
             "vitesse_moyenne_m_s": round(float(atlas["moyenne"][iy, ix]), 2),
             "facteur_charge": round(float(atlas["facteur_charge"][iy, ix]), 3),
             "aep_mwh": round(float(atlas["aep_mwh"][iy, ix]), 0),
             "altitude_m": round(float(geom.zt[iy, ix]), 0)}
        if geo is not None:
            la, lo = geo.to_latlon(s["x_m"], s["y_m"])
            s["lat"], s["lon"] = round(float(la), 5), round(float(lo), 5)
        sites.append(s)
    resume = {"hub_agl_m": atlas["hub_agl_m"], "instants": atlas["instants"], "courbe": atlas["courbe"],
              "rated_kw": atlas["rated_kw"], "rho_kg_m3": round(atlas["rho_kg_m3"], 4),
              "vitesse_moyenne_domaine_m_s": round(float(np.nanmean(atlas["moyenne"])), 2),
              "facteur_charge_domaine": round(float(np.nanmean(atlas["facteur_charge"])), 3),
              "sites": sites, "geotiff": tif,
              "limites": ("historique = instants du cache (saisons et heures de la generation, pas une annee "
                          "continue) ; aucune production mesuree ; courbe de reference unique ; sillages et "
                          "pertes non comptes ; aide a la decision a confirmer par une campagne de mesure")}
    import json
    (sortie / "atlas.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
    _figure(atlas, geom, sortie / "atlas.png")
    return resume


def _figure(atlas: dict, geom, chemin: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    ext = (0, geom.nx * geom.dx / 1000, 0, geom.ny * geom.dy / 1000)
    for ax, (cle, titre, cmap) in zip(axes, (("moyenne", f"Vitesse moyenne a {atlas['hub_agl_m']:g} m (m/s)", "viridis"),
                                             ("facteur_charge", f"Facteur de charge ({atlas['courbe']})", "magma"))):
        im = ax.imshow(atlas[cle], origin="lower", extent=ext, cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=0.8)
        for r, (iy, ix) in enumerate(atlas["sites"], 1):
            ax.plot(ix * geom.dx / 1000, iy * geom.dy / 1000, "w*", ms=14, mec="k")
            ax.annotate(str(r), (ix * geom.dx / 1000, iy * geom.dy / 1000), xytext=(6, 6), textcoords="offset points",
                        color="w", fontsize=11, weight="bold")
        ax.set_title(titre, fontsize=10)
        ax.set_xlabel("km")
    fig.savefig(chemin, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- atlas NATIONAL (phase 7)
# Plusieurs hauteurs en UNE passe du modele par instant, densite de l'air PAR MAILLE (le relief
# national va de 0 a 1 544 m : une densite moyenne du domaine fausserait la production de ~15 %),
# moments separes par groupe d'instants (validation, test, entrainement) pour combiner ensuite sans
# repasser le modele, et la reference « ERA5 seule » (profil ERA5 aux noeuds, base du modele
# residuel) sur les memes instants et les memes mailles. Les fonctions du Grand Tunis ci-dessus
# sont inchangees.
HAUTEURS_NATIONAL = (50.0, 100.0, 150.0)
COUCHES_NATIONAL = ("moyenne", "ecart_type", "densite_puissance_w_m2", "k", "c", "facteur_charge", "aep_mwh")
H_ECHELLE_ISA_M = 8500.0


def rho_isa(z_asl_m) -> np.ndarray:
    """Densite de l'air (kg/m3), atmosphere standard simplifiee du Grand Tunis (H = 8 500 m)."""
    return RHO_REF * np.exp(-np.asarray(z_asl_m, float) / H_ECHELLE_ISA_M)


def vitesse_era5_agl(geom, vecteur: np.ndarray, hauteurs) -> dict[float, np.ndarray] | None:
    """{h: vitesse ERA5 (profil aux noeuds, base du modele residuel) a h m au-dessus du sol} ; None
    pour un modele sans forcage."""
    if geom.interp_era5 is None or geom.forcage is None:
        return None
    from .forcage import N_SCAL
    uv = geom.interp_era5.dense_np(np.asarray(vecteur)[N_SCAL:])
    u = np.where(geom.valid, uv[0], np.nan)
    v = np.where(geom.valid, uv[1], np.nan)
    return {float(h): np.hypot(*champ_agl(u, v, geom, float(h))) for h in hauteurs}


def construire_hauteurs(wf, groupes: dict[str, list[str]], hauteurs=HAUTEURS_NATIONAL, era5: bool = True,
                        verbose: bool = True) -> dict:
    """Moments de la vitesse a chaque hauteur AGL, par source (« modele », « era5 ») et par groupe
    d'instants ({nom: stems}). Une passe du modele par instant (entree de production du cache)."""
    import time

    geom = wf.geom
    cum: dict[tuple[str, str, float], Cumul] = {}
    t0, n = time.time(), 0
    total = sum(len(s) for s in groupes.values())
    for g, stems in groupes.items():
        for stem in stems:
            S, hour, doy, stamp = wf.surface_from_cache(stem)
            vol = wf.predict(S, hour, doy, timestamp=stamp, reliability=False)
            ref = (vitesse_era5_agl(geom, geom.forcage.vecteur(stamp), hauteurs)
                   if (era5 and geom.forcage is not None) else None)
            for h in hauteurs:
                u, v = champ_agl(vol.u, vol.v, geom, float(h))
                cum.setdefault(("modele", g, float(h)), Cumul(geom.zt.shape)).ajouter(np.hypot(u, v))
                if ref is not None:
                    cum.setdefault(("era5", g, float(h)), Cumul(geom.zt.shape)).ajouter(ref[float(h)])
            n += 1
            if verbose and n % 25 == 0:
                print(f"  atlas national : {n}/{total} instants ({time.time() - t0:.0f} s)", flush=True)
    return {"hauteurs": [float(h) for h in hauteurs], "groupes": {g: len(s) for g, s in groupes.items()},
            "cumuls": cum}


def combiner(cumuls: dict, source: str, groupes, h: float) -> Cumul | None:
    """Somme des moments des `groupes` (les moments s'additionnent)."""
    out = None
    for g in groupes:
        c = cumuls.get((source, g, float(h)))
        if c is None:
            continue
        if out is None:
            out = Cumul(c.a.shape[1:])
        out.a += c.a
    return out


def couches(cum: Cumul, geom, h: float, courbe: PowerCurve, table=None) -> dict:
    """Vitesse moyenne, ecart type, densite de puissance 0,5 rho <v^3> (rho de la maille, ISA a
    z_terrain + h), Weibull (k, c) par les moments, facteur de charge et production annuelle de la
    courbe sous la loi de Weibull de la vitesse NORMALISEE en densite (IEC 61400-12-1 : v_n = v
    (rho / rho_ref)^(1/3), soit c_n = c (rho / rho_ref)^(1/3), k inchange) — exact, avec une seule
    table (k, c) a rho_ref."""
    m = cum.moments()
    k, c = weibull_moments(m["moyenne"], m["ecart_type"])
    rho = rho_isa(np.asarray(geom.zt, float) + h)
    table = table if table is not None else table_aep(courbe, courbe.rho_ref)
    fc = facteur_charge(k, c * (rho / courbe.rho_ref) ** (1.0 / 3.0), table)
    return {"n": m["n"], "moyenne": m["moyenne"], "ecart_type": m["ecart_type"],
            "densite_puissance_w_m2": 0.5 * rho * m["cube_moyen"], "k": k, "c": c,
            "facteur_charge": fc, "aep_mwh": fc * courbe.rated_kw * 8760.0 / 1000.0, "rho_kg_m3": rho}


def masque_terre(geom, z0_min: float = 0.005) -> np.ndarray:
    """Terre ferme (z0 >= 0,005 m, comme les candidats OS6) avec une colonne au-dessus du relief."""
    return (np.asarray(geom.z0) >= z0_min) & np.asarray(geom.valid).any(axis=0)


def meilleures_zones(fc: np.ndarray, geom, n: int, d_min_m: float, pente_max_deg: float,
                     z0_min: float = 0.005, polygone=None) -> list[tuple[int, int]]:
    """Les n mailles au plus fort facteur de charge, sur terre ferme, pente <= pente_max, a au moins
    d_min les unes des autres ; `polygone` [(lat, lon), ...] : seulement les mailles dedans (ex. le
    territoire tunisien, s'il est fourni ; aucun n'est invente)."""
    ok = np.isfinite(fc) & masque_terre(geom, z0_min) & (np.asarray(geom.slope) <= pente_max_deg)
    if polygone is not None:
        from .agriculture import latlon_noeuds
        from .os6 import _dans_polygone
        lat, lon = latlon_noeuds(geom)
        ok &= _dans_polygone(lat, lon, polygone)
    ordre = np.argsort(-np.where(ok, fc, -np.inf), axis=None)
    pris: list[tuple[int, int]] = []
    for flat in ordre:
        iy, ix = np.unravel_index(flat, fc.shape)
        if not ok[iy, ix]:
            break
        if all(math.hypot((iy - a) * geom.dy, (ix - b) * geom.dx) >= d_min_m for a, b in pris):
            pris.append((int(iy), int(ix)))
            if len(pris) == n:
                break
    return pris


def blocs_csv(couches_h: dict, geom, b: int):
    """Couches moyennees par blocs b x b mailles de terre (lisible dans un tableur : la grille
    complete reste dans les GeoTIFF et le .npz)."""
    import pandas as pd

    from .agriculture import latlon_noeuds
    from .atom_wind import blocs

    terre = masque_terre(geom)
    lat, lon = latlon_noeuds(geom)
    la, lo = blocs(lat, b), blocs(lon, b)
    part = blocs(terre.astype(float), b)
    lignes = []
    for h, cc in couches_h.items():
        moy = {k: blocs(np.where(terre, cc[k], np.nan), b) for k in COUCHES_NATIONAL}
        for (j, i), p in np.ndenumerate(part):
            if not p > 0 or not np.isfinite(moy["moyenne"][j, i]):
                continue
            lignes.append({"hauteur_m": h, "lat": round(float(la[j, i]), 4), "lon": round(float(lo[j, i]), 4),
                           "part_terre": round(float(p), 3),
                           **{k: round(float(moy[k][j, i]), 4) for k in COUCHES_NATIONAL}})
    return pd.DataFrame(lignes)
