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
