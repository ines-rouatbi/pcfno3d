"""Lot M5 — eolien (pcfno3d.windfarm, critere EOL1).

Tests unitaires sur des cas construits a la main (direction meteo sur les cas cardinaux,
sol vs mer, NaN sous le relief, densite, REWS sur profil connu, Weibull sur echantillon
connu, rose des vents), conformite aux schemas du contrat ATOM rev. 2 (copies dans
tests/data/atom_rev2), puis integration sur le mini-dataset : desactive = rien ne change,
active = intervalles de l'ensemble M2, API, MCP, et mecanisme EOL1 sur CPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pcfno3d.config import REPO_ROOT, ReliabilityCfg, WindFarmCfg
from pcfno3d.realdata import isa
from pcfno3d.windfarm import (RHO_REF, FarmCalc, FarmContext, PowerCurve, TurbineCalc,
                              _power, _state, alert_id, compute_farm,
                              density_normalized_speed, farm_from_dict, fit_weibull,
                              hub_density, hub_wind_kpi, iso_utc, load_farm,
                              meteo_direction, profile_at, rews, rotor_strips,
                              round_direction, round_veer, shear_alpha, weibull_aep,
                              wind_rose, windrose_document, wrap180)

SCHEMAS = Path(__file__).parent / "data" / "atom_rev2"
DEMO = REPO_ROOT / "configs" / "eolien" / "parc_demo.yaml"
WCFG = WindFarmCfg()


def _validator(name: str):
    """Validateur JSON Schema 2020-12 du contrat ATOM rev. 2 (refs vers common resolues)."""
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    docs = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in SCHEMAS.glob("*.json")}
    registry = referencing.Registry().with_resources(
        (d["$id"], referencing.Resource.from_contents(d)) for d in docs.values())
    return jsonschema.Draft202012Validator(docs[f"{name}.schema.json"], registry=registry)


def _valide(name: str, doc: dict) -> None:
    errs = sorted(_validator(name).iter_errors(doc), key=lambda e: list(e.path))
    assert not errs, [f"{list(e.path)} : {e.message}" for e in errs[:5]]


def _courbe(**kw) -> dict:
    base = {"rated_kw": 2000, "cut_in_ms": 3.0, "cut_out_ms": 25.0, "source": "test",
            "speed_ms": [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 25, 25.01],
            "power_kw": [0, 0, 70, 180, 340, 560, 840, 1180, 1520, 1820, 2000, 2000, 0]}
    return {**base, **kw}


def _colonnes(geom, n=2, hub=60.0, r=20.0):
    """Colonnes ou le rotor [hub - r, hub + r] est encadre par des noeuds valides, de relief
    le plus different possible (pour distinguer sol et mer)."""
    h = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    hv = np.where(geom.valid, h, np.nan)
    lo, hi = np.nanmin(hv, axis=0), np.nanmax(hv, axis=0)
    ok = (lo <= hub - r - 1) & (hi >= hub + r + 1)
    jj, ii = np.nonzero(ok)
    order = np.argsort(geom.zt[jj, ii])
    pick = [order[0], order[-1]] if n == 2 else list(order[np.linspace(0, len(order) - 1, n)
                                                         .astype(int)])
    return [(int(jj[k]), int(ii[k])) for k in pick]


def _parc(geom, cols, hub=60.0, d=40.0, **extra) -> dict:
    return {"exemple": True, "power_curves": {"g2": _courbe()},
            "turbines": [{"id": f"T{n}", "x_m": i * geom.dx, "y_m": j * geom.dy,
                          "hub_agl_m": hub, "rotor_d_m": d, "model": "g2", **extra}
                         for n, (j, i) in enumerate(cols)]}


def _volume(geom, u, v, stamp="2025-01-06 12:00:00"):
    from pcfno3d.inference import Volume
    w = np.zeros_like(u)
    nan = ~geom.valid
    return Volume(np.where(nan, np.nan, u).astype(np.float32),
                  np.where(nan, np.nan, v).astype(np.float32),
                  np.where(nan, np.nan, w).astype(np.float32), geom, timestamp=stamp)


def _h_agl(geom):
    return np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]


# ------------------------------------------------------------------ conventions
def test_direction_meteo_cas_cardinaux():
    """D'ou VIENT le vent : u > 0 (vers l'est) = vent d'ouest (270), v > 0 = vent du sud."""
    u = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
    v = np.array([0.0, 1.0, 0.0, -1.0, 1.0])
    np.testing.assert_allclose(meteo_direction(u, v), [270, 180, 90, 0, 225], atol=1e-9)
    assert ((meteo_direction(u, v) >= 0) & (meteo_direction(u, v) < 360)).all()


def test_veer_signe_horaire_et_bornes():
    """veer = dir(haut) - dir(bas) dans [-180, 180), positif dans le sens horaire (D9)."""
    assert wrap180(10 - 350) == pytest.approx(20)        # 350 -> 10 : horaire
    assert wrap180(350 - 10) == pytest.approx(-20)       # 10 -> 350 : antihoraire
    assert wrap180(180 - 0) == pytest.approx(-180)       # borne basse incluse
    assert wrap180(0 - 180) == pytest.approx(-180)
    assert -180 <= float(wrap180(359.999)) < 180


def test_arrondi_avant_modulo_aux_bornes(geom, cfg):
    """Regle m4 : arrondir AVANT le modulo. veer dans [-180, 180) (exclusiveMaximum 180),
    direction dans [0, 360)."""
    assert float(round_veer(179.96)) == -180.0                 # et non 180,0
    assert float(round_veer(-179.96)) == -180.0
    assert float(round_veer(179.94)) == 179.9
    assert float(round_direction(359.96)) == 0.0               # et non 360,0
    assert float(round_direction(359.94)) == 359.9
    assert float(round_direction(-0.04)) == 0.0
    # jusqu'au document du contrat : 179,96 ne sort jamais en 180,0
    farm = farm_from_dict(_parc(geom, _colonnes(geom)), geom, cfg)
    c = TurbineCalc(farm.turbines[0], farm.curves["g2"], "production", hub_speed=8.0,
                    rews=8.0, v_n=8.0, veer_deg=179.96, rho=1.2,
                    rho_source="atmosphere_standard", power_kw=800.0)
    doc = c.to_contract()
    assert doc["veer_deg"] == -180.0
    _valide("wind-farm", {"turbines": [doc], "power_curves": {},
                          "farm": {"power_kw": {"p10": None, "p50": 800.0, "p90": None},
                                   "rated_kw": 2000, "n_production": 1, "n_arret": 0}})
    c.veer_deg = -179.96
    assert c.to_contract()["veer_deg"] == -180.0
    c.hub_direction = 359.96
    det = FarmCalc(farm, [c], "2025-01-06 12:00:00", 0.0, None, None, 0, {}).details()[0]
    assert det["hub_direction_deg"] == 0.0


def test_identifiant_alerte_sha8_du_contrat():
    """id = sha8 de « kind|cible » : meme valeur que la fixture du contrat (alerts.json)."""
    assert alert_id("cisaillement", "T-03") == "bdec1bd1"
    assert alert_id("vent_fort_moyeu", "T-03") != alert_id("cisaillement", "T-03")
    # comme pcfno3d.alerts (M4) : pas d'instant invente
    assert iso_utc("2025-01-06 12:00:00") == "2025-01-06T12:00:00Z"
    with pytest.raises(ValueError):
        iso_utc(None)


# ------------------------------------------------------------------ courbe, densite
def test_courbe_cut_in_cut_out_et_densite_iec():
    c = PowerCurve("g", **{k: v for k, v in _courbe().items() if k != "source"})
    p = c.power([0, 2.9, 3.0, 8.0, 12, 25.0, 25.005, 26, np.nan])
    assert p.tolist() == [0, 0, 0, 840, 2000, 2000, 0, 0, 0]
    # IEC 61400-12-1 : v_n = v (rho / rho_ref)^(1/3) ; 0,9^3 = 0,729 -> v_n = 9 exactement
    assert density_normalized_speed(10.0, RHO_REF * 0.729) == pytest.approx(9.0)
    assert c.power(density_normalized_speed(10.0, RHO_REF * 0.729)) == pytest.approx(1180)
    with pytest.raises(ValueError):
        PowerCurve("x", [0, 5, 4], [0, 1, 2], 2, 3, 25)        # vitesses non croissantes
    with pytest.raises(ValueError, match="P > 0"):                # trou entre cut-in et out
        PowerCurve("x", [0, 3, 4, 5, 25], [0, 0, 0, 5, 5], 5, 3, 25)
    with pytest.raises(ValueError, match="cut-out"):              # table trop courte
        PowerCurve("x", [0, 3, 12], [0, 0, 5], 5, 3, 25)


def test_regle_unique_etat_et_puissance(geom, cfg):
    """Etat et puissance sur la MEME vitesse (REWS normalise en densite) : production <=>
    P > 0. Cas qui etait incoherent : moyeu juste sous le cut-out et rho > rho_ref."""
    c = PowerCurve("g", **{k: v for k, v in _courbe().items() if k != "source"})
    rng = np.random.default_rng(11)
    for v, rho in zip(rng.uniform(0, 35, 4000), rng.uniform(0.9, 1.4, 4000)):
        vn = float(density_normalized_speed(v, rho))
        assert (_state(vn, c) == "production") == (float(_power(v, rho, c)) > 0), (v, rho)
    for v in (3.0, 25.0, 25.0001):
        assert (_state(v, c) == "production") == (float(c.power(v)) > 0)
    cols = _colonnes(geom)
    vol = _volume(geom, np.full(geom.shape, 24.8), np.zeros(geom.shape))
    lourd = farm_from_dict(_parc(geom, cols, air_density=1.30), geom, cfg)
    for t in compute_farm(vol, lourd, WCFG).turbines:
        assert t.hub_speed == pytest.approx(24.8) and t.hub_speed < 25.0
        assert t.v_n == pytest.approx(24.8 * (1.30 / RHO_REF) ** (1 / 3)) and t.v_n > 25
        assert t.state == "arret_vent_fort" and t.power_kw == 0.0
    normal = farm_from_dict(_parc(geom, cols, air_density=RHO_REF), geom, cfg)
    for t in compute_farm(vol, normal, WCFG).turbines:
        assert t.state == "production" and t.power_kw == 2000.0


def test_courbes_de_demo_documentees_et_sous_betz():
    """Parc d'exemple : marque comme exemple, courbes sourcees, Cp <= 16/27 (Betz)."""
    geom = SimpleNamespace(nx=500, ny=250, dx=100.0, dy=100.0)
    farm = load_farm(DEMO, geom)
    assert farm.exemple and len(farm.turbines) == 3 and farm.sha256
    assert all("exemple" in t.name for t in farm.turbines)
    for t in farm.turbines:
        c = farm.curves[t.model]
        assert c.source and c.rho_ref == RHO_REF
        area = math.pi * (t.rotor_d_m / 2) ** 2
        v = c.speed_ms[c.speed_ms > 0]
        cp = c.power(v) * 1000.0 / (0.5 * RHO_REF * area * v ** 3)
        assert cp.max() <= 16 / 27, (t.model, cp.max())
    assert farm.curves["nrel-5mw-cp"].rated_kw == 5000           # NREL/TP-500-38060


def test_declaration_refuse_les_cas_ambigus(geom, cfg):
    cols = _colonnes(geom)
    ok = _parc(geom, cols)
    farm = farm_from_dict(ok, geom, cfg)
    assert [(t.j, t.i) for t in farm.turbines] == cols
    for mod, motif in (
            (lambda d: d["turbines"][0].update(model="inconnu"), "power_curves"),
            (lambda d: d["turbines"][1].update(id="T0"), "double"),
            (lambda d: d["turbines"][0].update(x_m=1e6), "hors du domaine"),
            (lambda d: d["turbines"][0].update(hub_agl_m=15.0), "au-dessus du sol"),
            (lambda d: (d["turbines"][0].pop("x_m"), d["turbines"][0].pop("y_m"),
                        d["turbines"][0].update(lat=36.8, lon=10.1)), "georeference")):
        d = json.loads(json.dumps(ok))
        mod(d)
        with pytest.raises(ValueError, match=motif):
            farm_from_dict(d, geom, cfg)


def test_densite_isa_surface_ou_mesuree(geom, cfg):
    cols = _colonnes(geom)
    farm = farm_from_dict(_parc(geom, cols), geom, cfg)
    t = farm.turbines[0]
    z_hub = float(geom.zt[t.j, t.i]) + t.hub_agl_m                 # ASL = relief + AGL
    rho, src, _ = hub_density(t, geom, None, 2000.0)
    assert src == "atmosphere_standard" and rho == pytest.approx(float(isa(z_hub)[2]))
    assert float(isa(0.0)[2]) == pytest.approx(1.225, abs=1e-3)
    # capteur visible dans la colonne : rho de surface ramene au moyeu par le rapport ISA
    rho_s = np.full((geom.ny, geom.nx), 1.10)
    sensor = np.zeros((geom.ny, geom.nx), bool)
    sensor[t.j, t.i] = True
    rho2, src2, info = hub_density(t, geom, FarmContext(farm.columns, None, rho_s, sensor), 2e3)
    attendu = 1.10 * float(isa(z_hub)[2]) / float(isa(geom.z_sensor[t.j, t.i])[2])
    assert src2 == "extrapolation_surface" and rho2 == pytest.approx(attendu)
    assert rho2 < 1.10 and info["distance_m"] == 0.0
    # aucun capteur visible dans le rayon -> ISA
    rho3, src3, _ = hub_density(t, geom, FarmContext(farm.columns, None, rho_s, ~sensor & False),
                                2e3)
    assert src3 == "atmosphere_standard"
    t.air_density = 1.05
    assert hub_density(t, geom, None, 2e3)[:2] == (1.05, "mesure")


# ------------------------------------------------------------------ profil, REWS, cisaillement
def test_rews_profil_connu():
    hs, w = rotor_strips(100.0, 80.0, 20)
    assert w.sum() == pytest.approx(1.0) and np.allclose(w, w[::-1])
    assert hs.min() > 60 and hs.max() < 140
    # profil uniforme : REWS = vitesse
    assert rews(np.full(20, 7.3), w) == pytest.approx(7.3)
    # profil lineaire v = a + b y : E[v^3] = a^3 + 3 a b^2 E[y^2], E[y^2] = R^2 / 4 (disque)
    a, b, r = 8.0, 0.05, 40.0
    attendu = (a ** 3 + 0.75 * a * b * b * r * r) ** (1 / 3)
    assert rews(a + b * (hs - 100.0), w) == pytest.approx(attendu, rel=1e-3)
    assert rews(a + b * (hs - 100.0), w) > a                    # moyenne du CUBE (Jensen)
    # loi de puissance : comparaison a une integration fine
    hf, wf_ = rotor_strips(100.0, 80.0, 4000)
    prof = lambda h: 9.0 * (h / 100.0) ** 0.25                  # noqa: E731
    assert rews(prof(hs), w) == pytest.approx(rews(prof(hf), wf_), rel=1e-3)
    assert rews(np.r_[np.full(19, 5.0), np.nan], w) != rews(np.full(20, 5.0), w)


def test_cisaillement_loi_de_puissance():
    v = lambda h: 8.0 * (h / 100.0) ** 0.2                      # noqa: E731
    assert float(shear_alpha(v(50), v(150), 50, 150)) == pytest.approx(0.2)
    assert np.isnan(shear_alpha(0.0, 5.0, 50, 150))
    # extrapolation sous le premier noeud : loi de puissance des deux premiers noeuds
    h = np.array([-40.0, 30.0, 80.0, 180.0])
    s = np.array([np.nan, v(30), v(80), v(180)])
    out = profile_at(h, s, np.zeros(4), np.array([10.0, 20.0, 30.0, 100.0]))
    assert out["extrapole"] and out["alpha_bas"] == pytest.approx(0.2)
    np.testing.assert_allclose(out["speed"][:3], [v(10), v(20), v(30)], rtol=1e-9)
    assert np.isnan(profile_at(h, s, np.zeros(4), np.array([500.0]))["speed"][0])


def test_hauteur_au_dessus_du_sol_et_pas_de_la_mer(geom, cfg):
    """vitesse = 0,1 x hauteur AU-DESSUS DU SOL : le moyeu a 60 m lit 6,0 m/s quel que soit
    le relief. Une confusion avec l'altitude (ASL) donnerait 0,1 x (60 + relief)."""
    cols = _colonnes(geom)
    assert abs(geom.zt[cols[0]] - geom.zt[cols[1]]) > 10           # reliefs differents
    farm = farm_from_dict(_parc(geom, cols), geom, cfg)
    vol = _volume(geom, 0.1 * _h_agl(geom), np.zeros(geom.shape))
    calc = compute_farm(vol, farm, WCFG)
    attendu_rews = (6.0 ** 3 + 0.75 * 6.0 * 0.1 ** 2 * 20.0 ** 2) ** (1 / 3)
    for c in calc.turbines:
        assert c.state == "production"
        assert c.hub_speed == pytest.approx(6.0, abs=1e-5)
        assert c.rews == pytest.approx(attendu_rews, rel=2e-3)
        assert c.shear_alpha == pytest.approx(1.0, rel=1e-6)     # v proportionnel a h
        assert c.info["hub_asl_m"] == pytest.approx(float(geom.zt[c.turbine.j, c.turbine.i])
                                                     + 60.0, abs=0.01)
    kpi = hub_wind_kpi(vol, 60.0)
    assert kpi["hub_wind_mean_ms"] == pytest.approx(6.0) and kpi["hub_wind_max_ms"] == 6.0
    assert kpi["hub_height_agl_m"] == 60.0


def test_nan_et_valeurs_sous_le_relief_ignorees(geom, cfg):
    cols = _colonnes(geom)
    farm = farm_from_dict(_parc(geom, cols), geom, cfg)
    u = 0.1 * _h_agl(geom)
    ref = compute_farm(_volume(geom, u, np.zeros(geom.shape)), farm, WCFG)
    # une valeur FINIE mais absurde sous le relief ne doit jamais etre lue
    vol = _volume(geom, u, np.zeros(geom.shape))
    vol.u = np.where(geom.valid, vol.u, 999.0).astype(np.float32)
    vol.v = np.where(geom.valid, vol.v, 999.0).astype(np.float32)
    calc = compute_farm(vol, farm, WCFG)
    for a, b in zip(calc.turbines, ref.turbines):
        assert a.hub_speed == b.hub_speed and a.rews == b.rews
    assert hub_wind_kpi(vol, 60.0) == hub_wind_kpi(_volume(geom, u, 0 * u), 60.0)
    # colonne entierement NaN : eolienne indisponible, tout a null dans le contrat
    j, i = cols[0]
    vol.u[:, j, i] = np.nan
    c = compute_farm(vol, farm, WCFG)
    doc = c.document()["turbines"][0]
    assert doc["state"] == "indisponible" and doc["hub_speed"]["p50"] is None
    assert doc["power_kw"] == {"p10": None, "p50": None, "p90": None}
    assert doc["air_density"] is None and doc["air_density_source"] is None
    assert c.document()["farm"]["n_production"] == 1 and c.n_arret == 0


def test_direction_et_veer_sur_un_volume(geom, cfg):
    """Vent DU NORD (v < 0) -> 0 ; direction tournant dans le sens horaire avec la hauteur
    (0,5 degre par metre) -> veer > 0, ~ 0,5 x (haut - bas du rotor)."""
    cols = _colonnes(geom)
    farm = farm_from_dict(_parc(geom, cols), geom, cfg)
    h = np.clip(_h_agl(geom), 0, None)
    d = np.radians(0.5 * h)
    s = 8.0
    vol = _volume(geom, -s * np.sin(d), -s * np.cos(d))
    for c in compute_farm(vol, farm, WCFG).turbines:
        assert c.hub_direction == pytest.approx(30.0, abs=3.0)
        assert c.veer_deg == pytest.approx(20.0, abs=4.0) and c.veer_deg > 0
    nord = _volume(geom, np.zeros(geom.shape), np.full(geom.shape, -6.0))
    for c in compute_farm(nord, farm, WCFG).turbines:
        assert c.hub_direction == pytest.approx(0.0, abs=1e-6) or \
            c.hub_direction == pytest.approx(360.0, abs=1e-6)
        assert c.veer_deg == pytest.approx(0.0, abs=1e-6)
        assert c.shear_alpha == pytest.approx(0.0, abs=1e-9)


def test_etats_et_alertes_du_contrat(geom, cfg):
    cols = _colonnes(geom, n=2)
    farm = farm_from_dict(_parc(geom, cols), geom, cfg)
    # v = 0,5 h : 30 m/s au moyeu (> cut-out 25, < 35) et alpha = 1 sur le rotor
    u = 0.5 * _h_agl(geom)
    calc = compute_farm(_volume(geom, u, np.zeros(geom.shape)), farm, WCFG)
    assert {c.state for c in calc.turbines} == {"arret_vent_fort"}
    assert calc.power_kw == 0.0 and calc.n_arret == 2
    items = calc.alerts(WCFG)
    kinds = {(a["kind"], a["target"]) for a in items}
    assert kinds == {(k, t.id) for k in ("vent_fort_moyeu", "cisaillement")
                     for t in farm.turbines}
    for a in items:
        assert a["id"] == alert_id(a["kind"], a["target"]) and a["turbine_id"] == a["target"]
        assert a["raised_at"] == "2025-01-06T12:00:00Z" and a["status"] == "ouverte"
    vf = [a for a in items if a["kind"] == "vent_fort_moyeu"]
    assert all(a["level"] == "important" and a["unit"] == "m/s" for a in vf)
    cis = [a for a in items if a["kind"] == "cisaillement"]
    assert all(a["level"] == "important" and a["threshold"] == WCFG.shear_important
               and a["value"] == pytest.approx(1.0) for a in cis)
    _valide("alerts", {"items": items})
    # vent faible : arret, aucune alerte
    calme = compute_farm(_volume(geom, np.full(geom.shape, 1.0), 0 * u), farm, WCFG)
    assert {c.state for c in calme.turbines} == {"arret_vent_faible"}
    assert calme.alerts(WCFG) == []


# ------------------------------------------------------------------ Weibull, AEP, rose
def test_weibull_retrouve_k_et_c():
    rng = np.random.default_rng(7)
    for k, c in ((2.1, 9.4), (1.6, 6.0), (3.0, 12.0)):
        x = c * rng.weibull(k, 20000)
        kk, cc = fit_weibull(x)
        assert kk == pytest.approx(k, rel=0.03) and cc == pytest.approx(c, rel=0.02)
    assert fit_weibull([5.0, 5.0, 5.0]) is None and fit_weibull([np.nan]) is None


def test_aep_analytique_et_densite():
    # courbe constante = rated entre 0 et 25 m/s : facteur de capacite = F(25) de Weibull
    plate = PowerCurve("p", [0, 25], [1000, 1000], 1000, 0.0, 25.0)
    aep, cf = weibull_aep(plate, 2.0, 10.0)
    assert cf == pytest.approx(1 - math.exp(-(25 / 10) ** 2), abs=1e-3)
    assert aep == pytest.approx(cf * 1000 * 8760 / 1000)
    c = PowerCurve("g", **{k: v for k, v in _courbe().items() if k != "source"})
    a1, _ = weibull_aep(c, 2.0, 8.0, RHO_REF)
    a2, _ = weibull_aep(c, 2.0, 8.0, 1.05)                      # air moins dense
    assert 0 < a2 < a1


def test_rose_des_vents_frequences_et_secteurs():
    d = np.array([0, 359, 14.9, 15, 90, 180, 270, 45, 10, 200])
    s = np.array([5, 7, 4, 8, 30, 2, 13, 0.1, np.nan, 9])
    n, nc, sec = wind_rose(s, d, 12, WCFG.speed_bin_edges_ms, 0.2)
    assert n == 9 and nc == 1                       # le calme compte, le NaN est exclu
    assert sum(x["frequency"] for x in sec) == pytest.approx(1.0, abs=0.01)
    assert [x["dir_center_deg"] for x in sec] == [30.0 * k for k in range(12)]
    f = {x["dir_center_deg"]: x for x in sec}
    c = 1 / 12                                      # calme reparti a parts egales
    assert f[0.0]["frequency"] == pytest.approx((3 + c) / 9, abs=1e-4)   # 0, 359, 14,9
    assert f[30.0]["frequency"] == pytest.approx((1 + c) / 9, abs=1e-4)  # 15
    assert f[60.0]["frequency"] == pytest.approx(c / 9, abs=1e-4)        # calme seul
    assert f[60.0]["speed_bins"][0] == pytest.approx(c / 9, abs=1e-4)    # 1re classe
    assert f[90.0]["speed_bins"][-1] == pytest.approx(1 / 9, abs=1e-4)   # classe ouverte
    for x in sec:
        assert sum(x["speed_bins"]) == pytest.approx(x["frequency"], abs=1e-3)
    # 14,96 s'arrondit a 15,0 avant le classement : secteur 30, comme sa valeur publiee
    _, _, sec2 = wind_rose([5.0], [14.96], 12, WCFG.speed_bin_edges_ms, 0.2)
    assert sec2[1]["frequency"] == 1.0
    # que des calmes : n_valid > 0, somme 1 ; aucun instant valide : pas de rose (None)
    n3, nc3, sec3 = wind_rose([0.1, 0.0], [np.nan, np.nan], 12, WCFG.speed_bin_edges_ms)
    assert n3 == nc3 == 2
    assert sum(x["frequency"] for x in sec3) == pytest.approx(1.0, abs=0.01)
    assert windrose_document("T", 0, 0, 60, ["2025-01-01"], [np.nan], [np.nan], WCFG) is None
    rng = np.random.default_rng(3)
    doc = windrose_document("T-01", 100.0, 200.0, 60.0,
                            ["2025-01-01 00:00", "2025-01-06 18:00"] * 250,
                            9.4 * rng.weibull(2.1, 500), rng.uniform(0, 360, 500), WCFG,
                            PowerCurve("g", **{k: v for k, v in _courbe().items()
                                               if k != "source"}), 1.2)
    assert sum(x["frequency"] for x in doc["sectors"]) == pytest.approx(1.0, abs=0.01)
    assert doc["period"] == {"from": "2025-01-01T00:00:00Z", "to": "2025-01-06T18:00:00Z"}
    assert 0 < doc["weibull"]["capacity_factor"] < 1 and doc["weibull"]["aep_mwh"] > 0
    _valide("windrose", doc)


# ------------------------------------------------------------------ integration (mini-jeu)
@pytest.fixture(scope="module")
def trained(cfg, index, norm):
    from pcfno3d.train import Trainer
    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg


@pytest.fixture(scope="module")
def instant(trained):
    from pcfno3d.cache import load_index
    return str(load_index(trained)["stem"].iloc[0])


@pytest.fixture
def parc_test(trained, geom, tmp_path):
    path = tmp_path / "parc.yaml"
    path.write_text(json.dumps(_parc(geom, _colonnes(geom, n=3))), encoding="utf-8")
    return path


@pytest.fixture
def wf_parc(trained, parc_test):
    """Modele avec un parc declare ; reliability en option. Configuration restauree apres."""
    from pcfno3d.inference import WindFieldModel
    made = []

    def make(reliability=None, members=3):
        trained.wind_farm.turbines = parc_test
        trained.inference.reliability = reliability
        trained.reliability.members = members
        wf = WindFieldModel(trained, device="cpu", verbose=False)
        made.append(wf)
        return wf
    try:
        yield make
    finally:
        trained.wind_farm.turbines = None
        trained.inference.reliability = None
        trained.reliability.members = ReliabilityCfg().members


def test_desactive_rien_ne_change(trained, instant):
    from pcfno3d.inference import WindFieldModel
    assert trained.wind_farm.turbines is None
    wf = WindFieldModel(trained, device="cpu", verbose=False)
    assert wf.farm is None
    vol = wf.predict_stem(instant)
    assert vol.farm_context is None


def test_parc_sans_fiabilite_contrat_et_valeur_servie(wf_parc, trained, instant):
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.windfarm import farm_report
    ref = WindFieldModel(trained, device="cpu", verbose=False).predict_stem(instant)
    wf = wf_parc()
    vol = wf.predict_stem(instant)
    np.testing.assert_array_equal(vol.u, ref.u)                 # le parc ne touche a rien
    assert vol.farm_context is not None and vol.farm_context.members_uv is None
    rep = farm_report(vol, wf)
    _valide("wind-farm", rep["wind_farm"])
    _valide("alerts", rep["alerts"])
    json.dumps(rep, allow_nan=False)
    assert rep["uncertainty"] == {"method": "aucune", "members": 1, "calibrated": False}
    for t in rep["wind_farm"]["turbines"]:
        assert t["hub_speed"]["p10"] is None and t["hub_speed"]["p90"] is None
        assert t["air_density_source"] in ("extrapolation_surface", "atmosphere_standard")
    assert rep["wind_farm"]["farm"]["power_kw"]["p10"] is None
    assert set(rep["kpi"]) == {"hub_wind_mean_ms", "hub_wind_max_ms", "hub_height_agl_m"}
    assert rep["parc"]["exemple"] is True


def test_parc_avec_ensemble_m2_intervalles(wf_parc, instant):
    wf = wf_parc(reliability="non_calibre", members=3)
    vol = wf.predict_stem(instant)
    g = wf.geom
    assert vol.farm_context.members_uv.shape == (3, 2, g.nz, 3)
    calc = compute_farm(vol, wf.farm, wf.cfg.wind_farm, calibration=wf.reliability)
    doc = calc.document()
    _valide("wind-farm", doc)
    f = doc["farm"]["power_kw"]
    assert f["p10"] <= f["p50"] <= f["p90"]
    assert f["p50"] == pytest.approx(sum(t["power_kw"]["p50"] for t in doc["turbines"]), abs=0.5)
    for c, t in zip(calc.turbines, doc["turbines"]):
        if t["state"] == "indisponible":
            continue
        hs = t["hub_speed"]
        assert hs["p10"] <= hs["p50"] <= hs["p90"]
        # D6 : intervalle symetrique +/- 1,2816 sigma autour de la passe servie
        assert hs["p90"] - hs["p50"] == pytest.approx(1.2816 * c.hub_sigma, abs=0.011)
        assert t["power_kw"]["p10"] <= t["power_kw"]["p50"] <= t["power_kw"]["p90"]
        assert 0 <= t["power_kw"]["p90"] <= t["rated_kw"]
    assert calc.uncertainty["calibrated"] is False and calc.uncertainty["members"] == 3


def test_api_et_mcp_production_eolienne(wf_parc, instant):
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp")
    from fastapi import HTTPException

    from service import app as svc
    from service import mcp_server
    wf = wf_parc()
    old, old_mcp = dict(svc.STATE), mcp_server._MODEL
    try:
        svc.STATE["model"] = wf
        svc.STATE["volumes"] = {}
        body = svc.wind_farm(stem=instant, _=None)
        mcp_server.set_model(wf)
        outil = mcp_server.production_eolienne(stem=instant)
        farm, wf.farm = wf.farm, None                       # parc non declare : 404
        with pytest.raises(HTTPException) as exc:
            svc.wind_farm(stem=instant, _=None)
        assert exc.value.status_code == 404
        with pytest.raises(ValueError, match="wind_farm.turbines"):
            mcp_server.production_eolienne(stem=instant)
        wf.farm = farm
    finally:
        svc.STATE.clear()
        svc.STATE.update(old)
        mcp_server._MODEL = old_mcp
        mcp_server._VOLUMES.clear()
    _valide("wind-farm", body["wind_farm"])
    json.dumps(body, allow_nan=False)
    assert outil["stem"] == instant and outil["wind_farm"] == body["wind_farm"]


def test_eol1_mecanisme_sur_le_mini_jeu(wf_parc, instant, monkeypatch):
    from pcfno3d.windfarm import eol1, truth_volume
    wf = wf_parc()
    r = eol1(wf, wf.farm, n_instants=2)
    assert r["instants"] == 2 and r["capteurs_visibles"] == 1.0
    assert r["seuil"] == 0.05 and r["parc"]["eoliennes"] == 3
    assert r["ecart_relatif"] is None or r["ecart_relatif"] >= 0
    assert isinstance(r["eol1_conforme"], (bool, type(None)))
    assert 0 <= r["accord_etats"] <= 1 and r["nmae_puissance"] >= 0
    json.dumps(r, allow_nan=False)
    # si la reconstruction EST la verite, l'ecart est nul et EOL1 conforme
    monkeypatch.setattr(wf, "predict", lambda S, hour, doy, timestamp=None, **kw:
                        truth_volume(wf.cfg, wf.geom, wf._stem_courant, timestamp))
    orig = wf.surface_from_cache

    def surface(stem):
        wf._stem_courant = stem
        return orig(stem)
    monkeypatch.setattr(wf, "surface_from_cache", surface)
    r0 = eol1(wf, wf.farm, n_instants=2)
    if r0["ecart_relatif"] is not None:
        assert r0["ecart_relatif"] == 0.0 and r0["eol1_conforme"] is True
    assert r0["nmae_puissance"] == 0.0 and r0["accord_etats"] == 1.0


def test_eol1_memes_instants_et_masques_que_dso1(wf_parc):
    """Les capteurs visibles d'EOL1 sont ceux de l'evaluation DSO1 (eval_loader, graine
    2000, position dans la selection synoptique), y compris en sous-echantillonnant."""
    from pcfno3d.cache import load_index
    from pcfno3d.dataset import eval_loader, is_synoptic, make_splits
    from pcfno3d.windfarm import eol1
    wf = wf_parc()
    splits = make_splits(wf.cfg, load_index(wf.cfg))
    te_s = splits.test[0]
    syn = np.array([is_synoptic(splits.stamp_of[x]) for x in te_s], dtype=bool)
    idx = np.nonzero(syn)[0] if syn.any() else np.arange(len(te_s))
    loader, stems = eval_loader(wf.cfg, wf.geom, splits, idx, 2000, keep=0.5)
    attendu = {str(st): int(b[1].sum()) for st, b in zip(stems, loader)}
    r = eol1(wf, wf.farm, keep=0.5, n_instants=2)
    assert r["instants"] == 2
    for row in r["par_instant"]:
        assert row["capteurs"] == attendu[row["stem"]], row["stem"]
        assert str(te_s[idx][row["position_dso1"]]) == row["stem"]


def test_alertes_jamais_sans_horodatage(geom, cfg):
    farm = farm_from_dict(_parc(geom, _colonnes(geom)), geom, cfg)
    vol = _volume(geom, 0.5 * _h_agl(geom), np.zeros(geom.shape), stamp=None)
    calc = compute_farm(vol, farm, WCFG)
    with pytest.raises(ValueError, match="horodatage"):
        calc.alerts(WCFG)
    out = calc.to_json(WCFG)
    assert out["alerts"] == {"items": []} and out["alertes_non_produites"]


def test_script_eol1_ecrit_dans_un_dossier_neuf(trained, parc_test, monkeypatch):
    import importlib.util
    for var, p in (("ANEO_DATA", trained.paths.data_dir), ("ANEO_CACHE", trained.paths.cache_dir),
                   ("ANEO_RUN", trained.paths.run_dir)):
        monkeypatch.setenv(var, str(p))
    spec = importlib.util.spec_from_file_location(
        "m5_script", REPO_ROOT / "scripts" / "m5_eol1_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    avant = {p: p.stat().st_mtime_ns for p in trained.paths.run_dir.rglob("*") if p.is_file()}
    rc = mod.main(["-c", "test", "--device", "cpu", "--parc", str(parc_test),
                   "--instants", "1", "--keep", "1.0", "0.5"])
    assert rc == 0
    for p, t in avant.items():
        assert p.stat().st_mtime_ns == t, p
    sessions = sorted((trained.paths.run_dir / "eolien_pcfno3d").glob("*/eol1_*"))
    resume = json.loads((sessions[-1] / "eol1.json").read_text(encoding="utf-8"))
    assert resume["critere"] == "EOL1" and set(resume["par_capteurs"]) == {"1", "0.5"}
    assert resume["verdict_sur_capteurs"] == 1.0
    assert (sessions[-1] / "eol1_keep1.json").exists()


def test_roses_du_parc_sur_une_serie(wf_parc):
    from pcfno3d.cache import load_index
    from pcfno3d.windfarm import farm_windroses
    wf = wf_parc()
    stems = [str(s) for s in load_index(wf.cfg)["stem"].iloc[:4]]
    docs, ecartees = farm_windroses(wf, wf.farm, stems)
    assert set(docs) | set(ecartees) == {t.id for t in wf.farm.turbines}
    assert not set(docs) & set(ecartees)
    for doc in docs.values():
        _valide("windrose", doc)
        if doc["n_valid"]:
            assert sum(x["frequency"] for x in doc["sectors"]) == pytest.approx(1.0, abs=0.01)
