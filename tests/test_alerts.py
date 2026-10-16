"""Lot M4 — alertes (pcfno3d.alerts, pcfno3d.an1, critere AN1).

Unitaires sur des tableaux construits a la main (id, schema, journal, persistance, entrees,
physique), integration sur le mini-dataset (moteur, simulation AN1, script GPU, API, MCP).

Les schemas du contrat ATOM rev. 2 sont recopies tels quels dans tests/data/atom_rev2/
depuis atom-dashboard, branche agent/rayen-audit-atom/contrat-v1 (commit b7b13b8) ; la
fixture d'alertes est dans fixtures/, hors du glob *.json des schemas (test_windfarm).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pcfno3d.alerts import (CHAMPS, KINDS, NIVEAUX, JournalAlertes, ModeleSites,
                            MoteurAlertes, SuiviDerive, agreger, ajuster_sites, alert_id,
                            alerte, alertes_contrat, alertes_couverture,
                            alertes_derive, alertes_disponibilite, alertes_entrees,
                            alertes_externes,
                            alertes_physique, charger_seuils, compter_ouvertes, derive_entrees,
                            drapeaux_derive, ecarts_capteurs, ecarts_spatiaux, groupes,
                            interp_colonne, iso_utc, mediane_pile, normaliser,
                            physique, regle_entrees, resume_stations, rho_instant,
                            stations_hors_plage, verifier_alerte, voisins)
from pcfno3d.config import AlertsCfg

REPO = Path(__file__).resolve().parents[1]
ATOM = REPO / "tests" / "data" / "atom_rev2"
CONTRAT = REPO / "tests" / "data" / "contract"
# cles du bloc `qualite` du rapport M3 (main) : le lot M4 n'y ajoute rien
CLES_QUALITE_M3 = {"instants", "stations_declarees", "stations_valides",
                   "interpolees_ou_signalees", "hors_plage", "densite", "anomalies",
                   "fraicheur", "couverture", "stations_a_masquer"}
CLES_PHYSICS = {"mass_residual_rms", "mass_residual_p95", "mass_residual_ok",
                "ground_condition_rms_ms", "rho_source"}


@pytest.fixture(scope="module")
def schema_alertes():
    """Validateur jsonschema du document /v1/alerts (refs vers common.schema.json)."""
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    docs = [json.loads((ATOM / f"{n}.schema.json").read_text(encoding="utf-8"))
            for n in ("alerts", "common")]
    reg = referencing.Registry().with_resources(
        [(d["$id"], referencing.Resource.from_contents(d)) for d in docs])
    return jsonschema.Draft202012Validator(docs[0], registry=reg)


def _valide(validator, doc: dict) -> None:
    erreurs = sorted(validator.iter_errors(doc), key=str)
    assert not erreurs, [e.message for e in erreurs[:5]]
    for a in doc["items"]:
        verifier_alerte(a)


def _publiee(a: dict, t: str = "2025-01-06T00:00:00Z") -> dict:
    return dict(a, raised_at=t)


# ------------------------------------------------------------------ briques
def test_id_sha8_de_kind_cible_et_cles_exactes(schema_alertes):
    a = alerte("capteur_derive", "c340_200", "important", "t", "c", "r",
               station_id="c340_200", location=(17000, 10000), value=1.8512, threshold=1,
               unit="m/s")
    assert a["id"] == hashlib.sha256(b"capteur_derive|c340_200").hexdigest()[:8]
    assert a["id"] == alert_id("capteur_derive", "c340_200")        # stable
    assert alert_id("station_muette", "c340_200") != a["id"]
    assert tuple(a) == CHAMPS and a["status"] == "ouverte" and a["raised_at"] is None
    with pytest.raises(ValueError, match="raised_at"):
        verifier_alerte(a)                            # pose par le journal, pas avant
    _valide(schema_alertes, {"items": [_publiee(a)]})
    # la fixture du contrat passe le meme controle (memes regles que le schema)
    fixture = json.loads((ATOM / "fixtures" / "capteur_en_derive_alerts.json")
                         .read_text(encoding="utf-8"))
    _valide(schema_alertes, fixture)
    with pytest.raises(ValueError):
        alerte("inconnu", "x", "important", "t", "c", "r")
    with pytest.raises(ValueError):
        alerte("hors_plage", "x", "grave", "t", "c", "r")
    assert set(KINDS) == set(schema_alertes.schema["properties"]["items"]["items"]
                             ["properties"]["kind"]["enum"])
    assert iso_utc("2025-01-06 03:00:00") == "2025-01-06T03:00:00Z"
    assert iso_utc("2025-01-06T04:00:00+01:00") == "2025-01-06T03:00:00Z"


def test_alertes_contrat_depuis_le_rapport_m3(tmp_path, schema_alertes):
    import pandas as pd

    from pcfno3d.contract import Grille, valider_fichier

    grille = Grille.depuis_meta(CONTRAT)
    r = valider_fichier(CONTRAT / "panne.csv", grille)
    al = alertes_contrat(r, grille.nx_s, grille.dx_s, grille.dy_s)
    muettes = [a for a in al if a["kind"] == "station_muette"]
    ids = r["qualite"]["fraicheur"]["stations_muettes"]["ids"]
    assert len(muettes) == len(ids) == 6
    # point_id 2 -> ix 1, iy 0 ; point_id 17 -> ix 4, iy 2 (treillis de 6 x 4 a 50 m)
    assert {a["target"] for a in muettes} >= {"c1_0", "c4_2"}
    a2 = next(a for a in muettes if a["target"] == "c1_0")
    assert a2["station_id"] == "c1_0" and a2["location"] == {"x_m": 50.0, "y_m": 0.0}
    assert "figee" in a2["cause"].lower()                          # point 2 est fige
    ins = [a for a in al if a["kind"] == "stations_insuffisantes"]
    assert len(ins) == 1 and ins[0]["level"] == "important" and ins[0]["target"] == "domaine"
    assert ins[0]["value"] == pytest.approx(r["qualite"]["stations_valides"]["part_pire_instant"])
    # hors plage : une temperature a 80 degC sur le point 5
    df = pd.read_csv(CONTRAT / "conforme.csv")
    df.loc[df["point_id"] == 5, "temperature_c"] = 80.0
    f = tmp_path / "hors_plage.csv"
    f.write_bytes(df.to_csv(index=False).encode("utf-8"))
    r2 = valider_fichier(f, grille)
    assert set(r2["qualite"]) == CLES_QUALITE_M3            # rapport M3 inchange
    ids = stations_hors_plage(f, grille.n_stations, grille.node_id_start,
                              r2["seuils"]["tol_densite_rel"])
    assert ids == [5]
    assert stations_hors_plage(CONTRAT / "conforme.csv", grille.n_stations,
                               grille.node_id_start, 0.02) == []
    hp = [a for a in alertes_contrat(r2, grille.nx_s, grille.dx_s, grille.dy_s,
                                     hors_plage=ids) if a["kind"] == "hors_plage"]
    assert [a["target"] for a in hp] == ["c4_0"]
    _valide(schema_alertes, {"items": [_publiee(a) for a in al + hp]})
    assert alertes_contrat({"qualite": None}, 6, 50.0, 50.0) == []


def test_groupes_en_treillis_et_interpolation_colonne(geom):
    g = groupes(geom.ny, geom.nx, 3)
    assert set(np.unique(g)) == set(range(9))
    j, i = np.nonzero(g == 4)
    assert (np.diff(np.unique(i)) == 3).all() and (np.diff(np.unique(j)) == 3).all()
    # champ lineaire en z : l'interpolation le retrouve entre deux niveaux valides
    zl = np.asarray(geom.z_levels, float)
    f = np.broadcast_to(zl[:, None, None], geom.shape).astype(np.float64)
    f = np.where(geom.valid, f, np.nan)
    kf = np.argmax(geom.valid, axis=0)
    z = zl[kf] + 7.0
    out = interp_colonne(f, z, geom)
    ok = geom.valid.any(axis=0) & (z <= zl[-1])
    np.testing.assert_allclose(out[ok], z[ok])
    # sous le premier noeud valide : la valeur de ce noeud, jamais NaN
    bas = interp_colonne(f, zl[kf] - 50.0, geom)
    np.testing.assert_allclose(bas[ok], zl[kf][ok])


def test_normalisation_robuste_et_persistance():
    rng = np.random.default_rng(0)
    d = rng.normal(0.0, 0.3, (2, 10, 20)) + np.array([1.0, -2.0])[:, None, None]
    d[:, 5, 5] += np.array([3.0, 0.0])                      # un capteur a +3 m/s
    d[:, 0, 0] = np.nan                                     # non evalue
    ec = normaliser(d, plancher=0.05)
    assert ec.centre == pytest.approx((1.0, -2.0), abs=0.1)   # derive commune retiree
    assert ec.echelle == pytest.approx(0.3, rel=0.25)
    assert np.isnan(ec.z[0, 0]) and ec.z[5, 5] > 8
    assert normaliser(np.zeros((2, 3, 3)), 0.05).echelle == 0.05      # plancher
    s = SuiviDerive(fenetre=3)
    for k in range(4):
        s.ajouter(f"2025-01-06T0{k}:00:00Z", ec)
    flag, st = s.drapeaux(kappa=6.0, min_m_s=0.5, min_instants=3)
    assert len(s.z) == 3 and st["n"][5, 5] == 3 and st["n"][0, 0] == 0
    assert flag[5, 5] and flag.sum() == 1
    # plancher en m/s : ~3 m/s d'ecart ne declenche pas sous un plancher de 5 m/s
    assert not s.drapeaux(kappa=6.0, min_m_s=5.0, min_instants=3)[0].any()


def test_derive_entrees_unites_fausses():
    mu = [-5.0, -5.0, 0.0, 15.0, 1.2, 60.0, 100000.0]
    sd = [1.0, 2.0, 0.02, 5.0, 0.03, 10.0, 150.0]
    rng = np.random.default_rng(1)
    S = np.stack([rng.normal(m, s, (10, 20)) for m, s in zip(mu, sd)]
                 + [np.ones((10, 20))]).astype(np.float32)
    sensor = S[7] > 0.5
    assert not any(x["derive"] for x in derive_entrees(S, sensor, mu, sd))
    K = S.copy()
    K[3] += 273.15                                          # kelvins au lieu de degC
    st = derive_entrees(K, sensor, mu, sd)
    assert [x["variable"] for x in st if x["derive"]] == ["temperature_c"]
    al = alertes_entrees(st)
    assert len(al) == 1 and al[0]["kind"] == "derive_entrees"
    assert al[0]["target"] == "domaine:temperature_c" and al[0]["level"] == "important"
    # v2 : un jour de vent fort (u a +4 sd, w a queue lourde) n'est plus une derive des
    # entrees (source des fausses alertes de la v1) ; la v1 l'aurait signale
    W = S.copy()
    W[0] += 4.0 * sd[0]
    W[2, :2] += 6.0 * sd[2]                                 # 10 % de w extremes
    stw = derive_entrees(W, sensor, mu, sd)
    assert not any(x["derive"] for x in stw)
    assert {x["variable"] for x in stw if not x["evaluee"]} == {"wind_u_m_s", "wind_v_m_s",
                                                                 "wind_w_m_s"}
    tous = ["wind_u_m_s", "wind_v_m_s", "wind_w_m_s", "temperature_c", "air_density_kg_m3",
            "relative_humidity_pct", "pressure_pa"]
    assert set(regle_entrees(stw, 3.0, 0.02, tous)) == {"wind_u_m_s", "wind_w_m_s"}   # v1
    # humidite en fraction : toutes les valeurs a plus de 5 sd -> detectee par la part
    H = S.copy()
    H[5] *= 0.01
    sth = derive_entrees(H, sensor, mu, sd, z_max=8.0, part_extreme=0.5)
    assert [x["variable"] for x in sth if x["derive"]] == ["relative_humidity_pct"]


def test_physique_d11_et_alerte_de_masse(geom):
    z = np.zeros(geom.shape, np.float32)
    rho, src = rho_instant(geom, "atmosphere_standard")
    assert src == "atmosphere_standard" and rho.shape == geom.shape
    p = physique(geom, z, z, z, rho, src, 1e-4)
    assert set(p) == CLES_PHYSICS
    assert p["mass_residual_rms"] == 0.0 and p["mass_residual_ok"] is True
    assert p["ground_condition_rms_ms"] == 0.0
    assert alertes_physique(p, 1e-4) == []
    # w = 1 m/s partout sauf au sommet : divergence verticale -> residu enorme
    w = np.ones(geom.shape, np.float32)
    w[-1] = 0.0
    p2 = physique(geom, z, z, w, rho, src, 1e-4)
    assert p2["mass_residual_ok"] is False
    al = alertes_physique(p2, 1e-4, 10.0, seuil_sol=0.1)
    assert {a["target"] for a in al} == {"domaine:masse", "domaine:sol"}
    assert next(a for a in al if a["target"] == "domaine:masse")["level"] == "critique"
    # extrapolation de surface : un facteur unique, sinon repli sur l'ISA
    S = np.zeros((8, geom.ny, geom.nx), np.float32)
    S[7] = 1.0
    from pcfno3d.realdata import isa
    S[4] = 1.02 * isa(geom.z_sensor)[2]
    r2, s2 = rho_instant(geom, "extrapolation_surface", S, S[7] > 0.5)
    assert s2 == "extrapolation_surface"
    np.testing.assert_allclose(r2, rho * 1.02, rtol=1e-5)
    assert rho_instant(geom, "extrapolation_surface", S, np.zeros_like(S[7], bool))[1] \
        == "atmosphere_standard"
    with pytest.raises(ValueError):
        rho_instant(geom, "verite_synthetique")


def test_couverture_par_maille_de_2_km(geom):
    cov = np.where(geom.valid, 0.1, np.nan).astype(np.float32)
    al = alertes_couverture(geom, cov, maille_m=1000.0, seuil=0.2)
    # mini-domaine 2 km x 1 km : deux mailles de 1 km, toutes deux mal couvertes
    assert {a["target"] for a in al} == {"m0_0", "m1_0"}
    assert all(a["level"] == "information" and a["value"] == pytest.approx(0.1) for a in al)
    assert alertes_couverture(geom, np.where(geom.valid, 0.9, np.nan), 1000.0, 0.2) == []


def test_alertes_externes_m5_et_agregation():
    ext = alertes_externes([{"kind": "cisaillement", "target": "T-03", "level": "information",
                             "title": "Cisaillement", "cause": "alpha > 0,3",
                             "recommendation": "Aucune action", "turbine_id": "T-03",
                             "location": {"x_m": 30000, "y_m": 15000}, "value": 0.34,
                             "threshold": 0.3}])
    assert ext[0]["id"] == alert_id("cisaillement", "T-03") and ext[0]["turbine_id"] == "T-03"
    with pytest.raises(ValueError, match="non accepte"):
        alertes_externes([{"kind": "capteur_derive", "target": "c1_1", "level": "important",
                           "title": "t", "cause": "c", "recommendation": "r"}])
    with pytest.raises(ValueError, match="id"):
        alertes_externes([{"id": "00000000", "kind": "vent_fort_moyeu", "target": "T-01",
                           "level": "important", "title": "t", "cause": "c",
                           "recommendation": "r"}])
    a = alerte("hors_plage", "c1_1", "avertissement", "t", "c", "r", station_id="c1_1")
    b = dict(a, level="critique")
    assert agreger([a], [b]) == [b]                       # meme id : le plus grave
    items = [_publiee(x) for x in agreger([a], ext)]
    assert compter_ouvertes(items) == {"information": 1, "avertissement": 1,
                                       "important": 0, "critique": 0}
    assert resume_stations(items, 10) == {"active": 9, "suspecte": 0, "muette": 0,
                                          "derive": 0, "exclue": 1}


def test_journal_cycle_de_vie_et_relecture(tmp_path, schema_alertes):
    path = tmp_path / "logs" / "alertes.jsonl"
    A = alerte("capteur_derive", "c1_1", "important", "t", "c", "r", station_id="c1_1")
    B = alerte("station_muette", "c2_2", "avertissement", "t", "c", "r", station_id="c2_2")
    j = JournalAlertes(path, retention_h=6.0)
    t1, t2, t3, t4 = (f"2025-01-06T{h:02d}:00:00Z" for h in (0, 3, 6, 12))
    i1 = j.mettre_a_jour([A, B], t1)
    assert {a["raised_at"] for a in i1} == {t1}
    i2 = j.mettre_a_jour([dict(A, level="critique")], t2)
    a2 = next(a for a in i2 if a["id"] == A["id"])
    assert a2["raised_at"] == t1 and a2["status"] == "ouverte" and a2["level"] == "critique"
    b2 = next(a for a in i2 if a["id"] == B["id"])
    assert b2["status"] == "close" and b2["raised_at"] == t1      # close, encore publiee
    _valide(schema_alertes, {"items": i2})
    lignes = [json.loads(x) for x in path.read_bytes().decode("utf-8").splitlines()]
    assert [e["evenement"] for e in lignes] == ["levee", "levee", "niveau", "close"]
    assert b"\r\n" not in path.read_bytes()
    # redemarrage : l'episode ouvert garde sa premiere levee
    j2 = JournalAlertes(path, retention_h=6.0)
    assert j2.dernier == t2 and set(j2.ouvertes) == {A["id"]}
    i3 = j2.mettre_a_jour([A], t3)
    assert next(a for a in i3 if a["id"] == A["id"])["raised_at"] == t1
    assert any(a["id"] == B["id"] and a["status"] == "close" for a in i3)   # 3 h < 6 h
    # instant anterieur : consultation, rien n'est ecrit
    taille = path.stat().st_size
    j2.mettre_a_jour([B], t1)
    assert path.stat().st_size == taille and j2.dernier == t3
    # au-dela de la retention, la close disparait ; une nouvelle levee repart de zero
    i4 = j2.mettre_a_jour([A, B], t4)
    assert next(a for a in i4 if a["id"] == B["id"])["raised_at"] == t4
    assert all(a["status"] == "ouverte" for a in i4)
    # journal en memoire seulement
    assert JournalAlertes(None).mettre_a_jour([A], t1)[0]["raised_at"] == t1


def test_seuils_regles_sur_la_validation_seulement(tmp_path):
    from pcfno3d.alerts import METHOD
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"jeu": "test", "methode": METHOD}), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        charger_seuils(p)
    # un JSON v1 (session du 16/10/2026) reste relisible ; une methode inconnue non
    p.write_text(json.dumps({"jeu": "validation", "methode": "alertes-m4-v1",
                             "derive_statistique": "innovation", "derive_kappa": 12.75}),
                 encoding="utf-8")
    assert charger_seuils(p)["derive_kappa"] == 12.75
    p.write_text(json.dumps({"jeu": "validation", "methode": "autre"}), encoding="utf-8")
    with pytest.raises(ValueError, match="methode"):
        charger_seuils(p)
    p.write_text(json.dumps({"jeu": "validation", "methode": METHOD,
                             "derive_statistique": "influence", "derive_kappa": 9.5}),
                 encoding="utf-8")
    assert charger_seuils(p)["derive_kappa"] == 9.5
    with pytest.raises(ValueError):
        AlertsCfg(derive_statistique="autre")
    with pytest.raises(ValueError):
        AlertsCfg(derive_min_instants=5, derive_fenetre=4)
    assert AlertsCfg().enabled is False                      # strictement opt-in
    assert AlertsCfg().derive_statistique == "spatiale"      # v2 : aucune passe du modele
    with pytest.raises(ValueError):
        AlertsCfg(derive_rayon=0)


# ------------------------------------------------------------------ detecteur spatial (v2)
def _champ_lisse(ny=30, nx=40, t=0.0, bruit=0.05, graine=0):
    """S (8, ny, nx) : vent lisse qui tourne avec t, petit bruit, tous capteurs presents."""
    rng = np.random.default_rng(graine)
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    S = np.zeros((8, ny, nx))
    S[0] = 5.0 + 0.05 * i + np.cos(t) + rng.normal(0, bruit, (ny, nx))
    S[1] = -2.0 + 0.03 * j + np.sin(t) + rng.normal(0, bruit, (ny, nx))
    S[7] = 1.0
    return S


def test_voisins_et_mediane_robuste():
    rng = np.random.default_rng(3)
    f = rng.normal(size=(7, 9))
    f[2, 3] = np.nan
    pile = voisins(f, 2)
    assert pile.shape == (24, 7, 9)
    assert np.isnan(pile[:, 0, 0]).sum() == 24 - 8           # coin : 8 voisins existent
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = np.nanmedian(pile, axis=0)
    np.testing.assert_allclose(mediane_pile(pile), ref)
    assert np.isnan(mediane_pile(np.full((4, 2, 2), np.nan))).all()


def test_detecteur_spatial_injection_et_capteurs_sains():
    """Derives connues (biais, gain, girouette) sur un champ lisse : signalees ; capteurs
    sains : rien ; un voisin en derive ne fausse pas la mediane ; deterministe."""
    cibles = {(10, 10): "biais", (10, 20): "gain", (20, 30): "girouette"}
    suivi = SuiviDerive(4, "vecteur")
    for k in range(4):
        S = _champ_lisse(t=0.3 * k, graine=k)
        for (j, i), typ in cibles.items():
            u, v = S[0, j, i], S[1, j, i]
            if typ == "biais":
                S[0, j, i] += 1.0
            elif typ == "gain":
                S[0, j, i], S[1, j, i] = 1.3 * u, 1.3 * v
            else:
                a = np.radians(30.0)
                S[0, j, i], S[1, j, i] = (u * np.cos(a) - v * np.sin(a),
                                          u * np.sin(a) + v * np.cos(a))
        ec = ecarts_spatiaux(S, S[7] > 0.5, rayon=2, rayon_echelle=2)
        ec2 = ecarts_spatiaux(S, S[7] > 0.5, rayon=2, rayon_echelle=2)
        np.testing.assert_array_equal(ec.z, ec2.z)            # deterministe
        assert ec.echelle_locale is not None and ec.echelle == pytest.approx(0.05, rel=0.5)
        suivi.ajouter(f"2025-01-06T{3 * k:02d}:00:00Z", ec)
    flag, st = suivi.drapeaux(kappa=6.0, min_m_s=0.5, min_instants=4)
    assert {tuple(x) for x in np.argwhere(flag)} == set(cibles)
    assert st["a"][10, 10] == pytest.approx(1.0, abs=0.15)    # le biais, en m/s
    # sans derive : aucun capteur signale
    s0 = SuiviDerive(4, "vecteur")
    for k in range(4):
        S = _champ_lisse(t=0.3 * k, graine=10 + k)
        s0.ajouter(f"2025-01-06T{3 * k:02d}:00:00Z", ecarts_spatiaux(S, S[7] > 0.5))
    assert not s0.drapeaux(6.0, 0.5, 4)[0].any()
    # capteurs absents : non evalues (NaN), jamais inventes
    S = _champ_lisse()
    S[7, :, :5] = 0.0
    ec = ecarts_spatiaux(S, S[7] > 0.5, min_voisins=3)
    assert np.isnan(ec.z[:, :5]).all() and np.isfinite(ec.z[:, 7:]).all()
    # alerte au format du contrat : seuil = kappa x echelle locale, valeur >= seuil
    from types import SimpleNamespace
    g = SimpleNamespace(stride=2, dx=100.0, dy=100.0)
    al = alertes_derive(g, flag, st, 6.0, 0.5, 2.0, "spatiale", rayon=2)
    assert len(al) == 3 and all(a["value"] >= a["threshold"] for a in al)
    assert "mediane" in al[0]["cause"]
    for a in al:
        verifier_alerte(dict(a, raised_at="2025-01-06T09:00:00Z"))


def test_modele_de_sites_appris_sur_l_entrainement(tmp_path):
    """Un ecart de site qui se repete (crete : +20 % du vent des voisins) est appris sur
    l'entrainement puis retire : la crete n'est plus suspecte, une derive l'est toujours."""
    def instant(k, crete=True, biais=0.0):
        S = _champ_lisse(t=0.4 * k, graine=100 + k)
        if crete:
            S[0, 15, 15] *= 1.2
            S[1, 15, 15] *= 1.2
        S[0, 5, 30] += biais
        return S, 3.0 * k
    m = ajuster_sites((instant(k) for k in range(12)), rayon=2, info={"jours": ["001"]})
    path = m.sauver(tmp_path / "sites_R2.npz")
    m2 = ModeleSites.charger(path, forme=(30, 40))
    np.testing.assert_array_equal(m2.coef, m.coef)
    assert m2.rayon == 2 and m2.info["jeu"] == "entrainement"
    with pytest.raises(ValueError, match="grille"):
        ModeleSites.charger(path, forme=(10, 10))
    with pytest.raises(ValueError, match="rayon"):
        ecarts_spatiaux(*instant(0)[:1], np.ones((30, 40), bool), rayon=1, sites=m2,
                        heure=0.0)
    # un modele marque autrement qu'« entrainement » est refuse
    meta = json.dumps({"jeu": "validation", "methode": "sites-lineaires-v1", "rayon": 2})
    with open(tmp_path / "faux.npz", "wb") as f:
        np.savez(f, coef=m.coef, meta=np.array(meta))
    with pytest.raises(ValueError, match="entrainement"):
        ModeleSites.charger(tmp_path / "faux.npz")
    sans, avec = SuiviDerive(4, "vecteur"), SuiviDerive(4, "vecteur")
    for k in range(20, 24):
        S, h = instant(k, biais=1.0)
        sans.ajouter(f"2025-01-07T{k - 20:02d}:00:00Z", ecarts_spatiaux(S, S[7] > 0.5))
        avec.ajouter(f"2025-01-07T{k - 20:02d}:00:00Z",
                     ecarts_spatiaux(S, S[7] > 0.5, sites=m2, heure=h))
    f_sans = sans.drapeaux(6.0, 0.5, 4)[0]
    f_avec = avec.drapeaux(6.0, 0.5, 4)[0]
    assert f_sans[15, 15] and f_sans[5, 30]                   # crete suspecte sans site
    assert not f_avec[15, 15] and f_avec[5, 30]               # crete expliquee, derive vue
    assert f_avec.sum() == 1


def test_persistance_vectorielle_et_combinee():
    from pcfno3d.alerts import Ecarts, combiner, composantes
    d = np.zeros((2, 3, 3))
    d[0, 1, 1] = 1.0
    s = np.full((3, 3), 0.1)
    v = SuiviDerive(2, "vecteur")
    v.ajouter("a", Ecarts(d, np.abs(d[0]), np.abs(d[0]) / 0.1, 0.1, (0.0, 0.0), s))
    v.ajouter("b", Ecarts(-d, np.abs(d[0]), np.abs(d[0]) / 0.1, 0.1, (0.0, 0.0), s))
    st = v.statistiques()                         # +1 puis -1 : le cumul s'annule
    assert st["a"][1, 1] == pytest.approx(0.0) and st["n"][1, 1] == 2
    with pytest.raises(ValueError):
        v.ajouter("c", Ecarts(d, d[0], d[0], 0.1, (0.0, 0.0)))      # sans echelle locale
    with pytest.raises(ValueError):
        SuiviDerive(2, "autre")
    c = combiner({"z": np.array([8.0]), "a": np.array([1.0]), "n": np.array([4]),
                  "echelle": 0.1}, {"z": np.array([2.0]), "n": np.array([3])})
    assert c["z"][0] == 5.0 and c["n"][0] == 3 and c["a"][0] == 1.0
    assert drapeaux_derive(c, 5.0, 0.5, 3)[0]
    assert composantes("combinee") == ("spatiale", "innovation")
    with pytest.raises(ValueError):
        composantes("autre")


def test_stations_insuffisantes_selon_le_reseau():
    """Treillis : part des mailles ; stations reelles : part des stations ATTENDUES, et
    pas d'alerte si ce nombre est inconnu (jamais critique en permanence)."""
    sensor = np.zeros((100, 100), bool)
    sensor[:2, :20] = True                           # 40 mailles sur 10 000
    assert alertes_disponibilite(sensor)[0]["level"] == "critique"          # treillis
    assert alertes_disponibilite(sensor, reseau="stations_aneo") == []      # inconnu
    assert alertes_disponibilite(sensor, reseau="stations_aneo", attendues=45) == []
    a = alertes_disponibilite(sensor, reseau="stations_aneo", attendues=100)
    assert a[0]["level"] == "important" and a[0]["value"] == pytest.approx(0.4)
    a = alertes_disponibilite(sensor, reseau="stations_aneo", disponibles=10, attendues=50)
    assert a[0]["level"] == "critique" and a[0]["value"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        AlertsCfg(reseau="autre")


def test_journal_ligne_tronquee_ignoree(tmp_path):
    """Arret brutal ou synchro Drive : une ligne tronquee n'empeche pas le demarrage."""
    path = tmp_path / "alertes.jsonl"
    A = alerte("capteur_derive", "c1_1", "important", "t", "c", "r", station_id="c1_1")
    JournalAlertes(path).mettre_a_jour([A], "2025-01-06T00:00:00Z")
    with open(path, "ab") as f:
        f.write(b'{"evenement": "levee", "instant": "2025-01-06T03:00')    # tronquee
    with pytest.warns(RuntimeWarning, match="illisible"):
        j = JournalAlertes(path)
    assert j.lignes_illisibles == 1 and set(j.ouvertes) == {A["id"]}
    assert j.dernier == "2025-01-06T00:00:00Z"
    items = j.mettre_a_jour([A], "2025-01-06T06:00:00Z")
    assert items[0]["raised_at"] == "2025-01-06T00:00:00Z"


# ------------------------------------------------------------------ integration
@pytest.fixture(scope="module")
def trained(cfg, index, norm):
    from pcfno3d.train import Trainer
    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg


@pytest.fixture(scope="module")
def wf(trained):
    from pcfno3d.inference import WindFieldModel
    return WindFieldModel(trained, device="cpu", verbose=False)


def _jour(cfg, split: str) -> list[str]:
    from pcfno3d.cache import load_index
    idx = load_index(cfg)
    return idx[idx["split"] == split].sort_values("timestamp")["stem"].tolist()


def test_ecarts_detectent_un_capteur_fausse(wf):
    """Un capteur a +8 m/s ressort, en innovation, loin devant les autres."""
    stem = _jour(wf.cfg, "test")[0]
    S, hour, doy, stamp = wf.surface_from_cache(stem)
    sensor = S[7] > 0.5
    j, i = (int(x[0]) for x in np.nonzero(sensor[2:-2, 2:-2]))
    j, i = j + 2, i + 2
    S2 = S.copy()
    S2[0, j, i] += 8.0
    vol = wf.predict(S2, hour, doy, timestamp=stamp)
    appels = []

    def membre(kept):
        appels.append(kept)
        return wf._member(S2, kept, hour, doy)

    ec = ecarts_capteurs(S2, sensor, wf.geom, (vol.u, vol.v), membre, 3,
                         ("innovation", "influence"))
    assert len(appels) == 9                                    # cote^2 passes, pas une par capteur
    assert all(not (k & ~sensor).any() for k in appels)        # jamais un capteur invente
    z = ec["innovation"].z
    assert np.isfinite(z[sensor]).all() and np.isnan(z[~sensor]).all()
    assert z[j, i] == np.nanmax(z) and z[j, i] > 10
    assert np.isfinite(ec["influence"].z[sensor]).all()


def test_moteur_document_conforme_et_valeur_servie_inchangee(wf, tmp_path, schema_alertes):
    """Chaine complete sur le jour de test : document conforme au schema rev. 2, journal
    ecrit, et la valeur servie n'est jamais modifiee (bit a bit)."""
    cfg = wf.cfg
    moteur = MoteurAlertes(cfg, journal=tmp_path / "alertes.jsonl")
    stems = _jour(cfg, "test")
    present = np.logical_and.reduce([wf.surface_from_cache(s)[0][7] > 0.5 for s in stems])
    jj, ii = np.nonzero(present[2:-2, 2:-2])
    jc, ic = int(jj[len(jj) // 2]) + 2, int(ii[len(ii) // 2]) + 2
    res = None
    for k, stem in enumerate(stems):
        S, hour, doy, stamp = wf.surface_from_cache(stem)
        S0 = S.copy()
        if k:                                       # un capteur fausse a partir de t1
            S[0, jc, ic] += 8.0
        vol = wf.predict(S, hour, doy, timestamp=stamp)
        u0, v0, w0 = vol.u.copy(), vol.v.copy(), vol.w.copy()
        res = moteur.evaluer_modele(wf, S, hour, doy, vol, cle=f"stem:{stem}")
        for a, b in ((vol.u, u0), (vol.v, v0), (vol.w, w0)):
            np.testing.assert_array_equal(a, b)
        if not k:
            np.testing.assert_array_equal(S, S0)
        _valide(schema_alertes, res.document())
        assert set(res.physics) == CLES_PHYSICS and res.derive["evaluee"] is True
        assert res.derive["passes"] == 0 and res.derive["statistique"] == "spatiale"
    assert set(res.document()) == {"items"}
    kinds = {a["kind"] for a in res.items}
    assert "capteur_derive" in kinds
    d = next(a for a in res.items if a["kind"] == "capteur_derive")
    assert d["target"] == f"c{ic * wf.geom.stride}_{jc * wf.geom.stride}"
    assert d["raised_at"] > iso_utc(wf.surface_from_cache(stems[0])[3])
    assert d["unit"] == "m/s" and d["value"] >= d["threshold"]
    assert res.ouvertes() == compter_ouvertes(res.items)
    assert (tmp_path / "alertes.jsonl").exists()
    # meme instant redemande : resultat memorise, rien de recalcule
    S, hour, doy, stamp = wf.surface_from_cache(stems[-1])
    assert moteur.evaluer_modele(wf, S, hour, doy, wf.predict(S, hour, doy, timestamp=stamp),
                                 cle=f"stem:{stems[-1]}") is res
    # instant anterieur : evalue sans derive (persistance chronologique)
    S, hour, doy, stamp = wf.surface_from_cache(stems[0])
    r0 = moteur.evaluer_modele(wf, S, hour, doy, wf.predict(S, hour, doy, timestamp=stamp))
    assert r0.derive["evaluee"] is False


def test_moteur_innovation_v1_toujours_disponible(wf, tmp_path):
    """La statistique v1 (innovation, cote^2 passes) reste selectionnable ; sans membre,
    elle n'est pas evaluee, alors que le detecteur spatial n'en a pas besoin."""
    import copy
    cfg = copy.copy(wf.cfg)
    cfg.alerts = AlertsCfg(derive_statistique="innovation")
    moteur = MoteurAlertes(cfg, journal=False)
    stem = _jour(cfg, "test")[0]
    S, hour, doy, stamp = wf.surface_from_cache(stem)
    vol = wf.predict(S, hour, doy, timestamp=stamp)
    res = moteur.evaluer_modele(wf, S, hour, doy, vol)
    assert res.derive["evaluee"] and res.derive["passes"] == 9
    st = wf.norm.stats
    sans = MoteurAlertes(cfg, journal=False).evaluer(
        wf.geom, S, (vol.u, vol.v, vol.w), stamp, None, st["surf_mu"], st["surf_sd"])
    assert sans.derive["evaluee"] is False and "membre" in sans.derive["raison"]
    cfg.alerts = AlertsCfg()                               # spatiale : pas de membre requis
    sp = MoteurAlertes(cfg, journal=False).evaluer(
        wf.geom, S, (vol.u, vol.v, vol.w), stamp, None, st["surf_mu"], st["surf_sd"])
    assert sp.derive["evaluee"] is True and sp.derive["passes"] == 0


def _sites_mini(wf):
    from pcfno3d import an1
    return an1.ajuster_sites_entrainement(wf, _jour(wf.cfg, "train"), (1, 2))


def test_simulation_an1_sur_le_mini_jeu(wf):
    """Mecanisme AN1 sur CPU : injection, detecteurs (spatial avec / sans site, innovation),
    reglage sur la validation, verdict sur le test. Le modele du mini-jeu est minuscule :
    on verifie le mecanisme (derives fortes et pannes detectees), pas le chiffre de v7, qui
    vient de scripts/m4_an1_colab.py sur GPU."""
    from pcfno3d import an1
    acfg = AlertsCfg()
    plan = (("biais", 4.0), ("biais", 8.0), ("gain", 1.0))
    sites = _sites_mini(wf)
    with pytest.raises(ValueError, match="hors entrainement"):
        an1.ajuster_sites_entrainement(wf, _jour(wf.cfg, "val"), (2,))
    dets = [an1.Detecteur("spatiale", 3, 2, 2, False), an1.Detecteur("spatiale", 3, 2, 2, True),
            an1.Detecteur("spatiale", 4, 1, 2, True), an1.Detecteur("innovation", 3),
            an1.Detecteur("combinee", 3, 2, 2, True)]
    assert len({d.cle for d in dets}) == len(dets)
    jours = [an1.simuler_jour(wf, _jour(wf.cfg, sp), acfg, np.random.default_rng(k), 6, 2,
                              dets, plan=plan, sites=sites, par_instant=(sp == "test"))
             for k, sp in enumerate(("val", "test"))]
    for j in jours:
        assert len(j["derives"]) == 6 and len(j["figees"]) == 2
        cells = [d["cellule"] for d in j["derives"]] + [f["cellule"] for f in j["figees"]]
        assert len(set(cells)) == 8
        assert all(j["figee_signalee"][6:])                            # regle M3
        assert all(d["ecart_moyen"] >= 4.0 - 1e-6 for d in j["derives"]
                   if d["type"] == "biais")
        assert len(j["physique"]) == 4 and set(j["physique"][0]) == CLES_PHYSICS
        assert set(j["detecteurs"]) == {d.cle for d in dets}
    # determinisme : meme graine, memes defauts, memes statistiques
    bis = an1.simuler_jour(wf, _jour(wf.cfg, "val"), acfg, np.random.default_rng(0), 6, 2,
                           dets[:2], plan=plan, sites=sites)
    for d in dets[:2]:
        np.testing.assert_array_equal(bis["detecteurs"][d.cle]["defauts"],
                                      jours[0]["detecteurs"][d.cle]["defauts"])
        np.testing.assert_array_equal(bis["detecteurs"][d.cle]["sains_hauts"],
                                      jours[0]["detecteurs"][d.cle]["sains_hauts"])
    choix = an1.choisir_detecteur(jours[:1], fdr_cible=0.025)
    assert choix["jeu"] == "validation" and choix["detecteur"] in {d.cle for d in dets}
    assert set(choix["alternatives"]) == {d.cle for d in dets}
    assert choix["derive_statistique"] in ("spatiale", "innovation", "combinee")
    for cle in (d.cle for d in dets if d.spatial):
        r = an1.evaluer_an1(jours[1:], cle, choix["alternatives"][cle]["kappa"], e_ref=1.0)
        assert r["derives_ref"] == 6 and r["detection"] >= 5 / 6, (cle, r)
    res = an1.evaluer_an1(jours[1:], choix["detecteur"], choix["derive_kappa"], e_ref=1.0)
    assert res["detection"] >= 5 / 6
    pan = res["pannes_hors_verdict"]                 # hors verdict (egalite exacte)
    assert pan["pannes"] == 2 and pan["detection_regle_m3"] == 1.0
    assert 0.0 <= res["fausses_alertes"] <= 1.0 and res["capteurs_sains"] > 100
    assert res["fausses_alertes_par_instant"] is not None
    assert res["taux_faux_positifs"] == res["fausses_alertes_n"] / res["capteurs_sains"]
    assert an1.PROTOCOLE == {"derives_par_jour": 40, "pannes_par_jour": 10,
                             "e_ref_m_s": 1.0}
    assert set(res["detection_par_type"]) == {"biais:4", "biais:8", "gain:1"}
    assert isinstance(res["an1_capteurs_conforme"], bool)
    # entrees : regle choisie sur la validation, appliquee au test
    reg = an1.choisir_regle_entrees(jours[:1])
    assert reg["jeu"] == "validation" and "wind_u_m_s" not in reg["entrees_variables"]
    assert reg["validation"]["detection"] == 1.0 and reg["validation"]["fausses_alertes"] == 0
    ent = an1.evaluer_entrees(jours[1:], {"z_max": reg["entrees_z_max"],
                                          "part_extreme": reg["entrees_part_extreme"],
                                          "variables": reg["entrees_variables"]})
    assert ent["detection"] == 1.0 and ent["instants"] == 4 and ent["fausses_alertes"] == 0
    assert an1.evaluer_entrees(jours)["instants"] == 8


def test_grille_des_detecteurs_fixee_d_avance():
    from pcfno3d import an1
    g = an1.detecteurs()
    assert g[0].statistique == "spatiale" and g[-1].statistique == "combinee"
    assert len(g) == len({d.cle for d in g}) == 2 * 36 + 2 * 3
    assert an1.Detecteur("spatiale", 4, 2, 2, True).seuils() == {
        "derive_statistique": "spatiale", "derive_fenetre": 4, "derive_min_instants": 4,
        "derive_rayon": 2, "derive_rayon_echelle": 2, "derive_sites": "sites_R2.npz"}
    with pytest.raises(ValueError):
        an1.Detecteur("spatiale", 4)                     # rayon requis


def test_script_an1_colab_dossier_neuf(trained, monkeypatch, tmp_path):
    """scripts/m4_an1_colab.py sur le mini-jeu : session neuve, rien d'ecrase, seuils de
    validation (et modele de sites) relisibles par le service ; hors protocole, AUCUN
    verdict officiel ; seconde lecture declaree."""
    import importlib.util
    for var, p in (("ANEO_DATA", trained.paths.data_dir), ("ANEO_CACHE", trained.paths.cache_dir),
                   ("ANEO_RUN", trained.paths.run_dir)):
        monkeypatch.setenv(var, str(p))
    spec = importlib.util.spec_from_file_location("m4_script",
                                                  REPO / "scripts" / "m4_an1_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    avant = {p: p.stat().st_mtime_ns for p in trained.paths.run_dir.rglob("*") if p.is_file()}
    assert mod.main(["-c", "test", "--device", "cpu", "--derives-par-jour", "6",
                     "--pannes-par-jour", "2", "--statistiques", "spatiale,innovation",
                     "--deuxieme-lecture"]) == 0
    for p, t in avant.items():
        assert p.stat().st_mtime_ns == t, p
    sessions = sorted((trained.paths.run_dir / "alertes_pcfno3d").glob("*/session_v2_*"))
    assert sessions
    s = sessions[-1]
    seuils = charger_seuils(s / "seuils_validation.json")
    assert seuils["jeu"] == "validation" and seuils["methode"] == "alertes-m4-v2"
    assert seuils["derive_statistique"] in ("spatiale", "innovation")
    assert {"entrees_z_max", "entrees_part_extreme", "entrees_variables",
            "alternatives"} <= set(seuils)
    assert len(list(s.glob("sites_R*.npz"))) == 3          # un modele par rayon de la grille
    t = json.loads((s / "an1_test.json").read_text(encoding="utf-8"))
    assert t["jeu"] == "test" and t["statut"] == "exploratoire"
    assert t["an1_conforme"] is None                 # hors protocole : pas de verdict
    assert t["lecture"] == "deuxieme" and "premiere lecture" in t["note"]
    assert t["premiere_lecture"]["an1_conforme"] is False
    assert t["reglages"] == {"derives_par_jour": 6, "pannes_par_jour": 2, "e_ref_m_s": 1.0}
    assert {"detection", "fausses_alertes", "taux_faux_positifs",
            "fausses_alertes_par_instant", "pannes_hors_verdict"} <= set(t["capteurs"])
    assert "fausses_par_variable" in t["entrees_regle_v1_diagnostic"]
    resume = json.loads((s / "resume.json").read_text(encoding="utf-8"))
    assert resume["physique_validation"]["mass_residual_rms"]["n"] > 0
    assert resume["statut"] == "exploratoire" and resume["an1_conforme"] is None
    # le service relit ces seuils (et le modele de sites retenu) sans rien recalculer
    import copy
    cfg = copy.copy(trained)
    cfg.alerts = AlertsCfg(enabled=True, seuils=s / "seuils_validation.json")
    moteur = MoteurAlertes(cfg, journal=False)
    assert moteur.acfg.derive_statistique == seuils["derive_statistique"]
    assert moteur.acfg.entrees_z_max == seuils["entrees_z_max"]
    assert (moteur.sites is not None) == bool(seuils.get("derive_sites"))
    # un seuil marque « test » est refuse par le service
    faux = tmp_path / "seuils_test.json"
    faux.write_text(json.dumps({**seuils, "jeu": "test"}), encoding="utf-8")
    cfg.alerts = AlertsCfg(enabled=True, seuils=faux)
    with pytest.raises(ValueError, match="validation"):
        MoteurAlertes(cfg, journal=False)


# ------------------------------------------------------------------ service
@pytest.fixture
def service_env(trained, monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp")

    def lancer(enabled: bool):
        conf = tmp_path / f"svc_{int(enabled)}.yaml"
        txt = (REPO / "configs" / "test.yaml").read_text(encoding="utf-8")
        if enabled:
            journal = (tmp_path / "alertes.jsonl").as_posix()
            txt += f"\nalerts:\n  enabled: true\n  journal: {journal}\n"
        conf.write_bytes(txt.encode("utf-8"))
        monkeypatch.setenv("PCFNO3D_CONFIG", str(conf))
        for var, p in (("PCFNO3D_DATA_DIR", trained.paths.data_dir),
                       ("PCFNO3D_CACHE_DIR", trained.paths.cache_dir),
                       ("PCFNO3D_RUN_DIR", trained.paths.run_dir)):
            monkeypatch.setenv(var, str(p))
        monkeypatch.setenv("PCFNO3D_DEVICE", "cpu")
        for var in ("PCFNO3D_API_KEYS", "PCFNO3D_INSECURE_OPEN", "PCFNO3D_QUOTA_MINUTE",
                    "PCFNO3D_QUOTA_JOUR", "PCFNO3D_UPSTREAM"):
            monkeypatch.delenv(var, raising=False)
        from service.app import create_app
        return create_app()

    from service.auth import ApiKeyStore, default_store_path
    path = default_store_path(trained.paths.run_dir)
    backup = path.read_bytes() if path.exists() else None
    path.unlink(missing_ok=True)
    cle, _ = ApiKeyStore(path).create("tests-alertes", scope="read")
    try:
        yield lancer, cle
    finally:
        path.unlink(missing_ok=True)
        if backup is not None:
            path.write_bytes(backup)
        from service import mcp_server
        mcp_server.set_alertes(None)


def _csv_instant(cfg, corrompre: bool = False) -> bytes:
    import pandas as pd
    meta = json.loads((cfg.paths.data_dir / "meta.json").read_text(encoding="utf-8"))
    day = sorted(cfg.paths.data_dir.glob("aneo_observations_day_*"))[-1]
    df = pd.read_csv(day)
    df = df[(df["timestamp_utc"] == df["timestamp_utc"].iloc[0])
            & (df["point_id"] < meta["noeuds_3d"]["point_id_debut"])]
    if corrompre:                                   # rejet : type illisible + doublon
        df = df.astype({"temperature_c": object})
        df.iloc[0, df.columns.get_loc("temperature_c")] = "abc"
        df = pd.concat([df, df.iloc[[5]]])
    return df.to_csv(index=False).encode("utf-8")


def _mcp(client, name, args=None):
    r = client.post("/mcp", headers={"Content-Type": "application/json",
                                     "Accept": "application/json, text/event-stream",
                                     "MCP-Protocol-Version": "2025-06-18"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": args or {}}})
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    assert res.get("isError") is not True, res
    return res.get("structuredContent") or json.loads(res["content"][0]["text"])


VOLATILS = {"inference_s", "controle_utc"}          # temps mesure, horloge murale


def _sans_volatils(x):
    if isinstance(x, dict):
        return {k: _sans_volatils(v) for k, v in x.items() if k not in VOLATILS}
    if isinstance(x, list):
        return [_sans_volatils(v) for v in x]
    return x


def _corps_main(wf, data: bytes, tmp_path) -> tuple[int, dict]:
    """Corps de l'upload tel que main le construit (avant le lot M4), recalcule a part."""
    from pcfno3d.contract import Grille, Seuils, valider_fichier
    f = tmp_path / "ref.csv"
    f.write_bytes(data)
    masque: dict = {}
    rapport = valider_fichier(f, Grille.depuis_meta(wf.geom.meta.raw), Seuils.upload(),
                              nom="i.csv", masque=masque)
    if rapport["verdict"] == "rejet":
        body = {"detail": f"CSV non conforme au contrat ANEO ({rapport['verdict']}) : "
                          + " ; ".join(rapport["motifs"]), "qualite": rapport}
        return 422, json.loads(json.dumps(body))
    vol = wf.predict_csv(f, keep=1.0, exclure=masque)
    valid = wf.geom.valid
    body = {"timestamp_utc": vol.timestamp, "capteurs_utilises": vol.n_sensors,
            "inference_s": 0.0, "vitesse_max_m_s": round(float(vol.speed[valid].max()), 3),
            "statut_qualite": rapport["verdict"],
            "avertissements": rapport["motifs"] + rapport["avertissements"],
            "qualite": rapport}
    return 200, json.loads(json.dumps(body))


def test_api_desactivee_rien_ne_change(service_env, trained, wf, tmp_path):
    """Alertes desactivees : corps COMPLET de l'upload (200 et 422) identique a celui de
    main, recursivement (hors temps mesure et horloge) ; rapport M3 sans cle ajoutee."""
    from fastapi.testclient import TestClient
    lancer, cle = service_env
    with TestClient(lancer(False), headers={"X-API-Key": cle}) as c:
        r = c.get("/api/alerts")
        assert r.status_code == 409 and "alerts.enabled" in r.json()["detail"]
        for corrompre, statut in ((False, 200), (True, 422)):
            data = _csv_instant(trained, corrompre)
            up = c.post("/api/reconstruct/upload", files={"file": ("i.csv", data, "text/csv")})
            code, attendu = _corps_main(wf, data, tmp_path)
            assert up.status_code == code == statut, up.text
            assert _sans_volatils(up.json()) == _sans_volatils(attendu)
            q = up.json()["qualite"]["qualite"]
            assert q is None or set(q) == CLES_QUALITE_M3
        out = _mcp(c, "alertes_actives")
        assert out["disponible"] is False
    assert not (Path(trained.paths.run_dir) / "logs" / "alertes.jsonl").exists()


def test_upload_hors_de_la_boucle_d_evenements(service_env, trained, monkeypatch):
    """Le calcul de l'upload (contrat, passes, alertes) tourne dans un thread du pool :
    pendant qu'il travaille, /api/health repond tout de suite."""
    import threading
    import time

    from fastapi.testclient import TestClient

    from service import app as svc
    lancer, cle = service_env
    libere, entre = threading.Event(), threading.Event()

    def lent(*args):
        entre.set()
        libere.wait(20)
        return {"ok": True}

    monkeypatch.setattr(svc, "_upload_calcul", lent)
    with TestClient(lancer(False), headers={"X-API-Key": cle}) as c:
        res = {}
        t = threading.Thread(target=lambda: res.update(r=c.post(
            "/api/reconstruct/upload", files={"file": ("i.csv", b"x", "text/csv")})))
        t.start()
        try:
            assert entre.wait(20)
            t0 = time.perf_counter()
            assert c.get("/api/health").status_code == 200
            assert time.perf_counter() - t0 < 5.0          # la boucle n'est pas bloquee
        finally:
            libere.set()
            t.join(30)
        assert res["r"].status_code == 200 and res["r"].json() == {"ok": True}


def test_api_alertes_activees(service_env, trained, schema_alertes, tmp_path):
    from fastapi.testclient import TestClient
    lancer, cle = service_env
    stems = _jour(trained, "test")
    with TestClient(lancer(True)) as c:
        assert c.get("/api/alerts").status_code == 401                 # cle requise
        h = {"X-API-Key": cle}
        for stem in stems:
            r = c.get("/api/alerts", params={"stem": stem}, headers=h)
            assert r.status_code == 200, r.text
            _valide(schema_alertes, r.json())
        assert c.get("/api/alerts", headers=h).json() == r.json()       # dernier calcul
        assert c.get("/api/alerts", params={"stem": "999_000000"},
                     headers=h).status_code == 404
        up = c.post("/api/reconstruct/upload", headers=h,
                    files={"file": ("i.csv", _csv_instant(trained), "text/csv")})
        assert up.status_code == 200, up.text
        body = up.json()
        _valide(schema_alertes, body["alertes"])
        assert set(body["physique"]) == CLES_PHYSICS
        c.headers.update(h)
        out = _mcp(c, "alertes_actives", {"stem": stems[0]})
        assert out["disponible"] is True and set(out["ouvertes_par_niveau"]) == set(NIVEAUX)
        assert all(a["status"] == "ouverte" for a in out["items"])
    assert (tmp_path / "alertes.jsonl").exists()
