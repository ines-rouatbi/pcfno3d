"""OS6 national (phase 5, configs/national/regle_os6_national.json) sur le mini-jeu : capteurs simules
comme a l'entrainement du candidat D, metrique au-dessus du sol, georeferencement de la grille, regle
hachee, choix de la graine de D sur la validation, sorties au format du Grand Tunis."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_forcage_era5 import _jeu_reel, cfg_era5, forcage  # noqa: F401  (fixtures)
from test_os6 import _contraintes, jeu  # noqa: F401  (fixture : mini-jeu reel entraine)

REPO = Path(__file__).resolve().parents[1]
REGLE = REPO / "configs" / "national" / "regle_os6_national.json"


def _script():
    spec = importlib.util.spec_from_file_location("os6_script_nat", REPO / "scripts" / "os6_reseau_aneo.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _regle(tmp_path, **kw) -> Path:
    """Une regle d'essai au format de la regle nationale (parametres reduits au mini-jeu), hachee."""
    p = json.loads(REGLE.read_text(encoding="utf-8"))["parametres"]
    p.update({"n_val": 2, "n_test": 2, "par_etape": 2, "n_max": 4, "tailles": [2, 4],
              "rayon_influence_m": 300.0, "graines_hasard": [0, 1],
              "contraintes": {"pas_candidats_m": 100.0, "distance_min_m": 200.0, "marge_bord_m": 0.0,
                              "altitude_max_m": 1e4, "pente_max_deg": 90.0, "z0_min_m": 0.0,
                              "n_max": 4}, **kw})
    f = tmp_path / "regle_essai.json"
    f.write_text(json.dumps({"statut": "PROPOSITION (essai)", "parametres": p}, indent=1), encoding="utf-8")
    (tmp_path / "regle_essai.json.sha256").write_text(
        hashlib.sha256(f.read_bytes()).hexdigest() + "  regle_essai.json\n", encoding="ascii")
    return f


# ---------------------------------------------------------------------- regle et choix du modele
def test_regle_nationale_hachee_et_coherente_avec_D():
    from pcfno3d import os6

    r, h = os6.charger_regle(REGLE)
    assert "PROPOSITION" in r["statut"]
    P = r["parametres"]
    # memes candidats que les capteurs simules de D (contraintes par defaut de l'OS6), budget 400
    defaut = os6.contraintes_dict(os6.Contraintes())
    assert {k: v for k, v in P["contraintes"].items() if k != "n_max"} == \
        {k: v for k, v in defaut.items() if k != "n_max"}
    assert P["contraintes"]["n_max"] == P["n_max"] == 400
    assert tuple(P["tailles"]) == os6.TAILLES_NATIONAL and max(P["tailles"]) >= 400
    v4 = json.loads((REPO / "configs" / "national" / "regle_v4_national.json").read_text(encoding="utf-8"))
    assert "25 m" in v4["candidats"]["D"]["capteurs_simules"] and P["hauteur_capteur_m"] == 25.0
    assert P["metrique"] == "sol" and P["georeferencement"] == "grille"
    assert P["seuil_m_s"] == 0.5


def test_regle_modifiee_refusee(tmp_path):
    from pcfno3d import os6

    f = tmp_path / "r.json"
    f.write_bytes(REGLE.read_bytes() + b" ")
    (tmp_path / "r.json.sha256").write_text(REGLE.with_suffix(".json.sha256").read_text(encoding="ascii"))
    with pytest.raises(ValueError, match="different"):
        os6.charger_regle(f)
    (tmp_path / "s.json").write_bytes(REGLE.read_bytes())
    with pytest.raises(ValueError, match="absente"):
        os6.charger_regle(tmp_path / "s.json")


def test_meilleure_graine_sur_la_validation(tmp_path):
    from pcfno3d import os6

    for g, scores in ((42, [3.0, 2.0, 2.5]), (43, [2.2, 1.7]), (44, [1.9])):
        d = tmp_path / f"national_1km_v4_D_s{g}"
        d.mkdir()
        (d / "history.json").write_text(json.dumps([{"epoch": i, "val_score": s} for i, s in enumerate(scores)]))
    c = os6.meilleure_graine(tmp_path)
    assert c["graine"] == 43 and c["scores_validation"] == {"42": 2.0, "43": 1.7, "44": 1.9}
    assert c["run_dir"].endswith("national_1km_v4_D_s43")
    with pytest.raises(FileNotFoundError):
        os6.meilleure_graine(tmp_path, graines=(42, 45))


# ---------------------------------------------------------------------- geometrie
def test_georef_grille_identique_a_l_entree_era5(jeu):  # noqa: F811
    from pcfno3d import os6
    from pcfno3d.geometry import Geometry

    g = Geometry(jeu)
    geo = os6.GeoRefGrille.depuis_geometrie(g)
    jj, ii = np.meshgrid(np.arange(g.ny), np.arange(g.nx), indexing="ij")
    lat, lon = geo.to_latlon(ii * g.dx, jj * g.dy)
    assert np.allclose(lat, g.interp_era5.lat) and np.allclose(lon, g.interp_era5.lon)
    x, y = geo.to_xy(lat, lon)
    assert np.allclose(x, ii * g.dx) and np.allclose(y, jj * g.dy)


def test_tranches_au_dessus_du_sol_comme_dso_hauteur_sol(jeu):  # noqa: F811
    from pcfno3d import os6
    from pcfno3d.geometry import Geometry

    sys.path.insert(0, str(REPO / "scripts"))
    from dso_hauteur_sol import BORNES, PART_MIN, masques_tranches

    assert tuple(BORNES) == os6.BORNES_SOL and PART_MIN == os6.PART_MIN_TRANCHE
    g = Geometry(jeu, device="cpu")
    lab = os6.etiquettes_sol(g)
    m, _ = masques_tranches(g)
    for t, (a, b) in enumerate(zip(BORNES[:-1], BORNES[1:])):
        assert np.array_equal(lab == t, m[f"{a:g}-{b:g} m"].cpu().numpy())
    assert np.array_equal(lab >= 0, m["0-1000 m (domaine)"].cpu().numpy())


def test_zones_exclues_geojson(jeu, tmp_path):  # noqa: F811
    from pcfno3d import os6
    from pcfno3d.geometry import Geometry

    g = Geometry(jeu)
    geo = os6.GeoRefGrille.depuis_geometrie(g)
    c = os6.Contraintes.charger(_contraintes(tmp_path))
    tous = os6.candidats(g, c, geo)
    # moitie ouest du domaine exclue (GeoJSON : lon, lat)
    la0, lo0 = geo.to_latlon(-10.0, -10.0)
    la1, lo1 = geo.to_latlon(g.nx * g.dx / 2, g.ny * g.dy + 10)
    poly = [[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1], [lo0, la0]]
    f = tmp_path / "zones.geojson"
    f.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"nom": "essai"}, "geometry": {"type": "Polygon", "coordinates": [poly]}}]}))
    z = os6.zones_geojson(f)
    assert z[0][0] == [la0, lo0]                                  # [lat, lon] pour Contraintes
    c.exclusions = z
    reste = os6.candidats(g, c, geo)
    assert 0 < len(reste) < len(tous) and (reste[:, 1] * g.dx > g.nx * g.dx / 2 - 1e-6).all()
    with pytest.raises(ValueError, match="non prise en charge"):
        f.write_text(json.dumps({"type": "Point", "coordinates": [0, 0]}))
        os6.zones_geojson(f)


# ---------------------------------------------------------------------- capteurs simules et passes
def test_surfaces_par_masque_identiques_a_la_surface_commune(jeu):  # noqa: F811
    from pcfno3d.inference import WindFieldModel
    from pcfno3d import os6

    wf = WindFieldModel(jeu, device="cpu", verbose=False)
    b = os6.Banc(wf, "validation", 1, verbose=False)
    s_raw, sensor, tfeat, y, j = b.ds[0]
    S, m = s_raw.numpy(), sensor.numpy() > 0.5
    extra = tfeat.numpy()[os6.N_TFEAT:]
    a = wf._members(S, [m, m], 12.0, 100, extra=extra)
    c = wf._members(S, [m, m], 12.0, 100, extra=extra, surfaces=[S.copy(), S.copy()])
    for x, z in zip(a, c):
        for p, q in zip(x, z):
            assert np.allclose(p, q, atol=1e-5)
    with pytest.raises(ValueError, match="une surface par masque"):
        wf._members(S, [m, m], 12.0, 100, extra=extra, surfaces=[S])


def test_capteurs_simules_comme_l_entrainement_de_D(jeu, monkeypatch):  # noqa: F811
    """Valeur = EntreeANEO.valeur_a_hauteur (25 m) + bruit 0,1 + 2 % ; meme tirage pour toutes les
    dispositions ; le capteur prime sur le pseudo-capteur ERA5 de sa maille ; sans capteur, l'ERA5 reste."""
    from pcfno3d import os6
    from pcfno3d.aneo_entree import EntreeANEO
    from pcfno3d.inference import WindFieldModel

    wf = WindFieldModel(jeu, device="cpu", verbose=False)
    g = wf.geom
    capt = os6.CapteursSimules(g, 25.0, graine=7)
    b = os6.Banc(wf, "validation", 1, verbose=False, capteurs=capt, metrique="sol")
    s_raw, sensor, _t, y, j = b.ds[0]
    S, base, Y = s_raw.numpy(), sensor.numpy() > 0.5, y.numpy()
    ok = np.isfinite(S[:7]).all(axis=0)
    treillis = np.argwhere(base & ok)                 # pseudo-capteurs ERA5 (et stations) de l'instant
    c1 = tuple(int(v) for v in treillis[0])
    libres = np.argwhere(~base & ok & np.asarray(g.valid).any(axis=0))
    c2 = tuple(int(v) for v in libres[0])
    vus = []
    orig = wf._members

    def espion(S0, masks, h, d, **kw):
        vus.append((masks, kw.get("surfaces")))
        return orig(S0, masks, h, d, **kw)
    monkeypatch.setattr(wf, "_members", espion)
    reseaux = {"vide": os6.masque(g, []), "un": os6.masque(g, [c1]), "deux": os6.masque(g, [c1, c2])}
    r = b.passe(reseaux)
    noms = list(reseaux)
    tout = [s for _m, ss in vus for s in ss]
    masques = [m for mm, _s in vus for m in mm]
    surf = dict(zip(noms, tout))
    msk = dict(zip(noms, masques))
    S_cache = np.load(b.ds.cache_dir / f"{b.ds.stems[j]}_S.npy")
    e = EntreeANEO(g, None, 0.0, (0.0, 0.0), (0, 0), 25.0)
    z = np.random.default_rng([7, 1000, int(j)]).standard_normal((2, g.ny, g.nx))
    for c in (c1, c2):
        uu, vv = e.valeur_a_hauteur(S_cache, Y, *c)
        sd = 0.1 + 0.02 * np.hypot(uu, vv)
        attendu = (uu + sd * z[0][c], vv + sd * z[1][c])
        assert np.allclose((surf["deux"][0][c], surf["deux"][1][c]), attendu, atol=1e-5)
    assert surf["un"][0][c1] == surf["deux"][0][c1]               # meme tirage, deux dispositions
    assert surf["vide"][0][c1] == S[0][c1] != surf["un"][0][c1]   # sans capteur : l'ERA5 de la maille
    assert msk["deux"][c2] and not msk["un"][c2]
    # metrique au-dessus du sol : tranches, domaine 0-1 km, critere = RMSE vectoriel du domaine
    for n in noms:
        assert set(r[n]["tranches"]["tranche"]) >= {"0-100 m", "0-1000 m (domaine)"}
        assert np.isfinite(r[n]["rmse_max_uv_sol"]) and r[n]["rmse_critere"] == r[n]["rmse_vect_sol"]


# ---------------------------------------------------------------------- bout en bout
def test_os6_national_de_bout_en_bout(jeu, tmp_path, monkeypatch):  # noqa: F811
    """--regle : placement (validation), courbe propre et courbe d'information ERA5 degradee (test),
    plan ; colonnes du Grand Tunis conservees, colonnes nationales en plus ; reprise ; autre regle refusee."""
    import pcfno3d.config as config_mod

    m = _script()
    monkeypatch.setattr(config_mod.Config, "load", classmethod(lambda cls, _n: jeu))
    deg = tmp_path / "deg.json"
    deg.write_text(json.dumps({"sigma_u": 1.0, "sigma_v": 1.0, "L_km": 100.0}))
    choix = tmp_path / "choix_modele.json"
    choix.write_text(json.dumps({"graine": 43, "scores_validation": {"43": 1.7}}))
    sortie = tmp_path / "os6_national"
    regle = _regle(tmp_path)
    argv = ["os6", "-c", "x", "--device", "cpu", "--regle", str(regle), "--sortie", str(sortie),
            "--degradation", str(deg), "--modele-info", str(choix), "--lot", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    assert m.main() == 0
    for f in ("candidats.csv", "ordre_validation.csv", "plan_aneo.csv", "plan_aneo.geojson",
              "courbe_test.csv", "courbe_test_par_niveau.csv", "courbe_test_par_tranche.csv",
              "courbe_test_era5_degradee.csv", "courbe_test_era5_degradee_par_niveau.csv",
              "courbe_test_era5_degradee_par_tranche.csv", "os6.json", "courbe.png", "carte.png",
              "regle_os6.json"):
        assert (sortie / f).exists(), f
    p = pd.read_csv(sortie / "plan_aneo.csv")
    assert list(p.columns) == ["id", "ordre", "etape", "iy", "ix", "score_erreur_m2_s2",
                               "gain_rmse_validation_m_s", "rmse_validation_apres_m_s", "x_m", "y_m",
                               "lat", "lon", "altitude_sol_m", "hauteur_mat_m", "gain_cumule_m_s"]
    assert len(p) == 4 and (p["hauteur_mat_m"] == 25.0).all()
    assert p[["lat", "lon", "gain_rmse_validation_m_s"]].notna().all().all()
    r = pd.read_csv(sortie / "courbe_test.csv")
    assert list(r.columns) == ["strategie", "n_capteurs", "rmse_vect", "pire_rmse_niveau",
                               "rmse_vect_sol", "rmse_max_uv_sol", "seuil_respecte"]
    assert set(r["strategie"]) == {"actuel", "optimise", "grille", "hasard"}
    assert (r["seuil_respecte"] == (r["rmse_max_uv_sol"] <= 0.5)).all()
    tt = pd.read_csv(sortie / "courbe_test_par_tranche.csv")
    assert {"tranche", "n", "rmse_u", "rmse_v", "retenue", "rmse_max_uv_sol"} <= set(tt.columns)
    rd = pd.read_csv(sortie / "courbe_test_era5_degradee.csv")
    assert list(rd.columns) == list(r.columns)
    assert not np.allclose(rd["rmse_vect"], r["rmse_vect"])     # l'ERA5 degradee change l'entree
    j = json.loads((sortie / "os6.json").read_text(encoding="utf-8"))
    assert j["regle_sha256"] == hashlib.sha256(regle.read_bytes()).hexdigest()
    assert j["choix_modele"]["graine"] == 43 and j["capteur_simule"]["hauteur_m"] == 25.0
    assert "rmse_max_uv_sol" in j["reseau_actuel"] and "comparaison_net1_sol" in j
    assert "nombre_qui_respecterait_le_seuil" in j["information_era5_degradee"]
    assert "mesures" in j["limites"]
    gj = json.loads((sortie / "plan_aneo.geojson").read_text(encoding="utf-8"))
    assert {f["properties"]["type"] for f in gj["features"]} == {"capteur ANEO propose", "station existante"}
    assert gj["metadata"]["georeferencement"] == "grille"
    # reprise : relit placement et courbes (le test n'est pas recalcule)
    avant = (sortie / "courbe_test_par_niveau.csv").stat().st_mtime_ns
    assert m.main() == 0
    assert (sortie / "courbe_test_par_niveau.csv").stat().st_mtime_ns == avant
    # un dossier deja calcule avec une autre regle est refuse
    (tmp_path / "x").mkdir()
    autre = _regle(tmp_path / "x", rayon_influence_m=500.0)
    monkeypatch.setattr(sys, "argv", [a if a != str(regle) else str(autre) for a in argv])
    with pytest.raises(SystemExit, match="autre regle"):
        m.main()
    monkeypatch.setattr(sys, "argv", argv + ["--contraintes", str(_contraintes(tmp_path))])
    with pytest.raises(SystemExit, match="fixe les contraintes"):
        m.main()


# ---------------------------------------------------------------------- carnet
def test_carnet_os6_national():
    """Carnet de la phase 5 : accord d'Ines en tete, commit fige, empreinte de la regle verifiee, aucun
    entrainement, graine de D choisie sur la validation ; le carnet du depot porte la regle actuelle."""
    sys.path.insert(0, str(REPO / "scripts"))
    import make_national_os6_nb as M

    sha = (REPO / "configs" / "national" / "regle_os6_national.json.sha256").read_text(encoding="ascii").split()[0]
    nb = M.os6_national("0123456789abcdef0123456789abcdef01234567")
    assert nb["metadata"]["colab"]["gpuType"] == "A100"
    texte = "\n".join("".join(c["source"]) for c in nb["cells"])
    tete = "".join(nb["cells"][0]["source"])
    assert "accord écrit d'Ines" in tete and "Aucun entraînement" in tete and "PROPOSITION" in tete
    assert "0123456789abcdef0123456789abcdef01234567" in texte and sha in texte
    assert "os6.meilleure_graine" in texte and "--regle" in texte and "os6_reseau_aneo.py" in texte
    for interdit in (" train\"", " train ", "evaluate", "dso_hauteur_sol.py"):
        assert interdit not in texte, interdit
    # le carnet du depot est genere depuis la regle actuelle (sinon : python scripts/make_national_os6_nb.py)
    f = REPO / "notebooks" / "current" / M.NOM
    assert f.exists() and sha in f.read_text(encoding="utf-8")
