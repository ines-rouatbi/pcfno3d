"""Regle configs/national/regle_national_gfs_v6.json : empreinte, valeurs recopiees, choix sur 2021 avec la marge
(+3 %), instants de novembre 2026, reutilisation du code de la v5.2 (sorties propres), lecture des verdicts par le
service (seulement pour la variante servie), carnet. CPU, aucun reseau, aucune donnee reelle."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from test_forcage_era5 import HAUTEURS, cfg_era5, forcage  # noqa: F401  (fixtures)
from test_gfs_v5_national import _selection, jeu_v5  # noqa: F401  (fixtures)
from test_os7_prevision import jeu_gfs  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]


def _charger(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, REPO / "scripts" / fichier)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V6 = _charger("juger_gfs_v6_t", "juger_national_gfs_v6.py")
REGLE = json.loads(V6.REGLE.read_text(encoding="utf-8"))


def test_regle_hachee_et_valeurs_recopiees():
    assert V6.verifier_regle() == "9babee638cf81cbb86b82fc54ccbb55ef7bdc1c5232ca294013139b6c2c1dd25"
    e1, e2 = REGLE["etape_1_choix_sur_validation_2021"], REGLE["etape_2_jugement_novembre_2026"]
    assert "<= +3 %" in e1["admissible"] and V6.MARGE_CHOIX == 0.03
    assert "borne haute de l'IC 95 %" in e1["admissible"] and "CHACUNE des trois cases" in e1["admissible"]
    assert "0,005 m/s" in e1["retenue"] and "(C2, puis C1, puis C3)" in e1["retenue"]
    assert (V6.EX_AEQUO, V6.ORDRE_SIMPLICITE) == (0.005, ("c2", "c1", "c3"))
    assert "validation_2021_v5_2/variantes.json" in e1["donnees"] and "aucun nouveau calcul" in e1["donnees"]
    assert "du 2026-11-01 00 UTC au 2026-11-23 21 UTC inclus, toutes les 3 heures : 184 instants" in e2["periode"]
    assert (V6.DEBUT, V6.FIN, V6.PAS_H) == ("2026-11-01T00:00", "2026-11-23T21:00", 3)
    assert "f003" in e2["heures_intermediaires"] and "f027" in e2["heures_intermediaires"]
    assert "n'est PAS calcule" in e2["sources"]
    # metrique et critere recopies de la v5.2 a l'identique
    r52 = json.loads(V6.REGLE_V5_2.read_text(encoding="utf-8"))
    assert REGLE["metrique"] == r52["metrique"]
    assert REGLE["critere"]["par_case"] == r52["critere"]["par_case"]
    assert REGLE["critere"]["cases_jugees"] == r52["critere"]["cases_jugees"]
    # les regles precedentes restent hachees et intactes
    assert V6.C.verifier_regle(V6.REGLE_V5_2).startswith("542e676c")


def test_instants_de_novembre_et_echeances_du_service():
    from pcfno3d.gfs import run_pour
    ts = V6.instants_novembre()
    assert len(ts) == 184 and ts[0] == pd.Timestamp("2026-11-01T00:00") and ts[-1] == pd.Timestamp("2026-11-23T21:00")
    t = pd.Timestamp("2026-11-05T03:00")
    assert [run_pour(t, e)[1] for e in (0, 24, 48)] == [3, 27, 51]
    t = pd.Timestamp("2026-11-05T06:00")
    assert [run_pour(t, e)[1] for e in (0, 24, 48)] == [0, 24, 48]


def _resultat(degr: dict, haut: float, rmse: float) -> dict:
    """Un resultat de variantes.json : memes degradations et borne haute dans les trois cases."""
    case = {"ic95": [haut - 0.1, haut], "par_aeroport": {a: {"degradation": d} for a, d in degr.items()}}
    return {"rmse_moyen_trois_cases": rmse, "details": {c: copy.deepcopy(case) for c in V6.CASES}}


def test_choix_avec_marge():
    ok = {"Tabarka": 0.029, "Sfax": -0.1}
    v = {"c1": _resultat({"Tabarka": 0.04, "Sfax": -0.2}, -0.1, 2.80),       # Tabarka +4 % : pas admissible
         "c3_a0.75_H100": _resultat({"Tabarka": 0.031, "Sfax": -0.1}, -0.1, 2.85),
         "c3_a0.25_H100": _resultat(ok, -0.05, 2.93),
         "c2_a0.25_H100": _resultat(ok, -0.02, 2.934),                        # ex aequo a 0,005 : la plus simple
         "c2_a0.5_H100": _resultat(ok, 0.01, 2.90)}                           # IC qui touche 0 : pas admissible
    s = V6.choisir(v)
    assert s["arret"] is False and s["retenue"] == "c2_a0.25_H100"
    adm = {x["nom"]: x["admissible"] for x in s["lignes"]}
    assert adm == {"c1": False, "c3_a0.75_H100": False, "c3_a0.25_H100": True, "c2_a0.25_H100": True,
                   "c2_a0.5_H100": False}
    rien = V6.choisir({"c1": _resultat({"Tabarka": 0.06}, -0.1, 2.8)})
    assert rien["arret"] is True and rien["retenue"] is None and "novembre 2026 n'est pas lu" in rien["raison"]


def test_commande_choisir_verifie_la_source_et_n_ecrase_rien(tmp_path):
    v52 = V6.C.verifier_regle(V6.REGLE_V5_2)
    noms = [x["nom"] for x in V6.C.variantes()]
    res = {n: _resultat({"Tabarka": 0.02, "Sfax": -0.1}, -0.05, 2.9 + i / 100) for i, n in enumerate(noms)}
    f = tmp_path / "variantes.json"
    f.write_text(json.dumps({"regle_sha256": "autre", "variantes": res}), encoding="utf-8")
    with pytest.raises(SystemExit, match="empreinte"):
        V6.commande_choisir(f, tmp_path / "v6")
    f.write_text(json.dumps({"regle_sha256": v52, "variantes": res}), encoding="utf-8")
    s = V6.commande_choisir(f, tmp_path / "v6")
    assert s["retenue"] == "c1" and s["variante"]["famille"] == "c1" and s["regle_sha256"] == V6.verifier_regle()
    assert (tmp_path / "v6" / "choix_2021_v6.md").exists()
    with pytest.raises(SystemExit, match="deja une selection"):
        V6.commande_choisir(f, tmp_path / "v6")


def test_jugement_novembre_par_le_code_de_la_v5_2(jeu_v5, tmp_path, monkeypatch):
    """Le code de la v5.2 avec l'empreinte, la periode et les sorties de la v6 ; sans C' s44 sans correction."""
    from pcfno3d import statut_service as S

    base, _, tn, obs, tous = jeu_v5
    V6.configurer_v5_pour_v6()
    monkeypatch.setattr(V6.J, "echantillon_2026", lambda: list(tous[8:]))    # mini periode (archive synthetique)
    sha = V6.verifier_regle()
    assert V6.C.verifier_regle() == sha
    sortie = tmp_path / "resultat"
    sortie.mkdir()
    (sortie / "controle_entree.json").write_text(json.dumps({"regle_sha256": sha, "passe": True}), encoding="utf-8")
    var = next(v for v in V6.C.variantes() if v["nom"] == "c2_a0.25_H100")
    sel = _selection(tmp_path, sha, variante=var)
    v = V6.J.juger(base, copy.deepcopy(base), sel, list(tn.values()), obs, sortie, device="cpu", verbose=False,
                   avec_cp_s44=False)
    assert v["regle_sha256"] == sha and v["candidat"]["nom"] == "c2_a0.25_H100"
    for f in ("verdicts_gfs_v6.json", "gfs_v6.md", "paires.csv", "bilan.csv"):
        assert (sortie / f).exists(), f
    assert not (sortie / "verdicts_gfs_v5.json").exists()
    assert "regle_national_gfs_v6" in (sortie / "gfs_v6.md").read_text(encoding="utf-8")
    lu = S.verdicts_gfs_v6(sortie / "verdicts_gfs_v6.json")
    assert set(lu["cases"]) == set(V6.CASES) and lu["candidat"]["nom"] == "c2_a0.25_H100"
    with pytest.raises(SystemExit, match="une seule lecture"):
        V6.J.juger(base, base, sel, list(tn.values()), obs, sortie, device="cpu", verbose=False, avec_cp_s44=False)


def test_service_lit_les_verdicts_v6_pour_la_variante_servie(tmp_path, monkeypatch):
    from pcfno3d import statut_service as S
    from service import atom_outils

    f = tmp_path / "verdicts_gfs_v6.json"
    cand = {"nom": "c3_a0.25_H100", "famille": "c3", "a0": 0.25, "H": 100.0}
    f.write_text(json.dumps({"regle_sha256": V6.verifier_regle(), "candidat": cand,
                             "cases": {c: "accepte" for c in V6.CASES}}), encoding="utf-8")
    monkeypatch.setenv("PCFNO3D_VERDICTS_GFS_V6", str(f))
    geom = SimpleNamespace(nx=388, dx=1000.0, ny=828, dy=1000.0)
    ck = "/drive/runs/national_1km_v5_gfs/best.pt"

    def wf(a0, h=100.0):
        inf = SimpleNamespace(ancrage_gfs_a0=a0, ancrage_gfs_h=h, checkpoint_temps_reel=ck)
        return SimpleNamespace(cfg=SimpleNamespace(inference=inf), geom=geom)

    assert set(atom_outils.verdicts_v6_de(wf(0.25))) == set(V6.CASES)       # la variante jugee est servie
    assert atom_outils.verdicts_v6_de(wf(0.75)) == {}                         # autre reglage servi : ignore
    s = S.statut(True, S.TEMPS_REEL, verdicts_v6=atom_outils.verdicts_v6_de(wf(0.25)), zones=S.regle_zones(),
                 lat=36.95, lon=8.75)
    assert s["statut"] == S.VALIDE and "regle_national_gfs_v6" in s["preuve"]  # partout, Tabarka compris
    f.write_text(json.dumps({"regle_sha256": "autre", "candidat": cand, "cases": {}}), encoding="utf-8")
    assert S.verdicts_gfs_v6() == {}


def test_carnet_v6():
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        g = _charger("make_gfs_v6_nb_t", "make_national_gfs_v6_nb.py")
    finally:
        sys.path.remove(str(REPO / "scripts"))
    nb = g.carnet_v6("a" * 40)
    x = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert nb["metadata"]["colab"]["gpuType"] == "L4"
    assert x.index("COMMIT_FIGE") < x.index("UNITES_DEJA_UTILISEES") < x.index("nvidia-smi") < x.index("drive.mount")
    assert "except FileNotFoundError" in x                                  # session sans GPU : liberee, pas plantee
    assert V6.verifier_regle() in x and "regle_national_gfs_v5_2.json" in x
    assert x.index('(GV6 / "selection.json").exists()') < x.index("zf.extract(m, CACHE)")
    assert x.index("juger_national_gfs_v6.py choisir") < x.index("pcfno3d.gfs") < x.index("historique_metar.py") \
        < x.index("juger_national_gfs_v6.py controle") < x.index("juger_national_gfs_v6.py juger")
    assert "--periode 2026-11-01T00:00,2026-11-23T21:00" in x and "--jours-impairs" not in x
    assert "--sans-cp-s44" not in x and "runtime.unassign()" in x
    for interdit in ("cdsapirc", "apikey", "rmtree(GV", "unlink(", "shutil.move("):
        assert interdit not in x, interdit
