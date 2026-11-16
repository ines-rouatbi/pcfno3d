"""Mode, date et statut des donnees servies (demande d'Ines du 16/11/2026)."""
import json

from pcfno3d import statut_service as st


def test_analyse_validee_et_temps_reel_national_experimental():
    assert st.statut(True, st.ANALYSE)["statut"] == st.VALIDE
    assert st.statut(True, st.ANALYSE, couche_10m=True)["statut"] == st.VALIDE
    assert st.statut(False, st.ANALYSE)["statut"] == st.VALIDE
    for mode in (st.TEMPS_REEL, st.PREVISION):
        for c in (False, True):
            s = st.statut(True, mode, couche_10m=c, j=1)
            assert s["statut"] == st.EXPERIMENTAL, (mode, c)
    assert "regle_national_gfs_v4" in st.statut(True, st.TEMPS_REEL)["preuve"]


def test_grand_tunis_temps_reel_et_prevision_v4_experimentaux_v5_valide():
    assert st.statut(False, st.TEMPS_REEL)["statut"] == st.EXPERIMENTAL
    assert st.statut(False, st.PREVISION, j=1)["statut"] == st.EXPERIMENTAL
    assert st.statut(False, st.PREVISION, j=1, prevision_os7=True)["statut"] == st.VALIDE
    assert st.statut(False, st.PREVISION, j=0, prevision_os7=True)["statut"] == st.EXPERIMENTAL


def test_verdicts_ignores_sans_empreinte_de_la_regle(tmp_path, monkeypatch):
    regle = tmp_path / "regle_national_gfs_v4.json"
    regle.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(st, "REGLE_GFS", regle)
    v = tmp_path / "verdicts.json"
    v.write_text(json.dumps({"regle_sha256": "abc", "cases": {"a / analyse": "accepte"}}), encoding="utf-8")
    assert st.verdicts_gfs_v4(v) == {}                       # regle pas encore hachee
    (tmp_path / "regle_national_gfs_v4.json.sha256").write_text("def  regle_national_gfs_v4.json\n")
    assert st.verdicts_gfs_v4(v) == {}                       # empreinte differente
    (tmp_path / "regle_national_gfs_v4.json.sha256").write_text("abc  regle_national_gfs_v4.json\n")
    ver = st.verdicts_gfs_v4(v)
    assert ver == {"a / analyse": True}
    assert st.statut(True, st.TEMPS_REEL, verdicts=ver)["statut"] == st.VALIDE
    assert st.statut(True, st.TEMPS_REEL, couche_10m=True, verdicts=ver)["statut"] == st.EXPERIMENTAL


def test_prevision_couche_10m_dit_gfs_brut():
    assert "GFS brut" in st.statut(True, st.PREVISION, couche_10m=True)["preuve"]


def test_la_regle_proposee_n_est_pas_encore_hachee():
    assert st.REGLE_GFS.exists()
    assert "PROPOSITION" in json.loads(st.REGLE_GFS.read_text(encoding="utf-8"))["statut"]
