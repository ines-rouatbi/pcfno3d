"""scripts/juger_national_gfs_v4.py : constantes recopiees de la regle, metrique, bootstrap des jours, critere."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("juger_gfs", REPO / "scripts" / "juger_national_gfs_v4.py")
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)


def test_regle_hachee_et_constantes_recopiees():
    sha = J.verifier_regle()
    regle = json.loads(J.REGLE.read_text(encoding="utf-8"))
    assert len(sha) == 64
    assert list(J.CASES) == regle["critere"]["cases_jugees"]
    assert "2000 tirages, graine 2030" in regle["metrique"]
    assert (J.TIRAGES, J.GRAINE) == (2000, 2030)
    assert "plus de 5 %" in regle["critere"]["par_source_et_echeance"] and J.DEGRADATION_MAX == 0.05
    assert "au plus 2 %" in regle["donnees_gfs"]["controle"] and J.ABSENTS_MAX == 0.02


def _paires(gain: float, degrade: str | None = None, n_jours: int = 20, graine: int = 0) -> pd.DataFrame:
    """12 aeroports, 8 instants par jour ; source a = obs + erreur (1 - gain) x celle de GFS."""
    rng = np.random.default_rng(graine)
    lignes = []
    for e in J.ECHEANCES:
        for d in range(n_jours):
            for h in range(8):
                for a in range(12):
                    uo, vo = rng.normal(0, 4, 2)
                    eg = rng.normal(0, 2.5, 2)
                    f = 1.3 if degrade == f"A{a:02d}" else (1 - gain)
                    lignes.append({"station": f"A{a:02d}", "jour": f"2022-01-{d + 1:02d}", "echeance_h": e,
                                   "stem": f"{d}_{h}", "u_obs": uo, "v_obs": vo,
                                   "u_gfs_brut": uo + eg[0], "v_gfs_brut": vo + eg[1],
                                   "u_a_national_gfs": uo + f * eg[0], "v_a_national_gfs": vo + f * eg[1],
                                   "u_b_gfs_oi_10m": (uo + f * eg[0]) if e == 0 else np.nan,
                                   "v_b_gfs_oi_10m": (vo + f * eg[1]) if e == 0 else np.nan})
    return pd.DataFrame(lignes)


def test_tirages_identiques_et_par_jour():
    w1, w2 = J.tirages_jours(list(range(10)), 50), J.tirages_jours(list(range(10)), 50)
    assert np.array_equal(w1, w2) and np.all(w1.sum(1) == 10)


def test_gain_net_accepte():
    p = _paires(0.10)
    v = J.juger(p, sorted(p["station"].unique()), "x" * 64)
    assert set(v["cases"]) == set(J.CASES)
    assert all(x == "accepte" for x in v["cases"].values()), v["cases"]
    d = v["details"]["a / analyse"]
    assert d["ecart"] < 0 and d["ic95"][1] <= 0


def test_aucun_gain_refuse():
    p = _paires(0.0)
    v = J.juger(p, sorted(p["station"].unique()), "x" * 64)
    # ecart nul : la borne haute de l'IC est > 0 (ou = 0 exactement) -> pas de gain prouve
    assert all(x in ("refuse", "accepte") for x in v["cases"].values())
    assert v["details"]["a / analyse"]["ecart"] == 0.0


def test_un_aeroport_degrade_refuse_meme_avec_gain_moyen():
    p = _paires(0.15, degrade="A03")
    v = J.juger(p, sorted(p["station"].unique()), "x" * 64)
    d = v["details"]["a / +24 h"]
    assert d["aeroports_degrades_plus_5_pct"] == ["A03"] and d["verdict"] == "refuse"


def test_aeroport_sans_paire_non_juge():
    p = _paires(0.10)
    stations = sorted(p["station"].unique()) + ["ZZZZ"]
    v = J.juger(p, stations, "x" * 64)
    assert v["details"]["a / analyse"]["verdict"] == "non juge"


def test_verdicts_lus_par_le_service(tmp_path, monkeypatch):
    from pcfno3d import statut_service as S
    p = _paires(0.10)
    v = J.juger(p, sorted(p["station"].unique()), J.verifier_regle())
    f = tmp_path / "verdicts_gfs_v4.json"
    f.write_text(json.dumps(v), encoding="utf-8")
    lu = S.verdicts_gfs_v4(f)
    assert lu == {c: True for c in J.CASES}
    assert S.statut(True, S.TEMPS_REEL, verdicts=lu)["statut"] == S.VALIDE
    v["regle_sha256"] = "0" * 64
    f.write_text(json.dumps(v), encoding="utf-8")
    assert S.verdicts_gfs_v4(f) == {}


# ---------------------------------------------------------------------- de bout en bout sur le mini-jeu
import copy  # noqa: E402
import shutil  # noqa: E402

import pytest  # noqa: E402

from test_forcage_era5 import cfg_era5, forcage  # noqa: E402,F401  (fixtures)
from test_os7_prevision import jeu_gfs  # noqa: E402,F401  (fixture)


def test_calculer_mini_jeu(jeu_gfs, tmp_path):  # noqa: F811
    """Passes du modele aux trois echeances, aeroport retire a l'analyse, OI a l'analyse seulement, verdicts lus
    par le service ; une seconde lecture dans le meme dossier est refusee."""
    c, p, _, _ = jeu_gfs
    base = copy.deepcopy(c)
    data = tmp_path / "jeu"
    shutil.copytree(c.paths.data_dir, data)
    base.apply_overrides({"paths.data_dir": data})
    ancien = json.loads((c.paths.data_dir / "stations.json").read_text(encoding="utf-8"))
    codes = [("DTTX", "DTTX Sfax"), ("DTTZ", "DTTZ Tozeur"), ("DTKA", "DTKA Tabarka")]
    (data / "stations.json").write_text(json.dumps(
        [{"nom": n, "code": k, "iy": s["iy"], "ix": s["ix"]} for (k, n), s in zip(codes, ancien)]), encoding="utf-8")
    gfs = tmp_path / "gfs"
    gfs.mkdir()
    src = Path(p.paths.forcage_era5)
    for e in J.ECHEANCES:
        for suf in (".csv", ".npz"):
            shutil.copy2(src.with_suffix(suf), gfs / f"forcage_gfs_f{e}{suf}")
    idx = pd.read_csv(c.paths.cache_dir / "index.csv")
    obs = tmp_path / "obs"
    obs.mkdir()
    lignes = [{"station_id": code, "timestamp_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "wind_speed_m_s": 3.0 + k,
               "wind_dir_deg": 200.0 + 40 * k, "wind_u_m_s": np.nan, "wind_v_m_s": np.nan}
              for t in pd.to_datetime(idx["timestamp"]) for k, (code, _) in enumerate(codes)]
    pd.DataFrame(lignes).drop_duplicates(["station_id", "timestamp_utc"]).to_csv(obs / "metar_2022.csv", index=False)

    sortie = tmp_path / "sortie"
    v = J.calculer(base, gfs, obs, sortie, device="cpu", verbose=False)
    paires = pd.read_csv(sortie / "paires.csv")
    assert set(paires["station"]) == {"Sfax", "Tozeur", "Tabarka"}
    assert set(paires["echeance_h"]) == set(J.ECHEANCES)
    a0, a24 = paires[paires["echeance_h"] == 0], paires[paires["echeance_h"] == 24]
    assert a0[["u_b_gfs_oi_10m", "u_era5", "u_a_national_gfs"]].notna().all().all()
    assert a24[["u_b_gfs_oi_10m", "u_era5"]].isna().all().all()
    # meme forcage aux trois echeances ici : GFS brut identique ; le modele differe (stations en entree a l'analyse)
    np.testing.assert_allclose(np.sort(a0["u_gfs_brut"]), np.sort(a24["u_gfs_brut"]))
    assert set(v["cases"]) == set(J.CASES) and v["instants"]["absents"] == 0
    assert (sortie / "gfs_v4.md").read_text(encoding="utf-8").startswith("# Règle")
    from pcfno3d import statut_service as S
    assert set(S.verdicts_gfs_v4(sortie / "verdicts_gfs_v4.json")) == set(J.CASES)
    with pytest.raises(SystemExit):
        J.calculer(base, gfs, obs, sortie, device="cpu", verbose=False)
