"""Contrat de donnees ANEO (cahier des charges, section 8) : critere DE1.

Trois jeux de non-regression (tests/data/contract/, regeneres par
scripts/make_contract_files.py) et un test par regle sur une copie mutee du fichier
conforme.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandera")

from pcfno3d.contract import (  # noqa: E402
    Grille,
    Seuils,
    chemin_rapport,
    controler,
    densite_gaz_parfaits,
    lire_csv,
    mettre_en_quarantaine,
    valeurs_figees,
    valider_dataframe,
    valider_fichier,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tests" / "data" / "contract"


@pytest.fixture(scope="module")
def grille() -> Grille:
    return Grille.depuis_meta(DATA)


@pytest.fixture
def brut() -> pd.DataFrame:
    """Le fichier conforme, tel que lu (une copie neuve par test)."""
    df, erreurs = lire_csv(DATA / "conforme.csv")
    assert not erreurs
    return df


def _regles(rapport: dict) -> set[str]:
    return {e["regle"] for e in rapport["erreurs"]}


def _erreur(rapport: dict, regle: str, colonne: str | None = None) -> dict:
    found = [e for e in rapport["erreurs"]
             if e["regle"] == regle and (colonne is None or e["colonne"] == colonne)]
    assert found, f"{regle}/{colonne} absent de {sorted(_regles(rapport))}"
    return found[0]


# ---------------------------------------------------------------------- non-regression
def test_fichier_conforme(grille):
    r = valider_fichier(DATA / "conforme.csv", grille)
    assert r["verdict"] == "conforme", r["motifs"]
    assert r["erreurs"] == [] and r["avertissements"] == []
    q = r["qualite"]
    assert q["instants"] == 4 and q["stations_declarees"] == 24
    assert q["stations_valides"]["part"] > 0.95
    ip = q["interpolees_ou_signalees"]
    assert ip["nombre"] == 2 and ip["source_interpolee"] == 2 and ip["mask_i_obs_0"] == 2
    assert q["densite"]["incoherentes"] == 0 and q["densite"]["ecart_rel_max"] < 1e-3
    assert q["densite"]["temperature_virtuelle"]["pression_de_vapeur"] == r["resume"]["lignes"]
    assert q["fraicheur"]["stations_muettes"]["nombre"] == 0
    assert all(v["nombre"] == 0 for v in q["hors_plage"].values())
    json.dumps(r)                                     # le rapport est du JSON pur


def test_fichier_corrompu_rejete_avec_toutes_les_erreurs(grille):
    r = valider_fichier(DATA / "corrompu.csv", grille)
    assert r["verdict"] == "rejet"
    # validation paresseuse : TOUTES les fautes du fichier sont rapportees d'un coup
    assert _regles(r) == {
        "type_reel", "type_entier", "type_horodatage", "valeur_manquante", "hors_plage",
        "point_id_inconnu", "mask_i_obs_invalide", "source_non_declaree",
        "source_incoherente", "doublon", "pas_incompatible", "hour_utc_incoherent",
        "densite_incoherente"}
    e = _erreur(r, "type_reel", "temperature_c")
    assert e["lignes"] == [2] and e["exemples"] == ["abc"] and e["gravite"] == "bloquant"
    assert _erreur(r, "type_reel", "wind_u_m_s")["exemples"] == ["12,5"]
    assert _erreur(r, "type_entier", "point_id")["exemples"] == ["x7"]
    assert _erreur(r, "point_id_inconnu")["exemples"] == [999.0]
    assert _erreur(r, "doublon")["nombre"] == 3
    assert _erreur(r, "hors_plage", "temperature_c")["gravite"] == "mesure"
    for col in ("temperature_c", "relative_humidity_pct", "pressure_pa", "wind_w_m_s"):
        assert _erreur(r, "hors_plage", col)["nombre"] == 1
        assert r["qualite"]["hors_plage"][col]["nombre"] == 1     # indicateur 8.3
    assert r["qualite"]["hors_plage"]["wind_u_m_s"]["nombre"] == 0
    assert any("erreurs bloquantes" in m for m in r["motifs"])
    # l'instant mal aligne (03:00:07) ne cree pas un faux instant ou tout serait muet
    assert r["qualite"]["instants"] == 4


def test_fichier_en_panne_en_quarantaine(grille):
    r = valider_fichier(DATA / "panne.csv", grille, Seuils(tuile_m=100.0))
    assert r["verdict"] == "quarantaine", r["motifs"]
    assert not [e for e in r["erreurs"] if e["gravite"] == "bloquant"]
    q = r["qualite"]
    fr = q["fraicheur"]
    assert fr["stations_muettes"]["ids"] == [1, 2, 17, 18, 23, 24]
    assert fr["valeurs_figees"]["ids"] == [1, 2]
    assert fr["valeurs_figees"]["par_variable"] == {"wind_u_m_s": 2, "wind_v_m_s": 2,
                                                    "temperature_c": 0}
    assert fr["absences"] == 12 and fr["stations_absentes"]["nombre"] == 0
    assert fr["stations_sans_mesure_dernier_instant"] == 4
    assert q["interpolees_ou_signalees"]["part"] == pytest.approx(1 / 3, abs=1e-3)
    assert q["interpolees_ou_signalees"]["stations_toujours_interpolees"] == 7
    assert q["stations_valides"]["part_pire_instant"] == pytest.approx(11 / 24, abs=1e-3)
    assert q["stations_valides"]["part"] >= Seuils().rejet_stations_valides
    # couverture : le coin nord-est (tuile y=1, x=2) n'a plus de station valide
    cov = q["couverture"]
    assert cov["tuiles_y_x"] == [2, 3]
    assert cov["part_tuiles_vides_pire_instant"] == pytest.approx(1 / 6, abs=1e-3)
    assert cov["part_valide_par_tuile"][1][2] == 0.25     # valide au 1er instant seulement
    motifs = " ".join(r["motifs"])
    assert "muettes" in motifs and "interpolees" in motifs and "stations valides" in motifs


def test_generateur_des_jeux_reproduit_les_verdicts(tmp_path, grille):
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_contract_files.py"),
                    str(tmp_path)], check=True, capture_output=True)
    g = Grille.depuis_meta(tmp_path)
    assert g == grille
    for name, attendu in (("conforme", "conforme"), ("corrompu", "rejet"),
                          ("panne", "quarantaine")):
        assert valider_fichier(tmp_path / f"{name}.csv", g)["verdict"] == attendu
        assert ((tmp_path / f"{name}.csv").read_bytes()
                == (DATA / f"{name}.csv").read_bytes()), f"{name}.csv a regenerer"


def test_fichier_journalier_du_mini_dataset(mini_dataset):
    """Le contrat suit les fichiers reels du generateur : un jour du mini-dataset passe."""
    path = sorted(mini_dataset.glob("aneo_observations_day_*"))[0]
    r = valider_fichier(path, Grille.depuis_meta(mini_dataset))
    assert r["verdict"] == "conforme", r["motifs"]
    assert r["resume"]["jour_du_fichier"]["day_index"] == 1
    assert r["resume"]["lignes_noeuds_3d"] > 0
    # le mini-dataset marque ~10 % de capteurs manquants (mask_i_obs = 0) : avertissement
    assert any("interpolees" in a for a in r["avertissements"])
    surface = valider_fichier(path, Grille.depuis_meta(mini_dataset), surface_seule=True)
    assert surface["resume"]["lignes_noeuds_3d"] == 0
    assert surface["verdict"] == "conforme"


# ---------------------------------------------------------------------- une regle a la fois
def _ligne(df: pd.DataFrame, pid: int, hour: int) -> int:
    return df.index[(df["point_id"] == pid) & (df["hour_utc"] == hour)][0]


def _set(df, pid, hour, col, value):
    df = df.astype({col: object}) if isinstance(value, str) else df
    df.loc[_ligne(df, pid, hour), col] = value
    return df


MUTATIONS = {
    "colonne_absente": lambda d: d.drop(columns="potential_temperature_k"),
    "valeur_manquante": lambda d: _set(d, 3, 0, "hour_utc", np.nan),
    "type_entier": lambda d: _set(d, 3, 0, "mask_i_obs", "un"),
    "type_reel": lambda d: _set(d, 3, 0, "pressure_pa", "1013 hPa"),
    "type_horodatage": lambda d: _set(d, 3, 0, "timestamp_utc", "07/01/2025 00:00"),
    "mask_i_obs_invalide": lambda d: _set(d, 3, 0, "mask_i_obs", 3),
    "source_non_declaree": lambda d: _set(d, 3, 0, "source_type", "STATION_X"),
    "point_id_inconnu": lambda d: _set(d, 3, 0, "point_id", 31),
    "source_incoherente": lambda d: _set(d, 26, 0, "source_type",
                                         "ANEO_PHYSICAL_SYNTHETIC_MULTITIME"),
    "doublon": lambda d: pd.concat([d, d.iloc[[-1]]]).reset_index(drop=True),
    "horodatage_decroissant": lambda d: pd.concat(
        [d[d["hour_utc"] == 3], d[d["hour_utc"] != 3]]).reset_index(drop=True),
    "pas_incompatible": lambda d: _set(d, 30, 9, "timestamp_utc", "2025-01-07 09:00:05"),
    "hour_utc_incoherent": lambda d: _set(d, 3, 6, "hour_utc", 7),
    "day_index_incoherent": lambda d: _set(d, 3, 6, "day_index", 5),
    "hors_plage": lambda d: _set(d, 3, 6, "wind_v_m_s", -75.0),
    "densite_incoherente": lambda d: _set(d, 3, 6, "air_density_kg_m3", 1.32),
}


@pytest.mark.parametrize("regle", sorted(MUTATIONS))
def test_une_regle(regle, brut, grille):
    r = valider_dataframe(MUTATIONS[regle](brut), grille, nom="test.csv")
    assert regle in _regles(r), r["erreurs"]
    gravite = _erreur(r, regle)["gravite"]
    # une anomalie de MESURE isolee (< 1 % des lignes) met en quarantaine ; le reste rejette
    assert r["verdict"] == ("quarantaine" if gravite == "mesure" else "rejet"), r["motifs"]


def test_jour_du_nom_de_fichier(brut, grille):
    ok = valider_dataframe(brut, grille, nom="aneo_observations_day_004_20250107_multitime.csv")
    assert ok["verdict"] == "conforme"
    faux = valider_dataframe(brut, grille,
                             nom="aneo_observations_day_004_20250108_multitime.csv")
    assert _erreur(faux, "jour_du_fichier")["nombre"] == len(brut)


def test_densite_temperature_virtuelle_et_tolerance(brut, grille):
    df = brut.copy()
    rho_gp, modes = densite_gaz_parfaits(df)
    moist = ((df["air_density_kg_m3"] - rho_gp) / rho_gp).abs()
    assert moist.max() < 1e-3 and modes["air_sec"] == 0
    dry = df.drop(columns=["water_vapor_pressure_pa"]).assign(relative_humidity_pct=np.nan)
    rho_dry, modes_dry = densite_gaz_parfaits(dry)
    assert modes_dry["air_sec"] == len(df)
    assert ((df["air_density_kg_m3"] - rho_dry) / rho_dry).abs().min() > moist.max()
    # sans pression de vapeur, l'humidite relative suffit (Magnus)
    _, modes_rh = densite_gaz_parfaits(df.drop(columns=["water_vapor_pressure_pa"]))
    assert modes_rh["humidite_relative"] == len(df)

    i = _ligne(df, 8, 3)
    for facteur, attendu in ((1.015, True), (1.025, False)):
        d = df.copy()
        d.loc[i, "air_density_kg_m3"] = round(rho_gp[i] * facteur, 5)
        r = valider_dataframe(d, grille)
        assert ("densite_incoherente" not in _regles(r)) is attendu, facteur


def test_valeurs_figees():
    vals = np.array([[1.0, 1.0, 1.0, 2.0],
                     [1.0, 2.0, 1.0, 2.0],
                     [np.nan, np.nan, np.nan, 1.0],
                     [3.0, 4.0, 4.0, 4.0]])
    f = valeurs_figees(vals, 3)
    assert f.tolist() == [[True, True, True, False], [False] * 4, [False] * 4,
                          [False, True, True, True]]
    assert not valeurs_figees(vals[:, :2], 3).any()            # trop peu d'instants


def test_station_muette_isolee_avertit_sans_bloquer(brut, grille):
    d = brut.drop(index=_ligne(brut, 10, 6))
    r = valider_dataframe(d, grille)
    assert r["verdict"] == "conforme"
    assert r["qualite"]["fraicheur"]["stations_muettes"]["ids"] == [10]
    assert r["qualite"]["fraicheur"]["absences"] == 1
    assert any("muette" in a for a in r["avertissements"])


def test_station_absente_du_fichier_listee(brut, grille):
    r = valider_dataframe(brut[brut["point_id"] != 12], grille)
    assert r["qualite"]["fraicheur"]["stations_absentes"]["ids"] == [12]
    assert any("absente" in a for a in r["avertissements"])


def test_seuils_des_interpolees(brut, grille):
    d = brut.copy()
    d.loc[d["point_id"].isin([1, 2, 3]), "mask_i_obs"] = 0       # (12 + 2) / 96 = 14,6 %
    r = valider_dataframe(d, grille)
    assert r["verdict"] == "conforme"
    assert any("interpolees" in a for a in r["avertissements"])
    d.loc[d["point_id"].isin(range(1, 9)), "mask_i_obs"] = 0     # (32 + 2) / 96 = 35 %
    r = valider_dataframe(d, grille)
    assert r["verdict"] == "quarantaine"


def test_score_de_qualite_faible_compte_comme_signale(brut, grille):
    d = brut.copy()
    d.loc[d["point_id"] == 5, "aneo_quality_score"] = 0.3
    q = valider_dataframe(d, grille)["qualite"]["interpolees_ou_signalees"]
    assert q["qualite_faible"] == 4


def test_trop_peu_de_stations_valides_rejette(brut, grille):
    d = brut.copy()
    d.loc[d["point_id"] <= 20, "mask_i_obs"] = 0
    r = valider_dataframe(d, grille)
    assert r["verdict"] == "rejet"
    assert any("reconstruction impossible" in m for m in r["motifs"])


def test_fichier_illisible_et_lignes_malformees(tmp_path, grille):
    vide = tmp_path / "vide.csv"
    vide.write_bytes(b"")
    r = valider_fichier(vide, grille)
    assert r["verdict"] == "rejet" and "fichier_illisible" in _regles(r)

    texte = (DATA / "conforme.csv").read_text(encoding="utf-8").splitlines()
    texte[5] += ",champ_en_trop"
    bad = tmp_path / "malforme.csv"
    bad.write_text("\n".join(texte) + "\n", encoding="utf-8")
    r = valider_fichier(bad, grille)
    assert r["verdict"] == "rejet"
    assert _erreur(r, "ligne_malformee")["exemples"][0].endswith(",champ_en_trop")


# ---------------------------------------------------------------------- quarantaine
def test_quarantaine_deplace_et_ne_supprime_rien(tmp_path, grille):
    src = tmp_path / "entree" / "corrompu.csv"
    src.parent.mkdir()
    shutil.copy2(DATA / "corrompu.csv", src)
    q = tmp_path / "quarantine"
    r = controler(src, grille, quarantaine=q)
    assert r["verdict"] == "rejet" and not src.exists()
    dest = q / "corrompu.csv"
    assert dest.read_bytes() == (DATA / "corrompu.csv").read_bytes()
    saved = json.loads(chemin_rapport(dest).read_text(encoding="utf-8"))
    assert saved["verdict"] == "rejet" and saved["quarantaine"]["mode"] == "deplacement"

    # meme nom une seconde fois, en copie : rien n'est ecrase, l'original reste en place
    shutil.copy2(DATA / "corrompu.csv", src)
    r2 = controler(src, grille, quarantaine=q, copier=True)
    assert src.exists() and dest.exists()
    assert Path(r2["quarantaine"]["fichier"]) == q / "corrompu_1.csv"
    assert chemin_rapport(q / "corrompu_1.csv").exists()


def test_fichier_conforme_reste_en_place(tmp_path, grille):
    src = tmp_path / "conforme.csv"
    shutil.copy2(DATA / "conforme.csv", src)
    r = controler(src, grille, quarantaine=tmp_path / "quarantine")
    assert r["verdict"] == "conforme" and src.exists()
    assert not (tmp_path / "quarantine").exists()


def test_quarantaine_fichier_compresse(tmp_path, grille):
    src = tmp_path / "x.csv.gz"
    src.write_bytes(b"pas du gzip")
    rapport = {"verdict": "rejet"}
    (tmp_path / "q").mkdir()
    (tmp_path / "q" / "x.csv.gz").write_bytes(b"deja la")
    dest = mettre_en_quarantaine(src, rapport, tmp_path / "q")
    assert dest.name == "x_1.csv.gz" and (tmp_path / "q" / "x.csv.gz").read_bytes() == b"deja la"


# ---------------------------------------------------------------------- CLI
def test_cli_validate(tmp_path, capsys):
    from pcfno3d.cli import main

    meta = str(DATA / "meta.json")
    assert main(["validate", str(DATA / "conforme.csv"), "--meta", meta]) == 0
    assert main(["validate", str(DATA / "panne.csv"), "--meta", meta]) == 1
    out = tmp_path / "rapport.json"
    code = main(["validate", str(DATA / "corrompu.csv"), "--meta", meta,
                 "--report", str(out)])
    assert code == 2
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "rejet"
    assert "[REJET]" in capsys.readouterr().out

    # plusieurs fichiers : le pire verdict fait le code ; la quarantaine copie
    q = tmp_path / "quarantine"
    code = main(["validate", str(DATA / "conforme.csv"), str(DATA / "panne.csv"),
                 "--meta", meta, "--quarantine", str(q), "--copier",
                 "--report", str(out)])
    assert code == 1
    assert (q / "panne.csv").exists() and (DATA / "panne.csv").exists()
    assert not (q / "conforme.csv").exists()
    assert [r["verdict"] for r in json.loads(out.read_text(encoding="utf-8"))] == \
        ["conforme", "quarantaine"]
