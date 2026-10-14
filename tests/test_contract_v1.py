"""Barriere de non-regression du contrat v1 (service PC-FNO3D -> tableau de bord).

Trois etages :
  1. les fixtures du contrat (copie de atom-dashboard/contract, cf. tests/contract/README.md)
     valident contre leurs schemas, et leurs binaires ont la taille declaree ;
  2. ce que le service produit AUJOURD'HUI (mini-dataset, petit modele, CPU), converti par
     `pcfno3d.contract_v1`, valide contre les schemas ;
  3. ce qu'il ne sait pas encore produire est en `xfail(strict=True)` avec le lot qui le
     livrera : le jour ou le lot rend la ressource conforme, le test passe, `strict` le fait
     echouer, et on retire le marqueur. La barriere se resserre lot apres lot.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

jsonschema = pytest.importorskip("jsonschema")
referencing = pytest.importorskip("referencing")

from jsonschema import Draft202012Validator  # noqa: E402
from referencing import Registry, Resource  # noqa: E402

from pcfno3d import contract_v1  # noqa: E402

CONTRACT = Path(__file__).resolve().parent / "contract"
SCHEMAS = CONTRACT / "schemas"
FIXTURES = CONTRACT / "fixtures"
SCENARIOS = ["normal", "capteur_en_derive", "gpu_eteint"]
FILE_SCHEMA = {
    "meta.json": "meta", "latest.json": "latest", "times.json": "times",
    "snapshot.json": "snapshot", "profile.json": "profile", "quality.json": "quality",
    "alerts.json": "alerts", "stations.json": "stations", "wind-farm.json": "wind-farm",
    "windrose.json": "windrose", "publish-manifest.json": "publish-manifest",
}
MAX_FILE_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------------- outils
def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _schemas() -> dict[str, dict]:
    return {p.name.removesuffix(".schema.json"): _load(p)
            for p in sorted(SCHEMAS.glob("*.schema.json"))}


SCHEMA = _schemas()
# les $ref entre schemas ("common.schema.json#/$defs/...") se resolvent par leur $id
REGISTRY = Registry().with_resources(
    (s["$id"], Resource.from_contents(s)) for s in SCHEMA.values())


def _validator(schema: dict) -> Draft202012Validator:
    return Draft202012Validator(schema, registry=REGISTRY,
                                format_checker=Draft202012Validator.FORMAT_CHECKER)


def _errors(name_or_schema, instance) -> list[str]:
    schema = SCHEMA[name_or_schema] if isinstance(name_or_schema, str) else name_or_schema
    return [f"{'/'.join(map(str, e.absolute_path)) or '/'} : {e.message}"
            for e in sorted(_validator(schema).iter_errors(instance), key=str)]


def assert_conforme(name_or_schema, instance) -> None:
    errs = _errors(name_or_schema, instance)
    assert not errs, "ecarts au contrat :\n  " + "\n  ".join(errs[:20])


def _prune_required(schema: dict, instance: dict, prefix: str = "") -> tuple[dict, list[str]]:
    """Copie du schema ou `required` ne garde que les cles presentes ; renvoie les retirees.

    Le reste (types, bornes, additionalProperties) s'applique toujours aux cles presentes.
    """
    out = copy.deepcopy(schema)
    removed = [prefix + k for k in out.get("required", []) if k not in instance]
    out["required"] = [k for k in out.get("required", []) if k in instance]
    for key, sub in out.get("properties", {}).items():
        if key in instance and isinstance(instance[key], dict) and "properties" in sub:
            out["properties"][key], more = _prune_required(sub, instance[key], f"{prefix}{key}.")
            removed += more
    return out, removed


def _f32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<f4")


# ---------------------------------------------------------------------------- 1. fixtures
def test_les_schemas_sont_du_json_schema_2020_12():
    assert set(SCHEMA) == {"common", *FILE_SCHEMA.values()}
    for name, s in SCHEMA.items():
        assert s["$schema"] == "https://json-schema.org/draft/2020-12/schema", name
        Draft202012Validator.check_schema(s)


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("fname", sorted(FILE_SCHEMA))
def test_fixture_conforme_a_son_schema(scenario, fname):
    assert_conforme(FILE_SCHEMA[fname], _load(FIXTURES / scenario / fname))


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_fixture_binaires_taille_et_relief(scenario):
    d = FIXTURES / scenario
    meta = _load(d / "meta.json")
    g = meta["grid"]
    nx, ny, nz = g["nx"], g["ny"], g["nz"]
    assert len(g["levels_asl_m"]) == nz
    expected = nx * ny * 4

    terrain = (d / "terrain.f32").read_bytes()
    assert len(terrain) == expected
    zt = _f32(terrain)
    assert np.isfinite(zt).all(), "le relief ne doit avoir aucun NaN"

    files = sorted((d / "fields").glob("*.f32"))
    assert len(files) == nz * len(meta["fields"])
    for k, z in enumerate(g["levels_asl_m"]):
        below = z <= zt
        for var in meta["fields"]:
            raw = (d / "fields" / f"{k}_{var}.f32").read_bytes()
            assert len(raw) == expected, f"{k}_{var}"
            a = _f32(raw)
            assert (np.isnan(a) == below).all(), f"{k}_{var} : NaN incoherent avec le relief"
        u, v, s = (_f32((d / "fields" / f"{k}_{n}.f32").read_bytes()) for n in ("u", "v", "speed"))
        m = np.isfinite(s)
        assert np.allclose(np.hypot(u[m], v[m]), s[m], atol=1e-3)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_fixture_manifeste_sha256_et_taille(scenario):
    d = FIXTURES / scenario
    for f in _load(d / "publish-manifest.json")["files"]:
        raw = (d / f["path"]).read_bytes()
        assert len(raw) == f["bytes"], f["path"]
        assert hashlib.sha256(raw).hexdigest() == f["sha256"], f["path"]


# ---------------------------------------------------------------------------- 2. service
GEOREF = {"origin_lat": 36.00, "origin_lon": 9.70, "rotation_deg": 0.0, "origin_assumed": True}


@pytest.fixture(scope="module")
def served(cfg, geom, index, norm):
    """Le service tel qu'il tourne : petit modele du mini-dataset, un instant de test."""
    from pcfno3d.inference import WindFieldModel
    from pcfno3d.train import Trainer

    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    wf = WindFieldModel(cfg, device="cpu", verbose=False)
    test = index[index["split"] == "test"]
    stem = str((test if len(test) else index)["stem"].iloc[0])
    volume = wf.predict_stem(stem)
    sha = contract_v1.file_sha256(wf.checkpoint_path)
    versions = contract_v1.to_versions(
        model=cfg.paths.run_dir.name, norm=contract_v1.file_sha256(cfg.paths.norm_stats)[:8],
        data=cfg.paths.data_dir.name, model_sha256=sha)
    return {"wf": wf, "stem": stem, "volume": volume, "versions": versions}


@pytest.fixture(scope="module")
def anchored_cfg(cfg):
    """Copie de la config avec l'ancrage SUPPOSE du domaine (36,00 N, 9,70 E)."""
    from dataclasses import replace

    c = copy.copy(cfg)
    c.real = replace(cfg.real, **GEOREF)
    return c


def test_meta_du_service_conforme(geom, anchored_cfg):
    meta = contract_v1.to_meta(geom, anchored_cfg)
    assert_conforme("meta", meta)
    assert meta["grid"]["nz"] == len(meta["grid"]["levels_asl_m"]) == geom.nz
    assert meta["georef"] == {"lat0": 36.0, "lon0": 9.7, "rotation_deg": 0.0, "suppose": True}
    assert meta["grid"]["valid_nodes"] == int(geom.valid.sum())


def test_meta_refuse_un_domaine_sans_ancrage(geom, cfg):
    from dataclasses import replace

    c = copy.copy(cfg)
    c.real = replace(cfg.real, origin_lat=None, origin_lon=None)
    with pytest.raises(ValueError, match="georef"):
        contract_v1.to_meta(geom, c)


def test_terrain_du_service_float32_le_j0_au_sud(cfg, geom):
    import pandas as pd

    raw = contract_v1.terrain_bytes(geom)
    assert len(raw) == geom.nx * geom.ny * 4
    zt = _f32(raw)
    assert np.isfinite(zt).all()
    # ordre et origine : la valeur a l'index j*nx + i est le relief du noeud (x=i*dx, y=j*dy)
    # tel que l'ecrit le generateur — y croissant vers le nord, donc j = 0 au sud
    grid = pd.read_csv(cfg.paths.data_dir / "aneo_grid_points.csv",
                       usecols=["point_id", "x_m", "y_m", "z_terrain_m"])
    surf = grid[grid["point_id"] < geom.node_id_start]
    lookup = {(round(x, 3), round(y, 3)): z
              for x, y, z in zip(surf["x_m"], surf["y_m"], surf["z_terrain_m"])}
    for j in range(geom.ny):
        for i in range(geom.nx):
            expected = lookup[(round(i * geom.dx, 3), round(j * geom.dy, 3))]
            assert zt[j * geom.nx + i] == pytest.approx(expected, abs=1e-3), (i, j)
    # little-endian explicite : relu en big-endian, le relief n'a plus de sens
    assert not np.allclose(np.frombuffer(raw, dtype=">f4"), zt)


def test_champs_du_service_float32_le_nan_hors_noeuds(served):
    volume = served["volume"]
    g = volume.geom
    files = contract_v1.fields_files(volume)
    assert len(files) == g.nz * len(contract_v1.FIELDS_TODAY)
    for k in range(g.nz):
        for var in contract_v1.FIELDS_TODAY:
            raw = files[f"fields/{k}_{var}.f32"]
            assert len(raw) == g.nx * g.ny * 4
            a = _f32(raw).reshape(g.ny, g.nx)
            assert (np.isnan(a) == ~g.valid[k]).all(), f"{k}_{var}"
            ref = {"u": volume.u, "v": volume.v, "w": volume.w, "speed": volume.speed}[var][k]
            np.testing.assert_array_equal(a[g.valid[k]], ref[g.valid[k]].astype(np.float32))
            assert not (a == -9999.0).any(), "-9999 est le format export_web, pas le contrat"
        u, v, s = (_f32(files[f"fields/{k}_{n}.f32"]) for n in ("u", "v", "speed"))
        m = np.isfinite(s)
        assert np.allclose(np.hypot(u[m], v[m]), s[m], atol=1e-3)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "relecture PR #1 (bloquant) : le contrat derive le NaN de `niveau <= relief`, le service "
    "du noeud absent du generateur, qui n'en cree aucun a moins de min_z_rel_m = 5 m du sol"))
def test_champs_du_service_nan_selon_la_regle_litterale_du_contrat(served, geom):
    # garde-fou : le mini-dataset doit contenir de tels noeuds, sinon ce test ne prouve rien
    agl = geom.z_levels[:, None, None] - geom.zt[None]
    if not ((agl > 0) & (agl < 5.0)).any():
        raise RuntimeError("mini-dataset sans noeud a moins de 5 m du sol : test sans objet")
    zt = _f32(contract_v1.terrain_bytes(geom))
    for k, z in enumerate(geom.z_levels):
        a = _f32(contract_v1.field_bytes(served["volume"], k, "u"))
        assert (np.isnan(a) == (z <= zt)).all()


def test_profil_du_service_conforme(served):
    volume = served["volume"]
    g = volume.geom
    for x, y in [(0.0, 0.0), (g.dx * 3.4, g.dy * 2.6), ((g.nx - 1) * g.dx, (g.ny - 1) * g.dy)]:
        prof = contract_v1.to_profile(volume, x, y)
        assert_conforme("profile", prof)
        assert len(prof["levels"]) == g.nz
        for lv in prof["levels"]:
            if lv["below_terrain"]:
                assert lv["u"] is None and lv["dir_from_deg"] is None
                continue
            assert lv["agl_m"] > 0
            d = (270.0 - np.degrees(np.arctan2(lv["v"], lv["u"]))) % 360.0
            if lv["dir_from_deg"] is not None:
                assert abs((d - lv["dir_from_deg"] + 540.0) % 360.0 - 180.0) < 1.5


def test_direction_meteo_et_calme():
    # vent d'ouest (souffle vers l'est) : vient de 270 ; vent du nord : vient de 0
    assert contract_v1.direction_from(5.0, 0.0) == 270.0
    assert contract_v1.direction_from(0.0, -5.0) == 0.0
    assert contract_v1.direction_from(0.0, 5.0) == 180.0
    assert contract_v1.direction_from(-5.0, 0.0) == 90.0
    assert contract_v1.direction_from(0.1, 0.1) is None          # < 0,2 m/s
    assert contract_v1.direction_from(float("nan"), 1.0) is None
    # un arrondi a 360,00 doit revenir a 0 (exclusiveMaximum: 360)
    assert contract_v1.direction_from(1e-6, -5.0) == 0.0


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="lot M2 : statut, couverture, fiabilite, intervalle p10-p90")
def test_profil_avec_fiabilite(served):
    prof = contract_v1.to_profile(served["volume"], 0.0, 0.0)
    assert prof["uncertainty"]["members"] > 1
    for lv in prof["levels"]:
        if not lv["below_terrain"]:
            assert lv["status"] in ("mesure", "estime")
            assert lv["coverage"] is not None and lv["reliability_class"] is not None
            assert None not in lv["speed_interval"].values()


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="lot M2 : champs sigma_u/v/w, coverage, reliability, status")
def test_meta_annonce_les_champs_de_fiabilite(geom, anchored_cfg):
    fields = set(contract_v1.to_meta(geom, anchored_cfg)["fields"])
    assert {"sigma_u", "sigma_v", "sigma_w", "coverage", "reliability", "status"} <= fields


def test_snapshot_partiel_du_service(served):
    snap = contract_v1.to_snapshot_partial(served["volume"], served["stem"], served["versions"])
    assert snap["snapshot_id"] == f"{served['stem']}_{served['versions']['model_sha256'][:8]}"
    pruned, missing = _prune_required(SCHEMA["snapshot"], snap)
    assert_conforme(pruned, snap)
    # exactement ce qui est annonce comme manquant : un lot qui ajoute une cle la retire
    # de SNAPSHOT_PENDING, et une cle perdue en route fait echouer ce test
    assert set(missing) == set(contract_v1.SNAPSHOT_PENDING)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="lots M2 + M3 + M4 + M5 : cles de contract_v1.SNAPSHOT_PENDING")
def test_snapshot_complet_du_service(served):
    snap = contract_v1.to_snapshot_partial(served["volume"], served["stem"], served["versions"])
    assert_conforme("snapshot", snap)


def test_manifeste_de_publication_du_service(served, geom, anchored_cfg):
    volume, versions = served["volume"], served["versions"]
    sid = contract_v1.snapshot_id(served["stem"], versions["model_sha256"])
    valid_time = contract_v1.iso_utc(volume.timestamp)
    files = {"meta.json": json.dumps(contract_v1.to_meta(geom, anchored_cfg)).encode(),
             "terrain.f32": contract_v1.terrain_bytes(geom),
             **contract_v1.fields_files(volume)}
    man = contract_v1.to_publish_manifest(sid, valid_time, versions, files,
                                          last_observation_at=valid_time)
    assert_conforme("publish-manifest", man)
    assert all(0 < f["bytes"] <= MAX_FILE_BYTES for f in man["files"])


def test_horodatage_et_identifiant():
    assert contract_v1.iso_utc("2025-01-06 12:00:00") == "2025-01-06T12:00:00Z"
    assert contract_v1.iso_utc("2025-01-06T13:00:00+01:00") == "2025-01-06T12:00:00Z"
    sha = "b2529442" + "0" * 56
    assert contract_v1.snapshot_id("017_120000", sha) == "017_120000_b2529442"
    with pytest.raises(ValueError):
        contract_v1.snapshot_id("017_120000", "b2529442")


# ---------------------------------------------------------------------------- 3. lots a venir
# Chaque ressource ci-dessous n'a pas encore d'adaptateur : l'appel leve AttributeError.
# Le lot qui l'ajoute fait passer le test, `strict` le signale, et on retire le marqueur.
PENDING_RESOURCES = [
    ("quality", "to_quality", "lot M2 : fiabilite agregee et calibration (DSO6)"),
    ("alerts", "to_alerts", "lot M4 : alertes (4 niveaux, JSONL)"),
    ("wind-farm", "to_wind_farm", "lot M5 : eolien (moyeu, REWS, puissance)"),
    ("stations", "to_stations", "lots M3 + M4 : stations muettes / derive ; liste reelle = tache 61"),
    ("windrose", "to_windrose", "lots M5 + M6 : Weibull sur l'historique publie"),
]


@pytest.mark.parametrize("schema_name,fn,lot", [
    pytest.param(s, f, lot, marks=pytest.mark.xfail(strict=True, raises=AttributeError,
                                                    reason=lot), id=s)
    for s, f, lot in PENDING_RESOURCES])
def test_ressource_a_venir(served, schema_name, fn, lot):
    produce = getattr(contract_v1, fn)
    assert_conforme(schema_name, produce(served["volume"]))


@pytest.mark.parametrize("schema_name", ["latest", "times"])
def test_ressource_du_worker(schema_name):
    pytest.skip(f"{schema_name} : produit par le Worker (lot F1) depuis R2, pas par le service")
