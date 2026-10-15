"""Lot M6 — publication au format du contrat ATOM rev. 2 (HIS1, DE3).

Instantanes du mini-dataset valides par les schemas Python (jsonschema, copie de
tests/data/atom_rev2/, branche agent/rayen-audit-atom/contrat-v1, commit b7b13b8) ET par
validate.mjs du contrat (node + ajv : node_modules du contrat trouve par la variable
ATOM_CONTRACT_NODE_MODULES ou dans ~/atom-verif/contract ; test ignore sinon). Protocole de
publication contre un Worker simule local (tests/worker_simule.py) avec coupures de
connexion : latest ne bascule jamais sur un instantane incomplet. Aucun appel reseau hors
127.0.0.1, aucun jeton reel.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from pcfno3d.config import ReliabilityCfg

REPO = Path(__file__).resolve().parents[1]
ATOM = REPO / "tests" / "data" / "atom_rev2"
DEMO = REPO / "configs" / "eolien" / "parc_demo.yaml"
GEN = "2025-01-12T12:00:02Z"
JETON = "jeton-de-test-" + "x" * 24          # faux jeton, uniquement pour le Worker simule
# ancrage suppose des configs Colab, pour la CLI et le rejeu
ANCRE = ["--set", "real.origin_lat=36.0", "--set", "real.origin_lon=9.7",
         "--set", "real.origin_assumed=true"]

FILE_SCHEMA = {"latest.json": "latest", "times.json": "times", "snapshot.json": "snapshot",
               "quality.json": "quality", "alerts.json": "alerts", "stations.json": "stations",
               "wind-farm.json": "wind-farm", "data-quality.json": "data-quality",
               "publish-manifest.json": "publish-manifest"}


# ------------------------------------------------------------------ outils
@pytest.fixture(scope="module")
def schemas():
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    docs = {p.name.replace(".schema.json", ""): json.loads(p.read_text(encoding="utf-8"))
            for p in ATOM.glob("*.schema.json")}
    reg = referencing.Registry().with_resources(
        [(d["$id"], referencing.Resource.from_contents(d)) for d in docs.values()])
    fmt = jsonschema.Draft202012Validator.FORMAT_CHECKER
    return {n: jsonschema.Draft202012Validator(d, registry=reg, format_checker=fmt)
            for n, d in docs.items()}


def valider_python(schemas, dossier: Path) -> list[str]:
    """Chaque JSON de l'instantane contre son schema (meme table que validate.mjs)."""
    erreurs = []

    def un(rel: str, nom: str) -> None:
        doc = json.loads((dossier / rel).read_bytes().decode("utf-8"))
        for e in schemas[nom].iter_errors(doc):
            erreurs.append(f"{rel} {list(e.absolute_path)} {e.message}")

    for rel, nom in FILE_SCHEMA.items():
        un(rel, nom)
    for p in sorted((dossier / "profiles").glob("*.json")):
        un(f"profiles/{p.name}", "profile")
    for p in sorted((dossier / "windrose").glob("*.json")):
        un(f"windrose/{p.name}", "windrose")
    for m in (dossier / "models").iterdir():
        un(f"models/{m.name}/meta.json", "meta")
        un(f"models/{m.name}/model-manifest.json", "model-manifest")
    return erreurs


def _node_modules() -> Path | None:
    cands = []
    if os.environ.get("ATOM_CONTRACT_NODE_MODULES"):
        cands.append(Path(os.environ["ATOM_CONTRACT_NODE_MODULES"]))
    cands.append(Path.home() / "atom-verif" / "contract" / "node_modules")
    for c in cands:
        if (c / "ajv" / "package.json").exists() and (c / "ajv-formats").exists():
            return c
    return None


@pytest.fixture(scope="module")
def validate_mjs(tmp_path_factory):
    """validate.mjs du contrat, dans un dossier temporaire : contract/{tools,schemas,
    node_modules} (jamais d'ecriture dans le depot du tableau de bord)."""
    if shutil.which("node") is None:
        pytest.skip("node absent")
    nm = _node_modules()
    if nm is None:
        pytest.skip("node_modules du contrat (ajv 8.17.1, ajv-formats 3.0.1) introuvable : "
                    "definir ATOM_CONTRACT_NODE_MODULES")
    root = tmp_path_factory.mktemp("contrat") / "contract"
    (root / "tools").mkdir(parents=True)
    shutil.copy(ATOM / "tools" / "validate.mjs", root / "tools" / "validate.mjs")
    shutil.copy(ATOM / "tools" / "package.json", root / "package.json")
    (root / "schemas").mkdir()
    for p in ATOM.glob("*.schema.json"):
        shutil.copy(p, root / "schemas" / p.name)
    shutil.copytree(nm, root / "node_modules")

    def lancer(*dossiers: Path) -> subprocess.CompletedProcess:
        return subprocess.run(["node", str(root / "tools" / "validate.mjs"),
                               *map(str, dossiers)], capture_output=True, timeout=300)
    return lancer


@pytest.fixture(scope="module")
def trained(cfg, index, norm):
    """Modele du mini-jeu, avec l'ancrage SUPPOSE des configs Colab (M6 refuse de publier
    un domaine sans ancrage) ; restaure a la fin du module."""
    from pcfno3d.train import Trainer
    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    avant = (cfg.real.origin_lat, cfg.real.origin_lon, cfg.real.origin_assumed)
    cfg.real.origin_lat, cfg.real.origin_lon, cfg.real.origin_assumed = 36.0, 9.7, True
    yield cfg
    cfg.real.origin_lat, cfg.real.origin_lon, cfg.real.origin_assumed = avant


@pytest.fixture(scope="module")
def parc(geom, tmp_path_factory) -> Path:
    """Meme parc que tests/test_m45.py : 3 eoliennes ou le rotor est encadre."""
    demo = yaml.safe_load(DEMO.read_text(encoding="utf-8"))
    h = np.asarray(geom.z_levels, float)[:, None, None] - geom.zt[None]
    hv = np.where(geom.valid, h, np.nan)
    ok = (np.nanmin(hv, axis=0) <= 39.0) & (np.nanmax(hv, axis=0) >= 81.0)
    jj, ii = np.nonzero(ok)
    pick = np.linspace(0, len(jj) - 1, 3).astype(int)
    data = {"exemple": True,
            "power_curves": {"generique-2MW": demo["power_curves"]["generique-2MW"]},
            "turbines": [{"id": f"T-0{n + 1}", "name": f"T-0{n + 1} (exemple)",
                          "x_m": float(ii[k] * geom.dx), "y_m": float(jj[k] * geom.dy),
                          "hub_agl_m": 60.0, "rotor_d_m": 40.0, "model": "generique-2MW"}
                         for n, k in enumerate(pick)]}
    path = tmp_path_factory.mktemp("parc") / "parc_mini.yaml"
    path.write_bytes(yaml.safe_dump(data, allow_unicode=True).encode("utf-8"))
    return path


def _instants(cfg, n: int = 3) -> list[str]:
    from pcfno3d.cache import load_index
    idx = load_index(cfg)
    return idx[idx["split"] == "test"].sort_values("timestamp")["stem"].tolist()[:n]


@pytest.fixture(scope="module")
def complet(trained, parc):
    """WindFieldModel avec M2 (non calibre, 3 membres) et M5 (parc), + moteur M4."""
    from pcfno3d.inference import WindFieldModel
    trained.wind_farm.turbines = parc
    trained.inference.reliability = "non_calibre"
    trained.reliability.members = 3
    try:
        wf = WindFieldModel(trained, device="cpu", verbose=False)
    finally:
        trained.wind_farm.turbines = None
        trained.inference.reliability = None
        trained.reliability.members = ReliabilityCfg().members
    return wf


@pytest.fixture(scope="module")
def nu(trained):
    """WindFieldModel sans aucun lot optionnel (M2, M4, M5 inactifs)."""
    from pcfno3d.inference import WindFieldModel
    return WindFieldModel(trained, device="cpu", verbose=False)


@pytest.fixture(scope="module")
def depot_complet(complet, trained, tmp_path_factory):
    """3 instants de test publies en local, dans l'ordre, tous lots actifs."""
    from pcfno3d.alerts import MoteurAlertes
    from pcfno3d.publish import DepotLocal, instantane_du_cache
    racine = tmp_path_factory.mktemp("pub_complet")
    depot = DepotLocal(racine)
    moteur = MoteurAlertes(trained, journal=racine / "alertes.jsonl")
    insts = []
    for stem in _instants(trained, 3):
        inst = instantane_du_cache(complet, stem, data_kind="test", depot=depot, moteur=moteur,
                                   generated_at=GEN, garder_champs=True)
        depot.ecrire(inst)
        insts.append(inst)
    return depot, insts


# ------------------------------------------------------------------ construction
def test_instantanes_conformes_aux_schemas(depot_complet, schemas):
    depot, insts = depot_complet
    for inst in insts:
        d = depot.dossier(inst.snapshot_id)
        assert valider_python(schemas, d) == []
        snap = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))
        assert snap["snapshot_id"] == inst.snapshot_id
        assert snap["snapshot_id"].endswith("_" + inst.versions["model_sha256"][:8])
        assert inst.versions["data_kind"] == "test"
        # M2, M4, M5 actifs : aucun bloc null
        assert snap["physics"] is not None and snap["stations_summary"] is not None
        assert snap["kpi"]["open_alerts"] is not None
        assert snap["kpi"]["share_high_reliability"] is not None
        assert [p["id"] for p in snap["points"]] == ["T-01", "T-02", "T-03"]
        assert all(p["profile"] for p in snap["points"])
        q = json.loads((d / "quality.json").read_text(encoding="utf-8"))
        assert q["uncertainty"]["method"] == "ensemble-masquage-v1"
        meta = json.loads((d / "models" / inst.sha8 / "meta.json").read_text(encoding="utf-8"))
        assert "sigma_speed" in meta["fields"] and "status" in meta["fields"]
        # le manifeste ne contient pas les fichiers du modele (D3)
        chemins = {f["path"] for f in inst.manifeste["files"]}
        assert not chemins & {"meta.json", "terrain.f32"}


def test_regle_b1_nan_sous_min_agl(depot_complet, geom):
    """NaN (f32) / 255 (u8) <=> z_asl - relief < min_agl_m, recalcule depuis terrain.f32."""
    depot, insts = depot_complet
    inst = insts[0]
    d = depot.dossier(inst.snapshot_id)
    meta = json.loads((d / "models" / inst.sha8 / "meta.json").read_text(encoding="utf-8"))
    g = meta["grid"]
    terr = np.fromfile(d / "models" / inst.sha8 / "terrain.f32", "<f4").reshape(g["ny"], g["nx"])
    z = np.asarray(g["levels_asl_m"], float)[:, None, None]
    sous = (z - terr.astype(float)[None]) < g["min_agl_m"]
    assert g["valid_nodes"] == int((~sous).sum())
    for v in meta["fields"]:
        enc = meta["encoding"][v]
        dt = "<f4" if enc["dtype"] == "float32" else "u1"
        a = np.fromfile(d / enc["file"], dt).reshape(g["nz"], g["ny"], g["nx"])
        absent = np.isnan(a) if dt == "<f4" else a == 255
        np.testing.assert_array_equal(absent, sous, err_msg=v)
    u = np.fromfile(d / "fields/u.f32", "<f4").reshape(sous.shape)
    vv = np.fromfile(d / "fields/v.f32", "<f4").reshape(sous.shape)
    s = np.fromfile(d / "fields/speed.f32", "<f4").reshape(sous.shape)
    assert np.nanmax(np.abs(np.hypot(u, vv) - s)) < 1e-3
    # le masque publie est celui du generateur sur le mini-dataset (aucun noeud retire)
    assert inst.info["noeuds_retires_min_agl"] == 0
    np.testing.assert_array_equal(~sous, geom.valid)


def test_validate_mjs_du_contrat(depot_complet, validate_mjs):
    """Les 3 instantanes du mini-jeu passent validate.mjs (rev. 2) : schemas, sha256 des deux
    manifestes, NaN <=> min_agl_m, profils, roses, open_alerts, reseau."""
    depot, insts = depot_complet
    r = validate_mjs(*[depot.dossier(i.snapshot_id) for i in insts])
    sortie = (r.stdout + r.stderr).decode("utf-8", "replace")
    print(sortie.encode("ascii", "backslashreplace").decode("ascii"))   # console cp1252
    assert r.returncode == 0, sortie
    assert "tous les instantan" in sortie


def test_validate_mjs_detecte_un_defaut(depot_complet, validate_mjs, tmp_path):
    """Controle negatif : un noeud valide mis a NaN (et le manifeste non mis a jour) est
    refuse par validate.mjs — le test precedent ne passe pas par defaut."""
    depot, insts = depot_complet
    inst = insts[0]
    copie = tmp_path / inst.snapshot_id
    shutil.copytree(depot.dossier(inst.snapshot_id), copie)
    u = np.fromfile(copie / "fields" / "u.f32", "<f4")
    u[int(np.nonzero(np.isfinite(u))[0][0])] = np.nan
    (copie / "fields" / "u.f32").write_bytes(u.astype("<f4").tobytes())
    r = validate_mjs(copie)
    sortie = (r.stdout + r.stderr).decode("utf-8", "replace")
    assert r.returncode == 1, sortie
    assert "sha256" in sortie and "NaN" in sortie


def test_sans_lots_optionnels(nu, trained, tmp_path, schemas, validate_mjs):
    """M2, M4, M5 inactifs : uncertainty « aucune », blocs null (B5), seuls u, v, w, speed,
    pas de profil ; toujours conforme (schemas + validate.mjs)."""
    from pcfno3d.publish import DepotLocal, instantane_du_cache
    depot = DepotLocal(tmp_path / "pub")
    inst = instantane_du_cache(nu, _instants(trained, 1)[0], data_kind="test", depot=depot,
                               generated_at=GEN)
    d = depot.ecrire(inst)
    assert valider_python(schemas, d) == []
    snap = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))
    assert snap["physics"] is None and snap["stations_summary"] is None
    assert snap["kpi"]["open_alerts"] is None and snap["kpi"]["share_high_reliability"] is None
    assert snap["points"] == [] and snap["assets"]["profile_template"] is None
    q = json.loads((d / "quality.json").read_text(encoding="utf-8"))
    assert q["uncertainty"] == {"method": "aucune", "members": 1, "calibrated": False}
    assert q["calibration"] is None and all(b["share_elevee"] is None for b in q["by_level"])
    meta = json.loads((d / "models" / inst.sha8 / "meta.json").read_text(encoding="utf-8"))
    assert meta["fields"] == ["u", "v", "w", "speed"]
    r = validate_mjs(d)
    assert r.returncode == 0, (r.stdout + r.stderr).decode("utf-8", "replace")


def test_historique_et_latest(depot_complet):
    """index/times.json du plus recent au plus ancien ; latest = le plus recent ; un
    instantane plus ancien ecrit apres ne fait pas reculer latest ; un instantane est
    immuable ; roses des vents cumulees sur l'historique, jamais sur le futur."""
    depot, insts = depot_complet
    items = depot.times()
    assert [t["snapshot_id"] for t in items] == [i.snapshot_id for i in reversed(insts)]
    assert depot.latest()["snapshot_id"] == insts[-1].snapshot_id
    with pytest.raises(FileExistsError):
        depot.ecrire(insts[0])
    n_valid = []
    for inst in insts:
        rose = json.loads(inst.fichiers["windrose/T-01.json"].decode("utf-8"))
        n_valid.append(rose["n_valid"])
        assert rose["period"]["to"] == inst.valid_time
    assert n_valid == [1, 2, 3]
    # times.json du dossier : l'historique au moment de l'instantane
    t0 = json.loads((depot.dossier(insts[0].snapshot_id) / "times.json").read_text("utf-8"))
    assert t0["total"] == 1


def test_direct_depuis_un_csv(complet, trained, tmp_path, schemas):
    """CSV de mesures (chemin de l'upload, contrat M3) : stem live_AAAAMMJJ_HHMMSS,
    data-quality.json tire du rapport du contrat (D12), conforme aux schemas."""
    from pcfno3d.alerts import MoteurAlertes
    from pcfno3d.publish import DepotLocal, instantane_du_csv
    csv = sorted(Path(trained.paths.data_dir).glob("aneo_observations_day_*"))[-1]
    depot = DepotLocal(tmp_path / "direct")
    inst = instantane_du_csv(complet, csv, data_kind="test", depot=depot,
                             moteur=MoteurAlertes(trained, journal=tmp_path / "j.jsonl"),
                             generated_at=GEN)
    assert inst.snapshot_id.startswith("live_") and inst.snapshot_id.endswith(inst.sha8)
    d = depot.ecrire(inst)
    assert valider_python(schemas, d) == []
    dq = json.loads((d / "data-quality.json").read_text(encoding="utf-8"))
    assert dq["files"]["received"] == 1
    cap = inst.info["capteurs"]                          # comptage exact, capteurs a 50 m
    assert cap["estime"] is False and 0 < cap["utilises"] <= cap["presents"]
    net = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))["network"]
    assert net["sensors_visible"] == cap["utilises"]
    assert dq["period"]["from"] <= inst.valid_time <= dq["period"]["to"]


def test_ecriture_locale_interrompue(nu, trained, tmp_path, monkeypatch):
    """HIS1 en local : une ecriture qui echoue au milieu ne laisse ni dossier d'instantane,
    ni dossier temporaire, et latest.json reste sur l'instantane precedent."""
    from pcfno3d.publish import DepotLocal, instantane_du_cache
    depot = DepotLocal(tmp_path / "pub")
    a, b = _instants(trained, 2)
    depot.ecrire(instantane_du_cache(nu, a, data_kind="test", depot=depot, generated_at=GEN))
    inst = instantane_du_cache(nu, b, data_kind="test", depot=depot, generated_at=GEN)
    vrai = Path.write_bytes
    n = {"k": 0}

    def panne(self, data):
        n["k"] += 1
        if n["k"] == 4:
            raise OSError("disque plein (simule)")
        return vrai(self, data)

    monkeypatch.setattr(Path, "write_bytes", panne)
    with pytest.raises(OSError):
        depot.ecrire(inst)
    monkeypatch.setattr(Path, "write_bytes", vrai)
    assert not depot.dossier(inst.snapshot_id).exists()
    assert [p.name for p in depot.snapshots.iterdir()] == [depot.latest()["snapshot_id"]]
    assert depot.latest()["snapshot_id"].startswith(a) and len(depot.times()) == 1
    depot.ecrire(inst)
    assert depot.latest()["snapshot_id"] == inst.snapshot_id


def test_latest_ne_recule_pas(nu, trained, tmp_path):
    from pcfno3d.publish import DepotLocal, instantane_du_cache
    depot = DepotLocal(tmp_path / "pub")
    a, b = _instants(trained, 2)
    depot.ecrire(instantane_du_cache(nu, b, data_kind="test", depot=depot, generated_at=GEN))
    depot.ecrire(instantane_du_cache(nu, a, data_kind="test", depot=depot, generated_at=GEN))
    assert depot.latest()["snapshot_id"].startswith(b)
    assert len(depot.times()) == 2


# ------------------------------------------------------------------ publication HTTP
@pytest.fixture
def serveur():
    sys.path.insert(0, str(Path(__file__).parent))
    from worker_simule import ServeurSimule
    with ServeurSimule(JETON) as s:
        yield s


def _client(serveur, **kw):
    from pcfno3d.publish import ClientPublication
    kw.setdefault("pause_s", 0.0)
    return ClientPublication(serveur.url, token=JETON, journal=None, **kw)


def test_publication_complete(serveur, depot_complet):
    depot, insts = depot_complet
    c = _client(serveur)
    for inst in insts:
        rep = c.publier(depot.dossier(inst.snapshot_id))
        assert rep["ok"] and rep["snapshot_id"] == inst.snapshot_id
    w = serveur.worker
    assert w.r2.json("latest.json")["snapshot_id"] == insts[-1].snapshot_id
    times = w.r2.json("index/times.json")
    assert [t["snapshot_id"] for t in times] == [i.snapshot_id for i in reversed(insts)]
    for t, inst in zip(reversed(times), insts):
        assert t["open_alerts"] == inst.open_alerts
    # fichiers du modele publies UNE fois (un seul commit de modele)
    commits_modele = [j for j in w.journal if j[1].endswith("/commit") and "/model/" in j[1]]
    assert len(commits_modele) == 1
    assert w.heartbeats == len(insts)
    # republier le meme instantane : idempotent, latest inchange
    rep = c.publier(depot.dossier(insts[0].snapshot_id))
    assert rep.get("unchanged") is True
    assert w.r2.json("latest.json")["snapshot_id"] == insts[-1].snapshot_id


def test_coupure_pendant_les_fichiers_ne_casse_pas_latest(serveur, depot_complet):
    """HIS1 : connexion coupee au milieu des PUT de l'instantane 2 -> echec, latest reste
    sur l'instantane 1 ; relancer la meme publication la termine."""
    from pcfno3d.publish import PublicationRefusee
    depot, insts = depot_complet
    c = _client(serveur, tentatives=1)
    c.publier(depot.dossier(insts[0].snapshot_id))
    w = serveur.worker
    sid2 = insts[1].snapshot_id
    w.couper(lambda m, p: m == "PUT" and f"/{sid2}/files/" in p and "alerts.json" in p)
    with pytest.raises(PublicationRefusee):
        c.publier(depot.dossier(sid2))
    assert w.r2.json("latest.json")["snapshot_id"] == insts[0].snapshot_id
    assert f"snapshots/{sid2}/_commit.json" not in w.r2.objets
    assert [t["snapshot_id"] for t in w.r2.json("index/times.json")] == [insts[0].snapshot_id]
    c.publier(depot.dossier(sid2))
    assert w.r2.json("latest.json")["snapshot_id"] == sid2


def test_coupure_au_commit(serveur, depot_complet):
    """Commit coupe AVANT traitement : rien ne bascule. Commit entierement traite (index et
    latest ecrits) mais reponse perdue : la reprise recoit « unchanged ». Ce n'est PAS
    « unchanged » qui prouve la publication (le Worker actuel le repond aussi quand il a
    plante apres _commit.json, voir test_echec_apres_commit_*) : c'est la relecture de
    /v1/times et /v1/latest faite par le client."""
    from pcfno3d.publish import PublicationRefusee
    depot, insts = depot_complet
    w = serveur.worker
    c = _client(serveur, tentatives=1)
    c.publier(depot.dossier(insts[0].snapshot_id))
    sid2 = insts[1].snapshot_id
    w.couper(lambda m, p: m == "POST" and p.endswith(f"/{sid2}/commit"))
    with pytest.raises(PublicationRefusee):
        c.publier(depot.dossier(sid2))
    assert w.r2.json("latest.json")["snapshot_id"] == insts[0].snapshot_id
    # reprises automatiques : la reponse du commit est perdue une fois
    c3 = _client(serveur, tentatives=3)
    c3.modeles_commites.add(insts[0].sha8)
    w.couper(lambda m, p: m == "POST" and p.endswith(f"/{sid2}/commit"), apres_traitement=True)
    rep = c3.publier(depot.dossier(sid2))
    assert rep.get("unchanged") is True
    assert w.r2.json("latest.json")["snapshot_id"] == sid2


def test_fichier_altere_refuse(serveur, depot_complet, tmp_path):
    """Un fichier modifie apres construction n'est jamais envoye ; un manifeste dont le
    sha256 ne correspond pas au depot est refuse par le Worker : latest inchange."""
    from pcfno3d.publish import PublicationRefusee
    depot, insts = depot_complet
    c = _client(serveur, tentatives=1)
    c.publier(depot.dossier(insts[0].snapshot_id))
    copie = tmp_path / insts[1].snapshot_id
    shutil.copytree(depot.dossier(insts[1].snapshot_id), copie)
    p = copie / "fields" / "u.f32"
    b = bytearray(p.read_bytes())
    b[0] ^= 0xFF
    p.write_bytes(bytes(b))
    with pytest.raises(PublicationRefusee, match="different du manifeste"):
        c.publier(copie)
    # commit d'un manifeste qui annonce un fichier jamais depose : refuse
    man = json.loads((copie / "publish-manifest.json").read_text(encoding="utf-8"))
    with pytest.raises(PublicationRefusee) as e:
        c._commit(f"/v1/publish/{man['snapshot_id']}", man)
    assert e.value.status == 422
    assert serveur.worker.r2.json("latest.json")["snapshot_id"] == insts[0].snapshot_id


def test_jeton_jamais_expose(serveur, depot_complet, capsys, monkeypatch):
    """Le jeton n'apparait ni dans le repr, ni dans les messages, ni dans les erreurs ;
    http vers un hote distant et instantane de test vers la production sont refuses AVANT
    toute requete ; jeton absent = erreur explicite."""
    from pcfno3d.publish import ClientPublication, PublicationRefusee
    depot, insts = depot_complet
    c = ClientPublication(serveur.url, token=JETON, pause_s=0.0, tentatives=1)
    c.publier(depot.dossier(insts[0].snapshot_id))
    mauvais = ClientPublication(serveur.url, token="autre-jeton", pause_s=0.0, tentatives=1)
    with pytest.raises(PublicationRefusee) as e:
        mauvais.publier(depot.dossier(insts[1].snapshot_id))
    sortie = capsys.readouterr()
    for texte in (repr(c), str(e.value), sortie.out, sortie.err):
        assert JETON not in texte and "autre-jeton" not in texte
    with pytest.raises(ValueError, match="https"):
        ClientPublication("http://atom.zelqane.com", token=JETON)
    # data_kind « test » : seulement vers un hote de bouclage ; hote normalise
    for url in ("https://atom.zelqane.com", "https://ATOM.Zelqane.com.",
                "https://exemple.invalid"):
        prod = ClientPublication(url, token=JETON, tentatives=1)
        with pytest.raises(PublicationRefusee, match="bouclage"):
            prod.publier(depot.dossier(insts[0].snapshot_id))
        assert prod.requetes == 0
    assert ClientPublication("http://localhost:1", token=JETON).local
    assert ClientPublication("http://[::1]:1", token=JETON).local
    monkeypatch.delenv("ATOM_PUBLISH_TOKEN", raising=False)
    with pytest.raises(ValueError, match="ATOM_PUBLISH_TOKEN"):
        ClientPublication(serveur.url)


def test_meta_immuable_par_modele(serveur, depot_complet, nu, trained, tmp_path):
    """Meme poids, autres champs (M2 inactif) : la meta publiee differe -> refus avant tout
    depot (le modele est immuable)."""
    from pcfno3d.publish import DepotLocal, PublicationRefusee, instantane_du_cache
    depot, insts = depot_complet
    _client(serveur).publier(depot.dossier(insts[0].snapshot_id))
    autre = DepotLocal(tmp_path / "nu")
    stem = _instants(trained, 3)[2]
    d = autre.ecrire(instantane_du_cache(nu, stem, data_kind="test", generated_at=GEN))
    with pytest.raises(PublicationRefusee, match="meta du modele"):
        _client(serveur).publier(d)


# ------------------------------------------------------------------ exports DE3
@pytest.mark.parametrize("moteur", ["netcdf4", "scipy"])
def test_export_netcdf(depot_complet, geom, tmp_path, moteur):
    if moteur == "netcdf4":
        pytest.importorskip("netCDF4")
    from pcfno3d.export_noeuds import _lire_netcdf, exporter_netcdf
    _, insts = depot_complet
    inst = insts[0]
    p = exporter_netcdf(inst, geom, tmp_path / f"{moteur}.nc", moteur=moteur)
    dims, attrs, variables = _lire_netcdf(p)
    assert (dims["altitude"], dims["y"], dims["x"]) == geom.shape
    assert variables["u"][0] == ("altitude", "y", "x")
    assert variables["u"][1]["units"] == "m s-1"
    assert variables["lat"][1]["units"] == "degrees_north"
    assert attrs["model_sha256"] == inst.versions["model_sha256"]
    assert attrs["version_normalisation"] == inst.versions["norm"]
    assert attrs["projection"] == "equirectangulaire_tangent_noeud_0_0"
    assert "statut" in variables and "sigma_u" in variables
    # CF : temps, grid_mapping, coordonnees de projection, ancrage suppose dit sur lat/lon
    assert {"time", "crs"} <= set(variables)
    assert variables["u"][1]["grid_mapping"] == "crs"
    assert variables["u"][1]["coordinates"] == "time lat lon"
    assert variables["x"][1]["standard_name"] == "projection_x_coordinate"
    assert variables["time"][1]["standard_name"] == "time"
    assert "SUPPOSEE" in variables["lat"][1]["long_name"]
    xr = pytest.importorskip("xarray")
    with xr.open_dataset(p) as ds:
        assert str(ds["time"].values)[:19] == inst.valid_time[:19]
        assert set(ds["u"].dims) == {"altitude", "y", "x"}
    # memes valeurs que fields/u.f32
    if moteur == "netcdf4":
        import netCDF4
        with netCDF4.Dataset(p) as ds:
            u = np.ma.filled(ds["u"][:], np.nan)
            lat = ds["lat"][:]
    else:
        from scipy.io import netcdf_file
        with netcdf_file(p, "r", mmap=False) as ds:
            u = ds.variables["u"][:].copy()
            lat = ds.variables["lat"][:].copy()
    np.testing.assert_array_equal(u, inst.champs.f32["u"])
    meta = json.loads(inst.modele["meta.json"].decode("utf-8"))
    assert lat.min() == pytest.approx(meta["georef"]["bounds"]["lat_min"], abs=1e-6)
    assert lat.max() == pytest.approx(meta["georef"]["bounds"]["lat_max"], abs=1e-6)


def test_export_csv_noeuds_virtuels(depot_complet, geom, tmp_path):
    import pandas as pd

    from pcfno3d.export_noeuds import COLONNES_CSV, exporter_csv
    _, insts = depot_complet
    inst = insts[0]
    p = exporter_csv(inst, geom, tmp_path / "noeuds.csv")
    assert b"\r\n" not in p.read_bytes()[:10000]
    df = pd.read_csv(p, keep_default_na=False, na_values=[""])
    assert tuple(df.columns) == COLONNES_CSV
    assert len(df) == int(inst.champs.valide.sum())
    assert set(df["statut"]) <= {"mesur\u00e9", "estim\u00e9"}
    assert set(df["classe_fiabilite"]) <= {"faible", "moyenne", "\u00e9lev\u00e9e"}
    assert (df["version_modele"] == f"test@{inst.sha8}").all()
    assert (df["version_normalisation"] == inst.versions["norm"]).all()
    assert (df["timestamp_utc"] == inst.valid_time).all()
    assert df["hauteur_sol_m"].min() >= 5.0 - 1e-6
    mes = df[df["statut"] == "mesur\u00e9"]
    assert mes["source"].str.match(r"^c\d+_\d+$").all()
    # meme valeur que le champ publie, au noeud
    r = df.iloc[0]
    i, j = int(round(r["x_m"] / geom.dx)), int(round(r["y_m"] / geom.dy))
    k = int(np.nonzero(np.asarray(geom.z_levels) == r["altitude_m"])[0][0])
    assert r["u"] == pytest.approx(float(inst.champs.f32["u"][k, j, i]), abs=6e-4)


def test_csv_refuse_par_le_schema_de_sortie(depot_complet, geom):
    """Le schema de sortie bloque une valeur hors plage avant livraison."""
    from pcfno3d.export_noeuds import noeuds_dataframe, schema_csv
    _, insts = depot_complet
    df = noeuds_dataframe(insts[0], geom)
    df.loc[df.index[0], "couverture"] = 1.5
    with pytest.raises(Exception, match="couverture"):
        schema_csv().validate(df, lazy=True)


# ------------------------------------------------------------------ CLI et rejeu
def test_cli_publish_local_puis_distant(trained, serveur, tmp_path, monkeypatch, capsys):
    from pcfno3d.cli import main
    stems = _instants(trained, 2)
    racine = tmp_path / "cli"
    base = ["--data-dir", str(trained.paths.data_dir), "--cache-dir",
            str(trained.paths.cache_dir), "--run-dir", str(trained.paths.run_dir), "-c",
            "test", *ANCRE]
    rc = main(base + ["publish", "--stem", stems[0], "--sortie", str(racine),
                      "--data-kind", "test", "--device", "cpu", "--exports"])
    assert rc == 0
    sid = json.loads((racine / "latest.json").read_text(encoding="utf-8"))["snapshot_id"]
    assert (racine / "exports" / sid / f"{sid}.nc").exists()
    assert (racine / "exports" / sid / f"{sid}_noeuds.csv").exists()
    monkeypatch.setenv("ATOM_PUBLISH_TOKEN", JETON)
    rc = main(base + ["publish", "--stem", stems[1], "--sortie", str(racine),
                      "--data-kind", "test", "--device", "cpu", "--remote", serveur.url])
    assert rc == 0
    assert serveur.worker.r2.json("latest.json")["snapshot_id"].startswith(stems[1])
    sortie = capsys.readouterr()
    assert JETON not in sortie.out + sortie.err


def test_rejeu_local(trained, tmp_path, monkeypatch, capsys):
    """scripts/m6_rejeu_colab.py --local sur 2 instants, puis reprise sans doublon."""
    sys.path.insert(0, str(REPO / "scripts"))
    import m6_rejeu_colab as rejeu
    monkeypatch.delenv("ATOM_PUBLISH_TOKEN", raising=False)
    args = ["-c", "test", "--data-dir", str(trained.paths.data_dir), "--cache-dir",
            str(trained.paths.cache_dir), "--run-dir", str(trained.paths.run_dir),
            "--sortie", str(tmp_path / "rejeu"), "--limite", "2", "--device", "cpu",
            "--data-kind", "test", *ANCRE]
    assert rejeu.main(args) == 0
    times = json.loads((tmp_path / "rejeu" / "index" / "times.json").read_text("utf-8"))
    assert times["total"] == 2
    assert rejeu.main(args + ["--limite", "3"]) == 0          # reprise : 1 seul nouveau
    times = json.loads((tmp_path / "rejeu" / "index" / "times.json").read_text("utf-8"))
    assert times["total"] == 3
    assert "deja publie" in capsys.readouterr().out


# ------------------------------------------------------------------ corrections de relecture
def test_sans_ancrage_refuse(nu, trained):
    """M6 n'invente pas de position : sans real.origin_lat/lon, aucune publication."""
    from pcfno3d.publish import instantane_du_cache
    avant = trained.real.origin_lat
    trained.real.origin_lat = None
    try:
        with pytest.raises(ValueError, match="ancrage"):
            instantane_du_cache(nu, _instants(trained, 1)[0], data_kind="test",
                                generated_at=GEN)
    finally:
        trained.real.origin_lat = avant


def test_open_alerts_null_sans_m4(nu, trained, tmp_path, schemas):
    """M4 inactif : open_alerts null dans times.json (local et dossier), jamais 0."""
    from pcfno3d.publish import DepotLocal, instantane_du_cache
    depot = DepotLocal(tmp_path / "pub")
    inst = instantane_du_cache(nu, _instants(trained, 1)[0], data_kind="test", depot=depot,
                               generated_at=GEN)
    assert inst.open_alerts is None
    d = depot.ecrire(inst)
    assert depot.times()[0]["open_alerts"] is None
    t = json.loads((d / "times.json").read_text(encoding="utf-8"))
    assert t["items"][0]["open_alerts"] is None
    assert valider_python(schemas, d) == []


def test_reseau_en_capteurs_50m(depot_complet, complet, geom):
    """B4 : total, visibles et perdus comptes en capteurs du treillis a 50 m (meme unite) ;
    depuis le cache, ESTIMATION round(fraction x stride^2) par maille, dite comme telle."""
    depot, insts = depot_complet
    inst = insts[0]
    d = depot.dossier(inst.snapshot_id)
    S, *_ = complet.surface_from_cache(inst.snapshot_id[:-9])
    k = geom.stride ** 2
    attendu = int(np.rint(np.clip(S[7], 0, 1) * k).sum())
    net = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))["network"]
    st = json.loads((d / "stations.json").read_text(encoding="utf-8"))["items"]
    assert net["sensors_visible"] == attendu and attendu > int((S[7] > 0.5).sum())
    perdus = sum(s["state"] in ("muette", "exclue") for s in st)
    assert net["sensors_total"] - net["sensors_visible"] == perdus
    assert inst.info["capteurs"]["estime"] is True
    dq = json.loads((d / "data-quality.json").read_text(encoding="utf-8"))
    assert dq["sensors"]["total"] == net["sensors_total"]
    assert dq["sensors"]["observed"] == net["sensors_visible"]


def test_stations_doc_identifiant_hors_treillis(geom):
    """Hauteur inconnue (identifiant non c<i>_<j>) : null (B5), jamais 0."""
    from pcfno3d.publish import stations_doc
    items = [{"status": "ouverte", "station_id": "ANEO-12", "kind": "station_muette",
              "location": {"x_m": 0.0, "y_m": 0.0}, "value": None}]
    out = stations_doc(items, geom, np.zeros((8, geom.ny, geom.nx), np.float32), GEN)
    assert out[0]["height_agl_m"] is None


def test_echec_apres_commit_worker_actuel(serveur, depot_complet):
    """Worker ACTUEL plante apres _commit.json : la reprise du commit repond « unchanged »
    sans rien reparer. Le client relit /v1/times et /v1/latest, rejoue le commit une fois,
    puis leve PublicationIncomplete (jamais un succes)."""
    from pcfno3d.publish import PublicationIncomplete
    depot, insts = depot_complet
    w = serveur.worker
    c = _client(serveur)
    c.publier(depot.dossier(insts[0].snapshot_id))
    sid2 = insts[1].snapshot_id
    w.echecs_apres_commit = 1
    with pytest.raises(PublicationIncomplete, match="Worker"):
        c.publier(depot.dossier(sid2))
    assert f"snapshots/{sid2}/_commit.json" in w.r2.objets
    assert w.r2.json("latest.json")["snapshot_id"] == insts[0].snapshot_id
    assert sid2 not in {t["snapshot_id"] for t in w.r2.json("index/times.json")}
    commits = [j for j in w.journal if j[1].endswith(f"/{sid2}/commit")]
    assert [j[2] for j in commits] == [500, 200, 200]          # echec, reprise, rejeu


def test_echec_apres_commit_worker_corrige(serveur, depot_complet):
    """Worker CORRIGE : la relance de la publication rejoue le commit, qui refait l'index
    et latest ; le client le verifie par relecture."""
    from pcfno3d.publish import PublicationRefusee
    depot, insts = depot_complet
    w = serveur.worker
    w.corrige = True
    c = _client(serveur, tentatives=1)
    c.publier(depot.dossier(insts[0].snapshot_id))
    sid2 = insts[1].snapshot_id
    w.echecs_apres_commit = 1
    with pytest.raises(PublicationRefusee):
        c.publier(depot.dossier(sid2))                         # 500, une seule tentative
    assert w.r2.json("latest.json")["snapshot_id"] == insts[0].snapshot_id
    rep = c.publier(depot.dossier(sid2))                       # relance : reparee
    assert rep.get("unchanged") is True
    assert w.r2.json("latest.json")["snapshot_id"] == sid2
    assert sid2 in {t["snapshot_id"] for t in w.r2.json("index/times.json")}
    assert c.est_publie(sid2, insts[1].valid_time) == (True, "ok")


def test_rejeu_ne_marque_ni_ne_purge_un_echec(trained, serveur, tmp_path, monkeypatch):
    """Rejeu --remote --purger contre le Worker actuel en panne apres _commit.json : code 3,
    dossier local garde, rien de marque publie ; Worker corrige : la relance termine, marque
    et purge."""
    sys.path.insert(0, str(REPO / "scripts"))
    import m6_rejeu_colab as rejeu
    monkeypatch.setenv("ATOM_PUBLISH_TOKEN", JETON)
    racine = tmp_path / "rejeu"
    args = ["-c", "test", "--data-dir", str(trained.paths.data_dir), "--cache-dir",
            str(trained.paths.cache_dir), "--run-dir", str(trained.paths.run_dir),
            "--sortie", str(racine), "--limite", "1", "--device", "cpu", "--data-kind", "test",
            "--remote", serveur.url, "--purger", *ANCRE]
    w = serveur.worker
    w.echecs_apres_commit = 10
    assert rejeu.main(args) == 3
    dossiers = [p for p in (racine / "snapshots").iterdir() if not p.name.startswith(".")]
    assert len(dossiers) == 1 and (dossiers[0] / "publish-manifest.json").exists()
    assert not (racine / "index" / "publies_distant.json").exists()
    w.echecs_apres_commit, w.corrige = 0, True
    assert rejeu.main(args) == 0
    assert not dossiers[0].exists()                              # purge apres verification
    pub = json.loads((racine / "index" / "publies_distant.json").read_text("utf-8"))
    assert pub[serveur.url] == [dossiers[0].name]


def test_redirection_jamais_suivie(depot_complet):
    """Une 302 du Worker devient une erreur : le jeton ne part jamais vers l'hote de la
    redirection (second serveur local qui enregistre les en-tetes)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from pcfno3d.publish import ClientPublication, PublicationRefusee
    vus = []

    class Cible(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _r(self):
            vus.append(dict(self.headers.items()))
            self.send_response(200)
            self.send_header("content-length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
        do_GET = do_PUT = do_POST = _r

    cible = ThreadingHTTPServer(("127.0.0.1", 0), Cible)
    url_cible = f"http://127.0.0.1:{cible.server_address[1]}"

    class Redirige(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _r(self):
            n = int(self.headers.get("content-length") or 0)
            self.rfile.read(n)
            self.send_response(302)
            self.send_header("location", url_cible + self.path)
            self.send_header("content-length", "0")
            self.end_headers()
        do_GET = do_PUT = do_POST = _r

    source = ThreadingHTTPServer(("127.0.0.1", 0), Redirige)
    for s in (cible, source):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        c = ClientPublication(f"http://127.0.0.1:{source.server_address[1]}", token=JETON,
                              tentatives=1, pause_s=0.0, journal=None)
        with pytest.raises(PublicationRefusee, match="redirection"):
            c._requete("PUT", "/v1/publish/x_0123abcd/files/a.json", b"{}")
        depot, insts = depot_complet
        with pytest.raises(PublicationRefusee, match="redirection"):
            c.publier(depot.dossier(insts[0].snapshot_id))
        assert vus == []                       # aucune requete, donc aucun jeton, chez la cible
    finally:
        for s in (cible, source):
            s.shutdown()
            s.server_close()


def test_index_local_repare(nu, trained, tmp_path):
    """Coupure entre le renommage du dossier et l'index : a l'ouverture, DepotLocal complete
    index/times.json et latest.json depuis les dossiers presents, et retire les dossiers
    temporaires abandonnes."""
    from pcfno3d.publish import DepotLocal, instantane_du_cache
    racine = tmp_path / "pub"
    depot = DepotLocal(racine)
    a, b = _instants(trained, 2)
    depot.ecrire(instantane_du_cache(nu, a, data_kind="test", depot=depot, generated_at=GEN))
    ib = instantane_du_cache(nu, b, data_kind="test", depot=depot, generated_at=GEN)
    depot.ecrire(ib)
    # etat apres coupure : index sans l'instantane b, latest encore sur a, temporaire laisse
    items = [t for t in depot.times() if t["snapshot_id"] != ib.snapshot_id]
    (racine / "index" / "times.json").write_bytes(json.dumps(
        {"total": 1, "next": None, "items": items}).encode("utf-8"))
    (racine / "latest.json").write_bytes(
        (depot.dossier(items[0]["snapshot_id"]) / "latest.json").read_bytes())
    (racine / "snapshots" / f".{ib.snapshot_id}.partiel123").mkdir()
    rouvert = DepotLocal(racine)
    assert ib.snapshot_id in rouvert.repares and "latest.json" in rouvert.repares
    assert [t["snapshot_id"] for t in rouvert.times()][0] == ib.snapshot_id
    assert len(rouvert.times()) == 2 and rouvert.times()[0]["open_alerts"] is None
    assert rouvert.latest()["snapshot_id"] == ib.snapshot_id
    assert not list((racine / "snapshots").glob(".*"))
    assert DepotLocal(racine).repares == []                    # rien a refaire ensuite


def test_exports_refaits_depuis_le_dossier(depot_complet, geom, trained, tmp_path,
                                           monkeypatch):
    """Exports DE3 depuis le dossier PUBLIE : seuls les manquants sont refaits, un CSV qui
    echoue a l'ecriture ne laisse pas de .tmp, valeurs = champs publies."""
    import pandas as pd

    from pcfno3d.export_noeuds import _lire_netcdf, chemins_exports, exporter_dossier
    depot, insts = depot_complet
    sid = insts[0].snapshot_id
    dossier = depot.dossier(sid)
    rc = trained.reliability
    ex = tmp_path / "exports"
    faits = exporter_dossier(dossier, geom, ex, rc.class_high, rc.class_mid)
    assert set(faits) == {"netcdf", "csv"}
    assert exporter_dossier(dossier, geom, ex, rc.class_high, rc.class_mid) == {}
    ch = chemins_exports(ex, sid)
    ch["csv"].unlink()
    assert set(exporter_dossier(dossier, geom, ex, rc.class_high, rc.class_mid)) == {"csv"}
    df = pd.read_csv(ch["csv"], keep_default_na=False, na_values=[""])
    assert len(df) == int(insts[0].champs.valide.sum())
    _, attrs, _ = _lire_netcdf(ch["netcdf"])
    assert attrs["snapshot_id"] == sid
    ch["csv"].unlink()

    def panne(self, *a, **k):
        raise OSError("disque plein (simule)")

    monkeypatch.setattr(pd.DataFrame, "to_csv", panne)
    with pytest.raises(OSError):
        exporter_dossier(dossier, geom, ex, rc.class_high, rc.class_mid)
    assert not ch["csv"].exists() and not list(ch["csv"].parent.glob(".*tmp*"))


def test_cli_instantane_existant(trained, tmp_path):
    """Instantane deja ecrit : configuration differente -> refus ; meme configuration ->
    reutilise tel quel, exports faits depuis CE dossier (aucun recalcul melange)."""
    from pcfno3d.cli import main
    stem = _instants(trained, 1)[0]
    racine = tmp_path / "cli"
    base = ["--data-dir", str(trained.paths.data_dir), "--cache-dir",
            str(trained.paths.cache_dir), "--run-dir", str(trained.paths.run_dir), "-c",
            "test", *ANCRE, "publish", "--stem", stem, "--sortie", str(racine),
            "--device", "cpu"]
    assert main(base + ["--data-kind", "test"]) == 0
    sid = json.loads((racine / "latest.json").read_text(encoding="utf-8"))["snapshot_id"]
    avant = (racine / "snapshots" / sid / "publish-manifest.json").read_bytes()
    assert main(base + ["--data-kind", "synthetique"]) == 1
    assert main(base + ["--data-kind", "test", "--exports"]) == 0
    assert (racine / "snapshots" / sid / "publish-manifest.json").read_bytes() == avant
    assert (racine / "exports" / sid / f"{sid}_noeuds.csv").exists()
