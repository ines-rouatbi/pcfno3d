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
    from pcfno3d.train import Trainer
    if not cfg.paths.ckpt_best.exists():
        Trainer(cfg, device="cpu", verbose=False).fit()
    return cfg


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
    """Commit coupe AVANT traitement : rien ne bascule. Commit traite mais reponse perdue :
    la reprise recoit « unchanged » et latest est bien le nouvel instantane."""
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
    prod = ClientPublication("https://atom.zelqane.com", token=JETON, tentatives=1)
    with pytest.raises(PublicationRefusee, match="test"):
        prod.publier(depot.dossier(insts[0].snapshot_id))
    assert prod.requetes == 0
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
            str(trained.paths.cache_dir), "--run-dir", str(trained.paths.run_dir), "-c", "test"]
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
            "--data-kind", "test"]
    assert rejeu.main(args) == 0
    times = json.loads((tmp_path / "rejeu" / "index" / "times.json").read_text("utf-8"))
    assert times["total"] == 2
    assert rejeu.main(args + ["--limite", "3"]) == 0          # reprise : 1 seul nouveau
    times = json.loads((tmp_path / "rejeu" / "index" / "times.json").read_text("utf-8"))
    assert times["total"] == 3
    assert "deja publie" in capsys.readouterr().out
