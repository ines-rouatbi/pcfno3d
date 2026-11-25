"""Trousseau sur Google Drive (service Colab) : une relecture peut rendre une copie PLUS ANCIENNE que la derniere
ecriture du service. Une session de compte creee a l'instant ne doit pas disparaitre (401 trois secondes apres la
connexion, 25/11/2026) ; une revocation ecrite ensuite par un autre processus doit, elle, etre prise en compte."""
from __future__ import annotations

import json
import os
import time

from service.auth import ApiKeyStore


def _perimer(path, contenu: str) -> None:
    """Remet une ancienne copie du fichier, comme une relecture de Drive non encore synchronisee."""
    path.write_text(contenu, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))   # attributs changes -> relecture


def test_une_copie_perimee_ne_fait_pas_perdre_la_session(tmp_path):
    f = tmp_path / "api_keys.json"
    store = ApiKeyStore(f)
    store.create("worker", scope="read")
    ancien = f.read_text(encoding="utf-8")                  # etat avant la connexion
    secret, _ = store.create("compte-abc", scope="admin")   # la connexion ecrit la session
    _perimer(f, ancien)
    assert store.verify(secret) is not None                 # la session reste valide
    assert store.verify(secret) is not None                 # et a la requete suivante aussi


def test_une_ecriture_plus_recente_d_un_autre_processus_est_lue(tmp_path):
    f = tmp_path / "api_keys.json"
    service = ApiKeyStore(f)
    secret, _ = service.create("partenaire", scope="read")
    assert service.verify(secret) is not None
    time.sleep(0.01)
    cli = ApiKeyStore(f)                                     # pcfno3d apikey revoke, autre processus
    assert cli.revoke("partenaire")
    assert service.verify(secret) is None                   # revocation prise en compte


def test_le_fichier_porte_son_horodatage(tmp_path):
    f = tmp_path / "api_keys.json"
    ApiKeyStore(f).create("x", scope="read")
    assert json.loads(f.read_text(encoding="utf-8"))["ecrit_ns"] > 0


def test_un_ancien_fichier_sans_horodatage_se_lit_au_demarrage(tmp_path):
    f = tmp_path / "api_keys.json"
    ApiKeyStore(f).create("x", scope="read")
    data = json.loads(f.read_text(encoding="utf-8"))
    data.pop("ecrit_ns")
    f.write_text(json.dumps(data), encoding="utf-8")
    assert [r.name for r in ApiKeyStore(f).records] == ["x"]
