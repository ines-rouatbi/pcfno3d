"""Tunnel nomme a URL fixe (atom-origin.zelqane.com) et cle enregistree par empreinte.

Deux secrets, deux sens, remis de main a main : les identifiants du tunnel arrivent sur
Colab, la cle de lecture en part. Ces tests verifient surtout qu'aucun des deux ne fuit :
ni dans un journal, ni dans un message d'erreur, ni sur disque.
"""

from __future__ import annotations

import json
import sys
import uuid

import pytest

TUNNEL_ID = str(uuid.uuid4())
SECRET_TUNNEL = "c2VjcmV0LWR1LXR1bm5lbC1xdWktbmUtZG9pdC1qYW1haXMtZnVpcg=="


def _creds(tunnel_id: str = TUNNEL_ID, **over) -> str:
    data = {"AccountTag": "compte", "TunnelSecret": SECRET_TUNNEL, "TunnelID": tunnel_id}
    data.update(over)
    return json.dumps(data)


# ---------------------------------------------------------------- cle par empreinte
def test_cle_enregistree_par_empreinte_sans_le_secret(tmp_path):
    from service.auth import ApiKeyStore, generate_key, hash_key

    secret = generate_key()
    store = ApiKeyStore(tmp_path / "api_keys.json")
    record = store.register_hash("atom-worker", hash_key(secret), scope="read")
    assert record.scope == "read"
    assert secret not in (tmp_path / "api_keys.json").read_text(encoding="utf-8")

    relu = ApiKeyStore(tmp_path / "api_keys.json")      # nouvelle session Colab
    assert relu.verify(secret).name == "atom-worker"
    assert relu.verify(generate_key()) is None
    # idempotent : la cellule peut etre relancee a chaque session
    assert relu.register_hash("atom-worker", hash_key(secret)).key_hash == record.key_hash
    assert len(relu.records) == 1


def test_empreinte_invalide_ou_portee_differente_refusees(tmp_path):
    from service.auth import ApiKeyStore, generate_key, hash_key

    store = ApiKeyStore(tmp_path / "api_keys.json")
    with pytest.raises(ValueError, match="SHA-256"):
        store.register_hash("x", "pcf_pas_une_empreinte")
    h = hash_key(generate_key())
    store.register_hash("x", h, scope="read")
    with pytest.raises(ValueError, match="portee read"):
        store.register_hash("x", h, scope="admin")       # pas d'elevation silencieuse


# ---------------------------------------------------------------- identifiants du tunnel
@pytest.mark.parametrize("creds, motif", [
    ("pas du json", "illisibles"),
    (json.dumps({"AccountTag": "a", "TunnelSecret": SECRET_TUNNEL}), "incomplets"),
    (_creds(tunnel_id=str(uuid.uuid4())), "AUTRE tunnel"),
])
def test_identifiants_refuses_sans_jamais_etre_affiches(creds, motif):
    from service.tunnel import check_credentials

    with pytest.raises(ValueError, match=motif) as exc:
        check_credentials(creds, TUNNEL_ID)
    assert SECRET_TUNNEL not in str(exc.value)


FAUX_CLOUDFLARED = r'''
import os, sys, time
args = sys.argv[1:]
assert args[:3] == ["tunnel", "--no-autoupdate", "run"], args
assert args[-1] == sys.argv[-1]
creds = os.environ.get("TUNNEL_CRED_CONTENTS", "")
if "{echec}" == "oui":
    print("ERR Unauthorized: invalid credentials", flush=True)
    sys.exit(1)
assert "TunnelSecret" in creds
print("INF Starting tunnel tunnelID=" + args[-1], flush=True)
print("INF Registered tunnel connection connIndex=0 location=cdg01 protocol=quic", flush=True)
time.sleep(60)
'''


def _faux(tmp_path, echec=False):
    script = tmp_path / "faux_cloudflared.py"
    script.write_text(FAUX_CLOUDFLARED.replace("{echec}", "oui" if echec else "non"),
                      encoding="utf-8")
    return [sys.executable, str(script)]


def test_tunnel_nomme_url_fixe_et_secret_hors_du_journal(tmp_path):
    from service.tunnel import start_named_tunnel

    log = tmp_path / "cloudflared.log"
    tun = start_named_tunnel(TUNNEL_ID, "atom-origin.zelqane.com", _creds(), port=8123,
                             timeout=30, log_path=log, command=_faux(tmp_path))
    try:
        assert tun.public_url == "https://atom-origin.zelqane.com"
        assert tun.host == "atom-origin.zelqane.com"
        assert tun.alive
        assert SECRET_TUNNEL not in log.read_text(encoding="utf-8")
        assert TUNNEL_ID in log.read_text(encoding="utf-8")
    finally:
        tun.stop()
    assert not tun.alive


def test_tunnel_nomme_refuse_explique_sans_reveler_le_secret(tmp_path):
    from service.tunnel import start_named_tunnel

    with pytest.raises(RuntimeError, match="Unauthorized") as exc:
        start_named_tunnel(TUNNEL_ID, "atom-origin.zelqane.com", _creds(), timeout=30,
                           log_path=tmp_path / "cf.log", command=_faux(tmp_path, echec=True))
    assert SECRET_TUNNEL not in str(exc.value)
