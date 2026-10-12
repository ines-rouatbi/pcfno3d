"""Ou et quand l'entrainement ecrit ses checkpoints.

Sur Colab, run_dir est le Drive : y ecrire last.pt (~7 Go pour la v7) a chaque epoque
coutait ~75 s par epoque. Avec paths.local_dir, l'entrainement ecrit en local et ne
recopie sur le Drive que toutes les `save_every` epoques, a la fin, et sur interruption.
Et le cache, qui ne change pas pendant l'entrainement, n'est plus jamais recopie.
"""

from __future__ import annotations

import shutil

import pytest
import torch


def _cfg(base, tmp_path, *, local=True, epochs=3, save_every=2, mirror=None):
    from pcfno3d.config import Config

    c = Config.load("test")
    over = {"paths.data_dir": base.paths.data_dir, "paths.cache_dir": base.paths.cache_dir,
            "paths.run_dir": tmp_path / "drive_run", "train.epochs": epochs,
            "train.save_every": save_every, "train.num_workers": 0}
    if local:
        over["paths.local_dir"] = tmp_path / "local_run"
    if mirror is not None:
        over["paths.cache_mirror"] = mirror
    c.apply_overrides(over)
    c.paths.mkdirs()
    shutil.copy2(base.paths.norm_stats, c.paths.norm_stats)    # meme normalisation
    return c


def test_last_ecrit_en_local_et_recopie_periodiquement(cfg, index, norm, tmp_path):
    from pcfno3d.train import Trainer

    c = _cfg(cfg, tmp_path, epochs=3, save_every=2)
    t = Trainer(c, device="cpu", verbose=False)

    # on observe l'etat du Drive apres chaque epoque
    vu_sur_le_drive = []
    original = t._save_atomic

    def espion(path, payload):
        original(path, payload)
        if path.name == "last.pt":
            durable = c.paths.ckpt_last
            vu_sur_le_drive.append(
                torch.load(durable, weights_only=False)["epoch"] if durable.exists() else None)

    t._save_atomic = espion
    t.fit()

    # epoque 1 (index 0) : rien sur le Drive ; la copie de l'epoque 2 n'a lieu qu'apres
    # l'ecriture locale, donc l'espion la voit a l'epoque 3
    assert vu_sur_le_drive == [None, None, 1]
    assert (c.paths.local_dir / "last.pt").exists()
    assert torch.load(c.paths.ckpt_last, weights_only=False)["epoch"] == 2   # fin : a jour
    assert c.paths.ckpt_best.exists()
    assert [s["raison"] for s in t.syncs] == ["epoque 2", "epoque 3"]
    assert not list(c.paths.run_dir.glob("*.tmp"))                         # copies atomiques


def test_best_recopie_seulement_s_il_a_change(cfg, index, norm, tmp_path, monkeypatch):
    from pcfno3d.train import Trainer

    c = _cfg(cfg, tmp_path, epochs=4, save_every=1)
    t = Trainer(c, device="cpu", verbose=False)
    scores = iter([0.5, 0.9, 0.9, 0.9])     # seule la 1re epoque ameliore
    monkeypatch.setattr(t.loss_fn, "val_score", lambda _parts: next(scores))
    t.fit()
    copies_best = [s for s in t.syncs if "best.pt" in s["fichiers"]]
    assert len(copies_best) == 1          # 1,8 Go recopie une fois, pas a chaque epoque
    assert all("last.pt" in s["fichiers"] for s in t.syncs)


def test_reprise_dans_une_session_neuve(cfg, index, norm, tmp_path):
    """Session Colab coupee net puis rouverte : disque local perdu, on repart du Drive.

    Coupure nette = le processus meurt, le `finally` ne s'execute pas : on le simule en
    neutralisant la copie « interruption ». Le nombre d'epoques ne change pas, comme sur
    Colab (un plan OneCycle ne s'allonge pas apres coup).
    """
    from pcfno3d.train import Trainer

    c = _cfg(cfg, tmp_path, epochs=3, save_every=2)
    t = Trainer(c, device="cpu", verbose=False)
    sync = t._sync
    t._sync = lambda reason="": None if reason == "interruption" else sync(reason)
    n = {"e": 0}
    validate = t.validate

    def coupe_a_la_3e(*a, **k):
        n["e"] += 1
        if n["e"] == 3:
            raise KeyboardInterrupt
        return validate(*a, **k)

    t.validate = coupe_a_la_3e
    with pytest.raises(KeyboardInterrupt):
        t.fit()
    assert torch.load(c.paths.ckpt_last, weights_only=False)["epoch"] == 1   # copie ep. 2

    shutil.rmtree(c.paths.local_dir)               # nouvelle machine Colab
    reprise = Trainer(c, device="cpu", verbose=False)
    assert reprise.start_epoch == 2                # repart de la copie du Drive
    assert (c.paths.local_dir / "best.pt").exists()   # best ramene pour la comparaison
    reprise.fit()
    assert torch.load(c.paths.ckpt_last, weights_only=False)["epoch"] == 2


def test_reprise_prefere_le_local_de_la_meme_session(cfg, index, norm, tmp_path):
    """Meme session : le local est plus recent que le Drive (copie periodique en retard)."""
    from pcfno3d.train import Trainer

    c = _cfg(cfg, tmp_path, epochs=2, save_every=2)
    Trainer(c, device="cpu", verbose=False).fit()          # local et Drive : epoque 1
    # le Drive en retard d'une epoque, comme entre deux copies periodiques
    ancien = torch.load(c.paths.ckpt_last, weights_only=False)
    ancien["epoch"] = 0
    torch.save(ancien, c.paths.ckpt_last)

    c.apply_overrides({"train.epochs": 3})
    reprise = Trainer(c, device="cpu", verbose=False)
    assert reprise.start_epoch == 2                        # depuis le local (epoque 1)


def test_interruption_sauve_le_travail_sur_le_drive(cfg, index, norm, tmp_path):
    from pcfno3d import train as train_mod

    # 4 epoques, pas de copie periodique avant la fin. (5 donnerait exactement 10 pas de
    # plan : OneCycleLR de PyTorch divise alors par zero dans sa phase de montee — cas
    # degenere des petites configurations, impossible a l'echelle reelle : 5 000 pas en v8)
    c = _cfg(cfg, tmp_path, epochs=4, save_every=10)
    t = train_mod.Trainer(c, device="cpu", verbose=False)
    n = {"epoques": 0}
    original = t.validate

    def validate_puis_coupe(*a, **k):
        n["epoques"] += 1
        if n["epoques"] == 3:
            raise KeyboardInterrupt                # la cellule est arretee
        return original(*a, **k)

    t.validate = validate_puis_coupe
    with pytest.raises(KeyboardInterrupt):
        t.fit()
    assert t.syncs[-1]["raison"] == "interruption"
    # les deux epoques terminees sont sur le Drive
    assert torch.load(c.paths.ckpt_last, weights_only=False)["epoch"] == 1


def test_sans_local_dir_ecriture_directe(cfg, index, norm, tmp_path):
    """Poste local : run_dir est deja un disque rapide, aucune copie n'a lieu."""
    from pcfno3d.train import Trainer

    c = _cfg(cfg, tmp_path, local=False, epochs=2, save_every=1)
    t = Trainer(c, device="cpu", verbose=False)
    t.fit()
    assert c.paths.ckpt_last.exists() and c.paths.ckpt_best.exists()
    assert t.syncs == []


# ---------------------------------------------------------------------------- cache
def test_l_entrainement_ne_touche_jamais_au_miroir_du_cache(cfg, index, norm, tmp_path):
    from pcfno3d.train import Trainer

    miroir = tmp_path / "miroir_drive"
    c = _cfg(cfg, tmp_path, epochs=2, save_every=1, mirror=miroir)
    Trainer(c, device="cpu", verbose=False).fit()
    assert not miroir.exists() or not any(miroir.iterdir())


def test_build_cache_ne_copie_que_les_jours_construits(cfg, geom, tmp_path):
    from pcfno3d.cache import build_cache
    from pcfno3d.config import Config

    c = Config.load("test")
    miroir = tmp_path / "miroir"
    c.apply_overrides({"paths.data_dir": cfg.paths.data_dir,
                       "paths.cache_dir": tmp_path / "cache_local",
                       "paths.run_dir": tmp_path / "run", "paths.cache_mirror": miroir})
    build_cache(c, geom, days=[1], verbose=False)
    premiers = sorted(p.name for p in miroir.iterdir())
    assert "index.csv" in premiers
    assert all(n.startswith("001_") or n == "index.csv" for n in premiers)

    avant = {p.name: p.stat().st_mtime_ns for p in miroir.glob("001_*")}
    build_cache(c, geom, days=[1, 2], verbose=False)     # le jour 1 existe deja
    apres = {p.name: p.stat().st_mtime_ns for p in miroir.glob("001_*")}
    assert avant == apres                                 # jamais recopie
    assert any(p.name.startswith("002_") for p in miroir.iterdir())
