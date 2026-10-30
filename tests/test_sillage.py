"""Sillage de Jensen (BF72) : cas analytiques."""

from __future__ import annotations

import math

import numpy as np
import pytest


def test_aligne_et_hors_sillage():
    from pcfno3d.sillage import jensen
    # deux eoliennes alignees nord-sud, 500 m d'ecart, vent du nord : la seconde (au sud) est dans le sillage
    u = jensen([0, 0], [500, 0], [90, 90], [8.0, 8.0], direction_deg=0.0, ct=[0.8, 0.8], k=0.075)
    attendu = 8.0 * (1 - (1 - math.sqrt(0.2)) / (1 + 2 * 0.075 * 500 / 90) ** 2)
    assert u[0] == pytest.approx(8.0) and u[1] == pytest.approx(attendu, rel=1e-9)
    # vent d'est : elles sont cote a cote, aucun sillage
    assert np.allclose(jensen([0, 0], [500, 0], [90, 90], [8.0, 8.0], direction_deg=90.0), 8.0)
    # vent du sud : c'est la premiere qui est a l'abri
    v = jensen([0, 0], [500, 0], [90, 90], [8.0, 8.0], direction_deg=180.0, ct=[0.8, 0.8])
    assert v[1] == pytest.approx(8.0) and v[0] < 8.0


def test_recouvrement_partiel_et_somme_quadratique():
    from pcfno3d.sillage import _recouvrement, ct_generique, jensen
    assert _recouvrement(0.0, 45, 60) == 1.0 and _recouvrement(200.0, 45, 60) == 0.0
    assert 0.0 < _recouvrement(80.0, 45, 60) < 1.0
    # trois en ligne : la troisieme recoit deux sillages, plus de deficit que la deuxieme
    u = jensen([0, 0, 0], [1000, 500, 0], [90] * 3, [9.0] * 3, direction_deg=0.0)
    assert u[0] > u[1] and u[2] < u[0]
    assert ct_generique(np.array([2.0, 8.0, 20.0, 30.0])).tolist()[0] == 0.0
    assert ct_generique(np.array([30.0]))[0] == 0.0
