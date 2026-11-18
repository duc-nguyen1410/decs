"""Which unstable modes the stability step saves (ecs_core/stability.py, modes_to_save; solver option
Nsave_modes): the most unstable first, at most Nsave_modes of them, all of them by default.
Run: python -m pytest tests"""
import numpy as np
import pytest

pytest.importorskip("dedalus.public")
from ecs_core.stability import modes_to_save

GROWTH = np.array([0.05 + 0j, 0.30 + 0.1j, 0.30 - 0.1j, -0.2 + 0j, 0.01 + 0j, 0.12 + 0j, -0.01 + 0j])


def test_all_unstable_modes_by_default_most_unstable_first():
    assert modes_to_save(GROWTH).tolist() == [1, 2, 5, 0, 4]


def test_only_the_leading_n_modes():
    assert modes_to_save(GROWTH, 3).tolist() == [1, 2, 5]
    assert modes_to_save(GROWTH, 10).tolist() == [1, 2, 5, 0, 4]   # fewer unstable than n: all of them
    assert modes_to_save(GROWTH, 0).tolist() == []


def test_no_unstable_mode():
    assert modes_to_save(-np.abs(GROWTH), 5).tolist() == []
