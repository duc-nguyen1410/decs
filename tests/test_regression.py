"""Regression test: every ECS type (EQ, TW, PO, RPO), with and without CFL, plus Arnoldi,
must give exactly the stored reference results (tests/data/regression_reference.npz).

A failure means the code now computes something different. If that is intended, check
the new results and regenerate the reference with `python tests/make_reference.py`.
The reference is exact on the machine and Dedalus/FFTW build it was made with; on
another machine, rounding may differ slightly, so differences up to RTOL are reported
but accepted there (set DECS_EXACT=0).
"""
import os
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("dedalus.public")
from regression_cases import run_cases

REFERENCE = Path(__file__).resolve().parent/'data'/'regression_reference.npz'
EXACT = os.environ.get('DECS_EXACT', '1') != '0'
RTOL = 1e-6


@pytest.fixture(scope='module')
def results():
    return run_cases()


@pytest.fixture(scope='module')
def reference():
    if not REFERENCE.exists():
        pytest.skip("No reference yet: run python tests/make_reference.py")
    return np.load(REFERENCE)


def test_same_cases(results, reference):
    assert sorted(results) == sorted(reference.files)


@pytest.mark.parametrize('key', ['EQ_cfl0', 'TW_cfl0', 'PO_cfl0', 'RPO_cfl0',
                                 'EQ_cfl1', 'TW_cfl1', 'PO_cfl1', 'RPO_cfl1',
                                 'EQ_cfl0_arnoldi', 'PO_cfl0_arnoldi',
                                 'EQ_cfl1_arnoldi', 'PO_cfl1_arnoldi'])
def test_matches_reference(results, reference, key):
    new, ref = results[key], reference[key]
    diff = np.max(np.abs(new - ref))/max(np.max(np.abs(ref)), 1e-300)
    if EXACT:
        assert np.array_equal(new, ref), f"{key}: max relative difference {diff:.3e}"
    else:
        assert diff < RTOL, f"{key}: max relative difference {diff:.3e}"
