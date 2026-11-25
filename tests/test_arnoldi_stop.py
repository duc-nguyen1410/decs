"""Arnoldi stability (ecs_core/stability.py, _stability_arnoldi): stopping once the leading modes
have converged (arnoldi_tol, arnoldi_nconv) and starting from guess modes (stability_guess,
arnoldi_warm_start). The linearised flow map is a fixed matrix with known multipliers.
Run: python -m pytest tests"""
import numpy as np
import pytest

pytest.importorskip("dedalus.public")
from ecs_core.stability import _stability_arnoldi

N = 300
LEADING = np.array([2.0, 1.6, 1.3])          # multipliers mu of the wanted modes (Tp = 1)


def operator(seed=0, perturb=0.0):
    """Symmetric matrix V diag(mu) V^T: LEADING, then N - 3 multipliers in [0, 0.9]."""
    rng = np.random.default_rng(seed)
    mu = np.concatenate([LEADING, rng.uniform(0.0, 0.9, N - LEADING.size)])
    V, _ = np.linalg.qr(rng.standard_normal((N, N)))
    A = V @ np.diag(mu) @ V.T
    if perturb:
        E = rng.standard_normal((N, N))
        A = A + perturb*(E + E.T)/np.linalg.norm(E + E.T, 2)
    return A


class Model:
    def size(self):
        return N


class FakeSolver:
    """The attributes of ECSSolver that _stability_arnoldi uses; DG v = A v."""
    def __init__(self, A, **opts):
        self.A, self.model, self.comm, self.odir = A, Model(), None, './'
        self.Neigen, self.projectNeutralDrift = 150, False
        self.arnoldi_tol, self.arnoldi_nconv = None, 3
        self.stability_guess, self.arnoldi_warm_start = None, False
        for k, v in opts.items():
            setattr(self, k, v)

    def unpack_xi(self, xi):
        return xi, 1.0, 0.0, 0.0, 0.0

    def return_map(self, *args):
        return None

    def return_map_derivative(self, x0, v, *args):
        return self.A @ v


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)      # stability/arnoldi_history.csv goes to odir = './'


def run(solver):
    np.random.seed(1)
    growthrate, _, _, res = _stability_arnoldi(solver, np.zeros(N))
    return growthrate, res


def test_without_tol_runs_neigen_iterations():
    growthrate, _ = run(FakeSolver(operator()))
    assert growthrate.size == 150


def test_history_has_one_row_per_iteration():
    growthrate, res = run(FakeSolver(operator(), arnoldi_tol=1e-8))
    rows = np.genfromtxt('stability/arnoldi_history.csv', delimiter=',', names=True)
    assert rows['step'].tolist() == list(range(1, growthrate.size + 1))
    last = rows[-1]
    assert np.allclose([last[f'lambda_real_{i}'] for i in range(3)], growthrate[:3].real)
    assert max(last[f'residual_{i}'] for i in range(3)) < 1e-8
    assert np.isnan(rows[0]['lambda_real_1'])      # one Ritz value after the first step


def test_stops_once_leading_modes_converge():
    growthrate, _ = run(FakeSolver(operator(), arnoldi_tol=1e-8))
    assert growthrate.size < 150
    assert np.allclose(growthrate[:3].real, np.log(LEADING), rtol=0, atol=1e-7)


def test_warm_start_converges_in_fewer_iterations():
    # continuation-like: the guess comes from a slightly different operator
    first = FakeSolver(operator(), arnoldi_tol=1e-8, arnoldi_warm_start=True)
    run(first)
    assert len(first.stability_guess) == 3
    A2 = operator(perturb=1e-4)
    cold, _ = run(FakeSolver(A2, arnoldi_tol=1e-8))
    warm, _ = run(FakeSolver(A2, arnoldi_tol=1e-8, stability_guess=first.stability_guess))
    assert warm.size <= 0.7*cold.size
    assert np.allclose(warm[:3].real, cold[:3].real, rtol=0, atol=1e-7)


def test_guess_of_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="stability_guess"):
        run(FakeSolver(operator(), stability_guess=[np.ones(N + 1)]))
