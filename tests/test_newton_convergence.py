"""Newton convergence bookkeeping (ecs_core/newton.py, ECSSolver.solve): a step that brings the residual
below tol on the last allowed iteration (max_iter) counts as converged, its solution is saved and its
stability computed. Before the fix the loop only tested convergence at the top of an iteration, so such a
solve returned success=False (seen as "Failed to initialize continuation" with a residual of 1e-10).
Newton steps are stubbed: each one divides the residual by 100.
Run: python -m pytest tests"""
import numpy as np
import pytest

pytest.importorskip("dedalus.public")
from ecs_core import newton
from ecs_core.newton import ECSSolver


class FakeModel:
    def __init__(self, n=4):
        self.n, self.saved = n, []

    def size(self):
        return self.n

    def set_state(self, x):
        self.x = np.array(x)

    def save_state(self, path):
        self.saved.append(path)

    def get_flow_properties(self):
        return {}


def make_solver(monkeypatch, max_iter, tol, r0=1.0):
    solver = ECSSolver.__new__(ECSSolver)
    solver.model, solver.max_iter, solver.tol = FakeModel(), max_iter, tol
    solver.odir, solver.save_ecs_history, solver.trust_radius = 'out/', False, 1.0
    solver.converged_calls = []
    solver._on_converged = lambda xi: solver.converged_calls.append(xi.copy())
    solver._evaluate = lambda xi: (np.full(4, r0/2), None)          # ||residual|| = r0
    solver.newton_step = lambda xi, res, phi: (np.zeros_like(xi), 0.0, 1.0, (res/100, phi))

    class Log:
        def write_header(self, s): pass
        def write(self, line): pass
        def close(self): pass
    monkeypatch.setattr(newton.output, 'RunLog', lambda s: Log())
    for name in ('reset_flow_properties', 'write_flow_properties', 'write_iteration_files'):
        monkeypatch.setattr(newton.output, name, lambda *a, **k: None)
    return solver


def test_converged_on_last_iteration_is_a_success(monkeypatch):
    # residuals 1, 1e-2, 1e-4, 1e-6: below tol only after the 3rd (= last) step
    solver = make_solver(monkeypatch, max_iter=3, tol=1e-5)
    _, success, res, _, _ = solver.solve(np.zeros(4))
    assert success and res < 1e-5
    assert len(solver.converged_calls) == 1


def test_converged_before_the_last_iteration(monkeypatch):
    solver = make_solver(monkeypatch, max_iter=6, tol=1e-5)
    _, success, _, _, _ = solver.solve(np.zeros(4))
    assert success and len(solver.converged_calls) == 1


def test_not_converged_is_a_failure(monkeypatch):
    solver = make_solver(monkeypatch, max_iter=2, tol=1e-5)
    _, success, res, _, _ = solver.solve(np.zeros(4))
    assert not success and res > 1e-5
    assert solver.converged_calls == []
