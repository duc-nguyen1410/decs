"""Time integration of FluidModel (physics/base.py) on a small SaltFinger grid.
Run: python -m pytest tests"""
import numpy as np
import pytest

de = pytest.importorskip("dedalus.public")
from physics import SaltFinger

PARAMS = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'init_dt': 2e-4}
T = 0.004            # 20 steps of init_dt


def make_model(**overrides):
    model = SaltFinger({**PARAMS, **overrides}, sizes=(32, 32), bounds=(0.6, 1.0))
    model.build_problems()
    return model


@pytest.fixture(scope='module')
def model():
    return make_model()


@pytest.fixture(scope='module')
def states(model):
    rng = np.random.default_rng(0)
    return [1e-2*rng.standard_normal(model.size()) for _ in range(2)]


def fresh_flow_map(model, x0, Tp):
    """Reference: integrate on a newly built solver (the behaviour before solver reuse)."""
    solver = model._new_ivp_solver(x0, Tp, reuse=False)
    model._integrate(solver, Tp)
    return model.get_state()


def test_reused_solver_matches_fresh_solver(model, states):
    x0, x1 = states
    model.flow_map(x0, T)                       # leave history in the cached solver
    reused = model.flow_map(x1, T)
    assert np.array_equal(reused, fresh_flow_map(model, x1, T))


def test_t_derivative_matches_fresh_solver(model, states):
    x0, x1 = states
    dt = 1e-6
    solver = model._new_ivp_solver(x1, dt, reuse=False)
    solver.step(dt)
    expected = (model.get_state() - x1)/dt
    model.flow_map(x0, T)
    assert np.array_equal(model.t_derivative(x1, dt), expected)


def test_fresh_build_does_not_corrupt_cached_solver(states):
    # Building a second solver from the same IVP problem invalidates the first one;
    # save_time_dependent_solution does this after every converged orbit.
    model = make_model()
    expected = make_model().flow_map(states[1], T)
    model.flow_map(states[0], T)
    model._integrate(model._new_ivp_solver(states[0], T, reuse=False), T)
    assert np.array_equal(model.flow_map(states[1], T), expected)


def test_callback_called_every_step_and_can_stop(model, states):
    calls = []
    model.flow_map(states[0], T, callback=lambda s: calls.append(s.iteration) and False)
    assert calls == list(range(1, 21))

    stopped = []
    def stop_at_5(solver):
        stopped.append(solver.iteration)
        return solver.iteration == 5
    model.flow_map(states[0], T, callback=stop_at_5)
    assert stopped == [1, 2, 3, 4, 5]


def test_set_param_rebuilds_solver(states):
    model = make_model()
    model.flow_map(states[0], T)                # cache a solver at Ra = 1e5
    model.set_param('Ra', 2e5)
    reference = make_model(Ra=2e5)
    assert np.array_equal(model.flow_map(states[1], T), reference.flow_map(states[1], T))


def test_observables(model):
    x_grid, z_grid = (b.global_grid(model.dist, scale=1) for b in model.bases)
    Lx = model.bounds[0]
    u = np.zeros((2,) + model.get_grid_shape())
    u[0] = np.sin(2*np.pi*z_grid)                               # <u^2>/2 = 1/4
    te = np.cos(2*np.pi*x_grid/Lx) + 0*z_grid                   # <te^2> = 1/2
    sa = np.zeros(model.get_grid_shape())
    x = np.concatenate([u.ravel(), te.ravel(), sa.ravel()])
    assert model.kinetic_energy(x) == pytest.approx(0.25, rel=1e-12)
    assert model.l2norm(x) == pytest.approx(1.0, rel=1e-12)    # sqrt(1/2 + 1/2)
