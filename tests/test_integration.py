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
    w = np.zeros((2,) + model.get_grid_shape())
    w[1] = np.cos(2*np.pi*x_grid/Lx) + 0*z_grid                 # <w^2>/2 = 1/4
    xw = np.concatenate([w.ravel(), te.ravel(), sa.ravel()])
    assert model.cross_flow_energy(x) == pytest.approx(0.0, abs=1e-14)
    assert model.cross_flow_energy(xw) == pytest.approx(0.25, rel=1e-12)


def test_observable_in_callback_does_not_change_trajectory(model, states):
    x0, _ = states
    plain = model.flow_map(x0, T)
    energies = []
    observed = model.flow_map(x0, T, callback=lambda s: energies.append(model.cross_flow_energy()))
    assert len(energies) == 20
    assert np.array_equal(plain, observed)


def test_current_state_matches_get_state_and_is_safe_in_callback(model, states):
    x0, _ = states
    plain = model.flow_map(x0, T)
    snaps = []
    observed = model.flow_map(x0, T, callback=lambda s: snaps.append(model.current_state()))
    assert np.array_equal(plain, observed)
    assert np.array_equal(snaps[-1], plain)


@pytest.mark.parametrize('axes,shift', [(('x',), {'ax': 3/32}), (('z',), {'az': -5/32}),
                                        (('z', 'x'), {'ax': -0.25, 'az': 0.125})])
def test_shifted_distance_matches_solver_shift_convention(model, states, axes, shift):
    """ECSSolver solves x0 = S(a) F(x0); shifted_distance(x, y) must return the a with
    S(a) y = x, here for y = S(-a) x built with the model's own translation."""
    from physics import Symmetry
    from tools.poincare import shifted_distance
    x = model.flow_map(states[0], T)                 # smooth state
    y = model.apply_symmetry(x, Symmetry(**{k: -v for k, v in shift.items()}))
    d, found = shifted_distance(model, x, y, axes)
    assert d < 1e-10
    assert found == pytest.approx(shift, abs=1e-10)


def test_find_symmetries_recovers_imposed_symmetry(model, states):
    """A state made symmetric under a shifted x-z reflection: find_symmetries must return that
    symmetry first, and the centering shift must remove its shifts."""
    from physics import Symmetry
    from tools import find_symmetries
    sigma = Symmetry(sx=-1, sz=-1, ax=0.3, az=-0.15)   # involution: sigma(sigma(x)) = x
    x = model.flow_map(states[0], T)                   # smooth state
    x = 0.5*(x + model.apply_symmetry(x, sigma))
    best = find_symmetries(model, x)[0]
    assert best['distance'] < 1e-12
    s = best['symmetry']
    assert (s.sx, s.sy, s.sz) == (-1, 1, -1)
    assert (s.ax, s.az) == pytest.approx((0.3, -0.15), abs=1e-12)
    x_c = model.apply_symmetry(x, Symmetry(**best['center']))
    centered = model.apply_symmetry(x_c, Symmetry(sx=-1, sz=-1))
    assert np.linalg.norm(centered - x_c) < 1e-12*np.linalg.norm(x_c)


def test_change_grid_round_trip_and_values(model, states):
    """Spectral interpolation 32^2 -> 48x40 (other Lx) -> 32^2 returns the state, and on the
    finer grid it equals the old state at the shared grid points."""
    from tools import change_grid
    x = model.flow_map(states[0], T)                   # smooth state
    fine = SaltFinger(PARAMS, sizes=(64, 48), bounds=(0.6, 1.0))
    x_fine = change_grid(model, fine)
    blocks, blocks_fine = x.reshape(-1, 32, 32), x_fine.reshape(-1, 64, 48)
    assert np.allclose(blocks_fine[:, ::2, ::3][:, :, :16], blocks[:, :, ::2][:, :, :16],
                       rtol=0, atol=1e-10*np.abs(x).max())
    back = change_grid(fine, model)
    assert np.linalg.norm(back - x) < 1e-12*np.linalg.norm(x)
    stretched = SaltFinger(PARAMS, sizes=(32, 32), bounds=(0.8, 1.0))   # new Lx: same grid values
    assert np.linalg.norm(change_grid(model, stretched) - x) < 1e-13*np.linalg.norm(x)


def test_change_grid_2d_to_3d(model, states):
    """A 2-D state on a 3-D grid is y-independent with v = 0."""
    from tools import change_grid
    x = model.flow_map(states[0], T)
    model3 = SaltFinger(PARAMS, sizes=(32, 4, 32), bounds=(0.6, 0.3, 1.0))
    x3 = change_grid(model, model3)
    u, rest = x3[:3*32*4*32].reshape(3, 32, 4, 32), x3[3*32*4*32:].reshape(-1, 32, 4, 32)
    u2, rest2 = x[:2*32*32].reshape(2, 32, 32), x[2*32*32:].reshape(-1, 32, 32)
    assert np.abs(u[1]).max() == 0
    for j in range(4):
        assert np.allclose(u[[0, 2], :, j], u2, rtol=0, atol=1e-13)
        assert np.allclose(rest[:, :, j], rest2, rtol=0, atol=1e-13)
