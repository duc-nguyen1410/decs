"""MultiShootECSSolver (ecs_core/multishoot.py) on a 32^2 SaltFinger grid."""
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("dedalus.public")
from physics import SaltFinger
from ecs_core import MultiShootECSSolver

REFERENCE = Path(__file__).resolve().parent/'data'/'regression_reference.npz'


@pytest.fixture(scope='module')
def model():
    m = SaltFinger({'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'init_dt': 2e-4},
                   sizes=(32, 32), bounds=(0.6, 1.0))
    m.build_problems()
    return m


@pytest.fixture(scope='module')
def x0(model):
    return model.flow_map(5e-2*np.random.default_rng(0).standard_normal(model.size()), 0.01)


def solver_for(model, m, **search):
    s = MultiShootECSSolver(model, {'n_segments': m})
    s.Tsearch, s.Rxsearch = int(search.get('Tsearch', 0)), int(search.get('Rxsearch', 0))
    s.Tp, s.ax = 0.02, search.get('ax', 0.0)
    return s


def test_one_segment_is_single_shooting_exactly():
    """m = 1 must reproduce every regression case (EQ, TW, PO, RPO, CFL, Arnoldi) bit for bit."""
    from regression_cases import run_cases
    if not REFERENCE.exists():
        pytest.skip("No reference yet: run python tests/make_reference.py")
    reference = np.load(REFERENCE)
    results = run_cases(MultiShootECSSolver, {'n_segments': 1})
    for key in reference.files:
        assert np.array_equal(results[key], reference[key]), key


def test_initial_segments_join_exactly(model, x0):
    s = solver_for(model, 3, Tsearch=True)
    xi = s.pack_xi(x0, s.Tp)
    assert xi.size == 3*model.size() + 1
    res, _ = s._evaluate(xi)
    N = model.size()
    assert np.all(res[:2*N] == 0)            # x_{i+1} = F^tau(x_i) for the inner segments
    assert np.linalg.norm(res[2*N:3*N]) > 0  # the orbit does not close


@pytest.mark.parametrize('search', [dict(Tsearch=True), dict(Tsearch=True, Rxsearch=True, ax=0.05)])
def test_jacobian_matches_finite_difference_of_residual(model, x0, search):
    """State rows of matvec(v) = -(r(xi + h v) - r(xi))/h, including the Tp and shift columns."""
    s = solver_for(model, 3, **search)
    xi = s.pack_xi(x0, s.Tp, s.ax)
    res, phi = s._evaluate(xi)
    matvec = s._jacobian(xi, phi)
    rng = np.random.default_rng(2)
    N3 = 3*model.size()
    for col in ['x'] + s._searched():
        v = np.zeros_like(xi)
        if col == 'x':
            v[:N3] = rng.standard_normal(N3)
        else:
            v[N3 + s._searched().index(col)] = 1.0
        h = 1e-6/np.linalg.norm(v)*np.linalg.norm(xi[:N3]) if col == 'x' else 1e-6
        fd = -(s._evaluate(xi + h*v)[0] - res)[:N3]/h
        jv = matvec(v)[:N3]
        tol = 1e-2 if col == 'Tp' else 1e-3   # d/dTp is approximated by the vector field
        assert np.linalg.norm(jv - fd) < tol*np.linalg.norm(fd), col
