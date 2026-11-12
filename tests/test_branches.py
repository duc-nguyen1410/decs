"""tools/branches.py on a 32^2 grid: reading unstable modes (real and complex), the guesses,
the symmetry note, and the classification of Newton results (with a stand-in solver)."""
import numpy as np
import pytest
import scipy.io

de = pytest.importorskip("dedalus.public")
from physics import SaltFinger, Symmetry
from tools import load_unstable_modes, branch_guesses, search_branches
from tools.branches import _drift_axes

PARAMS = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'init_dt': 2e-4, 'ext': '.nc'}
N = 32
PIXZ = Symmetry(sx=-1, sz=-1)


@pytest.fixture(scope='module')
def model():
    m = SaltFinger(PARAMS, sizes=(N, N), bounds=(0.6, 1.0))
    m.build_problems()
    return m


@pytest.fixture(scope='module')
def states(model):
    """A pixz-symmetric state x, a mode antisymmetric under pixz and two generic vectors."""
    rng = np.random.default_rng(3)
    s = [model.flow_map(1e-2*rng.standard_normal(model.size()), 0.004) for _ in range(4)]
    x = 0.5*(s[0] + model.apply_symmetry(s[0], PIXZ))
    anti = 0.5*(s[1] - model.apply_symmetry(s[1], PIXZ))
    return x, anti, s[2], s[3]


def write_stability(model, folder, lam, vectors):
    folder.mkdir()
    scipy.io.mmwrite(str(folder/'growthrate.mtx'), np.array(lam).reshape(1, -1))
    for i, v in enumerate(vectors):
        model.save_state(str(folder/f'unstable_mode_{i}'), v.real)
        if np.any(v.imag):
            model.save_state(str(folder/f'unstable_mode_{i}_imag'), v.imag)


def test_load_unstable_modes_pairs_and_complex_parts(model, states, tmp_path):
    x, anti, a, b = states
    v1 = a + 1j*b
    write_stability(model, tmp_path/'stab', [2.0, 0.5 + 3j, 0.5 - 3j, -1.0], [anti, v1, np.conj(v1)])
    modes = load_unstable_modes(model, tmp_path/'stab')
    assert [m[0] for m in modes] == [2.0, 0.5 + 3j]               # one per conjugate pair
    assert np.allclose(modes[0][1], anti) and np.allclose(modes[1][1], v1)


def test_branch_guesses(model, states, tmp_path):
    x, anti, a, b = states
    modes = [(2.0 + 0j, anti.astype(complex)), (0.5 + 3j, a + 1j*b)]
    g = branch_guesses(model, x, modes, eps=(1e-2,), n_phases=4, odir=str(tmp_path/'guesses'))
    assert len(g) == 2 + 4                                         # +- for real, 4 phases for Hopf
    assert [h['kind'] for h in g] == ['eq']*2 + ['po']*4
    assert g[2]['Tp'] == pytest.approx(2*np.pi/3) and g[0]['Tp'] is None
    for h in g:
        assert np.linalg.norm(h['x'] - x) == pytest.approx(1e-2*np.linalg.norm(x))
    assert np.allclose(g[1]['x'] - x, -(g[0]['x'] - x))           # the two signs
    # the antisymmetric mode breaks pixz: a drift is searched along x and z
    assert any((s.sx, s.sz) == (-1, -1) for s in g[0]['breaks'])
    assert _drift_axes(model, g[0]) == ['x', 'z']
    assert len(list((tmp_path/'guesses').glob('guess_*.nc'))) == 6
    assert (tmp_path/'guesses'/'guesses.csv').read_text().count('\n') == 7


class StandInSolver:
    """solve() returns preset 'converged' states, to test search_branches' bookkeeping."""
    def __init__(self, model, results):
        self.model, self.results, self.Tp, self.odir, self.calls = model, list(results), 0.2, '', []

    def solve(self, x0, **kw):
        self.calls.append(kw)
        x = self.results.pop(0)
        return x, True, 1e-10, np.linalg.norm(x), {}

    def unpack_xi(self, xi):
        return xi, self.Tp, 0, 0, 0


def test_search_branches_classifies_results(model, states, tmp_path):
    x, anti, a, b = states
    y = x + 0.3*anti                                    # a different state
    y_copy = model.apply_symmetry(y, Symmetry(sx=-1, sz=-1, ax=0.25))
    guesses = branch_guesses(model, x, [(2.0 + 0j, anti.astype(complex))], eps=(1e-2,))
    guesses.append({**guesses[0], 'kind': 'po', 'Tp': 0.6})
    solver = StandInSolver(model, [model.apply_symmetry(x, Symmetry(ax=0.1)), y, y_copy])
    res = search_branches(solver, x, guesses, odir=str(tmp_path/'search'))
    assert [r['status'] for r in res] == ['original', 'new', 'same as 1']
    assert solver.calls[0] == {'Tp': 0.2}
    assert solver.calls[2] == {'Tsearch': True, 'Tp': 0.6}
    assert (tmp_path/'search'/'branches.csv').exists()
