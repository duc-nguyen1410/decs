"""Several symmetries imposed together (physics/symmetry.py, SymmetryGroup): the order and
commutation checks, .sym files with several lines, the projection on a 32^2 SaltFinger grid,
the solver map G = P S F^Tp, the unchanged single-Symmetry map, and the stability analysis,
which must run without any imposed symmetry. Run: python -m pytest tests"""
import numpy as np
import pytest

from physics.symmetry import Symmetry, SymmetryGroup, symmetry_order, as_symmetry

PIXZ = Symmetry(sx=-1, sz=-1)                       # (x, z) -> (-x, -z)
SHIFT_REFLECT = Symmetry(sz=-1, ax=0.5, az=0.5)     # (x, z) -> (x + Lx/2, 1/2 - z)


# ---------------------------------------------------------------- pure python ---------------
def test_symmetry_order():
    assert symmetry_order(Symmetry()) == 1
    assert symmetry_order(PIXZ) == 2
    assert symmetry_order(SHIFT_REFLECT) == 2
    assert symmetry_order(Symmetry(ax=0.5)) == 2
    assert symmetry_order(Symmetry(ax=0.25)) == 4
    assert symmetry_order(Symmetry(sx=-1, ax=0.25)) == 2       # shift along a reflected axis
    assert symmetry_order(Symmetry(rz=1, az=0.25)) == 4        # screw
    assert symmetry_order(Symmetry(rz=1, az=0.125)) == 8
    assert symmetry_order(Symmetry(ay=0.25), dim=2) == 1       # no y in 2-D
    assert symmetry_order(Symmetry(az=0.25), bounded=True) == 1


def test_commutation_check():
    SymmetryGroup(PIXZ, SHIFT_REFLECT).check_commuting(dim=2)
    with pytest.raises(ValueError, match="do not commute"):
        SymmetryGroup(Symmetry(sx=-1), Symmetry(ax=0.25)).check_commuting(dim=2)


def test_load_multi_line_file(tmp_path):
    f = tmp_path / 'eq1.sym'
    f.write_text("% 2\n1 -1 1 -1 0 0 0\n# shift-reflect\n1 1 1 -1 0.5 0 0.5\n")
    g = SymmetryGroup().load_from_file(str(f))
    assert len(g) == 2 and g.is_nontrivial()
    assert str(g) == "[1.0 -1.0 1.0 -1.0 0.0 0.0 0.0] [1.0 1.0 1.0 -1.0 0.5 0.0 0.5]"
    with pytest.raises(ValueError, match="SymmetryGroup"):      # a plain Symmetry refuses it
        Symmetry().load_from_file(str(f))
    # several single-line files add up
    a, b = tmp_path / 'a.sym', tmp_path / 'b.sym'
    a.write_text("1 -1 1 -1 0 0 0")
    b.write_text("1 1 1 -1 0.5 0 0.5\n")
    assert str(SymmetryGroup().load_from_file(str(a), str(b))) == str(g)


def test_single_symmetry_file_unchanged(tmp_path):
    f = tmp_path / 'pixz.sym'
    f.write_text("1 -1 1 -1 0 0 0")                 # no trailing newline, as the shipped files
    s = Symmetry()
    s.load_from_file(str(f))
    assert (s.sx, s.sy, s.sz, s.ax, s.az, s.rz) == (-1, 1, -1, 0, 0, 0)
    assert str(s) == "[1.0 -1.0 1.0 -1.0 0.0 0.0 0.0]"


def test_as_symmetry():
    assert isinstance(as_symmetry([PIXZ, SHIFT_REFLECT]), SymmetryGroup)
    assert as_symmetry(PIXZ) is PIXZ
    assert not as_symmetry(None).is_nontrivial()
    assert not SymmetryGroup().is_nontrivial() and not SymmetryGroup(Symmetry()).is_nontrivial()


# ---------------------------------------------------------------- with Dedalus --------------
PARAMS = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'init_dt': 2e-4, 'ext': '.nc'}
N = 32


@pytest.fixture(scope='module')
def model():
    pytest.importorskip("dedalus.public")
    from physics import SaltFinger
    m = SaltFinger(PARAMS, sizes=(N, N), bounds=(0.6, 1.0))
    return m


@pytest.fixture(scope='module')
def state(model):
    return model.flow_map(1e-2*np.random.default_rng(5).standard_normal(model.size()), 0.004)


@pytest.fixture
def solver(model, tmp_path):
    from ecs_core import ECSSolver
    return ECSSolver(model, {'odir': str(tmp_path) + '/', 'save_log': False})


def test_projection_is_invariant_under_each_generator(model, state):
    g = SymmetryGroup(PIXZ, SHIFT_REFLECT)
    p = g.impose(model, state)
    norm = np.linalg.norm(p)
    assert norm > 0.1*np.linalg.norm(state)                    # not projected to zero
    for s in g:
        assert np.linalg.norm(model.apply_symmetry(p, s) - p) < 1e-12*norm
    assert np.linalg.norm(model.apply_symmetry(p, Symmetry(sx=-1, ax=0.5, az=0.5)) - p) < 1e-12*norm
    assert np.allclose(g.impose(model, p), p, atol=1e-13*norm)  # P^2 = P
    # P = (1 + s1 + s2 + s1 s2)/4, the average over the group
    s1 = model.apply_symmetry(state, PIXZ)
    s2 = model.apply_symmetry(state, SHIFT_REFLECT)
    s12 = model.apply_symmetry(s2, PIXZ)
    assert np.allclose(p, 0.25*(state + s1 + s2 + s12), atol=1e-13*norm)


def test_group_return_map_projects(solver, model, state):
    solver.sigma = [PIXZ, SHIFT_REFLECT]                     # a list becomes a SymmetryGroup
    assert isinstance(solver.sigma, SymmetryGroup)
    phi = solver.return_map(state, 0.002)
    ref = solver.sigma.impose(model, model.flow_map(state, 0.002))
    assert np.allclose(phi, ref, atol=1e-13*np.linalg.norm(ref))


def test_single_symmetry_return_map_unchanged(solver, model, state):
    solver.sigma = Symmetry(sx=-1, sz=-1)
    assert isinstance(solver.sigma, Symmetry)
    phi = solver.return_map(state, 0.002, ax=0.1)
    ref = model.apply_symmetry(model.apply_symmetry(model.flow_map(state, 0.002), Symmetry(ax=0.1)), PIXZ)
    assert np.array_equal(phi, ref)                             # sigma x, not a projection


def test_header_lists_all_symmetries(solver):
    from ecs_core import output
    solver.sigma = SymmetryGroup(PIXZ, SHIFT_REFLECT)
    lines = []

    class Log:
        def write(self, text):
            lines.append(text)
    output.RunLog.write_header(Log(), solver)
    text = "".join(lines)
    assert f"symmetry group: {PIXZ} {SHIFT_REFLECT}" in text


def test_stability_runs_without_imposed_symmetries(solver, model, state, monkeypatch):
    from ecs_core import stability as stab
    group = SymmetryGroup(PIXZ, SHIFT_REFLECT)
    solver.sigma = group
    seen = {}

    def fake_arnoldi(s, xi):
        seen['sigma'] = s.sigma
        x = xi[:model.size()]
        seen['map'] = s.return_map(x, 0.002)
        lam = np.array([-1.0 + 0j])
        return lam, np.exp(lam*0.002), np.zeros((model.size(), 1)), None

    monkeypatch.setattr(stab, '_stability_arnoldi', fake_arnoldi)
    monkeypatch.setattr(solver, 'residual', lambda xi: np.zeros(1))
    solver.stability_method = 'arnoldi'
    stab.stability(solver, solver.pack_xi(state, 0.002))
    assert not seen['sigma'].is_nontrivial()                    # no constraint on perturbations
    assert np.array_equal(seen['map'], model.flow_map(state, 0.002))
    assert solver.sigma is group                                # restored afterwards
