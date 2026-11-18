"""Rotation about z (physics/rotation.py, Symmetry(rz=...)) and helical tools (tools/helical.py).
The first tests are pure numpy; the others need Dedalus and use a small 3-D SaltFinger box."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

# load physics/rotation.py on its own: importing the physics package needs Dedalus
_spec = importlib.util.spec_from_file_location(
    'decs_rotation', Path(__file__).resolve().parent.parent/'physics'/'rotation.py')
rotation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rotation)
rotate_z_grid = rotation.rotate_z_grid

N, NZ, L = 8, 6, 0.5


def grid():
    x = np.arange(N)*L/N
    z = np.arange(NZ)/NZ
    return np.meshgrid(x, x, z, indexing='ij')


def rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


# --------------------------------------------------------------------------- numpy only
@pytest.mark.parametrize('vector', [False, True])
def test_four_quarter_turns_are_identity(vector):
    f = rand(3, N, N, NZ) if vector else rand(N, N, NZ)
    assert np.array_equal(rotate_z_grid(f, 4, vector), f)
    assert np.array_equal(rotate_z_grid(rotate_z_grid(f, 1, vector), 3, vector), f)
    assert np.array_equal(rotate_z_grid(f, -1, vector), rotate_z_grid(f, 3, vector))


def test_half_turn_is_point_reflection():
    """R_90^2 is (x, y) -> (-x, -y), with u and v changing sign: the reflections sx = sy = -1."""
    f = rand(3, N, N, NZ)
    perm = (-np.arange(N)) % N
    expected = f[:, perm][:, :, perm]*np.array([-1, -1, 1])[:, None, None, None]
    assert np.array_equal(rotate_z_grid(f, 2, vector=True), expected)


def test_rotation_of_analytic_fields():
    """[R q](x) = Q q(Q^-1 x), Q (x, y) = (-y, x), Q^-1 (x, y) = (y, -x)."""
    X, Y, Z = grid()
    k = 2*np.pi/L
    s = lambda x, y, z: np.sin(k*x + 0.3)*np.cos(2*k*y) + np.cos(2*np.pi*z)*np.sin(k*y)
    # scalar
    assert np.allclose(rotate_z_grid(s(X, Y, Z), 1), s(Y, -X, Z), atol=1e-13)
    # vector u = (a, b, c): u' = (-b, a, c) at Q^-1 x
    a = lambda x, y, z: np.sin(k*x)*np.cos(k*y + 0.1)
    b = lambda x, y, z: np.cos(2*k*x)
    c = lambda x, y, z: np.sin(k*y)*np.cos(2*np.pi*z)
    u = np.stack([a(X, Y, Z), b(X, Y, Z), c(X, Y, Z)])
    expected = np.stack([-b(Y, -X, Z), a(Y, -X, Z), c(Y, -X, Z)])
    assert np.allclose(rotate_z_grid(u, 1, vector=True), expected, atol=1e-13)


def test_rotation_turns_a_uniform_flow():
    u = np.zeros((3, N, N, NZ))
    u[0] = 1.0                                   # uniform flow along x
    r = rotate_z_grid(u, 1, vector=True)
    assert np.allclose(r[0], 0) and np.allclose(r[1], 1) and np.allclose(r[2], 0)


def test_rotation_needs_square_grid():
    with pytest.raises(ValueError):
        rotate_z_grid(rand(N, N + 2, NZ), 1)


# --------------------------------------------------------------------------- with Dedalus
PARAMS = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'init_dt': 2e-4}
T = 0.002            # 10 steps of init_dt


@pytest.fixture(scope='module')
def model():
    pytest.importorskip("dedalus.public")
    from physics import SaltFinger
    m = SaltFinger(PARAMS, sizes=(16, 16, 16), bounds=(L, L, 1.0))
    m.build_problems()
    return m


@pytest.fixture(scope='module')
def smooth(model):
    return model.flow_map(1e-2*np.random.default_rng(2).standard_normal(model.size()), T)


def test_symmetry_rz_matches_reflections(model, smooth):
    from physics import Symmetry
    x = smooth
    r2 = model.apply_symmetry(model.apply_symmetry(x, Symmetry(rz=1)), Symmetry(rz=1))
    assert np.allclose(r2, model.apply_symmetry(x, Symmetry(sx=-1, sy=-1)), atol=1e-13)
    assert np.allclose(model.apply_symmetry(x, Symmetry(rz=4)), x)       # rz is taken mod 4


def test_flow_map_commutes_with_rotation(model, smooth):
    """The equations are invariant under R_90: F(R x) = R F(x)."""
    from physics import Symmetry
    R = Symmetry(rz=1)
    a = model.flow_map(model.apply_symmetry(smooth, R), T)
    b = model.apply_symmetry(model.flow_map(smooth, T), R)
    assert np.linalg.norm(a - b) < 1e-10*np.linalg.norm(b)


def test_screw_projection_is_invariant_and_idempotent(model, smooth):
    from tools import screw_symmetry
    sigma = screw_symmetry(+1)
    xs = model.symmetrize(smooth, sigma, 4)
    assert np.linalg.norm(model.apply_symmetry(xs, sigma) - xs) < 1e-12*np.linalg.norm(xs)
    assert np.allclose(model.symmetrize(xs, sigma, 4), xs, atol=1e-13)


@pytest.mark.parametrize('h', [1, -1])
def test_screw_projection_turns_planar_shear_into_helix(model, h):
    """A planar shear sin(2 pi z) e_x projects onto a helix of half its amplitude, with the
    handedness of the screw; the diagnostics see it."""
    from tools import screw_symmetry, helical_diagnostics
    shape = model.get_grid_shape()
    Z = np.broadcast_to(np.arange(shape[2])/shape[2], shape)
    u = np.zeros((3,) + shape)
    u[0] = np.sin(2*np.pi*Z)
    x = np.concatenate([u.ravel(), np.zeros(2*u[0].size)])
    sigma = screw_symmetry(h)
    xs = model.symmetrize(x, sigma, 4)
    us = xs[:u.size].reshape(u.shape)
    expected = 0.5*np.stack([np.sin(2*np.pi*Z), h*np.cos(2*np.pi*Z)])
    assert np.allclose(us[:2], expected, atol=1e-13) and np.allclose(us[2], 0, atol=1e-13)
    d = helical_diagnostics(model, xs, sigma)
    assert d['winding_number'] == pytest.approx(-h)
    assert d['planar_fraction'] == pytest.approx(0, abs=1e-12)
    assert np.sign(d['mean_flow_helicity']) == h
    assert d['helicity'] == pytest.approx(d['mean_flow_helicity'])
    assert d['Hrel_mean'] == pytest.approx(h)        # a single-mode helix is Beltrami
    assert np.isnan(d['Hrel_fluct'])                 # no fluctuations
    assert d['screw_distance'] < 1e-13
    # the planar shear itself: no winding, planar, not screw symmetric
    d0 = helical_diagnostics(model, x, sigma)
    assert d0['winding_number'] == pytest.approx(0, abs=1e-12)
    assert d0['planar_fraction'] == pytest.approx(1)
    assert d0['screw_distance'] > 0.5


def test_grid_screw_matches_apply_symmetry(model, smooth):
    """The numpy screw (usable in 'sim' mode) equals model.apply_symmetry with screw_symmetry."""
    from tools import screw_symmetry, screw_image_grid, screw_project_grid
    shape = model.get_grid_shape()
    for h in (1, -1):
        a = screw_image_grid(smooth, shape, h)
        b = model.apply_symmetry(smooth, screw_symmetry(h))
        assert np.allclose(a, b, atol=1e-13)
        assert np.allclose(screw_project_grid(smooth, shape, h),
                           model.symmetrize(smooth, screw_symmetry(h), 4), atol=1e-13)


def test_best_screw_center_finds_shifted_axis(model, smooth):
    """A screw-symmetric state moved off the axis is found again by best_screw_center."""
    from tools import screw_project_grid, shift_grid, best_screw_center
    shape = model.get_grid_shape()
    xs = screw_project_grid(smooth, shape, 1)
    moved = shift_grid(xs, shape, 3, 5.5)
    sx, sy, r = best_screw_center(moved, shape, 1)
    assert r == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(shift_grid(moved, shape, sx, sy), screw_project_grid(shift_grid(moved, shape, sx, sy), shape, 1), atol=1e-12)


def test_c2x_grid_matches_reflections_and_group(model, smooth):
    """C2x equals Symmetry(sy=-1, sz=-1), commutes with the flow, and C2x sigma C2x = sigma^-1,
    so the group projection is invariant under both."""
    from physics import Symmetry
    from tools import c2x_image_grid, screw_image_grid, helical_group_project_grid
    shape = model.get_grid_shape()
    assert np.allclose(c2x_image_grid(smooth, shape), model.apply_symmetry(smooth, Symmetry(sy=-1, sz=-1)), atol=1e-13)
    a = c2x_image_grid(screw_image_grid(c2x_image_grid(smooth, shape), shape, 1), shape)
    b = screw_image_grid(screw_image_grid(screw_image_grid(smooth, shape, 1), shape, 1), shape, 1)
    assert np.allclose(a, b, atol=1e-13)
    xg = helical_group_project_grid(smooth, shape, 1)
    assert np.allclose(c2x_image_grid(xg, shape), xg, atol=1e-13)
    assert np.allclose(screw_image_grid(xg, shape, 1), xg, atol=1e-13)
    f = model.flow_map(xg, T)
    assert np.linalg.norm(c2x_image_grid(f, shape) - f) < 1e-10*np.linalg.norm(f)
