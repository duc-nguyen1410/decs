"""Single-mode magnetoconvection (physics/magneto.py, BoundedQuasiStaticMagnetoConvection_SingleMode):
the state vector and the state files keep the complex mode amplitudes, and x_derivative is the phase
rotation of the amplitudes. Run: python -m pytest tests"""
import numpy as np
import pytest

de = pytest.importorskip("dedalus.public")
from physics import BoundedQuasiStaticMagnetoConvection_SingleMode as SingleMode

PARAMS = {'Ra': 2.0, 'Pr': 1.0, 'Q': 1e6, 'kx': 8*np.pi, 'ky': 0.0, 'scale_Ra_c': True,
          'stress-free': True, 'init_dt': 0.005}
NZ = 32


@pytest.fixture(scope='module')
def model():
    m = SingleMode(PARAMS, sizes=(NZ,), bounds=(1.0,), bounded=True)
    m.use_CFL = False
    return m


def set_fields(model, seed=0):
    """Complex amplitudes u, v, w, te and real means U0, V0, T0 with all parts nonzero."""
    rng = np.random.default_rng(seed)
    for f in model.fields:
        f.change_scales(1)
        if f.name in model.complex_fields:
            f['g'] = rng.standard_normal(NZ) + 1j*rng.standard_normal(NZ)
        else:
            f['g'] = rng.standard_normal(NZ)
    return {f.name: f['g'].copy() for f in model.fields}


def test_state_keeps_complex_amplitudes(model):
    fields = set_fields(model)
    x = model.get_state()
    assert x.size == model.size() == NZ*(2*4 + 3)
    model.set_state(np.zeros_like(x))
    model.set_state(x)
    for f in model.fields:
        f.change_scales(1)
        assert np.allclose(f['g'], fields[f.name], rtol=0, atol=1e-14)


@pytest.mark.parametrize('ext', ['.nc', '.h5'])
def test_save_load_keeps_complex_amplitudes(model, tmp_path, ext):
    set_fields(model, seed=1)
    x = model.get_state()
    model.ext = ext
    model.save_state(str(tmp_path / 'state'))
    model.set_state(np.zeros_like(x))
    model.load_state(str(tmp_path / 'state'))
    assert np.allclose(model.get_state(), x, rtol=0, atol=1e-14)


def test_x_derivative_is_phase_rotation(model):
    """A shift by a in x multiplies each amplitude by exp(i kx a) and leaves the means."""
    set_fields(model, seed=2)
    x = model.get_state()
    a = 1e-5

    def shifted(a):
        model.set_state(x)
        for f in model.fields:
            if f.name in model.complex_fields:
                f.change_scales(1)
                f['g'] = f['g']*np.exp(1j*PARAMS['kx']*a)
        return model.get_state()
    central = (shifted(a) - shifted(-a))/(2*a)          # error ~ (kx a)^2 kx |x| / 6
    assert np.allclose(model.x_derivative(x), central, rtol=0, atol=1e-6*np.abs(central).max())


def test_flow_map_matches_step_loop(model):
    """flow_map (through get_state / set_state) gives the plain solver.step result."""
    z, = model.dist.local_grids(model.bases[0])
    for f in model.fields:
        f.change_scales(1)
        f['g'] = 0
    model.te['g'] = 1e-3*np.sin(np.pi*z)*(1 + 0.5j)
    x0 = model.get_state()
    solver = model.ivp_problem.build_solver(de.RK222)
    for _ in range(40):
        solver.step(PARAMS['init_dt'])
    stepped = model.get_state()
    assert np.isfinite(stepped).all() and np.abs(stepped - x0).max() > 0
    assert np.allclose(model.flow_map(x0, 40*PARAMS['init_dt']), stepped, rtol=0, atol=1e-13)


def test_set_param_Ra_matches_model_built_at_that_Ra():
    """set_param('Ra', value) takes Ra in the units of the params (Ra/Ra_c with scale_Ra_c), like the
    full model: the flow map after set_param equals that of a model built at the new Ra."""
    def build(Ra):
        m = SingleMode(dict(PARAMS, Ra=Ra), sizes=(NZ,), bounds=(1.0,), bounded=True)
        m.use_CFL = False
        return m
    a, b = build(2.0), build(3.0)
    a.set_param('Ra', 3.0)
    z, = b.dist.local_grids(b.bases[0])
    for m in (a, b):
        for f in m.fields:
            f.change_scales(1)
            f['g'] = 0
        m.te['g'] = 1e-3*np.sin(np.pi*z)*(1 + 0.5j)
    x0 = b.get_state()
    ya, yb = a.flow_map(x0, 20*PARAMS['init_dt']), b.flow_map(x0, 20*PARAMS['init_dt'])
    assert np.isfinite(ya).all()
    assert np.allclose(ya, yb, rtol=0, atol=1e-13)
