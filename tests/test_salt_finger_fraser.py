"""SaltFingerFraser (units of Fraser et al. 2025) against SaltFinger (Ra = 1) after the unit
conversion t = tau t_SF, u = u_SF/tau, T = T_SF/tau, S = (1 + eps) S_SF. Small 2-D box."""
import numpy as np
import pytest

dedalus = pytest.importorskip("dedalus.public")
from physics import SaltFinger, SaltFingerFraser                       # noqa: E402

PR, TAU, EPS = 5.0, 0.05, 0.5
L = 2*np.pi/0.6                       # about one fastest-growing finger wavelength (in d)
SIZES, BOUNDS = (16, 32), (L, 2*L)
DT = 0.1                              # SaltFinger time step (units d^2/kT)
T = 2.0                               # 20 steps


def from_saltfinger_state(x, eps, tau, dim=2):
    """SaltFinger state (u, T, S) -> SaltFingerFraser units: u/tau, T/tau, (1 + eps) S."""
    y = np.array(x, copy=True)
    n = y.size//(dim + 2)
    y[:(dim + 1)*n] /= tau
    y[(dim + 1)*n:] *= 1 + eps
    return y


def to_saltfinger_state(y, eps, tau, dim=2):
    """Inverse of from_saltfinger_state."""
    x = np.array(y, copy=True)
    n = x.size//(dim + 2)
    x[:(dim + 1)*n] *= tau
    x[(dim + 1)*n:] /= 1 + eps
    return x


def saltfinger(eps=EPS, **kw):
    m = SaltFinger({'Ra': 1.0, 'Pr': PR, 'tau': TAU, 'Rrho': 1/((1 + eps)*TAU), 'init_dt': DT},
                   sizes=SIZES, bounds=BOUNDS, **kw)
    m.build_problems()
    return m


def fraser(eps=EPS, **kw):
    m = SaltFingerFraser({'Sc': PR/TAU, 'tau': TAU, 'eps': eps, 'init_dt': TAU*DT},
                         sizes=SIZES, bounds=BOUNDS, **kw)
    m.build_problems()
    return m


@pytest.fixture(scope='module')
def x0():
    """A developed finger state in SaltFinger units."""
    m = saltfinger()
    x = 1e-3*np.random.default_rng(0).standard_normal(m.size())
    return m.flow_map(x, 400.0)


def test_parameters():
    m = SaltFingerFraser({'Pr': PR, 'tau': TAU, 'eps': EPS}, sizes=SIZES, bounds=BOUNDS)
    assert m.params['Sc'] == pytest.approx(PR/TAU) and 'Pr' not in m.params
    with pytest.raises(KeyError):
        SaltFingerFraser({'Sc': 100.0, 'tau': TAU}, sizes=SIZES, bounds=BOUNDS)


def test_state_conversion_round_trip(x0):
    y = from_saltfinger_state(x0, EPS, TAU, dim=2)
    assert np.allclose(to_saltfinger_state(y, EPS, TAU, dim=2), x0, rtol=0, atol=1e-14*np.abs(x0).max())


def test_flow_map_matches_saltfinger(x0):
    sf, fr = saltfinger(), fraser()
    xs = sf.flow_map(x0, T)
    xf = fr.flow_map(from_saltfinger_state(x0, EPS, TAU, dim=2), TAU*T)
    assert np.linalg.norm(xs - x0) > 1e-3*np.linalg.norm(x0)            # the state does evolve
    assert np.linalg.norm(to_saltfinger_state(xf, EPS, TAU, dim=2) - xs) < 1e-9*np.linalg.norm(xs)   # round-off (~1e-11)


def test_fluxes_match_saltfinger(x0):
    sf, fr = saltfinger(), fraser()
    sf.set_state(x0); ps = sf.get_flow_properties()
    fr.set_state(from_saltfinger_state(x0, EPS, TAU, dim=2)); pf = fr.get_flow_properties()
    assert pf['Fs'] == pytest.approx((1 + EPS)/TAU*ps['Fs'], rel=1e-12)
    assert pf['Ft'] == pytest.approx(ps['Ft']/TAU**2, rel=1e-12)
    for key in ('Nu', 'Sh'):
        assert pf[key] == pytest.approx(ps[key], rel=1e-12)
    assert pf['I']/pf['D'] == pytest.approx(ps['I']/ps['D'], rel=1e-10)


def test_set_param_eps_matches_new_model(x0):
    """Continuation in eps: set_param updates the equations in place."""
    y = from_saltfinger_state(x0, EPS, TAU, dim=2)
    m = fraser(eps=0.2)
    m.set_param('eps', EPS)
    assert np.allclose(m.flow_map(y, TAU*T), fraser().flow_map(y, TAU*T), rtol=0, atol=1e-13*np.abs(y).max())


def test_evp_growth_rates_scale_with_tau():
    """Motionless state: eigenvalues in units kS/d^2 are those of SaltFinger divided by tau."""
    zero = np.zeros(saltfinger(mode='ecs').size())
    ls, _ = saltfinger(mode='ecs').solve_evp(zero, N=4)
    lf, _ = fraser(mode='ecs').solve_evp(zero, N=4)
    assert ls[0].real > 0                                                 # fingers grow
    assert np.allclose(lf, ls/TAU, rtol=1e-8)


# --------------------------------------------------------------------------- reduced models
def reduced(model, eps=EPS, dt=TAU*DT, **kw):
    m = SaltFingerFraser({'Sc': PR/TAU, 'tau': TAU, 'eps': eps, 'model': model, 'init_dt': dt},
                         sizes=SIZES, bounds=BOUNDS, **kw)
    m.build_problems()
    return m


def test_model_names():
    assert reduced('mifsc').model_name == 'MIFSC'
    with pytest.raises(ValueError):
        reduced('quasilinear')
    with pytest.raises(KeyError):
        SaltFingerFraser({'Sc': 100.0, 'tau': TAU, 'eps': EPS, 'stokes': True}, sizes=SIZES, bounds=BOUNDS)


@pytest.mark.parametrize('model, flags', [('LPN', {'temperature_feedback': True}),
                                          ('IFSC', {'temperature_feedback': True, 'stokes': True})])
def test_lpn_ifsc_match_saltfinger_options(x0, model, flags):
    """'LPN' and 'IFSC' are SaltFinger's temperature_feedback (and stokes) options in other units."""
    sf = SaltFinger({'Ra': 1.0, 'Pr': PR, 'tau': TAU, 'Rrho': 1/((1 + EPS)*TAU), 'init_dt': DT, **flags},
                    sizes=SIZES, bounds=BOUNDS)
    sf.build_problems()
    xs = sf.flow_map(x0, T)
    xf = reduced(model).flow_map(from_saltfinger_state(x0, EPS, TAU, dim=2), TAU*T)
    assert np.linalg.norm(to_saltfinger_state(xf, EPS, TAU, dim=2) - xs) < 1e-9*np.linalg.norm(xs)


def _velocity(m, x):
    n = x.size//4
    return x[:2*n].reshape((2,) + tuple(m.get_grid_shape()))


@pytest.fixture(scope='module')
def mifsc_state(x0):
    """A MIFSC state with a mean flow: x0 plus a horizontal shear, integrated a little."""
    m = reduced('MIFSC')
    y = from_saltfinger_state(x0, EPS, TAU, dim=2)
    u = _velocity(m, y)
    z = np.arange(SIZES[1])*BOUNDS[1]/SIZES[1]
    u[0] += 0.3*np.abs(u).max()*np.sin(2*np.pi*z/BOUNDS[1])[None, :]
    return m, m.flow_map(y, TAU*T)


def test_ifsc_has_no_mean_flow(mifsc_state):
    m, y = mifsc_state
    x = reduced('IFSC').flow_map(y, TAU*T)
    U = _velocity(m, x)[0].mean(axis=0)
    assert np.abs(U).max() < 1e-10*np.abs(_velocity(m, x)).max()


def test_mifsc_fluctuations_are_inertia_free(mifsc_state):
    """MIFSC and IFSC differ only in the mean flow: one tiny IFSC step from a MIFSC state
    gives the same fluctuating velocity (determined by S alone)."""
    m, y = mifsc_state
    U = _velocity(m, y)[0].mean(axis=0)
    assert np.abs(U).max() > 1e-3*np.abs(_velocity(m, y)).max()           # there is a mean flow
    h = 1e-9
    x = reduced('IFSC', dt=h).flow_map(y, h)
    fluct = lambda v: v - v.mean(axis=1, keepdims=True)
    a, b = fluct(_velocity(m, y)), fluct(_velocity(m, x))
    assert np.linalg.norm(a - b) < 1e-6*np.linalg.norm(a)


def test_mifsc_mean_flow_equation(mifsc_state):
    """dt U = -dz <u' w'> + Sc dz^2 U, checked with one short step."""
    m, y = mifsc_state
    m.set_state(y)
    ex = m.coords.unit_vector_fields(m.dist)[0]
    rhs = ((-m._hmean(m.u @ dedalus.grad(m.u)) + PR/TAU*dedalus.lap(m._hmean(m.u))) @ ex).evaluate()
    rhs.change_scales(1)
    rhs = np.ravel(rhs.allgather_data('g').real)
    h = 1e-6
    x = reduced('MIFSC', dt=h).flow_map(y, h)
    dU = (_velocity(m, x)[0].mean(axis=0) - _velocity(m, y)[0].mean(axis=0))/h
    assert np.linalg.norm(dU - rhs) < 1e-3*np.linalg.norm(rhs)


def test_reduced_evp_growth_rates():
    """Motionless state: the fingers (k_perp != 0) of IFSC and MIFSC are the same; LPN is
    SaltFinger with temperature_feedback, divided by tau."""
    zero = np.zeros(reduced('IFSC', mode='ecs').size())
    li, _ = reduced('IFSC', mode='ecs').solve_evp(zero, N=4)
    lm, _ = reduced('MIFSC', mode='ecs').solve_evp(zero, N=4)
    assert li[0].real > 0 and np.allclose(lm, li, rtol=1e-8)
    sf = SaltFinger({'Ra': 1.0, 'Pr': PR, 'tau': TAU, 'Rrho': 1/((1 + EPS)*TAU), 'temperature_feedback': True},
                    sizes=SIZES, bounds=BOUNDS, mode='ecs')
    sf.build_problems()
    ls, _ = sf.solve_evp(zero, N=4)
    ll, _ = reduced('LPN', mode='ecs').solve_evp(zero, N=4)
    assert np.allclose(ll, ls/TAU, rtol=1e-8)
