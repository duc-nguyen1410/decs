"""
Find the symmetries of a state, e.g. of a solution converged without an imposed symmetry.

For every combination of reflections R (x -> -x, [y -> -y], z -> -z) the shift a along the
periodic directions minimising

    d(R, a) = || S(a) R x - x || / || x ||

is found: first on the grid by FFT cross-correlation (shifted_distance), then refined by Newton
iterations on the Fourier series of the correlation. The pair (R, a) is the Symmetry
s(sx, sy, sz, ax, ay, az) in the convention of ECSSolver.sigma (reflect, then shift), and d is
checked with model.apply_symmetry. Pure translations (R = identity) are tested for the
half-domain shifts only, since a = 0 is trivially a symmetry.
"""
import itertools
import logging
import numpy as np
from physics.symmetry import Symmetry
from .poincare import _blocks, shifted_distance

logger = logging.getLogger(__name__)


def _axis_names(model):
    return ('x', 'y', 'z') if model.dim == 3 else ('x', 'z')


def _refine_shift(model, x, y, shift_axes, a0, n_iter=20):
    """Newton iterations for the a maximising sum x(xi) y(xi + a L), started from a0 (dict)."""
    names = _axis_names(model)
    axes = [1 + names.index(n) for n in shift_axes]
    # C(m) = sum conj(X) Y over components / other axes;  corr(a) = Re sum_m C(m) e^{2 pi i m.a}
    C = sum((np.conj(np.fft.fftn(a, axes=axes))*np.fft.fftn(b, axes=axes))
            .sum(axis=tuple(i for i in range(a.ndim) if i not in axes))
            for a, b in zip(_blocks(model, x), _blocks(model, y)))
    m = np.meshgrid(*[np.fft.fftfreq(n, 1/n) for n in C.shape], indexing='ij')
    a = a_grid = np.array([a0['a' + n] for n in shift_axes])
    for _ in range(n_iter):
        e = C*np.exp(2j*np.pi*sum(mi*ai for mi, ai in zip(m, a)))
        grad = np.array([-(2*np.pi*mi*e).imag.sum() for mi in m])
        hess = np.array([[-(4*np.pi**2*mi*mj*e).real.sum() for mj in m] for mi in m])
        try:
            step = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        if np.any(np.abs(a + step - a_grid) > 1/np.array(C.shape)):  # left the grid cell
            break
        a = a + step
        if np.max(np.abs(step)) < 1e-15:
            break
    a = a - np.round(a)                                 # in [-1/2, 1/2]
    return {'a' + n: float(ai) for n, ai in zip(shift_axes, a)}


def _distance(model, x, sigma):
    return np.linalg.norm(model.apply_symmetry(x, sigma) - x)/np.linalg.norm(x)


def find_symmetries(model, x):
    """All reflection combinations with their best shift, sorted by distance (best first).

    Returns a list of dicts with
      'symmetry' : Symmetry with x ~ symmetry(x) (load into ECSSolver.sigma)
      'distance' : ||symmetry(x) - x|| / ||x||
      'center'   : shifts {'ax': ..} that move the state so that the reflected periodic
                   directions need no shift, i.e. x_c = Symmetry(**center)(x) satisfies
                   x_c ~ centered(x_c), centered = symmetry with those shifts set to 0
    """
    names = _axis_names(model)
    periodic = names[:-1] if model.bounded else names
    results = []
    for flips in itertools.product((1, -1), repeat=model.dim):
        refl = dict(zip(['s' + n for n in names], flips))
        reflected = [n for n in names if refl['s' + n] == -1]
        if not reflected:
            # pure translations: half-domain shifts along every subset of periodic directions
            for k in range(1, len(periodic) + 1):
                for sub in itertools.combinations(periodic, k):
                    sigma = Symmetry(**{'a' + n: 0.5 for n in sub})
                    results.append({'symmetry': sigma, 'distance': _distance(model, x, sigma),
                                    'center': {}})
            continue
        y = model.apply_symmetry(x, Symmetry(**refl))
        _, a0 = shifted_distance(model, x, y, periodic)
        shifts = _refine_shift(model, x, y, periodic, a0)
        sigma = Symmetry(**refl, **shifts)
        d = _distance(model, x, sigma)
        sigma0 = Symmetry(**refl, **a0)                  # keep the grid estimate if better
        d0 = _distance(model, x, sigma0)
        if d0 < d:
            sigma, d, shifts = sigma0, d0, a0
        # round shifts within round-off of 0 or 1/2 (e.g. 4.9999999999999994e-01)
        snapped = {k: np.round(2*v)/2 + 0.0 if abs(2*v - np.round(2*v)) < 1e-10 else v
                   for k, v in shifts.items()}
        sigma_s = Symmetry(**refl, **snapped)
        d_s = _distance(model, x, sigma_s)
        if d_s <= 2*d + 1e-14:
            sigma, d, shifts = sigma_s, d_s, snapped
        # x = S(a) R x  =>  x_c = S(b) x satisfies x_c = S(a + 2b) R x_c along reflected axes
        center = {'a' + n: -shifts['a' + n]/2 for n in reflected if 'a' + n in shifts}
        results.append({'symmetry': sigma, 'distance': d, 'center': center})
    results.sort(key=lambda r: r['distance'])
    return results


def reflections(model):
    """All combinations of reflections of the model's directions, identity first."""
    names = _axis_names(model)
    return [Symmetry(**{'s' + n: f for n, f in zip(names, flips)})
            for flips in itertools.product((1, -1), repeat=model.dim)]


def symmetry_distance(model, x, y):
    """min over reflections R and shifts a (periodic directions) of ||S(a) R y - x|| / ||x||.
    Returns (distance, sigma) with sigma(y) ~ x, e.g. to tell whether a converged solution x is
    a symmetric copy of a known one y. The shift is refined between grid points (Newton on the
    Fourier series of the correlation) and the distance computed at it with apply_symmetry."""
    names = _axis_names(model)
    periodic = names[:-1] if model.bounded else names
    best = (np.inf, None)
    for refl in reflections(model):
        yr = model.apply_symmetry(y, refl)
        d_grid, a0 = shifted_distance(model, x, yr, periodic)
        if d_grid > 2*best[0]:
            continue                                    # cannot beat the best after refinement
        refl_kw = dict(sx=refl.sx, sy=refl.sy, sz=refl.sz)
        for shifts in (_refine_shift(model, x, yr, periodic, a0), a0):
            sigma = Symmetry(**refl_kw, **shifts)
            d = _distance_to(model, x, y, sigma)
            if d < best[0]:
                best = (d, sigma)
    return best


def _distance_to(model, x, y, sigma):
    return np.linalg.norm(model.apply_symmetry(y, sigma) - x)/np.linalg.norm(x)


def nearest_solution(model, x, catalogue):
    """[(distance, name, sigma)] for the states of catalogue (dict name -> state vector),
    nearest first, with symmetry_distance(model, x, catalogue[name])."""
    found = [(*symmetry_distance(model, x, y), name) for name, y in catalogue.items()]
    return sorted([(d, name, sigma) for d, sigma, name in found], key=lambda r: r[0])
