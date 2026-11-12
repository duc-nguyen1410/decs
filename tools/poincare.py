"""
Poincare sections and recurrence search: initial guesses for periodic orbits (PO) and
relative periodic orbits (RPO).

PoincareSection records the crossings g(x(t)) = level of a trajectory, with g an observable
of the current model state (e.g. the cross-flow energy) and crossings counted only in one
direction (direction=+1: g increasing). A crossing is detected between two checks and then
refined: integrate from the last state before it with a time step found by regula falsi on g.

recurrence_search compares states (e.g. the section crossings) pairwise, minimised over
shifts in the periodic directions,
    d_ij = min_a || S(a) x_j - x_i || / || x_i ||,      S(a) f(x) = f(x + a L),
which is the residual convention of ECSSolver (x0 = S(a) F^Tp(x0)). The best pairs give
Tp = t_j - t_i and the shifts (ax, ay, az) as starting values for ECSSolver.solve.
"""
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PoincareSection:
    """Crossings g(x) = level of trajectories of a model.

    observable : function(model) -> float of the CURRENT model state, e.g.
                 lambda m: m.cross_flow_energy(). It is called between time steps, so it must
                 not change the layout of model.fields (evaluate on field copies).
    direction  : +1 (g increasing), -1 (g decreasing) or 0 (both).
    check_every: test for a crossing every this many time steps.
    refine     : refine each crossing to |g - level| <= tol*max(|level|, |g jump|), else
                 interpolate linearly between the two checks around it.
    """

    def __init__(self, model, observable, level=0.0, direction=1, check_every=1,
                 refine=True, tol=1e-10, max_refine=20):
        if direction not in (-1, 0, 1):
            raise ValueError("direction must be +1, -1 or 0")
        self.model = model
        self.observable = observable
        self.level = level
        self.direction = direction
        self.check_every = check_every
        self.refine = refine
        self.tol = tol
        self.max_refine = max_refine

    def _crossed(self, g0, g1):
        up = g0 < self.level <= g1
        down = g0 > self.level >= g1
        return {1: up, -1: down, 0: up or down}[self.direction]

    def run(self, x0, T, max_crossings=None, odir=None):
        """Integrate x0 over time T (stop after max_crossings) and return the crossings as a
        list of dicts {'t': time, 'x': state, 'g': observable}. With odir, also write
        odir/crossings.csv and the states odir/crossing_<i>. The state at the end of the run
        (time T, or when max_crossings stopped it) is kept in self.final_state."""
        brackets = []                       # (t0, x0, g0, t1, x1, g1) around each crossing
        prev = {'t': 0.0, 'x': np.array(x0, dtype=float)}
        self.model.set_state(prev['x'])
        prev['g'] = self.observable(self.model)

        def callback(solver):
            if solver.iteration % self.check_every:
                return False
            t, g, x = solver.sim_time, self.observable(self.model), self.model.current_state()
            if self._crossed(prev['g'], g):
                brackets.append((prev['t'], prev['x'], prev['g'], t, x, g))
            prev.update(t=t, x=x, g=g)
            return max_crossings is not None and len(brackets) >= max_crossings

        self.final_state = self.model.flow_map(x0, T, callback)   # before refining moves the state
        crossings = [self._locate(*b) for b in brackets]
        logger.info(f"Poincare section: {len(crossings)} crossings in t <= {T}")
        if odir is not None:
            self.save(crossings, odir)
        return crossings

    def _locate(self, t0, x0, g0, t1, x1, g1):
        """Crossing between (t0, x0) and (t1, x1)."""
        c = self.level
        if not self.refine:
            s = (c - g0)/(g1 - g0)
            return {'t': t0 + s*(t1 - t0), 'x': x0 + s*(x1 - x0), 'g': c}
        # regula falsi (Illinois) on tau in [0, t1 - t0], integrating from x0
        a, ga, b, gb = 0.0, g0 - c, t1 - t0, g1 - c
        scale = self.tol*max(abs(c), abs(g1 - g0))
        tau, x, g, side = b, x1, gb, 0
        for _ in range(self.max_refine):
            tau = (a*gb - b*ga)/(gb - ga)
            x = self.model.flow_map(x0, tau)
            g = self.observable(self.model) - c
            if abs(g) <= scale:
                break
            if np.sign(g) == np.sign(ga):
                a, ga = tau, g
                if side == -1:
                    gb *= 0.5
                side = -1
            else:
                b, gb = tau, g
                if side == 1:
                    ga *= 0.5
                side = 1
        else:
            logger.warning(f"Crossing near t = {t0 + tau:.6g} not refined to tol "
                           f"(|g - level| = {abs(g):.3e})")
        return {'t': t0 + tau, 'x': x, 'g': g + c}

    def save(self, crossings, odir):
        """crossings.csv (index, t, g) and the states crossing_<i> in odir."""
        if self.model.dist.comm.rank == 0:
            os.makedirs(odir, exist_ok=True)
            with open(os.path.join(odir, 'crossings.csv'), 'w') as f:
                f.write("i, t, g\n")
                for i, cr in enumerate(crossings):
                    f.write(f"{i}, {cr['t']:.15e}, {cr['g']:.15e}\n")
        for i, cr in enumerate(crossings):
            self.model.save_state(os.path.join(odir, f'crossing_{i}'), cr['x'])


def _blocks(model, x):
    """State vector -> list of arrays (components, *grid_shape), one per field."""
    shape = tuple(model.get_grid_shape())
    n = int(np.prod(shape))
    out, cursor = [], 0
    for field in model.fields:
        k = model.dim if len(field.tensorsig) > 0 else 1
        out.append(x[cursor:cursor + k*n].reshape((k,) + shape))
        cursor += k*n
    return out


def shifted_distance(model, x, y, shift_axes=('x',)):
    """min over grid shifts a of ||S(a) y - x|| / ||x|| along the periodic shift_axes.
    Returns (distance, {'ax': .., 'az': ..}), the shifts as fractions of the domain length,
    refined between grid points by a parabola through the minimum."""
    names = ('x', 'y', 'z') if model.dim == 3 else ('x', 'z')
    if model.bounded and names[-1] in shift_axes:
        raise ValueError(f"cannot shift along the bounded direction {names[-1]}")
    shift_axes = sorted(shift_axes, key=names.index)
    axes = [1 + names.index(name) for name in shift_axes]   # +1: component axis first
    xb, yb = _blocks(model, x), _blocks(model, y)
    # sum_xi x(xi) y(xi + k) for all grid shifts k (circular cross-correlation)
    corr = sum(np.fft.ifftn(np.conj(np.fft.fftn(a, axes=axes))*np.fft.fftn(b, axes=axes),
                            axes=axes).real.sum(axis=tuple(i for i in range(a.ndim) if i not in axes))
               for a, b in zip(xb, yb))
    d2 = np.dot(x, x) + np.dot(y, y) - 2*corr
    k = np.unravel_index(np.argmin(d2), d2.shape)
    shifts = {}
    for dim_i, (name, n) in enumerate(zip(shift_axes, d2.shape)):
        km, kp = list(k), list(k)
        km[dim_i], kp[dim_i] = (k[dim_i] - 1) % n, (k[dim_i] + 1) % n
        fm, f0, fp = d2[tuple(km)], d2[k], d2[tuple(kp)]
        den = fm - 2*f0 + fp
        frac = 0.5*(fm - fp)/den if den > 0 else 0.0
        a = (k[dim_i] + frac)/n
        shifts['a' + name] = a - np.round(a)        # in [-1/2, 1/2]
    # distance at the best grid shift computed directly (d2 above loses half the digits)
    diff = [np.roll(b, [-ki for ki in k], axis=axes) - a for a, b in zip(xb, yb)]
    return np.sqrt(sum(np.sum(dd**2) for dd in diff))/np.linalg.norm(x), shifts


def recurrence_search(model, states, times, shift_axes=(), T_min=0.0, T_max=np.inf, n_best=5):
    """Near-recurrences among states at times (e.g. Poincare crossings): the n_best pairs
    j > i with T_min <= t_j - t_i <= T_max, sorted by d_ij = min_a ||S(a) x_j - x_i|| / ||x_i||.
    Multiples of a period recur as well as the period itself: bound the search with T_max.
    shift_axes: periodic directions to minimise over, () for periodic orbits, ('x',) for
    RPO drifting in x. Returns dicts {'i', 'j', 'Tp', 'distance', 'ax', ...}."""
    found = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            Tp = times[j] - times[i]
            if not T_min <= Tp <= T_max:
                continue
            if shift_axes:
                d, shifts = shifted_distance(model, states[i], states[j], shift_axes)
            else:
                d, shifts = np.linalg.norm(states[j] - states[i])/np.linalg.norm(states[i]), {}
            found.append({'i': i, 'j': j, 'Tp': Tp, 'distance': d, **shifts})
    found.sort(key=lambda r: r['distance'])
    return found[:n_best]
