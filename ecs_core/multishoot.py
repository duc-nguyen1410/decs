"""
Multiple shooting for periodic orbits (PO) and relative periodic orbits (RPO).

The period is split into m segments of time tau = Tp/m, each with its own initial state:

Unknowns
--------
    xi = [x_1, ..., x_m, (Tp), (ax), (ay), (az)]

Equations
---------
    r_i = x_{i+1} - F^tau(x_i)                       i = 1 .. m-1
    r_m = x_1     - sigma S(ax, ay, az) F^tau(x_m)
plus the phase conditions of ECSSolver, on x_1. The Jacobian is block-bidiagonal; one
Jacobian-vector product integrates every segment once (one period in total), as in single
shooting. Perturbations grow over tau instead of Tp, which conditions long or strongly
unstable orbits better.

x_1 is "the solution": saving, flow properties, stability (Arnoldi on the full period map)
and time histories work as in ECSSolver. With m = 1 the solver is ECSSolver exactly.
"""
import numpy as np
from .newton import ECSSolver


class MultiShootECSSolver(ECSSolver):
    """ECSSolver with multiple shooting. params: those of ECSSolver plus
    'n_segments' (m, default 2). The initial segments are built by integrating x0."""

    def __init__(self, model, params=None):
        super().__init__(model, params)
        self.n_segments = (params or {}).get('n_segments', 2)

    # ------------------------------------------------------------------
    # Unknowns xi = [x_1, ..., x_m, (Tp), (ax), (ay), (az)]
    # ------------------------------------------------------------------
    def unpack_xi(self, xi):
        """xi -> (x_1, Tp, ax, ay, az); parameters not searched take their current value."""
        N, m = self.model.size(), self.n_segments
        values = {'Tp': self.Tp, 'ax': self.ax, 'ay': self.ay, 'az': self.az}
        for i, name in enumerate(self._searched()):
            values[name] = xi[m*N + i]
        return xi[:N], values['Tp'], values['ax'], values['ay'], values['az']

    def segments(self, xi):
        """The m segment states of xi, shape (m, N)."""
        N, m = self.model.size(), self.n_segments
        return xi[:m*N].reshape(m, N)

    def pack_xi(self, x, Tp, ax=0.0, ay=0.0, az=0.0):
        """xi from x_1 (the other segments by integrating it over Tp/m each) or from all m
        segments concatenated, plus the searched parameters."""
        N, m = self.model.size(), self.n_segments
        x = np.asarray(x, dtype=float)
        if x.size == N:
            segs = [x]
            for _ in range(m - 1):
                segs.append(self.model.flow_map(segs[-1], Tp/m))
            x = np.concatenate(segs)
        elif x.size != m*N:
            raise ValueError(f"x has size {x.size}; expected {N} (x_1) or {m*N} (all segments)")
        values = {'Tp': Tp, 'ax': ax, 'ay': ay, 'az': az}
        return np.concatenate([x, [values[name] for name in self._searched()]])

    # ------------------------------------------------------------------
    # Maps
    # ------------------------------------------------------------------
    def _segment_map(self, x, i, tau, ax, ay, az):
        """F^tau(x); for the last segment sigma S(ax, ay, az) F^tau(x)."""
        if i == self.n_segments - 1:
            return self.return_map(x, tau, ax, ay, az)
        return self.model.flow_map(x, tau)

    def _segment_derivative(self, x, dx, phi, i, tau, ax, ay, az):
        """Finite-difference derivative of segment map i at x along dx, phi = its map at x."""
        norm_v = np.linalg.norm(dx)
        if norm_v == 0:
            return np.zeros_like(dx)
        epsilon = self.eps_x/norm_v
        return (self._segment_map(x + epsilon*dx, i, tau, ax, ay, az) - phi)/epsilon

    def _evaluate(self, xi):
        """(residual vector, segment end states (m, N)); phase-condition rows are zero."""
        N, m = self.model.size(), self.n_segments
        segs = self.segments(xi)
        _, Tp, ax, ay, az = self.unpack_xi(xi)
        tau = Tp/m
        phi = np.empty((m, N))
        for i in range(m):
            phi[i] = self._segment_map(segs[i], i, tau, ax, ay, az)
        res = np.zeros_like(xi)
        res[:m*N] = (-phi + np.roll(segs, -1, axis=0)).ravel()
        return res, phi

    def _jacobian(self, xi, phi_base):
        """Return matvec(v), v = [dx_1, ..., dx_m, dTp, dax, ...] (sign as in ECSSolver):
            block i : DF_i dx_i - dx_{i+1} + f(phi_i)/m dTp  (+ dG/da da in the last block)
            row     : <d_p x_1, dx_1>   (phase condition of each searched parameter)."""
        N, m = self.model.size(), self.n_segments
        segs = self.segments(xi)
        x1, Tp, ax, ay, az = self.unpack_xi(xi)
        tau = Tp/m
        tangents = []
        for name in self._searched():
            if name == 'Tp':          # every segment time depends on Tp
                d_phi = [self._derivative('Tp', phi_base[i])/m for i in range(m)]
            else:                     # the shifts act on the last segment only
                d_phi = [None]*(m - 1) + [self._derivative(name, phi_base[m - 1])]
            tangents.append((d_phi, np.conj(self._derivative(name, x1))))

        def matvec(v):
            dX = v[:m*N].reshape(m, N)
            out = np.zeros_like(v)
            blocks = out[:m*N].reshape(m, N)
            for i in range(m):
                blocks[i] = (self._segment_derivative(segs[i], dX[i], phi_base[i], i, tau, ax, ay, az)
                             - dX[(i + 1) % m])
            for k, (d_phi, d_x1_conj) in enumerate(tangents):
                for i in range(m):
                    if d_phi[i] is not None:
                        blocks[i] += d_phi[i]*v[m*N + k]
                out[m*N + k] = np.matmul(d_x1_conj, dX[0])
            return out
        return matvec
