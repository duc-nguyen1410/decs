"""
Edge tracking along the boundary between the laminar and turbulent basins of attraction
(Skufca, Yorke & Eckhardt 2006; Schneider, Eckhardt & Yorke 2007).

States on a segment  x(lam) = x_a + lam (x_b - x_a),  lam in [0, 1],  x_a laminar, x_b turbulent:

    1. bisect : classify x(lam) by integrating it until the cross-flow energy
                Ecf = 1/2 <w^2> falls below E_lo (laminar) or rises above E_hi (turbulent);
                halve the bracket [lam_lo, lam_hi] until its two states are closer than
                bisect_tol (L2 norm of the difference).
    2. follow : integrate both bracket states in steps of dt_check until they are further
                apart than delta_max, or one of them leaves E_lo < Ecf < E_hi.
    3. refine : bisect again between the pair at the last check before that, and repeat
                until the edge trajectory reaches time T_track.

The bisection midpoint is the midpoint in the L2 norm of the perturbation x - x_a
(bisect_in='l2': lam = (lam_lo + lam_hi)/2) or in its cross-flow energy, which grows like
lam^2 (bisect_in='ecf': lam^2 = (lam_lo^2 + lam_hi^2)/2). Both converge to the same edge.

The laminar-side state of the bracket traces the edge trajectory. Its history (t, Ecf, L2
norm) is written to edge_history.csv in odir, and its latest state to edge_state.
"""
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EdgeTracker:
    """Edge tracking for a FluidModel (or any model with flow_map(x, T, callback),
    cross_flow_energy(x=None), l2norm(x) and save_state(name, x)).

    params (required): E_lo, E_hi (Ecf thresholds of the laminar / turbulent classification),
        T_max (longest integration to classify one state), dt_check (time between separation
        checks while following the edge), delta_max (separation that triggers a new bisection),
        bisect_tol (bracket width at which a bisection stops; bisect_tol < delta_max).
    params (optional): bisect_in ('l2' or 'ecf', default 'l2'), check_every (classify every
        this many time steps, default 10), odir (output directory, default './edge/').
    """

    def __init__(self, model, params):
        self.model = model
        self.E_lo = params['E_lo']
        self.E_hi = params['E_hi']
        self.T_max = params['T_max']
        self.dt_check = params['dt_check']
        self.delta_max = params['delta_max']
        self.bisect_tol = params['bisect_tol']
        self.bisect_in = params.get('bisect_in', 'l2')
        self.check_every = params.get('check_every', 10)
        self.odir = params.get('odir', './edge/')
        if self.bisect_in not in ('l2', 'ecf'):
            raise ValueError(f"bisect_in must be 'l2' or 'ecf', not {self.bisect_in!r}")
        if not 0 < self.E_lo < self.E_hi:
            raise ValueError("need 0 < E_lo < E_hi")
        if not self.bisect_tol < self.delta_max:
            raise ValueError("need bisect_tol < delta_max")
        self.history = []          # rows (t, Ecf, L2 norm) of the edge trajectory

    def classify(self, x):
        """'laminar' or 'turbulent': integrate x until Ecf leaves (E_lo, E_hi). A state still
        undecided after T_max is classified by comparing Ecf with sqrt(E_lo*E_hi)."""
        decided = []

        def callback(solver):
            if solver.iteration % self.check_every == 0:
                E = self.model.cross_flow_energy()
                if E < self.E_lo:
                    decided.append('laminar')
                elif E > self.E_hi:
                    decided.append('turbulent')
            return bool(decided)

        self.model.flow_map(x, self.T_max, callback)
        if decided:
            return decided[0]
        E = self.model.cross_flow_energy()
        logger.warning(f"Undecided after T_max = {self.T_max} (Ecf = {E:.6e}); "
                       "consider a larger T_max or other thresholds.")
        return 'laminar' if E < np.sqrt(self.E_lo*self.E_hi) else 'turbulent'

    def bisect(self, xa, xb):
        """Bisect the segment from laminar xa to turbulent xb; return the bracket states."""
        d = xb - xa
        dist = self.model.l2norm(d)
        lo, hi = 0.0, 1.0
        n = 0
        while (hi - lo)*dist > self.bisect_tol:
            mid = 0.5*(lo + hi) if self.bisect_in == 'l2' else np.sqrt(0.5*(lo**2 + hi**2))
            if self.classify(xa + mid*d) == 'laminar':
                lo = mid
            else:
                hi = mid
            n += 1
            logger.info(f"Bisection {n}: lambda in [{lo:.15f}, {hi:.15f}], "
                        f"distance {(hi - lo)*dist:.3e}")
        return xa + lo*d, xa + hi*d

    def follow(self, xa, xb):
        """Integrate the bracket pair in steps of dt_check until it separates; return the pair
        at the last check before that and the time advanced."""
        elapsed = 0.0
        while True:
            ya = self.model.flow_map(xa, self.dt_check)
            yb = self.model.flow_map(xb, self.dt_check)
            Ea, Eb = self.model.cross_flow_energy(ya), self.model.cross_flow_energy(yb)
            dist = self.model.l2norm(yb - ya)
            inside = self.E_lo < Ea < self.E_hi and self.E_lo < Eb < self.E_hi
            if dist > self.delta_max or not inside:
                return xa, xb, elapsed
            xa, xb = ya, yb
            elapsed += self.dt_check
            self._record(self._t + elapsed, xa)

    def track(self, x_lam, x_turb, T_track):
        """Track the edge from the segment between x_lam (laminar) and x_turb (turbulent)
        until time T_track. Returns the final edge state; the history is in self.history."""
        xa, xb = np.array(x_lam, dtype=float), np.array(x_turb, dtype=float)
        if self.classify(xa) != 'laminar':
            raise ValueError("x_lam is not classified laminar (check E_lo)")
        if self.classify(xb) != 'turbulent':
            raise ValueError("x_turb is not classified turbulent (check E_hi)")
        self.history = []
        self._t = 0.0
        self._init_csv()
        while True:
            xa, xb = self.bisect(xa, xb)
            if self._t == 0.0:
                self._record(0.0, xa)
            if self._t >= T_track:
                break
            xa, xb, dt = self.follow(xa, xb)
            if dt == 0.0:
                raise RuntimeError("Bracket states separated within one dt_check; "
                                   "reduce dt_check or bisect_tol.")
            self._t += dt
            logger.info(f"Edge at t = {self._t:.6f}: Ecf = {self.history[-1][1]:.6e}, "
                        f"L2 norm = {self.history[-1][2]:.6e}")
            self.model.save_state(os.path.join(self.odir, 'edge_state'), xa)
        return xa

    def _init_csv(self):
        if self.model.dist.comm.rank == 0:
            os.makedirs(self.odir, exist_ok=True)
            with open(os.path.join(self.odir, 'edge_history.csv'), 'w') as f:
                f.write("t, Ecf, L2norm\n")

    def _record(self, t, x):
        row = (t, self.model.cross_flow_energy(x), self.model.l2norm(x))
        self.history.append(row)
        if self.model.dist.comm.rank == 0:
            with open(os.path.join(self.odir, 'edge_history.csv'), 'a') as f:
                f.write(", ".join(f"{v:.12e}" for v in row) + "\n")
