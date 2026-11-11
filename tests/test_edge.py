"""EdgeTracker (ecs_core/edge.py) on a cheap bistable toy model with a known edge state.

Toy: dx/dt = -x (x - 1)(x - 3), dy/dt = -y. Laminar attractor (0, 0), turbulent attractor
(3, 0), and the edge {x = 1} with the edge state (1, 0) of Ecf = x^2/2 = 0.5."""
import numpy as np
import pytest

from ecs_core.edge import EdgeTracker


class ToyModel:
    dt = 1e-3

    class dist:
        class comm:
            rank = 0

    def __init__(self):
        self.state = np.zeros(2)

    @staticmethod
    def rhs(s):
        return np.array([-s[0]*(s[0] - 1)*(s[0] - 3), -s[1]])

    def flow_map(self, x0, T, callback=None):
        class Solver:
            iteration = 0
        solver, s = Solver(), np.array(x0, dtype=float)
        for _ in range(max(1, round(T/self.dt))):
            k1 = self.rhs(s); k2 = self.rhs(s + 0.5*self.dt*k1)
            k3 = self.rhs(s + 0.5*self.dt*k2); k4 = self.rhs(s + self.dt*k3)
            s = s + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
            self.state = s
            solver.iteration += 1
            if callback is not None and callback(solver):
                break
        return s.copy()

    def cross_flow_energy(self, x=None):
        s = self.state if x is None else x
        return 0.5*s[0]**2

    def l2norm(self, x):
        return float(np.linalg.norm(x))

    def save_state(self, name, x):
        pass


PARAMS = {'E_lo': 0.05, 'E_hi': 2.0, 'T_max': 20.0, 'dt_check': 0.5,
          'delta_max': 1e-2, 'bisect_tol': 1e-7, 'check_every': 10}


@pytest.mark.parametrize('bisect_in', ['l2', 'ecf'])
def test_edge_tracking_finds_edge_state(tmp_path, bisect_in):
    tracker = EdgeTracker(ToyModel(), {**PARAMS, 'bisect_in': bisect_in, 'odir': str(tmp_path)})
    edge = tracker.track([0.0, 0.0], [2.5, 1.0], T_track=10.0)
    assert edge == pytest.approx([1.0, 0.0], abs=1e-4)
    t, E, _ = np.array(tracker.history).T
    assert t[-1] >= 10.0 and np.all(np.diff(t) > 0)
    assert E[-1] == pytest.approx(0.5, abs=1e-2)   # recorded while the pair drifts apart
    assert len(np.loadtxt(tmp_path/'edge_history.csv', delimiter=',', skiprows=1)) == len(t)


def test_first_bisection_brackets_the_edge():
    tracker = EdgeTracker(ToyModel(), PARAMS)
    lo, hi = tracker.bisect(np.zeros(2), np.array([2.5, 1.0]))
    assert lo[0] < 1.0 < hi[0] and hi[0] - lo[0] < 1e-6


def test_wrong_initial_states_raise():
    tracker = EdgeTracker(ToyModel(), {**PARAMS, 'odir': None})
    with pytest.raises(ValueError, match='turbulent'):
        tracker.track([0.0, 0.0], [0.5, 0.0], T_track=1.0)
