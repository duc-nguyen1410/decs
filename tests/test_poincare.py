"""PoincareSection and recurrence search (ecs_core/poincare.py).

Toy: Hopf normal form dx/dt = x(1 - r^2) - y, dy/dt = y(1 - r^2) + x, a stable limit cycle
r = 1 of period 2 pi. Section y = 0 with y increasing: crossings at (1, 0), 2 pi apart."""
import numpy as np
import pytest

from ecs_core.poincare import PoincareSection, recurrence_search


class ToyModel:
    dt = 1e-2

    class dist:
        class comm:
            rank = 0

    def __init__(self):
        self.state = np.zeros(2)

    @staticmethod
    def rhs(s):
        r2 = s @ s
        return np.array([s[0]*(1 - r2) - s[1], s[1]*(1 - r2) + s[0]])

    def set_state(self, x):
        self.state = np.array(x, dtype=float)

    def current_state(self):
        return self.state.copy()

    def flow_map(self, x0, T, callback=None):
        class Solver:
            iteration, sim_time = 0, 0.0
        solver, s = Solver(), np.array(x0, dtype=float)
        n = max(1, int(round(T/self.dt)))
        h = T/n
        for _ in range(n):
            k1 = self.rhs(s); k2 = self.rhs(s + 0.5*h*k1)
            k3 = self.rhs(s + 0.5*h*k2); k4 = self.rhs(s + h*k3)
            s = s + h/6*(k1 + 2*k2 + 2*k3 + k4)
            self.state = s
            solver.iteration += 1
            solver.sim_time += h
            if callback is not None and callback(solver):
                break
        return s.copy()


def y_coordinate(model):
    return model.state[1]


def test_crossings_of_limit_cycle():
    section = PoincareSection(ToyModel(), y_coordinate, level=0.0, direction=1)
    crossings = section.run([1.0, -0.5], T=20.0)
    t = np.array([c['t'] for c in crossings])
    assert len(crossings) == 4                        # first at t = 0.46 (starts below y = 0)
    assert np.diff(t) == pytest.approx(2*np.pi, abs=1e-7)
    assert crossings[-1]['x'] == pytest.approx([1.0, 0.0], abs=1e-7)
    assert all(abs(c['g']) < 1e-9 for c in crossings)


def test_direction_and_max_crossings():
    both = PoincareSection(ToyModel(), y_coordinate, direction=0).run([1.0, -0.5], T=20.0)
    down = PoincareSection(ToyModel(), y_coordinate, direction=-1).run([1.0, -0.5], T=20.0)
    assert len(both) == 7 and len(down) == 3
    assert all(c['x'][0] < 0 for c in down)
    first = PoincareSection(ToyModel(), y_coordinate).run([1.0, -0.5], T=20.0, max_crossings=1)
    assert len(first) == 1


def test_without_refinement_is_less_accurate():
    coarse = PoincareSection(ToyModel(), y_coordinate, refine=False).run([1.0, -0.5], T=20.0)
    t = np.array([c['t'] for c in coarse])
    assert np.diff(t) == pytest.approx(2*np.pi, abs=1e-3)


def test_recurrence_search_finds_period():
    model = ToyModel()
    crossings = PoincareSection(model, y_coordinate).run([0.3, -0.1], T=40.0)
    best = recurrence_search(model, [c['x'] for c in crossings], [c['t'] for c in crossings],
                             T_max=8.0, n_best=1)[0]
    assert best['Tp'] == pytest.approx(2*np.pi, abs=1e-6)
