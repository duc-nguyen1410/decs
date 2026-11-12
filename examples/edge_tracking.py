"""
Edge tracking in the thermohaline-shear model at a subcritical point: the base state is linearly
stable at Ri = 30 (Pe = 10, Pr = 10, Lambda = 2, tau = 0.01), while a finite-amplitude branch,
bifurcating subcritically near Ri ~ 24, coexists with it.

    1. turbulent state by hysteresis: grow it at Ri = 20 (base state unstable), then raise Ri to 30
    2. thresholds of the classification from its cross-flow energy Ecf = 1/2 <w^2>
    3. bisect between the base state (zero fluctuations) and the turbulent state, follow the edge

LONG RUN: every bisection integrates about 20 states for up to T_max; use MPI on a cluster
(mpiexec -n 16 python3 edge_tracking.py). Not yet run to completion for these parameters: check
in step 1 that Ecf stays finite at Ri = 30 before trusting the thresholds.

Run from the examples folder:  python3 edge_tracking.py
"""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path
import logging
logger = logging.getLogger(__name__)
import numpy as np
from physics import ThermohalineShear
from tools import EdgeTracker

sizes = (128, 64)
bounds = (2*np.pi, 1.0)
params = {'Pe': 10.0, 'Ri': 20.0, 'Pr': 10.0, 'Lambda': 2.0, 'tau': 0.01,
          'init_dt': 1e-3,
          'ext': '.nc'}
model = ThermohalineShear(params=params, sizes=sizes, bounds=bounds)
model.build_problems()
model.use_CFL = False # fixed time steps: CFL can lose stability in long runs (see physics/base.py)

# 1. turbulent state at Ri = 30 by hysteresis from Ri = 20
ecf = lambda m: m.cross_flow_energy()
x = 1e-2*np.random.default_rng(0).standard_normal(model.size())
x = model.flow_map(x, 100.0)                     # saturate at Ri = 20
model.set_param('Ri', 30.0)                      # base state linearly stable from here on
E = []
x = model.flow_map(x, 150.0, callback=lambda s: (s.iteration % 100 == 0 and E.append(ecf(model))) and False)
E_turb = np.mean(E[len(E)//2:])
logger.info(f"Ri = 30: Ecf of the turbulent state {E_turb:.4e} (last value {E[-1]:.4e})")
if E[-1] < 1e-2*E_turb:
    raise RuntimeError("the state decayed at Ri = 30: no turbulent state to bracket")
model.save_state('test_edge/turbulent_Ri30', x)

# 2. classification: laminar below E_lo, turbulent above E_hi; the edge lies in between
tracker = EdgeTracker(model, {'E_lo': 1e-3*E_turb,
                              'E_hi': 0.5*E_turb,
                              'T_max': 50.0,        # longest integration to classify one state
                              'dt_check': 1.0,      # separation checks while following the edge
                              'delta_max': 1e-2*model.l2norm(x),
                              'bisect_tol': 1e-6*model.l2norm(x),
                              'bisect_in': 'l2',    # or 'ecf': bisect in the cross-flow energy
                              'odir': 'test_edge/'})

# 3. track the edge from the segment between the base state and the turbulent state
x_lam = np.zeros(model.size())                   # the model computes fluctuations about the base flow
x_edge = tracker.track(x_lam, x, T_track=200.0)  # writes edge_history.csv and edge_state
logger.info(f"edge state: Ecf {model.cross_flow_energy(x_edge):.4e}")
# x_edge is a starting guess for ECSSolver: an equilibrium, traveling wave or periodic orbit
# with one unstable direction (compare with the lower branch continued from Ri ~ 24).
