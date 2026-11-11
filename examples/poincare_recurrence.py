"""
Poincaré section and recurrence search, then Newton from the best recurrence.

Demonstrated on the relative periodic orbit RPO2 (Lx = 0.8): the section Ecf = level is crossed
once per period, and the recurrence search over x-shifts recovers the period Tp = 0.0738 and the
shift ax = 0.5, which are passed to the Newton solver. For a turbulent run, start from a state on
the attractor and integrate much longer (in chunks): see docs/finding_po_rpo.md.

Run from the examples folder:  python3 poincare_recurrence.py
(about 20 minutes on one core: the section takes a minute, Newton two iterations from the crossing)
"""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path
import logging
logger = logging.getLogger(__name__)
import numpy as np
from physics import SaltFinger
from ecs_core import ECSSolver, PoincareSection, recurrence_search

sizes = (128, 128)
bounds = (0.8, 1.0)
params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0,
          'init_dt': 1e-4,
          'ext': '.nc'}
model = SaltFinger(params=params, sizes=sizes, bounds=bounds)
model.build_problems()
model.use_CFL = False
model.load_state('Lx0.8_RPO2/solution')
x0 = model.get_state()

# 1. choose the section: the mean of Ecf = 1/2 <w^2> along a short probe run
ecf = lambda m: m.cross_flow_energy() # observable of the current state (safe between time steps)
E = []
model.flow_map(x0, 0.08, callback=lambda solver: E.append(ecf(model)) and False)
level = np.mean(E)
logger.info(f"Ecf in [{min(E):.4f}, {max(E):.4f}], section level {level:.4f}")

# 2. record the upward crossings Ecf = level (refined to |Ecf - level| <= 1e-10 relative)
section = PoincareSection(model, ecf, level=level, direction=+1)
crossings = section.run(x0, T=0.25, odir='test_poincare/') # writes crossings.csv and crossing_<i>
times = [c['t'] for c in crossings]
states = [c['x'] for c in crossings]
logger.info(f"{len(crossings)} crossings, intervals {np.diff(times)}")

# 3. near-recurrences min_a ||S(a) x_j - x_i|| / ||x_i||, minimised over shifts in x.
#    T_max bounds the search: every multiple of a period recurs too.
best = recurrence_search(model, states, times, shift_axes=('x',), T_max=0.12, n_best=3)
for r in best:
    logger.info(f"recurrence: Tp = {r['Tp']:.6f}, ax = {r['ax']:+.4f}, distance {r['distance']:.2e}")

# 4. Newton from the best recurrence: Tp and ax are ECSSolver's starting values directly
r = best[0]
nsolver = ECSSolver(model, {'max_iter': 20, 'tol': 1e-9, 'trust_radius': 20.0,
                            'krylov_dim': 50, 'krylov_dim_min': 5, 'gmres_min_error': 1e-20,
                            'computeStability': False, 'odir': 'test_poincare_RPO/'})
solution, success, res, norm, properties = nsolver.solve(x0=states[r['i']],
                                                         Tsearch=True, Rxsearch=True,
                                                         Tp=r['Tp'], ax=r['ax'])
logger.info(f"Newton: success {success}, residual {res:.3e}, Tp {nsolver.Tp:.8f}, ax {nsolver.ax:.6f}")
