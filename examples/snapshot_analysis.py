# Guesses for ECS from a simulation's snapshots: recurrences (PO / RPO), slow points (EQ / TW)
# and visits near known solutions. Here the "simulation" is the relative periodic orbit RPO2,
# so the recurrence search must return its period Tp = 0.0738 and shift ax = 0.5; on a
# turbulent run (e.g. simulate.py), point snap_dir to its snapshots and skip step 1.
# to run: python3 snapshot_analysis.py   (a few minutes)
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path

import logging
logger = logging.getLogger(__name__)
import numpy as np
import dedalus.public as de
from physics import SaltFinger
from tools import (load_snapshots, load_snapshot, lag_distances, recurrences, slow_points,
                   reference_distances, reflections, nearest_solution)

params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'init_dt': 2e-4, 'ext': '.nc',
          'odir': 'test_snapshots/'}
sizes, bounds = (128, 128), (0.8, 1.0)
snap_dir = params['odir'] + 'snapshots'

# 1. a simulation writing snapshots, as in simulate.py (4 periods, 25 snapshots per period)
sim = SaltFinger(params=params, sizes=sizes, bounds=bounds, mode='sim')
sim.build_problems()
solver = sim.ivp_problem.build_solver(de.RK222)
sim.load_state('Lx0.8_RPO2/solution')
sim.set_cfl(solver)
sim.set_snapshots(solver=solver, sim_dt=0.003)
solver.stop_sim_time = 0.3
while solver.proceed:
    solver.step(sim.CFL.compute_timestep())

# 2. analysis (an 'ecs' model: its symmetry operators assume the complex Fourier basis)
model = SaltFinger(params=params, sizes=sizes, bounds=bounds)
T, X = load_snapshots(model, snap_dir)
logger.info(f"{len(T)} snapshots, t = {T[0]:.3f} - {T[-1]:.3f}")

# recurrences, minimised over shifts in x and z: starting guesses (Tp, ax, az) for ECSSolver.
# Every multiple of a period recurs too, so bound the lags (here below 2 Tp = 0.148); the period
# is known to within a snapshot interval (Newton refines it)
dist, shifts = lag_distances(model, X, max_lag=40, shift_axes=('x', 'z'))
for r in recurrences(T, dist, shifts, leave=0.2, n_best=5):
    logger.info(f"return: t = {r['t']:.3f}  Tp = {r['Tp']:.4f}  ax = {r['ax']:+.4f}  "
                f"az = {r['az']:+.4f}  distance {r['distance']:.3f}")

# slow points: equilibrium / travelling-wave guesses (an orbit has none: speeds stay large)
for s in slow_points(T, dist, shifts, n=3, min_separation=0.05):
    logger.info(f"slow point: t = {s['t']:.3f}  speed {s['speed']:.3f} per snapshot  "
                f"drift ax = {s['ax']:+.4f}, az = {s['az']:+.4f}")

# visits near known solutions (with their reflections), and which one a snapshot is nearest to
catalogue = {}
for name in ('PO3', 'RPO2'):
    model.load_state(f'Lx0.8_{name}/solution')
    catalogue[name] = model.get_state()
for name, x in catalogue.items():
    d = reference_distances(model, X, [model.apply_symmetry(x, s) for s in reflections(model)],
                            ('x', 'z'))
    logger.info(f"distance to {name}: min {d.min():.3f}; below 0.05 at t = " +
                ", ".join(f"{t:.3f}" for t in T[d < 0.05]))
t, x = load_snapshot(model, snap_dir, 0.15)
for d, name, sigma in nearest_solution(model, x, catalogue):
    logger.info(f"snapshot t = {t:.3f}: {name} at distance {d:.3f} via {sigma}")
