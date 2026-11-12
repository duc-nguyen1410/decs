# Find the symmetries of a state (e.g. a solution converged without an imposed symmetry).
# to run: python3 find_symmetry.py   (or mpiexec -n 4 python3 find_symmetry.py)
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path

import logging
logger = logging.getLogger(__name__)
import numpy as np
from physics import SaltFinger, Symmetry
from tools import find_symmetries

state_file = 'Lx0.6_TM1/solution'     # without extension
sizes = (128, 128)
bounds = (0.6, 1.0)
tol = 1e-6                            # distance below which the state counts as symmetric
save_centered = False                 # also save the state shifted so the best symmetry has
                                      # no shift along its reflected directions

params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'ext': '.nc'}
model = SaltFinger(params=params, sizes=sizes, bounds=bounds)   # parameters do not matter here
model.load_state(state_file)
x = model.get_state()

results = find_symmetries(model, x)
logger.info("s sx sy sz ax ay az : ||sigma(x) - x||/||x||")
for r in results:
    flag = '  <- symmetric' if r['distance'] < tol else ''
    logger.info(f"{r['symmetry']} : {r['distance']:.3e}{flag}")

best = results[0]
logger.info(f"Best symmetry: {best['symmetry']}, distance {best['distance']:.3e}")
if best['distance'] > tol:
    logger.info(f"No symmetry within tol = {tol:g}.")
else:
    sym_file = state_file + '_best.sym'
    s = best['symmetry']
    if model.dist.comm.rank == 0:
        with open(sym_file, 'w') as f:
            f.write(f"{s.s:g} {s.sx:g} {s.sy:g} {s.sz:g} {s.ax:.16g} {s.ay:.16g} {s.az:.16g}")
    logger.info(f"Saved {sym_file} (load with solver.sigma.load_from_file)")

    if save_centered and best['center']:
        x_c = model.apply_symmetry(x, Symmetry(**best['center']))
        centered = Symmetry(s.s, s.sx, s.sy, s.sz,
                            *[0 if getattr(s, 's' + n) == -1 else getattr(s, 'a' + n) for n in 'xyz'])
        d = np.linalg.norm(model.apply_symmetry(x_c, centered) - x_c)/np.linalg.norm(x_c)
        logger.info(f"Centered state: symmetry {centered}, distance {d:.3e}")
        model.save_state(state_file + '_centered', x_c)
