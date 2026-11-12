# New solutions / branches from the unstable modes of a converged solution (any model).
#   1. guesses x + eps |x| Re(v e^{i phi}) along each unstable mode v (both signs; several phases
#      for a complex pair, whose guesses are periodic orbits with Tp ~ 2 pi / Im(lambda));
#   2. optionally, guesses further away: the slow points of the trajectories leaving x along
#      each mode (its unstable manifold);
#   3. Newton from every guess; each result is classified as 'original', 'new' or 'same as <k>'
#      (up to shifts and reflections), in branches/branches.csv.
# Near a bifurcation (small unstable lambda), step 1 is enough; far from it, use step 2.
# Newton costs ~10-20 min per iteration at 128^2 on one core: test at a coarse grid first
# (change_grid.py), or run with MPI.
# to run: python3 find_branches.py   (or mpiexec -n 16 python3 find_branches.py)
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path

import logging
logger = logging.getLogger(__name__)
from physics import SaltFinger
from ecs_core import ECSSolver
from tools import load_unstable_modes, branch_guesses, unstable_manifold_guesses, search_branches

solution_dir = 'Lx0.6_WF'            # solution.nc and stability/ (unstable modes) of the solution
base = 'eq'                          # 'eq' (equilibrium) or 'tw' (travelling wave)
eps = (1e-2, 1e-1)                   # perturbation sizes, relative to |x|
n_phases = 4                         # phases per complex mode
follow_manifold = False              # step 2 (integrates every guess for T_follow)
T_follow = 1.0
run_newton = True                    # step 3
odir = 'test_branches/'

params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'init_dt': 2e-4, 'ext': '.nc'}
model = SaltFinger(params=params, sizes=(128, 128), bounds=(0.6, 1.0))
model.build_problems()
model.load_state(solution_dir + '/solution')
x = model.get_state()

modes = load_unstable_modes(model, solution_dir + '/stability')
for lam, _ in modes:
    logger.info(f"unstable mode: lambda = {lam:.4f}" +
                (f"  (Hopf: Tp ~ {6.283185307/abs(lam.imag):.4f})" if lam.imag else "  (steady)"))

guesses = branch_guesses(model, x, modes, eps=eps, n_phases=n_phases, base=base, odir=odir)
if follow_manifold:
    guesses += unstable_manifold_guesses(model, x, modes, eps=1e-3, T=T_follow, dt=0.01,
                                         n_slow=2, base=base, odir=odir + 'manifold/')
for k, g in enumerate(guesses):
    logger.info(f"guess {k}: mode {g['mode']}, eps {g['eps']:g}, phase {g['phase']:.2f}, {g['kind']}"
                + (f", Tp {g['Tp']:.4f}" if g['Tp'] else '')
                + f"; keeps {[str(s) for s in g['keeps']]}, breaks {[str(s) for s in g['breaks']]}")

if run_newton:
    solver = ECSSolver(model, {'tol': 1e-9, 'max_iter': 20, 'trust_radius': 20.0, 'krylov_dim': 50,
                               'computeStability': True, 'Neigen': 20, 'odir': odir})
    # CFL steps are faster for equilibria / waves; orbits (Tsearch) use fixed steps, as in find_po.py
    model.use_CFL = all(g['kind'] in ('eq', 'tw') for g in guesses)
    results = search_branches(solver, x, guesses, odir=odir, Tp=0.2)
    for r in results:
        logger.info(f"guess {r['guess']} ({r['kind']}): {r['status']}, residual {r['residual']:.2e}")
