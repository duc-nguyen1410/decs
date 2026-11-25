# Find a helical (screw-symmetric) exact coherent state of 3-D salt-finger convection.
# to run: mpiexec -n 8 python3 find_helical.py
#
# The search is restricted to states invariant under the screw symmetry
#     sigma_h = T_z(h Lz/4) o R_90       (rotation by 90 deg about z, then shift by Lz/4)
# whose mean flow U(z) turns by 90 degrees every Lz/4: a helix instead of the planar shear
# of the 2-D solutions (see tools/helical.py). The box must be square: Lx = Ly, Nx = Ny.
#
# Inside this subspace, x- and y-shifts are not neutral (they move the rotation axis) but the
# z-shift is: search it (Rzsearch=True). A drift in z of a screw-symmetric state is the same
# as a rotation of the pattern, so a converged state with az != 0 is a rotating wave.
# Add Tsearch=True for helical periodic / relative periodic orbits.
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path

import logging
logger = logging.getLogger(__name__)
import numpy as np
from physics import SaltFinger
from ecs_core import ECSSolver
from tools import screw_symmetry, helical_guess, helical_diagnostics

# ---------------------------------------------------------------- settings
params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0,
          'init_dt': 2e-4,
          'ext': '.nc'}
L = 0.6                         # Lx = Ly
sizes = (32, 32, 64)            # (Nx, Ny, Nz), Nx = Ny; start coarse, refine with change_grid
handedness = +1                 # +1: sigma = Symmetry(rz=1, az=0.25), -1: az = -0.25

# Initial guess: 'from_2d' extends a 2-D solution uniformly in y and projects it onto the
# screw-symmetric states; 'from_3d' projects a 3-D state (e.g. a DNS snapshot, same grid).
guess = 'from_2d'
state_2d, sizes_2d = 'Lx0.6_TM1/solution', (128, 128)   # 2-D solution, its grid; Lx_2d = L
state_3d = 'helical_guess'                                # 3-D state file (without extension)
guess_scale = 2.0               # the projection halves the shear and the fingers; restore them

# Newton: Tp and the initial z-shift az (per Tp). TM1 drifts by az = -0.2446 per Tp = 0.2;
# a helical state may drift differently, the search adjusts it.
Tp = 0.2
az0 = -0.24463599362935448
odir = f"Lx{L}_Rrho{params['Rrho']}_helical_h{handedness:+d}/"

# ---------------------------------------------------------------- model and guess
model = SaltFinger(params=params, sizes=sizes, bounds=(L, L, 1.0))
sigma = screw_symmetry(handedness)

if guess == 'from_2d':
    model_2d = SaltFinger(params=params, sizes=sizes_2d, bounds=(L, 1.0), build=False)
    model_2d.load_state(state_2d)
    x0 = helical_guess(model_2d, model, sigma, scale=guess_scale)
else:
    model.load_state(state_3d)
    x0 = model.symmetrize(model.get_state(), sigma, 4)

def report(label, x):
    d = helical_diagnostics(model, x, sigma)
    logger.info(f"{label}: " + ", ".join(f"{k} = {v:.4g}" for k, v in d.items()))
    return d

report("Initial guess", x0)
model.save_state(odir + 'initial_guess', x0)

# ---------------------------------------------------------------- Newton-Krylov
nsolver_params = {'max_iter': 30,
                  'tol': 1e-9,
                  'trust_radius': 20.0,
                  'krylov_dim': 60,
                  'krylov_dim_min': 5,
                  'gmres_min_error': 1e-20,
                  'computeStability': False,  # 3-D: Arnoldi only (the EVP does not fit), costly
                  'Neigen': 40,
                  'odir': odir}
nsolver = ECSSolver(model, nsolver_params)
nsolver.model.use_CFL = True
nsolver.sigma = sigma                       # or nsolver.sigma.load_from_file('symmetry_screw.sym')

solution, success, res, norm, properties = nsolver.solve(x0=x0, Tp=Tp, Rzsearch=True, az=az0)

x, Tp, ax, ay, az = nsolver.unpack_xi(solution)
d = report("Solution" if success else "Last iterate", x)
logger.info(f"converged: {success}, residual {res:.3e}, Tp = {Tp}, az = {az}, "
            f"drift speed c = {az/Tp:.4g} Lz per unit time, "
            f"rotation rate = {2*np.pi*az/Tp:.4g} rad per unit time")
if model.dist.comm.rank == 0:
    with open(odir + 'helical_diagnostics.txt', 'w') as f:
        for k, v in {**d, 'Tp': Tp, 'az': az, 'residual': res}.items():
            f.write(f"{k} {v:.16g}\n")
