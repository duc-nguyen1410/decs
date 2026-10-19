# to run: mpiexec -n 4 python3 find_tw.py
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path

import logging
logger = logging.getLogger(__name__)
import numpy as np
import dedalus.public as de
from physics.salt_finger import SaltFinger
from ecs_core.ecs_core import ECSSolver

sizes = (128, 128)
bounds = (0.6, 1.0)
params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0,
          'init_dt': 2e-4,
          'ext': '.nc'}
model = SaltFinger(params=params, sizes=sizes, bounds=bounds)
model.build_problems()

nsolver_params = {'max_iter': 20, 
                  'tol': 1e-9, 
                  'trust_radius': 20.0,
                  'krylov_dim': 50,
                  'krylov_dim_min': 5,
                  'gmres_min_error': 1e-20,
                  'computeStability': True,
                  'Neigen': 60,
                  'projectNeutralDrift': True,
                  'odir':f"test_TW/"
                  }
nsolver = ECSSolver(model, nsolver_params)

nsolver.model.load_state('Lx0.6_TM1/solution')

nsolver.model.use_CFL = True
nsolver.model.save_snapshots = True # sate snapshot during Tp to plot a video if needed
nsolver.sigma.load_from_file('symmetry_pix.sym')

x0 = nsolver.model.get_state()
logger.info(f"Initial state norm: {np.linalg.norm(x0)}")
solution, success, res, norm, properties = nsolver.NewtonSolver(x0 = x0, 
                                                                Tp = 0.2, 
                                                                Rzsearch = True, 
                                                                az = -0.24463599362935448)