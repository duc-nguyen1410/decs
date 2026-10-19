# to run: mpiexec -n 4 python3 find_rpo.py
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
bounds = (0.8, 1.0)
Ra = 1e5
Pr = 7.0
Rrho = 40.0
tau = 0.01
params = {'Ra': Ra, 'Pr': Pr, 'tau': tau, 'Rrho': Rrho,
          'init_dt': 1e-4,
          'ext': '.nc'}
model = SaltFinger(params=params, sizes=sizes, bounds=bounds)
model.build_problems()

nsolver_params = {'max_iter': 20, 
                  'tol': 1e-9, 
                  'trust_radius': 20.0,
                  'krylov_dim': 50,
                  'krylov_dim_min': 5,
                  'gmres_min_error': 1e-20,
                  'computeStability': False,
                  'Neigen': 60,
                  'projectNeutralDrift': True,
                  'odir':f"test_RPO/"
                  }
nsolver = ECSSolver(model, nsolver_params)

# nsolver.model.use_CFL = True
nsolver.model.save_snapshots = True

nsolver.model.load_state('Lx0.8_RPO2/solution')

x0 = nsolver.model.get_state()
logger.info(f"Initial state norm: {np.linalg.norm(x0)}")
solution, success, res, norm, properties = nsolver.NewtonSolver(x0 = x0, 
                                                                Tsearch=True, 
                                                                Rxsearch=True, 
                                                                Rysearch=False, 
                                                                Rzsearch=False, 
                                                                Tp = 0.07380265343504276,
                                                                ax = 0.5, # ax = dx/Lx
                                                                ay = 0.0,
                                                                az = 0.0)