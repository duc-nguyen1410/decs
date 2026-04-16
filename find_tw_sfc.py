# to run: mpiexec -n 4 python3 main.py

import logging
logger = logging.getLogger(__name__)
import numpy as np
import dedalus.public as de
from physics.double_diffusion import SaltFinger
from ecs_core.ecs_core import ECSSolver

sizes = (128, 128)
bounds = (0.6, 1.0)
params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0}
model = SaltFinger(params=params, sizes=sizes, bounds=bounds)
model.build_ivp_problem()

nsolver_params = {'max_iterations': 100, 
                  'tol': 1e-8, 
                  'trust_radius': 20.0,
                  'krylov_dim_min': 40,
                  'gmres_min_error': 1e-6,
                  'computeStability': False
                  }
nsolver = ECSSolver(model, nsolver_params)

nsolver.model.load_state('TM1_Lx0.6_new.h5') # az = -0.2446059566700214

x0 = nsolver.model.get_state()
logger.info(f"Initial state norm: {np.linalg.norm(x0)}")
solution, success, res, norm, properties = nsolver.NewtonSolver(x0,
                                                                Rzsearch=True,
                                                                az=-0.24,
                                                                Tp = 0.2,
                                                                dt = 1e-3)