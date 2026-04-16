# to run: mpiexec -n 4 python3 main.py

import logging
logger = logging.getLogger(__name__)
import numpy as np
import dedalus.public as de
from physics.double_diffusion import SaltFinger
from ecs_core.ecs_core import ECSSolver

sizes = (128, 128)
bounds = (0.8, 1.0)
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

nsolver.model.load_state('RPO2_1_Lx0.8_new.h5')

x0 = nsolver.model.get_state()
logger.info(f"Initial state norm: {np.linalg.norm(x0)}")
solution, success, res, norm, properties = nsolver.NewtonSolver(x0,
                                                                Tsearch=True,
                                                                Rxsearch=True,
                                                                Tp = 0.07380265343504276,
                                                                ax = 0.5,
                                                                dt = 0.0001)