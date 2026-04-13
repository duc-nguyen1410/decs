# to run: mpiexec -n 4 python3 main.py

import logging
logger = logging.getLogger(__name__)
import numpy as np
import dedalus.public as de
from physics.double_diffusion import SaltFinger
from ecs_core.ecs_core import ECSSolver

sizes = (32, 32, 32)
bounds = (0.5, 0.5, 1.0)
domain = SaltFinger.create_domain(sizes, bounds)
params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0,
          'init_dt': 2e-3}
model = SaltFinger(domain, params)

nsolver_params = {'max_iterations': 100, 
                  'tol': 1e-8, 
                  'trust_radius': 20.0,
                  'krylov_dim_min': 40,
                  'gmres_min_error': 1e-6}
nsolver = ECSSolver(model, nsolver_params)

# model.set_initial_conditions(mode='random')
# model.load_state('CI1_Lx0.6.h5')
model.load_state('test_save_state.h5')
x0 = model.get_state()
logger.info(f"Initial state norm: {np.linalg.norm(x0)}")
solution = nsolver.NewtonSolver(x0,
                                Tsearch=False,
                                Rxsearch=False,
                                Rzsearch=False,
                                Tp = 0.02,
                                dt = 2e-4)

nsolver.stability(solution)