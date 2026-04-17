# to run: mpiexec -n 4 python3 main.py

import logging
logger = logging.getLogger(__name__)
import numpy as np
import dedalus.public as de
from physics.double_diffusion import SaltFinger
from ecs_core.ecs_core import ECSSolver
from continuation.continuation import Continuation

sizes = (128, 128)
bounds = (0.6, 1.0)
params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0}
model = SaltFinger(params=params, sizes=sizes, bounds=bounds)

nsolver_params = {'max_iterations': 100, 
                  'tol': 1e-8, 
                  'trust_radius': 20.0,
                  'krylov_dim_min': 40,
                  'gmres_min_error': 1e-6,
                  'computeStability': False
                  }
nsolver = ECSSolver(model, nsolver_params)

cont_params = {'mu_name': 'Lx',
               'odir': 'debug_continuation/',
               'Tsearch': False, 'Tp': 0.2,
               'Rxsearch': False, 'ax': 0.0,
               'Rzsearch': False, 'az': 0.0,
               }
Cont = Continuation(nsolver, cont_params)

# Cont.ECSSolver.model.set_initial_conditions(mode='random')
nsolver.model.load_state('CI1_Lx0.6_new.h5')
# x0 = Cont.ECSSolver.model.get_state()
# logger.info(f"Initial state norm: {np.linalg.norm(x0)}")
# continue the trivial state
Cont.arc_length_continuation(mu_start=0.6, dmu=0.001, n_steps=4, mu_target=1.0)