# to run: mpiexec -n 4 python3 main.py
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path 

import logging
logger = logging.getLogger(__name__)
import numpy as np
import dedalus.public as de
from physics.salt_finger import SaltFinger
from physics.symmetry import Symmetry
from ecs_core.ecs_core import ECSSolver
from continuation.continuation import Continuation

sizes = (128, 128)
bounds = (0.5, 1.0)
Ra = 1e5
Pr = 7.0
Rrho = 40.0
tau = 0.01
params = {'Ra': Ra, 'Pr': Pr, 'tau': tau, 'Rrho': Rrho,
          'init_dt': 2e-4,
          'ext': '.nc'}
model = SaltFinger(params=params, sizes=sizes, bounds=bounds)
model.build_problems()

nsolver_params = {'max_iterations': 5, 
                  'tol': 1e-9, 
                  'eps_t': 1e-5,
                  'eps_x': 1e-6,
                  'trust_radius': 20.0,
                  'krylov_dim': 50,
                  'krylov_dim_min': 5,
                  'gmres_min_error': 1e-20,
                  'computeStability': True,
                  'Neigen': 60,
                  'projectNeutralDrift': True,
                  }
nsolver = ECSSolver(model, nsolver_params)

nsolver.model.use_CFL = True
nsolver.sigma=Symmetry()
nsolver.sigma.load_from_file('symmetry_pixz.sym')

cont_params = {'mu_name': 'Lx',
               'odir': f"ContLx_up/",
               'Tsearch': False, 'Tp': 0.2,
               'Rxsearch': False, 'ax': 0.0,
               'Rzsearch': False, 'az': 0.0,
               'guess_error_min': 0.01,
               'guess_error_max': 50,
               }
Cont = Continuation(nsolver, cont_params)

Cont.ECSSolver.model.load_state('solution')

Cont.run_continuation(mu_start=0.6, dmu=0.005, n_steps=500, mu_target=0.3)