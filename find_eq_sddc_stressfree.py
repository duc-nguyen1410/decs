# to run: mpiexec -n 4 python3 main.py

import logging
logger = logging.getLogger(__name__)
import numpy as np
import dedalus.public as de
from physics.double_diffusion import ShearedDiffusiveConvection
from ecs_core.ecs_core import ECSSolver

params = {'Ra': 1e4, 'Pr': 7.0, 'tau': 0.01, 'Lambda': 2.0, 'Ri': 4.0,
          'stress-free': True}
model = ShearedDiffusiveConvection(params=params,
                                   sizes=(128, 128),
                                   bounds=(2*np.pi, 1.0),
                                   bounded=True,
                                   )
model.build_problems()

nsolver_params = {'odir':"eq_sddc_stressfree_Ra1e4_Ri4/",
                  'max_iterations': 100, 
                  'tol': 1e-8, 
                  'trust_radius': 2.0,
                  'krylov_dim_min': 40,
                  'gmres_min_error': 1e-6,
                  'computeStability': True
                  }
nsolver = ECSSolver(model, nsolver_params)

nsolver.model.load_state('sddc_stressfree/sample_sddc_Ra1e4_Ri4.h5')

x0 = nsolver.model.get_state()
logger.info(f"Initial state norm: {np.linalg.norm(x0)}")
solution, success, res, norm, properties = nsolver.NewtonSolver(x0,
                                                                Tp = 2,
                                                                dt = 0.02)