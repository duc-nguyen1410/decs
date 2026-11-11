"""
Periodic orbit by multiple shooting: the period is split into n_segments segments, each with its
own initial state, which conditions long or strongly unstable orbits better than single shooting.
Same solve arguments and output as find_po.py; the first segment is the saved solution.
Choose n_segments ~ lambda*Tp/1.5 (lambda: largest Floquet / Lyapunov exponent), see
docs/finding_po_rpo.md. For a relative periodic orbit add Rxsearch=True and ax, as in find_rpo.py.

Run from the examples folder:  python3 find_po_multishoot.py
(about 25 minutes on one core, mostly the stability analysis; PO3 is already converged, so Newton
stops at once: start from a perturbed state or another Tp to see it iterate)
"""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path
import logging
logger = logging.getLogger(__name__)
import numpy as np
from physics import SaltFinger
from ecs_core import MultiShootECSSolver

sizes = (128, 128)
bounds = (0.8, 1.0)
params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0,
          'init_dt': 3e-4,
          'ext': '.nc'}
model = SaltFinger(params=params, sizes=sizes, bounds=bounds)
model.build_problems()

n_segments = 3
nsolver_params = {'n_segments': n_segments,
                  'max_iter': 20,
                  'tol': 1e-9,
                  'trust_radius': 20.0*np.sqrt(n_segments), # bounds the step over all segments
                  'krylov_dim': 50,
                  'krylov_dim_min': 5,
                  'gmres_min_error': 1e-20,
                  'computeStability': True, # Floquet stability of the full period
                  'Neigen': 60,
                  'projectNeutralDrift': True,
                  'odir': f"test_PO_multishoot/"
                  }
nsolver = MultiShootECSSolver(model, nsolver_params)
nsolver.model.use_CFL = False
nsolver.model.load_state('Lx0.8_PO3/solution')
x0 = nsolver.model.get_state() # segments 2..m are built by integrating x0 over Tp/m each
logger.info(f"Initial state norm: {np.linalg.norm(x0)}")
solution, success, res, norm, properties = nsolver.solve(x0=x0,
                                                         Tsearch=True,
                                                         Tp=0.21522529995179002)
