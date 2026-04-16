# to run: mpiexec -n 4 python3 main.py

import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
# import dedalus.public as de
from physics.double_diffusion import SaltFinger, DiffusiveConvection

sizes = (32, 32, 32)
bounds = (0.5, 0.5, 1.0)
params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 2.0}
model = SaltFinger(params=params, sizes=sizes, bounds=bounds)
model.build_evp_problem()

x0 = model.get_state()

evals, emodes = model.solve_EVP(x0, N=20, target=5)

logger.info("Eigenvalues:")
for i, eval in enumerate(evals):
    logger.info(f"{i}: {eval}")
model.save_state('leading_evp.h5')
# model.show_state()
model.preview3D()
