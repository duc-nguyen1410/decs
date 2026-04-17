# to run: mpiexec -n 4 python3 simulate.py
import numpy as np
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import dedalus.public as de
from physics.double_diffusion import ShearedDiffusiveConvection 



params = {'Ra': 1e4, 'Pr': 7.0, 'tau': 0.01, 'Lambda': 2.0, 'Ri': 4.0,
          'stress-free': True}
model = ShearedDiffusiveConvection(params=params,
                                   sizes=(128, 128),
                                   bounds=(2*np.pi, 1.0),
                                   bounded=True,
                                   mode='sim')

model.build_problems()
solver = model.ivp_problem.build_solver(de.RK222)

model.set_initial_conditions(mode='horizontal_sin', scale=0.25)

model.set_CFL(solver=solver, initial_dt=2e-3, cadence=10, safety=0.2, threshold=0.1, max_dt=0.05)

while solver.proceed:
    timestep = model.CFL.compute_timestep()
    solver.step(timestep)
    if (solver.iteration-1) % 100 == 0:
        logger.info(f"Iteration: {solver.iteration}, Time: {solver.sim_time}, Timestep: {timestep}")
        model.preview()