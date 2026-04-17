# to run: mpiexec -n 4 python3 simulate.py
import numpy as np
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import dedalus.public as de
from physics.double_diffusion import SaltFinger # import your physics model here 

sizes = (128, 128)
bounds = (0.8, 1.0)
params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0}
SFC = SaltFinger(params=params, 
                 sizes=sizes, 
                 bounds=bounds, 
                 bounded=False,
                 mode='sim')
SFC.build_ivp_problem()
solver = SFC.ivp_problem.build_solver(de.RK222)

# SFC.set_initial_conditions(mode='random')

# get a state from h5 file
# SFC.load_state('TM1_Lx0.6_new.h5')
SFC.load_state('PO2_1_Lx0.8_new.h5')
# SFC.load_state('RPO2_1_Lx0.8_new.h5')
SFC.set_CFL(solver, initial_dt=1e-3, max_dt=1e-1)


while solver.proceed:
    timestep = SFC.CFL.compute_timestep()
    solver.step(timestep)
    if solver.iteration % 10 == 0:
        logger.info(f"Iteration: {solver.iteration}, Time: {solver.sim_time}, Timestep: {timestep}")
        SFC.preview()