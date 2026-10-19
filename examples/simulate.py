# to run: mpiexec -n 4 python3 simulate.py
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path

import numpy as np
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import dedalus.public as de
from physics.salt_finger import SaltFinger # import your physics model here 

sizes = (128, 128)
bounds = (0.5, 1.0)
Ra = 1e5
Pr = 0.25
Rrho = 40.0
tau = 0.01
params = {'Ra': Ra, 'Pr': Pr, 'tau': tau, 'Rrho': Rrho,
          'odir':f"Lx{bounds[0]}_Ra{Ra}_Pr{Pr}_Rrho{Rrho}_simulation/",
          'init_dt': 1e-4,
        #   'stokes': False,
        #   'temperature_feedback': False,
          'cadence': 10,
          'safety': 0.5,
          'threshold': 0.05,
          'min_change': 0.5,
          'max_change': 1.5,
          'max_dt': 0.1,
          'ext': '.nc'} # set extension of output file: .nc or .h5
SFC = SaltFinger(params=params, 
                 sizes=sizes, 
                 bounds=bounds, 
                 mode='sim')
SFC.build_problems() # Build IVP problem
solver = SFC.ivp_problem.build_solver(de.RK222)

SFC.set_initial_conditions(mode='random')

# SFC.load_state('state_file_path') # get a state
# SFC.add_perturbation() # pertub current state by a random noise

SFC.set_CFL(solver)

# saving data for post-processing if needed, as in Dedalus normally
SFC.set_snapshots(solver=solver,sim_dt=0.001)
SFC.set_flowproperties(solver=solver,sim_dt=0.001)
SFC.set_meanprofiles(solver=solver,sim_dt=0.001)

solver.stop_sim_time = 100.0
while solver.proceed:
    timestep = SFC.CFL.compute_timestep()
    solver.step(timestep)
    if (solver.iteration-1) % 100 == 0:
        logger.info(f"Iteration: {solver.iteration}, Time: {solver.sim_time}, Timestep: {timestep}")
        prop = SFC.get_flow_properties() # Get and print properties of current state, eg. Nusselt number
        logger.info(prop)
        SFC.preview() # Create a window to show current state
        
    if (solver.iteration-1) % 1000 == 0:
        SFC.save_state(params['odir']+f"step_{solver.iteration}") # save current state into .nc files using netCDF4 to open in Paravew if needed