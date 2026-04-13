# to run: mpiexec -n 4 python3 main.py

import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
# import dedalus.public as de
from physics.double_diffusion import SaltFinger, DiffusiveConvection

sizes = (32, 32, 32)
bounds = (0.5, 0.5, 1.0)
domain = SaltFinger.create_domain(sizes, bounds)

params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 2.0}
model = SaltFinger(domain, params)

evp_problem = model.get_EVP()
solver = evp_problem.build_solver()
# SFC.load_eq_state('Base.h5')
# SFC.load_eq_state('CI1_Lx0.6.h5')
# print("loaded state")
print("total size", model.size())
x0 = model.get_state()
model.save_state('test_save_state.h5')
model.load_state('test_save_state.h5')
evals, emodes = model.solve_EVP(evp_problem, x0, N=20, target=5)

print("Eigenvalues:")
for i, eval in enumerate(evals):
    print(f"{i}: {eval}")
model.save_state('leading_evp.h5')
# # solver.set_state(0, solver.subsystems[0])
model.show_state()
