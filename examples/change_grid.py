# Move a state to another grid and/or domain size (spectral interpolation), e.g. a 128^2
# solution to 48^2 to test the stability EVP, or a solution to a new Lx as a continuation guess.
# to run: python3 change_grid.py
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path

import logging
logger = logging.getLogger(__name__)
from physics import SaltFinger
from tools import change_grid

params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'ext': '.nc'}

# old state: grid and domain it was computed on
model_old = SaltFinger(params=params, sizes=(128, 128), bounds=(0.6, 1.0))
model_old.load_state('Lx0.6_WF/solution')

# new grid and domain: (Nx, Nz) / (Lx, Lz), or (Nx, Ny, Nz) / (Lx, Ly, Lz) to extend to 3-D
model_new = SaltFinger(params=params, sizes=(48, 48), bounds=(0.6, 1.0))

x_new = change_grid(model_old, model_new)
model_new.save_state('WF_Lx0.6_48x48', x_new)
# reconverge x_new on the new grid with ECSSolver before using it (see find_eq.py)
