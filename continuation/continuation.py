import numpy as np
import scipy.io
import os
import h5py
import logging
import dedalus.public as de
from mpi4py import MPI
logging.getLogger('solvers').setLevel(logging.WARNING)
logging.getLogger('subsystems').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def quadraticInterpolate(xn, sn, snew):
    """
    Vectorized Neville-style quadratic interpolation.
    xn: List of 3 state vectors [x0, x1, x2], each shape (N,)
    sn: List of 3 arclength values [s0, s1, s2]
    snew: Target arclength
    """
    x = np.array(xn) # Shape (3, N)
    s = np.array(sn)
    
    # Lagrange basis polynomials for quadratic interpolation
    L0 = (snew - s[1]) * (snew - s[2]) / ((s[0] - s[1]) * (s[0] - s[2]))
    L1 = (snew - s[0]) * (snew - s[2]) / ((s[1] - s[0]) * (s[1] - s[2]))
    L2 = (snew - s[0]) * (snew - s[1]) / ((s[2] - s[0]) * (s[2] - s[1]))
    
    return L0 * x[0] + L1 * x[1] + L2 * x[2]

class Continuation:
    def __init__(self, ECSSolver, params=None):
        self.ECSSolver = ECSSolver
        # main settings
        self.mu_name = params['mu_name'] if params and 'mu_name' in params else 1
        self.odir = params['odir'] if params and 'odir' in params else './'
        self.Tsearch = params['Tsearch'] if params and 'Tsearch' in params else False
        self.Rxsearch = params['Rxsearch'] if params and 'Rxsearch' in params else False
        self.Rzsearch = params['Rzsearch'] if params and 'Rzsearch' in params else False
        self.Tp = params['Tp'] if params and 'Tp' in params else 1.0
        self.ax = params['ax'] if params and 'ax' in params else 0.0
        self.ay = params['ay'] if params and 'ay' in params else 0.0
        self.az = params['az'] if params and 'az' in params else 0.0
        # sub-parameter
        self.mu_ref = params['mu_ref'] if params and 'mu_ref' in params else 1
        self.ds_min = params['ds_min'] if params and 'ds_min' in params else 1e-4
        self.ds_max = params['ds_max'] if params and 'ds_max' in params else 0.01
        self.guess_error_min = params['guess_error_min'] if params and 'guess_error_min' in params else 0.1   # acceptable *lower* bound for guesserr
        self.guess_error_max = params['guess_error_max'] if params and 'guess_error_max' in params else 10.0   # acceptable *upper* bound for guesserr

        # History for arclength
        self.isearch = 0
        self.mu_history = []
        self.x_history = []
        self.s_history = []
        self.norm_history = []
        # History of ECS's parameters for arclength
        self.Tp_history = []
        self.ax_history = []
        self.ay_history = []
        self.az_history = []

    
    def set_parameter(self, value):
        """Updates the specific parameter in the solver's model."""
        self.ECSSolver.model.set_param(self.mu_name, value)
        
    def get_parameter(self):
        return getattr(self.ECSSolver.model, self.mu_name)
    
    def step_continuation(self, x_guess, mu_val):
        """Directly calls the solver without subprocess overhead."""
        logger.info("\n")
        self.set_parameter(mu_val)
        
        self.ECSSolver.odir = self.odir + f'search-{self.isearch}/'
        self.ECSSolver.model.odir = self.ECSSolver.odir # update this for saving time-dependent solution later
        # Call your solver's main execution method
        result = self.ECSSolver.NewtonSolver(x_guess)

        self.isearch += 1
        return result

    def save_flow_properties(self, mu, properties, filename="flow_properties.csv"):
        if self.ECSSolver.model.dist.comm.rank == 0:
            file_path = os.path.join(self.odir, filename)
            file_exists = os.path.isfile(file_path)
            keys = list(properties.keys())
            if not file_exists:
                with open(file_path, mode='w') as header:
                    # Write header if file is new
                    header_line = f"{self.mu_name}, " + ", ".join(keys)
                    header.write(header_line + "\n")
            with open(file_path, mode='a') as f:
                # Append data
                values = [f"{mu:.12f}"] + [f"{float(properties[k]):.12f}" for k in keys]
                f.write(", ".join(values) + "\n")
    
    def arc_length_continuation(self, mu_start, dmu, n_steps=50, mu_target=None):
        """Pseudo-arclength loop using direct class calls."""
        
        logger.info(f"Starting Arclength Continuation on {self.mu_name}...")
        if self.ECSSolver.model.dist.comm.rank == 0:
            if not os.path.exists(self.odir):
                os.mkdir(self.odir)

        # Initialize: Need 3 points to start the quadratic predictor
        current_mu = mu_start
        current_x = self.ECSSolver.model.get_state() # Start from solver's current state

        for i in range(3):
            sol, success, res, norm, properties = self.step_continuation(current_x, current_mu)
            if not success:
                raise RuntimeError(f"Failed to initialize continuation at mu={current_mu}")
            logger.info(f"Search {self.isearch-1}: Success | {self.mu_name} = {current_mu:.6f} | Res = {res:.2e}")
            
            self.save_flow_properties(current_mu, properties)

            self.mu_history.append(current_mu)
            self.x_history.append(sol.copy())
            self.norm_history.append(norm)
            
            if i == 0:
                self.s_history.append(0.0)
            else:
                ds_init = np.sqrt(np.linalg.norm(self.x_history[-1] - self.x_history[-2])**2 + 
                                 (self.mu_history[-1] - self.mu_history[-2])**2)
                self.s_history.append(self.s_history[-1] + ds_init)
            
            current_mu += dmu
            current_x = sol.copy()

        # Main Predictor-Corrector Loop
        ds = self.s_history[-1] - self.s_history[-2]

        for step in range(n_steps-3):
            s_next = self.s_history[-1] + ds
            
            # Predict
            mu_pred = quadraticInterpolate(self.mu_history[-3:], self.s_history[-3:], s_next)
            x_pred = quadraticInterpolate(self.x_history[-3:], self.s_history[-3:], s_next)
            
            # Correct (Direct Newton Call)
            sol, success, res, norm, properties = self.step_continuation(x_pred, mu_pred)
            
            if success:
                self.save_flow_properties(current_mu, properties)

                self.mu_history.append(mu_pred)
                self.x_history.append(sol.copy())
                self.norm_history.append(norm)
                
                # Calculate actual arclength step taken
                ds_actual = np.sqrt(np.linalg.norm(self.x_history[-1] - self.x_history[-2])**2 + 
                                   (self.mu_history[-1] - self.mu_history[-2])**2)
                self.s_history.append(self.s_history[-1] + ds_actual)
                
                logger.info(f"Search {self.isearch-1}: Success | {self.mu_name} = {mu_pred:.6f} | Res = {res:.2e}")
                
                # Optional: Save every N steps
                # if step % 5 == 0:
                #     self.save_checkpoint(step)
            else:
                logger.info(f"Step {step}: Failed. Reducing step size...")
                ds /= 2.0
                if ds < self.ds_min:
                    logger.info("Step size too small. Convergence lost.")
                    break