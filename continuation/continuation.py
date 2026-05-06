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
        self.guesserrtarget = np.sqrt(self.guess_error_min * self.guess_error_max) # target guess error for adjusting ds
        self.predictor = params['predictor'] if params and 'predictor' in params else 'tangent' # 'quadratic' or 'tangent'
        self.Ndsadjust = params['Ndsadjust'] if params and 'Ndsadjust' in params else 5 # number of steps to wait before adjusting ds
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
        logger.info("")
        self.set_parameter(mu_val)
        
        self.ECSSolver.odir = self.odir + f'search-{self.isearch}/'
        self.ECSSolver.model.odir = self.ECSSolver.odir # update this for saving time-dependent solution later
        
        # Call your solver's main execution method
        result = self.ECSSolver.NewtonSolver(x0=x_guess,Tp=self.Tp,ax=self.ax,az=self.az)
        
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
                    header_line = f"{self.mu_name}, " + ", ".join(keys) + ", dir"
                    header.write(header_line + "\n")
            with open(file_path, mode='a') as f:
                # Append data
                values = [f"{mu:.12f}"] + [f"{float(properties[k]):.12f}" for k in keys]
                f.write(", ".join(values) + f", search-{self.isearch-1}/\n")
    def save_mu(self,mu):
        if self.ECSSolver.model.dist.comm.rank == 0:
            file_path = os.path.join(self.ECSSolver.odir, "mu.txt")
            # print(file_path)
            with open(file_path, mode='w') as file:
                file.write(f"{mu}")
    def check_residual(self, x_guess, mu_val):
        self.set_parameter(mu_val)
        # Initial guess includes state and parameters
        xi = np.concatenate([x_guess, 
                            [self.Tp] if self.Tsearch else [], 
                            [self.ax] if self.Rxsearch else [], 
                            [self.az] if self.Rzsearch else []])  
        nonlinear_res = self.ECSSolver.NonlinearOperator(xi)
        return np.linalg.norm(nonlinear_res)
    def arc_length_continuation(self, mu_start, dmu, n_steps=50, mu_target=None):
        """Pseudo-arclength loop using direct class calls."""
        N_ = self.ECSSolver.model.size()
        
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
            
            # save current mu
            self.save_mu(current_mu)

            self.save_flow_properties(current_mu, properties)

            self.mu_history.append(current_mu)
            self.x_history.append(sol[:N_].copy())
            self.norm_history.append(norm)
            if self.Tsearch:
                self.Tp_history.append(sol[N_+self.Tsearch-1])
            if self.Rxsearch:
                self.ax_history.append(sol[N_+self.Tsearch+self.Rxsearch-1])
            if self.Rzsearch:
                self.az_history.append(sol[N_+self.Tsearch+self.Rxsearch+self.Rzsearch-1])

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
            # Check if we've reached the target mu
            if mu_target is not None:
                if (self.mu_history[-1] - mu_target) * (self.mu_history[-2] - mu_target) < 0:
                    logger.info("Reached target mu.")
                    break

            success = False
            # Inner adjustment loop: Find a decent initial guess
            for iadjust in range(self.Ndsadjust):
                # Predict next point
                if self.predictor == 'quadratic':
                    # Predictor using quadratic interpolation in arclength space
                    s_next = self.s_history[-1] + ds
                    mu_pred = quadraticInterpolate(self.mu_history[-3:], self.s_history[-3:], s_next)
                    x_pred = quadraticInterpolate(self.x_history[-3:], self.s_history[-3:], s_next)
                    if self.Tsearch:
                        Tnew = quadraticInterpolate(self.Tp_history[-3:], self.s_history[-3:], s_next)
                        self.Tp = Tnew
                    if self.Rxsearch:
                        axnew = quadraticInterpolate(self.ax_history[-3:], self.s_history[-3:], s_next)
                        self.ax = axnew
                    if self.Rzsearch:
                        aznew = quadraticInterpolate(self.az_history[-3:], self.s_history[-3:], s_next)
                        self.az = aznew
                else:
                    # Predictor using tangent control
                    dx = self.x_history[-1] - self.x_history[-2]
                    dmu = self.mu_history[-1] - self.mu_history[-2]
                    norm = np.sqrt(np.linalg.norm(dx)**2 + dmu**2)
                    if norm < 1e-14:
                        raise RuntimeError("Tangent vector collapsed.")
                    tx = dx / norm
                    tmu = dmu / norm
                    x_pred = self.x_history[-1] + ds * tx
                    mu_pred = self.mu_history[-1] + ds * tmu
                    if self.Tsearch:
                        dTp = self.Tp_history[-1] - self.Tp_history[-2]
                        Tnew = self.Tp_history[-1] + ds * dTp / norm
                        self.Tp = Tnew
                    if self.Rxsearch:
                        dax = self.ax_history[-1] - self.ax_history[-2]
                        axnew = self.ax_history[-1] + ds * dax / norm
                        self.ax = axnew
                    if self.Rzsearch:
                        daz = self.az_history[-1] - self.az_history[-2]
                        aznew = self.az_history[-1] + ds * daz / norm
                        self.az = aznew
                guess_err = self.check_residual(x_pred, mu_pred)

                logger.info(f"dsmin == {self.ds_min}")
                logger.info(f"ds    == {ds}")
                logger.info(f"dsmax == {self.ds_max}")
                if self.Tsearch:
                    logger.info(f"Predicted Tp = {self.Tp:.6f}")
                if self.Rxsearch:
                    logger.info(f"Predicted ax = {self.ax:.6f}")
                if self.Rzsearch:                
                    logger.info(f"Predicted az = {self.az:.6f}")
                logger.info(f"Predicted {self.mu_name} = {mu_pred:.6f}")
                logger.info(f"guesserrmin == {self.guess_error_min}")
                logger.info(f"guesserr    == {guess_err}")
                logger.info(f"guesserrmax == {self.guess_error_max}")

                # Check residual of the guess to decide if we need to adjust ds before correction
                if self.guess_error_min <= guess_err <= self.guess_error_max:
                    # Guess is "just right" - move to full solve
                    break
                elif guess_err < self.guess_error_min:
                    # Guess is too good! We are being too conservative.
                    # Increase ds for the NEXT step, but use this guess now.
                    ds = min(ds * 1.5, self.ds_max)
                    break
                else:
                    # Guess error is too high (Newton will likely fail)
                    # Shrink ds and re-predict within this adjustment loop
                    ds = max(ds * (self.guesserrtarget / guess_err)**(1/3), self.ds_min)
                    if ds == self.ds_min:
                        break # Cannot shrink further, try solving anyway

            # Correct (Direct Newton Call)
            sol, success, res, norm, properties = self.step_continuation(x_pred, mu_pred)
            
            if success:
                # save current mu
                self.save_mu(mu_pred)
                self.save_flow_properties(mu_pred, properties)

                self.mu_history.append(mu_pred)
                self.x_history.append(sol[:N_].copy())
                self.norm_history.append(norm)
                if self.Tsearch:
                    self.Tp_history.append(sol[N_+self.Tsearch-1])
                if self.Rxsearch:
                    self.ax_history.append(sol[N_+self.Tsearch+self.Rxsearch-1])
                if self.Rzsearch:
                    self.az_history.append(sol[N_+self.Tsearch+self.Rxsearch+self.Rzsearch-1])
                
                # Calculate actual arclength step taken
                ds_actual = np.sqrt(np.linalg.norm(self.x_history[-1] - self.x_history[-2])**2 + 
                                   (self.mu_history[-1] - self.mu_history[-2])**2)
                self.s_history.append(self.s_history[-1] + ds_actual)
                
                logger.info(f"Search {self.isearch-1}: Success | {self.mu_name} = {mu_pred:.6f} | Res = {res:.2e}")
                # increase step size if convergence is good, decrease if not
                ds = min(ds * 1.3, self.ds_max)
            else:
                logger.info(f"Step {step}: Failed. Reducing step size...")
                ds = max(ds * 0.7, self.ds_min)
                
            
            