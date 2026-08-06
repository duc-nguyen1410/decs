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
        self.mu_name = (params or {}).get('mu_name', 'Lx') #
        self.continuation_type = (params or {}).get('continuation_type', 'arclength') # 'natural' or 'arclength'
        self.odir = (params or {}).get('odir', './')
        self.Tsearch = (params or {}).get('Tsearch', False)
        self.Rxsearch = (params or {}).get('Rxsearch', False)
        self.Rysearch = (params or {}).get('Rysearch', False)
        self.Rzsearch = (params or {}).get('Rzsearch', False)
        self.Tp = (params or {}).get('Tp', 1.0)
        self.ax = (params or {}).get('ax', 0.0)
        self.ay = (params or {}).get('ay', 0.0)
        self.az = (params or {}).get('az', 0.0)
        # sub-parameter
        self.mu_ref = (params or {}).get('mu_ref', 1)
        self.ds_min = (params or {}).get('ds_min', 1e-4)
        self.ds_max = (params or {}).get('ds_max', 0.01)
        self.guess_error_min = (params or {}).get('guess_error_min', 0.1)   # acceptable *lower* bound for guesserr
        self.guess_error_max = (params or {}).get('guess_error_max', 10.0)   # acceptable *upper* bound for guesserr
        self.guesserrtarget = np.sqrt(self.guess_error_min * self.guess_error_max) # target guess error for adjusting ds
        self.predictor = (params or {}).get('predictor', 'tangent') # 'quadratic' or 'tangent'
        self.Ndsadjust = (params or {}).get('Ndsadjust', 5) # number of steps to wait before adjusting ds
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

        current_search_dir = os.path.join(self.odir, f'search-{self.isearch}/')
        self.ECSSolver.odir = current_search_dir
        self.ECSSolver.model.odir = current_search_dir # update this for saving time-dependent solution later
        
        # Call your solver's main execution method
        result = self.ECSSolver.NewtonSolver(x0=x_guess,
                                             Tsearch=self.Tsearch,
                                             Rxsearch=self.Rxsearch,
                                             Rysearch=self.Rysearch,
                                             Rzsearch=self.Rzsearch,
                                             Tp=self.Tp,
                                             ax=self.ax if self.Rxsearch else 0.0,
                                             ay=self.ay if self.Rysearch else 0.0,
                                             az=self.az if self.Rzsearch else 0.0)

        # Extract success flag assuming result = (sol, success, res, norm, properties)
        success = result[1]
        if success:
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
                            [self.ay] if self.Rysearch else [], 
                            [self.az] if self.Rzsearch else []])  
        nonlinear_res = self.ECSSolver.NonlinearOperator(xi)
        return np.linalg.norm(nonlinear_res)
    def run_continuation(self, mu_start, dmu, n_steps=50, mu_target=None):
        """ Run the natural/arclength continuation process. """
        N_ = self.ECSSolver.model.size()
        self.ECSSolver.Tsearch = self.Tsearch
        self.ECSSolver.Rxsearch = self.Rxsearch
        self.ECSSolver.Rysearch = self.Rysearch
        self.ECSSolver.Rzsearch = self.Rzsearch
        munorm = abs(self.mu_ref) if abs(self.mu_ref) > 1e-12 else abs(mu_start)
        if munorm < 1e-12:
            munorm = 1.0  # Avoid division by zero
        
        logger.info(f"Starting {self.continuation_type.capitalize()} Continuation on {self.mu_name}...")
        if self.ECSSolver.model.dist.comm.rank == 0:
            os.makedirs(self.odir, exist_ok=True)

        # Initialize: Need 3 points to start the quadratic predictor
        current_mu = mu_start
        current_x = self.ECSSolver.model.get_state() # Start from solver's current state

        for i in range(3):
            # solution = [x,[Tp],[ax],[ay],[az]]
            sol, success, res, norm, properties = self.step_continuation(current_x, current_mu)
            if not success:
                raise RuntimeError(f"Failed to initialize continuation at mu={current_mu}")
            logger.info(f"Search {self.isearch-1}: Success | {self.mu_name} = {current_mu:.6f} | Res = {res:.2e}")

            # Extract only the state vector x (first N_ entries)
            x_sol = sol[:N_].copy()

            # save current mu
            self.save_mu(current_mu)
            self.save_flow_properties(current_mu, properties)

            self.mu_history.append(current_mu)
            self.x_history.append(x_sol)
            self.norm_history.append(norm)

            if self.Tsearch:
                self.Tp = sol[N_+self.Tsearch-1]
                self.Tp_history.append(self.Tp)
            if self.Rxsearch:
                self.ax = sol[N_+self.Tsearch+self.Rxsearch-1]
                self.ax_history.append(self.ax)
            if self.Rysearch:
                self.ay = sol[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1]
                self.ay_history.append(self.ay)
            if self.Rzsearch:
                self.az = sol[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1]
                self.az_history.append(self.az)

            if i == 0:
                self.s_history.append(0.0)
            else:
                x_norm = np.linalg.norm(self.x_history[-1])
                if x_norm < 1e-12:
                    x_norm = 1.0  # Avoid division by zero
                ds_init = np.sqrt((np.linalg.norm(self.x_history[-1] - self.x_history[-2])/x_norm)**2 + 
                                 ((self.mu_history[-1] - self.mu_history[-2])/munorm)**2)
                self.s_history.append(self.s_history[-1] + ds_init)
            
            current_mu += dmu
            current_x = x_sol # update state vector x

        # Main Predictor-Corrector Loop
        ds = self.s_history[-1] - self.s_history[-2]
        step = 0
        while step < (n_steps - 3):
            # Check if we've reached the target mu
            if mu_target is not None:
                if (self.mu_history[-1] - mu_target) * (self.mu_history[-2] - mu_target) < 0:
                    logger.info("Reached target mu.")
                    break

            success = False
            # Inner adjustment loop: Find a decent initial guess
            for iadjust in range(self.Ndsadjust):
                # Predict next point
                if self.continuation_type == "natural":
                    # ==========================================
                    # Natural Parameter Continuation Predictor
                    # ==========================================
                    dmu_step = ds  # Reusing ds variable as the step size for mu, or use a dedicated dmu

                    mu_pred = self.mu_history[-1] + dmu_step

                    # Linear extrapolation of x with respect to mu
                    dmu_prev = self.mu_history[-1] - self.mu_history[-2]
                    if abs(dmu_prev) < 1e-14:
                        raise RuntimeError("Parameter step collapsed in natural continuation.")

                    dx_dmu = (self.x_history[-1] - self.x_history[-2]) / dmu_prev
                    x_pred = self.x_history[-1] + dx_dmu * dmu_step

                    # Optional variables extrapolation w.r.t mu
                    if self.Tsearch:
                        dTp_dmu = (self.Tp_history[-1] - self.Tp_history[-2]) / dmu_prev
                        Tp_pred = self.Tp_history[-1] + dTp_dmu * dmu_step
                    else:
                        Tp_pred = None

                    if self.Rxsearch:
                        dax_dmu = (self.ax_history[-1] - self.ax_history[-2]) / dmu_prev
                        ax_pred = self.ax_history[-1] + dax_dmu * dmu_step
                    else:
                        ax_pred = None

                    if self.Rysearch:
                        day_dmu = (self.ay_history[-1] - self.ay_history[-2]) / dmu_prev
                        ay_pred = self.ay_history[-1] + day_dmu * dmu_step
                    else:
                        ay_pred = None

                    if self.Rzsearch:
                        daz_dmu = (self.az_history[-1] - self.az_history[-2]) / dmu_prev
                        az_pred = self.az_history[-1] + daz_dmu * dmu_step
                    else:
                        az_pred = None

                else:
                    # ==========================================
                    # Arclength Continuation Predictors
                    # ==========================================
                    if self.predictor == 'quadratic':
                        # Predictor using quadratic interpolation in arclength space
                        s_next = self.s_history[-1] + ds
                        mu_pred = quadraticInterpolate(self.mu_history[-3:], self.s_history[-3:], s_next)
                        x_pred = quadraticInterpolate(self.x_history[-3:], self.s_history[-3:], s_next)
                        Tp_pred = quadraticInterpolate(self.Tp_history[-3:], self.s_history[-3:], s_next) if self.Tsearch else None
                        ax_pred = quadraticInterpolate(self.ax_history[-3:], self.s_history[-3:], s_next) if self.Rxsearch else None
                        ay_pred = quadraticInterpolate(self.ay_history[-3:], self.s_history[-3:], s_next) if self.Rysearch else None
                        az_pred = quadraticInterpolate(self.az_history[-3:], self.s_history[-3:], s_next) if self.Rzsearch else None
                    else:
                        # Predictor using tangent control
                        dx = self.x_history[-1] - self.x_history[-2]
                        dmu_val = self.mu_history[-1] - self.mu_history[-2]
                        tangent_norm = np.sqrt(np.linalg.norm(dx)**2 + (dmu_val/munorm)**2)
                        if tangent_norm < 1e-14:
                            raise RuntimeError("Tangent vector collapsed.")
                        x_pred = self.x_history[-1] + ds * (dx / tangent_norm)
                        mu_pred = self.mu_history[-1] + ds * (dmu_val / tangent_norm)
                        Tp_pred = self.Tp_history[-1] + ds * (self.Tp_history[-1] - self.Tp_history[-2]) / tangent_norm if self.Tsearch else None
                        ax_pred = self.ax_history[-1] + ds * (self.ax_history[-1] - self.ax_history[-2]) / tangent_norm if self.Rxsearch else None
                        ay_pred = self.ay_history[-1] + ds * (self.ay_history[-1] - self.ay_history[-2]) / tangent_norm if self.Rysearch else None
                        az_pred = self.az_history[-1] + ds * (self.az_history[-1] - self.az_history[-2]) / tangent_norm if self.Rzsearch else None

                # Assign predicted aux values temporarily
                if self.Tsearch: self.Tp = Tp_pred
                if self.Rxsearch: self.ax = ax_pred
                if self.Rysearch: self.ay = ay_pred
                if self.Rzsearch: self.az = az_pred
                guess_err = self.check_residual(x_pred, mu_pred)

                logger.info(f"dsmin == {self.ds_min}")
                logger.info(f"ds    == {ds}")
                logger.info(f"dsmax == {self.ds_max}")
                if self.Tsearch:
                    logger.info(f"Predicted Tp = {self.Tp:.6f}")
                if self.Rxsearch:
                    logger.info(f"Predicted ax = {self.ax:.6f}")
                if self.Rysearch:
                    logger.info(f"Predicted ay = {self.ay:.6f}")
                if self.Rzsearch:                
                    logger.info(f"Predicted az = {self.az:.6f}")
                logger.info(f"Predicted {self.mu_name} = {mu_pred:.6f}")
                logger.info(f"guesserrmin == {self.guess_error_min}")
                logger.info(f"guesserr    == {guess_err}")
                logger.info(f"guesserrmax == {self.guess_error_max}")

                # Evaluate initial guess quality
                if guess_err <= self.guess_error_max:
                    break # Guess is acceptable, proceed to solve
                else:
                    # Guess error is too high (Newton will likely fail)
                    # Shrink ds and re-predict within this adjustment loop
                    ds_new = max(ds * (self.guesserrtarget / guess_err)**(1/3), self.ds_min)
                    if ds_new == ds: # Hit ds_min limit
                        break # Reached ds_min limit, attempt solve anyway
                    ds = ds_new

            # Correct (Direct Newton Call)
            sol, success, res, norm, properties = self.step_continuation(x_pred, mu_pred)
            
            if success:
                x_sol = sol[:N_].copy()
                # save current mu
                self.save_mu(mu_pred)
                self.save_flow_properties(mu_pred, properties)

                self.mu_history.append(mu_pred)
                self.x_history.append(x_sol)
                self.norm_history.append(norm)

                if self.Tsearch:
                    self.Tp = sol[N_+self.Tsearch-1]
                    self.Tp_history.append(self.Tp)
                if self.Rxsearch:
                    self.ax = sol[N_+self.Tsearch+self.Rxsearch-1]
                    self.ax_history.append(self.ax)
                if self.Rysearch:
                    self.ay = sol[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1]
                    self.ay_history.append(self.ay)
                if self.Rzsearch:
                    self.az = sol[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1]
                    self.az_history.append(self.az)
                
                # Calculate actual arclength step taken
                ds_actual = np.sqrt(np.linalg.norm(self.x_history[-1] - self.x_history[-2])**2 + 
                                   ((self.mu_history[-1] - self.mu_history[-2])/munorm)**2)
                self.s_history.append(self.s_history[-1] + ds_actual)
                
                logger.info(f"Search {self.isearch-1}: Success | {self.mu_name} = {mu_pred:.6f} | Res = {res:.2e}")

                # Modest step size growth (only if guess quality was very high)
                if guess_err < self.guess_error_min:
                    ds = min(ds * 1.15, self.ds_max)

                step += 1  # Advance step counter only on success
            else:
                logger.info(f"Step {step}: Failed. Reducing step size...")
                # if np.isclose(ds, self.ds_min):
                #     raise RuntimeError(f"Continuation stuck: Newton failed at minimum step size ds_min={self.ds_min}")
                # Restore attributes to last converged state
                if self.Tsearch and self.Tp_history: self.Tp = self.Tp_history[-1]
                if self.Rxsearch and self.ax_history: self.ax = self.ax_history[-1]
                if self.Rysearch and self.ay_history: self.ay = self.ay_history[-1]
                if self.Rzsearch and self.az_history: self.az = self.az_history[-1]
                ds = max(ds * 0.5, self.ds_min)
                
            
            