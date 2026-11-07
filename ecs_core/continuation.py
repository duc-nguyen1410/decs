"""
Parameter continuation of exact coherent states along a branch in mu
(any model parameter, e.g. 'Lx' or 'Ra').

Predictor-corrector loop
------------------------
    1. initialise : three Newton solves at mu_start, mu_start + dmu, mu_start + 2 dmu
    2. predict    : next point from the branch history (natural, quadratic or tangent)
    3. choose ds  : grow/shrink the step until the predicted residual lies in
                    [guess_error_min, guess_error_max]
    4. correct    : Newton solve (ECSSolver) at the predicted point
    5. record     : on success append to the history, else halve ds and retry

Arclength s uses the scaled metric  ds^2 = (|dx| / |x|)^2 + (dmu / max(|mu|, |mu_ref|))^2.
Searched ECS parameters (Tp, ax, ay, az) are continued alongside x and mu.
"""
import os
import logging
import numpy as np
from . import output

logger = logging.getLogger(__name__)


def quadratic_interpolate(xn, sn, snew):
    """Quadratic (Lagrange) interpolation of 3 points xn (scalars or arrays) at
    arclengths sn, evaluated at snew."""
    x = np.array(xn)
    s = np.array(sn)
    L0 = (snew - s[1])*(snew - s[2])/((s[0] - s[1])*(s[0] - s[2]))
    L1 = (snew - s[0])*(snew - s[2])/((s[1] - s[0])*(s[1] - s[2]))
    L2 = (snew - s[0])*(snew - s[1])/((s[2] - s[0])*(s[2] - s[1]))
    return L0*x[0] + L1*x[1] + L2*x[2]



class Continuation:
    def __init__(self, ECSSolver, params=None):
        p = params or {}
        self.ECSSolver = ECSSolver
        # main settings
        self.mu_name = p.get('mu_name', 'Lx')
        self.continuation_type = p.get('continuation_type', 'arclength')  # 'natural' or 'arclength'
        self.predictor = p.get('predictor', 'tangent')                    # arclength: 'tangent' or 'quadratic'
        self.odir = p.get('odir', './')
        # searched ECS parameters and their initial values
        self.Tsearch = p.get('Tsearch', False)
        self.Rxsearch = p.get('Rxsearch', False)
        self.Rysearch = p.get('Rysearch', False)
        self.Rzsearch = p.get('Rzsearch', False)
        self.Tp = p.get('Tp', 1.0)
        self.ax = p.get('ax', 0.0)
        self.ay = p.get('ay', 0.0)
        self.az = p.get('az', 0.0)
        # step-size control
        self.mu_ref = p.get('mu_ref', 1)          # scale of mu in the arclength metric
        self.ds_min = p.get('ds_min', 1e-4)
        self.ds_max = p.get('ds_max', 0.01)
        self.guess_error_min = p.get('guess_error_min', 0.1)   # accept predictions with residual
        self.guess_error_max = p.get('guess_error_max', 10.0)  # in [guess_error_min, guess_error_max]
        self.guesserrtarget = np.sqrt(self.guess_error_min*self.guess_error_max)
        self.Ndsadjust = p.get('Ndsadjust', 5)    # max predictor retries per step
        # branch history
        self.isearch = 0
        self.mu_history = []
        self.x_history = []
        self.s_history = []
        self.norm_history = []
        self.dir_history = []       # search-<i>/ folder of each point (for restarts)
        self.Tp_history = []
        self.ax_history = []
        self.ay_history = []
        self.az_history = []

    # ------------------------------------------------------------------
    # Searched ECS parameters (Tp, ax, ay, az)
    # ------------------------------------------------------------------
    def _searched(self):
        flags = (('Tp', self.Tsearch), ('ax', self.Rxsearch), ('ay', self.Rysearch), ('az', self.Rzsearch))
        return [name for name, on in flags if on]

    def _history(self, name):
        return getattr(self, f'{name}_history')

    def _set_searched(self, values):
        for name, value in values.items():
            setattr(self, name, value)

    def _restore_searched(self):
        """Reset the searched parameters to the last converged point."""
        for name in self._searched():
            if self._history(name):
                setattr(self, name, self._history(name)[-1])

    # ------------------------------------------------------------------
    # Continuation parameter mu
    # ------------------------------------------------------------------
    def set_parameter(self, value):
        """Set mu in the model."""
        self.ECSSolver.model.set_param(self.mu_name, value)

    def get_parameter(self):
        return getattr(self.ECSSolver.model, self.mu_name)

    # ------------------------------------------------------------------
    # Corrector and residual of a guess
    # ------------------------------------------------------------------
    def step_continuation(self, x_guess, mu_val):
        """Newton solve at mu_val from x_guess, writing into odir/search-<isearch>/.
        Returns the result of ECSSolver.solve: (xi, success, res, norm, properties)."""
        logger.info("")
        self.set_parameter(mu_val)
        search_dir = os.path.join(self.odir, f'search-{self.isearch}/')
        self.ECSSolver.odir = search_dir
        self.ECSSolver.model.odir = search_dir  # used when saving time-dependent solutions
        result = self.ECSSolver.solve(x0=x_guess,
                                             Tsearch=self.Tsearch, Rxsearch=self.Rxsearch,
                                             Rysearch=self.Rysearch, Rzsearch=self.Rzsearch,
                                             Tp=self.Tp,
                                             ax=self.ax if self.Rxsearch else 0.0,
                                             ay=self.ay if self.Rysearch else 0.0,
                                             az=self.az if self.Rzsearch else 0.0)
        if result[1]:
            self.isearch += 1
        return result

    def check_residual(self, x_guess, mu_val):
        """Newton residual norm of the guess (x_guess, current Tp/ax/...) at mu_val."""
        self.set_parameter(mu_val)
        xi = self.ECSSolver.pack_xi(x_guess, self.Tp, self.ax, self.ay, self.az)
        return np.linalg.norm(self.ECSSolver.residual(xi))

    # ------------------------------------------------------------------
    # Branch history
    # ------------------------------------------------------------------
    def _record(self, mu, sol, norm, properties):
        """Save and append a converged point; returns its state x."""
        x, Tp, ax, ay, az = self.ECSSolver.unpack_xi(sol)
        self.save_mu(mu)
        self.save_flow_properties(mu, properties)
        self.mu_history.append(mu)
        self.x_history.append(x)
        self.norm_history.append(norm)
        self.dir_history.append(f'search-{self.isearch-1}/')
        values = {'Tp': Tp, 'ax': ax, 'ay': ay, 'az': az}
        for name in self._searched():
            setattr(self, name, values[name])
            self._history(name).append(values[name])
        return x

    def _step_norm(self, x_curr, x_prev, mu_curr, mu_prev):
        """Arclength between two branch points in the scaled metric."""
        x_norm_val = max(np.linalg.norm(x_prev), 1e-12)
        mu_norm_val = max(abs(mu_curr), self._mu_scale_min)
        return np.sqrt((np.linalg.norm(x_curr - x_prev)/x_norm_val)**2 +
                       ((mu_curr - mu_prev)/mu_norm_val)**2)

    def _initialize(self, mu_start, dmu):
        """Three Newton solves at mu_start + i*dmu; returns the initial ds."""
        mu = mu_start
        x = self.ECSSolver.model.get_state()
        for _ in range(3):
            sol, success, res, norm, properties = self.step_continuation(x, mu)
            if not success:
                raise RuntimeError(f"Failed to initialize continuation at mu={mu}")
            logger.info(f"Search {self.isearch-1}: Success | {self.mu_name} = {mu:.6f} | Res = {res:.2e}")
            x = self._record(mu, sol, norm, properties)
            mu += dmu

        X, M = self.x_history, self.mu_history
        s1 = self._step_norm(X[-2], X[-3], M[-2], M[-3])
        s2 = self._step_norm(X[-1], X[-2], M[-1], M[-2])
        self.s_history = [0.0, s1, s1 + s2]
        logger.info("Initial s")
        for i, s in enumerate(self.s_history):
            logger.info(f"s[{i}] == {s}")
        self._log_ds(s2)
        self.save_checkpoint(s2)
        return s2

    # ------------------------------------------------------------------
    # Predictors: return (mu_pred, x_pred, {searched parameter: predicted value})
    # ------------------------------------------------------------------
    def _predict(self, ds):
        if self.continuation_type == "natural":
            return self._predict_natural(ds)
        if self.predictor == 'quadratic':
            return self._predict_quadratic(ds)
        return self._predict_tangent(ds)

    def _predict_natural(self, ds):
        """Linear extrapolation in mu. ds is the dimensionless step of the arclength
        metric, so the mu step is ds * max(|mu|, |mu_ref|) (same scaling as _step_norm)."""
        M, X = self.mu_history, self.x_history
        dmu_prev = M[-1] - M[-2]
        if abs(dmu_prev) < 1e-14:
            raise RuntimeError("Parameter step collapsed in natural continuation.")
        dmu_step = ds*max(abs(M[-1]), self._mu_scale_min)*np.sign(dmu_prev)
        mu_pred = M[-1] + dmu_step
        x_pred = X[-1] + ((X[-1] - X[-2])/dmu_prev)*dmu_step
        aux = {}
        for name in self._searched():
            h = self._history(name)
            aux[name] = h[-1] + ((h[-1] - h[-2])/dmu_prev)*dmu_step
        return mu_pred, x_pred, aux

    def _predict_quadratic(self, ds):
        """Quadratic extrapolation through the last 3 points in arclength."""
        s_next = self.s_history[-1] + ds
        s_pts = self.s_history[-3:]
        mu_pred = quadratic_interpolate(self.mu_history[-3:], s_pts, s_next)
        x_pred = quadratic_interpolate(self.x_history[-3:], s_pts, s_next)
        aux = {name: quadratic_interpolate(self._history(name)[-3:], s_pts, s_next)
               for name in self._searched()}
        return mu_pred, x_pred, aux

    def _predict_tangent(self, ds):
        """Step ds along the secant through the last 2 points (scaled metric)."""
        M, X = self.mu_history, self.x_history
        dx = X[-1] - X[-2]
        dmu = M[-1] - M[-2]
        daux = {name: self._history(name)[-1] - self._history(name)[-2] for name in self._searched()}

        scaled_sq_sum = (np.linalg.norm(dx)/max(np.linalg.norm(X[-2]), 1e-12))**2 \
            + (dmu/max(abs(M[-1]), self._mu_scale_min))**2
        for name, d in daux.items():
            scaled_sq_sum += (d/max(abs(self._history(name)[-2]), 1e-12))**2
        tangent_norm = np.sqrt(scaled_sq_sum)
        if tangent_norm < 1e-14:
            raise RuntimeError("Tangent vector collapsed.")

        x_pred = X[-1] + ds*(dx/tangent_norm)
        mu_pred = M[-1] + ds*(dmu/tangent_norm)
        aux = {name: self._history(name)[-1] + ds*(d/tangent_norm) for name, d in daux.items()}
        return mu_pred, x_pred, aux

    # ------------------------------------------------------------------
    # Step-size selection
    # ------------------------------------------------------------------
    def _log_ds(self, ds):
        logger.info(f"dsmin == {self.ds_min}")
        logger.info(f"ds    == {ds}")
        logger.info(f"dsmax == {self.ds_max}")

    def _choose_step(self, ds):
        """Predict, and adjust ds (up to Ndsadjust times) until the residual of
        the prediction lies in [guess_error_min, guess_error_max].
        Returns (mu_pred, x_pred, ds); sets self.Tp/ax/... to the predicted values."""
        for _ in range(self.Ndsadjust):
            mu_pred, x_pred, aux = self._predict(ds)
            self._set_searched(aux)

            if self.Tsearch and aux['Tp'] <= 0:
                logger.info(f"Non-positive Tp predicted ({aux['Tp']}). Shrinking ds...")
                ds = max(ds*0.5, self.ds_min)
                continue

            guess_err = self.check_residual(x_pred, mu_pred)
            self._log_ds(ds)
            for name in self._searched():
                logger.info(f"Predicted {name} = {getattr(self, name):.6f}")
            logger.info(f"Predicted {self.mu_name} = {mu_pred:.6f}")
            logger.info(f"guesserrmin == {self.guess_error_min}")
            logger.info(f"guesserr    == {guess_err}")
            logger.info(f"guesserrmax == {self.guess_error_max}")

            if self.guess_error_min <= guess_err <= self.guess_error_max:
                break                                   # good guess
            elif guess_err < self.guess_error_min:
                if np.isclose(ds, self.ds_max):
                    break                               # already at the largest step
                ds = np.clip(ds*1.15, self.ds_min, self.ds_max)
            else:                                       # Newton would likely fail
                ds = np.clip(ds*(self.guesserrtarget/guess_err)**(1/3), self.ds_min, self.ds_max)
        return mu_pred, x_pred, ds

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _setup(self):
        solver = self.ECSSolver
        solver.Tsearch, solver.Rxsearch = self.Tsearch, self.Rxsearch
        solver.Rysearch, solver.Rzsearch = self.Rysearch, self.Rzsearch
        self._mu_scale_min = abs(self.mu_ref) if abs(self.mu_ref) > 1e-12 else 1.0
        if solver.model.dist.comm.rank == 0:
            os.makedirs(self.odir, exist_ok=True)

    def run_continuation(self, mu_start, dmu, n_steps=50, mu_target=None):
        """Continue from the model's current state at mu_start for up to n_steps
        branch points (including the 3 initial ones), stopping early once mu
        crosses mu_target."""
        self._setup()
        logger.info(f"Starting {self.continuation_type.capitalize()} Continuation on {self.mu_name}...")
        ds = self._initialize(mu_start, dmu)
        self._continue(ds, n_steps - 3, mu_target)

    def restart_continuation(self, n_steps=50, mu_target=None):
        """Resume the continuation saved in self.odir and compute up to n_steps
        more branch points, stopping early once mu crosses mu_target.

        Uses odir/continuation_state.json (exact resume). Without it (runs made
        before checkpoints existed) the branch is rebuilt from odir/flow_properties.csv
        and the step size is estimated from the last step."""
        self._setup()
        state = output.read_continuation_state(self)
        exact = state is not None
        if not exact:
            logger.info("No continuation_state.json found; rebuilding the branch from flow_properties.csv.")
            state = output.read_branch_csv(self)
        if state['mu_name'] != self.mu_name:
            raise ValueError(f"Saved continuation is in '{state['mu_name']}', not '{self.mu_name}'.")
        missing = [name for name in self._searched() if name not in state]
        if missing:
            raise ValueError(f"Saved branch has no values for searched parameter(s) {missing}.")
        if len(state['mu']) < 3:
            raise RuntimeError(f"Need at least 3 converged points in {self.odir} to restart; "
                               f"found {len(state['mu'])}. Use run_continuation instead.")

        # branch history (states are reloaded only for the last 3 points: all the predictors need)
        self.mu_history = list(state['mu'])
        self.dir_history = list(state['dirs'])
        for name in self._searched():
            self._history(name)[:] = list(state[name])
            setattr(self, name, state[name][-1])
        self.isearch = 1 + max(int(d.strip('/').split('-')[-1]) for d in self.dir_history)
        self.x_history, self.norm_history = [], []
        model = self.ECSSolver.model
        for mu, d in zip(self.mu_history[-3:], self.dir_history[-3:]):
            self.set_parameter(mu)
            model.load_state(os.path.join(self.odir, d, 'solution'))
            x = model.get_state()
            self.x_history.append(x)
            self.norm_history.append(np.linalg.norm(x))

        if exact:
            self.s_history = list(state['s'])
            ds = state['ds']
        else:
            X, M = self.x_history, self.mu_history
            s1 = self._step_norm(X[-2], X[-3], M[-2], M[-3])
            s2 = self._step_norm(X[-1], X[-2], M[-1], M[-2])
            self.s_history = [0.0, s1, s1 + s2]
            ds = s2
        logger.info(f"Restarting {self.continuation_type} continuation on {self.mu_name} from "
                    f"{len(self.mu_history)} points, last {self.mu_name} = {self.mu_history[-1]}, "
                    f"next search-{self.isearch}, ds = {ds}")
        self._continue(ds, n_steps, mu_target)

    def _continue(self, ds, n_new, mu_target=None):
        """Predictor-corrector loop: add up to n_new branch points."""
        step = 0
        while step < n_new:
            M = self.mu_history
            if mu_target is not None and (M[-1] - mu_target)*(M[-2] - mu_target) <= 0:
                logger.info("Reached or crossed target mu.")
                break

            self._log_ds(ds)
            if ds > self.ds_max:
                logger.info(f"ds ({ds:.4f}) exceeded ds_max ({self.ds_max}). Capping to ds_max.")
                ds = self.ds_max
            elif ds < self.ds_min:
                logger.info(f"ds ({ds:.4f}) dropped below ds_min ({self.ds_min}). Raising to ds_min.")
                ds = self.ds_min

            mu_pred, x_pred, ds = self._choose_step(ds)
            sol, success, res, norm, properties = self.step_continuation(x_pred, mu_pred)

            if success:
                self._record(mu_pred, sol, norm, properties)
                ds_actual = self._step_norm(self.x_history[-1], self.x_history[-2],
                                            self.mu_history[-1], self.mu_history[-2])
                self.s_history.append(self.s_history[-1] + ds_actual)
                ds = np.clip(ds*1.05, self.ds_min, self.ds_max)
                logger.info(f"Search {self.isearch-1}: Success | {self.mu_name} = {mu_pred:.6f} | Res = {res:.2e}")
                self.save_checkpoint(ds)
                step += 1
            else:
                logger.info(f"Step {step}: Failed. Reducing step size...")
                if np.isclose(ds, self.ds_min):
                    raise RuntimeError(f"Continuation stuck: Newton failed at minimum step size ds_min={self.ds_min}")
                self._restore_searched()
                ds = max(ds*0.5, self.ds_min)

    # ------------------------------------------------------------------
    # Output (see output.py)
    # ------------------------------------------------------------------
    def save_flow_properties(self, mu, properties, filename="flow_properties.csv"):
        output.write_branch_point(self, mu, properties, filename)

    def save_mu(self, mu):
        output.write_mu(self.ECSSolver, mu)

    def save_checkpoint(self, ds):
        """Write odir/continuation_state.json, used by restart_continuation."""
        output.write_continuation_state(self, ds)
