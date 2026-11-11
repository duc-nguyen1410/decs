"""
Newton-Krylov (hookstep) solver for exact coherent states: equilibria (EQ),
travelling waves (TW), periodic orbits (PO) and relative periodic orbits (RPO).

Unknowns
--------
    xi = [x, (Tp), (ax), (ay), (az)]
x is the flattened flow state (model.get_state()); the period Tp and the shifts
ax, ay, az are appended only when searched (Tsearch, Rxsearch, ...).

Equations
---------
    residual(xi) = x - G(x),   G(x) = sigma S(ax, ay, az) F^Tp(x)
plus one phase condition per searched parameter, orthogonal to the
corresponding neutral direction (d/dt, d/dx, ...).

Module layout
-------------
    newton.py     ECSSolver: unknowns, maps G / DG, Jacobian, Newton loop
    krylov.py     GMRES with hookstep and trust region (pure numpy)
    stability.py  Floquet stability analysis
    output.py     run log, CSV and parameter files
"""
import logging
import numpy as np
from physics.symmetry import Symmetry
from . import krylov
from . import output
from . import stability as stability_mod

logging.getLogger('solvers').setLevel(logging.WARNING)
logging.getLogger('subsystems').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class ECSSolver:
    def __init__(self, model, params=None):
        p = params or {}
        self.model = model
        # searched unknowns (set again by solve)
        self.Tsearch = p.get('Tsearch', 0)
        self.Rxsearch = p.get('Rxsearch', 0)
        self.Rysearch = p.get('Rysearch', 0)
        self.Rzsearch = p.get('Rzsearch', 0)
        self.Tp = p.get('Tp', 0.2)
        self.ax = p.get('ax', 0.0)
        self.ay = p.get('ay', 0.0)
        self.az = p.get('az', 0.0)
        self.sigma = Symmetry()  # symmetry imposed on G (e.g. load_from_file)
        # output
        self.odir = p.get('odir', 'ecs_output/')
        self.model.odir = self.odir
        self.save_ecs_history = p.get('save_ecs_history', False)  # keep solution of every iteration
        self.save_log = p.get('save_log', True)
        self.log_filename = p.get('log_filename', 'output.log')
        # Newton
        self.tol = p.get('tol', 1e-6)
        self.max_iter = p.get('max_iter', 20)
        self.eps_x = p.get('eps_x', 1e-7)  # finite-difference step for DG
        self.eps_t = p.get('eps_t', 1e-6)  # time step for d/dt
        # GMRES + trust region
        self.krylov_dim = p.get('krylov_dim', 50)
        self.krylov_dim_min = p.get('krylov_dim_min', 20)
        self.gmres_min_error = p.get('gmres_min_error', 1e-20)
        self.trust_radius = p.get('trust_radius', 1.0)
        self.trust_radius_min = p.get('trust_radius_min', 1e-4)
        self.trust_radius_max = p.get('trust_radius_max', self.trust_radius)
        # stability
        self.computeStability = p.get('computeStability', False)
        self.projectNeutralDrift = p.get('projectNeutralDrift', False)
        self.Neigen = p.get('Neigen', 50)
        self.stability_method = p.get('stability_method', 'arnoldi')  # 'arnoldi' or 'evp'
        self.evp_target = p.get('evp_target', 0.0)  # sparse EVP: eigenvalues nearest this

    # ------------------------------------------------------------------
    # Unknowns xi = [x, (Tp), (ax), (ay), (az)]
    # ------------------------------------------------------------------
    def _searched(self):
        """Names of the searched parameters, in their order in xi."""
        flags = (('Tp', self.Tsearch), ('ax', self.Rxsearch), ('ay', self.Rysearch), ('az', self.Rzsearch))
        return [name for name, on in flags if on]

    def unpack_xi(self, xi):
        """xi -> (x, Tp, ax, ay, az); parameters not searched take their current value."""
        N = self.model.size()
        values = {'Tp': self.Tp, 'ax': self.ax, 'ay': self.ay, 'az': self.az}
        for i, name in enumerate(self._searched()):
            values[name] = xi[N+i]
        return xi[:N], values['Tp'], values['ax'], values['ay'], values['az']

    def pack_xi(self, x, Tp, ax=0.0, ay=0.0, az=0.0):
        """(x, Tp, ax, ay, az) -> xi, appending only the searched parameters."""
        values = {'Tp': Tp, 'ax': ax, 'ay': ay, 'az': az}
        return np.array(list(x) + [values[name] for name in self._searched()])

    def _sync_params(self, xi):
        """Keep self.Tp, self.ax, ... equal to the values in xi."""
        x, self.Tp, self.ax, self.ay, self.az = self.unpack_xi(xi)
        return x

    # ------------------------------------------------------------------
    # Maps
    # ------------------------------------------------------------------
    def return_map(self, x0, Tp: float, ax: float = 0, ay: float = 0, az: float = 0):
        """sigma S(ax, ay, az) F^Tp(x0)."""
        x = self.model.flow_map(x0, Tp)
        # relative symmetry (shift) for travelling waves / relative periodic orbits
        relative_sigma = Symmetry(ax=ax, ay=ay, az=az)
        if relative_sigma.is_nontrivial():
            x = self.model.apply_symmetry(x, relative_sigma)
        # imposed symmetry constraint
        if self.sigma.is_nontrivial():
            x = self.model.apply_symmetry(x, self.sigma)
        return x

    def return_map_derivative(self, x_base, x_perturb, phi_base, Tp: float, ax: float = 0, ay: float = 0, az: float = 0):
        """Finite-difference Jacobian-vector product (G(x_base + eps v) - G(x_base)) / eps,
        with phi_base = G(x_base)."""
        norm_v = np.linalg.norm(x_perturb)
        if norm_v == 0:
            return np.zeros_like(x_perturb)
        epsilon = self.eps_x/norm_v
        return (self.return_map(x_base + epsilon*x_perturb, Tp, ax, ay, az) - phi_base)/epsilon

    def _evaluate(self, xi):
        """(residual vector, G(x)) at xi; the phase-condition rows of the residual are zero."""
        x0, Tp, ax, ay, az = self.unpack_xi(xi)
        phi = self.return_map(x0, Tp, ax, ay, az)
        res = np.zeros_like(xi)
        res[:self.model.size()] = -phi + x0
        return res, phi

    def residual(self, xi):
        """Newton residual x - G(x) (zero in the phase-condition rows)."""
        return self._evaluate(xi)[0]

    def _derivative(self, name, x):
        """Derivative of the flow / shift of x with respect to parameter `name`: d/dt for Tp,
        L d/dx for ax (the shift is x -> x + ax*Lx, so d/dax = Lx d/dx), likewise ay, az."""
        if name == 'Tp':
            return self.model.t_derivative(x, self.eps_t)
        bounds = self.model.bounds
        derivative, L = {'ax': (self.model.x_derivative, bounds[0]),
                         'ay': (self.model.y_derivative, bounds[1]),
                         'az': (self.model.z_derivative, bounds[-1])}[name]
        return L*derivative(x)

    def _jacobian(self, xi, phi_base):
        """Return matvec(v): the Newton Jacobian at xi acting on v = [dx, dTp, dax, ...]:
            rows 0..N-1 : DG dx - dx + sum_p dG/dp * dp
            row  N+i    : <d_p x, dx>   (phase condition of the i-th searched parameter)
        The parameter derivatives are computed once here and reused for every Krylov vector."""
        N = self.model.size()
        x0, Tp, ax, ay, az = self.unpack_xi(xi)
        tangents = [(self._derivative(name, phi_base), np.conj(self._derivative(name, x0)))
                    for name in self._searched()]

        def matvec(v):
            dx = v[:N]
            out = np.zeros_like(v)
            out[:N] = self.return_map_derivative(x0, dx, phi_base, Tp, ax, ay, az) - dx
            for i, (d_phi, d_x0_conj) in enumerate(tangents):
                out[:N] += d_phi*v[N+i]
                out[N+i] = np.matmul(d_x0_conj, dx)
            return out
        return matvec

    def jacobian_vector(self, xi, xi_perturb, phi_base):
        """Newton Jacobian at xi applied to xi_perturb (see _jacobian)."""
        return self._jacobian(xi, phi_base)(xi_perturb)

    # ------------------------------------------------------------------
    # Newton step
    # ------------------------------------------------------------------
    def newton_step(self, xi, residual, phi_base):
        """Hookstep-GMRES Newton step from xi.
        Returns (dxi, linear-model error, next trust radius, (residual, G) at xi + dxi)."""
        logger.info("Starting GMRES ...")
        matvec = self._jacobian(xi, phi_base)
        Q, H, k, beta, w, res_sq = krylov.gmres_hookstep(
            matvec, residual, self.krylov_dim, self.trust_radius,
            kmin=self.krylov_dim_min, min_error=self.gmres_min_error)

        def evaluate(dx):
            res_trial, phi_trial = self._evaluate(xi + dx)
            return np.linalg.norm(res_trial), (res_trial, phi_trial)

        return krylov.trust_region_step(
            Q, H, k, beta, w, res_sq, self.trust_radius, np.linalg.norm(residual), evaluate,
            self.trust_radius_min, self.trust_radius_max)

    def solve(self, x0, Tsearch=False, Rxsearch=False, Rysearch=False, Rzsearch=False,
                     Tp=20.0, ax=0.0, ay=0.0, az=0.0):
        """Solve from initial guess x0 (and Tp, shifts).
        Returns (xi, success, residual norm, ||x||, properties)."""
        self.Tsearch, self.Rxsearch = int(bool(Tsearch)), int(bool(Rxsearch))
        self.Rysearch, self.Rzsearch = int(bool(Rysearch)), int(bool(Rzsearch))
        self.Tp = Tp
        self.ax = ax if Rxsearch else 0.0
        self.ay = ay if Rysearch else 0.0
        self.az = az if Rzsearch else 0.0
        N = self.model.size()
        if np.size(x0) != N:
            raise ValueError(f"Initial guess x0 has size {np.size(x0)}, but expected size is {N}.")

        log = output.RunLog(self)
        log.write_header(self)
        output.reset_flow_properties(self)  # one CSV per run, like output.log
        logger.info("Starting Newton solver ...")
        log.write('Starting Newton solver ...')

        xi = self.pack_xi(x0, self.Tp, self.ax, self.ay, self.az)
        residual, phi = self._evaluate(xi)
        norm_b = np.linalg.norm(residual)
        success = False

        for i in range(self.max_iter):
            x = self._sync_params(xi)
            # save the current iterate (every iteration if save_ecs_history, else overwrite)
            self.model.set_state(x)
            self.model.save_state(self.odir + (f'solution_iter_{i}' if self.save_ecs_history else 'solution'))
            output.write_flow_properties(self, xi)
            pstr = output.param_string(self, self.Tp, self.ax, self.ay, self.az)

            if i == 0:
                logger.info(f"Initialization, Residual norm: {norm_b:.5e}{pstr}")
                log.write(f"Initialization, Residual norm: {norm_b:.5e}{pstr}")
            if norm_b < self.tol:
                logger.info("Convergence achieved!")
                log.write("Convergence achieved!")
                success = True
                self._on_converged(xi)
                break

            dxi, error, tr, (residual, phi) = self.newton_step(xi, residual, phi)
            self.trust_radius = tr
            xi += dxi
            x = self._sync_params(xi)
            norm_b = np.linalg.norm(residual)

            pstr = output.param_string(self, self.Tp, self.ax, self.ay, self.az)
            logger.info(f"Iteration {i}, ||x||: {np.linalg.norm(x)}, Residual: {norm_b}, "
                        f"GMRES error: {error}, trust radius: {tr}{pstr}")
            log.write(f"Iteration {i}, Residual norm: {norm_b:.5e}, ||x||: {np.linalg.norm(x)}, "
                      f"GMRES error: {error}, trust radius: {tr}{pstr}")
            output.write_iteration_files(self, self.Tp, self.ax, self.ay, self.az, norm_b)
        log.close()

        x = self._sync_params(xi)
        self.model.set_state(x)
        flow_properties = self.model.get_flow_properties() or {}
        properties = {**output.search_values(self, self.Tp, self.ax, self.ay, self.az), **flow_properties}
        return xi, success, norm_b, np.linalg.norm(x), properties

    def _on_converged(self, xi):
        """Save the converged solution, its time history (orbits / waves) and stability."""
        x, Tp, ax, ay, az = self.unpack_xi(xi)
        self.model.set_state(x)
        self.model.save_state(self.odir + 'solution')
        if self._searched():
            self.model.save_time_dependent_solution(x, Tp, ax, ay, az)
        if self.computeStability:
            self.stability(xi)

    # ------------------------------------------------------------------
    # Analysis / output (implemented in stability.py and output.py)
    # ------------------------------------------------------------------
    def stability(self, xi):
        """Floquet stability of a converged solution xi (see stability.py)."""
        stability_mod.stability(self, xi)

    def save_flow_properties(self, xi, filename="flow_properties.csv"):
        output.write_flow_properties(self, xi, filename)
