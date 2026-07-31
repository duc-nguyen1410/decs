import numpy as np
import scipy.io
import os
import h5py
from scipy import sparse
from scipy import optimize
import logging
import dedalus.public as de
from mpi4py import MPI
from physics.symmetry import Symmetry
logging.getLogger('solvers').setLevel(logging.WARNING)
logging.getLogger('subsystems').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class ECSSolver:
    def __init__(self, model, params=None):
        self.model = model
        #
        self.Tsearch = (params or {}).get('Tsearch', False)
        self.Rxsearch = (params or {}).get('Rxsearch', False)
        self.Rysearch = (params or {}).get('Rysearch', False)
        self.Rzsearch = (params or {}).get('Rzsearch', False)
        self.sigma = Symmetry()
        # Set default parameters for the solver
        self.odir = (params or {}).get('odir', 'ecs_output/')
        self.model.odir = self.odir
        self.tol = (params or {}).get('tol', 1e-6)
        self.max_iter = (params or {}).get('max_iter', 20)
        self.Tp = (params or {}).get('Tp', 0.2)
        #
        self.d_tol = (params or {}).get('d_tol', 1e-7)
        self.gmres_min_error = (params or {}).get('gmres_min_error', 1e-10)
        self.trust_radius_min = (params or {}).get('trust_radius_min', 1e-4)
        self.trust_radius = (params or {}).get('trust_radius', 1.0)
        #
        self.krylov_dim = (params or {}).get('krylov_dim', 50)
        self.krylov_dim_min = (params or {}).get('krylov_dim_min', 20)
        # stability
        self.projectNeutralDrift = (params or {}).get('projectNeutralDrift', False)
        self.computeStability = (params or {}).get('computeStability', False)
        self.Neigen = (params or {}).get('Neigen', 50)
        # save updates of solution during the Newton iteration
        self.save_ecs_history = (params or {}).get('save_ecs_history', False)
        #
        self.save_log = (params or {}).get('save_log', True)
        self.log_filename = (params or {}).get('log_filename', 'output.log')

    def G(self, x0, Tp:float, ax:float=0, ay:float=0, az:float=0):
        ''' Return sigma*F^Tp(x0) '''
        x = self.model.F_Tp(x0, Tp)
        # apply a relative symmetry to determine traveling wave or relative periodic orbit
        relative_sigma = Symmetry(ax=ax,ay=ay,az=az)
        if relative_sigma.is_nontrivial():
            x = self.model.apply_symmetry(x, relative_sigma)
        # apply a specific symmetry as a constraint
        if self.sigma.is_nontrivial():
            x = self.model.apply_symmetry(x, self.sigma)
        return x
    def DG(self, x_base, x_perturb, phi_base, Tp:float, ax:float=0, ay:float=0, az:float=0):
        ''' Return (F^Tp(x0+dx) - F^Tp(x0)) / ||dx|| '''
        norm_v = np.linalg.norm(x_perturb)
        if norm_v == 0:
            return np.zeros_like(x_perturb)
        epsilon = self.d_tol / norm_v
        # logger.info(f"Computing DG with epsilon: {epsilon}, ||x_perturb||: {norm_v}")
        array_init = x_base + epsilon*x_perturb
        # logger.info(f"Initial state ||x_base + epsilon*x_perturb||: {np.linalg.norm(array_init)}, ||x_base||: {np.linalg.norm(x_base)}")
        array_final = self.G(array_init, Tp, ax, ay, az)
        # logger.info(f"G computed, ||G(x_base + epsilon*x_perturb)||: {np.linalg.norm(array_final)}, ||G(x_base)||: {np.linalg.norm(phi_base)}")
        array_out = (array_final-phi_base)/epsilon
        # logger.info(f"DG output computed, ||DG||: {np.linalg.norm(array_out)}")
        return array_out
    def LinearOperator(self, xi, xi_perturb, phi_base):
        ''' Linearized operator for the Newton iteration 
            (F^T(x0+dx) - F^T(x0)) / ||dx|| - dx
        '''
        N_ = self.model.size()
        
        T_temp, ax_temp, ay_temp, az_temp = self.Tp, 0.0, 0.0, 0.0
        if self.Tsearch:
            T_temp = xi[N_+self.Tsearch-1]
            delta_T = xi_perturb[N_+self.Tsearch-1]
        if self.Rxsearch:
            ax_temp = xi[N_+self.Tsearch+self.Rxsearch-1]
            delta_ax = xi_perturb[N_+self.Tsearch+self.Rxsearch-1]
        if self.Rysearch:
            ay_temp = xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1]
            delta_ay = xi_perturb[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1]
        if self.Rzsearch:
            az_temp = xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1]
            delta_az = xi_perturb[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1]
        
        x_base = np.copy(xi[:N_])
        delta_x = np.copy(xi_perturb[:N_])

        array_out = np.zeros_like(xi)
        array_out[:N_] = self.DG(x_base, delta_x, phi_base, T_temp, ax_temp, ay_temp, az_temp) - delta_x
        # logger.info(f"Linear operator applied, ||DG||: {np.linalg.norm(array_out[:N_])}, ||delta_x||: {np.linalg.norm(delta_x)}")
        
        if self.Tsearch: # Sensitivity to T + Phase Condition for T
            array_out[:N_] += self.model.t_derivative(phi_base, self.d_tol) * delta_T
            array_out[N_+self.Tsearch-1] = np.matmul(np.conj(self.model.t_derivative(x_base, self.d_tol)), delta_x)
        if self.Rxsearch: # Sensitivity to shift + Phase Condition (fixing the wave in x)
            array_out[:N_] += self.model.x_derivative(phi_base) * delta_ax
            array_out[N_+self.Tsearch+self.Rxsearch-1] = np.matmul(np.conj(self.model.x_derivative(x_base)), delta_x)
        if self.Rysearch: # Sensitivity to shift + Phase Condition (fixing the wave in y)
            array_out[:N_] += self.model.y_derivative(phi_base) * delta_ay
            array_out[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1] = np.matmul(np.conj(self.model.y_derivative(x_base)), delta_x)
        if self.Rzsearch: # Sensitivity to shift + Phase Condition (fixing the wave in z)
            array_out[:N_] += self.model.z_derivative(phi_base) * delta_az
            array_out[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1] = np.matmul(np.conj(self.model.z_derivative(x_base)), delta_x)
        return array_out
    def NonlinearOperator(self, xi):
        ''' Return sigma*F(x) - x '''
        N_ = self.model.size()
        xi_out = np.zeros_like(xi)
        state = xi[:N_]
        T_temp = xi[N_+self.Tsearch-1] if self.Tsearch else self.Tp
        ax_temp = xi[N_+self.Tsearch+self.Rxsearch-1] if self.Rxsearch else 0.0
        ay_temp = xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1] if self.Rysearch else 0.0
        az_temp = xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1] if self.Rzsearch else 0.0
        F_state = self.G(state, T_temp, ax_temp, ay_temp, az_temp)
        xi_out[:N_] = - F_state + state
        return xi_out  # Residual for the state variables
    def arnoldi_iteration_inner(self, xi_base, Q, phi_base, k:int):
        Qk = self.LinearOperator(xi_base, Q[:, k - 1], phi_base)
        # logger.info(f"Arnoldi iteration {k}, ||Qk|| before orthogonalization: {np.linalg.norm(Qk)}")
        Hk = np.zeros(k+1)
        for j in range(0, k):
            Hk[j] = np.matmul(np.conj(Q[:,j]), Qk)
            Qk = Qk- Hk[j]*Q[:,j]
        Hk[k] = np.linalg.norm(Qk)
        Qk = Qk/Hk[k]
        return Qk, Hk
    def arnoldi_iteration(self, x_base, phi_base, T:float, ax:float, ay:float, az:float, r, n:int):
        ''' Arnoldi iteration '''
        neutral_basis = []
        if self.projectNeutralDrift and np.linalg.norm(x_base) >= 1e-6:
            # Spatial x-derivative
            dudx = self.model.x_derivative(x_base)
            norm_x = np.linalg.norm(dudx)
            if norm_x > 1e-12:
                neutral_basis.append(dudx / norm_x)

            # Spatial z-derivative
            if not self.model.bounded:
                dudz = self.model.z_derivative(x_base)
                norm_z = np.linalg.norm(dudz)
                if norm_z > 1e-12:
                    neutral_basis.append(dudz / norm_z)

            # Time derivative
            if self.Tsearch:
                dudt = self.model.t_derivative(x_base, self.d_tol)
                norm_t = np.linalg.norm(dudt)
                if norm_t > 1e-12:
                    neutral_basis.append(dudt / norm_t)

            # Orthonormalize neutral directions against each other (Gram-Schmidt)
            ortho_neutral = []
            for u in neutral_basis:
                for u_prev in ortho_neutral:
                    u -= np.vdot(u_prev, u) * u_prev
                norm_u = np.linalg.norm(u)
                if norm_u > 1e-12:
                    ortho_neutral.append(u / norm_u)
            neutral_basis = ortho_neutral

        # Ensure starting vector is orthogonal to neutral direction
        def project_out(v):
            projected_v = v.copy()

            # SYMMETRY PROJECTION ONTO V+
            # if self.sigma.is_nontrivial():
            #     Sv = self.model.apply_symmetry(projected_v, self.sigma)
                # V-: anti-symmetric subspace
                # Computes 0.5 * (I + S) v : project all V- subspaces modes to zero
                # projected_v = 0.5 * (projected_v + Sv)  
                # V+: symmetric subspace
                # Computes 0.5 * (I - S) v : project all V+ subspaces modes to zero
                # projected_v = 0.5 * (projected_v - Sv)  

            # dudx_ref = self.model.x_derivative(x_base) # x-derivative
            # dudx_ref = dudx_ref / np.linalg.norm(dudx_ref) # normalization
            # gg1 = np.vdot(dudx_ref, dudx_ref)
            # g1v = np.vdot(dudx_ref, v)
            # projected_v -= dudx_ref * (g1v / gg1)

            # if not self.model.bounded:
            #     dudz_ref = self.model.z_derivative(x_base) # z-derivative
            #     dudz_ref = dudz_ref / np.linalg.norm(dudz_ref) # normalization
            #     gg2 = np.vdot(dudz_ref, dudz_ref)
            #     g2v = np.vdot(dudz_ref, v)
            #     projected_v -= dudz_ref * (g2v / gg2)
            # if self.Tsearch:
            #     dudt_ref = self.model.t_derivative(x_base, self.d_tol) # time-derivative
            #     dudt_ref = dudt_ref / np.linalg.norm(dudt_ref) # normalization
            #     gg = np.vdot(dudt_ref, dudt_ref) # <dudt_ref, dudt_ref>
            #     gv = np.vdot(dudt_ref, v) # <dudt_ref, v>
            #     projected_v -= dudt_ref * (gv / gg)

            # Project out neutral drift directions
            if self.projectNeutralDrift:
                for u in neutral_basis:
                    projected_v -= u * np.vdot(u, projected_v)
            return projected_v
        
        
                    
        # Arnoldi Initialization
        r = project_out(r)
        Q = np.zeros((r.size, n+1))
        H = np.zeros((n+1, n))
        Q[:,0] = r/np.linalg.norm(r)
        # Arnoldi Loop
        for k in range(1, n + 1):
            q_in = Q[:, k-1]
            if self.projectNeutralDrift:
                q_in = project_out(q_in) # Project input before applying L
                v = self.DG(x_base, q_in, phi_base, T, ax, ay, az) # Apply operator
                v = project_out(v) # Project output
            else:
                v = self.DG(x_base, q_in, phi_base, T, ax, ay, az)
            # Modified Gram-Schmidt (MGS)
            for j in range(0, k):
                H[j, k-1] = np.vdot(Q[:,j], v)
                v = v - H[j, k-1]*Q[:,j]
            H[k, k-1] = np.linalg.norm(v)
            if H[k, k - 1] < 1e-12:
                logger.info(f"Arnoldi Krylov subspace closed at step {k}")
                break
            Q[:,k] = v/H[k, k-1]
        return Q, H
    def Hookstep(self, H_, beta_, k_, tr):
        e1 = np.zeros(k_+1)
        e1[0] = beta_
        def fun(x_, F):
            r = np.matmul(F, x_) + e1
            return np.matmul(r, r)
        def Jacobian(x_, F):
            return 2*np.matmul(np.matmul(np.transpose(F), F), x_) + 2*np.matmul(np.transpose(F), e1)
        def constraint(x_):
            return tr*tr - np.matmul(np.transpose(x_), x_)
        def constraintJac(x_):
            return -2*x_
        ineq_cons = {'type': 'ineq','fun' : constraint,'jac' : constraintJac}
        w_init = np.zeros(k_)
        w_init[0] = 1e-3
        res = scipy.optimize.minimize(fun, w_init, args=(H_[0:k_+1,0:k_]), method='SLSQP', jac = Jacobian,
            constraints=(ineq_cons), options={'ftol': 1e-34, 'disp': False, 'maxiter': 100000000}, bounds=None)
        return res
    def GMRES(self, xi_base, xi_pert, phi_base, b, kmax, tr):
        xk = np.copy(xi_pert)
        logger.info("Starting GMRES ...")
        # logger.info(f"Initial perturbation norm: {np.linalg.norm(x_pert)}, Initial residual norm: {np.linalg.norm(b)}")
        r = self.LinearOperator(xi_base, xi_pert, phi_base) - b
        rho = np.linalg.norm(r)
        beta = rho
        b_norm = np.linalg.norm(b)
        # logger.info(f"Initial GMRES residual norm: {rho}, ||b||: {b_norm}")
        Q = np.zeros((xi_pert.size, kmax+1))
        H = np.zeros((kmax+1, kmax))
        
        min_error = np.inf
        min_vector = np.zeros(xi_pert.size)
        
        Q[:,0] = r/np.linalg.norm(r)
        best_k = 1
        for k in range(1, kmax):
            Q[:,k], H[:k+1,k-1] = self.arnoldi_iteration_inner(xi_base, Q[:,0:k], phi_base, k)
            # logger.info(f"Q[:,{k}] norm: {np.linalg.norm(Q[:,k])}, H[:{k+1},{k-1}] norm: {np.linalg.norm(H[:k+1,k-1])}")
            res = self.Hookstep(H, beta, k, tr)
            rho = np.linalg.norm(res.fun)
            
            # if MPI.COMM_WORLD.rank == 0:
            #     print(".", end='', flush=True)
            min_error = rho
            xk = np.matmul(Q[:,0:k], res.x)
            logger.info(f"GMRES iteration {k}, residual norm: {rho}")
            best_k = k
            if self.krylov_dim_min <= k and rho < self.gmres_min_error:
                break
        test = np.linalg.norm(self.NonlinearOperator(np.copy(xi_base)+xi_pert+xk))
        # logger.info(f"Initial optimal residual norm: {test}, min GMRES residual norm: {min_error}")
        tr_local = tr
        while test > 0.99*b_norm and tr_local > self.trust_radius_min:
            res = self.Hookstep(H, beta, best_k, tr_local)
            xk = np.matmul(Q[:,0:(best_k)], res.x)
            min_error = np.linalg.norm(res.fun)
            test = np.linalg.norm(self.NonlinearOperator(np.copy(xi_base)+xi_pert+xk))
            tr_local = 0.5*tr_local
            # logger.info(f"Hookstep-based optimal residual norm: {test}, min GMRES residual norm: {min_error}, trust radius: {tr_local}")
        return xi_pert + xk, min_error, tr_local
    def save_flow_properties(self, xi, filename="flow_properties.csv"):
        properties = self.model.get_flow_properties()
        N_ = self.model.size()
        # debug_x = self.model.get_state()
        # logger.info(f"||x||: {np.linalg.norm(debug_x)}")
        if self.model.dist.comm.rank == 0:
            file_path = os.path.join(self.odir, filename)
            file_exists = os.path.isfile(file_path)
            keys = list(properties.keys())
            if not file_exists:
                with open(file_path, mode='w') as header:
                    # Write header if file is new
                    header_line = ""
                    if self.Tsearch:
                        header_line = header_line + f"Tp, "
                    if self.Rxsearch:
                        header_line = header_line + f"ax, "
                    if self.Rysearch:
                        header_line = header_line + f"ay, "
                    if self.Rzsearch:
                        header_line = header_line + f"az, "
                    header_line = header_line + ", ".join(keys)
                    header.write(header_line + "\n")
            with open(file_path, mode='a') as f:
                # Append data
                values = []
                if self.Tsearch:
                    values = values + [f"{xi[N_+self.Tsearch-1]:.12f}"]
                if self.Rxsearch:
                    values = values + [f"{xi[N_+self.Tsearch+self.Rxsearch-1]:.12f}"]
                if self.Rysearch:
                    values = values + [f"{xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1]:.12f}"]
                if self.Rzsearch:
                    values = values + [f"{xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1]:.12f}"]
                values = values + [f"{float(properties[k]):.12f}" for k in keys]
                f.write(", ".join(values) + "\n")
            
            # Also print to terminal for real-time monitoring
            prop_strings = [f"{k}={float(properties[k]):.6f}" for k in keys]
            logger.info(f"Prop Check: {' | '.join(prop_strings)}")
    def stability(self,x):
        nonlinear_res = self.NonlinearOperator(x)
        norm_b = np.linalg.norm(nonlinear_res)
        if norm_b < 1e-6:
            logger.info('Solving linear stability problem around converged solution ...')
            if self.model.dist.comm.rank == 0:
                if not os.path.exists(self.odir+'stability/'):
                    os.mkdir(self.odir+'stability/')
            if self.sigma.is_nontrivial():
                self.sigma = Symmetry() # do not apply symmetry in linear stability analysis
            N_ = self.model.size()
            T_temp = x[N_+self.Tsearch-1] if self.Tsearch else self.Tp
            ax_temp = x[N_+self.Tsearch+self.Rxsearch-1] if self.Rxsearch else 0.0
            ay_temp = x[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1] if self.Rysearch else 0.0
            az_temp = x[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1] if self.Rzsearch else 0.0
            phi_base = self.G(x[:N_], T_temp, ax_temp, ay_temp, az_temp)
            # Floquet method
            Q, H_ = self.arnoldi_iteration(x[:N_], phi_base, T_temp, ax_temp, ay_temp, az_temp, np.random.rand(N_), self.Neigen) # <-- Ne iterations
            H = H_[0:-1,:]
            # get eigenvalue and eigenvector results, these are Floquet multipliers
            eigenvalues, eigenvectors_ = scipy.linalg.eig(H) 
            eigenvalues_abs = np.abs(eigenvalues)
            eigenvectors = np.matmul(Q[:,0:-1], eigenvectors_)
            growthrate = np.log(eigenvalues) / T_temp # convert to growth rate

            # Last row of H_ (h_{m+1,m})
            h_last = H_[-1, :]  # 1 x m
            # Residual for each Ritz pair
            res = np.zeros(eigenvalues.size)
            for i in range(eigenvalues.size):
                y = eigenvectors_[:, i]           # THIS is the eigenvector of H
                res[i] = abs(h_last @ y)

            # Sort modes by descending growth rate
            idx = np.argsort(growthrate.real)[::-1]
            growthrate = growthrate[idx]
            eigenvalues  = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            res = res[idx]

            if MPI.COMM_WORLD.rank == 0:
                # save Floquet multiplier
                scipy.io.mmwrite(self.odir+'stability/eigenvalues.mtx', eigenvalues.reshape(1, -1)) # Eigenvalues; .mtx = Matrix Market format
                # save growth rate
                scipy.io.mmwrite(self.odir+'stability/growthrate.mtx', growthrate.reshape(1, -1))
                # save residual
                scipy.io.mmwrite(self.odir+'stability/residual.mtx', [res])
            
            coords = [basis.global_grid(self.model.dist, scale=self.model.dealias) for basis in self.model.bases]
            coord_names = [basis.coord.name for basis in self.model.bases]
            unstable = np.where(growthrate.real > 0)[0]
            if MPI.COMM_WORLD.rank == 0 and unstable.size > 0:
                eigenvectors_unstable = eigenvectors[:, unstable]
                eigenvalues_unstable = eigenvalues[unstable]
                growthrate_unstable = growthrate[unstable]
                h5f = h5py.File(self.odir+'stability/eigen_unstable.h5', 'w') 
                h5f.create_dataset('/eigenvectors', data = eigenvectors_unstable) 
                h5f.create_dataset('/eigenvalues', data = eigenvalues_unstable) 
                h5f.create_dataset('/growthrate', data = growthrate_unstable) 
                for c_name, g_data in zip(coord_names, coords):
                    h5f.create_dataset(f'/{c_name}', data = g_data) 
                h5f.close()
            logger.info('Done!')
    def NewtonSolver(self, 
                    x0, 
                    Tsearch=False,
                    Rxsearch=False,
                    Rysearch=False,
                    Rzsearch=False,
                    Tp = 20.0, 
                    ax = 0.0, 
                    ay = 0.0,
                    az = 0.0):
        self.Tsearch = Tsearch
        self.Rxsearch = Rxsearch
        self.Rysearch = Rysearch
        self.Rzsearch = Rzsearch
        self.Tp = Tp
        N_ = self.model.size()
        if self.model.dist.comm.rank == 0:
            if not os.path.exists(self.odir):
                os.mkdir(self.odir)
            if os.path.exists(self.odir+self.log_filename):
                os.remove(self.odir+self.log_filename)
            log_file = open(self.odir+self.log_filename, 'w', buffering=1)
            log_file.writelines('---------------------------------\n')
            log_file.writelines('--- Parameters ------------------\n')
            log_file.writelines('---------------------------------\n')
            for key, value in self.model.params.items():
                log_file.writelines(f"{key}: {value}\n")
            log_file.writelines(f"dim: {self.model.dim}\n")
            log_file.writelines(f"sizes: {self.model.sizes}\n")
            log_file.writelines(f"bounds: {self.model.bounds}\n")
            log_file.writelines(f"bounded: {self.model.bounded}\n")
            log_file.writelines(f"dealias: {self.model.dealias}\n")
            log_file.writelines(f"mode: {self.model.mode}\n")
            log_file.writelines(f"CFL: {self.model.use_CFL}\n")
            log_file.writelines(f"initial dt: {self.model.init_dt}\n")
            if self.sigma.is_nontrivial():
                log_file.writelines(f"symmetry: {self.sigma.print()}\n")
            log_file.writelines(f"ecs_odir: {self.odir}\n")
            log_file.writelines(f"tol: {self.tol}\n")
            log_file.writelines(f"max_iter: {self.max_iter}\n")
            log_file.writelines(f"initial Tp: {self.Tp}\n")
            log_file.writelines(f"trust radius: {self.trust_radius}\n")
            log_file.writelines(f"krylov_dim: {self.krylov_dim}\n")
            log_file.writelines(f"krylov_dim_min: {self.krylov_dim_min}\n")
            log_file.writelines(f"projectNeutralDrift: {self.projectNeutralDrift}\n")
            log_file.writelines(f"computeStability: {self.computeStability}\n")
            log_file.writelines(f"Neigen: {self.Neigen}\n")
            log_file.writelines(f"save_ecs_history: {self.save_ecs_history}\n")
            log_file.writelines(f"save_snapshots: {self.model.save_snapshots}\n")
            log_file.writelines(f"save_flowproperties: {self.model.save_flowproperties}\n")
            log_file.writelines(f"save_meanprofiles: {self.model.save_meanprofiles}\n")
            log_file.writelines('---------------------------------\n')
            logger.info("Starting Newton solver ...")
            log_file.writelines('Starting Newton solver ...\n')
        # Initial guess includes state and parameters
        xi = np.concatenate([x0, 
                            [Tp] if self.Tsearch else [], 
                            [ax] if self.Rxsearch else [], 
                            [ay] if self.Rysearch else [],
                            [az] if self.Rzsearch else []]) 
        
        xi_pert = np.zeros_like(xi)
        success = False
        for i in range(self.max_iter):
            # logger.info("\n")
            self.model.set_state(xi[:N_])
            if self.save_ecs_history:
                # save the solution at each iteration for post-analysis of the convergence process if needed
                self.model.save_state(self.odir + f'solution_iter_{i}')
            else:
                # save a temporary solution at each iteration for debugging purposes or restarting if needed
                self.model.save_state(self.odir + 'solution_temp')
            
            nonlinear_res = self.NonlinearOperator(xi)
            norm_b = np.linalg.norm(nonlinear_res)
            self.save_flow_properties(xi)
            if i==0:
                logger.info(f"Initialization, Residual norm: {norm_b}" \
                            +f"{f", Tp: {xi[N_+self.Tsearch-1]}" if self.Tsearch else ""}" \
                            +f"{f", ax: {xi[N_+self.Tsearch+self.Rxsearch-1]}" if self.Rxsearch else ""}" \
                            +f"{f", ay: {xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1]}" if self.Rysearch else ""}" \
                            +f"{f", az: {xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1]}" if self.Rzsearch else ""}")
                if self.model.dist.comm.rank == 0:
                    log_file.writelines(f"Initialization, Residual norm: {norm_b}" \
                                        +f"{f", Tp: {xi[N_+self.Tsearch-1]}" if self.Tsearch else ""}" \
                                        +f"{f", ax: {xi[N_+self.Tsearch+self.Rxsearch-1]}" if self.Rxsearch else ""}" \
                                        +f"{f", ay: {xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1]}" if self.Rysearch else ""}" \
                                        +f"{f", az: {xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1]}" if self.Rzsearch else ""}\n")
            if norm_b < self.tol:
                logger.info("Convergence achieved!")
                if self.model.dist.comm.rank == 0:
                    log_file.writelines("Convergence achieved!\n")
                success = True
                # save the solution
                self.model.set_state(xi[:N_])
                self.model.save_state(self.odir + 'solution')
                # save best Tp

                # save best symmetry

                # save time-dependent data
                if self.Tsearch or self.Rxsearch or self.Rysearch or self.Rzsearch:
                    self.model.save_time_dependent_solution(xi[:N_],
                                                            xi[N_+self.Tsearch-1] if self.Tsearch else Tp,
                                                            xi[N_+self.Tsearch+self.Rxsearch-1] if self.Rxsearch else ax,
                                                            xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1] if self.Rysearch else ay,
                                                            xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1] if self.Rzsearch else az)
                # compute stability
                if self.computeStability:
                    self.stability(xi)
                break
            
            phi_base = self.G(xi[:N_], 
                              xi[N_+self.Tsearch-1] if self.Tsearch else self.Tp, 
                              xi[N_+self.Tsearch+self.Rxsearch-1] if self.Rxsearch else ax,
                              xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1] if self.Rysearch else ay,
                              xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1] if self.Rzsearch else az)
            dxi, error, tr = self.GMRES(xi, xi_pert, phi_base, nonlinear_res, self.krylov_dim, self.trust_radius)
            xi += dxi # Update the solution
            nonlinear_res = self.NonlinearOperator(xi)
            norm_b = np.linalg.norm(nonlinear_res)
            logger.info(f"Iteration {i}, ||x||: {np.linalg.norm(xi[:N_])}, Residual: {norm_b}, GMRES error: {error}, trust radius: {tr}")
            if self.model.dist.comm.rank == 0:
                log_file.writelines(f"Iteration {i}, Residual norm: {norm_b}, ||x||: {np.linalg.norm(xi[:N_])}, GMRES error: {error}, trust radius: {tr}" \
                                    +f"{f", Tp: {xi[N_+self.Tsearch-1]}" if self.Tsearch else ""}" \
                                    +f"{f", ax: {xi[N_+self.Tsearch+self.Rxsearch-1]}" if self.Rxsearch else ""}" \
                                    +f"{f", ay: {xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1]}" if self.Rysearch else ""}" \
                                    +f"{f", az: {xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1]}" if self.Rzsearch else ""}\n")

            # save T
            if self.Tsearch:
                T_file = open(self.odir+'T.txt', 'w')
                T_file.writelines(str(xi[N_+self.Tsearch-1]))
                T_file.close()

            # save shift speeds: ax, ay , az
            if self.Rxsearch:
                ax_file = open(self.odir+'ax.txt', 'w')
                ax_file.writelines(str(xi[N_+self.Tsearch+self.Rxsearch-1]))
                ax_file.close()
            if self.Rysearch:
                ay_file = open(self.odir+'ay.txt', 'w')
                ay_file.writelines(str(xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch-1]))
                ay_file.close()
            if self.Rzsearch:
                az_file = open(self.odir+'az.txt', 'w')
                az_file.writelines(str(xi[N_+self.Tsearch+self.Rxsearch+self.Rysearch+self.Rzsearch-1]))
                az_file.close()

            # save tolerance
            tol_file = open(self.odir+'tol.txt', 'w')
            tol_file.writelines(str(norm_b))
            tol_file.close()

        return xi, success, norm_b, np.linalg.norm(xi[:N_]), self.model.get_flow_properties()
    

    