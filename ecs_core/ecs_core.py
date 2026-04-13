import numpy as np
import scipy.io
import os
import h5py
from scipy import sparse
from scipy import optimize
import logging
import dedalus.public as de
from mpi4py import MPI
logging.getLogger('solvers').setLevel(logging.WARNING)
logging.getLogger('subsystems').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class NewtonSolver:
    def __init__(self, model, params=None):
        self.model = model
        self.IVP_problem = self.model.get_IVP()  
        #
        self.Tsearch = params['Tsearch'] if params and 'Tsearch' in params else False
        self.Rxsearch = params['Rxsearch'] if params and 'Rxsearch' in params else False
        self.Rzsearch = params['Rzsearch'] if params and 'Rzsearch' in params else False
        # Set default parameters for the solver
        self.odir = params['odir'] if params and 'odir' in params else './ecs_output/'
        self.tol = params['tol'] if params and 'tol' in params else 1e-8
        self.max_iter = params['max_iter'] if params and 'max_iter' in params else 20
        self.Tp = params['Tp'] if params and 'Tp' in params else 0.02
        self.d_tol = params['d_tol'] if params and 'd_tol' in params else 1e-7
        self.n_timesteps = 10
        self.gmres_min_error = params['gmres_min_error'] if params and 'gmres_min_error' in params else 1e-6
        self.trust_radius_min = params['trust_radius_min'] if params and 'trust_radius_min' in params else 1e-4
        self.trust_radius = params['trust_radius'] if params and 'trust_radius' in params else 1.0
        self.krylov_dim = params['krylov_dim'] if params and 'krylov_dim' in params else 50
        self.krylov_dim_min = params['krylov_dim_min'] if params and 'krylov_dim_min' in params else 20
        self.projectNeutralDrift = True
        self.Ne = 50
    def G(self, x0, Tp, ax, az):
        ''' Return sigma(ax, az)*F^Tp(x0) '''
        x = self.model.F_Tp(self.IVP_problem, x0, Tp)
        # symmetry operations on x using ax and az if needed
        # x = self.model.apply_symmetry(x, ax, az)
        return x
    def DG(self, x_base, x_perturb, phi_base, Tp, ax, az):
        ''' Return (F^Tp(x0+dx) - F^Tp(x0)) / ||dx|| '''
        norm_v = np.linalg.norm(x_perturb)
        if norm_v == 0:
            return np.zeros_like(x_perturb)
        epsilon = self.d_tol / norm_v
        # logger.info(f"Computing DG with epsilon: {epsilon}, ||x_perturb||: {norm_v}")

        array_init = x_base + epsilon*x_perturb
        # logger.info(f"Initial state ||x_base + epsilon*x_perturb||: {np.linalg.norm(array_init)}, ||x_base||: {np.linalg.norm(x_base)}")
        array_final = self.G(array_init, Tp, ax, az)
        # logger.info(f"G computed, ||G(x_base + epsilon*x_perturb)||: {np.linalg.norm(array_final)}, ||G(x_base)||: {np.linalg.norm(phi_base)}")
        array_out = (array_final-phi_base)/epsilon
        # logger.info(f"DG output computed, ||DG||: {np.linalg.norm(array_out)}")
        return array_out
    def LinearOperator(self, x, x_perturb, phi_base):
        ''' Linearized operator for the Newton iteration 
            (F^T(x0+dx) - F^T(x0)) / ||dx|| - dx
        '''
        N_ = self.model.size()
        
        T_temp, ax_temp, az_temp = self.Tp, 0.0, 0.0
        if self.Tsearch:
            T_temp = x[N_+self.Tsearch-1]
            delta_T = x_perturb[N_+self.Tsearch-1]
        if self.Rxsearch:
            ax_temp = x[N_+self.Tsearch+self.Rxsearch-1]
            delta_ax = x_perturb[N_+self.Tsearch+self.Rxsearch-1]
        if self.Rzsearch:
            az_temp = x[N_+self.Tsearch+self.Rxsearch+self.Rzsearch-1]
            delta_az = x_perturb[N_+self.Tsearch+self.Rxsearch+self.Rzsearch-1]

        
        x_base = np.copy(x[:N_])
        delta_x = np.copy(x_perturb[:N_])

        array_out = np.zeros_like(x)
        array_out[:N_] = self.DG(x_base, delta_x, phi_base, T_temp, ax_temp, az_temp) - delta_x
        # logger.info(f"Linear operator applied, ||DG||: {np.linalg.norm(array_out[:N_])}, ||delta_x||: {np.linalg.norm(delta_x)}")
        if self.Tsearch:
            array_out[:N_] += self.model.t_derivative(phi_base) * delta_T
            array_out[N_+self.Tsearch-1] = np.matmul(np.conj(self.model.t_derivative(x_base)), delta_x)
        if self.Rxsearch:
            array_out[:N_] += self.model.x_derivative(phi_base) * delta_ax
            array_out[N_+self.Tsearch+self.Rxsearch-1] = np.matmul(np.conj(self.model.x_derivative(x_base)), delta_x)
        if self.Rzsearch:
            array_out[:N_] += self.model.z_derivative(phi_base) * delta_az
            array_out[N_+self.Tsearch+self.Rxsearch+self.Rzsearch-1] = np.matmul(np.conj(self.model.z_derivative(x_base)), delta_x)
        return array_out
    def NonlinearOperator(self, x):
        ''' Return sigma*F(x) - x '''
        N_ = self.model.size()
        state = x[:N_]
        T_temp = x[N_+self.Tsearch-1] if self.Tsearch else self.Tp
        ax_temp = x[N_+self.Tsearch+self.Rxsearch-1] if self.Rxsearch else 0.0
        az_temp = x[N_+self.Tsearch+self.Rxsearch+self.Rzsearch-1] if self.Rzsearch else 0.0
        F_state = self.G(state, T_temp, ax_temp, az_temp)
        return (- F_state + state)  # Residual for the state variables
    def arnoldi_iteration_inner(self, x_base, Q, phi_base, k:int):
        Qk = self.LinearOperator(x_base, Q[:, k - 1], phi_base)
        # logger.info(f"Arnoldi iteration {k}, ||Qk|| before orthogonalization: {np.linalg.norm(Qk)}")
        Hk = np.zeros(k+1)
        for j in range(0, k):
            Hk[j] = np.matmul(np.conj(Q[:,j]), Qk)
            Qk = Qk- Hk[j]*Q[:,j]
        Hk[k] = np.linalg.norm(Qk)
        Qk = Qk/Hk[k]
        return Qk, Hk
    def arnoldi_iteration(self, x_base, phi_base, T, ax, az, r, n:int):
        ''' Arnoldi iteration '''
        # Ensure starting vector is orthogonal to neutral direction
        def project_out(v):
            dudt_ref = self.model.t_derivative(self.IVP_problem, x_base, self.d_tol) # time-derivative
            dudt_ref = dudt_ref / np.linalg.norm(dudt_ref) # normalization
            dudx_ref = self.model.x_derivative(x_base) # x-derivative
            dudx_ref = dudx_ref / np.linalg.norm(dudx_ref) # normalization
            dudz_ref = self.model.z_derivative(x_base) # z-derivative
            dudz_ref = dudz_ref / np.linalg.norm(dudz_ref) # normalization
            gg = np.vdot(dudt_ref, dudt_ref) # <dudt_ref, dudt_ref>
            gv = np.vdot(dudt_ref, v) # <dudt_ref, v>
            gg1 = np.vdot(dudx_ref, dudx_ref)
            gg2 = np.vdot(dudz_ref, dudz_ref)
            g1v = np.vdot(dudx_ref, v)
            g2v = np.vdot(dudz_ref, v)
            return v - dudt_ref * (gv / gg) - dudx_ref * (g1v / gg1) - dudz_ref * (g2v / gg2)
        if self.projectNeutralDrift:
            r = project_out(r)
        Q = np.zeros((r.size, n+1))
        H = np.zeros((n+1, n))
        Q[:,0] = r/np.linalg.norm(r)
        for k in range(1, n + 1):
            if self.projectNeutralDrift:
                q_in = project_out(Q[:, k-1]) # Project input before applying L
                v = self.DG(x_base, q_in, phi_base, T, ax, az) # Apply operator
                v = project_out(v) # Project output
            else:
                v = self.DG(x_base, Q[:, k - 1], phi_base, T, ax, az)
            for j in range(0, k): # Arnoldi orthogonalization
                H[j, k-1] = np.vdot(Q[:,j], v)
                v = v - H[j, k-1]*Q[:,j]
            H[k, k-1] = np.linalg.norm(v)
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
    def GMRES(self, x_base, x_pert, phi_base, b, kmax, tr):
        xk = np.copy(x_pert)
        # logger.info("Starting GMRES ...")
        # logger.info(f"Initial perturbation norm: {np.linalg.norm(x_pert)}, Initial residual norm: {np.linalg.norm(b)}")
        r = self.LinearOperator(x_base, x_pert, phi_base) - b
        rho = np.linalg.norm(r)
        beta = rho
        b_norm = np.linalg.norm(b)
        # logger.info(f"Initial GMRES residual norm: {rho}, ||b||: {b_norm}")
        Q = np.zeros((x_pert.size, kmax+1))
        H = np.zeros((kmax+1, kmax))
        
        min_error = np.inf
        min_vector = np.zeros(x_pert.size)
        
        Q[:,0] = r/np.linalg.norm(r)
        best_k = 1
        for k in range(1, kmax):
            Q[:,k], H[:k+1,k-1] = self.arnoldi_iteration_inner(x_base, Q[:,0:k], phi_base, k)
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
        test = np.linalg.norm(self.NonlinearOperator(np.copy(x_base)+x_pert+xk))
        # logger.info(f"Initial optimal residual norm: {test}, min GMRES residual norm: {min_error}")
        tr_local = tr
        while test > 0.99*b_norm and tr_local > self.trust_radius_min:
            res = self.Hookstep(H, beta, best_k, tr_local)
            xk = np.matmul(Q[:,0:(best_k)], res.x)
            min_error = np.linalg.norm(res.fun)
            test = np.linalg.norm(self.NonlinearOperator(np.copy(x_base)+x_pert+xk))
            tr_local = 0.5*tr_local
            # logger.info(f"Hookstep-based optimal residual norm: {test}, min GMRES residual norm: {min_error}, trust radius: {tr_local}")
        return x_pert + xk, min_error, tr_local
    
    def solve(self, 
              x0, 
              Tp=0.02, 
              ax = 0.0, 
              az = 0.0,
              dt=2e-4):
        self.model.init_dt = dt
        self.Tp = Tp
        N_ = self.model.size()
        logger.info("Starting Newton solver ...")
        x = np.concatenate([x0, 
                            [Tp] if self.Tsearch else [], 
                            [ax] if self.Rxsearch else [], 
                            [az] if self.Rzsearch else []])  # Initial guess includes state and parameters
        x_pert = np.zeros_like(x)
        
        for i in range(self.max_iter):
            self.model.set_state(x[:N_])
            self.model.preview()
            nonlinear_res = self.NonlinearOperator(x)
            norm_b = np.linalg.norm(nonlinear_res)
            # logger.info(f"Iteration {i}, Residual norm: {norm_b}, Tp: {x[N_+self.Tsearch-1] if self.Tsearch else Tp}, ax: {x[N_+self.Tsearch+self.Rxsearch-1] if self.Rxsearch else ax}, az: {x[N_+self.Tsearch+self.Rxsearch+self.Rzsearch-1] if self.Rzsearch else az}")
            if norm_b < self.tol:
                logger.info("Convergence achieved!")
                # save the solution to an h5 file
                self.model.set_state(x[:N_])
                self.model.save_state(self.odir + 'solution.h5')
                break
            T_temp = x[N_+self.Tsearch-1] if self.Tsearch else self.Tp
            ax_temp = x[N_+self.Tsearch+self.Rxsearch-1] if self.Rxsearch else 0.0
            az_temp = x[N_+self.Tsearch+self.Rxsearch+self.Rzsearch-1] if self.Rzsearch else 0.0
            phi_base = self.G(x[:N_], T_temp, ax_temp, az_temp)
            dx, error, tr = self.GMRES(x, x_pert, phi_base, nonlinear_res, self.krylov_dim, self.trust_radius)
            x += dx # Update the solution
            nonlinear_res = self.NonlinearOperator(x)
            norm_b = np.linalg.norm(nonlinear_res)
            logger.info(f"Iteration {i}, ||x||: {np.linalg.norm(x[:N_])}, Residual: {norm_b}, GMRES error: {error}, trust radius: {tr}")
            
        # else:
        #     logger.warning("Maximum iterations reached without convergence.")
        return x
    

    def stability(self,x):
        nonlinear_res = self.NonlinearOperator(x)
        norm_b = np.linalg.norm(nonlinear_res)
        if norm_b < 1e-6:
            logger.info('Solving linear stability problem around converged solution ...')
            if self.model.dist.comm.rank == 0:
                if not os.path.exists(self.odir+'stability/'):
                        os.mkdir(self.odir+'stability/')

            N_ = self.model.size()
            T_temp = x[N_+self.Tsearch-1] if self.Tsearch else self.Tp
            ax_temp = x[N_+self.Tsearch+self.Rxsearch-1] if self.Rxsearch else 0.0
            az_temp = x[N_+self.Tsearch+self.Rxsearch+self.Rzsearch-1] if self.Rzsearch else 0.0
            
            phi_base = self.G(x[:N_], T_temp, ax_temp, az_temp)
            # Floquet method
            Q, H_ = self.arnoldi_iteration(x[:N_], phi_base, T_temp, ax_temp, az_temp, np.random.rand(N_), self.Ne) # <-- Ne iterations
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
            
            xg = self.model.x_basis.global_grid(self.model.dist, scale=self.model.dealias) 
            zg = self.model.z_basis.global_grid(self.model.dist, scale=self.model.dealias)
            unstable = np.where(growthrate.real > 0)[0]
            if MPI.COMM_WORLD.rank == 0 and unstable.size > 0:
                eigenvectors_unstable = eigenvectors[:, unstable]
                eigenvalues_unstable = eigenvalues[unstable]
                growthrate_unstable = growthrate[unstable]
                h5f = h5py.File(self.odir+'stability/eigen_unstable.h5', 'w') 
                h5f.create_dataset('/eigenvectors', data = eigenvectors_unstable) 
                h5f.create_dataset('/eigenvalues', data = eigenvalues_unstable) 
                h5f.create_dataset('/growthrate', data = growthrate_unstable) 
                h5f.create_dataset('/xg', data = xg) 
                h5f.create_dataset('/zg', data = zg) 
                h5f.close()