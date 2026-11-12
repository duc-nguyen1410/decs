"""
Salt-finger convection in a fully periodic domain (validated model, paper submitted).
Builds on the shared DoubleDiffusion base in double_diffusion_base.py.
"""
import dedalus.public as de
import numpy as np
from mpi4py import MPI
import h5py
import matplotlib.pyplot as plt
from .double_diffusion_base import DoubleDiffusion


class SaltFinger(DoubleDiffusion):
    def build_problems(self):
        self.build_ivp_problem()
    def build_ivp_problem(self):
        tau_p = self.dist.Field(name='tau_p')
        tau_u = self.dist.VectorField(self.coords, name='tau_u')
        tau_te = self.dist.Field(name='tau_te')
        tau_sa = self.dist.Field(name='tau_sa')

        unit_vectors = self.coords.unit_vector_fields(self.dist)
        if self.dim == 2:
            ex, ez = unit_vectors
            ey = None # Or a zero-field if needed
        else:
            ex, ey, ez = unit_vectors

        # PARAMETERS / BACKGROUND FIELDS (The Knowns)
        self.Ra_param = self.dist.Field(name='Ra_param'); self.Ra_param['g'] = self.params['Ra']
        self.Pr_param = self.dist.Field(name='Pr_param'); self.Pr_param['g'] = self.params['Pr']
        self.Rrho_param = self.dist.Field(name='Rrho_param'); self.Rrho_param['g'] = self.params['Rrho']
        self.tau_param = self.dist.Field(name='tau_param'); self.tau_param['g'] = self.params['tau']
        
        ns = {'np': np,
              'Ra': self.Ra_param, 'Pr': self.Pr_param, 'Rrho': self.Rrho_param, 'tau': self.tau_param,
              'ex': ex, 'ey': ey, 'ez': ez,
              'w': self.u @ ez,
             }

        vars = [self.p, self.u, self.te, self.sa, 
                tau_p, tau_u, tau_te, tau_sa]
        self.ivp_problem = de.IVP(vars, namespace=ns)
        # Periodic Governing Equations nondimensionalized by thermal-diffusion-scaled velocity kappaT/H
        self.ivp_problem.add_equation("trace(grad(u)) + tau_p = 0")
        self.ivp_problem.add_equation("integ(p) = 0") 
        if self.params.get('stokes', False):
            self.ivp_problem.add_equation("grad(p) - Pr*lap(u) - Pr*Ra*(te-sa/Rrho)*ez + tau_u = 0")
        else:
            self.ivp_problem.add_equation("dt(u) + grad(p) - Pr*lap(u) - Pr*Ra*(te-sa/Rrho)*ez + tau_u = - u@grad(u)")
        if self.params.get('temperature_feedback', False):
            self.ivp_problem.add_equation("- lap(te) + w + tau_te = 0")
        else:
            self.ivp_problem.add_equation("dt(te) - lap(te) + w + tau_te = - u@grad(te)")
        self.ivp_problem.add_equation("dt(sa) - tau*lap(sa) + w + tau_sa = - u@grad(sa)")
        # Integral constraints for floating zero-means
        for v in ['u', 'te', 'sa']:
            self.ivp_problem.add_equation(f"integ({v}) = 0")
    def build_evp_problem(self, coupled=True):
        """Linearisation of the IVP about the base state (u_eq, te_eq, sa_eq), in a frame
        moving with velocity c (traveling waves): dt -> sigma, and -c.grad(q) wherever dt
        appears. Same options as the IVP ('stokes', 'temperature_feedback').
        coupled: base fields on all bases (any base state); False: z-only base fields
        (horizontally uniform base state, one EVP per horizontal wavenumber).
        Used by FluidModel.solve_evp; needs build_ivp_problem first (parameter fields)."""
        sigma = self.dist.Field(name='sigma')
        tau_p = self.dist.Field(name='tau_p_evp')
        tau_u = self.dist.VectorField(self.coords, name='tau_u_evp')
        tau_te = self.dist.Field(name='tau_te_evp')
        tau_sa = self.dist.Field(name='tau_sa_evp')
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        u_eq, te_eq, sa_eq = self.evp_base_fields(coupled)

        ns = {'Ra': self.Ra_param, 'Pr': self.Pr_param, 'Rrho': self.Rrho_param, 'tau': self.tau_param,
              'ez': ez, 'w': self.u @ ez, 'c': self.evp_frame_velocity(),
              'u_eq': u_eq, 'te_eq': te_eq, 'sa_eq': sa_eq,
              'sigma': sigma, 'tau_p': tau_p, 'tau_u': tau_u, 'tau_te': tau_te, 'tau_sa': tau_sa}
        variables = [self.p, self.u, self.te, self.sa, tau_p, tau_u, tau_te, tau_sa]
        evp = de.EVP(variables, eigenvalue=sigma, namespace=ns)
        evp.add_equation("trace(grad(u)) + tau_p = 0")
        evp.add_equation("integ(p) = 0")
        if self.params.get('stokes', False):
            evp.add_equation("grad(p) - Pr*lap(u) - Pr*Ra*(te-sa/Rrho)*ez + tau_u = 0")
        else:
            evp.add_equation("sigma*u - c@grad(u) + grad(p) - Pr*lap(u) - Pr*Ra*(te-sa/Rrho)*ez"
                             " + u@grad(u_eq) + u_eq@grad(u) + tau_u = 0")
        if self.params.get('temperature_feedback', False):
            evp.add_equation("- lap(te) + w + tau_te = 0")
        else:
            evp.add_equation("sigma*te - c@grad(te) - lap(te) + w + u@grad(te_eq) + u_eq@grad(te) + tau_te = 0")
        evp.add_equation("sigma*sa - c@grad(sa) - tau*lap(sa) + w + u@grad(sa_eq) + u_eq@grad(sa) + tau_sa = 0")
        for v in ['u', 'te', 'sa']:
            evp.add_equation(f"integ({v}) = 0")
        return evp

    def get_flow_properties(self):
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        w = self.u @ ez
        Ra = self.params['Ra']
        Pr = self.params['Pr']
        Rrho = self.params['Rrho']
        tau = self.params['tau']
        # Nusselt and Sherwood numbers 
        Ft = de.Average(w*self.te)
        Fs = de.Average(w*self.sa)
        Nu = 1 - de.Average(w*self.te)
        Sh = 1 - de.Average(w*self.sa)/tau
        # Energy
        grad_u = de.grad(self.u)
        grad_u2 = sum(de.dot(grad_u @ e, grad_u @ e) for e in self.coords.unit_vector_fields(self.dist)) # |grad u|^2, all components
        D = de.Average(Pr*grad_u2) # Viscous dissipation
        I = de.Average(Pr*Ra*(self.te-self.sa/Rrho)*w) # kinetic energy input by bouyancy
        I_t = de.Average(Pr*Ra*self.te*w) # kinetic energy input by temperature-induced bouyancy
        I_s = de.Average(-Pr*Ra*self.sa*w/Rrho) # kinetic energy input by salinity-induced bouyancy
        # .evaluate() returns a field object
        # ['g'] accesses the grid data
        Ft_val = Ft.evaluate()['g'].real
        Fs_val = Fs.evaluate()['g'].real
        # Nusselt and Sherwood numbers 
        Nu_val = Nu.evaluate()['g'].real
        Sh_val = Sh.evaluate()['g'].real
        D_val = D.evaluate()['g'].real
        I_val = I.evaluate()['g'].real
        I_t_val = I_t.evaluate()['g'].real
        I_s_val = I_s.evaluate()['g'].real
        if self.dist.comm.rank == 0:
            return {'Ft': float(Ft_val.item()),
                    'Fs': float(Fs_val.item()),
                    'Nu': float(Nu_val.item()),
                    'Sh': float(Sh_val.item()),
                    'D': float(D_val.item()),
                    'I': float(I_val.item()),
                    'I_t': float(I_t_val.item()),
                    'I_s': float(I_s_val.item()),
                    }
        else:
            return {}

    def set_snapshots(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        snapshots = solver.evaluator.add_file_handler(self.odir+'snapshots', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        snapshots.add_task(self.te, name='te')
        snapshots.add_task(self.sa, name='sa')
        snapshots.add_task(self.u, name='u')
        
    def set_flowproperties(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        w = self.u @ ez
        Ra = self.params['Ra']
        Pr = self.params['Pr']
        Rrho = self.params['Rrho']
        tau = self.params['tau']
        flowproperties = solver.evaluator.add_file_handler(self.odir+'flowproperties', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        flowproperties.add_task(1 - de.Average(w*self.te), name='Nu') # Nusselt number
        flowproperties.add_task(1 - de.Average(w*self.sa)/tau, name='Sh') # Sherwood number
        grad_u = de.grad(self.u)
        grad_u2 = sum(de.dot(grad_u @ e, grad_u @ e) for e in self.coords.unit_vector_fields(self.dist)) # |grad u|^2, all components
        flowproperties.add_task(de.Average(Pr*grad_u2), name='D') # Viscous dissipation
        flowproperties.add_task(de.Average(Pr*Ra*(self.te-self.sa/Rrho)*w), name='I') # kinetic energy input by bouyancy
        flowproperties.add_task(de.Average(Pr*Ra*self.te*w), name='I_t') # kinetic energy input by temperature-induced bouyancy
        flowproperties.add_task(de.Average(-Pr*Ra*self.sa*w/Rrho), name='I_s') # kinetic energy input by salinity-induced bouyancy

    def set_meanprofiles(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        if self.dim == 2:
            h_mean = lambda A: de.Average(A,'x')
        else:
            h_mean = lambda A: de.Average(A,('x', 'y'))
        meanprofiles = solver.evaluator.add_file_handler(self.odir+'meanprofiles', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        meanprofiles.add_task(h_mean(self.u), name='u') # horizontal averaged x-axis velocity
        meanprofiles.add_task(h_mean(self.te), name='te') # horizontal averaged temperature
        meanprofiles.add_task(h_mean(self.sa), name='sa') # horizontal averaged sanility
