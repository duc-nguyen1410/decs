"""
Thermohaline-shear instability, Radko (2019) (validated model, paper in preparation).
Builds on the shared DoubleDiffusion base in double_diffusion_base.py.
"""
import dedalus.public as de
import numpy as np
from mpi4py import MPI
import h5py
import matplotlib.pyplot as plt
from .double_diffusion_base import DoubleDiffusion


class ThermohalineShear(DoubleDiffusion):
    ''' Thermohaline-shear instability. Read paper Radko (2019) JFM '''
    def build_problems(self):
        self.build_ivp_problem()
    def build_ivp_problem(self):
        tau_p = self.dist.Field(name='tau_p')
        tau_u = self.dist.VectorField(self.coords, name='tau_u') # velocity
        tau_te = self.dist.Field(name='tau_te')
        tau_sa = self.dist.Field(name='tau_sa')
        
        unit_vectors = self.coords.unit_vector_fields(self.dist)
        if self.dim == 2:
            ex, ez = unit_vectors
            ey = None # Or a zero-field if needed
        else:
            ex, ey, ez = unit_vectors
        z, = self.dist.local_grids(self.bases[-1])

        grad_u = de.grad(self.u)
        grad_te = de.grad(self.te)
        grad_sa = de.grad(self.sa)
        lap_u = de.div(grad_u)
        lap_te = de.div(grad_te)
        lap_sa = de.div(grad_sa)

        dx = lambda A: de.Differentiate(A, self.coords['x']) 
        dz = lambda A: de.Differentiate(A, self.coords['z']) 

        # PARAMETERS / BACKGROUND FIELDS (The Knowns)
        self.Pe_param = self.dist.Field(name='Pe_param'); self.Pe_param['g'] = self.params['Pe']
        self.Ri_param = self.dist.Field(name='Ri_param'); self.Ri_param['g'] = self.params['Ri']
        self.Pr_param = self.dist.Field(name='Pr_param'); self.Pr_param['g'] = self.params['Pr']
        self.Lambda_param = self.dist.Field(name='Lambda_param'); self.Lambda_param['g'] = self.params['Lambda']
        self.tau_param = self.dist.Field(name='tau_param'); self.tau_param['g'] = self.params['tau']
        
        if self.params.get('infinite-Pr', False):
            self.Ra_param = self.dist.Field(name='Ra_param'); self.Ra_param['g'] = (4*(np.pi**2)*self.params['Ri']*self.params['Pe']**2)/(self.params['Pr']*(self.params['Lambda']-1))

        baru = self.dist.Field(bases=self.bases[-1]); baru['g'] = np.sin(2*np.pi*z)

        ns = {'np': np,
              'Pe': self.Pe_param, 'Pr': self.Pr_param, 'Lambda': self.Lambda_param, 'tau': self.tau_param, 'Ri': self.Ri_param,
              'ex': ex, 'ey': ey, 'ez': ez,
              'ux': self.u @ ex, 'w': self.u @ ez,
              'grad_u': grad_u, 'grad_te': grad_te, 'grad_sa': grad_sa, 
              'lap_u': lap_u, 'lap_te': lap_te, 'lap_sa': lap_sa,
              'dx': dx, 'dz': dz,
              'baru': baru,
             }
        if self.params.get('infinite-Pr', False):
            ns.update({'Ra': self.Ra_param})
        vars = [self.p, self.u, self.te, self.sa, 
                tau_p, tau_u, tau_te, tau_sa]
        self.ivp_problem = de.IVP(vars, namespace=ns)

        # Governing Equations
        self.ivp_problem.add_equation("trace(grad_u) + tau_p = 0")
        self.ivp_problem.add_equation("integ(p) = 0") 
        # Momentum equation
        if self.params.get('stokes', False):
            self.ivp_problem.add_equation("grad(p) - Pr/Pe*lap_u - (4*(np.pi**2)*Ri)/(Lambda-1)*(te-sa)*ez + tau_u = 0")
        elif self.params.get('infinite-Pr', False):
            self.ivp_problem.add_equation("grad(p) - 1.0/Pe*lap_u - Ra/(Pe**2)*(te-sa)*ez + tau_u = 0")
        else:
            self.ivp_problem.add_equation("dt(u) + baru*dx(u) + w*dz(baru)*ex + grad(p) - Pr/Pe*lap_u - (4*(np.pi**2)*Ri)/(Lambda-1)*(te-sa)*ez + tau_u = - u@grad(u)")
        # Temperature
        if self.params.get('zero-temp-feedback', False):
            self.ivp_problem.add_equation("dt(te) - (1.0/Pe)*lap_te - w + tau_te = 0")
        elif self.params.get('fully-zero-temp-feedback', False):
            self.ivp_problem.add_equation("- (1.0/Pe)*lap_te - w + tau_te = 0")
        else: 
            self.ivp_problem.add_equation("dt(te) + baru*dx(te) - (1.0/Pe)*lap_te - w + tau_te = - u@grad(te)")
        # Salinity
        self.ivp_problem.add_equation("dt(sa) + baru*dx(sa) - (tau/Pe)*lap_sa - Lambda*w + tau_sa = - u@grad(sa)")
        # Constraints
        self.ivp_problem.add_equation("integ(u) = 0") 
        self.ivp_problem.add_equation("integ(te) = 0") 
        self.ivp_problem.add_equation("integ(sa) = 0")

    def build_evp_problem(self, coupled=True):
        """Linearisation of the IVP about the base state (u_eq, te_eq, sa_eq) on top of the
        background shear baru = sin(2 pi z), in a frame moving with velocity c (traveling
        waves): dt -> sigma, and -c.grad(q) wherever dt appears. Same options as the IVP
        ('stokes', 'infinite-Pr', 'zero-temp-feedback', 'fully-zero-temp-feedback').
        coupled: base fields on all bases (any base state); False: z-only base fields
        (horizontally uniform base state, one EVP per horizontal wavenumber).
        Used by FluidModel.solve_evp; needs build_ivp_problem first (parameter fields)."""
        sigma = self.dist.Field(name='sigma')
        tau_p = self.dist.Field(name='tau_p_evp')
        tau_u = self.dist.VectorField(self.coords, name='tau_u_evp')
        tau_te = self.dist.Field(name='tau_te_evp')
        tau_sa = self.dist.Field(name='tau_sa_evp')
        ex = self.coords.unit_vector_fields(self.dist)[0]
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        z, = self.dist.local_grids(self.bases[-1])
        baru = self.dist.Field(bases=self.bases[-1]); baru['g'] = np.sin(2*np.pi*z)
        u_eq, te_eq, sa_eq = self.evp_base_fields(coupled)

        ns = {'np': np, 'Pe': self.Pe_param, 'Pr': self.Pr_param, 'Lambda': self.Lambda_param,
              'tau': self.tau_param, 'Ri': self.Ri_param,
              'ex': ex, 'ez': ez, 'w': self.u @ ez, 'c': self.evp_frame_velocity(),
              'dx': lambda A: de.Differentiate(A, self.coords['x']),
              'dz': lambda A: de.Differentiate(A, self.coords['z']),
              'baru': baru, 'u_eq': u_eq, 'te_eq': te_eq, 'sa_eq': sa_eq,
              'sigma': sigma, 'tau_p': tau_p, 'tau_u': tau_u, 'tau_te': tau_te, 'tau_sa': tau_sa}
        if self.params.get('infinite-Pr', False):
            ns['Ra'] = self.Ra_param
        variables = [self.p, self.u, self.te, self.sa, tau_p, tau_u, tau_te, tau_sa]
        evp = de.EVP(variables, eigenvalue=sigma, namespace=ns)
        evp.add_equation("trace(grad(u)) + tau_p = 0")
        evp.add_equation("integ(p) = 0")
        # Momentum
        if self.params.get('stokes', False):
            evp.add_equation("grad(p) - Pr/Pe*lap(u) - (4*(np.pi**2)*Ri)/(Lambda-1)*(te-sa)*ez + tau_u = 0")
        elif self.params.get('infinite-Pr', False):
            evp.add_equation("grad(p) - 1.0/Pe*lap(u) - Ra/(Pe**2)*(te-sa)*ez + tau_u = 0")
        else:
            evp.add_equation("sigma*u - c@grad(u) + baru*dx(u) + w*dz(baru)*ex + grad(p) - Pr/Pe*lap(u)"
                             " - (4*(np.pi**2)*Ri)/(Lambda-1)*(te-sa)*ez + u@grad(u_eq) + u_eq@grad(u) + tau_u = 0")
        # Temperature
        if self.params.get('zero-temp-feedback', False):
            evp.add_equation("sigma*te - c@grad(te) - (1.0/Pe)*lap(te) - w + tau_te = 0")
        elif self.params.get('fully-zero-temp-feedback', False):
            evp.add_equation("- (1.0/Pe)*lap(te) - w + tau_te = 0")
        else:
            evp.add_equation("sigma*te - c@grad(te) + baru*dx(te) - (1.0/Pe)*lap(te) - w"
                             " + u@grad(te_eq) + u_eq@grad(te) + tau_te = 0")
        # Salinity
        evp.add_equation("sigma*sa - c@grad(sa) + baru*dx(sa) - (tau/Pe)*lap(sa) - Lambda*w"
                         " + u@grad(sa_eq) + u_eq@grad(sa) + tau_sa = 0")
        for v in ['u', 'te', 'sa']:
            evp.add_equation(f"integ({v}) = 0")
        return evp

    def get_flow_properties(self):
        ex = self.coords.unit_vector_fields(self.dist)[0]
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        ux = self.u @ ex
        w = self.u @ ez
        Pr = self.params['Pr']
        Pe = self.params['Pe']
        Lambda = self.params['Lambda']
        tau = self.params['tau']
        Ri = self.params['Ri']
        z, = self.dist.local_grids(self.bases[-1])
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        baru = self.dist.Field(bases=self.bases[-1])
        baru['g'] = np.sin(2*np.pi*z) # background shear of the IVP (unit amplitude)
        Ft = de.Average(w*self.te)
        Fs = de.Average(w*self.sa)
        # Nusselt and Sherwood numbers 
        Nu = 1 + de.Average(w*self.te)*Pe
        Sh = 1 + de.Average(w*self.sa)*Pe/tau
        grad_u = de.grad(self.u)
        # D = D_base + D_fluc
        # D_base = np.sqrt(Pr/Ra) * 2 * (np.pi**2) * (A_s**2)
        # save D_fluc
        grad_u2 = sum(de.dot(grad_u @ e, grad_u @ e) for e in self.coords.unit_vector_fields(self.dist)) # |grad u|^2, all components
        D = Pr/Pe * de.Average(grad_u2) # Viscous dissipation
        I = (4*(np.pi**2)*Ri)/(Lambda-1)*de.Average((self.te-self.sa)*w) # kinetic energy input by bouyancy
        I_t = (4*(np.pi**2)*Ri)/(Lambda-1)*de.Average(self.te*w) # kinetic energy input by temperature-induced bouyancy
        I_s = -(4*(np.pi**2)*Ri)/(Lambda-1)*de.Average(self.sa*w) # kinetic energy input by salinity-induced bouyancy
        I_shear = de.Average(-ux * w * dz(baru)) # kinetic energy input by background shear
        
        # .evaluate() returns a field object
        Ft_val = Ft.evaluate()['g'].real
        Fs_val = Fs.evaluate()['g'].real
        Nu_val = Nu.evaluate()['g'].real
        Sh_val = Sh.evaluate()['g'].real
        D_val = D.evaluate()['g'].real
        I_val = I.evaluate()['g'].real
        I_t_val = I_t.evaluate()['g'].real
        I_s_val = I_s.evaluate()['g'].real
        I_shear_val = I_shear.evaluate()['g'].real
        if self.dist.comm.rank == 0:
            return {'Ft': float(Ft_val.item()),
                    'Fs': float(Fs_val.item()),
                    'Nu': float(Nu_val.item()),
                    'Sh': float(Sh_val.item()),
                    'D': float(D_val.item()),
                    'I_shear': float(I_shear_val.item()),
                    'I': float(I_val.item()),
                    'I_t': float(I_t_val.item()),
                    'I_s': float(I_s_val.item())}
        else:
            return {}

    def set_snapshots(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        snapshots = solver.evaluator.add_file_handler(self.odir+'snapshots', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        snapshots.add_task(self.u, name='u')
        snapshots.add_task(self.te, name='te')
        snapshots.add_task(self.sa, name='sa')
        
    def set_flowproperties(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        ex = self.coords.unit_vector_fields(self.dist)[0]
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        ux = self.u @ ex
        w = self.u @ ez
        Pe = self.params['Pe']
        Pr = self.params['Pr']
        Lambda = self.params['Lambda']
        tau = self.params['tau']
        Ri = self.params['Ri']
        z, = self.dist.local_grids(self.bases[-1])
        Lz = self.bases[-1].bounds[1]
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        baru = self.dist.Field(bases=self.bases[-1])
        baru['g'] = np.sin(2*np.pi*z) # background shear of the IVP (unit amplitude)

        flowproperties = solver.evaluator.add_file_handler(self.odir+'flowproperties', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        flowproperties.add_task(1+Pe*de.Average(w*self.te), name='Nu') # Nusselt number
        flowproperties.add_task(1+Pe/tau*de.Average(w*self.sa), name='Sh') # Sherwood number
        grad_u = de.grad(self.u)
        # D = D_base + D_fluc
        # D_base = np.sqrt(Pr/Ra) * 2 * (np.pi**2) * (A_s**2)
        # save D_fluc
        grad_u2 = sum(de.dot(grad_u @ e, grad_u @ e) for e in self.coords.unit_vector_fields(self.dist)) # |grad u|^2, all components
        flowproperties.add_task(Pr/Pe * de.Average(grad_u2), name='D') # Viscous dissipation
        flowproperties.add_task(de.Average(-ux * w * dz(baru)), name='I_shear') # kinetic energy input by background shear
        flowproperties.add_task((4*(np.pi**2)*Ri)/(Lambda-1)*de.Average((self.te-self.sa)*w), name='I') # kinetic energy input by bouyancy
        flowproperties.add_task((4*(np.pi**2)*Ri)/(Lambda-1)*de.Average(self.te*w), name='I_t') # kinetic energy input by temperature-induced bouyancy
        flowproperties.add_task(-(4*(np.pi**2)*Ri)/(Lambda-1)*de.Average(self.sa*w), name='I_s') # kinetic energy input by salinity-induced bouyancy

    def set_meanprofiles(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        if self.dim == 2:
            h_mean = lambda A: de.Average(A,'x')
        else:
            h_mean = lambda A: de.Average(A,('x', 'y'))
        meanprofiles = solver.evaluator.add_file_handler(self.odir+'meanprofiles', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        meanprofiles.add_task(h_mean(self.u), name='u') # horizontal averaged x-axis velocity
        meanprofiles.add_task(h_mean(self.te), name='te') # horizontal averaged temperature
        meanprofiles.add_task(h_mean(self.sa), name='sa') # horizontal averaged sanility
        u_prime = self.u - h_mean(self.u)
        te_prime = self.te - h_mean(self.te)
        sa_prime = self.sa - h_mean(self.sa)
        meanprofiles.add_task(de.sqrt(h_mean(u_prime**2)), name="u_rms")
        meanprofiles.add_task(de.sqrt(h_mean(te_prime**2)), name="te_rms")
        meanprofiles.add_task(de.sqrt(h_mean(sa_prime**2)), name="sa_rms")

