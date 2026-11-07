"""
Double-diffusive models that have NOT been validated yet:

    BoundedSaltFinger                salt fingers between walls
    DiffusiveConvection              diffusive convection (currently fails to build: self.sigma undefined)
    ShearedDiffusiveConvection       diffusive convection with background shear
    WallShearedDiffusiveConvection   wall-bounded sheared diffusive convection

Base class: double_diffusion_base.py. Validated models live in their own files:
    salt_finger.py         SaltFinger
    thermohaline_shear.py  ThermohalineShear
"""
import dedalus.public as de
import numpy as np
from mpi4py import MPI
import h5py
import matplotlib.pyplot as plt
from .double_diffusion_base import DoubleDiffusion 


class BoundedSaltFinger(DoubleDiffusion):
    def build_problems(self):
        self.build_ivp_problem()
    def build_ivp_problem(self):
        tau_p = self.dist.Field(name='tau_p')
        tau_u1 = self.dist.VectorField(self.coords, name='tau_u1', bases=self.bases[:-1])
        tau_te1 = self.dist.Field(name='tau_te1', bases=self.bases[:-1])
        tau_sa1 = self.dist.Field(name='tau_sa1', bases=self.bases[:-1])
        tau_u2 = self.dist.VectorField(self.coords, name='tau_u2', bases=self.bases[:-1])
        tau_te2 = self.dist.Field(name='tau_te2', bases=self.bases[:-1])
        tau_sa2 = self.dist.Field(name='tau_sa2', bases=self.bases[:-1])

        unit_vectors = self.coords.unit_vector_fields(self.dist)
        if self.dim == 2:
            ex, ez = unit_vectors
            ey = None # Or a zero-field if needed
        else:
            ex, ey, ez = unit_vectors
        
        lift_basis = self.bases[-1].derivative_basis(1)
        lift = lambda A: de.Lift(A, lift_basis, -1)

        grad_u = de.grad(self.u) + ez*lift(tau_u1) 
        grad_te = de.grad(self.te) + ez*lift(tau_te1) 
        grad_sa = de.grad(self.sa) + ez*lift(tau_sa1) 
        lap_u = de.div(grad_u)
        lap_te = de.div(grad_te)
        lap_sa = de.div(grad_sa)

        dx = lambda A: de.Differentiate(A, self.coords['x']) 
        dz = lambda A: de.Differentiate(A, self.coords['z']) 

        # PARAMETERS / BACKGROUND FIELDS (The Knowns)
        self.Ra_param = self.dist.Field(name='Ra_param'); self.Ra_param['g'] = self.params['Ra']
        self.Pr_param = self.dist.Field(name='Pr_param'); self.Pr_param['g'] = self.params['Pr']
        self.Rrho_param = self.dist.Field(name='Rrho_param'); self.Rrho_param['g'] = self.params['Rrho']
        self.tau_param = self.dist.Field(name='tau_param'); self.tau_param['g'] = self.params['tau']
        self.Ri_param = self.dist.Field(name='Ri_param'); self.Ri_param['g'] = self.params['Ri']

        ns = {'np': np,
              'Ra': self.Ra_param, 'Pr': self.Pr_param, 'Rrho': self.Rrho_param, 'tau': self.tau_param, 'Ri': self.Ri_param,
              'ex': ex, 'ey': ey, 'ez': ez,
              'w': self.u @ ez,
              'lift': lift,
              'grad_u': grad_u, 'grad_te': grad_te, 'grad_sa': grad_sa, 
              'lap_u': lap_u, 'lap_te': lap_te, 'lap_sa': lap_sa,
              'dx': dx, 'dz': dz,
             }
        
        vars = [self.p, self.u, self.te, self.sa, 
                tau_p, tau_u1, tau_te1, tau_sa1,
                tau_u2, tau_te2, tau_sa2]
        self.ivp_problem = de.IVP(vars, namespace=ns)
        # Periodic Governing Equations
        self.ivp_problem.add_equation("trace(grad_u) + tau_p = 0")
        self.ivp_problem.add_equation("integ(p) = 0") 
        # velocity nondimensionalization by free-fall velocity
        self.ivp_problem.add_equation("dt(u) + grad(p) - np.sqrt(Pr/Ra)*lap_u - (te-sa/Rrho)*ez + lift(tau_u2) = - u@grad_u")
        self.ivp_problem.add_equation("dt(te) - (1.0/np.sqrt(Pr*Ra))*lap_te + w + lift(tau_te2) = - u@grad_te")
        self.ivp_problem.add_equation("dt(sa) - (tau/np.sqrt(Pr*Ra))*lap_sa + w + lift(tau_sa2) = - u@grad_sa")
        self.ivp_problem.add_equation("te(z='left') = 0")
        self.ivp_problem.add_equation("te(z='right') = 0")
        self.ivp_problem.add_equation("sa(z='left') = 0")
        self.ivp_problem.add_equation("sa(z='right') = 0")
        if self.params.get('stress-free', False):
            self.ivp_problem.add_equation("w(z='left') = 0")
            self.ivp_problem.add_equation("dz(ux)(z='left') = 0")
            if self.dim==3:
                self.ivp_problem.add_equation("dz(uy)(z='left') = 0")
            self.ivp_problem.add_equation("w(z='right') = 0")
            self.ivp_problem.add_equation("dz(ux)(z='right') = 0")
            if self.dim==3:
                self.ivp_problem.add_equation("dz(uy)(z='right') = 0")
        else: # no-slip
            self.ivp_problem.add_equation("u(z='left') = 0")
            self.ivp_problem.add_equation("u(z='right') = 0")
    def get_flow_properties(self):
        # ns = self._get_base_namespace()
        # ex = ns['ex']
        # w = ns['w']
        ex = self.coords.unit_vector_fields(self.dist)[0]
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        ux = self.u @ ex
        w = self.u @ ez
        Ra = self.params['Ra']
        Pr = self.params['Pr']
        Rrho = self.params['Rrho']
        tau = self.params['tau']
        #
        z, = self.dist.local_grids(self.bases[-1])
        Lz = self.bases[-1].bounds[1]
        #
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        if self.dim == 2:
            h_mean = lambda A: de.Average(A,'x')
        else:
            h_mean = lambda A: de.Average(A,('x', 'y'))
        # Heat and salt fluxes
        Jt = 1-h_mean(dz(self.te))(z=0)
        Js = 1-h_mean(dz(self.sa))(z=0)
        # Nusselt and Sherwood numbers 
        Nu = 1+h_mean(np.sqrt(Pr*Ra)*w*self.te - dz(self.te))(z=Lz/2)
        Sh = 1+h_mean(np.sqrt(Pr*Ra)/tau*w*self.sa - dz(self.sa))(z=Lz/2)
        # Energy
        grad_u = de.grad(self.u)
        D = de.Average(Pr*(de.dot(grad_u @ ex, grad_u @ ex) + de.dot(grad_u @ ez, grad_u @ ez))) # Viscous dissipation
        I = de.Average(Pr*Ra*(self.te-self.sa/Rrho)*w) # kinetic energy input by bouyancy
        I_t = de.Average(Pr*Ra*self.te*w) # kinetic energy input by temperature-induced bouyancy
        I_s = de.Average(-Pr*Ra*self.sa*w/Rrho) # kinetic energy input by salinity-induced bouyancy
        # .evaluate() returns a field object
        # ['g'] accesses the grid data
        # Heat and salt fluxes
        Jt_val = Jt.evaluate()['g'].real
        Js_val = Js.evaluate()['g'].real
        # Nusselt and Sherwood numbers 
        Nu_val = Nu.evaluate()['g'].real
        Sh_val = Sh.evaluate()['g'].real
        D_val = D.evaluate()['g'].real
        I_val = I.evaluate()['g'].real
        I_t_val = I_t.evaluate()['g'].real
        I_s_val = I_s.evaluate()['g'].real
        if self.dist.comm.rank == 0:
            return {'Jt': float(Jt_val.item()),
                    'Js': float(Js_val.item()),
                    'Nu': float(Nu_val.item()),
                    'Sh': float(Sh_val.item()),
                    'D': float(D_val.item()),
                    'I': float(I_val.item()),
                    'I_t': float(I_t_val.item()),
                    'I_s': float(I_s_val.item())}
        else:
            return {}
        
    def set_snapshots(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        snapshots = solver.evaluator.add_file_handler(self.odir+'snapshots', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        snapshots.add_task(self.te, name='te')
        snapshots.add_task(self.sa, name='sa')
        snapshots.add_task(self.u, name='u')
        
    def set_flowproperties(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        ex = self.coords.unit_vector_fields(self.dist)[0]
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        w = self.u @ ez
        Ra = self.params['Ra']
        Pr = self.params['Pr']
        Rrho = self.params['Rrho']
        tau = self.params['tau']
        z, = self.dist.local_grids(self.bases[-1])
        Lz = self.bases[-1].bounds[1]
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        if self.dim == 2:
            h_mean = lambda A: de.Average(A,'x')
        else:
            h_mean = lambda A: de.Average(A,('x', 'y'))
        flowproperties = solver.evaluator.add_file_handler(self.odir+'flowproperties', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        flowproperties.add_task(1 - h_mean(dz(self.te))(z=Lz/2), name='Jt')
        flowproperties.add_task(1 - h_mean(dz(self.sa))(z=Lz/2), name='Js')
        flowproperties.add_task(1 + h_mean(np.sqrt(Pr*Ra)*w*self.te - dz(self.te))(z=Lz/2), name='Nu') # Nusselt number
        flowproperties.add_task(1 + h_mean(np.sqrt(Pr*Ra)/tau*w*self.sa - dz(self.sa))(z=Lz/2), name='Sh') # Sherwood number
        grad_u = de.grad(self.u)
        flowproperties.add_task(de.Average(Pr*(de.dot(grad_u @ ex, grad_u @ ex) + de.dot(grad_u @ ez, grad_u @ ez))), name='D') # Viscous dissipation
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


class DiffusiveConvection(DoubleDiffusion):
    def build_problems(self):
        self.build_ivp_problem()
        self.build_evp_problem()
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
        self.Lambda_param = self.dist.Field(name='Lambda_param'); self.Lambda_param['g'] = self.params['Lambda']
        self.tau_param = self.dist.Field(name='tau_param'); self.tau_param['g'] = self.params['tau']

        ns = {'np': np,
              'Ra': self.Ra_param, 'Pr': self.Pr_param, 'Lambda': self.Lambda_param, 'tau': self.tau_param,
              'ex': ex, 'ey': ey, 'ez': ez,
              'w': self.u @ ez,
             }
        vars = [self.p, self.u, self.te, self.sa, 
                tau_p, tau_u, tau_te, tau_sa]
        self.ivp_problem = de.IVP(vars, namespace=ns)
        # Periodic Governing Equations
        self.ivp_problem.add_equation("trace(grad(u)) + tau_p = 0")
        self.ivp_problem.add_equation("integ(p) = 0") 
        # velocity nondimensionalization by thermal diffusivity
        # self.ivp_problem.add_equation("dt(u) + grad(p) - Pr*lap(u) - Pr*Ra*(te-Lambda*sa)*ez + tau_u = - u@grad(u)")
        # self.ivp_problem.add_equation("dt(te) - lap(te) - w + tau_te = - u@grad(te)")
        # self.ivp_problem.add_equation("dt(sa) - tau*lap(sa) - w + tau_sa = - u@grad(sa)")
        # velocity nondimensionalization by free-fall velocity
        self.ivp_problem.add_equation("dt(u) + grad(p) - np.sqrt(Pr/Ra)*lap(u) - (te-Lambda*sa)*ez + tau_u = - u@grad(u)")
        self.ivp_problem.add_equation("dt(te) - (1.0/np.sqrt(Pr*Ra))*lap(te) - w + tau_te = - u@grad(te)")
        self.ivp_problem.add_equation("dt(sa) - (tau/np.sqrt(Pr*Ra))*lap(sa) - w + tau_sa = - u@grad(sa)")
        # Integral constraints for floating zero-means
        for v in ['u', 'te', 'sa']:
            self.ivp_problem.add_equation(f"integ({v}) = 0")
    def build_evp_problem(self):
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
        self.Lambda_param = self.dist.Field(name='Lambda_param'); self.Lambda_param['g'] = self.params['Lambda']
        self.tau_param = self.dist.Field(name='tau_param'); self.tau_param['g'] = self.params['tau']

        ns = {'np': np,
              'Ra': self.Ra_param, 'Pr': self.Pr_param, 'Lambda': self.Lambda_param, 'tau': self.tau_param,
              'ex': ex, 'ey': ey, 'ez': ez,
              'w': self.u @ ez,
              'u_eq': self.u_eq, 'te_eq': self.te_eq, 'sa_eq': self.sa_eq,
              'sigma': self.sigma
             }
        vars = [self.p, self.u, self.te, self.sa, 
                tau_p, tau_u, tau_te, tau_sa]
        # Define EVP
        self.evp_problem = de.EVP(vars, eigenvalue=self.sigma, namespace=ns)
        # Add equations here based on the linearized physics of salt finger convection
        self.evp_problem.add_equation("trace(grad(u)) + tau_p = 0")
        self.evp_problem.add_equation("integ(p) = 0") 
        self.evp_problem.add_equation("sigma*u + grad(p) - Pr*lap(u) - Pr*Ra*(te-Lambda*sa)*ez + u@grad(u_eq)+u_eq@grad(u) + tau_u = 0")
        self.evp_problem.add_equation("sigma*te - lap(te) - w + u@grad(te_eq)+u_eq@grad(te) + tau_te = 0")
        self.evp_problem.add_equation("sigma*sa - tau*lap(sa) - w + u@grad(sa_eq)+u_eq@grad(sa) + tau_sa = 0")
        # Integral constraints
        for v in ['u', 'te', 'sa']:
            self.evp_problem.add_equation(f"integ({v}) = 0")


class ShearedDiffusiveConvection(DoubleDiffusion):
    ''' Thermohaline-shear instability in paper Radko (2019) '''
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

        grad_u = de.grad(self.u)
        grad_te = de.grad(self.te)
        grad_sa = de.grad(self.sa)
        lap_u = de.div(grad_u)
        lap_te = de.div(grad_te)
        lap_sa = de.div(grad_sa)

        dx = lambda A: de.Differentiate(A, self.coords['x']) 
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        z, = self.dist.local_grids(self.bases[-1])

        # PARAMETERS / BACKGROUND FIELDS (The Knowns)
        self.Ra_param = self.dist.Field(name='Ra_param'); self.Ra_param['g'] = self.params['Ra']
        self.Pr_param = self.dist.Field(name='Pr_param'); self.Pr_param['g'] = self.params['Pr']
        self.Lambda_param = self.dist.Field(name='Lambda_param'); self.Lambda_param['g'] = self.params['Lambda']
        self.tau_param = self.dist.Field(name='tau_param'); self.tau_param['g'] = self.params['tau']
        self.Ri_param = self.dist.Field(name='Ri_param'); self.Ri_param['g'] = self.params['Ri']

        A_s = np.sqrt((self.params['Lambda']-1.0)/self.params['Ri'])
        baru = self.dist.Field(bases=self.bases[-1]); baru['g'] = A_s*np.sin(2*np.pi*z)

        ns = {'np': np,
              'Ra': self.Ra_param, 'Pr': self.Pr_param, 'Lambda': self.Lambda_param, 'tau': self.tau_param, 'Ri': self.Ri_param,
              'ex': ex, 'ey': ey, 'ez': ez,
              'ux': self.u @ ex, 'w': self.u @ ez,
              'grad_u': grad_u, 'grad_te': grad_te, 'grad_sa': grad_sa, 
              'lap_u': lap_u, 'lap_te': lap_te, 'lap_sa': lap_sa,
              'dx': dx, 'dz': dz,
              'baru': baru,
             }
        vars = [self.p, self.u, self.te, self.sa, 
                tau_p, tau_u, tau_te, tau_sa]
        # if self.params['stokes']:
        #     vars.append(tau_u)
        self.ivp_problem = de.IVP(vars, namespace=ns)
        # Governing Equations
        self.ivp_problem.add_equation("trace(grad_u) + tau_p = 0")
        self.ivp_problem.add_equation("integ(p) = 0") 
        if self.params.get('stokes', False):
            self.ivp_problem.add_equation("grad(p) - np.sqrt(Pr/Ra)*lap_u - (te-Lambda*sa)*ez + tau_u = 0")
        else:
            self.ivp_problem.add_equation("dt(u) + baru*dx(u) + w*dz(baru)*ex + grad(p) - np.sqrt(Pr/Ra)*lap_u - (te-Lambda*sa)*ez + tau_u = - u@grad(u)")
        self.ivp_problem.add_equation("dt(te) + baru*dx(te) - (1.0/np.sqrt(Pr*Ra))*lap_te - w + tau_te = - u@grad(te)")
        self.ivp_problem.add_equation("dt(sa) + baru*dx(sa) - (tau/np.sqrt(Pr*Ra))*lap_sa - w + tau_sa = - u@grad(sa)")
        self.ivp_problem.add_equation("integ(u) = 0") 
        self.ivp_problem.add_equation("integ(te) = 0") 
        self.ivp_problem.add_equation("integ(sa) = 0") 
    def get_flow_properties(self):
        ex = self.coords.unit_vector_fields(self.dist)[0]
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        ux = self.u @ ex
        w = self.u @ ez
        Pr = self.params['Pr']
        Ra = self.params['Ra']
        Lambda = self.params['Lambda']
        tau = self.params['tau']
        Ri = self.params['Ri']
        z, = self.dist.local_grids(self.bases[-1])
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        baru = self.dist.Field(bases=self.bases[-1])
        A_s = np.sqrt((Lambda-1.0)/Ri)
        baru['g'] = A_s*np.sin(2*np.pi*z)
        # Nusselt and Sherwood numbers 
        Nu = 1 + de.Average(w*self.te)*np.sqrt(Pr*Ra)
        Sh = 1 + de.Average(w*self.sa)*np.sqrt(Pr*Ra)/tau
        grad_u = de.grad(self.u)
        # D = D_base + D_fluc
        # D_base = np.sqrt(Pr/Ra) * 2 * (np.pi**2) * (A_s**2)
        # save D_fluc
        D = np.sqrt(Pr/Ra) * de.Average(de.dot(grad_u @ ex, grad_u @ ex) + de.dot(grad_u @ ez, grad_u @ ez)) # Viscous dissipation, TODO: complete 3D calculation
        I = de.Average((self.te-Lambda*self.sa)*w) # kinetic energy input by bouyancy
        I_t = de.Average(self.te*w) # kinetic energy input by temperature-induced bouyancy
        I_s = de.Average(-Lambda*self.sa*w) # kinetic energy input by salinity-induced bouyancy
        I_shear = de.Average(-ux * w * dz(baru)) # kinetic energy input by background shear
                
        # .evaluate() returns a field object
        # ['g'] accesses the grid data
        # Nusselt and Sherwood numbers 
        Nu_val = Nu.evaluate()['g'].real
        Sh_val = Sh.evaluate()['g'].real
        D_val = D.evaluate()['g'].real
        I_val = I.evaluate()['g'].real
        I_t_val = I_t.evaluate()['g'].real
        I_s_val = I_s.evaluate()['g'].real
        I_shear_val = I_shear.evaluate()['g'].real
        if self.dist.comm.rank == 0:
            return {'Nu': float(Nu_val.item()),
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
        Ra = self.params['Ra']
        Pr = self.params['Pr']
        Lambda = self.params['Lambda']
        tau = self.params['tau']
        Ri = self.params['Ri']
        z, = self.dist.local_grids(self.bases[-1])
        Lz = self.bases[-1].bounds[1]
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        baru = self.dist.Field(bases=self.bases[-1])
        A_s = np.sqrt((Lambda-1.0)/Ri)
        baru['g'] = A_s*np.sin(2*np.pi*z)
        if self.dim == 2:
            h_mean = lambda A: de.Average(A,'x')
        else:
            h_mean = lambda A: de.Average(A,('x', 'y'))
        flowproperties = solver.evaluator.add_file_handler(self.odir+'flowproperties', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        flowproperties.add_task(1+np.sqrt(Pr*Ra)*de.Average(w*self.te), name='Nu') # Nusselt number
        flowproperties.add_task(1+np.sqrt(Pr*Ra)/tau*de.Average(w*self.sa), name='Sh') # Sherwood number
        grad_u = de.grad(self.u)
        # D = D_base + D_fluc
        # D_base = np.sqrt(Pr/Ra) * 2 * (np.pi**2) * (A_s**2)
        # save D_fluc
        flowproperties.add_task(np.sqrt(Pr/Ra) * de.Average(de.dot(grad_u @ ex, grad_u @ ex) + de.dot(grad_u @ ez, grad_u @ ez)), name='D') # Viscous dissipation
        flowproperties.add_task(de.Average(-ux * w * dz(baru)), name='I_shear') # kinetic energy input by background shear
        flowproperties.add_task(de.Average((self.te-Lambda*self.sa)*w), name='I') # kinetic energy input by bouyancy
        flowproperties.add_task(de.Average(self.te*w), name='I_t') # kinetic energy input by temperature-induced bouyancy
        flowproperties.add_task(de.Average(-Lambda*self.sa*w), name='I_s') # kinetic energy input by salinity-induced bouyancy

    def set_meanprofiles(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        if self.dim == 2:
            h_mean = lambda A: de.Average(A,'x')
        else:
            h_mean = lambda A: de.Average(A,('x', 'y'))
        meanprofiles = solver.evaluator.add_file_handler(self.odir+'meanprofiles', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        meanprofiles.add_task(h_mean(self.u), name='u') # horizontal averaged x-axis velocity
        meanprofiles.add_task(h_mean(self.te), name='te') # horizontal averaged temperature
        meanprofiles.add_task(h_mean(self.sa), name='sa') # horizontal averaged sanility


class WallShearedDiffusiveConvection(DoubleDiffusion):
    def build_problems(self):
        self.build_ivp_problem()
    def build_ivp_problem(self):
        tau_p = self.dist.Field(name='tau_p')
        tau_u1 = self.dist.VectorField(self.coords, name='tau_u1', bases=self.bases[:-1])
        tau_te1 = self.dist.Field(name='tau_te1', bases=self.bases[:-1])
        tau_sa1 = self.dist.Field(name='tau_sa1', bases=self.bases[:-1])
        tau_u2 = self.dist.VectorField(self.coords, name='tau_u2', bases=self.bases[:-1])
        tau_te2 = self.dist.Field(name='tau_te2', bases=self.bases[:-1])
        tau_sa2 = self.dist.Field(name='tau_sa2', bases=self.bases[:-1])

        unit_vectors = self.coords.unit_vector_fields(self.dist)
        if self.dim == 2:
            ex, ez = unit_vectors
            ey = None # Or a zero-field if needed
        else:
            ex, ey, ez = unit_vectors

        lift_basis = self.bases[-1].derivative_basis(1)
        lift = lambda A: de.Lift(A, lift_basis, -1)

        grad_u = de.grad(self.u) + ez*lift(tau_u1) 
        grad_te = de.grad(self.te) + ez*lift(tau_te1) 
        grad_sa = de.grad(self.sa) + ez*lift(tau_sa1) 
        lap_u = de.div(grad_u)
        lap_te = de.div(grad_te)
        lap_sa = de.div(grad_sa)

        dx = lambda A: de.Differentiate(A, self.coords['x']) 
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        z, = self.dist.local_grids(self.bases[-1])
        Lz = self.bases[-1].bounds[1]
    
        # PARAMETERS / BACKGROUND FIELDS (The Knowns)
        self.Ra_param = self.dist.Field(name='Ra_param'); self.Ra_param['g'] = self.params['Ra']
        self.Ri_param = self.dist.Field(name='Ri_param'); self.Ri_param['g'] = self.params['Ri']
        self.Pr_param = self.dist.Field(name='Pr_param'); self.Pr_param['g'] = self.params['Pr']
        self.Lambda_param = self.dist.Field(name='Lambda_param'); self.Lambda_param['g'] = self.params['Lambda']
        self.tau_param = self.dist.Field(name='tau_param'); self.tau_param['g'] = self.params['tau']

        Uw = 1.0/np.sqrt(self.params['Ri'])
        baru = self.dist.Field(bases=self.bases[-1]); baru['g'] = (z-Lz/2)*Uw

        ns = {'np': np,
              'Ra': self.Ra_param, 'Pr': self.Pr_param, 'Lambda': self.Lambda_param, 'tau': self.tau_param, 'Ri': self.Ri_param,
              'ex': ex, 'ey': ey, 'ez': ez,
              'ux': self.u @ ex, 'w': self.u @ ez,
              'lift': lift,
              'grad_u': grad_u, 'grad_te': grad_te, 'grad_sa': grad_sa, 
              'lap_u': lap_u, 'lap_te': lap_te, 'lap_sa': lap_sa,
              'dx': dx, 'dz': dz,
              'baru': baru,
             }
        vars = [self.p, self.u, self.te, self.sa, 
                tau_p, tau_u1, tau_te1, tau_sa1,
                tau_u2, tau_te2, tau_sa2]
        self.ivp_problem = de.IVP(vars, namespace=ns)
        # Periodic Governing Equations
        self.ivp_problem.add_equation("trace(grad_u) + tau_p = 0")
        self.ivp_problem.add_equation("integ(p) = 0") 
        # velocity nondimensionalization by free-fall velocity
        self.ivp_problem.add_equation("dt(u) + baru*dx(u) + w*dz(baru)*ex + grad(p) - np.sqrt(Pr/Ra)*lap_u - (te-Lambda*sa)*ez + lift(tau_u2) = - u@grad_u")
        self.ivp_problem.add_equation("dt(te) + baru*dx(te) - (1.0/np.sqrt(Pr*Ra))*lap_te - w + lift(tau_te2) = - u@grad_te")
        self.ivp_problem.add_equation("dt(sa) + baru*dx(sa) - (tau/np.sqrt(Pr*Ra))*lap_sa - w + lift(tau_sa2) = - u@grad_sa")
        self.ivp_problem.add_equation("te(z='left') = 0")
        self.ivp_problem.add_equation("te(z='right') = 0")
        self.ivp_problem.add_equation("sa(z='left') = 0")
        self.ivp_problem.add_equation("sa(z='right') = 0")
        if self.params['stress-free']:
            self.ivp_problem.add_equation("w(z='left') = 0")
            self.ivp_problem.add_equation("dz(ux)(z='left') = 0")
            if self.dim==3:
                self.ivp_problem.add_equation("dz(uy)(z='left') = 0")
            self.ivp_problem.add_equation("w(z='right') = 0")
            self.ivp_problem.add_equation("dz(ux)(z='right') = 0")
            if self.dim==3:
                self.ivp_problem.add_equation("dz(uy)(z='right') = 0")
        else: # no-slip
            self.ivp_problem.add_equation("u(z='left') = 0")
            self.ivp_problem.add_equation("u(z='right') = 0")
    def get_flow_properties(self):
        ex = self.coords.unit_vector_fields(self.dist)[0]
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        ux = self.u @ ex
        w = self.u @ ez
        Ra = self.params['Ra']
        Pr = self.params['Pr']
        Lambda = self.params['Lambda']
        tau = self.params['tau']
        Ri = self.params['Ri']
        #
        z, = self.dist.local_grids(self.bases[-1])
        Lz = self.bases[-1].bounds[1]
        Uw = 1.0/np.sqrt(Ri)
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        if self.dim == 2:
            h_mean = lambda A: de.Average(A,'x')
        else:
            h_mean = lambda A: de.Average(A,('x', 'y'))
        # Heat and salt fluxes
        Jt = 1.0-h_mean(dz(self.te))(z=0)
        Js = 1.0-h_mean(dz(self.sa))(z=0)
        # Nusselt and Sherwood numbers 
        Nu = 1+h_mean(np.sqrt(Pr*Ra)*w*self.te - dz(self.te))(z=Lz/2)
        Sh = 1+h_mean(np.sqrt(Pr*Ra)/tau*w*self.sa - dz(self.sa))(z=Lz/2)
        grad_u = de.grad(self.u)
        D = np.sqrt(Pr/Ra) * (1 + de.Average(de.dot(grad_u @ ex, grad_u @ ex) + de.dot(grad_u @ ez, grad_u @ ez))) # Viscous dissipation, TODO: complete 3D calculation
        I = de.Average((self.te-Lambda*self.sa)*w) # kinetic energy input by bouyancy
        I_t = de.Average(self.te*w) # kinetic energy input by temperature-induced bouyancy
        I_s = de.Average(-Lambda*self.sa*w) # kinetic energy input by salinity-induced bouyancy
        tau_w = 1+0.5*(h_mean(dz(ux))(z=0)+h_mean(dz(ux))(z=Lz)) # wall-shear stress
        I_w = np.sqrt(Pr/Ra)*tau_w*Uw # kinetic energy input by wall-shear stress
        # .evaluate() returns a field object
        # ['g'] accesses the grid data
        # Heat and salt fluxes at wall
        Jt_val = Jt.evaluate()['g'].real
        Js_val = Js.evaluate()['g'].real
        # Nusselt and Sherwood numbers at mid-plane
        Nu_val = Nu.evaluate()['g'].real
        Sh_val = Sh.evaluate()['g'].real
        D_val = D.evaluate()['g'].real
        I_val = I.evaluate()['g'].real
        I_t_val = I_t.evaluate()['g'].real
        I_s_val = I_s.evaluate()['g'].real
        I_w_val = I_w.evaluate()['g'].real
        if self.dist.comm.rank == 0:
            return {'Jt': float(Jt_val.item()),
                    'Js': float(Js_val.item()),
                    'Nu': float(Nu_val.item()),
                    'Sh': float(Sh_val.item()),
                    'D': float(D_val.item()),
                    'I_w': float(I_w_val.item()),
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
        Ra = self.params['Ra']
        Pr = self.params['Pr']
        Lambda = self.params['Lambda']
        tau = self.params['tau']
        Ri = self.params['Ri']
        z, = self.dist.local_grids(self.bases[-1])
        Lz = self.bases[-1].bounds[1]
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        Uw = 1.0/np.sqrt(Ri)
        if self.dim == 2:
            h_mean = lambda A: de.Average(A,'x')
        else:
            h_mean = lambda A: de.Average(A,('x', 'y'))
        flowproperties = solver.evaluator.add_file_handler(self.odir+'flowproperties', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        flowproperties.add_task(1-h_mean(dz(self.te))(z=0), name='Jt') # heat flux at wall
        flowproperties.add_task(1-h_mean(dz(self.sa))(z=0), name='Js') # mass flux at wall
        flowproperties.add_task(1+h_mean(np.sqrt(Pr*Ra)*w*self.te - dz(self.te))(z=Lz/2), name='Nu') # Nusselt number
        flowproperties.add_task(1+h_mean(np.sqrt(Pr*Ra)/tau*w*self.sa - dz(self.sa))(z=Lz/2), name='Sh') # Sherwood number
        grad_u = de.grad(self.u)
        flowproperties.add_task(np.sqrt(Pr/Ra) * (1 + de.Average(de.dot(grad_u @ ex, grad_u @ ex) + de.dot(grad_u @ ez, grad_u @ ez))), name='D') # Viscous dissipation
        tau_w = 1+0.5*(h_mean(dz(ux))(z=0)+h_mean(dz(ux))(z=Lz)) # wall-shear stress
        I_w = np.sqrt(Pr/Ra)*tau_w*Uw # kinetic energy input by wall-shear stress
        flowproperties.add_task(tau_w, name='tau_w') # wall-shear stress
        flowproperties.add_task(I_w, name='I_w') # kinetic energy input by wall-shear stress
        flowproperties.add_task(de.Average((self.te-Lambda*self.sa)*w), name='I') # kinetic energy input by bouyancy
        flowproperties.add_task(de.Average(self.te*w), name='I_t') # kinetic energy input by temperature-induced bouyancy
        flowproperties.add_task(de.Average(-Lambda*self.sa*w), name='I_s') # kinetic energy input by salinity-induced bouyancy

    def set_meanprofiles(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        if self.dim == 2:
            h_mean = lambda A: de.Average(A,'x')
        else:
            h_mean = lambda A: de.Average(A,('x', 'y'))
        meanprofiles = solver.evaluator.add_file_handler(self.odir+'meanprofiles', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        meanprofiles.add_task(h_mean(self.u), name='u') # horizontal averaged x-axis velocity
        meanprofiles.add_task(h_mean(self.te), name='te') # horizontal averaged temperature
        meanprofiles.add_task(h_mean(self.sa), name='sa') # horizontal averaged sanility
