import dedalus.public as de
import numpy as np
from mpi4py import MPI
import h5py
import matplotlib.pyplot as plt
from .base import FluidModel 

class DoubleDiffusion(FluidModel):
    def __init__(self, params, sizes, bounds, bounded=False, mode='ecs', dealias=3/2):
        # Call the FluidModel __init__ first
        super().__init__(params, sizes, bounds, bounded, mode, dealias)
        # Add additional fields
        self.p = None
        self.u = None
        self.te = None
        self.sa = None
        self.u_eq = None
        self.te_eq = None
        self.sa_eq = None
        self.create_domain()
        self.build_fields()

    def create_domain(self):
        """ Creates a 2D or 3D domain/bases in the model """
        if self.dim == 2:
            Nx, Nz = self.sizes
            Lx, Lz = self.bounds
        elif self.dim == 3:
            Nx, Ny, Nz = self.sizes
            Lx, Ly, Lz = self.bounds
        else:
            raise ValueError("Sizes and bounds must be length 2 or 3.")
        MPI.COMM_WORLD.Barrier()
        # Only instantiate CartesianCoordinates and Distributor ONCE
        # Re-creating Distributor on every parameter evaluation leaks MPI communicators.
        if self.dist is None:
            if self.dim == 2:
                self.coords = de.CartesianCoordinates('x', 'z')
            else:
                self.coords = de.CartesianCoordinates('x', 'y', 'z')
            dtype = np.float64 if self.mode == 'sim' else np.complex128
            self.dist = de.Distributor(self.coords, dtype=dtype)

        # Horizontal x-basis (Always periodic)
        self.x_basis = None
        if self.mode=='sim':
            self.x_basis = de.RealFourier(self.coords['x'], size=Nx, bounds=(0, Lx), dealias=self.dealias)
        else:
            self.x_basis = de.ComplexFourier(self.coords['x'], size=Nx, bounds=(0, Lx), dealias=self.dealias)
        
        # Only if 3D, Always periodic
        if self.dim == 3:
            self.y_basis = None
            if self.mode=='sim':
                self.y_basis = de.RealFourier(self.coords['y'], size=Ny, bounds=(0, Ly), dealias=self.dealias)
            else:
                self.y_basis = de.ComplexFourier(self.coords['y'], size=Ny, bounds=(0, Ly), dealias=self.dealias)

        self.z_basis = None
        if self.bounded:
            # Use Chebyshev for bounded domains
            self.z_basis = de.ChebyshevT(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)
        else:
            # Use Fourier for fully periodic domains
            if self.mode=='sim':
                self.z_basis = de.RealFourier(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)
            else:
                self.z_basis = de.ComplexFourier(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)

        self.bases = None
        if self.dim == 2:
            self.bases = (self.x_basis, self.z_basis)
        else:
            self.bases = (self.x_basis, self.y_basis, self.z_basis)

    def build_fields(self):
        # pressure p (scalar)
        self.p = self.dist.Field(name='p', bases=self.bases)
        # velocity u (Vector)
        self.u = self.dist.VectorField(self.coords, name='u', bases=self.bases)
        # self.u.change_scales(self.dealias)
        self.u_eq = self.dist.VectorField(self.coords, name='u_eq', bases=self.bases)
        # temperature \theta (scalar)
        self.te = self.dist.Field(name='te', bases=self.bases)
        # self.te.change_scales(self.dealias)
        self.te_eq = self.dist.Field(name='te_eq', bases=self.bases)
        # salinity s (scalar)
        self.sa = self.dist.Field(name='sa', bases=self.bases)
        # self.sa.change_scales(self.dealias)
        self.sa_eq = self.dist.Field(name='sa_eq', bases=self.bases)
        # re-collect fields to field list
        self.fields = [self.u, self.te, self.sa]
        self.eq_fields = [self.u_eq, self.te_eq, self.sa_eq]

    def preview(self, field_idx=-1):
        """ Preview the current state using last field of the system. """
        data_g = self.fields[field_idx].allgather_data('g').real
        if self.dist.comm.rank == 0:
            xaxis = self.bases[0].global_grid(self.dist, scale=self.dealias)
            zaxis = self.bases[-1].global_grid(self.dist, scale=self.dealias)
            if self.dim == 3:
                data_g = data_g[:,0,:] # get a 2D slice
            # Initialize the figure only once
            if self.preview_fig is None:
                plt.ion()  # Turn on interactive mode
                self.preview_fig, self.preview_ax = plt.subplots(figsize=(4,3))
                self.preview_im = self.preview_ax.pcolormesh(xaxis.ravel(), zaxis.ravel(), data_g.T, 
                                             cmap='RdBu_r', shading='auto')
                self.preview_ax.set_xlabel('x')
                self.preview_ax.set_ylabel('z')
                self.preview_fig.colorbar(self.preview_im)
                # self.preview_ax.set_title("Salt Concentration") 
                self.preview_fig.canvas.draw()
                self.preview_fig.canvas.flush_events()  
            else:
                self.preview_im.set_array(data_g.T.ravel())
                v_min, v_max = np.min(data_g), np.max(data_g)
                self.preview_im.set_clim(vmin=v_min, vmax=v_max)
                # self.preview_ax.set_title(f"Salt Concentration at time {self.sim_time:.2f}")
                self.preview_fig.canvas.draw()
                self.preview_fig.canvas.flush_events()  

########################################################################################
########################################################################################
########################################################################################
########################################################################################
class SaltFinger(DoubleDiffusion):
    def build_problems(self):
        self.build_ivp_problem()
        # self.build_evp_problem()
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
    def build_evp_problem(self):
        sigma = self.dist.Field(name='sigma')
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
              'u_eq': self.u_eq, 'te_eq': self.te_eq, 'sa_eq': self.sa_eq,
              'sigma': sigma
             }
        vars = [self.p, self.u, self.te, self.sa, 
                tau_p, tau_u, tau_te, tau_sa]
        # Define EVP
        self.evp_problem = de.EVP(vars, eigenvalue=sigma, namespace=ns)
        # Add equations here based on the linearized physics of salt finger convection
        self.evp_problem.add_equation("trace(grad(u)) + tau_p = 0")
        self.evp_problem.add_equation("integ(p) = 0") 
        self.evp_problem.add_equation("sigma*u + grad(p) - Pr*lap(u) - Pr*Ra*(te-sa/Rrho)*ez + u@grad(u_eq)+u_eq@grad(u) + tau_u = 0")
        self.evp_problem.add_equation("sigma*te - lap(te) + w + u@grad(te_eq)+u_eq@grad(te) + tau_te = 0")
        self.evp_problem.add_equation("sigma*sa - tau*lap(sa) + w + u@grad(sa_eq)+u_eq@grad(sa) + tau_sa = 0")
        # Integral constraints
        for v in ['u', 'te', 'sa']:
            self.evp_problem.add_equation(f"integ({v}) = 0")
    def get_flow_properties(self):
        ex = self.coords.unit_vector_fields(self.dist)[0]
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
        D = de.Average(Pr*(de.dot(grad_u @ ex, grad_u @ ex) + de.dot(grad_u @ ez, grad_u @ ez))) # Viscous dissipation
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
        ex = self.coords.unit_vector_fields(self.dist)[0]
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
        