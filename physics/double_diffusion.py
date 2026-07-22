import dedalus.public as de
import numpy as np
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
            self.coords = de.CartesianCoordinates('x', 'z')
        elif self.dim == 3:
            Nx, Ny, Nz = self.sizes
            Lx, Ly, Lz = self.bounds
            self.coords = de.CartesianCoordinates('x', 'y', 'z')
        else:
            raise ValueError("Sizes and bounds must be length 2 or 3.")
        
        if self.mode=='sim':
            self.dist = de.Distributor(self.coords, dtype=np.float64)
        else:
            self.dist = de.Distributor(self.coords, dtype=np.complex128)

        # Horizontal x-basis (Always periodic)
        if self.mode=='sim':
            x_basis = de.RealFourier(self.coords['x'], size=Nx, bounds=(0, Lx), dealias=self.dealias)
        else:
            x_basis = de.ComplexFourier(self.coords['x'], size=Nx, bounds=(0, Lx), dealias=self.dealias)
        
        # Only if 3D, Always periodic
        if self.dim == 3:
            if self.mode=='sim':
                y_basis = de.RealFourier(self.coords['y'], size=Ny, bounds=(0, Ly), dealias=self.dealias)
            else:
                y_basis = de.ComplexFourier(self.coords['y'], size=Ny, bounds=(0, Ly), dealias=self.dealias)

        if self.bounded:
            # Use Chebyshev for bounded domains
            z_basis = de.ChebyshevT(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)
        else:
            # Use Fourier for fully periodic domains
            if self.mode=='sim':
                z_basis = de.RealFourier(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)
            else:
                z_basis = de.ComplexFourier(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)
        
        if self.dim == 2:
            self.bases = (x_basis, z_basis)
        else:
            self.bases = (x_basis, y_basis, z_basis)

    def build_fields(self):
        # pressure p (scalar)
        self.p = self.dist.Field(name='p', bases=self.bases)
        # velocity u (Vector)
        self.u = self.dist.VectorField(self.coords, name='u', bases=self.bases)
        self.u.change_scales(self.dealias)
        self.u_eq = self.dist.VectorField(self.coords, name='u_eq', bases=self.bases)
        # temperature \theta (scalar)
        self.te = self.dist.Field(name='te', bases=self.bases)
        self.te.change_scales(self.dealias)
        self.te_eq = self.dist.Field(name='te_eq', bases=self.bases)
        # salinity s (scalar)
        self.sa = self.dist.Field(name='sa', bases=self.bases)
        self.sa.change_scales(self.dealias)
        self.sa_eq = self.dist.Field(name='sa_eq', bases=self.bases)
        # re-collect fields to field list
        self.fields = [self.u, self.te, self.sa]
        self.eq_fields = [self.u_eq, self.te_eq, self.sa_eq]

    def preview(self):
        """ Preview the current state using last field of the system. """
        data_g = self.fields[-1].allgather_data('g').real
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
                       
    # def preview3D(self):
    #     """ Preview the current state using last field in 3D using isosurfaces. """
    #     if self.dim == 3:
    #         from mpl_toolkits.mplot3d import art3d
    #         # pip install scikit-image
    #         from skimage import measure # For Marching Cubes (isosurface)
    
    #         # Get the last field (usually Salinity or Temperature)
    #         data_g = self.fields[-1].allgather_data('g').real
            
    #         if self.dist.comm.rank == 0:
    #             # Get 1D axis arrays for the grid
    #             xg = self.bases[0].global_grid(self.dist, scale=self.dealias).ravel()
    #             yg = self.bases[1].global_grid(self.dist, scale=self.dealias).ravel()
    #             zg = self.bases[2].global_grid(self.dist, scale=self.dealias).ravel()
                
    #             # 1. Initialize Figure
    #             if self.preview_fig is None:
    #                 plt.ion()
    #                 self.preview_fig = plt.figure(figsize=(6, 5))
    #                 self.preview_ax = self.preview_fig.add_subplot(111, projection='3d')
    #                 self.preview_ax.set_xlabel('x')
    #                 self.preview_ax.set_ylabel('y')
    #                 self.preview_ax.set_zlabel('z')
    #             else:
    #                 self.preview_ax.clear() # Clear the previous frame
    #                 self.preview_ax.set_xlabel('x')
    #                 self.preview_ax.set_ylabel('y')
    #                 self.preview_ax.set_zlabel('z')

    #             # 2. Generate Isosurface using Marching Cubes
    #             # Choose a level (e.g., the mean of the field)
    #             level = (np.max(data_g) + np.min(data_g)) / 2
                
    #             try:
    #                 # verts: coordinates of vertices, faces: triangles
    #                 verts, faces, normals, values = measure.marching_cubes(data_g, level=level)
                    
    #                 # Scale vertices from index-space to physical-space
    #                 # Indices are (i, j, k) corresponding to (x, y, z)
    #                 verts[:, 0] = verts[:, 0] * (xg[1] - xg[0]) + xg[0]
    #                 verts[:, 1] = verts[:, 1] * (yg[1] - yg[0]) + yg[0]
    #                 verts[:, 2] = verts[:, 2] * (zg[1] - zg[0]) + zg[0]

    #                 # 3. Create a 3D PolyCollection (the mesh)
    #                 mesh = art3d.Poly3DCollection(verts[faces])
    #                 mesh.set_edgecolor('none')
    #                 mesh.set_alpha(0.6)
    #                 mesh.set_facecolor('royalblue')
                    
    #                 self.preview_ax.add_collection3d(mesh)
                    
    #                 # Set limits based on domain
    #                 self.preview_ax.set_xlim(xg.min(), xg.max())
    #                 self.preview_ax.set_ylim(yg.min(), yg.max())
    #                 self.preview_ax.set_zlim(zg.min(), zg.max())
                    
    #             except (ValueError, RuntimeError):
    #                 # Fallback if the field is uniform or level is outside range
    #                 self.preview_ax.text(0.5, 0.5, 0.5, "Field uniform - No surface", transform=self.preview_ax.transAxes)

    #             self.preview_fig.canvas.draw()
    #             self.preview_fig.canvas.flush_events()
    
class SaltFinger(DoubleDiffusion):
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
        
        ns = {'np': np,
              'Ra':self.params['Ra'],
              'Pr':self.params['Pr'],
              'Rrho':self.params['Rrho'],
              'tau':self.params['tau'],
              'ex': ex, 'ey': ey, 'ez': ez,
              'w': self.u @ ez,
             }
        if self.dist.comm.rank == 0:
            print(ns)
        vars = [self.p, self.u, self.te, self.sa, 
                tau_p, tau_u, tau_te, tau_sa]
        self.ivp_problem = de.IVP(vars, namespace=ns)
        # Periodic Governing Equations
        self.ivp_problem.add_equation("trace(grad(u)) + tau_p = 0")
        self.ivp_problem.add_equation("integ(p) = 0") 
        self.ivp_problem.add_equation("dt(u) + grad(p) - Pr*lap(u) - Pr*Ra*(te-sa/Rrho)*ez + tau_u = - u@grad(u)")
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
        ns = {'np': np,
              'Ra':self.params['Ra'],
              'Pr':self.params['Pr'],
              'Rrho':self.params['Rrho'],
              'tau':self.params['tau'],
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
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        w = self.u @ ez
        # Heat and salt fluxes
        Ft = de.Average(w*self.te)
        Fs = de.Average(w*self.sa)/self.params['tau']
        # Nusselt and Sherwood numbers 
        Nu = 1 - Ft
        Sh = 1 - Fs
        # .evaluate() returns a field object
        # ['g'] accesses the grid data
        # Nusselt and Sherwood numbers 
        Nu_val = Nu.evaluate()['g'].real
        Sh_val = Sh.evaluate()['g'].real
        if self.dist.comm.rank == 0:
            return {'Nu': float(Nu_val),
                    'Sh': float(Sh_val)}
        else:
            return None
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

        baru = self.dist.Field(bases=self.bases[-1])
        ns = {'np': np,
              'Ra':self.params['Ra'],
              'Pr':self.params['Pr'],
              'Rrho':self.params['Rrho'],
              'tau':self.params['tau'],
              'Ri':self.params['Ri'],
              'ex': ex, 'ey': ey, 'ez': ez,
              'w': self.u @ ez,
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
        self.ivp_problem.add_equation("dt(u) + grad(p) - np.sqrt(Pr/Ra)*lap_u - (te-sa/Rrho)*ez + lift(tau_u2) = - u@grad_u")
        self.ivp_problem.add_equation("dt(te) - (1.0/np.sqrt(Pr*Ra))*lap_te + w + lift(tau_te2) = - u@grad_te")
        self.ivp_problem.add_equation("dt(sa) - (tau/np.sqrt(Pr*Ra))*lap_sa + w + lift(tau_sa2) = - u@grad_sa")
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
        # ns = self._get_base_namespace()
        # ex = ns['ex']
        # w = ns['w']
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        w = self.u @ ez
        Ra = self.params['Ra']
        Pr = self.params['Pr']
        tau = self.params['tau']
        #
        z, = self.dist.local_grids(self.bases[-1])
        Lz = self.bases[-1].bounds[1]
        #
        # baru = self.dist.Field(bases=(self.bases[-1])) # base flow
        barT = self.dist.Field(bases=(self.bases[-1])) # base state of temperature: -y
        barS = self.dist.Field(bases=(self.bases[-1])) # base state of temperature: -y
        barT['g'] = -z
        barS['g'] = -z
        totT = barT + self.te
        totS = barS + self.sa
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        h_mean = lambda A: de.Average(A,'x')
        # Heat and salt fluxes
        Jt = -h_mean(dz(totT))(z=0)
        Js = -h_mean(dz(totS))(z=0)
        # Nusselt and Sherwood numbers 
        Nu = h_mean(np.sqrt(Pr*Ra)*w*totT - dz(totT))(z=Lz/2)
        Sh = h_mean(np.sqrt(Pr*Ra)/tau*w*totS - dz(totS))(z=Lz/2)
        # .evaluate() returns a field object
        # ['g'] accesses the grid data
        # Heat and salt fluxes
        Jt_val = Jt.evaluate()['g'].real
        Js_val = Js.evaluate()['g'].real
        # Nusselt and Sherwood numbers 
        Nu_val = Nu.evaluate()['g'].real
        Sh_val = Sh.evaluate()['g'].real
        if self.dist.comm.rank == 0:
            return {'Jt': float(Jt_val),
                    'Js': float(Js_val),
                    'Nu': float(Nu_val),
                    'Sh': float(Sh_val)}
        else:
            return None
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

        ns = {'np': np,
              'Ra':self.params['Ra'],
              'Pr':self.params['Pr'],
              'Lambda':self.params['Lambda'],
              'tau':self.params['tau'],
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

        ns = {'np': np,
              'Ra':self.params['Ra'],
              'Pr':self.params['Pr'],
              'Lambda':self.params['Lambda'],
              'tau':self.params['tau'],
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

        baru = self.dist.Field(bases=self.bases[-1])
        A_s = np.sqrt((self.params['Lambda']-1.0)/self.params['Ri'])
        
        z, = self.dist.local_grids(self.bases[-1])
        baru['g'] = A_s*np.sin(2*np.pi*z)

        ns = {'np': np,
              'Ra':self.params['Ra'],
              'Pr':self.params['Pr'],
              'Lambda':self.params['Lambda'],
              'tau':self.params['tau'],
              'Ri':self.params['Ri'],
              'ex': ex, 'ey': ey, 'ez': ez,
              'ux': self.u @ ex,
              'w': self.u @ ez,
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
        if 'stokes' in self.params:
            if self.params['stokes']:
                self.ivp_problem.add_equation("grad(p) - np.sqrt(Pr/Ra)*lap_u - (te-Lambda*sa)*ez + tau_u = 0")
            else:
                self.ivp_problem.add_equation("dt(u) + baru*dx(u) + w*dz(baru)*ex + grad(p) - np.sqrt(Pr/Ra)*lap_u - (te-Lambda*sa)*ez + tau_u = - u@grad(u)")
        else:
            self.ivp_problem.add_equation("dt(u) + baru*dx(u) + w*dz(baru)*ex + grad(p) - np.sqrt(Pr/Ra)*lap_u - (te-Lambda*sa)*ez + tau_u = - u@grad(u)")
        self.ivp_problem.add_equation("dt(te) + baru*dx(te) - (1.0/np.sqrt(Pr*Ra))*lap_te - w + tau_te = - u@grad(te)")
        self.ivp_problem.add_equation("dt(sa) + baru*dx(sa) - (tau/np.sqrt(Pr*Ra))*lap_sa - w + tau_sa = - u@grad(sa)")
        self.ivp_problem.add_equation("integ(u) = 0") 
        self.ivp_problem.add_equation("integ(te) = 0") 
        self.ivp_problem.add_equation("integ(sa) = 0") 
    def get_flow_properties(self):
        return None
        # ez = self.coords.unit_vector_fields(self.dist)[-1]
        # w = self.u @ ez
        # Ra = self.params['Ra']
        # Pr = self.params['Pr']
        # tau = self.params['tau']
        # #
        # z, = self.dist.local_grids(self.bases[-1])
        # Lz = self.bases[-1].bounds[1]
        # #
        # # baru = self.dist.Field(bases=(self.bases[-1])) # base flow
        # barT = self.dist.Field(bases=(self.bases[-1])) # base state of temperature: -y
        # barS = self.dist.Field(bases=(self.bases[-1])) # base state of temperature: -y
        # barT['g'] = -z
        # barS['g'] = -z
        # totT = barT + self.te
        # totS = barS + self.sa
        # dz = lambda A: de.Differentiate(A, self.coords['z']) 
        # h_mean = lambda A: de.Average(A,'x')
        # # Heat and salt fluxes
        # Jt = -h_mean(dz(totT))(z=0)
        # Js = -h_mean(dz(totS))(z=0)
        # # Nusselt and Sherwood numbers 
        # Nu = h_mean(np.sqrt(Pr*Ra)*w*totT - dz(totT))(z=Lz/2)
        # Sh = h_mean(np.sqrt(Pr*Ra)/tau*w*totS - dz(totS))(z=Lz/2)
        # # .evaluate() returns a field object
        # # ['g'] accesses the grid data
        # # Heat and salt fluxes
        # Jt_val = Jt.evaluate()['g'].real
        # Js_val = Js.evaluate()['g'].real
        # # Nusselt and Sherwood numbers 
        # Nu_val = Nu.evaluate()['g'].real
        # Sh_val = Sh.evaluate()['g'].real
        # if self.dist.comm.rank == 0:
        #     return {'Jt': float(Jt_val),
        #             'Js': float(Js_val),
        #             'Nu': float(Nu_val),
        #             'Sh': float(Sh_val)}
        # else:
        #     return None
class ShearedBoundedDiffusiveConvection(DoubleDiffusion):
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

        baru = self.dist.Field(bases=self.bases[-1])
        Uw = 1.0/np.sqrt(self.params['Ri'])
        
        z, = self.dist.local_grids(self.bases[-1])
        Lz = self.bases[-1].bounds[1]
        baru['g'] = (z-Lz/2)*Uw

        ns = {'np': np,
              'Ra':self.params['Ra'],
              'Pr':self.params['Pr'],
              'Lambda':self.params['Lambda'],
              'tau':self.params['tau'],
              'Ri':self.params['Ri'],
              'ex': ex, 'ey': ey, 'ez': ez,
              'ux': self.u @ ex,
              'w': self.u @ ez,
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
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        w = self.u @ ez
        Ra = self.params['Ra']
        Pr = self.params['Pr']
        tau = self.params['tau']
        #
        z, = self.dist.local_grids(self.bases[-1])
        Lz = self.bases[-1].bounds[1]
        #
        # baru = self.dist.Field(bases=(self.bases[-1])) # base flow
        barT = self.dist.Field(bases=(self.bases[-1])) # base state of temperature: -y
        barS = self.dist.Field(bases=(self.bases[-1])) # base state of temperature: -y
        barT['g'] = -z
        barS['g'] = -z
        totT = barT + self.te
        totS = barS + self.sa
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        h_mean = lambda A: de.Average(A,'x')
        # Heat and salt fluxes
        Jt = -h_mean(dz(totT))(z=0)
        Js = -h_mean(dz(totS))(z=0)
        # Nusselt and Sherwood numbers 
        Nu = h_mean(np.sqrt(Pr*Ra)*w*totT - dz(totT))(z=Lz/2)
        Sh = h_mean(np.sqrt(Pr*Ra)/tau*w*totS - dz(totS))(z=Lz/2)
        # .evaluate() returns a field object
        # ['g'] accesses the grid data
        # Heat and salt fluxes
        Jt_val = Jt.evaluate()['g'].real
        Js_val = Js.evaluate()['g'].real
        # Nusselt and Sherwood numbers 
        Nu_val = Nu.evaluate()['g'].real
        Sh_val = Sh.evaluate()['g'].real
        if self.dist.comm.rank == 0:
            return {'Jt': float(Jt_val),
                    'Js': float(Js_val),
                    'Nu': float(Nu_val),
                    'Sh': float(Sh_val)}
        else:
            return None
