import dedalus.public as de
import numpy as np
import h5py
import matplotlib.pyplot as plt
from .base import FluidModel 

class MagnetoConvection(FluidModel):
    def __init__(self, params, sizes, bounds, bounded=False, mode='ecs', dealias=3/2):
        # Call the FluidModel __init__ first
        super().__init__(params, sizes, bounds, bounded, mode, dealias)
        
        # Add additional fields
        # pressure p (scalar)
        self.p = None
        # velocity u (Vector)
        self.u = None
        self.u_eq = None
        # temperature \theta (scalar)
        self.te = None
        self.te_eq = None
        # electric potential \Phi (scalar)
        self.Phi = None
        self.Phi_eq = None

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
        # electric potential \Phi (scalar)
        self.Phi = self.dist.Field(name='Phi', bases=self.bases)
        self.Phi.change_scales(self.dealias)
        self.Phi_eq = self.dist.Field(name='Phi_eq', bases=self.bases)

        # Newton solver now sees [u, Phi, te], save 'te' as last element for preview if needed
        if self.dim==3:
            self.fields = [self.u, self.Phi, self.te] # for ECS
            self.eq_fields = [self.u_eq, self.Phi_eq, self.te_eq] # for EVP
        else: 
            self.fields = [self.u, self.te] # for ECS
            self.eq_fields = [self.u_eq, self.te_eq] # for EVP
    
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
    
class BoundedQuasiStaticMagnetoConvection(MagnetoConvection):
    def build_problems(self):
        self.build_ivp_problem()
    def build_ivp_problem(self):
        tau_p = self.dist.Field(name='tau_p')
        tau_u1 = self.dist.VectorField(self.coords, name='tau_u1', bases=self.bases[:-1])
        tau_te1 = self.dist.Field(name='tau_te1', bases=self.bases[:-1])
        tau_u2 = self.dist.VectorField(self.coords, name='tau_u2', bases=self.bases[:-1])
        tau_te2 = self.dist.Field(name='tau_te2', bases=self.bases[:-1])
        # Phi only needed for 3D or specific 2.5D setups
        if self.dim == 3:
            tau_Phi1 = self.dist.Field(name='tau_Phi1', bases=self.bases[:-1])
            tau_Phi2 = self.dist.Field(name='tau_Phi2', bases=self.bases[:-1])
            tau_Phi_gauge = self.dist.Field(name='tau_Phi_gauge')
        
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
        lap_u = de.div(grad_u)
        lap_te = de.div(grad_te)

        # --- Quasi-Static MHD Logic ---
        if self.dim == 3:
            grad_Phi = de.grad(self.Phi) + ez*lift(tau_Phi1) 
            lap_Phi = de.div(grad_Phi)
            # J is a 3D Vector
            J = - grad_Phi + de.cross(self.u, ez) # quasi-static MHD Ohm’s law
            Lorentz_force = de.cross(J, ez)
        else:
            # In 2D (xz), u = (ux, w). u x ez = -ux*ey. 
            # (u x ez) x ez = -ux*ex.
            # This bypasses the need for the Phi Poisson equation entirely.
            Lorentz_force = - (self.u @ ex) * ex

        dx = lambda A: de.Differentiate(A, self.coords['x']) 
        dz = lambda A: de.Differentiate(A, self.coords['z'])

        ns = {'np': np,
              'Ra':self.params['Ra'],
              'Pr':self.params['Pr'],
              'Q':self.params['Q'],
              'ex': ex, 'ey': ey, 'ez': ez,
              'ux': self.u @ ex,
              'w': self.u @ ez,
              'grad_u': grad_u, 'grad_te': grad_te,
              'lap_u': lap_u, 'lap_te': lap_te, 
              'dx': dx, 'dz': dz,
              'lift': lift,
              'Lorentz_force': Lorentz_force
             }
        # Variable List
        vars = [self.p, self.u, self.te, 
                tau_p, tau_u1, tau_te1, 
                tau_u2, tau_te2]
        if self.dim == 3:
            vars += [self.Phi, tau_Phi1, tau_Phi2, tau_Phi_gauge]
            ns.update({'lap_Phi': lap_Phi})
        
        self.ivp_problem = de.IVP(vars, namespace=ns)
        # Governing Equations
        self.ivp_problem.add_equation("dt(u) + grad(p) - Q*np.sqrt(Pr/Ra)*Lorentz_force - te*ez - np.sqrt(Pr/Ra)*lap_u + lift(tau_u2) = - u@grad_u")
        self.ivp_problem.add_equation("dt(te) - (1.0/np.sqrt(Pr*Ra))*lap_te - w + lift(tau_te2) = - u@grad_te")
        self.ivp_problem.add_equation("trace(grad_u) + tau_p = 0")
        self.ivp_problem.add_equation("integ(p) = 0") 

        if self.dim == 3:
            self.ivp_problem.add_equation("lap_Phi + lift(tau_Phi2) + tau_Phi_gauge = div(cross(u, ez))")
            self.ivp_problem.add_equation("integ(Phi) = 0")

        if self.params['stress-free']: # Stress-free boundary condition
            self.ivp_problem.add_equation("w(z='left') = 0") # No penetration
            self.ivp_problem.add_equation("dz(ux)(z='left') = 0") # Stress-free
            if self.dim==3:
                self.ivp_problem.add_equation("dz(uy)(z='left') = 0") # Stress-free
            self.ivp_problem.add_equation("w(z='right') = 0") # No penetration
            self.ivp_problem.add_equation("dz(ux)(z='right') = 0") # Stress-free
            if self.dim==3:
                self.ivp_problem.add_equation("dz(uy)(z='right') = 0") # Stress-free
        else: # no-slip
            self.ivp_problem.add_equation("u(z='left') = 0")
            self.ivp_problem.add_equation("u(z='right') = 0")
        # Isothermal
        self.ivp_problem.add_equation("te(z='left') = 0")
        self.ivp_problem.add_equation("te(z='right') = 0")
        # Insulating
        if self.dim == 3:
            self.ivp_problem.add_equation("dz(Phi)(z='left') = 0")
            self.ivp_problem.add_equation("dz(Phi)(z='right') = 0")
    
    # def get_flow_properties(self):
    #     ez = self.coords.unit_vector_fields(self.dist)[-1]
    #     w = self.w
    #     Ra = self.params['Ra']
    #     Pr = self.params['Pr']
    #     Q = self.params['Q']
    #     #
    #     z, = self.dist.local_grids(self.bases[-1])
        
    #     dz = lambda A: de.Differentiate(A, self.coords['z']) 
    #     h_mean = lambda A: de.Average(A,'x')
    #     # Nusselt number
    #     Nu_p = 1-dz(self.T0)(z=0) # plane Nusselt number
    #     # .evaluate() returns a field object
    #     # ['g'] accesses the grid data
    #     Nu_p_val = Nu_p.evaluate()['g'].real
    #     if self.dist.comm.rank == 0:
    #         return {'Nu_p': float(Nu_p_val)}
    #     else:
    #         return None