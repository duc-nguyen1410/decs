import dedalus.public as de
import numpy as np
from mpi4py import MPI
import h5py
import matplotlib.pyplot as plt
from .base import FluidModel 
# Helper function
def kc_to_minimize_critical_Ra(Q):
    # set Rayleigh number = critical Rayleigh number (based on eq 2.7 in paper Yan et al. (2019)
    # Coefficients of the polynomial (descending order)
    # 2*k_c^6 + 3*pi^2*k_c^4 + 0*k_c^3 + 0*k_c^2 + 0*k_c + (-pi^6 - Q*pi^4)
    coeffs = [2, 0, 3 * np.pi**2, 0, 0, 0, -(np.pi**6 + Q * np.pi**4)]
    # we can get an analytical solution for k_c by solving the polynomial equation 2*k_c^6 + 3*pi^2*k_c^4 - (pi^6 + Q*pi^4) = 0
    # Find roots
    roots = np.roots(coeffs)
    # Filter real roots
    real_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
    return real_roots
def critical_Ra(Q_val):
    k_c = kc_to_minimize_critical_Ra(Q_val)[0]  # Take the first real root
    Ra_c = (np.pi**2 + k_c**2)**3 / k_c**2 + Q_val * (np.pi**2 + k_c**2) / k_c**2
    return Ra_c

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
            x_basis = de.RealFourier(self.coords['x'], size=Nx, bounds=(0, Lx), dealias=self.dealias)
        else:
            x_basis = de.ComplexFourier(self.coords['x'], size=Nx, bounds=(0, Lx), dealias=self.dealias)
        
        # Only if 3D, Always periodic
        if self.dim == 3:
            self.y_basis = None
            if self.mode=='sim':
                y_basis = de.RealFourier(self.coords['y'], size=Ny, bounds=(0, Ly), dealias=self.dealias)
            else:
                y_basis = de.ComplexFourier(self.coords['y'], size=Ny, bounds=(0, Ly), dealias=self.dealias)

        self.z_basis = None
        if self.bounded:
            # Use Chebyshev for bounded domains
            z_basis = de.ChebyshevT(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)
        else:
            # Use Fourier for fully periodic domains
            if self.mode=='sim':
                z_basis = de.RealFourier(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)
            else:
                z_basis = de.ComplexFourier(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)

        self.bases = None
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
            # In 2D (xz), u = (ux, 0, w) -> u x ez = -ux*ey
            # J = grad_Phi + u x ez = grad_Phi -ux*ey
            # By applying the charge conservation, we can get lap_Phi = 0
            # Because we are using isulating B.C., it leads to grad_Phi = 0
            # so, J = -ux*ey
            # Lorentz_force = J x ez = -ux*ex.
            # This bypasses the need for the Phi Poisson equation entirely.
            Lorentz_force = - (self.u @ ex) * ex

        dx = lambda A: de.Differentiate(A, self.coords['x']) 
        dz = lambda A: de.Differentiate(A, self.coords['z'])

        Q = self.params['Q']
        Pr = self.params['Pr']
        Ra = self.params['Ra']
        use_scaled_Ra = self.params.get('scale_Ra_c', False)
        if use_scaled_Ra:
            Rac = critical_Ra(Q)
            Ra = Ra*Rac

        ns = {'np': np,
              'Ra': Ra,
              'Pr': Pr,
              'Q': Q,
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
    
    def get_flow_properties(self):
        ex = self.coords.unit_vector_fields(self.dist)[0]
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        ux = self.u @ ex
        w = self.u @ ez
        Q = self.params['Q']
        Pr = self.params['Pr']
        Ra = self.params['Ra']
        use_scaled_Ra = self.params.get('scale_Ra_c', False)
        if use_scaled_Ra:
            Rac = critical_Ra(Q)
            Ra = Ra*Rac
        #
        z, = self.dist.local_grids(self.bases[-1])
        dz = lambda A: de.Differentiate(A, self.coords['z']) 
        if self.dim == 2:
            h_mean = lambda A: de.Average(A,'x')
        else:
            h_mean = lambda A: de.Average(A,'x','y')
        # [1] plane Nusselt number
        Nu_p = 1-h_mean(dz(self.te))(z=0) # plane Nusselt number
        # [2] volume average
        L2_temp = np.sqrt(de.Average(self.te**2))
        # [3] 
        grad_u = de.grad(self.u)
        D = de.Average(np.sqrt(Pr/Ra)*(de.dot(grad_u @ ex, grad_u @ ex) + de.dot(grad_u @ ez, grad_u @ ez))) # Viscous dissipation
        I = de.Average(self.te*w) # kinetic energy input by temperature-induced bouyancy
        if self.dim == 2:
            # J = -ux*ey
            # Lorentz_force = J x ez = -ux*ex.
            # Lorentz_force * u = -ux*ux
            I_j = de.Average(-Q*np.sqrt(Pr/Ra)*ux*ux)
        else: # 3D
            grad_Phi = de.grad(self.Phi)
            lap_Phi = de.div(grad_Phi)
            # J is a 3D Vector
            J = - grad_Phi + de.cross(self.u, ez) # quasi-static MHD Ohm’s law
            Lorentz_force = de.cross(J, ez)
            I_j = de.Average(Q*np.sqrt(Pr/Ra)*de.dot(Lorentz_force*self.u))
        # .evaluate() returns a field object
        # ['g'] accesses the grid data
        Nu_p_val = Nu_p.evaluate()['g'].real
        L2_temp_val = L2_temp.evaluate()['g'].real
        D_val = D.evaluate()['g'].real
        I_val = I.evaluate()['g'].real
        I_j_val = I_j.evaluate()['g'].real
        if self.dist.comm.rank == 0:
            return {'Nu_p': float(Nu_p_val.item()),
                    'L2_temp': float(L2_temp_val.item()),
                    'D': float(D_val.item()),
                    'I': float(I_val.item()),
                    'I_j': float(I_j_val.item())}
        else:
            return {}


    def set_snapshots(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        snapshots = solver.evaluator.add_file_handler(self.odir+'snapshots', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        snapshots.add_task(self.u, name='u')
        snapshots.add_task(self.te, name='te')
        
    def set_flowproperties(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        ex = self.coords.unit_vector_fields(self.dist)[0]
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        w = self.u @ ez
        Q = self.params['Q']
        Pr = self.params['Pr']
        Ra = self.params['Ra']
        use_scaled_Ra = self.params.get('scale_Ra_c', False)
        if use_scaled_Ra:
            Rac = critical_Ra(Q)
            Ra = Ra*Rac
        if self.dim == 2:
            h_mean = lambda A: de.Average(A,'x')
        else:
            h_mean = lambda A: de.Average(A,'x','y')
        dz = lambda A: de.Differentiate(A, self.coords['z'])
        z, = self.dist.local_grids(self.bases[-1])
        flowproperties = solver.evaluator.add_file_handler(self.odir+'flowproperties', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        flowproperties.add_task(1-h_mean(dz(self.te))(z=0), name='Nu') # Nusselt number
        grad_u = de.grad(self.u)
        flowproperties.add_task(np.sum(de.Average(np.sqrt(Pr/Ra)*(de.dot(grad_u @ ex, grad_u @ ex) + de.dot(grad_u @ ez, grad_u @ ez)))), name='D') # Viscous dissipation
        flowproperties.add_task(de.Average(self.te*w), name='I') # kinetic energy input by bouyancy

    def set_meanprofiles(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        if self.dim == 2:
            h_mean = lambda A: de.Average(A,'x')
        else:
            h_mean = lambda A: de.Average(A,'x','y')
        ex = self.coords.unit_vector_fields(self.dist)[0]
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        meanprofiles = solver.evaluator.add_file_handler(self.odir+'meanprofiles', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        meanprofiles.add_task(h_mean(self.u@ex), name='u') # horizontal averaged x-axis velocity
        meanprofiles.add_task(h_mean(self.u@ez), name='w') # horizontal averaged z-axis velocity, optional, becasue wm=0
        meanprofiles.add_task(h_mean(self.te), name='te') # horizontal averaged temperature
