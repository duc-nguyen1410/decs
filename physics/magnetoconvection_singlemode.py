import dedalus.public as de
import numpy as np
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

class MagnetoConvection_SingleMode(FluidModel):
    def __init__(self, params, sizes, bounds, bounded=False, mode='ecs', dealias=3/2):
        # Call the FluidModel __init__ first
        super().__init__(params, sizes, bounds, bounded, mode, dealias)

        ## sigle-mode ansatz for the horizontal direction
        ##  ux = U0(z,t) + u(z,t) * exp(i*kx*x + i*ky*y) + c.c.
        ##  uy = V0(z,t) + v(z,t) * exp(i*kx*x + i*ky*y) + c.c.
        ##  uz =           w(z,t) * exp(i*kx*x + i*ky*y) + c.c.
        ##  temperature = T0(z,t) + te(z,t) * exp(i*kx*x + i*ky*y) + c.c.
        # pressure p (scalar)
        self.p = None
        # velocity u (Vector)
        self.U0 = None
        self.V0 = None
        self.u = None
        self.v = None
        self.w = None
        # J (Vector)
        self.Jx0 = None
        self.Jy0 = None
        self.Jz0 = None
        self.Jx = None
        self.Jy = None
        self.Jz = None
        # temperature \theta (scalar)
        self.T0 = None
        self.te = None
        # electric potential \Phi (scalar)
        self.Phi0 = None
        self.Phi = None

        self.create_domain()
        self.build_fields()

    def create_domain(self):
        """ Creates a 2D or 3D domain/bases in the model """
        if self.dim == 1:
            Nz, = self.sizes
            Lz, = self.bounds
            self.coords = de.Coordinate('z')
        else:
            raise ValueError("Sizes and bounds in single-mode ansatz must be length 1.")
        
        # single-mode ansatz must use complex-valued fields
        if self.dist is None:
            self.dist = de.Distributor(self.coords, dtype=np.complex128)

        if self.bounded:
            # Use Chebyshev for bounded domains
            z_basis = de.ChebyshevT(self.coords, size=Nz, bounds=(0, Lz), dealias=self.dealias)
        else:
            raise ValueError("The single-mode ansatz needs a bounded domain.")
        self.bases = (z_basis,)
        # print("# of bases: ",len(self.bases))
        
    def build_fields(self):
        # pressure p (scalar)
        self.p = self.dist.Field(name='p', bases=self.bases)
        # velocity u (Vector)
        self.U0 = self.dist.Field(name='U0', bases=self.bases)
        self.U0.change_scales(self.dealias)
        self.V0 = self.dist.Field(name='V0', bases=self.bases)
        self.u = self.dist.Field(name='u', bases=self.bases)
        self.v = self.dist.Field(name='v', bases=self.bases)
        self.w = self.dist.Field(name='w', bases=self.bases)
        self.U0.change_scales(self.dealias)
        self.V0.change_scales(self.dealias)
        self.u.change_scales(self.dealias)
        self.v.change_scales(self.dealias)
        self.w.change_scales(self.dealias)
        # J (Vector)
        self.Jx0 = self.dist.Field(name='Jx0', bases=self.bases)
        self.Jx0.change_scales(self.dealias)
        self.Jy0 = self.dist.Field(name='Jy0', bases=self.bases)
        self.Jy0.change_scales(self.dealias)
        self.Jz0 = self.dist.Field(name='Jz0', bases=self.bases)
        self.Jz0.change_scales(self.dealias)
        self.Jx = self.dist.Field(name='Jx', bases=self.bases)
        self.Jx.change_scales(self.dealias)
        self.Jy = self.dist.Field(name='Jy', bases=self.bases)
        self.Jy.change_scales(self.dealias)
        self.Jz = self.dist.Field(name='Jz', bases=self.bases)
        self.Jz.change_scales(self.dealias)
        # temperature \theta (scalar)
        self.T0 = self.dist.Field(name='T0', bases=self.bases)
        self.T0.change_scales(self.dealias)
        self.te = self.dist.Field(name='te', bases=self.bases)
        self.te.change_scales(self.dealias)
        # electric potential \Phi (scalar)
        self.Phi0 = self.dist.Field(name='Phi0', bases=self.bases)
        self.Phi0.change_scales(self.dealias)
        self.Phi = self.dist.Field(name='Phi', bases=self.bases)
        self.Phi.change_scales(self.dealias)

        # Newton solver now sees [u, Phi, te], save 'te' as last element for preview if needed
        self.fields = [self.u, self.v, self.w, self.U0, self.V0, \
                       self.te, self.T0] # for ECS
        
    def preview(self):
        """ Preview the current state using last field of the system. """
        T0_g = self.T0.allgather_data('g').real
        te_g = self.te.allgather_data('g').real
        if self.dist.comm.rank == 0:
            zaxis = self.bases[-1].global_grid(self.dist, scale=self.dealias)
            kx = self.params['kx']
            ky = self.params['ky']
            Nz, = self.sizes
            x = np.linspace(0, 2*np.pi/kx, Nz)
            y = 0.0
            # plot slice y=0
            ##  temperature = T0(z) + te(z) * exp(i*kx*x + i*ky*y) 
            temperature = (T0_g[:, None] + te_g[:, None] * np.exp(1j*kx*x[None, :] + 1j*ky*y)).real
            # Initialize the figure only once
            if self.preview_fig is None:
                plt.ion()  # Turn on interactive mode
                self.preview_fig, self.preview_ax = plt.subplots(figsize=(4,3))
                self.preview_im = self.preview_ax.pcolormesh(x, zaxis.ravel(), temperature, 
                                             cmap='RdBu_r', shading='auto')
                self.preview_ax.set_xlabel('x')
                self.preview_ax.set_ylabel('z')
                self.preview_fig.colorbar(self.preview_im)
                # self.preview_ax.set_title("Salt Concentration") 
                self.preview_fig.canvas.draw()
                self.preview_fig.canvas.flush_events()  
            else:
                self.preview_im.set_array(temperature.ravel())
                v_min, v_max = np.min(temperature), np.max(temperature)
                self.preview_im.set_clim(vmin=v_min, vmax=v_max)
                # self.preview_ax.set_title(f"Salt Concentration at time {self.sim_time:.2f}")
                self.preview_fig.canvas.draw()
                self.preview_fig.canvas.flush_events()  
    
class BoundedQuasiStaticMagnetoConvection_SingleMode(MagnetoConvection_SingleMode):
    def build_problems(self):
        self.build_ivp_problem()
    def build_ivp_problem(self):
        # Tau fields (scalars, no basis)
        tau_p = self.dist.Field(name='tau_p')
        tau_u1 = self.dist.Field(name='tau_u1'); tau_u2 = self.dist.Field(name='tau_u2')
        tau_v1 = self.dist.Field(name='tau_v1'); tau_v2 = self.dist.Field(name='tau_v2')
        tau_w1 = self.dist.Field(name='tau_w1'); tau_w2 = self.dist.Field(name='tau_w2')
        tau_U01 = self.dist.Field(name='tau_U01'); tau_U02 = self.dist.Field(name='tau_U02')
        tau_V01 = self.dist.Field(name='tau_V01'); tau_V02 = self.dist.Field(name='tau_V02')
        tau_te1 = self.dist.Field(name='tau_te1'); tau_te2 = self.dist.Field(name='tau_te2')
        tau_T01 = self.dist.Field(name='tau_T01'); tau_T02 = self.dist.Field(name='tau_T02')
        tau_Phi1 = self.dist.Field(name='tau_Phi1'); tau_Phi2 = self.dist.Field(name='tau_Phi2')
        
        # Substitutions and Lifts
        z = self.dist.local_grids(self.bases[-1])
        lift_basis1 = self.bases[-1].derivative_basis(1)
        lift_basis2 = self.bases[-1].derivative_basis(2)
        lift1 = lambda A: de.Lift(A, lift_basis1, -1)
        lift2 = lambda A: de.Lift(A, lift_basis2, -2)

        dz = lambda A: de.Differentiate(A, self.coords)
        i = 1j
        conj = lambda A: np.conj(A)

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
              'kx': self.params['kx'], 'ky': self.params['ky'], 
              'k2': self.params['kx']**2 + self.params['ky']**2,
              'dz': dz,
              'i': i, 'conj': conj,
              'lift1': lift1, 'lift2': lift2, 
             }
        # Variable List
        vars = [self.p, tau_p,
                self.U0, tau_U01, tau_U02, self.u, tau_u1, tau_u2, 
                self.V0, tau_V01, tau_V02, self.v, tau_v1, tau_v2,
                self.w, tau_w1, tau_w2,
                self.T0, tau_T01, tau_T02, self.te, tau_te1, tau_te2,
                self.Jx0, self.Jy0, self.Jx, self.Jy, self.Jz,
                self.Phi, tau_Phi1, tau_Phi2]
        
        self.ivp_problem = de.IVP(vars, namespace=ns)
        # Governing Equations
        ## momentum equation
        self.ivp_problem.add_equation("dt(u) + i*kx*p - np.sqrt(Pr/Ra)*(dz(dz(u))-k2*u) - Q*np.sqrt(Pr/Ra)*Jy + lift1(tau_u1) + lift2(tau_u2) = - U0*i*kx*u - V0*i*ky*u - w*dz(U0)")
        self.ivp_problem.add_equation("dt(v) + i*ky*p - np.sqrt(Pr/Ra)*(dz(dz(v))-k2*v) + Q*np.sqrt(Pr/Ra)*Jx + lift1(tau_v1) + lift2(tau_v2) = - U0*i*kx*v - V0*i*ky*v - w*dz(V0)")
        self.ivp_problem.add_equation("dt(w) + dz(p) - np.sqrt(Pr/Ra)*(dz(dz(w))-k2*w) - te  + lift1(tau_w1) + lift2(tau_w2) = - U0*i*kx*w - V0*i*ky*w")
        self.ivp_problem.add_equation("i*kx*u+i*ky*v+dz(w) + lift1(tau_p) = 0")
        self.ivp_problem.add_equation("dt(U0) - np.sqrt(Pr/Ra)*dz(dz(U0)) - Q*np.sqrt(Pr/Ra)*Jy0 + lift1(tau_U01) + lift2(tau_U02) = -dz(conj(w)*u+w*conj(u))")
        self.ivp_problem.add_equation("dt(V0) - np.sqrt(Pr/Ra)*dz(dz(V0)) + Q*np.sqrt(Pr/Ra)*Jx0 + lift1(tau_V01) + lift2(tau_V02) = -dz(conj(w)*v+w*conj(v))")
        ## temperature equation
        self.ivp_problem.add_equation("dt(te) - (1.0/np.sqrt(Pr*Ra))*(dz(dz(te))-k2*te) - w + lift1(tau_te1) + lift2(tau_te2) = - U0*i*kx*te - V0*i*ky*te - w*dz(T0)")
        self.ivp_problem.add_equation("dt(T0) - (1.0/np.sqrt(Pr*Ra))*dz(dz(T0)) + lift1(tau_T01) + lift2(tau_T02) = -dz(conj(w)*te+w*conj(te))")
        ##
        self.ivp_problem.add_equation("Jx + i*kx*Phi = v")
        self.ivp_problem.add_equation("Jy + i*ky*Phi = - u")
        self.ivp_problem.add_equation("Jz + dz(Phi) = 0")
        self.ivp_problem.add_equation("i*kx*Jx + i*ky*Jy + dz(Jz) + lift1(tau_Phi1) + lift2(tau_Phi2) = 0")
        self.ivp_problem.add_equation("Jx0 = V0")
        self.ivp_problem.add_equation("Jy0 = - U0")

        # self.ivp_problem.add_equation("p(z='left') = 0")
        self.ivp_problem.add_equation("integ(p) = 0") 

        # boundary condition
        self.ivp_problem.add_equation("w(z='left') = 0") # No penetration
        self.ivp_problem.add_equation("w(z='right') = 0") # No penetration
        if self.params['stress-free']: # Stress-free
            self.ivp_problem.add_equation("dz(u)(z='left') = 0")
            self.ivp_problem.add_equation("dz(u)(z='right') = 0")
            self.ivp_problem.add_equation("dz(v)(z='left') = 0")
            self.ivp_problem.add_equation("dz(v)(z='right') = 0")
            self.ivp_problem.add_equation("dz(U0)(z='left') = 0")
            self.ivp_problem.add_equation("dz(U0)(z='right') = 0")
            self.ivp_problem.add_equation("dz(V0)(z='left') = 0")
            self.ivp_problem.add_equation("dz(V0)(z='right') = 0")
        else: # noslip
            self.ivp_problem.add_equation("u(z='left') = 0") 
            self.ivp_problem.add_equation("u(z='right') = 0") 
            self.ivp_problem.add_equation("v(z='left') = 0") 
            self.ivp_problem.add_equation("v(z='right') = 0") 
            self.ivp_problem.add_equation("U0(z='left') = 0") 
            self.ivp_problem.add_equation("U0(z='right') = 0") 
            self.ivp_problem.add_equation("V0(z='left') = 0")
            self.ivp_problem.add_equation("V0(z='right') = 0")

        # Isothermal
        self.ivp_problem.add_equation("te(z='left') = 0")
        self.ivp_problem.add_equation("te(z='right') = 0")
        self.ivp_problem.add_equation("T0(z='left') = 0")
        self.ivp_problem.add_equation("T0(z='right') = 0")
        # Insulating
        self.ivp_problem.add_equation("dz(Phi)(z='left') = 0")
        self.ivp_problem.add_equation("dz(Phi)(z='right') = 0")

    def get_flow_properties(self):
        #
        z = self.dist.local_grids(self.bases[-1])
        
        dz = lambda A: de.Differentiate(A, self.coords) 
        # Nusselt number
        Nu_p = 1-dz(self.T0)(z=0) # plane Nusselt number
        # .evaluate() returns a field object
        # ['g'] accesses the grid data
        Nu_p_val = Nu_p.evaluate()['g'].real
        if self.dist.comm.rank == 0:
            return {'Nu_p': float(Nu_p_val.item())}
        else:
            return {}   
