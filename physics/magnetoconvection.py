import dedalus.public as de
import numpy as np
import h5py
from .base import FluidModel 

class MagnetoConvection(FluidModel):
    def __init__(self, params, sizes, bounds, bounded=False, mode='ecs', dealias=3/2):
        # Call the FluidModel __init__ first
        super().__init__(params, sizes, bounds, bounded, mode, dealias)
        
        # Add additional fields
        # pressure p (scalar)
        self.p = self.dist.Field(name='p', bases=self.all_bases)
        # velocity u (Vector)
        self.u = self.dist.VectorField(self.coords, name='u', bases=self.all_bases)
        self.u.change_scales(self.dealias)
        self.u_eq = self.dist.VectorField(self.coords, name='u_eq', bases=self.all_bases)
        # temperature \theta (scalar)
        self.te = self.dist.Field(name='te', bases=self.all_bases)
        self.te.change_scales(self.dealias)
        self.te_eq = self.dist.Field(name='te_eq', bases=self.all_bases)
        # electric potential \Phi (scalar)
        self.Phi = self.dist.Field(name='Phi', bases=self.all_bases)
        self.Phi.change_scales(self.dealias)
        self.Phi_eq = self.dist.Field(name='Phi_eq', bases=self.all_bases)

        # Newton solver now sees [u, Phi, te], save 'te' as last element for preview if needed
        self.state_fields.extend([self.u, self.Phi, self.te])
        # for EVP
        self.eq_fields.extend([self.u_eq, self.Phi_eq, self.te_eq])
    def rebuild_fields(self):
        # pressure p (scalar)
        self.p = self.dist.Field(name='p', bases=self.all_bases)
        # velocity u (Vector)
        self.u = self.dist.VectorField(self.coords, name='u', bases=self.all_bases)
        self.u.change_scales(self.dealias)
        self.u_eq = self.dist.VectorField(self.coords, name='u_eq', bases=self.all_bases)
        # temperature \theta (scalar)
        self.te = self.dist.Field(name='te', bases=self.all_bases)
        self.te.change_scales(self.dealias)
        self.te_eq = self.dist.Field(name='te_eq', bases=self.all_bases)
        # electric potential \Phi (scalar)
        self.Phi = self.dist.Field(name='Phi', bases=self.all_bases)
        self.Phi.change_scales(self.dealias)
        self.Phi_eq = self.dist.Field(name='Phi_eq', bases=self.all_bases)

        # Newton solver now sees [u, Phi, te], save 'te' as last element for preview if needed
        if self.dim==3:
            self.state_fields.extend([self.u, self.Phi, self.te]) # for ECS
            self.eq_fields.extend([self.u_eq, self.Phi_eq, self.te_eq]) # for EVP
        else: 
            self.state_fields.extend([self.u, self.te]) # for ECS
            self.eq_fields.extend([self.u_eq, self.te_eq]) # for EVP

    def _get_base_namespace(self):
        unit_vectors = self.coords.unit_vector_fields(self.dist)
        if len(unit_vectors) == 2:
            ex, ez = unit_vectors
            ey = None # Or a zero-field if needed
        else:
            ex, ey, ez = unit_vectors
        ns = {
            'ex': ex, 'ez': ez, 
            'w': self.u @ ez,
            'ux': self.u @ ex,
        }
        if ey is not None:
            ns['ey'] = ey
            ns['uy'] = self.u @ ey
        ns.update(self.params)
        return ns
    
class BoundedQuasiStaticMagnetoConvection(MagnetoConvection):
    def build_problems(self):
        self.build_ivp_problem()
    def build_ivp_problem(self):
        ns = self._get_base_namespace()
        tau_p = self.dist.Field(name='tau_p')
        tau_u1 = self.dist.VectorField(self.coords, name='tau_u1', bases=self.all_bases[:-1])
        tau_te1 = self.dist.Field(name='tau_te1', bases=self.all_bases[:-1])
        tau_u2 = self.dist.VectorField(self.coords, name='tau_u2', bases=self.all_bases[:-1])
        tau_te2 = self.dist.Field(name='tau_te2', bases=self.all_bases[:-1])
        # Phi only needed for 3D or specific 2.5D setups
        if self.dim == 3:
            tau_Phi1 = self.dist.Field(name='tau_Phi1', bases=self.all_bases[:-1])
            tau_Phi2 = self.dist.Field(name='tau_Phi2', bases=self.all_bases[:-1])
            tau_Phi_gauge = self.dist.Field(name='tau_Phi_gauge')
        
        lift_basis = self.z_basis.derivative_basis(1)
        lift = lambda A: de.Lift(A, lift_basis, -1)

        ex = ns['ex']
        ez = ns['ez']
        grad_u = de.grad(self.u) + ez*lift(tau_u1) 
        grad_te = de.grad(self.te) + ez*lift(tau_te1) 
        lap_u = de.div(grad_u)
        lap_te = de.div(grad_te)

        # --- Quasi-Static MHD Logic ---
        if self.dim == 3:
            grad_Phi = de.grad(self.Phi) + ez*lift(tau_Phi1) 
            lap_Phi = de.div(grad_Phi)
            # J is a 3D Vector
            J = - grad_Phi + de.cross(self.u, ez)
            Lorentz_force = de.cross(J, ez)
        else:
            # In 2D (xz), u = (ux, w). u x ez = -ux*ey. 
            # (u x ez) x ez = -ux*ex.
            # This bypasses the need for the Phi Poisson equation entirely.
            Lorentz_force = - (self.u @ ex) * ex

        # J = - grad_Phi + de.cross(self.u, ez) # quasi-static MHD Ohm’s law

        dx = lambda A: de.Differentiate(A, self.coords['x']) 
        dz = lambda A: de.Differentiate(A, self.coords['z'])

        ns.update({'np': np,
                   'grad_u': grad_u, 'grad_te': grad_te,
                   'lap_u': lap_u, 'lap_te': lap_te, 
                   'dx': dx, 'dz': dz,
                   'lift': lift,
                   'Lorentz_force': Lorentz_force
                   })
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
    
class QuasiStaticMagnetoConvection_Assatz(MagnetoConvection):
    def get_IVP(self):
        pass