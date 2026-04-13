import dedalus.public as de
import numpy as np
import h5py
from .base import FluidModel2D # The dot means "from the same folder"

class DoubleDiffusion(FluidModel2D):
    def __init__(self, domain, params):
        # Call the FluidModel2D __init__ first
        super().__init__(domain, params)
        
        # Add the specific salinity field
        self.sa = self.dist.Field(name='sa', bases=(self.x_basis, self.z_basis))
        self.sa.change_scales(self.dealias)
        self.tau_sa = self.dist.Field(name='tau_sa')
        self.sa_eq = self.dist.Field(name='sa_eq', bases=(self.x_basis, self.z_basis))

        # for bounded domain
        

        # Newton solver now sees [u, te, sa]
        self.state_fields.append(self.sa)

        self.eq_fields.append(self.sa_eq)

    def _get_base_namespace(self):
        ex, ez = self.coords.unit_vector_fields(self.dist)
        ns = {
            'ez': ez, 'ex': ex,
            'w': self.u @ ez,
            # Operators
            # 'grad': lambda A: de.grad(A),
            # 'lap': lambda A: de.div(de.grad(A)),
            # 'trace': lambda A: de.trace(A),
            # 'integ': lambda A: de.Integrate(A)
        }
        ns.update(self.params)
        return ns
    
    def load_state(self, filename):
        with h5py.File(filename, mode='r') as file:
            u_init = np.array(file.get('/u'))
            w_init = np.array(file.get('/w'))
            te_init = np.array(file.get('/t'))
            sa_init = np.array(file.get('/s'))
        self.set_state(np.concatenate([u_init.ravel(), w_init.ravel(), te_init.ravel(), sa_init.ravel()]))

class SaltFinger(DoubleDiffusion):
    def get_IVP(self):
        ns = self._get_base_namespace()
        vars = [self.p, self.u, self.te, self.sa, 
                self.tau_p, self.tau_u, self.tau_te, self.tau_sa]
        problem = de.IVP(vars, namespace=ns)
        # Periodic Governing Equations
        problem.add_equation("trace(grad(u)) + tau_p = 0")
        problem.add_equation("integ(p) = 0") 
        problem.add_equation("dt(u) + grad(p) - Pr*lap(u) - Pr*Ra*(te-sa/Rrho)*ez + tau_u = - u@grad(u)")
        problem.add_equation("dt(te) - lap(te) + w + tau_te = - u@grad(te)")
        problem.add_equation("dt(sa) - tau*lap(sa) + w + tau_sa = - u@grad(sa)")
        # Integral constraints for floating zero-means
        for v in ['u', 'te', 'sa']:
            problem.add_equation(f"integ({v}) = 0")
        return problem
    def get_EVP(self):
        ns = self._get_base_namespace()
        ns.update({'u_eq': self.u_eq, 'te_eq': self.te_eq, 'sa_eq': self.sa_eq,
                   'sigma': self.sigma})
        vars = [self.p, self.u, self.te, self.sa, 
                self.tau_p, self.tau_u, self.tau_te, self.tau_sa]
        # Define EVP
        problem = de.EVP(vars, eigenvalue=self.sigma, namespace=ns)
        # Add equations here based on the linearized physics of salt finger convection
        problem.add_equation("trace(grad(u)) + tau_p = 0")
        problem.add_equation("integ(p) = 0") 
        problem.add_equation("sigma*u + grad(p) - Pr*lap(u) - Pr*Ra*(te-sa/Rrho)*ez + u@grad(u_eq)+u_eq@grad(u) + tau_u = 0")
        problem.add_equation("sigma*te - lap(te) + w + u@grad(te_eq)+u_eq@grad(te) + tau_te = 0")
        problem.add_equation("sigma*sa - tau*lap(sa) + w + u@grad(sa_eq)+u_eq@grad(sa) + tau_sa = 0")
        # Integral constraints
        for v in ['u', 'te', 'sa']:
            problem.add_equation(f"integ({v}) = 0")
        return problem
    
class DiffusiveConvection(DoubleDiffusion):
    def get_IVP(self):
        ns = self._get_base_namespace()
        ns.update({'np': np})
        vars = [self.p, self.u, self.te, self.sa, 
                self.tau_p, self.tau_u, self.tau_te, self.tau_sa]
        problem = de.IVP(vars, namespace=ns)
        # Periodic Governing Equations
        problem.add_equation("trace(grad(u)) + tau_p = 0")
        problem.add_equation("integ(p) = 0") 
        # velocity nondimensionalization by thermal diffusivity
        # problem.add_equation("dt(u) + grad(p) - Pr*lap(u) - Pr*Ra*(te-Lambda*sa)*ez + tau_u = - u@grad(u)")
        # problem.add_equation("dt(te) - lap(te) - w + tau_te = - u@grad(te)")
        # problem.add_equation("dt(sa) - tau*lap(sa) - w + tau_sa = - u@grad(sa)")
        # velocity nondimensionalization by free-fall velocity
        problem.add_equation("dt(u) + grad(p) - np.sqrt(Pr/Ra)*lap(u) - (te-Lambda*sa)*ez + tau_u = - u@grad(u)")
        problem.add_equation("dt(te) - (1.0/np.sqrt(Pr*Ra))*lap(te) - w + tau_te = - u@grad(te)")
        problem.add_equation("dt(sa) - (tau/np.sqrt(Pr*Ra))*lap(sa) - w + tau_sa = - u@grad(sa)")
        # Integral constraints for floating zero-means
        for v in ['u', 'te', 'sa']:
            problem.add_equation(f"integ({v}) = 0")
        return problem
    def get_EVP(self):
        ns = self._get_base_namespace()
        ns.update({'u_eq': self.u_eq, 'te_eq': self.te_eq, 'sa_eq': self.sa_eq,
                   'sigma': self.sigma})
        vars = [self.p, self.u, self.te, self.sa, 
                self.tau_p, self.tau_u, self.tau_te, self.tau_sa]
        # Define EVP
        problem = de.EVP(vars, eigenvalue=self.sigma, namespace=ns)
        # Add equations here based on the linearized physics of salt finger convection
        problem.add_equation("trace(grad(u)) + tau_p = 0")
        problem.add_equation("integ(p) = 0") 
        problem.add_equation("sigma*u + grad(p) - Pr*lap(u) - Pr*Ra*(te-Lambda*sa)*ez + u@grad(u_eq)+u_eq@grad(u) + tau_u = 0")
        problem.add_equation("sigma*te - lap(te) - w + u@grad(te_eq)+u_eq@grad(te) + tau_te = 0")
        problem.add_equation("sigma*sa - tau*lap(sa) - w + u@grad(sa_eq)+u_eq@grad(sa) + tau_sa = 0")
        # Integral constraints
        for v in ['u', 'te', 'sa']:
            problem.add_equation(f"integ({v}) = 0")
        return problem
    
class ShearedDiffusiveConvection(DoubleDiffusion):
    def get_IVP(self):
        ns = self._get_base_namespace()
        tau_u1 = self.dist.VectorField(self.coords, name='tau_u1', bases=self.x_basis)
        tau_te1 = self.dist.Field(name='tau_te1', bases=self.x_basis)
        tau_sa1 = self.dist.Field(name='tau_sa1', bases=self.x_basis)
        tau_u2 = self.dist.VectorField(self.coords, name='tau_u2', bases=self.x_basis)
        tau_te2 = self.dist.Field(name='tau_te2', bases=self.x_basis)
        tau_sa2 = self.dist.Field(name='tau_sa2', bases=self.x_basis)

        ex, ez = self.coords.unit_vector_fields(self.dist)
        lift_basis = self.z_basis.derivative_basis(1)
        lift = lambda A: de.Lift(A, lift_basis, -1)

        grad_u = de.grad(self.u) + ez*lift(tau_u1) 
        grad_te = de.grad(self.te) + ez*lift(tau_te1) 
        grad_sa = de.grad(self.sa) + ez*lift(tau_sa1) 
        lap_u = de.div(grad_u)
        lap_te = de.div(grad_te)
        lap_sa = de.div(grad_sa)

        dx = lambda A: de.Differentiate(A, self.coords['x']) 
        dz = lambda A: de.Differentiate(A, self.coords['z']) 

        baru = self.dist.Field(bases=self.z_basis)
        Uw = 1.0/np.sqrt(self.params['Ri'])
        z, = self.dist.local_grids(self.z_basis)
        baru['g'] = z*Uw
        ns.update({'np': np,
                   'grad_u': grad_u, 'grad_te': grad_te, 'grad_sa': grad_sa, 
                   'lap_u': lap_u, 'lap_te': lap_te, 'lap_sa': lap_sa,
                   'dx': dx, 'dz': dz,
                   'baru': baru,
                   'lift': lift
                   })
        vars = [self.p, self.u, self.te, self.sa, 
                self.tau_p, tau_u1, tau_te1, tau_sa1,
                tau_u2, tau_te2, tau_sa2]
        problem = de.IVP(vars, namespace=ns)
        # Periodic Governing Equations
        problem.add_equation("trace(grad_u) + tau_p = 0")
        problem.add_equation("integ(p) = 0") 
        # velocity nondimensionalization by free-fall velocity
        problem.add_equation("dt(u) + baru*dx(u) + w*dz(baru)*ex + grad(p) - np.sqrt(Pr/Ra)*lap_u - (te-Lambda*sa)*ez + lift(tau_u2) = - u@grad_u")
        problem.add_equation("dt(te) + baru*dx(te) - (1.0/np.sqrt(Pr*Ra))*lap_te - w + lift(tau_te2) = - u@grad_te")
        problem.add_equation("dt(sa) + baru*dx(sa) - (tau/np.sqrt(Pr*Ra))*lap_sa - w + lift(tau_sa2) = - u@grad_sa")
        problem.add_equation("u(z='left') = 0")
        problem.add_equation("u(z='right') = 0")
        problem.add_equation("te(z='left') = 0")
        problem.add_equation("te(z='right') = 0")
        problem.add_equation("sa(z='left') = 0")
        problem.add_equation("sa(z='right') = 0")
        return problem
    