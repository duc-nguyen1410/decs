import dedalus.public as de
import numpy as np
import h5py
from .base import FluidModel 

class DoubleDiffusion(FluidModel):
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
        # salinity s (scalar)
        self.sa = self.dist.Field(name='sa', bases=self.all_bases)
        self.sa.change_scales(self.dealias)
        self.sa_eq = self.dist.Field(name='sa_eq', bases=self.all_bases)

        # Newton solver now sees [u, te, sa]
        self.state_fields = [self.u, self.te, self.sa]
        self.eq_fields = [self.u_eq, self.te_eq, self.sa_eq]

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
        # salinity s (scalar)
        self.sa = self.dist.Field(name='sa', bases=self.all_bases)
        self.sa.change_scales(self.dealias)
        self.sa_eq = self.dist.Field(name='sa_eq', bases=self.all_bases)
        # re-collect fields to field list
        self.state_fields = [self.u, self.te, self.sa]
        self.eq_fields = [self.u_eq, self.te_eq, self.sa_eq]
    
class SaltFinger(DoubleDiffusion):
    def build_problems(self):
        self.build_ivp_problem()
        self.build_evp_problem()
    def build_ivp_problem(self):
        ns = self._get_base_namespace()
        ns.update({'Ra':self.params['Ra'],
                   'Pr':self.params['Pr'],
                   'Rrho':self.params['Rrho'],
                   'tau':self.params['tau'],})
        tau_p = self.dist.Field(name='tau_p')
        tau_u = self.dist.VectorField(self.coords, name='tau_u')
        tau_te = self.dist.Field(name='tau_te')
        tau_sa = self.dist.Field(name='tau_sa')
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
        ns = self._get_base_namespace()
        ns.update({'Ra':self.params['Ra'],
                   'Pr':self.params['Pr'],
                   'Rrho':self.params['Rrho'],
                   'tau':self.params['tau'],})
        sigma = self.dist.Field(name='sigma')
        tau_p = self.dist.Field(name='tau_p')
        tau_u = self.dist.VectorField(self.coords, name='tau_u')
        tau_te = self.dist.Field(name='tau_te')
        tau_sa = self.dist.Field(name='tau_sa')
        ns.update({'u_eq': self.u_eq, 'te_eq': self.te_eq, 'sa_eq': self.sa_eq,
                   'sigma': sigma})
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
        ns = self._get_base_namespace()
        w = ns['w']
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
    
class DiffusiveConvection(DoubleDiffusion):
    def build_problems(self):
        self.build_ivp_problem()
        self.build_evp_problem()
    def build_ivp_problem(self):
        ns = self._get_base_namespace()
        ns.update({'Ra':self.params['Ra'],
                   'Pr':self.params['Pr'],
                   'Lambda':self.params['Lambda'],
                   'tau':self.params['tau'],})
        tau_p = self.dist.Field(name='tau_p')
        tau_u = self.dist.VectorField(self.coords, name='tau_u')
        tau_te = self.dist.Field(name='tau_te')
        tau_sa = self.dist.Field(name='tau_sa')
        ns.update({'np': np})
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
        ns = self._get_base_namespace()
        ns.update({'Ra':self.params['Ra'],
                   'Pr':self.params['Pr'],
                   'Lambda':self.params['Lambda'],
                   'tau':self.params['tau'],})
        tau_p = self.dist.Field(name='tau_p')
        tau_u = self.dist.VectorField(self.coords, name='tau_u')
        tau_te = self.dist.Field(name='tau_te')
        tau_sa = self.dist.Field(name='tau_sa')
        ns.update({'u_eq': self.u_eq, 'te_eq': self.te_eq, 'sa_eq': self.sa_eq,
                   'sigma': self.sigma})
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
    def build_problems(self):
        self.build_ivp_problem()
    def build_ivp_problem(self):
        ns = self._get_base_namespace()
        ns.update({'Ra':self.params['Ra'],
                   'Pr':self.params['Pr'],
                   'Lambda':self.params['Lambda'],
                   'tau':self.params['tau'],
                   'Ri':self.params['Ri']})
        # print("initial namespace:", ns)
        tau_p = self.dist.Field(name='tau_p')
        tau_u1 = self.dist.VectorField(self.coords, name='tau_u1', bases=self.all_bases[:-1])
        tau_te1 = self.dist.Field(name='tau_te1', bases=self.all_bases[:-1])
        tau_sa1 = self.dist.Field(name='tau_sa1', bases=self.all_bases[:-1])
        tau_u2 = self.dist.VectorField(self.coords, name='tau_u2', bases=self.all_bases[:-1])
        tau_te2 = self.dist.Field(name='tau_te2', bases=self.all_bases[:-1])
        tau_sa2 = self.dist.Field(name='tau_sa2', bases=self.all_bases[:-1])

        ez = ns['ez']
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
        # print("Ri=",self.params['Ri'])
        # print("Ra=",self.params['Ra'])
        # print("Pr=",self.params['Pr'])
        # print("tau=",self.params['tau'])
        # print("Lambda=",self.params['Lambda'])
        z, = self.dist.local_grids(self.z_basis)
        Lz = self.z_basis.bounds[1]
        # print("Lz=",Lz)
        baru['g'] = (z-Lz/2)*Uw
        ns.update({'np': np,
                   'grad_u': grad_u, 'grad_te': grad_te, 'grad_sa': grad_sa, 
                   'lap_u': lap_u, 'lap_te': lap_te, 'lap_sa': lap_sa,
                   'dx': dx, 'dz': dz,
                   'baru': baru,
                   'lift': lift
                   })
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
        ns = self._get_base_namespace()
        ex = ns['ex']
        w = ns['w']
        Ra = self.params['Ra']
        Pr = self.params['Pr']
        tau = self.params['tau']
        #
        z, = self.dist.local_grids(self.z_basis)
        Lz = self.z_basis.bounds[1]
        #
        # baru = self.dist.Field(bases=(self.z_basis)) # base flow
        barT = self.dist.Field(bases=(self.z_basis)) # base state of temperature: -y
        barS = self.dist.Field(bases=(self.z_basis)) # base state of temperature: -y
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
