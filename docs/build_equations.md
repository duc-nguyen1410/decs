To build new governing equations, you have to create a new class `NewProblem` from base class `FluidModel`

```python
from physics.base import FluidModel 

class NewProblem(FluidModel):
    def __init__(self, params, sizes, bounds, bounded=False, mode='ecs', dealias=3/2):
        # Call the FluidModel __init__ first
        super().__init__(params, sizes, bounds, bounded, mode, dealias)

        # Add additional fields, p, u, te, ...
        # for example:
        # pressure p (scalar)
        self.p = self.dist.Field(name='p', bases=self.all_bases)
        # velocity u (Vector)
        self.u = self.dist.VectorField(self.coords, name='u', bases=self.all_bases)
        self.u.change_scales(self.dealias)
        self.u_eq = self.dist.VectorField(self.coords, name='u_eq', bases=self.all_bases)
        # other fields
        # ...

        # replace key variable list for ECS solver
        self.state_fields = [self.u] # for ECS
        self.eq_fields = [self.u_eq] # for EVP, if needed
    def rebuild_fields(self):
        # pressure p (scalar)
        self.p = self.dist.Field(name='p', bases=self.all_bases)
        # velocity u (Vector)
        self.u = self.dist.VectorField(self.coords, name='u', bases=self.all_bases)
        self.u.change_scales(self.dealias)
        self.u_eq = self.dist.VectorField(self.coords, name='u_eq', bases=self.all_bases)

        # replace key variable list for ECS solver
        self.state_fields = [self.u, self.Phi, self.te] # for ECS
        self.eq_fields = [self.u_eq, self.Phi_eq, self.te_eq] # for EVP, if needed
        
```


Now we can build new spectific equations for IVP problem
```python
class ChildProblem(NewProblem):
    def build_problems(self):
        self.build_ivp_problem()
        # self.build_evp_problem()
    def build_ivp_problem(self):
        ns = self._get_base_namespace()

        # define sub-variables
        tau_p = ...
        tau_u1 = ...
        tau_u2 = ...
        lift = ...
        grad_u = ...
        lap_up = ...
        # set list of variables `vars` for IVP
        vars = [self.p, self.u]
        # update namespace 'ns' for parameters,... if needed
        ns.update({'Re':self.params['Re'], 

                    })
        # define IVP problem here
        self.ivp_problem = de.IVP(vars, namespace=ns)
        # Governing Equations
        self.ivp_problem.add_equation("dt(u) + grad(p) -(1/Re)*lap_u + lift(tau_u2) = - u@grad_u")
        self.ivp_problem.add_equation("trace(grad_u) + tau_p = 0")
        self.ivp_problem.add_equation("integ(p) = 0") 
        # other equations

    # similarly, we define EVP if needed
    # for standard ECS solver, only IVP problem is enough
    # just use evp problem if you want to use EVP
    def build_evp_problem(self):
        self.evp_problem = ...
        pass
```