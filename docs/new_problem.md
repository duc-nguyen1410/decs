To build new governing equations, you have to create a new class `NewProblem` from base class `FluidModel`. In general, this class acts as a base for multiple problems using the same number of fields (velocity, temperature, density,...).

```python
from physics.base import FluidModel 

class NewProblem(FluidModel):
    def __init__(self, params, sizes, bounds, bounded=False, mode='ecs', dealias=3/2):
        # Call the FluidModel __init__ first
        super().__init__(params, sizes, bounds, bounded, mode, dealias)

        # Add additional fields, p, u, te, ...
        # for example:
        # pressure p (scalar)
        self.p = None
        self.u = None
        self.u_eq = None # use this for EVP
        # other fields
        # ...
        # replace key variable list for ECS solver
        self.state_fields = [self.u] # for ECS
        self.eq_fields = [self.u_eq] # for EVP, if needed
        self.create_domain()
        self.build_fields()

    def create_domain(self):
        """ Creates a 2D or 3D domain/bases using Cartesian coordinates """
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
        self.p = self.dist.Field(name='p', bases=self.all_bases)
        # velocity u (Vector)
        self.u = self.dist.VectorField(self.coords, name='u', bases=self.all_bases)
        self.u_eq = self.dist.VectorField(self.coords, name='u_eq', bases=self.all_bases)

        # replace key variable list for ECS solver
        self.fields = [self.u, self.Phi, self.te] # for ECS
        self.eq_fields = [self.u_eq, self.Phi_eq, self.te_eq] # for EVP, if needed
        
```


Now we can build new spectific problems. The code should include a IVP problem, EVP problem for solving eigenvalue problem if needed for analytical solutions, 
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

    def get_flow_properties(self):
        """ Return flow properties of state """
        ...
        return {'Nu': float(Nu_val.item()),
                'D': float(D_val.item()),
                'I': float(I_val.item())}

    def set_snapshots(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        """  Save snapshots during simulation or TW/PO/RPO """
        snapshots = solver.evaluator.add_file_handler(self.odir+'snapshots', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        snapshots.add_task(self.te, name='te')
        snapshots.add_task(self.u, name='u')

    def set_flowproperties(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        """  Save flow properties (scalar value) during simulation or TW/PO/RPO """
        ...
        flowproperties = solver.evaluator.add_file_handler(self.odir+'flowproperties', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        flowproperties.add_task(1 - de.Average(w*self.te), name='Nu') # Nusselt number
        grad_u = de.grad(self.u)
        flowproperties.add_task(de.Average(Pr*(de.dot(grad_u @ ex, grad_u @ ex) + de.dot(grad_u @ ez, grad_u @ ez))), name='D') # Viscous dissipation
        flowproperties.add_task(de.Average(Pr*Ra*(self.te-self.sa/Rrho)*w), name='I') # kinetic energy input by bouyancy

    def set_meanprofiles(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        """  Save mean profiles (horizontal/vertical average) during simulation or TW/PO/RPO """
        ...
        meanprofiles = solver.evaluator.add_file_handler(self.odir+'meanprofiles', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        meanprofiles.add_task(h_mean(self.u), name='u') # horizontal averaged velocity
        meanprofiles.add_task(h_mean(self.te), name='te') # horizontal averaged temperature
        
```