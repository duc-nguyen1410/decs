# Adding a new problem

A model is a Python class that owns a Dedalus domain, its fields and an initial-value problem (IVP). Once it follows the conventions below, the [ECS solver](ecs_solver.md), stability analysis, continuation and saving/loading work with it unchanged.

There are two ways to start:
- **Same fields as an existing family** (e.g. velocity, temperature and salinity): subclass the family's base class, such as `DoubleDiffusion`, and only write the equations.
- **New set of fields**: subclass `FluidModel` and also create the domain and fields.

## Option 1: subclass an existing base class

`DoubleDiffusion` (`physics/double_diffusion_base.py`) already creates the domain and the fields `p`, `u`, `te`, `sa` (plus `u_eq`, `te_eq`, `sa_eq` for an eigenvalue problem). A new double-diffusive model only needs its equations and diagnostics. Use [`physics/salt_finger.py`](../physics/salt_finger.py) as a complete template.

```python
# physics/my_model.py
import numpy as np
import dedalus.public as de
from .double_diffusion_base import DoubleDiffusion


class MyModel(DoubleDiffusion):
    def build_problems(self):
        self.build_ivp_problem()

    def build_ivp_problem(self):
        tau_p = self.dist.Field(name='tau_p')
        tau_u = self.dist.VectorField(self.coords, name='tau_u')
        tau_te = self.dist.Field(name='tau_te')
        tau_sa = self.dist.Field(name='tau_sa')
        ez = self.coords.unit_vector_fields(self.dist)[-1]

        # Parameters as Dedalus fields named <name>_param, so that
        # model.set_param('Ra', value) (used by continuation) updates them in place.
        self.Ra_param = self.dist.Field(name='Ra_param'); self.Ra_param['g'] = self.params['Ra']
        self.Pr_param = self.dist.Field(name='Pr_param'); self.Pr_param['g'] = self.params['Pr']

        ns = {'Ra': self.Ra_param, 'Pr': self.Pr_param, 'ez': ez, 'w': self.u @ ez}
        variables = [self.p, self.u, self.te, self.sa, tau_p, tau_u, tau_te, tau_sa]
        self.ivp_problem = de.IVP(variables, namespace=ns)
        self.ivp_problem.add_equation("trace(grad(u)) + tau_p = 0")
        self.ivp_problem.add_equation("integ(p) = 0")
        self.ivp_problem.add_equation("dt(u) + grad(p) - Pr*lap(u) - Pr*Ra*(te - sa)*ez + tau_u = - u@grad(u)")
        self.ivp_problem.add_equation("dt(te) - lap(te) + w + tau_te = - u@grad(te)")
        self.ivp_problem.add_equation("dt(sa) - lap(sa) + w + tau_sa = - u@grad(sa)")
        for v in ['u', 'te', 'sa']:
            self.ivp_problem.add_equation(f"integ({v}) = 0")

    def get_flow_properties(self):
        """Scalar diagnostics of the current state, as a dict of floats."""
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        Nu = 1 - de.Average((self.u @ ez)*self.te)
        return {'Nu': float(Nu.evaluate()['g'].real.flat[0])}
```

## Option 2: subclass `FluidModel`

For a new set of fields, also write `__init__`, `create_domain` and `build_fields`. Copy them from [`physics/double_diffusion_base.py`](../physics/double_diffusion_base.py) and change the fields:

```python
from .base import FluidModel


class MyFamily(FluidModel):
    def __init__(self, params, sizes, bounds, bounded=False, mode='ecs', dealias=3/2):
        super().__init__(params, sizes, bounds, bounded, mode, dealias)
        self.create_domain()
        self.build_fields()

    def create_domain(self):
        ...  # set self.coords, self.dist (create once) and self.bases

    def build_fields(self):
        self.p = self.dist.Field(name='p', bases=self.bases)
        self.u = self.dist.VectorField(self.coords, name='u', bases=self.bases)
        self.b = self.dist.Field(name='b', bases=self.bases)
        self.fields = [self.u, self.b]          # state seen by the ECS solver; velocity first
        self.eq_fields = []                     # base-state fields of an EVP, if any
```

Then add `build_problems`, `build_ivp_problem` and `get_flow_properties` in the family class or in subclasses, as in Option 1.

## What the solver relies on

| Member | Used for |
|---|---|
| `self.fields` | the state vector: `get_state` / `set_state` flatten these fields in order |
| `self.fields[0]` must be the velocity | CFL time stepping and the sign changes of reflections |
| `self.ivp_problem` (built by `build_problems()`) | time integration `flow_map` |
| `self.bases` with a Fourier first ($x$) basis | shifts and reflections in $x$; $z$ may be Fourier (`bounded=False`) or Chebyshev (`bounded=True`) |
| `get_flow_properties()` returning a dict of floats | logs and `flow_properties.csv` |
| parameters as `<name>_param` fields | continuation in that parameter via `set_param` |
| `self.dist` created only once in `create_domain` | geometry continuation rebuilds the domain; re-creating the distributor leaks MPI communicators |

Optional:
- `set_snapshots`, `set_flowproperties`, `set_meanprofiles(solver, sim_dt, max_writes, mode)`: add Dedalus file handlers; used by simulations and, when `save_snapshots` etc. are on, to save the time history of a converged TW / PO / RPO. See [`physics/salt_finger.py`](../physics/salt_finger.py).
- `build_evp_problem(self, coupled=True)`: the IVP linearised about a base state, **returning** a `de.EVP`, for `model.solve_evp(...)` and `stability_method='evp'`. Replace `dt(q)` by `sigma*q - c@grad(q)` and each nonlinear term by its linearisation, e.g. `u@grad(u)` by `u@grad(u_eq) + u_eq@grad(u)`. Take the base fields from `self.evp_base_fields(coupled)` (full fields, or z-only copies for horizontally uniform base states) and the frame velocity `c` from `self.evp_frame_velocity()`. See [`physics/salt_finger.py`](../physics/salt_finger.py) and [`physics/thermohaline_shear.py`](../physics/thermohaline_shear.py).
- `preview()`: quick plot of the current state.

## Register the model

Export the class in [`physics/__init__.py`](../physics/__init__.py) so it can be imported as `from physics import MyModel`:

```python
from .my_model import MyModel
```

and add `'MyModel'` to `__all__`. Then check it with a short simulation (as in [`examples/simulate.py`](../examples/simulate.py)) before searching for ECS.
