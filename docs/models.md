# Physical models

All models are imported from the `physics` package and share the same constructor:

```python
from physics import SaltFinger
model = SaltFinger(params, sizes, bounds, bounded=False, mode='ecs', dealias=3/2)
model.build_problems()
```

| Argument | Meaning |
|---|---|
| `params` | dict of physical parameters (below) and run options |
| `sizes` | grid `(Nx, Nz)` or `(Nx, Ny, Nz)` |
| `bounds` | domain size `(Lx, Lz)` or `(Lx, Ly, Lz)` |
| `bounded` | `False`: Fourier in $z$ (periodic); `True`: Chebyshev in $z$ (walls) |
| `mode` | `'ecs'`: complex Fourier bases, needed for symmetries and shifts in the ECS solver; `'sim'`: real Fourier bases, more robust for long simulations with CFL |
| `dealias` | dealiasing factor |

Run options shared by all models (in `params`): `init_dt` (time step, default `2e-4`), `ext` (format of saved states: `'.nc'`, the default and preferred, opens directly in ParaView and post-processes quickly; or `'.h5'`), `odir`, `use_CFL` and the CFL settings `cadence`, `safety`, `threshold`, `min_change`, `max_change`, `max_dt`, and `save_snapshots`, `save_flowproperties`, `save_meanprofiles` (time histories written after a TW / PO / RPO converges).

| Model | File | Status |
|---|---|---|
| `SaltFinger` | `physics/salt_finger.py` | validated |
| `ThermohalineShear` | `physics/thermohaline_shear.py` | validated |
| `BoundedSaltFinger`, `DiffusiveConvection`, `ShearedDiffusiveConvection`, `WallShearedDiffusiveConvection` | `physics/double_diffusion.py` | not validated |
| `BoundedQuasiStaticMagnetoConvection`, `BoundedQuasiStaticMagnetoConvection_SingleMode` | `physics/magneto.py` | not validated |

# Double-diffusive convection
Double-diffusive convection (DDC) requires velocity ($\boldsymbol{U}$), temperature ($T$), salinity ($S$), and pressure ($p$). We decompose the velocity, temperature, and salinity into a base state ($\boldsymbol{U}_b, T_b, S_b$) and fluctuations ($\boldsymbol{u}, \theta, s$). Fluctuations are the computed variables in simulation, visualization, and the [ECS solver](ecs_solver.md). The base state depends on the specific problem. All double-diffusive models share the base class `DoubleDiffusion` (`physics/double_diffusion_base.py`), which creates the domain and the fields `u`, `te` ($\theta$), `sa` ($s$) and `p`.

## Salt finger convection `SaltFinger`
This is unbounded salt finger convection `bounded=False`. Base state is:

$$\boldsymbol{U}_b=\boldsymbol{0},\quad T_b=S_b=z$$

Nondimensional governing equations in terms of fluctuation using thermal diffusion scale:

$$
\begin{align}
    \nabla\cdot \boldsymbol{u} &= 0,\
    \partial_t \boldsymbol{u}+\boldsymbol{u}\cdot\nabla\boldsymbol{u} &= -\nabla p + Pr\nabla^{2}\boldsymbol{u} + PrRa\left(\theta-R_\rho^{-1} s\right)\mathbf{e}_z,\
    \partial_t \theta +\boldsymbol{u}\cdot\nabla \theta +w &= \nabla^{2} \theta,\
    \partial_t s +\boldsymbol{u}\cdot\nabla s +w &= \tau\nabla^{2} s
\end{align}
$$

with dimensionless parameters

$$
Ra=\frac{g\beta_T \Delta T^{*} H^{3}}{\kappa_T\nu}, \quad Pr=\frac{\nu}{\kappa_T}, \quad \tau=\frac{\kappa_S}{\kappa_T}, \quad R_\rho=\frac{\beta_T\Delta T^{*}}{\beta_S\Delta S^{*}}
$$

and constraints for homogeneous mode:

$$\int\int\int \boldsymbol{u} = \int\int\int \theta = \int\int\int s = 0$$

`params`: `Ra`, `Pr`, `tau`, `Rrho`. Optional flags (default `False`):
- `'stokes'`: drop $\partial_t \boldsymbol{u}$ and $\boldsymbol{u}\cdot\nabla\boldsymbol{u}$ (infinite-Prandtl, Stokes momentum balance);
- `'temperature_feedback'`: replace the temperature equation by the steady balance $w = \nabla^2\theta$.

Flow properties (`model.get_flow_properties()`): fluxes `Ft`, `Fs`, Nusselt and Sherwood numbers `Nu`, `Sh`, viscous dissipation `D`, and kinetic-energy input `I` (with its temperature and salinity parts `I_t`, `I_s`).

```python
from physics import SaltFinger
params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'init_dt': 2e-4, 'ext': '.nc'}
model = SaltFinger(params=params, sizes=(128, 128), bounds=(0.6, 1.0))
model.build_problems()
```

## Thermohaline-shear instability `ThermohalineShear`
Thermohaline-shear instability following Radko (2019, JFM), in a fully periodic domain `bounded=False` with a background shear flow

$$\bar{U}(z) = \sin(2\pi z)\,\mathbf{e}_x .$$

Governing equations for the fluctuations, as implemented:

$$
\begin{align}
    \nabla\cdot \boldsymbol{u} &= 0,\
    \partial_t \boldsymbol{u} + \bar{U}\partial_x\boldsymbol{u} + w\,\bar{U}'\,\mathbf{e}_x + \boldsymbol{u}\cdot\nabla\boldsymbol{u} &= -\nabla p + \frac{Pr}{Pe}\nabla^{2}\boldsymbol{u} + \frac{4\pi^2 Ri}{\Lambda-1}\left(\theta - s\right)\mathbf{e}_z,\
    \partial_t \theta + \bar{U}\partial_x\theta + \boldsymbol{u}\cdot\nabla \theta - w &= \frac{1}{Pe}\nabla^{2} \theta,\
    \partial_t s + \bar{U}\partial_x s + \boldsymbol{u}\cdot\nabla s - \Lambda w &= \frac{\tau}{Pe}\nabla^{2} s
\end{align}
$$

with the same zero-mean constraints on $\boldsymbol{u}$, $\theta$ and $s$.

`params`: `Pe`, `Ri`, `Pr`, `Lambda`, `tau`. Optional flags (default `False`):
- `'stokes'`: Stokes momentum balance (no $\partial_t\boldsymbol{u}$, shear advection or $\boldsymbol{u}\cdot\nabla\boldsymbol{u}$);
- `'infinite-Pr'`: infinite-Prandtl momentum balance with $Ra = 4\pi^2 Ri\,Pe^2/(Pr(\Lambda-1))$;
- `'zero-temp-feedback'`: linear temperature equation without advection;
- `'fully-zero-temp-feedback'`: steady temperature balance $w = -\nabla^2\theta/Pe$.

```python
from physics import ThermohalineShear
params = {'Pe': 100.0, 'Ri': 1.0, 'Pr': 7.0, 'Lambda': 2.0, 'tau': 0.01, 'init_dt': 1e-4}
model = ThermohalineShear(params=params, sizes=(128, 128), bounds=(1.0, 1.0))
model.build_problems()
```

## Models not yet validated
These classes exist in `physics/double_diffusion.py` and `physics/magneto.py` but have not been checked against published results. Use them with care.

| Model | `params` | Known issues |
|---|---|---|
| `BoundedSaltFinger` (`bounded=True`) | `Ra`, `Pr`, `tau`, `Rrho`, `Ri`, optional `stress-free` | `stress-free=True` fails to build (`ux` undefined) |
| `DiffusiveConvection` | `Ra`, `Pr`, `tau`, `Lambda` | fails to build (`self.sigma` undefined in its EVP) |
| `ShearedDiffusiveConvection` (periodic) | `Ra`, `Pr`, `tau`, `Lambda`, `Ri`, optional `stokes` | the commented draft below describes an older wall-bounded version |
| `WallShearedDiffusiveConvection` (`bounded=True`) | `Ra`, `Pr`, `tau`, `Lambda`, `Ri`, `stress-free` (required) | |
| `BoundedQuasiStaticMagnetoConvection` (`bounded=True`) | `Ra`, `Pr`, `Q`, optional `stress-free`, `scale_Ra_c` | |
| `BoundedQuasiStaticMagnetoConvection_SingleMode` (1-D, `bounded=True`) | `Ra`, `Pr`, `Q`, `kx`, `ky`, optional `stress-free`, `scale_Ra_c` | no $x$/$y$ symmetries or shifts |

Draft equations for some of these models are kept as comments in the source of this page.

<!-- 
## Diffusive convection `DiffusiveConvection`
This is unbounded diffusive convection `bounded=False`. Base state is:

$$\boldsymbol{U}_b=\boldsymbol{0},\quad T_b=S_b=-z$$

Nondimensional governing equations in terms of fluctuation using thermal diffusion scale:

$$
\begin{align}
    \nabla\cdot \boldsymbol{u} = 0,\\
    \partial_t \boldsymbol{u}+\boldsymbol{u}\cdot\nabla\boldsymbol{u} = -\nabla p + Pr\nabla^2\boldsymbol{u} + PrRa\left(\theta-\Lambda s\right)\mathbf{e}_z,\\
    \partial_t \theta +\boldsymbol{u}\cdot\nabla \theta -w = \nabla^2 \theta,\\
    \partial_t s +\boldsymbol{u}\cdot\nabla s -w = \tau\nabla^2 s
\end{align}
$$

with dimensionless parameters

$$Ra=\frac{g\beta_T \Delta
 T^{*} H^{3}}{\kappa_T\nu}, \quad Pr=\frac{\nu}{\kappa_T}, \quad \tau=\frac{\kappa_S}{\kappa_T}, \quad \Lambda=\frac{\beta_S\Delta S^{*}}{\beta_T\Delta T^{*}}$$

and constraints for homogeneous mode:

$$\int\int\int \boldsymbol{u} = \int\int\int \theta=\int\int\int s = 0$$

## Sheared diffusive convection `ShearedDiffusiveConvection`
This is vertical-bounded diffusive convection `bounded=True`. Base state is:

$$\boldsymbol{U}_b=\left(z-\frac{L_z}{2}\right)\frac{1}{\sqrt{Ri}}\boldsymbol{e}_x,\quad T_b=S_b=-z$$

Nondimensional governing equations in terms of fluctuation using free-fall velocity scale $U_{ff}=\sqrt{g\beta_T\Delta T^{*}H}$:

$$
\begin{align}
\nabla\cdot \boldsymbol{u} = 0,\\
\partial_t \boldsymbol{u} + U_b \partial_x u + w\partial_z U_b \boldsymbol{e}_x+\boldsymbol{u}\cdot\nabla\boldsymbol{u} = -\nabla p + \sqrt{\frac{Pr}{Ra}}\nabla^2\boldsymbol{u} + \left(\theta-\Lambda s\right)\mathbf{e}_z,\\
\partial_t \theta + U_b\partial_x \theta +\boldsymbol{u}\cdot\nabla \theta -w = \frac{1}{\sqrt{PrRa}} \nabla^2 \theta,\\
\partial_t s + U_b\partial_x s +\boldsymbol{u}\cdot\nabla s -w = \frac{\tau}{\sqrt{PrRa}}\nabla^2 s
\end{align}
$$

with dimensionless parameters

$$
Ra=\frac{g\beta_T \Delta T^{*} H^{3}}{\kappa_T\nu}, \quad Pr=\frac{\nu}{\kappa_T}, \quad \tau=\frac{\kappa_S}{\kappa_T}, \quad \Lambda=\frac{\beta_S\Delta S^{*}}{\beta_T\Delta T^{*}}, \quad Ri=\frac{g\beta_T\Delta T^{*}H}{U_w^{*2}}
 $$


Boundary condition:
- No-slip (standard, `'stress-free'=False` in `params`) at bottom and top walls:

$$\boldsymbol{u}=\theta=s=0$$

- Stress-free (optional, `'stress-free'=True` in `params`)  at bottom and top sides:

$$w=\theta=s=0, \quad \partial_z(u,v) = 0$$

```python
from physics.double_diffusion import ShearedDiffusiveConvection 
params = {'Ra': 1e4, 'Pr': 7.0, 'tau': 0.01, 'Lambda': 2.0, 'Ri': 4.0,
          'stress-free': False,
          'odir':"sddc_noslip/"}
model = ShearedDiffusiveConvection(params=params,
                                   sizes=(128, 128),
                                   bounds=(2*np.pi, 1.0),
                                   bounded=True,
                                   mode='sim')
```

# Magnetoconvection
## Bounded quasi-static magnetoconvection `BoundedQuasiStaticMagnetoConvection`
This is vertical-bounded domain `bounded=True`.

$$
\begin{align}
\partial_t \boldsymbol{u} + \boldsymbol{u}\cdot\nabla\boldsymbol{u} = -\nabla p + \sqrt{\frac{Pr}{Ra}}\nabla^2\boldsymbol{u} + \theta\mathbf{e}_z + Q\sqrt{\frac{Pr}{Ra}}\boldsymbol{J}\times\mathbf{e}_z,\\
\partial_t \theta + \boldsymbol{u}\cdot\nabla \theta - w = \frac{1}{\sqrt{PrRa}} \nabla^2 \theta,\\
\nabla\cdot \boldsymbol{u} = 0,\\
\boldsymbol{J} = -\nabla\Phi + \boldsymbol{u}\times\mathbf{e}_z,\\
\nabla\cdot \boldsymbol{J} = 0,\\
\end{align}
$$

with dimensionless parameters

$$
Ra=\frac{g\beta_T \Delta T^{*} H^{3}}{\kappa_T\nu}, \quad Pr=\frac{\nu}{\kappa_T}, \quad Q=\frac{B_0^{2} H^{2}}{\rho\nu\mu\eta}
$$

Boundary condition:
- No-slip (standard, `'stress-free'=False` in `params`) at bottom and top walls:
$$\boldsymbol{u}=\theta=0, \quad \partial_z \Phi = 0$$

- Stress-free (optional, `'stress-free'=True` in `params`)  at bottom and top sides:

$$w=\theta=0, \quad \partial_z(u,v) = 0, \quad \partial_z \Phi = 0$$

In 2D case, $\Phi$ will be removed. -->