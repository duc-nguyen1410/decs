Currently, we have created two models: 

# Double-diffusive convection
Double-diffusive convection (DDC) requires velocity ($\mathbf{U}$), temperature ($T$), salinity ($S$), and pressure ($p$). We decompose the velocity, temperature, and salinity into a base state ($\mathbf{U}_b, T_b, S_b$) and fluctuations ($\mathbf{u}, \theta, s$). Fluctuations will be used as main computing variables in simulation, their visualizations, and in [ECS solver](find_ecs.md). The base state depends on the following specific problems. 

## Salt finger convection `SaltFinger`
This is unbounded salt finger convection `bounded=False`. Base state is:

$$\mathbf{U}_b=\mathbf{0},\quad T_b=S_b=z$$

Nondimensional governing equations in terms of fluctuation using thermal diffusion scale:

$$
\begin{align}
    \nabla\cdot \mathbf{u} = 0,\\
    \partial_t \mathbf{u}+\mathbf{u}\cdot\nabla\mathbf{u} = -\nabla p + Pr\nabla^2\mathbf{u} + PrRa\left(\theta-R_\rho^{-1} s\right)\mathbf{e}_z,\\
    \partial_t \theta +\mathbf{u}\cdot\nabla \theta +w = \nabla^2 \theta,\\
    \partial_t s +\mathbf{u}\cdot\nabla s +w = \tau\nabla^2 s
\end{align}
$$

with dimensionless parameters

$$
Ra=\frac{g\beta_T \Delta T^* H^3}{\kappa_T\nu}, \quad Pr=\frac{\nu}{\kappa_T}, \quad \tau=\frac{\kappa_S}{\kappa_T}, \quad R_\rho=\frac{\beta_T\Delta T^*}{\beta_S\Delta S^*}
$$

and constraints for homogeneous mode:

$$\int\int\int \mathbf{u} = \int\int\int \theta = \int\int\int s = 0$$


## Diffusive convection `DiffusiveConvection`
This is unbounded diffusive convection `bounded=False`. Base state is:

$$\mathbf{U}_b=\mathbf{0},\quad T_b=S_b=-z$$

Nondimensional governing equations in terms of fluctuation using thermal diffusion scale:

$$
\begin{align}
    \nabla\cdot \mathbf{u} = 0,\\
    \partial_t \mathbf{u}+\mathbf{u}\cdot\nabla\mathbf{u} = -\nabla p + Pr\nabla^2\mathbf{u} + PrRa\left(\theta-\Lambda s\right)\mathbf{e}_z,\\
    \partial_t \theta +\mathbf{u}\cdot\nabla \theta -w = \nabla^2 \theta,\\
    \partial_t s +\mathbf{u}\cdot\nabla s -w = \tau\nabla^2 s
\end{align}
$$

with dimensionless parameters

$$Ra=\frac{g\beta_T \Delta
 T^* H^3}{\kappa_T\nu}, \quad Pr=\frac{\nu}{\kappa_T}, \quad \tau=\frac{\kappa_S}{\kappa_T}, \quad \Lambda=\frac{\beta_S\Delta S^*}{\beta_T\Delta T^*}$$

and constraints for homogeneous mode:

$$\int\int\int \mathbf{u} = \int\int\int \theta=\int\int\int s = 0$$

## Sheared diffusive convection `ShearedDiffusiveConvection`
This is vertical-bounded diffusive convection `bounded=True`. Base state is:

$$\mathbf{U}_b=\left(z-\frac{L_z}{2}\right)\frac{1}{\sqrt{Ri}}\mathbf{e}_x,\quad T_b=S_b=-z$$

Nondimensional governing equations in terms of fluctuation using free-fall velocity scale:

$$
\begin{align}
\nabla\cdot \mathbf{u} = 0,\\
\partial_t \mathbf{u} + U_b \partial_x u + w\partial_z U_b \mathbf{e}_x+\mathbf{u}\cdot\nabla\mathbf{u} = -\nabla p + \sqrt{\frac{Pr}{Ra}}\nabla^2\mathbf{u} + \left(\theta-\Lambda s\right)\mathbf{e}_z,\\
\partial_t \theta + U_b\partial_x \theta +\mathbf{u}\cdot\nabla \theta -w = \frac{1}{\sqrt{PrRa}} \nabla^2 \theta,\\
\partial_t s + U_b\partial_x s +\mathbf{u}\cdot\nabla s -w = \frac{\tau}{\sqrt{PrRa}}\nabla^2 s
\end{align}
$$

with dimensionless parameters

$$
Ra=\frac{g\beta_T \Delta T^* H^3}{\kappa_T\nu}, \quad Pr=\frac{\nu}{\kappa_T}, \quad \tau=\frac{\kappa_S}{\kappa_T}, \quad \Lambda=\frac{\beta_S\Delta S^*}{\beta_T\Delta T^*}, \quad Ri=\frac{g\beta_T\Delta T^*H}{U_w^{*2}}
 $$


Boundary condition:
- No-slip (standard, `'stress-free'=False` in `params`) at bottom and top walls:

$$\mathbf{u}=\theta=s=0$$

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
\partial_t \mathbf{u} + \mathbf{u}\cdot\nabla\mathbf{u} = -\nabla p + \sqrt{\frac{Pr}{Ra}}\nabla^2\mathbf{u} + \theta\mathbf{e}_z + Q\sqrt{\frac{Pr}{Ra}}\mathbf{J}\times\mathbf{e}_z,\\
\partial_t \theta + \mathbf{u}\cdot\nabla \theta - w = \frac{1}{\sqrt{PrRa}} \nabla^2 \theta,\\
\nabla\cdot \mathbf{u} = 0,\\
\mathbf{J} = -\nabla\Phi + \mathbf{u}\times\mathbf{e}_z,\\
\nabla\cdot \mathbf{J} = 0,\\
\end{align}
$$

with dimensionless parameters

$$
Ra=\frac{g\beta_T \Delta T^* H^3}{\kappa_T\nu}, \quad Pr=\frac{\nu}{\kappa_T}, \quad Q=\frac{B_0^2 H^2}{\rho\nu\mu\eta}
$$

Boundary condition:
<!-- - No-slip (standard, `'stress-free'=False` in `params`) at bottom and top walls:
$$\mathbf{u}=\theta=0, \quad \partial_z \Phi = 0$$ -->

- Stress-free (optional, `'stress-free'=True` in `params`)  at bottom and top sides:
- 
$$w=\theta=0, \quad \partial_z(u,v) = 0, \quad \partial_z \Phi = 0$$

In 2D case, $\Phi$ will be removed.