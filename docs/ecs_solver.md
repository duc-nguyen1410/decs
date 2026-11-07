# ECS solver

`ecs_core` finds exact coherent structures (ECS) of any model in `physics/` with a Newton–Krylov method, analyses their linear stability, and continues them in a parameter.

```python
from ecs_core import ECSSolver, Continuation
```

## The problem being solved

Let $F^{T}(x)$ be the flow map: the state $x$ (all model fields, flattened) integrated forward over time $T$. An ECS satisfies

$$x = \sigma\, S(a_x, a_y, a_z)\, F^{T}(x)$$

where $S$ shifts the state by $(a_x L_x, a_y L_y, a_z L_z)$ and $\sigma$ is an optional imposed symmetry. The unknowns are

$$\xi = [x,\ (T),\ (a_x),\ (a_y),\ (a_z)]$$

where the period and shifts are included only when searched. Each searched parameter adds one phase condition, which removes the neutral direction (time or space translation) that would otherwise make the solution non-unique.

| Solution | Searched unknowns |
|---|---|
| Equilibrium (EQ) | $x$ only; $T$ is a fixed integration time |
| Traveling wave (TW) | $x$, one or more shifts $a$ |
| Periodic orbit (PO) | $x$, $T$ |
| Relative periodic orbit (RPO) | $x$, $T$, one or more shifts $a$ |

Each Newton step solves $J\,\delta\xi = -r$ with GMRES, using only matrix-free products $J v$ (finite-difference time integrations), and limits the step with a hookstep trust region.

## `ECSSolver(model, params)`

`model` is a built model (after `model.build_problems()`). Options in `params`, with defaults:

| Key | Default | Meaning |
|---|---|---|
| `odir` | `'ecs_output/'` | output directory |
| `tol` | `1e-6` | convergence: Newton residual norm $\lVert x - G(x)\rVert$ |
| `max_iter` | `20` | maximum Newton iterations |
| `krylov_dim` | `50` | maximum Krylov vectors per Newton step |
| `krylov_dim_min` | `20` | minimum Krylov vectors before GMRES may stop on `gmres_min_error` |
| `gmres_min_error` | `1e-20` | GMRES stopping threshold (squared linear-model residual) |
| `trust_radius` | `1.0` | initial trust radius of the hookstep |
| `trust_radius_max` | `trust_radius` | largest radius the trust region may grow back to |
| `trust_radius_min` | `1e-4` | smallest radius |
| `eps_x` | `1e-7` | finite-difference step for $J v$ |
| `eps_t` | `1e-6` | time step for $\partial_t x$ (period phase condition) |
| `computeStability` | `False` | run stability analysis after convergence |
| `stability_method` | `'arnoldi'` | `'arnoldi'` (any solution) or `'evp'` (equilibria and traveling waves, see below) |
| `Neigen` | `50` | Arnoldi iterations; with `'evp'`, number of eigenvalues returned |
| `evp_target` | `0.0` | with `'evp'`: the sparse eigenvalue search returns eigenvalues nearest this value (keep 0, see below) |
| `projectNeutralDrift` | `False` | Arnoldi: project neutral (shift / time) directions out of the analysis; EVP: drop the modes that are neutral shifts of the base state and return `Neigen` other modes |
| `save_ecs_history` | `False` | keep the iterate of every Newton step (`solution_iter_<i>`) instead of overwriting `solution` |
| `log_filename` | `'output.log'` | run log name |

**Trust region.** If a step does not reduce the residual, the radius is halved and the step recomputed. For the next Newton step the radius is updated from the ratio of actual to predicted residual reduction: doubled (up to `trust_radius_max`) after a good step that was limited by the radius, halved after a poor one, otherwise kept. Each step is logged as `Trust region: used ..., actual/predicted reduction ..., next ...`.

### `solver.solve(x0, Tsearch=False, Rxsearch=False, Rysearch=False, Rzsearch=False, Tp=20.0, ax=0.0, ay=0.0, az=0.0)`

- `x0`: initial state, e.g. `model.get_state()` after `model.load_state(...)`.
- `Tp`: integration time for equilibria, initial period for orbits.
- `ax`, `ay`, `az`: initial shifts as fractions of the domain length (used when the matching `R*search` is on).

Returns `(xi, success, residual, norm, properties)`: the final unknowns, whether `tol` was reached, the residual norm, $\lVert x\rVert$, and a dict of searched parameters plus the model's flow properties. Use `solver.unpack_xi(xi)` to get `(x, Tp, ax, ay, az)`.

## Imposed symmetries

Restricting the search to a symmetry subspace removes neutral directions and usually speeds up convergence. Set `solver.sigma` before calling `solve`:

```python
solver.sigma.load_from_file('examples/symmetry_pixz.sym')
# or: from physics import Symmetry; solver.sigma = Symmetry(sx=-1, sz=-1)
```

A `.sym` file holds one line `s sx sy sz ax ay az` for $u(x,y,z) \to s\,u(s_x x + a_x,\ s_y y + a_y,\ s_z z + a_z)$:

| File | Line | Symmetry |
|---|---|---|
| [symmetry_pix.sym](../examples/symmetry_pix.sym) | `1 -1 1 1 0 0 0` | reflection $x \to -x$ |
| [symmetry_pixz.sym](../examples/symmetry_pixz.sym) | `1 -1 1 -1 0 0 0` | reflections $x \to -x$ and $z \to -z$ |

Reflections flip the sign of the matching velocity component; a $z$-reflection also flips the sign of scalar fields. The global parity `s` is stored but not currently applied.

## Stability analysis

With `computeStability=True`, a converged solution is analysed automatically (or call `solver.stability(xi)` yourself). The analysis is skipped unless the residual is below `1e-6`, and the imposed symmetry is not applied to perturbations. Two methods are available (`stability_method`); both write growth rates $\lambda$ and multipliers $\mu = e^{\lambda T}$ sorted from most unstable, plus the unstable modes.

### Arnoldi (`'arnoldi'`, default)

Arnoldi iteration on the linearised flow map gives the Floquet multipliers $\mu$ and growth rates $\lambda = \ln(\mu)/T$. It works for every solution type (it is the only option for periodic and relative periodic orbits), is matrix-free and runs in parallel. It only resolves the leading multipliers well. When many modes decay very slowly (e.g. salinity with small $\tau/Pe$), their multipliers cluster near 1 and `Neigen` Arnoldi vectors may not resolve them: check `residual.mtx`, or cross-check with the EVP.

### Eigenvalue problem (`'evp'`)

> **Use the EVP on low-resolution grids only (up to about 48², 64² at most), for testing and for cross-checking Arnoldi.** For a base state that varies in $x$, its memory grows like $(N_x N_z)^2$: at production resolutions (e.g. 128², about 100 GB) it fails with a `MemoryError`. `ECSSolver` detects this beforehand and falls back to Arnoldi; a direct `model.solve_evp` call raises `EVPTooLargeError`. To check a high-resolution solution, interpolate it to 32²–48², reconverge it there, and compare the EVP with Arnoldi at that grid (see the table below).

For an equilibrium, or a traveling wave in its co-moving frame, the growth rates are the eigenvalues of the linearised equations,

$$\sigma\, \mathcal{M} q + \mathcal{L}(x_{eq})\, q = 0, \qquad q \propto e^{\sigma t},$$

solved directly with a Dedalus eigenvalue problem. There is no time-stepping or finite-difference error, and shift-invert can find eigenvalues anywhere in the spectrum. For a traveling wave the frame velocity is $c = a L / T$ for each searched shift $a$. Available for `SaltFinger` and `ThermohalineShear` (any of their options). The same solver is available directly:

```python
growth_rates, modes = model.solve_evp(x_eq, N=10, target=6.0)        # equilibrium
growth_rates, modes = model.solve_evp(x_tw, N=10, frame_velocity=(cx, 0.0))   # traveling wave
```

How the EVP is solved depends on the base state:

| Base state | EVP | MPI | Cost |
|---|---|---|---|
| independent of $x$ (rest state, laminar shear) | one small EVP per horizontal wavenumber, solved densely (all eigenvalues) | yes, wavenumbers are spread over the processes | cheap |
| varies in $x$ (most ECS) | one coupled EVP for all modes | works in an MPI run, but **rank 0 does all the work**: Dedalus cannot couple directions that are split across processes, so rank 0 solves it on a single-process copy of the model and broadcasts the result while the other ranks wait | large: dense up to `dense_limit` (2000) unknowns, otherwise a sparse search for the `N` eigenvalues nearest `evp_target` |

For a coupled EVP, the sparse search finds the `N` (`Neigen`) eigenvalues **nearest `evp_target`** and returns them sorted by growth rate. Keep the default `evp_target = 0`. The leading modes of an ECS lie close to 0 relative to the bulk of the spectrum, which extends to large negative growth rates. With `Neigen = 60` at 32², the 60 eigenvalues nearest 0 reach down to $\mathrm{Re}\,\lambda \approx -48$ and contain all the leading modes that Arnoldi finds. A target far from the cluster converges much more slowly: with 60 modes at 32², `evp_target = 0` took 39 s, while 0.5 and 6 did not finish within 300 s.

**Neutral modes.** An ECS that is periodic in $x$ (and $z$) has neutral shift modes $\partial_x \mathbf{x}_{eq}$, $\partial_z \mathbf{x}_{eq}$ with $\lambda \approx 0$. With `projectNeutralDrift=True`, each EVP mode is compared with these directions: a mode is dropped (and logged) if more than 90 % of its norm lies in their span, and a few extra modes are searched so that `Neigen` modes are still returned. Directly: `model.solve_evp(x_eq, N, exclude=[...], exclude_overlap=0.9)`.

**Memory.** The base state couples almost every mode, so the factorized matrix of a coupled EVP is nearly dense. With $n \approx 5 N_x N_z$ unknowns in 2-D, it takes about $16 n^2$ bytes:

| Grid | unknowns $n$ | peak memory | time (`Neigen = 60`) |
|---|---|---|---|
| 32² | 5 125 | 0.7 GB (measured) | 64 s |
| 48² | 11 525 | 2.2 GB (measured) | 266 s |
| 64² | ~20 500 | ~7 GB (estimate) | ~15 min |
| 128² | ~82 000 | ~100 GB (estimate) | infeasible |

Before building anything, `solve_evp` estimates this and raises `EVPTooLargeError` if it exceeds `max_memory` (default: 80 % of the memory available). Inside `ECSSolver`, `stability_method='evp'` then logs a warning and falls back to Arnoldi, so a long Newton run still gets its stability results.

A parallel Newton run can therefore keep `stability_method='evp'`. The extra processes don't speed up the coupled EVP, though, and rank 0 also holds an extra copy of the model and the EVP matrices. For large problems you may prefer a separate single-process job on the saved solution (`model.load_state(...)`, then `model.solve_evp(model.get_state(), ...)`).

The EVP evaluates the nonlinear terms as exact spectral products with the base state, while the IVP (and so Newton and Arnoldi) uses dealiased products on the grid. A converged ECS is therefore exactly steady for the IVP but only approximately for the EVP, and the two methods agree to the spatial resolution error. Example, `SaltFinger` WF equilibrium ($Ra=10^5$, $L_x=0.6$):

| Grid | EVP leading $\lambda$ | Arnoldi leading $\lambda$ | EVP shift (neutral) mode | coupled EVP time |
|---|---|---|---|---|
| 32² | 5.38919 | 5.38891 | 0.0034 | 25 s |
| 48² | 5.38855 | 5.38853 | 0.00006 | 191 s |

A shift mode noticeably away from 0 indicates an under-resolved base state. The coupled EVP is practical up to roughly 48²–64² in 2-D; at higher resolution use Arnoldi, and the EVP on the base state interpolated to 48² as a cross-check.

## Parameter continuation

```python
cont = Continuation(solver, {'mu_name': 'Lx', 'odir': 'ContLx/', 'Tp': 0.2})
model.load_state('solution')     # converged starting solution
cont.run_continuation(mu_start=0.6, dmu=0.005, n_steps=500, mu_target=0.3)
```

`run_continuation(mu_start, dmu, n_steps=50, mu_target=None)` first solves at `mu_start`, `mu_start + dmu` and `mu_start + 2*dmu`. It then predicts each next point from the branch history, adjusts the step until the residual of the prediction is acceptable, corrects it with `solve`, and stops after `n_steps` points or once $\mu$ crosses `mu_target`.

| Key | Default | Meaning |
|---|---|---|
| `mu_name` | `'Lx'` | continued parameter: a model parameter (e.g. `'Ra'`) or `'Lx'`, `'Ly'`, `'Lz'`, `'Aspect'` |
| `continuation_type` | `'arclength'` | `'arclength'` or `'natural'` |
| `predictor` | `'tangent'` | arclength predictor: `'tangent'` (secant) or `'quadratic'` (last 3 points) |
| `odir` | `'./'` | output directory |
| `Tsearch`, `Rxsearch`, `Rysearch`, `Rzsearch` | `False` | solution type, as in `solve` |
| `Tp`, `ax`, `ay`, `az` | `1.0`, `0`, `0`, `0` | initial period / shifts |
| `mu_ref` | `1` | scale of $\mu$ in the arclength metric |
| `ds_min`, `ds_max` | `1e-4`, `0.01` | step-size bounds (dimensionless arclength) |
| `guess_error_min`, `guess_error_max` | `0.1`, `10` | accepted residual range of a predicted point |
| `Ndsadjust` | `5` | maximum step adjustments per point |

The arclength metric is $ds^2 = (\lVert\Delta x\rVert/\lVert x\rVert)^2 + (\Delta\mu/\max(|\mu|, |\mu_{ref}|))^2$. In natural continuation the step in $\mu$ is $ds \cdot \max(|\mu|, |\mu_{ref}|)$. A physical parameter can only be continued if the model stores it as a Dedalus field named `<name>_param` (e.g. `Ra_param`); geometry parameters rebuild the domain.

### Restarting a continuation

A stopped or crashed continuation can be resumed from its output directory. Create the model, solver and `Continuation` exactly as in the original run (same `odir`, `mu_name` and searched parameters), then call `restart_continuation` instead of `run_continuation`:

```python
cont = Continuation(solver, {'mu_name': 'Lx', 'odir': 'ContLx/', 'Tp': 0.2})
cont.restart_continuation(n_steps=200, mu_target=0.3)   # up to 200 more points
```

`restart_continuation(n_steps=50, mu_target=None)` reloads the converged states of the last three points from their `search-<i>/solution` files and continues numbering at the next `search-<i>`. Here `n_steps` counts new points only.

After every accepted point the continuation writes `odir/continuation_state.json` (step size, branch history and folders). With this file the restart continues exactly as an uninterrupted run would. Runs made before this file existed are restarted from `odir/flow_properties.csv` instead, with the step size estimated from the last step, so they continue along the same branch with slightly different step sizes. At least three converged points are needed.

## Output files

In `odir` of a Newton solve:

| File | Content |
|---|---|
| `output.log` | parameters and residual of every Newton iteration |
| `flow_properties.csv` | searched parameters and flow properties of every iterate |
| `solution.<ext>` | current iterate (the converged solution at the end) |
| `T.txt`, `ax.txt`, ... | current period / shifts |
| `tol.txt` | current residual |
| `snapshots/`, `flowproperties/`, `meanprofiles/` | time history of a converged TW / PO / RPO, if `model.save_snapshots` etc. are on |
| `stability/eigenvalues.mtx`, `growthrate.mtx`, `residual.mtx` | Floquet multipliers, growth rates, Ritz residuals (Matrix Market) |
| `stability/unstable_mode_<i>.<ext>` | unstable eigenmodes |

Each Newton solve starts a new `flow_properties.csv` and `output.log` in its `odir`.

`<ext>` is the model's `params['ext']`, by default `.nc` (NetCDF), the preferred format: solutions and eigenmodes open directly in ParaView and post-process quickly with `netCDF4` / `xarray`. `'.h5'` (HDF5) is the alternative.

Continuation writes each point to `odir/search-<i>/` (the files above plus `mu.txt`), a branch summary `odir/flow_properties.csv` with $\mu$, the flow properties and the search directory of every point, and the restart checkpoint `odir/continuation_state.json`.
