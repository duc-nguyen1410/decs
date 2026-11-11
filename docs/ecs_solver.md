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

## Multiple shooting

`MultiShootECSSolver(model, params)` (`ecs_core/multishoot.py`) finds periodic orbits and relative periodic orbits like `ECSSolver`, with the period split into $m$ segments (`'n_segments'`, default 2) of time $\tau = T_p/m$, each with its own initial state:

$$\xi = [x_1, \dots, x_m, T_p, a_x, \dots],\qquad
r_i = x_{i+1} - F^{\tau}(x_i)\ (i < m),\qquad r_m = x_1 - \sigma S(a) F^{\tau}(x_m),$$

with the phase conditions of `ECSSolver` on $x_1$. It is a drop-in replacement: same `solve` arguments, same options, same output; $x_1$ is the saved solution and the stability is that of the full period. The initial segments are obtained by integrating `x0`.

```python
from ecs_core import MultiShootECSSolver
solver = MultiShootECSSolver(model, {'n_segments': 4, 'tol': 1e-9, 'trust_radius': 20.0, 'odir': 'po_ms/'})
xi, success, residual, norm, properties = solver.solve(x0=x0, Tsearch=True, Tp=Tp)
```

- **Cost.** One Jacobian-vector product integrates each segment once, one period in total, as in single shooting; in serial an iteration costs the same. The unknowns and the Krylov basis are $m$ times larger (`krylov_dim` vectors of $m N$ numbers).
- **When it helps.** Perturbations grow over $\tau$ instead of $T_p$, so long or strongly unstable orbits are better conditioned: fewer GMRES vectors and a larger basin of convergence, e.g. from rough recurrences of a turbulent run. For short, mildly unstable orbits single shooting is as good.
- **Residual and trust region.** The residual norm adds the mismatches of all $m$ segments, and `trust_radius` applies to the whole step $(dx_1, \dots, dx_m)$.
- **Limitations.** Segments are integrated one after another (no parallelism over segments yet). With `Continuation`, predicted points are re-split by integrating $x_1$.

With `n_segments=1` the solver is `ECSSolver` exactly (tested bit for bit on the regression cases).

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

## Edge tracking

`EdgeTracker(model, params)` (`ecs_core/edge.py`) follows the edge of chaos, the boundary between the basins of attraction of the laminar state and of turbulence, by repeated bisection (Skufca, Yorke & Eckhardt 2006). States are classified by the cross-flow energy $E_{cf} = \tfrac12\langle w^2\rangle$ (`model.cross_flow_energy(x)`): a state is integrated until $E_{cf} < E_{lo}$ (laminar) or $E_{cf} > E_{hi}$ (turbulent).

```python
from ecs_core import EdgeTracker
tracker = EdgeTracker(model, {'E_lo': 1e-8, 'E_hi': 1e-3, 'T_max': 50.0, 'dt_check': 1.0,
                              'delta_max': 1e-2, 'bisect_tol': 1e-6, 'odir': 'edge/'})
x_edge = tracker.track(x_lam, x_turb, T_track=200.0)
```

1. **Bisect** on the segment $x_a + \lambda(x_b - x_a)$ from a laminar state $x_a$ to a turbulent state $x_b$ until the bracket states are closer than `bisect_tol` (L2 norm `model.l2norm`).
2. **Follow** both bracket states, checking every `dt_check`, until they are further apart than `delta_max` or one leaves $E_{lo} < E_{cf} < E_{hi}$.
3. **Refine**: bisect again between the pair from the last check before that; repeat until time `T_track`.

| Key | Default | Meaning |
|---|---|---|
| `E_lo`, `E_hi` | required | $E_{cf}$ thresholds of the laminar / turbulent classification; the edge must lie between them |
| `T_max` | required | longest integration to classify one state; still undecided states are classified by $E_{cf}$ against $\sqrt{E_{lo}E_{hi}}$, with a warning |
| `dt_check` | required | time between separation checks while following the edge |
| `delta_max` | required | separation (L2 norm) that triggers a new bisection |
| `bisect_tol` | required | bracket width (L2 norm) at which a bisection stops, below `delta_max` |
| `bisect_in` | `'l2'` | bisection midpoint in the L2 norm of the perturbation $x - x_a$ (`'l2'`) or in its cross-flow energy (`'ecf'`) |
| `check_every` | `10` | classify every this many time steps |
| `odir` | `'./edge/'` | output directory |

Output in `odir`: `edge_history.csv` (`t, Ecf, L2norm` of the edge trajectory) and `edge_state` (latest edge state). A converged edge trajectory is a good initial guess for an edge state with `ECSSolver` (an equilibrium, traveling wave or periodic orbit with one unstable direction).

## Poincaré sections and recurrence search

`PoincareSection(model, observable, level, direction)` (`ecs_core/poincare.py`) records where a trajectory crosses $g(x) = $ `level`, with $g$ an observable of the current model state, counted only when $g$ increases (`direction=+1`), decreases (`-1`) or both (`0`). Each crossing is detected between two time steps and refined by regula falsi: integrate from the last state before the crossing until $g$ equals `level` to a relative `tol`. `recurrence_search` then compares the crossings pairwise to find near-returns, which give initial guesses for `ECSSolver`.

```python
from ecs_core import PoincareSection, recurrence_search
section = PoincareSection(model, lambda m: m.cross_flow_energy(), level=E_mean, direction=1)
crossings = section.run(x0, T=50.0, odir='section/')      # [{'t', 'x', 'g'}, ...]
best = recurrence_search(model, [c['x'] for c in crossings], [c['t'] for c in crossings],
                         shift_axes=('x',), T_max=5.0)       # RPO drifting in x
r = best[0]
solver.solve(x0=crossings[r['i']]['x'], Tsearch=True, Rxsearch=True, Tp=r['Tp'], ax=r['ax'])
```

| Argument | Default | Meaning |
|---|---|---|
| `observable` | required | `function(model) -> float` of the **current** state. It runs between time steps, so it must not change the layout of `model.fields`: evaluate on field copies, as `cross_flow_energy()` and `current_state()` do (`get_state()` is not safe there) |
| `level`, `direction` | `0.0`, `+1` | the section $g = $ `level` and the crossing direction |
| `check_every` | `1` | test for a crossing every this many time steps |
| `refine`, `tol`, `max_refine` | `True`, `1e-10`, `20` | refine crossings to $\lvert g - \text{level}\rvert \le$ `tol` $\cdot\max(\lvert\text{level}\rvert, \lvert\Delta g\rvert)$; `refine=False` interpolates linearly between the two steps |

`run(x0, T, max_crossings=None, odir=None)` integrates up to `T` (or until `max_crossings`) and, with `odir`, writes `crossings.csv` (`i, t, g`) and the states `crossing_<i>`.

`recurrence_search(model, states, times, shift_axes=(), T_min=0, T_max=inf, n_best=5)` returns the `n_best` pairs $j > i$ sorted by

$$d_{ij} = \min_{a} \frac{\lVert S(a)\,x_j - x_i\rVert}{\lVert x_i\rVert},\qquad S(a) f(x) = f(x + a L),$$

as dicts with `i`, `j`, `Tp` $= t_j - t_i$, `distance` and the shifts (`ax`, `az`, ...). $S(a)$ is the shift convention of `ECSSolver`, so `Tp` and the shifts are its starting values directly. The minimum over shifts uses an FFT over the periodic `shift_axes`; the shift is refined between grid points by a parabola, while `distance` is that at the best grid shift. `shift_axes=()` compares without shifts (periodic orbits). Every multiple of a period is a recurrence too, so bound the search with `T_max`. A section of a symmetric orbit can be crossed several times per period (e.g. twice for PO3, whose Ecf has period $T_p/2$); the recurrence search still finds the true period. `shifted_distance(model, x, y, shift_axes)` gives the minimum for one pair.

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
