import numpy as np
import dedalus.public as de
from mpi4py import MPI
import gc
import matplotlib.pyplot as plt
from .symmetry import Symmetry, SymmetryOps
from .state_io import StateIO
import os
import time
import logging
logger = logging.getLogger(__name__)


class EVPTooLargeError(MemoryError):
    """The eigenvalue problem would not fit in memory (raised before anything is built)."""


def available_memory():
    """Memory available now, in bytes (Linux /proc/meminfo, else total physical memory)."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1])*1024
    except OSError:
        pass
    try:
        return os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        return float('inf')


class FluidModel(StateIO, SymmetryOps):
    def __init__(self, params, sizes, bounds, bounded=False, mode='ecs', dealias=3/2, comm=None):
        """
        :param float sizes: Grid mesh of domain (Nx, Nz) or (Nx, Ny, Nz)
        :param float bounds: Domain size (Lx, Lz) or (Lx, Ly, Lz)
        :param bool bounded: Is this a domain bounded in vertical (z) direction? If yes, model will use Chebyshev grid instead of Fourier mode
        :param str mode: The model will build dist with dtype=np.complex128 and bases with de.ComplexFourier as defaults to impose ECS's symmetry. However, CFL(u) will loss stability during long-term simulation due to ComplexFourier. Therefore, you need to set mode='sim' to use dtype=float64 and RealFourier instead, to get stability of CFL(u) operator for long-time simulation. 
        :param float dealias: Grid will be scaled for enhancing convective term
        :param comm: MPI communicator of the model (default MPI.COMM_WORLD)
        """
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.params = params
        self.sizes = sizes
        self.bounds = bounds
        self.dim = len(sizes)
        self.bounded = bounded
        self.mode = mode
        self.dealias = dealias
        self.coords = None
        self.dist = None
        self.bases = None

        self.init_dt = (params or {}).get('init_dt', 2e-4)
        self.odir = (params or {}).get('odir', "sim_output/")
        self.ext = (params or {}).get('ext', '.nc')

        # Registry: Newton solver will see [u, ...]
        self.fields = []
        self.ivp_problem = None
        self._ivp_cache = None       # (ivp_problem, IVP solver) reused by flow_map / t_derivative
        # For EVP
        self.eq_fields = []          # base state of the EVP (same layout as self.fields)
        self.eq_fields_z = None      # z-only copies, for base states that do not vary horizontally
        self.evp_problem = None
        self._evp_problems = {}      # coupled (True/False) -> EVP built by build_evp_problem
        self.frame_velocity = None   # constant vector field: velocity of the co-moving frame (TW)
        self._evp_serial = None      # (key, single-process copy) for coupled EVPs under MPI

        # CFL function
        self.use_CFL = (params or {}).get('use_CFL', False)
        self.CFL = None
        self.CFL_cadence = (params or {}).get('cadence', 10)
        self.CFL_safety = (params or {}).get('safety', 0.5)
        self.CFL_threshold = (params or {}).get('threshold', 0.05)
        self.CFL_min_change = (params or {}).get('min_change', 0.5)
        self.CFL_max_change = (params or {}).get('max_change', 1.5)
        self.CFL_max_dt = (params or {}).get('max_dt', 0.1)

        # Output options
        self.save_snapshots = (params or {}).get('save_snapshots', False)
        self.save_flowproperties = (params or {}).get('save_flowproperties', False)
        self.save_meanprofiles = (params or {}).get('save_meanprofiles', False)

        # Preview current state
        self.preview_fig = None
        self.preview_ax = None
        self.preview_im = None
    
    def set_param(self, name, value):
        """ Update a parameter and rebuild grid if needed. """
        logger.info(f"Setting parameter {name} = {value}")

        # Define geometry parameters that control domain bounds
        GEOMETRY_PARAMS = {'Lx', 'Ly', 'Lz', 'Aspect'}

        # Geometry / Domain Parameters (Requires Rebuilding Domain & Problem)
        if name in GEOMETRY_PARAMS:
            bounds_list = list(self.bounds)

            if name == 'Lx':
                bounds_list[0] = value
            elif name == 'Ly' and self.dim == 3:
                bounds_list[1] = value
            elif name == 'Lz':
                bounds_list[-1] = value
            elif name == 'Aspect':
                lz = bounds_list[-1]
                bounds_list[0] = value * lz
                if self.dim == 3:
                    bounds_list[1] = value * lz

            # Apply new bounds tuple
            self.bounds = tuple(bounds_list)

            # Re-initialize Dedalus spatial grid and bases
            self.create_domain()
            self.build_fields()

            # Clear references and force full rebuild for domain changes only
            self.ivp_problem = None
            gc.collect()
            self.comm.Barrier()
            
            self.build_problems()
            logger.info(f"Geometry parameter '{name}' updated to {value}. Full domain & problem rebuilt.")

        # Dynamic Physical Parameters (Updated IN-PLACE without rebuilding!)
        elif name in self.params:
            self.params[name] = value
            setattr(self, name, value)  # Update attribute so Dedalus equations see it
            # update parameter inside IVP problem directly
            attr_name = f"{name}_param"
            if hasattr(self, attr_name):
                param_field = getattr(self, attr_name)
                param_field['g'] = value
                self._ivp_cache = None  # parameter fields are built into the solver's matrices
                logger.info(f"Parameter '{name}' updated to {value}")
            else:
                logger.info(f"Parameter '{name}' doesn't exist inside model!")

        # Fallback for other existing instance attributes
        elif hasattr(self, name):
            setattr(self, name, value)
        else:
            raise KeyError(f"Parameter '{name}' is unknown (not in self.params or model attributes).")
    
    def get_grid_shape(self, dealias=False):
        """
        Returns (Nx, Nz) or (Nx, Ny, Nz) scaled by dealias of current domain
        """
        return tuple(basis.global_grid(self.dist, scale=self.dealias if dealias else 1).shape[i] 
                 for i, basis in enumerate(self.bases))
    
    def size(self, dealias=False):
        """
        Return total freedom elements on dealias-scaled grid data of all fields 
        """
        grid_shape = self.get_grid_shape(dealias=dealias)
        points_per_field = np.prod(grid_shape) # np.prod(grid_shape) gives Nx*Nz or Nx*Ny*Nz
        total_size = 0
        for field in self.fields:
            # Vector fields have 'dim' components (len > 0), Scalars do not
            num_components = self.dim if len(field.tensorsig) > 0 else 1
            # In 2D, num_components is 2; in 3D, it is 3
            total_size += num_components * points_per_field
        return total_size
    
    def get_state(self, dealias=False):
        """
        Return a vector of current state collected from all fields
        """
        data_slices = []
        for field in self.fields:
            # Gather from MPI processes and flatten
            if not dealias: field.change_scales(1.0)
            gdata = field.allgather_data('g').real
            data_slices.append(gdata.ravel())
        return np.concatenate(data_slices)
    
    def set_state(self, state_vector):
        """
        Load a state vector to global grid data of each field
        """
        grid_shape = self.get_grid_shape()
        points_per_field = int(np.prod(grid_shape))
        cursor = 0
        for field in self.fields:
            # Determine if we are dealing with a Vector (dim components) or Scalar (1)
            num_components = self.dim if len(field.tensorsig) > 0 else 1
            size = num_components * points_per_field
            # Reshape logic: (components, Nx, Nz) or (components, Nx, Ny, Nz)
            if num_components > 1:
                reshape_to = (num_components,) + grid_shape
            else:
                reshape_to = grid_shape
            # print(f"Setting field '{field.name}' with size {size} and reshape {reshape_to}")
            data = state_vector[cursor:cursor+size].reshape(reshape_to)
            field.load_from_global_grid_data(data)
            cursor += size

    def set_eq_state(self, state_vector):
        """
        Load a state vector to global grid data of each sub-field as a base state in eigenvalue problem
        """
        grid_shape = self.get_grid_shape()
        points_per_field = int(np.prod(grid_shape))
        cursor = 0
        
        for field in self.eq_fields:
            num_components = self.dim if len(field.tensorsig) > 0 else 1
            size = num_components * points_per_field
            
            if num_components > 1:
                reshape_to = (num_components,) + grid_shape
            else:
                reshape_to = grid_shape
                
            data = state_vector[cursor:cursor+size].reshape(reshape_to)
            field.load_from_global_grid_data(data)
            cursor += size


    def set_initial_conditions(self,mode = 'random', scale=1e-3):
        """ Set initial condition for each field """
        if mode == 'random':
            for field in self.fields:
                field.fill_random('g', seed=42, distribution='normal', scale=scale) # Random noise
        elif mode == 'vertical_sin':
            z = self.bases[-1].local_grid(self.dist, scale=self.dealias)
            for i, field in enumerate(self.fields):
                if i==0: # velocity
                    field.fill_random('g', seed=42, distribution='normal', scale=1e-2)
                else: # scalar fields
                    field['g'] = -scale*np.sin(2.0*np.pi*1*z)
        else:
            raise ValueError("Invalid mode for initial conditions")
    def add_perturbation(self,scale=1e-3):
        """ Add a perturbation to each field """
        for field in self.fields:
            field['g'] += scale * np.random.standard_normal(field['g'].shape)

    def set_cfl(self, solver):
        """ Set up the CFL condition for adaptive time-stepping. """
        self.CFL = de.CFL(solver, initial_dt=self.init_dt, cadence=self.CFL_cadence, safety=self.CFL_safety, threshold=self.CFL_threshold, 
                          max_change=self.CFL_max_change, min_change=self.CFL_min_change, max_dt=self.CFL_max_dt)
        self.CFL.add_velocity(self.fields[0]) # Assuming the first field is velocity; adjust if needed

    # ------------------------------------------------------------------
    # Linear stability from the eigenvalue problem (EVP)
    # ------------------------------------------------------------------
    def evp_base_fields(self, coupled):
        """Base-state fields for build_evp_problem: the full eq_fields (coupled EVP), or
        z-only copies (one small EVP per horizontal wavenumber)."""
        if coupled:
            return self.eq_fields
        if self.eq_fields_z is None:
            self.eq_fields_z = [
                self.dist.VectorField(self.coords, name=f.name + '_z', bases=self.bases[-1])
                if len(f.tensorsig) > 0 else self.dist.Field(name=f.name + '_z', bases=self.bases[-1])
                for f in self.eq_fields]
        return self.eq_fields_z

    def evp_frame_velocity(self):
        """Constant vector field c for the co-moving-frame term -c.grad(q) of an EVP."""
        if self.frame_velocity is None:
            self.frame_velocity = self.dist.VectorField(self.coords, name='c_frame')
        return self.frame_velocity

    def varies_horizontally(self, x, rtol=1e-10):
        """True if the state x depends on x (or y in 3-D)."""
        scale = max(np.linalg.norm(x), 1e-300)
        derivs = [self.x_derivative(x)] + ([self.y_derivative(x)] if self.dim == 3 else [])
        return any(np.linalg.norm(d) > rtol*scale for d in derivs)

    def evp_unknowns(self, coupled):
        """Number of unknowns of the EVP matrix: all modes (coupled) or one horizontal wavenumber."""
        if coupled not in self._evp_problems:
            self._evp_problems[coupled] = self.build_evp_problem(coupled=coupled)
        n = 0
        for var in self._evp_problems[coupled].variables:
            components = int(np.prod([cs.dim for cs in var.tensorsig]))
            n += components*int(np.prod([b.size for b in var.domain.bases]))
        if not coupled:
            n //= int(np.prod(self.sizes[:-1]))
        return n

    def evp_memory_estimate(self, coupled):
        """Rough peak memory (bytes) of one EVP subproblem: a horizontally varying base state
        couples almost all modes, so the factorized matrix is nearly dense (~ n^2 entries)."""
        return self.EVP_BYTES_PER_ENTRY*self.evp_unknowns(coupled)**2

    EVP_BYTES_PER_ENTRY = 16   # calibrated on SaltFinger 32^2 / 48^2 (see docs/ecs_solver.md)

    def solve_evp(self, x_eq, N=10, target=0.0, frame_velocity=None, coupled=None, dense_limit=2000,
                  ncc_cutoff=1e-6, max_memory=None, exclude=None, exclude_overlap=0.9):
        """
        Linear stability of a steady base state x_eq from the model's eigenvalue problem
            sigma M q + L(x_eq) q = 0,      perturbations q ~ exp(sigma t).
        Works for equilibria, and for traveling waves in their co-moving frame
        (frame_velocity = (cx, [cy,] cz), with c = a*L/Tp for a wave shifted by a*L per Tp).

        coupled : couple all directions (needed when x_eq varies horizontally). Dedalus cannot
                  couple distributed directions, so under MPI a coupled EVP is solved by rank 0
                  alone, on a single-process copy of the model (the others wait).
                  None: coupled exactly when x_eq varies horizontally. Otherwise one small EVP
                  per horizontal wavenumber, spread over the MPI processes.
        Subproblems up to dense_limit unknowns are solved densely (all eigenvalues), larger
        ones with a sparse shift-invert search for the N eigenvalues nearest `target`.
        ncc_cutoff: relative size below which coefficients of the base state are dropped
        when building the matrices (Dedalus default).
        max_memory: memory (GB) the EVP may use; default 80% of the memory available now.
        A coupled EVP needs ~ n^2 storage (n ~ 5 Nx Nz in 2-D): if the estimate exceeds this,
        EVPTooLargeError is raised before anything is built (128^2 needs ~100 GB).
        exclude: state vectors (e.g. the neutral shifts d/dx x_eq, d/dz x_eq) whose modes are
        dropped: a mode is dropped if more than `exclude_overlap` of its norm lies in their span.
        Extra modes are searched so that still N modes are returned.

        Returns (eigenvalues, modes): the N eigenvalues with the largest real part found,
        sorted by decreasing real part, and the modes as state vectors (real part on the grid,
        as get_state()). The model's state fields are overwritten.
        """
        comm = self.dist.comm
        if self.dist.dtype != np.complex128:
            raise ValueError("solve_evp needs complex fields: build the model with mode='ecs'.")
        if coupled is None:
            coupled = self.varies_horizontally(x_eq)
        n, need = self.evp_unknowns(coupled), self.evp_memory_estimate(coupled)
        limit = max_memory*1e9 if max_memory is not None else 0.8*comm.bcast(available_memory(), root=0)
        logger.info(f"EVP ({'coupled' if coupled else 'per wavenumber'}): {n} unknowns per subproblem, "
                    f"estimated memory {need/1e9:.1f} GB (limit {limit/1e9:.1f} GB)")
        if need > limit:
            raise EVPTooLargeError(
                f"The {'coupled ' if coupled else ''}EVP has {n} unknowns and needs about {need/1e9:.0f} GB "
                f"(limit {limit/1e9:.0f} GB). Use Arnoldi (stability_method='arnoldi'), or solve the EVP "
                f"on the base state interpolated to a lower resolution.")
        if coupled and comm.size > 1:
            return self._solve_evp_on_root(x_eq, N=N, target=target, frame_velocity=frame_velocity,
                                           dense_limit=dense_limit, ncc_cutoff=ncc_cutoff,
                                           max_memory=limit/1e9, exclude=exclude,
                                           exclude_overlap=exclude_overlap)
        if coupled not in self._evp_problems:
            self._evp_problems[coupled] = self.build_evp_problem(coupled=coupled)
        self.evp_problem = self._evp_problems[coupled]

        # base state and frame velocity
        self.set_eq_state(x_eq)
        if not coupled:  # z-only copies of the (horizontally uniform) base state
            first = (Ellipsis,) + (slice(0, 1),)*(self.dim - 1) + (slice(None),)
            for full, zonly in zip(self.eq_fields, self.eq_fields_z):
                full.change_scales(1)
                zonly.load_from_global_grid_data(full.allgather_data('g')[first])
        c = self.evp_frame_velocity()
        c['g'] = 0
        if frame_velocity is not None:
            for i, ci in enumerate(frame_velocity):
                c['g'][i] = ci

        # directions to exclude (neutral shifts): orthonormal real basis U, search a few extra modes
        U = None
        if exclude is not None and len(exclude) > 0:
            U, _ = np.linalg.qr(np.array(exclude, dtype=float).T)
        n_search = N + (2*U.shape[1] if U is not None else 0)

        self.comm.Barrier()
        solver = self.evp_problem.build_solver(matrix_coupling=[True]*self.dim, ncc_cutoff=ncc_cutoff) if coupled \
            else self.evp_problem.build_solver(ncc_cutoff=ncc_cutoff)
        found = []      # (eigenvalue, owner rank, local subproblem, eigenvector)
        for sp in solver.subproblems:
            solver.build_matrices([sp], ['M', 'L'])   # EVP matrices are built lazily
            n = sp.L_min.shape[0]
            if n <= dense_limit:
                solver.solve_dense(sp)
            else:
                solver.solve_sparse(sp, N=min(n_search, n - 2), target=target)
            lams = solver.eigenvalues
            ok = np.where(np.isfinite(lams) & (np.abs(lams) < 1e12))[0]   # drop infinite (algebraic) modes
            for j in ok[np.argsort(-lams[ok].real)][:n_search]:           # keep the most unstable
                found.append((lams[j], sp, solver.eigenvectors[:, j].copy()))
        found.sort(key=lambda t: -t[0].real)
        found = found[:n_search]

        # global selection: the N largest real parts over all processes
        everything = [(lam, rank, i) for rank, lams in enumerate(comm.allgather([t[0] for t in found]))
                      for i, lam in enumerate(lams)]
        everything.sort(key=lambda t: -t[0].real)
        eigenvalues, modes = [], []
        for lam, rank, i in everything:
            if len(modes) == N:
                break
            for var in solver.state:
                var['c'] = 0
            if rank == comm.rank:
                _, sp, vec = found[i]
                sp.subsystems[0].scatter(vec, solver.state)
            if U is not None:   # drop modes lying (almost) in the span of the excluded directions
                q = np.concatenate([self._grid_data(f).ravel() for f in self.fields])
                overlap = np.linalg.norm(U.T @ q)/max(np.linalg.norm(q), 1e-300)
                if overlap > exclude_overlap:
                    logger.info(f"EVP: dropped neutral mode lambda = {lam:.6g} (overlap {overlap:.3f})")
                    continue
            eigenvalues.append(lam)
            modes.append(self.get_state())
        return np.array(eigenvalues), np.array(modes).reshape(len(modes), -1).T

    @staticmethod
    def _grid_data(field):
        """Global complex grid data of a field at scale 1 (all ranks)."""
        field.change_scales(1)
        return field.allgather_data('g')

    def _serial_copy(self):
        """Single-process copy of this model (same class, parameters and domain) on rank 0,
        rebuilt only when the parameters or the domain change."""
        key = (repr(sorted(self.params.items())), tuple(self.sizes), tuple(self.bounds))
        if self._evp_serial is None or self._evp_serial[0] != key:
            copy = type(self)(dict(self.params), self.sizes, self.bounds, self.bounded, self.mode,
                              self.dealias, comm=MPI.COMM_SELF)
            copy.build_problems()
            self._evp_serial = (key, copy)
        return self._evp_serial[1]

    def _solve_evp_on_root(self, x_eq, **kwargs):
        """Coupled EVP under MPI: Dedalus cannot couple distributed directions, so rank 0 solves
        it on a single-process copy of the model while the other ranks wait; the result is
        broadcast to all ranks."""
        comm = self.dist.comm
        logger.info(f"Coupled EVP: solving on rank 0 only ({comm.size} processes in the run).")
        result, error = None, None
        if comm.rank == 0:
            try:
                result = self._serial_copy().solve_evp(x_eq, coupled=True, **kwargs)
            except Exception as exc:       # report on all ranks instead of hanging the others
                logger.exception("Coupled EVP failed on rank 0")
                error = f"{type(exc).__name__}: {exc}"
        # The other ranks wait without spinning: a blocking MPI wait polls at 100% CPU and
        # would compete with rank 0's multithreaded factorization.
        request = comm.Ibarrier()
        while not request.Test():
            time.sleep(0.1)
        result, error = comm.bcast((result, error), root=0)
        if error is not None:
            raise RuntimeError(f"Coupled EVP failed on rank 0: {error}")
        eigenvalues, modes = result
        if modes.shape[1] > 0:
            self.set_state(modes[:, -1])   # same side effect as the serial path: state fields overwritten
        return eigenvalues, modes

    def show_state(self):
        """
        Show the current state of the system.
        """
        # get state of last field, eg salinity in DDC or temperature in RBC
        data_g = self.fields[-1].allgather_data('g').real
        if self.dist.comm.rank == 0:
            if self.dim == 3:
                data_g = data_g[:,0,:] # get a 2D slice
            xaxis = self.bases[0].global_grid(self.dist, scale=self.dealias)
            zaxis = self.bases[-1].global_grid(self.dist, scale=self.dealias)
            fig, ax = plt.subplots(figsize=(4,3))
            im = ax.pcolormesh(xaxis.ravel(), zaxis.ravel(), data_g.T, 
                                            cmap='RdBu_r', shading='auto')
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            fig.colorbar(im)
            plt.show(block=True)
            
    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def _new_ivp_solver(self, x0, stop_time, reuse=False):
        """RK222 IVP solver starting from state x0 at t = 0.
        reuse: return the cached solver (built once per IVP problem; set_param drops it),
        which avoids rebuilding the matrices on every call. RK222 keeps no history between
        steps, so a reused solver gives the same result as a fresh one. Use reuse=False
        when attaching file handlers, which would otherwise stay on the cached solver.
        Building another solver from the same IVP problem invalidates the earlier one (its
        steps no longer match a fresh solver), so a fresh build also drops the cache. Do not
        build solvers from model.ivp_problem directly between flow_map calls."""
        if not reuse:
            self._ivp_cache = None
            solver = self.ivp_problem.build_solver(de.RK222)
        else:
            if self._ivp_cache is None or self._ivp_cache[0] is not self.ivp_problem:
                self._ivp_cache = (self.ivp_problem, self.ivp_problem.build_solver(de.RK222))
            solver = self._ivp_cache[1]
            # Handlers (incl. the nonlinear terms F and the CFL) fire only when their schedule
            # counter increases; reset them, or the first step after iteration = 0 would reuse
            # the F of the previous run.
            for handler in solver.evaluator.handlers:
                handler.last_wall_div = handler.last_sim_div = handler.last_iter_div = -1
        self.set_state(x0)
        solver.stop_sim_time = stop_time
        solver.sim_time = 0
        solver.iteration = 0
        solver.stop_wall_time = np.inf
        solver.stop_iteration = np.inf
        return solver

    def _integrate(self, solver, T, callback=None):
        """Advance solver to time T: CFL-adaptive steps (use_CFL), else equal steps of ~init_dt.
        callback(solver), if given, is called after every step; returning True stops the
        integration early (at solver.sim_time < T)."""
        if self.use_CFL:
            if self.CFL is None or self.CFL.solver is not solver:
                self.set_cfl(solver)
            else:  # reused solver: restart the CFL from init_dt, as a fresh one would
                self.CFL.stored_dt = self.init_dt
            while solver.proceed:
                dt = self.CFL.compute_timestep()
                if solver.sim_time + dt > T:
                    dt = T - solver.sim_time
                solver.step(dt)
                if callback is not None and callback(solver):
                    break
        else:
            num_steps = max(1, int(T/self.init_dt))
            dt = T/num_steps
            for i in range(num_steps):
                solver.step(dt)
                if callback is not None and callback(solver):
                    break

    def flow_map(self, x0, Tp, callback=None):
        """Flow map: integrate x0 forward by Tp and return the final state.
        callback(solver) is called after every step; returning True stops early
        (the state at that time is returned, solver.sim_time gives the time reached)."""
        gc.collect() # free unreachable objects in system
        self.comm.Barrier()
        solver = self._new_ivp_solver(x0, Tp, reuse=True)
        self._integrate(solver, Tp, callback)
        return self.get_state()

    def save_time_dependent_solution(self, x0, Tp:float, ax=0, ay=0, az=0):
        """Integrate a converged orbit/wave and write snapshots / flow properties /
        mean profiles (as enabled) over 2 periods, or one full domain shift for waves."""
        self.comm.Barrier()
        logger.info("Saving time-dependent solutions.")
        sim_time = Tp # for periodic orbit
        n_full_solution_steps = 100
        # for traveling wave and relative periodic orbit
        a_max = max(abs(ax),abs(ay),abs(az))
        if a_max<0.05:
            sim_time = 2 * Tp
        else:
            sim_time = 1/a_max * Tp
            n_full_solution_steps = 2*100

        solver = self._new_ivp_solver(x0, sim_time)
        if self.save_snapshots:
            self.set_snapshots(solver=solver, sim_dt=sim_time/n_full_solution_steps)
        if self.save_flowproperties:
            self.set_flowproperties(solver=solver, sim_dt=sim_time/n_full_solution_steps)
        if self.save_meanprofiles:
            self.set_meanprofiles(solver=solver, sim_dt=sim_time/n_full_solution_steps)
        self._integrate(solver, sim_time)

    
    def t_derivative(self, x, delta_T):
        self.comm.Barrier()
        # Return dF/dt (restores model state to x afterwards)
        solver = self._new_ivp_solver(x, delta_T, reuse=True)
        solver.step(delta_T)
        x_out = self.get_state()
        self.set_state(x)  # restore state so caller's view is unchanged
        return (x_out - x) / delta_T
    def x_derivative(self, x):
        # Return dF/dx
        self.set_state(x)
        data_slices = []
        for field in self.fields:
            x_der = de.Differentiate(field, self.coords['x']).evaluate()
            x_der.change_scales(1.0)
            gdata = x_der.allgather_data('g').real
            data_slices.append(gdata.ravel())
        return np.concatenate(data_slices)
    def y_derivative(self, x):
        # Return dF/dx
        self.set_state(x)
        data_slices = []
        for field in self.fields:
            y_der = de.Differentiate(field, self.coords['y']).evaluate()
            y_der.change_scales(1.0)
            gdata = y_der.allgather_data('g').real
            data_slices.append(gdata.ravel())
        return np.concatenate(data_slices)
    def z_derivative(self, x):
        # Return dF/dz
        self.set_state(x)
        data_slices = []
        for field in self.fields:
            z_der = de.Differentiate(field, self.coords['z']).evaluate()
            z_der.change_scales(1.0)
            gdata = z_der.allgather_data('g').real
            data_slices.append(gdata.ravel())
        return np.concatenate(data_slices)

    # ------------------------------------------------------------------
    # Observables of a state vector (set the model state to x)
    # ------------------------------------------------------------------
    @staticmethod
    def _volume_average(operand):
        """Volume average of a scalar expression, as a float on all ranks."""
        avg = de.Average(operand).evaluate()
        avg.change_scales(1)
        return float(avg.allgather_data('g').real.ravel()[0])

    def l2norm(self, x):
        """L2 norm of the flow field: sqrt(<sum over fields |q|^2>), <.> the volume average."""
        self.set_state(x)
        return np.sqrt(self._volume_average(sum(f @ f if len(f.tensorsig) > 0 else f*f
                                                for f in self.fields)))

    def kinetic_energy(self, x):
        """Kinetic energy 1/2 <|u|^2> of the velocity (the first field)."""
        self.set_state(x)
        u = self.fields[0]
        return 0.5*self._volume_average(u @ u)
