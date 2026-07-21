import numpy as np
import dedalus.public as de
import h5py
from mpi4py import MPI
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
class FluidModel:
    def __init__(self, params, sizes, bounds, bounded=False, mode='ecs', dealias=3/2):
        """
        :param float sizes: Grid mesh of domain (Nx, Nz) or (Nx, Ny, Nz)
        :param float bounds: Domain size (Lx, Lz) or (Lx, Ly, Lz)
        :param bool bounded: Is this a domain bounded in vertical (z) direction? If yes, model will use Chebyshev grid instead of Fourier mode
        :param str mode: The model will build dist with dtype=np.complex128 and bases with de.ComplexFourier as defaults to impose ECS's symmetry. However, CFL(u) will loss stability during long-term simulation due to ComplexFourier. Therefore, you need to set mode='sim' to use dtype=float64 and RealFourier instead, to get stability of CFL(u) operator for long-time simulation. 
        :param float dealias: Grid will be scaled for enhancing convective term
        """
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
        self.ext = (params or {}).get('ext', 'nc')

        # Registry: Newton solver will see [u, ...]
        self.fields = []
        self.ivp_problem = None
        # For EVP
        self.eq_fields = []
        self.evp_problem = None
        
        # CFL function
        self.CFL = None

        # Preview current state
        self.preview_fig = None
        self.preview_ax = None
        self.preview_im = None
    
    def set_param(self, name, value):
        """Update a parameter and return True if a domain rebuild is needed."""
        setattr(self, name, value)
        self.params[name] = value
        if name in self.params:
            self.params[name] = value 
            logger.info(f"{name} was updated to {self.params[name]}")
        else:
            logger.info(f"mu_name={name} is unkbown")
        
        rebuild_domain = False
        # Check if we modified geometry
        if name == 'Lx':
            rebuild_domain = True
            # If 2D (Lx, Lz), recreate the tuple with the new Lx
            if len(self.bounds) == 2:
                self.bounds = (value, self.bounds[1])
            # If 3D (Lx, Ly, Lz)
            elif len(self.bounds) == 3:
                self.bounds = (value, self.bounds[1], self.bounds[2])

        elif name == 'Ly' and len(self.bounds) == 3:
            rebuild_domain = True
            self.bounds = (self.bounds[0], value, self.bounds[2])
        
        elif name == 'Aspect' and len(self.bounds) == 3:
            rebuild_domain = True
            if len(self.bounds) == 2: # If 2D (Lx, Lz)
                self.bounds = (value, self.bounds[1])
            elif len(self.bounds) == 3: # If 3D (Lx, Ly, Lz)
                self.bounds = (value, value, self.bounds[2])

        elif name == 'Lz':
            rebuild_domain = True
            if len(self.bounds) == 2:
                self.bounds = (self.bounds[0], value)
            elif len(self.bounds) == 3:
                self.bounds = (self.bounds[0], self.bounds[1], value)
        
        if rebuild_domain:
            # Re-setup the domain/bases in the model
            self.create_domain() # Or your specific domain setup function
            # Re-initialize the Dedalus Problem with the new fields/bases
            self.build_fields()

        self.build_ivp_problem()
        self.build_evp_problem()
    
    def get_grid_shape(self):
        """
        Returns (Nx, Nz) or (Nx, Ny, Nz) scaled by dealias of current domain
        """
        return tuple(basis.global_grid(self.dist, scale=self.dealias).shape[i] 
                 for i, basis in enumerate(self.bases))
    
    def size(self):
        """
        Return total freedom elements on dealias-scaled grid data of all fields 
        """
        grid_shape = self.get_grid_shape()
        points_per_field = np.prod(grid_shape) # np.prod(grid_shape) gives Nx*Nz or Nx*Ny*Nz
        total_size = 0
        for field in self.fields:
            # Vector fields have 'dim' components (len > 0), Scalars do not
            num_components = self.dim if len(field.tensorsig) > 0 else 1
            # In 2D, num_components is 2; in 3D, it is 3
            total_size += num_components * points_per_field
        return total_size
    
    def get_state(self):
        """
        Return a vector of current state collected from all fields
        """
        data_slices = []
        for field in self.fields:
            # Gather from MPI processes and flatten
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

    def save_state(self, filename):
        """
        Save dealias-scaled grid data to a file
        """
        import os
        import h5py
        filename = filename + self.ext
        target_dir = os.path.dirname(filename)
        if self.dist.comm.rank == 0:
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir)
        self.dist.comm.Barrier() # Wait for directory to exist before writing
        # ext = os.path.splitext(filename)[1].lower()

        gathered_data = {}
        for field in self.fields:
            # Use the field name as the HDF5 dataset name
            gathered_data[field.name] = field.allgather_data('g').real

        if self.dist.comm.rank == 0:
            # --- HDF5 Path ---
            if self.ext == '.h5':
                with h5py.File(filename, mode='w') as f:
                    for name, data in gathered_data.items():
                        field_obj = next(obj for obj in self.fields if obj.name == name)
                        is_vector = len(field_obj.tensorsig) > 0
                        if is_vector:
                            # Generic component naming: u0, u1, u2 
                            # Corresponds to (u, w) in 2D or (u, v, w) in 
                            # print("shape of vector:", )
                            for i in range(data.shape[0]):
                                f.create_dataset(f'{name}_{i}', data=data[i])
                        else:
                            # Scalar field (e.g., te, sa)
                            f.create_dataset(name, data=data)
                    
                    # Save grid info for easy plotting later
                    # This handles (xg, zg) or (xg, yg, zg)
                    coords = [basis.global_grid(self.dist, scale=self.dealias) 
                            for basis in self.bases]
                    coord_names = [basis.coord.name for basis in self.bases]

                    for g_name, g_data in zip(coord_names, coords):
                        f.create_dataset(g_name, data=g_data)
                    
                    # Record dimensionality for easier post-processing
                    f.attrs['dim'] = self.dim
            
            # --- NetCDF Path ---
            elif self.ext == '.nc':
                try:
                    import netCDF4 as nc
                except ImportError:
                    raise ImportError("netCDF4 library is required for .nc output.")
                
                with nc.Dataset(filename, mode='w', format='NETCDF4') as ds:
                    ds.dim_attr = self.dim
                    coords = [basis.global_grid(self.dist, scale=self.dealias) for basis in self.bases]
                    coord_names = [basis.coord.name for basis in self.bases]
                    # We reverse the coord_names for ParaView so the last dim is X
                    coords = tuple(reversed(coords))
                    coord_names = tuple(reversed(coord_names))
                    
                    for c_name, g_data in zip(coord_names, coords):
                        # g_data is often 2D/3D from global_grid; we take the 1D slice
                        # x is 1st axis, y 2nd, z 3rd (standard Dedalus)
                        dim_size = g_data.size
                        ds.createDimension(c_name, dim_size)
                        # print(f"{c_name} size={dim_size} shape={np.shape(g_data)}") # checked: correct
                        v = ds.createVariable(c_name, 'f8', (c_name,))
                        v[:] = g_data.flatten()

                    # Add Fields
                    for name, data in gathered_data.items():
                        field_obj = next(obj for obj in self.fields if obj.name == name)
                        if len(field_obj.tensorsig) > 0: # is vector
                            for i in range(data.shape[0]):
                                var = ds.createVariable(f'{name}_{i}', 'f8', tuple(coord_names))
                                var[:] = data[i].T
                                # print(np.shape(data[i]))
                        else:
                            var = ds.createVariable(name, 'f8', tuple(coord_names))
                            var[:] = data.T
            else:
                raise ValueError(f"Unsupported file extension: {self.ext}")  
    def load_state(self, filename):
        """
        Load dealias-scaled grid data from a file
        """
        import os
        filename = filename + self.ext
        logger.info(f"Loading state from {filename}")
        # ext = os.path.splitext(filename)[1].lower()
        
        if self.ext == '.h5':
            # --- HDF5 Loading Logic ---
            with h5py.File(filename, mode='r') as f:
                # current_shape = self.get_grid_shape() # (Nx, [Ny], Nz)
                for field in self.fields:
                    # Check if this field is a Vector or Scalar
                    is_vector = len(field.tensorsig) > 0
                    if is_vector:
                        # Reconstruct the multi-component array (e.g., 2, Nx, Nz)
                        # We check how many components the field expects
                        num_comp = self.dim 
                        # Get the shape of one component to initialize the buffer
                        comp_shape = f[f'{field.name}_0'].shape
                        data = np.zeros((num_comp,) + comp_shape)
                        for i in range(num_comp):
                            data[i] = f[f'{field.name}_{i}'][:]
                    else:
                        # Scalar field (e.g., te, sa)
                        data = f[field.name][:]
                    # Load the gathered data into the distributed field
                    # Dedalus handles the distribution to different MPI ranks automatically
                    field.load_from_global_grid_data(data)
        elif self.ext == '.nc':
            # --- NetCDF Loading Logic ---
            try:
                import netCDF4 as nc
            except ImportError:
                raise ImportError("netCDF4 library is required to load .nc files.")
            
            with nc.Dataset(filename, mode='r') as ds:
                for field in self.fields:
                    is_vector = len(field.tensorsig) > 0
                    if is_vector:
                        num_comp = self.dim
                        # Reconstruct vector data
                        # Shapes in .nc are (Nx, [Ny], Nz) because we renamed labels, not data
                        comp_shape = ds.variables[f'{field.name}_0'].shape
                        data = np.zeros((num_comp,) + comp_shape)
                        for i in range(num_comp):
                            data[i] = ds.variables[f'{field.name}_{i}'][:]
                    else:
                        # Scalar field
                        data = ds.variables[field.name][:].T
                    
                    field.load_from_global_grid_data(data)
        else:
            raise ValueError(f"Unsupported file extension: {self.ext}")  

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

    def set_CFL(self, solver, initial_dt=0.001, cadence=10, safety=0.5, threshold=0.1,  max_change=1.5, min_change=0.5, max_dt=0.1):
        """ Set up the CFL condition for adaptive time-stepping. """
        self.CFL = de.CFL(solver, initial_dt=initial_dt, cadence=cadence, safety=safety, threshold=threshold, 
                          max_change=max_change, min_change=min_change, max_dt=max_dt)
        self.CFL.add_velocity(self.fields[0]) # Assuming the first field is velocity; adjust if needed

    def solve_EVP(self, x0, N=20, target=1.0):
        """
        Solve eigenvalue problem using Dedalus's EVP with a base state 'x0'. 
        Finding N eigenmodes near to target eigenvalue.
        """
        MPI.COMM_WORLD.Barrier()
        solver = self.evp_problem.build_solver()
        self.set_eq_state(x0)
        solver.solve_sparse(solver.subproblems[0], N=N, target=target)
        evals = solver.eigenvalues
        emodes = solver.eigenvectors
        sorted_indices = np.argsort(-evals.real)
        evals = evals[sorted_indices]
        emodes = emodes[:, sorted_indices]
        solver.set_state(sorted_indices[0], solver.subsystems[0])
        return evals, emodes
    
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
            
    def F_Tp(self, x0, Tp):
        MPI.COMM_WORLD.Barrier()
        solver = self.ivp_problem.build_solver(de.RK222)
        self.set_state(x0)
        solver.stop_sim_time = Tp
        solver.sim_time = 0
        solver.iteration = 0
        solver.stop_wall_time = np.inf
        solver.stop_iteration = np.inf
        
        num_steps = int(Tp/self.init_dt)
        dt = Tp/num_steps
        for i in range(num_steps):
            solver.step(dt)
        return self.get_state()
    def save_time_dependent_solution(self, x0, Tp, ax=0, az=0):
        MPI.COMM_WORLD.Barrier()
        solver = self.ivp_problem.build_solver(de.RK222)
        self.set_state(x0)

        # set T
        sim_time = Tp # for periodic orbit
        n_full_solution_steps = 100
        # for traveling wave and relative periodic orbit
        a_max = max(abs(ax),abs(az))
        if a_max<0.05:
            sim_time = 2 * Tp
        else:
            sim_time = 1/a_max * Tp
            n_full_solution_steps = 2*100


        solver.stop_sim_time = sim_time
        solver.sim_time = 0
        solver.iteration = 0
        solver.stop_wall_time = np.inf
        solver.stop_iteration = np.inf

        self.set_snapshots(solver=solver, sim_dt=sim_time/n_full_solution_steps)
        # self.set_timehistory(solver=solver,properties=properties)
        
        num_steps = int(sim_time/self.init_dt)
        dt = sim_time/num_steps
        for i in range(num_steps):
            solver.step(dt)

    
    def t_derivative(self, x, delta_T):
        MPI.COMM_WORLD.Barrier()
        # Return dF/dt
        solver = self.ivp_problem.build_solver(de.RK222)
        self.set_state(x)
        solver.step(delta_T)
        x_out = self.get_state()
        return (x_out - x) / delta_T
    def x_derivative(self, x):
        # Return dF/dx
        self.set_state(x)
        data_slices = []
        for field in self.fields:
            gdata = de.Differentiate(field, self.coords['x']).evaluate().allgather_data('g').real
            data_slices.append(gdata.ravel())
        return np.concatenate(data_slices)
    def z_derivative(self, x):
        # Return dF/dz
        self.set_state(x)
        data_slices = []
        for field in self.fields:
            gdata = de.Differentiate(field, self.coords['z']).evaluate().allgather_data('g').real
            data_slices.append(gdata.ravel())
        return np.concatenate(data_slices)
    
    def apply_symmetry_ax(self, field, ax):
        kx = self.bases[0].wavenumbers
        phase_shift = np.exp(1j * kx * ax)
        coeff = field.allgather_data('c')
        view = [np.newaxis] * coeff.ndim
        view[1] = slice(None) # Match the X-axis
        coeff *= phase_shift[tuple(view)]
        field.load_from_global_coeff_data(coeff)
    def apply_symmetry_ay(self, field, ay):
        if self.dim < 3:
            return # Do nothing if 2D
        ky = self.bases[1].wavenumbers
        phase_shift = np.exp(1j * ky * ay)
        coeff = field.allgather_data('c')
        # In 3D (comp, x, y, z), y is axis 2
        view = [np.newaxis] * coeff.ndim
        view[2] = slice(None) 
        coeff *= phase_shift[tuple(view)]
        field.load_from_global_coeff_data(coeff)

    def apply_symmetry_az(self, field, az):
        """Applies a translation in the z-direction (Fourier only)."""
        if self.bounded:
            raise NotImplementedError("Cannot use phase-shift for bounded (Chebyshev) z-basis.")
        kz = self.bases[-1].wavenumbers
        phase_shift = np.exp(1j * kz * az)
        coeff = field.allgather_data('c')
        # In 2D (comp, x, z), z is axis 2. In 3D (comp, x, y, z), z is axis 3.
        view = [np.newaxis] * coeff.ndim
        view[-1] = slice(None) # z is always the last axis
        coeff = coeff * phase_shift[tuple(view)]
        # print("coeff.imag",np.linalg.norm(coeff.imag))
        field.load_from_global_coeff_data(coeff)
        # field['c'] = coeff
    def apply_symmetry(self, x, ax=0, az=0):
        self.set_state(x)
        for field in self.fields:
            if ax != 0:
                self.apply_symmetry_ax(field, ax)
            if az != 0 and not self.bounded: # Only if z is Fourier!
                self.apply_symmetry_az(field, az)
        return self.get_state()