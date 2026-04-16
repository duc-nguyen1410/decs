import numpy as np
import dedalus.public as de
import h5py
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
class FluidModel:
    def __init__(self, params, sizes, bounds, bounded=False, dealias=3/2):
        """
        :param float sizes: Grid mesh of domain (Nx, Nz) or (Nx, Ny, Nz)
        :param float bounds: Domain size (Lx, Lz) or (Lx, Ly, Lz)
        :param bool bounded: Is this a domain bounded in vertical (z) direction? If yes, it requires Chebyshev grid
        :param float dealias: Grid will be scaled for enhancing convective term
        """
        self.params = params
        self.sizes = sizes
        self.bounds = bounds
        self.dim = len(sizes)
        self.bounded = bounded
        self.dealias = dealias
        self.coords = None
        self.dist = None
        self.x_basis = None
        self.y_basis = None
        self.z_basis = None
        self.all_bases = None
        self.create_domain()
        self.init_dt = 2e-4

        # Registry: Newton solver will see [u, ...]
        self.state_fields = []
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

    # @staticmethod
    def create_domain(self):
        """
        Creates a 2D or 3D domain/bases in the model.

        """
        
        if self.dim == 2:
            Nx, Nz = self.sizes
            Lx, Lz = self.bounds
            self.coords = de.CartesianCoordinates('x', 'z')
        elif self.dim == 3:
            Nx, Ny, Nz = self.sizes
            Lx, Ly, Lz = self.bounds
            self.coords = de.CartesianCoordinates('x', 'y', 'z')
        else:
            raise ValueError("Sizes and bounds must be length 2 or 3.")
        
        self.dist = de.Distributor(self.coords, dtype=np.complex128)

        # Horizontal x-basis (Always periodic)
        self.x_basis = de.ComplexFourier(self.coords['x'], size=Nx, bounds=(0, Lx), dealias=self.dealias)
        
        # Only if 3D, Always periodic
        if self.dim == 3:
            self.y_basis = de.ComplexFourier(self.coords['y'], size=Ny, bounds=(0, Ly), dealias=self.dealias)

        if self.bounded:
            # Use Chebyshev for bounded domains
            self.z_basis = de.ChebyshevT(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)
        else:
            # Use Fourier for fully periodic domains
            self.z_basis = de.ComplexFourier(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)
        
        if self.dim == 2:
            self.all_bases = (self.x_basis, self.z_basis)
        else:
            self.all_bases = (self.x_basis, self.y_basis, self.z_basis)
        
    
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
            return True

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
            self.rebuild_fields()

        self.build_ivp_problem()
        self.build_evp_problem()
    
    def get_grid_shape(self):
        """
        Returns (Nx, Nz) or (Nx, Ny, Nz) scaled by dealias of current domain
        """
        return tuple(basis.global_grid(self.dist, scale=self.dealias).shape[i] 
                 for i, basis in enumerate(self.all_bases))
    
    def size(self):
        """
        Return total freedom elements on dealias-scaled grid data of all fields 
        """
        grid_shape = self.get_grid_shape()
        points_per_field = np.prod(grid_shape) # np.prod(grid_shape) gives Nx*Nz or Nx*Ny*Nz
        total_size = 0
        for field in self.state_fields:
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
        for field in self.state_fields:
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
        for field in self.state_fields:
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
        target_dir = os.path.dirname(filename)
        if self.dist.comm.rank == 0:
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir)
        self.dist.comm.Barrier() # Wait for directory to exist before writing

        gathered_data = {}
        for field in self.state_fields:
            # Use the field name as the HDF5 dataset name
            gathered_data[field.name] = field.allgather_data('g').real

        if self.dist.comm.rank == 0:
            with h5py.File(filename, mode='w') as f:
                for name, data in gathered_data.items():
                    field_obj = next(obj for obj in self.state_fields if obj.name == name)
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
                grids = [basis.global_grid(self.dist, scale=self.dealias) 
                        for basis in self.all_bases]
                
                grid_names = ['xg', 'yg', 'zg'] if self.dim == 3 else ['xg', 'zg']
                
                for g_name, g_data in zip(grid_names, grids):
                    f.create_dataset(g_name, data=g_data)
                
                # Record dimensionality for easier post-processing
                f.attrs['dim'] = self.dim
    def load_state(self, filename):
        """
        Load dealias-scaled grid data from a file
        """
        logger.info(f"Loading state from {filename}")
        
        with h5py.File(filename, mode='r') as f:
            # current_shape = self.get_grid_shape() # (Nx, [Ny], Nz)

            for field in self.state_fields:
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
                
        logger.info("State loaded successfully.")

    def set_initial_conditions(self,mode = 'random', scale=1e-3):
        """
        Set initial condition for each field
        """
        if mode == 'random':
            for field in self.state_fields:
                field.fill_random('g', seed=42, distribution='normal', scale=scale) # Random noise
        elif mode == 'horizontal_sin':
            z = self.z_basis.local_grid(self.dist, scale=self.dealias)
            for i, field in enumerate(self.state_fields):
                if i==0: # velocity
                    field.fill_random('g', seed=42, distribution='normal', scale=1e-3)
                else: # scalar fields
                    field['g'] = -scale*np.sin(2.0*np.pi*1*z)
        else:
            raise ValueError("Invalid mode for initial conditions")

    def set_CFL(self, solver, initial_dt=0.001, cadence=10, safety=0.5, threshold=0.1,  max_change=1.5, min_change=0.5, max_dt=0.1):
        """
        Set up the CFL condition for adaptive time-stepping.
        """
        self.CFL = de.CFL(solver, initial_dt=initial_dt, cadence=cadence, safety=safety, threshold=threshold, 
                          max_change=max_change, min_change=min_change, max_dt=max_dt)
        self.CFL.add_velocity(self.u)

    def set_snapshots(self, solver, sim_dt=10.0, max_writes=1000, file_handler_mode = 'overwrite'):
        """
        Settings of saving data to 'snapshots/' for post-processing later
        """
        snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=sim_dt, max_writes=max_writes, mode=file_handler_mode)
        for field in self.state_fields:
            snapshots.add_task(field, name=field.name)
    def set_checkpoints(self, solver, sim_dt=100.0, max_writes=1, file_handler_mode = 'overwrite'):
        """
        Settings of saving all information of current state to 'checkpoints/' for reload/restart simulation later
        """
        checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=sim_dt, max_writes=max_writes, mode=file_handler_mode)
        checkpoints.add_tasks(solver.state)
    def set_timehistory(self, solver, properties, sim_dt=10.0, max_writes=1000, file_handler_mode = 'overwrite'):
        """
        Settings of saving properties of fluid flow to 'timehistory/' for post-processing later
        """
        timehistory = solver.evaluator.add_file_handler('', sim_dt=sim_dt, max_writes=max_writes, mode=file_handler_mode)
        for property in properties:
            timehistory.add_task(property, name=property.name)
    def preview(self):
        """ Preview the current state using last field of the system. """
        data_g = self.state_fields[-1].allgather_data('g').real
        if self.dist.comm.rank == 0:
            xaxis = self.x_basis.global_grid(self.dist, scale=self.dealias)
            zaxis = self.z_basis.global_grid(self.dist, scale=self.dealias)
            if self.dim == 3:
                data_g = data_g[:,0,:] # get a 2D slice
            # Initialize the figure only once
            if self.preview_fig is None:
                plt.ion()  # Turn on interactive mode
                self.preview_fig, self.preview_ax = plt.subplots(figsize=(4,3))
                self.preview_im = self.preview_ax.pcolormesh(xaxis.ravel(), zaxis.ravel(), data_g.T, 
                                             cmap='RdBu_r', shading='auto')
                self.preview_ax.set_xlabel('x')
                self.preview_ax.set_ylabel('z')
                self.preview_fig.colorbar(self.preview_im)
                # self.preview_ax.set_title("Salt Concentration") 
            else:
                self.preview_im.set_array(data_g.T.ravel())
                v_min, v_max = np.min(data_g), np.max(data_g)
                self.preview_im.set_clim(vmin=v_min, vmax=v_max)
                # self.preview_ax.set_title(f"Salt Concentration at time {self.sim_time:.2f}")
                self.preview_fig.canvas.draw()
                self.preview_fig.canvas.flush_events()         
    def preview3D(self):
        """ Preview the current state using last field in 3D using isosurfaces. """
        if self.dim == 3:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import art3d
            # pip install scikit-image
            from skimage import measure # For Marching Cubes (isosurface)
    
            # Get the last field (usually Salinity or Temperature)
            data_g = self.state_fields[-1].allgather_data('g').real
            
            if self.dist.comm.rank == 0:
                # Get 1D axis arrays for the grid
                xg = self.x_basis.global_grid(self.dist, scale=self.dealias).ravel()
                yg = self.y_basis.global_grid(self.dist, scale=self.dealias).ravel()
                zg = self.z_basis.global_grid(self.dist, scale=self.dealias).ravel()
                
                # 1. Initialize Figure
                if self.preview_fig is None:
                    plt.ion()
                    self.preview_fig = plt.figure(figsize=(6, 5))
                    self.preview_ax = self.preview_fig.add_subplot(111, projection='3d')
                    self.preview_ax.set_xlabel('x')
                    self.preview_ax.set_ylabel('y')
                    self.preview_ax.set_zlabel('z')
                else:
                    self.preview_ax.clear() # Clear the previous frame
                    self.preview_ax.set_xlabel('x')
                    self.preview_ax.set_ylabel('y')
                    self.preview_ax.set_zlabel('z')

                # 2. Generate Isosurface using Marching Cubes
                # Choose a level (e.g., the mean of the field)
                level = (np.max(data_g) + np.min(data_g)) / 2
                
                try:
                    # verts: coordinates of vertices, faces: triangles
                    verts, faces, normals, values = measure.marching_cubes(data_g, level=level)
                    
                    # Scale vertices from index-space to physical-space
                    # Indices are (i, j, k) corresponding to (x, y, z)
                    verts[:, 0] = verts[:, 0] * (xg[1] - xg[0]) + xg[0]
                    verts[:, 1] = verts[:, 1] * (yg[1] - yg[0]) + yg[0]
                    verts[:, 2] = verts[:, 2] * (zg[1] - zg[0]) + zg[0]

                    # 3. Create a 3D PolyCollection (the mesh)
                    mesh = art3d.Poly3DCollection(verts[faces])
                    mesh.set_edgecolor('none')
                    mesh.set_alpha(0.6)
                    mesh.set_facecolor('royalblue')
                    
                    self.preview_ax.add_collection3d(mesh)
                    
                    # Set limits based on domain
                    self.preview_ax.set_xlim(xg.min(), xg.max())
                    self.preview_ax.set_ylim(yg.min(), yg.max())
                    self.preview_ax.set_zlim(zg.min(), zg.max())
                    
                except (ValueError, RuntimeError):
                    # Fallback if the field is uniform or level is outside range
                    self.preview_ax.text(0.5, 0.5, 0.5, "Field uniform - No surface", transform=self.preview_ax.transAxes)

                self.preview_fig.canvas.draw()
                self.preview_fig.canvas.flush_events()
    def solve_EVP(self, x0, N=20, target=1.0):
        """
        Solve eigenvalue problem using Dedalus's EVP with a base state 'x0'. 
        Finding N eigenmodes near to target eigenvalue.
        """
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
        data_g = self.state_fields[-1].allgather_data('g').real
        if self.dist.comm.rank == 0:
            if self.dim == 3:
                data_g = data_g[:,0,:] # get a 2D slice
            xaxis = self.x_basis.global_grid(self.dist, scale=self.dealias)
            zaxis = self.z_basis.global_grid(self.dist, scale=self.dealias)
            fig, ax = plt.subplots(figsize=(4,3))
            im = ax.pcolormesh(xaxis.ravel(), zaxis.ravel(), data_g.T, 
                                            cmap='RdBu_r', shading='auto')
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            fig.colorbar(im)
            plt.show(block=True)
            
    def F_Tp(self, x0, Tp):
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

        self.set_snapshots(solver=solver, sim_dt=10.0)
        self.set_timehistory(solver=solver,properties=properties)
        
        num_steps = int(sim_time/self.init_dt)
        dt = sim_time/num_steps
        for i in range(num_steps):
            solver.step(dt)

    
    def t_derivative(self, x, delta_T):
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
        for field in self.state_fields:
            gdata = de.Differentiate(field, self.coords['x']).evaluate().allgather_data('g').real
            data_slices.append(gdata.ravel())
        return np.concatenate(data_slices)
    def z_derivative(self, x):
        # Return dF/dz
        self.set_state(x)
        data_slices = []
        for field in self.state_fields:
            gdata = de.Differentiate(field, self.coords['z']).evaluate().allgather_data('g').real
            data_slices.append(gdata.ravel())
        return np.concatenate(data_slices)
    
    def apply_symmetry_ax(self, field, ax):
        kx = self.x_basis.wavenumbers
        phase_shift = np.exp(1j * kx * ax)
        coeff = field.allgather_data('c')
        view = [np.newaxis] * coeff.ndim
        view[1] = slice(None) # Match the X-axis
        coeff *= phase_shift[tuple(view)]
        field.load_from_global_coeff_data(coeff)
    def apply_symmetry_ay(self, field, ay):
        if self.dim < 3:
            return # Do nothing if 2D
        ky = self.y_basis.wavenumbers
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
        kz = self.z_basis.wavenumbers
        phase_shift = np.exp(1j * kz * az)
        coeff = field.allgather_data('c')
        # In 2D (comp, x, z), z is axis 2. In 3D (comp, x, y, z), z is axis 3.
        view = [np.newaxis] * coeff.ndim
        view[-1] = slice(None) # z is always the last axis
        coeff *= phase_shift[tuple(view)]
        field.load_from_global_coeff_data(coeff)
    def apply_symmetry(self, x, ax=0, az=0):
        self.set_state(x)
        for field in self.state_fields:
            if ax != 0:
                self.apply_symmetry_ax(field, ax)
            if az != 0 and not self.bounded: # Only if z is Fourier!
                self.apply_symmetry_az(field, az)
        return self.get_state()