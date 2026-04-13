import numpy as np
import dedalus.public as de
import h5py
import matplotlib.pyplot as plt
class FluidModel2D:
    def __init__(self, domain, params):
        self.dim = domain['dim']
        self.coords = domain['coords']
        self.dist = domain['dist']
        self.x_basis = domain['x_basis']
        self.y_basis = domain['y_basis']
        self.z_basis = domain['z_basis']
        self.all_bases = domain['all_bases']
        self.dealias = domain['dealias']
        self.params = params
        self.init_dt = 2e-4

        # Core fields: Velocity, Pressure, Temperature
        self.u = self.dist.VectorField(self.coords, name='u', bases=self.all_bases)
        self.p = self.dist.Field(name='p', bases=self.all_bases)
        self.u.change_scales(self.dealias)
        
        # Registry: Newton solver sees [u]
        self.state_fields = [self.u]

        # For EVP
        self.sigma = self.dist.Field(name='sigma') # eigenvalues
        self.u_eq = self.dist.VectorField(self.coords, name='u_eq', bases=self.all_bases)
        
        self.eq_fields = [self.u_eq]

        # CFL function
        self.CFL = None

        # Preview current state
        self.preview_fig = None
        self.preview_ax = None
        self.preview_im = None

    @staticmethod
    def create_domain(sizes, bounds, bounded=False, dealias=3/2):
        """
        Creates a 2D or 3D domain based on input tuple lengths.

        sizes: (Nx, Nz) or (Nx, Ny, Nz)
        bounds: (Lx, Lz) or (Lx, Ly, Lz)
        """
        dim = len(sizes)
        if dim == 2:
            Nx, Nz = sizes
            Lx, Lz = bounds
            coords = de.CartesianCoordinates('x', 'z')
        elif dim == 3:
            Nx, Ny, Nz = sizes
            Lx, Ly, Lz = bounds
            coords = de.CartesianCoordinates('x', 'y', 'z')
        else:
            raise ValueError("Sizes and bounds must be length 2 or 3.")
        
        dist = de.Distributor(coords, dtype=np.complex128)

        # Horizontal x-basis (Always periodic)
        x_basis = de.ComplexFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
        
        # Only if 3D, Always periodic
        if dim == 3:
            y_basis = de.ComplexFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

        if bounded:
            # Use Chebyshev for bounded domains
            # Centering at 0 (-Lz/2, Lz/2) is standard for many cases with a shear
            z_basis = de.ChebyshevT(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
        else:
            # Use Fourier for fully periodic domains
            z_basis = de.ComplexFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
        
        if dim == 2:
            all_bases = (x_basis, z_basis)
        else:
            all_bases = (x_basis, y_basis, z_basis)
        
        return {
            'dim': dim,
            'coords': coords, 
            'dist': dist, 
            'x_basis': x_basis,
            'y_basis': y_basis if dim == 3 else None, 
            'z_basis': z_basis, 
            'all_bases': all_bases,  # Merged for easy field creation
            'dealias': dealias,
            'bounded': bounded,
        }
    
    def get_grid_shape(self):
        # Returns (Nx, Nz) or (Nx, Ny, Nz) based on the current domain
        return tuple(basis.global_grid(self.dist, scale=self.dealias).shape[i] 
                 for i, basis in enumerate(self.all_bases))
    
    def size(self):
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
        data_slices = []
        for field in self.state_fields:
            # Gather from MPI processes and flatten
            gdata = field.allgather_data('g').real
            data_slices.append(gdata.ravel())
        return np.concatenate(data_slices)
    
    def set_state(self, state_vector):
        dim =self.dim
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
                        # Corresponds to (u, w) in 2D or (u, v, w) in 3D
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
        if self.dist.comm.rank == 0:
            print(f"Loading state from {filename}")
        
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
                
        if self.dist.comm.rank == 0:
            print("State loaded successfully.")

    def set_initial_conditions(self,mode = 'random', scale=1e-3):
        if mode == 'random':
            for field in self.state_fields:
                field.fill_random('g', seed=42, distribution='normal', scale=scale) # Random noise
        elif mode == 'horizontal_sin':
            z = self.z_basis.local_grid(self.dist, scale=self.dealias)
            for i, field in enumerate(self.state_fields):
                if i==0: # velocity
                    field.fill_random('g', seed=42, distribution='normal', scale=1e-3)
                else: # scalar fields
                    field['g'] = -scale*np.sin(2.0*np.pi*2*z)
        else:
            raise ValueError("Invalid mode for initial conditions")

    def set_CFL(self, solver, initial_dt, cadence=10, safety=0.5, threshold=0.1,  max_change=1.5, min_change=0.5, max_dt=0.1):
        """
        Set up the CFL condition for adaptive time-stepping.
        """
        self.CFL = de.CFL(solver, initial_dt=initial_dt, cadence=cadence, safety=safety, threshold=threshold, 
                          max_change=max_change, min_change=min_change, max_dt=max_dt)
        self.CFL.add_velocity(self.u)
    

    def preview(self):
        """ Preview the current state of the system. """
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
    
    def solve_EVP(self, evp_problem, x0, N=20, target=1.0):
        solver = evp_problem.build_solver()
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
            
    def F_Tp(self, ivp_problem, x0, Tp):
        solver = ivp_problem.build_solver(de.RK222)
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
    
    def t_derivative(self, ivp_problem, x, delta_T):
        # Return dF/dt
        solver = ivp_problem.build_solver(de.RK222)
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