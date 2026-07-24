import numpy as np
import dedalus.public as de
import h5py
from mpi4py import MPI
import matplotlib.pyplot as plt
from .symmetry import Symmetry
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
        self.ext = (params or {}).get('ext', '.nc')

        # Registry: Newton solver will see [u, ...]
        self.fields = []
        self.ivp_problem = None
        # For EVP
        self.eq_fields = []
        self.evp_problem = None
        
        # CFL function
        self.use_CFL = False
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

        self.build_problems()
    
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
        logger.info(f"Saving state to {filename}")

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
                            data[i] = ds.variables[f'{field.name}_{i}'][:].T
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

    # def set_CFL(self, solver, initial_dt=0.001, cadence=10, safety=0.5, threshold=0.1,  max_change=1.5, min_change=0.5, max_dt=0.1):
    #     """ Set up the CFL condition for adaptive time-stepping. """
    #     self.CFL = de.CFL(solver, initial_dt=initial_dt, cadence=cadence, safety=safety, threshold=threshold, 
    #                       max_change=max_change, min_change=min_change, max_dt=max_dt)
    #     self.CFL.add_velocity(self.fields[0]) # Assuming the first field is velocity; adjust if needed

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
        if self.use_CFL:
            self.set_CFL(solver, initial_dt=self.init_dt)
            while solver.proceed:
                dt = self.CFL.compute_timestep()
                if solver.sim_time + dt > Tp:
                    dt = Tp - solver.sim_time
                solver.step(dt)
        else:
            num_steps = int(Tp/self.init_dt)
            dt = Tp/num_steps
            for i in range(num_steps):
                solver.step(dt)
        return self.get_state()
    def save_time_dependent_solution(self, x0, Tp:float, ax=0, ay=0, az=0):
        MPI.COMM_WORLD.Barrier()
        logger.info("Saving time-dependent solutions.")
        solver = self.ivp_problem.build_solver(de.RK222)
        self.set_state(x0)

        # set T
        sim_time = Tp # for periodic orbit
        n_full_solution_steps = 100
        # for traveling wave and relative periodic orbit
        a_max = max(abs(ax),abs(ay),abs(az))
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
        self.set_timehistory(solver=solver, sim_dt=sim_time/n_full_solution_steps)
        
        if self.use_CFL:
            self.set_CFL(solver, initial_dt=self.init_dt)
            while solver.proceed:
                dt = self.CFL.compute_timestep()
                if solver.sim_time + dt > Tp:
                    dt = Tp - solver.sim_time
                solver.step(dt)
        else:
            num_steps = int(Tp/self.init_dt)
            dt = Tp/num_steps
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
    
    def translation_ax(self, field, ax:float):
        ''' Shift field by dx=ax*Lx in the periodic x-direction (Fourier only). '''
        kx = self.bases[0].wavenumbers
        dx = ax*self.bounds[0]
        phase_shift = np.exp(1j * kx * dx)
        field_coeff = field.allgather_data('c')
        if self.dim == 2:
            field_coeff *= phase_shift[:, np.newaxis]
        elif self.dim == 3:
            field_coeff *= phase_shift[:, np.newaxis, np.newaxis]
        field.load_from_global_coeff_data(field_coeff)
    
    def translation_ay(self, field, ay:float):
        ''' Shift field by dy=ay*Ly in the periodic y-direction (Fourier only). '''
        if self.dim < 3:
            return # Do nothing if 2D
        ky = self.bases[1].wavenumbers
        dy = ay*self.bounds[1]
        phase_shift = np.exp(1j * ky * dy)
        field_coeff = field.allgather_data('c')
        field_coeff *= phase_shift[np.newaxis, :, np.newaxis]
        field.load_from_global_coeff_data(field_coeff)

    def translation_az(self, field, az:float):
        ''' Shift field by dz=az*Lz in the periodic y-direction (Fourier only). '''
        if self.bounded:
            raise NotImplementedError("Cannot use phase-shift for bounded (Chebyshev) z-basis.")
        kz = self.bases[-1].wavenumbers
        dz = az*self.bounds[-1]
        phase_shift = np.exp(1j * kz * dz)
        field_coeff = field.allgather_data('c')
        if self.dim == 2:
            field_coeff *= phase_shift[np.newaxis,:]
        elif self.dim == 3:
            field_coeff *= phase_shift[np.newaxis,np.newaxis,:]
        field.load_from_global_coeff_data(field_coeff)

    def reflection_x(self,field):
        coeff = field.allgather_data('c')
        def build_reflection_perm_x(kx, tol=1e-12):
            N = len(kx)
            perm = np.zeros(N, dtype=int)
            for i, k in enumerate(kx):
                # find index of -k
                j = np.where(np.abs(kx + k) < tol)[0]
                if len(j) == 0:
                    # must be Nyquist → self-map
                    perm[i] = i
                else:
                    perm[i] = j[0]
            return perm
        # apply kx -> -kx permutation
        kx = self.bases[0].wavenumbers
        # build permutation: k -> -k
        perm_x = build_reflection_perm_x(kx)
        if len(field.tensorsig) > 0: # is velocity vector
            coeff_ref  = coeff[:, perm_x, :] if self.dim==2 else coeff[:, perm_x, :, :]
            coeff_ref[0] *= -1   # horizontal velocity is odd, change sign of ux only
        else:
            coeff_ref  = coeff[perm_x, :] if self.dim==2 else coeff[perm_x, :, :]
        
        field.load_from_global_coeff_data(coeff_ref)
    def reflection_y(self,field):
        if self.dim < 3:
            return # Do nothing if 2D
        coeff = field.allgather_data('c')
        def build_reflection_perm_y(ky, tol=1e-12):
            N = len(ky)
            perm = np.zeros(N, dtype=int)
            for i, k in enumerate(ky):
                # find index of -k
                j = np.where(np.abs(ky + k) < tol)[0]
                if len(j) == 0:
                    # must be Nyquist → self-map
                    perm[i] = i
                else:
                    perm[i] = j[0]
            return perm
        # apply kx -> -kx permutation
        ky = self.bases[1].wavenumbers
        # build permutation: k -> -k
        perm_y = build_reflection_perm_y(ky)
        if len(field.tensorsig) > 0: # is velocity vector
            coeff_ref  = coeff[:, :, perm_y, :]
            coeff_ref[1] *= -1   # spanwise velocity is odd
        else:
            coeff_ref  = coeff[:, perm_y, :]

        field.load_from_global_coeff_data(coeff_ref)

    def reflection_z(self,field):
        coeff = field.allgather_data('c')
        def build_reflection_perm_z(kz, tol=1e-12):
            N = len(kz)
            perm = np.zeros(N, dtype=int)
            for i, k in enumerate(kz):
                # find index of -k
                j = np.where(np.abs(kz + k) < tol)[0]
                if len(j) == 0:
                    # must be Nyquist → self-map
                    perm[i] = i
                else:
                    perm[i] = j[0]
            return perm
        # apply kz -> -kz permutation
        kz = self.bases[-1].wavenumbers
        # build permutation: k -> -k
        perm_z = build_reflection_perm_z(kz)
        if len(field.tensorsig) > 0: # is velocity vector
            coeff_ref  = coeff[:, :, perm_z] if self.dim==2 else coeff[:, :, :, perm_z]
            coeff_ref[-1] *= -1   # vertical velocity is odd
        else:
            coeff_ref  = coeff[:, perm_z] if self.dim==2 else coeff[:, :, perm_z]
            coeff_ref *= -1 # change sign of scalar
        # enforce zero mode for odd variable
        # zero_idx = np.where(kz == 0)[0][0]
        # coeff_ref[0, :, zero_idx] = 0
        field.load_from_global_coeff_data(coeff_ref)

    def apply_symmetry(self, x, sigma:Symmetry):
        self.set_state(x)
        for field in self.fields:
            # apply reflection symmetries
            if sigma.sx == -1:
                self.reflection_x(field)
            if sigma.sy == -1:
                self.reflection_y(field)
            if sigma.sz == -1:
                self.reflection_z(field)
            # apply translation symmetries
            if sigma.ax != 0:
                self.translation_ax(field, sigma.ax)
            if sigma.ay != 0:
                self.translation_ay(field, sigma.ay)
            if sigma.az != 0 and not self.bounded: # Only if z is Fourier!
                self.translation_az(field, sigma.az)
        return self.get_state()