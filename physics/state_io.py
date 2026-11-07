"""
Saving / loading the flow state (self.fields) to .h5 or .nc files.
Mixed into FluidModel; uses self.fields, self.bases, self.dist, self.dim, self.ext.
"""
import os
import logging
import numpy as np
import h5py

logger = logging.getLogger(__name__)


class StateIO:
    def save_state(self, filename, state_vector=None, dealias=False):
        """
        Save dealias-scaled grid data to a file
        """
        if state_vector is not None:
            self.set_state(state_vector)
        filename = filename + self.ext
        target_dir = os.path.dirname(filename)
        if self.dist.comm.rank == 0:
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir)
        self.dist.comm.Barrier() # Wait for directory to exist before writing
        logger.info(f"Saving state to {filename}")

        gathered_data = {}
        for field in self.fields:
            if not dealias: field.change_scales(1.0)
            # Use the field name as the HDF5 dataset name
            gathered_data[field.name] = field.allgather_data('g').real # shape [dim][nx][ny][nz]

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
                            for i in range(data.shape[0]):
                                f.create_dataset(f'{name}_{i}', data=data[i])
                        else:
                            # Scalar field (e.g., te, sa)
                            f.create_dataset(name, data=data)
                    
                    # Save grid info for easy plotting later
                    # This handles (xg, zg) or (xg, yg, zg)
                    coords = [basis.global_grid(self.dist, scale=self.dealias if dealias else 1) 
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
                    coords = [basis.global_grid(self.dist, scale=self.dealias if dealias else 1) for basis in self.bases]
                    coord_names = [basis.coord.name for basis in self.bases]
                    # Reverse for C-order / NetCDF / ParaView standard [z, y, x]
                    coords = tuple(reversed(coords))
                    coord_names = tuple(reversed(coord_names))
                    
                    for c_name, g_data in zip(coord_names, coords):
                        # g_data is often 2D/3D from global_grid; we take the 1D slice
                        # x is 1st axis, y 2nd, z 3rd (standard Dedalus)
                        dim_size = g_data.size
                        ds.createDimension(c_name, dim_size)
                        v = ds.createVariable(c_name, 'f8', (c_name,))
                        v[:] = g_data.flatten()

                    # Add Fields
                    # Dedalus use shape [dim][nz][ny][nx]
                    for name, data in gathered_data.items():
                        working_field = next(field for field in self.fields if field.name == name)
                        is_vector = len(working_field.tensorsig) > 0
                        if is_vector: # is vector
                            for i in range(self.dim):
                                var = ds.createVariable(f'{name}_{i}', 'f8', tuple(coord_names))
                                var[:] = data[i].T # Transpose spatial dimensions [nx, ny, nz] -> [nz, ny, nx]
                        else:
                            var = ds.createVariable(name, 'f8', tuple(coord_names))
                            var[:] = data.T # Transpose spatial dimensions [nx, ny, nz] -> [nz, ny, nx]
            else:
                raise ValueError(f"Unsupported file extension: {self.ext}")  
        self.dist.comm.Barrier()

    def load_state(self, filename):
        """
        Load dealias-scaled grid data from a file
        """
        filename = filename + self.ext
        logger.info(f"Loading state from {filename}")
        
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
                        # Get shape on disk: [nz, ny, nx] (or [nz, nx] in 2D)
                        comp_shape_nc = ds.variables[f'{field.name}_0'].shape
                        # Reconstruct components with original Dedalus layout: [num_comp, nx, ny, nz]
                        # We transpose comp_shape_nc back to [nx, ny, nz]
                        comp_shape_dedalus = comp_shape_nc[::-1]
                        data = np.zeros((num_comp,) + comp_shape_dedalus)
                        for i in range(num_comp):
                            # Read [nz, ny, nx] and transpose back to [nx, ny, nz]
                            data[i] = ds.variables[f'{field.name}_{i}'][:].T # saved data' shape is [nz][ny][nx]
                    else:
                        # Read scalar field [nz, ny, nx] and transpose back to [nx, ny, nz]
                        data = ds.variables[field.name][:].T
                    # Distribute global data back into Dedalus field using (Dim, Nx, [Ny], Nz)
                    field.load_from_global_grid_data(data)
        else:
            raise ValueError(f"Unsupported file extension: {self.ext}")
