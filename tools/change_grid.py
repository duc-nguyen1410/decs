# to run: python3 change_grid.py
import numpy as np
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import RegularGridInterpolator
rescale3d = True
filename = "CI1_Lx0.5_new"
Nx_new = 96
Ny_new = 96
Nz_new = 192
Lx_new = 0.5
Ly_new = 0.5
Lz_new = 1.0


with h5py.File(filename+".h5", 'r') as old_f, h5py.File("rescaled_"+filename+".h5", 'w') as new_f:
    print(f"Old keys detected: {list(old_f.keys())}")
    xg, zg = old_f['xg'][:].ravel(), old_f['zg'][:].ravel()
    Nx, Nz = len(xg), len(zg)
    Lx_old, Lz_old = (xg[1]-xg[0]) * Nx, (zg[1]-zg[0]) * Nz
    # Handle Y for 3D
    if 'yg' in old_f and rescale3d:
        yg = old_f['yg'][:].ravel()
        Ny = len(yg)
        Ly_old = (yg[1]-yg[0]) * Ny
    else:
        yg, Ny, Ly_old = [0.0], 1, 0.0
    print(f"Lx_old = {Lx_old}, Ly_old = {Ly_old}, Lz_old = {Lz_old}")
    print(f"Nx_old = {Nx}, Ny_old = {Ny}, Nz_old = {Nz}")

    # Step 1: Change domain size first
    xg_new = np.linspace(0, Lx_new, Nx_new)
    zg_new = np.linspace(0, Lz_new, Nz_new)
    xg = np.linspace(0, Lx_new, Nx) # update old coordinates with new domain size
    zg = np.linspace(0, Lz_new, Nz)

    
    # Step 2: Interpolate the fields if changing the grid
    interpolate = False
    if rescale3d:
        yg_new = np.linspace(0, Ly_new, Ny_new)
        yg = np.linspace(0, Ly_new, Ny) # update old coordinates with new domain size
        # Create 3D meshgrid
        X_new, Y_new, Z_new = np.meshgrid(xg_new, yg_new, zg_new, indexing='ij')
        pts = np.vstack([X_new.ravel(), Y_new.ravel(), Z_new.ravel()]).T
        new_shape = (Nx_new, Ny_new, Nz_new)
        old_axes = (xg, yg, zg)
        if Nx_new!=Nx or Nz_new!=Nz or Ny_new!=Ny:
            interpolate = True
    else:
        # Fallback to 2D
        X_new, Z_new = np.meshgrid(xg_new, zg_new, indexing='ij')
        pts = np.vstack([X_new.ravel(), Z_new.ravel()]).T
        new_shape = (Nx_new, Nz_new)
        old_axes = (xg, zg)
        if Nx_new!=Nx or Nz_new!=Nz:
            interpolate = True

    # Process Datasets
    for key in old_f.keys():
        if key in ['xg', 'yg', 'zg']:
            continue
            
        obj = old_f[key]
        
        if isinstance(obj, h5py.Group):
            old_f.copy(obj, new_f, name=key)
        elif isinstance(obj, h5py.Dataset):
            if obj.shape == (): # Scalar
                new_f.create_dataset(key, data=obj[()])
            else:
                # Interpolate
                old_data = obj[:]
                if interpolate:
                    if Ny == 1:
                        # add plane if old data id 2D
                        # Setup interpolator on the 2D plane
                        interp = RegularGridInterpolator((xg, zg), 
                                                        old_data, bounds_error=False, fill_value=None)
                        pts_for_interp = np.vstack([X_new.ravel(), Z_new.ravel()]).T
                        # Interpolate: This maps the 2D plane onto every Y-slice of the 3D volume
                        data_new = interp(pts_for_interp).reshape(Nx_new, Ny_new, Nz_new)
                    else:
                        # Using method='linear' is default and fast
                        interp = RegularGridInterpolator(old_axes, old_data, 
                                                        bounds_error=False, fill_value=None)
                        data_new = interp(pts).reshape(new_shape)
                else:
                    data_new = old_data
                if Ny==1 and key=='u_1':
                    new_f.create_dataset('u_1', data=np.zeros_like(data_new)) # create u1 as zero field
                    new_f.create_dataset('u_2', data=data_new) # save u_1 as u_2
                else:
                    new_f.create_dataset(key, data=data_new)
                print(f"Rescaled {key} to shape {new_shape}")

    # Write new coordinates
    new_f.create_dataset('xg', data=xg_new)
    new_f.create_dataset('zg', data=zg_new)
    if rescale3d and yg is not None:
        new_f.create_dataset('yg', data=yg_new)
        


