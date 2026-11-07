"""
DoubleDiffusion: shared base class for all double-diffusive models
(temperature + salinity): domain, fields and preview.
"""
import dedalus.public as de
import numpy as np
from mpi4py import MPI
import h5py
import matplotlib.pyplot as plt
from .base import FluidModel 


class DoubleDiffusion(FluidModel):
    def __init__(self, params, sizes, bounds, bounded=False, mode='ecs', dealias=3/2, comm=None):
        # Call the FluidModel __init__ first
        super().__init__(params, sizes, bounds, bounded, mode, dealias, comm)
        # Add additional fields
        self.p = None
        self.u = None
        self.te = None
        self.sa = None
        self.u_eq = None
        self.te_eq = None
        self.sa_eq = None
        self.create_domain()
        self.build_fields()

    def create_domain(self):
        """ Creates a 2D or 3D domain/bases in the model """
        if self.dim == 2:
            Nx, Nz = self.sizes
            Lx, Lz = self.bounds
        elif self.dim == 3:
            Nx, Ny, Nz = self.sizes
            Lx, Ly, Lz = self.bounds
        else:
            raise ValueError("Sizes and bounds must be length 2 or 3.")
        self.comm.Barrier()
        # Only instantiate CartesianCoordinates and Distributor ONCE
        # Re-creating Distributor on every parameter evaluation leaks MPI communicators.
        if self.dist is None:
            if self.dim == 2:
                self.coords = de.CartesianCoordinates('x', 'z')
            else:
                self.coords = de.CartesianCoordinates('x', 'y', 'z')
            dtype = np.float64 if self.mode == 'sim' else np.complex128
            self.dist = de.Distributor(self.coords, dtype=dtype, comm=self.comm)

        # Horizontal x-basis (Always periodic)
        self.x_basis = None
        if self.mode=='sim':
            self.x_basis = de.RealFourier(self.coords['x'], size=Nx, bounds=(0, Lx), dealias=self.dealias)
        else:
            self.x_basis = de.ComplexFourier(self.coords['x'], size=Nx, bounds=(0, Lx), dealias=self.dealias)
        
        # Only if 3D, Always periodic
        if self.dim == 3:
            self.y_basis = None
            if self.mode=='sim':
                self.y_basis = de.RealFourier(self.coords['y'], size=Ny, bounds=(0, Ly), dealias=self.dealias)
            else:
                self.y_basis = de.ComplexFourier(self.coords['y'], size=Ny, bounds=(0, Ly), dealias=self.dealias)

        self.z_basis = None
        if self.bounded:
            # Use Chebyshev for bounded domains
            self.z_basis = de.ChebyshevT(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)
        else:
            # Use Fourier for fully periodic domains
            if self.mode=='sim':
                self.z_basis = de.RealFourier(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)
            else:
                self.z_basis = de.ComplexFourier(self.coords['z'], size=Nz, bounds=(0, Lz), dealias=self.dealias)

        self.bases = None
        if self.dim == 2:
            self.bases = (self.x_basis, self.z_basis)
        else:
            self.bases = (self.x_basis, self.y_basis, self.z_basis)

    def build_fields(self):
        # pressure p (scalar)
        self.p = self.dist.Field(name='p', bases=self.bases)
        # velocity u (Vector)
        self.u = self.dist.VectorField(self.coords, name='u', bases=self.bases)
        # self.u.change_scales(self.dealias)
        self.u_eq = self.dist.VectorField(self.coords, name='u_eq', bases=self.bases)
        # temperature \theta (scalar)
        self.te = self.dist.Field(name='te', bases=self.bases)
        # self.te.change_scales(self.dealias)
        self.te_eq = self.dist.Field(name='te_eq', bases=self.bases)
        # salinity s (scalar)
        self.sa = self.dist.Field(name='sa', bases=self.bases)
        # self.sa.change_scales(self.dealias)
        self.sa_eq = self.dist.Field(name='sa_eq', bases=self.bases)
        # re-collect fields to field list
        self.fields = [self.u, self.te, self.sa]
        self.eq_fields = [self.u_eq, self.te_eq, self.sa_eq]

    def preview(self, field_idx=-1):
        """ Preview the current state using last field of the system. """
        data_g = self.fields[field_idx].allgather_data('g').real
        if self.dist.comm.rank == 0:
            xaxis = self.bases[0].global_grid(self.dist, scale=self.dealias)
            zaxis = self.bases[-1].global_grid(self.dist, scale=self.dealias)
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
                self.preview_fig.canvas.draw()
                self.preview_fig.canvas.flush_events()  
            else:
                self.preview_im.set_array(data_g.T.ravel())
                v_min, v_max = np.min(data_g), np.max(data_g)
                self.preview_im.set_clim(vmin=v_min, vmax=v_max)
                # self.preview_ax.set_title(f"Salt Concentration at time {self.sim_time:.2f}")
                self.preview_fig.canvas.draw()
                self.preview_fig.canvas.flush_events()  
