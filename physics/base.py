import numpy as np
import dedalus.public as de

import matplotlib.pyplot as plt
class FluidModel2D:
    def __init__(self, domain, params):
        self.coords = domain['coords']
        self.dist = domain['dist']
        self.x_basis = domain['x_basis']
        self.z_basis = domain['z_basis']
        self.dealias = domain['dealias']
        self.params = params
        self.init_dt = 2e-4

        # Core fields: Velocity, Pressure, Temperature
        self.u = self.dist.VectorField(self.coords, name='u', bases=(self.x_basis, self.z_basis))
        self.p = self.dist.Field(name='p', bases=(self.x_basis, self.z_basis))
        self.te = self.dist.Field(name='te', bases=(self.x_basis, self.z_basis))
        self.u.change_scales(self.dealias)
        self.te.change_scales(self.dealias)
        
        # Registry: Newton solver sees [u, te]
        self.state_fields = [self.u, self.te]

        # Standard Convection Taus
        self.tau_p = self.dist.Field(name='tau_p')
        self.tau_u = self.dist.VectorField(self.coords, name='tau_u')
        self.tau_te = self.dist.Field(name='tau_te')


        

        # For EVP
        self.sigma = self.dist.Field(name='sigma') # eigenvalues
        self.u_eq = self.dist.VectorField(self.coords, name='u_eq', bases=(self.x_basis, self.z_basis))
        self.te_eq = self.dist.Field(name='te_eq', bases=(self.x_basis, self.z_basis))
        self.eq_fields = [self.u_eq, self.te_eq]

        # CFL function
        self.CFL = None

        # Preview current state
        self.preview_fig = None
        self.preview_ax = None
        self.preview_im = None

    @staticmethod
    def create_domain(Nx, Nz, Lx, Lz, bounded=False, dealias=3/2):
        coords = de.CartesianCoordinates('x', 'z')
        dist = de.Distributor(coords, dtype=np.complex128)
        x_basis = de.ComplexFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
        z_basis = de.ComplexFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
        if bounded:
            z_basis = de.ChebyshevT(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
        return {
            'coords': coords, 
            'dist': dist, 
            'x_basis': x_basis, 
            'z_basis': z_basis, 
            'dealias': dealias,
            'bounded': bounded,
        }
    
    def get_grid_shape(self):
        Nx = self.x_basis.global_grid(self.dist, scale=self.dealias).shape[0]
        Nz = self.z_basis.global_grid(self.dist, scale=self.dealias).shape[1]
        return Nx, Nz
    
    def size(self):
        Nx, Nz = self.get_grid_shape()
        return (1 + len(self.state_fields)) * Nx * Nz
    
    def get_state(self):
        data_slices = []
        for field in self.state_fields:
            # Gather from MPI processes and flatten
            gdata = field.allgather_data('g').real
            data_slices.append(gdata.ravel())
        return np.concatenate(data_slices)
    
    def set_state(self, state_vector):
        Nx, Nz = self.get_grid_shape()
        cursor = 0
        for i, field in enumerate(self.state_fields):
            if i==0:
                size = 2 * Nx * Nz
                data = state_vector[cursor:cursor+size].reshape((2, Nx, Nz))
                field.load_from_global_grid_data(data)
            else:
                size = Nx * Nz
                data = state_vector[cursor:cursor+size].reshape((Nx, Nz))
                field.load_from_global_grid_data(data)
            cursor += size
    def set_eq_state(self, state_vector):
        Nx, Nz = self.get_grid_shape()
        cursor = 0
        for i, field in enumerate(self.eq_fields):
            if i==0: # [u, te, ...], u first
                size = 2 * Nx * Nz
                data = state_vector[cursor:cursor+size].reshape((2, Nx, Nz))
                field.load_from_global_grid_data(data)
            else:
                size = Nx * Nz
                data = state_vector[cursor:cursor+size].reshape((Nx, Nz))
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
                    if data.ndim == 3: # Vector field (components, x, z)
                        f.create_dataset(f'{name}_u', data=data[0])
                        f.create_dataset(f'{name}_w', data=data[1])
                    else: # Scalar field (x, z)
                        f.create_dataset(name, data=data)
                
                # Save grid info for easy plotting later
                xg = self.x_basis.global_grid(self.dist, scale=self.dealias)
                zg = self.z_basis.global_grid(self.dist, scale=self.dealias)
                f.create_dataset('xg', data=xg)
                f.create_dataset('zg', data=zg)


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
        data_g = self.state_fields[-1].allgather_data('g').real
        if self.dist.comm.rank == 0:
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
        # Return dF/dt at x
        solver = ivp_problem.build_solver(de.RK222)
        self.set_state(x)
        solver.step(delta_T)
        x_out = self.get_state()
        return (x_out - x) / delta_T
    def x_derivative(self, x):
        # Return dF/dax at x using finite differences
        self.set_state(x)
        dudx = de.Differentiate(self.u, self.coords['x']).evaluate().allgather_data('g').real
        dtdx = de.Differentiate(self.te, self.coords['x']).evaluate().allgather_data('g').real
        dsdx = de.Differentiate(self.sa, self.coords['x']).evaluate().allgather_data('g').real
        return np.concatenate([dudx.ravel(), dtdx.ravel(), dsdx.ravel()])
    def z_derivative(self, x):
        # Return dF/daz at x using finite differences
        self.set_state(x)
        dudz = de.Differentiate(self.u, self.coords['z']).evaluate().allgather_data('g').real
        dtdz = de.Differentiate(self.te, self.coords['z']).evaluate().allgather_data('g').real
        dsdz = de.Differentiate(self.sa, self.coords['z']).evaluate().allgather_data('g').real
        return np.concatenate([dudz.ravel(), dtdz.ravel(), dsdz.ravel()])
    