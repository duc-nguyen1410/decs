import numpy as np
from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)
class Symmetry:
    def __init__(self, s:int=1, sx:int=1, sy:int=1, sz:int=1, ax:float=0, ay:float=0, az:float=0):
        """
        Defines a symmetry operation: S [u(x,y,z)] = s * u(sx*x + ax, sy*y + ay, sz*z + az)

        Parameters:
        s  : Global parity (usually 1 or -1)
        sx : x-reflection (1: no reflection, -1: x -> -x)
        sy : y-reflection (1: no reflection, -1: y -> -y)
        sz : z-reflection (1: no reflection, -1: z -> -z)
        ax : x-shift (translation normalized by Lx)
        ay : y-shift (translation normalized by Ly)
        az : z-shift (translation normalized by Lz)
        """
        self.s = s
        self.sx, self.sy, self.sz = sx, sy, sz
        self.ax, self.ay, self.az = ax, ay, az
    def __str__(self):
        return f"[{self.s} {self.sx} {self.sy} {self.sz} {self.ax} {self.ay} {self.az}]"

    def load_from_file(self, filename:str):
        with open(filename, 'r') as f:
            logger.info(f"Loading symmetry from file {filename}")
            # Format: s sx sy sz ax ay az
            data = [float(x) for x in f.readline().split()]
            if len(data)==7:
                self.s, self.sx, self.sy, self.sz, self.ax, self.ay, self.az = data
                logger.info("Loaded symmetry: " + str(self))
            else:
                logger.warning(f"File {filename} contains invalid symmetry format! Expected 7 values, got {len(data)}.")
        

    def is_nontrivial(self):
        return not (self.s == 1 and self.sx == 1 and self.sy == 1 and self.sz == 1 and 
                    self.ax == 0 and self.ay == 0 and self.az == 0)
    


def _reflection_perm(k, tol=1e-12):
    """Index permutation mapping each Fourier wavenumber k to -k
    (the Nyquist mode, which has no partner, maps to itself)."""
    perm = np.arange(len(k))
    for i, ki in enumerate(k):
        j = np.where(np.abs(k + ki) < tol)[0]
        if len(j) > 0:
            perm[i] = j[0]
    return perm


class SymmetryOps:
    """Reflections and translations of the flow state, mixed into FluidModel.
    Uses self.fields, self.bases, self.bounds, self.dim, self.bounded."""

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
        # apply kx -> -kx permutation
        kx = self.bases[0].wavenumbers
        # build permutation: k -> -k
        perm_x = _reflection_perm(kx)
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
        # apply kx -> -kx permutation
        ky = self.bases[1].wavenumbers
        # build permutation: k -> -k
        perm_y = _reflection_perm(ky)
        if len(field.tensorsig) > 0: # is velocity vector
            coeff_ref  = coeff[:, :, perm_y, :]
            coeff_ref[1] *= -1   # spanwise velocity is odd
        else:
            coeff_ref  = coeff[:, perm_y, :]

        field.load_from_global_coeff_data(coeff_ref)

    def reflection_z(self,field):
        if self.bounded: # Apply z -> -z reflection for bounded (Chebyshev) z-basis
            # Apply Chebyshev parity polynomial mapping: T_n(-z) = (-1)^n * T_n(z)
            # Get total number of Chebyshev modes along z
            Nz = field['c'].shape[-1]
            parity = (-1) ** np.arange(Nz)  # [1, -1, 1, -1, ...]

            # Reshape parity for broadcasting against field['c'] shape
            view = [1] * field['c'].ndim
            view[-1] = Nz
            parity = parity.reshape(view)

            # Apply Chebyshev parity reflection z -> -z
            field['c'] *= parity

            # Change the sign of fields
            if len(field.tensorsig) > 0:
                # Vector field (e.g., velocity u = (u, [v], w))
                # Vertical velocity component (w) is odd under z -> -z
                field['c'][-1] *= -1
            else: 
                # Scalar field (e.g., Temperature or Salinity)
                # Flip sign if scalar fluctuation is antisymmetric about midplane
                field['c'] *= -1
        else: # Use Fourier permutation for periodic bases.
            coeff = field.allgather_data('c')
            # apply kz -> -kz permutation
            kz = self.bases[-1].wavenumbers
            # build permutation: k -> -k
            perm_z = _reflection_perm(kz)
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
