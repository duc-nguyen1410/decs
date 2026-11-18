import numpy as np
import logging
from .rotation import rotate_z_grid
logger = logging.getLogger(__name__)
class Symmetry:
    def __init__(self, s:int=1, sx:int=1, sy:int=1, sz:int=1, ax:float=0, ay:float=0, az:float=0,
                 rz:int=0):
        """
        Defines a symmetry operation: S [u(x,y,z)] = s * u(sx*x + ax, sy*y + ay, sz*z + az),
        applied as  shift o rotation o reflections  (reflections first, shifts last).

        Parameters:
        s  : Global parity (usually 1 or -1)
        sx : x-reflection (1: no reflection, -1: x -> -x)
        sy : y-reflection (1: no reflection, -1: y -> -y)
        sz : z-reflection (1: no reflection, -1: z -> -z)
        ax : x-shift (translation normalized by Lx)
        ay : y-shift (translation normalized by Ly)
        az : z-shift (translation normalized by Lz)
        rz : number of 90-degree rotations about the z-axis through the origin, counter-clockwise
             seen from +z (3-D only, needs Lx == Ly and Nx == Ny). The rotation moves the points
             and turns (u, v) with them: [R q](x, y, z) = Q q(Q^-1 (x, y), z), Q (x, y) = (-y, x).
             Screw (helical) symmetry: Symmetry(rz=1, az=0.25), see tools/helical.py.
        """
        self.s = s
        self.sx, self.sy, self.sz = sx, sy, sz
        self.ax, self.ay, self.az = ax, ay, az
        self.rz = int(round(rz)) % 4
    def __str__(self):
        rot = f" {self.rz}" if self.rz else ""   # 7 values (no rotation) as before
        return f"[{self.s} {self.sx} {self.sy} {self.sz} {self.ax} {self.ay} {self.az}{rot}]"

    def load_from_file(self, filename:str):
        """Read one symmetry `s sx sy sz ax ay az [rz]` from a .sym file (lines starting with
        % or # are skipped). A file listing several symmetries needs SymmetryGroup.load_from_file."""
        logger.info(f"Loading symmetry from file {filename}")
        lines = _symmetry_lines(filename)
        if len(lines) > 1:
            raise ValueError(f"{filename} lists {len(lines)} symmetries: load it with "
                             "SymmetryGroup().load_from_file(...) to impose all of them.")
        data = lines[0] if lines else []
        if len(data) in (7, 8):
            self.s, self.sx, self.sy, self.sz, self.ax, self.ay, self.az = data[:7]
            self.rz = int(round(data[7])) % 4 if len(data) == 8 else 0
            logger.info("Loaded symmetry: " + str(self))
        else:
            logger.warning(f"File {filename} contains invalid symmetry format! Expected 7 or 8 values, got {len(data)}.")
        return self

    def impose(self, model, x):
        """The map applied by ECSSolver.return_map: sigma x."""
        return model.apply_symmetry(x, self)

    def is_nontrivial(self):
        return not (self.s == 1 and self.sx == 1 and self.sy == 1 and self.sz == 1 and 
                    self.ax == 0 and self.ay == 0 and self.az == 0 and self.rz == 0)
    


def _symmetry_lines(filename):
    """Numeric rows of a .sym file; blank lines and lines starting with % or # are skipped
    (a Channelflow-style first line '% N' gives the number of symmetries)."""
    rows = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line[0] not in '%#':
                rows.append([float(v) for v in line.split()])
    return rows


def _affine_map(sigma, dim, bounded):
    """Point map g(p) = L p + b (coordinates normalised by L_i, so shifts are mod 1) with
    (sigma u)(p) = (sign) u(g(p)), as applied by SymmetryOps.apply_symmetry: reflections M,
    then rotation R, then shift T(a) give g(p) = M Q^-1 (p + a). Exact (Fraction) entries."""
    from fractions import Fraction
    refl, shift = [sigma.sx, sigma.sy, sigma.sz], [sigma.ax, sigma.ay, sigma.az]
    if dim == 2:                                # no y direction
        refl[1], shift[1] = 1, 0
    if bounded:                                 # the z-shift is not applied on a Chebyshev basis
        shift[2] = 0
    M = np.diag([int(round(v)) for v in refl]).astype(object)
    rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=object)   # Q^-1 (x, y) = (y, -x)
    Qinv = np.eye(3, dtype=int).astype(object)
    for _ in range(getattr(sigma, 'rz', 0) % 4):
        Qinv = rot.dot(Qinv)
    L = M.dot(Qinv)
    b = L.dot(np.array([Fraction(v).limit_denominator(10**6) for v in shift], dtype=object))
    return L, b


def _same_map(A, B):
    (L1, b1), (L2, b2) = A, B
    return np.array_equal(L1, L2) and all((u - v) % 1 == 0 for u, v in zip(b1, b2))


def _compose(A, B):
    """(A o B)(p) = A(B(p))."""
    (L1, b1), (L2, b2) = A, B
    return L1.dot(L2), L1.dot(b2) + b1


def symmetry_order(sigma, dim=3, bounded=False, max_order=64):
    """Smallest n >= 1 with sigma^n = identity (2 for reflections and half shifts, 4 for the
    screw Symmetry(rz=1, az=0.25)). dim and bounded as in the model (2-D: y is ignored;
    bounded: the z-shift is ignored, as in apply_symmetry)."""
    g = _affine_map(sigma, dim, bounded)
    identity = (np.eye(3, dtype=int).astype(object), np.zeros(3, dtype=object))
    gn = g
    for n in range(1, max_order + 1):
        if _same_map(gn, identity):
            return n
        gn = _compose(g, gn)
    raise ValueError(f"Symmetry {sigma} has no finite order <= {max_order}: it cannot be "
                     "imposed by projection (use rational shifts).")


class SymmetryGroup:
    """Several commuting symmetries imposed together, e.g. in ECSSolver.sigma:

        solver.sigma = SymmetryGroup().load_from_file('symmetry_eq1.sym')   # several lines
        solver.sigma = SymmetryGroup().load_from_file('symmetry_pixz.sym',
                                                      'symmetry_piz_taux0.5_tauz0.5.sym')
        solver.sigma = SymmetryGroup(Symmetry(sx=-1, sz=-1), Symmetry(sz=-1, ax=0.5, az=0.5))
        solver.sigma = [sym1, sym2]           # a list is converted to a SymmetryGroup

    The solver then uses G(x) = P S(a) F^Tp(x), with P the projection onto the states
    invariant under every generator:  P = P_1 P_2 ... P_m,  P_i = (1/n_i) sum_k sigma_i^k
    (n_i the order of sigma_i; P_i = (1 + sigma_i)/2 for an involution). For commuting
    generators this is the average over the whole group they generate (for two involutions
    (1 + s1 + s2 + s1 s2)/4); non-commuting generators are rejected. Imposing only the
    product s1 s2 would be weaker: it keeps states symmetric under s1 s2 but not under s1, s2.
    A plain Symmetry in solver.sigma keeps the old map G = sigma S(a) F^Tp."""

    def __init__(self, *generators):
        self.generators = []
        self.orders = []          # None: computed from the symmetry (symmetry_order)
        for g in generators:
            self.add(g)

    def add(self, sigma, order=None):
        """Add a generator (a Symmetry); `order` overrides the computed n with sigma^n = 1."""
        if not isinstance(sigma, Symmetry):
            raise TypeError(f"SymmetryGroup generators must be Symmetry objects, got {type(sigma)}")
        self.generators.append(sigma)
        self.orders.append(order)
        return self

    def load_from_file(self, *filenames):
        """Add the symmetries of one or more .sym files, one `s sx sy sz ax ay az [rz]` per
        line (an optional Channelflow-style first line '% N' and # comments are skipped)."""
        for filename in filenames:
            logger.info(f"Loading symmetries from file {filename}")
            for data in _symmetry_lines(filename):
                if len(data) not in (7, 8):
                    raise ValueError(f"{filename}: expected 7 or 8 values per line, got {len(data)}")
                self.add(Symmetry(*data[:7], rz=data[7] if len(data) == 8 else 0))
        logger.info("Loaded symmetry group: " + str(self))
        return self

    def __iter__(self):
        return iter(self.generators)

    def __len__(self):
        return len(self.generators)

    def __str__(self):
        return " ".join(str(g) for g in self.generators) if self.generators else "[]"

    def is_nontrivial(self):
        return any(g.is_nontrivial() for g in self.generators)

    def check_commuting(self, dim=3, bounded=False):
        """Raise ValueError unless every pair of generators commutes."""
        maps = [_affine_map(g, dim, bounded) for g in self.generators]
        for i in range(len(maps)):
            for j in range(i):
                if not _same_map(_compose(maps[i], maps[j]), _compose(maps[j], maps[i])):
                    raise ValueError(f"Symmetries {self.generators[j]} and {self.generators[i]} "
                                     "do not commute: their projections cannot be combined.")

    def group_orders(self, dim=3, bounded=False):
        return [n if n is not None else symmetry_order(g, dim, bounded)
                for g, n in zip(self.generators, self.orders)]

    def impose(self, model, x):
        """The map applied by ECSSolver.return_map: the projection P x."""
        dim, bounded = model.dim, model.bounded
        self.check_commuting(dim, bounded)
        for g, n in zip(self.generators, self.group_orders(dim, bounded)):
            if g.is_nontrivial():
                x = model.symmetrize(x, g, n)
        return x


def as_symmetry(sigma):
    """Symmetry / SymmetryGroup unchanged; a list or tuple of Symmetry -> SymmetryGroup;
    None -> the identity Symmetry()."""
    if sigma is None:
        return Symmetry()
    if isinstance(sigma, (list, tuple)):
        return SymmetryGroup(*sigma)
    return sigma

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

    def rotation_z(self, field, quarter_turns:int):
        ''' Rotate field by quarter_turns*90 degrees about the z-axis through the origin
        (exact permutation of the grid values, see physics/rotation.py). '''
        if int(quarter_turns) % 4 == 0:
            return
        if self.dim != 3:
            raise ValueError("Rotation about z needs a 3-D domain.")
        Nx, Ny = self.get_grid_shape()[:2]
        if Nx != Ny or not np.isclose(self.bounds[0], self.bounds[1]):
            raise ValueError(f"Rotation about z needs Lx == Ly and Nx == Ny, got "
                             f"(Lx, Ly) = {self.bounds[:2]}, (Nx, Ny) = {(Nx, Ny)}.")
        field.change_scales(1)
        data = field.allgather_data('g')
        field.load_from_global_grid_data(rotate_z_grid(data, quarter_turns, len(field.tensorsig) > 0))

    def symmetrize(self, x, sigma:Symmetry, order:int):
        """Projection of x onto the states invariant under sigma: (1/n) sum_k sigma^k x,
        for a symmetry of order n (sigma^n = identity), e.g. n = 4 for the screw
        Symmetry(rz=1, az=0.25), n = 2 for a reflection."""
        total = np.array(x, dtype=float, copy=True)
        y = np.array(x, copy=True)
        for _ in range(order - 1):
            y = self.apply_symmetry(y, sigma)
            total += y
        return total/order

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
            # apply rotation about z (after reflections, before shifts)
            if getattr(sigma, 'rz', 0):
                self.rotation_z(field, sigma.rz)
            # apply translation symmetries
            if sigma.ax != 0:
                self.translation_ax(field, sigma.ax)
            if sigma.ay != 0:
                self.translation_ay(field, sigma.ay)
            if sigma.az != 0 and not self.bounded: # Only if z is Fourier!
                self.translation_az(field, sigma.az)
        return self.get_state()
