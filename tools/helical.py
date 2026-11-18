"""
Helical (screw-symmetric) states of 3-D models in a square periodic box (Lx = Ly, Nx = Ny).

The screw symmetry  sigma_h = T_z(h Lz/4) o R_90,  h = +1 or -1,  has order 4 (sigma_h^4 = 1).
A state with sigma_h x = x has a horizontally averaged flow U(z) = (U, V) that turns by 90
degrees every Lz/4: with Z = U + iV, only the vertical Fourier modes m = -h, 3h, -5h, ... of
Z survive, so there is no net mean flow and no planar shear (sin(2 pi z) e_x has m = +1 and
m = -1 alike). The leading mode is the helix

    h = +1 :  U(z) = A (cos 2 pi z/Lz, -sin 2 pi z/Lz),   mean-flow helicity > 0
    h = -1 :  U(z) = A (cos 2 pi z/Lz, +sin 2 pi z/Lz),   mean-flow helicity < 0

sigma_h and its inverse define the same invariant subspace, so Symmetry(rz=1, az=0.25) and
Symmetry(rz=3, az=-0.25) select the same handedness. The two handednesses are mirror images
of each other (x -> -x maps one onto the other).

    from tools import screw_symmetry, helical_guess, helical_diagnostics
    sigma = screw_symmetry(+1)                       # Symmetry(rz=1, az=0.25)
    x0 = helical_guess(model_2d, model_3d, sigma)    # 2-D state -> 3-D screw-symmetric guess
    print(helical_diagnostics(model_3d, x0, sigma))
"""
import numpy as np
from physics.symmetry import Symmetry
from .change_grid import change_grid


def screw_symmetry(handedness=1):
    """Screw symmetry T_z(handedness*Lz/4) o R_90 (order 4); handedness = +1 or -1."""
    if handedness not in (1, -1):
        raise ValueError("handedness must be +1 or -1")
    return Symmetry(rz=1, az=0.25*handedness)


def helical_guess(model_2d, model_3d, sigma=None, scale=1.0):
    """3-D screw-symmetric initial guess from the current state of a 2-D model (e.g. TM1):
    the state is extended uniformly in y (change_grid), then projected onto the states
    invariant under sigma (default screw_symmetry(+1)) by averaging its 4 images. The
    projection keeps half of the planar mean shear (as a helix) and averages the fingers of
    the four rotated copies, so a factor `scale` (e.g. 2) can restore the amplitude.
    Returns the state vector of model_3d (also loaded into model_3d)."""
    sigma = screw_symmetry(+1) if sigma is None else sigma
    x = change_grid(model_2d, model_3d)
    x = scale*model_3d.symmetrize(x, sigma, 4)
    model_3d.set_state(x)
    return x


# ----------------------------------------------------------------------------
# Screw operations on grid state vectors (pure numpy, any mode, also 'sim')
# A state vector holds the velocity (3, Nx, Ny, Nz) followed by scalar fields (Nx, Ny, Nz),
# as model.get_state() returns it; shape = model.get_grid_shape(). Needs Nx = Ny, Nz % 4 = 0.
# ----------------------------------------------------------------------------
def _split(x, shape):
    n = int(np.prod(shape))
    return np.asarray(x[:3*n]).reshape((3,) + tuple(shape)), np.asarray(x[3*n:]).reshape((-1,) + tuple(shape))


def _join(u, s):
    return np.concatenate([u.ravel(), s.ravel()])


def screw_image_grid(x, shape, handedness=1):
    """sigma_h x = T_z(h Lz/4) R_90 x on the grid (same as model.apply_symmetry(x,
    screw_symmetry(h)), but without Dedalus): rotate, then q(z) -> q(z + h Lz/4)."""
    from physics.rotation import rotate_z_grid
    if shape[2] % 4:
        raise ValueError("the screw shift Lz/4 needs Nz divisible by 4")
    u, s = _split(x, shape)
    roll = -handedness*shape[2]//4
    u = np.roll(rotate_z_grid(u, 1, vector=True), roll, axis=-1)
    s = np.roll(np.stack([rotate_z_grid(f, 1) for f in s]), roll, axis=-1)
    return _join(u, s)


def screw_project_grid(x, shape, handedness=1):
    """Projection (1/4) sum_k sigma_h^k x onto the screw-symmetric states (numpy)."""
    total, y = np.array(x, dtype=float), np.array(x, dtype=float)
    for _ in range(3):
        y = screw_image_grid(y, shape, handedness)
        total += y
    return total/4


def c2x_image_grid(x, shape):
    """Rotation by 180 degrees about the x-axis: q(x, y, z) -> q(x, -y, -z) with
    (u, v, w) -> (u, -v, -w) and the scalars (fluctuations about T_b = S_b = z) changing sign;
    equals Symmetry(sy=-1, sz=-1). It keeps the handedness of a helix and conjugates the screw
    to its inverse, so with screw_symmetry it generates a dihedral group of order 8."""
    u, s = _split(x, shape)
    py = (-np.arange(shape[1])) % shape[1]
    pz = (-np.arange(shape[2])) % shape[2]
    u = u[:, :, py][:, :, :, pz]*np.array([1, -1, -1])[:, None, None, None]
    s = -s[:, :, py][:, :, :, pz]
    return _join(u, s)


def helical_group_project_grid(x, shape, handedness=1, c2=True):
    """Projection onto the states invariant under the screw (and, if c2, also C2x):
    (1/2)(1 + C2x) (1/4) sum_k sigma_h^k x."""
    y = screw_project_grid(x, shape, handedness)
    return 0.5*(y + c2x_image_grid(y, shape)) if c2 else y


def shift_grid(x, shape, sx, sy):
    """q(x, y) -> q(x + sx dx, y + sy dy), shifts in grid spacings (fractional: spectral)."""
    u, s = _split(x, shape)
    out = []
    for f in (u, s):
        F = np.fft.fft2(f, axes=(1, 2))
        kx = np.fft.fftfreq(shape[0])*2*np.pi
        ky = np.fft.fftfreq(shape[1])*2*np.pi
        F *= np.exp(1j*(kx[:, None]*sx + ky[None, :]*sy))[None, :, :, None]
        out.append(np.fft.ifft2(F, axes=(1, 2)).real)
    return _join(*out)


def best_screw_center(x, shape, handedness=1, n_sub=2):
    """Horizontal shift that puts the state's best screw axis on the rotation axis of
    screw_symmetry (the z-axis through the origin). Scans shifts on a grid of 1/n_sub grid
    spacings (n_sub = 2 includes the cell centres, the other C4 centres of a square lattice)
    and returns (sx, sy, retained) maximising retained = ||P y||/||y||, y = shift_grid(x,
    sx, sy), P = screw_project_grid. A z-shift commutes with the screw and is not needed."""
    best = (0.0, 0.0, -1.0)
    norm = np.linalg.norm(x)
    for sx in np.arange(0, shape[0], 1/n_sub):
        for sy in np.arange(0, shape[1], 1/n_sub):
            y = shift_grid(x, shape, sx, sy)
            r = np.linalg.norm(screw_project_grid(y, shape, handedness))/norm
            if r > best[2]:
                best = (float(sx), float(sy), float(r))
    return best


# ----------------------------------------------------------------------------
# Diagnostics (numpy on the state vector, global grid at scale 1)
# ----------------------------------------------------------------------------
def _velocity(model, x):
    """Velocity (3, Nx, Ny, Nz) of state vector x (velocity is the first field)."""
    if model.dim != 3:
        raise ValueError("helical diagnostics need a 3-D model")
    shape = tuple(model.get_grid_shape())
    n = int(np.prod(shape))
    return np.asarray(x[:3*n]).real.reshape((3,) + shape)


def _wavenumbers(N, L):
    return 2*np.pi*np.fft.fftfreq(N, d=L/N)


def mean_flow(model, x):
    """(z, U(z), V(z)): horizontally averaged horizontal velocity of state x."""
    u = _velocity(model, x)
    Nz = u.shape[-1]
    z = np.arange(Nz)*model.bounds[-1]/Nz
    return z, u[0].mean(axis=(0, 1)), u[1].mean(axis=(0, 1))


def winding_number(U, V):
    """Energy-weighted turning of the mean-flow direction over one vertical period:
    sum_m m |Z_m|^2 / sum_m |Z_m|^2 for Z = U + iV = sum_m Z_m exp(2 pi i m z/Lz), so a helix
    turning m times counter-clockwise (seen from +z) gives m, a planar shear (Z_m and Z_-m of
    equal size) gives 0. Unlike counting the turns of the angle, it is defined when the mean
    flow passes through zero. 0 for a vanishing mean flow."""
    Z = np.fft.fft(U + 1j*V)
    E = np.abs(Z)**2
    if E.sum() == 0:
        return 0.0
    m = np.fft.fftfreq(len(Z), d=1/len(Z))
    return float(np.sum(m*E)/E.sum())


def _curl(model, u):
    """Curl of a periodic velocity array (3, Nx, Ny, Nz) (spectral derivatives)."""
    kx, ky, kz = (_wavenumbers(N, L) for N, L in zip(u.shape[1:], model.bounds))
    uh = np.fft.fftn(u, axes=(1, 2, 3))
    d = lambda c, k, axis: np.fft.ifftn(1j*k.reshape([-1 if a == axis else 1 for a in range(3)])*uh[c],
                                        axes=(0, 1, 2)).real
    return np.stack([d(2, ky, 1) - d(1, kz, 2),
                     d(0, kz, 2) - d(2, kx, 0),
                     d(1, kx, 0) - d(0, ky, 1)])


def helicity(model, x):
    """Volume-averaged helicity <u . (curl u)> of state x (spectral derivatives)."""
    u = _velocity(model, x)
    return float(np.mean(np.sum(u*_curl(model, u), axis=0)))


def relative_helicities(model, x):
    """Relative helicities as in Fraser et al. (2025, JFM 1020 R1), H_rel[a] =
    <a . curl a>/sqrt(<|a|^2> <|curl a|^2>), in [-1, 1] (1: maximally helical, Beltrami),
    for the mean flow U = (U(z), V(z), 0) and for the fluctuations u' = u - U.
    Returns (H_rel[U], H_rel[u']); nan where the field is below 1e-10 of the velocity rms
    (its value would only describe round-off)."""
    u = _velocity(model, x)
    U = np.zeros_like(u)
    U[:2] = u[:2].mean(axis=(1, 2), keepdims=True)
    floor = 1e-20*np.mean(np.sum(u*u, axis=0))

    def rel(a):
        a2 = np.mean(np.sum(a*a, axis=0))
        if a2 <= floor or a2 == 0:
            return float('nan')
        w = _curl(model, a)
        n = np.sqrt(a2*np.mean(np.sum(w*w, axis=0)))
        return float(np.mean(np.sum(a*w, axis=0))/n) if n > 0 else float('nan')
    return rel(U), rel(u - U)


def mean_flow_helicity(model, x):
    """Helicity <U . (curl U)> of the mean flow U(z) = (U, V, 0) alone: <V U' - U V'>."""
    _, U, V = mean_flow(model, x)
    kz = _wavenumbers(len(U), model.bounds[-1])
    dz = lambda f: np.fft.ifft(1j*kz*np.fft.fft(f)).real
    return float(np.mean(V*dz(U) - U*dz(V)))


def helical_diagnostics(model, x, sigma=None):
    """Summary of the helical character of state x:
      'mean_flow_rms'       sqrt(<U^2 + V^2>_z)
      'planar_fraction'     (l1 - l2)/(l1 + l2), l1 >= l2 the eigenvalues of the covariance of
                            (U, V) over z: 1 for a planar shear, 0 for a circular helix
      'winding_number'      turns of the mean-flow direction over Lz, counter-clockwise seen from
                            +z positive: -1 for the screw_symmetry(+1) helix, +1 for (-1)
      'mean_flow_helicity'  <U . curl U> of the mean flow
      'helicity'            <u . curl u> of the full velocity
      'Hrel_mean', 'Hrel_fluct'  relative helicities of the mean flow and of the fluctuations
                            (relative_helicities, as in Fraser et al. 2025)
      'screw_distance'      ||sigma x - x||/||x||  (only if sigma is given; needs Dedalus)
    With no mean flow (rms below 1e-10 of the velocity rms) planar_fraction is nan and
    winding_number 0: their values would only describe round-off."""
    z, U, V = mean_flow(model, x)
    E = np.mean(U**2 + V**2)
    u_rms = np.sqrt(np.mean(_velocity(model, x)[:2]**2)*2)
    if E > (1e-10*u_rms)**2:
        # planar part: largest eigenvalue of the 2x2 covariance of (U, V)
        C = np.array([[np.mean(U*U), np.mean(U*V)], [np.mean(U*V), np.mean(V*V)]])
        lam = np.linalg.eigvalsh(C)
        planar = float((lam[1] - lam[0])/(lam[1] + lam[0]))
        winding = winding_number(U, V)
    else:
        planar, winding = float('nan'), 0.0
    out = {'mean_flow_rms': float(np.sqrt(E)),
           'planar_fraction': planar,
           'winding_number': winding,
           'mean_flow_helicity': mean_flow_helicity(model, x),
           'helicity': helicity(model, x)}
    out['Hrel_mean'], out['Hrel_fluct'] = relative_helicities(model, x)
    if sigma is not None:
        out['screw_distance'] = float(np.linalg.norm(model.apply_symmetry(x, sigma) - x)
                                      / np.linalg.norm(x))
    return out
