"""
Rotation by quarter turns about the z-axis, on global grid data (pure numpy).

On a square periodic grid (Nx = Ny, Lx = Ly, points x_i = i*L/N) a rotation by 90 degrees
about the z-axis through the origin maps grid points to grid points, so it is an exact
permutation of grid values, with no interpolation:

    [R q](x, y, z) = Q q(Q^-1 (x, y), z),    Q (x, y) = (-y, x)

    scalar / w :  f'[i, j, k] = f[j, (N - i) mod N, k]
    (u, v)     :  (u', v') = (-v, u), each taken at the same mapped point

Used by SymmetryOps.rotation_z (physics/symmetry.py); kept free of Dedalus and MPI so it
can be tested on plain arrays.
"""
import numpy as np


def rotate_z_grid(data, quarter_turns, vector=False):
    """Rotate grid data by quarter_turns*90 degrees (counter-clockwise seen from +z).

    data   : scalar array (Nx, Ny, Nz), or vector array (3, Nx, Ny, Nz) with components (u, v, w)
    vector : True if data is a vector field (its horizontal components are rotated too)
    """
    q = int(quarter_turns) % 4
    data = np.asarray(data)
    if q == 0:
        return data.copy()
    ax = 1 if vector else 0                 # index of the x axis
    if data.ndim != ax + 3:
        raise ValueError(f"expected a {'vector (3, Nx, Ny, Nz)' if vector else 'scalar (Nx, Ny, Nz)'}"
                         f" array, got shape {data.shape}")
    N = data.shape[ax]
    if data.shape[ax + 1] != N:
        raise ValueError(f"rotation about z needs Nx == Ny, got {data.shape[ax]} and {data.shape[ax + 1]}")
    perm = (-np.arange(N)) % N
    out = data
    for _ in range(q):
        # move the points: out'[i, j] = out[j, -i]
        out = np.swapaxes(out, ax, ax + 1)
        out = out[:, perm] if vector else out[perm]
        if vector:                          # rotate the components: (u, v, w) -> (-v, u, w)
            out = np.stack([-out[1], out[0], out[2]])
    return np.ascontiguousarray(out)
