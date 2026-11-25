"""
Linear (Floquet) stability of a converged solution: Arnoldi iteration on the
linearised flow map DG, optionally with the neutral (drift) directions projected out.

MPI: as in GMRES (krylov.py), the Arnoldi basis Q, the Hessenberg matrix and the Ritz
vectors exist on rank 0 only; every rank runs the time stepper on the vector broadcast
from rank 0, so all ranks apply DG to the same vector. Only the Ritz vectors of the modes
that are saved are formed (Nsave_modes of them), not all Neigen.
"""
import os
import logging
import numpy as np
import scipy.io
import scipy.linalg
from physics.symmetry import Symmetry
from physics.base import EVPTooLargeError
from .krylov import is_root, bcast_value, bcast_array

logger = logging.getLogger(__name__)


def neutral_basis(solver, x_base):
    """Orthonormal basis of the neutral directions of x_base: spatial shifts
    (d/dx, d/dy in 3-D, d/dz if periodic in z) and, for orbits, d/dt."""
    model = solver.model
    if np.linalg.norm(x_base) < 1e-6:
        return []
    candidates = [model.x_derivative(x_base)]
    if model.dim == 3:
        candidates.append(model.y_derivative(x_base))
    if not model.bounded:
        candidates.append(model.z_derivative(x_base))
    if solver.Tsearch:
        candidates.append(model.t_derivative(x_base, solver.eps_t))

    basis = []
    for u in candidates:
        norm_u = np.linalg.norm(u)
        if norm_u <= 1e-12:
            continue
        u = u/norm_u
        for u_prev in basis:  # Gram-Schmidt against the directions kept so far
            u = u - np.vdot(u_prev, u)*u_prev
        norm_u = np.linalg.norm(u)
        if norm_u > 1e-12:
            basis.append(u/norm_u)
    return basis


def arnoldi(matvec, r, n, project=None, comm=None):
    """n Arnoldi steps on matvec from start vector r (MGS). If project is given,
    it is applied to the start vector and to each operator input and output.
    Returns (Q, H) with Q (len(r), n+1), H (n+1, n); stops early if the
    Krylov subspace closes (remaining columns stay zero).
    MPI (comm): r is only read on rank 0, Q and H are None on the other ranks; matvec and
    project are called on every rank with the vector broadcast from rank 0."""
    root = is_root(comm)
    Q = H = None
    if root:
        if project is not None:
            r = project(r)
        Q = np.zeros((r.size, n+1))
        H = np.zeros((n+1, n))
        Q[:, 0] = r/np.linalg.norm(r)
    for k in range(1, n+1):
        q_in = bcast_array(comm, Q[:, k-1] if root else None)
        if project is not None:
            v = project(matvec(project(q_in)))
        else:
            v = matvec(q_in)
        closed = False
        if root:
            for j in range(k):
                H[j, k-1] = np.vdot(Q[:, j], v)
                v = v - H[j, k-1]*Q[:, j]
            H[k, k-1] = np.linalg.norm(v)
            if H[k, k-1] < 1e-12:
                logger.info(f"Arnoldi Krylov subspace closed at step {k}")
                closed = True
            else:
                Q[:, k] = v/H[k, k-1]
        if bcast_value(comm, closed):
            break
    return Q, H


def modes_to_save(growthrate, nsave=None):
    """Indices of the unstable modes (Re(growthrate) > 0) to save, most unstable first; at most nsave of
    them (None: all). A complex pair counts as two modes, so an odd nsave can cut a pair: its partner is
    the complex conjugate and carries no new information."""
    unstable = np.where(growthrate.real > 0.0)[0]
    unstable = unstable[np.argsort(-growthrate.real[unstable], kind='stable')]
    return unstable if nsave is None else unstable[:max(int(nsave), 0)]


def stability(solver, xi, residual_tol=1e-6):
    """Linear stability of the converged solution xi, by solver.stability_method:
        'arnoldi' : Arnoldi iteration on the linearised flow map (any solution type);
        'evp'     : the model's eigenvalue problem (equilibria and traveling waves only).
    Saves stability/{eigenvalues,growthrate}.mtx (Floquet multipliers exp(lambda Tp) and
    growth rates lambda, most unstable first), residual.mtx (Arnoldi Ritz residuals) and the
    unstable modes in solver.odir/stability: unstable_mode_<i> (real part) and, for complex modes,
    unstable_mode_<i>_imag -- all of them, or only the solver.Nsave_modes most unstable (i = 0 is the most
    unstable). The eigenvalue files always list all Neigen modes. Skipped if xi is not converged."""
    model = solver.model
    comm = model.dist.comm
    if np.linalg.norm(solver.residual(xi)) >= residual_tol:
        return
    method = solver.stability_method
    logger.info(f'Solving linear stability problem around converged solution ({method}) ...')
    out_dir = os.path.join(solver.odir, 'stability')
    if comm.rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    original_symmetry = solver.sigma
    # full stability: no symmetry constraint on perturbations (removes a single Symmetry and
    # every generator of a SymmetryGroup; the original object is restored below)
    solver.sigma = Symmetry()
    try:
        if method == 'arnoldi':
            growthrate, eigenvalues, eigenvectors, res = _stability_arnoldi(solver, xi)
        elif method == 'evp':
            growthrate, eigenvalues, eigenvectors, res = _stability_evp(solver, xi)
        else:
            raise ValueError(f"Unknown stability_method '{method}': use 'arnoldi' or 'evp'.")

        if comm.rank == 0:
            scipy.io.mmwrite(os.path.join(out_dir, 'eigenvalues.mtx'), eigenvalues.reshape(1, -1))
            scipy.io.mmwrite(os.path.join(out_dir, 'growthrate.mtx'), growthrate.reshape(1, -1))
            if res is not None:
                scipy.io.mmwrite(os.path.join(out_dir, 'residual.mtx'), [res])

        unstable = np.where(growthrate.real > 0.0)[0]
        if unstable.size > 0:
            keep = modes_to_save(growthrate, solver.Nsave_modes)
            logger.info(f"There are {unstable.size} unstable modes; saving {keep.size}.")
            if callable(eigenvectors):      # Arnoldi: Ritz vectors formed on rank 0 on demand
                modes = eigenvectors(keep)
            else:
                modes = comm.bcast(eigenvectors[:, keep] if comm.rank == 0 else None, root=0)
            for i, mode in enumerate(modes.T):
                model.save_state(filename=os.path.join(out_dir, f'unstable_mode_{i}'),
                                 state_vector=mode.real)
                if np.any(mode.imag):       # complex mode: save_state keeps real data only
                    model.save_state(filename=os.path.join(out_dir, f'unstable_mode_{i}_imag'),
                                     state_vector=mode.imag)
    finally:
        solver.sigma = original_symmetry
    logger.info('Done!')


def _stability_arnoldi(solver, xi):
    """Arnoldi on the linearised flow map DG: Ritz values are the Floquet multipliers.
    Returns (growthrate, eigenvalues, ritz_vectors, res), most unstable first, on every rank.
    ritz_vectors(cols) is collective and returns the Ritz vectors of those columns (complex,
    shape (N, len(cols))) on every rank: they are formed on rank 0 from its Arnoldi basis,
    only for the modes asked for."""
    comm = solver.comm
    root = is_root(comm)
    x0, Tp, ax, ay, az = solver.unpack_xi(xi)
    phi_base = solver.return_map(x0, Tp, ax, ay, az)
    project = None
    if solver.projectNeutralDrift:
        basis = neutral_basis(solver, x0)

        def project(v):
            v = v.copy()
            for u in basis:
                v -= u*np.vdot(u, v)
            return v

    apply_DG = lambda v: solver.return_map_derivative(x0, v, phi_base, Tp, ax, ay, az)
    # start vector drawn on rank 0 only (numpy's global generator, so np.random.seed
    # reproduces it) and broadcast by arnoldi: every rank perturbs with the same vector
    r = np.random.rand(solver.model.size()) if root else None
    Q, H_ = arnoldi(apply_DG, r, solver.Neigen, project, comm)

    eigvecs_H, summary = None, None
    if root:
        eigenvalues, eigvecs_H = scipy.linalg.eig(H_[0:-1, :])
        growthrate = np.log(eigenvalues)/Tp
        res = np.abs(H_[-1, :] @ eigvecs_H)  # Ritz residuals |h_{m+1,m} y_m|
        idx = np.argsort(growthrate.real)[::-1]  # most unstable first
        eigvecs_H = eigvecs_H[:, idx]
        summary = (growthrate[idx], eigenvalues[idx], res[idx])
    growthrate, eigenvalues, res = bcast_value(comm, summary)

    def ritz_vectors(cols):
        modes = None
        if root:
            Y = eigvecs_H[:, np.asarray(cols, dtype=int)]
            # real and imaginary parts separately: no complex copy of the whole basis
            modes = np.matmul(Q[:, 0:-1], Y.real) + 1j*np.matmul(Q[:, 0:-1], Y.imag)
        return bcast_array(comm, modes)

    return growthrate, eigenvalues, ritz_vectors, res


def _stability_evp(solver, xi):
    """Eigenvalues lambda of the model's EVP about the steady base state (traveling waves
    in their co-moving frame, c = a L / Tp per searched shift). With projectNeutralDrift,
    the neutral shift modes are dropped and Neigen other modes are returned."""
    if solver.Tsearch:
        raise ValueError("stability_method='evp' needs a steady base state (equilibrium or traveling "
                         "wave); use 'arnoldi' for periodic and relative periodic orbits.")
    model = solver.model
    x0, Tp, ax, ay, az = solver.unpack_xi(xi)
    shifts = [ax, ay, az] if model.dim == 3 else [ax, az]
    frame_velocity = [a*L/Tp for a, L in zip(shifts, model.bounds)]
    exclude = neutral_basis(solver, x0) if solver.projectNeutralDrift else None
    try:
        growthrate, eigenvectors = model.solve_evp(x0, N=solver.Neigen, target=solver.evp_target,
                                                   frame_velocity=frame_velocity, exclude=exclude)
    except EVPTooLargeError as exc:
        logger.warning(f"{exc} Falling back to Arnoldi for this stability analysis.")
        return _stability_arnoldi(solver, xi)
    return growthrate, np.exp(growthrate*Tp), eigenvectors, None
