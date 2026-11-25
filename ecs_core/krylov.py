"""
Krylov and trust-region building blocks for the Newton solver.

Pure numpy/scipy: nothing here knows about Dedalus or the flow model. The
Jacobian enters only as a callable ``matvec(v)`` and the nonlinear residual as
a callable ``evaluate(dx)``, so these functions can be tested on small matrices.

Notation (Newton step  A dx = -b):
    Q (n, k+1)   orthonormal Krylov basis, Q[:, 0] = -b / ||b||
    H (k+1, k)   Hessenberg matrix, A Q[:, :k] = Q H
    w (k,)       step coefficients, dx = Q[:, :k] w
    res_sq       squared residual of the linear model  || H w + beta e1 ||^2

MPI (comm given, comm.size > 1): the Krylov algebra runs on rank 0 only. Rank 0 owns Q, H, w
and res_sq (None on the other ranks); each matvec input is broadcast from rank 0, so every
rank runs the (collective) time stepper on the same vector; every decision that changes the
control flow (stop GMRES, shrink the trust region) is taken on rank 0 and broadcast. Memory
of the basis is therefore n*(k+1) values on rank 0 and nothing on the others, independent of
the number of ranks. With comm None or a single rank the code is plain serial numpy.
"""
import logging
import numpy as np
import scipy.optimize

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Root-only linear algebra: helpers (duck-typed mpi4py communicator)
# ----------------------------------------------------------------------
def is_root(comm):
    """True on rank 0, and always without a (multi-rank) communicator."""
    return comm is None or comm.rank == 0


def _parallel(comm):
    return comm is not None and comm.size > 1


def bcast_value(comm, value):
    """value of rank 0 on every rank (small Python objects: scalars, flags, short arrays)."""
    return comm.bcast(value, root=0) if _parallel(comm) else value


def bcast_array(comm, a):
    """numpy array of rank 0 on every rank (buffer broadcast; other ranks pass None)."""
    if not _parallel(comm):
        return a
    meta = comm.bcast((a.shape, a.dtype.str) if comm.rank == 0 else None, root=0)
    buf = np.ascontiguousarray(a) if comm.rank == 0 else np.empty(meta[0], dtype=np.dtype(meta[1]))
    comm.Bcast(buf, root=0)
    return buf


def orthogonalize(Q, k, v):
    """Modified Gram-Schmidt of v against Q[:, :k]. Returns (v, h) with h[:k] the
    projections and h[k] = ||v|| after orthogonalisation (v is not normalised)."""
    h = np.zeros(k+1)
    for j in range(k):
        h[j] = np.matmul(np.conj(Q[:, j]), v)
        v = v - h[j]*Q[:, j]
    h[k] = np.linalg.norm(v)
    return v, h


def arnoldi_step(matvec, Q, k):
    """One Arnoldi step (k >= 1, serial): orthogonalise A Q[:, k-1] against Q[:, :k]
    with modified Gram-Schmidt. Returns the new basis vector and column k-1 of H."""
    v, h = orthogonalize(Q, k, matvec(Q[:, k-1]))
    return v/h[k], h


def hookstep(H, beta, k, radius):
    """Minimise || H[:k+1, :k] w + beta e1 ||^2 subject to ||w|| <= radius.
    Returns (w, res_sq)."""
    F = H[0:k+1, 0:k]
    e1 = np.zeros(k+1)
    e1[0] = beta

    def fun(w):
        r = np.matmul(F, w) + e1
        return np.matmul(r, r)

    def jac(w):
        return 2*np.matmul(np.matmul(np.transpose(F), F), w) + 2*np.matmul(np.transpose(F), e1)

    constraint = {'type': 'ineq',
                  'fun': lambda w: radius*radius - np.matmul(np.transpose(w), w),
                  'jac': lambda w: -2*w}
    w0 = np.zeros(k)
    w0[0] = 1e-3
    res = scipy.optimize.minimize(fun, w0, method='SLSQP', jac=jac, constraints=(constraint,),
                                  options={'ftol': 1e-34, 'disp': False, 'maxiter': 100000000})
    return res.x, res.fun


def gmres_hookstep(matvec, b, kmax, radius, kmin=1, min_error=0.0, comm=None):
    """Build a Krylov basis for A dx = -b, solving the trust-region (hookstep)
    problem after every step. Stops after kmax steps, or once k >= kmin and the
    linear-model error drops below min_error.

    matvec(v) is called on every rank with the same v (collective); b is only read on rank 0.
    Returns (Q, H, k, beta, w, res_sq) with Q of shape (n, k+1); with a multi-rank comm,
    Q, H, beta, w and res_sq are None on ranks other than 0 (k is the same on every rank)."""
    root = is_root(comm)
    Q = H = beta = w = res_sq = None
    if root:
        r = -b
        beta = np.linalg.norm(r)
        Q = np.zeros((b.size, kmax+1))
        H = np.zeros((kmax+1, kmax))
        Q[:, 0] = r/beta if beta > 0 else r

    for k in range(1, kmax+1):
        v = matvec(bcast_array(comm, Q[:, k-1] if root else None))
        stop = False
        if root:
            v, h = orthogonalize(Q, k, v)
            Q[:, k], H[:k+1, k-1] = v/h[k], h
            w, res_sq = hookstep(H, beta, k, radius)
            # NOTE: res_sq is the *squared* linear-model residual (logged as "residual norm")
            logger.info(f"GMRES iteration {k}, residual norm: {np.abs(res_sq):.5e}")
            stop = kmin <= k and np.abs(res_sq) < min_error
        if bcast_value(comm, stop):
            break
    return (Q[:, :k+1] if root else None), H, k, beta, w, res_sq


def trust_region_step(Q, H, k, beta, w, res_sq, radius, b_norm, evaluate,
                      radius_min, radius_max, comm=None):
    """Accept a step dx = Q[:, :k] w, shrinking the radius until the true
    residual decreases, then choose the radius for the next Newton step.

    evaluate(dx) -> (residual norm at xi + dx, payload); the payload of the
    accepted step is returned so the caller can reuse it.

    Next radius from ratio = actual / predicted residual reduction:
        ratio > 0.75 and step limited by the radius -> double (<= radius_max)
        ratio < 0.25                                -> halve
        otherwise                                   -> keep
    Returns (dx, error, next_radius, payload).

    MPI (comm): Q, H, beta, w and res_sq are only read on rank 0; dx is formed there and
    broadcast, evaluate(dx) is called on every rank (collective), and dx, error and
    next_radius are returned on every rank."""
    root = is_root(comm)
    b_norm = bcast_value(comm, b_norm)
    dx = bcast_array(comm, np.matmul(Q[:, 0:k], w) if root else None)
    test, payload = evaluate(dx)
    test = bcast_value(comm, test)

    while test > 0.99*b_norm and radius > radius_min:
        radius = 0.5*radius
        if root:
            w, res_sq = hookstep(H, beta, k, radius)
        dx = bcast_array(comm, np.matmul(Q[:, 0:k], w) if root else None)
        test, payload = evaluate(dx)
        test = bcast_value(comm, test)

    decision = None
    if root:
        predicted = b_norm - np.sqrt(max(res_sq, 0.0))
        ratio = (b_norm - test)/predicted if predicted > 0 else 0.0
        on_boundary = np.linalg.norm(w) >= 0.99*radius
        if ratio > 0.75 and on_boundary:
            radius_next = min(2.0*radius, radius_max)
        elif ratio < 0.25:
            radius_next = 0.5*radius
        else:
            radius_next = radius
        logger.info(f"Trust region: used {radius:.4g}, actual/predicted reduction {ratio:.3f}, next {radius_next:.4g}")
        decision = (np.abs(res_sq), radius_next)
    error, radius_next = bcast_value(comm, decision)
    return dx, error, max(radius_next, radius_min), payload
