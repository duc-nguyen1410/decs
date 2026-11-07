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
"""
import logging
import numpy as np
import scipy.optimize

logger = logging.getLogger(__name__)


def arnoldi_step(matvec, Q, k):
    """One Arnoldi step (k >= 1): orthogonalise A Q[:, k-1] against Q[:, :k]
    with modified Gram-Schmidt. Returns the new basis vector and column k-1 of H."""
    v = matvec(Q[:, k-1])
    h = np.zeros(k+1)
    for j in range(k):
        h[j] = np.matmul(np.conj(Q[:, j]), v)
        v = v - h[j]*Q[:, j]
    h[k] = np.linalg.norm(v)
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


def gmres_hookstep(matvec, b, kmax, radius, kmin=1, min_error=0.0):
    """Build a Krylov basis for A dx = -b, solving the trust-region (hookstep)
    problem after every step. Stops after kmax steps, or once k >= kmin and the
    linear-model error drops below min_error.

    Returns (Q, H, k, beta, w, res_sq) with Q of shape (n, k+1)."""
    r = -b
    beta = np.linalg.norm(r)
    Q = np.zeros((b.size, kmax+1))
    H = np.zeros((kmax+1, kmax))
    Q[:, 0] = r/beta if beta > 0 else r

    for k in range(1, kmax+1):
        Q[:, k], H[:k+1, k-1] = arnoldi_step(matvec, Q, k)
        w, res_sq = hookstep(H, beta, k, radius)
        # NOTE: res_sq is the *squared* linear-model residual (logged as "residual norm")
        logger.info(f"GMRES iteration {k}, residual norm: {np.abs(res_sq):.5e}")
        if kmin <= k and np.abs(res_sq) < min_error:
            break
    return Q[:, :k+1], H, k, beta, w, res_sq


def trust_region_step(Q, H, k, beta, w, res_sq, radius, b_norm, evaluate,
                      radius_min, radius_max):
    """Accept a step dx = Q[:, :k] w, shrinking the radius until the true
    residual decreases, then choose the radius for the next Newton step.

    evaluate(dx) -> (residual norm at xi + dx, payload); the payload of the
    accepted step is returned so the caller can reuse it.

    Next radius from ratio = actual / predicted residual reduction:
        ratio > 0.75 and step limited by the radius -> double (<= radius_max)
        ratio < 0.25                                -> halve
        otherwise                                   -> keep
    Returns (dx, error, next_radius, payload)."""
    dx = np.matmul(Q[:, 0:k], w)
    test, payload = evaluate(dx)

    while test > 0.99*b_norm and radius > radius_min:
        radius = 0.5*radius
        w, res_sq = hookstep(H, beta, k, radius)
        dx = np.matmul(Q[:, 0:k], w)
        test, payload = evaluate(dx)

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
    return dx, np.abs(res_sq), max(radius_next, radius_min), payload
