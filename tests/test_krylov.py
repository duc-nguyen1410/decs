"""ecs_core.krylov on small dense matrices (no Dedalus needed)."""
import numpy as np
from ecs_core import krylov

rng = np.random.default_rng(0)
n = 8
A = np.eye(n) + 0.3*rng.standard_normal((n, n))
b = rng.standard_normal(n)
matvec = lambda v: A @ v


def test_gmres_solves_linear_system_with_large_radius():
    # gmres_hookstep solves A dx = b (newton.py passes A = DG - I and b = x - G(x))
    Q, H, k, beta, w, res_sq = krylov.gmres_hookstep(matvec, b, kmax=n, radius=1e6)
    dx = Q[:, :k] @ w
    assert np.linalg.norm(A @ dx - b) < 1e-6*np.linalg.norm(b)


def test_hookstep_respects_radius():
    Q, H, k, beta, w, res_sq = krylov.gmres_hookstep(matvec, b, kmax=n, radius=1e-2)
    assert np.linalg.norm(w) <= 1e-2*(1 + 1e-6)


def test_trust_region_grows_radius_after_good_limited_step():
    radius = 1.0   # full GMRES step has ||w|| ~ 15: limited, but reduces the residual > 1 %
    Q, H, k, beta, w, res_sq = krylov.gmres_hookstep(matvec, b, kmax=n, radius=radius)
    evaluate = lambda dx: (np.linalg.norm(b - A @ dx), None)   # exactly linear problem
    dx, error, radius_next, _ = krylov.trust_region_step(
        Q, H, k, beta, w, res_sq, radius, np.linalg.norm(b), evaluate,
        radius_min=1e-4, radius_max=4.0)
    assert np.linalg.norm(b - A @ dx) < np.linalg.norm(b)
    assert radius_next == 2*radius
