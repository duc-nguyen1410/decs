"""Regression cases for the ECS solver: two Newton iterations of every solution type
(EQ, TW, PO, RPO), with fixed time steps and with CFL, plus Arnoldi stability, on a
32^2 SaltFinger grid. Used by test_regression.py and make_reference.py."""
import tempfile
import numpy as np
from physics import SaltFinger
from ecs_core import ECSSolver
from ecs_core import stability as stability_mod

CASES = {
    'EQ':  dict(Tp=0.02),
    'TW':  dict(Tp=0.02, Rzsearch=True, az=-0.1),
    'PO':  dict(Tp=0.02, Tsearch=True),
    'RPO': dict(Tp=0.02, Tsearch=True, Rxsearch=True, ax=0.05),
}
ARNOLDI_CASES = ('EQ', 'PO')


def run_cases(solver_cls=ECSSolver, extra_params=None):
    """Return {name: array}: the Newton unknowns xi after 2 iterations of every case
    ('<type>_cfl<0|1>') and the sorted Arnoldi multipliers ('<type>_cfl<0|1>_arnoldi')."""
    model = SaltFinger({'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'init_dt': 2e-4, 'ext': '.h5'},
                       sizes=(32, 32), bounds=(0.6, 1.0))
    model.build_problems()
    x0 = 5e-2*np.random.default_rng(0).standard_normal(model.size())
    x0 = model.flow_map(x0, 0.01)        # smooth the noise a little

    results = {}
    odir = tempfile.mkdtemp() + '/'
    for cfl in (False, True):
        model.use_CFL = cfl
        for name, kwargs in CASES.items():
            solver = solver_cls(model, {**(extra_params or {}), 'max_iter': 2, 'tol': 1e-14, 'trust_radius': 1.0,
                                       'krylov_dim': 6, 'krylov_dim_min': 6, 'odir': odir})
            xi = solver.solve(x0=x0.copy(), **kwargs)[0]
            key = f"{name}_cfl{int(cfl)}"
            results[key] = xi
            if name in ARNOLDI_CASES:    # on the final iterate (not converged: call Arnoldi directly)
                np.random.seed(1)        # Arnoldi start vector
                solver.Neigen = 5
                multipliers = stability_mod._stability_arnoldi(solver, xi)[1]
                results[key + '_arnoldi'] = np.sort_complex(multipliers)
    return results
