"""
New solutions / branches from the unstable modes of a converged solution x (any model).

    modes = load_unstable_modes(model, 'EQ/stability')       # [(lam, v)], v complex
    guesses = branch_guesses(model, x, modes, eps=(1e-2, 1e-1), odir='branches/')
    found = search_branches(solver, x, guesses, odir='branches/')   # Newton from each guess

A guess is x + eps |x| w/|w|, w = Re(v e^{i phi}): both signs for a real eigenvalue, n_phases
phases for a complex pair (the sign is the phase pi). What it may converge to, for an
equilibrium (base='eq') or a travelling wave (base='tw', in its co-moving frame):
    real lam        -> steady bifurcation: a new equilibrium (travelling wave)
    lam = mu + i w  -> Hopf bifurcation: a periodic orbit (relative periodic orbit), Tp ~ 2 pi/w
A mode that breaks a reflection symmetry of x can also give a drifting solution along the
reflected direction ('note' names the symmetries of x the mode keeps and breaks; imposing the
kept ones in Newton removes neutral directions). Newton from x + eps v reaches a new branch near
the bifurcation (lam small); far from it, follow the unstable manifold instead
(unstable_manifold_guesses: integrate x + eps w and take its slow points). search_branches
tells new solutions from the original x and from each other with symmetry_distance.
"""
import os
import csv
import logging
from pathlib import Path
import numpy as np
import scipy.io
import h5py
from physics.symmetry import Symmetry
from .symmetry_search import find_symmetries, symmetry_distance

logger = logging.getLogger(__name__)


def load_unstable_modes(model, stab_dir):
    """[(lam, v)]: the unstable eigenvalues lam (most unstable first) and their modes v (complex
    state vectors), one per complex-conjugate pair (the one with Im lam >= 0). Reads the files
    of ECSSolver's stability analysis: growthrate.mtx with unstable_mode_<i>[_imag], or the
    older eigen_unstable.h5 (modes on the dealiased grid)."""
    stab_dir = Path(stab_dir)
    if (stab_dir/f'unstable_mode_0{model.ext}').exists():
        lam = scipy.io.mmread(str(stab_dir/'growthrate.mtx')).ravel()
        lam = lam[lam.real > 0]
        vs = []
        for i in range(len(lam)):
            model.load_state(str(stab_dir/f'unstable_mode_{i}'))
            v = model.get_state().astype(complex)
            if (stab_dir/f'unstable_mode_{i}_imag{model.ext}').exists():
                model.load_state(str(stab_dir/f'unstable_mode_{i}_imag'))
                v += 1j*model.get_state()
            elif abs(lam[i].imag) > 0:
                logger.warning(f"mode {i} (lambda = {lam[i]:.4f}) has no imaginary part on disk "
                               "(saved before it was kept): its phases reduce to +-Re v")
            vs.append(v)
    elif (stab_dir/'eigen_unstable.h5').exists():
        with h5py.File(stab_dir/'eigen_unstable.h5', 'r') as f:
            lam, V = f['growthrate'][:], f['eigenvectors'][:]
        vs = [_from_dealiased(model, V[:, i]) for i in range(V.shape[1])]
    else:
        raise FileNotFoundError(f"no unstable modes in {stab_dir}")
    keep = [i for i in range(len(lam))
            if lam[i].imag >= 0 or not np.any(np.isclose(lam, np.conj(lam[i])))]
    return [(complex(lam[i]), vs[i]) for i in keep]


def _from_dealiased(model, v):
    """State vector (grid scale 1) of a complex vector given on the dealiased grid."""
    n = v.size//sum(model.dim if len(f.tensorsig) else 1 for f in model.fields)
    shape = (int(round(n**(1/model.dim))),)*model.dim
    if int(np.prod(shape)) != n:
        raise ValueError("cannot infer the dealiased grid shape of a non-cubic grid")
    parts = []
    for comp in (v.real, v.imag):
        cursor = 0
        for field in model.fields:
            k = model.dim if len(field.tensorsig) else 1
            data = comp[cursor:cursor + k*n].reshape(((k,) if k > 1 else ()) + shape)
            field.load_from_global_grid_data(data)
            cursor += k*n
        parts.append(model.get_state())
    return parts[0] + 1j*parts[1]


def _symmetry_note(model, x, w, sym_tol=1e-6):
    """Symmetries of x that the perturbation w keeps (sigma w = w) and breaks (sigma w = -w)."""
    keeps, breaks = [], []
    for r in find_symmetries(model, x):
        sigma = r['symmetry']
        if r['distance'] > sym_tol or not sigma.is_nontrivial():
            continue
        sw = model.apply_symmetry(w, sigma)
        if np.linalg.norm(sw - w) < 1e-3*np.linalg.norm(w):
            keeps.append(sigma)
        elif np.linalg.norm(sw + w) < 1e-3*np.linalg.norm(w):
            breaks.append(sigma)
    return keeps, breaks


def branch_guesses(model, x, modes, eps=(1e-2,), n_phases=4, base='eq', odir=None):
    """Initial guesses near x along each unstable mode (see the module docstring). Returns a
    list of dicts: 'x' (the guess), 'mode' (index in modes), 'lam', 'eps', 'phase', 'kind'
    ('eq', 'tw', 'po' or 'rpo'), 'Tp' (2 pi / Im lam for a Hopf mode, else None), 'keeps' /
    'breaks' (symmetries of x the perturbation keeps / breaks). With odir, each guess is saved
    as guess_<k> with a table guesses.csv."""
    steady, hopf = {'eq': ('eq', 'po'), 'tw': ('tw', 'rpo')}[base]
    guesses = []
    for m, (lam, v) in enumerate(modes):
        complex_mode = abs(lam.imag) > 1e-12 and np.any(v.imag)
        phases = 2*np.pi*np.arange(n_phases)/n_phases if complex_mode else (0.0, np.pi)
        for phase in phases:
            w = (v*np.exp(1j*phase)).real
            keeps, breaks = _symmetry_note(model, x, w)
            for e in eps:
                guesses.append({'x': x + e*np.linalg.norm(x)*w/np.linalg.norm(w), 'mode': m,
                                'lam': lam, 'eps': e, 'phase': float(phase),
                                'kind': hopf if abs(lam.imag) > 1e-12 else steady,
                                'Tp': 2*np.pi/abs(lam.imag) if abs(lam.imag) > 1e-12 else None,
                                'keeps': keeps, 'breaks': breaks})
    if odir is not None:
        _save_table(model, guesses, odir)
    return guesses


def unstable_manifold_guesses(model, x, modes, eps=1e-3, T=1.0, dt=0.01, n_slow=3,
                              shift_axes=('x', 'z'), n_phases=2, base='eq', odir=None):
    """Guesses away from x: integrate x + eps w (w as in branch_guesses) for a time T, recording
    every dt, and keep the n_slow slowest states of each trajectory (lag-1 distance, drift
    removed, after it has left x by 10 eps). Slow states with a drift are marked 'tw', with
    the drift as starting shift. Same dicts as branch_guesses, plus 't' and the shifts."""
    from .timeseries import lag_distances, slow_points
    if model.bounded:
        shift_axes = [a for a in shift_axes if a != 'z']
    guesses = []
    for g in branch_guesses(model, x, modes, eps=(eps,), n_phases=n_phases, base=base):
        T_rec, S = [], []
        def record(solver):
            if not T_rec or solver.sim_time >= T_rec[-1] + dt:
                T_rec.append(solver.sim_time)
                S.append(model.current_state())
        model.flow_map(g['x'], T, callback=record)
        X = np.array(S).reshape((len(S), -1) + tuple(model.get_grid_shape()))
        dist, shifts = lag_distances(model, X, 1, shift_axes)
        left = np.array([symmetry_distance(model, s, x)[0] for s in S]) > 10*eps
        dist[~left, 1] = np.nan                                  # still next to x
        for p in slow_points(np.array(T_rec), dist, shifts, n=n_slow, min_separation=5*dt):
            drift = {k: v*1.0 for k, v in p.items() if k.startswith('a') and abs(v) > 1e-3}
            guesses.append({**g, 'x': S[p['i']], 't': p['t'], 'speed': p['speed'],
                            'kind': ('tw' if drift else 'eq') if g['Tp'] is None else g['kind'],
                            **{k: v/dt for k, v in drift.items()}})   # shift per unit time
    if odir is not None:
        _save_table(model, guesses, odir)
    return guesses


def _save_table(model, guesses, odir):
    os.makedirs(odir, exist_ok=True)
    rows = []
    for k, g in enumerate(guesses):
        model.save_state(os.path.join(odir, f'guess_{k}'), g['x'])
        rows.append({'guess': k, 'mode': g['mode'], 'lam_re': g['lam'].real, 'lam_im': g['lam'].imag,
                     'eps': g['eps'], 'phase': g['phase'], 'kind': g['kind'], 'Tp': g['Tp'],
                     't': g.get('t'), 'keeps': ' '.join(map(str, g['keeps'])),
                     'breaks': ' '.join(map(str, g['breaks']))})
    if model.dist.comm.rank == 0:
        with open(os.path.join(odir, 'guesses.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def _drift_axes(model, g):
    """Directions to search a drift in: those measured on the trajectory ('ax', ... in g), else
    the periodic directions reflected by a symmetry the perturbation breaks, else x."""
    axes = [n for n in 'xyz' if f'a{n}' in g]
    if not axes:
        axes = [n for n in 'xyz' if any(getattr(s, 's' + n) == -1 for s in g['breaks'])]
    if model.bounded:
        axes = [n for n in axes if n != 'z']
    return [n for n in axes if model.dim == 3 or n != 'y'] or ['x']


def search_branches(solver, x, guesses, odir, Tp=None, solve_kwargs=None, new_tol=1e-3):
    """Newton (solver.solve) from every guess, each run in odir/guess_<k>/, and classify the
    converged results as 'original' (within new_tol of x up to symmetries), 'new', or
    'same as <k>' (an earlier new one). Solve arguments by kind (override with solve_kwargs =
    {'eq': {...}, ...}): eq: fixed Tp (Tp, default solver.Tp); po: Tsearch from the guess Tp;
    tw / rpo: as eq / po with a shift search along the drift (measured by
    unstable_manifold_guesses, else the directions of the reflections the perturbation breaks). Returns the list of results (dicts) and writes
    odir/branches.csv."""
    model = solver.model
    Tp = solver.Tp if Tp is None else Tp
    found, results = [], []
    for k, g in enumerate(guesses):
        kw = {'eq': dict(Tp=Tp), 'tw': dict(Tp=Tp), 'po': dict(Tsearch=True, Tp=g['Tp']),
              'rpo': dict(Tsearch=True, Tp=g['Tp'])}[g['kind']]
        if g['kind'] in ('tw', 'rpo'):
            for n in _drift_axes(model, g):
                kw[f'R{n}search'] = True
                kw[f'a{n}'] = g.get(f'a{n}', 0.0)*kw['Tp']      # drift per unit time -> per Tp
        kw.update((solve_kwargs or {}).get(g['kind'], {}))
        solver.odir = os.path.join(odir, f'guess_{k}/')
        model.odir = solver.odir
        logger.info(f"guess {k}: mode {g['mode']} (lambda = {g['lam']:.4f}), eps {g['eps']:g}, "
                    f"phase {g['phase']:.2f}, {g['kind']}, solve({kw})")
        xi, success, res, _, props = solver.solve(x0=g['x'], **kw)
        x_new = solver.unpack_xi(xi)[0]
        status, d_orig = 'not converged', np.nan
        if success:
            d_orig = symmetry_distance(model, x_new, x)[0]
            if d_orig < new_tol:
                status = 'original'
            else:
                same = [j for j, y in found if symmetry_distance(model, x_new, y)[0] < new_tol]
                status = f'same as {same[0]}' if same else 'new'
                if not same:
                    found.append((k, x_new))
        results.append({'guess': k, 'mode': g['mode'], 'eps': g['eps'], 'phase': g['phase'],
                        'kind': g['kind'], 'success': success, 'residual': res,
                        'distance_to_original': d_orig, 'status': status,
                        **{n: props[n] for n in ('Tp', 'ax', 'ay', 'az') if n in props}})
        logger.info(f"guess {k}: {status} (residual {res:.2e}, distance to original {d_orig:.3e})")
    if model.dist.comm.rank == 0:
        os.makedirs(odir, exist_ok=True)
        with open(os.path.join(odir, 'branches.csv'), 'w', newline='') as f:
            fields = list(dict.fromkeys(k for r in results for k in r))
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
    return results
