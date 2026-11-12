"""
Move a state to another grid and/or domain size by spectral interpolation: the Fourier /
Chebyshev coefficients of every field are copied mode by mode (truncated or zero-padded).
Modes keep their wavenumber index, so a new domain size stretches the state (a mode with m
periods in Lx_old has m periods in Lx_new). A 2-D state can be extended to 3-D: it is put
on the ky = 0 plane, with spanwise velocity v = 0.

    model_old.load_state('solution')
    x_new = change_grid(model_old, model_new)
    model_new.save_state('solution_new', x_new)
"""
import numpy as np


def _mode_keys(basis, L):
    """One key per coefficient: (wavenumber index m, occurrence) for Fourier (RealFourier
    stores cos and sin of each m), (polynomial degree, 0) for Chebyshev."""
    if not hasattr(basis, 'wavenumbers'):
        return [(i, 0) for i in range(basis.size)]
    keys, seen = [], {}
    for m in np.rint(basis.wavenumbers*L/(2*np.pi)).astype(int):
        keys.append((m, seen.get(m, 0)))
        seen[m] = seen.get(m, 0) + 1
    return keys


def _index_map(keys_old, keys_new, nyquist=None):
    """Index pairs (old, new) of the coefficients present on both grids, without |m| = nyquist."""
    pos = {k: i for i, k in enumerate(keys_old)}
    pairs = [(pos[k], i) for i, k in enumerate(keys_new) if k in pos and abs(k[0]) != nyquist]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def change_grid(model_old, model_new):
    """State of model_old (its current fields) on the grid / domain of model_new.
    Both models must be the same physical model in the same mode and with the same z basis;
    model_new may be 3-D when model_old is 2-D. Returns model_new.get_state()."""
    if model_old.mode != model_new.mode or model_old.bounded != model_new.bounded:
        raise ValueError("models must have the same mode and the same z basis (bounded or not)")
    if model_new.dim < model_old.dim:
        raise ValueError("cannot change a 3-D state to 2-D")
    names_old = [b.coord.name for b in model_old.bases]
    maps = []
    for b_new, L_new in zip(model_new.bases, model_new.bounds):
        keys_new = _mode_keys(b_new, L_new)
        name = b_new.coord.name
        if name in names_old:
            i = names_old.index(name)
            b_old = model_old.bases[i]
            # the Nyquist mode of the coarser Fourier grid has no counterpart on the finer one
            nyquist = (min(b_old.size, b_new.size)//2
                       if hasattr(b_new, 'wavenumbers') and b_old.size != b_new.size else None)
            maps.append(_index_map(_mode_keys(b_old, model_old.bounds[i]), keys_new, nyquist))
        else:                                    # y added to a 2-D state: ky = 0 only
            maps.append(_index_map([(0, 0)], keys_new))
    idx_old = np.ix_(*[m[0] for m in maps])
    idx_new = np.ix_(*[m[1] for m in maps])
    for f_old, f_new in zip(model_old.fields, model_new.fields):
        c_old = f_old.allgather_data('c')
        c_new = np.zeros_like(f_new.allgather_data('c'))
        vector = len(f_old.tensorsig) > 0
        if model_new.dim > model_old.dim:
            if vector:                           # (u, w) -> (u, 0, w)
                c_old = np.stack([c_old[0], np.zeros_like(c_old[0]), c_old[1]])
            c_old = np.expand_dims(c_old, axis=-2)   # y axis before z
        if vector:
            c_new[(slice(None),) + idx_new] = c_old[(slice(None),) + idx_old]
        else:
            c_new[idx_new] = c_old[idx_old]
        f_new.load_from_global_coeff_data(c_new)
    return model_new.get_state()
