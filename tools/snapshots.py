"""
Read states from Dedalus snapshot files, e.g. those written by model.set_snapshots during a
simulation (a folder of <name>_s1.h5, <name>_s2.h5, ... with tasks named after model.fields).

    t, x = load_snapshot(model, 'run/snapshots', 120.0)      # state vector nearest t = 120
    T, X = load_snapshots(model, 'run/snapshots', (20, 300)) # X[i].ravel() is a state vector
"""
from pathlib import Path
import numpy as np
import h5py


def snapshot_files(folder):
    """The files <name>_s<k>.h5 in folder, in write order (s2 before s10)."""
    files = [f for f in Path(folder).glob('*_s*.h5') if f.stem.rsplit('_s', 1)[-1].isdigit()]
    if not files:
        raise FileNotFoundError(f"no snapshot files *_s<k>.h5 in {folder}")
    return sorted(files, key=lambda f: int(f.stem.rsplit('_s', 1)[-1]))


def _read(model, h, index):
    """Snapshot(s) `index` (int or slice) of the open file h as (..., C, *grid): the fields of
    the model in order, vector components first, so that one snapshot .ravel() is a state."""
    parts = []
    for field in model.fields:
        data = h['tasks'][field.name][index]
        if len(field.tensorsig) == 0:                    # scalar: add a component axis
            data = np.expand_dims(data, -model.dim - 1)
        parts.append(data)
    out = np.concatenate(parts, axis=-model.dim - 1)
    if out.shape[-model.dim:] != tuple(model.get_grid_shape()):
        raise ValueError(f"snapshot grid {out.shape[-model.dim:]} differs from the model grid "
                         f"{tuple(model.get_grid_shape())}")
    return out


def load_snapshot(model, folder, t, tol=None):
    """(t_found, x): the state vector of the snapshot nearest to time t. With tol, raise if no
    snapshot lies within tol of t."""
    best = None
    for f in snapshot_files(folder):
        with h5py.File(f, 'r') as h:
            times = h['scales/sim_time'][:]
            if len(times) == 0:
                continue
            k = int(np.argmin(np.abs(times - t)))
            if best is None or abs(times[k] - t) < abs(best[0] - t):
                best = (float(times[k]), f, k)
    if best is None or (tol is not None and abs(best[0] - t) > tol):
        raise ValueError(f"no snapshot near t = {t}" + (f" (nearest {best[0]:.4f})" if best else ""))
    with h5py.File(best[1], 'r') as h:
        return best[0], _read(model, h, best[2]).ravel().astype(np.float64)


def load_snapshots(model, folder, t_range=(-np.inf, np.inf), dtype=np.float32):
    """(T, X): times (n,) and states (n, C, *grid) of all snapshots with
    t_range[0] <= t < t_range[1]. float32 (default) halves the memory of long series."""
    T, X = [], []
    for f in snapshot_files(folder):
        with h5py.File(f, 'r') as h:
            times = h['scales/sim_time'][:]
            idx = np.flatnonzero((times >= t_range[0]) & (times < t_range[1]))
            if len(idx) == 0:
                continue
            a, b = idx[0], idx[-1] + 1
            X.append(_read(model, h, slice(a, b)).astype(dtype))
            T.append(times[a:b])
    if not T:
        raise ValueError(f"no snapshots with {t_range[0]} <= t < {t_range[1]} in {folder}")
    return np.concatenate(T), np.concatenate(X)
