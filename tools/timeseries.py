"""
Recurrences and slow points of a long series of states (e.g. DNS snapshots from load_snapshots),
with the distances minimised over shifts along periodic directions. All snapshots are compared
at once with batched FFTs, so thousands of snapshots take minutes, where recurrence_search
(pairwise, for a few Poincare crossings) would take hours.

    T, X = load_snapshots(model, 'run/snapshots', (20, 300))
    dist, shifts = lag_distances(model, X, max_lag=50, shift_axes=('x', 'z'))
    returns = recurrences(T, dist, shifts, leave=0.5)       # PO / RPO guesses: Tp, ax, az
    slow = slow_points(T, dist, shifts)                     # EQ / TW guesses
    d = reference_distances(model, X, [x_sol], ('x', 'z'))  # visits near a known state

Shifts follow ECSSolver: dist[i, L] = min_a ||S(a) x_{i+L} - x_i|| / ||x_i||, so a returned
(Tp, ax, az) is a starting guess for solve(Tp=..., ax=..., az=...). The minimum over shifts is
refined between grid points (Newton on the Fourier series of the correlation), so small distances
are not limited by the grid spacing. They are accurate to the square root of round-off, since
||x - y||^2 = |x|^2 + |y|^2 - 2C cancels: about 1e-4 (relative) for float32 states, 1e-8 for
float64 (load_snapshots(..., dtype=np.float64)).
"""
import numpy as np


def _shift_axes(model, shift_axes):
    names = ('x', 'y', 'z') if model.dim == 3 else ('x', 'z')
    if model.bounded and names[-1] in shift_axes:
        raise ValueError(f"cannot shift along the bounded direction {names[-1]}")
    return sorted(shift_axes, key=names.index), names


def _spectra(model, X, shift_axes, batch=200):
    """FFT of X (n, C, *grid) along the shift axes, batch by batch, and the squared norms."""
    shift_axes, names = _shift_axes(model, shift_axes)
    axes = tuple(2 + names.index(n) for n in shift_axes)
    flat = X.reshape(len(X), -1)
    n2 = np.einsum('ij,ij->i', flat, flat, dtype=np.float64)
    if not axes:
        return X, n2
    ctype = np.complex64 if X.dtype == np.float32 else np.complex128
    Xh = np.concatenate([np.fft.rfftn(X[s:s + batch], axes=axes).astype(ctype)
                         for s in range(0, len(X), batch)])
    return Xh, n2


def _correlation(model, A, B, shift_axes):
    """C[b, k] = sum_xi a_b(xi) b_b(xi + k) for all grid shifts k along shift_axes (the other
    axes and the components summed), from the spectra A, B of _spectra, in double precision."""
    shift_axes, names = _shift_axes(model, shift_axes)
    shift_idx = [names.index(n) for n in shift_axes]
    summed = (1,) + tuple(2 + j for j in range(model.dim) if j not in shift_idx)
    P = (np.conj(A).astype(np.complex128)*B).sum(axis=summed)
    if not shift_axes:
        return P.real
    grid = model.get_grid_shape()
    return np.fft.irfftn(P, s=[grid[j] for j in shift_idx], axes=range(1, 1 + len(shift_idx)))


def _min_over_shifts(na, nb, C, shift_axes, n_newton=8):
    """min over shifts a of ||S(a) b - a||^2 = na + nb - 2 C(a), for a batch: the grid minimum
    of C (b, *shift grid), refined between grid points by Newton iterations on the Fourier
    series of C (exact for band-limited states, so no grid-spacing floor). Returns the squared
    distances (b,) and the shifts {'ax': (b,), ...} (fractions of the domain, in [-1/2, 1/2])."""
    if not shift_axes:
        return na + nb - 2*C, {}
    b, shape = C.shape[0], C.shape[1:]
    axes = tuple(range(1, 1 + len(shape)))
    r = np.arange(b)
    idx = np.unravel_index(np.argmax(C.reshape(b, -1), axis=1), shape)
    a_grid = np.stack([i/n for i, n in zip(idx, shape)], axis=1)          # (b, k)
    F = np.fft.fftn(C, axes=axes)/np.prod(shape)                           # C(a) = Re sum F e^{2 pi i m.a}
    m = [2j*np.pi*np.fft.fftfreq(n, 1/n).reshape([-1 if j == i else 1 for j in range(len(shape))])
         for i, n in enumerate(shape)]
    a = a_grid.copy()
    for _ in range(n_newton):
        E = F*np.exp(sum(mi*a[:, i].reshape((b,) + (1,)*len(shape)) for i, mi in enumerate(m)))
        grad = np.stack([(mi*E).real.sum(axis=axes) for mi in m], axis=1)
        hess = np.stack([np.stack([(mi*mj*E).real.sum(axis=axes) for mj in m], axis=1) for mi in m], axis=1)
        ok = np.all(np.linalg.eigvalsh(hess) < 0, axis=1)                  # near a maximum only
        step = np.zeros_like(a)
        step[ok] = -np.linalg.solve(hess[ok], grad[ok][..., None])[..., 0]
        step[np.any(np.abs(a + step - a_grid)*np.array(shape) > 1, axis=1)] = 0   # stay in the cell
        a += step
        if np.abs(step).max() < 1e-14:
            break
    Cmax = (F*np.exp(sum(mi*a[:, i].reshape((b,) + (1,)*len(shape)) for i, mi in enumerate(m)))).real.sum(axis=axes)
    Cmax = np.maximum(Cmax, C[(r, *idx)])                                  # never worse than the grid
    a = a - np.round(a)
    return na + nb - 2*Cmax, {'a' + name: a[:, i] for i, name in enumerate(shift_axes)}


def lag_distances(model, X, max_lag, shift_axes=('x',), batch=200):
    """dist (n, max_lag + 1): dist[i, L] = min_a ||S(a) x_{i+L} - x_i|| / ||x_i|| (NaN where
    i + L >= n), and shifts {'ax': (n, max_lag + 1), ...} the minimising a."""
    shift_axes, _ = _shift_axes(model, shift_axes)
    Xh, n2 = _spectra(model, X, shift_axes, batch)
    n = len(X)
    dist = np.full((n, max_lag + 1), np.nan, np.float32)
    dist[:, 0] = 0
    shifts = {'a' + name: np.full((n, max_lag + 1), np.nan, np.float32) for name in shift_axes}
    for s in shifts.values():
        s[:, 0] = 0
    for L in range(1, min(max_lag, n - 1) + 1):
        for s0 in range(0, n - L, batch):
            s1 = min(n - L, s0 + batch)
            C = _correlation(model, Xh[s0:s1], Xh[s0 + L:s1 + L], shift_axes)
            f0, sh = _min_over_shifts(n2[s0:s1], n2[s0 + L:s1 + L], C, shift_axes)
            dist[s0:s1, L] = np.sqrt(np.maximum(f0, 0)/n2[s0:s1])
            for key, a in sh.items():
                shifts[key][s0:s1, L] = a
    return dist, shifts


def recurrences(T, dist, shifts, leave=0.5, n_best=25, exclude=(5, 2)):
    """Returns of the series: lags L where dist[i, L] is a local minimum in L after the
    distance has first risen above `leave` (so near-identical neighbouring snapshots and slow
    passages do not count). The n_best smallest, at least exclude = (di, dL) apart in (i, L),
    as dicts with 'i', 'lag', 't', 'Tp', 'distance', the shifts ('ax', ...) and 'min_on_way'
    (smallest distance at shorter lags), best first."""
    d = np.where(np.isfinite(dist), dist, np.nan)
    left = np.fmax.accumulate(np.nan_to_num(d, nan=0), axis=1)     # max over lags 1..L
    ok = np.zeros(d.shape, bool)
    ok[:, 2:-1] = (d[:, 2:-1] <= d[:, 1:-2]) & (d[:, 2:-1] <= d[:, 3:]) & (left[:, 1:-2] >= leave)
    score = np.where(ok, d, np.inf)
    found, taken = [], np.zeros(d.shape, bool)
    for flat in np.argsort(score, axis=None):
        i, L = np.unravel_index(flat, d.shape)
        if not np.isfinite(score[i, L]) or len(found) >= n_best:
            break
        if taken[i, L]:
            continue
        found.append({'i': int(i), 'lag': int(L), 't': float(T[i]), 'Tp': float(T[i + L] - T[i]),
                      'distance': float(d[i, L]), **{k: float(v[i, L]) for k, v in shifts.items()},
                      'min_on_way': float(np.nanmin(d[i, 1:L]))})
        taken[max(0, i - exclude[0]):i + exclude[0] + 1, max(0, L - exclude[1]):L + exclude[1] + 1] = True
    return found


def slow_points(T, dist, shifts, n=40, min_separation=1.0):
    """Snapshots where the flow changes least between successive snapshots, drift removed
    (speed = dist[i, 1], per snapshot interval): starting guesses for equilibria, and for
    travelling waves with phase speed c = shift*L / (T[i+1] - T[i]). The n slowest, at least
    min_separation apart in time, as dicts with 'i', 't', 'speed' and the lag-1 shifts."""
    speed = dist[:, 1]
    picked = []
    for i in np.argsort(np.where(np.isfinite(speed), speed, np.inf)):
        if not np.isfinite(speed[i]) or len(picked) >= n:
            break
        if all(abs(T[i] - T[j]) >= min_separation for j in picked):
            picked.append(i)
    return [{'i': int(i), 't': float(T[i]), 'speed': float(speed[i]),
             **{k: float(v[i, 1]) for k, v in shifts.items()}} for i in picked]


def reference_distances(model, X, refs, shift_axes=('x',), batch=300):
    """d (n,): d[i] = min over the reference states r (list of state vectors, e.g. a solution and
    its reflections) and shifts a of ||S(a) r - x_i|| / ||x_i||: when the series visits them."""
    shift_axes, _ = _shift_axes(model, shift_axes)
    Xh, n2 = _spectra(model, X, shift_axes, batch)
    best = np.full(len(X), np.inf)
    for r in refs:
        Rh, r2 = _spectra(model, np.asarray(r, X.dtype).reshape((1,) + X.shape[1:]), shift_axes)
        for s0 in range(0, len(X), batch):
            C = _correlation(model, Xh[s0:s0 + batch], Rh, shift_axes)
            f0, _ = _min_over_shifts(n2[s0:s0 + batch], r2[0], C, shift_axes)
            best[s0:s0 + batch] = np.minimum(best[s0:s0 + batch], np.sqrt(np.maximum(f0, 0)/n2[s0:s0 + batch]))
    return best


def exponential_rate(T, D, window):
    """Growth rate lam of D ~ exp(lam t) over window = (t0, t1): least-squares slope of ln D.
    Compare with eigenvalues: approach to / escape from a solution (lam < 0 / lam > 0)."""
    w = (np.asarray(T) >= window[0]) & (np.asarray(T) <= window[1])
    if w.sum() < 2:
        raise ValueError(f"fewer than 2 samples in {window}")
    return float(np.polyfit(np.asarray(T)[w], np.log(np.asarray(D)[w]), 1)[0])
