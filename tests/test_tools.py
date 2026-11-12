"""tools/: snapshot reading, batched distances of a series (lag_distances, reference_distances
against shifted_distance), recurrences, slow points and symmetry_distance, on a 32^2 grid."""
import numpy as np
import pytest

de = pytest.importorskip("dedalus.public")
h5py = pytest.importorskip("h5py")
from physics import SaltFinger, Symmetry
from tools import (load_snapshot, load_snapshots, lag_distances, recurrences, slow_points,
                   reference_distances, symmetry_distance, nearest_solution, shifted_distance)

PARAMS = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'init_dt': 2e-4}
N = 32


@pytest.fixture(scope='module')
def model():
    m = SaltFinger(PARAMS, sizes=(N, N), bounds=(0.6, 1.0))
    m.build_problems()
    return m


@pytest.fixture(scope='module')
def smooth(model):
    """Five different smooth states."""
    rng = np.random.default_rng(1)
    return [model.flow_map(1e-2*rng.standard_normal(model.size()), 0.004) for _ in range(5)]


def shift(model, x, **a):
    return model.apply_symmetry(x, Symmetry(**a))


def write_snapshots(folder, chunks):
    """Dedalus-like files snapshots_s<k>.h5 from {k: (times, states (n, 4, N, N))}."""
    folder.mkdir()
    for k, (t, X) in chunks.items():
        with h5py.File(folder/f'snapshots_s{k}.h5', 'w') as h:
            h['scales/sim_time'] = t
            h['tasks/u'], h['tasks/te'], h['tasks/sa'] = X[:, :2], X[:, 2], X[:, 3]


def test_load_snapshots_order_and_states(model, smooth, tmp_path):
    X = np.array([x.reshape(4, N, N) for x in smooth])
    # s10 holds the latest times: files must be read in numeric, not text, order
    write_snapshots(tmp_path/'snaps', {1: ([0.0, 0.1], X[:2]), 2: ([0.2, 0.3], X[2:4]),
                                       10: ([0.4], X[4:])})
    T, Y = load_snapshots(model, tmp_path/'snaps', dtype=np.float64)
    assert np.array_equal(T, [0.0, 0.1, 0.2, 0.3, 0.4]) and np.array_equal(Y, X)
    T, Y = load_snapshots(model, tmp_path/'snaps', (0.1, 0.35))
    assert np.allclose(T, [0.1, 0.2, 0.3]) and Y.dtype == np.float32
    t, x = load_snapshot(model, tmp_path/'snaps', 0.29)
    assert t == 0.3 and np.array_equal(x, smooth[3])
    with pytest.raises(ValueError):
        load_snapshot(model, tmp_path/'snaps', 0.7, tol=0.05)


@pytest.mark.parametrize('axes', [('x', 'z'), ('x',), ()])
def test_lag_and_reference_distances_match_direct_distances(model, smooth, axes):
    """lag_distances: the distance is that of the returned shift (computed with the model's own
    translation) and not above the grid minimum of shifted_distance; reference_distances: the
    same minimum as lag_distances on the pair."""
    X = np.array([x.reshape(4, N, N) for x in smooth])
    dist, shifts = lag_distances(model, X, max_lag=3, shift_axes=axes)
    for i, L in [(0, 1), (1, 3), (2, 2)]:
        a = {k: float(v[i, L]) for k, v in shifts.items()}
        direct = np.linalg.norm(shift(model, smooth[i + L], **a) - smooth[i])/np.linalg.norm(smooth[i])
        assert dist[i, L] == pytest.approx(direct, rel=1e-5)
        if axes:
            assert dist[i, L] <= shifted_distance(model, smooth[i], smooth[i + L], axes)[0]*(1 + 1e-6)
    assert np.isnan(dist[4, 1])
    refs = smooth[:2]
    d = reference_distances(model, X, refs, axes)
    for i in range(5):
        pair = [lag_distances(model, np.array([X[i], r.reshape(4, N, N)]), 1, axes)[0][0, 1] for r in refs]
        assert d[i] == pytest.approx(min(pair), rel=1e-5, abs=1e-6)


def test_recurrences_and_slow_points_of_a_drifting_periodic_series(model, smooth):
    """x_j = S(-j a) y_{j mod 3}: returns at lag 3 with shift 3a (S(3a) x_{j+3} = x_j); one
    nearly frozen pair (x_7 ~ shifted x_6) is the slowest point."""
    a = {'ax': 2/N, 'az': -1/N}
    X = [shift(model, smooth[j % 3], ax=-j*a['ax'], az=-j*a['az']) for j in range(12)]
    X[7] = shift(model, X[6], ax=-a['ax'], az=-a['az']) + 1e-3*smooth[4]
    T = 0.1*np.arange(12)
    X = np.array([x.reshape(4, N, N) for x in X])
    dist, shifts = lag_distances(model, X, max_lag=6, shift_axes=('x', 'z'))
    found = recurrences(T, dist, shifts, leave=0.5, n_best=3, exclude=(0, 0))
    good = [r for r in found if r['i'] not in (4, 6, 7)]          # untouched by the frozen pair
    assert good and good[0]['lag'] == 3 and good[0]['Tp'] == pytest.approx(0.3)
    assert good[0]['distance'] < 1e-3
    assert (good[0]['ax'], good[0]['az']) == pytest.approx((3*a['ax'], 3*a['az']), abs=1e-6)
    slow = slow_points(T, dist, shifts, n=1)[0]
    assert slow['i'] == 6 and slow['speed'] < 1e-2
    assert (slow['ax'], slow['az']) == pytest.approx((a['ax'], a['az']), abs=1e-3)


def test_symmetry_distance_and_nearest_solution(model, smooth):
    sigma = Symmetry(sx=-1, ax=5/N, az=-3/N)
    x = smooth[0]
    y = model.apply_symmetry(x, sigma)          # y = sigma(x); find s with s(y) = x
    d, s = symmetry_distance(model, x, y)
    assert d < 1e-12
    assert np.linalg.norm(model.apply_symmetry(y, s) - x) < 1e-12*np.linalg.norm(x)
    best = nearest_solution(model, x, {'other': smooth[1], 'copy': y})
    assert [b[1] for b in best] == ['copy', 'other'] and best[0][0] < 1e-12


def test_distances_are_refined_between_grid_points(model, smooth):
    """A copy shifted by a fraction of a grid cell (and reflected) is at distance ~0 (to the
    square root of round-off), not at the grid-shift floor (above 1e-3 here)."""
    x = smooth[0]
    b = {'ax': 0.37/N, 'az': -0.21/N}
    y = shift(model, x, **b)
    pair = np.array([x.reshape(4, N, N), y.reshape(4, N, N)])
    dist, shifts = lag_distances(model, pair, 1, ('x', 'z'))      # S(a) y = x: a = -b
    assert dist[0, 1] < 1e-7                  # sqrt of round-off: d^2 = |x|^2 + |y|^2 - 2C
    assert (shifts['ax'][0, 1], shifts['az'][0, 1]) == pytest.approx((-b['ax'], -b['az']), abs=1e-9)
    assert reference_distances(model, pair[1:], [x], ('x', 'z'))[0] < 1e-7
    assert reference_distances(model, pair[1:].astype(np.float32), [x], ('x', 'z'))[0] < 1e-3   # float32 round-off
    yr = model.apply_symmetry(x, Symmetry(sx=-1, ax=0.29/N, az=0.43/N))
    assert symmetry_distance(model, x, yr)[0] < 1e-12
    grid_only = min(np.linalg.norm(np.roll(y.reshape(4, N, N), (k, l), axis=(1, 2)).ravel() - x)
                    for k in (-1, 0, 1) for l in (-1, 0, 1))/np.linalg.norm(x)
    assert grid_only > 1e-3                                         # the floor that is avoided
