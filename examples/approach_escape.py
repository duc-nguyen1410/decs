# Approach to and escape from an equilibrium: distance of a trajectory to the equilibrium (up to
# shifts and reflections) on a log scale, against exp(lambda t) with the equilibrium's eigenvalues.
# Straight stretches whose slope equals an eigenvalue show that the trajectory passes near the
# equilibrium along that eigen-direction (e.g. near a homoclinic / heteroclinic connection).
#
# Here: a DNS state (Lx = 0.6, t = 100.5) that passes within 0.06 of the equilibrium WF.
#   scale = None : the real pass. The escape follows WF's leading eigenvalue (5.39); the
#                  approach starts at distance ~1, where the dynamics are nonlinear, so its
#                  slope is no eigenvalue.
#   scale = 1e-3 : the same deviation from WF shrunk 1000 times: the dynamics are linear and the
#                  slopes must equal eigenvalues (a check of the method and of the eigenvalues).
# to run: python3 approach_escape.py   (a few minutes)
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path

import logging
logger = logging.getLogger(__name__)
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from physics import SaltFinger
from tools import reference_distances, reflections, symmetry_distance, exponential_rate

eq_dir = 'Lx0.6_WF'                  # equilibrium: solution.nc and stability/growthrate.mtx
start_state = 'Lx0.6_WF/dns_t100.5'  # a state that will pass near the equilibrium
t_start = 100.5                      # its time (for the time axis only)
T_run = 2.5                          # integration time
dt_sample = 0.005                    # time between recorded states
scale = None                         # None: real pass; e.g. 1e-3: deviation shrunk to linear
windows = None                       # fit windows {'approach': (t0, t1), 'escape': (t0, t1)},
                                     # None: middle third (in log distance) of the fall / rise
odir = 'test_approach_escape/'

params = {'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'init_dt': 2e-4, 'ext': '.nc'}
model = SaltFinger(params=params, sizes=(128, 128), bounds=(0.6, 1.0))
model.build_problems()
model.use_CFL = True
model.load_state(eq_dir + '/solution')
x_eq = model.get_state()
copies = [model.apply_symmetry(x_eq, s) for s in reflections(model)]   # with shifts: the group orbit
lam = scipy.io.mmread(eq_dir + '/stability/growthrate.mtx').ravel()     # most unstable first
lam_u = lam[np.argmax(lam.real)]                                        # leading unstable
lam_s = lam[lam.real < 0][np.argmax(lam[lam.real < 0].real)]            # weakest stable
logger.info(f"eigenvalues: leading {lam_u:.4f}, weakest stable {lam_s:.4f}")

model.load_state(start_state)
x0 = model.get_state()
if scale is not None:            # the copy of the equilibrium nearest x0, plus the shrunk deviation
    d0, sigma = symmetry_distance(model, x0, x_eq)
    x_near = model.apply_symmetry(x_eq, sigma)
    x0 = x_near + scale*(x0 - x_near)
    logger.info(f"start {d0:.3f} from {eq_dir} (via {sigma}), deviation scaled by {scale:g}")

# integrate and record the state every dt_sample
T, S = [], []
def record(solver):
    if not T or solver.sim_time >= T[-1] + dt_sample:
        T.append(solver.sim_time)
        S.append(model.current_state())      # float64: distances resolved down to ~1e-8
model.flow_map(x0, T_run, callback=record)
T = t_start + np.array(T)
X = np.array(S).reshape((len(S), -1) + tuple(model.get_grid_shape()))
D = reference_distances(model, X, copies, shift_axes=('x', 'z'))
del X, S

# fit windows: by default the middle third (in ln D) of the fall before and the rise after the minimum
k = int(np.argmin(D))
if windows is None:
    def middle_third(idx, top):
        lo, hi = np.log(D[k]), np.log(top)
        sel = idx[(np.log(D[idx]) >= lo + (hi - lo)/3) & (np.log(D[idx]) <= lo + 2*(hi - lo)/3)]
        return (T[sel].min(), T[sel].max())
    windows = {'approach': middle_third(np.arange(0, k), D[:k].max()),
               'escape': middle_third(np.arange(k + 1, len(D)), D[k + 1:].max())}
rates = {name: exponential_rate(T, D, w) for name, w in windows.items()}
logger.info(f"closest pass: distance {D[k]:.3e} at t = {T[k]:.3f}")
for (name, w), ref in zip(windows.items(), (lam_s, lam_u)):
    logger.info(f"{name}: fitted rate {rates[name]:+.3f} over t = {w[0]:.3f}-{w[1]:.3f}; "
                f"eigenvalue {ref.real:+.3f}" + (f" (imag {ref.imag:+.2f})" if ref.imag else ""))

# figure as in Zheng, Tuckerman & Schneider, Fig. 7(b): lines anchored at the fit windows
fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
ax.semilogy(T, D, 'k', label='trajectory')
for (name, w), ref, c in zip(windows.items(), (lam_s, lam_u), ('tab:blue', 'tab:red')):
    i0 = np.argmin(np.abs(T - w[0]))
    ax.semilogy(T, D[i0]*np.exp(ref.real*(T - T[i0])), '--', color=c,
                label=fr'{name}: $e^{{\lambda t}}$, $\lambda$ = {ref.real:.3f} (fit {rates[name]:.3f})')
    ax.axvspan(*w, color=c, alpha=0.08)
ax.set_ylim(0.5*D.min(), 2*D.max())
ax.set_xlabel('t')
ax.set_ylabel(r'$\min_{\sigma}\,\|x(t) - \sigma x_{eq}\| / \|x(t)\|$')
ax.set_title(f"{start_state} vs {eq_dir}" + (f", deviation x{scale:g}" if scale else ''), fontsize=9)
ax.legend(fontsize=7.5)
Path(odir).mkdir(exist_ok=True)
fig.savefig(odir + 'approach_escape.png', dpi=130)
np.savez(odir + 'approach_escape.npz', t=T, d=D, lam_u=lam_u, lam_s=lam_s,
         **{f'window_{n}': w for n, w in windows.items()}, **{f'rate_{n}': r for n, r in rates.items()})
logger.info(f"wrote {odir}approach_escape.png and .npz")
