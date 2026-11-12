# Approach / escape analysis straight from stored snapshots (no re-integration): distance of every
# snapshot in a window to an equilibrium (up to shifts and reflections), with exp(lambda t) of
# its eigenvalues. Cheap for screening many passes; the snapshot interval must be short against
# 1/|lambda| for the slopes to mean anything (re-integrate with approach_escape.py otherwise).
# Needs a snapshot series: set snap_dir to your run (here the Lx = 0.6 DNS, not in the repo).
# to run: python3 approach_escape_snapshots.py
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
from tools import load_snapshots, reference_distances, reflections, exponential_rate

snap_dir = ('/mnt/d/Projects/2026/2026_ecs_saltfinger/Publication_data/Samples/'
            'Rrho40_Ra1e5_Pr7_tau0.01/Lx0.6_DNS/snapshots')
eq_dir = 'Lx0.6_WF'                  # equilibrium: solution.nc and stability/growthrate.mtx
t_window = (99.0, 104.0)             # snapshots to analyse (a pass near the equilibrium)
windows = {'approach': (100.3, 100.9), 'escape': (101.9, 102.2)}   # fit windows (set by eye)
odir = 'test_approach_escape_snapshots/'

model = SaltFinger({'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0}, sizes=(128, 128), bounds=(0.6, 1.0))
model.load_state(eq_dir + '/solution')
x_eq = model.get_state()
lam = scipy.io.mmread(eq_dir + '/stability/growthrate.mtx').ravel()
lam_u = lam[np.argmax(lam.real)]
lam_s = lam[lam.real < 0][np.argmax(lam[lam.real < 0].real)]

T, X = load_snapshots(model, snap_dir, t_window, dtype=np.float64)
D = reference_distances(model, X, [model.apply_symmetry(x_eq, s) for s in reflections(model)], ('x', 'z'))
k = int(np.argmin(D))
logger.info(f"{len(T)} snapshots; closest pass {D[k]:.3e} at t = {T[k]:.2f}")
rates = {name: exponential_rate(T, D, w) for name, w in windows.items()}
for (name, w), ref in zip(windows.items(), (lam_s, lam_u)):
    logger.info(f"{name}: fitted rate {rates[name]:+.3f} over t = {w[0]}-{w[1]}; eigenvalue {ref.real:+.3f}")

fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
ax.semilogy(T, D, 'k.-', label='snapshots')
for (name, w), ref, c in zip(windows.items(), (lam_s, lam_u), ('tab:blue', 'tab:red')):
    i0 = np.argmin(np.abs(T - w[0]))
    ax.semilogy(T, D[i0]*np.exp(ref.real*(T - T[i0])), '--', color=c,
                label=fr'{name}: $\lambda$ = {ref.real:.3f} (fit {rates[name]:.3f})')
    ax.axvspan(*w, color=c, alpha=0.08)
ax.set_ylim(0.5*D.min(), 2*D.max())
ax.set_xlabel('t')
ax.set_ylabel(r'$\min_{\sigma}\,\|x(t) - \sigma x_{eq}\| / \|x(t)\|$')
ax.legend(fontsize=7.5)
Path(odir).mkdir(exist_ok=True)
fig.savefig(odir + 'approach_escape_snapshots.png', dpi=130)
np.savez(odir + 'approach_escape_snapshots.npz', t=T, d=D, **{f'rate_{n}': r for n, r in rates.items()})
logger.info(f"wrote {odir}approach_escape_snapshots.png and .npz")
