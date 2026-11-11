"""
Figure of the Poincaré section and recurrence search on the relative periodic orbit RPO2
(Lx = 0.8), for docs/finding_po_rpo.md:

    left : Ecf(t) over three periods, the section level and the refined crossings
    right: recurrence plot d(t_i, t_j) = min_a ||S(a) x(t_j) - x(t_i)|| / ||x(t_i)|| (x-shifts),
           whose dark diagonals at t_j - t_i = Tp, 2 Tp, ... are the returns of the orbit

Run from the examples folder:  python3 plot_poincare_figure.py   (a few minutes)
Writes ../docs/media/poincare_rpo2.png.
"""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent # Get decs paths
sys.path.insert(0, str(root_dir)) # Insert into sys.path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from physics import SaltFinger
from ecs_core import PoincareSection, recurrence_search, shifted_distance

Tp = 0.07380265343504276
model = SaltFinger(params={'Ra': 1e5, 'Pr': 7.0, 'tau': 0.01, 'Rrho': 40.0, 'init_dt': 1e-4, 'ext': '.nc'},
                   sizes=(128, 128), bounds=(0.8, 1.0))
model.build_problems()
model.use_CFL = False
model.load_state('Lx0.8_RPO2/solution')
x0 = model.get_state()
ecf = lambda m: m.cross_flow_energy()

# Ecf every step and a state every `every` steps over three periods
T, every = 3*Tp, 20
t_E, E, t_X, X = [], [], [0.0], [x0]
def record(solver):
    t_E.append(solver.sim_time); E.append(ecf(model))
    if solver.iteration % every == 0:
        t_X.append(solver.sim_time); X.append(model.current_state())
    return False
model.flow_map(x0, T, callback=record)
level = np.mean(E)

# the section itself (refined crossings) and the recurrence search on its crossings
crossings = PoincareSection(model, ecf, level=level, direction=+1).run(x0, T)
t_c = np.array([c['t'] for c in crossings])
best = recurrence_search(model, [c['x'] for c in crossings], t_c, shift_axes=('x',), T_max=1.5*Tp, n_best=1)[0]

# recurrence plot of the sampled states
n = len(X)
D = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        D[i, j] = D[j, i] = shifted_distance(model, X[i], X[j], ('x',))[0]

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True,
                        gridspec_kw={'width_ratios': [1.35, 1]})
ax = axs[0]
ax.plot(np.array(t_E)/Tp, E, color='0.25', lw=1.0, label='Ecf(t)')
ax.axhline(level, color='tab:blue', ls='--', lw=1.0, label=f'section Ecf = {level:.2f}')
ax.plot(t_c/Tp, [c['g'] for c in crossings], 'o', color='tab:red', ms=7, label='crossings (upward)')
for k in range(4):
    ax.axvline(k, color='0.8', lw=0.6, zorder=0)
ax.set_xlabel('t / Tp'); ax.set_ylabel('Ecf')
ax.set_title(f'RPO2: crossings every {np.mean(np.diff(t_c)):.7f} (Tp = {Tp:.7f})', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
ax = axs[1]
im = ax.imshow(D, origin='lower', extent=[0, t_X[-1]/Tp, 0, t_X[-1]/Tp], cmap='magma', vmin=0, vmax=np.percentile(D, 95))
ax.set_xlabel('$t_j$ / Tp'); ax.set_ylabel('$t_i$ / Tp')
ax.set_title(f"recurrence plot (x-shifts removed)\nbest return: Tp = {best['Tp']:.6f}, ax = {best['ax']:+.3f}, "
             f"d = {best['distance']:.0e}", fontsize=10)
fig.colorbar(im, ax=ax, label='$d(t_i, t_j)$')
out = root_dir/'docs'/'media'/'poincare_rpo2.png'
fig.savefig(out, dpi=130)
print(f"wrote {out}: {len(crossings)} crossings, best return {best}")
