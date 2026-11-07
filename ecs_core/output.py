"""
File and log output for the Newton solver: run log, flow_properties.csv,
per-iteration parameter files and the solution-property summary.
All writes happen on MPI rank 0 only.
"""
import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


def search_values(solver, Tp, ax, ay, az):
    """Ordered {name: value} of the searched parameters (and wave speeds c = a*Tp*L
    for relative equilibria), e.g. {'Tp': ..} or {'ax': .., 'cx': ..}."""
    out = {}
    if solver.Tsearch:
        out['Tp'] = Tp
    for name, a, on, L in (('x', ax, solver.Rxsearch, solver.model.bounds[0]),
                           ('y', ay, solver.Rysearch, solver.model.bounds[1]),
                           ('z', az, solver.Rzsearch, solver.model.bounds[-1])):
        if on:
            out[f'a{name}'] = a
            if not solver.Tsearch:
                out[f'c{name}'] = a*Tp*L
    return out


def param_string(solver, Tp, ax, ay, az):
    """', Tp: ..., ax: ...' for the searched parameters (used in log lines)."""
    s = ""
    if solver.Tsearch:  s += f", Tp: {Tp}"
    if solver.Rxsearch: s += f", ax: {ax}"
    if solver.Rysearch: s += f", ay: {ay}"
    if solver.Rzsearch: s += f", az: {az}"
    return s


class RunLog:
    """Line-buffered run log (output.log), written on rank 0 only."""

    def __init__(self, solver):
        self.file = None
        if solver.model.dist.comm.rank == 0:
            os.makedirs(solver.odir, exist_ok=True)
            self.file = open(os.path.join(solver.odir, solver.log_filename), 'w', buffering=1)

    def write(self, line):
        if self.file is not None:
            self.file.write(line + "\n")

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def write_header(self, solver):
        model = solver.model
        lines = ['---------------------------------',
                 '--- Parameters ------------------',
                 '---------------------------------']
        lines += [f"{key}: {value}" for key, value in model.params.items()]
        lines += [f"dim: {model.dim}", f"sizes: {model.sizes}", f"bounds: {model.bounds}",
                  f"bounded: {model.bounded}", f"dealias: {model.dealias}", f"mode: {model.mode}",
                  f"CFL: {model.use_CFL}", f"initial dt: {model.init_dt}"]
        if solver.sigma.is_nontrivial():
            lines.append(f"symmetry: {str(solver.sigma)}")
        lines += [f"ecs_odir: {solver.odir}", f"tol: {solver.tol}", f"max_iter: {solver.max_iter}",
                  f"initial Tp: {solver.Tp}", f"trust radius: {solver.trust_radius}",
                  f"krylov_dim: {solver.krylov_dim}", f"krylov_dim_min: {solver.krylov_dim_min}",
                  f"projectNeutralDrift: {solver.projectNeutralDrift}",
                  f"computeStability: {solver.computeStability}", f"stability_method: {solver.stability_method}",
                  f"Neigen: {solver.Neigen}",
                  f"save_ecs_history: {solver.save_ecs_history}",
                  f"save_snapshots: {model.save_snapshots}",
                  f"save_flowproperties: {model.save_flowproperties}",
                  f"save_meanprofiles: {model.save_meanprofiles}",
                  '---------------------------------']
        for line in lines:
            self.write(line)


def reset_flow_properties(solver, filename="flow_properties.csv"):
    """Remove the CSV of a previous run in solver.odir, so a new Newton run starts a fresh file."""
    if solver.model.dist.comm.rank == 0:
        file_path = os.path.join(solver.odir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def write_flow_properties(solver, xi, filename="flow_properties.csv"):
    """Append searched parameters + model flow properties of xi to a CSV file."""
    x0, Tp, ax, ay, az = solver.unpack_xi(xi)
    solver.model.set_state(x0)
    properties = solver.model.get_flow_properties()
    if solver.model.dist.comm.rank != 0:
        return
    params = search_values(solver, Tp, ax, ay, az)
    keys = list(properties.keys())
    file_path = os.path.join(solver.odir, filename)
    if not os.path.isfile(file_path):
        with open(file_path, mode='w') as f:
            f.write(", ".join(list(params.keys()) + keys) + "\n")
    with open(file_path, mode='a') as f:
        values = [f"{v:.12f}" for v in params.values()]
        values += [f"{float(properties[k]):.12f}" for k in keys]
        f.write(", ".join(values) + "\n")
    # Also print to terminal for real-time monitoring
    logger.info(f"Prop Check: {' | '.join(f'{k}={float(properties[k]):.6f}' for k in keys)}")


def write_iteration_files(solver, Tp, ax, ay, az, residual_norm):
    """Current searched parameters (T.txt, ax.txt, ...) and residual (tol.txt)."""
    if solver.model.dist.comm.rank != 0:
        return
    files = {}
    if solver.Tsearch:  files['T.txt'] = Tp
    if solver.Rxsearch: files['ax.txt'] = ax
    if solver.Rysearch: files['ay.txt'] = ay
    if solver.Rzsearch: files['az.txt'] = az
    files['tol.txt'] = residual_norm
    for name, value in files.items():
        with open(os.path.join(solver.odir, name), 'w') as f:
            f.write(str(value))


# ----------------------------------------------------------------------
# Continuation output
# ----------------------------------------------------------------------
def write_mu(solver, mu):
    """Continuation parameter of the current search (odir/search-<i>/mu.txt)."""
    if solver.model.dist.comm.rank == 0:
        with open(os.path.join(solver.odir, "mu.txt"), mode='w') as f:
            f.write(f"{mu}")


def write_branch_point(cont, mu, properties, filename="flow_properties.csv"):
    """Append mu, flow properties and the search directory of a converged branch
    point to <continuation odir>/flow_properties.csv."""
    if cont.ECSSolver.model.dist.comm.rank != 0:
        return
    keys = list(properties.keys())
    file_path = os.path.join(cont.odir, filename)
    if not os.path.isfile(file_path):
        with open(file_path, mode='w') as f:
            f.write(f"{cont.mu_name}, " + ", ".join(keys) + ", dir\n")
    with open(file_path, mode='a') as f:
        values = [f"{mu:.12f}"] + [f"{float(properties[k]):.12f}" for k in keys]
        f.write(", ".join(values) + f", search-{cont.isearch-1}/\n")


CHECKPOINT = "continuation_state.json"


def write_continuation_state(cont, ds):
    """Checkpoint of a continuation (odir/continuation_state.json): everything
    restart_continuation needs besides the saved states in search-<i>/."""
    if cont.ECSSolver.model.dist.comm.rank != 0:
        return
    state = {'mu_name': cont.mu_name,
             'continuation_type': cont.continuation_type,
             'predictor': cont.predictor,
             'searched': cont._searched(),
             'isearch': cont.isearch,
             'ds': float(ds),
             'mu': [float(m) for m in cont.mu_history],
             's': [float(s) for s in cont.s_history],
             'dirs': list(cont.dir_history)}
    for name in cont._searched():
        state[name] = [float(v) for v in cont._history(name)]
    path = os.path.join(cont.odir, CHECKPOINT)
    with open(path + '.tmp', 'w') as f:
        json.dump(state, f, indent=1)
    os.replace(path + '.tmp', path)   # never leave a half-written checkpoint


def read_continuation_state(cont):
    """Checkpoint written by write_continuation_state, or None if there is none."""
    path = os.path.join(cont.odir, CHECKPOINT)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_branch_csv(cont, filename="flow_properties.csv"):
    """Rebuild the branch from the summary CSV of a continuation (for runs without
    a checkpoint): mu, searched parameters (Tp, ax, ...) and folder of each point."""
    path = os.path.join(cont.odir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No {CHECKPOINT} or {filename} in {cont.odir}: nothing to restart from.")
    with open(path) as f:
        rows = [line.strip().split(', ') for line in f if line.strip()]
    header, rows = rows[0], rows[1:]
    state = {'mu_name': header[0],
             'mu': [float(r[0]) for r in rows],
             'dirs': [r[-1] for r in rows]}
    for name in ('Tp', 'ax', 'ay', 'az'):
        if name in header:
            i = header.index(name)
            state[name] = [float(r[i]) for r in rows]
    return state
