"""
Tools around the ECS solver that need only a physical model: initial guesses (Poincare
sections, recurrences and slow points of snapshot series, edge tracking), symmetry detection,
reading snapshots, grid changes, guesses for new branches from unstable modes and helical
(screw-symmetric) states.

    from tools import PoincareSection, recurrence_search, EdgeTracker, find_symmetries
"""
from .edge import EdgeTracker
from .poincare import PoincareSection, recurrence_search, shifted_distance
from .symmetry_search import find_symmetries, reflections, symmetry_distance, nearest_solution
from .change_grid import change_grid
from .snapshots import snapshot_files, load_snapshot, load_snapshots
from .timeseries import (lag_distances, recurrences, slow_points, reference_distances,
                         exponential_rate)
from .branches import (load_unstable_modes, branch_guesses, unstable_manifold_guesses,
                       search_branches)
from .helical import (screw_symmetry, helical_guess, mean_flow, winding_number, helicity,
                      mean_flow_helicity, relative_helicities, helical_diagnostics,
                      screw_image_grid, screw_project_grid, shift_grid, best_screw_center,
                      c2x_image_grid, helical_group_project_grid)

__all__ = ['EdgeTracker', 'PoincareSection', 'recurrence_search', 'shifted_distance',
           'find_symmetries', 'reflections', 'symmetry_distance', 'nearest_solution',
           'change_grid', 'snapshot_files', 'load_snapshot', 'load_snapshots',
           'lag_distances', 'recurrences', 'slow_points', 'reference_distances',
           'exponential_rate', 'load_unstable_modes', 'branch_guesses',
           'unstable_manifold_guesses', 'search_branches',
           'screw_symmetry', 'helical_guess', 'mean_flow', 'winding_number', 'helicity',
           'mean_flow_helicity', 'relative_helicities', 'helical_diagnostics',
           'screw_image_grid', 'screw_project_grid', 'shift_grid', 'best_screw_center',
           'c2x_image_grid', 'helical_group_project_grid']
