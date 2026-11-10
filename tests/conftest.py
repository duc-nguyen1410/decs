import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repository root

# FFTW 'measure' planning picks transform algorithms by timing, so results can change in
# the last bits from one run to the next. 'estimate' is deterministic, which lets the
# regression test compare results exactly.
try:
    from dedalus.tools.config import config
    config['transforms-fftw']['PLANNING_RIGOR'] = 'estimate'
    config['parallelism-fftw']['PLANNING_RIGOR'] = 'estimate'
except ImportError:      # Dedalus not installed: only the pure-numpy tests run
    pass
