"""Regenerate tests/data/regression_reference.npz from the current code.

Run this ONLY after a change that is meant to alter the numerical results (a new time
stepper, a different algorithm, ...), once you have checked the new results are right:

    python tests/make_reference.py

Then commit the updated .npz together with the change.
"""
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import conftest  # noqa: F401  (repository path and deterministic FFTW planning)
import numpy as np
from regression_cases import run_cases

if __name__ == '__main__':
    out = here/'data'/'regression_reference.npz'
    out.parent.mkdir(exist_ok=True)
    results = run_cases()
    np.savez(out, **results)
    print(f"Wrote {len(results)} reference arrays to {out}")
