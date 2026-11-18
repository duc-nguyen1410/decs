"""
Physical models built on Dedalus.

    from physics import SaltFinger, Symmetry
"""
# Base classes and tools
from .base import FluidModel
from .symmetry import Symmetry
from .double_diffusion_base import DoubleDiffusion

# Validated models
from .salt_finger import SaltFinger
from .salt_finger_fraser import SaltFingerFraser
from .thermohaline_shear import ThermohalineShear

# Not yet validated
from .double_diffusion import (BoundedSaltFinger, DiffusiveConvection,
                               ShearedDiffusiveConvection, WallShearedDiffusiveConvection)
from .magneto import (BoundedQuasiStaticMagnetoConvection,
                      BoundedQuasiStaticMagnetoConvection_SingleMode)

__all__ = [
    'FluidModel', 'Symmetry', 'DoubleDiffusion',
    'SaltFinger', 'SaltFingerFraser', 'ThermohalineShear',
    'BoundedSaltFinger', 'DiffusiveConvection', 'ShearedDiffusiveConvection',
    'WallShearedDiffusiveConvection',
    'BoundedQuasiStaticMagnetoConvection', 'BoundedQuasiStaticMagnetoConvection_SingleMode',
]
