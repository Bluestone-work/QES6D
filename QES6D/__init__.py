import os
import sys

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

from .model import QES6D, QES6D_o, QES6D_StrongHead, QES6D_EffNetV2, QES6D_EffNetV2_Advanced
from .loss import GeodesicPlusAxisLoss, RobustEulerAxisLoss

__version__ = "0.1.0"
__all__ = [
    "QES6D",
    "QES6D_o",
    "QES6D_StrongHead",
    "QES6D_EffNetV2",
    "QES6D_EffNetV2_Advanced",
    "GeodesicPlusAxisLoss",
    "RobustEulerAxisLoss",
]
