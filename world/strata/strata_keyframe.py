"""
Data model representing an individual elevation keyframe for strata profiles.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class StrataKeyframe:
    """
    Immutable specification holding elevation, speed, and color thresholds.
    """

    elevation: float
    speed_multiplier: float
    color: Tuple[int, int, int]
