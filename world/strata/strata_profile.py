"""Data model holding compiled strata profile metadata and pre-baked LUTs."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from world.strata.strata_keyframe import StrataKeyframe


@dataclass(frozen=True, slots=True)
class StrataProfile:
    """Holds compiled strata metadata, color LUT array, and speed LUT array."""

    profile_name: str
    keyframes: Tuple[StrataKeyframe, ...]
    color_lut: NDArray[np.uint8]
    speed_lut: NDArray[np.float32]

    def get_color(self, elevation: float) -> Tuple[int, int, int]:
        """Samples RGB color tuple from pre-baked LUT at given elevation."""
        idx: int = int(np.clip(elevation * 1023.0, 0.0, 1023.0))
        r, g, b = self.color_lut[idx]
        return int(r), int(g), int(b)

    def get_speed(self, elevation: float) -> float:
        """Samples kinematic speed multiplier from pre-baked LUT at height."""
        idx: int = int(np.clip(elevation * 1023.0, 0.0, 1023.0))
        return float(self.speed_lut[idx])

    @property
    def inverted_lut(self) -> NDArray[np.uint8]:
        """Returns contiguous colormap array reversed along elevation axis."""
        return np.ascontiguousarray(self.color_lut[::-1])
