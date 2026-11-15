"""
Deterministic 2D Value Noise generator using vectorized NumPy math.
"""

import numpy as np
from numpy.typing import NDArray


class ValueNoise:
    """
    Evaluates continuous 2D Value Noise fields via scalar lattice hashing.
    """

    def __init__(self, seed: int = 420) -> None:
        """
        Initializes master seed for bitwise coordinate hashing.
        """
        self.seed: np.uint32 = np.uint32(seed & 0xFFFFFFFF)

    def evaluate(
        self,
        x_grid: NDArray[np.float32],
        y_grid: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Evaluates 2D Value Noise across coordinate arrays in [-1, 1]."""
        floor_x = np.floor(x_grid)
        floor_y = np.floor(y_grid)
        x0 = floor_x.astype(np.int32)
        y0 = floor_y.astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        xf = x_grid - floor_x
        yf = y_grid - floor_y

        u_fade = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
        v_fade = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

        v00 = self._hash_2d(x0, y0)
        v10 = self._hash_2d(x1, y0)
        v01 = self._hash_2d(x0, y1)
        v11 = self._hash_2d(x1, y1)

        nx0 = v00 + u_fade * (v10 - v00)
        nx1 = v01 + u_fade * (v11 - v01)
        raw_val = nx0 + v_fade * (nx1 - nx0)

        return np.clip(raw_val, -1.0, 1.0)

    def _hash_2d(
        self,
        i: NDArray[np.int32],
        j: NDArray[np.int32],
    ) -> NDArray[np.float32]:
        """
        Hashes integer lattice coordinates into scalar float32 values.
        """
        p1: np.uint32 = np.uint32(374761393)
        p2: np.uint32 = np.uint32(668265263)
        p3: np.uint32 = np.uint32(1274126177)
        p4: np.uint32 = np.uint32(2654435761)

        h = (i.astype(np.uint32) * p1) + (j.astype(np.uint32) * p2)
        h = h + self.seed
        h = (h ^ (h >> 13)) * p3
        h = (h ^ (h >> 16)) * p4
        raw_bits = h ^ (h >> 15)

        return (raw_bits.astype(np.float32) / 2147483648.0) - 1.0
