"""
Deterministic 2D Simplex noise generator using vectorized NumPy math.
"""

import math
import random
from typing import Final, List
import numpy as np
from numpy.typing import NDArray

F2: Final[float] = 0.5 * (math.sqrt(3.0) - 1.0)
G2: Final[float] = (3.0 - math.sqrt(3.0)) / 6.0

GRAD2: Final[List[List[int]]] = [
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [0, 1], [0, -1]
]

GRAD2_ARR: Final[NDArray[np.float32]] = np.array(
    GRAD2, dtype=np.float32
)


class SimplexNoise:
    """
    Evaluates 2D Simplex noise fields from a deterministic integer seed.
    """

    def __init__(self, seed: int = 420) -> None:
        """
        Initializes permutation lookup tables deterministically from seed.
        """
        self.seed: int = seed
        rng = random.Random(seed)
        perm = list(range(256))
        rng.shuffle(perm)

        doubled_perm = perm + perm
        self._p_arr: NDArray[np.int32] = np.array(
            doubled_perm, dtype=np.int32
        )
        self._perm_grad_arr: NDArray[np.int32] = np.array(
            [i % 8 for i in doubled_perm], dtype=np.int32
        )

    def evaluate(
        self,
        x_grid: NDArray[np.float32],
        y_grid: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Evaluates 2D Simplex noise across coordinate arrays in [-1.0, 1.0].
        """
        s_val = (x_grid + y_grid) * F2
        i_coord = np.floor(x_grid + s_val).astype(np.int32)
        j_coord = np.floor(y_grid + s_val).astype(np.int32)

        t_val = (i_coord + j_coord) * G2
        x0 = x_grid - (i_coord - t_val)
        y0 = y_grid - (j_coord - t_val)

        i1 = np.where(x0 > y0, 1, 0)
        j1 = np.where(x0 > y0, 0, 1)

        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + (2.0 * G2)
        y2 = y0 - 1.0 + (2.0 * G2)

        ii = i_coord & 255
        jj = j_coord & 255

        p = self._p_arr
        g_arr = GRAD2_ARR
        p_g = self._perm_grad_arr

        gi0 = p_g[p[ii + p[jj]]]
        gi1 = p_g[p[ii + i1 + p[jj + j1]]]
        gi2 = p_g[p[ii + 1 + p[jj + 1]]]

        g0 = g_arr[gi0]
        g1 = g_arr[gi1]
        g2 = g_arr[gi2]

        t0 = np.maximum(0.0, 0.5 - (x0 * x0) - (y0 * y0))
        t1 = np.maximum(0.0, 0.5 - (x1 * x1) - (y1 * y1))
        t2 = np.maximum(0.0, 0.5 - (x2 * x2) - (y2 * y2))

        t0_sq = t0 * t0
        t1_sq = t1 * t1
        t2_sq = t2 * t2

        n0 = (t0_sq * t0_sq) * (g0[..., 0] * x0 + g0[..., 1] * y0)
        n1 = (t1_sq * t1_sq) * (g1[..., 0] * x1 + g1[..., 1] * y1)
        n2 = (t2_sq * t2_sq) * (g2[..., 0] * x2 + g2[..., 1] * y2)

        raw_val = 70.0 * (n0 + n1 + n2)
        return np.clip(raw_val, -1.0, 1.0)
