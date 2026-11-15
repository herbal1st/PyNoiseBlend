"""
Deterministic 2D Perlin noise generator using quintic smoothstep.
"""

import math
import random
from typing import Final, List
import numpy as np
from numpy.typing import NDArray

GRAD2: Final[List[List[int]]] = [
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [0, 1], [0, -1]
]

GRAD2_ARR: Final[NDArray[np.float32]] = np.array(
    GRAD2, dtype=np.float32
)


class PerlinNoise:
    """
    Evaluates 2D Perlin noise fields from a deterministic integer seed.
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
        """Evaluates 2D Perlin noise across coordinate arrays in [-1, 1]."""
        floor_x = np.floor(x_grid)
        floor_y = np.floor(y_grid)
        cell_x = floor_x.astype(np.int32) & 255
        cell_y = floor_y.astype(np.int32) & 255

        xf = x_grid - floor_x
        yf = y_grid - floor_y

        u_fade = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
        v_fade = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

        p = self._p_arr
        g_arr = GRAD2_ARR
        p_g = self._perm_grad_arr

        aa = p_g[p[p[cell_x] + cell_y]]
        ab = p_g[p[p[cell_x] + cell_y + 1]]
        ba = p_g[p[p[cell_x + 1] + cell_y]]
        bb = p_g[p[p[cell_x + 1] + cell_y + 1]]

        g00 = g_arr[aa]
        g10 = g_arr[ba]
        g01 = g_arr[ab]
        g11 = g_arr[bb]

        dot00 = g00[..., 0] * xf + g00[..., 1] * yf
        dot10 = g10[..., 0] * (xf - 1.0) + g10[..., 1] * yf
        dot01 = g01[..., 0] * xf + g01[..., 1] * (yf - 1.0)
        dot11 = g11[..., 0] * (xf - 1.0) + g11[..., 1] * (yf - 1.0)

        x1 = dot00 + u_fade * (dot10 - dot00)
        x2 = dot01 + u_fade * (dot11 - dot01)
        raw_val = x1 + v_fade * (x2 - x1)

        return np.clip(raw_val, -1.0, 1.0)
