"""Extracts strided 2D slope gradient matrices directly from ChunkManager."""

import math
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from world.chunks.chunk_manager import ChunkManager


class ViewportHeightSampler:
    """Samples chunk slope gradients into persistent lightmap-sized buffers."""

    def __init__(self) -> None:
        """Initializes persistent gradient buffers and dimension tracker."""
        self._target_shape: Tuple[int, int] = (0, 0)
        self._gx_buffer: NDArray[np.float32] = np.empty(
            (0, 0), dtype=np.float32
        )
        self._gy_buffer: NDArray[np.float32] = np.empty(
            (0, 0), dtype=np.float32
        )

    def sample_viewport_gradients(
        self,
        focus_x: float,
        focus_y: float,
        view_w: int,
        view_h: int,
        chunk_manager: ChunkManager,
        target_w: int,
        target_h: int,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Extracts 2D gx, gy slope matrices strided to lightmap size."""
        origin_x = focus_x - (float(view_w) / 2.0)
        origin_y = focus_y - (float(view_h) / 2.0)

        sample_wx = np.floor(
            np.linspace(
                origin_x,
                origin_x + float(max(1, view_w - 1)),
                target_w,
            )
        ).astype(np.int32)
        sample_wy = np.floor(
            np.linspace(
                origin_y,
                origin_y + float(max(1, view_h - 1)),
                target_h,
            )
        ).astype(np.int32)

        return self._sample_at_world_coords(
            sample_wx, sample_wy, chunk_manager
        )

    def sample_tile_gradients(
        self,
        snapped_ox: float,
        snapped_oy: float,
        view_w: int,
        view_h: int,
        chunk_manager: ChunkManager,
        tile_size: int,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Samples one gradient per visible tile at exact tile centroids."""
        cols = (view_w // tile_size) + 2
        rows = (view_h // tile_size) + 2
        t_f = float(tile_size)

        sample_wx = np.floor(
            snapped_ox + ((np.arange(cols, dtype=np.float32) + 0.5) * t_f)
        ).astype(np.int32)
        sample_wy = np.floor(
            snapped_oy + ((np.arange(rows, dtype=np.float32) + 0.5) * t_f)
        ).astype(np.int32)

        return self._sample_at_world_coords(
            sample_wx, sample_wy, chunk_manager
        )

    def _sample_at_world_coords(
        self,
        sample_wx: NDArray[np.int32],
        sample_wy: NDArray[np.int32],
        chunk_manager: ChunkManager,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Populates persistent gradient buffers from loaded chunks."""
        target_w = int(sample_wx.size)
        target_h = int(sample_wy.size)

        if (target_h, target_w) != self._target_shape:
            self._target_shape = (target_h, target_w)
            self._gx_buffer = np.zeros(
                (target_h, target_w), dtype=np.float32
            )
            self._gy_buffer = np.zeros(
                (target_h, target_w), dtype=np.float32
            )
        else:
            self._gx_buffer.fill(0.0)
            self._gy_buffer.fill(0.0)

        c_sz = chunk_manager.chunk_size
        min_cx = math.floor(float(sample_wx[0]) / float(c_sz))
        max_cx = math.floor(float(sample_wx[-1]) / float(c_sz))
        min_cy = math.floor(float(sample_wy[0]) / float(c_sz))
        max_cy = math.floor(float(sample_wy[-1]) / float(c_sz))

        for cy in range(min_cy, max_cy + 1):
            chk_wy = cy * c_sz
            y0 = int(np.searchsorted(sample_wy, chk_wy, side="left"))
            y1 = int(np.searchsorted(sample_wy, chk_wy + c_sz, side="left"))
            if y0 >= y1:
                continue

            ly = (sample_wy[y0:y1] - chk_wy)[:, None]

            for cx in range(min_cx, max_cx + 1):
                chunk = chunk_manager.chunks.get((cx, cy))
                if chunk is None:
                    continue

                chk_wx = cx * c_sz
                x0 = int(np.searchsorted(sample_wx, chk_wx, side="left"))
                x1 = int(
                    np.searchsorted(sample_wx, chk_wx + c_sz, side="left")
                )
                if x0 >= x1:
                    continue

                lx = sample_wx[x0:x1] - chk_wx
                self._gx_buffer[y0:y1, x0:x1] = chunk.gx[ly, lx]
                self._gy_buffer[y0:y1, x0:x1] = chunk.gy[ly, lx]

        return self._gx_buffer, self._gy_buffer
