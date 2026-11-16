"""Extracts strided 2D slope gradient matrices directly from ChunkManager."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from world.chunks.chunk_manager import ChunkManager


@dataclass(frozen=True, slots=True)
class FallbackGradientGrid:
    """World-anchored coarse gradient fallback for unloaded chunks."""

    gx: NDArray[np.float32]
    gy: NDArray[np.float32]
    origin_x: float
    origin_y: float
    step_scale: float


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
        fallback: Optional[FallbackGradientGrid] = None,
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
            sample_wx, sample_wy, chunk_manager, fallback
        )

    def sample_tile_gradients(
        self,
        snapped_ox: float,
        snapped_oy: float,
        view_w: int,
        view_h: int,
        chunk_manager: ChunkManager,
        tile_size: int,
        fallback: Optional[FallbackGradientGrid] = None,
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
            sample_wx, sample_wy, chunk_manager, fallback
        )

    def _sample_at_world_coords(
        self,
        sample_wx: NDArray[np.int32],
        sample_wy: NDArray[np.int32],
        chunk_manager: ChunkManager,
        fallback: Optional[FallbackGradientGrid] = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Populates persistent gradient buffers from chunks or fallback."""
        target_w = int(sample_wx.size)
        target_h = int(sample_wy.size)
        self._ensure_buffers(target_h, target_w)
        if target_w <= 0 or target_h <= 0:
            return self._gx_buffer, self._gy_buffer

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

            sub_wy = sample_wy[y0:y1]
            ly = (sub_wy - chk_wy)[:, None]

            for cx in range(min_cx, max_cx + 1):
                chk_wx = cx * c_sz
                x0 = int(np.searchsorted(sample_wx, chk_wx, side="left"))
                x1 = int(
                    np.searchsorted(sample_wx, chk_wx + c_sz, side="left")
                )
                if x0 >= x1:
                    continue

                chunk = chunk_manager.chunks.get((cx, cy))
                if chunk is None:
                    self._fill_missing_slice(
                        x0, x1, y0, y1, sample_wx[x0:x1], sub_wy, fallback
                    )
                    continue

                lx = sample_wx[x0:x1] - chk_wx
                self._gx_buffer[y0:y1, x0:x1] = chunk.gx[ly, lx]
                self._gy_buffer[y0:y1, x0:x1] = chunk.gy[ly, lx]

        return self._gx_buffer, self._gy_buffer

    def _ensure_buffers(self, target_h: int, target_w: int) -> None:
        """Allocates persistent gradient matrices when output size changes."""
        if (target_h, target_w) == self._target_shape:
            return
        self._target_shape = (target_h, target_w)
        self._gx_buffer = np.zeros((target_h, target_w), dtype=np.float32)
        self._gy_buffer = np.zeros((target_h, target_w), dtype=np.float32)

    def _fill_missing_slice(
        self,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
        sub_wx: NDArray[np.int32],
        sub_wy: NDArray[np.int32],
        fallback: Optional[FallbackGradientGrid],
    ) -> None:
        """Fills an unloaded chunk region from fallback grid or zeros."""
        if fallback is None:
            self._gx_buffer[y0:y1, x0:x1] = 0.0
            self._gy_buffer[y0:y1, x0:x1] = 0.0
            return

        fb_h, fb_w = fallback.gx.shape
        if fb_h <= 0 or fb_w <= 0:
            self._gx_buffer[y0:y1, x0:x1] = 0.0
            self._gy_buffer[y0:y1, x0:x1] = 0.0
            return

        inv_step = np.float32(1.0 / max(1e-4, fallback.step_scale))
        ox = np.float32(fallback.origin_x)
        oy = np.float32(fallback.origin_y)
        fx = np.rint(
            (sub_wx.astype(np.float32) - ox) * inv_step
        ).astype(np.int32)
        fy = np.rint(
            (sub_wy.astype(np.float32) - oy) * inv_step
        ).astype(np.int32)

        vx = (fx >= 0) & (fx < fb_w)
        vy = (fy >= 0) & (fy < fb_h)
        if not (np.any(vx) and np.any(vy)):
            self._gx_buffer[y0:y1, x0:x1] = 0.0
            self._gy_buffer[y0:y1, x0:x1] = 0.0
            return

        if np.all(vx) and np.all(vy):
            self._gx_buffer[y0:y1, x0:x1] = fallback.gx[
                fy[:, None], fx[None, :]
            ]
            self._gy_buffer[y0:y1, x0:x1] = fallback.gy[
                fy[:, None], fx[None, :]
            ]
            return

        self._gx_buffer[y0:y1, x0:x1] = 0.0
        self._gy_buffer[y0:y1, x0:x1] = 0.0
        x_idx = np.nonzero(vx)[0]
        y_idx = np.nonzero(vy)[0]
        xs = int(x_idx[0])
        xe = int(x_idx[-1]) + 1
        ys = int(y_idx[0])
        ye = int(y_idx[-1]) + 1

        ix = fx[xs:xe]
        iy = fy[ys:ye]
        dst_y = slice(y0 + ys, y0 + ye)
        dst_x = slice(x0 + xs, x0 + xe)
        self._gx_buffer[dst_y, dst_x] = fallback.gx[iy[:, None], ix[None, :]]
        self._gy_buffer[dst_y, dst_x] = fallback.gy[iy[:, None], ix[None, :]]
