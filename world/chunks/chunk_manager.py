"""Spatial chunk manager orchestrating elliptical streaming and blits."""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import pygame

from world.chunks.chunk import Chunk
from world.chunks.chunk_provider import ChunkProvider


class ChunkManager:
    """Manages infinite spatial chunk caching, streaming, and blitting."""

    def __init__(self, provider: ChunkProvider) -> None:
        """Initializes storage, binds provider, and clamps margins."""
        self.provider: ChunkProvider = provider
        self.chunk_size: int = provider.chunk_profile.chunk_size
        self.chunks: Dict[Tuple[int, int], Chunk] = {}

        self.load_margin: float = max(
            0.75, provider.chunk_profile.load_margin
        )
        self.unload_margin: float = max(
            self.load_margin + 0.25, provider.chunk_profile.unload_margin
        )

        self._center_cx: float = 0.0
        self._center_cy: float = 0.0
        self._inv_a_load_sq: float = 1.0
        self._inv_b_load_sq: float = 1.0
        self._has_boundary: bool = False

    def set_provider(self, provider: ChunkProvider) -> None:
        """Rebinds active ChunkProvider and flushes loaded chunks."""
        self.provider = provider
        self.chunk_size = provider.chunk_profile.chunk_size
        self.load_margin = max(0.75, provider.chunk_profile.load_margin)
        self.unload_margin = max(
            self.load_margin + 0.25, provider.chunk_profile.unload_margin
        )
        self.clear()

    def rebake_all_surfaces(self, new_lut: NDArray[np.uint8]) -> None:
        """Updates provider LUT and re-indexes cached chunk surfaces."""
        self.provider.set_color_lut(new_lut)
        for chunk in self.chunks.values():
            self.provider.rebake_chunk_surface(chunk, new_lut)

    def update_focus(
        self,
        focus_x: float,
        focus_y: float,
        view_w: float,
        view_h: float,
        max_new_chunks: Optional[int] = None,
    ) -> int:
        """Streams chunks center-outward and purges distant entries."""
        c_sz: float = float(self.chunk_size)
        center_cx: float = focus_x / c_sz
        center_cy: float = focus_y / c_sz
        int_cx: int = math.floor(center_cx)
        int_cy: int = math.floor(center_cy)

        half_w: float = view_w / (2.0 * c_sz)
        half_h: float = view_h / (2.0 * c_sz)

        corner_factor: float = math.sqrt(2.0)
        a_load: float = (half_w * corner_factor) + self.load_margin
        b_load: float = (half_h * corner_factor) + self.load_margin
        a_unload: float = (half_w * corner_factor) + self.unload_margin
        b_unload: float = (half_h * corner_factor) + self.unload_margin

        inv_a_load_sq: float = 1.0 / max(1e-4, a_load * a_load)
        inv_b_load_sq: float = 1.0 / max(1e-4, b_load * b_load)
        inv_a_unload_sq: float = 1.0 / max(1e-4, a_unload * a_unload)
        inv_b_unload_sq: float = 1.0 / max(1e-4, b_unload * b_unload)

        self._center_cx = center_cx
        self._center_cy = center_cy
        self._inv_a_load_sq = inv_a_load_sq
        self._inv_b_load_sq = inv_b_load_sq
        self._has_boundary = True

        search_x: int = math.ceil(a_load)
        search_y: int = math.ceil(b_load)
        missing: List[Tuple[float, int, int]] = []

        for dy in range(-search_y, search_y + 1):
            for dx in range(-search_x, search_x + 1):
                tcx: int = int_cx + dx
                tcy: int = int_cy + dy

                diff_x: float = float(tcx) + 0.5 - center_cx
                diff_y: float = float(tcy) + 0.5 - center_cy

                dist_sq: float = (
                    (diff_x * diff_x * inv_a_load_sq)
                    + (diff_y * diff_y * inv_b_load_sq)
                )

                if dist_sq <= 1.0 and (tcx, tcy) not in self.chunks:
                    radial_sq = (diff_x * diff_x) + (diff_y * diff_y)
                    missing.append((radial_sq, tcx, tcy))

        if max_new_chunks is not None and len(missing) > max_new_chunks:
            missing.sort(key=lambda item: item[0])
            to_generate = missing[:max_new_chunks]
        else:
            to_generate = missing

        for _, tcx, tcy in to_generate:
            chunk = self.provider.generate_chunk(tcx, tcy)
            self.chunks[(tcx, tcy)] = chunk

        for coord_key in list(self.chunks.keys()):
            cx_idx, cy_idx = coord_key
            diff_x = float(cx_idx) + 0.5 - center_cx
            diff_y = float(cy_idx) + 0.5 - center_cy

            dist_sq = (
                (diff_x * diff_x * inv_a_unload_sq)
                + (diff_y * diff_y * inv_b_unload_sq)
            )

            if dist_sq > 1.0:
                del self.chunks[coord_key]

        return max(0, len(missing) - len(to_generate))

    def draw_chunks(
        self,
        surface: pygame.Surface,
        focus_x: float,
        focus_y: float,
    ) -> None:
        """Renders visible chunks using top-left anchor grid snapping."""
        rw: float = float(surface.get_width())
        rh: float = float(surface.get_height())
        min_cx, max_cx, min_cy, max_cy = self._compute_visible_bounds(
            focus_x, focus_y, rw, rh
        )

        c_sz: float = float(self.chunk_size)
        focus_cx: float = focus_x / c_sz
        focus_cy: float = focus_y / c_sz
        center_px: float = rw / 2.0
        center_py: float = rh / 2.0

        anchor_x: int = int(
            round(center_px + ((float(min_cx) - focus_cx) * c_sz))
        )
        anchor_y: int = int(
            round(center_py + ((float(min_cy) - focus_cy) * c_sz))
        )

        for cy in range(min_cy, max_cy + 1):
            rel_cy: int = cy - min_cy
            chk_y: int = anchor_y + (rel_cy * self.chunk_size)

            for cx in range(min_cx, max_cx + 1):
                chunk = self.chunks.get((cx, cy))
                if chunk is None or chunk.surface is None:
                    continue

                rel_cx: int = cx - min_cx
                chk_x: int = anchor_x + (rel_cx * self.chunk_size)
                surface.blit(chunk.surface, (chk_x, chk_y))

    def are_visible_chunks_loaded(
        self,
        focus_x: float,
        focus_y: float,
        view_w: float,
        view_h: float,
    ) -> bool:
        """Returns True when every chunk intersecting viewport is baked."""
        min_cx, max_cx, min_cy, max_cy = self._compute_visible_bounds(
            focus_x, focus_y, view_w, view_h
        )
        for cy in range(min_cy, max_cy + 1):
            for cx in range(min_cx, max_cx + 1):
                chunk = self.chunks.get((cx, cy))
                if chunk is None or chunk.surface is None:
                    return False
        return True

    def get_elevation_at(self, world_x: float, world_y: float) -> float:
        """Retrieves continuous elevation at global world coordinates."""
        cx: int = math.floor(world_x / float(self.chunk_size))
        cy: int = math.floor(world_y / float(self.chunk_size))
        chunk = self.chunks.get((cx, cy))
        if chunk is None:
            return 0.0

        lx: int = int(world_x) % self.chunk_size
        ly: int = int(world_y) % self.chunk_size
        return chunk.get_elevation_at(lx, ly)

    def get_gradient_at(
        self,
        world_x: float,
        world_y: float,
    ) -> Tuple[float, float]:
        """Retrieves continuous slope gradients (gx, gy) at world point."""
        cx: int = math.floor(world_x / float(self.chunk_size))
        cy: int = math.floor(world_y / float(self.chunk_size))
        chunk = self.chunks.get((cx, cy))
        if chunk is None:
            return 0.0, 0.0

        lx: int = int(world_x) % self.chunk_size
        ly: int = int(world_y) % self.chunk_size
        return chunk.get_gradient_at(lx, ly)

    def is_point_inside_load_boundary(
        self,
        world_x: float,
        world_y: float,
    ) -> bool:
        """Checks if world point lies within active streaming ellipse."""
        if not self._has_boundary:
            return True

        c_sz: float = float(self.chunk_size)
        diff_x: float = (world_x / c_sz) - self._center_cx
        diff_y: float = (world_y / c_sz) - self._center_cy

        dist_sq: float = (
            (diff_x * diff_x * self._inv_a_load_sq)
            + (diff_y * diff_y * self._inv_b_load_sq)
        )
        return dist_sq <= 1.0

    def clear(self) -> None:
        """Purges all loaded chunks and releases surface memory from RAM."""
        self.chunks.clear()
        self._has_boundary = False

    def _compute_visible_bounds(
        self,
        focus_x: float,
        focus_y: float,
        view_w: float,
        view_h: float,
    ) -> Tuple[int, int, int, int]:
        """Computes inclusive chunk coordinate bounds covering viewport."""
        c_sz: float = float(self.chunk_size)
        focus_cx: float = focus_x / c_sz
        focus_cy: float = focus_y / c_sz
        half_w_chunks: float = view_w / (2.0 * c_sz)
        half_h_chunks: float = view_h / (2.0 * c_sz)

        min_cx: int = math.floor(focus_cx - half_w_chunks)
        max_cx: int = math.ceil(focus_cx + half_w_chunks)
        min_cy: int = math.floor(focus_cy - half_h_chunks)
        max_cy: int = math.ceil(focus_cy + half_h_chunks)
        return min_cx, max_cx, min_cy, max_cy
