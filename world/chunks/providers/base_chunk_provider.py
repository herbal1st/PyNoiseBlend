"""Shared base provider for LUT translation, isolines, and surface baking."""

from typing import Optional
import numpy as np
from numpy.typing import NDArray
import pygame
import config
from core.contour_engine import bake_contour_net
from world.chunks.chunk import Chunk
from world.chunks.chunk_profile import ChunkProfile
from world.chunks.chunk_provider import ChunkProvider


class BaseChunkProvider(ChunkProvider):
    """Provides shared color indexing, contour baking, and SDL conversion."""

    def __init__(
        self,
        chunk_profile: ChunkProfile,
        color_lut: NDArray[np.uint8],
        net_enabled: bool = False,
        net_intervals: float = config.DEFAULT_NET_INTERVALS,
        net_bandwidth: float = config.DEFAULT_NET_BANDWIDTH,
        tile_stepping_enabled: bool = False,
        tile_size: int = config.DEFAULT_TILE_SIZE,
    ) -> None:
        """Initializes shared chunk dimensions, LUT, and visual settings."""
        self._chunk_profile: ChunkProfile = chunk_profile
        self.color_lut: NDArray[np.uint8] = color_lut
        self.net_enabled: bool = net_enabled
        self.net_intervals: float = net_intervals
        self.net_bandwidth: float = net_bandwidth
        self.tile_stepping_enabled: bool = tile_stepping_enabled
        self.tile_size: int = tile_size

    @property
    def chunk_profile(self) -> ChunkProfile:
        """Supplies chunk spatial dimensions and streaming buffer margins."""
        return self._chunk_profile

    def set_color_lut(self, lut: NDArray[np.uint8]) -> None:
        """Updates the active color look-up table for chunk generation."""
        self.color_lut = lut

    def update_visual_settings(
        self,
        net_enabled: bool,
        net_intervals: float,
        net_bandwidth: float,
        tile_stepping_enabled: bool,
        tile_size: int,
    ) -> None:
        """Updates runtime isoline and tile stepping configurations."""
        self.net_enabled = net_enabled
        self.net_intervals = net_intervals
        self.net_bandwidth = net_bandwidth
        self.tile_stepping_enabled = tile_stepping_enabled
        self.tile_size = tile_size

    def rebake_chunk_surface(
        self,
        chunk: Chunk,
        lut: NDArray[np.uint8],
    ) -> None:
        """Re-indexes cached elevation into new LUT without recomputing."""
        self.set_color_lut(lut)
        c_sz: int = chunk.size

        if self.net_enabled and self.tile_stepping_enabled:
            field_padded = np.pad(chunk.elevation, 1, mode="edge")
            rgb_buffer = self._bake_padded_tile_contour_rgb(
                field_padded, c_sz
            )
        else:
            rgb_buffer = self._render_field_to_rgb(chunk.elevation)

        surface = self._bake_surface_from_rgb(
            rgb_buffer, c_sz, target_surface=chunk.surface
        )
        chunk.set_surface(surface)

    def _bake_padded_tile_contour_rgb(
        self,
        field_padded: NDArray[np.float32],
        size: int,
    ) -> NDArray[np.uint8]:
        """Evaluates tiled contours on a 1px halo grid and crops core RGB."""
        full_buffer = bake_contour_net(
            heightfield=field_padded,
            lut=self.color_lut,
            intervals=self.net_intervals,
            bandwidth=self.net_bandwidth,
            tile_stepping=True,
            bg_color=config.COLOR_NET_BACKGROUND,
        )
        return np.ascontiguousarray(full_buffer[1 : size + 1, 1 : size + 1])

    def _render_field_to_rgb(
        self,
        field: NDArray[np.float32],
    ) -> NDArray[np.uint8]:
        """Translates continuous elevation grid to contiguous RGB buffer."""
        if self.net_enabled:
            buffer = bake_contour_net(
                heightfield=field,
                lut=self.color_lut,
                intervals=self.net_intervals,
                bandwidth=self.net_bandwidth,
                tile_stepping=self.tile_stepping_enabled,
                bg_color=config.COLOR_NET_BACKGROUND,
            )
            return np.ascontiguousarray(buffer)

        indices = np.clip(field * 1023.0, 0, 1023).astype(np.int32)
        return np.ascontiguousarray(self.color_lut[indices])

    @staticmethod
    def _bake_surface_from_rgb(
        rgb_buffer: NDArray[np.uint8],
        size: int,
        target_surface: Optional[pygame.Surface] = None,
    ) -> pygame.Surface:
        """Writes RGB buffer directly into display-matched SDL surface."""
        if (
            target_surface is not None
            and target_surface.get_size() == (size, size)
        ):
            surface = target_surface
        else:
            surface = pygame.Surface((size, size))
            if pygame.display.get_surface() is not None:
                surface = surface.convert()

        rgb_wh = np.transpose(rgb_buffer, (1, 0, 2))
        pygame.surfarray.blit_array(surface, rgb_wh)
        return surface
