"""Chunk provider synthesizing isolated procedural noise for tools."""

from typing import Optional
import numpy as np
from numpy.typing import NDArray
import pygame
import config
from utils.noise.noise_generator import NoiseGenerator, NoiseProfile
from world.chunks.chunk import Chunk
from world.chunks.chunk_profile import ChunkProfile
from world.chunks.providers.base_chunk_provider import BaseChunkProvider


class NoiseChunkProvider(BaseChunkProvider):
    """Synthesizes raw noise grids with tile stepping and contour nets."""

    def __init__(
        self,
        generator: NoiseGenerator,
        chunk_profile: ChunkProfile,
        color_lut: NDArray[np.uint8],
        net_enabled: bool = False,
        net_intervals: float = config.DEFAULT_NET_INTERVALS,
        net_bandwidth: float = config.DEFAULT_NET_BANDWIDTH,
        tile_stepping_enabled: bool = False,
        tile_size: int = config.DEFAULT_TILE_SIZE,
    ) -> None:
        """Binds noise engine and shared chunk/visual configuration."""
        super().__init__(
            chunk_profile=chunk_profile,
            color_lut=color_lut,
            net_enabled=net_enabled,
            net_intervals=net_intervals,
            net_bandwidth=net_bandwidth,
            tile_stepping_enabled=tile_stepping_enabled,
            tile_size=tile_size,
        )
        self._generator: NoiseGenerator = generator

    @property
    def generator(self) -> NoiseGenerator:
        """Returns active underlying NoiseGenerator instance."""
        return self._generator

    def set_generator(self, generator: NoiseGenerator) -> None:
        """Rebinds active noise generator when switching presets."""
        self._generator = generator

    def set_scale(self, scale: float) -> None:
        """Updates runtime noise wavelength scale in place."""
        self._generator.set_scale(scale)

    def set_offset(self, offset_x: float, offset_y: float) -> None:
        """Updates runtime noise spatial translation offset in place."""
        self._generator.set_offset(offset_x, offset_y)

    def sync_profile_parameters(self, prof: NoiseProfile) -> None:
        """Synchronizes all fractal parameters with the active profile."""
        self._generator.sync_from_profile(prof)

    def generate_chunk(
        self,
        cx: int,
        cy: int,
        target_surface: Optional[pygame.Surface] = None,
    ) -> Chunk:
        """Samples continuous noise and bakes hardware SDL surface."""
        c_sz: int = self._chunk_profile.chunk_size
        wx: float = float(cx * c_sz)
        wy: float = float(cy * c_sz)
        t_sz = self.tile_size if self.tile_stepping_enabled else 0

        if self.net_enabled and self.tile_stepping_enabled:
            pad_res = c_sz + 2
            field_padded = self._generator.sample_grid(
                origin_x=wx - 1.0,
                origin_y=wy - 1.0,
                width=pad_res,
                height=pad_res,
                tile_size=t_sz,
            )
            field = np.ascontiguousarray(
                field_padded[1 : c_sz + 1, 1 : c_sz + 1]
            )
            rgb_buffer = self._bake_padded_tile_contour_rgb(field_padded, c_sz)
        else:
            raw_field = self._generator.sample_grid(
                origin_x=wx,
                origin_y=wy,
                width=c_sz,
                height=c_sz,
                tile_size=t_sz,
            )
            field = np.ascontiguousarray(raw_field[:c_sz, :c_sz])
            rgb_buffer = self._render_field_to_rgb(field)

        surface = self._bake_surface_from_rgb(
            rgb_buffer, c_sz, target_surface=target_surface
        )
        return Chunk(
            cx=cx, cy=cy, size=c_sz, elevation=field, surface=surface
        )

    def sample_viewport_field(
        self,
        origin_x: float,
        origin_y: float,
        width: int,
        height: int,
        sub_factor: int,
    ) -> NDArray[np.float32]:
        """Samples compact un-upscaled float32 field for viewport preview."""
        t_sz = self.tile_size if self.tile_stepping_enabled else 0
        return self._generator.sample_grid(
            origin_x=origin_x,
            origin_y=origin_y,
            width=width,
            height=height,
            step=1.0,
            tile_size=t_sz,
            sub_factor=sub_factor,
            upscale=False,
        )
