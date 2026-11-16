"""Computes a compact 3D solar hillshade and ambient RGB lightmap."""

import math
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import pygame


class VectorizedHeightShadowEngine:
    """Generates a unified 2D RGB lightmap in persistent NumPy buffers."""

    _cached_dim: Tuple[int, int] = (0, 0)
    _lightmap_rgb: Optional[NDArray[np.uint8]] = None
    _illum_buffer: Optional[NDArray[np.float32]] = None
    _empty_surf: Optional[pygame.Surface] = None

    @classmethod
    def compute_unified_lightmap(
        cls,
        gx_matrix: NDArray[np.float32],
        gy_matrix: NDArray[np.float32],
        azimuth_deg: float,
        altitude_deg: float,
        steepness: float,
        ambient_ratio: float,
        shadow_intensity: int,
        highlight_intensity: int,
        ambient_rgb: Tuple[int, int, int],
        neutral_exposure: float = 0.85,
    ) -> pygame.Surface:
        """Evaluates 3D solar dot-product shading and ambient tinting."""
        grid_h, grid_w = gx_matrix.shape
        if grid_h < 2 or grid_w < 2:
            if cls._empty_surf is None:
                cls._empty_surf = pygame.Surface((1, 1))
                cls._empty_surf.fill((255, 255, 255))
            return cls._empty_surf

        cls._ensure_buffers(grid_w, grid_h)
        assert cls._lightmap_rgb is not None
        assert cls._illum_buffer is not None

        az_rad = math.radians(azimuth_deg)
        alt_rad = math.radians(max(0.0, min(90.0, altitude_deg)))
        cos_alt = math.cos(alt_rad)

        lx = math.cos(az_rad) * cos_alt
        ly = math.sin(az_rad) * cos_alt

        direct_weight = max(0.0, min(1.0, 1.0 - ambient_ratio))
        max_sh = (float(shadow_intensity) / 255.0) * direct_weight
        max_hl = (float(highlight_intensity) / 255.0) * direct_weight

        slopes = ((gx_matrix * lx) + (gy_matrix * ly)) * steepness
        hl_ratio = np.clip(slopes, 0.0, 1.0) * max_hl
        sh_ratio = np.clip(-slopes, 0.0, 1.0) * max_sh

        base_exp = max(0.50, min(1.00, float(neutral_exposure)))
        cls._illum_buffer.fill(base_exp)
        cls._illum_buffer += hl_ratio
        cls._illum_buffer -= sh_ratio
        np.clip(cls._illum_buffer, 0.0, 1.0, out=cls._illum_buffer)

        amb_r, amb_g, amb_b = ambient_rgb
        if amb_r == 255 and amb_g == 255 and amb_b == 255:
            scaled = np.clip(
                cls._illum_buffer * 255.0, 0.0, 255.0
            ).astype(np.uint8)
            cls._lightmap_rgb[:, :, 0] = scaled
            cls._lightmap_rgb[:, :, 1] = scaled
            cls._lightmap_rgb[:, :, 2] = scaled
        else:
            cls._lightmap_rgb[:, :, 0] = np.clip(
                cls._illum_buffer * float(amb_r), 0.0, 255.0
            ).astype(np.uint8)
            cls._lightmap_rgb[:, :, 1] = np.clip(
                cls._illum_buffer * float(amb_g), 0.0, 255.0
            ).astype(np.uint8)
            cls._lightmap_rgb[:, :, 2] = np.clip(
                cls._illum_buffer * float(amb_b), 0.0, 255.0
            ).astype(np.uint8)

        return pygame.image.frombuffer(
            cls._lightmap_rgb, (grid_w, grid_h), "RGB"
        )

    @classmethod
    def _ensure_buffers(cls, grid_w: int, grid_h: int) -> None:
        """Allocates persistent RGB and float32 buffers on size change."""
        if (grid_w, grid_h) != cls._cached_dim:
            cls._cached_dim = (grid_w, grid_h)
            cls._lightmap_rgb = np.full(
                (grid_h, grid_w, 3), 255, dtype=np.uint8
            )
            cls._illum_buffer = np.ones(
                (grid_h, grid_w), dtype=np.float32
            )

        if cls._empty_surf is None:
            cls._empty_surf = pygame.Surface((1, 1))
            cls._empty_surf.fill((255, 255, 255))
