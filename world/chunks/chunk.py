"""
Continuous spatial chunk data model holding elevation and baked surface.
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import pygame


class Chunk:
    """
    Container for a continuous raster chunk and its pre-baked surface.
    """

    def __init__(
        self,
        cx: int,
        cy: int,
        size: int,
        elevation: NDArray[np.float32],
        gx: Optional[NDArray[np.float32]] = None,
        gy: Optional[NDArray[np.float32]] = None,
        surface: Optional[pygame.Surface] = None,
    ) -> None:
        """
        Initializes spatial coordinates, elevation grid, and surface.
        """
        self.cx: int = cx
        self.cy: int = cy
        self.size: int = size
        self.world_x: int = cx * size
        self.world_y: int = cy * size
        self.elevation: NDArray[np.float32] = elevation

        if gx is not None:
            self.gx: NDArray[np.float32] = gx
        else:
            self.gx = np.zeros((size, size), dtype=np.float32)

        if gy is not None:
            self.gy: NDArray[np.float32] = gy
        else:
            self.gy = np.zeros((size, size), dtype=np.float32)

        self.surface: Optional[pygame.Surface] = surface
        self.is_dirty: bool = surface is None

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """
        Returns chunk bounding box as (world_x, world_y, width, height).
        """
        return self.world_x, self.world_y, self.size, self.size

    @property
    def center(self) -> Tuple[float, float]:
        """
        Returns continuous world coordinate center of the chunk.
        """
        half_size: float = float(self.size) / 2.0
        return self.world_x + half_size, self.world_y + half_size

    def get_elevation_at(self, local_x: int, local_y: int) -> float:
        """
        Retrieves local scalar elevation clamped to chunk boundary.
        """
        clamped_x: int = max(0, min(self.size - 1, local_x))
        clamped_y: int = max(0, min(self.size - 1, local_y))
        return float(self.elevation[clamped_y, clamped_x])

    def get_gradient_at(
        self,
        local_x: int,
        local_y: int,
    ) -> Tuple[float, float]:
        """
        Retrieves local slope gradient (gx, gy) clamped to boundary.
        """
        clamped_x: int = max(0, min(self.size - 1, local_x))
        clamped_y: int = max(0, min(self.size - 1, local_y))
        return (
            float(self.gx[clamped_y, clamped_x]),
            float(self.gy[clamped_y, clamped_x]),
        )

    def set_surface(self, surface: pygame.Surface) -> None:
        """
        Binds pre-rendered Pygame surface and clears dirty flag.
        """
        self.surface = surface
        self.is_dirty = False

    def clear_surface(self) -> None:
        """
        Releases Pygame surface memory and marks chunk as dirty.
        """
        self.surface = None
        self.is_dirty = True
