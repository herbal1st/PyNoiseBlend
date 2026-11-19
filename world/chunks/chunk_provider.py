"""Abstract base specification defining the chunk provider contract."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import pygame

from world.chunks.chunk import Chunk
from world.chunks.chunk_profile import ChunkProfile


class ChunkProvider(ABC):
    """Abstract contract decoupling spatial chunking from generation."""

    @property
    @abstractmethod
    def chunk_profile(self) -> ChunkProfile:
        """Supplies chunk spatial dimensions and streaming buffer margins."""
        pass

    @abstractmethod
    def set_color_lut(self, lut: NDArray[np.uint8]) -> None:
        """Updates the active color look-up table for chunk generation."""
        pass

    @abstractmethod
    def generate_chunk(
        self,
        cx: int,
        cy: int,
        target_surface: Optional[pygame.Surface] = None,
    ) -> Chunk:
        """Generates and bakes a Chunk container for grid coordinates."""
        pass

    @abstractmethod
    def rebake_chunk_surface(
        self,
        chunk: Chunk,
        lut: NDArray[np.uint8],
    ) -> None:
        """Re-indexes cached elevation into new LUT without recomputing."""
        pass

    @abstractmethod
    def sample_viewport_field(
        self,
        origin_x: float,
        origin_y: float,
        width: int,
        height: int,
        sub_factor: int,
    ) -> NDArray[np.float32]:
        """Samples compact un-upscaled float32 field for viewport preview."""
        pass
