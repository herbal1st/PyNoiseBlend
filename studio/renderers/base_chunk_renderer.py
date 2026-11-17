"""Shared chunk-streaming and coarse-preview base renderer."""

import math
from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import pygame
import config
from core.contour_engine import (
    draw_culled_contour_net_to_surface,
    draw_tile_contour_edges_to_surface,
)
from studio.renderers.base_renderer import BaseRenderer
from studio.state import StudioState
from world.chunks.chunk_manager import ChunkManager
from world.chunks.chunk_provider import ChunkProvider


class BaseChunkRenderer(BaseRenderer):
    """Coordinates scrubbing viewport previews and progressive chunk fills."""

    def __init__(
        self,
        screen: pygame.Surface,
        chunk_manager: ChunkManager,
    ) -> None:
        """Initializes chunk manager, preview surface, and scratch."""
        super().__init__(screen)
        self.chunk_manager: ChunkManager = chunk_manager
        self._preview_surface: pygame.Surface = pygame.Surface(
            (self.width, self.height)
        )
        self._scratch_surfaces: Dict[Tuple[int, int], pygame.Surface] = {}
        self._last_scrub_field: Optional[NDArray[np.float32]] = None
        self._has_preview: bool = False
        self._is_refining: bool = False
        self._chunks_stale_from_scrub: bool = False
        self._preview_cam_x: float = 0.0
        self._preview_cam_y: float = 0.0
        self._preview_origin_x: float = 0.0
        self._preview_origin_y: float = 0.0
        self._preview_step_scale: float = 1.0

    def _invalidate_preview(self) -> None:
        """Clears cached coarse preview and resets progressive state."""
        self._has_preview = False
        self._is_refining = False
        self._chunks_stale_from_scrub = False
        self._last_scrub_field = None
        self._preview_origin_x = 0.0
        self._preview_origin_y = 0.0
        self._preview_step_scale = 1.0

    def _mark_chunks_stale(self) -> None:
        """Flags cached chunks for progressive rebuild without scrub."""
        self._chunks_stale_from_scrub = True

    def _render_chunks_pipeline(
        self,
        state: StudioState,
        provider: ChunkProvider,
    ) -> None:
        """Executes live scrubbing preview, progressive refine, or pan."""
        if state.is_scrubbing:
            self._chunks_stale_from_scrub = True
            self._is_refining = True
            self._render_scrub_preview(state, provider)
            return

        if state.needs_refine_chunks or self._chunks_stale_from_scrub:
            self.chunk_manager.clear()
            state.needs_refine_chunks = False
            self._chunks_stale_from_scrub = False
            self._is_refining = True

        if self._is_refining:
            remaining = self.chunk_manager.update_focus(
                state.cam_x,
                state.cam_y,
                float(self.width),
                float(self.height),
                max_new_chunks=config.CHUNK_REFINE_BUDGET_PER_FRAME,
            )
            self._draw_refinement_frame(state)
            if remaining == 0:
                self._is_refining = False
                self._has_preview = False
                self._last_scrub_field = None
            return

        self.chunk_manager.update_focus(
            state.cam_x,
            state.cam_y,
            float(self.width),
            float(self.height),
        )
        self.screen.fill(config.COLOR_BG)
        self.chunk_manager.draw_chunks(self.screen, state.cam_x, state.cam_y)

    def _render_scrub_preview(
        self,
        state: StudioState,
        provider: ChunkProvider,
    ) -> None:
        """Renders single coarse viewport grid without chunk churn."""
        origin_x = state.cam_x - (float(self.width) / 2.0)
        origin_y = state.cam_y - (float(self.height) / 2.0)
        _, active_lut = state.active_palette

        if state.tile_stepping_enabled:
            self._bake_tiled_preview(
                state, provider, active_lut, origin_x, origin_y
            )
        else:
            self._bake_smooth_preview(
                state, provider, active_lut, origin_x, origin_y
            )

        self._preview_cam_x = state.cam_x
        self._preview_cam_y = state.cam_y
        self._has_preview = True
        self.screen.blit(self._preview_surface, (0, 0))

    def _bake_smooth_preview(
        self,
        state: StudioState,
        provider: ChunkProvider,
        lut: NDArray[np.uint8],
        origin_x: float,
        origin_y: float,
    ) -> None:
        """Samples (N+1) smooth coarse field and upscales onto preview."""
        sub_factor = config.SCRUB_SUB_SAMPLE_FACTOR
        field = provider.sample_viewport_field(
            origin_x=origin_x,
            origin_y=origin_y,
            width=self.width,
            height=self.height,
            sub_factor=sub_factor,
        )
        self._last_scrub_field = field
        self._preview_origin_x = origin_x
        self._preview_origin_y = origin_y
        self._preview_step_scale = float(sub_factor)

        if state.net_enabled:
            draw_culled_contour_net_to_surface(
                surface=self._preview_surface,
                coarse_wh=np.ascontiguousarray(field.T),
                lut=lut,
                intervals=state.net_intervals,
                bandwidth=state.net_bandwidth,
                bg_color=config.COLOR_NET_BACKGROUND,
            )
            return

        indices = np.clip(field * 1023.0, 0, 1023).astype(np.int32)
        rgb_buffer = lut[indices]
        h_dim, w_dim = field.shape
        scratch_surf = self._get_scratch_surface(w_dim, h_dim)
        transposed = np.transpose(rgb_buffer, (1, 0, 2))
        pygame.surfarray.blit_array(scratch_surf, transposed)
        pygame.transform.smoothscale(
            scratch_surf, (self.width, self.height), self._preview_surface
        )

    def _bake_tiled_preview(
        self,
        state: StudioState,
        provider: ChunkProvider,
        lut: NDArray[np.uint8],
        origin_x: float,
        origin_y: float,
    ) -> None:
        """Samples world-snapped tile centroids for crisp preview."""
        tile_sz = state.tile_size
        snapped_ox = float(math.floor(origin_x / tile_sz) * tile_sz)
        snapped_oy = float(math.floor(origin_y / tile_sz) * tile_sz)
        shift_x = int(round(origin_x - snapped_ox))
        shift_y = int(round(origin_y - snapped_oy))

        field = provider.sample_viewport_field(
            origin_x=snapped_ox,
            origin_y=snapped_oy,
            width=self.width + tile_sz,
            height=self.height + tile_sz,
            sub_factor=1,
        )
        self._last_scrub_field = field
        self._preview_origin_x = snapped_ox + (0.5 * float(tile_sz))
        self._preview_origin_y = snapped_oy + (0.5 * float(tile_sz))
        self._preview_step_scale = float(tile_sz)

        h_dim, w_dim = field.shape
        full_w = w_dim * tile_sz
        full_h = h_dim * tile_sz
        tile_surf = self._get_scratch_surface(full_w, full_h)

        if state.net_enabled:
            draw_tile_contour_edges_to_surface(
                surface=tile_surf,
                heightfield=field,
                lut=lut,
                intervals=state.net_intervals,
                tile_size=tile_sz,
                bg_color=config.COLOR_NET_BACKGROUND,
            )
        else:
            indices = np.clip(field * 1023.0, 0, 1023).astype(np.int32)
            rgb_buffer = lut[indices]
            transposed = np.transpose(rgb_buffer, (1, 0, 2))
            small_surf = self._get_scratch_surface(w_dim, h_dim)
            pygame.surfarray.blit_array(small_surf, transposed)
            pygame.transform.scale(small_surf, (full_w, full_h), tile_surf)

        self._preview_surface.blit(tile_surf, (-shift_x, -shift_y))

    def _draw_refinement_frame(self, state: StudioState) -> None:
        """Draws fallback preview underneath progressively filled chunks."""
        self.screen.fill(config.COLOR_BG)
        visible_ready = self.chunk_manager.are_visible_chunks_loaded(
            state.cam_x, state.cam_y, float(self.width), float(self.height)
        )
        if self._has_preview and not visible_ready:
            offset_x = int(round(self._preview_cam_x - state.cam_x))
            offset_y = int(round(self._preview_cam_y - state.cam_y))
            self.screen.blit(self._preview_surface, (offset_x, offset_y))

        self.chunk_manager.draw_chunks(self.screen, state.cam_x, state.cam_y)

    def _get_scratch_surface(
        self,
        width: int,
        height: int,
    ) -> pygame.Surface:
        """Returns cached offscreen surface matching dimensions."""
        key = (width, height)
        surface = self._scratch_surfaces.get(key)
        if surface is None:
            surface = pygame.Surface(key)
            self._scratch_surfaces[key] = surface
        return surface
