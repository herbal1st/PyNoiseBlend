"""Unified lightmap atmosphere overlay manager with single-pass blend."""

import math
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import pygame

from world.chunks.chunk_manager import ChunkManager
from world.lighting.ambient_palette import AmbientPaletteResolver
from world.lighting.height_shadow_engine import VectorizedHeightShadowEngine
from world.lighting.lighting_profile import LightingProfile
from world.lighting.time_clock import DayNightClock
from world.lighting.viewport_height_sampler import ViewportHeightSampler


class AtmosphereOverlayManager:
    """Manages solar progression and single-pass unified lightmap blits."""

    def __init__(self, profile: LightingProfile) -> None:
        """Initializes lighting profile, orbit clock, and surface caches."""
        self.profile: LightingProfile = profile
        self.clock: DayNightClock = DayNightClock(
            day_cycle_duration=profile.day_cycle_duration,
            start_time_ratio=profile.start_time_ratio,
        )
        self.sampler: ViewportHeightSampler = ViewportHeightSampler()
        self._small_surface: Optional[pygame.Surface] = None
        self._scaled_surface: Optional[pygame.Surface] = None
        self._tile_scaled_surface: Optional[pygame.Surface] = None

    def set_profile(self, profile: LightingProfile) -> None:
        """Switches active lighting preset and resets the orbit clock."""
        self.profile = profile
        self.clock = DayNightClock(
            day_cycle_duration=profile.day_cycle_duration,
            start_time_ratio=profile.start_time_ratio,
        )

    def update(self, dt: float) -> None:
        """Advances orbit clock by frame delta time in seconds."""
        self.clock.update(dt)

    def draw_overlay(
        self,
        target_surface: pygame.Surface,
        focus_x: float,
        focus_y: float,
        chunk_manager: ChunkManager,
        azimuth_deg: float,
        altitude_deg: float,
        steepness: float,
        ambient_ratio: float,
        neutral_exposure: float,
        sub_sample_override: Optional[int] = None,
        tile_size: int = 0,
    ) -> None:
        """Computes and blits the unified lightmap via BLEND_MULT."""
        surf_w, surf_h = target_surface.get_size()
        if tile_size > 1:
            snapped_ox, snapped_oy, shift_x, shift_y = (
                self._compute_tile_snap(focus_x, focus_y, surf_w, surf_h, tile_size)
            )
            gx, gy = self.sampler.sample_tile_gradients(
                snapped_ox=snapped_ox,
                snapped_oy=snapped_oy,
                view_w=surf_w,
                view_h=surf_h,
                chunk_manager=chunk_manager,
                tile_size=tile_size,
            )
        else:
            shift_x, shift_y = 0, 0
            base_sub = self.profile.lightmap_sub_sample
            eff_sub = (
                max(base_sub, sub_sample_override)
                if sub_sample_override is not None
                else base_sub
            )
            eff_sub = max(1, eff_sub)
            target_w = max(2, surf_w // eff_sub)
            target_h = max(2, surf_h // eff_sub)

            gx, gy = self.sampler.sample_viewport_gradients(
                focus_x=focus_x,
                focus_y=focus_y,
                view_w=surf_w,
                view_h=surf_h,
                chunk_manager=chunk_manager,
                target_w=target_w,
                target_h=target_h,
            )

        self._render_from_gradients(
            target_surface=target_surface,
            gx=gx,
            gy=gy,
            azimuth_deg=azimuth_deg,
            altitude_deg=altitude_deg,
            steepness=steepness,
            ambient_ratio=ambient_ratio,
            neutral_exposure=neutral_exposure,
            tile_size=tile_size,
            shift_x=shift_x,
            shift_y=shift_y,
        )

    def draw_overlay_from_field(
        self,
        target_surface: pygame.Surface,
        field: NDArray[np.float32],
        step_scale: float,
        azimuth_deg: float,
        altitude_deg: float,
        steepness: float,
        ambient_ratio: float,
        neutral_exposure: float,
        tile_size: int = 0,
        focus_x: float = 0.0,
        focus_y: float = 0.0,
    ) -> None:
        """Computes and blits lightmap directly from a scrub heightfield."""
        if field.shape[0] < 2 or field.shape[1] < 2:
            return

        gy, gx = np.gradient(field)
        inv_step = np.float32(1.0 / max(1.0, float(step_scale)))
        gx_scaled = (gx * inv_step).astype(np.float32)
        gy_scaled = (gy * inv_step).astype(np.float32)

        shift_x, shift_y = 0, 0
        if tile_size > 1:
            surf_w, surf_h = target_surface.get_size()
            _, _, shift_x, shift_y = self._compute_tile_snap(
                focus_x, focus_y, surf_w, surf_h, tile_size
            )

        self._render_from_gradients(
            target_surface=target_surface,
            gx=gx_scaled,
            gy=gy_scaled,
            azimuth_deg=azimuth_deg,
            altitude_deg=altitude_deg,
            steepness=steepness,
            ambient_ratio=ambient_ratio,
            neutral_exposure=neutral_exposure,
            tile_size=tile_size,
            shift_x=shift_x,
            shift_y=shift_y,
        )

    def draw_cached_overlay(self, target_surface: pygame.Surface) -> None:
        """Reuses last scaled lightmap surface during chunk refinement."""
        if self._scaled_surface is None:
            return
        if self._scaled_surface.get_size() != target_surface.get_size():
            return
        target_surface.blit(
            self._scaled_surface,
            (0, 0),
            special_flags=pygame.BLEND_MULT,
        )

    def _render_from_gradients(
        self,
        target_surface: pygame.Surface,
        gx: NDArray[np.float32],
        gy: NDArray[np.float32],
        azimuth_deg: float,
        altitude_deg: float,
        steepness: float,
        ambient_ratio: float,
        neutral_exposure: float,
        tile_size: int = 0,
        shift_x: int = 0,
        shift_y: int = 0,
    ) -> None:
        """Runs shadow engine and blits scaled lightmap onto target."""
        ambient_rgb: Tuple[int, int, int] = (
            AmbientPaletteResolver.resolve_ambient_color(
                self.clock.normalized_time, self.profile
            )
        )
        raw_lightmap = VectorizedHeightShadowEngine.compute_unified_lightmap(
            gx_matrix=gx,
            gy_matrix=gy,
            azimuth_deg=azimuth_deg,
            altitude_deg=altitude_deg,
            steepness=steepness,
            ambient_ratio=ambient_ratio,
            shadow_intensity=self.profile.shadow_intensity,
            highlight_intensity=self.profile.highlight_intensity,
            ambient_rgb=ambient_rgb,
            neutral_exposure=neutral_exposure,
        )
        self._blit_lightmap(
            target_surface,
            raw_lightmap,
            tile_size=tile_size,
            shift_x=shift_x,
            shift_y=shift_y,
        )

    def _blit_lightmap(
        self,
        target_surface: pygame.Surface,
        raw_lightmap: pygame.Surface,
        tile_size: int = 0,
        shift_x: int = 0,
        shift_y: int = 0,
    ) -> None:
        """Scales compact lightmap to screen size and multiplies."""
        lm_w, lm_h = raw_lightmap.get_size()
        if lm_w <= 1 or lm_h <= 1:
            return

        surf_w, surf_h = target_surface.get_size()
        if (
            self._small_surface is None
            or self._small_surface.get_size() != (lm_w, lm_h)
        ):
            self._small_surface = pygame.Surface((lm_w, lm_h)).convert()

        self._small_surface.blit(raw_lightmap, (0, 0))

        if (
            self._scaled_surface is None
            or self._scaled_surface.get_size() != (surf_w, surf_h)
        ):
            self._scaled_surface = pygame.Surface((surf_w, surf_h)).convert()

        if tile_size > 1:
            full_w = lm_w * tile_size
            full_h = lm_h * tile_size
            if (
                self._tile_scaled_surface is None
                or self._tile_scaled_surface.get_size() != (full_w, full_h)
            ):
                self._tile_scaled_surface = pygame.Surface(
                    (full_w, full_h)
                ).convert()
            pygame.transform.scale(
                self._small_surface, (full_w, full_h), self._tile_scaled_surface
            )
            self._scaled_surface.blit(
                self._tile_scaled_surface, (-shift_x, -shift_y)
            )
        else:
            pygame.transform.smoothscale(
                self._small_surface, (surf_w, surf_h), self._scaled_surface
            )

        target_surface.blit(
            self._scaled_surface,
            (0, 0),
            special_flags=pygame.BLEND_MULT,
        )

    @staticmethod
    def _compute_tile_snap(
        focus_x: float,
        focus_y: float,
        view_w: int,
        view_h: int,
        tile_size: int,
    ) -> Tuple[float, float, int, int]:
        """Snaps viewport origin to world tile grid and returns shift."""
        origin_x = focus_x - (float(view_w) / 2.0)
        origin_y = focus_y - (float(view_h) / 2.0)
        snapped_ox = float(math.floor(origin_x / tile_size) * tile_size)
        snapped_oy = float(math.floor(origin_y / tile_size) * tile_size)
        shift_x = int(round(origin_x - snapped_ox))
        shift_y = int(round(origin_y - snapped_oy))
        return snapped_ox, snapped_oy, shift_x, shift_y
