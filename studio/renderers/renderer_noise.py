"""Renderer for isolated procedural noise fields via ChunkManager."""

from typing import List, Tuple
import pygame
import config
from studio.renderers.base_chunk_renderer import BaseChunkRenderer
from studio.renderers.parameter_matrix import draw_parameter_matrix
from studio.state import StudioState
from utils.noise.noise_generator import NoiseGenerator, NoiseProfile
from world.chunks.chunk_manager import ChunkManager
from world.chunks.providers.noise_chunk_provider import NoiseChunkProvider


class RendererNoise(BaseChunkRenderer):
    """Coordinates raw noise chunk streaming, previews, and HUD cards."""

    def __init__(self, screen: pygame.Surface, state: StudioState) -> None:
        """Initializes provider and chunk manager from active state."""
        pal_name, active_lut = state.active_palette
        gen = NoiseGenerator(state.active_noise_profile)
        gen.set_scale(state.active_noise_profile.scale)
        gen.set_offset(state.noise_offset_x, state.noise_offset_y)

        self.provider: NoiseChunkProvider = NoiseChunkProvider(
            generator=gen,
            chunk_profile=state.active_chunk_profile,
            color_lut=active_lut,
            net_enabled=state.net_enabled,
            net_intervals=state.net_intervals,
            net_bandwidth=state.net_bandwidth,
            tile_stepping_enabled=state.tile_stepping_enabled,
            tile_size=state.tile_size,
        )
        super().__init__(screen, ChunkManager(self.provider))

        prof = state.active_noise_profile
        self._last_noise_key: str = prof.profile_name
        self._last_noise_scale: float = prof.scale
        self._last_octaves: int = prof.octaves
        self._last_decay: float = prof.decay
        self._last_lacunarity: float = prof.lacunarity
        self._last_seed: int = prof.seed
        self._last_noise_type: str = prof.noise_type
        self._last_offset_x: float = state.noise_offset_x
        self._last_offset_y: float = state.noise_offset_y
        self._last_palette_name: str = pal_name
        self._last_net_enabled: bool = state.net_enabled
        self._last_tile_size: int = state.tile_size
        self._last_tile_stepping: bool = state.tile_stepping_enabled

    def render(self, state: StudioState, fps: float) -> None:
        """Syncs state, runs preview/refinement pipeline, and draws HUD."""
        self._sync_provider_state(state)
        self._render_chunks_pipeline(state, self.provider)

        self._draw_hud(state, fps)
        self.draw_studio_footer(
            active_mode=3,
            extra_ctrl="[WASD]: Pan | [Click/Enter]: Edit Seed",
        )

    def _sync_provider_state(self, state: StudioState) -> None:
        """Pushes pending noise, scale, offset, visual, and LUT updates."""
        prof = state.active_noise_profile
        pal_name, active_lut = state.active_palette

        tile_changed = (
            state.tile_size != self._last_tile_size
            or state.tile_stepping_enabled != self._last_tile_stepping
        )
        net_changed = state.net_enabled != self._last_net_enabled
        profile_key_changed = prof.profile_name != self._last_noise_key
        palette_changed = pal_name != self._last_palette_name

        scale_changed = abs(prof.scale - self._last_noise_scale) > 1e-4
        offset_changed = (
            abs(state.noise_offset_x - self._last_offset_x) > 1e-4
            or abs(state.noise_offset_y - self._last_offset_y) > 1e-4
        )
        params_changed = (
            prof.octaves != self._last_octaves
            or abs(prof.decay - self._last_decay) > 1e-4
            or abs(prof.lacunarity - self._last_lacunarity) > 1e-4
            or prof.seed != self._last_seed
            or prof.noise_type != self._last_noise_type
        )

        if tile_changed or net_changed:
            self.provider.update_visual_settings(
                net_enabled=state.net_enabled,
                net_intervals=state.net_intervals,
                net_bandwidth=state.net_bandwidth,
                tile_stepping_enabled=state.tile_stepping_enabled,
                tile_size=state.tile_size,
            )
            self._last_net_enabled = state.net_enabled
            self._last_tile_size = state.tile_size
            self._last_tile_stepping = state.tile_stepping_enabled

        if profile_key_changed:
            gen = NoiseGenerator(prof)
            gen.set_scale(prof.scale)
            gen.set_offset(state.noise_offset_x, state.noise_offset_y)
            self.provider.set_generator(gen)
            self._record_profile_snapshot(prof, state)
            self._invalidate_preview()
            self.chunk_manager.clear()
        elif params_changed or scale_changed or offset_changed:
            self.provider.sync_profile_parameters(prof)
            self.provider.set_offset(state.noise_offset_x, state.noise_offset_y)
            self._record_profile_snapshot(prof, state)

            if params_changed:
                self._invalidate_preview()
                self.chunk_manager.clear()
            elif not state.is_scrubbing:
                self._mark_chunks_stale()

        if palette_changed:
            self.provider.set_color_lut(active_lut)
            self._last_palette_name = pal_name

        needs_tile_regen = state.tile_stepping_enabled and (
            net_changed or (palette_changed and state.net_enabled)
        )
        if tile_changed or needs_tile_regen:
            self._invalidate_preview()
            self.chunk_manager.clear()
        elif net_changed or palette_changed:
            self._has_preview = False
            self.chunk_manager.rebake_all_surfaces(active_lut)

    def _record_profile_snapshot(
        self, prof: NoiseProfile, state: StudioState
    ) -> None:
        """Caches snapshot of profile and offset variables."""
        self._last_noise_key = prof.profile_name
        self._last_noise_scale = prof.scale
        self._last_octaves = prof.octaves
        self._last_decay = prof.decay
        self._last_lacunarity = prof.lacunarity
        self._last_seed = prof.seed
        self._last_noise_type = prof.noise_type
        self._last_offset_x = state.noise_offset_x
        self._last_offset_y = state.noise_offset_y

    def _draw_hud(self, state: StudioState, fps: float) -> None:
        """Renders raw noise metadata and interactive parameter table."""
        prof = state.active_noise_profile
        pal_name, _ = state.active_palette

        step_tag = (
            f"ACTIVE ({state.tile_size}px)"
            if state.tile_stepping_enabled
            else "OFF (Smooth)"
        )
        net_tag = "ACTIVE" if state.net_enabled else "OFF"
        fallback_tag = (
            f" [FALLBACK: {config.RESOLUTION_FALLBACK_DIVIDER}x]"
            if state.fallback_latched
            else ""
        )

        hud_lines: List[Tuple[str, Tuple[int, int, int]]] = [
            (
                f"MODE 3: RAW NOISE INSPECTOR | TILE STEP: {step_tag} | "
                f"NET: {net_tag}",
                config.COLOR_HUD_TEXT_ACTIVE,
            ),
            (
                f"PROFILE: {prof.profile_name} | PALETTE: {pal_name}",
                config.COLOR_HUD_TEXT_NORMAL,
            ),
            (
                f"TYPE: {prof.noise_type} | SCALE: {prof.scale:.1f} | "
                f"OCTAVES: {prof.octaves}",
                config.COLOR_HUD_TEXT_NORMAL,
            ),
            (
                f"CHUNKS: {len(self.chunk_manager.chunks)} | "
                f"POS: ({state.cam_x:.1f}, {state.cam_y:.1f}) | "
                f"FPS: {fps:.1f}{fallback_tag}",
                config.COLOR_HUD_TEXT_NORMAL,
            ),
        ]

        card1_h = self.draw_card(
            x=10,
            y=10,
            lines=hud_lines,
            bg_rgba=config.COLOR_HUD_CARD_2D,
            pad=8,
        )

        draw_parameter_matrix(
            screen=self.screen,
            font=self.font,
            state=state,
            x=10,
            y=10 + card1_h + 6,
            is_mode_world=False,
        )
