"""Renderer for composite procedural worlds using chunk providers."""

from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
import pygame
import config
from studio.renderers.base_chunk_renderer import BaseChunkRenderer
from studio.renderers.parameter_matrix import draw_parameter_matrix
from studio.state import StudioState
from world.chunks.chunk_manager import ChunkManager
from world.chunks.providers.world_chunk_provider import WorldChunkProvider


class RendererWorld(BaseChunkRenderer):
    """Coordinates infinite composite terrain streaming and HUD cards."""

    def __init__(self, screen: pygame.Surface, state: StudioState) -> None:
        """Initializes formula-driven world provider and chunk manager."""
        pal_name, active_lut = state.active_palette
        self.provider: WorldChunkProvider = WorldChunkProvider(
            world_profile=state.active_world_profile,
            chunk_profile=state.active_chunk_profile,
            noise_profiles=state.working_noise_profiles,
            strata_lut=active_lut,
            formula_override=state.formula,
            world_seed_override=state.world_seed,
            base_profile_key=state.world_base_profile_key,
            macro_profile_key=state.world_macro_profile_key,
            crust_profile_key=state.world_crust_profile_key,
            net_enabled=state.net_enabled,
            net_intervals=state.net_intervals,
            net_bandwidth=state.net_bandwidth,
            tile_stepping_enabled=state.tile_stepping_enabled,
            tile_size=state.tile_size,
        )
        self.provider.set_layer_offsets(
            state.world_base_offset_x,
            state.world_base_offset_y,
            state.world_macro_offset_x,
            state.world_macro_offset_y,
            state.world_crust_offset_x,
            state.world_crust_offset_y,
        )
        super().__init__(screen, ChunkManager(self.provider))

        self._last_world_profile_key: str = (
            state.active_world_profile.profile_name
        )
        self._last_formula_name: str = state.formula.name
        self._last_world_seed: int = state.world_seed
        self._last_base_key: str = state.world_base_profile_key
        self._last_macro_key: str = state.world_macro_profile_key
        self._last_crust_key: str = state.world_crust_profile_key

        self._last_layer_snapshots: Dict[int, Tuple] = {}
        self._record_layer_snapshots(state)

        self._last_base_ox: float = state.world_base_offset_x
        self._last_base_oy: float = state.world_base_offset_y
        self._last_macro_ox: float = state.world_macro_offset_x
        self._last_macro_oy: float = state.world_macro_offset_y
        self._last_crust_ox: float = state.world_crust_offset_x
        self._last_crust_oy: float = state.world_crust_offset_y

        self._last_palette_name: str = pal_name
        self._last_net_enabled: bool = state.net_enabled
        self._last_tile_size: int = state.tile_size
        self._last_tile_stepping: bool = state.tile_stepping_enabled

    def render(self, state: StudioState, fps: float) -> None:
        """Syncs recipes, runs preview/refinement pipeline, and draws."""
        self._sync_provider_state(state)
        self._render_chunks_pipeline(state, self.provider)

        self._draw_hud(state, fps)
        self.draw_studio_footer(
            active_mode=4,
            extra_ctrl="[WASD]: Pan | [Click/Enter]: Edit Seed",
        )

    def _sync_provider_state(self, state: StudioState) -> None:
        """Synchronizes world recipe, formula, scales, offsets, and LUT."""
        wp = state.active_world_profile
        formula = state.formula
        pal_name, active_lut = state.active_palette

        world_profile_changed = (
            wp.profile_name != self._last_world_profile_key
            or formula.name != self._last_formula_name
        )
        world_seed_changed = state.world_seed != self._last_world_seed
        bindings_changed = (
            state.world_base_profile_key != self._last_base_key
            or state.world_macro_profile_key != self._last_macro_key
            or state.world_crust_profile_key != self._last_crust_key
        )

        tile_changed = (
            state.tile_size != self._last_tile_size
            or state.tile_stepping_enabled != self._last_tile_stepping
        )
        net_changed = state.net_enabled != self._last_net_enabled
        palette_changed = pal_name != self._last_palette_name

        offsets_changed = (
            abs(state.world_base_offset_x - self._last_base_ox) > 1e-4
            or abs(state.world_base_offset_y - self._last_base_oy) > 1e-4
            or abs(state.world_macro_offset_x - self._last_macro_ox) > 1e-4
            or abs(state.world_macro_offset_y - self._last_macro_oy) > 1e-4
            or abs(state.world_crust_offset_x - self._last_crust_ox) > 1e-4
            or abs(state.world_crust_offset_y - self._last_crust_oy) > 1e-4
        )

        layer_params_changed, scale_only_changed = self._check_layer_changes(
            state
        )

        if world_profile_changed:
            self._rebuild_provider(state, active_lut)
            return

        if world_seed_changed:
            self.provider.sync_world_seed(
                state.world_seed, state.working_noise_profiles
            )
            self._last_world_seed = state.world_seed
            self._invalidate_preview()
            self.chunk_manager.clear()

        if bindings_changed or layer_params_changed:
            for col in (0, 1, 2):
                key = state.get_layer_profile_key(col)
                prof = state.get_layer_profile(col)
                if key is not None and prof is not None:
                    self.provider.sync_layer_profile(col, key, prof)

            self._record_layer_snapshots(state)
            self._last_base_key = state.world_base_profile_key
            self._last_macro_key = state.world_macro_profile_key
            self._last_crust_key = state.world_crust_profile_key

            if bindings_changed or not scale_only_changed:
                self._invalidate_preview()
                self.chunk_manager.clear()
            elif not state.is_scrubbing:
                self._mark_chunks_stale()

        if offsets_changed:
            self.provider.set_layer_offsets(
                state.world_base_offset_x,
                state.world_base_offset_y,
                state.world_macro_offset_x,
                state.world_macro_offset_y,
                state.world_crust_offset_x,
                state.world_crust_offset_y,
            )
            self._last_base_ox = state.world_base_offset_x
            self._last_base_oy = state.world_base_offset_y
            self._last_macro_ox = state.world_macro_offset_x
            self._last_macro_oy = state.world_macro_offset_y
            self._last_crust_ox = state.world_crust_offset_x
            self._last_crust_oy = state.world_crust_offset_y
            if not state.is_scrubbing:
                self._mark_chunks_stale()

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

    def _rebuild_provider(
        self,
        state: StudioState,
        active_lut: NDArray[np.uint8],
    ) -> None:
        """Reconstructs WorldChunkProvider when profile or formula changes."""
        self.provider = WorldChunkProvider(
            world_profile=state.active_world_profile,
            chunk_profile=state.active_chunk_profile,
            noise_profiles=state.working_noise_profiles,
            strata_lut=active_lut,
            formula_override=state.formula,
            world_seed_override=state.world_seed,
            base_profile_key=state.world_base_profile_key,
            macro_profile_key=state.world_macro_profile_key,
            crust_profile_key=state.world_crust_profile_key,
            net_enabled=state.net_enabled,
            net_intervals=state.net_intervals,
            net_bandwidth=state.net_bandwidth,
            tile_stepping_enabled=state.tile_stepping_enabled,
            tile_size=state.tile_size,
        )
        self.provider.set_layer_offsets(
            state.world_base_offset_x,
            state.world_base_offset_y,
            state.world_macro_offset_x,
            state.world_macro_offset_y,
            state.world_crust_offset_x,
            state.world_crust_offset_y,
        )
        self.chunk_manager.set_provider(self.provider)
        self._invalidate_preview()
        self._last_world_profile_key = state.active_world_profile.profile_name
        self._last_formula_name = state.formula.name
        self._last_world_seed = state.world_seed
        self._last_base_key = state.world_base_profile_key
        self._last_macro_key = state.world_macro_profile_key
        self._last_crust_key = state.world_crust_profile_key
        self._record_layer_snapshots(state)

    def _check_layer_changes(self, state: StudioState) -> Tuple[bool, bool]:
        """Detects parameter changes across active layer profiles."""
        any_changed = False
        scale_only = True

        for col in (0, 1, 2):
            prof = state.get_layer_profile(col)
            if prof is None:
                continue

            snap = self._last_layer_snapshots.get(col)
            if snap is None:
                any_changed = True
                scale_only = False
                continue

            last_scale, last_oct, last_dec, last_lac, last_seed, last_type = snap
            s_diff = abs(prof.scale - last_scale) > 1e-4
            p_diff = (
                prof.octaves != last_oct
                or abs(prof.decay - last_dec) > 1e-4
                or abs(prof.lacunarity - last_lac) > 1e-4
                or prof.seed != last_seed
                or prof.noise_type != last_type
            )

            if s_diff or p_diff:
                any_changed = True
                if p_diff:
                    scale_only = False

        return any_changed, scale_only

    def _record_layer_snapshots(self, state: StudioState) -> None:
        """Stores snapshots of parameters for Base, Relief, and Crust."""
        for col in (0, 1, 2):
            prof = state.get_layer_profile(col)
            if prof is not None:
                self._last_layer_snapshots[col] = (
                    prof.scale,
                    prof.octaves,
                    prof.decay,
                    prof.lacunarity,
                    prof.seed,
                    prof.noise_type,
                )

    def _draw_hud(self, state: StudioState, fps: float) -> None:
        """Renders world metadata and interactive parameter table."""
        wp = state.active_world_profile
        pal_name, _ = state.active_palette

        step_tag = (
            f"ACTIVE ({state.tile_size}px)"
            if state.tile_stepping_enabled
            else "OFF (Smooth)"
        )
        net_tag = "ACTIVE" if state.net_enabled else "OFF"

        hud_lines: List[Tuple[str, Tuple[int, int, int]]] = [
            (
                f"MODE 4: PROCEDURAL WORLD | TILE STEP: {step_tag} | "
                f"NET: {net_tag}",
                config.COLOR_HUD_TEXT_ACTIVE,
            ),
            (
                f"RECIPE: {wp.profile_name} [SEED: {state.world_seed}] | "
                f"PALETTE: {pal_name}",
                config.COLOR_HUD_TEXT_NORMAL,
            ),
            (
                f"FORMULA: {state.formula.name}",
                config.COLOR_HUD_TEXT_NORMAL,
            ),
            (
                f"CHUNKS : {len(self.chunk_manager.chunks)} | "
                f"POS: ({state.cam_x:.1f}, {state.cam_y:.1f}) | "
                f"FPS: {fps:.1f}",
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
            is_mode_world=True,
        )
