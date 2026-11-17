"""Renderer for Mode 5 dynamic solar lighting and hillshading studio."""

from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
import pygame
import config
from studio.renderers.base_chunk_renderer import BaseChunkRenderer
from studio.state import StudioState
from utils.noise.noise_generator import NoiseProfile
from world.chunks.chunk_manager import ChunkManager
from world.chunks.providers.world_chunk_provider import WorldChunkProvider
from world.chunks.world_profile import WorldProfile
from world.lighting.atmosphere_overlay import AtmosphereOverlayManager

LIGHT_ROW_PRESET: int = 0
LIGHT_ROW_WORLD: int = 1
LIGHT_ROW_SCALE: int = 2
LIGHT_ROW_AZIMUTH: int = 3
LIGHT_ROW_ALTITUDE: int = 4
LIGHT_ROW_CONTRAST: int = 5
LIGHT_ROW_AMBIENT: int = 6
LIGHT_ROW_ORBIT_SPEED: int = 7
LIGHT_ROW_TIME: int = 8
LIGHT_ROW_EXPOSURE: int = 9
LIGHT_MATRIX_ROW_COUNT: int = 10

LIGHT_ORBIT_SCALES: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)

_SOLAR_ROWS: Tuple[Tuple[int, str], ...] = (
    (LIGHT_ROW_PRESET, "PRESET    :"),
    (LIGHT_ROW_WORLD, "WORLD     :"),
    (LIGHT_ROW_SCALE, "SCALE     :"),
    (LIGHT_ROW_AZIMUTH, "AZIMUTH   :"),
    (LIGHT_ROW_ALTITUDE, "ALTITUDE  :"),
    (LIGHT_ROW_CONTRAST, "CONTRAST  :"),
    (LIGHT_ROW_AMBIENT, "AMBIENT   :"),
    (LIGHT_ROW_ORBIT_SPEED, "ORBIT SPD :"),
    (LIGHT_ROW_TIME, "TIME/DAY  :"),
    (LIGHT_ROW_EXPOSURE, "EXPOSURE  :"),
)


class RendererLighting(BaseChunkRenderer):
    """Renders composite terrain under 3D solar hillshading and ambient."""

    def __init__(self, screen: pygame.Surface, state: StudioState) -> None:
        """Initializes gradient-enabled provider, chunks, and atmosphere."""
        pal_name, active_lut = state.active_palette
        self.provider: WorldChunkProvider = self._build_provider(
            state, active_lut
        )
        super().__init__(screen, ChunkManager(self.provider))
        self.atmosphere: AtmosphereOverlayManager = AtmosphereOverlayManager(
            state.active_lighting_profile
        )

        self._last_lighting_key: str = (
            state.active_lighting_profile.profile_name
        )
        self._last_palette_name: str = pal_name
        self._last_net_enabled: bool = state.net_enabled
        self._last_tile_size: int = state.tile_size
        self._last_tile_stepping: bool = state.tile_stepping_enabled
        self._last_struct_sig: Tuple = self._compute_struct_signature(state)
        self._last_scale_sig: Tuple = self._compute_scale_signature(state)

    def update(self, state: StudioState, dt: float) -> None:
        """Advances solar orbit clock and synchronizes lighting preset."""
        lp = state.active_lighting_profile
        if lp.profile_name != self._last_lighting_key:
            self.atmosphere.set_profile(lp)
            self._last_lighting_key = lp.profile_name

        if state.light_time_dirty:
            self.atmosphere.clock.set_normalized_time(state.light_time_ratio)
            state.light_time_dirty = False

        self.atmosphere.clock.time_scale = state.light_orbit_scale
        self.atmosphere.clock.is_paused = state.light_orbit_scale <= 0.0
        self.atmosphere.update(dt)

        if not self.atmosphere.clock.is_paused and lp.day_cycle_duration > 0.0:
            state.light_time_ratio = self.atmosphere.clock.normalized_time
            deg_step = (
                360.0 * dt * state.light_orbit_scale / lp.day_cycle_duration
            )
            state.light_azimuth_deg = (
                state.light_azimuth_deg - deg_step
            ) % 360.0

    def render(self, state: StudioState, fps: float) -> None:
        """Syncs world chunks, applies unified lightmap, and draws HUD."""
        self._sync_provider_state(state)
        self._render_chunks_pipeline(state, self.provider)
        self._draw_lighting_overlay(state)

        self._draw_hud(state, fps)
        self.draw_studio_footer(
            active_mode=5,
            extra_ctrl="[WASD]: Pan",
        )

    def _draw_lighting_overlay(self, state: StudioState) -> None:
        """Selects scrub-field or hybrid chunk-sampled lightmap path."""
        tile_sz = state.tile_size if state.tile_stepping_enabled else 0

        if state.is_scrubbing and self._last_scrub_field is not None:
            step_scale = (
                float(tile_sz)
                if tile_sz > 1
                else float(config.SCRUB_SUB_SAMPLE_FACTOR)
            )
            self.atmosphere.draw_overlay_from_field(
                target_surface=self.screen,
                field=self._last_scrub_field,
                step_scale=step_scale,
                azimuth_deg=state.light_azimuth_deg,
                altitude_deg=state.light_altitude_deg,
                steepness=state.light_contrast,
                ambient_ratio=state.light_ambient_ratio,
                neutral_exposure=state.light_exposure,
                tile_size=tile_sz,
                focus_x=state.cam_x,
                focus_y=state.cam_y,
                field_origin=(
                    self._preview_origin_x,
                    self._preview_origin_y,
                ),
            )
            return

        fb_field = (
            self._last_scrub_field
            if (self._is_refining and self._last_scrub_field is not None)
            else None
        )
        self.atmosphere.draw_overlay(
            target_surface=self.screen,
            focus_x=state.cam_x,
            focus_y=state.cam_y,
            chunk_manager=self.chunk_manager,
            azimuth_deg=state.light_azimuth_deg,
            altitude_deg=state.light_altitude_deg,
            steepness=state.light_contrast,
            ambient_ratio=state.light_ambient_ratio,
            neutral_exposure=state.light_exposure,
            tile_size=tile_sz,
            fallback_field=fb_field,
            fallback_origin=(
                self._preview_origin_x,
                self._preview_origin_y,
            ),
            fallback_step=self._preview_step_scale,
        )

    def _resolve_world_source(
        self, state: StudioState
    ) -> Tuple[
        WorldProfile,
        int,
        Dict[str, NoiseProfile],
        str,
        str,
        str,
        Tuple[float, float, float, float, float, float],
        float,
    ]:
        """Resolves active Mode 4 live state or baseline world.yaml build."""
        if state.light_world_source_index == 0:
            return (
                state.active_world_profile,
                state.world_seed,
                state.working_noise_profiles,
                state.world_base_profile_key,
                state.world_macro_profile_key,
                state.world_crust_profile_key,
                (
                    state.world_base_offset_x,
                    state.world_base_offset_y,
                    state.world_macro_offset_x,
                    state.world_macro_offset_y,
                    state.world_crust_offset_x,
                    state.world_crust_offset_y,
                ),
                1.0,
            )

        idx = (state.light_world_source_index - 1) % len(state.world_keys)
        wp = state.world_profiles[state.world_keys[idx]]
        zoom_ratio, ox, oy = state.get_light_preset_transform(wp.profile_name)
        return (
            wp,
            wp.world_seed,
            state.default_noise_profiles,
            wp.base_noise_profile,
            wp.macro_noise_profile,
            wp.crust_noise_profile,
            (ox, oy, ox, oy, ox, oy),
            zoom_ratio,
        )

    def _compute_struct_signature(self, state: StudioState) -> Tuple:
        """Captures structural world identity excluding scale and offsets."""
        wp, seed, profiles, b_k, m_k, c_k, _, _ = self._resolve_world_source(
            state
        )
        layer_snaps = []
        for key in (b_k, m_k, c_k):
            prof = profiles.get(key)
            if prof is not None:
                layer_snaps.append((
                    key,
                    prof.noise_type,
                    prof.seed,
                    prof.octaves,
                    round(prof.decay, 4),
                    round(prof.lacunarity, 4),
                    prof.sub_sample_factor,
                ))
        return (
            state.light_world_source_index,
            wp.profile_name,
            state.formula.name,
            seed,
            tuple(layer_snaps),
        )

    def _compute_scale_signature(self, state: StudioState) -> Tuple:
        """Captures current layer scales and offsets for in-place sync."""
        _, _, profiles, b_k, m_k, c_k, offsets, zoom_r = (
            self._resolve_world_source(state)
        )
        scales = []
        for key in (b_k, m_k, c_k):
            prof = profiles.get(key)
            scales.append(round(prof.scale * zoom_r, 4) if prof else 0.0)
        return (tuple(scales), tuple(round(v, 4) for v in offsets))

    def _build_provider(
        self,
        state: StudioState,
        active_lut: NDArray[np.uint8],
    ) -> WorldChunkProvider:
        """Constructs gradient-enabled WorldChunkProvider for Mode 5."""
        wp, seed, profiles, b_k, m_k, c_k, offsets, zoom_r = (
            self._resolve_world_source(state)
        )
        provider = WorldChunkProvider(
            world_profile=wp,
            chunk_profile=state.active_chunk_profile,
            noise_profiles=profiles,
            strata_lut=active_lut,
            formula_override=state.formula,
            world_seed_override=seed,
            base_profile_key=b_k,
            macro_profile_key=m_k,
            crust_profile_key=c_k,
            net_enabled=state.net_enabled,
            net_intervals=state.net_intervals,
            net_bandwidth=state.net_bandwidth,
            tile_stepping_enabled=state.tile_stepping_enabled,
            tile_size=state.tile_size,
            compute_gradients=True,
        )
        b_p = profiles.get(b_k)
        m_p = profiles.get(m_k)
        c_p = profiles.get(c_k)
        provider.set_layer_scales(
            (b_p.scale * zoom_r) if b_p else 100.0,
            (m_p.scale * zoom_r) if m_p else 100.0,
            (c_p.scale * zoom_r) if c_p else 100.0,
        )
        provider.set_layer_offsets(*offsets)
        return provider

    def _sync_provider_state(self, state: StudioState) -> None:
        """Synchronizes world source, scales, offsets, visuals, and LUTs."""
        pal_name, active_lut = state.active_palette
        tile_changed = (
            state.tile_size != self._last_tile_size
            or state.tile_stepping_enabled != self._last_tile_stepping
        )
        net_changed = state.net_enabled != self._last_net_enabled
        palette_changed = pal_name != self._last_palette_name

        struct_sig = self._compute_struct_signature(state)
        if struct_sig != self._last_struct_sig:
            self.provider = self._build_provider(state, active_lut)
            self.chunk_manager.set_provider(self.provider)
            self._invalidate_preview()
            self._last_struct_sig = struct_sig
            self._last_scale_sig = self._compute_scale_signature(state)
            self._last_palette_name = pal_name
            self._last_net_enabled = state.net_enabled
            self._last_tile_size = state.tile_size
            self._last_tile_stepping = state.tile_stepping_enabled
            return

        scale_sig = self._compute_scale_signature(state)
        if scale_sig != self._last_scale_sig:
            _, _, profiles, b_k, m_k, c_k, offsets, zoom_r = (
                self._resolve_world_source(state)
            )
            b_p = profiles.get(b_k)
            m_p = profiles.get(m_k)
            c_p = profiles.get(c_k)
            self.provider.set_layer_scales(
                (b_p.scale * zoom_r) if b_p else 100.0,
                (m_p.scale * zoom_r) if m_p else 100.0,
                (c_p.scale * zoom_r) if c_p else 100.0,
            )
            self.provider.set_layer_offsets(*offsets)
            self._last_scale_sig = scale_sig
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

    def _draw_hud(self, state: StudioState, fps: float) -> None:
        """Renders lighting metadata card and Solar Control Deck."""
        lp = state.active_lighting_profile
        pal_name, _ = state.active_palette

        step_tag = (
            f"ACTIVE ({state.tile_size}px)"
            if state.tile_stepping_enabled
            else "OFF (Smooth)"
        )
        net_tag = "ACTIVE" if state.net_enabled else "OFF"

        hud_lines: List[Tuple[str, Tuple[int, int, int]]] = [
            (
                f"MODE 5: SOLAR LIGHTING STUDIO | TILE: {step_tag} | "
                f"NET: {net_tag}",
                config.COLOR_HUD_TEXT_ACTIVE,
            ),
            (
                f"LIGHT : {lp.profile_name} | "
                f"WORLD: {state.active_light_world_label}",
                config.COLOR_HUD_TEXT_NORMAL,
            ),
            (
                f"PALETTE: {pal_name} | "
                f"POS: ({state.cam_x:.1f}, {state.cam_y:.1f})",
                config.COLOR_HUD_TEXT_NORMAL,
            ),
            (
                f"CHUNKS : {len(self.chunk_manager.chunks)} | "
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
        self._draw_solar_control_deck(state, x=10, y=10 + card1_h + 6)

    def _draw_solar_control_deck(
        self,
        state: StudioState,
        x: int,
        y: int,
    ) -> None:
        """Renders interactive solar parameter table and records hitboxes."""
        pad = 8
        line_h = self.font.get_linesize() + 2
        lbl_w = 110
        val_w = 195
        box_w = pad + lbl_w + val_w + pad
        box_h = pad + ((len(_SOLAR_ROWS) + 1) * line_h) + pad

        bg_surf = self.get_translucent_card_surface(
            box_w, box_h, config.COLOR_HUD_CARD_2D
        )
        self.screen.blit(bg_surf, (x, y))
        pygame.draw.rect(
            self.screen,
            config.COLOR_SELECTION,
            (x, y, box_w, box_h),
            2,
        )

        hdr_surf = self.font.render(
            "SOLAR CONTROL DECK", True, config.COLOR_HUD_TEXT_ACTIVE
        )
        self.screen.blit(hdr_surf, (x + pad, y + pad))

        state.light_hitboxes.clear()
        for idx, (row_id, row_lbl) in enumerate(_SOLAR_ROWS):
            ry = y + pad + ((idx + 1) * line_h)
            lbl_surf = self.font.render(
                row_lbl, True, config.COLOR_HUD_TEXT_NORMAL
            )
            self.screen.blit(lbl_surf, (x + pad, ry))

            cx = x + pad + lbl_w
            cell_rect = pygame.Rect(cx, ry - 1, val_w - 4, line_h - 1)
            state.light_hitboxes[row_id] = cell_rect

            is_focused = state.selected_light_row == row_id
            if is_focused:
                pygame.draw.rect(
                    self.screen,
                    config.COLOR_SELECTION,
                    cell_rect,
                    2,
                )

            val_str = self._format_solar_value(state, row_id)
            val_col = (
                config.COLOR_INPUT_ACTIVE
                if is_focused
                else config.COLOR_HUD_TEXT_NORMAL
            )
            val_surf = self.font.render(val_str, True, val_col)
            self.screen.blit(val_surf, (cx + 4, ry))

    def _format_solar_value(self, state: StudioState, row_id: int) -> str:
        """Formats display string for each Solar Control Deck row."""
        if row_id == LIGHT_ROW_PRESET:
            return f"[ {state.active_lighting_profile.profile_name} ]"
        if row_id == LIGHT_ROW_WORLD:
            return f"[ {state.active_light_world_label} ]"
        if row_id == LIGHT_ROW_AZIMUTH:
            return f"[ {state.light_azimuth_deg:.1f} deg ]"
        if row_id == LIGHT_ROW_ALTITUDE:
            return f"[ {state.light_altitude_deg:.1f} deg ]"
        if row_id == LIGHT_ROW_CONTRAST:
            return f"[ {state.light_contrast:.1f} ]"
        if row_id == LIGHT_ROW_AMBIENT:
            return f"[ {state.light_ambient_ratio:.2f} ]"
        if row_id == LIGHT_ROW_ORBIT_SPEED:
            if state.light_orbit_scale <= 0.0:
                return "[ PAUSED ]"
            return f"[ {state.light_orbit_scale:.1f}x ]"
        if row_id == LIGHT_ROW_TIME:
            return f"[ {state.light_time_ratio:.2f} ]"
        if row_id == LIGHT_ROW_EXPOSURE:
            return f"[ {state.light_exposure:.2f} ]"
        if row_id == LIGHT_ROW_SCALE:
            _, _, profiles, b_k, _, _, _, zoom_r = self._resolve_world_source(
                state
            )
            b_prof = profiles.get(b_k)
            eff_scale = (b_prof.scale * zoom_r) if b_prof else 0.0
            return f"[ {eff_scale:.1f} px ]"
        return "[ - ]"
