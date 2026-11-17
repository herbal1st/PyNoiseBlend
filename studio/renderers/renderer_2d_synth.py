"""2D spatial synthesizer renderer generating surfaces and nets."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import pygame
import config
from core.contour_engine import (
    ContourScratch,
    create_contour_scratch,
    draw_contour_net_to_surface,
    draw_culled_contour_net_to_surface,
    draw_tile_contour_edges_to_surface,
)
from core.formulas import PRESETS_2D
from studio.renderers.base_renderer import BaseRenderer
from studio.state import StudioState


class Renderer2DSynth(BaseRenderer):
    """Bakes 2D float32 heightfields into surfaces or topographical nets."""

    def __init__(self, screen: pygame.Surface) -> None:
        """Initializes base renderer, target surface, and scratch cache."""
        super().__init__(screen)
        self.rendered_surface: pygame.Surface = pygame.Surface(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        )
        self._scratch_surfaces: Dict[Tuple[int, int], pygame.Surface] = {}
        self._contour_scratch: Optional[ContourScratch] = None

    def update_surface(
        self,
        heightfield: NDArray[np.float32],
        lut: NDArray[np.uint8],
        state: StudioState,
    ) -> None:
        """Colors compact or native heightfields and scales onto target."""
        if state.net_enabled:
            if state.tile_stepping_enabled:
                draw_tile_contour_edges_to_surface(
                    surface=self.rendered_surface,
                    heightfield=heightfield,
                    lut=lut,
                    intervals=state.net_intervals,
                    tile_size=state.tile_size,
                    bg_color=config.COLOR_NET_BACKGROUND,
                )
            else:
                self._render_net_surface(heightfield, lut, state)
            return

        h_dim, w_dim = heightfield.shape
        indices = np.clip(heightfield * 1023.0, 0, 1023).astype(np.int32)
        rgb_buffer = lut[indices]
        transposed = np.transpose(rgb_buffer, (1, 0, 2))

        if w_dim == self.width and h_dim == self.height:
            pygame.surfarray.blit_array(self.rendered_surface, transposed)
            return

        scratch = self._get_scratch_surface(w_dim, h_dim)
        pygame.surfarray.blit_array(scratch, transposed)

        if state.tile_stepping_enabled:
            target_w = w_dim * state.tile_size
            target_h = h_dim * state.tile_size
            scaled = pygame.transform.scale(scratch, (target_w, target_h))
            self.rendered_surface.blit(scaled, (0, 0))
            return

        pygame.transform.smoothscale(
            scratch,
            (self.width, self.height),
            self.rendered_surface,
        )

    def render(self, state: StudioState, fps: float) -> None:
        """Blits baked 2D surface, HUD metadata, and studio footer."""
        self.screen.blit(self.rendered_surface, (0, 0))
        self._draw_hud(state, fps)
        self.draw_studio_footer(active_mode=2)

    def _get_contour_scratch(self) -> ContourScratch:
        """Lazily allocates reusable (W, H) buffers for contours."""
        if self._contour_scratch is None:
            self._contour_scratch = create_contour_scratch(
                self.width, self.height
            )
        return self._contour_scratch

    def _get_scratch_surface(
        self,
        width: int,
        height: int,
    ) -> pygame.Surface:
        """Returns cached offscreen surface matching compact grid."""
        key = (width, height)
        surface = self._scratch_surfaces.get(key)
        if surface is None:
            surface = pygame.Surface(key)
            self._scratch_surfaces[key] = surface
        return surface

    def _render_net_surface(
        self,
        heightfield: NDArray[np.float32],
        lut: NDArray[np.uint8],
        state: StudioState,
    ) -> None:
        """Draws crisp isolines directly using culled or native paths."""
        h_dim, w_dim = heightfield.shape
        if w_dim != self.width or h_dim != self.height:
            draw_culled_contour_net_to_surface(
                surface=self.rendered_surface,
                coarse_wh=np.ascontiguousarray(heightfield.T),
                lut=lut,
                intervals=state.net_intervals,
                bandwidth=state.net_bandwidth,
                bg_color=config.COLOR_NET_BACKGROUND,
            )
            return

        scratch = self._get_contour_scratch()
        np.copyto(scratch.field_buf, heightfield.T)
        draw_contour_net_to_surface(
            surface=self.rendered_surface,
            heightfield_wh=scratch.field_buf,
            lut=lut,
            intervals=state.net_intervals,
            bandwidth=state.net_bandwidth,
            bg_color=config.COLOR_NET_BACKGROUND,
            scratch=scratch,
        )

    def _draw_hud(self, state: StudioState, fps: float) -> None:
        """Renders stacked blueprint metadata and active channel cards."""
        preset = state.preset_2d
        pal_name, _ = state.active_palette
        p_idx = state.preset_index_2d + 1
        total_p = len(PRESETS_2D)

        if preset.composite_index < 0:
            type_tag = "SOLO"
            persp_tag = "N/A (SOLO)"
        elif preset.uses_relief and preset.uses_crust:
            type_tag = "3-LAYER"
            persp_tag = "WARPED" if state.is_warped else "RADIAL"
        else:
            type_tag = "2-LAYER"
            persp_tag = "WARPED" if state.is_warped else "RADIAL"

        step_tag = (
            f"ACTIVE ({state.tile_size}px)"
            if state.tile_stepping_enabled
            else "OFF (Smooth)"
        )
        net_tag = "ACTIVE" if state.net_enabled else "OFF"

        hud_lines: List[Tuple[str, Tuple[int, int, int]]] = [
            (
                f"MODE 2: 2D SPATIAL SYNTHESIZER | TILE: {step_tag} | "
                f"NET: {net_tag}",
                config.COLOR_HUD_TEXT_ACTIVE,
            ),
            (
                f"PALETTE: {pal_name} | PERSP: {persp_tag}",
                config.COLOR_HUD_TEXT_NORMAL,
            ),
            (
                f"PRESET [{p_idx}/{total_p}] [{type_tag}] {preset.name}",
                config.COLOR_HUD_TEXT_NORMAL,
            ),
            (
                f"MATH  : {preset.shorthand}",
                config.COLOR_HUD_TEXT_NORMAL,
            ),
            (
                f"STATS : FPS: {fps:.1f} | VIEW: {self.width}x{self.height}",
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

        self._draw_channel_strip(
            x=10,
            y=10 + card1_h + 6,
            state=state,
        )

    def _draw_channel_strip(
        self,
        x: int,
        y: int,
        state: StudioState,
    ) -> None:
        """Draws channel strip and records clickable layer hitboxes."""
        col_active = (85, 155, 255)
        col_muted = (140, 145, 155)
        col_divider = (80, 85, 95)
        col_label = (220, 220, 220)
        col_silenced = (90, 95, 105)

        pad = 8
        preset = state.preset_2d

        if preset.uses_base:
            b_col = (
                col_active if state.selected_panel_index == 0 else col_muted
            )
            b_text = f"B: {state.base_cycles:.2f}"
            b_id: Optional[int] = 0
        else:
            b_col = col_silenced
            b_text = "B: Silenced"
            b_id = None

        if preset.uses_relief:
            r_col = (
                col_active if state.selected_panel_index == 1 else col_muted
            )
            r_text = f"R: {state.relief_cycles:.2f}"
            r_id: Optional[int] = 1
        else:
            r_col = col_silenced
            r_text = "R: Silenced"
            r_id = None

        if preset.uses_crust:
            c_col = (
                col_active if state.selected_panel_index == 2 else col_muted
            )
            c_text = f"C: {state.crust_cycles:.2f}"
            c_id: Optional[int] = 2
        else:
            c_col = col_silenced
            c_text = "C: Silenced"
            c_id = None

        segments: List[Tuple[str, Tuple[int, int, int], Optional[int]]] = [
            ("CHANNEL FOCUS: [ ", col_label, None),
            (b_text, b_col, b_id),
            (" | ", col_divider, None),
            (r_text, r_col, r_id),
            (" | ", col_divider, None),
            (c_text, c_col, c_id),
            (" ] cycles", col_label, None),
        ]

        rendered = [
            (self.font.render(text, True, color), ch_id)
            for text, color, ch_id in segments
        ]
        box_w = sum(surf.get_width() for surf, _ in rendered) + (pad * 2)
        box_h = max(surf.get_height() for surf, _ in rendered) + (pad * 2)

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

        state.channel_hitboxes.clear()
        curr_x = x + pad
        for surf, ch_id in rendered:
            self.screen.blit(surf, (curr_x, y + pad))
            if ch_id is not None:
                state.channel_hitboxes[ch_id] = pygame.Rect(
                    curr_x - 4, y, surf.get_width() + 8, box_h
                )
            curr_x += surf.get_width()
