"""Base rendering helpers and shared HUD card components."""

from typing import Dict, List, Optional, Tuple
import pygame
import config


class BaseRenderer:
    """Provides shared typography and padded HUD background blitting."""

    def __init__(self, screen: pygame.Surface) -> None:
        """Initializes typography resources and card surface cache."""
        self.screen: pygame.Surface = screen
        self.width: int = screen.get_width()
        self.height: int = screen.get_height()
        self.font: pygame.font.Font = pygame.font.SysFont(
            "monospace", 13, bold=True
        )
        self._card_bg_cache: Dict[
            Tuple[int, int, Tuple[int, int, int, int]], pygame.Surface
        ] = {}

    def get_translucent_card_surface(
        self,
        width: int,
        height: int,
        bg_rgba: Tuple[int, int, int, int],
    ) -> pygame.Surface:
        """Returns cached translucent surface for given size and RGBA."""
        key = (width, height, bg_rgba)
        cached = self._card_bg_cache.get(key)
        if cached is not None:
            return cached

        if len(self._card_bg_cache) >= 32:
            self._card_bg_cache.clear()

        bg_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        bg_surf.fill(bg_rgba)
        self._card_bg_cache[key] = bg_surf
        return bg_surf

    def draw_card(
        self,
        x: int,
        y: int,
        lines: List[Tuple[str, Tuple[int, int, int]]],
        bg_rgba: Tuple[int, int, int, int],
        pad: int = 8,
        border_color: Optional[Tuple[int, int, int]] = None,
        border_width: int = 2,
    ) -> int:
        """Renders dynamic auto-sizing card and returns its total height."""
        rendered = [
            self.font.render(text, True, color) for text, color in lines
        ]
        line_h = self.font.get_linesize()
        max_w = max(surf.get_width() for surf in rendered)

        box_w = max_w + (pad * 2)
        box_h = (len(rendered) * line_h) + (pad * 2)

        bg_surf = self.get_translucent_card_surface(box_w, box_h, bg_rgba)
        self.screen.blit(bg_surf, (x, y))

        if border_color is not None:
            pygame.draw.rect(
                self.screen,
                border_color,
                (x, y, box_w, box_h),
                border_width,
            )

        for idx, surf in enumerate(rendered):
            self.screen.blit(surf, (x + pad, y + pad + (idx * line_h)))

        return box_h

    def draw_studio_footer(
        self,
        active_mode: int = 1,
        extra_ctrl: str = "",
    ) -> None:
        """Renders combined two-line studio nav and controls card."""
        mode_tags = [
            ("[1] Wave Math", 1),
            ("[2] 2D Strata", 2),
            ("[3] Raw Noise", 3),
            ("[4] World Map", 4),
        ]
        text_parts = [
            f"*{lbl}*" if mid == active_mode else lbl
            for lbl, mid in mode_tags
        ]
        nav_str = "STUDIO: " + " | ".join(text_parts)

        formula_str = "[TAB]: Formula | " if active_mode != 3 else ""
        build_str = "[B]: Build | " if active_mode in (3, 4) else ""
        persp_str = "[P]: Persp | " if active_mode == 2 else ""
        base_ctrl = (
            f"[1-4]: View | {formula_str}{build_str}"
            f"[LMB/RMB/Arrows]: Nav/Zoom (+SHIFT: Layer) | "
            f"[SPACE]: Step | [C/I]: Palette | {persp_str}"
            f"[N]: Net | [R]: Reset"
        )
        ctrl_str = f"{base_ctrl} | {extra_ctrl}" if extra_ctrl else base_ctrl

        pad = 6
        line_h = self.font.get_linesize()
        box_h = (2 * line_h) + (pad * 2)
        ry = self.height - box_h - 10

        footer_lines: List[Tuple[str, Tuple[int, int, int]]] = [
            (nav_str, config.COLOR_HUD_TEXT_ACTIVE),
            (ctrl_str, config.COLOR_HUD_TEXT_NORMAL),
        ]
        self.draw_card(
            x=10,
            y=ry,
            lines=footer_lines,
            bg_rgba=config.COLOR_HUD_CARD_2D,
            pad=pad,
        )
