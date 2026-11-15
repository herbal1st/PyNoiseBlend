"""Interactive parameter matrix deck renderer for Modes 3 and 4."""

import time
from typing import Dict, Optional, Tuple
import pygame
import config
from studio.state import StudioState
from utils.noise.noise_generator import NoiseProfile

_BASE_ROWS: Tuple[Tuple[int, str], ...] = (
    (config.ROW_PROFILE, "PROFILE   :"),
    (config.ROW_SCALE, "SCALE     :"),
    (config.ROW_OCTAVES, "OCTAVES   :"),
    (config.ROW_PERSIST, "PERSIST   :"),
    (config.ROW_LACUNARITY, "LACUNARITY:"),
    (config.ROW_SUBSAMPLE, "SUB-SAMPLE:"),
    (config.ROW_SEED, "SEED      :"),
)

_MATRIX_BG_CACHE: Dict[
    Tuple[int, int, Tuple[int, int, int, int]], pygame.Surface
] = {}


def draw_parameter_matrix(
    screen: pygame.Surface,
    font: pygame.font.Font,
    state: StudioState,
    x: int,
    y: int,
    is_mode_world: bool,
) -> int:
    """Renders interactive parameter matrix table and records hitboxes."""
    pad = 8
    line_h = font.get_linesize() + 2
    lbl_w = 115
    col_w = 155 if is_mode_world else 180
    num_cols = 3 if is_mode_world else 1

    box_w = pad + lbl_w + (num_cols * col_w) + pad
    total_rows = config.MATRIX_ROW_COUNT if is_mode_world else 7
    box_h = pad + ((total_rows + 1) * line_h) + pad

    bg_surf = _get_matrix_bg_surface(box_w, box_h, config.COLOR_HUD_CARD_2D)
    screen.blit(bg_surf, (x, y))

    pygame.draw.rect(
        screen,
        config.COLOR_SELECTION,
        (x, y, box_w, box_h),
        config.PANEL_SELECTION_WIDTH,
    )

    state.matrix_hitboxes.clear()
    state.channel_hitboxes.clear()

    _draw_header_row(
        screen, font, state, x, y, pad, lbl_w, col_w, line_h, is_mode_world
    )

    rows = list(_BASE_ROWS)
    if is_mode_world:
        rows.append((config.ROW_WORLD_SEED, "WORLD SEED:"))

    for row_idx, (row_id, row_lbl) in enumerate(rows):
        ry = y + pad + ((row_idx + 1) * line_h)
        lbl_surf = font.render(row_lbl, True, config.COLOR_HUD_TEXT_NORMAL)
        screen.blit(lbl_surf, (x + pad, ry))

        if row_id == config.ROW_WORLD_SEED:
            _draw_world_seed_cell(
                screen, font, state, x, ry, pad, lbl_w, col_w, line_h
            )
            continue

        for col in range(num_cols):
            cx = x + pad + lbl_w + (col * col_w)
            _draw_matrix_cell(
                screen,
                font,
                state,
                row_id,
                col,
                cx,
                ry,
                col_w,
                line_h,
                is_mode_world,
            )

    return box_h


def _get_matrix_bg_surface(
    width: int,
    height: int,
    bg_rgba: Tuple[int, int, int, int],
) -> pygame.Surface:
    """Returns cached translucent surface for parameter matrix deck."""
    key = (width, height, bg_rgba)
    cached = _MATRIX_BG_CACHE.get(key)
    if cached is not None:
        return cached

    if len(_MATRIX_BG_CACHE) >= 8:
        _MATRIX_BG_CACHE.clear()

    bg_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    bg_surf.fill(bg_rgba)
    _MATRIX_BG_CACHE[key] = bg_surf
    return bg_surf


def _draw_header_row(
    screen: pygame.Surface,
    font: pygame.font.Font,
    state: StudioState,
    x: int,
    y: int,
    pad: int,
    lbl_w: int,
    col_w: int,
    line_h: int,
    is_mode_world: bool,
) -> None:
    """Draws layer column headers and registers channel tab hitboxes."""
    hdr_y = y + pad
    lbl_surf = font.render("LAYERS  :", True, config.COLOR_HUD_TEXT_ACTIVE)
    screen.blit(lbl_surf, (x + pad, hdr_y))

    col_headers = (
        (("BASE (D)", 0), ("RELIEF (M)", 1), ("CRUST (C)", 2))
        if is_mode_world
        else (("NOISE LAYER", 0),)
    )
    for col_idx, (hdr_text, ch_id) in enumerate(col_headers):
        cx = x + pad + lbl_w + (col_idx * col_w)
        is_col_focus = state.selected_matrix_col == col_idx
        is_silenced = (
            is_mode_world and col_idx == 1 and not state.formula.uses_relief
        )

        hdr_col = (
            config.COLOR_HUD_TEXT_MUTED
            if is_silenced
            else (
                config.COLOR_HUD_TEXT_ACTIVE
                if is_col_focus
                else config.COLOR_HUD_TEXT_NORMAL
            )
        )
        t_surf = font.render(f"[ {hdr_text} ]", True, hdr_col)
        screen.blit(t_surf, (cx + 4, hdr_y))
        state.channel_hitboxes[ch_id] = pygame.Rect(
            cx, hdr_y - 2, col_w - 4, line_h
        )


def _draw_matrix_cell(
    screen: pygame.Surface,
    font: pygame.font.Font,
    state: StudioState,
    row: int,
    col: int,
    cx: int,
    ry: int,
    col_w: int,
    line_h: int,
    is_mode_world: bool,
) -> None:
    """Renders single data cell with selection and input highlighting."""
    cell_rect = pygame.Rect(cx, ry - 1, col_w - 6, line_h - 1)
    state.matrix_hitboxes[(row, col)] = cell_rect

    is_focused = (
        state.selected_matrix_row == row and state.selected_matrix_col == col
    )
    is_silenced = is_mode_world and col == 1 and not state.formula.uses_relief

    prof: Optional[NoiseProfile] = (
        state.get_layer_profile(col)
        if is_mode_world
        else state.active_noise_profile
    )
    val_text = _format_cell_value(state, prof, row, col, is_silenced)

    if is_focused and not is_silenced:
        pygame.draw.rect(
            screen,
            config.COLOR_SELECTION,
            cell_rect,
            config.PANEL_SELECTION_WIDTH,
        )

    text_col = (
        config.COLOR_HUD_TEXT_MUTED
        if is_silenced
        else (
            config.COLOR_INPUT_ACTIVE
            if is_focused
            else config.COLOR_HUD_TEXT_NORMAL
        )
    )
    t_surf = font.render(val_text, True, text_col)
    screen.blit(t_surf, (cx + 4, ry))


def _draw_world_seed_cell(
    screen: pygame.Surface,
    font: pygame.font.Font,
    state: StudioState,
    x: int,
    ry: int,
    pad: int,
    lbl_w: int,
    col_w: int,
    line_h: int,
) -> None:
    """Renders the dedicated master world seed row cell."""
    cx = x + pad + lbl_w
    cell_w = (col_w * 3) - 6
    cell_rect = pygame.Rect(cx, ry - 1, cell_w, line_h - 1)
    state.matrix_hitboxes[(config.ROW_WORLD_SEED, 0)] = cell_rect

    is_focused = state.selected_matrix_row == config.ROW_WORLD_SEED
    is_typing = (
        state.is_editing_seed
        and state.editing_seed_coord == (config.ROW_WORLD_SEED, 0)
    )

    if is_typing:
        val_text = _format_typing_seed(state.seed_input_buffer)
        text_col = config.COLOR_INPUT_ACTIVE
    else:
        val_text = f"[ {state.world_seed} ]"
        text_col = (
            config.COLOR_INPUT_ACTIVE
            if is_focused
            else config.COLOR_HUD_TEXT_NORMAL
        )

    if is_focused:
        pygame.draw.rect(
            screen,
            config.COLOR_SELECTION,
            cell_rect,
            config.PANEL_SELECTION_WIDTH,
        )

    t_surf = font.render(val_text, True, text_col)
    screen.blit(t_surf, (cx + 4, ry))


def _format_cell_value(
    state: StudioState,
    prof: Optional[NoiseProfile],
    row: int,
    col: int,
    is_silenced: bool,
) -> str:
    """Formats cell display string matching the active parameter."""
    if is_silenced:
        return "[ Silenced ]"
    if prof is None:
        return "[ None ]"

    if row == config.ROW_PROFILE:
        return f"[ {prof.profile_name} ]"
    if row == config.ROW_SCALE:
        return f"[ {prof.scale:.1f} px ]"
    if row == config.ROW_OCTAVES:
        return f"[ {prof.octaves} ]"
    if row == config.ROW_PERSIST:
        return f"[ {prof.decay:.2f} ]"
    if row == config.ROW_LACUNARITY:
        return f"[ {prof.lacunarity:.2f} ]"
    if row == config.ROW_SUBSAMPLE:
        return f"[ {prof.sub_sample_factor}x ]"
    if row == config.ROW_SEED:
        is_typing = (
            state.is_editing_seed
            and state.editing_seed_coord == (config.ROW_SEED, col)
        )
        if is_typing:
            return _format_typing_seed(state.seed_input_buffer)
        return f"[ {prof.seed} ]"

    return "[ - ]"


def _format_typing_seed(buffer: str) -> str:
    """Formats active seed text buffer with a blinking cursor."""
    cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
    return f"[ > {buffer}{cursor} < ]"
