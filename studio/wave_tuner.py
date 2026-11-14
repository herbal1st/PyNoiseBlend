"""Mode 1 and Mode 2 wave channel focus, offset shifting, and scaling."""

from typing import TYPE_CHECKING, List, Tuple
import config

if TYPE_CHECKING:
    from studio.state import StudioState


def get_valid_channels(state: "StudioState") -> Tuple[int, ...]:
    """Returns valid selectable channel indices for active mode."""
    if state.active_mode == 3:
        return (0,)

    if state.active_mode == 2:
        preset = state.preset_2d
        channels: List[int] = []
        if preset.uses_base:
            channels.append(0)
        if preset.uses_relief:
            channels.append(1)
        if preset.uses_crust:
            channels.append(2)
        return tuple(channels) if channels else (0,)

    formula = state.formula
    channels = [0]
    if formula.uses_relief:
        channels.append(1)
    if formula.uses_crust:
        channels.append(2)
    if state.active_mode == 1:
        channels.append(3)
    return tuple(channels)


def move_channel_focus(state: "StudioState", direction: int) -> None:
    """Moves channel focus across valid inputs for active mode."""
    channels = get_valid_channels(state)
    if state.selected_panel_index in channels:
        cur_pos = channels.index(state.selected_panel_index)
        next_pos = (cur_pos + direction) % len(channels)
        state.selected_panel_index = channels[next_pos]
    else:
        state.selected_panel_index = channels[0]

    if state.active_mode == 1:
        state.needs_rebuild_1d = True


def clamp_channel_focus(state: "StudioState") -> None:
    """Clamps channel focus to valid channels for the active mode."""
    channels = get_valid_channels(state)
    if state.selected_panel_index not in channels:
        state.selected_panel_index = channels[0]
        if state.active_mode == 1:
            state.needs_rebuild_1d = True


def shift_wave_offsets(
    state: "StudioState",
    ndx: float,
    ndy: float,
    single_layer: bool,
) -> None:
    """Applies normalized 2D drag deltas to Mode 1/2 wave channels."""
    if not single_layer or state.selected_panel_index == 0:
        state.base_offset_x += ndx
        state.base_offset_y += ndy
    if not single_layer or state.selected_panel_index == 1:
        state.relief_offset_x += ndx
        state.relief_offset_y += ndy
    if not single_layer or state.selected_panel_index == 2:
        state.crust_offset_x += ndx
        state.crust_offset_y += ndy


def scale_wave_cycles(
    state: "StudioState",
    factor: float,
    single_layer: bool,
) -> None:
    """Scales Mode 1/2 wave cycles with 2D center anchoring."""
    scale_all = (not single_layer) or (
        state.active_mode == 1 and state.selected_panel_index == 3
    )
    anchor_2d = state.active_mode == 2

    if scale_all or state.selected_panel_index == 0:
        new_base, ratio = _compute_scaled_cycle(
            state.base_cycles,
            config.BASE_MIN_CYCLES,
            config.BASE_MAX_CYCLES,
            factor,
        )
        if new_base != state.base_cycles:
            state.base_cycles = new_base
            if anchor_2d:
                state.base_offset_x *= ratio

    if scale_all or state.selected_panel_index == 1:
        new_relief, ratio = _compute_scaled_cycle(
            state.relief_cycles,
            config.RELIEF_MIN_CYCLES,
            config.RELIEF_MAX_CYCLES,
            factor,
        )
        if new_relief != state.relief_cycles:
            state.relief_cycles = new_relief
            if anchor_2d:
                state.relief_offset_y *= ratio

    if scale_all or state.selected_panel_index == 2:
        new_crust, ratio = _compute_scaled_cycle(
            state.crust_cycles,
            config.CRUST_MIN_CYCLES,
            config.CRUST_MAX_CYCLES,
            factor,
        )
        if new_crust != state.crust_cycles:
            state.crust_cycles = new_crust
            if anchor_2d and _is_warped_2layer(state):
                state.crust_offset_y *= ratio


def _compute_scaled_cycle(
    old_val: float,
    min_val: float,
    max_val: float,
    factor: float,
) -> Tuple[float, float]:
    """Clamps scaled cycle frequency and returns (new_val, old_over_new)."""
    new_val = max(min_val, min(max_val, old_val * factor))
    ratio = old_val / new_val if new_val != old_val else 1.0
    return new_val, ratio


def _is_warped_2layer(state: "StudioState") -> bool:
    """Checks if active Mode 2 preset is a warped 2-layer formula."""
    preset = state.preset_2d
    return (
        preset.composite_index >= 0
        and not preset.uses_relief
        and state.is_warped
    )
