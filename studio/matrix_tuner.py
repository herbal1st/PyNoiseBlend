"""Matrix parameter stepping, camera-anchored scaling, and resets."""

from typing import TYPE_CHECKING
import config
from utils.noise.noise_generator import SUB_SAMPLE_FACTORS

if TYPE_CHECKING:
    from studio.state import StudioState
    from utils.noise.noise_generator import NoiseProfile

ROW_PROFILE: int = 0
ROW_SCALE: int = 1
ROW_OCTAVES: int = 2
ROW_PERSIST: int = 3
ROW_LACUNARITY: int = 4
ROW_SUBSAMPLE: int = 5
ROW_SEED: int = 6
ROW_WORLD_SEED: int = 7
ROW_BUDGET: int = 8
MATRIX_ROW_COUNT: int = 9


def is_discrete_matrix_row(row: int) -> bool:
    """Checks if a matrix row index represents a discrete parameter."""
    return row in (
        ROW_PROFILE,
        ROW_OCTAVES,
        ROW_SUBSAMPLE,
        ROW_SEED,
        ROW_WORLD_SEED,
    )


def move_matrix_row(state: "StudioState", direction: int) -> None:
    """Steps matrix row selection up or down."""
    max_rows = MATRIX_ROW_COUNT if state.active_mode == 4 else 7
    state.selected_matrix_row = (
        state.selected_matrix_row + direction
    ) % max_rows


def move_matrix_column(state: "StudioState", direction: int) -> None:
    """Steps matrix column selection across Base, Relief, and Crust."""
    if state.active_mode != 4:
        return

    cols = [0]
    if state.formula.uses_relief:
        cols.append(1)
    if state.formula.uses_crust:
        cols.append(2)

    if state.selected_matrix_col in cols:
        cur_idx = cols.index(state.selected_matrix_col)
        next_idx = (cur_idx + direction) % len(cols)
        state.selected_matrix_col = cols[next_idx]
    else:
        state.selected_matrix_col = cols[0]

    state.selected_panel_index = state.selected_matrix_col


def shift_world_offsets(
    state: "StudioState",
    dx: float,
    dy: float,
) -> None:
    """Applies world pixel deltas to the focused Mode 4 noise layer."""
    col = state.selected_matrix_col
    if col == 0:
        state.world_base_offset_x += dx
        state.world_base_offset_y += dy
    elif col == 1:
        state.world_macro_offset_x += dx
        state.world_macro_offset_y += dy
    elif col == 2:
        state.world_crust_offset_x += dx
        state.world_crust_offset_y += dy


def cycle_layer_profile(
    state: "StudioState",
    col: int,
    direction: int,
) -> None:
    """Cycles assigned noise profile key for a Mode 4 layer column."""
    cur_key = state.get_layer_profile_key(col)
    if cur_key not in state.noise_keys:
        return
    idx = (state.noise_keys.index(cur_key) + direction) % len(state.noise_keys)
    set_layer_profile_key(state, col, state.noise_keys[idx])


def set_layer_profile_key(
    state: "StudioState",
    col: int,
    key: str,
) -> None:
    """Rebinds the noise profile for a specific layer column."""
    if key not in state.working_noise_profiles:
        return
    if col == 0:
        state.world_base_profile_key = key
    elif col == 1:
        state.world_macro_profile_key = key
    elif col == 2:
        state.world_crust_profile_key = key
    state.mark_discrete_change()


def step_matrix_param(
    state: "StudioState",
    direction: int,
    factor: float = 1.0,
    dt: float = 0.0,
    single_layer: bool = False,
) -> None:
    """Dispatches adjustment to the active matrix parameter cell."""
    is_mode_world = state.active_mode == 4
    col = state.selected_matrix_col if is_mode_world else 0
    row = state.selected_matrix_row

    if is_mode_world and col == 1 and not state.formula.uses_relief:
        return

    if row == ROW_PROFILE:
        if is_mode_world:
            cycle_layer_profile(state, col, direction)
        else:
            state.cycle_noise_profile(direction)
        return

    if row == ROW_BUDGET and is_mode_world:
        rate = config.BUDGET_SCRUB_RATE * (dt if dt > 0.0 else 0.05)
        state.set_crust_budget(state.crust_budget + (float(direction) * rate))
        return

    if row == ROW_WORLD_SEED:
        state.world_seed = (state.world_seed + direction) & 0xFFFFFFFF
        state.mark_discrete_change()
        return

    if row == ROW_SCALE:
        if is_mode_world:
            scale_world_layers(state, factor, single_layer=single_layer)
        else:
            _scale_single_layer(state, 0, factor)
        state.mark_scrubbing()
        return

    prof = (
        state.get_layer_profile(col)
        if is_mode_world
        else state.active_noise_profile
    )
    if prof is None:
        return

    _step_profile_field(state, prof, row, direction, dt)


def scale_world_layers(
    state: "StudioState",
    factor: float,
    single_layer: bool,
) -> None:
    """Scales all active layers or only focused layer anchored on cam."""
    if single_layer:
        _scale_single_layer(state, state.selected_matrix_col, factor)
        return

    _scale_single_layer(state, 0, factor)
    if state.formula.uses_relief:
        _scale_single_layer(state, 1, factor)
    if state.formula.uses_crust:
        _scale_single_layer(state, 2, factor)


def reset_noise_tuning(state: "StudioState") -> None:
    """Restores active Mode 3 noise profile to YAML baseline."""
    key = state.active_noise_profile.profile_name
    if key in state.default_noise_profiles:
        state.working_noise_profiles[key] = (
            state.default_noise_profiles[key].clone()
        )
    state.noise_offset_x = 0.0
    state.noise_offset_y = 0.0
    state.mark_discrete_change()


def reset_world_tuning(state: "StudioState", reset_all: bool) -> None:
    """Restores Mode 4 layer profiles and offsets to YAML baselines."""
    wp = state.active_world_profile
    if reset_all:
        state.world_seed = wp.world_seed
        state.world_base_profile_key = wp.base_noise_profile
        state.world_macro_profile_key = wp.macro_noise_profile
        state.world_crust_profile_key = wp.crust_noise_profile
        state.crust_budget = config.CRUST_BUDGET
        state.relief_budget = config.RELIEF_BUDGET

    layer_keys = (
        state.world_base_profile_key,
        state.world_macro_profile_key,
        state.world_crust_profile_key,
    )
    for col, key in enumerate(layer_keys):
        if reset_all or state.selected_matrix_col == col:
            if key in state.default_noise_profiles:
                state.working_noise_profiles[key] = (
                    state.default_noise_profiles[key].clone()
                )

    if reset_all or state.selected_matrix_col == 0:
        state.world_base_offset_x = 0.0
        state.world_base_offset_y = 0.0
    if reset_all or state.selected_matrix_col == 1:
        state.world_macro_offset_x = 0.0
        state.world_macro_offset_y = 0.0
    if reset_all or state.selected_matrix_col == 2:
        state.world_crust_offset_x = 0.0
        state.world_crust_offset_y = 0.0

    state.mark_discrete_change()


def _step_profile_field(
    state: "StudioState",
    prof: "NoiseProfile",
    row: int,
    direction: int,
    dt: float,
) -> None:
    """Applies discrete or continuous adjustment to a NoiseProfile."""
    if row == ROW_OCTAVES:
        prof.octaves = max(
            config.OCTAVE_MIN,
            min(config.OCTAVE_MAX, prof.octaves + direction),
        )
        state.mark_discrete_change()
    elif row == ROW_PERSIST:
        rate = config.DECAY_STEP_RATE * dt if dt > 0.0 else 0.05
        prof.decay = max(
            config.DECAY_MIN,
            min(config.DECAY_MAX, prof.decay + (direction * rate)),
        )
        state.mark_scrubbing()
    elif row == ROW_LACUNARITY:
        rate = config.LACUNARITY_STEP_RATE * dt if dt > 0.0 else 0.05
        prof.lacunarity = max(
            config.LACUNARITY_MIN,
            min(
                config.LACUNARITY_MAX,
                prof.lacunarity + (direction * rate),
            ),
        )
        state.mark_scrubbing()
    elif row == ROW_SUBSAMPLE:
        factors = SUB_SAMPLE_FACTORS
        cur_idx = (
            factors.index(prof.sub_sample_factor)
            if prof.sub_sample_factor in factors
            else 0
        )
        next_idx = (cur_idx + direction) % len(factors)
        prof.sub_sample_factor = factors[next_idx]
        state.mark_discrete_change()
    elif row == ROW_SEED:
        prof.seed = (prof.seed + direction) & 0xFFFFFFFF
        state.mark_discrete_change()


def _scale_single_layer(
    state: "StudioState",
    col: int,
    factor: float,
) -> None:
    """Scales feature scale while keeping view anchored on camera."""
    if state.active_mode == 3:
        old_s = state.active_noise_profile.scale
        new_s = _clamp_scale(old_s * factor)
        if new_s != old_s:
            ratio = new_s / old_s
            state.active_noise_profile.scale = new_s
            state.noise_offset_x = _project_anchored_offset(
                state.cam_x, state.noise_offset_x, ratio
            )
            state.noise_offset_y = _project_anchored_offset(
                state.cam_y, state.noise_offset_y, ratio
            )
        return

    prof = state.get_layer_profile(col)
    if prof is None:
        return

    old_s = prof.scale
    new_s = _clamp_scale(old_s * factor)
    if new_s == old_s:
        return

    ratio = new_s / old_s
    prof.scale = new_s

    if col == 0:
        state.world_base_offset_x = _project_anchored_offset(
            state.cam_x, state.world_base_offset_x, ratio
        )
        state.world_base_offset_y = _project_anchored_offset(
            state.cam_y, state.world_base_offset_y, ratio
        )
    elif col == 1:
        state.world_macro_offset_x = _project_anchored_offset(
            state.cam_x, state.world_macro_offset_x, ratio
        )
        state.world_macro_offset_y = _project_anchored_offset(
            state.cam_y, state.world_macro_offset_y, ratio
        )
    elif col == 2:
        state.world_crust_offset_x = _project_anchored_offset(
            state.cam_x, state.world_crust_offset_x, ratio
        )
        state.world_crust_offset_y = _project_anchored_offset(
            state.cam_y, state.world_crust_offset_y, ratio
        )


def _clamp_scale(value: float) -> float:
    """Clamps noise scale within configured engine boundaries."""
    return max(config.NOISE_MIN_SCALE, min(config.NOISE_MAX_SCALE, value))


def _project_anchored_offset(
    cam: float,
    offset: float,
    ratio: float,
) -> float:
    """Projects a spatial offset around the camera anchor after scaling."""
    return cam - ((cam - offset) * ratio)
