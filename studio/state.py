"""Mutable runtime selections and studio tuning state machine."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import pygame
import config
from core.formulas import FORMULA_CATALOG, PRESETS_2D, FormulaPreset, Preset2D
from studio.matrix_tuner import ROW_WORLD_SEED
from utils.noise.noise_generator import NoiseGenerator, NoiseProfile
from world.chunks.chunk_profile import ChunkProfile, ChunkProfileLoader
from world.chunks.world_profile import WorldProfile, WorldProfileLoader
from world.lighting.lighting_profile import (
    LightingProfile,
    LightingProfileLoader,
)
from world.strata.strata_loader import StrataLoader
from world.strata.strata_profile import StrataProfile


@dataclass(slots=True)
class StudioState:
    """Coordinates shared formula, palette, camera, and viewport state."""

    active_mode: int = config.DEFAULT_VIEW_MODE
    composite_index: int = 0
    preset_index_2d: int = 3
    selected_panel_index: int = 0

    selected_matrix_row: int = 1
    selected_matrix_col: int = 0

    crust_budget: float = config.CRUST_BUDGET
    relief_budget: float = config.RELIEF_BUDGET

    base_cycles: float = config.DEFAULT_BASE_CYCLES
    relief_cycles: float = config.DEFAULT_RELIEF_CYCLES
    crust_cycles: float = config.DEFAULT_CRUST_CYCLES

    base_offset_x: float = 0.0
    base_offset_y: float = 0.0
    relief_offset_x: float = 0.0
    relief_offset_y: float = 0.0
    crust_offset_x: float = 0.0
    crust_offset_y: float = 0.0

    noise_offset_x: float = 0.0
    noise_offset_y: float = 0.0

    world_base_offset_x: float = 0.0
    world_base_offset_y: float = 0.0
    world_macro_offset_x: float = 0.0
    world_macro_offset_y: float = 0.0
    world_crust_offset_x: float = 0.0
    world_crust_offset_y: float = 0.0

    world_seed: int = 101
    world_base_profile_key: str = "MICRO_SIMPLEX"
    world_macro_profile_key: str = "MID_VALUE"
    world_crust_profile_key: str = "MACRO_VALUE"

    is_editing_seed: bool = False
    seed_input_buffer: str = ""
    editing_seed_coord: Tuple[int, int] = (-1, -1)

    tile_stepping_enabled: bool = False
    tile_size: int = config.DEFAULT_TILE_SIZE
    net_enabled: bool = config.DEFAULT_NET_ENABLED
    net_intervals: float = config.DEFAULT_NET_INTERVALS
    net_bandwidth: float = config.DEFAULT_NET_BANDWIDTH
    is_inverted: bool = False
    is_warped: bool = False
    compute_gradients: bool = config.ENABLE_SLOPE_GRADIENTS

    cam_x: float = 0.0
    cam_y: float = 0.0

    needs_rebuild_1d: bool = True
    needs_rebuild_2d: bool = True
    needs_refine_chunks: bool = False
    is_scrubbing: bool = False
    scrub_timer: float = 0.0

    selected_light_row: int = 2
    light_azimuth_deg: float = 90.0
    light_altitude_deg: float = 10.0
    light_contrast: float = 135.0
    light_shadow_intensity: float = 170.0
    light_highlight_intensity: float = 120.0
    light_ambient_ratio: float = 0.25
    light_orbit_scale: float = 1.0
    light_day_cycle: float = 60.0
    light_time_ratio: float = 0.30
    light_exposure: float = 0.85
    light_sub_sample: int = 4
    light_time_dirty: bool = False
    light_world_source_index: int = 0
    light_preset_transforms: Dict[str, Tuple[float, float, float]] = field(
        default_factory=dict
    )
    light_hitboxes: Dict[int, pygame.Rect] = field(default_factory=dict)

    lighting_profiles: Dict[str, LightingProfile] = field(
        default_factory=LightingProfileLoader.load_profiles
    )
    lighting_keys: List[str] = field(default_factory=list)
    active_lighting_index: int = 0

    active_palette_index: int = 0
    channel_hitboxes: Dict[int, pygame.Rect] = field(default_factory=dict)
    matrix_hitboxes: Dict[Tuple[int, int], pygame.Rect] = field(
        default_factory=dict
    )

    strata_profiles: Dict[str, StrataProfile] = field(
        default_factory=StrataLoader.load_profiles
    )
    palette_keys: List[str] = field(default_factory=list)

    chunk_profiles: Dict[str, ChunkProfile] = field(
        default_factory=ChunkProfileLoader.load_profiles
    )
    default_noise_profiles: Dict[str, NoiseProfile] = field(
        default_factory=NoiseGenerator.load_profiles
    )
    working_noise_profiles: Dict[str, NoiseProfile] = field(
        default_factory=dict
    )
    noise_keys: List[str] = field(default_factory=list)
    active_noise_index: int = 0

    world_profiles: Dict[str, WorldProfile] = field(
        default_factory=WorldProfileLoader.load_profiles
    )
    world_keys: List[str] = field(default_factory=list)
    active_world_index: int = 0

    def __post_init__(self) -> None:
        """Initializes sorted keys, live working profiles, and indices."""
        self.palette_keys = list(self.strata_profiles.keys())
        self.noise_keys = list(self.default_noise_profiles.keys())
        self.world_keys = list(self.world_profiles.keys())
        self.lighting_keys = list(self.lighting_profiles.keys())

        self.working_noise_profiles = {
            k: v.clone() for k, v in self.default_noise_profiles.items()
        }

        if config.ACTIVE_WORLD_PROFILE in self.world_keys:
            self.active_world_index = self.world_keys.index(
                config.ACTIVE_WORLD_PROFILE
            )

        wp = self.active_world_profile
        self.world_seed = wp.world_seed
        self.world_base_profile_key = wp.base_noise_profile
        self.world_macro_profile_key = wp.macro_noise_profile
        self.world_crust_profile_key = wp.crust_noise_profile

        if config.ACTIVE_LIGHTING_PROFILE in self.lighting_keys:
            self.active_lighting_index = self.lighting_keys.index(
                config.ACTIVE_LIGHTING_PROFILE
            )
        self.reset_lighting_tuning()

    @property
    def formula(self) -> FormulaPreset:
        """Returns the active 1D composite formula preset."""
        return FORMULA_CATALOG[self.composite_index]

    @property
    def preset_2d(self) -> Preset2D:
        """Returns the active 2D planar projection preset."""
        return PRESETS_2D[self.preset_index_2d]

    @property
    def active_strata(self) -> StrataProfile:
        """Returns the currently active StrataProfile model."""
        key = self.palette_keys[self.active_palette_index]
        return self.strata_profiles[key]

    @property
    def active_palette(self) -> Tuple[str, NDArray[np.uint8]]:
        """Returns active palette name and contiguous (1024, 3) LUT array."""
        prof = self.active_strata
        lut = prof.inverted_lut if self.is_inverted else prof.color_lut
        name = (
            f"[INV] {prof.profile_name}"
            if self.is_inverted
            else prof.profile_name
        )
        return name, lut

    @property
    def active_noise_profile(self) -> NoiseProfile:
        """Returns working NoiseProfile specification for Mode 3."""
        key = self.noise_keys[self.active_noise_index]
        return self.working_noise_profiles[key]

    @property
    def active_world_profile(self) -> WorldProfile:
        """Returns currently active WorldProfile master recipe."""
        key = self.world_keys[self.active_world_index]
        return self.world_profiles[key]

    @property
    def active_chunk_profile(self) -> ChunkProfile:
        """Returns active ChunkProfile sizing and margin rules."""
        return self.chunk_profiles.get(
            config.ACTIVE_CHUNK_PROFILE,
            list(self.chunk_profiles.values())[0],
        )

    @property
    def active_lighting_profile(self) -> LightingProfile:
        """Returns currently active LightingProfile preset."""
        key = self.lighting_keys[self.active_lighting_index]
        return self.lighting_profiles[key]

    @property
    def active_light_world_label(self) -> str:
        """Returns display label for the active Mode 5 world source."""
        if self.light_world_source_index == 0:
            return "MODE 4 (LIVE)"
        idx = (self.light_world_source_index - 1) % len(self.world_keys)
        return self.world_keys[idx]

    def set_crust_budget(self, new_crust: float) -> None:
        """Updates crust and relief budget within safe clamping bounds."""
        clamped = max(
            config.CRUST_BUDGET_MIN, min(config.CRUST_BUDGET_MAX, new_crust)
        )
        self.crust_budget = round(clamped, 4)
        self.relief_budget = round(1.0 - self.crust_budget, 4)
        self.mark_scrubbing()

    def get_layer_profile_key(self, col: int) -> Optional[str]:
        """Returns the noise profile key bound to given layer column."""
        if col == 0:
            return self.world_base_profile_key
        if col == 1:
            return self.world_macro_profile_key
        if col == 2:
            return self.world_crust_profile_key
        return None

    def get_layer_profile(self, col: int) -> Optional[NoiseProfile]:
        """Returns the mutable working NoiseProfile for a layer column."""
        key = self.get_layer_profile_key(col)
        if key is not None and key in self.working_noise_profiles:
            return self.working_noise_profiles[key]
        return None

    def start_seed_editing(self, row: int, col: int) -> None:
        """Enters text typing mode for the specified seed cell."""
        self.is_editing_seed = True
        norm_col = 0 if row == ROW_WORLD_SEED else col
        self.editing_seed_coord = (row, norm_col)

        if row == ROW_WORLD_SEED:
            self.seed_input_buffer = str(self.world_seed)
            return

        prof = (
            self.get_layer_profile(norm_col)
            if self.active_mode == 4
            else self.active_noise_profile
        )
        self.seed_input_buffer = str(prof.seed) if prof else "0"

    def commit_seed_editing(self) -> None:
        """Validates typed buffer, commits seed, and exits edit mode."""
        if not self.is_editing_seed:
            return

        try:
            val = int(self.seed_input_buffer) & 0xFFFFFFFF
        except ValueError:
            self.cancel_seed_editing()
            return

        row, col = self.editing_seed_coord
        if row == ROW_WORLD_SEED:
            self.world_seed = val
        else:
            prof = (
                self.get_layer_profile(col)
                if self.active_mode == 4
                else self.active_noise_profile
            )
            if prof is not None:
                prof.seed = val

        self.cancel_seed_editing()
        self.mark_discrete_change()

    def cancel_seed_editing(self) -> None:
        """Aborts seed typing mode without modifying parameters."""
        self.is_editing_seed = False
        self.seed_input_buffer = ""
        self.editing_seed_coord = (-1, -1)

    def mark_scrubbing(self) -> None:
        """Engages coarse scrubbing mode and resets the settle timer."""
        self.is_scrubbing = True
        self.scrub_timer = config.SCRUB_SETTLE_DELAY
        self.needs_rebuild_1d = True
        self.needs_rebuild_2d = True

    def mark_discrete_change(self) -> None:
        """Cancels active scrubbing and marks viewports for crisp rebuild."""
        self.is_scrubbing = False
        self.scrub_timer = 0.0
        self.needs_rebuild_1d = True
        self.needs_rebuild_2d = True

    def settle_scrubbing(self) -> None:
        """Immediately settles active scrubbing into full-detail rebuild."""
        if not self.is_scrubbing:
            return
        self.is_scrubbing = False
        self.scrub_timer = 0.0
        self.needs_rebuild_1d = True
        self.needs_rebuild_2d = True
        self.needs_refine_chunks = True

    def tick_timers(self, dt: float) -> None:
        """Advances settle debounce timer and triggers final refinement."""
        if not self.is_scrubbing:
            return
        self.scrub_timer -= dt
        if self.scrub_timer <= 0.0:
            self.settle_scrubbing()

    def reset_wave_channels(self, reset_all: bool) -> None:
        """Restores focused or all wave cycles and spatial offsets."""
        all_channels = reset_all or (self.selected_panel_index == 3)
        if all_channels or self.selected_panel_index == 0:
            self.base_cycles = config.DEFAULT_BASE_CYCLES
            self.base_offset_x = 0.0
            self.base_offset_y = 0.0
        if all_channels or self.selected_panel_index == 1:
            self.relief_cycles = config.DEFAULT_RELIEF_CYCLES
            self.relief_offset_x = 0.0
            self.relief_offset_y = 0.0
        if all_channels or self.selected_panel_index == 2:
            self.crust_cycles = config.DEFAULT_CRUST_CYCLES
            self.crust_offset_x = 0.0
            self.crust_offset_y = 0.0
        if all_channels:
            self.crust_budget = config.CRUST_BUDGET
            self.relief_budget = config.RELIEF_BUDGET
        self.mark_discrete_change()

    def cycle_formula_1d(self, direction: int) -> None:
        """Cycles 1D composite formula catalog and aligns 2D preset."""
        self.composite_index = (
            self.composite_index + direction
        ) % len(FORMULA_CATALOG)

        for idx, p in enumerate(PRESETS_2D):
            if p.composite_index == self.composite_index:
                self.preset_index_2d = idx
                break
        self.mark_discrete_change()

    def cycle_formula_2d(self, direction: int) -> None:
        """Cycles 2D planar presets and syncs 1D pointer when non-solo."""
        self.preset_index_2d = (
            self.preset_index_2d + direction
        ) % len(PRESETS_2D)

        target_comp = self.preset_2d.composite_index
        if target_comp >= 0:
            self.composite_index = target_comp
        self.mark_discrete_change()

    def cycle_noise_profile(self, direction: int) -> None:
        """Cycles active noise profile in Mode 3."""
        self.active_noise_index = (
            self.active_noise_index + direction
        ) % len(self.noise_keys)
        self.mark_discrete_change()

    def cycle_world_profile(self, direction: int) -> None:
        """Cycles active world noise build and updates layer keys."""
        self.active_world_index = (
            self.active_world_index + direction
        ) % len(self.world_keys)
        wp = self.active_world_profile
        self.world_seed = wp.world_seed
        self.world_base_profile_key = wp.base_noise_profile
        self.world_macro_profile_key = wp.macro_noise_profile
        self.world_crust_profile_key = wp.crust_noise_profile
        self.mark_discrete_change()

    def cycle_lighting_profile(self, direction: int) -> None:
        """Cycles active solar lighting preset forward or backward."""
        self.active_lighting_index = (
            self.active_lighting_index + direction
        ) % len(self.lighting_keys)
        self.reset_lighting_tuning()

    def cycle_light_world_source(self, direction: int) -> None:
        """Cycles Mode 5 world source between Mode 4 Live and YAML builds."""
        total = 1 + len(self.world_keys)
        self.light_world_source_index = (
            self.light_world_source_index + direction
        ) % total
        self.mark_discrete_change()

    def reset_lighting_tuning(self) -> None:
        """Restores Mode 5 solar parameters from active lighting preset."""
        lp = self.active_lighting_profile
        self.light_azimuth_deg = lp.start_azimuth_deg
        self.light_altitude_deg = lp.start_altitude_deg
        self.light_contrast = lp.terrain_steepness
        self.light_shadow_intensity = float(lp.shadow_intensity)
        self.light_highlight_intensity = float(lp.highlight_intensity)
        self.light_ambient_ratio = lp.ambient_ratio
        self.light_day_cycle = lp.day_cycle_duration
        self.light_orbit_scale = 0.0 if lp.day_cycle_duration <= 0.0 else 1.0
        self.light_time_ratio = lp.start_time_ratio
        self.light_exposure = lp.neutral_exposure
        self.light_sub_sample = lp.lightmap_sub_sample
        self.light_preset_transforms.clear()
        self.light_time_dirty = True

    def get_light_preset_transform(
        self,
        preset_key: str,
    ) -> Tuple[float, float, float]:
        """Returns (zoom_ratio, offset_x, offset_y) for a Mode 5 preset."""
        return self.light_preset_transforms.get(preset_key, (1.0, 0.0, 0.0))

    def scale_light_preset(
        self,
        preset_key: str,
        factor: float,
    ) -> None:
        """Scales a Mode 5 world preset proportionally around camera."""
        wp = self.world_profiles.get(preset_key)
        if wp is None:
            return

        keys = [wp.base_noise_profile]
        if self.formula.uses_relief:
            keys.append(wp.macro_noise_profile)
        if self.formula.uses_crust:
            keys.append(wp.crust_noise_profile)

        old_ratio, off_x, off_y = self.get_light_preset_transform(
            preset_key
        )
        eff_scales = []
        for key in keys:
            prof = self.default_noise_profiles.get(key)
            if prof is not None and prof.scale > 0.0:
                eff_scales.append(prof.scale * old_ratio)

        if not eff_scales:
            return

        min_f = max(config.NOISE_MIN_SCALE / s for s in eff_scales)
        max_f = min(config.NOISE_MAX_SCALE / s for s in eff_scales)
        clamped_factor = (
            max(min_f, min(max_f, factor)) if min_f <= max_f else factor
        )
        new_ratio = old_ratio * clamped_factor
        if abs(new_ratio - old_ratio) <= 1e-7:
            return

        new_ox = self.cam_x - ((self.cam_x - off_x) * clamped_factor)
        new_oy = self.cam_y - ((self.cam_y - off_y) * clamped_factor)
        self.light_preset_transforms[preset_key] = (
            new_ratio,
            new_ox,
            new_oy,
        )

    def cycle_palette(self, direction: int) -> None:
        """Cycles active color palette forward or backward."""
        self.active_palette_index = (
            self.active_palette_index + direction
        ) % len(self.palette_keys)
        self.mark_discrete_change()

    def toggle_inversion(self) -> None:
        """Toggles color ramp inversion on and off."""
        self.is_inverted = not self.is_inverted
        self.mark_discrete_change()

    def toggle_perspective(self) -> None:
        """Toggles spatial perspective between standard/radial and warped."""
        self.is_warped = not self.is_warped
        self.mark_discrete_change()

    def toggle_net_mode(self) -> None:
        """Toggles isoline relief net display mode."""
        self.net_enabled = not self.net_enabled
        self.mark_discrete_change()

    def step_tile_size(self, direction: int) -> None:
        """Cycles tile stepping power-of-two sizes and toggles off at ends."""
        if direction > 0:
            if not self.tile_stepping_enabled:
                self.tile_stepping_enabled = True
                self.tile_size = config.MIN_TILE_SIZE
            else:
                next_sz = self.tile_size * 2
                if next_sz > config.MAX_TILE_SIZE:
                    self.tile_stepping_enabled = False
                else:
                    self.tile_size = next_sz
        else:
            if not self.tile_stepping_enabled:
                self.tile_stepping_enabled = True
                self.tile_size = config.MAX_TILE_SIZE
            else:
                next_sz = self.tile_size // 2
                if next_sz < config.MIN_TILE_SIZE:
                    self.tile_stepping_enabled = False
                else:
                    self.tile_size = next_sz
        self.mark_discrete_change()

    def sync_to_mode(self, mode: int) -> None:
        """Aligns active formulas and clears hitboxes on mode switch."""
        self.active_mode = mode
        self.channel_hitboxes.clear()
        self.matrix_hitboxes.clear()
        self.light_hitboxes.clear()
        self.cancel_seed_editing()

        if mode in (3, 4):
            self.selected_matrix_row = 1
            self.selected_matrix_col = 0
        elif mode == 5:
            self.selected_light_row = 2

        if mode == 1 and self.preset_2d.composite_index < 0:
            self.composite_index = 0
            for idx, p in enumerate(PRESETS_2D):
                if p.composite_index == 0:
                    self.preset_index_2d = idx
                    break
        self.mark_discrete_change()
