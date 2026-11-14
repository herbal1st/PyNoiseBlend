"""Event processing and continuous input handling for the studio."""

import math
from typing import Tuple
import pygame
import config
from studio import matrix_tuner, wave_tuner
from studio.state import StudioState


class InputHandler:
    """Handles keyboard events, unified mouse drags, tuning, and panning."""

    def __init__(self) -> None:
        """Initializes active mouse drag button and cursor tracking."""
        self.active_mouse_button: int = 0
        self.drag_start: Tuple[int, int] = (0, 0)

    def handle_input(self, state: StudioState, dt: float) -> bool:
        """Processes discrete events and continuous held keys. False quits."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if state.is_editing_seed:
                if not self._handle_seed_text_event(state, event):
                    return False
                continue

            if event.type == pygame.KEYDOWN:
                if not self._handle_key_down(state, event):
                    return False
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button in (1, 3)
                and self.active_mouse_button == 0
            ):
                self._handle_mouse_down(state, event.button, event.pos)
            elif (
                event.type == pygame.MOUSEBUTTONUP
                and event.button == self.active_mouse_button
            ):
                self.active_mouse_button = 0
                keys = pygame.key.get_pressed()
                if not (keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]):
                    state.settle_scrubbing()
            elif (
                event.type == pygame.MOUSEMOTION
                and self.active_mouse_button != 0
            ):
                self._handle_mouse_motion(state, event.pos, dt)

        if not state.is_editing_seed:
            self._handle_held_keys(state, dt)

        return True

    def _handle_seed_text_event(
        self,
        state: StudioState,
        event: pygame.event.Event,
    ) -> bool:
        """Captures alphanumeric keyboard input while seed field is active."""
        if event.type != pygame.KEYDOWN:
            return True

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            state.commit_seed_editing()
        elif event.key == pygame.K_ESCAPE:
            state.cancel_seed_editing()
        elif event.key == pygame.K_BACKSPACE:
            state.seed_input_buffer = state.seed_input_buffer[:-1]
        elif event.unicode.isdigit() and len(state.seed_input_buffer) < 10:
            state.seed_input_buffer += event.unicode

        return True

    def _handle_key_down(
        self,
        state: StudioState,
        event: pygame.event.Event,
    ) -> bool:
        """Routes discrete key press events."""
        if event.key == pygame.K_ESCAPE:
            return False

        mods = pygame.key.get_mods()
        shift_held = bool(mods & pygame.KMOD_SHIFT)

        if event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            state.sync_to_mode(int(event.unicode))
            self._clamp_focus_for_active_mode(state)
        elif event.key == pygame.K_SPACE:
            state.step_tile_size(-1 if shift_held else 1)
        elif event.key == pygame.K_TAB:
            direction = -1 if shift_held else 1
            if state.active_mode in (1, 4):
                state.cycle_formula_1d(direction)
                self._clamp_focus_for_active_mode(state)
            elif state.active_mode == 2:
                state.cycle_formula_2d(direction)
                self._clamp_focus_for_active_mode(state)
        elif event.key == pygame.K_b:
            direction = -1 if shift_held else 1
            if state.active_mode == 3:
                state.cycle_noise_profile(direction)
            elif state.active_mode == 4:
                state.cycle_world_profile(direction)
        elif event.key == pygame.K_c:
            state.cycle_palette(-1 if shift_held else 1)
        elif event.key == pygame.K_i:
            state.toggle_inversion()
        elif event.key == pygame.K_p:
            if state.active_mode == 2:
                state.toggle_perspective()
        elif event.key == pygame.K_n:
            state.toggle_net_mode()
        elif event.key == pygame.K_r:
            self._reset_tuning(state, mods)
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if state.active_mode in (3, 4):
                is_seed_row = state.selected_matrix_row in (
                    config.ROW_SEED,
                    config.ROW_WORLD_SEED,
                )
                if is_seed_row:
                    state.start_seed_editing(
                        state.selected_matrix_row, state.selected_matrix_col
                    )
        elif event.key in (pygame.K_UP, pygame.K_DOWN):
            direction = -1 if event.key == pygame.K_UP else 1
            if state.active_mode in (3, 4):
                if shift_held:
                    matrix_tuner.move_matrix_column(state, direction)
                else:
                    matrix_tuner.move_matrix_row(state, direction)
            else:
                wave_tuner.move_channel_focus(state, direction)
        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
            if state.active_mode in (3, 4):
                if StudioState.is_discrete_row(state.selected_matrix_row):
                    direction = 1 if event.key == pygame.K_RIGHT else -1
                    matrix_tuner.step_matrix_param(state, direction)

        return True

    def _handle_mouse_down(
        self,
        state: StudioState,
        button: int,
        pos: Tuple[int, int],
    ) -> None:
        """Selects clicked cell, toggles parameters, or engages drag."""
        shift_held = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)

        if state.active_mode in (3, 4):
            handled, start_drag = self._handle_matrix_mouse_down(
                state, button, pos, shift_held
            )
            if handled:
                if start_drag:
                    self.active_mouse_button = button
                    self.drag_start = pos
                return

        if state.active_mode == 1:
            panel_h = config.SCREEN_HEIGHT // config.PANEL_COUNT
            panel_idx = min(
                config.PANEL_COUNT - 1, max(0, pos[1] // panel_h)
            )
            if panel_idx not in wave_tuner.get_valid_channels(state):
                return
            if state.selected_panel_index != panel_idx:
                state.selected_panel_index = panel_idx
                state.needs_rebuild_1d = True
            self.active_mouse_button = button
            self.drag_start = pos
            return

        if state.active_mode == 2:
            for channel_idx, rect in state.channel_hitboxes.items():
                if rect.collidepoint(pos):
                    state.selected_panel_index = channel_idx
                    return

        self.active_mouse_button = button
        self.drag_start = pos

    def _handle_matrix_mouse_down(
        self,
        state: StudioState,
        button: int,
        pos: Tuple[int, int],
        shift_held: bool,
    ) -> Tuple[bool, bool]:
        """Handles matrix cell and tab clicks. Returns (handled, drag)."""
        for (row, col), rect in state.matrix_hitboxes.items():
            if not rect.collidepoint(pos):
                continue

            state.selected_matrix_row = row
            state.selected_matrix_col = col
            state.selected_panel_index = col
            direction = -1 if (button == 3 or shift_held) else 1

            if row in (config.ROW_SEED, config.ROW_WORLD_SEED):
                if button == 1 and not shift_held:
                    state.start_seed_editing(row, col)
                else:
                    matrix_tuner.step_matrix_param(state, direction)
                return True, False

            if StudioState.is_discrete_row(row):
                matrix_tuner.step_matrix_param(state, direction)
                return True, False

            return True, True

        for col_idx, rect in state.channel_hitboxes.items():
            if rect.collidepoint(pos):
                state.selected_matrix_col = col_idx
                state.selected_panel_index = col_idx
                return True, False

        return False, False

    def _handle_mouse_motion(
        self,
        state: StudioState,
        pos: Tuple[int, int],
        dt: float,
    ) -> None:
        """Dispatches active left or right mouse drag deltas."""
        dx = pos[0] - self.drag_start[0]
        dy = pos[1] - self.drag_start[1]
        self.drag_start = pos
        if dx == 0 and dy == 0:
            return

        shift_held = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)

        if self.active_mouse_button == 1:
            self._handle_left_drag(state, dx, dy, shift_held)
        elif self.active_mouse_button == 3 and dx != 0:
            factor = math.exp(float(dx) * config.MOUSE_ZOOM_SENSITIVITY)
            if state.active_mode in (3, 4):
                if not StudioState.is_discrete_row(state.selected_matrix_row):
                    direction = 1 if dx > 0 else -1
                    matrix_tuner.step_matrix_param(
                        state,
                        direction,
                        factor=factor,
                        dt=dt,
                        single_layer=shift_held,
                    )
            else:
                wave_tuner.scale_wave_cycles(
                    state, factor, single_layer=shift_held
                )
                state.mark_scrubbing()

    def _handle_left_drag(
        self,
        state: StudioState,
        dx: int,
        dy: int,
        shift_held: bool,
    ) -> None:
        """Moves all layers together, or only focused layer with SHIFT."""
        if state.active_mode in (1, 2):
            ndx = float(dx) / float(config.SCREEN_WIDTH)
            ndy = float(dy) / float(config.SCREEN_HEIGHT)
            single_layer = shift_held and state.selected_panel_index != 3
            wave_tuner.shift_wave_offsets(state, ndx, ndy, single_layer)
            state.mark_scrubbing()
            return

        if not shift_held:
            state.cam_x -= float(dx)
            state.cam_y -= float(dy)
            return

        if state.active_mode == 3:
            state.noise_offset_x += float(dx)
            state.noise_offset_y += float(dy)
        elif state.active_mode == 4:
            matrix_tuner.shift_world_offsets(state, float(dx), float(dy))

        state.mark_scrubbing()

    def _handle_held_keys(self, state: StudioState, dt: float) -> None:
        """Processes held keys for camera panning and continuous scaling."""
        keys = pygame.key.get_pressed()

        if state.active_mode in (3, 4):
            step = config.TEST_PAN_SPEED * dt
            if keys[pygame.K_w]:
                state.cam_y -= step
            if keys[pygame.K_s]:
                state.cam_y += step
            if keys[pygame.K_a]:
                state.cam_x -= step
            if keys[pygame.K_d]:
                state.cam_x += step

        move_left = keys[pygame.K_LEFT]
        move_right = keys[pygame.K_RIGHT]
        if move_left == move_right or dt <= 0.0:
            return

        direction = 1 if move_right else -1
        shift_held = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)

        if state.active_mode in (3, 4):
            if StudioState.is_discrete_row(state.selected_matrix_row):
                return
            step_scale = 1.0 + (config.NOISE_SCALE_ZOOM_RATE * dt)
            factor = step_scale if move_right else (1.0 / step_scale)
            matrix_tuner.step_matrix_param(
                state,
                direction,
                factor=factor,
                dt=dt,
                single_layer=shift_held,
            )
            return

        step_scale = 1.0 + (config.CYCLE_ZOOM_RATE * dt)
        factor = step_scale if move_right else (1.0 / step_scale)
        wave_tuner.scale_wave_cycles(state, factor, single_layer=shift_held)
        state.mark_scrubbing()

    def _clamp_focus_for_active_mode(self, state: StudioState) -> None:
        """Clamps channel or matrix column focus for the active mode."""
        wave_tuner.clamp_channel_focus(state)
        if state.active_mode == 4:
            matrix_tuner.move_matrix_column(state, 0)

    def _reset_tuning(self, state: StudioState, mods: int) -> None:
        """Resets all parameters, or only the focused layer with SHIFT."""
        reset_all = not bool(mods & pygame.KMOD_SHIFT)
        if reset_all:
            state.cam_x = 0.0
            state.cam_y = 0.0

        if state.active_mode == 3:
            matrix_tuner.reset_noise_tuning(state)
        elif state.active_mode == 4:
            matrix_tuner.reset_world_tuning(state, reset_all=reset_all)
        else:
            state.reset_wave_channels(reset_all=reset_all)
