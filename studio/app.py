"""Master studio application loop coordinating input, math, and renderers."""

from typing import Dict
import numpy as np
from numpy.typing import NDArray
import pygame
import config
from core.wave_synthesizer import (
    CoordinateMesh2D,
    build_coarse_mesh_2d,
    build_coordinate_mesh_2d,
    build_scrub_mesh_2d,
    build_wave_frame_1d,
    build_wave_frame_2d,
)
from studio.input_handler import InputHandler
from studio.renderers.renderer_1d import Renderer1D
from studio.renderers.renderer_2d_synth import Renderer2DSynth
from studio.renderers.renderer_lighting import RendererLighting
from studio.renderers.renderer_noise import RendererNoise
from studio.renderers.renderer_world import RendererWorld
from studio.state import StudioState
from utils.error_handler import install_crash_handler

install_crash_handler()


class TerrainStudioApp:
    """Coordinates window lifecycle, 5 viewports, and dirty re-bakes."""

    def __init__(self) -> None:
        """Initializes Pygame, state, coordinate meshes, and renderers."""
        pygame.init()
        self.screen: pygame.Surface = pygame.display.set_mode(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        )
        pygame.display.set_caption(config.APP_TITLE)
        self.clock: pygame.time.Clock = pygame.time.Clock()

        self.state: StudioState = StudioState()
        self.input_handler: InputHandler = InputHandler()

        self.x_coords: NDArray[np.float32] = np.linspace(
            0.0,
            1.0,
            config.SCREEN_WIDTH,
            dtype=np.float32,
        )
        self.continuous_mesh: CoordinateMesh2D = build_coordinate_mesh_2d(
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT,
        )
        self.scrub_mesh_2d: CoordinateMesh2D = build_scrub_mesh_2d(
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT,
            config.SCRUB_SUB_SAMPLE_FACTOR,
        )

        self.coarse_meshes: Dict[int, CoordinateMesh2D] = {
            sz: build_coarse_mesh_2d(
                config.SCREEN_WIDTH, config.SCREEN_HEIGHT, sz
            )
            for sz in (8, 16, 32)
        }

        self.renderer_1d: Renderer1D = Renderer1D(self.screen)
        self.renderer_2d: Renderer2DSynth = Renderer2DSynth(self.screen)
        self.renderer_noise: RendererNoise = RendererNoise(
            self.screen, self.state
        )
        self.renderer_world: RendererWorld = RendererWorld(
            self.screen, self.state
        )
        self.renderer_lighting: RendererLighting = RendererLighting(
            self.screen, self.state
        )

    def run(self) -> None:
        """Executes main event polling, wave math, and frame drawing."""
        running = True
        try:
            while running:
                raw_dt = self.clock.tick(config.FPS) / 1000.0
                dt = min(0.05, max(0.001, raw_dt))

                running = self.input_handler.handle_input(self.state, dt)
                if not running:
                    break

                self.state.tick_timers(dt)
                self.state.check_resolution_fallback(
                    self.clock.get_fps(), dt
                )
                if self.state.active_mode == 5:
                    self.renderer_lighting.update(self.state, dt)
                self._render_active_mode()
                pygame.display.flip()
        finally:
            pygame.quit()

    def _render_active_mode(self) -> None:
        """Delegates evaluation and rendering to the active viewport mode."""
        _, active_lut = self.state.active_palette
        fps = self.clock.get_fps()

        if self.state.active_mode == 1:
            if self.state.needs_rebuild_1d:
                frame_1d = build_wave_frame_1d(
                    x_coordinates=self.x_coords,
                    formula=self.state.formula,
                    base_cycles=self.state.base_cycles,
                    relief_cycles=self.state.relief_cycles,
                    crust_cycles=self.state.crust_cycles,
                    tile_stepping=self.state.tile_stepping_enabled,
                    tile_size=self.state.tile_size,
                    base_offset_x=self.state.base_offset_x,
                    relief_offset_x=self.state.relief_offset_x,
                    crust_offset_x=self.state.crust_offset_x,
                )
                self.renderer_1d.update_surface(self.state, frame_1d)
                self.state.needs_rebuild_1d = False

            self.renderer_1d.render()

        elif self.state.active_mode == 2:
            if self.state.needs_rebuild_2d:
                mesh = self._select_mesh_2d()
                heightfield_2d = build_wave_frame_2d(
                    mesh=mesh,
                    preset=self.state.preset_2d,
                    base_cycles=self.state.base_cycles,
                    relief_cycles=self.state.relief_cycles,
                    crust_cycles=self.state.crust_cycles,
                    is_warped=self.state.is_warped,
                    base_offset_x=self.state.base_offset_x,
                    base_offset_y=self.state.base_offset_y,
                    relief_offset_x=self.state.relief_offset_x,
                    relief_offset_y=self.state.relief_offset_y,
                    crust_offset_x=self.state.crust_offset_x,
                    crust_offset_y=self.state.crust_offset_y,
                )
                self.renderer_2d.update_surface(
                    heightfield=heightfield_2d,
                    lut=active_lut,
                    state=self.state,
                )
                self.state.needs_rebuild_2d = False

            self.renderer_2d.render(self.state, fps)

        elif self.state.active_mode == 3:
            self.renderer_noise.render(self.state, fps)

        elif self.state.active_mode == 4:
            self.renderer_world.render(self.state, fps)

        elif self.state.active_mode == 5:
            self.renderer_lighting.render(self.state, fps)

    def _select_mesh_2d(self) -> CoordinateMesh2D:
        """Selects active 2D coordinate mesh based on stepping and scrub."""
        if self.state.tile_stepping_enabled:
            return self.coarse_meshes.get(
                self.state.tile_size, self.continuous_mesh
            )
        if self.state.is_scrubbing:
            return self.scrub_mesh_2d
        return self.continuous_mesh


def run_app() -> None:
    """Boots the standalone terrain studio application."""
    TerrainStudioApp().run()
