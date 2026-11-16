"""Composite world chunk provider orchestrating noise and formulas."""

import math
from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import config
from core.formulas import FORMULA_CATALOG, FormulaPreset
from utils.error_handler import fail_fast
from utils.noise.noise_generator import NoiseGenerator, NoiseProfile
from utils.resampler import bilinear_resample_2d
from world.chunks.chunk import Chunk
from world.chunks.chunk_profile import ChunkProfile
from world.chunks.providers.base_chunk_provider import BaseChunkProvider
from world.chunks.world_profile import WorldProfile


class WorldChunkProvider(BaseChunkProvider):
    """Evaluates multi-noise modulation, halos, and color mapping."""

    def __init__(
        self,
        world_profile: WorldProfile,
        chunk_profile: ChunkProfile,
        noise_profiles: Dict[str, NoiseProfile],
        strata_lut: NDArray[np.uint8],
        formula_override: Optional[FormulaPreset] = None,
        world_seed_override: Optional[int] = None,
        base_profile_key: Optional[str] = None,
        macro_profile_key: Optional[str] = None,
        crust_profile_key: Optional[str] = None,
        net_enabled: bool = False,
        net_intervals: float = config.DEFAULT_NET_INTERVALS,
        net_bandwidth: float = config.DEFAULT_NET_BANDWIDTH,
        tile_stepping_enabled: bool = False,
        tile_size: int = config.DEFAULT_TILE_SIZE,
        compute_gradients: bool = config.ENABLE_SLOPE_GRADIENTS,
    ) -> None:
        """Binds profiles, sets up noise engines, and links formula."""
        super().__init__(
            chunk_profile=chunk_profile,
            color_lut=strata_lut,
            net_enabled=net_enabled,
            net_intervals=net_intervals,
            net_bandwidth=net_bandwidth,
            tile_stepping_enabled=tile_stepping_enabled,
            tile_size=tile_size,
        )
        self.world_profile: WorldProfile = world_profile
        self.world_seed: int = (
            world_seed_override
            if world_seed_override is not None
            else world_profile.world_seed
        )

        self.base_profile_key: str = (
            base_profile_key or world_profile.base_noise_profile
        )
        self.macro_profile_key: str = (
            macro_profile_key or world_profile.macro_noise_profile
        )
        self.crust_profile_key: str = (
            crust_profile_key or world_profile.crust_noise_profile
        )
        self.formula: FormulaPreset = (
            formula_override
            if formula_override is not None
            else FORMULA_CATALOG[0]
        )
        self.compute_gradients: bool = compute_gradients

        self.base_noise: NoiseGenerator = self._build_noise_engine(
            self.base_profile_key, noise_profiles, seed_offset=101
        )
        self.macro_noise: Optional[NoiseGenerator] = (
            self._build_noise_engine(
                self.macro_profile_key, noise_profiles, seed_offset=202
            )
            if self.formula.uses_relief
            else None
        )
        self.crust_noise: Optional[NoiseGenerator] = (
            self._build_noise_engine(
                self.crust_profile_key, noise_profiles, seed_offset=303
            )
            if self.formula.uses_crust
            else None
        )

    def set_compute_gradients(self, enabled: bool) -> None:
        """Toggles slope gradient calculation during chunk baking."""
        self.compute_gradients = enabled

    def set_formula(self, formula: FormulaPreset) -> None:
        """Updates active formula specification dynamically."""
        self.formula = formula

    def set_layer_scales(
        self,
        base_scale: float,
        macro_scale: float,
        crust_scale: float,
    ) -> None:
        """Updates noise layer scale wavelengths in place."""
        self.base_noise.set_scale(base_scale)
        if self.macro_noise is not None:
            self.macro_noise.set_scale(macro_scale)
        if self.crust_noise is not None:
            self.crust_noise.set_scale(crust_scale)

    def set_layer_offsets(
        self,
        base_x: float,
        base_y: float,
        macro_x: float,
        macro_y: float,
        crust_x: float,
        crust_y: float,
    ) -> None:
        """Updates noise layer spatial world offsets in place."""
        self.base_noise.set_offset(base_x, base_y)
        if self.macro_noise is not None:
            self.macro_noise.set_offset(macro_x, macro_y)
        if self.crust_noise is not None:
            self.crust_noise.set_offset(crust_x, crust_y)

    def sync_layer_profile(
        self,
        channel: int,
        key: str,
        prof: NoiseProfile,
    ) -> None:
        """Updates parameters for a specified layer without losing offsets."""
        seed_offset = 101 if channel == 0 else (202 if channel == 1 else 303)
        eff_seed = (self.world_seed + prof.seed + seed_offset) & 0xFFFFFFFF

        gen: Optional[NoiseGenerator]
        if channel == 0:
            self.base_profile_key = key
            gen = self.base_noise
        elif channel == 1:
            self.macro_profile_key = key
            gen = self.macro_noise
        else:
            self.crust_profile_key = key
            gen = self.crust_noise

        if gen is None:
            return

        gen.sync_from_profile(
            prof, seed_override=eff_seed, sync_sub_sample=True
        )

    def sync_world_seed(
        self,
        new_seed: int,
        noise_profiles: Dict[str, NoiseProfile],
    ) -> None:
        """Updates master world seed and refreshes all layer seed hashes."""
        self.world_seed = new_seed
        bindings = (
            (0, self.base_profile_key),
            (1, self.macro_profile_key),
            (2, self.crust_profile_key),
        )
        for channel, key in bindings:
            if key in noise_profiles:
                self.sync_layer_profile(channel, key, noise_profiles[key])

    def generate_chunk(
        self,
        cx: int,
        cy: int,
        sub_sample_override: Optional[int] = None,
    ) -> Chunk:
        """Generates elevation, optional gradients, and bakes surface."""
        chunk_sz = self._chunk_profile.chunk_size
        t_sz = self.tile_size if self.tile_stepping_enabled else 0
        world_floor = self._get_world_floor(sub_sample_override)
        sub_factor = self._resolve_sub_sample(world_floor)

        wx_origin = float(cx * chunk_sz)
        wy_origin = float(cy * chunk_sz)

        if self.tile_stepping_enabled and self.compute_gradients:
            h, gx, gy, rgb_buffer = self._generate_tiled_chunk_data(
                wx_origin, wy_origin, chunk_sz, t_sz
            )
            surface = self._bake_surface_from_rgb(rgb_buffer, chunk_sz)
            return Chunk(
                cx=cx,
                cy=cy,
                size=chunk_sz,
                elevation=h,
                gx=gx,
                gy=gy,
                surface=surface,
            )

        needs_tile_halo = self.net_enabled and self.tile_stepping_enabled
        needs_halo = self.compute_gradients or needs_tile_halo

        gx_out: Optional[NDArray[np.float32]] = None
        gy_out: Optional[NDArray[np.float32]] = None

        if needs_halo:
            h_padded = self._sample_padded_elevation(
                wx_origin, wy_origin, chunk_sz, sub_factor, world_floor, t_sz
            )
            if self.compute_gradients:
                h, gx_out, gy_out = self._extract_elevation_and_gradients(
                    h_padded, chunk_sz, sub_factor
                )
            else:
                h = self._extract_elevation_only(
                    h_padded, chunk_sz, sub_factor
                )

            if needs_tile_halo:
                rgb_buffer = self._bake_padded_tile_contour_rgb(
                    h_padded, chunk_sz
                )
            else:
                rgb_buffer = self._render_field_to_rgb(h)
        else:
            h = self._sample_unpadded_elevation(
                wx_origin, wy_origin, chunk_sz, sub_factor, world_floor, t_sz
            )
            rgb_buffer = self._render_field_to_rgb(h)

        surface = self._bake_surface_from_rgb(rgb_buffer, chunk_sz)
        return Chunk(
            cx=cx,
            cy=cy,
            size=chunk_sz,
            elevation=h,
            gx=gx_out,
            gy=gy_out,
            surface=surface,
        )

    def sample_viewport_field(
        self,
        origin_x: float,
        origin_y: float,
        width: int,
        height: int,
        sub_factor: int,
    ) -> NDArray[np.float32]:
        """Samples compact un-upscaled float32 field for viewport preview."""
        if self.tile_stepping_enabled:
            return self._evaluate_formula_grid(
                ox=origin_x,
                oy=origin_y,
                width=width,
                height=height,
                step=1.0,
                tile_size=self.tile_size,
                upscale=False,
            )

        safe_sub = max(1, sub_factor)
        eval_w = (width // safe_sub) + 1
        eval_h = (height // safe_sub) + 1
        return self._evaluate_formula_grid(
            ox=origin_x,
            oy=origin_y,
            width=eval_w,
            height=eval_h,
            step=float(safe_sub),
            tile_size=0,
            upscale=False,
        )

    def _get_world_floor(
        self,
        sub_sample_override: Optional[int],
    ) -> int:
        """Computes world and fallback sub-sampling floor divider."""
        base_floor = max(1, self.world_profile.sub_sample_factor)
        if sub_sample_override is not None:
            return max(base_floor, sub_sample_override)
        return base_floor

    def _resolve_sub_sample(self, world_floor: int) -> int:
        """Determines finest active layer divider for composite grid."""
        if self.tile_stepping_enabled:
            return 1
        active = [max(world_floor, self.base_noise.profile.sub_sample_factor)]
        if self.formula.uses_relief and self.macro_noise is not None:
            active.append(
                max(world_floor, self.macro_noise.profile.sub_sample_factor)
            )
        if self.formula.uses_crust and self.crust_noise is not None:
            active.append(
                max(world_floor, self.crust_noise.profile.sub_sample_factor)
            )
        return max(1, min(active))

    def _generate_tiled_chunk_data(
        self,
        wx_origin: float,
        wy_origin: float,
        chunk_sz: int,
        tile_sz: int,
    ) -> Tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.uint8],
    ]:
        """Computes tile-centroid heights and smooth tile-to-tile slopes."""
        tiles_per_chunk = max(1, chunk_sz // tile_sz)
        pad_tiles = tiles_per_chunk + 2
        t_f = float(tile_sz)

        h_tile_pad = self._evaluate_formula_grid(
            ox=wx_origin + (0.5 * t_f) - t_f,
            oy=wy_origin + (0.5 * t_f) - t_f,
            width=pad_tiles,
            height=pad_tiles,
            step=t_f,
            tile_size=0,
            upscale=False,
        )
        gy_full, gx_full = np.gradient(h_tile_pad)
        inv_step = np.float32(1.0 / t_f)

        h_tile = h_tile_pad[1:-1, 1:-1]
        gx_tile = (gx_full[1:-1, 1:-1] * inv_step).astype(np.float32)
        gy_tile = (gy_full[1:-1, 1:-1] * inv_step).astype(np.float32)

        h = np.ascontiguousarray(
            np.repeat(np.repeat(h_tile, tile_sz, axis=0), tile_sz, axis=1)[
                :chunk_sz, :chunk_sz
            ].astype(np.float32)
        )
        gx = np.ascontiguousarray(
            np.repeat(np.repeat(gx_tile, tile_sz, axis=0), tile_sz, axis=1)[
                :chunk_sz, :chunk_sz
            ].astype(np.float32)
        )
        gy = np.ascontiguousarray(
            np.repeat(np.repeat(gy_tile, tile_sz, axis=0), tile_sz, axis=1)[
                :chunk_sz, :chunk_sz
            ].astype(np.float32)
        )

        if self.net_enabled:
            h_exp_pad = np.repeat(
                np.repeat(h_tile_pad, tile_sz, axis=0), tile_sz, axis=1
            )
            h_halo = np.ascontiguousarray(
                h_exp_pad[
                    tile_sz - 1 : tile_sz + chunk_sz + 1,
                    tile_sz - 1 : tile_sz + chunk_sz + 1,
                ].astype(np.float32)
            )
            rgb_buffer = self._bake_padded_tile_contour_rgb(h_halo, chunk_sz)
        else:
            rgb_buffer = self._render_field_to_rgb(h)

        return h, gx, gy, rgb_buffer

    def _sample_padded_elevation(
        self,
        wx_origin: float,
        wy_origin: float,
        chunk_sz: int,
        sub_factor: int,
        world_floor: int,
        tile_size: int,
    ) -> NDArray[np.float32]:
        """Samples composite heightfield including 1-sample gradient halo."""
        step = float(sub_factor)
        pad_res = (
            (chunk_sz // sub_factor) + 3
            if sub_factor > 1
            else chunk_sz + 2
        )
        return self._evaluate_formula_grid(
            ox=wx_origin - step,
            oy=wy_origin - step,
            width=pad_res,
            height=pad_res,
            step=step,
            tile_size=tile_size,
            world_floor=world_floor,
            composite_sub=sub_factor,
        )

    def _sample_unpadded_elevation(
        self,
        wx_origin: float,
        wy_origin: float,
        chunk_sz: int,
        sub_factor: int,
        world_floor: int,
        tile_size: int,
    ) -> NDArray[np.float32]:
        """Samples composite heightfield without a 1-sample border halo."""
        if sub_factor > 1:
            eval_res = (chunk_sz // sub_factor) + 1
            h_coarse = self._evaluate_formula_grid(
                ox=wx_origin,
                oy=wy_origin,
                width=eval_res,
                height=eval_res,
                step=float(sub_factor),
                tile_size=0,
                world_floor=world_floor,
                composite_sub=sub_factor,
            )
            return bilinear_resample_2d(h_coarse, chunk_sz, chunk_sz)

        h_native = self._evaluate_formula_grid(
            ox=wx_origin,
            oy=wy_origin,
            width=chunk_sz,
            height=chunk_sz,
            step=1.0,
            tile_size=tile_size,
            world_floor=world_floor,
            composite_sub=1,
        )
        return np.ascontiguousarray(h_native)

    @staticmethod
    def _extract_elevation_only(
        h_padded: NDArray[np.float32],
        chunk_sz: int,
        sub_factor: int,
    ) -> NDArray[np.float32]:
        """Trims 1-sample halo from elevation and upsamples if needed."""
        h_core = h_padded[1:-1, 1:-1].astype(np.float32)
        if sub_factor > 1:
            return bilinear_resample_2d(h_core, chunk_sz, chunk_sz)
        return np.ascontiguousarray(h_core.copy())

    @staticmethod
    def _extract_elevation_and_gradients(
        h_padded: NDArray[np.float32],
        chunk_sz: int,
        sub_factor: int,
    ) -> Tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        """Computes gradient halo, trims edges, and upsamples if needed."""
        gy_full, gx_full = np.gradient(h_padded)
        gx_core = gx_full[1:-1, 1:-1].astype(np.float32)
        gy_core = gy_full[1:-1, 1:-1].astype(np.float32)
        h_core = h_padded[1:-1, 1:-1].astype(np.float32)

        if sub_factor > 1:
            inv_step = np.float32(1.0 / float(sub_factor))
            gx_core *= inv_step
            gy_core *= inv_step
            return (
                bilinear_resample_2d(h_core, chunk_sz, chunk_sz),
                bilinear_resample_2d(gx_core, chunk_sz, chunk_sz),
                bilinear_resample_2d(gy_core, chunk_sz, chunk_sz),
            )

        return (
            np.ascontiguousarray(h_core.copy()),
            np.ascontiguousarray(gx_core),
            np.ascontiguousarray(gy_core),
        )

    def _evaluate_formula_grid(
        self,
        ox: float,
        oy: float,
        width: int,
        height: int,
        step: float,
        tile_size: int = 0,
        upscale: bool = True,
        world_floor: int = 1,
        composite_sub: int = 1,
    ) -> NDArray[np.float32]:
        """Samples noise layers and routes them through active formula."""
        if tile_size > 1 or not upscale:
            b = self.base_noise.sample_grid(
                ox, oy, width, height,
                step=step, tile_size=tile_size,
                upscale=False, bypass_profile_sub=True,
            )
            r = (
                self.macro_noise.sample_grid(
                    ox, oy, width, height,
                    step=step, tile_size=tile_size,
                    upscale=False, bypass_profile_sub=True,
                )
                if self.formula.uses_relief and self.macro_noise is not None
                else np.zeros_like(b)
            )
            c = (
                self.crust_noise.sample_grid(
                    ox, oy, width, height,
                    step=step, tile_size=tile_size,
                    upscale=False, bypass_profile_sub=True,
                )
                if self.formula.uses_crust and self.crust_noise is not None
                else np.zeros_like(b)
            )
            raw = self.formula.func(b, r, c)
            return np.clip(raw, 0.0, 1.0).astype(np.float32)

        base_sub = max(world_floor, self.base_noise.profile.sub_sample_factor)
        b = self._sample_layer_grid(
            self.base_noise, ox, oy, width, height,
            step, base_sub, composite_sub,
        )

        if self.formula.uses_relief and self.macro_noise is not None:
            m_sub = max(
                world_floor, self.macro_noise.profile.sub_sample_factor
            )
            r = self._sample_layer_grid(
                self.macro_noise, ox, oy, width, height,
                step, m_sub, composite_sub,
            )
        else:
            r = np.zeros_like(b)

        if self.formula.uses_crust and self.crust_noise is not None:
            c_sub = max(
                world_floor, self.crust_noise.profile.sub_sample_factor
            )
            c = self._sample_layer_grid(
                self.crust_noise, ox, oy, width, height,
                step, c_sub, composite_sub,
            )
        else:
            c = np.zeros_like(b)

        raw = self.formula.func(b, r, c)
        return np.clip(raw, 0.0, 1.0).astype(np.float32)

    def _sample_layer_grid(
        self,
        gen: NoiseGenerator,
        ox: float,
        oy: float,
        width: int,
        height: int,
        step: float,
        layer_sub: int,
        composite_sub: int,
    ) -> NDArray[np.float32]:
        """Evaluates a single layer at its divider and aligns to grid."""
        if layer_sub <= composite_sub or width < 2 or height < 2:
            return gen.sample_grid(
                ox, oy, width, height,
                step=step, tile_size=0,
                upscale=False, bypass_profile_sub=True,
            )

        ratio = float(layer_sub) / float(composite_sub)
        coarse_w = int(math.ceil(float(width - 1) / ratio)) + 1
        coarse_h = int(math.ceil(float(height - 1) / ratio)) + 1
        coarse = gen.sample_grid(
            ox, oy, coarse_w, coarse_h,
            step=step * ratio, tile_size=0,
            upscale=False, bypass_profile_sub=True,
        )
        return self._upsample_fencepost(coarse, width, height, ratio)

    @staticmethod
    def _upsample_fencepost(
        coarse: NDArray[np.float32],
        width: int,
        height: int,
        ratio: float,
    ) -> NDArray[np.float32]:
        """Bilinearly resamples fencepost grid onto exact target points."""
        src_h, src_w = coarse.shape
        x_pos = np.arange(width, dtype=np.float32) / np.float32(ratio)
        x0 = np.clip(np.floor(x_pos).astype(np.int32), 0, src_w - 2)
        wx = (x_pos - x0)[None, :]
        horiz = coarse[:, x0] + ((coarse[:, x0 + 1] - coarse[:, x0]) * wx)

        y_pos = np.arange(height, dtype=np.float32) / np.float32(ratio)
        y0 = np.clip(np.floor(y_pos).astype(np.int32), 0, src_h - 2)
        wy = (y_pos - y0)[:, None]
        out = horiz[y0, :] + ((horiz[y0 + 1, :] - horiz[y0, :]) * wy)
        return np.ascontiguousarray(out.astype(np.float32))

    def _build_noise_engine(
        self,
        profile_key: str,
        noise_profiles: Dict[str, NoiseProfile],
        seed_offset: int,
    ) -> NoiseGenerator:
        """Constructs NoiseGenerator with additive world seed offset."""
        if profile_key not in noise_profiles:
            fail_fast(
                f"Noise profile '{profile_key}' not found in noise.yaml."
            )

        raw_cfg = noise_profiles[profile_key]
        eff_seed = (self.world_seed + raw_cfg.seed + seed_offset) & 0xFFFFFFFF
        mod_cfg = raw_cfg.clone()
        mod_cfg.seed = eff_seed
        return NoiseGenerator(mod_cfg)
