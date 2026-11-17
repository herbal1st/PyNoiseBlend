"""Multi-octave fractal noise coordinator generating float32 grids."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import yaml

import config
from utils.error_handler import fail_fast
from utils.noise.perlin import PerlinNoise
from utils.noise.simplex import SimplexNoise
from utils.noise.value import ValueNoise
from utils.resampler import bilinear_resample_2d

NOISE_TYPES: Tuple[str, ...] = ("SIMPLEX", "PERLIN", "VALUE")
SUB_SAMPLE_FACTORS: Tuple[int, ...] = (1, 2, 4, 8)


@dataclass(slots=True)
class NoiseProfile:
    """Mutable specification model holding fractal noise parameters."""

    profile_name: str
    noise_type: str
    seed: int
    scale: float
    octaves: int
    decay: float
    lacunarity: float
    sub_sample_factor: int

    def clone(self) -> "NoiseProfile":
        """Produces an independent duplicate of this configuration."""
        return NoiseProfile(
            profile_name=self.profile_name,
            noise_type=self.noise_type,
            seed=self.seed,
            scale=self.scale,
            octaves=self.octaves,
            decay=self.decay,
            lacunarity=self.lacunarity,
            sub_sample_factor=self.sub_sample_factor,
        )


class NoiseGenerator:
    """Coordinates multi-octave FBM accumulation and grid sampling."""

    def __init__(self, profile: NoiseProfile) -> None:
        """Binds cloned profile configuration and instantiates engine."""
        self.profile: NoiseProfile = profile.clone()
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.engine: Union[SimplexNoise, PerlinNoise, ValueNoise]
        self._build_engine()

    def set_scale(self, scale: float) -> None:
        """Updates feature scale in place without engine reset."""
        clamped = max(
            config.NOISE_MIN_SCALE, min(config.NOISE_MAX_SCALE, scale)
        )
        self.profile.scale = clamped

    def set_offset(self, offset_x: float, offset_y: float) -> None:
        """Updates spatial world translation offset in place."""
        self.offset_x = offset_x
        self.offset_y = offset_y

    def set_octaves(self, octaves: int) -> None:
        """Updates octave count in place clamped within safe bounds."""
        clamped = max(config.OCTAVE_MIN, min(config.OCTAVE_MAX, octaves))
        self.profile.octaves = clamped

    def set_decay(self, decay: float) -> None:
        """Updates fractal persistence decay factor in place."""
        clamped = max(config.DECAY_MIN, min(config.DECAY_MAX, decay))
        self.profile.decay = clamped

    def set_lacunarity(self, lacunarity: float) -> None:
        """Updates fractal lacunarity frequency multiplier in place."""
        clamped = max(
            config.LACUNARITY_MIN, min(config.LACUNARITY_MAX, lacunarity)
        )
        self.profile.lacunarity = clamped

    def set_seed(self, seed: int) -> None:
        """Updates integer randomization seed and rebuilds engine."""
        self.profile.seed = int(seed) & 0xFFFFFFFF
        self._build_engine()

    def set_noise_type(self, noise_type: str) -> None:
        """Updates noise algorithm type and rebuilds engine."""
        normalized = noise_type.upper()
        if normalized not in NOISE_TYPES:
            return
        self.profile.noise_type = normalized
        self._build_engine()

    def set_sub_sample_factor(self, factor: int) -> None:
        """Updates coarse sampling divider in place."""
        if factor in SUB_SAMPLE_FACTORS:
            self.profile.sub_sample_factor = factor

    def sync_from_profile(
        self,
        prof: NoiseProfile,
        seed_override: Optional[int] = None,
        sync_sub_sample: bool = True,
    ) -> None:
        """Synchronizes fractal parameters and engine state from profile."""
        self.set_scale(prof.scale)
        self.set_octaves(prof.octaves)
        self.set_decay(prof.decay)
        self.set_lacunarity(prof.lacunarity)
        if sync_sub_sample:
            self.set_sub_sample_factor(prof.sub_sample_factor)

        target_seed = (
            int(seed_override) & 0xFFFFFFFF
            if seed_override is not None
            else prof.seed
        )
        if self.profile.seed != target_seed:
            self.set_seed(target_seed)

        if self.profile.noise_type != prof.noise_type:
            self.set_noise_type(prof.noise_type)

    def sample_grid(
        self,
        origin_x: float,
        origin_y: float,
        width: int,
        height: int,
        step: float = 1.0,
        tile_size: int = 0,
        sub_sample_override: Optional[int] = None,
        upscale: bool = True,
        bypass_profile_sub: bool = False,
    ) -> NDArray[np.float32]:
        """Samples continuous or coarse float32 noise field in [0.0, 1.0]."""
        if tile_size > 1:
            return self._sample_tiled_grid(
                origin_x=origin_x,
                origin_y=origin_y,
                width=width,
                height=height,
                tile_size=tile_size,
                upscale=upscale,
            )

        if bypass_profile_sub:
            sub_factor = 1
        else:
            base_sub = self.profile.sub_sample_factor
            raw_sub = (
                max(base_sub, sub_sample_override)
                if sub_sample_override is not None
                else base_sub
            )
            sub_factor = max(1, raw_sub)

        return self._sample_smooth_grid(
            origin_x=origin_x,
            origin_y=origin_y,
            width=width,
            height=height,
            step=step,
            sub_factor=sub_factor,
            upscale=upscale,
        )

    @classmethod
    def load_profiles(
        cls,
        library_path: Optional[Path] = None,
    ) -> Dict[str, NoiseProfile]:
        """Parses YAML configuration into dictionary of NoiseProfiles."""
        root_dir = Path(__file__).resolve().parents[2]
        cfg_path = library_path or (root_dir / "profiles" / "noise.yaml")

        if not cfg_path.exists():
            fail_fast(f"Noise profile file missing: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        raw_profiles: Dict[str, Any] = data.get("profiles", {})
        profiles: Dict[str, NoiseProfile] = {}

        for p_name, p_data in raw_profiles.items():
            profiles[p_name] = NoiseProfile(
                profile_name=p_name,
                noise_type=str(p_data.get("noise_type", "SIMPLEX")),
                seed=int(p_data.get("seed", 420)),
                scale=float(p_data.get("scale", 100.0)),
                octaves=int(p_data.get("octaves", 1)),
                decay=float(p_data.get("decay", 0.5)),
                lacunarity=float(p_data.get("lacunarity", 2.0)),
                sub_sample_factor=int(p_data.get("sub_sample_factor", 1)),
            )

        return profiles

    def _build_engine(self) -> None:
        """Instantiates underlying noise engine matching active type/seed."""
        n_type: str = self.profile.noise_type.upper()
        if n_type == "PERLIN":
            self.engine = PerlinNoise(self.profile.seed)
        elif n_type == "VALUE":
            self.engine = ValueNoise(self.profile.seed)
        else:
            self.engine = SimplexNoise(self.profile.seed)

    def _sample_tiled_grid(
        self,
        origin_x: float,
        origin_y: float,
        width: int,
        height: int,
        tile_size: int,
        upscale: bool,
    ) -> NDArray[np.float32]:
        """Evaluates world-snapped tile centroids with optional expansion."""
        tile_sz_f = float(tile_size)
        safe_w = max(1, width)
        safe_h = max(1, height)

        tx_start = math.floor(origin_x / tile_sz_f)
        ty_start = math.floor(origin_y / tile_sz_f)
        tx_end = math.floor((origin_x + float(safe_w) - 1.0) / tile_sz_f)
        ty_end = math.floor((origin_y + float(safe_h) - 1.0) / tile_sz_f)

        snapped_x = (
            np.arange(tx_start, tx_end + 1, dtype=np.float32) + 0.5
        ) * tile_sz_f
        snapped_y = (
            np.arange(ty_start, ty_end + 1, dtype=np.float32) + 0.5
        ) * tile_sz_f
        x_mesh, y_mesh = np.meshgrid(snapped_x, snapped_y)

        coarse_field = self._evaluate_mesh(x_mesh, y_mesh)
        if not upscale:
            return np.ascontiguousarray(coarse_field)

        shift_x = max(
            0, int(round(origin_x - (float(tx_start) * tile_sz_f)))
        )
        shift_y = max(
            0, int(round(origin_y - (float(ty_start) * tile_sz_f)))
        )
        expanded_y = np.repeat(coarse_field, tile_size, axis=0)
        expanded_xy = np.repeat(expanded_y, tile_size, axis=1)
        sliced = expanded_xy[
            shift_y : shift_y + safe_h, shift_x : shift_x + safe_w
        ].astype(np.float32)
        return np.ascontiguousarray(sliced)

    def _sample_smooth_grid(
        self,
        origin_x: float,
        origin_y: float,
        width: int,
        height: int,
        step: float,
        sub_factor: int,
        upscale: bool,
    ) -> NDArray[np.float32]:
        """Evaluates (N+1) fencepost or native grid with resample."""
        if sub_factor > 1:
            eval_w = (width // sub_factor) + 1
            eval_h = (height // sub_factor) + 1
            eval_step = step * float(sub_factor)
        else:
            eval_w = width
            eval_h = height
            eval_step = step

        x_coords = origin_x + (
            np.arange(eval_w, dtype=np.float32) * eval_step
        )
        y_coords = origin_y + (
            np.arange(eval_h, dtype=np.float32) * eval_step
        )
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

        clipped = self._evaluate_mesh(x_mesh, y_mesh)
        if sub_factor > 1 and upscale:
            resampled = bilinear_resample_2d(clipped, height, width)
            return np.ascontiguousarray(resampled)

        return np.ascontiguousarray(clipped)

    def _evaluate_mesh(
        self,
        x_mesh: NDArray[np.float32],
        y_mesh: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Evaluates multi-octave FBM field across coordinate mesh."""
        sx = x_mesh - self.offset_x if self.offset_x != 0.0 else x_mesh
        sy = y_mesh - self.offset_y if self.offset_y != 0.0 else y_mesh

        eval_h, eval_w = sx.shape
        result = np.zeros((eval_h, eval_w), dtype=np.float32)
        total_amp: float = 0.0
        freq: float = 1.0 / max(0.001, self.profile.scale)
        amp: float = 1.0

        for _ in range(max(1, self.profile.octaves)):
            layer = self.engine.evaluate(sx * freq, sy * freq)
            result += layer * amp
            total_amp += amp
            freq *= self.profile.lacunarity
            amp *= self.profile.decay

        if total_amp > 0.0:
            result /= total_amp

        normalized = (result + 1.0) * 0.5
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)
