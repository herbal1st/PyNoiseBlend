"""Parser and compiler generating vectorized 1D LUTs from strata profiles."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

try:
    import yaml
except ImportError:
    yaml = None

from utils.error_handler import fail_fast
from world.strata.strata_keyframe import StrataKeyframe
from world.strata.strata_profile import StrataProfile


class StrataLoader:
    """Parses strata YAML profiles and bakes 1,024-slot LUT arrays."""

    @classmethod
    def load_profiles(
        cls,
        library_path: Optional[Path] = None,
    ) -> Dict[str, StrataProfile]:
        """Loads diagnostic heatmaps and all strata profiles from YAML."""
        profiles: Dict[str, StrataProfile] = cls._build_builtin_heatmaps()

        root_dir = Path(__file__).resolve().parents[2]
        cfg_path = library_path or (root_dir / "profiles" / "strata.yaml")

        if not cfg_path.exists():
            return profiles

        if yaml is None:
            fail_fast("PyYAML is not installed. Run: pip install pyyaml")

        with open(cfg_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        raw_profiles: Dict[str, Any] = data.get("profiles", {})
        for name, profile_data in raw_profiles.items():
            profiles[name] = cls._compile_profile(name, profile_data)

        return profiles

    @classmethod
    def _compile_profile(
        cls,
        name: str,
        profile_data: Dict[str, Any],
    ) -> StrataProfile:
        """Validates keyframes and constructs immutable StrataProfile."""
        raw_keyframes = profile_data.get("keyframes", [])
        if not raw_keyframes:
            fail_fast(f"Strata profile '{name}' has no defined keyframes.")

        keyframes = cls._normalize_keyframes(name, raw_keyframes)
        return cls._create_profile(name, keyframes)

    @classmethod
    def _create_profile(
        cls,
        name: str,
        keyframes: List[StrataKeyframe],
    ) -> StrataProfile:
        """Bakes normal, inverted, and speed LUTs into a StrataProfile."""
        color_lut, speed_lut = cls._bake_luts(keyframes)
        inverted_lut = np.ascontiguousarray(color_lut[::-1])
        return StrataProfile(
            profile_name=name,
            keyframes=tuple(keyframes),
            color_lut=color_lut,
            inverted_lut=inverted_lut,
            speed_lut=speed_lut,
        )

    @classmethod
    def _normalize_keyframes(
        cls,
        name: str,
        raw_keyframes: List[Any],
    ) -> List[StrataKeyframe]:
        """Validates keyframe thresholds and bounds within 0.0 and 1.0."""
        parsed: List[StrataKeyframe] = []
        last_elevation: float = -1.0

        for idx, item in enumerate(raw_keyframes):
            if len(item) != 3:
                fail_fast(
                    f"Profile '{name}' keyframe #{idx} must have 3 entries: "
                    f"[elevation, speed, [R, G, B]]."
                )

            elev = float(item[0])
            speed = float(item[1])
            rgb_raw = item[2]

            if elev < last_elevation:
                fail_fast(
                    f"Profile '{name}' keyframes must be non-decreasing. "
                    f"Found {elev:.3f} after {last_elevation:.3f}."
                )
            last_elevation = elev

            if speed < 0.0:
                fail_fast(
                    f"Profile '{name}' speed ({speed}) must be >= 0."
                )

            if len(rgb_raw) != 3:
                fail_fast(
                    f"Profile '{name}' keyframe #{idx} RGB must have "
                    f"3 items."
                )

            r = int(np.clip(rgb_raw[0], 0, 255))
            g = int(np.clip(rgb_raw[1], 0, 255))
            b = int(np.clip(rgb_raw[2], 0, 255))

            parsed.append(
                StrataKeyframe(
                    elevation=elev,
                    speed_multiplier=speed,
                    color=(r, g, b),
                )
            )

        if parsed[0].elevation > 0.0:
            parsed.insert(
                0,
                StrataKeyframe(
                    elevation=0.0,
                    speed_multiplier=parsed[0].speed_multiplier,
                    color=parsed[0].color,
                ),
            )

        if parsed[-1].elevation < 1.0:
            parsed.append(
                StrataKeyframe(
                    elevation=1.0,
                    speed_multiplier=parsed[-1].speed_multiplier,
                    color=parsed[-1].color,
                )
            )

        return parsed

    @classmethod
    def _bake_luts(
        cls,
        keyframes: List[StrataKeyframe],
    ) -> Tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """Linearly interpolates RGB and Speed values across 1,024 slots."""
        xp = np.array([kf.elevation for kf in keyframes], dtype=np.float32)
        r_pts = np.array([kf.color[0] for kf in keyframes], dtype=np.float32)
        g_pts = np.array([kf.color[1] for kf in keyframes], dtype=np.float32)
        b_pts = np.array([kf.color[2] for kf in keyframes], dtype=np.float32)
        s_pts = np.array(
            [kf.speed_multiplier for kf in keyframes], dtype=np.float32
        )

        sample_points = np.linspace(0.0, 1.0, 1024, dtype=np.float32)

        r_interp = np.interp(sample_points, xp, r_pts)
        g_interp = np.interp(sample_points, xp, g_pts)
        b_interp = np.interp(sample_points, xp, b_pts)
        speed_interp = np.interp(sample_points, xp, s_pts)

        r_lut = np.clip(np.round(r_interp), 0, 255).astype(np.uint8)
        g_lut = np.clip(np.round(g_interp), 0, 255).astype(np.uint8)
        b_lut = np.clip(np.round(b_interp), 0, 255).astype(np.uint8)

        color_lut = np.stack([r_lut, g_lut, b_lut], axis=-1)
        speed_lut = speed_interp.astype(np.float32)

        return np.ascontiguousarray(color_lut), np.ascontiguousarray(speed_lut)

    @classmethod
    def _build_builtin_heatmaps(cls) -> Dict[str, StrataProfile]:
        """Pre-bakes diagnostic heatmaps as baseline StrataProfiles."""
        heatmaps: Dict[str, StrataProfile] = {}

        grey_kfs = [
            StrataKeyframe(0.0, 1.0, (0, 0, 0)),
            StrataKeyframe(1.0, 1.0, (255, 255, 255)),
        ]
        heatmaps["GREYSCALE"] = cls._create_profile("GREYSCALE", grey_kfs)

        magma_kfs = [
            StrataKeyframe(0.00, 1.0, (0, 0, 4)),
            StrataKeyframe(0.20, 1.0, (40, 11, 84)),
            StrataKeyframe(0.40, 1.0, (101, 21, 110)),
            StrataKeyframe(0.60, 1.0, (182, 54, 121)),
            StrataKeyframe(0.80, 1.0, (251, 136, 97)),
            StrataKeyframe(0.95, 1.0, (254, 216, 154)),
            StrataKeyframe(1.00, 1.0, (252, 253, 191)),
        ]
        heatmaps["MAGMA"] = cls._create_profile("MAGMA", magma_kfs)

        turbo_kfs = [
            StrataKeyframe(0.00, 1.0, (48, 18, 59)),
            StrataKeyframe(0.15, 1.0, (70, 134, 251)),
            StrataKeyframe(0.35, 1.0, (27, 229, 181)),
            StrataKeyframe(0.55, 1.0, (164, 252, 60)),
            StrataKeyframe(0.75, 1.0, (251, 185, 56)),
            StrataKeyframe(0.90, 1.0, (227, 73, 24)),
            StrataKeyframe(1.00, 1.0, (122, 4, 3)),
        ]
        heatmaps["TURBO"] = cls._create_profile("TURBO", turbo_kfs)

        viridis_kfs = [
            StrataKeyframe(0.00, 1.0, (68, 1, 84)),
            StrataKeyframe(0.25, 1.0, (59, 82, 139)),
            StrataKeyframe(0.50, 1.0, (33, 145, 140)),
            StrataKeyframe(0.75, 1.0, (94, 201, 98)),
            StrataKeyframe(1.00, 1.0, (253, 231, 37)),
        ]
        heatmaps["VIRIDIS"] = cls._create_profile("VIRIDIS", viridis_kfs)

        return heatmaps
