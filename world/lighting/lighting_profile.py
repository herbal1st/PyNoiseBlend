"""Data model and YAML loader for solar lighting and atmosphere profiles."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

from utils.error_handler import fail_fast


@dataclass(frozen=True, slots=True)
class LightingProfile:
    """Immutable specification holding solar and ambient lighting rules."""

    profile_name: str
    lightmap_sub_sample: int
    neutral_exposure: float
    day_cycle_duration: float
    start_time_ratio: float
    start_azimuth_deg: float
    start_altitude_deg: float
    terrain_steepness: float
    ambient_ratio: float
    shadow_intensity: int
    highlight_intensity: int
    ambient_keyframes: Tuple[Tuple[float, int, int, int], ...]


class LightingProfileLoader:
    """Loads and validates solar lighting presets from lighting.yaml."""

    @classmethod
    def load_profiles(
        cls,
        library_path: Optional[Path] = None,
    ) -> Dict[str, LightingProfile]:
        """Parses lighting YAML configuration into LightingProfile objects."""
        root_dir = Path(__file__).resolve().parents[2]
        cfg_path = library_path or (root_dir / "profiles" / "lighting.yaml")

        if not cfg_path.exists():
            fail_fast(f"Lighting profile file missing: {cfg_path}")

        if yaml is None:
            fail_fast("PyYAML is not installed. Run: pip install pyyaml")

        with open(cfg_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        raw_profiles: Dict[str, Any] = data.get("profiles", {})
        if not raw_profiles:
            fail_fast(f"No lighting profiles defined in {cfg_path}")

        profiles: Dict[str, LightingProfile] = {}
        for name, p_data in raw_profiles.items():
            profiles[name] = cls._parse_profile(name, p_data)

        return profiles

    @classmethod
    def _parse_profile(
        cls,
        name: str,
        p_data: Dict[str, Any],
    ) -> LightingProfile:
        """Extracts and clamps lighting parameters for a single preset."""
        lm_sub = max(1, min(16, int(p_data.get("lightmap_sub_sample", 4))))
        exposure = max(
            0.50, min(1.00, float(p_data.get("neutral_exposure", 0.85)))
        )
        dur = max(0.0, float(p_data.get("day_cycle_duration", 60.0)))
        start_t = max(
            0.0, min(1.0, float(p_data.get("start_time_ratio", 0.30)))
        )
        azim = float(p_data.get("start_azimuth_deg", 90.0)) % 360.0
        alt = max(
            0.0, min(90.0, float(p_data.get("start_altitude_deg", 10.0)))
        )
        steep = max(1.0, float(p_data.get("terrain_steepness", 135.0)))
        amb_ratio = max(
            0.0, min(1.0, float(p_data.get("ambient_ratio", 0.25)))
        )
        sh_int = max(0, min(255, int(p_data.get("shadow_intensity", 170))))
        hl_int = max(0, min(255, int(p_data.get("highlight_intensity", 120))))

        raw_kfs: List[Any] = list(p_data.get("ambient_keyframes", []))
        parsed_kfs: List[Tuple[float, int, int, int]] = []
        for item in raw_kfs:
            if len(item) != 4:
                fail_fast(
                    f"Lighting profile '{name}' keyframe must have 4 items: "
                    f"[time_ratio, R, G, B]."
                )
            t_rat = max(0.0, min(1.0, float(item[0])))
            r_val = max(0, min(255, int(item[1])))
            g_val = max(0, min(255, int(item[2])))
            b_val = max(0, min(255, int(item[3])))
            parsed_kfs.append((t_rat, r_val, g_val, b_val))

        parsed_kfs.sort(key=lambda kf: kf[0])

        return LightingProfile(
            profile_name=name,
            lightmap_sub_sample=lm_sub,
            neutral_exposure=exposure,
            day_cycle_duration=dur,
            start_time_ratio=start_t,
            start_azimuth_deg=azim,
            start_altitude_deg=alt,
            terrain_steepness=steep,
            ambient_ratio=amb_ratio,
            shadow_intensity=sh_int,
            highlight_intensity=hl_int,
            ambient_keyframes=tuple(parsed_kfs),
        )
