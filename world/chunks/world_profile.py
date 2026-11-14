"""Data model and YAML loader for composite world recipe configurations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None

from utils.error_handler import fail_fast


@dataclass(frozen=True, slots=True)
class WorldProfile:
    """Immutable recipe linking noise layers into a world build."""

    profile_name: str
    world_seed: int
    sub_sample_factor: int
    base_noise_profile: str
    macro_noise_profile: str
    crust_noise_profile: str


class WorldProfileLoader:
    """Loads and validates world noise-engine recipes from world.yaml."""

    @classmethod
    def load_profiles(
        cls,
        library_path: Optional[Path] = None,
    ) -> Dict[str, WorldProfile]:
        """Parses world builds connecting procedural noise layers."""
        root_dir = Path(__file__).resolve().parents[2]
        cfg_path = library_path or (root_dir / "profiles" / "world.yaml")

        if not cfg_path.exists():
            fail_fast(f"World profile file missing: {cfg_path}")

        if yaml is None:
            fail_fast("PyYAML is not installed. Run: pip install pyyaml")

        with open(cfg_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        raw_profiles: Dict[str, Any] = data.get("profiles", {})
        profiles: Dict[str, WorldProfile] = {}

        for p_name, p_data in raw_profiles.items():
            profiles[p_name] = WorldProfile(
                profile_name=p_name,
                world_seed=int(p_data.get("world_seed", 420)),
                sub_sample_factor=int(p_data.get("sub_sample_factor", 1)),
                base_noise_profile=str(
                    p_data.get("base_noise_profile", "MICRO_SIMPLEX")
                ),
                macro_noise_profile=str(
                    p_data.get("macro_noise_profile", "MACRO_VALUE")
                ),
                crust_noise_profile=str(
                    p_data.get("crust_noise_profile", "MACRO_PERLIN")
                ),
            )

        return profiles
