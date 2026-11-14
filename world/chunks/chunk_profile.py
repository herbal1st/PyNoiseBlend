"""
Data model and loader for spatial chunk streaming and sizing configurations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import sys

try:
    import yaml
except ImportError:
    yaml = None

from utils.error_handler import fail_fast


@dataclass(frozen=True, slots=True)
class ChunkProfile:
    """
    Immutable data model holding chunk dimensions and streaming margins.
    """

    profile_name: str
    chunk_size: int
    load_margin: float
    unload_margin: float


class ChunkProfileLoader:
    """
    Loads and validates spatial chunk profiles from chunk.yaml.
    """

    @classmethod
    def load_profiles(
        cls,
        library_path: Optional[Path] = None,
    ) -> Dict[str, ChunkProfile]:
        """
        Parses chunk YAML configuration and enforces power-of-2 dimensions.
        """
        root_dir = Path(__file__).resolve().parents[2]
        cfg_path = library_path or (root_dir / "profiles" / "chunk.yaml")

        if not cfg_path.exists():
            fail_fast(f"Chunk profile file missing: {cfg_path}")

        if yaml is None:
            fail_fast("PyYAML is not installed. Run: pip install pyyaml")

        with open(cfg_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        raw_profiles: Dict[str, Any] = data.get("profiles", {})
        profiles: Dict[str, ChunkProfile] = {}

        for name, p_data in raw_profiles.items():
            size = int(p_data.get("chunk_size", 128))
            is_pot = (size > 0) and (size & (size - 1) == 0)
            if not is_pot:
                fail_fast(
                    f"Chunk profile '{name}' chunk_size ({size}) must be "
                    f"a power of 2 (e.g. 64, 128, 256)."
                )

            load_m = float(p_data.get("load_margin", 1.0))
            unload_m = float(p_data.get("unload_margin", 1.75))

            profiles[name] = ChunkProfile(
                profile_name=name,
                chunk_size=size,
                load_margin=load_m,
                unload_margin=unload_m,
            )

        return profiles
