"""Application entry point for the PyNoiseBlend simulation engine."""

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from studio.app import run_app


def main() -> None:
    """Initializes runtime configurations and boots the studio window."""
    run_app()


if __name__ == "__main__":
    main()
