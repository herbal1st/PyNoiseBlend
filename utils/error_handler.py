"""
Console error handling utilities providing graceful crash pauses on exit.
"""

import sys
import traceback
from types import TracebackType
from typing import Optional, Type


def fail_fast(message: str, exit_code: int = 1) -> None:
    """
    Prints a formatted fatal error, pauses for user input, and terminates.
    """
    print(f"\n[Fatal Error] {message}")
    _wait_for_user_exit()
    sys.exit(exit_code)


def install_crash_handler() -> None:
    """
    Attaches a global exception hook to prevent console auto-closing on crash.
    """
    sys.excepthook = _unhandled_exception_hook


def _unhandled_exception_hook(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_tb: Optional[TracebackType],
) -> None:
    """
    Intercepts uncaught exceptions, prints traceback, and pauses console.
    """
    print("\n[Application Crash - Unhandled Exception]")
    traceback.print_exception(exc_type, exc_value, exc_tb)
    _wait_for_user_exit()
    sys.exit(1)


def _wait_for_user_exit() -> None:
    """
    Prompts user to press Enter, handling non-interactive environments safely.
    """
    try:
        input("\nPress Enter to close this window...")
    except (EOFError, OSError):
        pass
