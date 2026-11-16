"""World time-of-day clock tracking normalized orbit and pause state."""


class DayNightClock:
    """Tracks elapsed time-of-day ratio with adjustable speed and pause."""

    def __init__(
        self,
        day_cycle_duration: float = 60.0,
        start_time_ratio: float = 0.30,
        time_scale: float = 1.0,
    ) -> None:
        """Initializes cycle duration, starting ratio, and speed scale."""
        self.day_cycle_duration: float = max(0.0, day_cycle_duration)
        self.time_scale: float = max(0.0, time_scale)
        self.is_paused: bool = self.day_cycle_duration <= 0.0
        clamped_start = max(0.0, min(1.0, start_time_ratio))
        self.elapsed_time: float = (
            clamped_start * self.day_cycle_duration
            if self.day_cycle_duration > 0.0
            else 0.0
        )
        self._static_time_ratio: float = clamped_start

    @property
    def normalized_time(self) -> float:
        """Returns time-of-day ratio in range [0.0, 1.0)."""
        if self.day_cycle_duration <= 0.0:
            return self._static_time_ratio
        return (
            self.elapsed_time % self.day_cycle_duration
        ) / self.day_cycle_duration

    def set_normalized_time(self, ratio: float) -> None:
        """Jumps clock to a specific normalized time-of-day [0.0, 1.0)."""
        wrapped = ratio % 1.0
        self._static_time_ratio = wrapped
        if self.day_cycle_duration > 0.0:
            self.elapsed_time = wrapped * self.day_cycle_duration

    def toggle_pause(self) -> None:
        """Toggles orbit clock progression on and off."""
        self.is_paused = not self.is_paused

    def update(self, dt: float) -> None:
        """Advances elapsed clock time by delta time scaled by time_scale."""
        if self.is_paused or dt <= 0.0 or self.day_cycle_duration <= 0.0:
            return
        self.elapsed_time += dt * self.time_scale
