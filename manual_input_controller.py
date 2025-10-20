"""Utilities for handling manual driver inputs with time-based integration."""


class ManualSteeringController:
    """Keyboard-based steering integrator that emulates CARLA's manual control."""

    def __init__(self, max_abs_steer=0.7, steer_speed_per_sec=0.5):
        """
        Args:
            max_abs_steer (float): Maximum absolute steering value that can be accumulated.
            steer_speed_per_sec (float): Rate at which steering grows per second when a key is held.
        """
        self._steer_cache = 0.0
        self.max_abs_steer = float(max_abs_steer)
        self.steer_speed_per_sec = float(steer_speed_per_sec)

    def update(self, steer_left: bool, steer_right: bool, dt_seconds: float) -> float:
        """Update steering cache based on key states and elapsed time.

        Args:
            steer_left: Whether the left steering key is pressed.
            steer_right: Whether the right steering key is pressed.
            dt_seconds: Elapsed time since last update in seconds.

        Returns:
            float: Clamped steering value in [-max_abs_steer, max_abs_steer].
        """
        dt = max(0.0, float(dt_seconds))
        increment = self.steer_speed_per_sec * dt

        if steer_left and not steer_right:
            if self._steer_cache > 0.0:
                self._steer_cache = 0.0
            self._steer_cache -= increment
        elif steer_right and not steer_left:
            if self._steer_cache < 0.0:
                self._steer_cache = 0.0
            self._steer_cache += increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = max(-self.max_abs_steer, min(self.max_abs_steer, self._steer_cache))
        return self._steer_cache

    def reset(self) -> None:
        """Reset the cached steering value."""
        self._steer_cache = 0.0
