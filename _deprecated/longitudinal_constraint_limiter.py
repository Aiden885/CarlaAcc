"""
Longitudinal constraint limiter for ACC torque commands.
Applies national-standard limits using sliding-window averages.
"""
from collections import deque
import math


class LongitudinalConstraintLimiter:
    def __init__(self, config, torque_converter):
        self.enabled = bool(getattr(config, "enable_longitudinal_constraints", False))
        self.dt = float(getattr(config, "fixed_delta_seconds", 0.05) or 0.05)

        self.a_max = float(getattr(config, "longitudinal_a_max_ms2", 2.0))
        self.decel_avg_max = float(getattr(config, "longitudinal_decel_avg_max_ms2", 3.0))
        self.decel_avg_window_s = float(getattr(config, "longitudinal_decel_avg_window_s", 2.0))
        self.accel_avg_max = float(getattr(config, "longitudinal_accel_avg_max_ms2", self.a_max))
        self.accel_avg_window_s = float(getattr(config, "longitudinal_accel_avg_window_s", 2.0))
        self.accel_soft_alpha = float(getattr(config, "longitudinal_accel_soft_alpha", 0.2))
        self.jerk_avg_max = float(getattr(config, "longitudinal_decel_jerk_avg_max_ms3", 2.5))
        self.jerk_avg_window_s = float(getattr(config, "longitudinal_decel_jerk_avg_window_s", 1.0))

        self.c_rr = float(getattr(config, "rolling_resistance_coeff", 0.012))
        self.cdA = float(getattr(config, "aero_cdA", 0.74))
        self.rho = float(getattr(config, "air_density_kg_m3", 1.225))
        self.driveline_efficiency = float(getattr(config, "driveline_efficiency", 1.0))
        if self.driveline_efficiency <= 0.0:
            self.driveline_efficiency = 1.0
        self.effective_mass_factor = float(
            getattr(config, "longitudinal_effective_mass_factor", 1.0)
        )
        if self.effective_mass_factor <= 0.0:
            self.effective_mass_factor = 1.0

        self.mass = float(getattr(torque_converter, "vehicle_mass", 0.0))
        self.wheel_radius = float(getattr(torque_converter, "wheel_radius", 0.0))
        self.gear_ratio = float(getattr(torque_converter, "total_gear_ratio", 1.0))
        if self.gear_ratio <= 0.0:
            self.gear_ratio = 1.0

        self.decel_window = deque(maxlen=max(1, int(round(self.decel_avg_window_s / self.dt))))
        self.accel_window = deque(maxlen=max(1, int(round(self.accel_avg_window_s / self.dt))))
        self.jerk_window = deque(maxlen=max(1, int(round(self.jerk_avg_window_s / self.dt))))

        self.prev_speed_ms = None
        self.prev_accel = None
        self.prev_limited_torque = 0.0

    def reset(self):
        self.prev_speed_ms = None
        self.prev_accel = None
        self.prev_limited_torque = 0.0
        self.decel_window.clear()
        self.accel_window.clear()
        self.jerk_window.clear()

    @staticmethod
    def _mean_non_none(values):
        total = 0.0
        count = 0
        for value in values:
            if value is None:
                continue
            total += value
            count += 1
        return total / count if count else 0.0

    def _resist_force(self, speed_ms):
        g = 9.80665
        f_roll = self.c_rr * self.mass * g
        f_drag = 0.5 * self.rho * self.cdA * speed_ms * speed_ms
        return f_roll + f_drag

    def limit(self, torque_cmd, speed_ms, accel_ms2=None):
        if not self.enabled:
            return torque_cmd

        use_accel = None
        if accel_ms2 is not None and math.isfinite(accel_ms2):
            use_accel = float(accel_ms2)

        if self.prev_speed_ms is None:
            self.prev_speed_ms = speed_ms
            self.prev_accel = use_accel if use_accel is not None else 0.0
            self.prev_limited_torque = torque_cmd
            return torque_cmd

        if use_accel is not None:
            acc = use_accel
        else:
            acc = (speed_ms - self.prev_speed_ms) / self.dt
        jerk = 0.0
        if self.prev_accel is not None:
            jerk = (acc - self.prev_accel) / self.dt

        self.prev_speed_ms = speed_ms
        self.prev_accel = acc

        if acc < 0.0:
            self.decel_window.append(-acc)
            self.jerk_window.append(abs(jerk))
            self.accel_window.append(None)
        else:
            self.decel_window.append(None)
            self.jerk_window.append(None)
            if acc > 0.0:
                self.accel_window.append(acc)
            else:
                self.accel_window.append(None)

        avg_decel = self._mean_non_none(self.decel_window)
        avg_accel = self._mean_non_none(self.accel_window)
        avg_jerk = self._mean_non_none(self.jerk_window)

        resist = self._resist_force(speed_ms)
        torque_cmd_limited = torque_cmd

        # Acceleration limit - feedback-based with windowed average.
        # 只在加速度平均值超阈值时限幅，允许短时超调。
        if avg_accel > self.accel_avg_max and torque_cmd > 0:
            effective_mass = self.mass * self.effective_mass_factor
            force_max = effective_mass * self.a_max + resist
            torque_max = force_max * self.wheel_radius / (self.gear_ratio * self.driveline_efficiency)
            if torque_cmd_limited > torque_max:
                alpha = max(0.0, min(self.accel_soft_alpha, 1.0))
                torque_cmd_limited = torque_cmd_limited + alpha * (torque_max - torque_cmd_limited)

        # Average deceleration limit (braking samples only).
        if torque_cmd_limited < 0.0 and avg_decel > self.decel_avg_max:
            brake_force_max = max(0.0, self.mass * self.decel_avg_max - resist)
            brake_torque_max = brake_force_max * self.wheel_radius / (self.gear_ratio * self.driveline_efficiency)
            torque_cmd_limited = max(torque_cmd_limited, -brake_torque_max)

        # Average jerk limit during braking.
        if torque_cmd_limited < 0.0 and avg_jerk > self.jerk_avg_max:
            torque_delta_max = (
                self.mass * self.jerk_avg_max * self.wheel_radius /
                (self.gear_ratio * self.driveline_efficiency)
            ) * self.dt
            min_torque = self.prev_limited_torque - torque_delta_max
            if torque_cmd_limited < min_torque:
                torque_cmd_limited = min_torque

        self.prev_limited_torque = torque_cmd_limited
        return torque_cmd_limited
