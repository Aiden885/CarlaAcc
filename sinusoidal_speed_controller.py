import carla
import math
import time


class SinusoidalSpeedController:
    """
    正弦波速度控制器
    用于控制目标车辆以正弦波模式变化的速度行驶
    """

    def __init__(self, vehicle, base_speed=20.0, amplitude=10.0, period=30.0):
        """
        初始化正弦速度控制器

        参数:
        vehicle - 要控制的Carla车辆对象
        base_speed - 基础速度 (km/h)
        amplitude - 正弦波幅度 (km/h)
        period - 正弦波周期 (秒)
        """
        self.vehicle = vehicle
        self.base_speed = base_speed
        self.amplitude = amplitude
        self.period = period
        self.start_time = time.time()
        self.tm = None  # 交通管理器引用

    def set_traffic_manager(self, tm):
        """设置交通管理器引用"""
        self.tm = tm

    def get_current_desired_speed(self):
        """
        获取当前的期望速度（不实际应用控制）
        返回当前正弦波计算出的期望速度值
        """
        current_time = time.time() - self.start_time
        phase = (2 * math.pi * current_time) / self.period
        desired_speed = self.base_speed + self.amplitude * math.sin(phase)

        # 确保速度为正值
        if desired_speed < 5.0:
            desired_speed = 5.0

        return desired_speed

    def update(self):
        """更新目标车辆的速度"""
        if self.vehicle is None:
            return

        # 计算当前的期望速度
        desired_speed = self.get_current_desired_speed()

        # 方式1：通过交通管理器控制速度（推荐）
        if self.tm:
            # 假设限速是30km/h，计算相对于限速的百分比差
            speed_limit = 50
            percentage_diff = ((speed_limit - desired_speed) / speed_limit) * 100
            self.tm.vehicle_percentage_speed_difference(self.vehicle, percentage_diff)

            print(f"Target vehicle speed set to {desired_speed:.2f} km/h (percentage diff: {percentage_diff:.2f}%)")

        # 方式2：直接控制车辆（备用方案）
        else:
            # 将km/h转换为m/s
            target_speed = desired_speed / 3.6

            # 获取当前车辆速度
            current_velocity = self.vehicle.get_velocity()
            current_speed = math.sqrt(current_velocity.x ** 2 + current_velocity.y ** 2)

            # 获取当前控制状态
            control = self.vehicle.get_control()

            # 简单的比例控制
            if target_speed > current_speed:
                # 需要加速
                control.throttle = min(1.0, (target_speed - current_speed) / 10.0)
                control.brake = 0.0
            else:
                # 需要减速
                control.throttle = 0.0
                control.brake = min(1.0, (current_speed - target_speed) / 5.0)

            # 保持原有的转向控制（由Carla自动驾驶处理）
            # 应用控制
            self.vehicle.apply_control(control)

            print(f"Target vehicle direct speed control: {desired_speed:.2f} km/h")

    def reset_timer(self):
        """重置计时器，从新的时间点开始正弦波"""
        self.start_time = time.time()

    def set_parameters(self, base_speed=None, amplitude=None, period=None):
        """
        动态修改正弦波参数

        参数:
        base_speed - 新的基础速度 (km/h)，None表示不修改
        amplitude - 新的振幅 (km/h)，None表示不修改
        period - 新的周期 (秒)，None表示不修改
        """
        if base_speed is not None:
            self.base_speed = base_speed
        if amplitude is not None:
            self.amplitude = amplitude
        if period is not None:
            self.period = period

        print(f"Updated parameters: base_speed={self.base_speed}, amplitude={self.amplitude}, period={self.period}")

    def get_speed_info(self):
        """
        获取速度控制器的当前状态信息
        返回字典包含当前期望速度、基础速度、振幅、周期等信息
        """
        current_time = time.time() - self.start_time
        phase = (2 * math.pi * current_time) / self.period

        return {
            'current_desired_speed': self.get_current_desired_speed(),
            'base_speed': self.base_speed,
            'amplitude': self.amplitude,
            'period': self.period,
            'elapsed_time': current_time,
            'phase': phase,
            'phase_degrees': math.degrees(phase) % 360
        }