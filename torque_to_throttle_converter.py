#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车轮扭矩到油门转换器
将SPPVT输出的车轮扭矩转换为CARLA的油门/刹车控制值
"""

import math


class TorqueToThrottleConverter:
    """
    车轮扭矩到油门转换器

    假设SPPVT输出的是期望的车轮扭矩（N·m），需要转换为油门值（0-1）
    """

    def __init__(self, vehicle):
        """
        初始化转换器

        参数:
        vehicle - CARLA车辆对象
        """
        self.vehicle = vehicle

        # 从车辆获取物理参数
        self._load_vehicle_parameters()

        # 初始化PID控制器（用于精确控制）
        self.pid_enabled = True
        self.pid_kp = 0.1
        self.pid_ki = 0.01
        self.pid_kd = 0.05
        self.integral_error = 0.0
        self.prev_error = 0.0

        print(f"TorqueToThrottleConverter 初始化完成")
        print(f"  车辆质量: {self.mass} kg")
        print(f"  车轮半径: {self.wheel_radius} m")
        print(f"  最大发动机扭矩: {self.max_engine_torque} N·m")
        print(f"  传动比: {self.total_gear_ratio}")

    def _load_vehicle_parameters(self):
        """从CARLA车辆加载物理参数"""
        physics = self.vehicle.get_physics_control()

        # 基本参数
        self.mass = physics.mass  # kg
        self.drag_coefficient = physics.drag_coefficient

        # 车轮参数（假设4个车轮相同，取第一个）
        self.wheel_radius = physics.wheels[0].radius / 100.0  # 转换cm到m
        self.num_drive_wheels = 4  # 四轮驱动（Audi e-tron）

        # 发动机参数
        self.max_engine_torque = max([p.y for p in physics.torque_curve])

        # 传动系统参数
        self.gear_ratio = 1.0  # 电动车单速
        self.final_ratio = physics.final_ratio
        self.drivetrain_efficiency = 0.90  # 假设90%效率

        # 计算总传动比
        self.total_gear_ratio = self.gear_ratio * self.final_ratio * self.drivetrain_efficiency

        # 空气阻力相关（需要估算迎风面积）
        # 对于Audi e-tron: 长4.9m × 宽2.0m × 高1.6m
        # 估算迎风面积 A ≈ 2.0 × 1.6 = 3.2 m²
        self.frontal_area = 3.2  # m²
        self.air_density = 1.225  # kg/m³ (标准大气压)

        # 滚动阻力系数
        self.rolling_resistance = 0.015  # 典型值
        self.gravity = 9.81  # m/s²

    def wheel_torque_to_acceleration(self, total_wheel_torque, current_speed_ms):
        """
        从总车轮扭矩计算期望加速度

        参数:
        total_wheel_torque - 总车轮扭矩 (N·m)，所有驱动轮的总和
        current_speed_ms - 当前车速 (m/s)

        返回:
        期望加速度 (m/s²)
        """
        # 1. 从车轮扭矩计算驱动力
        drive_force = total_wheel_torque / self.wheel_radius

        # 2. 计算阻力
        # 空气阻力: F_drag = 0.5 × ρ × C_d × A × v²
        air_drag = 0.5 * self.air_density * self.drag_coefficient * \
                   self.frontal_area * (current_speed_ms ** 2)

        # 滚动阻力: F_roll = μ_roll × m × g
        rolling_drag = self.rolling_resistance * self.mass * self.gravity

        # 3. 计算净力和加速度
        net_force = drive_force - air_drag - rolling_drag
        acceleration = net_force / self.mass

        return acceleration

    def acceleration_to_engine_torque(self, desired_accel, current_speed_ms):
        """
        从期望加速度反算发动机扭矩

        参数:
        desired_accel - 期望加速度 (m/s²)
        current_speed_ms - 当前车速 (m/s)

        返回:
        需要的发动机扭矩 (N·m)
        """
        # 1. 计算需要的净力
        required_net_force = desired_accel * self.mass

        # 2. 计算阻力
        air_drag = 0.5 * self.air_density * self.drag_coefficient * \
                   self.frontal_area * (current_speed_ms ** 2)
        rolling_drag = self.rolling_resistance * self.mass * self.gravity

        # 3. 计算需要的驱动力
        required_drive_force = required_net_force + air_drag + rolling_drag

        # 4. 从驱动力计算需要的车轮扭矩
        required_wheel_torque = required_drive_force * self.wheel_radius

        # 5. 从车轮扭矩反算发动机扭矩
        # T_wheel = T_engine × gear_ratio × final_ratio × efficiency
        # T_engine = T_wheel / (gear_ratio × final_ratio × efficiency)
        required_engine_torque = required_wheel_torque / self.total_gear_ratio

        return required_engine_torque

    def engine_torque_to_throttle(self, engine_torque):
        """
        从发动机扭矩计算油门值

        参数:
        engine_torque - 发动机扭矩 (N·m)

        返回:
        油门值 (0-1)
        """
        # 假设油门与发动机扭矩线性相关（简化）
        throttle = engine_torque / self.max_engine_torque

        # 限制范围
        throttle = max(0.0, min(1.0, throttle))

        return throttle

    def wheel_torque_to_throttle(self, total_wheel_torque, current_speed_kmh):
        """
        从车轮扭矩直接转换为油门/刹车值（前馈控制）

        参数:
        total_wheel_torque - 总车轮扭矩 (N·m)，正值为驱动，负值为制动
        current_speed_kmh - 当前车速 (km/h)

        返回:
        (throttle, brake) - 油门值和刹车值，范围都是 0-1
        """
        # 转换速度单位
        current_speed_ms = current_speed_kmh / 3.6

        # 1. 从车轮扭矩计算期望加速度
        desired_accel = self.wheel_torque_to_acceleration(
            total_wheel_torque, current_speed_ms
        )

        # 2. 从加速度反算发动机扭矩
        required_engine_torque = self.acceleration_to_engine_torque(
            desired_accel, current_speed_ms
        )

        # 3. 判断是加速还是制动
        if required_engine_torque >= 0:
            # 加速
            throttle = self.engine_torque_to_throttle(required_engine_torque)
            brake = 0.0
        else:
            # 制动（需要刹车）
            throttle = 0.0
            # 将负扭矩转换为刹车力
            # 简化映射：brake = |required_torque| / max_brake_torque
            # 假设最大制动扭矩为1000 N·m（来自wheel参数）
            max_brake_torque = 1000.0
            brake_torque_needed = abs(required_engine_torque * self.total_gear_ratio)
            brake = min(1.0, brake_torque_needed / (max_brake_torque * self.num_drive_wheels))

        return throttle, brake

    def wheel_torque_to_throttle_with_pid(self, total_wheel_torque, current_speed_kmh, dt=0.05):
        """
        从车轮扭矩转换为油门/刹车值（前馈 + PID反馈控制）

        参数:
        total_wheel_torque - 总车轮扭矩 (N·m)
        current_speed_kmh - 当前车速 (km/h)
        dt - 时间步长 (s)

        返回:
        (throttle, brake) - 油门值和刹车值，范围都是 0-1
        """
        # 前馈控制：基于模型计算初始油门
        throttle_ff, brake_ff = self.wheel_torque_to_throttle(
            total_wheel_torque, current_speed_kmh
        )

        if not self.pid_enabled:
            return throttle_ff, brake_ff

        # PID反馈控制（需要测量实际加速度）
        # 注意：这里需要在实际应用时测量真实加速度
        # 当前仅返回前馈值

        # TODO: 添加PID反馈
        # 1. 测量实际加速度
        # 2. 计算误差: error = desired_accel - actual_accel
        # 3. PID计算修正量
        # 4. throttle = throttle_ff + correction

        return throttle_ff, brake_ff

    def sppvt_torque_to_control(self, sppvt_output, current_speed_kmh):
        """
        SPPVT输出转换为CARLA控制命令

        参数:
        sppvt_output - SPPVT控制器输出（假设为车轮扭矩 N·m）
        current_speed_kmh - 当前车速 (km/h)

        返回:
        (throttle, brake) - 油门和刹车值
        """
        # 假设SPPVT输出的是总车轮扭矩
        total_wheel_torque = sppvt_output

        # 转换为油门/刹车
        throttle, brake = self.wheel_torque_to_throttle(
            total_wheel_torque, current_speed_kmh
        )

        return throttle, brake

    def reset_pid(self):
        """重置PID控制器状态"""
        self.integral_error = 0.0
        self.prev_error = 0.0

    def set_pid_parameters(self, kp=None, ki=None, kd=None):
        """设置PID参数"""
        if kp is not None:
            self.pid_kp = kp
        if ki is not None:
            self.pid_ki = ki
        if kd is not None:
            self.pid_kd = kd

        print(f"PID参数更新: Kp={self.pid_kp}, Ki={self.pid_ki}, Kd={self.pid_kd}")


def test_converter():
    """测试转换器"""
    print("\n" + "="*60)
    print("测试车轮扭矩到油门转换器")
    print("="*60)

    import carla

    # 连接CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 生成测试车辆
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

    vehicle.set_simulate_physics(True)

    for _ in range(20):
        world.tick()

    try:
        # 创建转换器
        converter = TorqueToThrottleConverter(vehicle)

        # 测试不同的车轮扭矩
        test_cases = [
            (500.0, 0.0),     # 500 N·m，静止
            (1000.0, 0.0),    # 1000 N·m，静止
            (2000.0, 30.0),   # 2000 N·m，30 km/h
            (3000.0, 60.0),   # 3000 N·m，60 km/h
            (-500.0, 30.0),   # -500 N·m（制动），30 km/h
            (-1000.0, 60.0),  # -1000 N·m（制动），60 km/h
        ]

        print("\n【测试结果】")
        print("车轮扭矩(N·m)\t速度(km/h)\t油门\t刹车")
        print("-" * 60)

        for wheel_torque, speed in test_cases:
            throttle, brake = converter.wheel_torque_to_throttle(wheel_torque, speed)
            print(f"{wheel_torque:8.1f}\t{speed:8.1f}\t{throttle:.3f}\t{brake:.3f}")

        print("\n✓ 转换器测试完成")

    finally:
        vehicle.destroy()


if __name__ == "__main__":
    test_converter()
