#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发动机扭矩到油门转换器
将SPPVT输出的发动机扭矩(N·m)转换为CARLA的油门/刹车控制值(0-1)

核心原理：
1. 根据车速计算发动机转速(RPM)
2. 从扭矩曲线插值得到该RPM下的最大可用扭矩
3. 油门值 = 需求扭矩 / 最大可用扭矩
4. 负扭矩通过刹车系统实现
"""

import math
import numpy as np


class TorqueToThrottleConverter:
    """
    发动机扭矩到油门转换器

    假设：SPPVT输出的是发动机扭矩(N·m)
    目标：转换为CARLA的油门值(0-1)
    """

    def __init__(self, vehicle):
        """
        初始化转换器

        参数:
        vehicle - CARLA车辆对象
        """
        self.vehicle = vehicle

        # 从CARLA加载车辆物理参数
        self._load_vehicle_parameters()

        print(f"\n{'='*60}")
        print(f"TorqueToThrottleConverter 初始化完成 (完整RPM模型)")
        print(f"{'='*60}")
        print(f"车辆物理参数:")
        print(f"  车轮半径: {self.wheel_radius:.3f} m")
        print(f"  档位传动比: {self.gear_ratio:.3f}")
        print(f"  最终传动比: {self.final_ratio:.3f}")
        print(f"  总传动比: {self.total_gear_ratio:.3f}")
        print(f"  最大RPM: {self.max_rpm:.0f}")
        print(f"\n发动机扭矩曲线:")
        for rpm, torque in zip(self.torque_curve_rpm, self.torque_curve_torque):
            print(f"  {rpm:6.0f} RPM → {torque:6.1f} N·m")
        print(f"  最大扭矩: {self.max_engine_torque:.1f} N·m")
        print(f"\n制动系统:")
        print(f"  单轮最大制动扭矩: {self.max_brake_torque_per_wheel:.1f} N·m")
        print(f"  驱动轮数量: {self.num_drive_wheels}")
        print(f"{'='*60}\n")

    def _load_vehicle_parameters(self):
        """从CARLA车辆加载物理参数"""
        physics = self.vehicle.get_physics_control()

        # === 车轮参数 ===
        # 假设4个车轮相同，取第一个
        self.wheel_radius = physics.wheels[0].radius / 100.0  # cm转m
        self.num_drive_wheels = 4  # Audi e-tron是四轮驱动

        # === 传动系统参数 ===
        # 档位传动比（电动车通常是单速，取第一档）
        self.gear_ratio = physics.forward_gears[0].ratio if physics.forward_gears else 1.0

        # 最终传动比（差速器）
        self.final_ratio = physics.final_ratio

        # 总传动比 = 档位比 × 最终比
        self.total_gear_ratio = self.gear_ratio * self.final_ratio

        # === 发动机参数 ===
        self.max_rpm = physics.max_rpm

        # 提取扭矩曲线数据
        # physics.torque_curve 是 Vector2D 列表，x=RPM, y=扭矩(N·m)
        self.torque_curve_rpm = np.array([point.x for point in physics.torque_curve])
        self.torque_curve_torque = np.array([point.y for point in physics.torque_curve])

        # 计算最大扭矩
        self.max_engine_torque = np.max(self.torque_curve_torque)

        # === 制动系统参数 ===
        self.max_brake_torque_per_wheel = physics.wheels[0].max_brake_torque  # N·m

    def _calculate_engine_rpm(self, speed_kmh):
        """
        根据车速计算发动机转速

        参数:
        speed_kmh - 车速 (km/h)

        返回:
        发动机转速 (RPM)
        """
        # 1. 车速单位转换: km/h → m/s
        speed_ms = speed_kmh / 3.6

        # 2. 车轮角速度 (rad/s)
        # ω = v / r
        wheel_angular_velocity = speed_ms / self.wheel_radius

        # 3. 车轮转速 (RPM)
        # RPM = ω × 60 / (2π)
        wheel_rpm = wheel_angular_velocity * 60.0 / (2.0 * math.pi)

        # 4. 发动机转速 (RPM)
        # 发动机RPM = 车轮RPM × 传动比
        engine_rpm = wheel_rpm * self.total_gear_ratio

        # 5. 限制在合理范围内
        # 电动车可能从0 RPM开始，但扭矩曲线通常从负值或0开始
        engine_rpm = max(0.0, min(engine_rpm, self.max_rpm))

        return engine_rpm

    def _get_max_torque_at_rpm(self, rpm):
        """
        从扭矩曲线插值得到该RPM下的最大可用扭矩

        参数:
        rpm - 发动机转速 (RPM)

        返回:
        该RPM下的最大扭矩 (N·m)
        """
        # 使用线性插值
        # 如果RPM超出范围，np.interp会使用边界值
        max_torque = np.interp(rpm, self.torque_curve_rpm, self.torque_curve_torque)

        return max_torque

    def _brake_torque_to_brake_value(self, engine_brake_torque):
        """
        将发动机制动扭矩转换为刹车值

        参数:
        engine_brake_torque - 发动机制动扭矩 (N·m, 正值)

        返回:
        刹车值 (0-1)
        """
        # 1. 通过传动系统放大到车轮
        # 车轮制动扭矩 = 发动机扭矩 × 传动比
        wheel_brake_torque_total = engine_brake_torque * self.total_gear_ratio

        # 2. 分配到各个驱动轮
        # 假设平均分配
        per_wheel_brake_torque = wheel_brake_torque_total / self.num_drive_wheels

        # 3. 归一化为刹车值 (0-1)
        # brake_value = 需求扭矩 / 最大制动扭矩
        brake_value = per_wheel_brake_torque / self.max_brake_torque_per_wheel

        # 4. 限制在合理范围
        brake_value = min(1.0, max(0.0, brake_value))

        return brake_value

    def engine_torque_to_throttle(self, desired_engine_torque, current_speed_kmh):
        """
        主接口：发动机扭矩 → 油门/刹车

        这是核心转换函数，实现完整的RPM模型：
        1. 根据车速计算发动机RPM
        2. 从扭矩曲线插值得到该RPM的最大扭矩
        3. 计算油门 = 需求扭矩 / 最大扭矩

        参数:
        desired_engine_torque - SPPVT输出的发动机扭矩 (N·m)
                               正值=驱动, 负值=制动
        current_speed_kmh - 当前车速 (km/h)

        返回:
        (throttle, brake) - 油门值和刹车值 (0-1)
        """
        # === 1. 判断是驱动还是制动 ===
        if desired_engine_torque >= 0:
            # --- 驱动模式：通过油门控制 ---

            # 2. 计算当前发动机RPM
            current_rpm = self._calculate_engine_rpm(current_speed_kmh)

            # 3. 获取该RPM下的最大可用扭矩
            max_available_torque = self._get_max_torque_at_rpm(current_rpm)

            # 4. 计算油门值
            # throttle = 需求扭矩 / 最大可用扭矩
            if max_available_torque > 0:
                throttle = desired_engine_torque / max_available_torque
            else:
                # 防御性代码：如果最大扭矩为0，油门为0
                throttle = 0.0

            # 5. 限制在[0, 1]范围
            throttle = min(1.0, max(0.0, throttle))

            brake = 0.0

        else:
            # --- 制动模式：通过刹车控制 ---

            throttle = 0.0

            # 将负扭矩转换为刹车值
            brake = self._brake_torque_to_brake_value(abs(desired_engine_torque))

        return throttle, brake

    def get_conversion_info(self, desired_engine_torque, current_speed_kmh):
        """
        获取转换过程的详细信息（用于调试）

        参数:
        desired_engine_torque - 需求发动机扭矩 (N·m)
        current_speed_kmh - 当前车速 (km/h)

        返回:
        包含转换详情的字典
        """
        current_rpm = self._calculate_engine_rpm(current_speed_kmh)
        max_available_torque = self._get_max_torque_at_rpm(current_rpm)
        throttle, brake = self.engine_torque_to_throttle(desired_engine_torque, current_speed_kmh)

        return {
            'desired_torque': desired_engine_torque,
            'current_speed_kmh': current_speed_kmh,
            'current_rpm': current_rpm,
            'max_available_torque': max_available_torque,
            'throttle': throttle,
            'brake': brake,
            'torque_utilization': (desired_engine_torque / max_available_torque * 100) if max_available_torque > 0 else 0
        }


# ============================================================================
# 测试代码
# ============================================================================

def test_converter():
    """测试转换器的完整功能"""
    print("\n" + "="*60)
    print("测试发动机扭矩到油门转换器 (完整RPM模型)")
    print("="*60)

    import carla
    import sys
    import io

    # 设置控制台输出为UTF-8（Windows）
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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

        # === 测试1: 不同速度和扭矩组合 ===
        print("\n【测试1: 不同速度和扭矩的转换结果】")
        print(f"\n{'速度':<10} {'需求扭矩':<12} {'RPM':<10} {'最大扭矩':<12} {'油门':<8} {'刹车':<8} {'扭矩利用率':<10}")
        print("-" * 90)

        test_cases = [
            # (速度 km/h, 需求扭矩 N·m)
            (0,    200),   # 静止，中等扭矩
            (0,    500),   # 静止，大扭矩
            (20,   300),   # 低速，中等扭矩
            (40,   400),   # 中速，大扭矩
            (60,   600),   # 高速，很大扭矩
            (80,   400),   # 更高速
            (100,  300),   # 很高速
            (120,  200),   # 极高速，中等扭矩
            (60,   -200),  # 制动测试
            (80,   -400),  # 大制动
        ]

        for speed, torque in test_cases:
            info = converter.get_conversion_info(torque, speed)

            print(f"{speed:<10.0f} {torque:<12.0f} {info['current_rpm']:<10.0f} "
                  f"{info['max_available_torque']:<12.1f} {info['throttle']:<8.3f} "
                  f"{info['brake']:<8.3f} {info['torque_utilization']:<10.1f}%")

        # === 测试2: 固定扭矩，不同速度（观察RPM影响）===
        print("\n【测试2: 固定扭矩400 N·m，不同速度的油门变化】")
        print(f"\n{'速度(km/h)':<12} {'RPM':<10} {'最大扭矩':<12} {'油门':<8}")
        print("-" * 50)

        fixed_torque = 400.0
        for speed in range(0, 121, 10):
            info = converter.get_conversion_info(fixed_torque, speed)
            print(f"{speed:<12.0f} {info['current_rpm']:<10.0f} "
                  f"{info['max_available_torque']:<12.1f} {info['throttle']:<8.3f}")

        # === 测试3: 固定速度，不同扭矩（验证线性关系）===
        print("\n【测试3: 固定速度60 km/h，不同扭矩的油门变化】")
        print(f"\n{'需求扭矩(N·m)':<15} {'油门':<8}")
        print("-" * 30)

        fixed_speed = 60.0
        for torque in range(0, 801, 100):
            info = converter.get_conversion_info(torque, fixed_speed)
            print(f"{torque:<15.0f} {info['throttle']:<8.3f}")

        print("\n✓ 转换器测试完成")

    finally:
        vehicle.destroy()


if __name__ == "__main__":
    test_converter()
