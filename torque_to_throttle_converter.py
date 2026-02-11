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

    正扭矩：SPPVT输出发动机扭矩(N·m) → 油门开度(0-1)
    负扭矩：SPPVT输出减速度(m/s²) → 刹车开度(0-1)

    所有物理参数均从CARLA Audi e-tron真实测量获得
    """

    # === CARLA Audi e-tron 真实物理参数（通过carla_vehicle_params_inspector.py测得）===
    VEHICLE_MASS = 2370.00  # kg - 车辆质量
    WHEEL_RADIUS = 0.37  # m - 车轮有效半径
    MAX_BRAKE_TORQUE_PER_WHEEL = 1000.0  # N·m - 单轮最大制动扭矩
    NUM_DRIVE_WHEELS = 4  # 驱动轮数量
    TOTAL_GEAR_RATIO = 9.204  # 总传动比（第1档）

    # 理论最大减速度 = (总制动扭矩 / 轮半径) / 质量
    # = (4 × 1000 / 0.37) / 2370 = 4.56 m/s²
    THEORETICAL_MAX_DECEL = 4.56  # m/s²

    def __init__(self, vehicle):
        """
        初始化转换器

        参数:
        vehicle - CARLA车辆对象
        """
        self.vehicle = vehicle

        # 使用硬编码的真实物理参数
        self.vehicle_mass = self.VEHICLE_MASS
        self.wheel_radius = self.WHEEL_RADIUS
        self.max_brake_torque_per_wheel = self.MAX_BRAKE_TORQUE_PER_WHEEL
        self.num_drive_wheels = self.NUM_DRIVE_WHEELS
        self.total_gear_ratio = self.TOTAL_GEAR_RATIO
        self.theoretical_max_decel = self.THEORETICAL_MAX_DECEL

        # 从CARLA加载其他参数（扭矩曲线等）
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
        """
        从CARLA加载无法硬编码的参数（主要是扭矩曲线）
        物理参数（质量、轮半径等）已硬编码为真实测量值
        """
        physics = self.vehicle.get_physics_control()

        # === 传动系统参数（用于打印显示）===
        self.gear_ratio = physics.forward_gears[0].ratio if physics.forward_gears else 1.0
        self.final_ratio = physics.final_ratio

        # === 发动机参数 ===
        self.max_rpm = physics.max_rpm

        # 提取扭矩曲线数据（必须从CARLA读取，无法硬编码）
        # physics.torque_curve 是 Vector2D 列表，x=RPM, y=扭矩(N·m)
        self.torque_curve_rpm = np.array([point.x for point in physics.torque_curve])
        self.torque_curve_torque = np.array([point.y for point in physics.torque_curve])

        # 计算最大扭矩
        self.max_engine_torque = np.max(self.torque_curve_torque)

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

    def deceleration_to_brake(self, decel_ms2):
        """
        减速度 → 刹车开度（物理模型，基于CARLA真实参数）

        完全符合CARLA底层实现：
        applied_torque = brake × max_brake_torque

        物理推导：
        1. F_total = m × |a|                    (牛顿第二定律)
        2. F_per_wheel = F_total / num_wheels   (分配到各轮)
        3. T_per_wheel = F_per_wheel × r        (力矩 = 力 × 半径)
        4. brake = T_per_wheel / T_max          (归一化到CARLA范围)

        参数:
            decel_ms2: 减速度 (m/s²，负值)
                      例如 -2.5 表示 2.5 m/s² 减速

        返回:
            brake: 刹车开度 [0, 1]
        """
        # 1. 取绝对值
        abs_decel = abs(decel_ms2)

        # 2. 减速度 → 总制动力 (牛顿第二定律: F = ma)
        F_total = self.vehicle_mass * abs_decel

        # 3. 分配到各轮
        F_per_wheel = F_total / self.num_drive_wheels

        # 4. 制动力 → 制动扭矩 (T = F × r)
        T_per_wheel = F_per_wheel * self.wheel_radius

        # 5. 归一化为刹车值（符合CARLA公式）
        # 使得：applied_torque = brake × max_brake_torque = T_per_wheel ✓
        brake = T_per_wheel / self.max_brake_torque_per_wheel

        # 6. 限制范围
        brake = min(1.0, max(0.0, brake))

        # 7. 超限警告
        if brake >= 0.99 and abs_decel > 0.1:
            print(f"⚠️ 需求减速度 {abs_decel:.2f} m/s² 接近/超过极限！")
            print(f"   该车型理论最大减速度约 {self.theoretical_max_decel:.2f} m/s²")

        return brake

    def _engine_torque_to_brake_value(self, engine_brake_torque):
        """
        发动机级别的制动扭矩 → 刹车值

        物理链路：
        1. 发动机扭矩 × 传动比 = 车轮总制动扭矩
        2. 车轮总扭矩 ÷ 轮数 = 单轮制动扭矩
        3. 单轮扭矩 ÷ 最大制动扭矩 = 刹车值

        符合CARLA底层公式：applied_torque = brake × max_brake_torque

        参数:
            engine_brake_torque: 发动机级别的制动扭矩 (N·m, 正值)

        返回:
            brake_value: 刹车值 [0, 1]
        """
        # 1. 通过传动系统放大到车轮
        wheel_brake_torque_total = engine_brake_torque * self.total_gear_ratio

        # 2. 分配到各个驱动轮
        per_wheel_brake_torque = wheel_brake_torque_total / self.num_drive_wheels

        # 3. 归一化为刹车值（符合CARLA公式）
        brake_value = per_wheel_brake_torque / self.max_brake_torque_per_wheel

        # 4. 限制在[0, 1]范围
        brake_value = min(1.0, max(0.0, brake_value))

        # 5. 超限警告
        if brake_value >= 0.99 and engine_brake_torque > 10.0:
            # 计算等效减速度（用于警告显示）
            total_brake_force = wheel_brake_torque_total / self.wheel_radius
            equiv_decel = total_brake_force / self.vehicle_mass
            print(f"⚠️ 制动需求接近极限！")
            print(f"   需求扭矩: {engine_brake_torque:.1f} N·m (发动机级别)")
            print(f"   等效减速度: {equiv_decel:.2f} m/s²")
            print(f"   理论最大: {self.theoretical_max_decel:.2f} m/s²")

        return brake_value

    def engine_torque_to_throttle(self, desired_engine_torque, current_speed_kmh):
        """
        主接口：统一处理加速/减速转换（统一扭矩单位）

        加速模式（正值）：
            发动机扭矩 (N·m) → 油门开度 [0,1]
            1. 根据车速计算发动机RPM
            2. 从扭矩曲线插值得到该RPM的最大扭矩
            3. 油门 = 需求扭矩 / 最大扭矩

        减速模式（负值）：
            发动机扭矩 (N·m) → 车轮制动扭矩 → 刹车开度 [0,1]
            1. 发动机扭矩 × 传动比 = 车轮制动扭矩
            2. 车轮扭矩 ÷ 轮数 = 单轮制动扭矩
            3. 单轮扭矩 ÷ 最大制动扭矩 = 刹车值

        参数:
            desired_engine_torque: 发动机扭矩 (N·m)
                - 正值：驱动扭矩
                - 负值：制动扭矩
            current_speed_kmh: 当前车速 (km/h)

        返回:
            (throttle, brake): 油门值和刹车值 [0, 1]
        """
        if desired_engine_torque >= 0:
            # === 加速模式：扭矩 → 油门 ===
            # 计算当前发动机RPM
            current_rpm = self._calculate_engine_rpm(current_speed_kmh)

            # 获取该RPM下的最大可用扭矩
            max_available_torque = self._get_max_torque_at_rpm(current_rpm)

            # 计算油门值
            if max_available_torque > 0:
                throttle = desired_engine_torque / max_available_torque
            else:
                throttle = 0.0

            # 限制在[0, 1]范围
            throttle = min(1.0, max(0.0, throttle))
            brake = 0.0

        else:
            # === 减速模式：扭矩 → 刹车 ===
            throttle = 0.0
            brake = self._engine_torque_to_brake_value(abs(desired_engine_torque))

        return throttle, brake

    def throttle_to_engine_torque(self, throttle: float, current_speed_kmh: float) -> float:
        """
        油门开度 → 发动机扭矩（engine_torque_to_throttle 的反向计算）

        engine_torque = throttle * max_torque_at_current_rpm

        参数:
            throttle: 油门开度 [0, 1]
            current_speed_kmh: 当前车速 (km/h)

        返回:
            发动机扭矩 (N·m)
        """
        current_rpm = self._calculate_engine_rpm(current_speed_kmh)
        max_available_torque = self._get_max_torque_at_rpm(current_rpm)
        return throttle * max_available_torque

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

