#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于PID的加速度到油门转换器
适用于CARLA的实际控制需求

原理：
SPPVT输出期望加速度 → PID控制器 → 油门/刹车值
不再依赖扭矩曲线，直接闭环控制
"""

import math


class AccelerationToThrottlePID:
    """
    加速度到油门的PID控制器

    这是针对CARLA实际情况的实用方案：
    1. 避免了油门-扭矩非线性映射的问题
    2. 使用闭环控制自动适应车辆特性
    3. 简单可靠
    """

    def __init__(self, vehicle):
        """
        初始化PID控制器

        参数:
        vehicle - CARLA车辆对象（用于获取当前速度）
        """
        self.vehicle = vehicle

        # PID参数（需要根据实际效果调整）
        self.kp = 0.3  # 比例增益
        self.ki = 0.05  # 积分增益
        self.kd = 0.1  # 微分增益

        # PID状态
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_speed = 0.0

        # 时间步长（CARLA默认0.05s）
        self.dt = 0.05

        # 前馈增益（可选，用于改善响应速度）
        self.feedforward_gain = 0.15

        print(f"\n{'='*60}")
        print(f"AccelerationToThrottlePID 初始化")
        print(f"{'='*60}")
        print(f"PID参数: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
        print(f"前馈增益: {self.feedforward_gain}")
        print(f"{'='*60}\n")

    def get_current_acceleration(self):
        """
        计算当前实际加速度 (m/s²)

        返回:
        当前加速度
        """
        # 获取当前速度
        velocity = self.vehicle.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        # 计算加速度（速度变化率）
        acceleration = (current_speed - self.prev_speed) / self.dt

        # 更新历史速度
        self.prev_speed = current_speed

        return acceleration

    def reset(self):
        """重置PID状态"""
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_speed = 0.0
        print("PID控制器已重置")

    def set_pid_parameters(self, kp=None, ki=None, kd=None, feedforward=None):
        """
        动态调整PID参数

        参数:
        kp - 比例增益
        ki - 积分增益
        kd - 微分增益
        feedforward - 前馈增益
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
        if feedforward is not None:
            self.feedforward_gain = feedforward

        print(f"PID参数更新: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}, FF={self.feedforward_gain}")

    def acceleration_to_throttle(self, desired_acceleration):
        """
        主接口：期望加速度 → 油门/刹车

        参数:
        desired_acceleration - 期望加速度 (m/s²)
                              正值=加速，负值=减速

        返回:
        (throttle, brake) - 油门和刹车值 (0-1)
        """
        # 1. 测量当前实际加速度
        actual_acceleration = self.get_current_acceleration()

        # 2. 计算误差
        error = desired_acceleration - actual_acceleration

        # 3. PID计算
        # 比例项
        p_term = self.kp * error

        # 积分项（带抗积分饱和）
        self.integral_error += error * self.dt
        # 限制积分项，防止积分饱和
        self.integral_error = max(-5.0, min(5.0, self.integral_error))
        i_term = self.ki * self.integral_error

        # 微分项
        d_term = self.kd * (error - self.prev_error) / self.dt

        # 4. 前馈项（基于期望加速度的开环预估）
        feedforward = self.feedforward_gain * desired_acceleration

        # 5. 总控制量
        control_output = feedforward + p_term + i_term + d_term

        # 6. 分离为油门和刹车
        if control_output > 0:
            # 加速
            throttle = min(1.0, max(0.0, control_output))
            brake = 0.0
        else:
            # 减速
            throttle = 0.0
            brake = min(1.0, max(0.0, -control_output))

        # 7. 更新状态
        self.prev_error = error

        return throttle, brake

    def get_debug_info(self, desired_accel):
        """
        获取调试信息

        返回:
        包含PID各项详细信息的字典
        """
        actual_accel = self.get_current_acceleration()
        error = desired_accel - actual_accel

        return {
            'desired_acceleration': desired_accel,
            'actual_acceleration': actual_accel,
            'error': error,
            'p_term': self.kp * error,
            'i_term': self.ki * self.integral_error,
            'd_term': self.kd * (error - self.prev_error) / self.dt,
            'feedforward': self.feedforward_gain * desired_accel,
            'integral_error': self.integral_error
        }


# ============================================================================
# 简化版本：直接映射（备用方案）
# ============================================================================

class SimpleAccelerationMapper:
    """
    简单的加速度到油门映射（无PID反馈）

    使用经验公式，适合快速原型
    """

    def __init__(self, vehicle=None):
        """
        初始化简单映射器

        参数:
        vehicle - CARLA车辆对象（可选，此方案不需要反馈）
        """
        # 映射参数（根据CARLA实际测试调整）
        self.accel_to_throttle_gain = 0.25  # 加速度→油门的增益
        self.brake_gain = 0.20  # 减速度→刹车的增益

        print(f"\n简单加速度映射器初始化")
        print(f"  加速增益: {self.accel_to_throttle_gain}")
        print(f"  制动增益: {self.brake_gain}\n")

    def acceleration_to_throttle(self, desired_acceleration):
        """
        简单映射：加速度 → 油门/刹车

        参数:
        desired_acceleration - 期望加速度 (m/s²)

        返回:
        (throttle, brake)
        """
        if desired_acceleration > 0:
            # 加速
            throttle = min(1.0, desired_acceleration * self.accel_to_throttle_gain)
            brake = 0.0
        else:
            # 减速
            throttle = 0.0
            brake = min(1.0, abs(desired_acceleration) * self.brake_gain)

        return throttle, brake


# ============================================================================
# 测试代码
# ============================================================================

def test_controllers():
    """测试两种控制器"""
    import carla
    import time
    import sys
    import io

    # 设置控制台输出为UTF-8（Windows）
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("\n" + "="*60)
    print("测试加速度到油门转换器")
    print("="*60)

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
        # 测试PID控制器
        print("\n【测试1: PID控制器】")
        pid_controller = AccelerationToThrottlePID(vehicle)

        # 测试不同的目标加速度
        test_cases = [
            (2.0, "中等加速"),
            (4.0, "强加速"),
            (-2.0, "中等制动"),
            (0.0, "保持速度")
        ]

        for target_accel, description in test_cases:
            print(f"\n目标: {description} ({target_accel:.1f} m/s²)")

            for i in range(20):
                throttle, brake = pid_controller.acceleration_to_throttle(target_accel)

                # 应用控制
                control = carla.VehicleControl()
                control.throttle = throttle
                control.brake = brake
                vehicle.apply_control(control)
                world.tick()

                # 每5帧输出一次
                if i % 5 == 0:
                    debug = pid_controller.get_debug_info(target_accel)
                    print(f"  帧{i}: 油门={throttle:.3f}, 刹车={brake:.3f}, "
                          f"误差={debug['error']:.3f} m/s²")

        # 测试简单映射器
        print("\n\n【测试2: 简单映射器】")
        simple_mapper = SimpleAccelerationMapper()

        for target_accel, description in test_cases:
            throttle, brake = simple_mapper.acceleration_to_throttle(target_accel)
            print(f"{description} ({target_accel:.1f} m/s²): "
                  f"油门={throttle:.3f}, 刹车={brake:.3f}")

        print("\n✓ 测试完成")

    finally:
        vehicle.destroy()


if __name__ == "__main__":
    test_controllers()
