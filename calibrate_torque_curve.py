#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过实验标定完整的扭矩曲线
原理：在不同速度下施加固定油门，测量加速度，反推实际扭矩
"""

import carla
import math
import time
import numpy as np
import sys
import io
import json

# 设置控制台输出为UTF-8（Windows）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class TorqueCurveCalibrator:
    """扭矩曲线标定器"""

    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world

        # 从CARLA加载车辆参数
        physics = vehicle.get_physics_control()
        self.mass = physics.mass
        self.drag_coef = physics.drag_coefficient
        self.frontal_area = 3.2  # Audi e-tron估算值 (m²)
        self.air_density = 1.225  # kg/m³
        self.rolling_resistance = 0.015

        self.wheel_radius = physics.wheels[0].radius / 100.0  # m
        self.gear_ratio = physics.forward_gears[0].ratio if physics.forward_gears else 1.0
        self.final_ratio = physics.final_ratio
        self.total_gear_ratio = self.gear_ratio * self.final_ratio

        # 已知的扭矩曲线数据点（用于验证）
        self.known_torque_curve = {
            0: 749.0,
            4000: 749.0,
            5000: 610.0,
            6000: 507.0,
            7000: 435.0,
            8000: 381.0,
            9000: 338.0
        }

        print(f"\n扭矩曲线标定器初始化:")
        print(f"  车辆质量: {self.mass} kg")
        print(f"  传动比: {self.total_gear_ratio:.3f}")
        print(f"  车轮半径: {self.wheel_radius:.3f} m")

    def calculate_rpm(self, speed_kmh):
        """从车速计算RPM"""
        speed_ms = speed_kmh / 3.6
        wheel_angular_vel = speed_ms / self.wheel_radius
        wheel_rpm = wheel_angular_vel * 60.0 / (2.0 * math.pi)
        engine_rpm = wheel_rpm * self.total_gear_ratio
        return engine_rpm

    def measure_acceleration(self, target_speed_kmh, throttle_value, duration=3.0):
        """
        在指定速度下施加固定油门，测量加速度

        参数:
        target_speed_kmh - 目标车速 (km/h)
        throttle_value - 油门值 (0-1)
        duration - 测量时长 (s)

        返回:
        (actual_speed, acceleration, rpm)
        """
        print(f"\n测量: 速度={target_speed_kmh:.0f} km/h, 油门={throttle_value:.2f}")

        # 1. 先加速到目标速度附近
        print(f"  加速到目标速度...")
        for _ in range(200):
            current_velocity = self.vehicle.get_velocity()
            current_speed = math.sqrt(current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2) * 3.6

            if current_speed < target_speed_kmh - 5:
                # 还未达到，加速
                control = carla.VehicleControl()
                control.throttle = 0.7
                control.brake = 0.0
            elif current_speed > target_speed_kmh + 5:
                # 超过了，减速
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 0.3
            else:
                # 接近目标速度
                break

            self.vehicle.apply_control(control)
            self.world.tick()

        # 2. 稳定一下
        for _ in range(30):
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 0.0
            self.vehicle.apply_control(control)
            self.world.tick()

        # 3. 记录初始状态
        initial_velocity = self.vehicle.get_velocity()
        initial_speed = math.sqrt(initial_velocity.x**2 + initial_velocity.y**2 + initial_velocity.z**2)

        # 4. 施加固定油门并测量
        print(f"  施加油门并测量加速度...")
        start_time = time.time()
        speed_samples = []

        while time.time() - start_time < duration:
            # 施加固定油门
            control = carla.VehicleControl()
            control.throttle = throttle_value
            control.brake = 0.0
            self.vehicle.apply_control(control)

            # 测量速度
            current_velocity = self.vehicle.get_velocity()
            current_speed_ms = math.sqrt(current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2)

            speed_samples.append({
                'time': time.time() - start_time,
                'speed': current_speed_ms
            })

            self.world.tick()

        # 5. 计算平均加速度（线性拟合）
        times = np.array([s['time'] for s in speed_samples])
        speeds = np.array([s['speed'] for s in speed_samples])

        # 使用最小二乘法拟合
        coeffs = np.polyfit(times, speeds, 1)
        acceleration = coeffs[0]  # m/s²

        # 平均速度和RPM
        avg_speed_ms = np.mean(speeds)
        avg_speed_kmh = avg_speed_ms * 3.6
        avg_rpm = self.calculate_rpm(avg_speed_kmh)

        print(f"  结果: 实际速度={avg_speed_kmh:.1f} km/h, "
              f"RPM={avg_rpm:.0f}, 加速度={acceleration:.3f} m/s²")

        return avg_speed_kmh, acceleration, avg_rpm

    def calculate_torque_from_acceleration(self, speed_kmh, acceleration):
        """
        从加速度反算发动机扭矩

        参数:
        speed_kmh - 车速 (km/h)
        acceleration - 测量的加速度 (m/s²)

        返回:
        发动机扭矩 (N·m)
        """
        speed_ms = speed_kmh / 3.6

        # 1. 计算需要的净力
        required_net_force = acceleration * self.mass

        # 2. 计算阻力
        air_drag = 0.5 * self.air_density * self.drag_coef * self.frontal_area * (speed_ms ** 2)
        rolling_drag = self.rolling_resistance * self.mass * 9.81

        # 3. 总驱动力 = 净力 + 阻力
        drive_force = required_net_force + air_drag + rolling_drag

        # 4. 车轮扭矩
        wheel_torque = drive_force * self.wheel_radius

        # 5. 发动机扭矩
        engine_torque = wheel_torque / self.total_gear_ratio

        return engine_torque

    def calibrate_full_curve(self, throttle_value=0.8):
        """
        标定完整扭矩曲线

        参数:
        throttle_value - 测试用的油门值 (0-1)

        返回:
        校准后的扭矩曲线数据
        """
        print(f"\n{'='*60}")
        print(f"开始标定扭矩曲线 (油门={throttle_value})")
        print(f"{'='*60}")

        # 测试速度点（覆盖整个工作范围）
        test_speeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

        calibration_results = []

        for speed in test_speeds:
            try:
                # 测量加速度
                actual_speed, accel, rpm = self.measure_acceleration(speed, throttle_value, duration=2.0)

                # 反算扭矩
                measured_torque = self.calculate_torque_from_acceleration(actual_speed, accel)

                # 根据油门值归一化到满油门扭矩
                estimated_max_torque = measured_torque / throttle_value

                # 与已知数据对比
                known_torque = np.interp(rpm,
                                         list(self.known_torque_curve.keys()),
                                         list(self.known_torque_curve.values()))

                error_percent = (estimated_max_torque - known_torque) / known_torque * 100 if known_torque > 0 else 0

                result = {
                    'target_speed_kmh': speed,
                    'actual_speed_kmh': actual_speed,
                    'rpm': rpm,
                    'acceleration_ms2': accel,
                    'measured_torque_nm': measured_torque,
                    'estimated_max_torque_nm': estimated_max_torque,
                    'known_torque_nm': known_torque,
                    'error_percent': error_percent
                }

                calibration_results.append(result)

                print(f"  估算最大扭矩: {estimated_max_torque:.1f} N·m, "
                      f"已知值: {known_torque:.1f} N·m, "
                      f"误差: {error_percent:+.1f}%")

            except Exception as e:
                print(f"  ❌ 测量失败: {e}")
                continue

        return calibration_results

    def save_calibration_results(self, results, filename='torque_curve_calibration.json'):
        """保存标定结果"""
        # 构建完整的扭矩曲线
        calibrated_curve = []
        for r in results:
            calibrated_curve.append({
                'rpm': r['rpm'],
                'torque': r['estimated_max_torque_nm']
            })

        # 与已知曲线合并
        merged_curve = {}

        # 添加已知点
        for rpm, torque in self.known_torque_curve.items():
            merged_curve[rpm] = {'torque': torque, 'source': 'CARLA_API'}

        # 添加校准点
        for point in calibrated_curve:
            rpm = int(point['rpm'])
            merged_curve[rpm] = {
                'torque': point['torque'],
                'source': 'Calibrated'
            }

        # 排序
        sorted_curve = sorted(merged_curve.items())

        output = {
            'vehicle': 'vehicle.audi.etron',
            'calibration_throttle': 0.8,
            'vehicle_parameters': {
                'mass_kg': self.mass,
                'gear_ratio': self.total_gear_ratio,
                'wheel_radius_m': self.wheel_radius
            },
            'raw_measurements': results,
            'merged_torque_curve': [
                {'rpm': rpm, 'torque_nm': data['torque'], 'source': data['source']}
                for rpm, data in sorted_curve
            ]
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n标定结果已保存到: {filename}")
        print(f"  合并后的扭矩曲线包含 {len(sorted_curve)} 个数据点")


def main():
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
        # 创建标定器
        calibrator = TorqueCurveCalibrator(vehicle, world)

        # 执行标定
        results = calibrator.calibrate_full_curve(throttle_value=0.8)

        # 保存结果
        calibrator.save_calibration_results(results)

        # 输出总结
        print(f"\n{'='*60}")
        print(f"标定完成总结")
        print(f"{'='*60}")
        print(f"\nRPM范围覆盖:")
        rpms = [r['rpm'] for r in results]
        print(f"  最小RPM: {min(rpms):.0f}")
        print(f"  最大RPM: {max(rpms):.0f}")
        print(f"  数据点数: {len(rpms)}")

        print(f"\n平均误差:")
        errors = [abs(r['error_percent']) for r in results]
        print(f"  平均绝对误差: {np.mean(errors):.1f}%")
        print(f"  最大绝对误差: {np.max(errors):.1f}%")

    finally:
        vehicle.destroy()


if __name__ == "__main__":
    main()
