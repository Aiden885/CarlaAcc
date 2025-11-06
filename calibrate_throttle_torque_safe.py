#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全的油门-扭矩双变量标定脚本（瞬时测量法）
测试矩阵：速度(5-60 km/h) × 油门(0.1-1.0)

测量方法：
1. 稳定到目标速度（±3 km/h，持续1秒）
2. 突然施加固定油门
3. 瞬时测量0.5秒内的加速度
4. 从加速度反算发动机扭矩

安全特性：
1. 碰撞检测传感器
2. 固定安全位置重生
3. 横向PID控制（防撞护栏）
4. 实时状态监控
5. 自动异常处理
"""

import carla
import math
import time
import numpy as np
import json
import sys
import io

# 设置控制台输出为UTF-8（Windows）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 导入横向PID控制器（从acc_updated.py照搬）
from lateral_pid_controller import LateralPIDController
from vehicle_utils import VehicleUtils


class SafeThrottleTorqueCalibrator:
    """安全的油门-扭矩标定器"""

    def __init__(self, client, world):
        self.client = client
        self.world = world
        self.vehicle = None
        self.collision_sensor = None

        # 碰撞检测
        self.collision_occurred = False
        self.collision_history = []

        # 车辆物理参数
        self.mass = None
        self.drag_coef = None  # 从CARLA获取（纯Cd值）

        # 根据CARLA源码确认的物理常数
        self.air_density = 1.225  # kg/m³ (标准大气压，15°C)

        # Audi e-tron实际参数（查询得到）
        self.frontal_area = 2.65  # m² (Audi e-tron迎风面积)
        self.rolling_resistance = 0.015  # 典型乘用车轮胎值
        self.wheel_radius = None
        self.total_gear_ratio = None

        # 测试配置（双变量标定：速度 × 油门）
        self.test_speeds = list(range(5, 65, 5))  # [5, 10, 15, ..., 60]
        self.test_throttles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # 安全参数
        self.max_retry_per_point = 3

        # 安全位置（将在setup时从spawn_points选择）
        self.safe_spawn = None

        # 横向PID控制器（从acc_updated.py照搬）
        # CARLA API模式：误差单位为米，需要更大的增益
        self.lateral_pid = LateralPIDController(kp=0.02, ki=0.02, kd=0.4)

        print(f"\n{'='*70}")
        print(f"安全油门-扭矩标定器初始化（瞬时测量法）")
        print(f"{'='*70}")
        print(f"测试速度: {len(self.test_speeds)} 个点 ({min(self.test_speeds)}-{max(self.test_speeds)} km/h)")
        print(f"测试油门: {len(self.test_throttles)} 个点 (0.1-1.0)")
        print(f"总测试点: {len(self.test_speeds) * len(self.test_throttles)}")
        print(f"测量方法: 稳定速度 → 施加油门 → 瞬时测量(0.5s)")
        print(f"预计时间: 10-15 分钟")
        print(f"{'='*70}\n")

    def setup_vehicle(self):
        """生成车辆并配置传感器"""
        print("正在生成测试车辆...")

        # 生成车辆
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]

        # 使用指定的安全位置（Town04中间车道，远离墙壁）
        # 用户提供的安全坐标
        safe_location = carla.Location(x=-343.766754, y=33.918137, z=1.099138)

        # 在这个位置附近找到最近的waypoint
        map_obj = self.world.get_map()
        waypoint = map_obj.get_waypoint(safe_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        # 使用waypoint的transform作为生成点
        self.safe_spawn = waypoint.transform

        print(f"✓ 使用指定安全位置: x={self.safe_spawn.location.x:.2f}, y={self.safe_spawn.location.y:.2f}")

        # 生成车辆
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, self.safe_spawn)

        if self.vehicle is None:
            # 如果指定位置失败，尝试使用地图的spawn点
            print("  指定位置失败，尝试其他spawn点...")
            spawn_points = map_obj.get_spawn_points()
            for i, spawn_point in enumerate(spawn_points):
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if self.vehicle is not None:
                    self.safe_spawn = spawn_point
                    print(f"✓ 使用备用spawn点 #{i}")
                    break

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle at any spawn point")

        self.vehicle.set_simulate_physics(True)

        # 等待稳定
        for _ in range(20):
            self.world.tick()

        # 加载物理参数
        physics = self.vehicle.get_physics_control()
        self.mass = physics.mass
        self.drag_coef = physics.drag_coefficient
        self.wheel_radius = physics.wheels[0].radius / 100.0
        gear_ratio = physics.forward_gears[0].ratio if physics.forward_gears else 1.0
        self.total_gear_ratio = gear_ratio * physics.final_ratio

        # 设置碰撞传感器
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        print(f"✓ 车辆生成成功: {self.vehicle.type_id}")
        print(f"  质量: {self.mass} kg")
        print(f"  传动比: {self.total_gear_ratio:.3f}")
        print(f"  车轮半径: {self.wheel_radius:.3f} m")
        print(f"  阻力系数: {self.drag_coef}")
        print(f"  迎风面积: {self.frontal_area} m²")
        print(f"✓ 碰撞传感器已激活")
        print(f"✓ 横向PID控制器已就绪\n")

    def _on_collision(self, event):
        """碰撞回调"""
        self.collision_occurred = True
        self.collision_history.append({
            'time': time.time(),
            'other_actor': event.other_actor.type_id if event.other_actor else 'Unknown',
            'impulse': math.sqrt(event.normal_impulse.x**2 +
                                event.normal_impulse.y**2 +
                                event.normal_impulse.z**2)
        })
        print(f"⚠️  检测到碰撞！对象: {event.other_actor.type_id if event.other_actor else 'Unknown'}")

    def reset_to_safe_position(self):
        """重置车辆到安全位置"""
        self.vehicle.set_transform(self.safe_spawn)

        # 停止车辆
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        self.vehicle.apply_control(control)

        # 重置碰撞标志
        self.collision_occurred = False

        # 等待稳定
        for _ in range(30):
            self.world.tick()

        time.sleep(0.5)

    def get_current_speed_kmh(self):
        """获取当前速度 (km/h)"""
        velocity = self.vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed_ms * 3.6

    def stabilize_to_speed(self, target_speed_kmh, tolerance=3.0, stable_duration=1.0, timeout=20.0):
        """
        稳定到目标速度（新方案：真正的速度稳定）

        参数:
        target_speed_kmh - 目标速度
        tolerance - 速度容差 (km/h)
        stable_duration - 需要稳定持续的时间 (秒)
        timeout - 超时时间

        返回:
        成功返回True，失败返回False
        """
        start_time = time.time()
        stable_start_time = None

        while time.time() - start_time < timeout:
            current_speed = self.get_current_speed_kmh()
            error = target_speed_kmh - current_speed

            # 检查碰撞
            if self.collision_occurred:
                print(f"  ✗ 稳定速度时发生碰撞")
                return False

            # 检查是否在容差范围内
            if abs(error) < tolerance:
                # 开始计时稳定时长
                if stable_start_time is None:
                    stable_start_time = time.time()
                elif time.time() - stable_start_time >= stable_duration:
                    # 已稳定足够时间
                    return True
            else:
                # 超出容差，重置稳定计时
                stable_start_time = None

            # PID纵向控制（精细控制以稳定速度）
            control = carla.VehicleControl()

            # 比例控制
            kp_speed = 0.05
            if error > 0:
                # 需要加速
                control.throttle = min(0.6, max(0.1, kp_speed * error))
                control.brake = 0.0
            else:
                # 需要减速
                control.throttle = 0.0
                control.brake = min(0.5, max(0.05, -kp_speed * error))

            # 横向控制（修复符号）
            lane_offset = VehicleUtils.get_lane_offset(self.vehicle, self.world)
            # 符号约定：carla_perception返回"左负右正"
            # PID控制器期望：车偏左为负，需要向右转=负转向
            lateral_error =lane_offset  # 反转符号
            steer_output = self.lateral_pid.update(lateral_error, dt=0.05)
            control.steer = steer_output

            self.vehicle.apply_control(control)
            self.world.tick()

        print(f"  ✗ 稳定速度超时")
        return False

    def measure_torque_at_point(self, target_speed_kmh, throttle_value):
        """
        测量单个(速度, 油门)点的扭矩（瞬时测量法）

        新方案：
        1. 稳定到目标速度
        2. 突然施加固定油门
        3. 测量前0.5秒的瞬时加速度（速度变化还很小）
        4. 从加速度反算扭矩

        返回:
        成功返回扭矩值(N·m)，失败返回None
        """
        # 1. 稳定到目标速度
        print(f"    稳定速度到 {target_speed_kmh} km/h...")
        if not self.stabilize_to_speed(target_speed_kmh, tolerance=3.0, stable_duration=1.0):
            return None

        # 2. 记录稳定时的速度（作为基准）
        stable_speed_ms = self.vehicle.get_velocity()
        stable_speed = math.sqrt(stable_speed_ms.x**2 + stable_speed_ms.y**2 + stable_speed_ms.z**2)

        # 3. 突然施加固定油门，测量瞬时加速度
        # 关键：只测量0.5秒，此时速度变化还很小
        instant_measurement_duration = 0.5  # 瞬时测量时长
        start_time = time.time()
        speed_samples = []

        print(f"    施加油门 {throttle_value:.1f}，瞬时测量...")

        while time.time() - start_time < instant_measurement_duration:
            # 检查碰撞
            if self.collision_occurred:
                print(f"    ✗ 测量过程中发生碰撞")
                return None

            # 纵向控制：施加固定油门
            control = carla.VehicleControl()
            control.throttle = throttle_value
            control.brake = 0.0

            # 横向控制（修复符号）
            lane_offset = VehicleUtils.get_lane_offset(self.vehicle, self.world)
            # 符号约定：carla_perception返回"左负右正"
            # PID控制器期望：车偏左为负，需要向右转=负转向
            lateral_error = lane_offset  # 反转符号
            steer_output = self.lateral_pid.update(lateral_error, dt=0.05)
            control.steer = steer_output

            self.vehicle.apply_control(control)

            # 记录速度
            velocity = self.vehicle.get_velocity()
            speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_samples.append({
                'time': time.time() - start_time,
                'speed': speed_ms
            })

            self.world.tick()

        # 4. 数据有效性检查
        if len(speed_samples) < 5:  # 至少5帧数据（0.5秒 / 0.05秒）
            print(f"    ✗ 数据点不足")
            return None

        # 5. 计算瞬时加速度（线性拟合）
        times = np.array([s['time'] for s in speed_samples])
        speeds = np.array([s['speed'] for s in speed_samples])

        # 使用最小二乘法拟合速度曲线
        coeffs = np.polyfit(times, speeds, 1)
        instant_acceleration = coeffs[0]  # m/s² (斜率即为加速度)

        # 6. 使用稳定时的速度作为基准速度（而非测量期间的平均速度）
        # 因为测量时间很短，速度还没有明显变化
        base_speed_kmh = stable_speed * 3.6

        print(f"    测量完成: 基准速度={base_speed_kmh:.1f} km/h, 瞬时加速度={instant_acceleration:.3f} m/s²")

        # 7. 加速度合理性检查
        if abs(instant_acceleration) > 10.0:  # 加速度不应该超过10 m/s²
            print(f"    ✗ 加速度异常: {instant_acceleration:.2f} m/s²")
            return None

        # 8. 从加速度反算扭矩（使用稳定时的速度）
        # 计算阻力
        air_drag = 0.5 * self.air_density * self.drag_coef * \
                   self.frontal_area * (stable_speed ** 2)
        rolling_drag = self.rolling_resistance * self.mass * 9.81

        # 净力
        net_force = instant_acceleration * self.mass

        # 驱动力
        drive_force = net_force + air_drag + rolling_drag

        # 车轮扭矩
        wheel_torque = drive_force * self.wheel_radius

        # 发动机扭矩
        engine_torque = wheel_torque / self.total_gear_ratio

        print(f"    ✓ 计算扭矩: {engine_torque:.1f} N·m")

        return engine_torque

    def calibrate(self):
        """执行完整标定"""
        print(f"\n{'='*70}")
        print(f"开始标定实验")
        print(f"{'='*70}\n")

        results = []
        total_points = len(self.test_speeds) * len(self.test_throttles)
        current_point = 0
        failed_points = 0

        for speed in self.test_speeds:
            print(f"\n--- 测试速度: {speed} km/h ---")

            for throttle in self.test_throttles:
                current_point += 1
                print(f"[{current_point}/{total_points}] 速度={speed} km/h, 油门={throttle:.1f}...", end=' ')

                retry_count = 0
                success = False
                measured_torque = None

                while retry_count < self.max_retry_per_point and not success:
                    # 重置到安全位置
                    if retry_count > 0:
                        print(f"\n    重试 {retry_count}/{self.max_retry_per_point}...")
                        self.reset_to_safe_position()

                    # 测量
                    measured_torque = self.measure_torque_at_point(speed, throttle)

                    if measured_torque is not None:
                        success = True
                    else:
                        retry_count += 1
                        self.reset_to_safe_position()

                if success:
                    results.append({
                        'speed_kmh': speed,
                        'throttle': throttle,
                        'engine_torque_nm': measured_torque
                    })
                    print(f"✓ 扭矩={measured_torque:.1f} N·m")
                else:
                    failed_points += 1
                    print(f"✗ 失败（已重试{self.max_retry_per_point}次）")
                    # 继续下一个点，不中断整个流程

        print(f"\n{'='*70}")
        print(f"标定完成")
        print(f"{'='*70}")
        print(f"成功: {len(results)}/{total_points} 个点")
        print(f"失败: {failed_points}/{total_points} 个点")
        print(f"成功率: {len(results)/total_points*100:.1f}%")

        if self.collision_history:
            print(f"\n碰撞事件: {len(self.collision_history)} 次")

        return results

    def save_results(self, results, filename='throttle_torque_map.json'):
        """保存标定结果"""
        output = {
            'vehicle': 'vehicle.audi.etron',
            'calibration_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'vehicle_parameters': {
                'mass_kg': self.mass,
                'gear_ratio': self.total_gear_ratio,
                'wheel_radius_m': self.wheel_radius
            },
            'test_configuration': {
                'speed_range_kmh': [min(self.test_speeds), max(self.test_speeds)],
                'speed_step_kmh': 5,
                'throttle_values': self.test_throttles,
                'total_points': len(self.test_speeds) * len(self.test_throttles)
            },
            'calibration_data': results,
            'statistics': {
                'total_points': len(self.test_speeds) * len(self.test_throttles),
                'successful_points': len(results),
                'failed_points': len(self.test_speeds) * len(self.test_throttles) - len(results),
                'collision_count': len(self.collision_history)
            }
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ 结果已保存到: {filename}")
        print(f"  数据点: {len(results)}")

    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")

        try:
            # 先停用自动驾驶
            if self.vehicle is not None and self.vehicle.is_alive:
                self.vehicle.set_autopilot(False)
                print("  ✓ 已停用自动驾驶")
        except:
            pass

        try:
            # 停止碰撞传感器
            if self.collision_sensor is not None and self.collision_sensor.is_alive:
                self.collision_sensor.stop()
                print("  ✓ 已停止碰撞传感器")
        except:
            pass

        try:
            # 销毁碰撞传感器
            if self.collision_sensor is not None:
                self.collision_sensor.destroy()
                self.collision_sensor = None
                print("  ✓ 已销毁碰撞传感器")
        except:
            pass

        try:
            # 销毁车辆
            if self.vehicle is not None:
                self.vehicle.destroy()
                self.vehicle = None
                print("  ✓ 已销毁车辆")
        except:
            pass

        print("✓ 资源清理完成")


def main():
    print("\n" + "="*70)
    print("安全油门-扭矩标定实验（瞬时测量法）")
    print("="*70)

    # 连接CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # 指定使用Town04地图
    current_world = client.get_world()
    current_map = current_world.get_map().name

    if 'Town04' not in current_map:
        print(f"\n正在加载地图 Town04...")
        world = client.load_world('Town04')
        print(f"✓ 地图加载完成")
    else:
        print(f"\n使用当前地图: {current_map}")
        world = current_world

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    print("✓ 同步模式已启用\n")

    # 创建标定器
    calibrator = SafeThrottleTorqueCalibrator(client, world)

    try:
        # 设置车辆
        calibrator.setup_vehicle()

        # 执行标定
        results = calibrator.calibrate()

        # 保存结果
        calibrator.save_results(results)

    except KeyboardInterrupt:
        print("\n\n用户中断实验")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        calibrator.cleanup()

        # 恢复异步模式
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("✓ 异步模式已恢复")


if __name__ == "__main__":
    main()
