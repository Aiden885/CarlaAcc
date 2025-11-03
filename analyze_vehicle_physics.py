#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析CARLA车辆的物理参数
获取质量、阻力系数、车轮半径、发动机扭矩曲线等
"""

import carla
import sys
import io

# 设置控制台输出为UTF-8（Windows）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def analyze_vehicle_physics(vehicle):
    """分析车辆物理参数"""
    print("\n" + "="*60)
    print("车辆物理参数分析")
    print("="*60)

    # 获取物理控制
    physics = vehicle.get_physics_control()

    # 基本参数
    print("\n【基本参数】")
    print(f"质量 (mass): {physics.mass} kg")
    print(f"重心 (center_of_mass): {physics.center_of_mass}")
    print(f"阻力系数 (drag_coefficient): {physics.drag_coefficient}")

    # 车轮参数
    print("\n【车轮参数】")
    print(f"车轮数量: {len(physics.wheels)}")
    for i, wheel in enumerate(physics.wheels):
        print(f"\n车轮 {i}:")
        print(f"  位置: {wheel.position}")
        print(f"  半径: {wheel.radius} cm")
        print(f"  半径: {wheel.radius / 100.0} m")  # 转换为米
        print(f"  轮胎摩擦系数: {wheel.tire_friction}")
        print(f"  最大制动扭矩: {wheel.max_brake_torque} N·m")
        print(f"  最大手刹扭矩: {wheel.max_handbrake_torque} N·m")
        print(f"  最大转向角: {wheel.max_steer_angle}°")

    # 发动机参数
    print("\n【发动机参数】")
    print(f"最大转速 (max_rpm): {physics.max_rpm} RPM")
    print(f"最大空转转速 (moi): {physics.moi}")
    print(f"阻尼率 (damping_rate_full_throttle): {physics.damping_rate_full_throttle}")
    print(f"零阻尼率 (damping_rate_zero_throttle_clutch_engaged): {physics.damping_rate_zero_throttle_clutch_engaged}")
    print(f"离合器断开阻尼率 (damping_rate_zero_throttle_clutch_disengaged): {physics.damping_rate_zero_throttle_clutch_disengaged}")

    # 发动机扭矩曲线
    print("\n【发动机扭矩曲线】")
    print(f"扭矩曲线点数: {len(physics.torque_curve)}")
    print("RPM → 扭矩(N·m)")
    for point in physics.torque_curve:
        print(f"  {point.x:.0f} RPM → {point.y:.1f} N·m")

    # 获取最大扭矩
    max_torque = max([p.y for p in physics.torque_curve])
    print(f"\n最大发动机扭矩: {max_torque} N·m")

    # 变速箱参数
    print("\n【变速箱参数】")
    print(f"自动变速箱: {physics.use_gear_autobox}")
    print(f"换档时间: {physics.gear_switch_time} s")
    print(f"离合器强度: {physics.clutch_strength} kg·m²/s")
    print(f"最终传动比: {physics.final_ratio}")

    # 前进档传动比
    print(f"\n前进档传动比:")
    for i, ratio in enumerate(physics.forward_gears):
        print(f"  档位 {i+1}: {ratio}")

    # 转向曲线
    print("\n【转向曲线】")
    if hasattr(physics, 'steering_curve'):
        print(f"转向曲线点数: {len(physics.steering_curve)}")
        for point in physics.steering_curve:
            print(f"  速度 {point.x:.1f} → 转向比 {point.y:.3f}")

    return physics


def test_throttle_response(vehicle, world):
    """测试不同油门下的加速度响应"""
    print("\n" + "="*60)
    print("测试油门-加速度响应")
    print("="*60)

    import time
    import math

    # 测试点：不同油门值
    throttle_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []

    for throttle in throttle_values:
        print(f"\n测试油门: {throttle}")

        # 先停车
        for _ in range(30):
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0
            vehicle.apply_control(control)
            world.tick()

        time.sleep(0.5)

        # 记录初始速度
        v_initial = vehicle.get_velocity()
        v_initial_mag = math.sqrt(v_initial.x**2 + v_initial.y**2 + v_initial.z**2)

        # 施加固定油门
        for _ in range(100):  # 5秒（假设20fps）
            control = carla.VehicleControl()
            control.throttle = throttle
            control.brake = 0.0
            vehicle.apply_control(control)
            world.tick()

        # 记录最终速度
        v_final = vehicle.get_velocity()
        v_final_mag = math.sqrt(v_final.x**2 + v_final.y**2 + v_final.z**2)

        # 计算平均加速度
        delta_v = v_final_mag - v_initial_mag
        delta_t = 5.0  # 5秒
        avg_accel = delta_v / delta_t

        results.append({
            'throttle': throttle,
            'v_initial': v_initial_mag,
            'v_final': v_final_mag,
            'accel': avg_accel
        })

        print(f"  初始速度: {v_initial_mag:.2f} m/s")
        print(f"  最终速度: {v_final_mag:.2f} m/s")
        print(f"  平均加速度: {avg_accel:.3f} m/s²")

    print("\n【油门-加速度映射表】")
    print("油门\t加速度(m/s²)")
    for r in results:
        print(f"{r['throttle']:.1f}\t{r['accel']:.3f}")

    return results


def main():
    print("\n启动车辆物理参数分析工具...")

    # 连接CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print("成功连接到CARLA服务器")
    except Exception as e:
        print(f"连接CARLA失败: {e}")
        return

    # 生成测试车辆
    vehicle = None
    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]

        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"生成测试车辆: {vehicle.type_id}")

        vehicle.set_simulate_physics(True)

        # 等待车辆稳定
        for _ in range(20):
            world.tick()

        # 分析物理参数
        physics = analyze_vehicle_physics(vehicle)

        # 测试油门响应（可选，耗时较长）
        print("\n是否测试油门响应？这将需要约1分钟时间。")
        print("（测试将自动进行，无需输入）")

        # 自动进行测试
        # test_results = test_throttle_response(vehicle, world)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if vehicle is not None:
            vehicle.destroy()
            print("\n测试车辆已销毁")


if __name__ == "__main__":
    main()
