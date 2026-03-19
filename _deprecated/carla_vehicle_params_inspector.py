#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 车辆物理参数检查器
用于读取并打印 CARLA 车辆的所有物理参数
帮助确定哪些参数可以从 CARLA 直接获取
"""

import carla
import numpy as np
import json


def inspect_vehicle_physics(vehicle):
    """
    检查并打印车辆的所有物理参数

    参数:
        vehicle: CARLA 车辆对象
    返回:
        dict: 包含所有物理参数的字典
    """
    print(f"\n{'='*80}")
    print(f"CARLA 车辆物理参数检查器")
    print(f"{'='*80}")
    print(f"车辆类型: {vehicle.type_id}")
    print(f"车辆ID: {vehicle.id}")
    print(f"{'='*80}\n")

    params = {}

    # 获取物理控制对象
    physics = vehicle.get_physics_control()

    # ========== 基础参数 ==========
    print("📊 基础物理参数:")
    print("-" * 80)

    # 车辆质量（关键参数！）
    if hasattr(physics, 'mass'):
        params['mass'] = physics.mass
        print(f"  ✅ 车辆质量 (mass):                    {physics.mass:.2f} kg")
    else:
        print(f"  ❌ 车辆质量 (mass):                    不可用")
        params['mass'] = None

    # 质心位置
    if hasattr(physics, 'center_of_mass'):
        com = physics.center_of_mass
        params['center_of_mass'] = {'x': com.x, 'y': com.y, 'z': com.z}
        print(f"  ✅ 质心位置 (center_of_mass):          x={com.x:.3f}, y={com.y:.3f}, z={com.z:.3f} m")
    else:
        print(f"  ❌ 质心位置 (center_of_mass):          不可用")
        params['center_of_mass'] = None

    # 阻力系数
    if hasattr(physics, 'drag_coefficient'):
        params['drag_coefficient'] = physics.drag_coefficient
        print(f"  ✅ 阻力系数 (drag_coefficient):        {physics.drag_coefficient:.4f}")
    else:
        print(f"  ❌ 阻力系数 (drag_coefficient):        不可用")
        params['drag_coefficient'] = None

    # 其他参数
    try:
        params['max_rpm'] = physics.max_rpm
        print(f"  ✅ 最大转速 (max_rpm):                 {physics.max_rpm:.0f} RPM")
    except:
        print(f"  ❌ 最大转速 (max_rpm):                 不可用")
        params['max_rpm'] = None

    try:
        params['moi'] = physics.moi
        print(f"  ✅ 转动惯量 (moi):                     {physics.moi:.2f} kg·m²")
    except:
        print(f"  ❌ 转动惯量 (moi):                     不可用")
        params['moi'] = None

    try:
        params['damping_rate_full_throttle'] = physics.damping_rate_full_throttle
        print(f"  ✅ 全油门阻尼率:                       {physics.damping_rate_full_throttle:.4f}")
    except:
        params['damping_rate_full_throttle'] = None

    try:
        params['damping_rate_zero_throttle_clutch_engaged'] = physics.damping_rate_zero_throttle_clutch_engaged
        print(f"  ✅ 零油门阻尼率(离合器接合):           {physics.damping_rate_zero_throttle_clutch_engaged:.4f}")
    except:
        params['damping_rate_zero_throttle_clutch_engaged'] = None

    try:
        params['damping_rate_zero_throttle_clutch_disengaged'] = physics.damping_rate_zero_throttle_clutch_disengaged
        print(f"  ✅ 零油门阻尼率(离合器分离):           {physics.damping_rate_zero_throttle_clutch_disengaged:.4f}")
    except:
        params['damping_rate_zero_throttle_clutch_disengaged'] = None

    # ========== 传动系统 ==========
    print(f"\n⚙️  传动系统参数:")
    print("-" * 80)

    # 最终传动比
    params['final_ratio'] = physics.final_ratio
    print(f"  ✅ 最终传动比 (final_ratio):           {physics.final_ratio:.4f}")

    # 档位
    params['forward_gears'] = []
    if hasattr(physics, 'forward_gears') and physics.forward_gears:
        print(f"  ✅ 前进档位数:                         {len(physics.forward_gears)}")
        for i, gear in enumerate(physics.forward_gears):
            gear_info = {
                'gear_num': i + 1,
                'ratio': gear.ratio,
                'down_ratio': gear.down_ratio,
                'up_ratio': gear.up_ratio
            }
            params['forward_gears'].append(gear_info)
            print(f"     档位 {i+1}: 传动比={gear.ratio:.4f}, 降档比={gear.down_ratio:.4f}, 升档比={gear.up_ratio:.4f}")
    else:
        print(f"  ❌ 前进档位:                           不可用")

    # ========== 车轮参数 ==========
    print(f"\n🛞 车轮物理参数:")
    print("-" * 80)

    params['wheels'] = []
    wheel_names = ['前左', '前右', '后左', '后右']

    if hasattr(physics, 'wheels') and physics.wheels:
        print(f"  ✅ 车轮数量:                           {len(physics.wheels)}")

        for i, wheel in enumerate(physics.wheels):
            wheel_name = wheel_names[i] if i < len(wheel_names) else f"车轮{i}"
            print(f"\n  --- {wheel_name}轮 ---")

            wheel_info = {'name': wheel_name}

            # 车轮半径（重要！）
            wheel_info['radius'] = wheel.radius / 100.0  # cm → m
            print(f"    半径 (radius):                     {wheel.radius:.2f} cm = {wheel.radius/100.0:.3f} m")

            # 最大制动扭矩（关键参数！）
            wheel_info['max_brake_torque'] = wheel.max_brake_torque
            print(f"    最大制动扭矩 (max_brake_torque):   {wheel.max_brake_torque:.1f} N·m")

            # 手刹扭矩
            wheel_info['max_handbrake_torque'] = wheel.max_handbrake_torque
            print(f"    手刹扭矩 (max_handbrake_torque):   {wheel.max_handbrake_torque:.1f} N·m")

            # 最大转向角
            wheel_info['max_steer_angle'] = wheel.max_steer_angle
            print(f"    最大转向角 (max_steer_angle):      {wheel.max_steer_angle:.2f}°")

            # 轮胎摩擦系数
            wheel_info['tire_friction'] = wheel.tire_friction
            print(f"    轮胎摩擦系数 (tire_friction):      {wheel.tire_friction:.3f}")

            # 阻尼率
            wheel_info['damping_rate'] = wheel.damping_rate
            print(f"    阻尼率 (damping_rate):             {wheel.damping_rate:.3f}")

            # 位置
            pos = wheel.position
            wheel_info['position'] = {'x': pos.x, 'y': pos.y, 'z': pos.z}
            print(f"    位置 (position):                   x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f} cm")

            # 侧向刚度
            try:
                wheel_info['lat_stiff_max_load'] = wheel.lat_stiff_max_load
                wheel_info['lat_stiff_value'] = wheel.lat_stiff_value
                print(f"    侧向刚度值:                        {wheel.lat_stiff_value:.2f}")
            except:
                pass

            # 纵向刚度
            try:
                wheel_info['long_stiff_value'] = wheel.long_stiff_value
                print(f"    纵向刚度值:                        {wheel.long_stiff_value:.2f}")
            except:
                pass

            params['wheels'].append(wheel_info)

    # ========== 发动机扭矩曲线 ==========
    print(f"\n🔧 发动机扭矩曲线:")
    print("-" * 80)

    if hasattr(physics, 'torque_curve') and physics.torque_curve:
        params['torque_curve'] = []
        print(f"  ✅ 扭矩曲线数据点数:                   {len(physics.torque_curve)}")
        print(f"\n  RPM       扭矩 (N·m)")
        print("  " + "-" * 30)

        for point in physics.torque_curve:
            params['torque_curve'].append({'rpm': point.x, 'torque': point.y})
            print(f"  {point.x:6.0f}    {point.y:8.2f}")

        # 计算峰值扭矩
        torques = [p.y for p in physics.torque_curve]
        max_torque = max(torques)
        max_torque_rpm = [p.x for p in physics.torque_curve if p.y == max_torque][0]
        params['max_engine_torque'] = max_torque
        params['max_engine_torque_rpm'] = max_torque_rpm

        print(f"\n  ✅ 峰值扭矩:                           {max_torque:.2f} N·m @ {max_torque_rpm:.0f} RPM")
    else:
        print(f"  ❌ 扭矩曲线:                           不可用")
        params['torque_curve'] = None

    # ========== 计算衍生参数 ==========
    print(f"\n📐 计算衍生参数:")
    print("-" * 80)

    # 总传动比（第一档）
    if params['forward_gears'] and params['final_ratio']:
        total_gear_ratio = params['forward_gears'][0]['ratio'] * params['final_ratio']
        params['total_gear_ratio_first'] = total_gear_ratio
        print(f"  ✅ 总传动比(第1档):                    {total_gear_ratio:.4f}")

    # 理论最大减速度
    if params['wheels'] and params['mass']:
        num_wheels = len(params['wheels'])
        avg_radius = np.mean([w['radius'] for w in params['wheels']])
        total_max_brake_torque = sum([w['max_brake_torque'] for w in params['wheels']])

        # F = T / r
        total_brake_force = total_max_brake_torque / avg_radius

        # a = F / m
        max_decel = total_brake_force / params['mass']
        params['theoretical_max_deceleration'] = max_decel

        print(f"  ✅ 理论最大减速度:                     {max_decel:.2f} m/s²")
        print(f"     (基于: {num_wheels}轮, 总扭矩{total_max_brake_torque:.0f}N·m, 质量{params['mass']:.0f}kg)")

    print(f"\n{'='*80}\n")

    return params


def save_params_to_json(params, filename='vehicle_physics_params.json'):
    """保存参数到 JSON 文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"✅ 参数已保存到: {filename}\n")


def main():
    """主函数：连接 CARLA 并检查车辆参数"""
    spawned_vehicles = []  # 记录自己生成的车辆，用于最后清理

    try:
        # 连接 CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        print("🔍 搜索场景中的车辆...")
        vehicles = world.get_actors().filter('vehicle.*')

        if not vehicles:
            print("⚠️  场景中没有车辆，自动生成测试车辆...\n")

            # 生成测试车辆（使用与主程序相同的车型）
            blueprint_library = world.get_blueprint_library()

            # 尝试生成 Audi e-tron（与主程序一致）
            vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]

            # 获取生成点
            spawn_points = world.get_map().get_spawn_points()
            if spawn_points:
                spawn_transform = spawn_points[0]
                test_vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)

                if test_vehicle:
                    spawned_vehicles.append(test_vehicle)
                    vehicles = [test_vehicle]
                    print(f"✅ 已生成测试车辆: {test_vehicle.type_id}\n")
                else:
                    print("❌ 无法生成测试车辆！")
                    return
            else:
                print("❌ 找不到生成点！")
                return
        else:
            print(f"✅ 找到 {len(vehicles)} 辆车辆\n")

        # 检查每辆车
        all_params = {}
        for i, vehicle in enumerate(vehicles):
            params = inspect_vehicle_physics(vehicle)
            all_params[f"vehicle_{i}_{vehicle.type_id}"] = params

        # 保存到文件
        save_params_to_json(all_params)

        # 生成 Python 配置代码
        print("=" * 80)
        print("📝 生成的 Python 配置代码（可直接复制到 torque_to_throttle_converter.py）:")
        print("=" * 80)

        for key, params in all_params.items():
            if params.get('mass'):
                print(f"\n# {key}")
                print(f"VEHICLE_MASS = {params['mass']:.2f}  # kg")

                if params['wheels']:
                    avg_radius = np.mean([w['radius'] for w in params['wheels']])
                    print(f"WHEEL_RADIUS = {avg_radius:.4f}  # m")

                    avg_brake_torque = np.mean([w['max_brake_torque'] for w in params['wheels']])
                    print(f"MAX_BRAKE_TORQUE_PER_WHEEL = {avg_brake_torque:.1f}  # N·m")

                if params.get('theoretical_max_deceleration'):
                    print(f"THEORETICAL_MAX_DECEL = {params['theoretical_max_deceleration']:.2f}  # m/s²")

                if params.get('total_gear_ratio_first'):
                    print(f"TOTAL_GEAR_RATIO = {params['total_gear_ratio_first']:.4f}")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理自己生成的测试车辆
        if spawned_vehicles:
            print("\n🧹 清理测试车辆...")
            for vehicle in spawned_vehicles:
                vehicle.destroy()
            print("✅ 测试车辆已清理\n")


if __name__ == '__main__':
    main()
