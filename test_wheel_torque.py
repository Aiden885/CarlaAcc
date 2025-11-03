#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 WheelPhysicsControl 的 torque 参数
验证是否可以直接控制车轮扭矩
"""

import carla
import time
import math
import sys
import io

# 设置控制台输出为UTF-8（Windows）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def get_vehicle_speed(vehicle):
    """获取车辆速度 (km/h)"""
    velocity = vehicle.get_velocity()
    speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed_ms * 3.6


def update_spectator(world, vehicle):
    """更新观察者位置，跟随车辆 - 侧面跟随视角"""
    spectator = world.get_spectator()
    vehicle_transform = vehicle.get_transform()

    # 侧面跟随视角：从车辆左侧5米、上方3米观察
    spectator_transform = carla.Transform(
        vehicle_transform.location + carla.Location(x=-5, y=-5, z=3),
        carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw + 45)
    )
    spectator.set_transform(spectator_transform)


def main():
    print("\n" + "="*60)
    print("测试 WheelPhysicsControl 车轮扭矩控制")
    print("="*60)

    # 连接CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print("✅ 成功连接到CARLA服务器")
    except Exception as e:
        print(f"❌ 连接CARLA失败: {e}")
        return

    # 生成测试车辆
    vehicle = None
    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]

        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"✅ 生成测试车辆: {vehicle.type_id}")
        print(f"   车辆ID: {vehicle.id}")

        vehicle.set_simulate_physics(True)

        print("设置观察者视角...")
        update_spectator(world, vehicle)

        for _ in range(20):
            world.tick()

        print("✅ 观察者已就位，请查看CARLA窗口\n")

    except Exception as e:
        print(f"❌ 生成车辆失败: {e}")
        if vehicle:
            vehicle.destroy()
        return

    try:
        # 先让车辆停稳
        print("让车辆停稳...")
        for _ in range(50):
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0
            vehicle.apply_control(control)
            update_spectator(world, vehicle)
            world.tick()

        time.sleep(1.0)

        # 测试 WheelPhysicsControl 扭矩控制
        print("\n" + "="*60)
        print("开始测试 WheelPhysicsControl 扭矩控制")
        print("="*60)

        # 1. 获取车辆当前的物理设置
        try:
            physics_control = vehicle.get_physics_control()
            print("✅ 成功获取车辆物理控制")
            print(f"   当前车轮数量: {len(physics_control.wheels)}")

            # 打印当前车轮配置
            print("\n当前车轮物理参数:")
            for i, wheel in enumerate(physics_control.wheels):
                print(f"   车轮 {i}: tire_friction={wheel.tire_friction}, "
                      f"max_steer_angle={wheel.max_steer_angle}, "
                      f"radius={wheel.radius}")

        except Exception as e:
            print(f"❌ 获取物理控制失败: {e}")
            return

        # 2. 关闭自动变速箱
        print("\n关闭自动变速箱...")
        physics_control.use_gear_autobox = False

        # 3. 检查 WheelPhysicsControl 是否有 torque 参数
        print("\n检查 WheelPhysicsControl 的可用参数...")
        test_wheel = carla.WheelPhysicsControl()
        available_params = dir(test_wheel)
        print(f"   WheelPhysicsControl 可用属性: {[p for p in available_params if not p.startswith('_')]}")

        if 'torque' in available_params:
            print("   ✅ 找到 'torque' 参数！")
        else:
            print("   ❌ 未找到 'torque' 参数")
            print("   可能的原因：")
            print("   1. CARLA版本不支持此参数")
            print("   2. 该参数名称可能不同")
            return

        # 4. 定义测试扭矩 (N·m)
        target_torque_rl = 500.0  # 后左轮
        target_torque_rr = 500.0  # 后右轮

        print(f"\n准备施加扭矩:")
        print(f"   后左轮: {target_torque_rl} N·m")
        print(f"   后右轮: {target_torque_rr} N·m")
        print(f"   前轮: 0.0 N·m (假设后驱)")

        # 5. 创建车轮控制对象
        try:
            wheel_fl = carla.WheelPhysicsControl(
                torque=0.0,
                steer_angle=0.0,
                brake_torque=0.0
            )
            wheel_fr = carla.WheelPhysicsControl(
                torque=0.0,
                steer_angle=0.0,
                brake_torque=0.0
            )
            wheel_rl = carla.WheelPhysicsControl(
                torque=target_torque_rl,
                steer_angle=0.0,
                brake_torque=0.0
            )
            wheel_rr = carla.WheelPhysicsControl(
                torque=target_torque_rr,
                steer_angle=0.0,
                brake_torque=0.0
            )
            print("✅ 成功创建车轮控制对象")

        except TypeError as e:
            print(f"❌ 创建车轮控制对象失败: {e}")
            print("   可能的原因：")
            print("   1. WheelPhysicsControl 的参数名称或类型不正确")
            print("   2. 需要提供更多参数")

            # 尝试使用原有参数 + 新参数
            print("\n尝试保留原有参数，仅修改扭矩...")
            try:
                wheel_fl = physics_control.wheels[0]
                wheel_fr = physics_control.wheels[1]
                wheel_rl = physics_control.wheels[2]
                wheel_rr = physics_control.wheels[3]

                # 尝试直接设置 torque 属性
                wheel_rl.torque = target_torque_rl
                wheel_rr.torque = target_torque_rr
                print("✅ 尝试直接设置 torque 属性")

            except Exception as e2:
                print(f"❌ 也失败了: {e2}")
                return

        # 6. 将车轮列表打包
        physics_control.wheels = [wheel_fl, wheel_fr, wheel_rl, wheel_rr]

        # 7. 应用物理设置
        print("\n应用车轮扭矩控制...")
        try:
            vehicle.apply_physics_control(physics_control)
            print("✅ 成功应用物理控制")
        except Exception as e:
            print(f"❌ 应用物理控制失败: {e}")
            return

        # 8. 观察车辆运动
        print("\n持续10秒施加扭矩，观察车辆运动...")
        start_time = time.time()
        initial_speed = get_vehicle_speed(vehicle)
        frame_count = 0

        while time.time() - start_time < 10.0:
            try:
                # 每帧重新应用物理控制
                vehicle.apply_physics_control(physics_control)

                update_spectator(world, vehicle)
                world.tick()

                frame_count += 1

                # 每0.5秒输出一次状态
                elapsed = time.time() - start_time
                if int(elapsed * 2) != int((elapsed - 0.01) * 2):
                    speed = get_vehicle_speed(vehicle)
                    location = vehicle.get_transform().location
                    print(f"  t={elapsed:.1f}s: 速度={speed:.2f} km/h, "
                          f"位置=({location.x:.1f}, {location.y:.1f})")

            except Exception as e:
                print(f"❌ 执行过程中出错: {e}")
                import traceback
                traceback.print_exc()
                break

        final_speed = get_vehicle_speed(vehicle)
        final_location = vehicle.get_transform().location

        print(f"\n结果:")
        print(f"  总帧数: {frame_count}")
        print(f"  速度变化: {initial_speed:.2f} → {final_speed:.2f} km/h")
        print(f"  位移: {math.sqrt(final_location.x**2 + final_location.y**2):.2f} m")

        if abs(final_speed - initial_speed) > 1.0:
            print("\n✅ WheelPhysicsControl 扭矩控制有效！车辆产生了运动")
        else:
            print("\n❌ WheelPhysicsControl 扭矩控制似乎没有效果")
            print("   可能的原因：")
            print("   1. 扭矩值太小")
            print("   2. 需要同时设置其他参数（如手刹、档位等）")
            print("   3. CARLA版本不支持此功能")

    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if vehicle is not None:
            vehicle.destroy()
            print("\n✅ 测试车辆已销毁")


if __name__ == "__main__":
    main()
