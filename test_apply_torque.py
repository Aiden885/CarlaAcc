#!/usr/bin/env python3
"""
测试 CARLA 的 command.ApplyTorque
"""

import carla
import time
import math

def get_vehicle_speed(vehicle):
    """获取车辆速度 (km/h)"""
    velocity = vehicle.get_velocity()
    speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed_ms * 3.6

def update_spectator(world, vehicle):
    """更新观察者位置，跟随车辆 - 鸟瞰视角（固定朝向）"""
    spectator = world.get_spectator()
    vehicle_transform = vehicle.get_transform()

    # 鸟瞰视角：从上方20米向下看，朝向固定（不跟随车辆旋转）
    spectator_transform = carla.Transform(
        vehicle_transform.location + carla.Location(z=20),
        carla.Rotation(pitch=-90, yaw=0)  # yaw固定为0，不跟随车辆
    )
    spectator.set_transform(spectator_transform)

def main():
    print("\n" + "="*60)
    print("测试 command.ApplyTorque")
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
        # 先让车辆停止
        print("让车辆停止...")
        for _ in range(50):
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0
            vehicle.apply_control(control)
            update_spectator(world, vehicle)
            world.tick()

        time.sleep(1.0)

        # 测试 command.ApplyTorque
        print("\n" + "="*60)
        print("开始测试 command.ApplyTorque")
        print("="*60)

        initial_speed = get_vehicle_speed(vehicle)
        print(f"初始速度: {initial_speed:.2f} km/h")

        # 创建扭矩向量 (绕Z轴施加扭矩，使车辆旋转)
        torque_z = 10000000000000000.0  # 尝试不同的值
        torque = carla.Vector3D(x=0, y=0, z=torque_z)

        print(f"施加扭矩: {torque}")
        print(f"   车辆ID: {vehicle.id}")

        # 创建 ApplyTorque 命令
        try:
            command = carla.command.ApplyTorque(vehicle.id, torque)
            print("✅ 成功创建 ApplyTorque 命令")
        except Exception as e:
            print(f"❌ 创建命令失败: {e}")
            return

        # 执行命令
        print("\n持续5秒施加扭矩...")
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < 30.0:
            try:
                # 每帧都施加扭矩
                response = client.apply_batch_sync([command], True)

                # 检查响应
                if response and len(response) > 0:
                    if response[0].error:
                        print(f"❌ 命令执行错误: {response[0].error}")
                        break

                update_spectator(world, vehicle)
                world.tick()

                frame_count += 1

                # 每0.5秒输出一次状态
                elapsed = time.time() - start_time
                if int(elapsed * 2) != int((elapsed - 0.01) * 2):
                    speed = get_vehicle_speed(vehicle)
                    rotation = vehicle.get_transform().rotation
                    print(f"  t={elapsed:.1f}s: 速度={speed:.2f} km/h, Yaw={rotation.yaw:.1f}°")

            except Exception as e:
                print(f"❌ 执行命令时出错: {e}")
                import traceback
                traceback.print_exc()
                break

        final_speed = get_vehicle_speed(vehicle)
        final_rotation = vehicle.get_transform().rotation

        print(f"\n结果:")
        print(f"  总帧数: {frame_count}")
        print(f"  速度变化: {initial_speed:.2f} → {final_speed:.2f} km/h")
        print(f"  最终Yaw角: {final_rotation.yaw:.1f}°")

        if abs(final_rotation.yaw) > 1.0 or abs(final_speed - initial_speed) > 1.0:
            print("\n✅ ApplyTorque 命令有效！车辆产生了运动")
        else:
            print("\n❌ ApplyTorque 命令似乎没有效果")

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
