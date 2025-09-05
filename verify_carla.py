import carla
import random
import time


def main():
    # 用于存储我们生成的actor的列表
    actor_list = []

    try:
        # 步骤 1: 连接到 CARLA 服务器
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)  # 设置5秒超时

        print("成功连接到 CARLA 服务器！")

        # 步骤 2: 获取世界对象
        world = client.get_world()
        current_map = world.get_map()
        print(f"当前地图名称: {current_map.name}")

        # 获取上帝视角（Spectator）的控制器
        spectator = world.get_spectator()

        # 步骤 3: 获取车辆蓝图并准备生成车辆
        blueprint_library = world.get_blueprint_library()

        # 优先尝试寻找特斯拉 Model 3，如果找不到，则随机选择一辆车
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        if not vehicle_bp:
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

        # 从地图的推荐出生点中随机选择一个
        spawn_points = current_map.get_spawn_points()
        if not spawn_points:
            print("当前地图中没有可用的出生点！")
            return

        spawn_point = random.choice(spawn_points)

        # 尝试生成车辆
        print("正在尝试生成一辆汽车...")
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        if vehicle is not None:
            actor_list.append(vehicle)
            print(f"成功生成车辆！ ID: {vehicle.id}, 类型: {vehicle.type_id}")

            # 步骤 4: 验证车辆状态
            location = vehicle.get_location()
            print(f"车辆已生成在坐标: X={location.x:.2f}, Y={location.y:.2f}, Z={location.z:.2f}")

            # --- 新增代码：调整视角 ---
            print("正在调整视角...")
            # 获取车辆的变换信息（位置和旋转）
            vehicle_transform = vehicle.get_transform()
            # 计算一个合适的摄像机位置：在车辆后方10米，上方5米
            # 我们通过车辆的“前向矢量”来确定“后方”在哪里
            spectator_location = vehicle_transform.location - 10 * vehicle_transform.get_forward_vector() + carla.Location(
                z=5)
            # 创建一个新的变换给摄像机，让它朝向车辆的方向并稍微向下看
            spectator_transform = carla.Transform(spectator_location,
                                                  carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw))
            # 将摄像机设置到新的位置和角度
            spectator.set_transform(spectator_transform)
            print("视角调整完成。")
            # --- 新增代码结束 ---

            # 步骤 5: 短暂等待，以便在模拟器窗口中观察
            print("脚本将等待10秒钟，请在CARLA窗口中查看车辆...")
            time.sleep(10)  # 延长等待时间以便观察
            print("等待结束。")

        else:
            print("生成车辆失败。请检查服务器终端是否有错误信息。")
            print("可能的原因是该位置有碰撞物，或资源加载问题。")

    except Exception as e:
        print(f"连接或执行过程中发生错误: {e}")
        print("请确保你已经启动了CARLA服务器 (运行 CarlaUE4.exe)。")

    finally:
        # 步骤 6: 清理环境，销毁所有生成的actor
        print("正在销毁已生成的actor...")
        if actor_list:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print("清理完成。脚本退出。")


if __name__ == '__main__':
    main()