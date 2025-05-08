import carla
import random
import math
import time

# 连接到CARLA服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 获取当前世界
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# 获取可用的生成点
spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)

# 生成前车
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
lead_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
print("前车已生成")

# 指定的前后车距离（米）
distance = 8.0

# 获取前车的变换
lead_transform = lead_vehicle.get_transform()
forward_vector = lead_transform.get_forward_vector()

# 计算自车的生成位置（前车后方指定距离）
ego_location = lead_transform.location + carla.Location(
    x=-forward_vector.x * distance,
    y=-forward_vector.y * distance,
    z=0.0
)

# 创建自车的变换
ego_transform = carla.Transform(
    location=ego_location,
    rotation=lead_transform.rotation
)

# 生成自车
ego_vehicle_bp = blueprint_library.find('vehicle.audi.a2')
ego_vehicle = world.spawn_actor(ego_vehicle_bp, ego_transform)
print(f"自车已生成，与前车距离为 {distance} 米")

# 设置前车自动驾驶
lead_vehicle.set_autopilot(True)

try:
    # 主循环
    while True:
        # 更新世界
        world.tick()

        # 计算并打印实际距离
        loc1 = lead_vehicle.get_location()
        loc2 = ego_vehicle.get_location()
        actual_distance = math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2 )
        print(f"两车之间的实际距离: {actual_distance:.2f} 米")

        # 将spectator放置在两车之间
        spectator = world.get_spectator()
        mid_location = carla.Location(
            (loc1.x + loc2.x) / 2,
            (loc1.y + loc2.y) / 2,
            max(loc1.z, loc2.z) + 10.0
        )
        spectator.set_transform(
            carla.Transform(
                mid_location,
                carla.Rotation(pitch=-90, yaw=0, roll=0)
            )
        )

        time.sleep(0.1)

except KeyboardInterrupt:
    # 清理
    print("清理...")
    lead_vehicle.destroy()
    ego_vehicle.destroy()