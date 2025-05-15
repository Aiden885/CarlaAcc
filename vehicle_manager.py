import carla
import math


class VehicleManager:
    def __init__(self, world, client):
        self.world = world
        self.client = client
        self.vehicles = []
        self.ego_vehicle = None
        self.target_vehicle = None
        self.tm = None
        self.tm_port = 8000

    def spawn_vehicles(self):
        # 获取蓝图库和地图
        blueprint_library = self.world.get_blueprint_library()
        map = self.world.get_map()

        # 选择车辆蓝图
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        ego_vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]

        # 获取道路上的有效生成点
        spawn_points = map.get_spawn_points()[0]

        # 获取生成点的车道
        waypoint = map.get_waypoint(spawn_points.location)
        lanes = []
        current_waypoint = waypoint
        for _ in range(3):
            lanes.append(current_waypoint)
            current_waypoint = current_waypoint.get_left_lane()
            if current_waypoint is None:
                break

        # 初始化车辆列表
        locations = []

        # 在中间车道生成目标车辆
        if len(lanes) > 1:  # 确保中间车道存在
            target_lane = lanes[1]  # 中间车道
            transform = target_lane.transform
            transform.location.z += 0.5  # 调整高度以避免地面碰撞
            locations.append(transform.location)
            target_vehicle = self.world.spawn_actor(vehicle_bp, transform)
            self.vehicles.append(target_vehicle)
            self.target_vehicle = target_vehicle
        else:
            raise ValueError("无法找到中间车道，无法生成目标车辆")

        # 在同一车道上生成自车，位于目标车辆后方20米
        ego_spawn_point = spawn_points
        ego_spawn_point.location = locations[0]  # 与目标车辆同车道
        ego_spawn_point.location.x -= 20  # 后方20米
        ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, ego_spawn_point)
        self.vehicles.append(ego_vehicle)
        self.ego_vehicle = ego_vehicle

        # 设置交通管理器
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_global_distance_to_leading_vehicle(2.0)
        self.tm.set_synchronous_mode(True)

        # 为目标车辆启用自动驾驶
        self.target_vehicle.set_autopilot(True, self.tm_port)
        self.tm.auto_lane_change(self.target_vehicle, False)

        # 初始速度设置（将在TargetVehicleController中动态调整）
        self.tm.vehicle_percentage_speed_difference(self.target_vehicle, 50.0)

        return self.ego_vehicle, self.target_vehicle

    def destroy_vehicles(self):
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.set_autopilot(False)
                vehicle.destroy()
        self.vehicles = []