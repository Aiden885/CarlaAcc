#!/usr/bin/env python
"""
CARLA API感知模块 - 修复版
使用CARLA官方推荐的向量方法计算横向偏移和航向误差
"""

import carla
import math


class CarlaPerception:
    """使用CARLA API的感知模块（修复版）"""

    def __init__(self, world, ego_vehicle, target_vehicle):
        """
        初始化CARLA API感知模块

        Args:
            world: CARLA世界对象
            ego_vehicle: 自车（carla.Vehicle）
            target_vehicle: 前车目标（carla.Vehicle）
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.target_vehicle = target_vehicle
        self.map = world.get_map()

        print("✅ CARLA API感知模块初始化完成（修复版）")

    def get_vehicle_distance(self):
        """
        获取自车与前车的沿车道距离（方案B：waypoint.s差值）

        Returns:
            float: 距离（米）
        """
        ego_location = self.ego_vehicle.get_location()
        target_location = self.target_vehicle.get_location()

        # 获取两车的waypoint
        ego_wp = self.map.get_waypoint(ego_location, project_to_road=True)
        target_wp = self.map.get_waypoint(target_location, project_to_road=True)

        # 优先使用waypoint.s计算沿道路距离
        if (ego_wp.road_id == target_wp.road_id and
            ego_wp.section_id == target_wp.section_id):
            distance = target_wp.s - ego_wp.s
            if distance < 0:
                distance = abs(distance)
        else:
            distance = ego_location.distance(target_location)

        return distance

    def get_lane_offset(self):
        """
        计算自车相对车道中心的横向偏移（用于PID转向控制）

        Returns:
            float: 横向偏移量（米），左负右正
        """
        ego_location = self.ego_vehicle.get_location()
        waypoint = self.map.get_waypoint(ego_location, project_to_road=True)

        # 使用CARLA官方推荐的向量方法
        lane_center = waypoint.transform.location
        forward = waypoint.transform.get_forward_vector()

        # 计算右向量（垂直于前向）
        right = carla.Vector3D(x=-forward.y, y=forward.x, z=0)
        right_length = math.sqrt(right.x**2 + right.y**2)

        if right_length > 1e-6:
            right_normalized = carla.Vector3D(
                x=right.x / right_length,
                y=right.y / right_length,
                z=0
            )
        else:
            right_normalized = carla.Vector3D(x=0, y=1, z=0)

        # 计算车辆位置到车道中心的向量
        diff = carla.Vector3D(
            x=ego_location.x - lane_center.x,
            y=ego_location.y - lane_center.y,
            z=0
        )

        # 投影到右向量上得到横向偏移
        # 正值表示车在车道中心右侧，负值表示在左侧
        lateral_offset = diff.x * right_normalized.x + diff.y * right_normalized.y

        return lateral_offset

    def get_front_axle_offset(self):
        """
        计算前轴中心相对车道中心的横向偏移（Stanley算法专用）
        使用CARLA官方推荐的向量方法

        Returns:
            float: 横向偏移（米），左负右正
        """
        # 获取车辆物理参数
        vehicle_physics = self.ego_vehicle.get_physics_control()
        front_axle_distance = vehicle_physics.wheels[0].position.x / 100.0  # cm -> m

        # 获取车辆变换信息
        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_yaw = math.radians(ego_transform.rotation.yaw)

        # 计算前轴位置
        front_axle_x = ego_location.x + front_axle_distance * math.cos(ego_yaw)
        front_axle_y = ego_location.y + front_axle_distance * math.sin(ego_yaw)
        front_axle_location = carla.Location(x=front_axle_x, y=front_axle_y, z=ego_location.z)

        # 获取前轴位置的waypoint
        front_axle_wp = self.map.get_waypoint(
            front_axle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        # 使用向量方法计算横向偏移
        lane_center = front_axle_wp.transform.location
        forward = front_axle_wp.transform.get_forward_vector()

        # 计算右向量
        right = carla.Vector3D(x=-forward.y, y=forward.x, z=0)
        right_length = math.sqrt(right.x**2 + right.y**2)

        if right_length > 1e-6:
            right_normalized = carla.Vector3D(
                x=right.x / right_length,
                y=right.y / right_length,
                z=0
            )
        else:
            right_normalized = carla.Vector3D(x=0, y=1, z=0)

        # 前轴到车道中心的向量
        diff = carla.Vector3D(
            x=front_axle_x - lane_center.x,
            y=front_axle_y - lane_center.y,
            z=0
        )

        # 投影到右向量
        lateral_offset = diff.x * right_normalized.x + diff.y * right_normalized.y

        return lateral_offset

    def get_heading_error(self):
        """
        计算车辆朝向与道路切线的角度差（Stanley算法专用）

        使用前轴位置获取路径航向，提前感知弯道

        Returns:
            float: 航向误差（弧度），路径航向 - 车辆航向
        """
        # 获取车辆物理参数
        vehicle_physics = self.ego_vehicle.get_physics_control()
        front_axle_distance = vehicle_physics.wheels[0].position.x / 100.0

        # 获取车辆变换信息
        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_yaw = math.radians(ego_transform.rotation.yaw)

        # 计算前轴位置
        front_axle_x = ego_location.x + front_axle_distance * math.cos(ego_yaw)
        front_axle_y = ego_location.y + front_axle_distance * math.sin(ego_yaw)
        front_axle_location = carla.Location(x=front_axle_x, y=front_axle_y, z=ego_location.z)

        # 获取前轴位置的waypoint
        waypoint = self.map.get_waypoint(
            front_axle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        # 获取路径朝向
        path_yaw = math.radians(waypoint.transform.rotation.yaw)

        # 计算航向误差
        heading_error = path_yaw - ego_yaw

        # 归一化到[-π, π]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        return heading_error

    def get_lane_angle_deviation(self):
        """
        计算车辆朝向与道路切线的角度差（度）

        Returns:
            float: 角度差（度），左负右正
        """
        ego_location = self.ego_vehicle.get_location()
        ego_rotation = self.ego_vehicle.get_transform().rotation

        lane_waypoint = self.map.get_waypoint(ego_location, project_to_road=True)

        ego_yaw = ego_rotation.yaw
        lane_yaw = lane_waypoint.transform.rotation.yaw

        angle_deviation = ego_yaw - lane_yaw

        # 归一化到[-180, 180]
        while angle_deviation > 180:
            angle_deviation -= 360
        while angle_deviation < -180:
            angle_deviation += 360

        return angle_deviation

    def validate_same_lane(self):
        """
        验证ego和target是否在同一车道

        Returns:
            bool: True if same lane
        """
        ego_wp = self.map.get_waypoint(self.ego_vehicle.get_location())
        target_wp = self.map.get_waypoint(self.target_vehicle.get_location())

        same_lane = (ego_wp.lane_id == target_wp.lane_id and
                     ego_wp.road_id == target_wp.road_id)

        return same_lane

    def get_debug_info(self):
        """
        获取调试信息（可选）

        Returns:
            dict: 包含详细感知信息的字典
        """
        ego_location = self.ego_vehicle.get_location()
        target_location = self.target_vehicle.get_location()

        ego_wp = self.map.get_waypoint(ego_location)
        target_wp = self.map.get_waypoint(target_location)

        return {
            'distance': self.get_vehicle_distance(),
            'lane_offset': self.get_lane_offset(),
            'angle_deviation': self.get_lane_angle_deviation(),
            'same_lane': self.validate_same_lane(),
            'ego_road_id': ego_wp.road_id,
            'ego_section_id': ego_wp.section_id,
            'ego_lane_id': ego_wp.lane_id,
            'ego_s': ego_wp.s,
            'target_road_id': target_wp.road_id,
            'target_section_id': target_wp.section_id,
            'target_lane_id': target_wp.lane_id,
            'target_s': target_wp.s,
            'euclidean_distance': ego_location.distance(target_location),
            'front_axle_offset': self.get_front_axle_offset(),
            'heading_error_deg': math.degrees(self.get_heading_error())
        }


if __name__ == '__main__':
    print("CARLA Perception Module - 修复版测试模式")
    print("请确保CARLA服务器正在运行")

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        vehicles = world.get_actors().filter('vehicle.*')

        if len(vehicles) >= 2:
            ego = vehicles[0]
            target = vehicles[1]

            perception = CarlaPerception(world, ego, target)
            debug_info = perception.get_debug_info()

            print("\n=== CARLA Perception Debug Info (修复版) ===")
            for key, value in debug_info.items():
                print(f"{key:20s}: {value}")
        else:
            print("⚠️ 需要至少2辆车进行测试")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请确保CARLA服务器正在运行")