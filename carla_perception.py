#!/usr/bin/env python
"""
CARLA API感知模块
使用CARLA原生API获取前车距离和车道偏移，替代视觉传感器方案
"""

import carla
import math


class CarlaPerception:
    """使用CARLA API的感知模块（精简版）"""

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

        print("✅ CARLA API感知模块初始化完成")

    def get_vehicle_distance(self):
        """
        获取自车与前车的沿车道距离（方案B：waypoint.s差值）

        优先使用OpenDRIVE的s坐标计算沿道路距离，
        跨路段时fallback到欧几里得距离

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
            # 同一road section，使用s坐标差值（沿道路距离）
            distance = target_wp.s - ego_wp.s

            # 处理负值情况（target在ego后方）
            if distance < 0:
                # 如果s坐标差为负，说明target在后方，使用绝对值
                distance = abs(distance)
        else:
            # 跨road/section，使用欧几里得距离作为近似
            distance = ego_location.distance(target_location)

        return distance

    def get_lane_offset(self):
        """
        计算自车相对车道中心的横向偏移（用于PID转向控制）

        使用投影方法计算横向偏移，符号约定：左负右正

        Returns:
            float: 横向偏移量（米），左负右正
        """
        ego_location = self.ego_vehicle.get_location()

        # 获取车道中心waypoint（投影到最近车道）
        lane_waypoint = self.map.get_waypoint(
            ego_location,
            project_to_road=True
        )

        # 计算横向偏移
        lane_center = lane_waypoint.transform.location
        dx = ego_location.x - lane_center.x
        dy = ego_location.y - lane_center.y

        # 投影到道路垂直方向（计算真实横向偏移）
        lane_yaw = math.radians(lane_waypoint.transform.rotation.yaw)
        lateral_offset = -dx * math.sin(lane_yaw) + dy * math.cos(lane_yaw)

        return lateral_offset

    def get_lane_angle_deviation(self):
        """
        计算车辆朝向与道路切线的角度差（可选，用于高级控制）

        Returns:
            float: 角度差（度），左负右正
        """
        ego_location = self.ego_vehicle.get_location()
        ego_rotation = self.ego_vehicle.get_transform().rotation

        # 获取车道waypoint
        lane_waypoint = self.map.get_waypoint(ego_location, project_to_road=True)

        # 计算角度差
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
        验证ego和target是否在同一车道（调试用）

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
            'euclidean_distance': ego_location.distance(target_location)
        }


if __name__ == '__main__':
    # 简单测试（需要CARLA服务器运行）
    print("CARLA Perception Module - 独立测试模式")
    print("请确保CARLA服务器正在运行")

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        # 获取所有车辆
        vehicles = world.get_actors().filter('vehicle.*')

        if len(vehicles) >= 2:
            ego = vehicles[0]
            target = vehicles[1]

            perception = CarlaPerception(world, ego, target)
            debug_info = perception.get_debug_info()

            print("\n=== CARLA Perception Debug Info ===")
            for key, value in debug_info.items():
                print(f"{key:20s}: {value}")
        else:
            print("⚠️ 需要至少2辆车进行测试")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请确保CARLA服务器正在运行")