"""
Vehicle utility functions for ACC system
车辆工具函数模块 - 提供几何计算和车辆状态查询
"""
import math
import carla


class VehicleUtils:
    """车辆相关的静态工具函数"""

    @staticmethod
    def get_vehicle_speed(vehicle):
        """
        计算车辆速度（km/h）

        Args:
            vehicle: CARLA车辆对象

        Returns:
            float: 车速（km/h）
        """
        velocity = vehicle.get_velocity()
        speed_m_s = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_kmh = speed_m_s * 3.6
        return speed_kmh

    @staticmethod
    def get_vehicle_distance(vehicle1, vehicle2):
        """
        计算两个车辆之间的距离（米）

        Args:
            vehicle1: CARLA车辆对象1
            vehicle2: CARLA车辆对象2

        Returns:
            float: 距离（米），如果任一车辆为None则返回inf
        """
        if vehicle1 is None or vehicle2 is None:
            return float('inf')

        loc1 = vehicle1.get_location()
        loc2 = vehicle2.get_location()

        distance = math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)
        return distance

    @staticmethod
    def find_best_target(track_id, projected_points):
        """
        优化的目标选择算法
        选择车道内（Y坐标在[-3, 3]范围内）且最近的目标

        Args:
            track_id: 跟踪目标列表，每个元素为 [x, y, ...]
            projected_points: 投影点列表（当前未使用，保留以兼容）

        Returns:
            int: 最佳目标的索引，如果没有找到则返回-1
        """
        current_target_idx = -1
        min_distance = float('inf')

        for idx in range(len(track_id)):
            if -3 < track_id[idx][1] < 3:  # Y坐标在车道内
                if track_id[idx][0] < min_distance:  # 选择最近的
                    min_distance = track_id[idx][0]
                    current_target_idx = idx

        return current_target_idx

    @staticmethod
    def get_lane_offset(ego_vehicle, world):
        """
        获取车辆相对于车道中心的偏移量

        Args:
            ego_vehicle: 自车CARLA对象
            world: CARLA世界对象

        Returns:
            float: 车道偏移量（米），正值表示偏右，负值表示偏左
        """
        if world is None:
            return 0.0

        carla_map = world.get_map()
        if carla_map is None:
            return 0.0

        vehicle_location = ego_vehicle.get_location()
        current_waypoint = carla_map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if current_waypoint is None:
            return 0.0

        lane_center = current_waypoint.transform.location
        lane_direction = current_waypoint.transform.get_forward_vector()

        to_center_vector = carla.Vector3D(
            lane_center.x - vehicle_location.x,
            lane_center.y - vehicle_location.y,
            0
        )

        right_direction = carla.Vector3D(
            -lane_direction.y,
            lane_direction.x,
            0
        ).make_unit_vector()

        offset = to_center_vector.dot(right_direction)
        return offset