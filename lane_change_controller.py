"""
前车换道控制器
使用CARLA Traffic Manager的force_lane_change实现前车换道控制
"""

import time


class LaneChangeController:
    """
    换道控制器

    功能：控制前车通过Traffic Manager进行换道
    在换道期间禁用constant velocity，使用TM保持目标速度
    """

    def __init__(self, traffic_manager, target_vehicle):
        """
        初始化换道控制器

        Args:
            traffic_manager: CARLA Traffic Manager实例
            target_vehicle: 目标车辆（前车）
        """
        self.tm = traffic_manager
        self.vehicle = target_vehicle

        # 换道状态管理
        self.is_changing_lane = False
        self.lane_change_start_time = None
        self.lane_change_duration = 5.0  # 换道持续时间（秒）
        self.target_speed_during_change = 50.0  # 换道时的目标速度

    def change_lane_left(self, current_target_speed_kmh):
        """
        向左换道

        Args:
            current_target_speed_kmh: 当前目标速度 (km/h)
        """
        # 记录目标速度
        self.target_speed_during_change = current_target_speed_kmh

        # 禁用constant velocity
        self.vehicle.disable_constant_velocity()

        # 设置TM速度控制：让车辆尽量保持目标速度
        speed_limit = self.vehicle.get_speed_limit()
        if speed_limit and speed_limit > 0:
            percentage_diff = ((speed_limit - current_target_speed_kmh) / speed_limit) * 100.0
            self.tm.vehicle_percentage_speed_difference(self.vehicle, percentage_diff)

        # 降低安全检查，提高换道成功率
        self.tm.distance_to_leading_vehicle(self.vehicle, 0.5)  # 减小安全距离
        self.tm.ignore_vehicles_percentage(self.vehicle, 30.0)  # 部分忽略周围车辆

        # 发送换道指令
        self.tm.force_lane_change(self.vehicle, False)  # to_right=False表示向左

        # 标记换道状态
        self.is_changing_lane = True
        self.lane_change_start_time = time.time()

        print(f"\n🚗 前车换道指令: 向左换道 (TM速度控制: {current_target_speed_kmh:.1f} km/h)")

    def change_lane_right(self, current_target_speed_kmh):
        """
        向右换道

        Args:
            current_target_speed_kmh: 当前目标速度 (km/h)
        """
        # 记录目标速度
        self.target_speed_during_change = current_target_speed_kmh

        # 禁用constant velocity
        self.vehicle.disable_constant_velocity()

        # 设置TM速度控制：让车辆尽量保持目标速度
        speed_limit = self.vehicle.get_speed_limit()
        if speed_limit and speed_limit > 0:
            percentage_diff = ((speed_limit - current_target_speed_kmh) / speed_limit) * 100.0
            self.tm.vehicle_percentage_speed_difference(self.vehicle, percentage_diff)

        # 降低安全检查，提高换道成功率
        self.tm.distance_to_leading_vehicle(self.vehicle, 0.5)  # 减小安全距离
        self.tm.ignore_vehicles_percentage(self.vehicle, 30.0)  # 部分忽略周围车辆

        # 发送换道指令
        self.tm.force_lane_change(self.vehicle, True)  # to_right=True表示向右

        # 标记换道状态
        self.is_changing_lane = True
        self.lane_change_start_time = time.time()

        print(f"\n🚗 前车换道指令: 向右换道 (TM速度控制: {current_target_speed_kmh:.1f} km/h)")

    def update(self):
        """
        更新换道状态
        检查换道是否完成

        Returns:
            bool: True表示换道已完成，需要恢复constant velocity
        """
        if not self.is_changing_lane:
            return False

        # 检查换道时间是否到达
        elapsed = time.time() - self.lane_change_start_time
        if elapsed >= self.lane_change_duration:
            self.is_changing_lane = False
            print("✅ 前车换道完成，恢复constant velocity控制")
            return True

        return False

    def is_lane_changing(self):
        """
        检查是否正在换道

        Returns:
            bool: True表示正在换道
        """
        return self.is_changing_lane
