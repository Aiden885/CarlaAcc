import carla
import time


class CutOutScenarioManager:
    """
    管理切出工况：
    - cut_vehicle：初始跟车目标，触发后向旁边车道切出
    - lead_vehicle：始终保持在本车道的前车，切出完成后切换跟车目标到它
    """

    def __init__(self, world, traffic_manager, ego_vehicle, cut_vehicle, lead_vehicle,
                 blueprint_library, trigger_time_s=10.0, change_to_right=True):
        self.world = world
        self.map = world.get_map()
        self.tm = traffic_manager
        self.ego_vehicle = ego_vehicle
        self.cut_vehicle = cut_vehicle      # 将要切出的车（初始跟车目标）
        self.lead_vehicle = lead_vehicle    # 始终留在本车道的前车（切出后跟随它）
        self.trigger_time_s = trigger_time_s
        self.change_to_right = change_to_right  # 优先方向（仍会尝试另一侧）

        self._lane_change_triggered = False
        self._lane_change_completed = False
        self._scenario_completed = False
        self._last_force_time = None
        self._target_lane_ids = []
        self._force_retry_count = 0
        self._ego_lane_id_at_trigger = None

        # TM 设置：禁用自动换道，必要时再启用
        port = self.tm.get_port()
        self.cut_vehicle.set_autopilot(True, port)
        self.lead_vehicle.set_autopilot(True, port)
        self.tm.auto_lane_change(self.cut_vehicle, False)
        self.tm.auto_lane_change(self.lead_vehicle, False)
        self.tm.ignore_lights_percentage(self.cut_vehicle, 100.0)
        self.tm.ignore_lights_percentage(self.lead_vehicle, 100.0)

    def maybe_trigger_cut_out(self, elapsed_time_s: float):
        if not self.cut_vehicle or self._scenario_completed:
            return

        wp = self.map.get_waypoint(self.cut_vehicle.get_location(), project_to_road=True)
        ego_wp = self.map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True)

        # 已触发变道则重试直到离开本车道
        if self._lane_change_triggered:
            if self._last_force_time is None or (elapsed_time_s - self._last_force_time) >= 3.0:
                self._force_both_directions(wp)
                self._last_force_time = elapsed_time_s
            # 如果已离开本车道，停止重试
            if self.has_cut_vehicle_left_lane():
                self._lane_change_triggered = False
                self._lane_change_completed = True
                self._scenario_completed = True
            return
        if self._lane_change_completed:
            self._scenario_completed = True
            return

        if elapsed_time_s < self.trigger_time_s:
            return

        # 记录原车道
        self._ego_lane_id_at_trigger = ego_wp.lane_id if ego_wp else None

        # 收集左右可用车道
        left_wp = wp.get_left_lane()
        right_wp = wp.get_right_lane()
        targets = []
        if right_wp:
            targets.append((True, right_wp))
        if left_wp:
            targets.append((False, left_wp))
        if not targets:
            print(f"⚠️ 切出触发失败：当前车道无可用左右侧车道 (lane_id={wp.lane_id})")
            return

        # 按优先方向排序
        targets.sort(key=lambda t: 0 if t[0] == self.change_to_right else 1)
        self._target_lane_ids = [t[1].lane_id for t in targets]

        # 触发后开启自动换道并同时尝试两个方向
        self.tm.auto_lane_change(self.cut_vehicle, True)
        self._force_both_directions(wp)
        self._lane_change_triggered = True
        self._last_force_time = elapsed_time_s
        print(f"🚗 触发切出车辆强制变道 (t={elapsed_time_s:.1f}s) | from lane {wp.lane_id} "
              f"to targets {self._target_lane_ids}")

    def has_cut_vehicle_left_lane(self, margin: float = 0.1) -> bool:
        """
        切出车是否已经离开与自车相同的lane。
        判定：lane_id 不同且（同road/section），如果记录了原lane_id，则要求不同于原lane_id；
        同时相对自车车道中心的偏移已跨过车道线 margin。
        """
        if not self.cut_vehicle:
            return False
        ego_wp = self.map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True)
        cut_wp = self.map.get_waypoint(self.cut_vehicle.get_location(), project_to_road=True)
        if not ego_wp or not cut_wp:
            return False

        if not (ego_wp.road_id == cut_wp.road_id and ego_wp.section_id == cut_wp.section_id):
            return False
        if self._target_lane_ids:
            return cut_wp.lane_id in self._target_lane_ids

        return False

    def _force_both_directions(self, wp):
        """强制向左变道（放宽安全限制）。"""
        self.tm.auto_lane_change(self.cut_vehicle, True)
        self.tm.ignore_vehicles_percentage(self.cut_vehicle, 100.0)
        self.tm.distance_to_leading_vehicle(self.cut_vehicle, 0.1)

        # 只尝试左侧变道
        left_wp = wp.get_left_lane()
        if left_wp:
            self.tm.force_lane_change(self.cut_vehicle, False)  # False = 向左
            self._force_retry_count += 1
            loc = self.cut_vehicle.get_location()
            print(f"↻ 强制向左变道 | retry={self._force_retry_count} | pos=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f}) | target_lane={left_wp.lane_id}")
        else:
            print(f"❌ 无左侧车道，无法切出 | current_lane={wp.lane_id}")
