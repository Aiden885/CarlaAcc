import math

import carla


class CutInScenarioManager:
    """
    管理切入工况：生成侧向车辆、按时间触发强制变道、判断是否进入本车道。
    """

    def __init__(self, world, traffic_manager, ego_vehicle, lead_vehicle,
                 blueprint_library, enable_from_left=True,
                 trigger_time_s=5.0, lateral_threshold_m=0.5,
                 vehicle_blueprint='vehicle.tesla.model3',
                 side_spawn_location: carla.Location = None,
                 side_back_offset: float = 0.0,
                 side_spawn_z_lift: float = 0.0,
                 lane_entry_margin: float = 0.1):
        self.world = world
        self.map = world.get_map()
        self.tm = traffic_manager
        self.ego_vehicle = ego_vehicle
        self.lead_vehicle = lead_vehicle
        self.blueprint_library = blueprint_library
        self.trigger_time_s = trigger_time_s
        self.lateral_threshold_m = lateral_threshold_m
        self.enable_from_left = enable_from_left
        self.vehicle_blueprint = vehicle_blueprint
        self.side_spawn_location = side_spawn_location
        self.side_back_offset = side_back_offset
        self.side_spawn_z_lift = side_spawn_z_lift
        self.lane_entry_margin = lane_entry_margin

        self.cut_in_vehicle = None
        self._lane_change_triggered = False
        self._last_force_time = None
        self._target_lane_id = None
        self._force_retry_count = 0
        self._force_to_right = True  # True 表示向右变道

        self._spawn_side_vehicle()

    # ------------------------------------------------------------------ public API
    def maybe_trigger_cut_in(self, elapsed_time_s: float):
        if not self.cut_in_vehicle:
            return

        wp = self.map.get_waypoint(self.cut_in_vehicle.get_location(), project_to_road=True)
        ego_wp = self.map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True)

        # 如果已触发但尚未到目标车道，则周期性重试
        if self._lane_change_triggered:
            # 已触发变道，检查是否到达目标或自车车道
            if (self._target_lane_id is not None and wp.lane_id == self._target_lane_id) or (
                    ego_wp and wp.lane_id == ego_wp.lane_id):
                return  # 已到达，无需重试
            # 未到达则持续重试（按间隔节流，直到成功）
            if self._last_force_time is None or (elapsed_time_s - self._last_force_time) >= 1.0:
                self.tm.auto_lane_change(self.cut_in_vehicle, True)
                self.tm.ignore_vehicles_percentage(self.cut_in_vehicle, 100.0)
                self.tm.distance_to_leading_vehicle(self.cut_in_vehicle, 0.1)
                self.tm.force_lane_change(self.cut_in_vehicle, self._force_to_right)
                self._last_force_time = elapsed_time_s
                self._force_retry_count += 1
                loc = self.cut_in_vehicle.get_location()
                direction = "右" if self._force_to_right else "左"
                print(f"↻ 重试切入强制向{direction}变道 | lane {wp.lane_id}→{self._target_lane_id} "
                      f"| retry={self._force_retry_count} | pos=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f})")
            return

        if elapsed_time_s < self.trigger_time_s:
            return

        offset_curr = self._lateral_offset_to_lane(self.cut_in_vehicle.get_location(), ego_wp)

        # 评估左右两侧哪个更接近自车车道中心
        left_wp = wp.get_left_lane()
        right_wp = wp.get_right_lane()
        left_offset = None
        right_offset = None
        if left_wp:
            left_offset = self._lateral_offset_to_lane(left_wp.transform.location, ego_wp)
        if right_wp:
            right_offset = self._lateral_offset_to_lane(right_wp.transform.location, ego_wp)

        # 选择绝对偏移更小的方向
        choose_left = False
        choose_right = False
        if left_offset is not None and right_offset is not None:
            choose_left = abs(left_offset) < abs(right_offset)
            choose_right = not choose_left
        elif left_offset is not None:
            choose_left = True
        elif right_offset is not None:
            choose_right = True
        else:
            print(f"⚠️ 切入触发失败：当前车道无可用左右侧车道 (lane_id={wp.lane_id})")
            return

        self._force_to_right = choose_right
        target_wp = right_wp if choose_right else left_wp

        # 临时允许自动换道，确保强制变道生效
        self.tm.auto_lane_change(self.cut_in_vehicle, True)
        self.tm.ignore_vehicles_percentage(self.cut_in_vehicle, 100.0)
        self.tm.distance_to_leading_vehicle(self.cut_in_vehicle, 0.1)
        self.tm.force_lane_change(self.cut_in_vehicle, self._force_to_right)
        self._lane_change_triggered = True
        self._last_force_time = elapsed_time_s
        self._target_lane_id = target_wp.lane_id if target_wp else None
        self._force_retry_count = 0
        direction = "右" if self._force_to_right else "左"
        print(f"🚗 触发切入车辆强制向{direction}变道 (t={elapsed_time_s:.1f}s) | "
              f"from lane {wp.lane_id} to lane {target_wp.lane_id} | "
              f"current_offset={offset_curr:.2f}m | left_offset={left_offset} | right_offset={right_offset}")

    def should_switch_to_cut_in(self) -> bool:
        """
        判断切入车是否已经进入自车当前车道并在前方，满足横向阈值后即可切换目标。
        """
        if not self.cut_in_vehicle:
            return False
        if not self._lane_change_triggered:
            return False

        ego_wp = self.map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True)
        cand_wp = self.map.get_waypoint(self.cut_in_vehicle.get_location(), project_to_road=True)

        # 必须在前方
        s_gap = cand_wp.s - ego_wp.s
        if s_gap <= 0:
            return False

        lateral = self._lateral_offset_to_lane(self.cut_in_vehicle.get_location(), ego_wp)

        same_lane = (ego_wp.road_id == cand_wp.road_id and
                     ego_wp.section_id == cand_wp.section_id and
                     ego_wp.lane_id == cand_wp.lane_id)
        same_road_section = (ego_wp.road_id == cand_wp.road_id and
                             ego_wp.section_id == cand_wp.section_id)

        # 允许提前切换：同一road/section且“距离车道边界≥0.1m”（车身中心已跨过车道线0.1m）
        lane_half_width = getattr(ego_wp, 'lane_width', 3.75) / 2.0
        distance_from_boundary = lane_half_width - abs(lateral)
        margin = getattr(self, 'lane_entry_margin', 0.1)
        entered_lane = distance_from_boundary >= margin  # 车身中心进入车道内侧 margin 米

        return same_lane or (same_road_section and entered_lane)

    # ------------------------------------------------------------------ helpers
    def _spawn_side_vehicle(self):
        """在侧向车道生成切入车辆。"""
        if self.side_spawn_location is not None:
            side_wp = self.map.get_waypoint(self.side_spawn_location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if side_wp is None:
                print("⚠️ 切入工况：侧向固定坐标不在可行驶车道，尝试使用相邻车道生成")
                side_wp = None  # fallback to lane-based below
            else:
                # 可选后移：沿车道方向向后移动指定距离（保持同车道）
                back_offset = getattr(self, 'side_back_offset', 0.0)
                z_lift = getattr(self, 'side_spawn_z_lift', 0.0)
                base_tf = side_wp.transform
                if back_offset > 0:
                    yaw_rad = math.radians(base_tf.rotation.yaw)
                    dx = -back_offset * math.cos(yaw_rad)
                    dy = -back_offset * math.sin(yaw_rad)
                    base_tf.location.x += dx
                    base_tf.location.y += dy
                base_tf.location.z += z_lift  # 抬升避免穿地
                transform = base_tf
                print(f"切入车预设生成点 road={side_wp.road_id}, lane={side_wp.lane_id}, back_offset={back_offset}, z_lift={z_lift}")
        else:
            lead_wp = self.map.get_waypoint(self.lead_vehicle.get_location(), project_to_road=True)
            side_wp = lead_wp.get_left_lane() if self.enable_from_left else lead_wp.get_right_lane()

            # 若指定侧无车道，尝试另一侧
            if side_wp is None:
                side_wp = lead_wp.get_right_lane() if self.enable_from_left else lead_wp.get_left_lane()
                self.enable_from_left = not self.enable_from_left

            if side_wp is None:
                print("⚠️ 切入工况初始化失败：无可用侧向车道")
                return

            transform = side_wp.transform
            transform.location.z += 0.1
        bp = self.blueprint_library.filter(self.vehicle_blueprint)[0]
        vehicle = self.world.try_spawn_actor(bp, transform)
        if vehicle is None:
            print(f"⚠️ 切入车辆生成失败，位置可能被占用：{transform.location}")
            return

        # 初始方向占位，实际触发时根据相对自车位置决定
        self._force_to_right = True

        port = self.tm.get_port()
        vehicle.set_autopilot(True, port)
        # 禁用自动换道，防止未到触发时间自行变道
        self.tm.auto_lane_change(vehicle, False)
        self.tm.ignore_lights_percentage(vehicle, 100.0)
        self.cut_in_vehicle = vehicle
        print(f"✅ 切入车辆已生成，位置: {transform.location}, 将在 {self.trigger_time_s:.1f}s 后触发切入（方向运行时决定）")

    @staticmethod
    def _lateral_offset_to_lane(location: carla.Location, lane_wp) -> float:
        """计算某点相对给定车道中心的横向偏移，左负右正。"""
        center = lane_wp.transform.location
        yaw = math.radians(lane_wp.transform.rotation.yaw)
        dx = location.x - center.x
        dy = location.y - center.y
        return -dx * math.sin(yaw) + dy * math.cos(yaw)
