import numpy as np
import carla
from enum import Enum
import math


class ACCMode(Enum):
    CRUISE = 1  # 定速巡航
    FOLLOW = 2  # 跟车（使用轨迹跟踪）
    STOP = 3  # 刹停
    EMERGENCY = 4  # 紧急制动


class ACCPlanningControl:
    def __init__(self, ego_vehicle, target_speed_kmh=40.0, time_gap=2.0, max_follow_distance=50.0):
        self.ego_vehicle = ego_vehicle
        self.target_speed = target_speed_kmh / 3.6  # 转换为 m/s
        self.time_gap = time_gap
        self.max_follow_distance = max_follow_distance
        self.min_safe_distance = 5.0
        self.max_accel = 2.0
        self.max_decel = -3.0
        self.control_dt = 0.05  # 控制步长改为0.05s（20Hz）
        self.prev_accel = 0.0
        self.prev_steer = 0.0
        self.smooth_alpha = 0.7  # 平滑因子
        self.mode = ACCMode.CRUISE

        # === 新增：PID控制器参数（纵向） ===
        self.pid_kp = 0.5  # 比例增益
        self.pid_ki = 0.01  # 积分增益
        self.pid_kd = 0.05  # 微分增益
        self.pid_prev_error = 0.0
        self.pid_integral = 0.0
        self.desired_distance = 15.0  # 期望跟车距离

        # === 新增：Stanley控制器参数（横向） ===
        self.stanley_k = 0.5  # Stanley控制器增益
        self.max_steer_angle = 0.5  # 最大转向角（弧度）

        # === 保留原有的CRUISE模式参数 ===
        self.speed_kp = 1.0
        self.speed_ki = 0.05
        self.speed_kd = 0.01
        self.lane_kp = 0.2
        self.lane_kd = 0.1
        self.speed_error_sum = 0.0
        self.prev_speed_error = 0.0
        self.prev_lane_error = 0.0

        # 初始化控制状态
        if not hasattr(self, 'prev_control'):
            self.prev_control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)

        # CARLA世界引用
        self.world = None
        if hasattr(ego_vehicle, 'get_world'):
            self.world = ego_vehicle.get_world()

        # 车道信息
        self.current_waypoint = None
        self.lane_width = 3.5

    def get_ego_state(self):
        """获取本车状态"""
        velocity = self.ego_vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        acceleration = self.ego_vehicle.get_acceleration()
        accel = acceleration.x
        return speed, accel

    def compute_safe_distance(self, ego_speed):
        """计算安全距离"""
        return max(ego_speed * self.time_gap + self.min_safe_distance, self.min_safe_distance)

    def get_lane_offset(self):
        """获取车辆相对于车道中心的偏移量"""
        if self.world is None:
            return 0.0

        carla_map = self.world.get_map()
        if carla_map is None:
            return 0.0

        vehicle_location = self.ego_vehicle.get_location()
        self.current_waypoint = carla_map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if self.current_waypoint is None:
            return 0.0

        lane_center = self.current_waypoint.transform.location
        lane_direction = self.current_waypoint.transform.get_forward_vector()

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

    def normalize_angle(self, angle):
        """将角度标准化到[-pi, pi]范围内"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def pid_longitudinal_control(self, distance_error):
        """PID纵向控制器"""
        # PID控制
        self.pid_integral += distance_error * self.control_dt

        # 限制积分项防止积分饱和
        self.pid_integral = np.clip(self.pid_integral, -10.0, 10.0)

        # 计算微分项
        pid_derivative = (distance_error - self.pid_prev_error) / self.control_dt

        # PID输出
        pid_output = (self.pid_kp * distance_error +
                      self.pid_ki * self.pid_integral +
                      self.pid_kd * pid_derivative)

        # 更新上一次误差
        self.pid_prev_error = distance_error

        # 限制输出范围
        return np.clip(pid_output, self.max_decel, self.max_accel)

### 横向控制器效果对比
    def stanley_original(self, trajectory_waypoints):
        """原始Stanley控制器（存在后方点问题）"""
        if not trajectory_waypoints:
            return 0.0

        # 获取自车状态
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        # 获取车速（避免除零错误）
        velocity = self.ego_vehicle.get_velocity()
        speed = max(0.1, math.sqrt(velocity.x ** 2 + velocity.y ** 2))

        # 找到最近的轨迹点（原始方法 - 可能选择后方点）
        closest_waypoint = None
        min_distance = float('inf')

        for waypoint in trajectory_waypoints:
            # 计算自车到轨迹点的距离
            dist = math.sqrt((vehicle_location.x - waypoint.world_x) ** 2 +
                             (vehicle_location.y - waypoint.world_y) ** 2)
            if dist < min_distance:
                min_distance = dist
                closest_waypoint = waypoint

        if closest_waypoint is None:
            return 0.0

        # 计算横向偏差
        wp_direction_x = math.cos(closest_waypoint.tangent_angle)
        wp_direction_y = math.sin(closest_waypoint.tangent_angle)

        dx = vehicle_location.x - closest_waypoint.world_x
        dy = vehicle_location.y - closest_waypoint.world_y
        lateral_error = dx * wp_direction_y - dy * wp_direction_x

        # 计算航向偏差
        heading_error = self.normalize_angle(closest_waypoint.tangent_angle - vehicle_yaw)

        # Stanley控制律
        crosstrack_term = math.atan2(self.stanley_k * lateral_error, speed)
        steer = heading_error + crosstrack_term

        # 限制转向角
        steer_output = np.clip(steer, -self.max_steer_angle, self.max_steer_angle)

        print(f"Stanley原始 - 最近点距离: {min_distance:.2f}m, "
              f"横向偏差: {lateral_error:.3f}m, 航向偏差: {math.degrees(heading_error):.1f}°, "
              f"转向输出: {math.degrees(steer_output):.1f}°")

        return steer_output

    def stanley_front_filtered(self, trajectory_waypoints):
        """Stanley控制器 - 前方点过滤版"""
        if not trajectory_waypoints:
            return 0.0

        # 获取自车状态
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        # 获取车速
        velocity = self.ego_vehicle.get_velocity()
        speed = max(0.1, math.sqrt(velocity.x ** 2 + velocity.y ** 2))

        # 只考虑前方的轨迹点
        front_waypoints = []
        vehicle_forward_vector = carla.Vector3D(
            math.cos(vehicle_yaw),
            math.sin(vehicle_yaw),
            0.0
        )

        for waypoint in trajectory_waypoints:
            # 计算车辆到轨迹点的向量
            to_waypoint = carla.Vector3D(
                waypoint.world_x - vehicle_location.x,
                waypoint.world_y - vehicle_location.y,
                0.0
            )

            # 检查轨迹点是否在车辆前方（点积 > 0）
            dot_product = vehicle_forward_vector.dot(to_waypoint)
            if dot_product > 0:  # 只保留前方的点
                front_waypoints.append(waypoint)

        if not front_waypoints:
            print("Stanley前方过滤 - 警告：没有找到前方轨迹点")
            return 0.0

        # 在前方轨迹点中找最近的
        closest_waypoint = None
        min_distance = float('inf')

        for waypoint in front_waypoints:
            dist = math.sqrt((vehicle_location.x - waypoint.world_x) ** 2 +
                             (vehicle_location.y - waypoint.world_y) ** 2)
            if dist < min_distance:
                min_distance = dist
                closest_waypoint = waypoint

        if closest_waypoint is None:
            return 0.0

        # 计算横向偏差
        wp_direction_x = math.cos(closest_waypoint.tangent_angle)
        wp_direction_y = math.sin(closest_waypoint.tangent_angle)

        dx = vehicle_location.x - closest_waypoint.world_x
        dy = vehicle_location.y - closest_waypoint.world_y
        lateral_error = dx * wp_direction_y - dy * wp_direction_x

        # 计算航向偏差
        heading_error = self.normalize_angle(closest_waypoint.tangent_angle - vehicle_yaw)

        # Stanley控制律
        crosstrack_term = math.atan2(self.stanley_k * lateral_error, speed)
        steer = heading_error + crosstrack_term

        # 限制转向角
        steer_output = np.clip(steer, -self.max_steer_angle, self.max_steer_angle)

        print(f"Stanley前方过滤 - 总点数: {len(trajectory_waypoints)}, 前方点数: {len(front_waypoints)}, "
              f"选择距离: {min_distance:.2f}m, 横向偏差: {lateral_error:.3f}m, "
              f"航向偏差: {math.degrees(heading_error):.1f}°, 转向输出: {math.degrees(steer_output):.1f}°")

        return steer_output

    def stanley_lookahead(self, trajectory_waypoints):
        """Stanley控制器 - 前瞻距离版"""
        if not trajectory_waypoints:
            return 0.0

        # 获取自车状态
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        # 获取车速
        velocity = self.ego_vehicle.get_velocity()
        speed = max(0.1, math.sqrt(velocity.x ** 2 + velocity.y ** 2))

        # 计算前瞻距离
        lookahead_distance = max(8.0, speed * 1.5)  # 至少8米，或者1.5秒前瞻

        # 只考虑前方的轨迹点
        front_waypoints = []
        vehicle_forward_vector = carla.Vector3D(
            math.cos(vehicle_yaw),
            math.sin(vehicle_yaw),
            0.0
        )

        for waypoint in trajectory_waypoints:
            # 计算车辆到轨迹点的向量
            to_waypoint = carla.Vector3D(
                waypoint.world_x - vehicle_location.x,
                waypoint.world_y - vehicle_location.y,
                0.0
            )

            # 检查轨迹点是否在车辆前方
            dot_product = vehicle_forward_vector.dot(to_waypoint)
            if dot_product > 0:
                front_waypoints.append(waypoint)

        if not front_waypoints:
            print("Stanley前瞻 - 警告：没有找到前方轨迹点")
            return 0.0

        # 在前方轨迹点中找到最接近前瞻距离的点
        target_waypoint = None
        min_distance_diff = float('inf')

        for waypoint in front_waypoints:
            dist = math.sqrt((vehicle_location.x - waypoint.world_x) ** 2 +
                             (vehicle_location.y - waypoint.world_y) ** 2)
            distance_diff = abs(dist - lookahead_distance)
            if distance_diff < min_distance_diff:
                min_distance_diff = distance_diff
                target_waypoint = waypoint

        if target_waypoint is None:
            # 如果没找到合适的点，选择最远的前方点
            max_distance = 0.0
            for waypoint in front_waypoints:
                dist = math.sqrt((vehicle_location.x - waypoint.world_x) ** 2 +
                                 (vehicle_location.y - waypoint.world_y) ** 2)
                if dist > max_distance:
                    max_distance = dist
                    target_waypoint = waypoint

        if target_waypoint is None:
            return 0.0

        # 计算横向偏差
        wp_direction_x = math.cos(target_waypoint.tangent_angle)
        wp_direction_y = math.sin(target_waypoint.tangent_angle)

        dx = vehicle_location.x - target_waypoint.world_x
        dy = vehicle_location.y - target_waypoint.world_y
        lateral_error = dx * wp_direction_y - dy * wp_direction_x

        # 计算航向偏差
        heading_error = self.normalize_angle(target_waypoint.tangent_angle - vehicle_yaw)

        # Stanley控制律
        crosstrack_term = math.atan2(self.stanley_k * lateral_error, speed)
        steer = heading_error + crosstrack_term

        # 限制转向角
        steer_output = np.clip(steer, -self.max_steer_angle, self.max_steer_angle)

        # 计算实际选择的距离
        actual_distance = math.sqrt((vehicle_location.x - target_waypoint.world_x) ** 2 +
                                    (vehicle_location.y - target_waypoint.world_y) ** 2)

        print(f"Stanley前瞻 - 目标前瞻: {lookahead_distance:.1f}m, 实际距离: {actual_distance:.1f}m, "
              f"前方点数: {len(front_waypoints)}, 横向偏差: {lateral_error:.3f}m, "
              f"航向偏差: {math.degrees(heading_error):.1f}°, 转向输出: {math.degrees(steer_output):.1f}°")

        return steer_output

    def stanley_arc_length(self, trajectory_waypoints):
        """Stanley控制器 - 基于弧长版（适用于有distance_from_start属性的轨迹点）"""
        if not trajectory_waypoints:
            return 0.0

        # 获取自车状态
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        # 获取车速
        velocity = self.ego_vehicle.get_velocity()
        speed = max(0.1, math.sqrt(velocity.x ** 2 + velocity.y ** 2))

        # 检查轨迹点是否有distance_from_start属性
        if not hasattr(trajectory_waypoints[0], 'distance_from_start'):
            print("Stanley弧长 - 警告：轨迹点没有distance_from_start属性，使用前瞻方法")
            return self.stanley_lookahead(trajectory_waypoints)

        # 找到车辆在轨迹上的大致投影位置
        vehicle_projection_s = None
        min_dist_for_projection = float('inf')

        for waypoint in trajectory_waypoints:
            dist = math.sqrt((vehicle_location.x - waypoint.world_x) ** 2 +
                             (vehicle_location.y - waypoint.world_y) ** 2)
            if dist < min_dist_for_projection:
                min_dist_for_projection = dist
                vehicle_projection_s = waypoint.distance_from_start

        if vehicle_projection_s is None:
            return 0.0

        # 计算目标弧长位置
        lookahead_distance = max(8.0, speed * 1.5)
        target_s = vehicle_projection_s + lookahead_distance

        # 找到目标弧长处的轨迹点
        target_waypoint = None
        min_s_diff = float('inf')

        for waypoint in trajectory_waypoints:
            waypoint_s = waypoint.distance_from_start
            if waypoint_s >= vehicle_projection_s:  # 确保是前方点
                s_diff = abs(waypoint_s - target_s)
                if s_diff < min_s_diff:
                    min_s_diff = s_diff
                    target_waypoint = waypoint

        if target_waypoint is None:
            # 如果没找到合适的前瞻点，选择最远的前方点
            max_s = vehicle_projection_s
            for waypoint in trajectory_waypoints:
                if waypoint.distance_from_start > max_s:
                    max_s = waypoint.distance_from_start
                    target_waypoint = waypoint

        if target_waypoint is None:
            return 0.0

        # 计算横向偏差
        wp_direction_x = math.cos(target_waypoint.tangent_angle)
        wp_direction_y = math.sin(target_waypoint.tangent_angle)

        dx = vehicle_location.x - target_waypoint.world_x
        dy = vehicle_location.y - target_waypoint.world_y
        lateral_error = dx * wp_direction_y - dy * wp_direction_x

        # 计算航向偏差
        heading_error = self.normalize_angle(target_waypoint.tangent_angle - vehicle_yaw)

        # Stanley控制律
        crosstrack_term = math.atan2(self.stanley_k * lateral_error, speed)
        steer = heading_error + crosstrack_term

        # 限制转向角
        steer_output = np.clip(steer, -self.max_steer_angle, self.max_steer_angle)

        print(f"Stanley弧长 - 车辆弧长: {vehicle_projection_s:.1f}m, 目标弧长: {target_s:.1f}m, "
              f"实际弧长: {target_waypoint.distance_from_start:.1f}m, 横向偏差: {lateral_error:.3f}m, "
              f"航向偏差: {math.degrees(heading_error):.1f}°, 转向输出: {math.degrees(steer_output):.1f}°")

        return steer_output

    def pure_pursuit_control(self, trajectory_waypoints):
        """Pure Pursuit控制器（作为对比参考）"""
        if not trajectory_waypoints:
            return 0.0

        # 获取自车状态
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        # 获取车速
        velocity = self.ego_vehicle.get_velocity()
        speed = max(0.1, math.sqrt(velocity.x ** 2 + velocity.y ** 2))

        # 计算前瞻距离（Pure Pursuit通常使用更大的前瞻距离）
        lookahead_distance = max(10.0, speed * 2.0)

        # 找到前瞻距离处的目标点
        target_point = None
        min_distance_diff = float('inf')

        vehicle_forward = carla.Vector3D(math.cos(vehicle_yaw), math.sin(vehicle_yaw), 0.0)

        for waypoint in trajectory_waypoints:
            # 计算距离
            dist = math.sqrt((vehicle_location.x - waypoint.world_x) ** 2 +
                             (vehicle_location.y - waypoint.world_y) ** 2)

            # 检查是否在前方
            to_waypoint = carla.Vector3D(
                waypoint.world_x - vehicle_location.x,
                waypoint.world_y - vehicle_location.y,
                0.0
            )

            if vehicle_forward.dot(to_waypoint) > 0:  # 确保在前方
                distance_diff = abs(dist - lookahead_distance)
                if distance_diff < min_distance_diff:
                    min_distance_diff = distance_diff
                    target_point = waypoint

        if target_point is None:
            print("Pure Pursuit - 警告：没有找到合适的目标点")
            return 0.0

        # Pure Pursuit控制律
        # 计算目标点在车辆坐标系下的位置
        dx = target_point.world_x - vehicle_location.x
        dy = target_point.world_y - vehicle_location.y

        # 转换到车辆坐标系
        cos_yaw = math.cos(vehicle_yaw)
        sin_yaw = math.sin(vehicle_yaw)

        x_body = cos_yaw * dx + sin_yaw * dy
        y_body = -sin_yaw * dx + cos_yaw * dy

        # Pure Pursuit公式
        L = math.sqrt(dx * dx + dy * dy)  # 实际距离
        curvature = 2.0 * y_body / (L * L)

        # 假设车辆轴距为2.5米
        wheelbase = 2.5
        steer_angle = math.atan(wheelbase * curvature)

        # 限制转向角
        steer_output = np.clip(steer_angle, -self.max_steer_angle, self.max_steer_angle)

        print(f"Pure Pursuit - 前瞻距离: {lookahead_distance:.1f}m, 实际距离: {L:.1f}m, "
              f"目标点: ({target_point.world_x:.1f}, {target_point.world_y:.1f}), "
              f"车身坐标: ({x_body:.1f}, {y_body:.1f}), 曲率: {curvature:.3f}, "
              f"转向: {math.degrees(steer_output):.1f}°")

        return steer_output



    def plan(self, target_info):
        """规划参考状态"""
        ego_speed, _ = self.get_ego_state()

        if target_info is None or len(target_info) < 8 or not all(np.isfinite(target_info)):
            self.mode = ACCMode.CRUISE
            return self.target_speed, self.max_follow_distance, 0.0, 0.0

        dist = float(target_info[0])  # 纵向距离
        rel_vel = float(target_info[6])  # 纵向相对速度

        if not all(np.isfinite([dist, rel_vel])):
            self.mode = ACCMode.CRUISE
            return self.target_speed, self.max_follow_distance, 0.0, 0.0

        lead_speed = ego_speed + rel_vel
        safe_dist = self.compute_safe_distance(ego_speed)

        # 决定控制模式
        if dist > self.max_follow_distance:
            self.mode = ACCMode.CRUISE
            ref_speed = self.target_speed
            ref_dist = self.max_follow_distance
        elif dist < self.min_safe_distance:
            self.mode = ACCMode.EMERGENCY
            ref_speed = 0.0
            ref_dist = self.min_safe_distance
        elif lead_speed < 0.1 and dist < safe_dist:
            self.mode = ACCMode.STOP
            ref_speed = 0.0
            ref_dist = self.min_safe_distance
        else:
            self.mode = ACCMode.FOLLOW
            ref_speed = min(lead_speed, self.target_speed)
            ref_dist = safe_dist

        return ref_speed, ref_dist, 0.0, 0.0

    def cruise_control(self, lane_offset=None):
        """CRUISE模式下的车道保持和速度控制（保持原有逻辑）"""
        ego_speed, ego_accel = self.get_ego_state()

        if lane_offset is None:
            lane_offset = self.get_lane_offset()

        # 纵向控制：PID速度控制
        speed_error = self.target_speed - ego_speed
        self.speed_error_sum += speed_error * self.control_dt
        speed_error_diff = (speed_error - self.prev_speed_error) / self.control_dt

        self.speed_error_sum = max(min(self.speed_error_sum, 5.0), -5.0)

        accel = (self.speed_kp * speed_error +
                 self.speed_ki * self.speed_error_sum +
                 self.speed_kd * speed_error_diff)

        accel = np.clip(accel, self.max_decel, self.max_accel)
        self.prev_speed_error = speed_error

        # 横向控制：基于车道偏移的PD控制
        lane_error_diff = (lane_offset - self.prev_lane_error) / self.control_dt
        steer = (self.lane_kp * lane_offset + self.lane_kd * lane_error_diff)
        steer = np.clip(steer, -self.max_steer_angle, self.max_steer_angle)
        self.prev_lane_error = lane_offset

        # 平滑控制输出
        accel = self.smooth_alpha * accel + (1 - self.smooth_alpha) * self.prev_accel
        steer = self.smooth_alpha * steer + (1 - self.smooth_alpha) * self.prev_steer
        self.prev_accel = accel
        self.prev_steer = steer

        control = self.control_to_vehicle(accel, steer)

        print(f"CRUISE Mode - Speed: {ego_speed:.2f}/{self.target_speed:.2f} m/s, "
              f"Lane Offset: {lane_offset:.2f} m, Accel: {accel:.2f} m/s², "
              f"Steer: {steer:.2f} rad")

        return control

    def follow_control_with_trajectory(self, target_info, trajectory_waypoints):
        """FOLLOW模式下的PID纵向+Stanley横向控制（使用轨迹点）"""
        ego_speed, _ = self.get_ego_state()

        # === 纵向控制：基于目标距离的PID ===
        if target_info is not None and len(target_info) >= 8:
            # 使用雷达检测的距离信息
            current_distance = float(target_info[0])
            target_speed = ego_speed + float(target_info[6])  # 前车速度

            # 动态调整期望距离
            dynamic_desired_distance = max(self.desired_distance, ego_speed * self.time_gap)
            distance_error = current_distance - dynamic_desired_distance

            # 速度限制
            if target_speed < self.target_speed:
                max_ref_speed = target_speed
            else:
                max_ref_speed = self.target_speed

            # PID纵向控制
            accel = self.pid_longitudinal_control(distance_error)

            # 考虑前车速度的前馈控制
            if abs(float(target_info[6])) > 0.1:  # 如果前车在加减速
                feedforward = 0.5 * float(target_info[6]) / 3.6  # 前馈项
                accel += feedforward

        else:
            # 没有目标信息，进入CRUISE模式
            self.mode = ACCMode.CRUISE
            return self.cruise_control()

        # === 横向控制：Stanley控制器使用轨迹点 ===
        if trajectory_waypoints and len(trajectory_waypoints) > 0:
            # 使用轨迹点进行横向控制
            steer = self.pure_pursuit_control(trajectory_waypoints)
        else:
            # 没有轨迹点，使用车道保持
            lane_offset = self.get_lane_offset()
            steer = self.lane_kp * lane_offset

        # 限制控制量
        accel = np.clip(accel, self.max_decel, self.max_accel)
        steer = np.clip(steer, -self.max_steer_angle, self.max_steer_angle)

        # 平滑控制输出
        accel = self.smooth_alpha * accel + (1 - self.smooth_alpha) * self.prev_accel
        steer = self.smooth_alpha * steer + (1 - self.smooth_alpha) * self.prev_steer
        self.prev_accel = accel
        self.prev_steer = steer

        control = self.control_to_vehicle(accel, steer)

        # 调试输出
        print(f"FOLLOW Mode - Distance: {current_distance:.1f}m (目标: {dynamic_desired_distance:.1f}m), "
              f"误差: {distance_error:.1f}m, 速度: {ego_speed:.1f}m/s, "
              f"加速度: {accel:.2f}m/s², 转向: {math.degrees(steer):.1f}°")

        return control

    def control_to_vehicle(self, accel, steer):
        """将加速度和转向角转换为车辆控制命令"""
        control = carla.VehicleControl()
        control.manual_gear_shift = False
        control.gear = 1

        # 纵向控制
        if accel > 0:
            control.throttle = min(accel / self.max_accel, 1.0)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-accel / -self.max_decel, 1.0)

        # 横向控制（归一化到 [-1, 1]）
        control.steer = steer / self.max_steer_angle

        # 平滑处理
        control.throttle = self.smooth_alpha * control.throttle + (1 - self.smooth_alpha) * self.prev_control.throttle
        control.brake = self.smooth_alpha * control.brake + (1 - self.smooth_alpha) * self.prev_control.brake
        control.steer = self.smooth_alpha * control.steer + (1 - self.smooth_alpha) * self.prev_control.steer

        self.prev_control = control
        return control

    def update(self, target_info, lane_offset=None):
        """更新控制逻辑（原有接口，不使用轨迹）"""
        if not hasattr(self, 'prev_control'):
            self.prev_control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)

        # 规划参考状态
        ref_speed, ref_dist, _, _ = self.plan(target_info)

        # 根据模式执行控制
        if self.mode == ACCMode.CRUISE:
            return self.cruise_control(lane_offset)
        elif self.mode in [ACCMode.FOLLOW, ACCMode.STOP, ACCMode.EMERGENCY]:
            # FOLLOW模式但没有轨迹信息，回退到基本控制
            return self.follow_control_with_trajectory(target_info, [])
        else:
            return self.cruise_control(lane_offset)

    def update_with_trajectory(self, target_info, trajectory_data):
        """新接口：使用轨迹信息的控制更新"""
        if not hasattr(self, 'prev_control'):
            self.prev_control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)

        # 检查轨迹数据有效性
        if not trajectory_data or not trajectory_data.get('is_valid', False):
            # 轨迹无效，回退到原有控制
            return self.update(target_info)

        # 规划参考状态
        ref_speed, ref_dist, _, _ = self.plan(target_info)

        # 获取轨迹点
        trajectory_waypoints = trajectory_data.get('waypoints', [])

        # 根据模式执行控制
        if self.mode == ACCMode.CRUISE:
            return self.cruise_control()
        elif self.mode in [ACCMode.FOLLOW, ACCMode.STOP, ACCMode.EMERGENCY]:
            # FOLLOW模式使用轨迹进行控制
            return self.follow_control_with_trajectory(target_info, trajectory_waypoints)
        else:
            return self.cruise_control()

    # === 新增：轨迹相关的辅助方法 ===
    def get_trajectory_preview_distance(self, ego_speed):
        """根据车速计算轨迹预瞄距离"""
        return max(10.0, ego_speed * 2.0)  # 至少10米，或者2秒前瞻

    def filter_trajectory_points(self, waypoints, preview_distance):
        """过滤轨迹点，只保留预瞄距离内的点"""
        if not waypoints:
            return []

        filtered_points = []
        for wp in waypoints:
            if wp.distance_from_start <= preview_distance:
                filtered_points.append(wp)

        return filtered_points

    def get_curvature_based_speed_limit(self, trajectory_waypoints):
        """基于轨迹曲率计算速度限制"""
        if not trajectory_waypoints:
            return self.target_speed

        max_curvature = max([wp.curvature for wp in trajectory_waypoints])

        # 根据曲率限制速度（经验公式）
        if max_curvature > 0.1:  # 急弯
            speed_limit = min(self.target_speed, 15.0)
        elif max_curvature > 0.05:  # 中等弯道
            speed_limit = min(self.target_speed, 20.0)
        else:  # 直道或缓弯
            speed_limit = self.target_speed

        return speed_limit