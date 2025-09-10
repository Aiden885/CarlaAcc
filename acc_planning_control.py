import numpy as np
import carla
from enum import Enum
import math
# 导入SPPVT控制器
from sppvt_longitudinal_control import sppvt_longitudinal_control
# 在文件顶部添加导入
from two_mode_controller import two_mode_control, three_mode_control_with_force_mode, get_force_mode_recommendation
import time


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
        self.smooth_alpha = 0.4  # 平滑因子

        # === 新增：PID控制器参数（纵向） ===
        self.pid_kp = 0.5  # 比例增益
        self.pid_ki = 0.01  # 积分增益
        self.pid_kd = 0.05  # 微分增益
        self.pid_prev_error = 0.0
        self.pid_integral = 0.0
        self.desired_distance = 15.0  # 期望跟车距离

        self.max_steer_angle = 0.4  # 最大转向角（弧度）

        # === 保留原有的CRUISE模式参数 ===
        self.speed_kp = 0.8
        self.speed_ki = 0
        self.speed_kd = 0
        self.lane_kp = 0.04
        self.lane_kd = 0
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

        # === 新增：前车重检测强制控制属性 ===
        self.last_target_lost_time = None
        self.speed_when_target_lost = None
        self.force_control_start_time = None
        self.force_control_duration = 8.0  # 强制控制持续时间（秒）

        # === 新增：多帧平均检测属性 ===
        self.target_history = []  # 存储最近几帧的检测结果
        self.history_size = 10  # 历史帧数
        self.min_valid_frames = 2  # 最少需要的有效检测帧数

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

    def cruise_control(self, lane_offset=None, target_info=None):
        """车道保持和速度控制（增加多帧平均检测逻辑）"""
        ego_speed, ego_accel = self.get_ego_state()

        if lane_offset is None:
            lane_offset = self.get_lane_offset()
        print(f"CRUISE - Lane Offset: {lane_offset:.1f}m")

        # === 新增：多帧平均检测逻辑 ===
        # 更新检测历史
        self.target_history.append(target_info)
        if len(self.target_history) > self.history_size:
            self.target_history.pop(0)

        # 统计有效检测
        valid_detections = [t for t in self.target_history if t is not None]

        # 判断有效的前车信息
        if len(valid_detections) >= self.min_valid_frames:
            # 使用最新的有效检测
            effective_target_info = valid_detections[-1]
            target_status = f"有效检测 ({len(valid_detections)}/{len(self.target_history)}帧)"
        else:
            # 检测不足，认为无前车
            effective_target_info = None
            target_status = f"检测不足 ({len(valid_detections)}/{len(self.target_history)}帧)"

        print(f"🎯 前车检测: {target_status}")

        # === 前车重检测强制控制逻辑（使用effective_target_info）===
        force_mode = None
        force_control_active = False

        if effective_target_info is not None:
            # 有前车的情况
            current_distance = float(effective_target_info[0])
            print(f"📏 传递给两模式: 距离={current_distance:.1f}m, 速度={ego_speed * 3.6:.1f}km/h")
            print(f"📏 effective_target_info长度: {len(effective_target_info)}")
            print(f"📏 effective_target_info内容: {effective_target_info[:3]}...")  # 只显示前3个元素

            # 检查前车重新检测
            if self.last_target_lost_time is not None:
                # 前车重新出现，检查是否需要强制控制
                speed_increase = ego_speed - (self.speed_when_target_lost or ego_speed)

                if speed_increase > 2.8:  # 速度增加超过2.8 m/s (约10 km/h)
                    # 启动强制控制
                    if self.force_control_start_time is None:
                        self.force_control_start_time = time.time()
                        # 根据距离情况选择强制模式
                        force_mode = get_force_mode_recommendation(ego_speed, current_distance)
                        print(f"🚨 前车重新检测，速度增加{speed_increase * 3.6:.1f}km/h，启动强制{force_mode}控制")

                    # 检查是否还在强制控制期间
                    force_duration = time.time() - self.force_control_start_time
                    if force_duration < self.force_control_duration:
                        # 继续使用之前确定的强制模式
                        force_mode = get_force_mode_recommendation(ego_speed, current_distance)
                        force_control_active = True
                        print(f"🛡️ 强制{force_mode}控制中 ({force_duration:.1f}/{self.force_control_duration:.1f}秒)")
                    else:
                        # 强制期结束
                        self.force_control_start_time = None
                        print(f"✅ 强制控制结束，恢复自主选择")
                else:
                    print(f"📍 前车重新检测，速度变化不大({speed_increase * 3.6:.1f}km/h)，正常控制")

            # 清除前车丢失状态
            self.last_target_lost_time = None
            self.speed_when_target_lost = None

        else:
            # 无前车情况
            if self.last_target_lost_time is None:
                # 刚刚丢失前车，记录状态
                self.last_target_lost_time = time.time()
                self.speed_when_target_lost = ego_speed
                print(f"📍 前车丢失，记录速度: {ego_speed * 3.6:.1f} km/h")

            # 清除强制控制状态
            if self.force_control_start_time is not None:
                self.force_control_start_time = None
                print(f"📍 前车持续丢失，清除强制控制状态")

        # === 使用两模式控制（使用effective_target_info）===
        if effective_target_info is not None and len(effective_target_info) >= 8 and all(
                np.isfinite(effective_target_info)):
            current_distance = float(effective_target_info[0])

            if force_control_active and force_mode:
                # 使用强制模式的两模式控制
                accel, control_info = three_mode_control_with_force_mode(
                    ego_speed, current_distance, self.target_speed, force_mode=force_mode
                )
                print(f"🛡️ Force-Mode: {control_info['mode']} - {control_info['message']}")
            else:
                # 使用正常的两模式控制
                accel, control_info = two_mode_control(ego_speed, current_distance, self.target_speed)
                print(f"Two-Mode: {control_info['mode']} - {control_info['message']}")

        else:
            # 没有目标，使用速度控制模式
            accel, control_info = two_mode_control(ego_speed, None, self.target_speed)
            print(f"No Target: {control_info['mode']} - {control_info['message']}")

        # 横向控制：基于车道偏移的PD控制
        lane_error_diff = (lane_offset - self.prev_lane_error) / self.control_dt
        steer = (self.lane_kp * lane_offset + self.lane_kd * lane_error_diff)
        steer = np.clip(steer, -self.max_steer_angle, self.max_steer_angle)
        self.prev_lane_error = lane_offset

        # 平滑控制输出
        accel = np.clip(accel, self.max_decel, self.max_accel)
        accel = self.smooth_alpha * accel + (1 - self.smooth_alpha) * self.prev_accel
        steer = self.smooth_alpha * steer + (1 - self.smooth_alpha) * self.prev_steer
        self.prev_accel = accel
        self.prev_steer = steer

        control = self.control_to_vehicle(accel, steer)

        # 显示控制状态
        control_type = "FORCE" if force_control_active else "NORMAL"
        print(f"CRUISE Mode ({control_type}) - Speed: {ego_speed:.2f}/{self.target_speed:.2f} m/s, "
              f"Lane Offset: {lane_offset:.2f} m, Accel: {accel:.2f} m/s², "
              f"Steer: {steer:.2f} rad")

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


