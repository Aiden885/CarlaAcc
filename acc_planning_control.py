import numpy as np
import carla
from enum import Enum
import math
# 导入SPPVT控制器
from sppvt_longitudinal_control import sppvt_longitudinal_control
# 在文件顶部添加导入
from three_mode_controller import three_mode_control


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
        """车道保持和速度控制（新增PID纵向控制基于target_info）"""
        ego_speed, ego_accel = self.get_ego_state()

        if lane_offset is None:
            lane_offset = self.get_lane_offset()
        print(f"CRUISE - Lane Offset: {lane_offset:.1f}m")

        # === 新增：使用三模式控制 ===
        if target_info is not None and len(target_info) >= 8 and all(np.isfinite(target_info)):
            current_distance = float(target_info[0])
            if current_distance is None:
                print("YYYYYYYYYYYYYYYYYYYYYYY")
            # 使用三模式控制
            accel, control_info = three_mode_control(ego_speed, current_distance, self.target_speed)

            print(f"Three-Mode: {control_info['mode']} - {control_info['message']}")

        else:
            # 没有目标，使用速度控制模式
            accel, control_info = three_mode_control(ego_speed, None, self.target_speed)
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

        print(f"CRUISE Mode - Speed: {ego_speed:.2f}/{self.target_speed:.2f} m/s, "
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


