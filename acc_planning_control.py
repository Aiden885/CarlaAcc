import numpy as np
import cvxpy as cp
import carla
from enum import Enum
import math


class ACCMode(Enum):
    CRUISE = 1  # 定速巡航
    FOLLOW = 2  # 跟车
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
        self.control_dt = 0.1  # 控制步长
        self.prev_accel = 0.0
        self.prev_steer = 0.0
        self.smooth_alpha = 0.7  # 平滑因子
        self.horizon = 3  # MPC 预测时域
        # MPC 权重
        self.Q_dist = 1.0  # 纵向距离权重
        self.Q_vel = 1.0  # 速度权重
        self.R_accel = 1.0  # 加速度控制权重
        self.Q_y = 0.5  # 横向偏移权重
        self.Q_yvel = 0.5  # 横向速度权重
        self.R_steer = 1.0  # 转向控制权重
        self.max_steer_angle = 0.5  # 最大转向角（弧度）
        self.mode = ACCMode.CRUISE
        if not hasattr(self, 'prev_control'):
            self.prev_control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)

        # CRUISE模式的PID参数
        self.speed_kp = 1.0  # 速度比例增益
        self.speed_ki = 0.05  # 速度积分增益
        self.speed_kd = 0.01  # 速度微分增益
        self.lane_kp = 0.2  # 车道保持比例增益
        self.lane_kd = 0.1  # 车道保持微分增益

        # PID控制器状态变量
        self.speed_error_sum = 0.0
        self.prev_speed_error = 0.0
        self.prev_lane_error = 0.0

        # 保存CARLA世界引用，用于车道线检测
        self.world = None
        if hasattr(ego_vehicle, 'get_world'):
            self.world = ego_vehicle.get_world()

        # 车道信息
        self.current_waypoint = None
        self.lane_width = 3.5  # 默认车道宽度

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
        """获取车辆相对于车道中心的偏移量

        这个函数可以根据需要使用摄像头或其他传感器来检测车道线
        目前使用CARLA API直接获取车辆位置与车道中心的偏移

        Returns:
            float: 车道偏移量，正值表示车辆偏向右侧，负值表示偏向左侧
        """
        if self.world is None:
            return 0.0

        # 获取地图引用
        carla_map = self.world.get_map()
        if carla_map is None:
            return 0.0

        # 获取车辆当前位置
        vehicle_location = self.ego_vehicle.get_location()

        # 获取当前位置的路点
        self.current_waypoint = carla_map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if self.current_waypoint is None:
            return 0.0

        # 计算车辆相对于车道中心的横向偏移
        lane_center = self.current_waypoint.transform.location
        lane_direction = self.current_waypoint.transform.get_forward_vector()

        # 计算车辆位置到车道中心的向量
        to_center_vector = carla.Vector3D(
            lane_center.x - vehicle_location.x,
            lane_center.y - vehicle_location.y,
            0  # 忽略高度差异
        )

        # 计算车道方向的垂直向量（车道的右侧方向）
        right_direction = carla.Vector3D(
            -lane_direction.y,  # 右向量
            lane_direction.x,
            0
        ).make_unit_vector()

        # 计算偏移量（正值表示车辆在车道右侧，负值表示在左侧）
        offset = to_center_vector.dot(right_direction)

        # 更新车道宽度信息
        left_waypoint = self.current_waypoint.get_left_lane()
        right_waypoint = self.current_waypoint.get_right_lane()

        if left_waypoint is not None and right_waypoint is not None:
            left_loc = left_waypoint.transform.location
            right_loc = right_waypoint.transform.location
            self.lane_width = math.sqrt((left_loc.x - right_loc.x) ** 2 + (left_loc.y - right_loc.y) ** 2)

        return offset

    def plan(self, target_info):
        """规划参考状态"""
        ego_speed, _ = self.get_ego_state()
        if target_info is None or len(target_info) < 8 or not all(np.isfinite(target_info)):
            self.mode = ACCMode.CRUISE
            return self.target_speed, 1000.0, 0.0, 0.0  # 使用1000.0替代无穷大

        dist = float(target_info[0])  # 纵向距离
        rel_vel = float(target_info[6])  # 纵向相对速度
        y_offset = float(target_info[1])  # 横向偏移
        y_vel = -float(target_info[7])  # 横向速度（假设前车横向速度为 0）

        if not all(np.isfinite([dist, rel_vel, y_offset, y_vel])):
            self.mode = ACCMode.CRUISE
            return self.target_speed, 1000.0, 0.0, 0.0  # 使用1000.0替代无穷大

        lead_speed = ego_speed + rel_vel
        safe_dist = self.compute_safe_distance(ego_speed)

        if dist > self.max_follow_distance:
            self.mode = ACCMode.CRUISE
            ref_speed = self.target_speed
            ref_dist = 1000.0  # 使用1000.0替代无穷大
        elif dist < self.min_safe_distance:
            self.mode = ACCMode.EMERGENCY
            ref_speed = 0.0
            ref_dist = 0.0
        elif lead_speed < 0.1 and dist < safe_dist:
            self.mode = ACCMode.STOP
            ref_speed = 0.0
            ref_dist = self.min_safe_distance
        else:
            self.mode = ACCMode.FOLLOW
            ref_speed = min(lead_speed, self.target_speed)
            ref_dist = safe_dist

        ref_y_offset = 0.0  # 目标横向偏移为 0
        ref_y_vel = 0.0  # 目标横向速度为 0
        return ref_speed, ref_dist, ref_y_offset, ref_y_vel

    def cruise_control(self, lane_offset=None):
        """CRUISE模式下的车道保持和速度控制

        Args:
            lane_offset: 车道偏移量，正值表示车辆偏向右侧，负值表示偏向左侧
                        如果为None，则自动调用get_lane_offset获取

        Returns:
            carla.VehicleControl: 车辆控制指令
        """
        ego_speed, ego_accel = self.get_ego_state()

        # 如果没有提供车道偏移，则自动获取
        if lane_offset is None:
            lane_offset = self.get_lane_offset()

        # 纵向控制：PID速度控制
        speed_error = self.target_speed - ego_speed
        self.speed_error_sum += speed_error * self.control_dt
        speed_error_diff = (speed_error - self.prev_speed_error) / self.control_dt

        # 防止积分饱和
        self.speed_error_sum = max(min(self.speed_error_sum, 5.0), -5.0)

        # PID计算加速度
        accel = (self.speed_kp * speed_error +
                 self.speed_ki * self.speed_error_sum +
                 self.speed_kd * speed_error_diff)

        # 限制加速度范围
        accel = np.clip(accel, self.max_decel, self.max_accel)

        # 记录当前误差用于下一次计算
        self.prev_speed_error = speed_error

        # 横向控制：基于车道偏移的PD控制
        lane_error_diff = (lane_offset - self.prev_lane_error) / self.control_dt

        # PD计算转向角
        steer = (self.lane_kp * lane_offset + self.lane_kd * lane_error_diff)

        # 限制转向角范围
        steer = np.clip(steer, -self.max_steer_angle, self.max_steer_angle)

        # 记录当前车道偏移用于下一次计算
        self.prev_lane_error = lane_offset

        # 平滑控制输出
        accel = self.smooth_alpha * accel + (1 - self.smooth_alpha) * self.prev_accel
        steer = self.smooth_alpha * steer + (1 - self.smooth_alpha) * self.prev_steer
        self.prev_accel = accel
        self.prev_steer = steer

        # 转换为车辆控制命令
        control = self.control_to_vehicle(accel, steer)

        print(f"CRUISE Mode - Speed: {ego_speed:.2f}/{self.target_speed:.2f} m/s, "
              f"Lane Offset: {lane_offset:.2f} m, Accel: {accel:.2f} m/s², "
              f"Steer: {steer:.2f} rad, Throttle: {control.throttle:.2f}, "
              f"Brake: {control.brake:.2f}, Steer Control: {control.steer:.2f}")

        return control

    def mpc_control(self, ref_speed, ref_dist, ref_y_offset, ref_y_vel, dist, rel_vel, ego_speed, y_offset, y_vel):
        """MPC控制器"""
        # 检查输入有效性
        if self.mode == ACCMode.CRUISE:
            # CRUISE模式下只检查速度和横向控制相关参数
            valid_inputs = [ego_speed, y_offset, y_vel, ref_speed, ref_y_offset, ref_y_vel]
            if not all(np.isfinite(valid_inputs)):
                print("Invalid MPC inputs for CRUISE mode, using previous controls")
                return self.prev_accel, self.prev_steer
        else:
            # 其他模式检查所有参数
            if not all(np.isfinite(
                    [dist, rel_vel, ego_speed, y_offset, y_vel, ref_speed, ref_dist, ref_y_offset, ref_y_vel])):
                print("Invalid MPC inputs, using previous controls")
                return self.prev_accel, self.prev_steer

        x = np.array([dist, rel_vel, ego_speed, y_offset, y_vel], dtype=np.float64)
        u = cp.Variable((self.horizon, 2))
        states = [x]

        A = np.array([[1, self.control_dt, 0, 0, 0],
                      [0, 1, -self.control_dt, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, self.control_dt],
                      [0, 0, 0, 0, 1]], dtype=np.float64)
        B = np.array([[0, 0],
                      [0, 0],
                      [self.control_dt, 0],
                      [0, 0],
                      [0, self.control_dt]], dtype=np.float64)

        cost = 0
        constraints = []

        for t in range(self.horizon):
            state = states[-1]
            dist_t, rel_vel_t, ego_speed_t, y_offset_t, y_vel_t = state
            a_t, delta_t = u[t, 0], u[t, 1]

            # 根据不同模式设置成本函数
            if self.mode == ACCMode.CRUISE:
                cost += self.Q_vel * cp.square(ego_speed_t - ref_speed)
            elif self.mode == ACCMode.STOP:
                cost += self.Q_dist * cp.square(dist_t - ref_dist) + self.Q_vel * cp.square(ego_speed_t)
            else:
                cost += self.Q_dist * cp.square(dist_t - ref_dist) + self.Q_vel * cp.square(ego_speed_t - ref_speed)

            cost += self.Q_y * cp.square(y_offset_t - ref_y_offset) + self.Q_yvel * cp.square(y_vel_t - ref_y_vel)
            cost += self.R_accel * cp.square(a_t) + self.R_steer * cp.square(delta_t)

            # 状态更新，使用初始 ego_speed 线性化
            dist_next = dist_t + self.control_dt * rel_vel_t
            rel_vel_next = rel_vel_t - self.control_dt * a_t
            ego_speed_next = ego_speed_t + self.control_dt * a_t
            y_offset_next = y_offset_t + self.control_dt * y_vel_t
            y_vel_next = y_vel_t + self.control_dt * (ego_speed * delta_t)  # 使用初始 ego_speed

            next_state = cp.vstack([dist_next, rel_vel_next, ego_speed_next, y_offset_next, y_vel_next])
            states.append(next_state)

        constraints += [u[:, 0] >= self.max_decel, u[:, 0] <= self.max_accel,
                        u[:, 1] >= -self.max_steer_angle, u[:, 1] <= self.max_steer_angle]

        try:
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(solver=cp.ECOS)
            if prob.status != cp.OPTIMAL or u.value is None:
                print("MPC optimization failed, using previous controls")
                return self.prev_accel, self.prev_steer
            return float(u.value[0, 0]), float(u.value[0, 1])
        except cp.SolverError:
            print("MPC solver error, using previous controls")
            return self.prev_accel, self.prev_steer

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
            control.brake = min(-0.5 * accel / -self.max_decel, 1.0)

        # 横向控制（归一化到 [-1, 1]）
        control.steer = min(max(steer / self.max_steer_angle, -1.0), 1.0)

        # 平滑处理
        control.throttle = self.smooth_alpha * control.throttle + (1 - self.smooth_alpha) * self.prev_control.throttle
        control.brake = self.smooth_alpha * control.brake + (1 - self.smooth_alpha) * self.prev_control.brake
        control.steer = self.smooth_alpha * control.steer + (1 - self.smooth_alpha) * self.prev_control.steer

        self.prev_control = control
        return control

    def update(self, target_info, lane_offset=None):
        """更新控制逻辑

        Args:
            target_info: 目标车辆信息
            lane_offset: 车道偏移量，可选参数。如果提供，则使用传入的值；否则在CRUISE模式下自动获取

        Returns:
            carla.VehicleControl: 车辆控制指令
        """
        if not hasattr(self, 'prev_control'):
            self.prev_control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)

        ego_speed, ego_accel = self.get_ego_state()
        ref_speed, ref_dist, ref_y_offset, ref_y_vel = self.plan(target_info)

        # CRUISE模式下使用基于车道的控制
        if self.mode == ACCMode.CRUISE:
            return self.cruise_control(lane_offset)

        # 其他模式使用MPC控制
        if target_info is not None and len(target_info) >= 8 and all(np.isfinite(target_info)):
            dist = float(target_info[0])
            rel_vel = float(target_info[6])
            y_offset = float(target_info[1])  # 使用目标信息中的横向偏移
            y_vel = -float(target_info[7])
        else:
            dist = 1000.0  # 使用大而有限的值代替无穷大
            rel_vel = 0.0
            # 如果没有提供车道偏移，则自动获取
            if lane_offset is None:
                y_offset = self.get_lane_offset()
            else:
                y_offset = lane_offset
            y_vel = 0.0

        accel, steer = self.mpc_control(ref_speed, ref_dist, ref_y_offset, ref_y_vel, dist, rel_vel, ego_speed,
                                        y_offset, y_vel)

        # 平滑控制输出
        accel = self.smooth_alpha * accel + (1 - self.smooth_alpha) * self.prev_accel
        steer = self.smooth_alpha * steer + (1 - self.smooth_alpha) * self.prev_steer
        self.prev_accel = accel
        self.prev_steer = steer

        control = self.control_to_vehicle(accel, steer)

        print(f"Mode: {self.mode}, Ego Speed: {ego_speed:.2f} m/s, "
              f"Dist: {dist:.2f} m, Y Offset: {y_offset:.2f} m, "
              f"Ref Speed: {ref_speed:.2f} m/s, Accel: {accel:.2f} m/s², "
              f"Steer: {steer:.2f} rad, Throttle: {control.throttle:.2f}, "
              f"Brake: {control.brake:.2f}, Steer Control: {control.steer:.2f}")

        return control