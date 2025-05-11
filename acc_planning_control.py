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
    def __init__(self, ego_vehicle, target_speed_kmh=60.0, time_gap=2.0, max_follow_distance=50.0):
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
        self.horizon = 10  # MPC 预测时域
        # MPC 权重
        self.Q_dist = 1.0  # 纵向距离权重
        self.Q_vel = 0.5  # 速度权重
        self.R_accel = 0.2  # 加速度控制权重
        self.Q_y = 2.0  # 横向偏移权重
        self.Q_yvel = 0.5  # 横向速度权重
        self.R_steer = 0.1  # 转向控制权重
        self.max_steer_angle = 0.5  # 最大转向角（弧度）
        self.mode = ACCMode.CRUISE
        if not hasattr(self, 'prev_control'):
            self.prev_control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)

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

    def plan(self, target_info):
        """规划参考状态"""
        ego_speed, _ = self.get_ego_state()
        if target_info is None or len(target_info) < 8 or not all(np.isfinite(target_info)):
            self.mode = ACCMode.CRUISE
            return self.target_speed, np.inf, 0.0, 0.0  # ref_speed, ref_dist, ref_y_offset, ref_y_vel

        dist = float(target_info[0])  # 纵向距离
        rel_vel = float(target_info[6])  # 纵向相对速度
        y_offset = float(target_info[1])  # 横向偏移
        y_vel = -float(target_info[7])  # 横向速度（假设前车横向速度为 0）

        if not all(np.isfinite([dist, rel_vel, y_offset, y_vel])):
            self.mode = ACCMode.CRUISE
            return self.target_speed, np.inf, 0.0, 0.0

        lead_speed = ego_speed + rel_vel
        safe_dist = self.compute_safe_distance(ego_speed)

        if dist > self.max_follow_distance:
            self.mode = ACCMode.CRUISE
            ref_speed = self.target_speed
            ref_dist = np.inf
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

    def mpc_control(self, ref_speed, ref_dist, ref_y_offset, ref_y_vel, dist, rel_vel, ego_speed, y_offset, y_vel):
        # 检查输入是否有效
        if not all(
                np.isfinite([dist, rel_vel, ego_speed, y_offset, y_vel, ref_speed, ref_dist, ref_y_offset, ref_y_vel])):
            print("Invalid MPC inputs, using previous controls")
            return self.prev_accel, self.prev_steer

        # 初始状态向量 [dist, rel_vel, ego_speed, y_offset, y_vel]
        x = np.array([dist, rel_vel, ego_speed, y_offset, y_vel], dtype=np.float64)

        # 定义控制变量 [加速度, 转向角]
        u = cp.Variable((self.horizon, 2))
        states = [x]

        # 定义状态转移矩阵 A（固定部分）
        A = np.array([[1, self.control_dt, 0, 0, 0],
                      [0, 1, -self.control_dt, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, self.control_dt],
                      [0, 0, 0, 0, 1]], dtype=np.float64)

        # 定义固定的 B 矩阵（仅包含常数项）
        B_fixed = np.array([[0, 0],
                            [0, 0],
                            [self.control_dt, 0],
                            [0, 0],
                            [0, 0]], dtype=np.float64)

        cost = 0
        constraints = []

        # 遍历预测horizon
        for t in range(self.horizon):
            state = states[-1]
            dist_t, rel_vel_t, ego_speed_t, y_offset_t, y_vel_t = state
            a_t, delta_t = u[t, 0], u[t, 1]

            # 定义成本函数
            if self.mode == "CRUISE":
                cost += self.Q_vel * cp.square(ego_speed_t - ref_speed)
            elif self.mode == "STOP":
                cost += self.Q_dist * cp.square(dist_t - ref_dist) + self.Q_vel * cp.square(ego_speed_t)
            else:
                cost += self.Q_dist * cp.square(dist_t - ref_dist) + self.Q_vel * cp.square(ego_speed_t - ref_speed)

            cost += self.Q_y * cp.square(y_offset_t - ref_y_offset) + self.Q_yvel * cp.square(y_vel_t - ref_y_vel)
            cost += self.R_accel * cp.square(a_t) + self.R_steer * cp.square(delta_t)

            # 状态更新（直接使用 cvxpy 表达式）
            dist_next = dist_t + self.control_dt * rel_vel_t
            rel_vel_next = rel_vel_t - self.control_dt * a_t
            ego_speed_next = ego_speed_t + self.control_dt * a_t
            y_offset_next = y_offset_t + self.control_dt * y_vel_t
            y_vel_next = y_vel_t + self.control_dt * (ego_speed_t * delta_t)  # 动态部分

            # 组合下一状态
            next_state = cp.vstack([dist_next, rel_vel_next, ego_speed_next, y_offset_next, y_vel_next])
            states.append(next_state)

        # 添加控制约束
        constraints += [u[:, 0] >= self.max_decel, u[:, 0] <= self.max_accel,
                        u[:, 1] >= -self.max_steer_angle, u[:, 1] <= self.max_steer_angle]

        # 求解优化问题
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
            control.brake = min(-accel / -self.max_decel, 1.0)

        # 横向控制（归一化到 [-1, 1]）
        control.steer = min(max(steer / self.max_steer_angle, -1.0), 1.0)

        # 平滑处理
        control.throttle = self.smooth_alpha * control.throttle + (1 - self.smooth_alpha) * self.prev_control.throttle
        control.brake = self.smooth_alpha * control.brake + (1 - self.smooth_alpha) * self.prev_control.brake
        control.steer = self.smooth_alpha * control.steer + (1 - self.smooth_alpha) * self.prev_control.steer

        self.prev_control = control
        return control

    def update(self, target_info):
        """更新控制逻辑"""
        if not hasattr(self, 'prev_control'):
            self.prev_control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)

        ego_speed, ego_accel = self.get_ego_state()
        ref_speed, ref_dist, ref_y_offset, ref_y_vel = self.plan(target_info)

        if target_info is not None and len(target_info) >= 8 and all(np.isfinite(target_info)):
            dist = float(target_info[0])
            rel_vel = float(target_info[6])
            y_offset = float(target_info[1])
            y_vel = -float(target_info[7])
        else:
            dist = np.inf
            rel_vel = 0.0
            y_offset = 0.0  # 无目标时假设保持当前车道
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