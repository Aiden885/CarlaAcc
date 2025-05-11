import numpy as np
import cvxpy as cp
import carla
from enum import Enum
import math

class ACCMode(Enum):
    CRUISE = 1      # 定速巡航
    FOLLOW = 2      # 跟车
    STOP = 3       # 刹停
    EMERGENCY = 4   # 紧急制动

class ACCPlanningControl:
    def __init__(self, ego_vehicle, target_speed_kmh=60.0, time_gap=2.0, max_follow_distance=50.0):
        self.ego_vehicle = ego_vehicle
        self.target_speed = target_speed_kmh / 3.6
        self.time_gap = time_gap
        self.max_follow_distance = max_follow_distance
        self.min_safe_distance = 5.0
        self.max_accel = 2.0
        self.max_decel = -3.0
        self.control_dt = 0.1
        self.prev_accel = 0.0
        self.smooth_alpha = 0.7
        self.horizon = 10
        self.Q_dist = 1.0
        self.Q_vel = 0.5
        self.R_accel = 0.2
        self.mode = ACCMode.CRUISE
        if not hasattr(self, 'prev_control'):
            self.prev_control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)

    def get_ego_state(self):
        velocity = self.ego_vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        acceleration = self.ego_vehicle.get_acceleration()
        accel = acceleration.x
        return speed, accel

    def compute_safe_distance(self, ego_speed):
        return max(ego_speed * self.time_gap + self.min_safe_distance, self.min_safe_distance)

    def plan(self, target_info):
        ego_speed, _ = self.get_ego_state()
        if target_info is None or len(target_info) < 7 or not all(np.isfinite(target_info)):
            self.mode = ACCMode.CRUISE
            return self.target_speed, np.inf
        dist = float(target_info[0])
        rel_vel = float(target_info[6])
        if not np.isfinite(dist) or not np.isfinite(rel_vel):
            self.mode = ACCMode.CRUISE
            return self.target_speed, np.inf
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
        return ref_speed, ref_dist

    def mpc_control(self, ref_speed, ref_dist, dist, rel_vel, ego_speed):
        if self.mode == ACCMode.EMERGENCY:
            return self.max_decel
        if not all(np.isfinite([dist, rel_vel, ego_speed, ref_speed, ref_dist])):
            print("Invalid MPC inputs, using previous acceleration")
            return self.prev_accel
        x = np.array([dist, rel_vel, ego_speed], dtype=np.float64)
        u = cp.Variable(self.horizon)
        states = [x]
        A = np.array([[1, self.control_dt, 0],
                      [0, 1, -self.control_dt],
                      [0, 0, 1]], dtype=np.float64)
        B = np.array([0, 0, self.control_dt], dtype=np.float64)  # 修改为 (3,)
        cost = 0
        for t in range(self.horizon):
            state = states[-1]
            dist_t, rel_vel_t, ego_speed_t = state
            if self.mode == ACCMode.CRUISE:
                cost += self.Q_vel * cp.square(ego_speed_t - ref_speed)
            elif self.mode == ACCMode.STOP:
                cost += self.Q_dist * cp.square(dist_t - ref_dist) + self.Q_vel * cp.square(ego_speed_t)
            else:
                cost += self.Q_dist * cp.square(dist_t - ref_dist) + self.Q_vel * cp.square(ego_speed_t - ref_speed)
            cost += self.R_accel * cp.square(u[t])
            next_state = A @ state + B * u[t]  # B 是 (3,)，结果为 (3,)
            states.append(next_state)
        constraints = [u >= self.max_decel, u <= self.max_accel]
        try:
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(solver=cp.ECOS)
            if prob.status != cp.OPTIMAL or u.value is None:
                print("MPC optimization failed, using previous acceleration")
                return self.prev_accel
            return float(u.value[0])
        except cp.SolverError:
            print("MPC solver error, using previous acceleration")
            return self.prev_accel

    def accel_to_control(self, accel):
        control = carla.VehicleControl()
        control.manual_gear_shift = False
        control.gear = 1
        if accel > 0:
            control.throttle = min(accel / self.max_accel, 1.0)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-accel / -self.max_decel, 1.0)
        control.throttle = self.smooth_alpha * control.throttle + (1 - self.smooth_alpha) * self.prev_control.throttle
        control.brake = self.smooth_alpha * control.brake + (1 - self.smooth_alpha) * self.prev_control.brake
        self.prev_control = control
        return control

    def update(self, target_info):
        if not hasattr(self, 'prev_control'):
            self.prev_control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)
        ego_speed, ego_accel = self.get_ego_state()
        ref_speed, ref_dist = self.plan(target_info)
        dist = float(target_info[0]) if target_info is not None and len(target_info) >= 7 and np.isfinite(target_info[0]) else np.inf
        rel_vel = float(target_info[6]) if target_info is not None and len(target_info) >= 7 and np.isfinite(target_info[6]) else 0.0
        accel = self.mpc_control(ref_speed, ref_dist, dist, rel_vel, ego_speed)
        accel = self.smooth_alpha * accel + (1 - self.smooth_alpha) * self.prev_accel
        self.prev_accel = accel
        control = self.accel_to_control(accel)
        print(f"Mode: {self.mode}, Ego Speed: {ego_speed:.2f} m/s, "
              f"Dist: {dist:.2f} m, Ref Speed: {ref_speed:.2f} m/s, "
              f"Accel: {accel:.2f} m/s², Throttle: {control.throttle:.2f}, Brake: {control.brake:.2f}")
        return control

def main():
    # 示例主函数，用于测试
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 获取 ego vehicle（需与 acc.py 一致）
    ego_vehicle = None
    for actor in world.get_actors():
        if 'vehicle.audi.etron' in actor.type_id:
            ego_vehicle = actor
            break
    if ego_vehicle is None:
        raise ValueError("Ego vehicle not found")

    # 初始化 ACC 控制器
    acc_controller = ACCPlanningControl(ego_vehicle, target_speed_kmh=60.0, time_gap=2.0)

    try:
        # 模拟感知数据（实际应从 acc.py 获取）
        target_info = [20.0, 0.0, 0.0, 4.8, 1.8, 1.5, -2.0, 0.0, 0.0, 1]  # [x, y, z, w, l, h, vx, vy, vz, track_id]
        while True:
            world.tick()
            control = acc_controller.update(target_info)
            ego_vehicle.apply_control(control)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

if __name__ == '__main__':
    main()