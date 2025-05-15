import carla
import threading
import time
import math
import numpy as np

class TargetVehicleController:
    def __init__(self, client, target_vehicle, map, speed_range=(30.0, 70.0)):
        self.client = client
        self.target_vehicle = target_vehicle
        self.map = map
        self.tm = client.get_trafficmanager(8000)
        self.min_speed = speed_range[0]  # km/h
        self.max_speed = speed_range[1]  # km/h

        # 速度控制参数
        self.base_speed_diff = 20.0  # 对应于70%的限速
        self.curve_speed_diff =40.0  # 对应于30%的限速
        self.current_speed_diff = self.base_speed_diff
        self.target_speed_diff = self.base_speed_diff

        # 平滑过渡参数
        self.speed_change_rate = 0.2  # 每次更新速度差的变化率
        self.straight_oscillation_period = 60.0  # 直道上速度变化的周期（秒）
        self.oscillation_amplitude = 10.0  # 直道上速度变化的幅度（百分比差值）

        # 更新线程控制
        self.update_running = False
        self.update_thread = None
        self.update_interval = 0.1  # 秒

        # 弯道检测参数
        self.look_ahead_distance = 20.0  # 前方检测距离（米）
        self.yaw_change_threshold = 10.0  # 偏航角变化阈值（度）

        # 坡度检测参数
        self.pitch_threshold = 5.0  # 坡度阈值（度）
        self.uphill_speed_diff = 50.0  # 上坡时增加速度差值
        self.downhill_speed_diff = 30.0  # 下坡时减少速度差值

        # 记录起始时间，用于周期性速度变化
        self.start_time = time.time()

    def start_update_thread(self):
        if self.update_thread is not None and self.update_thread.is_alive():
            return
        self.update_running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

    def stop_update_thread(self):
        self.update_running = False
        if self.update_thread is not None:
            self.update_thread.join(timeout=2.0)

    def _update_loop(self):
        last_update_time = time.time()
        while self.update_running:
            current_time = time.time()
            dt = current_time - last_update_time

            # 检查当前道路状态
            is_curved = self.is_curved_road()
            pitch = self.get_road_pitch()

            # 根据道路类型和坡度设置目标速度差值
            if abs(pitch) > self.pitch_threshold:
                if pitch > 0:  # 上坡
                    self.target_speed_diff = self.uphill_speed_diff
                else:  # 下坡
                    self.target_speed_diff = self.downhill_speed_diff
            elif is_curved:
                self.target_speed_diff = self.curve_speed_diff
            else:
                elapsed_time = current_time - self.start_time
                oscillation = self.oscillation_amplitude * math.sin(
                    2 * math.pi * elapsed_time / self.straight_oscillation_period)
                self.target_speed_diff = max(10.0, min(90.0, self.base_speed_diff + oscillation))

            # 平滑过渡到目标速度
            if abs(self.current_speed_diff - self.target_speed_diff) > 0.1:
                direction = 1 if self.target_speed_diff > self.current_speed_diff else -1
                change_amount = self.speed_change_rate * direction
                self.current_speed_diff += change_amount
                min_diff = (1.0 - self.max_speed / 100.0) * 100.0
                max_diff = (1.0 - self.min_speed / 100.0) * 100.0
                self.current_speed_diff = max(min_diff, min(max_diff, self.current_speed_diff))
                self.tm.vehicle_percentage_speed_difference(self.target_vehicle, self.current_speed_diff)

            last_update_time = current_time
            time.sleep(self.update_interval)

    def is_curved_road(self):
        vehicle_transform = self.target_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)

        # 获取前方路点
        next_waypoints = vehicle_waypoint.next(self.look_ahead_distance)
        if not next_waypoints:
            return False

        # 计算偏航角变化
        current_yaw = vehicle_waypoint.transform.rotation.yaw
        max_yaw_change = 0.0
        for wp in next_waypoints:
            wp_yaw = wp.transform.rotation.yaw
            yaw_diff = abs(wp_yaw - current_yaw)
            yaw_diff = min(yaw_diff, 360 - yaw_diff)  # 考虑偏航角的周期性
            max_yaw_change = max(max_yaw_change, yaw_diff)
            current_yaw = wp_yaw

        return max_yaw_change > self.yaw_change_threshold

    def get_road_pitch(self):
        vehicle_transform = self.target_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)
        return vehicle_waypoint.transform.rotation.pitch

    def get_current_speed(self):
        velocity = self.target_vehicle.get_velocity()
        speed_m_s = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        return speed_m_s * 3.6