import carla
import math
import numpy as np
import cv2
import csv
import time
import lane_detection
import kalman_filter
import radar_cluster
import pygame
import threading
from acc_planning_control import ACCPlanningControl


# 这个类负责以正弦波模式控制目标车辆的速度
class SinusoidalSpeedController:
    def __init__(self, vehicle, base_speed=30.0, amplitude=10.0, period=30.0):
        """
        初始化正弦速度控制器

        参数:
        vehicle - 要控制的Carla车辆
        base_speed - 基础速度 (km/h)
        amplitude - 正弦波幅度 (km/h)
        period - 正弦波周期 (秒)
        """
        self.vehicle = vehicle
        self.base_speed = base_speed
        self.amplitude = amplitude
        self.period = period
        self.start_time = time.time()
        self.tm = None

    def set_traffic_manager(self, tm):
        """设置交通管理器引用"""
        self.tm = tm

    def update(self):
        """更新目标车辆的速度"""
        if self.vehicle is None:
            return

        # 计算当前的正弦速度
        current_time = time.time() - self.start_time
        phase = (2 * math.pi * current_time) / self.period
        speed_factor = self.base_speed + self.amplitude * math.sin(phase)

        # 确保速度为正值
        if speed_factor < 5.0:
            speed_factor = 5.0

        # 转换为百分比速度差
        # Traffic Manager使用百分比差值: 0表示遵守限速，正值表示低于限速的百分比
        if self.tm:
            # 假设限速是50km/h，我们要计算相对于限速的百分比差
            speed_limit = 50.0  # 假设的限速值
            percentage_diff = ((speed_limit - speed_factor) / speed_limit) * 100
            self.tm.vehicle_percentage_speed_difference(self.vehicle, percentage_diff)

            # 打印当前设定的速度
            print(f"Target vehicle speed set to {speed_factor:.2f} km/h (percentage diff: {percentage_diff:.2f}%)")
        else:
            # 如果没有交通管理器，直接控制速度
            # 将km/h转换为m/s
            target_speed = speed_factor / 3.6

            # 获取当前车辆速度
            current_velocity = self.vehicle.get_velocity()
            current_speed = math.sqrt(current_velocity.x ** 2 + current_velocity.y ** 2 + current_velocity.z ** 2)

            # 计算需要的加速度
            control = self.vehicle.get_control()

            # 简单的比例控制
            if target_speed > current_speed:
                # 需要加速
                control.throttle = min(1.0, (target_speed - current_speed) / 5.0)
                control.brake = 0.0
            else:
                # 需要减速
                control.throttle = 0.0
                control.brake = min(1.0, (current_speed - target_speed) / 5.0)

            # 保持原有的转向控制（由Carla自动驾驶处理）
            # 应用控制
            self.vehicle.apply_control(control)

            print(f"Target vehicle direct speed control: {speed_factor:.2f} km/h")


class acc:
    def __init__(self):
        self.tracker = kalman_filter.RadarTracker()
        self.lane_detector = lane_detection.LaneDetector()
        self.radar_point_cluster = radar_cluster.RadarClusterNode()
        self.max_follow_distance = 50
        self.radar_detections = []
        self.latest_camera_image = None
        self.radar_2_world = []
        self.world_2_camera = []
        self.cluster = []
        self.track_id = []
        self.image_width = 1280
        self.image_height = 720
        self.target_vehicle = None
        self.start_time = None
        self.csv_file = None
        self.csv_writer = None
        self.init_carla()
        self.init_csv()

    def init_carla(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        self.world = self.client.load_world('Town04', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.world.load_map_layer(carla.MapLayer.Buildings)
        blueprint_library = self.world.get_blueprint_library()
        map = self.world.get_map()
        self.manual_mode = False
        self.manual_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        if self.manual_mode:
            pygame.init()
            pygame.display.set_mode((200, 200))
            pygame.key.set_repeat(50, 50)
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        ego_vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]
        spawn_points = map.get_spawn_points()[0]
        waypoint = map.get_waypoint(spawn_points.location)
        lanes = []
        current_waypoint = waypoint
        for _ in range(3):
            lanes.append(current_waypoint)
            current_waypoint = current_waypoint.get_left_lane()
            if current_waypoint is None:
                break
        vehicles = []
        locations = []
        if len(lanes) > 1:
            target_lane = lanes[1]
            transform = target_lane.transform
            transform.location.z += 0.5
            locations.append(transform.location)
            target_vehicle = self.world.spawn_actor(vehicle_bp, transform)
            vehicles.append(target_vehicle)
            self.target_vehicle = target_vehicle

            # 初始化正弦速度控制器
            self.target_speed_controller = SinusoidalSpeedController(
                vehicle=target_vehicle,
                base_speed=30.0,  # 基础速度30km/h
                amplitude=10.0,  # 振幅10km/h (速度会在20-40km/h之间变化)
                period=10.0  # 10秒一个周期
            )
        else:
            raise ValueError("无法找到中间车道，无法生成目标车辆")
        ego_spawn_point = spawn_points
        ego_spawn_point.location = locations[0]
        ego_spawn_point.location.x -= 20
        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, ego_spawn_point)
        vehicles.append(self.ego_vehicle)
        self.vehicles = vehicles
        tm = self.client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.0)
        tm.set_synchronous_mode(True)
        self.tm_port = tm.get_port()

        # 让目标车辆使用自动驾驶来处理横向控制
        for vehicle in vehicles[:-1]:
            vehicle.set_autopilot(True, self.tm_port)
            tm.auto_lane_change(vehicle, False)
            # 初始设置一个速度，后面会由控制器动态调整
            tm.vehicle_percentage_speed_difference(vehicle, 60.0)

        # 设置速度控制器的交通管理器
        if hasattr(self, 'target_speed_controller'):
            self.target_speed_controller.set_traffic_manager(tm)

        if not self.manual_mode:
            vehicles[-1].set_autopilot(True, self.tm_port)
            tm.auto_lane_change(vehicles[-1], False)
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('range', '100.0')
        radar_bp.set_attribute('horizontal_fov', '120.0')
        radar_bp.set_attribute('vertical_fov', '30.0')
        radar_bp.set_attribute('points_per_second', '10000')
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.5))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('points_per_second', '1000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('lower_fov', '-10')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)

    def init_csv(self):
        self.csv_file = open('speed_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Time(s)', 'Ego_Speed(km/h)', 'Target_Speed(km/h)', 'Target_Desired_Speed(km/h)','Distance(m)'])
        print("CSV file 'speed_data.csv' created and header written.")

    def get_vehicle_speed(self, vehicle):
        velocity = vehicle.get_velocity()
        speed_m_s = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_kmh = speed_m_s * 3.6
        return speed_kmh

    def radar_callback(self, radar_data):
        self.radar_points = []
        self.filted_points = []
        ego_velocity = self.get_vehicle_speed(self.ego_vehicle) / 3.6
        velocity_tolerance = 1.0
        for detection in radar_data:
            try:
                distance = detection.depth
                azimuth = math.degrees(detection.azimuth)
                altitude = math.degrees(detection.altitude)
                velocity = detection.velocity
                x = distance * math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth))
                y = -distance * math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth))
                z = distance * math.sin(math.radians(altitude))
                vx = velocity * math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth))
                vy = velocity * math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth))
                vz = velocity * math.sin(math.radians(altitude))
                expected_static_velocity = -ego_velocity * math.cos(math.radians(azimuth)) * math.cos(
                    math.radians(altitude))
                if z > -0.5:
                    self.radar_points.append([x, y, z, vx, vy, vz, velocity])
                    if abs(velocity - expected_static_velocity) > velocity_tolerance:
                        self.filted_points.append([x, y, z, vx, vy, vz, velocity])
            except AttributeError as e:
                print(f"AttributeError: {e}. Raw detection: {detection}")
        if self.filted_points:
            self.cluster = self.radar_point_cluster.radar_cluster(self.filted_points)
            if self.cluster:
                self.track_id = self.tracker.update(self.cluster)
                for track in self.track_id:
                    if not all(np.isfinite(track)):
                        print(f"Invalid track data: {track}")
                        self.track_id = []
                        break
            else:
                self.track_id = []
                print("No clusters found")
        else:
            self.track_id = []
            print("No filtered points")

    def camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        self.latest_camera_image = array

    def lidar_callback(self, lidar_data):
        points = []
        for point in lidar_data:
            x = point.point.x
            y = point.point.y
            z = point.point.z
            intensity = point.intensity
            points.append([x, y, z, intensity])
        self.latest_lidar_points = points

    def get_extrinsic_params(self, radar_sensor, camera_sensor):
        self.radar_2_world = radar_sensor.get_transform().get_matrix()
        self.world_2_camera = np.array(camera_sensor.get_transform().get_inverse_matrix())

    def project_radar_to_camera(self, radar_points, image_width=1280, image_height=720, fov=90):
        fx = image_width / (2.0 * np.tan(fov * np.pi / 360.0))
        fy = image_height / (2.0 * np.tan(fov * np.pi / 360.0))
        cx = image_width / 2
        cy = image_height / 2
        projected_points = []
        for x, y, z, w, l, h, vx, vy, vz, id in radar_points:
            radar_point = np.array([x, y, z, 1])
            world_point = np.dot(self.radar_2_world, radar_point)
            camera_point = np.dot(self.world_2_camera, world_point)
            point_in_camera_coords = np.array([
                camera_point[1],
                camera_point[2] * -1,
                camera_point[0]])
            u = cx + (fx * point_in_camera_coords[0] / point_in_camera_coords[2])
            v = cy + (fy * point_in_camera_coords[1] / point_in_camera_coords[2])
            ipm_point = np.dot(self.lane_detector.M, np.array([u, v - 300, 1]))
            ipm_point[0] = ipm_point[0] / ipm_point[2]
            ipm_point[1] = ipm_point[1] / ipm_point[2]
            projected_points.append([int(u), int(v), int(ipm_point[0]), int(ipm_point[1])])
        return projected_points

    def get_ipm_transform_matrix(self, camera_sensor, K, image_width=1280, image_height=720):
        camera_height = camera_sensor.get_transform().location.z
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        src_points = np.float32([
            [image_width * 0.2, image_height],
            [image_width * 0.8, image_height],
            [image_width * 0.6, image_height * 0.4],
            [image_width * 0.4, image_height * 0.4]
        ])
        ground_width = 10.0
        ground_length = 20.0
        dst_points = np.float32([
            [-ground_width / 2, 0],
            [ground_width / 2, 0],
            [ground_width / 2, ground_length],
            [-ground_width / 2, ground_length]
        ])
        H, _ = cv2.findHomography(src_points, dst_points)
        print("Camera IPM Transformation Matrix (Homography H):")
        print(H)
        return H

    def handle_manual_control(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.manual_mode = not self.manual_mode
                    print(f"Switched to {'manual' if self.manual_mode else 'autonomous'} mode")
                    if self.manual_mode:
                        self.manual_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        if self.manual_mode:
            keys = pygame.key.get_pressed()
            throttle_increment = 0.1
            steer_increment = 0.1
            if keys[pygame.K_w]:
                self.manual_control.throttle = min(self.manual_control.throttle + throttle_increment, 1.0)
            if keys[pygame.K_s]:
                self.manual_control.throttle = max(self.manual_control.throttle - throttle_increment, 0.0)
            if keys[pygame.K_a]:
                self.manual_control.steer = max(self.manual_control.steer - steer_increment, -1.0)
            if keys[pygame.K_d]:
                self.manual_control.steer = min(self.manual_control.steer + steer_increment, 1.0)
            if keys[pygame.K_SPACE]:
                self.manual_control.brake = 1.0
                self.manual_control.throttle = 0.0
            else:
                self.manual_control.brake = 0.0
            if not (keys[pygame.K_a] or keys[pygame.K_d]):
                self.manual_control.steer *= 0.9
            self.ego_vehicle.apply_control(self.manual_control)
            print(
                f"Manual: Throttle={self.manual_control.throttle:.2f}, Steer={self.manual_control.steer:.2f}, Brake={self.manual_control.brake:.2f}")

    def get_vehicle_distance(self, vehicle1, vehicle2):
        """计算两个车辆之间的距离（米）"""
        if vehicle1 is None or vehicle2 is None:
            return float('inf')  # 如果任一车辆不存在，返回无穷大

        loc1 = vehicle1.get_location()
        loc2 = vehicle2.get_location()

        # 计算2维欧氏距离
        distance = math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2 )
        return distance

    def generate_target(self):
        acc_controller = ACCPlanningControl(self.ego_vehicle, target_speed_kmh=40.0, time_gap=2.0,
                                            max_follow_distance=self.max_follow_distance)
        try:
            print("Starting radar, camera, LIDAR, and ACC control...")
            self.get_extrinsic_params(self.radar, self.camera)
            self.start_time = time.time()
            while True:
                # 更新目标车辆的速度（正弦波模式）
                current_desired_speed = 0.0
                if hasattr(self, 'target_speed_controller'):
                    self.target_speed_controller.update()
                    # 获取当前期望速度用于记录
                    current_time = time.time() - self.target_speed_controller.start_time
                    phase = (2 * math.pi * current_time) / self.target_speed_controller.period
                    current_desired_speed = self.target_speed_controller.base_speed + self.target_speed_controller.amplitude * math.sin(
                        phase)

                self.world.tick()
                if self.manual_mode:
                    self.handle_manual_control()
                else:
                    if self.latest_camera_image is not None:
                        image_with_radar = self.latest_camera_image.copy()
                        ego_speed = self.get_vehicle_speed(self.ego_vehicle)
                        target_speed = self.get_vehicle_speed(self.target_vehicle) if self.target_vehicle else 0.0

                        # 计算两车之间的距离
                        vehicle_distance = self.get_vehicle_distance(self.ego_vehicle, self.target_vehicle)

                        # 记录数据，增加目标期望速度
                        current_time = time.time() - self.start_time
                        self.csv_writer.writerow([current_time, ego_speed, target_speed, current_desired_speed,vehicle_distance])
                        self.csv_file.flush()

                        # 在画面上增加显示目标期望速度
                        cv2.putText(image_with_radar, f"Ego Speed: {ego_speed:.2f} km/h",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(image_with_radar, f"Target Speed: {target_speed:.2f} km/h",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        # cv2.putText(image_with_radar, f"Target Desired: {current_desired_speed:.2f} km/h",
                        #             (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

                        lane_windows, lane_image = self.lane_detector.lane_detect(image_with_radar)
                        track_id = self.track_id.copy() if self.track_id is not None else []
                        target_info = None
                        if track_id:
                            projected_points = self.project_radar_to_camera(track_id)
                            current_target_idx = -1
                            for idx in range(len(track_id)):
                                u, v, ipm_u, ipm_v = projected_points[idx]
                                color = (255, 0, 0)
                                cv2.circle(image_with_radar, (u, v), 5, color, -1)
                                if ipm_v < 0:
                                    ipm_v = 0
                                curve_left_dis = 0
                                curve_right_dis = 0
                                if ((-2 + curve_left_dis) < track_id[idx][1] < (2 + curve_right_dis)):
                                    if ((current_target_idx != -1) and (
                                            track_id[idx][0] < track_id[current_target_idx][0])) or (
                                            current_target_idx == -1):
                                        current_target_idx = idx
                            if current_target_idx >= 0:
                                cv2.circle(image_with_radar, (
                                    projected_points[current_target_idx][0], projected_points[current_target_idx][1]),
                                           10,
                                           (255, 255, 255), -1)
                                cv2.putText(image_with_radar, "id=" + str(track_id[current_target_idx][-1]),
                                            (projected_points[current_target_idx][0] + 5,
                                             projected_points[current_target_idx][1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 225, 100), 2)
                                target_info = track_id[current_target_idx]
                                print(f"Selected target info: {target_info}")
                                if not all(np.isfinite(target_info)):
                                    print("Invalid target info, skipping control")
                                    target_info = None
                        control = acc_controller.update(target_info)
                        self.ego_vehicle.apply_control(control)
                        cv2.imshow("Radar and Objects on Camera", image_with_radar)
                        cv2.waitKey(1)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print("Cleaning up...")
            cv2.destroyAllWindows()
            self.csv_file.close()
            self.destroy()

    def destroy(self):
        self.radar.stop()
        self.camera.stop()
        self.lidar.stop()
        self.radar.destroy()
        self.camera.destroy()
        self.lidar.destroy()
        for vehicle in self.vehicles:
            vehicle.set_autopilot(False, self.tm_port)
            vehicle.destroy()
        print(f"Destroyed {len(self.vehicles)} vehicles, radar, camera, and LIDAR.")


def main():
    acc_actor = acc()
    thread_1 = threading.Thread(target=acc_actor.radar.listen, args=(acc_actor.radar_callback,), name='T1')
    thread_2 = threading.Thread(target=acc_actor.camera.listen, args=(acc_actor.camera_callback,), name='T2')
    thread_3 = threading.Thread(target=acc_actor.lidar.listen, args=(acc_actor.lidar_callback,), name='T3')
    thread_1.start()
    thread_2.start()
    thread_3.start()
    try:
        acc_actor.generate_target()
    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        thread_1.join()
        thread_2.join()
        thread_3.join()
        print("All threads terminated.")


if __name__ == '__main__':
    main()