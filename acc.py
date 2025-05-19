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
from collections import deque
from acc_planning_control import ACCPlanningControl
from sinusoidal_speed_controller import SinusoidalSpeedController

from realtime_trajectory_manager import BSplineTrajectoryManager, RealTimeTrajectoryBuffer


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
        self.target_speed_controller = None

        # === 新增：轨迹管理系统 ===
        # 1. 创建轨迹缓冲区
        self.trajectory_buffer = RealTimeTrajectoryBuffer(
            max_length=50.0,  # 最大轨迹长度50米
            spatial_resolution=0.3  # 空间分辨率0.3米/点
        )

        # 2. 创建轨迹管理器
        self.trajectory_manager = BSplineTrajectoryManager(
            trajectory_buffer=self.trajectory_buffer,
            min_points_for_fit=5,  # 最少5个点才进行拟合
            fit_interval=3  # 每3帧更新一次拟合
        )

        # 3. 性能监控
        self.frame_times = deque(maxlen=100)
        self.last_frame_time = None

        # 4. 轨迹状态监控
        self.trajectory_stats = {
            'total_updates': 0,
            'successful_fits': 0,
            'last_waypoint_count': 0
        }

        self.init_carla()
        self.init_csv()


    def init_carla(self):
        # 初始化 Carla 客户端
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        try:
            self.world = self.client.get_world()
            self.world = self.client.load_world('Town05', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load map Town05: {e}")

        # 获取蓝图库和地图
        self.blueprint_library = self.world.get_blueprint_library()
        map = self.world.get_map()

        # #设置同步模式和时间
        # settings = self.world.get_settings()
        # settings.synchronous_mode = True
        # settings.fixed_delta_seconds = 0.05
        # self.world.apply_settings(settings)


        # 获取车辆蓝图
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        ego_vehicle_bp = self.blueprint_library.filter('vehicle.audi.etron')[0]

        # 定义固定生成点（上坡 Town05）
        fixed_point = carla.Location(x=10.663731, y=-203.651886, z=0.5)

        # 找到最近的 waypoint
        waypoint = map.get_waypoint(fixed_point, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            raise RuntimeError("Failed to find a valid waypoint near the specified location")

        # 设置前车生成点（基于 waypoint）
        spawn_point = waypoint.transform
        spawn_point.location.z += 0.2  # 略微抬高以避免地面碰撞

        # 生成目标车辆
        vehicles = []
        target_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if target_vehicle is None:
            raise RuntimeError("Failed to spawn target vehicle at waypoint location")
        vehicles.append(target_vehicle)
        self.target_vehicle = target_vehicle
        self.target_vehicle.set_autopilot(True)

        # 初始化正弦速度控制器
        self.target_speed_controller = SinusoidalSpeedController(
            vehicle=target_vehicle,
            base_speed=20.0,  # 基础速度 20km/h
            amplitude=10.0,  # 振幅 10km/h (速度在 10-30km/h 之间变化)
            period=10.0  # 10秒一个周期
        )

        # 生成自车（后方 20 米）
        ego_spawn_point = carla.Transform()
        ego_spawn_point.location = spawn_point.location
        ego_spawn_point.location.x += 20
        ego_spawn_point.rotation = spawn_point.rotation
        self.ego_vehicle = self.world.try_spawn_actor(ego_vehicle_bp, ego_spawn_point)
        self.ego_vehicle.set_autopilot(True)



        if self.ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")
        self.vehicles = vehicles
        self.ego_vehicle.set_autopilot(True)
        vehicles.append(self.ego_vehicle)


        # 设置交通管理器
        tm = self.client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.0)
        tm.set_synchronous_mode(True)
        self.tm_port = tm.get_port()
        #自车不变道
        tm.auto_lane_change(self.ego_vehicle,False)

        # 目标车辆自动驾驶设置
        for vehicle in vehicles:
            vehicle.set_autopilot(True, self.tm_port)
            tm.auto_lane_change(vehicle, False)
            tm.vehicle_percentage_speed_difference(vehicle, 30.0)

        # 设置速度控制器的交通管理器
        if self.target_speed_controller:
            self.target_speed_controller.set_traffic_manager(tm)

        # 设置所有交通信号灯为绿色
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)  # 锁定为绿色，防止自动切换
        print(f"Set {len(traffic_lights)} traffic lights to green")

        # 配置传感器
        radar_bp = self.blueprint_library.find('sensor.other.radar')
        RADAR_CONFIG = {
            'range': '100.0',
            'horizontal_fov': '120.0',
            'vertical_fov': '30.0',
            'points_per_second': '20000'
        }
        for attr, value in RADAR_CONFIG.items():
            radar_bp.set_attribute(attr, value)
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)

        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.5))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)

        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
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
        self.csv_writer.writerow(['Time(s)', 'Ego_Speed(km/h)', 'Target_Speed(km/h)', 'Target_Desired_Speed(km/h)',
                                  'Distance(m)', 'Target_World_X(m)', 'Target_World_Y(m)', 'Processing_Time(ms)'])
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

        # 修复版的手动控制处理

    def get_vehicle_distance(self, vehicle1, vehicle2):
        """计算两个车辆之间的距离（米）"""
        if vehicle1 is None or vehicle2 is None:
            return float('inf')

        loc1 = vehicle1.get_location()
        loc2 = vehicle2.get_location()

        # 计算2维欧氏距离
        distance = math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)
        return distance

    def find_best_target(self, track_id, projected_points):
        """优化的目标选择算法"""
        current_target_idx = -1
        min_distance = float('inf')

        for idx in range(len(track_id)):
            # 简化的车道判断
            if -2 < track_id[idx][1] < 2:  # Y坐标在车道内
                if track_id[idx][0] < min_distance:  # 选择最近的
                    min_distance = track_id[idx][0]
                    current_target_idx = idx

        return current_target_idx

    def visualize_trajectory_on_image_fast(self, image):
        """可视化轨迹（使用新的轨迹点）"""
        # 获取最近的轨迹点（使用新接口）
        recent_waypoints = self.trajectory_manager.get_control_waypoints(lookahead_distance=30.0)

        if len(recent_waypoints) < 2:
            return

        # 获取自车位置
        ego_location = self.ego_vehicle.get_location()
        ego_x, ego_y = ego_location.x, ego_location.y

        # 可视化参数
        scale = 20
        img_center_x = self.image_width // 2
        img_center_y = self.image_height

        # 绘制轨迹线和点
        points_to_draw = []
        for wp in recent_waypoints:
            # 转换为图像坐标（简化投影）
            rel_x = wp.world_x - ego_x
            rel_y = wp.world_y - ego_y

            img_x = int(img_center_x + rel_y * scale)
            img_y = int(img_center_y - rel_x * scale)

            # 边界检查
            if 0 <= img_x < self.image_width and 0 <= img_y < self.image_height:
                points_to_draw.append((img_x, img_y, wp))

        # 绘制轨迹线（绿色）
        for i in range(1, len(points_to_draw)):
            cv2.line(image, points_to_draw[i - 1][:2], points_to_draw[i][:2], (0, 255, 0), 2)

        # 绘制轨迹点
        for img_x, img_y, wp in points_to_draw[-10:]:  # 只绘制最后10个点
            # 根据曲率改变点的颜色
            if wp.curvature > 0.1:  # 弯道
                color = (0, 255, 255)  # 黄色
            else:  # 直道
                color = (0, 255, 0)  # 绿色
            cv2.circle(image, (img_x, img_y), 3, color, -1)

            # 绘制切线方向
            if len(points_to_draw) > 5:  # 只在有足够点时绘制
                tangent_length = 15
                end_x = int(img_x + tangent_length * math.cos(wp.tangent_angle))
                end_y = int(img_y - tangent_length * math.sin(wp.tangent_angle))
                cv2.arrowedLine(image, (img_x, img_y), (end_x, end_y), (255, 0, 0), 1)

    def update_display_text(self, image, ego_speed, target_speed, desired_speed, trajectory_info, processing_time):
        """更新显示文本，增加轨迹信息"""
        texts = [
            (f"Ego Speed: {ego_speed:.1f} km/h", (10, 30), (0, 255, 0)),
            (f"Target Speed: {target_speed:.1f} km/h", (10, 60), (0, 0, 255)),
            (f"Target Desired: {desired_speed:.1f} km/h", (10, 90), (255, 0, 255)),
            (f"Process Time: {processing_time:.2f}ms", (10, 120), (255, 255, 255))
        ]

        # 添加轨迹信息
        if trajectory_info:
            texts.extend([
                (f"Traj Points: {trajectory_info['waypoint_count']}", (10, 150), (255, 255, 0)),
                (f"Traj Length: {trajectory_info['total_length']:.1f}m", (10, 180), (255, 255, 0)),
                (f"Curvature: {trajectory_info.get('current_curvature', 0.0):.3f}", (10, 210), (255, 255, 0))
            ])

        # 添加平均帧率显示
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1000.0 / avg_frame_time
            texts.append((f"FPS: {fps:.1f}", (10, 240), (0, 255, 255)))

        for text, pos, color in texts:
            cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def show_performance_stats(self):
        """显示性能统计"""
        # 帧率统计
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1000.0 / avg_frame_time
            print(f"\nFrame Rate Statistics:")
            print(f"  Average FPS: {fps:.1f}")
            print(f"  Average frame time: {avg_frame_time:.2f} ms")
            print(f"  Max frame time: {np.max(self.frame_times):.2f} ms")

        # 轨迹处理统计
        traj_stats = self.trajectory_manager.get_performance_stats()
        if traj_stats:
            print(f"\nTrajectory Processing Statistics:")
            for key, value in traj_stats.items():
                print(f"  {key}: {value}")

    def generate_target(self):
        """主循环 - 集成轨迹处理"""
        acc_controller = ACCPlanningControl(
            self.ego_vehicle,
            target_speed_kmh=30.0,
            time_gap=2.0,
            max_follow_distance=self.max_follow_distance
        )

        try:

            self.get_extrinsic_params(self.radar, self.camera)
            self.start_time = time.time()

            frame_count = 0

            while True:
                # 帧时间监控
                current_frame_time = time.perf_counter()
                if self.last_frame_time is not None:
                    frame_time = (current_frame_time - self.last_frame_time) * 1000
                    self.frame_times.append(frame_time)
                self.last_frame_time = current_frame_time

                # 更新目标车辆的速度
                current_desired_speed = 0.0
                if self.target_speed_controller:
                    self.target_speed_controller.update()
                    current_desired_speed = self.target_speed_controller.get_current_desired_speed()

                self.world.tick()


                # 轨迹处理性能计时
                traj_start_time = time.perf_counter()
                if self.latest_camera_image is not None:
                    image_with_radar = self.latest_camera_image.copy()
                    ego_speed = self.get_vehicle_speed(self.ego_vehicle)
                    target_speed = self.get_vehicle_speed(self.target_vehicle) if self.target_vehicle else 0.0
                    vehicle_distance = self.get_vehicle_distance(self.ego_vehicle, self.target_vehicle)

                    #目标检测和轨迹处理 ===
                    track_id = self.track_id.copy() if self.track_id is not None else []
                    target_info = None
                    trajectory_waypoint = None
                    trajectory_info = None

                    if track_id:
                        try:
                            projected_points = self.project_radar_to_camera(track_id)
                            current_target_idx = self.find_best_target(track_id, projected_points)

                            # 绘制所有检测到的目标
                            for idx in range(min(len(track_id), len(projected_points))):
                                if len(projected_points[idx]) >= 2:
                                    u, v = projected_points[idx][0], projected_points[idx][1]
                                    cv2.circle(image_with_radar, (u, v), 5, (255, 0, 0), -1)

                            if current_target_idx >= 0 and current_target_idx < len(projected_points):
                                # 绘制选中的目标
                                u, v = projected_points[current_target_idx][0], projected_points[current_target_idx][1]
                                cv2.circle(image_with_radar, (u, v), 10, (255, 255, 255), -1)

                                cv2.putText(image_with_radar, f"id={track_id[current_target_idx][-1]:.0f}",
                                            (u + 5, v), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 225, 100), 2)

                                target_info = track_id[current_target_idx]
                                if len(target_info) >= 9 and all(np.isfinite(target_info[:9])):
                                    # === 核心：将目标信息传递给轨迹管理器 ===
                                    trajectory_waypoint = self.trajectory_manager.add_target_info_realtime(
                                        target_info, self.ego_vehicle
                                    )
                                    self.trajectory_stats['total_updates'] += 1

                                    # 获取轨迹信息用于显示和记录
                                    trajectory_info = self.trajectory_buffer.get_info()
                                    if trajectory_waypoint:
                                        trajectory_info['current_curvature'] = trajectory_waypoint.curvature
                                        trajectory_info['current_tangent_angle'] = trajectory_waypoint.tangent_angle
                                        self.trajectory_stats['successful_fits'] += 1

                        except Exception as e:
                            print(f"Target detection/trajectory error: {e}")

                    # 计算轨迹处理时间
                    traj_processing_time = (time.perf_counter() - traj_start_time) * 1000

                    # === 准备传递给控制器的轨迹数据 ===
                    control_trajectory = None
                    if self.trajectory_buffer.is_valid():
                        # 获取前瞻轨迹点用于控制
                        control_waypoints = self.trajectory_manager.get_control_waypoints(lookahead_distance=30.0)
                        if control_waypoints:
                            control_trajectory = {
                                'waypoints': control_waypoints,
                                'target_waypoint': trajectory_waypoint,
                                'is_valid': True,
                                'waypoint_count': len(control_waypoints),
                                'curvature_profile': self.trajectory_manager.get_trajectory_curvature_profile()
                            }


                    # 记录数据（增加轨迹信息）
                    target_world_x = trajectory_waypoint.world_x if trajectory_waypoint else 0.0
                    target_world_y = trajectory_waypoint.world_y if trajectory_waypoint else 0.0
                    target_curvature = trajectory_waypoint.curvature if trajectory_waypoint else 0.0
                    target_tangent_angle = trajectory_waypoint.tangent_angle if trajectory_waypoint else 0.0

                    current_time = time.time() - self.start_time
                    self.csv_writer.writerow([
                        current_time, ego_speed, target_speed, current_desired_speed,
                        vehicle_distance, target_world_x, target_world_y, traj_processing_time,
                        trajectory_info['waypoint_count'] if trajectory_info else 0,
                        trajectory_info['total_length'] if trajectory_info else 0.0,
                        target_curvature, target_tangent_angle
                    ])
                    self.csv_file.flush()

                    # 显示信息
                    if frame_count % 5 == 0:  # 每5帧更新一次显示
                        self.update_display_text(image_with_radar, ego_speed, target_speed,
                                                 current_desired_speed, trajectory_info, traj_processing_time)

                    # 可视化轨迹
                    self.visualize_trajectory_on_image_fast(image_with_radar)

                    # 车道线检测（简化版本）
                    try:
                        lane_windows, lane_image, detected_windows = self.lane_detector.lane_detect(image_with_radar)
                        valid_row = None
                        for row in lane_windows:
                            if len(row) == 6 and row[2] == 1 and row[5] == 1:
                                valid_row = row
                                break
                        if valid_row is not None:
                            lane_center = (valid_row[0] + valid_row[3]) / 2
                            cv2.putText(image_with_radar, f"Lane Center: {lane_center:.1f} px", (10, 270),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    except Exception as e:
                        print(f"Lane detection error: {e}")

                    # ===将轨迹传递给控制器 ===
                    try:
                        if control_trajectory and control_trajectory['is_valid']:
                            # 使用轨迹信息进行控制
                            control = acc_controller.update_with_trajectory(target_info, control_trajectory)

                        else:
                            # 回退到原有的点控制
                            control = acc_controller.update(target_info)

                        self.ego_vehicle.apply_control(control)
                        #print(control)
                    except Exception as e:
                        print(f"ACC control error: {e}")

                    # 显示图像
                    cv2.imshow("Radar and Objects on Camera", image_with_radar)
                    cv2.imshow("lane_image", lane_image)
                    # cv2.imwrite("C:\App\carla\CARLA_0.9.14\images\lane_frame_" +str(frame_count) + ".jpg", lane_image)
                    cv2.waitKey(1)

                    frame_count += 1

        except KeyboardInterrupt:
            print("\nStopped by user.")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up...")
            self.show_performance_stats()
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
            vehicle.destroy()
        self.ego_vehicle.destroy()
        print(f"Destroyed {len(self.vehicles)} vehicles, ego vehicle, radar, camera, and LIDAR.")

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