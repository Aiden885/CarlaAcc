import carla
import math
import numpy as np
import cv2
import csv
import time
import lane_detection
import kalman_filter
import radar_cluster
import threading

# Pygame相关
import pygame
from pygame.locals import *

# ACC相关模块
from acc_planning_control import ACCPlanningControl
from sinusoidal_speed_controller import SinusoidalSpeedController
from three_mode_controller import calculate_three_mode_desired_distance, set_three_mode_parameters, get_three_mode_status
from acc_decision import ACCDecisionModule, ACCCommand, ACCState

# 导入显示管理器
from display_manager import DisplayManager


class acc:
    def __init__(self):
        # === 显示管理器初始化 ===
        self.display_manager = DisplayManager(1280, 720)
        self.show_opencv = True  # 是否显示OpenCV窗口

        # === 原有的传感器和检测模块 ===
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

        # === ACC决策模块 ===
        self.acc_decision = ACCDecisionModule(initial_V3_kmh=50.0, initial_G1_m=15.0, initial_time_gap=2.0)
        self.acc_decision.set_debug(True)

        # === 控制状态 ===
        self.acc_control_active = False
        self.manual_control_active = True
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0

        # === 运行控制 ===
        self.running = True

        # 初始化CARLA
        self.init_carla()
        self.init_csv()

        # 初始化三模式参数
        self._sync_three_mode_parameters()

    def init_carla(self):
        # 初始化 Carla 客户端
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        try:
            self.world = self.client.get_world()
            self.world = self.client.load_world('Town05', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load map Town05: {e}")

        # 设置同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 60.0
        self.world.apply_settings(settings)

        # 获取蓝图库和地图
        self.blueprint_library = self.world.get_blueprint_library()
        map = self.world.get_map()

        # 获取车辆蓝图
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        ego_vehicle_bp = self.blueprint_library.filter('vehicle.audi.etron')[0]

        # 定义固定生成点
        fixed_point = carla.Location(x=0.663731, y=-203.651886, z=0.5)
        waypoint = map.get_waypoint(fixed_point, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            raise RuntimeError("Failed to find a valid waypoint near the specified location")

        # 生成目标车辆
        spawn_point = waypoint.transform
        spawn_point.location.z += 0.05

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
            base_speed=70,
            amplitude=5.0,
            period=10.0
        )

        # 生成自车
        ego_spawn_point = carla.Transform()
        ego_spawn_point.location = spawn_point.location
        ego_spawn_point.location.x += 20
        ego_spawn_point.rotation = spawn_point.rotation
        self.ego_vehicle = self.world.try_spawn_actor(ego_vehicle_bp, ego_spawn_point)

        if self.ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")
        self.vehicles = vehicles
        self.ego_vehicle.set_autopilot(False)

        # 初始化显示管理器的相机
        self.display_manager.init_camera_manager(self.ego_vehicle)

        # 设置交通管理器
        tm = self.client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.0)
        tm.set_synchronous_mode(False)
        self.tm_port = tm.get_port()
        tm.auto_lane_change(self.ego_vehicle, False)

        # 目标车辆设置
        for vehicle in vehicles:
            vehicle.set_autopilot(True, self.tm_port)
            tm.auto_lane_change(vehicle, False)
            tm.vehicle_percentage_speed_difference(vehicle, 30.0)

        if self.target_speed_controller:
            self.target_speed_controller.set_traffic_manager(tm)

        # 设置交通灯
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)

        # 配置传感器
        # 雷达
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

        # 相机（用于OpenCV处理）
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.5))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)

        # 激光雷达
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('points_per_second', '1000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('lower_fov', '-10')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)

    def init_csv(self):
        """初始化CSV文件"""
        self.csv_file = open('speed_data_integrated.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Time(s)',
            'Ego_Speed(km/h)',
            'Target_Speed(km/h)',
            'Actual_Distance(m)',
            'Desired_Distance(m)',
            'Control_Mode',
            'Lane_Offset',
            'ACC_State',
            'ACC_Active',
            'V3_Setting',
            'G1_Setting',
            'G2_Setting',
            'Cruise_Mode',
            'Manual_Throttle',
            'Manual_Brake',
            'Manual_Steer'
        ])

    def _sync_three_mode_parameters(self):
        """同步ACC决策参数到三模式控制器"""
        acc_params = self.acc_decision.get_current_parameters()
        set_three_mode_parameters(
            V1_kmh=0,
            V2_kmh=30,
            V3_kmh=acc_params['V3_kmh'],
            G1_m=acc_params['G1_m'],
            G2_s=acc_params['G2_s']
        )

    def handle_keyboard_input(self):
        """处理键盘输入"""
        keys = pygame.key.get_pressed()

        # === 人工介入检测（优先级最高）===
        if self.acc_control_active:
            # 检测任何手动控制输入
            manual_input_detected = (
                    keys[K_w] or keys[K_UP] or  # 油门
                    keys[K_s] or keys[K_DOWN] or  # 刹车
                    keys[K_a] or keys[K_LEFT] or  # 右转
                    keys[K_d] or keys[K_RIGHT] or  # 右转
                    keys[K_SPACE]  # 手刹
            )

            if manual_input_detected:
                ego_speed = self.get_vehicle_speed(self.ego_vehicle)
                # 根据输入类型确定指令
                if keys[K_w] or keys[K_UP]:
                    command = ACCCommand.THROTTLE
                    input_type = "油门"
                else:
                    command = ACCCommand.BRAKE
                    input_type = "刹车/转向/手刹"

                # 立即处理人工介入
                state, mode, msg = self.acc_decision.process_command(command, ego_speed)

                # 更新控制状态 - 修正逻辑
                acc_params = self.acc_decision.get_current_parameters()
                self.acc_control_active = acc_params['is_active']
                self.manual_control_active = True  # 人工介入后强制设为手动模式

                print(f"🚨 人工{input_type}介入: {msg}")
                print(f"💡 当前状态: {self.acc_decision.current_state.value}, 需按1键重新激活ACC")

        # 基础车辆控制（只在手动模式下有效）
        if self.manual_control_active:
            # 油门
            if keys[K_w] or keys[K_UP]:
                self.throttle = min(1.0, self.throttle + 0.02)
            else:
                self.throttle = max(0.0, self.throttle - 0.05)

            # 刹车
            if keys[K_s] or keys[K_DOWN]:
                self.brake = min(1.0, self.brake + 0.05)
            else:
                self.brake = max(0.0, self.brake - 0.1)

            # 转向
            if keys[K_a] or keys[K_LEFT]:
                self.steer = max(-1.0, self.steer - 0.05)
            elif keys[K_d] or keys[K_RIGHT]:
                self.steer = min(1.0, self.steer + 0.05)
            else:
                self.steer = self.steer * 0.9

        # 手刹
        hand_brake = keys[K_SPACE]

        # 应用手动控制
        if self.ego_vehicle and self.manual_control_active:
            control = carla.VehicleControl()
            control.throttle = self.throttle
            control.brake = self.brake
            control.steer = self.steer
            control.hand_brake = hand_brake
            self.ego_vehicle.apply_control(control)

    def handle_events(self):
        """处理事件"""
        # 处理显示管理器的事件
        display_events = self.display_manager.handle_display_events()

        for event_type, event_data in display_events:
            if event_type == 'quit':
                self.running = False
                return

            elif event_type == 'keydown':
                # 退出
                if event_data == K_ESCAPE:
                    self.running = False

                # OpenCV窗口控制
                elif event_data == K_o:
                    self.show_opencv = not self.show_opencv
                    print(f"OpenCV窗口: {'开启' if self.show_opencv else '关闭'}")

                elif event_data == K_p:
                    debug_state = not self.acc_decision.debug
                    self.acc_decision.set_debug(debug_state)
                    print(f"ACC调试模式: {'开启' if debug_state else '关闭'}")

                # ACC控制
                elif event_data == K_1:
                    self._process_acc_command(ACCCommand.ENGAGE)

                elif event_data == K_2:
                    self._process_acc_command(ACCCommand.EXIT)

                elif event_data == K_3:
                    self._process_acc_command(ACCCommand.CRUISE_MODE)

                elif event_data == K_q:
                    if self.acc_control_active:
                        self._process_acc_command(ACCCommand.INCREASE_SPEED)

                elif event_data == K_e:
                    if self.acc_control_active:
                        self._process_acc_command(ACCCommand.DECREASE_SPEED)

                elif event_data == K_r:
                    if self.acc_control_active:
                        self._process_acc_command(ACCCommand.INCREASE_DISTANCE)

                elif event_data == K_t:
                    if self.acc_control_active:
                        self._process_acc_command(ACCCommand.DECREASE_DISTANCE)

    def _process_acc_command(self, command):
        """处理ACC指令"""
        if not self.ego_vehicle:
            return

        ego_speed = self.get_vehicle_speed(self.ego_vehicle)
        target_distance = self.get_vehicle_distance(self.ego_vehicle, self.target_vehicle)
        has_target = target_distance < 50.0

        # === 新增：直接处理增速/减速指令 ===
        if command == ACCCommand.INCREASE_SPEED and self.acc_control_active:
            if hasattr(self, 'acc_controller') and self.acc_controller:
                current_target = self.acc_controller.target_speed * 3.6  # 转换为km/h
                new_target = min(120.0, current_target + 1.0)  # 增加1 km/h
                self.acc_controller.target_speed = new_target / 3.6  # 转换回m/s
                print(f"🎯 增速: {current_target:.1f} → {new_target:.1f} km/h")
                return

        elif command == ACCCommand.DECREASE_SPEED and self.acc_control_active:
            if hasattr(self, 'acc_controller') and self.acc_controller:
                current_target = self.acc_controller.target_speed * 3.6  # 转换为km/h
                new_target = max(20.0, current_target - 1.0)  # 减少1 km/h，最低20km/h
                self.acc_controller.target_speed = new_target / 3.6  # 转换回m/s
                print(f"🎯 减速: {current_target:.1f} → {new_target:.1f} km/h")
                return

        # === 调试：处理指令前的状态 ===
        print(f"\n🔍 处理ACC指令调试:")
        print(f"   指令: {command.value}")
        print(f"   处理前状态: {self.acc_decision.current_state.value}")
        print(f"   处理前控制模式: {self.acc_decision.current_control_mode}")
        print(f"   当前速度: {ego_speed:.1f} km/h")
        print(f"   有前车: {has_target}")

        # === 其他指令继续用原来的决策模块逻辑 ===
        state, mode, msg = self.acc_decision.process_command(
            command, ego_speed, has_target, target_distance if has_target else None)

        # === 调试：处理指令后的状态 ===
        print(f"   处理后状态: {state.value}")
        print(f"   处理后控制模式: {mode.value if mode else None}")
        print(f"   状态转移消息: {msg}")

        acc_params = self.acc_decision.get_current_parameters()
        self.acc_control_active = acc_params['is_active']

        # 修正：只有在ACC激活时才设为自动模式
        if self.acc_control_active:
            self.manual_control_active = False  # ACC激活时关闭手动模式
        # 如果ACC未激活，保持当前的manual_control_active状态

        print(f"   acc_control_active更新为: {self.acc_control_active}")
        print(f"   manual_control_active更新为: {self.manual_control_active}")
        print(f"   is_active从参数: {acc_params['is_active']}")

        # 同步参数（但不包括增速/减速，因为那些直接修改了target_speed）
        if command not in [ACCCommand.INCREASE_SPEED, ACCCommand.DECREASE_SPEED]:
            self._sync_three_mode_parameters()

        print(f"ACC指令 {command.value}: {msg}")
        print(f"🎯 当前速度: {ego_speed:.1f} km/h, ACC激活: {self.acc_control_active}")

        # 显示当前巡航速度（如果ACC激活且有控制器）
        if self.acc_control_active and hasattr(self, 'acc_controller') and self.acc_controller:
            current_cruise_speed = self.acc_controller.target_speed * 3.6
            print(f"🎯 当前巡航速度: {current_cruise_speed:.1f} km/h")

    def get_system_info(self):
        """获取系统状态信息，用于显示"""
        ego_speed = self.get_vehicle_speed(self.ego_vehicle)
        target_distance = self.get_vehicle_distance(self.ego_vehicle, self.target_vehicle)
        has_target = target_distance < 50.0
        acc_params = self.acc_decision.get_current_parameters()
        acc_status = self.acc_decision.get_status_info()

        # 获取当前巡航速度
        cruise_speed_kmh = 0.0
        if hasattr(self, 'acc_controller') and self.acc_controller:
            cruise_speed_kmh = self.acc_controller.target_speed * 3.6

        return {
            'ego_speed': ego_speed,
            'target_distance': target_distance,
            'has_target': has_target,
            'acc_active': self.acc_control_active,
            'acc_state': acc_status['state_description'],
            'cruise_mode': acc_params.get('cruise_mode_active', False),
            'cruise_speed_kmh': cruise_speed_kmh,  # 新增：当前巡航速度
            'V3_kmh': acc_params['V3_kmh'],  # 保留：最大速度限制
            'G1_m': acc_params['G1_m'],
            'G2_s': acc_params['G2_s'],
            'throttle': self.throttle,
            'brake': self.brake,
            'steer': self.steer
        }
    # === 以下是原有的传感器回调和处理函数 ===

    def get_vehicle_speed(self, vehicle):
        velocity = vehicle.get_velocity()
        speed_m_s = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_kmh = speed_m_s * 3.6
        return speed_kmh

    def calculate_desired_following_distance(self, ego_speed_kmh, time_gap=2.0, min_distance=5.0):
        """使用三模式控制计算期望跟车距离"""
        ego_speed_ms = ego_speed_kmh / 3.6
        desired_distance, control_mode = calculate_three_mode_desired_distance(ego_speed_ms)
        return desired_distance, control_mode

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
        else:
            self.track_id = []

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

    def get_vehicle_distance(self, vehicle1, vehicle2):
        """计算两个车辆之间的距离（米）"""
        if vehicle1 is None or vehicle2 is None:
            return float('inf')

        loc1 = vehicle1.get_location()
        loc2 = vehicle2.get_location()

        distance = math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)
        return distance

    def find_best_target(self, track_id, projected_points):
        """优化的目标选择算法"""
        current_target_idx = -1
        min_distance = float('inf')

        for idx in range(len(track_id)):
            if -3 < track_id[idx][1] < 3:  # Y坐标在车道内
                if track_id[idx][0] < min_distance:  # 选择最近的
                    min_distance = track_id[idx][0]
                    current_target_idx = idx

        return current_target_idx

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

    def generate_target(self):
        """主循环 - 完整集成ACC决策、控制和显示"""
        # 创建ACC控制器
        acc_controller = ACCPlanningControl(
            self.ego_vehicle,
            target_speed_kmh=80,
            time_gap=2.0,
            max_follow_distance=self.max_follow_distance
        )

        try:
            self.get_extrinsic_params(self.radar, self.camera)
            self.start_time = time.time()
            frame_count = 0

            print("\n=== ACC Integrated Control System ===")
            print("系统将在Pygame窗口中显示CARLA画面和ACC控制信息")
            print("\n键盘控制:")
            print("  1: ACC开启  2: ACC退出  3: 定速巡航")
            print("  Q/E: 增速/降速  R/T: 增距/降距")
            print("  W/S: 油门/刹车  A/D: 转向")
            print("  C: 切换视角  I: 信息显示  O: OpenCV窗口")
            print("  H: 帮助  P: 调试模式  ESC: 退出")
            print("\n")

            while self.running:
                # 处理事件
                self.handle_events()
                self.handle_keyboard_input()

                # 更新前车速度控制
                if self.target_speed_controller:
                    self.target_speed_controller.update()

                # 世界更新
                if self.world:
                    self.world.tick()

                self.display_manager.tick(60)  # 60 FPS

                # 获取车辆状态
                ego_speed = self.get_vehicle_speed(self.ego_vehicle)
                target_speed = self.get_vehicle_speed(self.target_vehicle) if self.target_vehicle else 0.0
                vehicle_distance = self.get_vehicle_distance(self.ego_vehicle, self.target_vehicle)
                has_target = vehicle_distance < 50.0

                # 获取ACC决策输出
                acc_params = self.acc_decision.get_current_parameters()
                acc_status = self.acc_decision.get_status_info()

                # === OpenCV图像处理（用于雷达和车道检测） ===
                if self.latest_camera_image is not None:
                    image_with_radar = self.latest_camera_image.copy()

                    # 目标检测和轨迹处理
                    track_id = self.track_id.copy() if self.track_id is not None else []
                    target_info = None

                    if track_id:
                        try:
                            projected_points = self.project_radar_to_camera(track_id)
                            current_target_idx = self.find_best_target(track_id, projected_points)

                            # 绘制检测目标
                            for idx in range(min(len(track_id), len(projected_points))):
                                if len(projected_points[idx]) >= 2:
                                    u, v = projected_points[idx][0], projected_points[idx][1]
                                    cv2.circle(image_with_radar, (u, v), 5, (255, 0, 0), -1)

                            if current_target_idx >= 0 and current_target_idx < len(projected_points):
                                u, v = projected_points[current_target_idx][0], projected_points[current_target_idx][1]
                                cv2.circle(image_with_radar, (u, v), 10, (255, 255, 255), -1)
                                cv2.putText(image_with_radar, f"id={track_id[current_target_idx][-1]:.0f}",
                                            (u + 5, v), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 225, 100), 2)
                                target_info = track_id[current_target_idx]

                        except Exception as e:
                            print(f"Target detection error: {e}")

                    # 车道检测
                    lane_center = 510
                    try:
                        lane_windows, lane_image, detected_windows = self.lane_detector.lane_detect(image_with_radar)
                        valid_row = None
                        for row in lane_windows:
                            if len(row) == 6 and row[2] == 1 and row[5] == 1:
                                valid_row = row
                                break
                        if valid_row is not None:
                            lane_center = (valid_row[0] + valid_row[3]) / 2
                    except Exception as e:
                        print(f"Lane detection error: {e}")

                    # 在OpenCV图像上添加ACC状态信息
                    y_offset = 10
                    cv2.putText(image_with_radar, f"ACC: {acc_status['state_description']}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_offset += 25

                    cv2.putText(image_with_radar, f"Active: {'YES' if self.acc_control_active else 'NO'}",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0) if self.acc_control_active else (255, 255, 255), 2)
                    y_offset += 25

                    if acc_params.get('cruise_mode_active', False):
                        cv2.putText(image_with_radar, "CRUISE MODE", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        y_offset += 25

                    # 显示OpenCV窗口（如果启用）
                    if self.show_opencv:
                        cv2.imshow("Radar and Lane Detection", image_with_radar)
                        cv2.waitKey(1)

                # === 车辆控制 ===
                # 检查是否应该进行ACC控制 - 基于控制模式
                decision_output = self.acc_decision.get_decision_output(ego_speed, vehicle_distance)

                # === 调试输出：检查每个判断条件 ===
                print(f"\n=== ACC控制判断调试 ===")
                print(f"1. acc_control_active: {self.acc_control_active}")
                print(f"2. decision_output['control_enabled']: {decision_output['control_enabled']}")
                print(f"3. is_in_active_control_mode(): {self.acc_decision.is_in_active_control_mode()}")
                print(f"4. current_state: {self.acc_decision.current_state.value}")
                print(f"5. current_control_mode: {decision_output.get('current_control_mode', 'None')}")
                print(f"6. manual_control_active: {self.manual_control_active}")
                print(
                    f"7. 综合判断结果: {self.acc_control_active and decision_output['control_enabled'] and self.acc_decision.is_in_active_control_mode()}")
                print("=== 调试结束 ===\n")

                # === 重要：只有在非手动控制模式下才执行ACC控制 ===
                if (self.acc_control_active and
                        decision_output['control_enabled'] and
                        self.acc_decision.is_in_active_control_mode() and
                        not self.manual_control_active):  # 新增条件：确保不在手动控制模式
                    # ACC控制模式 - 只有在主动控制模式下才执行
                    print("✅ 进入ACC控制执行分支")
                    try:
                        lane_offset = (lane_center - 510) / 150

                        # === 调试target_info ===
                        print(f"🎯 target_info调试:")
                        print(f"   target_info类型: {type(target_info)}")
                        print(f"   target_info值: {target_info}")
                        if target_info is not None:
                            print(
                                f"   target_info长度: {len(target_info) if hasattr(target_info, '__len__') else 'No length'}")
                        print(f"   force_cruise_mode: {decision_output['force_cruise_mode']}")

                        # 根据定速巡航模式决定是否使用目标信息
                        if decision_output['force_cruise_mode']:
                            # 定速巡航模式：忽略前车
                            print("🚗 执行定速巡航控制 (忽略前车)")
                            control = acc_controller.cruise_control(lane_offset, None)
                        else:
                            # 正常ACC模式：使用前车信息
                            print(f"🚗 执行自适应ACC控制 (使用前车信息: {target_info is not None})")
                            control = acc_controller.cruise_control(lane_offset, target_info)

                        if control.brake < 0.01:
                            control.brake = 0
                        self.ego_vehicle.apply_control(control)

                        # 显示当前控制模式
                        current_mode = decision_output.get('current_control_mode', 'Unknown')
                        print(f"🎮 ACC executing control mode: {current_mode}")

                    except Exception as e:
                        print(f"❌ ACC control error详细信息:")
                        print(f"   错误类型: {type(e).__name__}")
                        print(f"   错误消息: {str(e)}")
                        print(f"   target_info: {target_info}")
                        print(f"   lane_offset: {(lane_center - 510) / 150}")
                        import traceback
                        print(f"   完整错误堆栈:")
                        traceback.print_exc()

                else:
                    # 不满足控制条件时的提示
                    print("❌ 未进入ACC控制分支")
                    if self.acc_control_active:
                        acc_state = self.acc_decision.current_state.value
                        control_mode = decision_output.get('current_control_mode', 'None')
                        is_active_mode = self.acc_decision.is_in_active_control_mode()
                        print(f"💡 原因分析: State={acc_state}, ControlMode={control_mode}, "
                              f"ActiveMode={is_active_mode}, ControlEnabled={decision_output['control_enabled']}, "
                              f"ManualActive={self.manual_control_active}")
                    else:
                        if self.acc_decision.current_state == ACCState.ADAPTIVE_HISTORY_STANDBY:
                            print("💡 提示: ACC处于待命状态，按1键可重新激活")
                        else:
                            print("💡 原因: ACC未激活 (acc_control_active=False)")

                # === 数据记录 ===
                current_time = time.time() - self.start_time
                desired_distance, control_mode = self.calculate_desired_following_distance(ego_speed)

                self.csv_writer.writerow([
                    current_time,
                    ego_speed,
                    target_speed,
                    vehicle_distance,
                    desired_distance,
                    control_mode,
                    self.get_lane_offset(),
                    acc_status['state_description'],
                    self.acc_control_active,
                    acc_params['V3_kmh'],
                    acc_params['G1_m'],
                    acc_params['G2_s'],
                    acc_params.get('cruise_mode_active', False),
                    self.throttle,
                    self.brake,
                    self.steer
                ])
                self.csv_file.flush()

                # === Pygame渲染 ===
                system_info = self.get_system_info()
                self.display_manager.render_display(system_info)

                frame_count += 1

        except KeyboardInterrupt:
            print("\nStopped by user.")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up...")
            cv2.destroyAllWindows()
            self.csv_file.close()
            self.display_manager.destroy()
            self.destroy()

    def destroy(self):
        # 停止传感器
        self.radar.stop()
        self.camera.stop()
        self.lidar.stop()

        # 销毁传感器
        self.radar.destroy()
        self.camera.destroy()
        self.lidar.destroy()

        # 销毁车辆
        for vehicle in self.vehicles:
            vehicle.destroy()
        self.ego_vehicle.destroy()

        # 恢复异步模式
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        print(f"Destroyed {len(self.vehicles)} vehicles, ego vehicle, sensors, and restored settings.")


def main():
    acc_actor = acc()

    # 创建传感器监听线程
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
        # 停止传感器监听
        acc_actor.radar.stop()
        acc_actor.camera.stop()
        acc_actor.lidar.stop()

        # 等待线程结束
        thread_1.join()
        thread_2.join()
        thread_3.join()
        print("All threads terminated.")


if __name__ == '__main__':
    main()