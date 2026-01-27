#在lks_tcp_s9.py基础上修改，改为simulink11版本
import carla
import random
import time
import math
import numpy as np
import cv2
import pygame
import threading
import open3d as o3d
import csv
import queue
import ipdb
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import socket
import sys
import struct


class lks:
    def __init__(self, case_num):
        self.case_num = case_num
        self.max_follow_distance = 50
        self.radar_2_world = []
        self.world_2_camera = []
        self.csv_file = None
        self.csv_writer = None
        self.count = 0
        self.counts=[]
        self.errors_lks = []
        self.nearest_lane_points_history = []
        self.lks_steer = 0.0
        self.system_state = 1
        self.system_state_show = ["LKS In Control", "LKS Ready", "LKS Off"]
        self.in_control_steps = []
        self.current_speed = 0
        self.current_speed_list = []
        self.curvature_list = []
        self.ey_list = []
        self.lks_control_errors = []
        self.timestepT = 0
        self.timestepT1 = 0

        self.ey_t1 = 0
        self.edelta_t1 = 0
        self.theta_t1 = 0

        self.init_carla()
        self.init_csv()
        self.init_tcp()
        self.video_save_path = r'/home/zc/Code/CBDES/acc/data/uphill.mp4'
        self.video_writer = cv2.VideoWriter(self.video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
        droneConfig = {"fx": 3872.0, "fy": 3872.0, "px": 1968.0,"py": 484.0,
                # rotation matrix R (in deg)
                "yaw":    0.0, "pitch":  90.0, "roll":   90.0,
                # vehicle coords of camera origin
                "XCam": 44,"YCam": 0,"ZCam": 70}
        self.outputRes = (int(2 * droneConfig["py"]), int(2 * droneConfig["px"]))
        self.M = np.array([[-1.79115981e-17, -1.09191138e-01,  9.76832146e+01],
                            [6.98593773e-02,  5.82161478e-01, -3.81432200e+02],
                            [1.40841983e-19,  1.20281297e-03, -6.49519004e-01]])

    def init_tcp(self):
        HOST = "127.0.0.1"       # Change to remote PC IP if needed
        SEND_PORT = 25026
        RCV_PORT = 25028
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允许端口重用
        self.tcp_server.bind((HOST, SEND_PORT))
        self.tcp_server.listen()
        print(f"[SENDER] Waiting for receiver...")
        self.send_conn = None #, self.addr = self.server.accept()

        self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允许端口重用
        self.tcp_client.bind((HOST, RCV_PORT))
        self.tcp_client.listen()
        print(f"[RECVER] Waiting for SENDET...")
        self.rcv_conn = None #, self.addr = self.server.accept()


    def init_carla(self):

        # 先初始化 pygame（必须在 load_world 之前，避免 RTX 5080 Vulkan 冲突）
        pygame.init()
        WIDTH, HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        # Connect to the CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        # 智能加载地图：只在需要时才加载，避免不必要的重载导致GPU崩溃
        current_map_name = self.world.get_map().name
        if 'acc_30km_new' not in current_map_name:
            print(f"当前地图: {current_map_name}，加载目标地图: acc_30km_new")
            self.world = self.client.load_world(
                'acc_30km_new',
                carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles
            )
        else:
            print(f"地图已是 acc_30km_new ({current_map_name})，跳过加载")

        # Get all actors in the world
        actors = self.world.get_actors()

        # Filter for traffic lights
        traffic_lights = [actor for actor in actors if actor.type_id.startswith('traffic.traffic_light')]
        self.look_dis = 0
        # Set all traffic lights to green
        for traffic_light in traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.set_green_time(1000.0)  # Set green time to a large value (in seconds)
            traffic_light.set_red_time(0.0)  # Set red time to 0
            traffic_light.set_yellow_time(0.0)  # Set yellow time to 0

        # Get the blueprint library and map
        blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        self.manual_mode = True
        self.manual_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        # pygame 已在 load_world 之前初始化
        

        # Select vehicle blueprints
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        # ipdb.set_trace()
        # self.print_bp(vehicle_bp)
        ego_vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]

        # Get all valid spawn points on the road
        spawn_points = self.map.get_spawn_points()[0]

        
        self.whethe_end = True

        if self.case_num == 1:
            self.error_save_file_name = "results/_follow_up_1"
            followed_car_point = carla.Location(x=1964.847778, y=-4824.012207, z=0.5)
            ego_point = carla.Location(x=1963.010132, y=-4841.013672, z=0.5)
            self.end_point = carla.Location(x=2275.941650, y=-1905.315552, z=0.5)
        elif self.case_num == 2:
            self.error_save_file_name = "results/_follow_up_2"
            followed_car_point = carla.Location(x=2275.941650, y=-1905.315552, z=0.5)
            ego_point = carla.Location(x=2273.844727, y=-1933.583374, z=0.5)
            self.end_point = carla.Location(x=2511.432617, y=1281.097046, z=0.5)
        elif self.case_num == 3:
            self.error_save_file_name = "results/_follow_up_3"
            followed_car_point = carla.Location(x=2511.432617, y=1281.097046, z=0.5)
            ego_point = carla.Location(x=2508.690918, y=1242.117798, z=0.5)
            self.end_point = carla.Location(x=2695.179932, y=3894.786377, z=0.5)
        elif self.case_num == 4:
            self.error_save_file_name = "results/_follow_right_1"
            followed_car_point = carla.Location(x=1654.013672, y=6322.584473, z=0.5)
            ego_point = carla.Location(x=1665.092773, y=6334.560059, z=0.5)
            self.end_point = carla.Location(x=743.696716, y=5596.785645, z=0.5)
        elif self.case_num == 5:
            self.error_save_file_name = "results/_follow_right_2"
            followed_car_point = carla.Location(x=743.696716, y=5596.785645, z=0.5)
            ego_point = carla.Location(x=752.821411, y=5607.135742, z=0.5)
            self.end_point = carla.Location(x=-971.171997, y=3982.007080, z=0.5)
        elif self.case_num == 6:
            self.error_save_file_name = "results/_follow_right_3"
            followed_car_point = carla.Location(x=-971.171997, y=3982.007080, z=0.5)
            ego_point = carla.Location(x=-963.105347, y=4000.134277, z=0.5)
            self.end_point = carla.Location(x=-2039.404297, y=2537.271240, z=0.5)
        elif self.case_num == 7:
            self.error_save_file_name = "results/_follow_right_4"
            followed_car_point = carla.Location(x=-2039.404297, y=2537.271240, z=0.5)
            ego_point = carla.Location(x=-2034.824341, y=2557.189209, z=0.5)
            self.end_point = carla.Location(x=-2526.125977, y=1440.714844, z=0.5)
        elif self.case_num == 8:
            self.error_save_file_name = "results/_follow_right_5"
            followed_car_point = carla.Location(x=-2842.643799, y=648.184204, z=0.5)
            ego_point = carla.Location(x=-2841.705322, y=692.136292, z=0.5)
            self.end_point = carla.Location(x=-3027.574219, y=-1116.691284, z=0.5)
        elif self.case_num == 9:
            self.error_save_file_name = "results/_follow_right_6"
            followed_car_point = carla.Location(x=-3027.574219, y=-1116.691284, z=0.5)
            ego_point = carla.Location(x=-3032.802002, y=-1103.960449, z=0.5)
            self.end_point = carla.Location(x=-2846.395264, y=-2943.590820, z=0.5)
        elif self.case_num == 10:
            self.error_save_file_name = "results/_follow_right_7"
            followed_car_point = carla.Location(x=-2846.395264, y=-2943.590820, z=0.5)
            ego_point = carla.Location(x=-2851.281738, y=-2936.522461, z=0.5)
            self.end_point = carla.Location(x=-2171.305176, y=-4618.055664, z=0.5)
        elif self.case_num == 11:
            self.error_save_file_name = "results/_follow_right_8"
            followed_car_point = carla.Location(x=-2171.305176, y=-4618.055664, z=0.5)
            ego_point = carla.Location(x=-2190.767822, y=-4601.566895, z=0.5)
            self.end_point = carla.Location(x=-911.749756, y=-6107.719238, z=0.5)
        elif self.case_num == 12:
            self.error_save_file_name = "results/_follow_right_9"
            followed_car_point = carla.Location(x=-911.749756, y=-6107.719238, z=0.5)
            ego_point = carla.Location(x=-937.299377, y=-6110.980957, z=0.5)
            self.end_point = carla.Location(x=492.051514, y=-6452.419922, z=0.5)
        print(f"Test Case {self.case_num} selected.")

        waypoint = self.map.get_waypoint(followed_car_point, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            raise RuntimeError("Failed to find a valid waypoint near the specified location")

        # import ipdb;ipdb.set_trace()
        # 设置前车生成点（基于 waypoint）
        spawn_point = waypoint.transform
        lanes  = []
        current_waypoint = waypoint

        lanes.append(current_waypoint.get_left_lane())
        # lanes.append(current_waypoint.get_right_lane())
        lanes.append(current_waypoint)
        lanes.append(current_waypoint.get_right_lane())
        self.ego_gt_lane_points = []
        self.vehicles = []
        locations = []

        #add three cars
        for lane in lanes:
            if lane is not None:
                transform = lane.transform
                transform.location.z += 0.1
                locations.append(transform.location)
        
        # Spawn the ego vehicle
        ego_spawn_point = spawn_point
        #reset location
        temp_transform = carla.Transform()
        temp_transform.location = ego_point
        temp_transform.rotation = ego_spawn_point.rotation
        relative_point = carla.Location(x=0.0, y=1.0, z=0.0)
        if self.case_num == 12:
            relative_point = carla.Location(x=0.0, y=0.76, z=0.0)
        b_point_world = temp_transform.transform(relative_point)
        ego_spawn_point.location = b_point_world

        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, ego_spawn_point)

        # reset location



        # Enable autopilot for all vehicles
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_global_distance_to_leading_vehicle(1.0)
        self.tm.set_synchronous_mode(True)
        self.tm_port = self.tm.get_port()
        
        for vehicle in self.vehicles:
            vehicle.set_autopilot(True, self.tm_port)
            self.tm.auto_lane_change(vehicle, False)
        if not self.manual_mode:
            self.ego_vehicle.set_autopilot(True, self.tm_port)
            self.tm.auto_lane_change(self.ego_vehicle, False)
        # tm.auto_lane_change(self.ego_vehicle, False)

        # Get the vehicle size 
        bounding_box = self.ego_vehicle.bounding_box

        # Extract dimensions (full length, width, height in meters)
        self.ego_vehicle_length = 2 * bounding_box.extent.x  # Longitudinal (front-to-back)
        self.ego_vehicle_width = 2 * bounding_box.extent.y   # Lateral (side-to-side)
        self.ego_vehicle_height = 2 * bounding_box.extent.z  # Vertical

        # Set up radar sensor on the ego vehicle
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('range', '100.0')
        radar_bp.set_attribute('horizontal_fov', '120.0')
        radar_bp.set_attribute('vertical_fov', '60.0')
        radar_bp.set_attribute('points_per_second', '20000')
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)
        self.radar_2_world = self.radar.get_transform().get_matrix()
        self.world_2_radar = self.radar.get_transform().get_inverse_matrix()

        # Set up camera sensor on the ego vehicle
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '120')
        camera_bp.set_attribute('focal_distance', '100000.0')  # 焦距
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.5))
        self.image_width = camera_bp.get_attribute('image_size_x').as_int()
        self.image_height = camera_bp.get_attribute('image_size_y').as_int()
        self.fov = camera_bp.get_attribute('fov').as_float()
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        self.world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())
        self.camera_transform = self.camera.get_transform()
        self.front_video_writer = cv2.VideoWriter(
            '/home/zc/Code/CBDES/acc/images/videos/front_camera.avi',
            cv2.VideoWriter_fourcc(*'mp4v'),
            20.0,
            (self.image_width, self.image_height)
        )

        bev_camera_bp = blueprint_library.find('sensor.camera.rgb')
        bev_camera_bp.set_attribute('image_size_x', '1920')
        bev_camera_bp.set_attribute('image_size_y', '1080')
        bev_camera_bp.set_attribute('fov', '120')
        bev_camera_bp.set_attribute('focal_distance', '100.0')  # 焦距
        bev_camera_transform = carla.Transform(carla.Location(x=15, z=30), carla.Rotation(pitch=270))
        self.bev_camera = self.world.spawn_actor(bev_camera_bp, bev_camera_transform, attach_to=self.ego_vehicle)

        pygame_camera_bp = blueprint_library.find('sensor.camera.rgb')
        pygame_camera_bp.set_attribute('image_size_x', '800')
        pygame_camera_bp.set_attribute('image_size_y', '600')
        pygame_camera_bp.set_attribute('fov', '120')
        pygame_camera_bp.set_attribute('focal_distance', '100.0')  # 焦距
        pygame_camera_transform = carla.Transform(carla.Location(x=-5.0, z=6.0), carla.Rotation(pitch=-25.0))
        self.pygame_camera = self.world.spawn_actor(pygame_camera_bp, pygame_camera_transform, attach_to=self.ego_vehicle)
        self.image_queue = queue.Queue()

        self.ego_vehicle_wheelbase, self.radar2rearaxle = self.set_vehicle_parameters()
        self.off_state = 0  # LKS On

        #imu
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0.0, z=1.5))
        self.imu_sensor = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.ego_vehicle)
        self.imu_sensor.listen(lambda data: self._imu_callback(data))

    def _imu_callback(self, imu_data):
        """IMU传感器的回调函数，在独立的线程中被调用。"""
        # 从IMU数据中直接读取车辆坐标系下的线性加速度 (m/s²)
        accel = imu_data.accelerometer
        # 车辆坐标系下，横向加速度即为Y轴分量（根据左手坐标系，左为正）
        self.lateral_accel = -accel.y
        # 注意：如果需要在世界坐标系下使用，需进行坐标转换（见后续说明）

    def get_current_lateral_acceleration(self):
        """获取当前计算出的横向加速度值。"""
        return self.lateral_accel

    def set_vehicle_parameters(self):
        radar_location = [self.radar.get_transform().location.x, 
                          self.radar.get_transform().location.y]
        
        physics_control = self.ego_vehicle.get_physics_control()
        front_left_wheel_location = [physics_control.wheels[0].position.x,
                                    physics_control.wheels[0].position.y]
        front_right_wheel_location = [physics_control.wheels[1].position.x,
                                        physics_control.wheels[1].position.y]
        front_axle_rel_location = [(front_left_wheel_location[0] + front_right_wheel_location[0]) / 200,
                                      (front_left_wheel_location[1] + front_right_wheel_location[1]) / 200] 

        rear_left_wheel_location = [physics_control.wheels[2].position.x, 
                                    physics_control.wheels[2].position.y]
        rear_right_wheel_location = [physics_control.wheels[3].position.x,  
                                     physics_control.wheels[3].position.y]
        rear_axle_rel_location = [(rear_left_wheel_location[0] + rear_right_wheel_location[0]) / 200,
                                   (rear_left_wheel_location[1] + rear_right_wheel_location[1]) / 200]
        wheelbase = math.sqrt((front_axle_rel_location[0]-rear_axle_rel_location[0])**2 +
                               (front_axle_rel_location[1]-rear_axle_rel_location[1])**2)
        dis_radar_to_rear_axle = math.sqrt((radar_location[0]-rear_axle_rel_location[0])**2 +
                                            (radar_location[1]-rear_axle_rel_location[1])**2)
        return wheelbase, dis_radar_to_rear_axle




    def get_vehicle_speed(self, vehicle):
        velocity = vehicle.get_velocity()  # carla.Vector3D
        speed_m_s = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed_m_s * 3.6  # Convert m/s to km/h
        return speed_m_s

   
    def draw_lane_error(self, new_value):
        current_time_index = len(self.lane_error_x)
        self.lane_error_x.append(current_time_index)
        self.lane_error_y.append(new_value)
        if len(self.lane_error_x) > self.lane_max_point:
            self.lane_error_x.pop(0)
            self.lane_error_y.pop(0)
            self.lane_error_x[:] = range(len(self.lane_error_x))
        self.line.set_data(self.lane_error_x, self.lane_error_y)
        self.ax.relim()
        self.autoscale_view()

        return self.line


    def pygame_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        img = np.array(array)
        self.image_queue.put(array, block=False)  # Non-blocking


    def get_gt_lane_points(self, vehicle, range_meters=50.0):
        """
        在 CARLA 中绘制自车前方指定范围内的车道线。

        参数：
            world: carla.World 对象
            vehicle: carla.Vehicle 对象（自车）
            range_meters: 前方范围（米），默认 50 米
        """
        # 获取自车位置和朝向
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)  # 转换为弧度
        world2radar = self.radar.get_transform().get_inverse_matrix()
        radar2world = self.radar.get_transform().get_matrix()
        # 获取自车所在的 waypoint
        waypoint = self.map.get_waypoint(vehicle_location, project_to_road=True)
        

        # 初始化用于存储车道线的点
        lane_points = []
        right_lane_points = []
        left_lane_points = []

        # 向前搜索 50 米内的 waypoints
        distance = 0.0
        current_waypoint = waypoint
        step = 0.5  # 步长，单位为米
        while distance < range_meters:
            # 获取下一个 waypoint（步长约为 1 米）
            next_waypoints = current_waypoint.next(step)
            if not next_waypoints:
                break
            current_waypoint = next_waypoints[0]
            distance += step

            # 获取车道线的标记类型
            left_marking = current_waypoint.left_lane_marking
            right_marking = current_waypoint.right_lane_marking

            # 获取车道线的几何位置（左右车道线）
            lane_width = current_waypoint.lane_width
            right_lane_point = current_waypoint.transform.location + carla.Location(
                x=-lane_width / 2 * math.sin(vehicle_yaw),
                y=lane_width / 2 * math.cos(vehicle_yaw)
            )
            left_lane_point = current_waypoint.transform.location + carla.Location(
                x=lane_width / 2 * math.sin(vehicle_yaw),
                y=-lane_width / 2 * math.cos(vehicle_yaw)
            )

            # 根据车道线类型决定颜色和绘制
            def get_marking_color(marking):
                if marking.type == carla.LaneMarkingType.Solid:
                    return carla.Color(255, 255, 255)  # 白色实线
                elif marking.type == carla.LaneMarkingType.Broken:
                    return carla.Color(255, 255, 0)   # 黄色虚线
                else:
                    return carla.Color(128, 128, 128)  # 灰色（其他类型）

            # 添加车道线点
            right_lane_points.append(right_lane_point)
            left_lane_points.append(left_lane_point)
            lane_points.append([left_lane_point.x, left_lane_point.y, left_lane_point.z, 1,\
                                right_lane_point.x, right_lane_point.y, right_lane_point.z, 1])
        vis_lane_points = np.array(lane_points.copy())
        front_lane_points = []
        for i in range(len(lane_points)):
            lane_points[i][:3] = np.dot(world2radar, np.array(lane_points[i][:4]))[:3]
            lane_points[i][4:7] = np.dot(world2radar, np.array(lane_points[i][4:]))[:3]
            # lane_points[i][0] += self.radar2rearaxle
            # lane_points[i][4] += self.radar2rearaxle
            lane_points[i][0] = lane_points[i][0] + 2 - self.ego_vehicle.bounding_box.extent.x
            lane_points[i][4] = lane_points[i][4] + 2 - self.ego_vehicle.bounding_box.extent.x
            if lane_points[i][0] > -1 and lane_points[i][4] > -1:
                front_lane_points.append(lane_points[i])
            lane_points[i][1] = -lane_points[i][1]
            lane_points[i][5] = -lane_points[i][5]
            front_lane_points = front_lane_points[:60]
        return front_lane_points

    def bev_camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        bev_camera_image = array
        self.bev_img = np.array(bev_camera_image).copy()


    # Function to handle manual control
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
            throttle_increment = 0.5
            steer_increment = 0.1
            
            # Throttle control
            if keys[pygame.K_w]:
                self.manual_control.throttle = min(self.manual_control.throttle + throttle_increment, 1.0)
            if keys[pygame.K_s]:
                self.manual_control.throttle = max(self.manual_control.throttle - throttle_increment, 0.0)
            
            # Steering control
            if keys[pygame.K_a]:
                self.manual_control.steer = max(self.manual_control.steer - steer_increment, -1.0)
            if keys[pygame.K_d]:
                self.manual_control.steer = min(self.manual_control.steer + steer_increment, 1.0)
            
            # Brake control
            if keys[pygame.K_SPACE]:
                self.manual_control.brake = 1.0
                self.manual_control.throttle = 0.0
            else:
                self.manual_control.brake = 0.0
            
        
    def init_csv(self):
        self.csv_file = open('loc.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # self.csv_writer.writerow(['Time(s)', 'gt_y(m)',
        #                           'target_y(m)', "error"])
        self.csv_writer.writerow(['Time(s)',  'gt_x(m)', 'gt_y(m)',
                                  'target_x(m)', 'target_y(m)', "error"])
        print("CSV file 'loc.csv' created and header written.")


    def get_brake(self, control):
        break_status = control.brake
        if break_status > 0:
            is_braking = 1
            bool_is_braking = True
        else:
            is_braking = 0
            bool_is_braking = False
        # cv2.putText(self.bev_img, f"Brake Status: {bool_is_braking}", (20, 400), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return is_braking

    def get_turnSignals(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.ego_vehicle.set_light_state(carla.VehicleLightState.LeftBlinker)
        elif keys[pygame.K_RIGHT]:
            self.ego_vehicle.set_light_state(carla.VehicleLightState.RightBlinker)
        elif keys[pygame.K_DOWN]:
            self.ego_vehicle.set_light_state(carla.VehicleLightState.NONE)


        light_state = self.ego_vehicle.get_light_state()
        self.blinker = ""
        # 检查左转向灯是否开启
        if light_state & carla.VehicleLightState.LeftBlinker:
            is_lane_change_signal = 1
            bool_lane_change_signal = True
            self.blinker = "LeftBlinker"
            print("左转向灯正在闪烁")
            
        # 检查右转向灯是否开启
        elif light_state & carla.VehicleLightState.RightBlinker:
            is_lane_change_signal = 1
            bool_lane_change_signal = True
            self.blinker = "RightBlinker"
            print("右转向灯正在闪烁")
        else:
            is_lane_change_signal = 0
            bool_lane_change_signal = False
        
        return is_lane_change_signal

    def LKS_sendto_simulink(self):
        control = self.ego_vehicle.get_control()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.off_state = 1  # LKS Off
        off_state = self.off_state

        #brake
        brake_state = self.get_brake(control)
        # print("brake state: ", brake_state)
        
        #turnSignals
        turnSignals_state = self.get_turnSignals()
        # print("turnSignals state: ", turnSignals_state)

        #activeSteer
        active_steer = control.steer - self.lks_steer
        # print("active steer: ", active_steer)

        #x
        vehicle_x = 0

        #y
        vehicle_y = 0

        #cx cy boolLD
        valid_data = []
        boolLD = []
        for i in range(len(self.ego_gt_lane_points)):
            if self.ego_gt_lane_points[i][3] ==1.0:
                valid_data.append(self.ego_gt_lane_points[i])
            boolLD.append(int(self.ego_gt_lane_points[i][3]))
        # if len(self.ego_gt_lane_points)-len(valid_data)>25:
        #     print("lane detection failed")
        lane_center_x = []
        lane_center_y = []
        for copule_point in self.ego_gt_lane_points:
            lane_center_x.append((copule_point[0] + copule_point[4]) / 2)
            lane_center_y.append((copule_point[1] + copule_point[5]) / 2)

        # print("boolLD: ", boolLD)


        #v
        velocity = self.ego_vehicle.get_velocity()
        vehicle_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        vehicle_speed_kmh = vehicle_speed * 3.6
        self.current_speed = vehicle_speed_kmh
        self.current_speed_list.append(self.current_speed)

        ey_t1 = self.ey_t1
        edelta_t1 = self.edelta_t1
        theta_t1 = self.theta_t1

        self.timestepT = self.world.get_snapshot().frame
        timestepT = self.timestepT
        timestepT1 = self.timestepT1
        
        # print("vehicle speed: ", vehicle_speed_kmh," km/h")

        #communicate with simulink
        if self.send_conn == None:
            print("preparing to send")
            self.send_conn, addr = self.tcp_server.accept()
        tcp_boolLD = np.array([boolLD], dtype=np.float64).tobytes(order='F')
        tcp_brake = np.array([brake_state], dtype=np.float64).tobytes(order='F')
        tcp_turnSignals = np.array([turnSignals_state], dtype=np.float64).tobytes(order='F')
        tcp_activeSteer = np.array([active_steer], dtype=np.float64).tobytes(order='F')
        tcp_off_state = np.array([off_state], dtype=np.float64).tobytes(order='F')

        tcp_speed = np.array([vehicle_speed], dtype=np.float64).tobytes(order='F')
        tcp_cx = np.array(lane_center_x, dtype=np.float64).tobytes(order='F')
        tcp_cy = np.array(lane_center_y, dtype=np.float64).tobytes(order='F')
        if self.curvature < 0.2:
            is_curvature = 0
        else:
            is_curvature = 1
        tcp_curvature = np.array([is_curvature], dtype=np.float64).tobytes(order='F')
        tcp_ey_t1 = np.array([ey_t1], dtype=np.float64).tobytes(order='F')
        tcp_edelta_t1 = np.array([edelta_t1], dtype=np.float64).tobytes(order='F')
        tcp_theta_t1 = np.array([theta_t1], dtype=np.float64).tobytes(order='F')
        tcp_timestepT = np.array([timestepT], dtype=np.float64).tobytes(order='F')
        tcp_timestepT1 = np.array([timestepT1], dtype=np.float64).tobytes(order='F')

        message = tcp_boolLD + tcp_brake + tcp_turnSignals + tcp_activeSteer + tcp_off_state + \
                  tcp_speed + tcp_cx + tcp_cy + tcp_curvature + tcp_ey_t1 + tcp_edelta_t1 + \
                  tcp_theta_t1 + tcp_timestepT + tcp_timestepT1
        # ipdb.set_trace()
        # print("Sending data to Simulink... ", len(message), " bytes")
        try:
            self.send_conn.sendall(message)
        except BrokenPipeError:
            pass
        except socket.error as e:
            print(f"网络错误: {e}", file=sys.stderr)
        except KeyboardInterrupt:
            print("用户中断。")
        # time.sleep(0.1)


        # ipdb.set_trace()

        

    def LKS_receivefrom_simulink(self):
        if self.rcv_conn is None:
            try:
                self.rcv_conn, rcv_addr = self.tcp_client.accept()
                print(f"TCP接收连接已建立: {rcv_addr}")
            except socket.error as e:
                print(f"接受连接失败: {e}")
        
        # 配置参数：2个float32，每个4字节
        DOUBLE_SIZE_BYTES = 11  # float32 / single precision
        EXPECTED_FLOATS = 8   # 接收2个float数
        TOTAL_EXPECTED_BYTES = EXPECTED_FLOATS * DOUBLE_SIZE_BYTES  # 8字节
        BUFFER_SIZE = 1024    # 接收缓冲区大小
        
        try:
            # --- 优化版数据接收循环 ---
            # TCP是流式协议，需要循环确保收到完整数据包
            data_buffer = b''
            bytes_received = 0
            
            # 设置接收超时（可选，避免无限等待）
            # self.rcv_conn.settimeout(2.0)
            
            while bytes_received < TOTAL_EXPECTED_BYTES:
                # 计算还需要多少字节
                bytes_to_read = TOTAL_EXPECTED_BYTES - bytes_received
                
                # 接收数据（最多接收剩余需要的字节数）
                chunk = self.rcv_conn.recv(min(bytes_to_read, BUFFER_SIZE))
                
                # 检查连接是否已关闭
                if len(chunk) == 0:
                    print("连接被对端关闭")
                    self.rcv_conn.close()
                    self.rcv_conn = None
                
                data_buffer += chunk
                bytes_received += len(chunk)
            
            # 成功接收16字节，解析为2个dobule
            if len(data_buffer) == TOTAL_EXPECTED_BYTES:
                # 使用struct解析字节数据
                # 'ff' 格式字符串表示两个float32（4字节）
                systemState, offState, vState, LDState, brakeState, LCState, ey, theta, raw_theta, max_theta, edelta_t1 = struct.unpack('ddddddddddd', data_buffer)
                
            
                
                # 返回两个float值
            else:
                print(f"数据包大小不匹配: 期望 {TOTAL_EXPECTED_BYTES} 字节, 实际 {len(data_buffer)} 字节")
                
        except socket.timeout:
            print("接收数据超时")
            
        except socket.error as e:
            print(f"Socket错误: {e}")
            # 发生错误时关闭连接
            if self.rcv_conn:
                self.rcv_conn.close()
                self.rcv_conn = None
            

        if offState==1.0:
            cv2.putText(self.bev_img, f"LKS State: Off", (20, 600), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            return


        cv2.putText(self.bev_img, f"Speed: {self.current_speed:.2f} km/h", (20, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(self.bev_img, f"Low Speed: {vState}", (20, 200), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(self.bev_img, f"Lane Detect Failed: {LDState}", (20, 300), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(self.bev_img, f"Brake Applied: {bool(brakeState)}", (20, 400), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(self.bev_img, f"LC State: {bool(LCState)}", (20, 500), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.putText(self.bev_img, self.blinker, (20, 900), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        if systemState==0.0:
            if self.system_state == 1:
                self.in_control_steps.append(self.count)
            cv2.putText(self.bev_img, f"LKS State: In Control", (20, 600), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            steer = -theta
            self.manual_control.steer = 1*min(max(steer, -1.0), 1.0)
            self.lks_steer = self.manual_control.steer
            self.system_state = 0
            self.ey_t1 = ey
            self.edelta_t1 = edelta_t1
            self.theta_t1 = theta
            self.timestepT1 = self.world.get_snapshot().frame
            # ey = eng.workspace['ey'][-1][-1][-1]
            
        elif systemState > 0.0:
            cv2.putText(self.bev_img, f"LKS State: Ready", (20, 600), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            self.system_state = 1
            ey = 0
            self.ey_t1 = 0
            self.edelta_t1 = 0
            self.theta_t1 = -self.ego_vehicle.get_control().steer
            self.timestepT1 = self.world.get_snapshot().frame
        
        self.ey_list.append(ey)


    def save_results(self):
        data = np.column_stack((self.counts, self.errors_lks, self.nearest_lane_points_history, self.current_speed_list))
        average_error = np.mean(np.abs(self.lks_control_errors))
        average_error = round(average_error,4)
        print(f"Average absolute lane error: {average_error:.4f} m")
        self.error_save_file_name = self.error_save_file_name + '_e' + str(average_error)
        txt_name = self.error_save_file_name + '.txt'
        np.savetxt(txt_name, data, fmt="%.6f", header="Count\tLane Error (m)\tNearest Point\tSpeed")
        # np.savetxt(txt_name, data, fmt="%.6f", header="Count\tLane Error (m)")
        print(f"Lane errors saved to {txt_name}")
        

    def radius_three_points(self, P1, P2, P3):
        x1, y1 = P1
        x2, y2 = P2
        x3, y3 = P3
        a = x2 - x1; b = y2 - y1
        c = x3 - x1; d = y3 - y1
        e = a*(x1+x2) + b*(y1+y2)
        f = c*(x1+x3) + d*(y1+y3)
        g = 2*(a*(y3-y2) - b*(x3-x2))
        if abs(g) < 1e-6: return float('inf')
        cx = (d*e - b*f) / g
        cy = (a*f - c*e) / g
        r = ((x1-cx)**2 + (y1-cy)**2)**0.5
        return r


    def get_turning_radius_three_points(self, vehicle, dist1=5.0, dist2=15.0, dist3=25.0):
        """
        Use three points ahead of the vehicle at fixed distances.
        Recommended distances: 8m, 16m, 24m → works perfectly for R ≥ 50m
        """
        loc = vehicle.get_location()
        wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)

        # Get three waypoints ahead
        wp1 = wp.next(dist1)[0] #if wp.next(dist1) else wp
        wp2 = wp.next(dist2)[0] #if wp.next(dist2) else wp
        wp3 = wp.next(dist3)[0] #if wp.next(dist3) else wp

        # Transform to vehicle local frame (x=forward, y=left)
        ego_tf = vehicle.get_transform()
        ego_loc = ego_tf.location
        yaw_rad = np.deg2rad(ego_tf.rotation.yaw)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)

        def to_local(p):
            dx = p.x - ego_loc.x
            dy = p.y - ego_loc.y
            x_local = dx * cos_y + dy * sin_y
            y_local = -dx * sin_y + dy * cos_y   # left is positive
            return x_local, y_local

        p1 = to_local(wp1.transform.location)
        p2 = to_local(wp2.transform.location)
        p3 = to_local(wp3.transform.location)

        radius = self.radius_three_points(p1, p2, p3)
        return radius 


    def plot_results(self, ax1, ax2, ax3):
        ax1.plot(self.counts, self.errors_lks,c='r',ls='-', marker='o', mec='b',mfc='w')  
        ax2.plot(self.counts, self.current_speed_list,c='r',ls='-', marker='o', mec='g',mfc='w')  
        curvature = 1000/self.get_turning_radius_three_points(self.ego_vehicle)
        self.curvature_list.append(curvature)
        ax3.plot(self.counts, self.curvature_list, c='r',ls='-', marker='o', mec='y',mfc='w') 
        ax1.axhline(y=0, color='k', linestyle='--')
        for i in self.in_control_steps:
            ax1.axvline(x=i, color='g', linestyle='--')
            ax2.axvline(x=i, color='g', linestyle='--')
            ax3.axvline(x=i, color='g', linestyle='--')
        plt.tight_layout()
        plt.pause(0.1)


    def generate_target(self):
        try:
            print("Starting radar, camera, LIDAR, and object detection ROS publishing (press Ctrl+C to stop)...")
            self.start_time = time.time()
            self.lane_error = []
            
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1  # 10 FPS
            self.world.apply_settings(settings)

            print(f"当前参数：")
            print(f"fixed_delta_seconds: {settings.fixed_delta_seconds}")
            print(f"max_substep_delta_time: {settings.max_substep_delta_time}")
            print(f"max_substeps: {settings.max_substeps}")

            plt.ion()
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

            # 初始化空的线条对象
            # line1, = ax1.plot([], [], 'ro-', markersize=3, label='Lateral Error')
            # line2, = ax2.plot([], [], 'go-', markersize=3, label='Speed')
            # line3, = ax3.plot([], [], 'yo-', markersize=3, label='Curvature')
            line1, = ax1.plot([], [], c='r',ls='-', marker='o', mec='b',mfc='w')
            line2, = ax2.plot([], [], c='r',ls='-', marker='o', mec='g',mfc='w')
            line3, = ax3.plot([], [], c='r',ls='-', marker='o', mec='y',mfc='w')
            # line4, = ax4.plot([], [], c='r',ls='-', marker='o', mec='m',mfc='w')
            
            # 存储控制步骤的竖线引用
            self.control_lines_ax1 = []
            self.control_lines_ax2 = [] 
            self.control_lines_ax3 = []
            # self.control_lines_ax4 = []
            
            # 记录已绘制的控制步骤
            self.drawn_control_steps = set()

            ax1.set_title("LKS Lateral Error Over Time")
            ax1.set_xlabel("Time Step")
            ax1.set_ylabel("Error [m]")
            ax1.set_ylim(-1.2, 0.4)
            # ax1.legend()
            ax1.axhline(y=0, color='k', linestyle='--')  # 水平参考线只需绘制一次

            ax2.set_title("Vehicle Speed")
            ax2.set_xlabel("Time Step")
            ax2.set_ylabel("Speed [km/h]")
            ax2.set_ylim(0, 120)
            # ax2.legend()

            ax3.set_title("Road Curvature")
            ax3.set_xlabel("Time Step")
            ax3.set_ylabel(r'Curvature [$\times 10^{-3}$/m]')
            ax3.set_ylim(-0.5, 4.5)
            # ax3.legend()

            # ax4.set_title("LKS Preview Error Over Time")
            # ax4.set_xlabel("Time Step")
            # ax4.set_ylabel("ey [m]")
            # ax4.set_ylim(-1.2, 0.4)
            
            # 初始绘制
            plt.tight_layout()
            plt.pause(0.1)

            # import ipdb
            # ipdb.set_trace()
            
            while True:
                total_start_time = time.time()

                self.world.tick()
                
                current_speed_limit = self.ego_vehicle.get_speed_limit()
                # print(f"Current speed limit: {current_speed_limit} km/h")
                for vehicle in self.vehicles:
                    self.tm.vehicle_percentage_speed_difference(vehicle, -200)
                self.tm.vehicle_percentage_speed_difference(self.ego_vehicle, -500)

                if not self.image_queue.empty():
                    img = np.array(self.image_queue.get_nowait())
                    surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                    if pygame.display.get_active():
                        self.screen.blit(surface, (0, 0))
                        pygame.display.flip()

                vehicle_speed = self.current_speed/3.6
                if vehicle_speed < 10.0:
                    throttle = 0.8
                else:
                    throttle = 0.8
                
                current_lat_accel = self.get_current_lateral_acceleration()
                if abs(current_lat_accel) > 3.0:
                    print(f"Current lateral acceleration: {current_lat_accel:.4f} m/s²")
                
                self.manual_control = carla.VehicleControl(throttle=throttle, steer=0.0, brake=0.0)
                self.ego_gt_lane_points = self.get_gt_lane_points(self.ego_vehicle)
                self.curvature = 1000/self.get_turning_radius_three_points(self.ego_vehicle)

                self.LKS_sendto_simulink()
                self.LKS_receivefrom_simulink()

                nearest_lane_point = self.ego_gt_lane_points[0]
                error_lks = -(nearest_lane_point[1]+nearest_lane_point[5])/2
                self.errors_lks.append(error_lks)
                self.counts.append(self.count)
                self.nearest_lane_points_history.append(nearest_lane_point[0])
                if self.system_state == 0:
                    self.lks_control_errors.append(error_lks)
                
                
                # 只更新新增的数据点
                line1.set_data(self.counts, self.errors_lks)
                line2.set_data(self.counts, self.current_speed_list)
                
                # self.curvature = 1000/self.get_turning_radius_three_points(self.ego_vehicle)
                self.curvature_list.append(self.curvature)
                line3.set_data(self.counts, self.curvature_list)
                # line4.set_data(self.counts, self.ey_list)
                
                # 动态调整x轴范围，显示所有数据但自动滚动
                if self.counts:
                    current_max_x = max(self.counts)
                    # 显示最近100个点或所有点（如果少于100个）
                    # x_min = max(0, current_max_x - 100) if len(self.counts) > 100 else 0
                    x_min = 0
                    ax1.set_xlim(x_min, current_max_x + 5)  # 留一点右边距
                    ax2.set_xlim(x_min, current_max_x + 5)
                    ax3.set_xlim(x_min, current_max_x + 5)
                    # ax4.set_xlim(x_min, current_max_x + 5)
                
                # 只在新增控制步骤时添加竖线
                if self.count in self.in_control_steps and self.count not in self.drawn_control_steps:
                    line_ax1 = ax1.axvline(x=self.count, color='g', linestyle='--', alpha=0.7)
                    line_ax2 = ax2.axvline(x=self.count, color='g', linestyle='--', alpha=0.7)
                    line_ax3 = ax3.axvline(x=self.count, color='g', linestyle='--', alpha=0.7)
                    # line_ax4 = ax4.axvline(x=self.count, color='g', linestyle='--', alpha=0.7)
                    
                    self.control_lines_ax1.append(line_ax1)
                    self.control_lines_ax2.append(line_ax2)
                    self.control_lines_ax3.append(line_ax3)
                    # self.control_lines_ax4.append(line_ax4)
                    self.drawn_control_steps.add(self.count)

                plt.tight_layout()
                plt.pause(0.001)  # 大幅减少暂停时间

                if self.manual_mode:
                    self.handle_manual_control()
                self.ego_vehicle.apply_control(self.manual_control)
                
                if self.bev_img is not None:
                    cv2.namedWindow('bev', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('bev', 800, 600)
                    cv2.imshow('bev', self.bev_img)
                    cv2.waitKey(1)
                
                if self.whethe_end:
                    vehicle_location = self.ego_vehicle.get_transform().location
                    dis = np.sqrt(math.pow(vehicle_location.x - self.end_point.x, 2) + math.pow(vehicle_location.y - self.end_point.y, 2))
                    if dis < 5:
                        self.save_results()
                        fig_name = self.error_save_file_name + '.png'
                        plt.savefig(fig_name)
                        break
                
                self.count += 1
                time.sleep(0.01)
                total_end_time = time.time()
                # print(f"Total loop time: {total_end_time - total_start_time:.4f} seconds")
                
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cv2.destroyAllWindows()
            self.tcp_server.close()
            self.tcp_client.close()

    def destroy(self):
        # Cleanup
        self.radar.stop()
        self.camera.stop()
        # self.lidar.stop()
        self.radar.destroy()
        self.camera.destroy()
        self.imu_sensor.destroy()
        # self.lidar.destroy()
        for vehicle in self.vehicles:
            vehicle.set_autopilot(False, self.tm_port)
            vehicle.destroy()
        print(f"Destroyed {len(self.vehicles)} vehicles, radar, camera, and LIDAR.")


def main():
    para = argparse.ArgumentParser(description="LKS Test in CARLA")
    para.add_argument("--case", type=int, default=8)
    args = para.parse_args()
    case_num = args.case
    if os.path.exists("lane_error.txt"):
        os.remove("lane_error.txt")
    lks_actor = lks(case_num)
    
    thread_3 = threading.Thread(target=lks_actor.bev_camera.listen(lks_actor.bev_camera_callback), name='T3')
    thread_4 = threading.Thread(target=lks_actor.pygame_camera.listen(lks_actor.pygame_callback), name='T4')
    # thread_5 = threading.Thread(target=lks_actor.imu_sensor.listen(lks_actor.imu_callback), name='T5')

    thread_3.start()
    thread_4.start()

    thread_3.join()
    thread_4.join()
    time.sleep(0.3)
    lks_actor.generate_target()

if __name__ == '__main__':
    main()