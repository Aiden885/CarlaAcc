import carla
import math
import numpy as np
import cv2
import threading
from lane_detection import LaneDetector
from kalman_filter import RadarTracker
from radar_cluster import RadarClusterNode
from acc_planning_control import ACCPlanningControl
from vehicle_manager import VehicleManager
from sensors_manager import SensorsManager
from target_vehicle_controller import TargetVehicleController



class ACC:
    def __init__(self, target_speed_range=(20.0, 50.0)):
        self.tracker = RadarTracker()
        self.lane_detector = LaneDetector()
        self.radar_point_cluster = RadarClusterNode()
        self.max_follow_distance = 50

        self.radar_detections = []
        self.latest_camera_image = None
        self.radar_points = []
        self.filted_points = []
        self.radar_2_world = []
        self.world_2_camera = []
        self.cluster = []
        self.track_id = []

        self.image_width = 1280
        self.image_height = 720

        self.target_vehicle = None
        self.ego_vehicle = None
        self.manual_mode = False
        self.manual_control = None

        # 初始化CARLA连接
        self.init_carla()

        # 初始化目标车辆控制器（负责根据道路形状动态调整速度）
        self.target_controller = TargetVehicleController(
            self.client,
            self.target_vehicle,
            self.world.get_map(),
            speed_range=target_speed_range
        )


    def init_carla(self):
        # 连接到CARLA服务器
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        # 加载带有建筑和停放车辆的Town04地图
        self.world = self.client.load_world('Town04', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

        # 初始化车辆管理器
        self.vehicle_manager = VehicleManager(self.world, self.client)

        # 生成车辆
        self.ego_vehicle, self.target_vehicle = self.vehicle_manager.spawn_vehicles()

        # 初始化传感器管理器
        self.sensors_manager = SensorsManager(self.world, self.ego_vehicle)
        self.radar = self.sensors_manager.setup_radar(self.radar_callback)
        self.camera = self.sensors_manager.setup_camera(self.camera_callback)
        self.lidar = self.sensors_manager.setup_lidar(self.lidar_callback)

    def get_ego_speed(self, vehicle):
        velocity = vehicle.get_velocity()  # carla.Vector3D
        speed_m_s = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_kmh = speed_m_s * 3.6  # 转换m/s为km/h
        return speed_m_s

    # 雷达回调
    def radar_callback(self, radar_data):
        self.radar_points = []
        self.filted_points = []
        ego_velocity = self.get_ego_speed(self.ego_vehicle)
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

    # 相机回调
    def camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        self.latest_camera_image = array

    # 激光雷达回调
    def lidar_callback(self, lidar_data):
        points = []
        for point in lidar_data:
            x = point.point.x
            y = point.point.y
            z = point.point.z
            intensity = point.intensity
            points.append([x, y, z, intensity])
        self.latest_lidar_points = points

    # 获取外部参数（从雷达到相机）
    def get_extrinsic_params(self, radar_sensor, camera_sensor):
        self.radar_2_world = radar_sensor.get_transform().get_matrix()
        self.world_2_camera = np.array(camera_sensor.get_transform().get_inverse_matrix())

    # 将雷达点投影到相机图像
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

    def handle_manual_control(self):
        # 处理手动控制逻辑（与原始代码相同）
        import pygame
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

            # Reset steer if no input
            if not (keys[pygame.K_a] or keys[pygame.K_d]):
                self.manual_control.steer *= 0.9  # Dampen steering

            self.ego_vehicle.apply_control(self.manual_control)
            print(
                f"Manual: Throttle={self.manual_control.throttle:.2f}, Steer={self.manual_control.steer:.2f}, Brake={self.manual_control.brake:.2f}")

    def generate_target(self):
        acc_controller = ACCPlanningControl(self.ego_vehicle, target_speed_kmh=40.0, time_gap=2.0,
                                            max_follow_distance=self.max_follow_distance)
        try:
            print("Starting radar, camera, LIDAR, and ACC control...")
            self.get_extrinsic_params(self.radar, self.camera)

            # 启动目标车辆控制器更新线程
            self.target_controller.start_update_thread()

            while True:
                self.world.tick()
                if self.manual_mode:
                    self.handle_manual_control()
                else:
                    if self.latest_camera_image is not None:
                        image_with_radar = self.latest_camera_image.copy()

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
                                if not all(np.isfinite(target_info)):
                                    target_info = None
                        control = acc_controller.update(target_info)
                        self.ego_vehicle.apply_control(control)

                        # 显示当前目标车辆信息
                        current_speed = self.target_controller.get_current_speed()
                        road_status = "弯道" if self.target_controller.is_curved_road() else "直道"
                        cv2.putText(image_with_radar, f"目标车速: {current_speed:.1f} km/h, 道路: {road_status}",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                        cv2.imshow("Radar and Objects on Camera", image_with_radar)
                        cv2.waitKey(1)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cv2.destroyAllWindows()
            self.target_controller.stop_update_thread()

    def destroy(self):
        # 清理资源
        self.sensors_manager.destroy_sensors()
        self.vehicle_manager.destroy_vehicles()
        print("Destroyed all resources.")


def main():
    # 可以在这里设置目标车辆的速度范围（km/h）
    acc_actor = ACC(target_speed_range=(20.0, 50.0))

    # 启动传感器监听线程
    thread_1 = threading.Thread(target=acc_actor.radar.listen(acc_actor.radar_callback), name='T1')
    thread_2 = threading.Thread(target=acc_actor.camera.listen(acc_actor.camera_callback), name='T2')
    thread_3 = threading.Thread(target=acc_actor.lidar.listen(acc_actor.lidar_callback), name='T3')

    thread_1.start()
    thread_2.start()
    thread_3.start()

    acc_actor.generate_target()


if __name__ == '__main__':
    main()