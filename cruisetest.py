import carla
import math
import numpy as np
import cv2
import kalman_filter
import lane_detection
import radar_cluster
import pygame
import threading
import time
from acc_planning_control import ACCPlanningControl, ACCMode

#用于单独测试Cuise mode功能

class CruiseTest:
    def __init__(self):
        self.tracker = kalman_filter.RadarTracker()
        self.radar_point_cluster = radar_cluster.RadarClusterNode()
        self.max_follow_distance = 150  # 增大最大跟车距离，确保进入CRUISE模式
        self.lane_detector = lane_detection.LaneDetector()
        self.radar_detections = []
        self.latest_camera_image = None
        self.radar_2_world = []
        self.world_2_camera = []
        self.cluster = []
        self.track_id = []

        self.image_width = 1280
        self.image_height = 720

        # 车道信息
        self.lane_offset = 0.0
        self.current_waypoint = None
        self.left_waypoint = None
        self.right_waypoint = None

        self.target_vehicle = None
        self.init_carla()

    def init_carla(self):
        # Connect to the CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        # 加载Town04地图，这个地图有较长的直道，适合测试CRUISE模式
        self.world = self.client.load_world('Town04', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

        # 关闭建筑物，减少视觉干扰
        self.world.unload_map_layer(carla.MapLayer.Buildings)

        # 获取蓝图库和地图
        blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        self.manual_mode = False
        self.manual_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        # 初始化pygame用于手动控制
        if self.manual_mode:
            pygame.init()
            pygame.display.set_mode((200, 200))
            pygame.key.set_repeat(50, 50)

        # 选择车辆蓝图
        ego_vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]

        # 获取有效生成点
        spawn_points = self.map.get_spawn_points()
        # 选择一个直道上的生成点
        spawn_point = None
        for point in spawn_points:
            # 获取该点的路段信息
            waypoint = self.map.get_waypoint(point.location)
            # 检查是否为直道
            if abs(waypoint.transform.rotation.yaw) % 90 < 5 or abs(waypoint.transform.rotation.yaw) % 90 > 85:
                spawn_point = point
                break

        if spawn_point is None:
            spawn_point = spawn_points[0]  # 如果没找到合适的直道，使用第一个生成点

        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, spawn_point)

        # 设置交通管理器
        tm = self.client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm_port = tm.get_port()

        # 设置传感器
        # 设置雷达传感器
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('range', '100.0')
        radar_bp.set_attribute('horizontal_fov', '120.0')
        radar_bp.set_attribute('vertical_fov', '30.0')
        radar_bp.set_attribute('points_per_second', '10000')
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)

        # 设置摄像头传感器
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.5))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)

        # 设置LIDAR传感器
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('points_per_second', '1000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('lower_fov', '-10')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)

    def get_ego_speed(self, vehicle):
        velocity = vehicle.get_velocity()
        speed_m_s = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        return speed_m_s

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


    def get_lane_info(self):
        """使用CARLA API获取车道信息和车辆相对于车道中心的偏移"""
        # 获取车辆当前位置
        vehicle_location = self.ego_vehicle.get_location()

        # 获取当前位置的路点
        self.current_waypoint = self.map.get_waypoint(vehicle_location, project_to_road=True,
                                                      lane_type=carla.LaneType.Driving)

        if self.current_waypoint is None:
            return 0.0

        # 获取左右车道线的路点
        self.left_waypoint = self.current_waypoint.get_left_lane()
        self.right_waypoint = self.current_waypoint.get_right_lane()

        # 计算车辆相对于车道中心的横向偏移
        # CARLA中车道中心是以路点(waypoint)的位置表示的
        vehicle_transform = self.ego_vehicle.get_transform()

        # 计算车辆位置到车道中心的向量
        lane_center = self.current_waypoint.transform.location
        lane_direction = self.current_waypoint.transform.get_forward_vector()

        # 计算车辆位置到车道中心的向量
        to_center_vector = carla.Vector3D(lane_center.x - vehicle_location.x,
                                          lane_center.y - vehicle_location.y,
                                          0)  # 忽略高度差异

        # 计算车道方向的垂直向量（车道的右侧方向）
        right_direction = carla.Vector3D(lane_direction.y, -lane_direction.x, 0).make_unit_vector()

        # 计算偏移量（正值表示车辆在车道右侧，负值表示在左侧）
        offset = to_center_vector.dot(right_direction)

        self.lane_offset = offset

        # 计算车道宽度（如果有左右车道线）
        lane_width = 3.5  # 默认车道宽度
        if self.left_waypoint is not None and self.right_waypoint is not None:
            left_loc = self.left_waypoint.transform.location
            right_loc = self.right_waypoint.transform.location
            lane_width = math.sqrt((left_loc.x - right_loc.x) ** 2 + (left_loc.y - right_loc.y) ** 2)

        # 归一化偏移量，转换为[-1,1]范围，其中0表示车道中心，±1表示车道边缘
        normalized_offset = offset / (lane_width / 2)

        return offset, normalized_offset, lane_width

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

    def draw_lane_info(self, image, offset, normalized_offset, lane_width):
        """在图像上绘制车道信息"""
        h, w = image.shape[:2]

        # 绘制车道中心线和车辆位置指示
        center_x = w // 2
        bottom_y = h - 50

        # 绘制车道（简化表示）
        lane_width_px = 200  # 像素表示的车道宽度
        left_line_x = center_x - lane_width_px // 2
        right_line_x = center_x + lane_width_px // 2

        # 绘制车道线
        cv2.line(image, (left_line_x, bottom_y), (left_line_x, bottom_y - 100), (255, 255, 0), 2)
        cv2.line(image, (right_line_x, bottom_y), (right_line_x, bottom_y - 100), (255, 255, 0), 2)

        # 绘制车道中心线
        cv2.line(image, (center_x, bottom_y), (center_x, bottom_y - 100), (255, 255, 255), 1)

        # 计算车辆在图像中的位置
        vehicle_pos_x = center_x + int(normalized_offset * lane_width_px / 2)

        # 绘制车辆位置
        cv2.circle(image, (vehicle_pos_x, bottom_y - 50), 5, (0, 0, 255), -1)

        # 绘制偏移信息文本
        offset_text = f"Lane Offset: {offset:.2f} m"
        width_text = f"Lane Width: {lane_width:.2f} m"

        cv2.putText(image, offset_text, (50, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, width_text, (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return image

    def test_cruise_mode(self):
        # 创建ACC控制器，设置较高的目标速度和较大的最大跟车距离，以便于进入CRUISE模式
        acc_controller = ACCPlanningControl(self.ego_vehicle, target_speed_kmh=40.0, time_gap=2.0,
                                            max_follow_distance=self.max_follow_distance)

        try:
            print("Starting CRUISE mode test...")
            print("The vehicle should maintain its lane and drive at target speed.")
            self.get_extrinsic_params(self.radar, self.camera)

            # 创建状态显示窗口
            cv2.namedWindow("CRUISE Mode Status", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("CRUISE Mode Status", 400, 200)

            mode_colors = {
                ACCMode.CRUISE: (0, 255, 0),  # 绿色
                ACCMode.FOLLOW: (255, 255, 0),  # 黄色
                ACCMode.STOP: (0, 0, 255),  # 红色
                ACCMode.EMERGENCY: (0, 0, 255)  # 红色
            }

            while True:
                self.world.tick()

                if self.manual_mode:
                    self.handle_manual_control()
                else:
                    if self.latest_camera_image is not None:
                        image_with_radar = self.latest_camera_image.copy()

                        # 使用CARLA API获取车道信息和偏移量
                        offset, normalized_offset, lane_width = self.get_lane_info()

                        # 在图像上显示车道信息
                        image_with_lane = self.draw_lane_info(image_with_radar.copy(), offset, normalized_offset,
                                                              lane_width)

                        # 获取跟踪目标
                        track_id = self.track_id.copy() if self.track_id is not None else []
                        target_info = None

                        if track_id:
                            projected_points = self.project_radar_to_camera(track_id)
                            current_target_idx = -1

                            # 在图像上显示所有目标
                            for idx in range(len(track_id)):
                                if len(projected_points[idx]) >= 2:  # 确保有足够的点坐标
                                    u, v = projected_points[idx][0:2]
                                    color = (255, 0, 0)
                                    cv2.circle(image_with_lane, (u, v), 5, color, -1)

                                # 目标选择逻辑（保持原来的逻辑）
                                if ((-2) < track_id[idx][1] < 2):
                                    if ((current_target_idx != -1) and (
                                            track_id[idx][0] < track_id[current_target_idx][0])) or (
                                            current_target_idx == -1):
                                        current_target_idx = idx

                            # 标记选中的目标
                            if current_target_idx >= 0 and len(projected_points[current_target_idx]) >= 2:
                                u, v = projected_points[current_target_idx][0:2]
                                cv2.circle(image_with_lane, (u, v), 10, (255, 255, 255), -1)
                                cv2.putText(image_with_lane, "id=" + str(track_id[current_target_idx][-1]),
                                            (u + 5, v), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 225, 100), 2)
                                target_info = track_id[current_target_idx]

                                # 验证目标有效性
                                if not all(np.isfinite(target_info)):
                                    print("Invalid target info, skipping control")
                                    target_info = None

                        # 应用ACC控制，传递车道偏移信息
                        control = acc_controller.update(target_info, offset)
                        self.ego_vehicle.apply_control(control)

                        # 显示当前模式信息
                        status_image = np.zeros((200, 400, 3), dtype=np.uint8)
                        mode_text = f"Mode: {acc_controller.mode.name}"
                        speed_text = f"Speed: {acc_controller.get_ego_state()[0]:.2f} m/s"
                        target_text = f"Target Speed: {acc_controller.target_speed:.2f} m/s"
                        lane_text = f"Lane Offset: {offset:.2f} m"

                        cv2.putText(status_image, mode_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    mode_colors.get(acc_controller.mode, (255, 255, 255)), 2)
                        cv2.putText(status_image, speed_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)
                        cv2.putText(status_image, target_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)
                        cv2.putText(status_image, lane_text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)

                        cv2.imshow("CRUISE Mode Status", status_image)
                        cv2.imshow("Camera View with Lane Info", image_with_lane)

                        key = cv2.waitKey(1)
                        # 按ESC退出
                        if key == 27:
                            break

                        # 按空格键切换到CRUISE模式（强制）
                        if key == 32:  # 空格键
                            print("Forcing CRUISE mode")
                            acc_controller.mode = ACCMode.CRUISE

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self.radar.stop()
            self.camera.stop()
            self.lidar.stop()
            cv2.destroyAllWindows()

            # 清理资源
            self.radar.destroy()
            self.camera.destroy()
            self.lidar.destroy()
            self.ego_vehicle.destroy()
            print("Test completed and resources cleaned up.")


def main():
    cruise_test = CruiseTest()

    # 启动传感器线程
    thread_1 = threading.Thread(target=cruise_test.radar.listen(cruise_test.radar_callback), name='T1')
    thread_2 = threading.Thread(target=cruise_test.camera.listen(cruise_test.camera_callback), name='T2')
    thread_3 = threading.Thread(target=cruise_test.lidar.listen(cruise_test.lidar_callback), name='T3')

    thread_1.start()
    thread_2.start()
    thread_3.start()

    # 等待传感器初始化
    time.sleep(1)

    # 运行测试
    cruise_test.test_cruise_mode()


if __name__ == '__main__':
    main()